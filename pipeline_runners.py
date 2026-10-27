from __future__ import annotations

import json
import os
import time
from collections import Counter
from datetime import datetime, timezone

try:
    from .runtime import bootstrap
except ImportError:
    from runtime import bootstrap


def _task_header(legacy, title: str) -> None:
    legacy._log("=" * 72)
    legacy._log(title)
    legacy._log("=" * 72)
    legacy._log(f"Output directory: {legacy.OUT_DIR}")
    legacy._log(f"Models: {legacy.MODEL_LIST}")
    legacy._log(f"Run mode: {'FULL' if legacy.RUN_FULL else 'SAMPLED'}")


def _prepare_model_dir(legacy, model_name: str, task_id: str = "") -> str:
    model_out_dir = os.path.join(legacy.OUT_DIR, legacy.model_out_tag(model_name))
    os.makedirs(model_out_dir, exist_ok=True)
    legacy._log(f"[Model] {model_name} -> {model_out_dir}")
    meta = {
        "model": model_name,
        "task": task_id,
        "version": "V18",
        "backend": getattr(legacy, "BACKEND", ""),
        "temperature": getattr(legacy, "TEMPERATURE", 0.0),
        "seed": getattr(legacy, "SEED", 42),
        "run_full": getattr(legacy, "RUN_FULL", False),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "out_dir": model_out_dir,
    }
    meta_path = os.path.join(model_out_dir, f"run_metadata_{task_id}.json" if task_id else "run_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return model_out_dir


def _load_dataset(legacy, ds: str):
    """Load dataset, returning (comments, markets, positions) or None if data is missing."""
    try:
        comments, markets = legacy.load_data(ds)
        positions = legacy.load_positions(ds)
        return comments, markets, positions
    except FileNotFoundError as e:
        legacy._log(f"[SKIP] {ds}: {e}", "warning")
        return None


def _iter_topics(legacy, qualified: dict, ds: str):
    topics_present = sorted(set(v["topic"] for v in qualified.values()))
    for topic in topics_present:
        if ds == "polymarket":
            cfg = legacy.TOPIC_CONFIG_POLYMARKET.get(topic, legacy.DEFAULT_TOPIC_CFG_POLYMARKET)
        else:
            cfg = legacy.TOPIC_CONFIG.get(topic, legacy.DEFAULT_TOPIC_CFG)

        all_topic_mids = [mid for mid, v in qualified.items() if v["topic"] == topic]
        cap = legacy.N_MARKETS_PER_TOPIC.get(ds)
        if legacy.RUN_FULL or topic in legacy.TOPIC_NO_CAP:
            cap = None
        if cap is not None and len(all_topic_mids) > cap:
            all_topic_mids = legacy.random.sample(all_topic_mids, cap)
        yield topic, cfg, set(all_topic_mids)


def _build_data_plans(legacy) -> dict:
    """Pre-compute all dataset structures once so every model in MODEL_LIST
    sees exactly the same markets, pairs, and windows.

    The random market-capping in _iter_topics advances Python's global random
    state; calling it once per dataset (instead of once per model per dataset)
    ensures the cap is applied identically across all models.

    Returns
    -------
    plans : dict
        {ds: {
            "comments", "markets", "positions",
            "qualified", "comments_q",
            "task2_all",
            "all_wins",
            "topic_plans" : [(topic, cfg, topic_mids), ...],
            "topic_data"  : {topic: {cfg, topic_mids, comments_t, labeled_t,
                                      task2_t, wins_t, wins_count_t}},
        }}
    """
    plans: dict = {}
    for ds in ["polymarket", "manifold"]:
        loaded = _load_dataset(legacy, ds)
        if loaded is None:
            continue
        comments, markets, positions = loaded
        legacy._log(
            f"[Plan:{ds}] comments={len(comments)} markets={len(markets)} "
            f"positions={len(positions)}"
        )

        qualified = legacy.select_qualified_markets(
            comments, markets, prices_dir=legacy.PRICES_DIRS[ds]
        )
        if not qualified:
            legacy._log(f"[Plan:{ds}] no qualified markets — skipping")
            continue

        topic_dist = Counter(v["topic"] for v in qualified.values())
        legacy._log(
            f"[Plan:{ds}] qualified={len(qualified)} topic_dist={dict(topic_dist)}"
        )

        qual_mids = set(qualified.keys())
        comments_q = comments[
            comments["market_id"].astype(str).isin(qual_mids)
        ].reset_index(drop=True)

        task2_all = legacy.build_task2_pairs(comments_q, positions, ds)
        all_wins = legacy.build_all_windows(
            comments_q, prices_dir=legacy.PRICES_DIRS[ds], markets=markets
        )

        # ── Freeze random market-cap selection HERE (once per dataset) ──────
        topic_plans = list(_iter_topics(legacy, qualified, ds))
        legacy._log(
            f"[Plan:{ds}] topic_plans="
            f"{[(t, len(mids)) for t, _cfg, mids in topic_plans]}"
        )

        # Save shared task2 CSV once (not duplicated per model)
        if not task2_all.empty:
            shared_csv = os.path.join(legacy.OUT_DIR, f"task2_pairs_{ds}_all.csv")
            task2_all.to_csv(shared_csv, index=False)
            legacy._log(f"[Plan:{ds}] saved {shared_csv}")

        # Pre-compute per-topic slices (identical for every model)
        topic_data: dict = {}
        for topic, cfg, topic_mids in topic_plans:
            comments_t = comments[
                comments["market_id"].astype(str).isin(topic_mids)
            ].reset_index(drop=True)
            labeled_t = legacy.filter_labeled(comments_t)
            task2_t = (
                task2_all[
                    task2_all["market_id"].astype(str).isin(topic_mids)
                ].reset_index(drop=True)
                if not task2_all.empty
                else task2_all
            )
            wins_t = [w for w in all_wins if str(w["market_id"]) in topic_mids]
            wins_count_t = [
                w
                for w in legacy.build_all_windows_comment_count(
                    comments_t, prices_dir=legacy.PRICES_DIRS[ds], markets=markets
                )
                if str(w["market_id"]) in topic_mids
            ]
            topic_data[topic] = {
                "cfg": cfg,
                "topic_mids": topic_mids,
                "comments_t": comments_t,
                "labeled_t": labeled_t,
                "task2_t": task2_t,
                "wins_t": wins_t,
                "wins_count_t": wins_count_t,
            }

        plans[ds] = {
            "comments": comments,
            "markets": markets,
            "positions": positions,
            "qualified": qualified,
            "comments_q": comments_q,
            "task2_all": task2_all,
            "all_wins": all_wins,
            "topic_plans": topic_plans,
            "topic_data": topic_data,
        }

    return plans


# ── Pipeline runners ────────────────────────────────────────────────────────


def run_task01_pipeline() -> None:
    legacy, adapter = bootstrap()
    os.makedirs(legacy.OUT_DIR, exist_ok=True)
    log_path = legacy._setup_logging(legacy.OUT_DIR)
    adapter.set_log_dir(legacy.OUT_DIR)
    legacy._RUN_STATE_PATH = os.path.join(legacy.OUT_DIR, "run_state.json")

    _task_header(legacy, "CommitBench V18 Task0+1")
    legacy._log(f"Log file: {log_path}")
    if not legacy.check_backend():
        return

    # Build data plans once — all models share exactly the same dataset
    plans = _build_data_plans(legacy)
    if not plans:
        legacy._log("[ERROR] No valid datasets found. Aborting.")
        return

    t_global = time.time()
    for model_name in legacy.MODEL_LIST:
        legacy.MODEL = model_name
        model_out_dir = _prepare_model_dir(legacy, model_name, task_id="task01")

        for ds, plan in plans.items():
            legacy._log(f"[Dataset] {ds}")
            comments  = plan["comments"]
            positions = plan["positions"]

            for topic, td in plan["topic_data"].items():
                cfg = td["cfg"]
                topic_mids = td["topic_mids"]
                comments_t = td["comments_t"]
                labeled_t = td["labeled_t"]
                task2_t = td["task2_t"]
                wins_t = td["wins_t"]

                out_subdir = os.path.join(model_out_dir, ds, topic)
                os.makedirs(out_subdir, exist_ok=True)

                legacy._log(
                    f"[Topic] {ds}/{topic} markets={len(topic_mids)} "
                    f"labeled={len(labeled_t)} task2_pairs={len(task2_t)} "
                    f"windows={len(wins_t)}"
                )

                legacy.run_phase0(labeled_t, task2_t, wins_t, f"{ds}/{topic}")
                t0_result = legacy.run_task0(
                    comments_t, f"{ds}/{topic}", out_subdir, n_each=cfg["n_task0"]
                )
                pos_pool_t0 = t0_result["pos_pool"] if t0_result is not None else None
                legacy.run_task1(
                    labeled_t,
                    f"{ds}/{topic}",
                    out_subdir,
                    n_each=cfg["n_task1"],
                    comments_nopos=comments_t,
                    explicit_pos_pool=pos_pool_t0,
                )

                # ── Within-user DiD (Experiment 3) ──────────────────────────
                pre_did = legacy.build_within_user_pre_comments(
                    comments_t, positions, f"{ds}/{topic}",
                )
                if len(pre_did) >= 10:
                    legacy.run_task1_did(
                        pre_did, f"{ds}/{topic}", out_subdir,
                        n_each=cfg["n_task1"],
                    )
                else:
                    legacy._log(
                        f"  [DiD] {ds}/{topic}: insufficient pre-commitment comments "
                        f"(n={len(pre_did)}), skipping DiD"
                    )

    elapsed = time.time() - t_global
    legacy._log(f"Task0+1 finished in {elapsed:.1f}s")


def run_task2_pipeline() -> None:
    legacy, adapter = bootstrap()
    os.makedirs(legacy.OUT_DIR, exist_ok=True)
    log_path = legacy._setup_logging(legacy.OUT_DIR)
    adapter.set_log_dir(legacy.OUT_DIR)
    legacy._RUN_STATE_PATH = os.path.join(legacy.OUT_DIR, "run_state.json")

    _task_header(legacy, "CommitBench V18 Task2")
    legacy._log(f"Log file: {log_path}")
    if not legacy.check_backend():
        return

    plans = _build_data_plans(legacy)
    if not plans:
        legacy._log("[ERROR] No valid datasets found. Aborting.")
        return

    t_global = time.time()
    for model_name in legacy.MODEL_LIST:
        legacy.MODEL = model_name
        model_out_dir = _prepare_model_dir(legacy, model_name, task_id="task2")

        for ds, plan in plans.items():
            positions = plan["positions"]
            comments = plan["comments"]

            for topic, td in plan["topic_data"].items():
                cfg = td["cfg"]
                topic_mids = td["topic_mids"]
                comments_t = td["comments_t"]
                task2_t = td["task2_t"]

                out_subdir = os.path.join(model_out_dir, ds, topic)
                os.makedirs(out_subdir, exist_ok=True)

                legacy._log(f"[Topic] {ds}/{topic} task2_pairs={len(task2_t)}")

                legacy.run_task2(task2_t, f"{ds}/{topic}", out_subdir, n_each=cfg["n_task2"])
                legacy.run_task2_binary(
                    task2_t, f"{ds}/{topic}", out_subdir,
                    n_each=cfg["n_task2"], model=model_name
                )
                legacy.run_task2_multi_threshold(
                    comments_t,
                    positions,
                    f"{ds}/{topic}",
                    out_subdir,
                    n_each=cfg["n_task2"],
                    model=model_name,
                )

    elapsed = time.time() - t_global
    legacy._log(f"Task2 finished in {elapsed:.1f}s")


def run_task3_pipeline() -> None:
    legacy, adapter = bootstrap()
    os.makedirs(legacy.OUT_DIR, exist_ok=True)
    log_path = legacy._setup_logging(legacy.OUT_DIR)
    adapter.set_log_dir(legacy.OUT_DIR)
    legacy._RUN_STATE_PATH = os.path.join(legacy.OUT_DIR, "run_state.json")

    _task_header(legacy, "CommitBench V18 Task3")
    legacy._log(f"Log file: {log_path}")
    if not legacy.check_backend():
        return

    plans = _build_data_plans(legacy)
    if not plans:
        legacy._log("[ERROR] No valid datasets found. Aborting.")
        return

    t_global = time.time()
    for model_name in legacy.MODEL_LIST:
        legacy.MODEL = model_name
        model_out_dir = _prepare_model_dir(legacy, model_name, task_id="task3")

        for ds, plan in plans.items():
            for topic, td in plan["topic_data"].items():
                cfg = td["cfg"]
                topic_mids = td["topic_mids"]
                wins_t = td["wins_t"]
                wins_count_t = td["wins_count_t"]

                out_subdir = os.path.join(model_out_dir, ds, topic)
                os.makedirs(out_subdir, exist_ok=True)

                legacy._log(
                    f"[Topic] {ds}/{topic} windows={len(wins_t)} "
                    f"windows_count={len(wins_count_t)}"
                )

                legacy.run_task3(
                    wins_t, f"{ds}/{topic}", out_subdir,
                    n_win=cfg["n_task3"], variant="time"
                )
                legacy.run_task3(
                    wins_count_t, f"{ds}/{topic}", out_subdir,
                    n_win=cfg["n_task3"], variant="count"
                )

    elapsed = time.time() - t_global
    legacy._log(f"Task3 finished in {elapsed:.1f}s")
