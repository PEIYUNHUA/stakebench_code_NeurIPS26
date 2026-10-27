"""
E9b: freeze the source instance list for the translation-manipulation experiment.

Design decision D1 (E9_multilingual_design.md): do not rebuild an evaluation set. Take the
already-published English evaluation instances, sample a stratified subset, freeze it, and
use that one list for every arm.

Two requirements the runbook implies but does not spell out, both enforced here:

  1. The instance must be covered by ALL target models. The en arm doubles as a consistency
     check against the published logs (RUNBOOK 4.4) and every arm must be perfectly paired
     across models. An instance only some models were scored on can do neither job. The
     workbook's E6 caveat records that model coverage in the released logs is uneven, so
     this is filtered explicitly rather than assumed.

  2. The gold label must be identical across all target model logs. E3b found gold labels
     that differ between closed-model and open-model logs (G3: Polymarket 18, Manifold 89).
     Those keys are dropped and counted, not silently deduplicated by glob order.

No LLM calls. No writes outside rebuttal/results/e9/.

Outputs (rebuttal/results/e9/):
  e9b_source_instances.csv   frozen instances, text_en + gold
  e9b_freeze_report.json     counts, drop reasons, strata, instance-list hash
"""
import os, sys, json, glob, hashlib
import pandas as pd
import numpy as np

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "stakebench_code_NeurIPS26"))
import config

LOGS = os.path.join(ROOT, "commitbench_v18", "commitbench_v18")
OUT = os.path.join(ROOT, "rebuttal", "results", "e9")
os.makedirs(OUT, exist_ok=True)

# The six evaluated models. The open-weight names are the paper's checkpoints; in this run
# they are served through OpenRouter because the local GPU is 8 GB and cannot host qwen3-32b.
TARGET_LOG_DIRS = [
    "Qwen3-8B", "Qwen3-32B", "Llama3.2-3B",
    "Openrouter-GPT-5.5", "Openrouter-CLAUDE-HAIKU-4.5", "Openrouter-GEMINI-2.5-FLASH",
]

N_G1 = 300          # 150 YES / 150 NO
N_G2 = 200
SEED = config.SEED  # 42


def instance_key(market_id, created_at, text):
    """Published logs carry no comment_id; build a stable one from the identifying triple."""
    raw = f"{market_id}||{created_at}||{text}"
    return hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()[:16]


def load_task(pattern, gt_norm):
    """Load one task across all target model logs; returns the long frame with keys."""
    frames = []
    for mdir in TARGET_LOG_DIRS:
        for f in glob.glob(os.path.join(LOGS, mdir, "*", "*", pattern)):
            parts = f.replace("\\", "/").split("/")
            d = pd.read_csv(f)
            d["platform"] = parts[-3]
            d["topic"] = parts[-2]
            d["log_model"] = mdir
            frames.append(d)
    if not frames:
        raise SystemExit(f"no logs matched {pattern}")
    a = pd.concat(frames, ignore_index=True)
    a["text"] = a["text"].astype(str)
    a["question"] = a["question"].astype(str)
    a["comment_id"] = [instance_key(m, c, t)
                       for m, c, t in zip(a.market_id, a.created_at, a.text)]
    a["gt"] = gt_norm(a["gt"])
    return a


def _stratified(df, n, rng):
    """Proportional allocation over platform/topic strata, then top-up to exactly n."""
    if n >= len(df):
        return df.copy()
    share = df.stratum.value_counts(normalize=True)
    picked = []
    for s, frac in share.items():
        sub = df[df.stratum == s]
        k = min(int(round(frac * n)), len(sub))
        if k > 0:
            picked.append(sub.sample(k, random_state=rng.randint(0, 10 ** 6)))
    out = pd.concat(picked, ignore_index=True) if picked else df.head(0).copy()
    if len(out) < n:
        rest = df[~df.comment_id.isin(out.comment_id)]
        if len(rest):
            out = pd.concat([out, rest.sample(min(n - len(out), len(rest)),
                                              random_state=SEED)], ignore_index=True)
    return out.sample(min(n, len(out)), random_state=SEED).reset_index(drop=True)


def freeze(task_name, pattern, gt_norm, n_target, balance):
    a = load_task(pattern, gt_norm)
    n_rows, n_keys = len(a), a.comment_id.nunique()

    cov = a.groupby("comment_id")["log_model"].nunique()
    full_cov = set(cov[cov == len(TARGET_LOG_DIRS)].index)
    dropped_coverage = n_keys - len(full_cov)

    golds = a[a.comment_id.isin(full_cov)].groupby("comment_id")["gt"].nunique()
    consistent = set(golds[golds == 1].index)
    dropped_gold = len(full_cov) - len(consistent)

    uniq = a[a.comment_id.isin(consistent)].drop_duplicates("comment_id").set_index("comment_id")
    pool = uniq.reset_index()[["comment_id", "market_id", "created_at", "question",
                               "text", "gt", "platform", "topic"]].rename(
        columns={"text": "text_en"})
    for col in ("has_position", "position_size"):
        pool[col] = uniq[col].values if col in uniq.columns else np.nan
    pool["stratum"] = pool.platform + "/" + pool.topic

    rng = np.random.RandomState(SEED)
    if balance:
        # NB: use pool["gt"], never pool.gt -- DataFrame.gt is the greater-than method.
        half = n_target // 2
        parts = [_stratified(pool[pool["gt"] == balance[0]], half, rng),
                 _stratified(pool[pool["gt"] == balance[1]], n_target - half, rng)]
        sample = pd.concat(parts, ignore_index=True)
    else:
        sample = _stratified(pool, n_target, rng)

    sample = sample.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    sample["task"] = task_name

    report = dict(
        task=task_name, log_rows=int(n_rows), unique_keys=int(n_keys),
        dropped_incomplete_coverage=int(dropped_coverage),
        dropped_gold_inconsistent=int(dropped_gold),
        eligible_pool=int(len(pool)), sampled=int(len(sample)),
        gold_distribution={str(k): int(v) for k, v in sample["gt"].value_counts().items()},
        strata={str(k): int(v) for k, v in sample.stratum.value_counts().items()},
    )
    return sample, report


def main():
    g1, rep1 = freeze("G1", "*_task0_0shot.csv",
                      lambda s: s.astype(str).str.upper().map({"TRUE": "YES", "FALSE": "NO"}),
                      N_G1, balance=("YES", "NO"))
    g2, rep2 = freeze("G2", "*_task1.csv",
                      lambda s: s.astype(str).str.upper(), N_G2, balance=None)

    cols = ["task", "comment_id", "market_id", "created_at", "platform", "topic",
            "stratum", "question", "text_en", "gt", "has_position", "position_size"]
    frozen = pd.concat([g1, g2], ignore_index=True)[cols]
    path = os.path.join(OUT, "e9b_source_instances.csv")
    frozen.to_csv(path, index=False, encoding="utf-8")

    digest = hashlib.sha256(
        "\n".join(sorted(frozen.task + ":" + frozen.comment_id)).encode()).hexdigest()
    rep = dict(seed=SEED, target_models=TARGET_LOG_DIRS, n_total=int(len(frozen)),
               instance_list_sha256=digest, tasks=[rep1, rep2])
    with open(os.path.join(OUT, "e9b_freeze_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)

    for r in (rep1, rep2):
        print(f"[{r['task']}] log_rows={r['log_rows']} unique={r['unique_keys']} "
              f"drop_coverage={r['dropped_incomplete_coverage']} "
              f"drop_gold={r['dropped_gold_inconsistent']} pool={r['eligible_pool']} "
              f"sampled={r['sampled']} gold={r['gold_distribution']}")
    print(f"\nwrote {path}  n={len(frozen)}")
    print(f"instance_list_sha256 = {digest}")


if __name__ == "__main__":
    main()
