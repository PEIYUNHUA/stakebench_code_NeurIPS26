"""
E9d: run the evaluation arms over the frozen instances.

RUNBOOK 4.4: bypass pipeline_runners' market-sampling logic and call the inference layer
directly against the frozen instance table. Prompts come from prompt_factory unchanged
(decision D6) -- the instructions stay English, only the comment text is swapped for a
translation. Nothing in stakebench_code_NeurIPS26/ is modified; profiles and prompt builders
are imported and used as they are.

Arms per model per task:
  en                     original English text, the control arm
  <lang>                 translated text, the manipulation
  <lang>_rt              back-translated text, separates translation loss from surface language

Operational requirements:
  - resumable: one CSV per (model, arm, task); completed comment_ids are skipped on restart
  - budget: every response's OpenRouter cost is recorded and the run aborts at --max-spend
  - off-label outputs are preserved verbatim in raw_response and counted, never silently
    coerced into a valid label (RUNBOOK 5.6)

Usage:
  python rebuttal/analysis/e9/e9d_run_arms.py --limit 30 --tag _smoke \
      --models openrouter:gemini-2.5-flash,openrouter:claude-haiku-4.5 --max-spend 2.0

Outputs: rebuttal/results/e9/predictions/e9d_predictions_<model>_<arm>_<task>.csv
"""
import os, re, sys, json, time, argparse, threading, dataclasses
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "stakebench_code_NeurIPS26"))

import config
from model_profiles import resolve_profile
from prompt_factory import build_task0_prompt, build_task1_prompt
from parsing import normalize_label

# The paper's open-weight models ran locally under transformers. This machine has an 8 GB
# GPU and cannot host qwen3-32b, so they are served through OpenRouter instead.
#
# Only the served endpoint changes. The prompt profile stays the paper's own entry, because
# resolve_profile("openrouter:qwen3-8b") would detect family "openrouter" and silently fall
# back to the gpt-4o default -- different system_style, different max_new_tokens, and a
# generic format suffix instead of the qwen3/llama3 one. That would change the prompt, which
# RUNBOOK 1.4 forbids. So: take the paper's profile, swap hf_model_id only.
OPENROUTER_SERVING = {
    "qwen3:8b":    "qwen/qwen3-8b",
    "qwen3:32b":   "qwen/qwen3-32b",
    "llama3.2:3b": "meta-llama/llama-3.2-3b-instruct",
}


def profile_for(model_alias):
    """Paper's prompt profile, pointed at an OpenRouter endpoint when needed."""
    prof = resolve_profile(model_alias)
    served = OPENROUTER_SERVING.get(model_alias)
    if served:
        prof = dataclasses.replace(prof, hf_model_id=served)
    return prof

OUT = os.path.join(ROOT, "rebuttal", "results", "e9")
PRED = os.path.join(OUT, "predictions")
os.makedirs(PRED, exist_ok=True)

BASE = "https://openrouter.ai/api/v1"
CEILING = 4096          # matches llm_adapter's MAX_TOKENS_CEILING
LANGS = ["spa_Latn", "ron_Latn", "swh_Latn"]

_spend_lock = threading.Lock()
_spend = 0.0


class BudgetExceeded(RuntimeError):
    pass


def split_prompt(p):
    return (p.split("[CB_SYSTEM]", 1)[1].split("[/CB_SYSTEM]", 1)[0].strip(),
            p.split("[CB_USER]", 1)[1].split("[/CB_USER]", 1)[0].strip())


def call_openrouter(profile, system_text, user_text, key, max_spend):
    """One instance. Mirrors llm_adapter._query_openrouter: retries, token auto-scaling."""
    global _spend
    msgs = ([{"role": "system", "content": system_text}] if profile.use_system_prompt else [])
    msgs.append({"role": "user",
                 "content": user_text + ("\n/no_think" if profile.add_no_think else "")})
    headers = {"Authorization": f"Bearer {key}", "HTTP-Referer": "https://bristol.ac.uk",
               "X-Title": "CommitBench", "Content-Type": "application/json"}
    max_tokens = profile.max_new_tokens
    cost = 0.0
    while True:
        body = None
        for attempt in range(5):
            with _spend_lock:
                if _spend >= max_spend:
                    raise BudgetExceeded(f"spend {_spend:.4f} >= cap {max_spend}")
            try:
                r = requests.post(f"{BASE}/chat/completions", headers=headers, timeout=180,
                                  json={"model": profile.hf_model_id, "messages": msgs,
                                        "temperature": profile.temperature,
                                        "max_tokens": max_tokens})
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(2.0 * (2 ** attempt)); continue
                r.raise_for_status()
                body = r.json()
                break
            except BudgetExceeded:
                raise
            except Exception:
                if attempt == 4:
                    return "[REQUEST_FAILED]", cost, max_tokens
                time.sleep(2.0 * (2 ** attempt))
        if body is None:
            return "[REQUEST_FAILED]", cost, max_tokens

        usage = body.get("usage") or {}
        c = float(usage.get("cost") or 0.0)
        cost += c
        with _spend_lock:
            _spend += c

        choices = body.get("choices") or []
        if not choices:
            return "[NULL_CONTENT]", cost, max_tokens
        ch = choices[0]
        content = (ch.get("message") or {}).get("content")
        if content is not None and content.strip():
            return content, cost, max_tokens
        # reasoning models can burn the whole budget before emitting a label
        if ch.get("finish_reason") == "length" and max_tokens < CEILING:
            max_tokens = min(max_tokens * 4, CEILING)
            continue
        return "[NULL_CONTENT]", cost, max_tokens


def arm_text(row, arm, trans_lookup):
    if arm == "en":
        return row.text_en
    lang = arm[:-3] if arm.endswith("_rt") else arm
    rec = trans_lookup.get((row.comment_id, row.task, lang))
    if rec is None:
        return None
    return rec[1] if arm.endswith("_rt") else rec[0]


def run_cell(model_alias, arm, task, inst, trans_lookup, key, max_spend, workers):
    profile = profile_for(model_alias)
    safe_model = re.sub(r"[^A-Za-z0-9]+", "-", model_alias).strip("-")
    path = os.path.join(PRED, f"e9d_predictions_{safe_model}_{arm}_{task}.csv")

    done = set()
    if os.path.exists(path):
        prev = pd.read_csv(path)
        done = set(prev.comment_id.astype(str))
    todo = inst[~inst.comment_id.astype(str).isin(done)]
    if todo.empty:
        print(f"  [{model_alias} | {arm} | {task}] already complete ({len(done)})")
        return path, 0.0

    task_id = "task0" if task == "G1" else "task1"
    rows, lock = [], threading.Lock()
    t0 = time.time()

    def work(row):
        text = arm_text(row, arm, trans_lookup)
        if text is None:
            return None
        if task == "G1":
            prompt = build_task0_prompt(profile, str(row.question), str(text), zeroshot=False)
        else:
            prompt = build_task1_prompt(profile, str(row.question), str(text), topic=row.topic)
        system_text, user_text = split_prompt(prompt)
        raw, cost, final_mt = call_openrouter(profile, system_text, user_text, key, max_spend)
        pred = normalize_label(raw, task_id)
        rec = dict(market_id=row.market_id, comment_id=row.comment_id, arm=arm,
                   lang=("" if arm == "en" else arm.replace("_rt", "")),
                   question=row.question, text=text, gt=row["gt"], pred=pred,
                   correct=(str(pred).upper() == str(row["gt"]).upper()),
                   has_position=row.has_position, position_size=row.position_size,
                   raw_response=raw, model=model_alias, task=task,
                   platform=row.platform, topic=row.topic,
                   cost_usd=round(cost, 6), final_max_tokens=final_mt)
        with lock:
            rows.append(rec)
            n = len(rows)
            if n % 25 == 0 or n == len(todo):
                print(f"  [{model_alias} | {arm} | {task}] {n}/{len(todo)} "
                      f"spend=${_spend:.3f} {time.time()-t0:.0f}s", flush=True)
        return rec

    aborted = False
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(work, [r for _, r in todo.iterrows()]))
    except BudgetExceeded as e:
        aborted = True
        print(f"  !! BUDGET STOP: {e}")

    if rows:
        df = pd.DataFrame(rows)
        if os.path.exists(path):
            df = pd.concat([pd.read_csv(path), df], ignore_index=True)
        df.to_csv(path, index=False, encoding="utf-8")
    if aborted:
        raise BudgetExceeded("budget cap reached")
    return path, sum(r["cost_usd"] for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma-separated model aliases")
    ap.add_argument("--tasks", default="G1,G2")
    ap.add_argument("--limit", type=int, default=0, help="first N instances per task")
    ap.add_argument("--translations", default="e9b_translations.csv")
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-spend", type=float, default=15.0, help="hard USD cap for this run")
    ap.add_argument("--workers", type=int, default=config.OPENROUTER_CONCURRENCY)
    ap.add_argument("--arms", default="", help="override arm list, comma separated")
    args = ap.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY not set")

    inst = pd.read_csv(os.path.join(OUT, "e9b_source_instances.csv"))
    if args.limit:
        inst = inst.groupby("task", group_keys=False).head(args.limit).reset_index(drop=True)
    tr = pd.read_csv(os.path.join(OUT, args.translations))
    trans_lookup = {(r.comment_id, r.task, r.lang): (r.text_translated, r.text_backtranslated)
                    for r in tr.itertuples()}

    arms = args.arms.split(",") if args.arms else ["en"] + LANGS + [l + "_rt" for l in LANGS]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    # Fail fast on missing translations. arm_text returns None when a translation is absent
    # and the instance is then skipped, which would silently produce a short arm and break
    # the complete-pairing requirement. Better to refuse to start than to discover it in the
    # pairing report after paying for the calls.
    missing = []
    for task in tasks:
        for cid in inst[inst.task == task].comment_id:
            for lang in {a.replace("_rt", "") for a in arms if a != "en"}:
                if (cid, task, lang) not in trans_lookup:
                    missing.append((task, cid, lang))
    if missing:
        langs_missing = sorted({m[2] for m in missing})
        raise SystemExit(
            f"{len(missing)} missing translations across {langs_missing}.\n"
            f"e.g. {missing[:3]}\n"
            f"Run e9a_translate.py over the full frozen instance list before this.")

    planned = sum(len(inst[inst.task == t]) for t in tasks) * len(arms) * len(models)
    print(f"models={models}\narms={arms}\ntasks={tasks}\n"
          f"instances/task={ {t: int((inst.task==t).sum()) for t in tasks} }\n"
          f"planned calls={planned}  spend cap=${args.max_spend}\n")

    manifest = []
    try:
        for model in models:
            for task in tasks:
                sub = inst[inst.task == task]
                if sub.empty:
                    continue
                for arm in arms:
                    p, c = run_cell(model, arm, task, sub, trans_lookup, key,
                                    args.max_spend, args.workers)
                    manifest.append(dict(model=model, arm=arm, task=task,
                                         path=os.path.basename(p), cost=round(c, 4)))
    except BudgetExceeded:
        print("\nABORTED ON BUDGET CAP -- partial results are on disk and resumable.")

    with open(os.path.join(OUT, f"e9d_run_manifest{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(dict(models=models, arms=arms, tasks=tasks, limit=args.limit,
                       total_spend_usd=round(_spend, 4), cells=manifest), f, indent=2)
    print(f"\nTOTAL SPEND this run: ${_spend:.4f}")


if __name__ == "__main__":
    main()
