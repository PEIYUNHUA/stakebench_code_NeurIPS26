"""
E9f: agreement between this run's `en` control arm and the published v18 logs.

RUNBOOK 4.4: the en arm re-scores instances the paper already scored, so the two should
mostly agree. What the number is for:

  It establishes that this run is COMPARABLE to the published evaluation. It does NOT
  reproduce the paper's numbers and must never be described that way. Two reasons:
    - closed endpoints move; gpt-5.5 / claude-haiku-4.5 / gemini-2.5-flash today are not
      guaranteed to be the checkpoints scored for the paper;
    - the open-weight models ran locally under transformers for the paper and are served
      through OpenRouter here, so decoding stack, quantisation and batching all differ.
  So even a perfect protocol would not give 100% agreement.

RUNBOOK 10.2: agreement below 0.85 is a report-back condition, because it suggests the
endpoint or the environment changed in a way that needs investigating before the manipulation
result can be interpreted.

Outputs: e9_control_consistency.csv
"""
import os, re, sys, glob, argparse
import pandas as pd

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "rebuttal", "results", "e9")
PRED = os.path.join(OUT, "predictions")
LOGS = os.path.join(ROOT, "commitbench_v18", "commitbench_v18")

# this run's alias -> published log directory
LOG_DIR = {
    "qwen3:8b": "Qwen3-8B",
    "qwen3:32b": "Qwen3-32B",
    "llama3.2:3b": "Llama3.2-3B",
    "openrouter:gpt-5.5": "Openrouter-GPT-5.5",
    "openrouter:claude-haiku-4.5": "Openrouter-CLAUDE-HAIKU-4.5",
    "openrouter:gemini-2.5-flash": "Openrouter-GEMINI-2.5-FLASH",
}
PATTERN = {"G1": "*_task0_0shot.csv", "G2": "*_task1.csv"}


def published(model_alias, task, keys):
    """Published predictions for the given instance keys, keyed the same way as the freeze."""
    import hashlib
    frames = []
    for f in glob.glob(os.path.join(LOGS, LOG_DIR[model_alias], "*", "*", PATTERN[task])):
        frames.append(pd.read_csv(f))
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    d["text"] = d["text"].astype(str)
    d["comment_id"] = [
        hashlib.sha1(f"{m}||{c}||{t}".encode("utf-8", "replace")).hexdigest()[:16]
        for m, c, t in zip(d.market_id, d.created_at, d.text)
    ]
    d = d[d.comment_id.isin(keys)].drop_duplicates("comment_id")
    # task0 logs store the label as True/False; the frozen instances and this run's
    # predictions use YES/NO. Without this mapping every G1 cell scores 0.0 agreement.
    d["pred_published"] = (d["pred"].astype(str).str.upper()
                           .replace({"TRUE": "YES", "FALSE": "NO"}))
    return d.set_index("comment_id")[["pred_published"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    files = glob.glob(os.path.join(PRED, "e9d_predictions_*_en_*.csv"))
    if not files:
        raise SystemExit("no en-arm prediction files found")
    cur = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    cur = cur[cur.arm == "en"]
    cur["pred"] = cur["pred"].astype(str).str.upper()

    rows = []
    for (model, task), g in cur.groupby(["model", "task"]):
        if model not in LOG_DIR:
            continue
        pub = published(model, task, set(g.comment_id))
        if pub is None or pub.empty:
            continue
        j = g.set_index("comment_id").join(pub, how="inner")
        if j.empty:
            continue
        agree = float((j["pred"] == j["pred_published"]).mean())
        rows.append(dict(model=model, task=task, n_matched=len(j),
                         agreement=round(agree, 4),
                         acc_this_run=round(float((j["pred"] == j["gt"]).mean()), 4),
                         acc_published=round(float((j["pred_published"] == j["gt"]).mean()), 4),
                         flag=("BELOW 0.85 - investigate before interpreting"
                               if agree < 0.85 else "")))
    out = pd.DataFrame(rows).sort_values(["task", "model"])
    out.to_csv(os.path.join(OUT, f"e9_control_consistency{args.tag}.csv"), index=False)
    print(out.to_string(index=False))
    print("\nAgreement measures comparability with the published run, not reproduction of it.")
    print("Open-weight models: local transformers (paper) vs OpenRouter serving (here).")
    print("Closed models: the endpoint may have been updated since the paper was scored.")
    low = out[out.agreement < 0.85]
    if len(low):
        print(f"\nRUNBOOK 10.2 report-back: {len(low)} cell(s) below 0.85 agreement.")


if __name__ == "__main__":
    main()
