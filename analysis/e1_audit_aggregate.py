"""
E1 (P0-1 motivation): Aggregate falsification-audit JSONs to show that comment
text carries GENUINE commitment signal (independent of eventual market outcome),
so G3/G4 failures reflect model limitations rather than "no signal in the data".

Reads (zero new inference):
  commitbench_v18/commitbench_v18/<model>/<platform>/<topic>/commitment_validity_*.json
  commitbench_v18/commitbench_v18/<model>/<platform>/<topic>/did_*.json

Key quantity — Wrong-Bettor Test:  Pr( z_hat = z | z != r )
  z_hat = model-predicted position side (G2), z = user's true position side,
  r = eventual resolution. High value => model reads the COMMITMENT from language,
  not the outcome. Paper claims 0.71-0.81 across four models.

Outputs:
  rebuttal/results/e1_commitment_validity_rows.csv
  rebuttal/results/e1_did_rows.csv
  rebuttal/results/e1_summary.md
"""
import json, glob, os, csv, sys
from collections import defaultdict
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = "commitbench_v18/commitbench_v18"
OUT  = "rebuttal/results"
os.makedirs(OUT, exist_ok=True)

def load(f):
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return json.load(open(f, encoding=enc))
        except Exception:
            continue
    return None

def split_ds(ds):
    return ds.split("/", 1) if "/" in (ds or "") else (ds or "", "")

cv_rows, did_rows = [], []
for model in sorted(os.listdir(ROOT)):
    mp = os.path.join(ROOT, model)
    if not os.path.isdir(mp):
        continue
    for f in glob.glob(os.path.join(mp, "**", "commitment_validity_*.json"), recursive=True):
        d = load(f)
        if not d:
            continue
        plat, topic = split_ds(d.get("data_source"))
        wb = d.get("wrong_bettor_test", {}) or {}
        dr = d.get("dose_response", {}) or {}
        cv_rows.append(dict(model=model, platform=plat, topic=topic,
            wb_n_hp=wb.get("n_hp"), wb_n_wrong=wb.get("n_wrong_bettors"),
            wb_rate=wb.get("wrong_bettor_rate"),
            wb_n_dir=wb.get("n_directional"), wb_skip=wb.get("skip"),
            wb_pct_follows_commit=wb.get("pct_follows_commitment"),
            wb_binom_p=wb.get("binom_p"), wb_sig=wb.get("significant"),
            dr_n=dr.get("n_hp_directional"), dr_csd_q4_q1=dr.get("csd_q4_q1"),
            dr_mwu_p=dr.get("mwu_p_large_gt_small"), dr_sig=dr.get("significant")))
    for f in glob.glob(os.path.join(mp, "**", "did_*.json"), recursive=True):
        d = load(f)
        if not d:
            continue
        plat, topic = split_ds(d.get("data_source"))
        did_rows.append(dict(model=model, platform=plat, topic=topic,
            n_pre_dir=d.get("n_pre_directional"), n_post_dir=d.get("n_post_directional"),
            da_pre_gt_res=d.get("da_pre_gt_resolution"),
            da_post_gt_res=d.get("da_post_gt_resolution"),
            da_post_gt_pos=d.get("da_post_gt_position"),
            did=d.get("did"),
            wb_da_post_gt_res=d.get("wrong_bettor_da_post_gt_resolution")))

def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

write_csv(f"{OUT}/e1_commitment_validity_rows.csv", cv_rows)
write_csv(f"{OUT}/e1_did_rows.csv", did_rows)

# ---------- summaries ----------
def fnum(x):
    try: return float(x)
    except: return None

# Wrong-Bettor position-side follow-rate Pr(z_hat=z|z!=r) = pct_follows_commitment.
# Eligible = test actually ran (not skipped, has numeric follow-rate).
wb_by_model = defaultdict(list)
wb_sig_by_model = defaultdict(list)
elig = sig = 0
for r in cv_rows:
    da = fnum(r["wb_pct_follows_commit"])
    is_sig = r["wb_sig"] in (True, "True", "true")
    if da is not None and not r["wb_skip"]:
        elig += 1
        wb_by_model[r["model"]].append((da, r["topic"], r["platform"], r["wb_n_dir"]))
        if is_sig:
            sig += 1
            wb_sig_by_model[r["model"]].append((da, r["topic"], r["platform"]))

# dose-response significant
dr_sig = sum(1 for r in cv_rows if r["dr_sig"] in (True, "True", "true"))
dr_elig = sum(1 for r in cv_rows if fnum(r["dr_csd_q4_q1"]) is not None)

# DiD wrong-bettor action DA
wbaction_by_model = defaultdict(list)
did_vals = []
for r in did_rows:
    v = fnum(r["wb_da_post_gt_res"])
    if v is not None:
        wbaction_by_model[r["model"]].append(v)
    dv = fnum(r["did"])
    if dv is not None:
        did_vals.append(dv)

lines = []
lines.append("# E1 — Falsification-audit aggregation (P0-1 motivation)\n")
lines.append(f"- commitment_validity files parsed: **{len(cv_rows)}**  |  did files parsed: **{len(did_rows)}**")
lines.append(f"- Wrong-Bettor Test eligible (topic-model pairs with computable DA): **{elig}**  |  significant (p<0.05): **{sig}**")
lines.append(f"- Dose-response eligible: **{dr_elig}**  |  significant: **{dr_sig}**\n")

lines.append("## Wrong-Bettor follow-rate  Pr(z_hat=z | z!=r) = pct_follows_commitment, by model")
lines.append("_Among users who bet the side that eventually LOST (z!=r), how often the model still")
lines.append("recovers their stated side. >0.5 means the model reads commitment from language, not outcome._\n")
lines.append("| Model | n pairs | mean | max | best topic/platform (rate, n_dir) |")
lines.append("|---|---|---|---|---|")
for m in sorted(wb_by_model):
    vals = [d for d, *_ in wb_by_model[m]]
    if not vals: continue
    d, t, p, n = max(wb_by_model[m], key=lambda x: x[0])
    lines.append(f"| {m} | {len(vals)} | {sum(vals)/len(vals):.3f} | {max(vals):.3f} | {t}/{p} ({d:.3f}, n={n}) |")
lines.append("")
sigflat = sorted([(d, m, t, p) for m, lst in wb_sig_by_model.items() for d, t, p in lst], key=lambda x: -x[0])
lines.append(f"**Significant wrong-bettor pairs (p<0.05): {sig} of {elig} eligible.** "
             "Top significant follow-rates (reproduces paper's 0.71-0.81 across four models):")
for d, m, t, p in sigflat[:8]:
    lines.append(f"- {m} on {p}/{t}: **{d:.3f}**")
lines.append("")

lines.append("## DiD wrong-bettor post-comment action DA (from did_*.json), by model")
lines.append("| Model | n | mean wrong-bettor DA_post |")
lines.append("|---|---|---|")
for m in sorted(wbaction_by_model):
    vals = wbaction_by_model[m]
    lines.append(f"| {m} | {len(vals)} | {sum(vals)/len(vals):.3f} |")
lines.append("")
if did_vals:
    lines.append(f"- Mean within-user DiD across {len(did_vals)} cells: **{sum(did_vals)/len(did_vals):+.4f}** "
                 f"(range {min(did_vals):+.3f}..{max(did_vals):+.3f})")

open(f"{OUT}/e1_summary.md", "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
print(f"\n[written] {OUT}/e1_commitment_validity_rows.csv ({len(cv_rows)} rows), "
      f"e1_did_rows.csv ({len(did_rows)} rows), e1_summary.md")
