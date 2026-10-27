"""
Write the E9 pre-registration and preflight records (RUNBOOK 1.6 / 7).

Freezing this before the production run is what makes "we did not pick languages after
seeing the results" checkable. It records the git hash, the hash of the frozen instance
list, the language set, the model set, the translation model and its generation parameters,
the primary metric and the seed.

Run this BEFORE the production arms. Re-running after results exist will refuse to
overwrite an existing pre-registration unless --force is given.
"""
import os, sys, json, glob, hashlib, platform, subprocess, argparse, datetime

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "stakebench_code_NeurIPS26"))
import config

OUT = os.path.join(ROOT, "rebuttal", "results", "e9")

LANGS = ["spa_Latn", "ron_Latn", "swh_Latn"]

# gpt-5.5 is excluded from BOTH tasks. It is a reasoning model whose measured cost is
# $0.0038 (G1) / $0.0126 (G2) per instance -- roughly 90% of the total budget for one of six
# models -- and the remaining OpenRouter balance cannot cover it. Excluding it from G1 as
# well keeps the model set identical across tasks, so the per-model-then-average estimand is
# taken over the same five models everywhere instead of a ragged set.
#
# Cost: the paper's flagship closed OpenAI model is absent. The closed-model evidence rests
# on claude-haiku-4.5 and gemini-2.5-flash. This must be stated in the write-up; it is a
# budget constraint, not a finding about gpt-5.5.
MODELS_G1 = ["qwen3:8b", "qwen3:32b", "llama3.2:3b",
             "openrouter:gemini-2.5-flash", "openrouter:claude-haiku-4.5"]
MODELS_G2 = list(MODELS_G1)
EXCLUDED = {"openrouter:gpt-5.5": "budget: ~90% of remaining balance for one model"}


def sh(cmd):
    try:
        return subprocess.check_output(cmd, cwd=ROOT, shell=True,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    path = os.path.join(OUT, "e9_preregistration.json")
    if os.path.exists(path) and not args.force:
        raise SystemExit(f"{path} already exists; use --force only if you intend to re-freeze")

    with open(os.path.join(OUT, "e9b_freeze_report.json"), encoding="utf-8") as f:
        freeze = json.load(f)
    tparams = {}
    tp = os.path.join(OUT, "e9a_translation_params.json")
    if os.path.exists(tp):
        with open(tp, encoding="utf-8") as f:
            tparams = json.load(f)

    prereg = dict(
        experiment="E9 multilingual translation manipulation (v2)",
        created_utc=datetime.datetime.utcnow().isoformat() + "Z",
        git_hash=sh("git rev-parse HEAD"),
        git_dirty=bool(sh("git status --porcelain")),
        instance_list_sha256=freeze["instance_list_sha256"],
        n_instances=freeze["n_total"],
        instances_per_task={t["task"]: t["sampled"] for t in freeze["tasks"]},
        languages=LANGS,
        arms=["en"] + LANGS + [l + "_rt" for l in LANGS],
        models_G1=MODELS_G1, models_G2=MODELS_G2,
        translation=tparams,
        primary_metric="balanced accuracy vs the 0.5 floor",
        estimand="per-model delta (language arm - en arm), then mean over models",
        inference=dict(seed=config.SEED, temperature=config.TEMPERATURE,
                       concurrency=config.OPENROUTER_CONCURRENCY,
                       prompts="prompt_factory.build_task0_prompt / build_task1_prompt, "
                               "unmodified; instructions stay English"),
        tests=dict(paired="exact McNemar per model x language x task",
                   interval="paired bootstrap clustered on market_id, 2000 draws",
                   multiplicity="BH-FDR over language x model x task"),
        secondary_metrics=["off-label output rate", "neutral rate (G2)"],
        # 5 models x 7 arms x 500 instances = 17,500 calls. The cap was 11,000 under the
        # earlier six-model/partial plan; leaving it there would have the run trip its own
        # stopping rule. Spend cap stays at $18 against a measured $1.6 projection and a
        # $22.06 balance, so budget is no longer the binding constraint once gpt-5.5 is out.
        stopping_rules=dict(openrouter_call_cap=18000, spend_cap_usd=18.0,
                            failure_rate=0.05, off_label_rate=0.30,
                            numeral_preservation_floor=0.90),
        boundaries=[
            "machine-translated text is not authentic non-English prediction-market discourse",
            "G4 is not evaluated",
            "the quality filter and commitment lexicon are not validated in these languages",
        ],
    )

    preflight = dict(
        created_utc=prereg["created_utc"],
        python=sys.version.split()[0], platform=platform.platform(),
        packages={m: __import__(m).__version__ for m in
                  ("pandas", "numpy", "sklearn", "scipy", "transformers", "torch")},
        gpu=sh('python -c "import torch;print(torch.cuda.get_device_name(0))"'),
        nllb_cached=bool(glob.glob(os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface", "hub",
            "models--facebook--nllb-200-distilled-1.3B"))),
        openrouter_key_present=bool(os.environ.get("OPENROUTER_API_KEY")),
        note=("open-weight models are served through OpenRouter in this run because the local "
              "GPU is 8 GB; the paper ran them locally under transformers. Prompt profiles are "
              "the paper's own entries with only hf_model_id redirected."),
    )

    for p, obj in ((path, prereg), (os.path.join(OUT, "e9_preflight.json"), preflight)):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        print(f"wrote {p}")
    print(json.dumps(prereg, indent=2, ensure_ascii=False)[:1200])


if __name__ == "__main__":
    main()
