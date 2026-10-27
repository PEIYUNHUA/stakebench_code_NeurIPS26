# Analysis scripts

These scripts reproduce the robustness analyses of Section 5.5 and the camera-ready appendices from the
released model outputs. They use paths relative to a working directory that contains
`stakebench_code_NeurIPS26/` (this repository) and `commitbench_v18/commitbench_v18/` (the dataset's
`model_outputs/` folder, copied or symlinked under that name), and write to `rebuttal/results/`
(create it first). Run them from that working directory, e.g.

```bash
mkdir -p rebuttal/results
python stakebench_code_NeurIPS26/analysis/e6_signal_ceiling.py
```

| Script | Paper location | Output (`rebuttal/results/`) |
|---|---|---|
| `e6_signal_ceiling.py`, `e6b_group_splits.py` | Sec. 5.5 lexical recoverability probe | `e6_*.csv`, `e6b_*` |
| `e1_audit_aggregate.py`, `e1b_multiplicity.py` | Sec. 5.5 Wrong-Bettor audit (BH-corrected) | `e1_*`, `e1b_*` |
| `e2_stratification.py`, `e2b_user_retention.py` | Sec. 5.5 activity-skew stress test | `e2_*`, `e2b_*` |
| `e4_g4_probe.py` | Sec. 5.5 G4 window-level probe | `e4_*.csv` incl. the window records |
| `e5_protocol_robustness.py` | Appendix C protocol variants | `e5_*` |
| `e3_examples.py`, `e3b_example_verification.py` | Appendix qualitative examples | `e3_*`, `e3b_*` |
| `e7_language_coverage.py`, `e8_committed_language.py` | dataset card (*Multilingual headroom*); Appendix E marker table | `e7_*`, `e8_*` |
| `e9a_language_census.py`, `e9c_cue_portability.py`, `e9/` | Appendix translation probe | `e9a_*`, `e9/*` |

The result files themselves are not distributed; each script regenerates its outputs (including the G4 window pool `e4_windows_<platform>.csv`) from `model_outputs/` and the benchmark files.

Requirements: the packages in `../requirements.txt` plus `scikit-learn`. `e9/` additionally needs `torch`,
`sacrebleu` and an `OPENROUTER_API_KEY` environment variable for the closed-model arms; its scripts are
numbered in run order (`e9_preregistration.py` → `e9b_freeze_instances.py` → `e9a_translate.py` →
`e9a_fidelity_audit.py` → `e9d_run_arms.py` → `e9e_analyze.py` → `e9f_control_consistency.py` →
`e9g_summary.py`).
