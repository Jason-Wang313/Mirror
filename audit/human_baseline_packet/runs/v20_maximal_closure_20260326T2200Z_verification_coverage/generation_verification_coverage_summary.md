# Data Generation and Verification Coverage

- Run ID: `v20_maximal_closure_20260326T2200Z_verification_coverage`
- Experiments reported: `8`
- Human-verified rows/items (artifact-counted): `675`
- Static schema-verified tasks (artifact-counted): `0`

## Coverage Table

| Experiment | Primary label source | Human verified (n) | Verification depth | Static schema verified (n) | Notes |
| --- | --- | ---: | --- | ---: | --- |
| Exp1 | Programmatic scoring against fixed answer keys | 339 | 5-rater audit, unanimous=0.938 | 0 | Human verification sourced from multi-rater audit summary. |
| Exp2 | Programmatic / deterministic pipeline outputs | 0 | No dedicated human-audit artifact in this stage | 0 | Bench-level QA remains in reproducibility scripts. |
| Exp3 | Programmatic / deterministic pipeline outputs | 0 | No dedicated human-audit artifact in this stage | 0 | Pair-balanced diagnostics and bootstrap checks are automated. |
| Exp4 | Hybrid (programmatic + targeted human annotation packet) | 25 | Targeted annotation packet | 0 | Counts sourced from human_annotations_report.json. |
| Exp5 | Programmatic / deterministic pipeline outputs | 0 | No dedicated human-audit artifact in this stage | 0 | Adversarial-condition checks are scripted. |
| Exp6 | Programmatic + curated adversarial metadata | 0 | See manuscript appendix note for full-verification run details | 0 | This table reports machine-readable artifacts only. |
| Exp8 | Programmatic / deterministic pipeline outputs | 0 | No dedicated human-audit artifact in this stage | 0 | Scaling analysis is computed from benchmark outputs. |
| Exp9 | Programmatic routing outcomes + static task verification | 311 | 5-rater audit core=261, targeted packet=50, unanimous=0.747 | 0 | Static domain-component checks: 0/0 passed. |

## Notes

- This table is artifact-driven: it summarizes machine-readable evidence currently present in-repo.
- Human vs programmatic coverage can be expanded by adding experiment-specific verification artifacts.
