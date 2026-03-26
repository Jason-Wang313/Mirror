# Exp9 Instance-Level Abstention Baseline Comparison

- Run ID: `v20_final_hardening_20260326A_instance_c1c2_all_paradigms`
- Completed models: 16
- Skipped models: 0

## Macro Summary (Across Completed Models)

| Strategy | Mean Weak CFR | Mean Weak CFR Reduction vs No Routing | Mean Autonomy | Mean Escalation | Mean Overall Failure | N Models |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_routing` | 64.4% | 0.0% | 100.0% | 0.0% | 61.6% | 16 |
| `mirror_domain_routing` | 0.0% | 100.0% | 51.7% | 48.3% | 30.5% | 16 |
| `confidence_threshold_budget_matched` | 31.6% | 50.7% | 51.7% | 48.3% | 31.0% | 16 |
| `self_consistency_budget_matched` | 31.0% | 52.1% | 51.7% | 48.3% | 31.1% | 16 |
| `calibrated_confidence_budget_matched` | 30.5% | 52.0% | 51.7% | 48.3% | 29.7% | 16 |
| `conformal_style_budget_matched` | 1.6% | 93.8% | 6.2% | 93.8% | 1.2% | 16 |

## Notes

- Frame label: `Condition 1-2 / Paradigms 1-3`
- Conditions: `[1, 2]` | Paradigms: `[1, 2, 3]` | `api_success=true` rows only.
- Weak-domain policy: `median_or_bottom_k` (fallback `k=2`) from merged Exp1 natural accuracy.
- `mirror_domain_routing` escalates all weak-domain components.
- Strategy set: `robust_v2` | Calibration: `transformed_platt` | Matched-budget mode: `domain_budget`.
- `confidence_threshold_budget_matched` and `self_consistency_budget_matched` match MIRROR domain-routing escalation budget per model.
- `calibrated_confidence_budget_matched` uses transformed-Platt calibrated confidence uncertainty when enabled.
- `conformal_style_budget_matched` is a split-calibrated thresholding baseline over confidence uncertainty.
- Conformal target grid: `[0.1, 0.15, 0.2, 0.25, 0.3]`.

## Robust Frontier Summary (Macro)

| Strategy | Autonomy Target | Mean Weak CFR | Mean Escalation | N Models |
| --- | ---: | ---: | ---: | ---: |
| `confidence_threshold_frontier` | 30.0% | 17.4% | 70.0% | 16 |
| `confidence_threshold_frontier` | 40.0% | 23.6% | 60.0% | 16 |
| `confidence_threshold_frontier` | 50.0% | 30.5% | 50.0% | 16 |
| `confidence_threshold_frontier` | 60.0% | 36.8% | 40.0% | 16 |
| `confidence_threshold_frontier` | 70.0% | 42.6% | 30.0% | 16 |
| `confidence_threshold_frontier` | 80.0% | 48.9% | 20.0% | 16 |
| `confidence_threshold_frontier` | 90.0% | 55.9% | 10.0% | 16 |
| `self_consistency_frontier` | 30.0% | 17.6% | 70.0% | 16 |
| `self_consistency_frontier` | 40.0% | 23.6% | 60.0% | 16 |
| `self_consistency_frontier` | 50.0% | 29.3% | 50.0% | 16 |
| `self_consistency_frontier` | 60.0% | 36.7% | 40.0% | 16 |
| `self_consistency_frontier` | 70.0% | 42.5% | 30.0% | 16 |
| `self_consistency_frontier` | 80.0% | 48.7% | 20.0% | 16 |
| `self_consistency_frontier` | 90.0% | 55.8% | 10.0% | 16 |
| `calibrated_confidence_frontier` | 30.0% | 16.8% | 70.0% | 16 |
| `calibrated_confidence_frontier` | 40.0% | 22.6% | 60.0% | 16 |
| `calibrated_confidence_frontier` | 50.0% | 29.3% | 50.0% | 16 |
| `calibrated_confidence_frontier` | 60.0% | 35.6% | 40.0% | 16 |
| `calibrated_confidence_frontier` | 70.0% | 41.7% | 30.0% | 16 |
| `calibrated_confidence_frontier` | 80.0% | 48.7% | 20.0% | 16 |
| `calibrated_confidence_frontier` | 90.0% | 56.1% | 10.0% | 16 |

## Conformal Target-Error Grid (Macro)

| Target Error | Mean Weak CFR | Mean Autonomy | Mean Escalation | N Models |
| ---: | ---: | ---: | ---: | ---: |
| 0.10 | 0.0% | 0.0% | 100.0% | 16 |
| 0.15 | 0.0% | 0.0% | 100.0% | 16 |
| 0.20 | 1.2% | 4.5% | 95.5% | 16 |
| 0.25 | 1.6% | 6.2% | 93.8% | 16 |
| 0.30 | 2.2% | 8.3% | 91.7% | 16 |