# Exp9 Instance-Level Abstention Baseline Comparison

- Run ID: `v20_final_hardening_20260326A_instance_legacy_c1_p3`
- Completed models: 16
- Skipped models: 0

## Macro Summary (Across Completed Models)

| Strategy | Mean Weak CFR | Mean Weak CFR Reduction vs No Routing | Mean Autonomy | Mean Escalation | Mean Overall Failure | N Models |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_routing` | 70.0% | 0.0% | 100.0% | 0.0% | 68.5% | 16 |
| `mirror_domain_routing` | 0.0% | 100.0% | 51.4% | 48.6% | 34.6% | 16 |
| `confidence_threshold_budget_matched` | 31.4% | 56.0% | 51.4% | 48.6% | 35.5% | 16 |
| `self_consistency_budget_matched` | 31.4% | 56.1% | 51.4% | 48.6% | 34.8% | 16 |
| `calibrated_confidence_budget_matched` | 37.1% | 48.6% | 51.4% | 48.6% | 34.5% | 16 |
| `conformal_style_budget_matched` | 3.7% | 87.1% | 14.7% | 85.3% | 3.3% | 16 |

## Notes

- Frame label: `Condition 1 / Paradigm 3 (Legacy)`
- Conditions: `[1]` | Paradigms: `[3]` | `api_success=true` rows only.
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
| `confidence_threshold_frontier` | 30.0% | 18.2% | 70.0% | 16 |
| `confidence_threshold_frontier` | 40.0% | 24.6% | 59.9% | 16 |
| `confidence_threshold_frontier` | 50.0% | 30.6% | 50.0% | 16 |
| `confidence_threshold_frontier` | 60.0% | 37.6% | 40.1% | 16 |
| `confidence_threshold_frontier` | 70.0% | 45.3% | 30.0% | 16 |
| `confidence_threshold_frontier` | 80.0% | 53.3% | 20.0% | 16 |
| `confidence_threshold_frontier` | 90.0% | 61.5% | 9.9% | 16 |
| `self_consistency_frontier` | 30.0% | 16.9% | 70.0% | 16 |
| `self_consistency_frontier` | 40.0% | 23.4% | 59.9% | 16 |
| `self_consistency_frontier` | 50.0% | 30.6% | 50.0% | 16 |
| `self_consistency_frontier` | 60.0% | 37.5% | 40.1% | 16 |
| `self_consistency_frontier` | 70.0% | 45.1% | 30.0% | 16 |
| `self_consistency_frontier` | 80.0% | 52.7% | 20.0% | 16 |
| `self_consistency_frontier` | 90.0% | 60.2% | 9.9% | 16 |
| `calibrated_confidence_frontier` | 30.0% | 22.3% | 70.0% | 16 |
| `calibrated_confidence_frontier` | 40.0% | 29.6% | 59.9% | 16 |
| `calibrated_confidence_frontier` | 50.0% | 36.6% | 50.0% | 16 |
| `calibrated_confidence_frontier` | 60.0% | 42.6% | 40.1% | 16 |
| `calibrated_confidence_frontier` | 70.0% | 49.4% | 30.0% | 16 |
| `calibrated_confidence_frontier` | 80.0% | 56.1% | 20.0% | 16 |
| `calibrated_confidence_frontier` | 90.0% | 62.8% | 9.9% | 16 |

## Conformal Target-Error Grid (Macro)

| Target Error | Mean Weak CFR | Mean Autonomy | Mean Escalation | N Models |
| ---: | ---: | ---: | ---: | ---: |
| 0.10 | 0.1% | 1.6% | 98.4% | 16 |
| 0.15 | 0.1% | 1.7% | 98.3% | 16 |
| 0.20 | 2.0% | 9.1% | 90.9% | 16 |
| 0.25 | 3.7% | 14.7% | 85.3% | 16 |
| 0.30 | 4.6% | 16.5% | 83.5% | 16 |