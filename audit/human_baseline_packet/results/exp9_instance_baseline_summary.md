# Exp9 Instance-Level Abstention Baseline Comparison

- Run ID: `v20_hardening_human20_lock_v4_instance`
- Completed models: 11
- Skipped models: 5

## Macro Summary (Across Completed Models)

| Strategy | Mean Weak CFR | Mean Weak CFR Reduction vs No Routing | Mean Autonomy | Mean Escalation | Mean Overall Failure | N Models |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_routing` | 66.7% | 0.0% | 100.0% | 0.0% | 65.6% | 11 |
| `mirror_domain_routing` | 0.0% | 100.0% | 52.8% | 47.2% | 33.8% | 11 |
| `confidence_threshold_budget_matched` | 31.5% | 53.2% | 52.8% | 47.2% | 34.4% | 11 |
| `self_consistency_budget_matched` | 30.8% | 53.6% | 52.8% | 47.2% | 33.6% | 11 |
| `conformal_style` | 2.9% | 89.8% | 12.1% | 87.9% | 3.0% | 11 |

## Notes

- Frame: Condition 1, Paradigm 3, `api_success=true` rows only.
- Weak-domain policy: `median_or_bottom_k` (fallback `k=2`) from merged Exp1 natural accuracy.
- `mirror_domain_routing` escalates all weak-domain components.
- `confidence_threshold_budget_matched` and `self_consistency_budget_matched` match MIRROR domain-routing escalation budget per model.
- `conformal_style` is a split-calibrated thresholding baseline over the confidence-uncertainty proxy.
