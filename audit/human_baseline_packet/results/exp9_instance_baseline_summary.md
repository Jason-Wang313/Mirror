# Exp9 Instance-Level Abstention Baseline Comparison

- Run ID: `v20_bulletproof_hardening_v3_instance`
- Completed models: 16
- Skipped models: 0

## Macro Summary (Across Completed Models)

| Strategy | Mean Weak CFR | Mean Weak CFR Reduction vs No Routing | Mean Autonomy | Mean Escalation | Mean Overall Failure | N Models |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_routing` | 70.0% | 0.0% | 100.0% | 0.0% | 68.5% | 16 |
| `mirror_domain_routing` | 0.0% | 100.0% | 51.4% | 48.6% | 34.6% | 16 |
| `confidence_threshold_budget_matched` | 31.4% | 56.0% | 51.4% | 48.6% | 35.5% | 16 |
| `self_consistency_budget_matched` | 31.4% | 56.1% | 51.4% | 48.6% | 34.8% | 16 |
| `conformal_style` | 3.7% | 87.1% | 14.7% | 85.3% | 3.3% | 16 |

## Notes

- Frame: Condition 1, Paradigm 3, `api_success=true` rows only.
- Weak-domain policy: `median_or_bottom_k` (fallback `k=2`) from merged Exp1 natural accuracy.
- `mirror_domain_routing` escalates all weak-domain components.
- `confidence_threshold_budget_matched` and `self_consistency_budget_matched` match MIRROR domain-routing escalation budget per model.
- `conformal_style` is a split-calibrated thresholding baseline over the confidence-uncertainty proxy.
