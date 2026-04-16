# Exp9 Weak-Domain Policy Frontier Summary

- Run ID: `v20_policy_frontier_20260326T1632Z`
- Conditions: `[1]` | Paradigms: `[1, 2, 3]`
- Models covered: `16`

## Macro Frontier

| Family | Slice | Weak CFR | Escalation | Autonomy | System Success | Cost Proxy | Latency Proxy (ms) | N |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| absolute_threshold | all | 0.000 | 0.563 | 0.437 | 0.743 | 6.9315 | 15772.9 | 96 |
| absolute_threshold | matched_escalation | 0.000 | 0.547 | 0.453 | 0.737 | 6.8889 | 15623.4 | 16 |
| absolute_threshold | matched_autonomy | 0.000 | 0.547 | 0.453 | 0.737 | 6.8889 | 15623.4 | 16 |
| bottom_k | all | 0.000 | 0.309 | 0.691 | 0.584 | 6.2434 | 13357.9 | 64 |
| bottom_k | matched_escalation | 0.000 | 0.488 | 0.512 | 0.696 | 6.7286 | 15060.7 | 16 |
| bottom_k | matched_autonomy | 0.000 | 0.488 | 0.512 | 0.696 | 6.7286 | 15060.7 | 16 |
| median_or_bottom_k | all | 0.000 | 0.488 | 0.512 | 0.696 | 6.7286 | 15060.7 | 16 |
| median_or_bottom_k | matched_escalation | 0.000 | 0.488 | 0.512 | 0.696 | 6.7286 | 15060.7 | 16 |
| median_or_bottom_k | matched_autonomy | 0.000 | 0.488 | 0.512 | 0.696 | 6.7286 | 15060.7 | 16 |
| quantile_threshold | all | 0.000 | 0.350 | 0.650 | 0.611 | 6.3545 | 13748.0 | 80 |
| quantile_threshold | matched_escalation | 0.000 | 0.488 | 0.512 | 0.696 | 6.7286 | 15060.7 | 16 |
| quantile_threshold | matched_autonomy | 0.000 | 0.488 | 0.512 | 0.696 | 6.7286 | 15060.7 | 16 |

## Notes

- `median_or_bottom_k` is treated as the canonical reference policy.
- `matched_escalation` and `matched_autonomy` slices choose nearest policy settings per model.
- Cost/latency columns are packet-derived operational proxies, not claim-critical primary evidence.
