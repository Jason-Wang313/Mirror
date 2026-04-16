# Non-Oracle Utility Sensitivity (Exp9)

- Run ID: `v20_maximal_closure_20260326T2200Z_non_oracle`
- Generated (UTC): 2026-03-26T18:11:48+00:00
- Components analyzed: 98744

## Resolver Accuracy Inputs

- Deployment packet observed resolver accuracy: 0.5018518518518519
- Resolver eval rows: 540
- Cost/latency profile available: True

## Condition-Level Sensitivity (q = resolver correctness)

| q | C1 System Success | C4 System Success | C1 Weak Fail | C4 Weak Fail | C4-C1 Success Delta | C1 Cost | C4 Cost | C1 Latency | C4 Latency |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 0.3725 | 0.4735 | 0.6508 | 0.5313 | 0.1010 | 5.3941 | 5.4092 | 10256.2 | 10435.5 |
| 0.50 | 0.3725 | 0.4745 | 0.6508 | 0.5299 | 0.1020 | 5.3941 | 5.4092 | 10256.2 | 10435.5 |
| 0.60 | 0.3725 | 0.5234 | 0.6508 | 0.4596 | 0.1509 | 5.3941 | 5.4092 | 10256.2 | 10435.5 |
| 0.70 | 0.3725 | 0.5732 | 0.6508 | 0.3879 | 0.2007 | 5.3941 | 5.4092 | 10256.2 | 10435.5 |
| 0.80 | 0.3725 | 0.6231 | 0.6508 | 0.3163 | 0.2506 | 5.3941 | 5.4092 | 10256.2 | 10435.5 |
| 0.90 | 0.3725 | 0.6730 | 0.6508 | 0.2446 | 0.3005 | 5.3941 | 5.4092 | 10256.2 | 10435.5 |
| 1.00 | 0.3725 | 0.7229 | 0.6508 | 0.1729 | 0.3504 | 5.3941 | 5.4092 | 10256.2 | 10435.5 |

## Cost/Latency-Aware Pareto Frontier

| q | Condition | System Success | Autonomy | Escalation | Exp. Cost | Exp. Latency (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | C4 | 0.7229 | 0.5013 | 0.4987 | 5.4092 | 10435.5 |
| 0.50 | C1 | 0.3725 | 1.0000 | 0.0000 | 5.3941 | 10256.2 |
| 0.50 | C1 | 0.3725 | 1.0000 | 0.0000 | 5.3941 | 10256.2 |
| 0.60 | C1 | 0.3725 | 1.0000 | 0.0000 | 5.3941 | 10256.2 |
| 0.70 | C1 | 0.3725 | 1.0000 | 0.0000 | 5.3941 | 10256.2 |
| 0.80 | C1 | 0.3725 | 1.0000 | 0.0000 | 5.3941 | 10256.2 |
| 0.90 | C1 | 0.3725 | 1.0000 | 0.0000 | 5.3941 | 10256.2 |
| 1.00 | C1 | 0.3725 | 1.0000 | 0.0000 | 5.3941 | 10256.2 |

## Primary Takeaway

C4 retains a positive system-success advantage over C1 across resolver-quality sensitivity (q=0.50..1.00), while explicitly trading autonomy for reliability and operational overhead.
