# Non-Oracle Utility Sensitivity (Exp9)

- Run ID: `v20_stanford_hardening_20260325T2130Z_non_oracle`
- Generated (UTC): 2026-03-25T22:30:32+00:00
- Components analyzed: 98744

## Resolver Accuracy Inputs

- Deployment packet observed resolver accuracy: 0.5018518518518519
- Resolver eval rows: 540

## Condition-Level Sensitivity (q = resolver correctness)

| q | C1 System Success | C4 System Success | C1 Weak Fail | C4 Weak Fail | C4-C1 Success Delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 0.3725 | 0.4745 | 0.6508 | 0.5299 | 0.1020 |
| 0.70 | 0.3725 | 0.5732 | 0.6508 | 0.3879 | 0.2007 |
| 0.75 | 0.3725 | 0.5982 | 0.6508 | 0.3521 | 0.2257 |
| 0.80 | 0.3725 | 0.6231 | 0.6508 | 0.3163 | 0.2506 |
| 0.85 | 0.3725 | 0.6480 | 0.6508 | 0.2804 | 0.2756 |
| 0.90 | 0.3725 | 0.6730 | 0.6508 | 0.2446 | 0.3005 |
| 0.95 | 0.3725 | 0.6979 | 0.6508 | 0.2087 | 0.3254 |
| 1.00 | 0.3725 | 0.7229 | 0.6508 | 0.1729 | 0.3504 |

## Primary Takeaway

C4 retains a positive system-success advantage over C1 across resolver-quality sensitivity (q=0.50..1.00), while explicitly trading autonomy for reliability.
