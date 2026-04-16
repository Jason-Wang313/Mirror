# Parse/API Missingness Analysis

- Run ID: `v20_final_hardening_20260326B_parse_missingness`
- Created (UTC): 2026-03-26T16:57:32+00:00
- Exp1 rows: 47318
- Exp9 component rows: 124872
- Primary imputation mode: `all`
- MNAR bound mode: `moderate` (alpha=0.50)

## Missingness Overview

- Exp1 parse-missing rate: 24.35%
- Exp9 API-missing rate: 27.72%

## Sensitivity: Exp1 Natural Accuracy

| Mode | Mean Nat.Acc | Models |
| --- | ---: | ---: |
| `complete` | 0.3917 | 13 |
| `conservative` | 0.2507 | 18 |
| `ipw` | 0.3833 | 13 |

## Sensitivity: Exp9 Weak-Domain CFR

| Mode | Mean C1 Weak CFR | Mean C4 Weak CFR | Mean Reduction C1→C4 | Models(C1/C4) |
| --- | ---: | ---: | ---: | ---: |
| `complete` | 0.6450 | 0.1890 | 0.7290 | 16/14 |
| `conservative` | 0.7059 | 0.3485 | 0.5549 | 17/17 |
| `ipw` | 0.6463 | 0.1889 | 0.7294 | 16/14 |
| `mnar_bound` | NA | NA | [0.1227, 0.7779] | 17 |

## Headline Delta Summary

- C1→C4 reduction delta vs complete-case (conservative): -0.1741
- C1→C4 reduction delta vs complete-case (IPW): 0.0003
- C1→C4 reduction MNAR bounds: [0.1227, 0.7779]
- CCE anchor movement under this missingness stress: [0.0000, 0.0000]

## Mechanism Diagnostics

- Exp1 logistic AUC (observed vs missing): 0.9725274788550037
- Exp9 logistic AUC (observed vs missing): 0.9999339835494443
- Full per-feature chi-square tests and coefficient tables are in the JSON output.
