# Parse/API Missingness Analysis

- Run ID: `v20_stanford_hardening_20260325T2130Z_parse_canonical`
- Created (UTC): 2026-03-25T22:55:02+00:00
- Exp1 rows: 10996
- Exp9 component rows: 123552
- Primary imputation mode: `all`

## Missingness Overview

- Exp1 parse-missing rate: 4.96%
- Exp9 API-missing rate: 28.02%

## Sensitivity: Exp1 Natural Accuracy

| Mode | Mean Nat.Acc | Models |
| --- | ---: | ---: |
| `complete` | 0.4555 | 5 |
| `conservative` | 0.4376 | 5 |
| `ipw` | 0.4553 | 5 |

## Sensitivity: Exp9 Weak-Domain CFR

| Mode | Mean C1 Weak CFR | Mean C4 Weak CFR | Mean Reduction C1→C4 | Models(C1/C4) |
| --- | ---: | ---: | ---: | ---: |
| `complete` | 0.6469 | 0.1860 | 0.7349 | 16/14 |
| `conservative` | 0.7077 | 0.3462 | 0.5595 | 17/17 |
| `ipw` | 0.6483 | 0.1860 | 0.7349 | 16/14 |

## Mechanism Diagnostics

- Exp1 logistic AUC (observed vs missing): 0.9327047409536334
- Exp9 logistic AUC (observed vs missing): 0.9999326769324504
- Full per-feature chi-square tests and coefficient tables are in the JSON output.
