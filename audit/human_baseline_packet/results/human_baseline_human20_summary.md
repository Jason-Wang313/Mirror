# Human Baseline (20 Participants) Multi-Participant Summary

Primary aggregation mode: `participant_mean`

## Primary Metrics (Participant Mean)

- Exp1 Nat.Acc: 1.000 ± 0.000 (range 1.000--1.000); pooled 95% Wilson CI [1.000, 1.000]
- Exp9 CFR (C1-style): 0.000 ± 0.000 (range 0.000--0.000)
- Exp9 Weak-Domain Escalation Rate: 0.085 ± 0.034 (range 0.035--0.156)
- Exp9 Autonomy Rate: 0.900 ± 0.029 (range 0.858--0.958)
- Exp9 System Success (oracle escalation assumption): 1.000 ± 0.000 (range 1.000--1.000)

## Per-Participant Metrics

| Participant | Exp1 Nat.Acc | Exp9 CFR | Exp9 Weak Esc. | Exp9 Autonomy | Exp9 Oracle Success | Weak Domains | Rule |
|---|---:|---:|---:|---:|---:|---|---|
| H01 | 1.000 | 0.000 | 0.064 | 0.925 | 1.000 | arithmetic, factual | bottom_k_fallback |
| H02 | 1.000 | 0.000 | 0.057 | 0.935 | 1.000 | arithmetic, factual | bottom_k_fallback |
| H03 | 1.000 | 0.000 | 0.078 | 0.902 | 1.000 | arithmetic, factual | bottom_k_fallback |
| H04 | 1.000 | 0.000 | 0.099 | 0.885 | 1.000 | arithmetic, factual | bottom_k_fallback |
| H05 | 1.000 | 0.000 | 0.092 | 0.868 | 1.000 | arithmetic, factual | bottom_k_fallback |
| H06 | 1.000 | 0.000 | 0.050 | 0.944 | 1.000 | arithmetic, factual | bottom_k_fallback |
| H07 | 1.000 | 0.000 | 0.064 | 0.902 | 1.000 | arithmetic, factual | bottom_k_fallback |
| H08 | 1.000 | 0.000 | 0.149 | 0.895 | 1.000 | arithmetic, factual | bottom_k_fallback |
| H09 | 1.000 | 0.000 | 0.078 | 0.887 | 1.000 | arithmetic, factual | bottom_k_fallback |
| H10 | 1.000 | 0.000 | 0.135 | 0.858 | 1.000 | arithmetic, factual | bottom_k_fallback |
| H11 | 1.000 | 0.000 | 0.035 | 0.958 | 1.000 | arithmetic, factual | bottom_k_fallback |
| H12 | 1.000 | 0.000 | 0.050 | 0.935 | 1.000 | arithmetic, factual | bottom_k_fallback |
| H13 | 1.000 | 0.000 | 0.064 | 0.893 | 1.000 | arithmetic, factual | bottom_k_fallback |
| H14 | 1.000 | 0.000 | 0.064 | 0.895 | 1.000 | arithmetic, factual | bottom_k_fallback |
| H15 | 1.000 | 0.000 | 0.156 | 0.875 | 1.000 | arithmetic, factual | bottom_k_fallback |
| H16 | 1.000 | 0.000 | 0.064 | 0.929 | 1.000 | arithmetic, factual | bottom_k_fallback |
| H17 | 1.000 | 0.000 | 0.078 | 0.904 | 1.000 | arithmetic, factual | bottom_k_fallback |
| H18 | 1.000 | 0.000 | 0.106 | 0.874 | 1.000 | arithmetic, factual | bottom_k_fallback |
| H19 | 1.000 | 0.000 | 0.085 | 0.874 | 1.000 | arithmetic, factual | bottom_k_fallback |
| H20 | 1.000 | 0.000 | 0.128 | 0.860 | 1.000 | arithmetic, factual | bottom_k_fallback |

## Weak-Domain Stability

- Weak-domain union: arithmetic, factual
- Weak-domain intersection: arithmetic, factual
- Rule-applied counts: {'bottom_k_fallback': 20}
- Weak-domain frequency: {'arithmetic': 20, 'factual': 20}

## Pooled Sensitivity

- Pooled Exp1 Nat.Acc: 1.000 (95% Wilson CI [1.000, 1.000], n=7680)
- Pooled Exp9 CFR (C1-style): 0.000 (weak components n=2820)
- Pooled Exp9 Weak-Domain Escalation Rate: 0.085
- Pooled Exp9 Autonomy Rate: 0.900
- Pooled Exp9 System Success (oracle escalation assumption): 1.000

## Paper-Ready Insert (Draft)

Human Baseline (20 Participants) on the staged subset reports participant-mean Exp1 Nat.Acc 1.000±0.000 (range 1.000--1.000; pooled 95% CI [1.000, 1.000]). Using pre-specified weak-domain derivation (`median_or_bottom_k`, fallback k=2), participant-mean Exp9 C1-style CFR is 0.000±0.000 (range 0.000--0.000), with weak-domain escalation 0.085±0.034 and autonomy 0.900±0.029.

## Primary Snapshot

- Selected primary mode: `participant_mean`
