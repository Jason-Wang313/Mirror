# Human Baseline (3 Participants) Multi-Participant Summary

Primary aggregation mode: `participant_mean`

## Primary Metrics (Participant Mean)

- Exp1 Nat.Acc: 1.000 ± 0.000 (range 1.000--1.000); pooled 95% Wilson CI [0.993, 1.000]
- Exp9 CFR (C1-style): 0.000 ± 0.000 (range 0.000--0.000)
- Exp9 Weak-Domain Escalation Rate: 0.163 ± 0.090 (range 0.060--0.226)
- Exp9 Autonomy Rate: 0.838 ± 0.077 (range 0.781--0.926)
- Exp9 System Success (oracle escalation assumption): 1.000 ± 0.000 (range 1.000--1.000)

## Per-Participant Metrics

| Participant | Exp1 Nat.Acc | Exp9 CFR | Exp9 Weak Esc. | Exp9 Autonomy | Exp9 Oracle Success | Weak Domains | Rule |
|---|---:|---:|---:|---:|---:|---|---|
| P1 | 1.000 | 0.000 | 0.060 | 0.926 | 1.000 | arithmetic, factual | bottom_k_fallback |
| P2 | 1.000 | 0.000 | 0.202 | 0.809 | 1.000 | arithmetic, factual | bottom_k_fallback |
| P3 | 1.000 | 0.000 | 0.226 | 0.781 | 1.000 | arithmetic, factual | bottom_k_fallback |

## Weak-Domain Stability

- Weak-domain union: arithmetic, factual
- Weak-domain intersection: arithmetic, factual
- Rule-applied counts: {'bottom_k_fallback': 3}
- Weak-domain frequency: {'arithmetic': 3, 'factual': 3}

## Pooled Sensitivity

- Pooled Exp1 Nat.Acc: 1.000 (95% Wilson CI [0.993, 1.000], n=576)
- Pooled Exp9 CFR (C1-style): 0.000 (weak components n=252)
- Pooled Exp9 Weak-Domain Escalation Rate: 0.163
- Pooled Exp9 Autonomy Rate: 0.838
- Pooled Exp9 System Success (oracle escalation assumption): 1.000

## Paper-Ready Insert (Draft)

Human Baseline (3 Participants) (3 participants) on the staged subset reports participant-mean Exp1 Nat.Acc 1.000±0.000 (range 1.000--1.000; pooled 95% CI [0.993, 1.000]). Using pre-specified weak-domain derivation (`median_or_bottom_k`, fallback k=2), participant-mean Exp9 C1-style CFR is 0.000±0.000 (range 0.000--0.000), with weak-domain escalation 0.163±0.090 and autonomy 0.838±0.077.

## Primary Snapshot

- Selected primary mode: `participant_mean`
