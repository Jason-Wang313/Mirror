# Human Baseline P2 Summary

## Core Metrics

- Exp1 Nat.Acc: 1.000 (95% Wilson CI [0.980, 1.000], n=192)
- Exp9 CFR (C1-style): 0.000 (weak components n=84)
- Exp9 Weak-Domain Escalation Rate: 0.202
- Exp9 Autonomy Rate: 0.809
- Exp9 System Success (oracle escalation assumption): 1.000

## Domain Split (from Exp1)

- Weak-domain rule requested: `median_or_bottom_k`
- Weak-domain rule applied: `bottom_k_fallback`
- Weak domains (active for Exp9): arithmetic, factual
- Weak domains (< median): (none)
- Bottom-2 fallback set: arithmetic, factual
- Strong domains: linguistic, logical, procedural, social, spatial, temporal

## Per-Domain Nat.Acc

| Domain | n | Correct | Nat.Acc |
|---|---:|---:|---:|
| arithmetic | 24 | 24 | 1.000 |
| factual | 24 | 24 | 1.000 |
| linguistic | 24 | 24 | 1.000 |
| logical | 24 | 24 | 1.000 |
| procedural | 24 | 24 | 1.000 |
| social | 24 | 24 | 1.000 |
| spatial | 24 | 24 | 1.000 |
| temporal | 24 | 24 | 1.000 |

## Notes

- This report assumes Exp9 decisions use: `PROCEED`, `USE_TOOL`, or `FLAG_FOR_REVIEW`.
- CFR denominator follows MIRROR convention: all weak-domain components.
- If median-split yields no weak domains, `median_or_bottom_k` applies a deterministic bottom-k fallback to keep the denominator estimable.

## Paper-Ready Insert (Draft)

Human Baseline P2 pilot on the staged subset reports Exp1 Nat.Acc 1.000 (95% CI [0.980, 1.000]). Using weak domains defined from this human baseline p2 Exp1 profile (rule `bottom_k_fallback`), Exp9 C1-style CFR is 0.000, with weak-domain escalation rate 0.202 and autonomy 0.809.
