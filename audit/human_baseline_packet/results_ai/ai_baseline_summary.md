# AI Baseline (Synthetic, Not Human) Summary

## Core Metrics

- Exp1 Nat.Acc: 1.000 (95% Wilson CI [0.954, 1.000], n=80)
- Exp9 CFR (C1-style): 0.000 (weak components n=0)
- Exp9 Weak-Domain Escalation Rate: 0.000
- Exp9 Autonomy Rate: 1.000
- Exp9 System Success (oracle escalation assumption): 1.000

## Domain Split (from Exp1)

- Weak domains (< median): (none)
- Strong domains: arithmetic, factual, linguistic, logical, procedural, social, spatial, temporal

## Per-Domain Nat.Acc

| Domain | n | Correct | Nat.Acc |
|---|---:|---:|---:|
| arithmetic | 10 | 10 | 1.000 |
| factual | 10 | 10 | 1.000 |
| linguistic | 10 | 10 | 1.000 |
| logical | 10 | 10 | 1.000 |
| procedural | 10 | 10 | 1.000 |
| social | 10 | 10 | 1.000 |
| spatial | 10 | 10 | 1.000 |
| temporal | 10 | 10 | 1.000 |

## Notes

- This report assumes Exp9 decisions use: `PROCEED`, `USE_TOOL`, or `FLAG_FOR_REVIEW`.
- CFR denominator follows MIRROR convention: all weak-domain components.

## Paper-Ready Insert (Draft)

AI Baseline (Synthetic, Not Human) pilot on the staged subset reports Exp1 Nat.Acc 1.000 (95% CI [0.954, 1.000]). Using weak domains defined from this ai baseline (synthetic, not human) Exp1 profile, Exp9 C1-style CFR is 0.000, with weak-domain escalation rate 0.000 and autonomy 1.000.
