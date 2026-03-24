# Human-Baseline Reporting Template

## Protocol Snapshot

- Exp1 subset size: `{EXP1_N}`
- Exp9 subset size: `{EXP9_N_TASKS}` tasks (`{EXP9_N_COMPONENTS}` components)
- Decision schema: `PROCEED | USE_TOOL | FLAG_FOR_REVIEW`
- Weak-domain definition: domain `Nat.Acc` below median from human Exp1 profile

## Core Metrics

- Exp1 Nat.Acc: `{EXP1_NAT_ACC}` (95% CI `{EXP1_CI}`)
- Exp9 CFR (C1-style): `{EXP9_CFR}`
- Exp9 weak-domain escalation rate: `{EXP9_WEAK_ESC_RATE}`
- Exp9 autonomy rate: `{EXP9_AUTONOMY}`
- Exp9 system success (oracle escalation assumption): `{EXP9_SYSTEM_SUCCESS}`

## Domain Summary

### Weak domains

`{WEAK_DOMAINS}`

### Strong domains

`{STRONG_DOMAINS}`

## Paper-Ready Paragraph (Draft)

Human-baseline pilot on the staged subset reports Exp1 Nat.Acc `{EXP1_NAT_ACC}`
(95% CI `{EXP1_CI}`). Using weak domains defined from this human Exp1 profile,
Exp9 C1-style CFR is `{EXP9_CFR}`, with weak-domain escalation rate
`{EXP9_WEAK_ESC_RATE}` and autonomy `{EXP9_AUTONOMY}`.

## Reproducibility

- Packet manifest: `audit/human_baseline_packet/packet_manifest.json`
- Scoring output: `audit/human_baseline_packet/results/human_baseline_summary.json`
- Scoring markdown: `audit/human_baseline_packet/results/human_baseline_summary.md`
