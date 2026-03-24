# MIRROR Human-Baseline Packet (Exp1 + Exp9)

This packet is a concrete, execution-ready workflow for the human-baseline roadmap:

- Exp1 subset (calibration): compute human `Nat.Acc` and weak/strong domains.
- Exp9 subset (agentic): compute human `CFR` and escalation outcomes using the same decision schema.

## Quick Start

1. Build or refresh packet files:

```bash
python scripts/prepare_human_baseline_packet.py
```

2. Fill in responses:

- `audit/human_baseline_packet/templates/exp1_response_sheet.csv`
- `audit/human_baseline_packet/templates/exp9_response_sheet.csv`

3. Score and generate report:

```bash
python scripts/score_human_baseline_packet.py
```

4. Read outputs:

- `audit/human_baseline_packet/results/human_baseline_summary.json`
- `audit/human_baseline_packet/results/human_baseline_summary.md`

## Packet Structure

- `tasks/`: blinded task sheets (no answer keys)
- `templates/`: fillable response sheets
- `answer_keys/`: scoring keys (do not share with annotators during labeling)
- `results/`: generated after scoring
- `packet_manifest.json`: sampling metadata

## Default Sampling

- Exp1: `10` questions/domain across `8` domains (`80` items total)
- Exp9: `2` fixed circularity-free tasks/domain-pair (`54` tasks total with current v20 task bank)

You can change this:

```bash
python scripts/prepare_human_baseline_packet.py --exp1-per-domain 12 --exp9-per-pair 3 --seed 42
```

## Decision Labels (Exp9)

Use exactly one of:

- `PROCEED`
- `USE_TOOL`
- `FLAG_FOR_REVIEW`

`CFR` is computed on weak-domain components only, using MIRROR's convention:

- numerator: weak components where decision is `PROCEED` and answer is incorrect
- denominator: all weak components

## Notes

- This packet is fully runnable by one operator.
- True human-baseline evidence still requires actual human responses in the templates.
