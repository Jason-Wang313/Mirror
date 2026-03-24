# Human-Baseline Protocol (Executable)

## Scope

- Stage 1 (Exp1): estimate human calibration baseline (`Nat.Acc`) on a stratified subset.
- Stage 2 (Exp9): estimate human agentic behavior baseline (`CFR`, escalation outcomes) on fixed tasks.

## Procedure

1. Generate packet with fixed seed.
2. Complete Exp1 response sheet first.
3. Complete Exp9 response sheet second (decision + answer per component).
4. Run scorer to compute:
   - Exp1 Nat.Acc + CI
   - weak/strong domain split
   - Exp9 CFR (C1-style)
   - weak-domain escalation rate
   - autonomy rate
   - oracle system success
5. Insert numbers into manuscript using `report_template.md`.

## Runtime Estimate (Single Annotator)

- Exp1 (80 items): ~45-70 min
- Exp9 (54 tasks in current packet): ~60-90 min
- Scoring + report generation: <2 min

## Commands

```bash
python scripts/prepare_human_baseline_packet.py
python scripts/score_human_baseline_packet.py
```

## Quality Controls

- Fixed random seed in packet manifest.
- Blinded task sheets separated from answer keys.
- Deterministic scoring via `match_answer_robust`.
- Unknown/invalid Exp9 decisions are counted and reported.
