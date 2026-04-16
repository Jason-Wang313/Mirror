# Targeted Mechanistic Probe (Open-Weight Models)

- Run ID: `v20_maximal_closure_20260326T2200Z_mechanistic_probe`
- Models scored: `14`
- Mean AUC(monitor->correctness): `0.563`
- Mean AUC(monitor->proceed): `0.486`
- Mean control alignment gap (high-low proceed): `-0.022`
- Mean mismatch score: `0.085`
- Status: `pass`

## Notes

- This is a targeted behavioral probe, not hidden-state mechanistic tracing.
- `mismatch_score = (AUC_monitor->correctness - 0.5) - (high-low proceed gap)`.
- Higher mismatch implies monitoring signals are stronger than control translation.
