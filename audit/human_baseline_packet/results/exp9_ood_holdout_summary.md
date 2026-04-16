# Exp9 OOD Holdout-Domain Stress Summary

- Run ID: `v20_maximal_closure_20260326T2200Z_ood_holdout`
- Conditions: `[1]` | Paradigms: `[1, 2, 3]`
- Models covered: `16`
- Holdout slices: `128`
- Policy match rate (full vs OOD): `1.000`
- Mean delta system success (OOD-full): `0.0000`
- Mean delta autonomy (OOD-full): `0.0000`
- Mean delta escalation (OOD-full): `0.0000`
- OOD generalization status: `pass`

## Notes

- Full policy = canonical `median_or_bottom_k` built using all 8 domains.
- OOD policy = hold one domain out when constructing weak-domain threshold.
- Near-zero mean deltas imply domain-holdout robustness for routing behavior.
