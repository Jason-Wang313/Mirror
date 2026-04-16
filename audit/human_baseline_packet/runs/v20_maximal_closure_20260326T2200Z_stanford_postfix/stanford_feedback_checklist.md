# Stanford Feedback Checklist

- Run ID: `v20_maximal_closure_20260326T2200Z_stanford_postfix`
- Pass: 22/23
- Fail: 1

## Checks

| ID | Pass | Description |
| --- | --- | --- |
| `parse_missingness` | yes | Parse/API missingness analysis is reported in manuscript. |
| `mnar_bounds_reporting` | yes | Explicit MNAR bound scenarios are reported for headline metrics. |
| `exp3_difficulty_control` | yes | Exp3 difficulty-controlled ablation is reported. |
| `cce_main_text_formula_example` | yes | CCE is formalized in main text with a worked example. |
| `mci_channel_robustness` | yes | MCI channel-disagreement diagnostics are reported. |
| `non_oracle_utility_main_text` | yes | Non-oracle/fallible-resolver utility appears in main claim path. |
| `utility_frontier_prominence` | yes | Cost/latency-aware Pareto or frontier reporting appears in main text path. |
| `instance_baseline_expanded` | yes | Instance-level baselines are discussed beyond narrow legacy frame. |
| `instance_baseline_robust_set` | yes | Robust baseline set includes calibrated confidence and target-error grid framing. |
| `mapping_validity` | yes | Exp9 domain-component mapping validation is reported. |
| `related_work_expanded` | yes | Recent metacognitive-control/abstention related work is integrated in main text. |
| `baseline_main_text_prominence` | yes | Main text explicitly foregrounds budget-matched and frontier baseline comparisons. |
| `weak_domain_frontier_reporting` | yes | Weak-domain policy-family frontier (median/bottom-k/absolute/quantile) is reported. |
| `exp3_expanded_sample_size_disclosure` | yes | Manuscript discloses expanded Exp3 pair-level scale beyond the 112-task v2 bank. |
| `hard_packet_v2_integration` | yes | Hard human packet v2 evidence is integrated (summary artifact + manuscript mention). |
| `proper_score_primary_claim_path` | yes | Main claim path explicitly prioritizes strictly proper scoring (Brier/log/ECE) over MIRROR-gap-only framing. |
| `cfr_plus_utility_coreport` | yes | CFR is co-reported with system success/autonomy/cost/latency in claim-critical sections. |
| `baseline_operating_point_disclosure` | yes | Baseline operating points disclose matched budget/autonomy/escalation details. |
| `sar_ai_interpretation_block` | yes | SAR/AI interpretation is explicitly provided with scale intuition in manuscript text. |
| `ood_holdout_evidence` | yes | OOD holdout-domain routing stress artifact is present and passes stability status. |
| `mechanistic_probe_evidence` | yes | Targeted open-weight mechanistic probe artifact is present with sufficient model coverage. |
| `verification_coverage_table_evidence` | yes | Generation/verification coverage table artifact exists and is marked complete. |
| `hard_v2_cohort_completion` | no | Hard-v2 cohort execution is complete with at least 20 validated participant pairs. |

## Actionable Items

- `hard_v2_cohort_completion`: Hard-v2 cohort execution is complete with at least 20 validated participant pairs. (Expect human_baseline_hardv2_cohort_summary.json status='complete' and participant_pairs_validated_ok >= 20.)