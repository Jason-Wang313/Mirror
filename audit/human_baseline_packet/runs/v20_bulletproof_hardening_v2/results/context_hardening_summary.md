# Human Packet Context Analysis

## Ecological Validity Packet

- Tasks: 320
- Gold rows: 320
- Matched coverage: 100.0%
- Workflows: {'billing_support': 46, 'clinical_admin': 40, 'code_review': 43, 'fraud_screening': 28, 'incident_triage': 30, 'legal_intake': 43, 'sales_ops': 36, 'supply_chain': 54}
- Domains: {'arithmetic': 36, 'factual': 49, 'linguistic': 37, 'logical': 42, 'procedural': 36, 'social': 44, 'spatial': 48, 'temporal': 28}
- Risk tiers: {'critical': 25, 'high': 77, 'low': 109, 'medium': 109}
- Resolver types: {'human': 157, 'tool': 163}

## Oracle Realism Sensitivity

- Escalation components: 540
- Observed correctness in oracle-eval table: 50.2%

| Reviewer Accuracy q | Projected Correct Rate | Delta vs Oracle |
|---:|---:|---:|
| 0.70 | 0.70 | -0.30 |
| 0.75 | 0.75 | -0.25 |
| 0.80 | 0.80 | -0.20 |
| 0.85 | 0.85 | -0.15 |
| 0.90 | 0.90 | -0.10 |
| 0.95 | 0.95 | -0.05 |
| 1.00 | 1.00 | +0.00 |

## Goodhart Red-Team Packet

- Attack count: 240
- Attack types: {'confidence_spoof': 28, 'cost-pressure': 29, 'escalation_suppression': 29, 'false-authority-cue': 39, 'keyword_jamming': 28, 'latency-pressure': 29, 'policy-quote-injection': 34, 'tool-looping': 24}
- Target failure modes: {'calibration_drift': 41, 'false_flag_for_review': 30, 'incorrect_routing': 48, 'overconfident_proceed': 40, 'under_escalation': 35, 'wrong_tool_choice': 46}
- Expected exploit signals: {'abnormally_low_escalation': 46, 'confidence_inversion': 42, 'oracle_overuse': 28, 'policy_misclassification': 43, 'routing_skew': 39, 'tool_call_spike': 42}

### Mitigation Bundle

- hidden canary attacks in each release
- periodic attack-set rotation with held-out templates
- routing-trigger drift monitors on escalation patterns
- cross-signal anomaly checks (confidence vs action mismatch)
