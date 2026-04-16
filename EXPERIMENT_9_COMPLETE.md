# Experiment 9: The Knowing-Doing Gap — Rebuild Complete

## Status: Rebuilt to Full Spec ✅

All five core files refactored/rebuilt to match the NeurIPS 2026 spec.

---

## Audit Findings (Pre-Rebuild)

| Requirement | Old Status | New Status |
|---|---|---|
| Subcategory-level analysis (40 subcats × 12 models = 480 pts) | ❌ Model-level only (N=6) | ✅ Subcategory level |
| Fixed task set as co-primary (300 fixed, circularity_free) | ❌ 20 shared tasks, excluded from analysis | ✅ Primary path in both runner and analyzer |
| Circularity defense (`circularity_free` flag) | ❌ Not in task or trial records | ✅ In every task and trial record |
| Partial r ≈ 0 contingency + null reporting | ⚠️ Partial (model-level only) | ✅ Per MIRROR level, BH-corrected, honest null block |
| 4 Conditions (Uninformed/Self-informed/Instructed/Constrained) | ❌ Single condition | ✅ Implemented |
| 3 Spec Paradigms (Autonomous/Checkpoint/No-Tool Behavioral) | ❌ 5 wrong paradigms | ✅ Exactly 3 spec paradigms |
| KDI metric | ❌ Missing | ✅ Implemented |
| Escalation curve | ❌ Missing | ✅ Implemented |
| Routing comparison | ❌ Missing | ✅ Implemented |
| BCa bootstrap (10,000 iterations) | ❌ 1,000 percentile | ✅ BCa, 10,000 |
| BH FDR correction | ❌ Missing | ✅ Implemented |
| Mixed-effects model | ❌ Missing | ✅ statsmodels MixedLM + fallback |
| 12 models | ❌ 6 models | ✅ 12 models |
| Controls 2, 3, 4, 5, 6 | ❌ Missing | ✅ All implemented |

---

## Files Rebuilt

### 1. `mirror/experiments/agentic_paradigms.py` ✅
**Paradigms (exactly 3, matching spec):**
- `AutonomousToolUseParadigm` (P1) — spec classification rules
- `CheckpointDecisionsParadigm` (P2) — PROCEED/USE_TOOL/FLAG_FOR_REVIEW
- `NoToolBehavioralParadigm` (P3) — hedging/decomp/token/error_type per component

**Condition injection:**
- `build_condition_prefix(condition, domain_a, domain_b, score_a, score_b)` — Conditions 1–4
- `build_false_score_prefix(...)` — Control 2 inverted scores
- `classify_error_type(answer, correct, hedge_count)` — overconfident_precise vs cautious_approximate

---

### 2. `mirror/scoring/agentic_metrics.py` ✅
**Metrics:**
- `compute_cfr_udr_subcategory(results, condition, paradigm, circularity_free_only)` — subcategory-level
- `compute_cfr_model_level(results, ...)` — for escalation curve
- `compute_kdi(mirror_gap, appropriate_action_rate)` — Knowing-Doing Index
- `compute_kdi_table(subcategory_metrics, mirror_gaps)` — KDI distribution per model
- `compute_paradigm3_signals(results, ...)` — behavioral signal aggregation
- `compute_oracle_cfr(results, ...)` — Control 5
- `compute_routing_comparison(results, ...)` — no/accuracy/MIRROR/oracle routing
- `interpret_partial_r(partial_r)` — pre-registered null reporting
- `interpret_paradigm3_null(r)` — RLHF confound interpretation

**Domain constants:**
- `DOMAINS` — 8 domains
- `SUBCATEGORIES` — 5 subcategories per domain (40 total)
- `all_subcategory_keys()` — list of 40 (domain, subcat) tuples

---

### 3. `scripts/generate_exp9_tasks.py` ✅
**Task schema** (all fields per spec):
```
task_id, task_type, circularity_free, target_model,
domain_a, domain_b, subcategory_a, subcategory_b,
difficulty_a, difficulty_b, correct_answer_a, correct_answer_b,
answer_type_a, answer_type_b, task_text, part1_text, part2_text
```
**Task sets:**
- 300 fixed tasks (circularity_free=True) — from FIXED_DOMAIN_PAIRS × templates
- 300 tailored tasks (circularity_free=False) — model-specific strong/weak pairings
- `--pilot` flag for quick 8-task sanity check

**Content:**
- 13 domain-pair template sets (60 hand-crafted templates)
- 5 generic fallback templates
- Subcategory assignment via round-robin per domain

---

### 4. `scripts/run_experiment_9.py` ✅
**Trial record schema** (all fields including new):
```
model, task_id, condition (1-4), paradigm (1-3), is_false_score_control,
circularity_free, domain_a/b, subcategory_a/b, strength_a/b,
component_a/b_decision/correct/answer/tool_used/deferred/externally_routed,
exp1_accuracy_a/b, mirror_gap_a/b,
hedge_count/decomp_count/token_count/error_type (P3),
raw_response, api_success, timestamp
```

**Modes:**
- `--mode pilot` — 2 models, C1, P1+2, 5 tasks (~60 API calls)
- `--mode full` — 12 models, 4 conditions, 3 paradigms, 600 tasks (~86,400 API calls)
- `--mode control2` — false score injection (150 fixed tasks per model)

**Key implementation notes:**
- Condition 4 external routing: `apply_condition4_routing()` pre-empts model on weak-domain components (exp1_accuracy < 0.50 → force use_tool, assume correct)
- `SKIP_COMBOS = {(4, 3)}` — Condition 4 × Paradigm 3 skipped (spec: duplicate of C1P3)
- 12 models in `MODELS_FULL`
- Preserved: `call_with_retry`, `split_response_into_parts`, `classify_section_p1`

---

### 5. `scripts/analyze_experiment_9.py` ✅
**Primary analysis grain: subcategory level (model × subcategory)**

**Outputs:**
1. Money Plot (PRIMARY: fixed tasks only; SUPPLEMENTARY: all tasks)
2. Escalation Curve (4 conditions) with shape interpretation
3. CFR/UDR table per model
4. KDI table
5. Partial correlation table (per MIRROR level 0–3 + composite)
6. Routing comparison (4 strategies)
7. Paradigm convergence (Control 6)
8. Paradigm 3 behavioral signal correlations
9. Control 2 (false score injection) analysis
10. Control 3 (cross-model dissociation) + permutation test
11. Cohen's d (strong vs weak domain behaviour)
12. Oracle baseline (Control 5)

**Statistical framework (all required elements):**
- `bootstrap_bca(x, y, n=10_000)` — BCa bootstrap CI
- `benjamini_hochberg(p_values)` — BH FDR correction
- `mixed_effects_approximation(...)` — statsmodels MixedLM + within-group demeaned fallback
- `partial_corr(x, y, z)` — partial correlation controlling for accuracy
- `cohens_d(group1, group2)` — effect size
- `_permutation_test_dissociation(...)` — Control 3 permutation test

---

## Running the Experiment

```bash
# Step 1: Generate tasks (pilot sanity check)
python scripts/generate_exp9_tasks.py --pilot

# Step 2: Generate full task set
python scripts/generate_exp9_tasks.py

# Step 3: Pilot run (verify end-to-end)
python scripts/run_experiment_9.py --mode pilot

# Step 4: Full run (overnight, ~$100 API cost)
python scripts/run_experiment_9.py --mode full

# Step 5: Control 2 (false score injection)
python scripts/run_experiment_9.py --mode control2

# Step 6: Analysis
python scripts/analyze_experiment_9.py --run-id <RUN_ID>

# Primary-only (faster, circularity-free fixed tasks)
python scripts/analyze_experiment_9.py --run-id <RUN_ID> --primary-only
```

---

## Pre-Registration Checklist (before Step 4)

Per spec: publish on OSF before running Phase 3 (data collection).

Pre-registered predictions:
1. CFR will NOT decrease by > 10% between Condition 1 and 2 (knowing-doing gap)
2. CFR WILL decrease by > 40% between Condition 1 and 4 (external control works)
3. MIRROR-CFR correlation > 0 (r > 0.3) on fixed tasks (money plot)
4. Paradigm 3 behavioral signals correlate with MIRROR gap (r > 0.2)

---

## Success Criteria
The experiment succeeds if ANY of the following hold:
1. MIRROR-CFR correlation significant on fixed tasks (money plot shows real trend)
2. Escalation curve shows knowing-doing gap (flat C1→2→3, drop at C4)
3. Paradigm 3 behavioral signals correlate with MIRROR (RLHF confound eliminated)
4. MIRROR routing outperforms accuracy routing
5. Cross-model dissociations exist on identical tasks
