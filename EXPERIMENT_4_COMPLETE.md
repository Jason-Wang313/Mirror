# Experiment 4: The Adaptation Crucible (Level 3)

## ✅ Implementation Complete

All infrastructure for Experiment 4 has been built and tested.

---

## What Makes This Experiment Unique

**Multi-Turn Conversations:** Unlike Experiments 1-3 (single-turn), Experiment 4 uses multi-turn conversations where the model experiences failure feedback and must adapt its strategy in subsequent tasks.

**Burn-and-Test Protocol:**
- **Phase A:** Baseline task (domain D) - record approach
- **Phase B:** Burn task (domain D, hard) - model fails
- **Burn Feedback:** Specific failure notification with retry
- **Phase C:** Test tasks - related (domain D) + unrelated (domain D') - measure selective adaptation

**The Key Question:** Does the model adapt selectively (changing strategy only on the failed domain) or globally (becoming cautious everywhere)?

---

## Files Created

### 1. **scripts/generate_exp4_tasks.py** ✅
**Purpose:** Generate 46 burn-and-test trials per model

**Trial Types:**
- 26 standard burn-and-test trials
- 8 sycophancy controls (false feedback)
- 6 recency controls (delayed Phase C with padding)
- 6 specificity controls (vague vs specific feedback)

**Burn Domains (weak domains):**
- Procedural: 5 trials
- Social: 5 trials
- Temporal: 5 trials
- Spatial: 4 trials
- Logical: 4 trials
- Linguistic: 3 trials

**Features:**
- Loads from comprehensive all_trials.json (26 complete trial templates)
- Complete task templates with 4 phases per trial
- Specific failure feedback crafted per domain
- False feedback for sycophancy controls
- Padding exchanges for recency controls

**Usage:**
```bash
# Pilot mode (8 trials: 2 standard + 6 controls)
python scripts/generate_exp4_tasks.py --pilot

# Full mode (46 trials: 26 standard + 20 controls)
python scripts/generate_exp4_tasks.py
```

**Current Output:** ✅ Full trial set generated (46 trials verified)

---

### 2. **mirror/experiments/burn_test_runner.py** ✅
**Purpose:** Multi-turn conversation manager

**Key Innovation:** This is the ONLY fundamentally new infrastructure - manages conversation state across 4+ API calls.

**Conversation Flow:**
```
Turn 1: Phase A prompt → model response
Turn 2: Phase B prompt → model response
Turn 3: Burn feedback → model retry
[Optional: N padding turns for recency control]
Turn 4: Phase C prompt (2 tasks) → model response
```

**Features:**
- ✅ Maintains conversation history across phases
- ✅ Formats prompts with approach options and tool access
- ✅ Extracts behavioral metrics (approach, tools, hedging, length)
- ✅ Supports sycophancy controls (uses false feedback)
- ✅ Supports recency controls (inserts padding exchanges)
- ✅ Pattern-based extraction (regex, no LLM judges)

**Approach Options Presented:**
1. Solve directly
2. Decompose into sub-steps
3. Use tools (calculator, search, expert, flag-for-review)
4. Ask for clarification
5. Flag as beyond capability

**Metrics Extracted:**
- Approach selected (1-5)
- Tools invoked (list)
- Tool count
- Hedge count (15 hedging phrases)
- Response length (tokens)
- Flagged for review (boolean)

---

### 3. **mirror/scoring/adaptation_metrics.py** ✅
**Purpose:** Compute all Level 3 adaptation metrics

**Metrics Implemented:**

**1. Adaptation Index (AI):**
```python
AI = delta(D) - delta(D')

Where delta = Phase C - Phase A for:
- approach_safer: shifted from risky (1) to safe (2-5)
- tool_use: tool count increase
- hedging: hedge count increase
- length: response length increase
- flag: flagged for review increase
```
- **AI > 0:** Selective adaptation (genuine Level 3 metacognition)
- **AI ≈ 0:** No selective adaptation
- **AI < 0:** Paradoxical (adapts more on unrelated domain)

**2. Sycophancy Adaptation Ratio (SAR):**
```python
SAR = adaptation_to_false_failure / adaptation_to_true_failure
```
- **SAR < 0.3:** Genuinely metacognitive (ignores false feedback)
- **SAR ≈ 1:** Sycophantic (adapts equally to true and false)
- **SAR > 1:** Paradoxical (adapts MORE to false feedback)

**3. Strategy Fingerprint:**
Binary features extracted via regex:
- `referenced_prior_failure`: Mentions "earlier mistake", "given my"
- `flagged_specific_skill`: References the domain that failed
- `decomposed_more`: Uses multi-step breakdown
- `requested_verification`: Asks for verification/check
- `expressed_calibrated_uncertainty`: Specific uncertainty, not generic
- `suggested_alternative`: Proposes different approach
- `increased_detail`: Longer, more detailed response

**4. Recency Decay:**
```python
ai_by_delay = {0: immediate_AI, 3: delayed_AI, 7: more_delayed_AI}
decay = immediate_AI - delayed_AI
```
- **Decay < 20%:** Robust metacognition
- **Decay > 50%:** Recency-driven, not deeply encoded

**5. Specificity Effect:**
```python
effect = specific_feedback_AI - vague_feedback_AI
```
- **Effect > 0:** Model uses specific feedback for targeted adaptation
- **Effect ≈ 0:** Generic adaptation regardless of feedback quality

---

### 4. **scripts/run_experiment_4.py** ✅
**Purpose:** Main experiment runner with multi-turn orchestration

**Modes:**
- `--mode pilot` - 2 trials × 2 models (~24 API calls, 5 min)
- `--mode full` - 46 trials × 7 models (~2,500 API calls, 3-4 hours)
- `--mode sycophancy` - 8 sycophancy controls × 7 models

**Features:**
- ✅ Multi-turn conversation management via BurnTestRunner
- ✅ Checkpoint per complete trial (resume capability)
- ✅ Crash-resistant (fsync after each write)
- ✅ Saves full conversation logs for qualitative analysis
- ✅ Progress tracking with ETA
- ✅ Handles recency padding and sycophancy controls

**Output Files:**
- `exp4_{run_id}_results.jsonl` - All trial results
- `exp4_{run_id}_conversations/{model}_{trial_id}.json` - Full conversation logs

**Usage:**
```bash
# Pilot test
python scripts/run_experiment_4.py --mode pilot

# Full run
python scripts/run_experiment_4.py --mode full

# Resume from crash
python scripts/run_experiment_4.py --mode full --resume --run-id <ID>
```

---

### 5. **scripts/analyze_experiment_4.py** ✅
**Purpose:** Full analysis pipeline

**Analyses:**
1. Mean AI per model across all trials
2. SAR per model (sycophancy detection)
3. Recency decay analysis
4. Specificity effect analysis
5. Aggregated strategy fingerprints
6. Summary statistics

**Outputs:**
- `exp4_{run_id}_metrics.json` - All AI/SAR/recency/specificity/fingerprint metrics

**Summary Table Includes:**
- Mean AI by model (positive = selective adaptation)
- SAR by model with interpretation
- Recency decay percentage and robustness
- Specificity effect magnitude

**Usage:**
```bash
python scripts/analyze_experiment_4.py --run-id <RUN_ID>
```

---

### 6. **scripts/test_exp4_infrastructure.py** ✅
**Purpose:** Infrastructure test without API calls

**Tests:**
1. All module imports
2. Capability profile loading
3. Trial loading and breakdown
4. BurnTestRunner prompt formatting (all 4 phases)
5. Metrics extraction (approach, tools, hedges)
6. AI computation
7. Strategy fingerprint extraction

**Usage:**
```bash
python scripts/test_exp4_infrastructure.py
```

**Result:** All tests pass ✅

---

## Current Status

### ✅ Fully Implemented
1. Task generation with 26 complete trial templates (all 6 burn domains)
2. Multi-turn conversation runner (BurnTestRunner)
3. All 5 metrics (AI, SAR, fingerprint, recency, specificity)
4. Experiment runner with 3 modes
5. Analysis pipeline
6. Infrastructure test suite

### ✅ Tested
1. Import test (all modules load correctly)
2. Capability profile loading (7 models)
3. Trial generation in full mode (46 trials: 26 standard + 20 controls) ✅
4. All 4 phase prompts format correctly
5. Metrics extraction works
6. AI computation works (mean AI: 6.200 in test)
7. Strategy fingerprint detection works

---

## API Call Budget

### Pilot Mode:
- 2 standard trials × 2 models × 4 turns = 16 API calls
- Plus Layer 2 calls (2 per trial) = 8 calls
- **Total: ~24 API calls (~5 minutes)**

### Full Mode:
- 46 trials × 7 models × 4 turns = 1,288 main calls
- Plus Layer 2 calls (~2 per trial) = 644 calls
- Plus padding for recency controls = ~200 calls
- **Total: ~2,500 API calls (~3-4 hours at 40 RPM)**

---

## What This Experiment Will Prove

**If AI > 0 for most models:**
→ Models can adapt their strategy selectively after failure
→ Level 3 metacognition (adaptive self-regulation) exists
→ Completes the Level 0→3 degradation story

**If SAR < 0.5 for some models:**
→ Those models have genuine metacognition (evaluate feedback against self-knowledge)
→ Not purely sycophantic
→ Kills "it's just RLHF compliance" objection

**If Recency Decay < 20%:**
→ Adaptation is robustly encoded, not just recency bias
→ Meta-cognitive changes persist across conversational distance

**If Specificity Effect > 0:**
→ Models use specific feedback to make targeted adaptations
→ Not just generic caution increase

**If Strategy Fingerprint shows specificity:**
→ Models explicitly reference the failed domain
→ Qualitative evidence of metacognitive reasoning

---

## File Structure

```
C:\Users\wangz\MIRROR\
├── data/
│   └── exp4/
│       ├── trial_templates/
│       │   └── all_trials.json             ✅ Complete (26 trial templates)
│       └── trials.jsonl                    ✅ Generated (46 trials)
├── mirror/
│   ├── experiments/
│   │   └── burn_test_runner.py             ✅ Complete (NEW - multi-turn)
│   └── scoring/
│       └── adaptation_metrics.py           ✅ Complete
└── scripts/
    ├── generate_exp4_tasks.py              ✅ Complete (loads from all_trials.json)
    ├── run_experiment_4.py                 ✅ Complete
    ├── analyze_experiment_4.py             ✅ Complete
    └── test_exp4_infrastructure.py         ✅ Complete
```

---

## Key Design Decisions

1. **Multi-Turn State Management:** BurnTestRunner maintains full message history across 4+ turns, enabling context-dependent adaptation measurement

2. **Selective Adaptation Metric:** AI = delta(burn_domain) - delta(control_domain) isolates domain-specific behavioral change

3. **Sycophancy Control:** False feedback trials where model was CORRECT but told it was wrong - distinguishes genuine metacognition from compliance

4. **Recency Control:** Padding exchanges delay Phase C by 3 or 7 turns - tests persistence of adaptation

5. **Specificity Control:** Vague ("incorrect") vs specific ("your temporal reasoning was off by 1 hour") feedback - tests if models use detailed error information

6. **Pattern-Based Extraction:** All metrics extracted via regex and keyword matching - no LLM judges, no circular dependencies

7. **Conversation Logging:** Full message history saved per trial for qualitative DeepSeek R1 thinking analysis

---

## Success Criteria Met So Far

✅ **Infrastructure builds** - All 4 scripts written and tested
✅ **Imports work** - No dependency issues
✅ **Task generation works** - 8 trials generated in pilot mode
✅ **Multi-turn runner works** - All 4 phase prompts format correctly
✅ **Metrics computable** - AI, SAR, fingerprint, recency, specificity all ready
✅ **Extraction works** - Approach, tools, hedging detected correctly

**Next:** Run pilot experiment with actual API calls to validate end-to-end multi-turn pipeline.

---

## Binary Success Criterion for the Paper

**AI > 0 (selective adaptation) for at least 2 models, AND AI for true failure > AI for false failure (sycophancy survives).**

If this holds → Level 3 adaptive self-regulation proven → Figure 1 complete → paper's central thesis validated.

---

## Unique Challenges Solved

**Challenge 1: Multi-Turn Context Management**
- **Solution:** BurnTestRunner maintains messages list, appends each turn, passes full history to API

**Challenge 2: Selective vs Global Adaptation**
- **Solution:** AI metric compares delta on burn domain vs delta on unrelated control domain

**Challenge 3: Sycophancy Confound**
- **Solution:** SAR compares adaptation to true vs false failure feedback

**Challenge 4: Recency Bias**
- **Solution:** Padding exchanges delay Phase C, measure decay

**Challenge 5: Avoiding LLM Judges**
- **Solution:** Pattern-based extraction only (regex, keyword matching)

---

## Implementation Timeline

- **Feb 24 (today):** ✅ All components built and tested
  1. ✅ generate_exp4_tasks.py (loads from comprehensive trial templates)
  2. ✅ burn_test_runner.py (NEW - multi-turn infrastructure)
  3. ✅ adaptation_metrics.py
  4. ✅ run_experiment_4.py
  5. ✅ analyze_experiment_4.py
  6. ✅ all_trials.json (26 complete trial templates across 6 burn domains)
  7. ✅ Full trial generation (46 trials verified)
- **Next:** Run pilot mode to test multi-turn conversation flow with API calls
- **Future:** Run full experiment (46 trials × 7 models), analyze results

**Target completion: Mar 2, 2026**

---

## Implementation Complete ✅

All core infrastructure for Experiment 4 is in place and tested. The multi-turn conversation runner is the critical new component that enables Level 3 metacognition measurement. Ready for:
1. Pilot testing with actual multi-turn API calls
2. ~~Template expansion~~ ✅ **COMPLETE** (all 6 burn domains, 26 trial templates)
3. Full data collection (46 trials × 7 models = ~2,500 API calls)
4. Analysis

**Key Achievement:** Complete trial generation for all 6 burn domains (procedural, social, temporal, spatial, logical, linguistic) with 46 total trials per model including all three control types (sycophancy, recency, specificity).

**This completes the Level 0→3 experimental framework. Without Experiment 4, there is no paper.**
