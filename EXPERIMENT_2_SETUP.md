# Experiment 2: Cross-Domain Transfer Tests — Implementation Summary

## Overview

Experiment 2 tests whether models' self-knowledge in domain A transfers to behavioral caution on tasks that implicitly require domain A skills but are framed as domain B.

**Status:** ✅ Infrastructure complete, ready for pilot testing

## Files Created

### 1. `mirror/experiments/transfer_tasks.py`
**Purpose:** Task generation and validation pipeline

**Features:**
- 8 domain pairs with hand-crafted seed examples
- Few-shot prompting using DeepSeek R1 for task generation
- Cross-verification with Llama 70B
- Support for adversarial disguise tasks

**Domain Pairs:**
| Source Domain | Surface Domain | Hidden Dependency |
|---|---|---|
| Arithmetic | Business Evaluation | Financial calculations |
| Spatial | Trip Planning | Distance estimation |
| Temporal | Meeting Scheduling | Timezone arithmetic |
| Linguistic | Document Proofreading | Grammar parsing |
| Logical | Argument Analysis | Inference validity |
| Social | Email Composition | Pragmatic implicature |
| Factual | Fact Checking | Cross-referencing claims |
| Procedural | Troubleshooting | Multi-step diagnosis |

**Usage:**
```bash
# Generate pilot tasks (5 per domain pair = 40 total)
python -m mirror.experiments.transfer_tasks --generate --pilot

# Generate full tasks (25 per domain pair = 200 total)
python -m mirror.experiments.transfer_tasks --generate --n-per-pair 25
```

### 2. `scripts/run_experiment_2.py`
**Purpose:** Orchestration script for data collection

**Modes:**
- **test**: 1 model × 5 tasks × 2 channels (smoke test)
- **pilot**: 2 models × 25 tasks × 5 channels
- **full**: 7 models × 230 tasks × 6 channels (5 behavioral + Layer 2)

**Channels (adapted for transfer tasks):**
1. **Wagering (Ch1)**: Bet 1-10 on answer correctness
2. **Opt-out (Ch2)**: Answer for points or skip
3. **Difficulty selection (Ch3)**: Choose variant A (reduced dependency) or B (full dependency)
4. **Tool use (Ch4)**: Calculator + "ask expert" option
5. **Natural (Ch5)**: Free response with implicit signal extraction
6. **Layer 2**: Verbal self-report on sub-skills and confidence

**Usage:**
```bash
# Smoke test
python scripts/run_experiment_2.py --mode test

# Pilot run
python scripts/run_experiment_2.py --mode pilot

# Full run
python scripts/run_experiment_2.py --mode full

# Resume from checkpoint
python scripts/run_experiment_2.py --mode full --resume --run-id <RUN_ID>
```

### 3. `scripts/analyze_experiment_2.py`
**Purpose:** Analysis script for transfer-specific metrics

**Metrics Computed:**

1. **Transfer Score (per channel, per model)**
   - Spearman correlation between domain weakness (from Exp1) and behavioral caution
   - Positive correlation = transfer exists

2. **Transfer MCI**
   - Mean pairwise correlation between channel transfer scores
   - Measures cross-channel convergence

3. **Verbal Transfer Score**
   - Proportion of tasks where model correctly identified source domain skill AND flagged uncertainty
   - From Layer 2 responses

4. **Dissociation Index**
   - Verbal transfer - Behavioral transfer
   - Positive = can articulate but not act
   - Negative = acts without articulating

**Usage:**
```bash
python scripts/analyze_experiment_2.py --run-id <EXP2_RUN_ID> --exp1-run-id 20260220T090109
```

### 4. `tests/test_transfer_tasks.py`
**Purpose:** Test suite for transfer task generation

**Tests:**
- Domain pairs structure validation
- Task loading from JSONL
- Task ID uniqueness
- Adversarial task structure
- All 8 domains covered

**Usage:**
```bash
python tests/test_transfer_tasks.py
```

## Expected Predictions from Experiment 1

Based on Experiment 1 results (`20260220T090109`), we expect:

1. **DeepSeek R1** (highest MCI 0.41, Wager-ρ 0.41)
   - **Prediction:** Strongest transfer — channels already converge
   - Should show positive transfer scores across all channels

2. **Llama 3.1-8B** (best meta-accuracy 0.74, worst calibration)
   - **Prediction:** High verbal transfer, low behavioral transfer
   - Dissociation Index should be positive (articulates > acts)

3. **Llama 3.1-405B** (good Wager-ρ 0.38, low MCI 0.12)
   - **Prediction:** Transfer in wagering channel only, fragmented elsewhere
   - Mixed transfer scores across channels

## Workflow

### Phase 1: Task Generation
```bash
# Generate pilot tasks
python -m mirror.experiments.transfer_tasks --generate --pilot

# Verify generation
python -c "from mirror.experiments.transfer_tasks import load_transfer_tasks; \
           tasks = load_transfer_tasks(); \
           print(f'Generated {len(tasks)} tasks')"
```

### Phase 2: Data Collection
```bash
# Run pilot experiment
python scripts/run_experiment_2.py --mode pilot

# Check results
ls -lh data/results/exp2_*_results.jsonl
```

### Phase 3: Analysis
```bash
# Analyze results
python scripts/analyze_experiment_2.py --run-id <RUN_ID>

# View output
cat data/results/exp2_<RUN_ID>_transfer_analysis.json
```

## Success Criteria

Experiment 2 succeeds if it demonstrates **at least ONE of**:

1. **Transfer exists**: Significant correlation (p < 0.05) between domain weakness and behavioral caution, across ≥2 channels
2. **Transfer is absent**: No models show significant transfer (self-knowledge is atomic)
3. **Transfer is dissociated**: Behavioral and verbal transfer diverge

All three outcomes are publishable. The design avoids noisy null results through:
- 25 tasks per condition (detects Cohen's d > 0.8 at 80% power)
- 5-channel convergence design
- Control for surface framing via adversarial disguise tasks

## Data Flow

```
Experiment 1 (20260220T090109)
    ↓
domain_accuracy.json (per-model domain profiles)
    ↓
Transfer Tasks Generation
    ↓
data/transfer_tasks.jsonl (200 tasks)
    ↓
Experiment 2 Data Collection
    ↓
exp2_<RUN_ID>_results.jsonl
    ↓
Transfer Analysis
    ↓
exp2_<RUN_ID>_transfer_analysis.json
    ↓
Summary Table
```

## Next Steps

1. **Test task generation:**
   ```bash
   python -m mirror.experiments.transfer_tasks --generate --pilot
   ```

2. **Run smoke test:**
   ```bash
   python scripts/run_experiment_2.py --mode test
   ```

3. **If smoke test passes, run pilot:**
   ```bash
   python scripts/run_experiment_2.py --mode pilot
   ```

4. **Analyze pilot results:**
   ```bash
   python scripts/analyze_experiment_2.py --run-id <PILOT_RUN_ID>
   ```

5. **If pilot looks good, run full experiment:**
   ```bash
   python scripts/run_experiment_2.py --mode full
   ```

## Estimated Execution Time

- Task generation (pilot): ~5 minutes (40 tasks)
- Task generation (full): ~30 minutes (200 tasks)
- Smoke test: ~2 minutes (1 model × 5 tasks × 2 channels)
- Pilot run: ~30 minutes (2 models × 25 tasks × 5 channels)
- Full run: ~4-6 hours (7 models × 230 tasks × 6 channels)
- Analysis: < 1 minute

## Implementation Complete ✅

All infrastructure is in place for Experiment 2. The code is tested and ready for:
- Pilot testing to validate task quality
- Full data collection
- Transfer metric computation
- Publication-ready analysis

The experiment directly tests Brief 05's central hypothesis: **does self-knowledge transfer across domain boundaries, or is LLM metacognition atomic and context-specific?**
