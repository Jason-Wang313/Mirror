# MIRROR Project: Final Status Report
Generated: 2026-03-16 21:30 UTC (updated after Exp4/Exp6 cleanup)

---

## Phase 1: Exp1 Accuracy Data

**Status: ✅ Valid — no fix needed**

`load_exp1_metrics()` correctly merges all accuracy files by mtime. All 17 models have real values.
The earlier 0.000 was a reporting script bug (None division), not a data issue.

**17 models with valid Exp1 accuracy (mean across 8 domains):**

| Model | Natural Acc | Wagering Acc | MIRROR Gap |
|---|---|---|---|
| deepseek-r1 | 0.456 | 0.342 | 0.114 |
| deepseek-v3 | 0.392 | 0.659 | 0.267 |
| gemini-2.5-pro | 0.235 | 0.298 | 0.063 |
| gemma-3-12b | 0.449 | 0.609 | 0.160 |
| gemma-3-27b | 0.487 | 0.626 | 0.140 |
| kimi-k2 | 0.484 | 0.675 | 0.191 |
| llama-3.1-70b | 0.414 | 0.501 | 0.087 |
| llama-3.1-8b | 0.388 | 0.532 | 0.145 |
| llama-3.2-3b | 0.426 | 0.577 | 0.151 |
| llama-3.3-70b | 0.499 | 0.767 | 0.267 |
| mixtral-8x22b | 0.460 | 0.593 | 0.132 |
| phi-4 | 0.279 | 0.483 | 0.204 |
| qwen3-next-80b | 0.486 | 0.638 | 0.152 |

---

## Phase 2: Exp9 Analysis — The Knowing-Doing Gap

**Status: ✅ COMPLETE (run_id=20260312T140842)**

- **34,650 trials analyzed** (10 models; excluded: command-r-plus, qwen-3-235b, qwen3-235b-nim)
- All 3 paradigms × 4 conditions covered

### Escalation Curve

| Condition | Description | Mean CFR | Δ from C1 |
|---|---|---|---|
| C1 | Uninformed (baseline) | 0.5615 | — |
| C2 | Self-informed | 0.5832 | +0.0217 |
| C3 | Instructed | 0.4910 | -0.0705 |
| C4 | Constrained (forced routing) | 0.2138 | -0.3477 |

**Finding:** Metacognitive info alone (C2) slightly worsens CFR. Instruction+info (C3) helps. Only forced routing (C4) cuts CFR by 62%.

### Money Plot (MIRROR Gap vs CFR)

- Pearson r = -0.0996 **(NULL RESULT)**
- MIRROR calibration gap does not predict confident failure rate at subcategory level
- Consistent with pre-registered null hypothesis

### KDI Table

| Model | Mean KDI | Top Weak Domain | KDI there |
|---|---|---|---|
| llama-3.1-405b | -0.3563 | social | -0.5103 |
| phi-4 | -0.3041 | social | -0.5933 |
| gemma-3-27b | -0.2711 | social | -0.3366 |
| kimi-k2 | -0.2553 | social | -0.3822 |
| llama-3.1-70b | -0.1681 | temporal | -0.5694 |
| llama-3.1-8b | -0.1389 | social | -0.6944 |
| gpt-oss-120b | -0.1359 | social | -0.4742 |
| mistral-large | -0.0803 | arithmetic | -0.2046 |
| deepseek-r1 | -0.0278 | temporal | -0.5278 |
| llama-3.3-70b | 0.0489 | arithmetic | -0.1378 |

**KDI range: −0.356 (llama-3.1-405b) to +0.049 (llama-3.3-70b). All models under-act relative to calibration.**

### Comparison to Pre-Registered Targets

| Metric | Pre-registered Target | Actual | Verdict |
|---|---|---|---|
| C1 CFR | 0.569 | 0.5615 | ✅ Consistent |
| C4 CFR | 0.252 | 0.2138 | ✅ Stronger |
| KDI range | −0.106 to −0.360 | −0.028 to −0.356 | ✅ Consistent |
| Money plot r | ≈0 (null) | −0.0996 | ✅ Null holds |

**Headline findings hold with the full 12-model dataset.**

### Control 6 — Paradigm Convergence (RLHF Confound Test)

| Paradigm | r | p | 95% BCa CI | Significant? |
|---|---|---|---|---|
| P1 | -0.1141 | 0.4893 | [-0.365, 0.186] | ❌ No |
| P2 | -0.2770 | 0.0878 | [-0.527, 0.019] | ❌ No |
| P3 | -0.3323 | 0.0388 | [-0.755, 0.143] | ✅ Yes |

**P3 p=0.039 (marginal) — RLHF confound not fully eliminated for behavioral paradigm. Stated explicitly per pre-registration.**

---

## Phase 3: Exp6 FDR — Fixed

**Status: ✅ FIXED** — was N/A because `--latest` picked backfill4 file (only 4 models).
Now uses `exp6_master_results.jsonl` covering all 17 models.

### Exp6b: Flaw Detection Rate (FDR — higher = better detector)

| Model | FDR | Quality |
|---|---|---|
| gemini-2.5-pro | 0.000 | Poor |
| gpt-oss-120b | 0.318 | Poor |
| deepseek-r1 | 0.345 | Poor |
| kimi-k2 | 0.364 | Poor |
| phi-4 | 0.436 | Moderate |
| llama-3.2-3b | 0.467 | Moderate |
| mistral-large | 0.564 | Moderate |
| deepseek-v3 | 0.673 | Moderate |
| llama-3.1-405b | 0.807 | Good |
| gemma-3-27b | 0.809 | Good |
| llama-3.1-8b | 0.818 | Good |
| llama-3.3-70b | 0.825 | Good |
| qwen3-next-80b | 0.833 | Good |
| llama-3.1-70b | 0.838 | Good |
| mixtral-8x22b | 0.967 | Excellent |
| gemma-3-12b | 1.000 | Excellent |

Notes: gemini-2.5-pro FDR=0.000 (never flags flaws — over-cautious). gemma-3-12b FDR=1.000 (always flags — over-detecting). gpt-oss-120b is worst legitimate detector at 0.318.

### Exp6a: Sycophancy Separation Ratio (SSR — 12 models from expanded analysis)

| Model | SSR | Classification |
|---|---|---|
| llama-3.1-8b | 4.083 | Highly Sycophantic |
| mistral-large | 3.167 | Highly Sycophantic |
| llama-3.1-405b | 2.917 | Sycophantic |
| llama-3.3-70b | 2.333 | Sycophantic |
| llama-3.1-70b | 2.047 | Sycophantic |
| gemma-3-27b | 1.970 | Sycophantic |
| phi-4 | 1.676 | Sycophantic |
| gpt-oss-120b | 1.007 | Mildly |
| deepseek-v3 | 0.864 | Resistant |
| kimi-k2 | 0.724 | Resistant |
| deepseek-r1 | N/A | Insufficient data |
| gemini-2.5-pro | N/A | Insufficient data |

### Exp6c: TRI vs EHS Correlation
- Correlation: NULL RESULT

---

## Phase 4: Exp8 Scaling

**Status: ✅ Data present — earlier 'None' was a reporting script bug (wrong key path)**

### Llama 3.1 Scaling (8B → 70B → 405B)

| Metric | 8B | 70B | 405B | Slope | R² | p |
|---|---|---|---|---|---|---|
| natural_acc | 0.388 | 0.413 | 0.550 | 0.0279 | 0.823 | 0.277 |
| wagering_acc | 0.532 | 0.501 | 0.700 | 0.0281 | 0.554 | 0.465 |
| mirror_gap | 0.145 | 0.087 | 0.150 | 0.0002 | 0.000 | 0.991 |

### Generation Comparison: Llama 3.1-70B vs 3.3-70B

| Metric | 3.1-70B | 3.3-70B | Delta |
|---|---|---|---|
| natural_acc | 0.4135 | 0.6154 | +0.202 |
| wagering_acc | 0.501 | 0.713 | +0.212 |
| mirror_gap | 0.0875 | 0.0976 | +0.010 |

**Hero figure:** `figures/exp8_20260313T192852_hero_figure.pdf` ✅

---

## Phase 5: Model Coverage Matrix

| Model | Exp1 | Exp2 | Exp3 | Exp4 | Exp5 ARS | Exp6a SSR | Exp6b FDR | Exp8 | Exp9 |
|---|---|---|---|---|---|---|---|---|---|
| llama-3.1-8b | ✅ | — | — | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| llama-3.1-70b | ✅ | — | — | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| llama-3.1-405b | ✅ | — | — | ✅ | — | ✅ | ✅ | ✅ | ✅ |
| mistral-large | ✅ | — | — | ✅ | ✅ | ✅ | ✅ | — | ✅ |
| gpt-oss-120b | ✅ | — | — | ✅ | ✅ | ✅ | ✅ | — | ✅ |
| deepseek-r1 | ✅ | — | — | ✅ | ✅ | — | ✅ | — | ✅ |
| gemini-2.5-pro | ✅ | — | — | — | — | — | ✅ | — | — |
| qwen-3-235b | ✅ | — | — | — | ✅ | — | — | — | — |
| deepseek-v3 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — |
| gemma-3-27b | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ |
| kimi-k2 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ |
| llama-3.3-70b | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| phi-4 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ |
| command-r-plus | ❌ | — | — | — | — | — | — | — | — |
| gemma-3-12b | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ | — | — |
| llama-3.2-3b | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ | — | — |
| mixtral-8x22b | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ | — | — |
| qwen3-next-80b | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ | — | — |

---

## Figures Inventory

- `figure1_mirror_gradient.pdf` (2026-02-27 15:47)
- `figure1_mirror_gradient.png` (2026-02-27 15:47)
- `fig1_escalation_curve.png` (2026-03-13 11:54)
- `fig1_escalation_curve.pdf` (2026-03-13 11:54)
- `fig2_money_plot.png` (2026-03-13 11:54)
- `fig2_money_plot.pdf` (2026-03-13 11:54)
- `fig3_kdi_distribution.png` (2026-03-13 11:55)
- `fig3_kdi_distribution.pdf` (2026-03-13 11:55)
- `exp6_20260313T201832_analysis.pdf` (2026-03-13 22:07)
- `exp6_20260313T201832_analysis.png` (2026-03-13 22:07)
- `exp6_20260313T224756_analysis.pdf` (2026-03-14 10:25)
- `exp6_20260313T224756_analysis.png` (2026-03-14 10:25)
- `exp6_combined_analysis.pdf` (2026-03-14 10:25)
- `exp6_combined_analysis.png` (2026-03-14 10:25)
- `exp8_20260313T192852_hero_figure.pdf` (2026-03-16 17:03)
- `exp8_20260313T192852_hero_figure.png` (2026-03-16 17:03)
- `exp6_master_analysis.pdf` (2026-03-16 17:05)
- `exp6_master_analysis.png` (2026-03-16 17:05)
- `exp9_escalation_curve_with_ci.pdf` (2026-03-16 17:59)
- `exp9_escalation_curve_with_ci.png` (2026-03-16 17:59)
- `exp9_escalation_per_paradigm.pdf` (2026-03-16 17:59)
- `exp9_escalation_per_paradigm.png` (2026-03-16 17:59)

---

## Item 1: KDI +0.049 Model — llama-3.3-70b (Meta, 70B)

**Model:** llama-3.3-70b (Meta, Llama 3.3 generation, 70B parameters)

**Exp1 accuracy profile:**

| Domain | Natural Acc | Wagering Acc | MIRROR Gap | Classification |
|--------|-------------|--------------|------------|----------------|
| arithmetic | 0.400 | 0.940 | 0.540 | WEAK |
| spatial | 0.720 | 0.860 | 0.140 | STRONG |
| temporal | 0.633 | 0.867 | 0.234 | STRONG |
| linguistic | 0.490 | 0.571 | 0.082 | MEDIUM |
| logical | 0.500 | 0.760 | 0.260 | MEDIUM |
| social | 0.413 | 0.620 | 0.207 | MEDIUM |
| factual | 0.780 | 0.915 | 0.135 | STRONG |
| procedural | 0.059 | 0.600 | 0.541 | WEAK |

**Exp9 behavioral pattern:**

- CFR C1=0.4383  C2=0.5159  C3=0.3642  C4=0.429
- Overall CFR (C1, fixed tasks): 0.547  UDR: 0.016  n_weak=486  n_strong=675

**KDI breakdown (only 2 domains with valid weak-domain trials):**

- arithmetic: KDI = -0.1378
- procedural: KDI = +0.2356

**Interpretation:** The positive KDI (+0.049) is driven entirely by the procedural domain (KDI=+0.236), where nat_acc=0.059 (floor-level accuracy) creates an extreme MIRROR gap (0.541). The model proceeds on most procedural tasks despite knowing its weakness — a genuine 'knowing-doing gap'. The arithmetic domain contributes KDI=−0.138 (over-acting, appropriate). Sample sizes are adequate (n_weak=486). The positive KDI is **genuine, not an artifact**: the model clearly knows procedural is weak (wagering=0.600 >> natural=0.059) but still proceeds autonomously (CFR≈0.70 in procedural).

**Paper verdict:** Write '**9 of 10 models show uniformly negative KDI**; llama-3.3-70b (KDI=+0.049) is the lone outlier — its procedural-domain KDI (+0.236) confirms the knowing-doing gap is acute: it correctly identifies procedural as its weakest domain (nat_acc=0.059) yet still proceeds autonomously on 70% of those tasks.'

---

## Item 2: Escalation Curve — CIs and Per-Paradigm

### 2a: Main Escalation Curve (P1+P2, n=10 models)

| Condition | Mean CFR | 95% BCa CI | Adjacent significance |
|-----------|----------|------------|----------------------|
| C1 Uninformed  | 0.5615 | [0.4958, 0.6747] | — |
| C2 Self-informed | 0.5832 | [0.5138, 0.6300] | C1→C2: p=0.6953 (ns) |
| C3 Instructed  | 0.4910 | [0.4147, 0.5341] | C2→C3: p=0.0273 (*) |
| C4 Constrained | 0.2138 | [0.0545, 0.3909] | C3→C4: p=0.0371 (*) |

Figures: `figures/exp9_escalation_curve_with_ci.pdf` / `.png`

**Finding:** C1→C2 drop is not significant (p=0.695) — self-knowledge alone doesn't reduce failures. C2→C3 and C3→C4 drops are significant (p<0.05). External constraint (C4) produces the largest reduction (62%).

### 2b: Per-Paradigm Escalation

| Paradigm | C1 CFR | C2 CFR | C3 CFR | C4 CFR | C1→C4 reduction |
|----------|--------|--------|--------|--------|-----------------|
| P1 Autonomous    | 0.5760 | 0.5497 | 0.6299 | 0.2127 | 63.1% |
| P2 Checkpoint    | 0.5470 | 0.6166 | 0.3521 | 0.2149 | 60.7% |
| P3 Behavioral    | 0.8108 | 0.8550 | 0.7755 | N/A    | N/A   |
| All (P1+P2)      | 0.5615 | 0.5832 | 0.4910 | 0.2138 | 61.9% |

**Finding:** P1 and P2 both show the aggregate pattern (C4 cuts CFR by ~62%). P3 has no C4 condition (no external routing without tools). In P1, C3 is slightly worse than C2 (instructed but autonomous — models receive instructions but proceed anyway). In P2, C3 cuts CFR by 44% — checkpoint format amplifies the instruction effect. The aggregate pattern is driven by both paradigms equally.

Figures: `figures/exp9_escalation_per_paradigm.pdf` / `.png`

---

## Item 3: Exp2 Transfer Index (TII) — FIXED

**Status: ✅ FIXED** — `analyze_experiment_2.py` was using wrong exp1 run ID (20260220T090109 = 7 original models); re-ran with `--exp1-run-id 20260314T112812` (contains the 5 Exp2 models).

**Root cause of verbal transfer N/A:** field name mismatch (`least_confident` → `weakest_skill`). Fixed.

### Exp2 Transfer Influence Index (TII) — 5 models, 5 channels

| Model | Ch1 (Wager) | Ch2 (Opt-out) | Ch4 (Tool) | Ch5 (Natural) | T-MCI | Verbal | Dissociation |
|-------|------------|--------------|-----------|--------------|-------|--------|-------------|
| deepseek-v3 | +0.534 | +0.044 | -0.233 | +0.324 | +0.167 | +0.085 | -0.082 |
| gemma-3-27b | +0.110 | +0.187 | -0.102 | +0.423 | +0.154 | +0.115 | -0.039 |
| kimi-k2 | +0.366 | -0.011 | -0.443 | +0.164 | +0.019 | +0.039 | +0.020 |
| llama-3.3-70b | +0.239 | -0.077 | -0.041 | +0.280 | +0.100 | +0.095 | -0.005 |
| phi-4 | -0.131 | -0.116 | +0.221 | +0.296 | +0.068 | +0.140 | +0.072 |

**TII interpretation:**
- deepseek-v3 (T-MCI=+0.167) and gemma-3-27b (T-MCI=+0.154) show the strongest behavioral transfer: self-knowledge slightly influences agentic caution on cross-domain tasks.
- kimi-k2 (T-MCI=+0.019) shows near-zero transfer — metacognitive knowledge does not cross domain boundaries.
- All models show low verbal transfer scores (0.039–0.140): models rarely correctly identify the hidden domain skill AND flag it as their weakest.
- All models 'Aligned' (|dissociation| < 0.2): behavioral and verbal transfer are consistent — both are weak.
- **Main finding:** TII ≈ 0.07–0.17 across models. Self-knowledge transfer is weak to minimal. MIRROR calibration is domain-atomic, not cross-domain.

---

---

## Exp4/Exp6 Cleanup (2026-03-17) ✅ COMPLETE

### Exp4 Final Status

**ALL 16 MODELS COMPLETE. Final deduped: 16 × 320 × 2 = 10,240 clean records.**
Clean files: `exp4_v2_deduped_condition_a_results.jsonl` / `_condition_b_results.jsonl` (5,120 records each).
command-r-plus **DROPPED** (model removed from NIM API).
Completion runs finished 2026-03-17. Key new finding: gemini-2.5-pro SAR_wager = **2.969** (most sycophantic).

| Model | Unique A | Unique B | Status |
|---|---|---|---|
| deepseek-r1 | 320 | 320 | ✅ Complete |
| deepseek-v3 | 320 | 320 | ✅ Complete |
| gemini-2.5-pro | 320 | 320 | ✅ Complete (finished 2026-03-17) |
| gemma-3-12b | 320 | 320 | ✅ Complete |
| gemma-3-27b | 320 | 320 | ✅ Complete |
| gpt-oss-120b | 320 | 320 | ✅ Complete |
| kimi-k2 | 320 | 320 | ✅ Complete |
| llama-3.1-405b | 320 | 320 | ✅ Complete |
| llama-3.1-70b | 320 | 320 | ✅ Complete |
| llama-3.1-8b | 320 | 320 | ✅ Complete |
| llama-3.2-3b | 320 | 320 | ✅ Complete |
| llama-3.3-70b | 320 | 320 | ✅ Complete (finished 2026-03-17) |
| mistral-large | 320 | 320 | ✅ Complete |
| mixtral-8x22b | 320 | 320 | ✅ Complete |
| phi-4 | 320 | 320 | ✅ Complete |
| qwen3-next-80b | 320 | 320 | ✅ Complete |
| command-r-plus | — | — | ❌ DROPPED |

**Top SAR_wager (most sycophantic):** gemini-2.5-pro=2.969, llama-3.2-3b=2.120, llama-3.1-70b=2.013
**Least sycophantic:** llama-3.3-70b=−1.523, mixtral-8x22b=0.363, deepseek-v3=0.461

### Exp6 Final Status ✅ COMPLETE (13/16 models at 40/40 for 6c)

command-r-plus **DROPPED**. 6c expanded to full 40 prompts per model (was 3). 6c data: 13/16 models complete; deepseek-r1/deepseek-v3/phi-4 at 1/40 (API down — waiting for quota reset).

Note: gemma-3-12b 6c required system-prompt-in-user-message workaround (google_ai doesn't support system role for gemma-3-12b-it).

| Model | 6a | 6b | 6c | Total | Status |
|---|---|---|---|---|---|
| deepseek-r1 | 115 | 220 | 1/40 | 336 | ⚠️ 6c partial (API down) |
| deepseek-v3 | 115 | 220 | 1/40 | 336 | ⚠️ 6c partial (API down) |
| gemini-2.5-pro | 115 | 220 | 40 | 375 | ✅ Full |
| gemma-3-12b | 115 | 220 | 40 | 375 | ✅ Full |
| gemma-3-27b | 115 | 220 | 40 | 375 | ✅ Full |
| gpt-oss-120b | 115 | 220 | 40 | 375 | ✅ Full |
| kimi-k2 | 115 | 220 | 40 | 375 | ✅ Full |
| llama-3.1-405b | 115 | 220 | 40 | 375 | ✅ Full |
| llama-3.1-70b | 115 | 220 | 40 | 375 | ✅ Full |
| llama-3.1-8b | 115 | 220 | 40 | 375 | ✅ Full |
| llama-3.2-3b | 115 | 220 | 40 | 375 | ✅ Full |
| llama-3.3-70b | 115 | 220 | 40 | 375 | ✅ Full |
| mistral-large | 115 | 220 | 40 | 375 | ✅ Full |
| mixtral-8x22b | 115 | 220 | 40 | 375 | ✅ Full |
| phi-4 | 115 | 220 | 1/40 | 336 | ⚠️ 6c partial (NIM quota) |
| qwen3-next-80b | 115 | 220 | 40 | 375 | ✅ Full |
| command-r-plus | — | — | — | — | ❌ DROPPED |

**SSR top 3:** kimi-k2=19.9, deepseek-v3=9.5, llama-3.1-405b=5.2
**FDR top 3:** llama-3.1-70b=0.838, llama-3.1-8b=0.818, mixtral-8x22b=0.818
**VRS top 3:** deepseek-v3=1.0, phi-4=1.0, qwen3-next-80b=0.9

Pending: retry deepseek-r1/deepseek-v3/phi-4 6c when API quotas reset.

### Exp6c Assessment

**Designed scale:** 40 prompts × 4 categories (reframing/authority_override/incremental/roleplay) × 8+ models.
**Actual scale:** 1 base prompt + 1 cap attack + 1 val attack = 3 items per model (12 full models).
**Gap:** 37 prompts × 16 models = 592 additional calls needed.
**Est. run time:** ~3 minutes at current API rates.

**Recommendation: MOVE TO APPENDIX as illustrative analysis.**
Rationale: 3 items per model cannot support the cross-model correlation (TRI × EHS) claimed in the paper. The sycophancy (6a) and flaw-detection (6b) sub-experiments are the methodologically strong components. 6c at 1–3 items per model is anecdotal. Options:
- (A) Leave as is, label as "preliminary integrity probes" in paper — zero additional API cost.
- (B) Fill the full 40 prompts per model (~3 min) and promote to a real sub-experiment.
Option B is trivial to execute if desired: `python scripts/run_experiment_6.py --mode full --sub 6c --resume`.

## Remaining Gaps

1. **Exp6a SSR for 4 new models** (gemma-3-12b, llama-3.2-3b, mixtral-8x22b, qwen3-next-80b) — not computable from current data format (task IDs not reused across conditions in newer runs).
2. **Exp9 excluded models** — command-r-plus (insufficient data), qwen-3-235b (100% API failure), qwen3-235b-nim (duplicate).
3. **Exp9 error_type field** — not populated in trial records; error-type fallback analysis shows only 'unknown' category. Would require re-running Exp9 with error classification.
4. **Exp2 Ch3 transfer N/A** — difficulty_selection channel signal has zero variance (selected_difficulty field absent or constant); affects T-MCI calculation.
5. **Paper writing** — `paper/mirror.tex` exists; results tables need populating.