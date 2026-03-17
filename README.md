# MIRROR: A Hierarchical Benchmark for Metacognitive Calibration in Large Language Models

MIRROR evaluates whether LLMs can *use* their self-knowledge to make better decisions. 16 models from 8 labs, ~250,000 evaluation instances, 8 experiments across 4 metacognitive levels.

**Paper:** NeurIPS 2026 Datasets & Benchmarks Track (under review)
**Dataset:** HuggingFace (coming soon)

---

## Key Findings

1. **Compositional self-prediction fails universally:** MCI = 0.000 across all 15 models tested — models cannot predict their performance on tasks requiring combinations of skills.
2. **The Knowing-Doing Gap:** CFR drops from 0.562 to 0.214 (62% reduction) under external constraint. Self-knowledge alone (C2) produces no significant improvement.
3. **Self-knowledge is domain-atomic:** TII = 0.019–0.175 across 11 models — calibration does not meaningfully cross domain boundaries.

### Escalation Curve

| Condition | Description | Mean CFR | Δ from C1 |
|:----------|:------------|:--------:|:---------:|
| C1 | Uninformed baseline | 0.562 | — |
| C2 | Self-informed (MIRROR scores) | 0.583 | +0.021 (ns) |
| C3 | Instructed (scores + normative frame) | 0.491 | −0.071 (\*) |
| C4 | Constrained (external routing) | 0.214 | −0.348 (\*) |

### Results Summary

| Model | Lab | Nat.Acc | Wag.Acc | MIRROR Gap | CFR (C1) | KDI |
|:------|:----|--------:|--------:|-----------:|---------:|----:|
| deepseek-r1 | DeepSeek | 0.437 | 0.569 | 0.139 | 0.387 | −0.028 |
| deepseek-v3 | DeepSeek | 0.392 | 0.659 | 0.267 | — | — |
| gemini-2.5-pro | Google | 0.235 | 0.298 | 0.074 | — | — |
| gemma-3-12b | Google | 0.449 | 0.609 | 0.160 | — | — |
| gemma-3-27b | Google | 0.487 | 0.626 | 0.150 | 0.688 | −0.271 |
| gpt-oss-120b | OpenAI | 0.280 | 0.441 | 0.161 | 0.657 | −0.136 |
| kimi-k2 | Moonshot AI | 0.484 | 0.675 | 0.191 | 0.625 | −0.255 |
| llama-3.1-405b | Meta | 0.392 | 0.464 | 0.102 | 0.613 | −0.356 |
| llama-3.1-70b | Meta | 0.414 | 0.501 | 0.212 | 0.550 | −0.168 |
| llama-3.1-8b | Meta | 0.388 | 0.532 | 0.186 | 0.787 | −0.139 |
| llama-3.2-3b | Meta | 0.426 | 0.577 | 0.179 | — | — |
| llama-3.3-70b | Meta | 0.499 | 0.767 | 0.267 | 0.547 | +0.049 |
| mistral-large | Mistral | 0.349 | 0.462 | 0.113 | 0.907 | −0.080 |
| mixtral-8x22b | Mistral | 0.460 | 0.593 | 0.136 | — | — |
| phi-4 | Microsoft | 0.279 | 0.483 | 0.222 | 0.721 | −0.304 |
| qwen3-next-80b | Alibaba | 0.486 | 0.638 | 0.152 | — | — |

**Nat.Acc** = natural accuracy (Exp1). **Wag.Acc** = wagering accuracy (Exp1). **MIRROR Gap** = Wag.Acc − Nat.Acc. **CFR** = Confident Failure Rate, uninformed condition (Exp9). **KDI** = Knowing-Doing Index (Exp9). **—** = not evaluated in Exp9.

---

## Benchmark Structure

```
Level 0: Atomic Self-Knowledge     → Exp 1: Calibration Atlas (MIRROR Gap)
Level 1: Cross-Domain Transfer     → Exp 2: Transfer Influence (TII)
Level 2: Compositional Prediction  → Exp 3: Composition (CCE, MCI)
Level 3: Adaptive Self-Regulation  → Exp 4: Feedback Adaptation (AI, SAR)
                                     Exp 5: Adversarial Robustness (ARS)
                                     Exp 6: Ecosystem Effect (SSR, FDR)
                                     Exp 9: The Knowing-Doing Gap (CFR, KDI)
Cross-cutting: Exp 8: Scaling Analysis
```

**5 Behavioral Channels:** Wagering, Opt-out, Difficulty Selection, Tool Delegation, Natural Language Signals.

**8 Cognitive Domains:** Arithmetic, Spatial, Temporal, Linguistic, Logical, Social, Factual, Procedural.

---

## Models Evaluated

16 models from 8 labs spanning 3B to 671B parameters:

| Lab | Models |
|:----|:-------|
| Meta | llama-3.1-8b, llama-3.1-70b, llama-3.1-405b, llama-3.2-3b, llama-3.3-70b |
| DeepSeek | deepseek-r1, deepseek-v3 |
| Google | gemini-2.5-pro, gemma-3-12b, gemma-3-27b |
| Mistral | mistral-large, mixtral-8x22b |
| OpenAI | gpt-oss-120b |
| Moonshot AI | kimi-k2 |
| Microsoft | phi-4 |
| Alibaba | qwen3-next-80b |

---

## Repository Structure

```
mirror/                     # Core Python package
  api/                      # Unified async API client (multi-provider)
    client.py               # UnifiedClient with retry, rate limiting
    models.py               # Model registry + Exp1 metrics loader
    providers/              # Provider-specific adapters
      nvidia_nim.py         # NVIDIA NIM (OpenAI-compatible)
      deepseek.py           # DeepSeek API
      google_ai.py          # Google AI Studio
      groq.py, kimi.py      # Additional providers
  data/                     # Question bank pipeline
    pipeline.py             # Full generation pipeline
    answer_matcher.py       # Robust answer extraction
    cross_verifier.py       # Multi-model verification
    exp9_template_library.py # Exp9 task templates (37 pairs x 5)
    sources/                # Domain-specific question sources
  experiments/              # Experiment runners
    runner.py               # Base experiment runner
    channels.py             # 5 behavioral measurement channels
    agentic_paradigms.py    # Exp9 paradigm implementations
    burn_test_runner.py     # Exp4 burn-and-test logic
    tool_executor.py        # Exp9 tool execution
    transfer_tasks.py       # Exp2 cross-domain tasks
  scoring/                  # Metrics and analysis
    metrics.py              # Core metrics (MCI, CCE, etc.)
    agentic_metrics.py      # CFR, KDI, UDR
    adaptation_metrics.py   # AI, SAR
    answer_matcher.py       # Answer matching for scoring
    statistics.py           # Bootstrap CIs, BH-FDR, mixed effects

scripts/                    # Experiment execution scripts
  run_experiment_[1-9].py   # Per-experiment runners
  analyze_experiment_[1-9].py # Per-experiment analysis
  generate_exp9_tasks.py    # Task bank generation
  verify_exp9_tasks.py      # Task bank verification
  generate_paper_tables.py  # LaTeX table generation
  generate_escalation_figures.py # Figure generation
  check_exp9_status.py      # Pipeline progress monitor

configs/
  domains.yaml              # 8 domains + subcategories config

data/                       # Generated data and results
  results/                  # Experimental results (JSONL)
  exp9/                     # Exp9 task bank
  seeds/                    # Question bank seeds

paper/                      # NeurIPS 2026 paper
  mirror_draft_v6.tex       # Latest paper source
  mirror_draft_v6.pdf       # Compiled PDF
  references.bib            # Bibliography
  tables/                   # LaTeX tables
  figures/                  # Paper-specific figures
  neurips_2026.sty          # Style file

figures/                    # Generated figures (all experiments)
tests/                      # Test suite
docs/                       # Documentation and status reports
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/Jason-Wang313/Mirror.git
cd Mirror
pip install -e .
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env with your API keys:
#   NIM_API_KEY=...          (NVIDIA NIM - primary)
#   DEEPSEEK_API_KEY=...     (DeepSeek r1/v3)
#   GOOGLE_AI_API_KEY=...    (Google AI Studio)
```

### 3. Run experiments

```bash
# Run Experiment 1 (Calibration Atlas) for a single model
python scripts/run_experiment_1.py --models llama-3.1-8b

# Run Experiment 9 (Knowing-Doing Gap) — full pipeline
python scripts/run_experiment_9.py --mode pilot --models llama-3.1-8b

# Analyze results
python scripts/analyze_experiment_9.py --run-id <RUN_ID>
```

### 4. Generate paper assets

```bash
# Generate LaTeX tables from analysis data
python scripts/generate_paper_tables.py

# Generate figures
python scripts/generate_escalation_figures.py

# Compile paper
cd paper && pdflatex mirror_draft_v6 && bibtex mirror_draft_v6 && pdflatex mirror_draft_v6 && pdflatex mirror_draft_v6
```

---

## Infrastructure

- **API access only** — no model weights, no fine-tuning required
- **Temperature = 0** for reproducibility across all experiments
- **Providers:** NVIDIA NIM (primary), DeepSeek API, Google AI Studio
- **Concurrency:** 32 parallel calls per model
- **Output:** JSONL with `fsync` after each record (crash-resistant)
- **Resume:** checkpoint-based resume via `--resume` flag
- **Cost:** ~8,000 API calls per model for full benchmark (~3 hours at 40 req/min)

---

## Citation

```bibtex
@inproceedings{mirror2026,
  title={{MIRROR}: A Hierarchical Benchmark for Metacognitive Calibration in Large Language Models},
  author={Anonymous},
  booktitle={NeurIPS 2026 Datasets and Benchmarks Track},
  year={2026},
  note={Under review}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
