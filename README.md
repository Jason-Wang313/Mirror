# MIRROR: A Hierarchical Benchmark for Metacognitive Calibration in Large Language Models

**MIRROR** (Metacognitive Integrity and Recursive Reasoning in Operationalized Response) is a benchmark that evaluates whether LLMs can *use* their self-knowledge to make better decisions. We test 16 models from 8 labs across ~250,000 evaluation instances.

📄 **Paper:** NeurIPS 2026 Datasets & Benchmarks Track (under review)  
📦 **Dataset:** [HuggingFace](https://huggingface.co/datasets/Jason-Wang313/MIRROR) (coming soon)  
🏗️ **Status:** All 8 experiments complete. Paper submitted.

---

## Key Findings

**1. Compositional self-prediction fails universally.**  
The Metacognitive Convergence Index (MCI) is exactly **0.000** across all 15 models tested. Models that calibrate accurately in individual domains completely fail to compose those calibrations for multi-domain tasks.

**2. The Knowing-Doing Gap.**  
Models possess accurate self-knowledge but don't act on it. External architectural constraint reduces the Confident Failure Rate (CFR) from **0.562 to 0.214** — a **62% reduction**. Providing models with their own calibration scores produces no significant improvement (*p* > 0.05).

**3. Self-knowledge is domain-atomic.**  
Transfer Influence Index (TII) ranges from 0.019 to 0.175 across 11 models. Calibration does not meaningfully cross domain boundaries.

---

## Results Summary

### Main Results (Table 3 in paper)

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

**Nat.Acc** = natural accuracy (Exp1). **Wag.Acc** = wagering accuracy (Exp1). **MIRROR Gap** = Wag.Acc − Nat.Acc (overconfidence). **CFR** = Confident Failure Rate, uninformed condition (Exp9). **KDI** = Knowing-Doing Index (Exp9). "—" = model not evaluated in Exp9.

### Experiment 9: Escalation Curve

| Condition | Description | Mean CFR | Δ from C1 |
|:----------|:------------|:--------:|:---------:|
| C1 | Uninformed baseline | 0.562 | — |
| C2 | Self-informed (MIRROR scores) | 0.583 | +0.021 (ns) |
| C3 | Instructed (scores + normative frame) | 0.491 | −0.071 (*) |
| C4 | Constrained (external routing) | 0.214 | −0.348 (*) |

Self-knowledge alone (C2) adds nothing. Only external constraint (C4) works.

---

## Benchmark Structure

MIRROR decomposes LLM metacognition into **4 levels** × **5 behavioral channels** × **8 experiments**:

```
Level 0: Atomic Self-Knowledge     → Exp 1: Calibration Atlas
Level 1: Cross-Domain Transfer     → Exp 2: Transfer Influence (TII)
Level 2: Compositional Prediction  → Exp 3: Composition (CCE, MCI)
Level 3: Adaptive Self-Regulation  → Exp 4: Feedback Adaptation (AI, SAR)
                                   → Exp 5: Adversarial Robustness (ARS)
                                   → Exp 6: Ecosystem Effect (SSR, FDR)
                                   → Exp 9: Knowing-Doing Gap (CFR, KDI)
Cross-cutting: Exp 8: Scaling Analysis
```

**5 Behavioral Channels:** Wagering, Opt-out, Difficulty Selection, Tool Delegation, Natural Language Signals.

---

## Models Evaluated

16 models from 8 labs (3B to >1T parameters):

| Lab | Models |
|:----|:-------|
| DeepSeek | deepseek-r1, deepseek-v3 |
| Google | gemini-2.5-pro, gemma-3-12b, gemma-3-27b |
| OpenAI | gpt-oss-120b |
| Moonshot AI | kimi-k2 |
| Meta | llama-3.1-8b, llama-3.1-70b, llama-3.1-405b, llama-3.2-3b, llama-3.3-70b |
| Mistral | mistral-large, mixtral-8x22b |
| Microsoft | phi-4 |
| Alibaba | qwen3-next-80b |

---

## Repository Structure

```
Mirror/
├── README.md
├── paper/
│   ├── mirror_draft_v6.tex          # NeurIPS 2026 D&B submission
│   ├── references.bib
│   ├── figures/
│   │   ├── exp9_escalation_curve_with_ci.png
│   │   ├── fig3_kdi_distribution.png
│   │   ├── fig2_money_plot.png
│   │   ├── exp8_hero_figure.png
│   │   ├── exp9_escalation_per_paradigm.png
│   │   └── figure1_mirror_gradient.png
│   └── tables/
│       ├── table1_main_results.tex
│       └── table2_comparison.tex
├── experiments/
│   ├── exp1_self_knowledge_atlas/
│   ├── exp2_cross_domain_transfer/
│   ├── exp3_compositional_prediction/
│   ├── exp4_adaptation_crucible/
│   ├── exp5_adversarial_robustness/
│   ├── exp6_ecosystem_effect/
│   ├── exp8_scaling_analysis/
│   └── exp9_knowing_doing_gap/
├── analysis/
│   ├── generate_figures.py
│   └── compute_metrics.py
├── data/
│   ├── exp1/                        # Raw JSONL results
│   ├── exp2/
│   ├── ...
│   └── exp9/
├── prompts/                         # All evaluation prompts
│   ├── exp1_templates/
│   ├── ...
│   └── exp9_templates/
└── eval/                            # pip-installable evaluation suite
    ├── setup.py
    ├── mirror_eval/
    │   ├── __init__.py
    │   ├── run_benchmark.py
    │   └── scoring.py
    └── README.md
```

---

## Quick Start

### Evaluate a new model

```bash
pip install mirror-eval  # coming soon

mirror-eval --model your-model-name --api-key $API_KEY --experiments all
```

Running the full benchmark requires ~8,000 API calls (~3 hours at 40 req/min). Only API access is needed — no weights, no activations, no fine-tuning.

### Reproduce results

```bash
git clone https://github.com/Jason-Wang313/Mirror.git
cd Mirror

# Install dependencies
pip install -r requirements.txt

# Run analysis on existing data
python analysis/compute_metrics.py --experiment all
python analysis/generate_figures.py
```

---

## Infrastructure

All experiments use `temperature=0` for reproducibility.

| API Provider | Models |
|:-------------|:-------|
| NVIDIA NIM (free tier) | Llama family, Mistral, Gemma-3-27b, Phi-4, Kimi-K2, GPT-OSS-120b, Qwen3 |
| DeepSeek API | deepseek-r1, deepseek-v3 |
| Google AI Studio | gemini-2.5-pro, gemma-3-12b |

Concurrency: 32 parallel calls per model. All results stored as JSONL with `fsync` after each record for crash resistance.

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

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

For questions about the benchmark, open an issue or contact the authors (details after de-anonymization).
