# Reproducing MIRROR Paper Results

## Prerequisites

```bash
pip install -r requirements.txt
# Configure API keys in .env (see README.md)
```

## Experiment 1: Self-Knowledge Atlas

```bash
# Run on a single model
python scripts/run_experiment_1.py --models llama-3.1-8b

# Analyze results
python scripts/analyze_experiment_1.py --run-id <RUN_ID>
```

## Experiment 9: The Knowing-Doing Gap

```bash
# Pilot mode (8 tasks, quick test)
python scripts/run_experiment_9.py --mode pilot --models llama-3.1-8b

# Full mode (all 597 tasks × 4 conditions × 3 paradigms)
python scripts/run_experiment_9.py --mode full --models llama-3.1-8b

# Analyze results
python scripts/analyze_experiment_9.py --run-id <RUN_ID>
```

## Generate Paper Tables

```bash
python scripts/generate_paper_tables.py
# Outputs: paper/tables/table1_main_results.tex through table5_experiment_summary.tex
```

## Generate Figures

```bash
# Escalation curve (Figure 2)
python scripts/generate_escalation_figures.py

# KDI distribution (Figure 3)
python scripts/plot_exp9_figures.py
```

## Compile Paper

```bash
cd paper
pdflatex mirror_draft_v16 && bibtex mirror_draft_v16 && pdflatex mirror_draft_v16 && pdflatex mirror_draft_v16
```

## Temperature Ablation (Appendix M)

```bash
python scripts/temperature_ablation.py
# Runs Exp1 + Exp9 at temperature=0.7 on 3 models
# Outputs: paper/supplementary/temperature_ablation.json
```

## All Other Experiments

```bash
python scripts/run_experiment_2.py --models <MODEL>
python scripts/run_experiment_3.py --models <MODEL>
python scripts/run_experiment_4.py --models <MODEL>
python scripts/run_experiment_5.py --models <MODEL>
python scripts/run_experiment_6.py --models <MODEL>
python scripts/run_experiment_8.py  # Uses existing Exp1 data
```

## Notes

- All experiments use `temperature=0` by default for reproducibility
- Results are written as JSONL with `fsync` for crash resistance
- Use `--resume` flag to resume interrupted runs
- Concurrency: 32 parallel API calls by default
