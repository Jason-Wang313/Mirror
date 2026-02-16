# mirror-bench

Research infrastructure for **Project MIRROR** — measuring metacognitive capacity in large language models.

## Setup

1. **Install the package:**
   ```bash
   pip install -e .
   ```

2. **Configure API keys:**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

3. **Run the API test:**
   ```bash
   python tests/test_api.py
   ```

## Structure

- `mirror/api/` — Unified API client for all LLM providers
- `mirror/experiments/` — Experiment runners (8 experiments)
- `mirror/channels/` — 5 behavioral measurement channels
- `mirror/scoring/` — Scoring pipelines
- `data/` — Question bank and seed data
- `results/` — Raw experimental results and API logs
- `analysis/` — Analysis scripts and figures
- `paper/` — LaTeX paper

## Models

The project tests 8 models:
- **Scaling family:** Llama 3.1 (8B, 70B, 405B)
- **Frontier:** Gemini 2.5 Pro, DeepSeek R1
- **Diversity:** Mistral Large 3, Qwen 3 235B, GPT OSS 120B

## License

Research use only.
