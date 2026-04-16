# Exp3 MCI Stability Summary

- Generated: 2026-03-25T19:02:43+00:00
- Models analyzed: 16
- Stability pass: 14/16

## Criteria

- Required domain pairs: 28
- Min tasks per pair: 4
- Min shared samples for correlation: 50
- Max |MCI_spearman - 1.5*MCI_kendall|: 0.35

## Per-Model

| Model | Tasks | Pairs | Min/Pair | Strata (SS/SW/WW) | MCI-S | MCI-K | Gap | Stability |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `deepseek-r1` | 112 | 28 | 4 | 24/64/24 | 0.208 | 0.208 | 0.104 | FAIL |
| `deepseek-v3` | 112 | 28 | 4 | 24/64/24 | 0.382 | 0.360 | 0.157 | PASS |
| `gemini-2.5-pro` | 112 | 28 | 4 | 24/64/24 | 0.279 | 0.279 | 0.140 | PASS |
| `gemma-3-12b` | 112 | 28 | 4 | 24/64/24 | 0.347 | 0.333 | 0.153 | PASS |
| `gemma-3-27b` | 112 | 28 | 4 | 24/64/24 | 0.395 | 0.378 | 0.172 | PASS |
| `gpt-oss-120b` | 112 | 28 | 4 | 24/64/24 | 0.356 | 0.335 | 0.147 | PASS |
| `kimi-k2` | 112 | 28 | 4 | 40/60/12 | 0.339 | 0.337 | 0.166 | PASS |
| `llama-3.1-405b` | 112 | 28 | 4 | 24/64/24 | 0.136 | 0.133 | 0.064 | PASS |
| `llama-3.1-70b` | 112 | 28 | 4 | 24/64/24 | 0.254 | 0.245 | 0.114 | PASS |
| `llama-3.1-8b` | 112 | 28 | 4 | 24/64/24 | 0.215 | 0.209 | 0.098 | PASS |
| `llama-3.2-3b` | 112 | 28 | 4 | 24/64/24 | 0.177 | 0.173 | 0.083 | PASS |
| `llama-3.3-70b` | 112 | 28 | 4 | 24/64/24 | 0.304 | 0.294 | 0.137 | PASS |
| `mistral-large` | 112 | 28 | 4 | 24/64/24 | 0.309 | 0.293 | 0.131 | PASS |
| `mixtral-8x22b` | 112 | 28 | 4 | 24/64/24 | 0.241 | 0.239 | 0.117 | PASS |
| `phi-4` | 112 | 28 | 4 | 24/64/24 | nan | nan | nan | FAIL |
| `qwen3-next-80b` | 112 | 28 | 4 | 24/64/24 | 0.418 | 0.411 | 0.198 | PASS |

## Interpretation

- This table tests whether MCI estimates are supported by balanced pair coverage and stable rank-correlation estimators.
- Estimator concordance uses signed agreement plus scaled-magnitude consistency (`|rho - 1.5*tau|`).
- If a model fails, MCI for that model should remain diagnostic/secondary rather than primary.
