# Exp3 MCI Stability Summary

- Generated: 2026-03-25T15:44:22+00:00
- Models analyzed: 13
- Stability pass: 1/13

## Criteria

- Required domain pairs: 28
- Min tasks per pair: 4
- Min shared samples for correlation: 50
- Max |MCI_spearman - MCI_kendall|: 0.15

## Per-Model

| Model | Tasks | Pairs | Min/Pair | Strata (SS/SW/WW) | MCI-S | MCI-K | Gap | Stability |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `gemma-3-12b` | 112 | 28 | 4 | 24/64/24 | 0.347 | 0.626 | 0.278 | FAIL |
| `gemma-3-27b` | 112 | 28 | 4 | 24/64/24 | 0.395 | 0.671 | 0.276 | FAIL |
| `gpt-oss-120b` | 112 | 28 | 4 | 24/64/24 | 0.356 | 0.535 | 0.179 | FAIL |
| `kimi-k2` | 112 | 28 | 4 | 40/60/12 | 0.339 | 0.678 | 0.338 | FAIL |
| `llama-3.1-405b` | 112 | 28 | 4 | 24/64/24 | 0.136 | 0.279 | 0.143 | PASS |
| `llama-3.1-70b` | 112 | 28 | 4 | 24/64/24 | 0.254 | 0.468 | 0.215 | FAIL |
| `llama-3.1-8b` | 112 | 28 | 4 | 24/64/24 | 0.215 | 0.414 | 0.200 | FAIL |
| `llama-3.2-3b` | 112 | 28 | 4 | 24/64/24 | 0.177 | 0.369 | 0.192 | FAIL |
| `llama-3.3-70b` | 112 | 28 | 4 | 24/64/24 | 0.304 | 0.624 | 0.320 | FAIL |
| `mistral-large` | 112 | 28 | 4 | 24/64/24 | 0.309 | 0.523 | 0.214 | FAIL |
| `mixtral-8x22b` | 112 | 28 | 4 | 24/64/24 | 0.241 | 0.681 | 0.441 | FAIL |
| `phi-4` | 112 | 28 | 4 | 24/64/24 | nan | nan | nan | FAIL |
| `qwen3-next-80b` | 112 | 28 | 4 | 24/64/24 | 0.418 | 0.752 | 0.334 | FAIL |

## Interpretation

- This table tests whether MCI estimates are supported by balanced pair coverage and stable rank-correlation estimators.
- If a model fails, MCI for that model should remain diagnostic/secondary rather than primary.
