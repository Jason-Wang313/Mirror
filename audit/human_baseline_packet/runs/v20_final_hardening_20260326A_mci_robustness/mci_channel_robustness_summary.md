# MCI Channel Robustness (Exp3 v2)

- Run ID: `v20_final_hardening_20260326A_mci_robustness`
- Generated (UTC): 2026-03-26T10:45:32+00:00
- Models analyzed: 15

## Per-Model Variant MCI (Spearman)

| Model | Full | No-Wagering | Leave-Out Natural | Leave-Out Layer2 |
| --- | ---: | ---: | ---: | ---: |
| `deepseek-r1` | 0.2082 | 0.2082 | NA | NA |
| `deepseek-v3` | 0.3822 | 0.3169 | 0.4841 | 0.3456 |
| `gemini-2.5-pro` | 0.2793 | 0.1996 | 0.2252 | 0.4132 |
| `gemma-3-12b` | 0.3474 | 0.3229 | 0.3405 | 0.3787 |
| `gemma-3-27b` | 0.3950 | 0.2888 | 0.4283 | 0.4678 |
| `gpt-oss-120b` | 0.3559 | 0.1577 | 0.4659 | 0.4442 |
| `kimi-k2` | 0.3392 | 0.1943 | 0.2408 | 0.5826 |
| `llama-3.1-405b` | 0.1363 | -0.0499 | 0.1812 | 0.2777 |
| `llama-3.1-70b` | 0.2538 | 0.0512 | 0.3289 | 0.3813 |
| `llama-3.1-8b` | 0.2149 | 0.0994 | 0.2434 | 0.3019 |
| `llama-3.2-3b` | 0.1769 | 0.0529 | 0.2280 | 0.2499 |
| `llama-3.3-70b` | 0.3039 | 0.1382 | 0.4145 | 0.3592 |
| `mistral-large` | 0.3086 | 0.2465 | 0.2835 | 0.3959 |
| `mixtral-8x22b` | 0.2409 | 0.1113 | 0.2335 | 0.3777 |
| `qwen3-next-80b` | 0.4184 | 0.3227 | 0.3526 | 0.5801 |

## Rank Stability

- Full vs No-Wagering rank Spearman: 0.7714285714285712
- Full vs Leave-Out-Natural rank Spearman: 0.7934065934065934
- Full vs Leave-Out-Layer2 rank Spearman: 0.7010989010989012

## Sign Shifts

- Full→No-Wagering sign shifts: 1
- Full→Leave-Out-Natural sign shifts: 0
- Full→Leave-Out-Layer2 sign shifts: 0
