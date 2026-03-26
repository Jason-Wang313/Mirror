# Exp3 Difficulty-Controlled Ablation

- Run ID: `v20_final_hardening_20260326B_exp3_difficulty`
- Generated (UTC): 2026-03-26T16:57:58+00:00
- Models analyzed: 16
- Bootstrap samples: 1200

## Controlled CCE by Model

| Model | Balanced CCE | 95% CI | Mean Conf | Mean Acc | Overconfidence Gap | Balanced N |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `deepseek-r1` | 0.5000 | [0.4722, 0.5278] | 0.5000 | 0.1389 | 0.3611 | 72 |
| `deepseek-v3` | 0.4583 | [0.3028, 0.6139] | 0.8306 | 0.4028 | 0.4278 | 72 |
| `gemini-2.5-pro` | 0.6493 | [0.5000, 0.8306] | 0.9938 | 0.3472 | 0.6465 | 72 |
| `gemma-3-12b` | 0.5222 | [0.3806, 0.7000] | 0.9125 | 0.4028 | 0.5097 | 72 |
| `gemma-3-27b` | 0.5576 | [0.3567, 0.6972] | 0.9007 | 0.3750 | 0.5257 | 72 |
| `gpt-oss-120b` | 0.3640 | [0.2175, 0.7017] | 0.9453 | 0.6000 | 0.3453 | 15 |
| `kimi-k2` | 0.4847 | [0.2907, 0.7056] | 0.9792 | 0.5000 | 0.4792 | 36 |
| `llama-3.1-405b` | 0.4792 | [0.3389, 0.6167] | 0.7264 | 0.3472 | 0.3792 | 72 |
| `llama-3.1-70b` | 0.4285 | [0.2556, 0.6100] | 0.9285 | 0.5417 | 0.3868 | 72 |
| `llama-3.1-8b` | 0.5903 | [0.4222, 0.7208] | 0.8708 | 0.3472 | 0.5236 | 72 |
| `llama-3.2-3b` | 0.5319 | [0.4611, 0.7278] | 0.8542 | 0.4306 | 0.4236 | 72 |
| `llama-3.3-70b` | 0.4861 | [0.3111, 0.6028] | 0.8181 | 0.3472 | 0.4708 | 72 |
| `mistral-large` | 0.6381 | [0.4917, 0.7889] | 0.9269 | 0.3056 | 0.6214 | 72 |
| `mixtral-8x22b` | 0.7806 | [0.6028, 0.9306] | 0.9875 | 0.2083 | 0.7792 | 72 |
| `phi-4` | 0.5000 | [0.5000, 0.5000] | 0.5000 | 0.0000 | 0.5000 | 72 |
| `qwen3-next-80b` | 0.4979 | [0.3278, 0.6806] | 0.9715 | 0.4861 | 0.4854 | 72 |

## Universal-Failure Checks

- All models overconfident (gap > 0): `True`
- All models balanced CCE > 0.20: `True`
- All models balanced CCE > 0.30: `True`
- Models with balanced CCE >= 0.434: `14/16`

## Interpretation

Difficulty-controlled, pair-balanced estimates preserve a large compositional calibration error profile across models, indicating the Exp3 signal is not an artifact of strong/weak mixture imbalance.
