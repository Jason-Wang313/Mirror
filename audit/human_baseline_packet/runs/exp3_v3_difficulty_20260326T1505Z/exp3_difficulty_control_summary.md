# Exp3 Difficulty-Controlled Ablation

- Run ID: `exp3_v3_difficulty_20260326T1505Z`
- Generated (UTC): 2026-03-26T16:31:10+00:00
- Models analyzed: 15
- Bootstrap samples: 800

## Controlled CCE by Model

| Model | Balanced CCE | 95% CI | Mean Conf | Mean Acc | Overconfidence Gap | Balanced N |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `deepseek-r1` | 0.5031 | [0.4917, 0.5184] | 0.5156 | 0.2118 | 0.3038 | 288 |
| `deepseek-v3` | 0.5133 | [0.4700, 0.5868] | 0.7899 | 0.3492 | 0.4407 | 252 |
| `gemini-2.5-pro` | 0.6687 | [0.5878, 0.7219] | 0.8565 | 0.3542 | 0.5024 | 288 |
| `gemma-3-12b` | 0.5563 | [0.5233, 0.6080] | 0.7285 | 0.1875 | 0.5410 | 288 |
| `gemma-3-27b` | 0.5557 | [0.5069, 0.6070] | 0.7675 | 0.3576 | 0.4099 | 288 |
| `gpt-oss-120b` | 0.4279 | [0.3856, 0.4794] | 0.5077 | 0.2812 | 0.2265 | 288 |
| `kimi-k2` | 0.5764 | [0.4651, 0.6359] | 0.9368 | 0.3958 | 0.5410 | 144 |
| `llama-3.1-405b` | 0.5082 | [0.4764, 0.5600] | 0.6731 | 0.2882 | 0.3849 | 288 |
| `llama-3.1-70b` | 0.5035 | [0.4912, 0.5093] | 0.5066 | 0.0035 | 0.5031 | 288 |
| `llama-3.1-8b` | 0.6181 | [0.5839, 0.6588] | 0.7260 | 0.2743 | 0.4517 | 288 |
| `llama-3.2-3b` | 0.5486 | [0.4259, 0.6426] | 0.4764 | 0.3889 | 0.0875 | 72 |
| `llama-3.3-70b` | 0.5625 | [0.4317, 0.5710] | 0.6042 | 0.0417 | 0.5625 | 24 |
| `mistral-large` | 0.5801 | [0.5117, 0.6391] | 0.8746 | 0.3704 | 0.5042 | 162 |
| `mixtral-8x22b` | 0.6984 | [0.6086, 0.7241] | 0.8738 | 0.2604 | 0.6134 | 288 |
| `qwen3-next-80b` | 0.5625 | [0.4690, 0.6093] | 0.9528 | 0.4167 | 0.5362 | 288 |

## Universal-Failure Checks

- All models overconfident (gap > 0): `True`
- All models balanced CCE > 0.20: `True`
- All models balanced CCE > 0.30: `True`
- Models with balanced CCE >= 0.434: `14/15`

## Interpretation

Difficulty-controlled, pair-balanced estimates preserve a large compositional calibration error profile across models, indicating the Exp3 signal is not an artifact of strong/weak mixture imbalance.
