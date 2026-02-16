# Setup Guide for mirror-bench

## Quick Start

### 1. Install Dependencies

```bash
pip install -e .
```

This will install the `mirror` package in editable mode along with all dependencies.

### 2. Configure API Keys

Edit the `.env` file and add your API keys:

```bash
# Edit .env and fill in your API keys
# The file has been created from .env.example
```

You need API keys for:
- **NVIDIA NIM** - Get from https://build.nvidia.com/
- **Google AI Studio** - Get from https://aistudio.google.com/
- **DeepSeek** - Get from https://platform.deepseek.com/

### 3. Verify NVIDIA NIM Model IDs (Important!)

The model IDs in `mirror/api/models.py` are educated guesses. Verify them against the actual NVIDIA NIM catalog:

```bash
python tests/verify_nvidia_models.py
```

If any model IDs are incorrect, the script will show available alternatives. Update the model IDs in `mirror/api/models.py` accordingly.

### 4. Run API Tests

Test all three providers:

```bash
python tests/test_api.py
```

This will:
- Send a test query to each provider (NVIDIA NIM, Google AI, DeepSeek)
- Verify response format and token tracking
- Save logs to `results/api_logs/`

If all tests pass, you're ready to go! ✅

## Next Steps

After setup:
1. Verify the `.env` file has all API keys filled in
2. Run the verification script to check NVIDIA model IDs
3. Run the test script to confirm all providers work
4. Start implementing experiments (future briefs)

## Troubleshooting

### Missing API Keys
If you see "API key environment variable not set", make sure:
1. The `.env` file exists in the repository root
2. All API keys are filled in (not the placeholder values)
3. There are no extra spaces around the `=` sign

### NVIDIA NIM Model IDs Wrong
If NVIDIA NIM tests fail with "model not found":
1. Run `python tests/verify_nvidia_models.py`
2. Update the model IDs in `mirror/api/models.py`
3. Re-run the tests

### Google AI SDK Issues
If you see import errors for `google.genai`:
```bash
pip install --upgrade google-genai
```

### Rate Limiting
The default rate limit is 30 requests per minute per provider. If you hit rate limits:
1. Adjust limits in `mirror/api/rate_limiter.py`
2. Or reduce `max_concurrent` in batch operations

## Testing Individual Components

### Test model registry:
```python
from mirror.api import get_model, list_models

# Get a model config
model = get_model("llama-3.1-8b")
print(model)

# List all models
print(list_models())

# List by role
print(list_models(role="scaling_small"))
```

### Test a single provider:
```python
from mirror.api import UnifiedClient

client = UnifiedClient()
response = client.complete_sync(
    model="llama-3.1-8b",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response["content"])
```

### Test async batch:
```python
import asyncio
from mirror.api import UnifiedClient

async def test_batch():
    client = UnifiedClient()
    messages_list = [
        [{"role": "user", "content": "What is 2+2?"}],
        [{"role": "user", "content": "What is 3+3?"}],
    ]
    responses = await client.complete_batch(
        model="llama-3.1-8b",
        messages_list=messages_list,
        max_concurrent=2,
    )
    for r in responses:
        print(r["content"])

asyncio.run(test_batch())
```

## Project Structure

```
mirror-bench/
├── mirror/                    # Main package
│   ├── api/                   # ✅ Unified API layer (this brief)
│   │   ├── client.py          # UnifiedClient
│   │   ├── models.py          # Model registry
│   │   ├── rate_limiter.py    # Rate limiting
│   │   └── providers/         # Provider implementations
│   ├── experiments/           # ⏳ Experiment runners (future)
│   ├── channels/              # ⏳ Behavioral channels (future)
│   ├── scoring/               # ⏳ Scoring pipelines (future)
│   └── utils/                 # Shared utilities
│
├── data/                      # ⏳ Question bank (future)
├── results/                   # Experimental results + logs
├── analysis/                  # ⏳ Analysis scripts (future)
├── tests/                     # Tests
└── .env                       # Your API keys (not in git)
```

Legend: ✅ Complete | ⏳ Future brief
