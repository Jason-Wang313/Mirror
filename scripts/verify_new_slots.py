"""
Parallel verification of 5 new NIM model slots for Exp4.
Uses NVIDIA_NIM_API_KEY_3 for all tests.
Tests candidates in priority order within each slot.
"""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI

BASE_URL = "https://integrate.api.nvidia.com/v1"
TEST_MESSAGES = [{"role": "user", "content": "What is 2+2? Reply with just the number."}]

SLOTS = {
    "slot1_llama3.2-3b": [
        "meta/llama-3.2-3b-instruct",
    ],
    "slot2_llama3.2-1b": [
        "meta/llama-3.2-1b-instruct",
    ],
    "slot3_mixtral": [
        "mistralai/mixtral-8x22b-instruct-v0.1",
        "mistralai/mistral-small-24b-instruct-2501",
        "mistralai/mistral-nemo-12b-instruct",
    ],
    "slot4_new_lab": [
        "writer/palmyra-x5",
        "ibm/granite-3.1-8b-instruct",
        "nvidia/nemotron-4-340b-instruct",
        "upstage/solar-10.7b-instruct",
    ],
    "slot5_gemma_smaller": [
        "google/gemma-3-12b-it",
        "google/gemma-2-9b-it",
        "google/gemma-3-4b-it",
    ],
}


async def test_model(client: AsyncOpenAI, model_id: str, slot_name: str) -> dict:
    """Test a single model, return result dict."""
    try:
        resp = await client.chat.completions.create(
            model=model_id,
            messages=TEST_MESSAGES,
            max_tokens=5,
            timeout=30,
        )
        content = resp.choices[0].message.content.strip() if resp.choices else ""
        return {"slot": slot_name, "model": model_id, "status": "OK",
                "response": content, "error": None}
    except Exception as e:
        err = str(e)
        # Extract HTTP status code if present
        status = "ERROR"
        if "404" in err:
            status = "404_NOT_FOUND"
        elif "429" in err:
            status = "429_RATE_LIMIT"
        elif "401" in err:
            status = "401_AUTH"
        elif "400" in err:
            status = "400_BAD_REQUEST"
        elif "timeout" in err.lower():
            status = "TIMEOUT"
        return {"slot": slot_name, "model": model_id, "status": status,
                "response": None, "error": err[:200]}


async def verify_slot(client: AsyncOpenAI, slot_name: str, candidates: list) -> dict:
    """Try each candidate in order, return first that succeeds."""
    print(f"[{slot_name}] Testing {len(candidates)} candidate(s)...", flush=True)
    for model_id in candidates:
        print(f"  [{slot_name}] Trying {model_id} ...", flush=True)
        result = await test_model(client, model_id, slot_name)
        if result["status"] == "OK":
            print(f"  [{slot_name}] PASS: {model_id} → '{result['response']}'", flush=True)
            return {"slot": slot_name, "verified": True,
                    "model_id": model_id, "response": result["response"]}
        else:
            print(f"  [{slot_name}] FAIL ({result['status']}): {model_id}", flush=True)
    return {"slot": slot_name, "verified": False,
            "model_id": None, "response": None,
            "tried": candidates}


async def main():
    key3 = os.getenv("NVIDIA_NIM_API_KEY_3")
    if not key3:
        print("ERROR: NVIDIA_NIM_API_KEY_3 not found in .env", flush=True)
        sys.exit(1)

    print(f"Using NVIDIA_NIM_API_KEY_3: {key3[:12]}...", flush=True)
    print(f"Testing {len(SLOTS)} slots in parallel...\n", flush=True)

    client = AsyncOpenAI(api_key=key3, base_url=BASE_URL)

    tasks = [verify_slot(client, slot_name, candidates)
             for slot_name, candidates in SLOTS.items()]
    results = await asyncio.gather(*tasks)

    print("\n" + "="*70)
    print("VERIFICATION RESULTS")
    print("="*70)
    verified = []
    failed = []
    for r in results:
        if r["verified"]:
            print(f"  PASS  {r['slot']:30s}  {r['model_id']}")
            verified.append(r)
        else:
            tried = r.get("tried", [])
            print(f"  FAIL  {r['slot']:30s}  (tried: {', '.join(tried)})")
            failed.append(r)

    print(f"\nVerified: {len(verified)}/5 slots")
    print(f"Failed:   {len(failed)}/5 slots")

    # Print a machine-readable summary for the next step
    print("\n--- VERIFIED MODELS ---")
    for r in verified:
        print(f"VERIFIED|{r['slot']}|{r['model_id']}")

    await client.close()
    return verified


if __name__ == "__main__":
    asyncio.run(main())
