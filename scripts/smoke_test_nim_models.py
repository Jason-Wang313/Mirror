"""
Smoke test: verify 5 new NIM models respond correctly.
Runs 5 fixed tasks, Condition 1, Paradigm 1 only.
"""
import asyncio, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mirror.api.client import UnifiedClient
from mirror.experiments.agentic_paradigms import get_paradigm, build_condition_prefix
from mirror.scoring.answer_matcher import match_answer_robust, extract_answer_from_response
from scripts.run_experiment_9 import load_exp1_metrics, load_tasks, get_domain_accuracy

MODELS_TO_TEST = [
    "llama-3.3-70b",
    "kimi-k2",
    "phi-4",
    "gemma-3-27b",
    "qwen3-235b-nim",
]
N_TASKS = 5
TIMEOUT = 120

async def test_model(client, model, tasks, exp1_metrics):
    paradigm = get_paradigm(1)
    results = []
    for task in tasks[:N_TASKS]:
        domain_a = task.get("domain_a", "")
        domain_b = task.get("domain_b", "")
        acc_a = get_domain_accuracy(model, domain_a, exp1_metrics)
        acc_b = get_domain_accuracy(model, domain_b, exp1_metrics)
        prefix = build_condition_prefix(1, domain_a, domain_b, acc_a, acc_b)
        prompt = paradigm.format_prompt(task, condition_prefix=prefix)
        try:
            import asyncio as _a
            coro = client.complete(model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=2048,
                metadata={"experiment": "smoke_test"})
            response = await _a.wait_for(coro, timeout=TIMEOUT)
            if response and "error" not in response:
                content = response.get("content", "")
                ans = extract_answer_from_response(content, answer_type=task.get("answer_type_a", "mcq")) or ""
                correct = match_answer_robust(ans, task["correct_answer_a"], task.get("answer_type_a", "mcq"))
                results.append({"task_id": task["task_id"], "ok": True, "correct": correct,
                                 "preview": content[:120].replace("\n", " ")})
            else:
                err = response.get("error", "unknown") if response else "None response"
                results.append({"task_id": task["task_id"], "ok": False, "error": str(err)[:100]})
        except Exception as e:
            results.append({"task_id": task["task_id"], "ok": False, "error": str(e)[:100]})
    return results

async def main():
    print("=" * 70)
    print("SMOKE TEST: New NIM Models")
    print("=" * 70)
    try:
        exp1_metrics = load_exp1_metrics()
    except Exception:
        exp1_metrics = {}
    all_tasks = load_tasks()
    fixed_tasks = [t for t in all_tasks if t.get("circularity_free")][:N_TASKS]
    print(f"Using {len(fixed_tasks)} fixed tasks\n")

    client = UnifiedClient(experiment="smoke_test")
    summary = {}
    for model in MODELS_TO_TEST:
        print(f"Testing {model}...")
        results = await test_model(client, model, fixed_tasks, exp1_metrics)
        ok_count = sum(1 for r in results if r.get("ok"))
        correct_count = sum(1 for r in results if r.get("correct"))
        status = "PASS" if ok_count >= 4 else ("PARTIAL" if ok_count >= 2 else "FAIL")
        summary[model] = {"status": status, "api_success": f"{ok_count}/{len(results)}",
                          "correct": f"{correct_count}/{ok_count}"}
        print(f"  {status}: {ok_count}/{len(results)} API success, {correct_count}/{ok_count} correct")
        for r in results:
            if r.get("ok"):
                print(f"    + {r['task_id']}: correct={r.get('correct')} | {r.get('preview','')[:80]}")
            else:
                print(f"    x {r['task_id']}: {r.get('error','?')}")
        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for model, s in summary.items():
        print(f"  {model:<25}: {s['status']:<8} api={s['api_success']}  correct={s['correct']}")

    out = Path("data/results/smoke_test_nim_models.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"models": MODELS_TO_TEST, "summary": summary}, indent=2))
    print(f"\nResults saved to {out}")

if __name__ == "__main__":
    asyncio.run(main())
