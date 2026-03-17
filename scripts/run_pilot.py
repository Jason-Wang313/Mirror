"""
Pilot experiment runner for MIRROR.

Runs 50 questions × 2 models × 5 channels + Layer 2 self-report.
Prints a summary table and saves all results to data/results/.

Usage:
    python scripts/run_pilot.py
    python scripts/run_pilot.py --models llama-3.1-8b gemini-2.5-pro
    python scripts/run_pilot.py --n-questions 10 --channels 1 2 5
    python scripts/run_pilot.py --domains arithmetic logical
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path when run as a script
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mirror.api.client import UnifiedClient
from mirror.experiments.runner import ExperimentRunner
from mirror.scoring.metrics import compute_all_metrics


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODELS = ["llama-3.1-8b", "llama-3.1-70b"]
DEFAULT_N_QUESTIONS = 50
DEFAULT_CHANNELS = [1, 2, 3, 4, 5, "layer2"]
DEFAULT_DOMAINS = None  # All domains


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(results: list[dict]) -> None:
    """Print a formatted summary table of results."""
    print("\n" + "=" * 80)
    print("PILOT EXPERIMENT SUMMARY")
    print("=" * 80)

    # Group by model
    by_model: dict = defaultdict(list)
    for r in results:
        model = r.get("model", "unknown")
        by_model[model].append(r)

    for model, model_results in sorted(by_model.items()):
        print(f"\nModel: {model}")
        print("-" * 70)

        # Per-domain accuracy from Channel 5 (natural)
        by_domain: dict = defaultdict(list)
        for r in model_results:
            if r.get("channel") == 5 and r.get("answer_correct") is not None:
                by_domain[r.get("domain", "unknown")].append(r["answer_correct"])

        if by_domain:
            print(f"{'Domain':<16} {'N':>4} {'Accuracy':>10} {'Parse%':>8}")
            print(f"  {'-'*50}")
            all_correct = []
            for domain in sorted(by_domain.keys()):
                corrects = by_domain[domain]
                acc = sum(corrects) / len(corrects)
                all_correct.extend(corrects)
                print(f"  {domain:<14} {len(corrects):>4} {acc:>10.1%}")

            overall_acc = sum(all_correct) / len(all_correct) if all_correct else 0
            print(f"  {'OVERALL':<14} {len(all_correct):>4} {overall_acc:>10.1%}")

        # Channel-level stats
        print(f"\n  Channel statistics:")
        channel_ids = [1, 2, 3, 4, 5, "layer2"]
        for ch in channel_ids:
            ch_results = [r for r in model_results if r.get("channel") == ch]
            if not ch_results:
                continue

            n_total = len(ch_results)
            n_parse_ok = sum(1 for r in ch_results if r.get("parse_success"))
            n_errors = sum(1 for r in ch_results if r.get("error"))

            parse_rate = n_parse_ok / n_total if n_total else 0
            ch_label = f"Ch{ch}" if isinstance(ch, int) else "L2"
            print(f"    {ch_label}: {n_total:3d} calls | parse: {parse_rate:.0%} | errors: {n_errors}")

            # Channel-specific metrics
            if ch == 1:
                bets = [r["parsed"].get("bet") for r in ch_results if r.get("parsed", {}).get("bet") is not None]
                if bets:
                    print(f"         mean bet: {sum(bets)/len(bets):.1f} | range: {min(bets)}-{max(bets)}")

            elif ch == 2:
                skipped = sum(1 for r in ch_results if r.get("parsed", {}).get("skipped"))
                print(f"         skip rate: {skipped/n_total:.1%}")

            elif ch == 3:
                chose_a = sum(1 for r in ch_results if r.get("parsed", {}).get("choice") == "A")
                skipped = sum(1 for r in ch_results if r.get("skipped"))
                n_valid = n_total - skipped
                if n_valid > 0:
                    print(f"         chose A (hard): {chose_a/n_valid:.1%} | skipped (no pair): {skipped}")

            elif ch == 4:
                tool_uses = [
                    len(r.get("parsed", {}).get("tools_used", []))
                    for r in ch_results
                    if not r.get("error")
                ]
                if tool_uses:
                    n_with_tools = sum(1 for t in tool_uses if t > 0)
                    print(f"         tool use rate: {n_with_tools/len(tool_uses):.1%} | avg tools: {sum(tool_uses)/len(tool_uses):.1f}")

            elif ch == 5:
                lengths = [r.get("parsed", {}).get("response_length", 0) for r in ch_results]
                hedges = [r.get("parsed", {}).get("hedging_count", 0) for r in ch_results]
                if lengths:
                    print(f"         avg words: {sum(lengths)/len(lengths):.0f} | avg hedges: {sum(hedges)/len(hedges):.1f}")

            elif ch == "layer2":
                confs = [r["parsed"].get("confidence") for r in ch_results if r.get("parsed", {}).get("confidence") is not None]
                if confs:
                    print(f"         mean confidence: {sum(confs)/len(confs):.1f} | range: {min(confs)}-{max(confs)}")

        print()

    # Latency stats
    latencies = [r.get("latency_ms", 0) for r in results if r.get("latency_ms", 0) > 0]
    if latencies:
        print(f"\nLatency: mean={sum(latencies)/len(latencies):.0f}ms | max={max(latencies):.0f}ms")

    total_calls = len(results)
    n_errors = sum(1 for r in results if r.get("error"))
    n_parse_fail = sum(1 for r in results if not r.get("parse_success") and not r.get("error") and not r.get("skipped"))
    print(f"Total calls: {total_calls} | errors: {n_errors} | parse failures: {n_parse_fail}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MIRROR pilot experiment runner"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model names to evaluate",
    )
    parser.add_argument(
        "--n-questions",
        type=int,
        default=DEFAULT_N_QUESTIONS,
        help="Max questions per domain (default: 50)",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help="Channels to run (1-5, layer2). Default: all",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Domains to include. Default: all",
    )
    parser.add_argument(
        "--questions-path",
        default="data/questions.jsonl",
        help="Path to questions.jsonl",
    )
    parser.add_argument(
        "--results-dir",
        default="data/results",
        help="Directory for saving results",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run identifier for checkpointing. Auto-generated if not set.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoint (uses --run-id)",
    )

    args = parser.parse_args()

    # Parse channels (mixed int/str)
    if args.channels:
        channels = []
        for ch in args.channels:
            if ch == "layer2":
                channels.append("layer2")
            else:
                channels.append(int(ch))
    else:
        channels = DEFAULT_CHANNELS

    run_id = args.run_id
    if run_id is None:
        run_id = f"pilot_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"

    print("=" * 60)
    print("MIRROR Pilot Experiment")
    print("=" * 60)
    print(f"Models:     {', '.join(args.models)}")
    print(f"Channels:   {channels}")
    print(f"Domains:    {args.domains or 'all'}")
    print(f"Max/domain: {args.n_questions}")
    print(f"Run ID:     {run_id}")
    print("=" * 60)

    # Initialize client and runner
    client = UnifiedClient(
        log_dir=str(Path(args.results_dir) / "api_logs"),
        experiment=run_id,
    )
    runner = ExperimentRunner(
        client=client,
        questions_path=args.questions_path,
        results_dir=args.results_dir,
        checkpoint_interval=50,
    )

    # Run experiment
    results = runner.run(
        models=args.models,
        channels=channels,
        domains=args.domains,
        max_questions_per_domain=args.n_questions,
        prompt_variant="default",
        run_id=run_id,
    )

    # Save final results
    results_path = Path(args.results_dir) / f"{run_id}.jsonl"
    runner.save_results(results, str(results_path))

    # Compute and save metrics
    print("\nComputing metrics...")
    metrics = compute_all_metrics(results)

    metrics_path = Path(args.results_dir) / f"{run_id}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        import json
        json.dump(metrics, f, indent=2, default=str)
    print(f"✅ Metrics saved → {metrics_path}")

    # MCI per model
    from mirror.scoring.metrics import compute_mci, compute_channel_dissociation_matrix

    mci_output = {}
    by_model: dict = defaultdict(list)
    for r in results:
        by_model[r.get("model", "unknown")].append(r)

    for model, model_results in by_model.items():
        mci_output[model] = {
            "mci": compute_mci(model_results),
            "dissociation": compute_channel_dissociation_matrix(model_results),
        }

    mci_path = Path(args.results_dir) / f"{run_id}_mci.json"
    with open(mci_path, "w", encoding="utf-8") as f:
        json.dump(mci_output, f, indent=2, default=str)
    print(f"✅ MCI saved → {mci_path}")

    # Print summary
    print_summary_table(results)

    return results


if __name__ == "__main__":
    main()
