"""
Master pipeline orchestrator for question bank generation.

Usage:
    python -m mirror.data.pipeline                    # Full pipeline
    python -m mirror.data.pipeline --stage download   # Just download
    python -m mirror.data.pipeline --stage seeds      # Just seed selection
    python -m mirror.data.pipeline --stage generate   # Just variation generation
    python -m mirror.data.pipeline --stage verify     # Just verification
    python -m mirror.data.pipeline --pilot            # Pilot run: 200 per domain
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from .sources.download_all import download_all_datasets, load_all_datasets
from .seed_selector import run_seed_selection
from .variation_generator import run_variation_generation
from .counterfactual import run_counterfactual_generation
from .deduplicator import run_deduplication
from .cross_verifier import run_verification
from .difficulty_validator import run_difficulty_validation
from .provenance import build_provenance_table, compile_final_question_bank


STAGES = [
    "download",       # Download all source datasets
    "seeds",          # Select and categorize seeds
    "generate",       # Generate controlled variations
    "counterfactual", # Generate counterfactual subset
    "dedup",          # Embedding deduplication
    "verify",         # Cross-LLM verification
    "difficulty",     # Difficulty calibration
    "compile",        # Compile final questions.jsonl
    "provenance",     # Build provenance table
    "report",         # Generate verification report
]


def stage_download():
    """Stage 1: Download all source datasets."""
    print("\n" + "="*60)
    print("STAGE 1: Downloading Source Datasets")
    print("="*60)

    download_all_datasets()


def stage_seeds(pilot_mode: bool):
    """Stage 2: Select seeds."""
    print("\n" + "="*60)
    print("STAGE 2: Selecting Seeds")
    print("="*60)

    # Load all datasets
    all_questions = load_all_datasets()

    if len(all_questions) == 0:
        print("⚠️  No questions loaded. Did you run download stage?")
        return

    # Select seeds
    run_seed_selection(all_questions, pilot_mode=pilot_mode)


def stage_generate(pilot_mode: bool):
    """Stage 3: Generate variations."""
    print("\n" + "="*60)
    print("STAGE 3: Generating Variations")
    print("="*60)

    run_variation_generation(pilot_mode=pilot_mode)


def stage_counterfactual(pilot_mode: bool):
    """Stage 4: Generate counterfactuals."""
    print("\n" + "="*60)
    print("STAGE 4: Generating Counterfactuals")
    print("="*60)

    run_counterfactual_generation(pilot_mode=pilot_mode)


def stage_dedup():
    """Stage 5: Deduplication."""
    print("\n" + "="*60)
    print("STAGE 5: Deduplicating Questions")
    print("="*60)

    dedup_stats = run_deduplication()
    return {"dedup_stats": dedup_stats}


def stage_verify(pilot_mode: bool):
    """Stage 6: Cross-LLM verification."""
    print("\n" + "="*60)
    print("STAGE 6: Cross-LLM Verification")
    print("="*60)

    verification_stats = run_verification(pilot_mode=pilot_mode)
    return {"verification_stats": verification_stats}


def stage_difficulty():
    """Stage 7: Difficulty validation."""
    print("\n" + "="*60)
    print("STAGE 7: Difficulty Validation")
    print("="*60)

    difficulty_stats = run_difficulty_validation()
    return {"difficulty_stats": difficulty_stats}


def stage_compile():
    """Stage 8: Compile final question bank."""
    print("\n" + "="*60)
    print("STAGE 8: Compiling Final Question Bank")
    print("="*60)

    total = compile_final_question_bank()
    return {"total_questions": total}


def stage_provenance():
    """Stage 9: Build provenance table."""
    print("\n" + "="*60)
    print("STAGE 9: Building Provenance Table")
    print("="*60)

    provenance_stats = build_provenance_table()
    return {"provenance_stats": provenance_stats}


def stage_report(all_stats: dict, start_time: float, pilot_mode: bool):
    """Stage 10: Generate verification report."""
    print("\n" + "="*60)
    print("STAGE 10: Generating Verification Report")
    print("="*60)

    duration_minutes = (time.time() - start_time) / 60.0

    report = {
        "pipeline_run": {
            "timestamp": datetime.now().isoformat(),
            "mode": "pilot" if pilot_mode else "full",
            "duration_minutes": round(duration_minutes, 2),
        },
        "seed_stats": all_stats.get("seed_stats", {}),
        "generation_stats": all_stats.get("generation_stats", {}),
        "counterfactual_stats": all_stats.get("counterfactual_stats", {}),
        "dedup_stats": all_stats.get("dedup_stats", {}),
        "verification_stats": all_stats.get("verification_stats", {}),
        "difficulty_stats": all_stats.get("difficulty_stats", {}),
        "final_count": all_stats.get("total_questions", 0),
        "provenance_stats": all_stats.get("provenance_stats", {}),
    }

    # Save report
    with open("data/verification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("✅ Verification report saved to data/verification_report.json")
    print(f"\n{'='*60}")
    print(f"Pipeline Complete!")
    print(f"Duration: {duration_minutes:.1f} minutes")
    print(f"Total Questions: {report['final_count']}")
    print(f"{'='*60}")


def run_pipeline(stages_to_run: list = None, pilot_mode: bool = False):
    """
    Run the complete pipeline.

    Args:
        stages_to_run: List of stage names to run. If None, runs all.
        pilot_mode: If True, runs in pilot mode (fewer questions, faster)
    """
    if stages_to_run is None:
        stages_to_run = STAGES

    start_time = time.time()
    all_stats = {}

    print("="*60)
    print("MIRROR Question Bank Pipeline")
    print("="*60)
    print(f"Mode: {'PILOT' if pilot_mode else 'FULL'}")
    print(f"Stages to run: {', '.join(stages_to_run)}")
    print("="*60)

    stage_functions = {
        "download": lambda: stage_download(),
        "seeds": lambda: stage_seeds(pilot_mode),
        "generate": lambda: stage_generate(pilot_mode),
        "counterfactual": lambda: stage_counterfactual(pilot_mode),
        "dedup": lambda: stage_dedup(),
        "verify": lambda: stage_verify(pilot_mode),
        "difficulty": lambda: stage_difficulty(),
        "compile": lambda: stage_compile(),
        "provenance": lambda: stage_provenance(),
        "report": lambda: stage_report(all_stats, start_time, pilot_mode),
    }

    for stage_name in stages_to_run:
        if stage_name not in stage_functions:
            print(f"⚠️  Unknown stage: {stage_name}")
            continue

        print(f"\n{'#'*60}")
        print(f"# Running stage: {stage_name}")
        print(f"{'#'*60}")

        try:
            result = stage_functions[stage_name]()
            if result:
                all_stats.update(result)
        except Exception as e:
            print(f"\n❌ Stage '{stage_name}' failed: {e}")
            import traceback
            traceback.print_exc()
            print("\nContinuing to next stage...")

    print("\n" + "="*60)
    print("Pipeline execution complete!")
    print("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MIRROR Question Bank Pipeline")

    parser.add_argument(
        "--stage",
        type=str,
        help=f"Run only this stage. Options: {', '.join(STAGES)}",
    )

    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run in pilot mode (200 questions per domain instead of 625)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even if outputs exist",
    )

    args = parser.parse_args()

    # Determine which stages to run
    if args.stage:
        if args.stage not in STAGES:
            print(f"❌ Unknown stage: {args.stage}")
            print(f"Available stages: {', '.join(STAGES)}")
            return

        stages_to_run = [args.stage]
    else:
        stages_to_run = STAGES

    # Run pipeline
    run_pipeline(stages_to_run=stages_to_run, pilot_mode=args.pilot)


if __name__ == "__main__":
    main()
