"""
Experiment runner — orchestrates MIRROR behavioral measurement.

Runs questions through behavioral channels across multiple models with:
  - Checkpointing (resume on interrupt)
  - Rate limiting (configurable delays per provider)
  - Structured logging (every call logged)
  - Parse failure tracking (never skip a question)

Usage:
    runner = ExperimentRunner(
        client=UnifiedClient(experiment="pilot"),
        questions_path="data/questions.jsonl",
    )
    results = runner.run(
        models=["llama-3.1-8b", "gemini-2.5-pro"],
        channels=[1, 2, 3, 4, 5],
        layers=[1, 2],
        domains=None,                  # None = all domains
        max_questions_per_domain=50,
        prompt_variant="default",
    )
    runner.save_results(results, "data/results/pilot_run.jsonl")
"""

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..api.client import UnifiedClient
from ..api.models import get_model
from .channels import (
    build_prompt,
    build_layer2_prompt,
    parse_response,
    parse_layer2,
)
from .layers import (
    build_channel3_pairs_index,
    get_max_tokens,
    pair_questions_for_difficulty_selection,
)
from ..scoring.answer_matcher import match_answer_robust


# ---------------------------------------------------------------------------
# Provider delay configuration (seconds between calls)
# ---------------------------------------------------------------------------

PROVIDER_DELAYS = {
    "nvidia_nim": 0.5,
    "google_ai": 1.0,
    "deepseek": 0.5,
}

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds, doubles each retry


# ---------------------------------------------------------------------------
# ExperimentRunner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """
    Orchestrates MIRROR behavioral measurement experiments.

    Sequential model execution, sequential channels per question,
    checkpoint/resume on interrupt.
    """

    def __init__(
        self,
        client: UnifiedClient,
        questions_path: str = "data/questions.jsonl",
        results_dir: str = "data/results",
        checkpoint_interval: int = 50,
    ):
        """
        Args:
            client: UnifiedClient instance (Brief 01 API)
            questions_path: Path to questions.jsonl
            results_dir: Directory to save results and checkpoints
            checkpoint_interval: Save checkpoint every N questions
        """
        self.client = client
        self.questions_path = Path(questions_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = checkpoint_interval

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        models: list[str],
        channels: list[int] = None,
        layers: list[int] = None,
        domains: Optional[list[str]] = None,
        max_questions_per_domain: Optional[int] = None,
        prompt_variant: str = "default",
        run_id: Optional[str] = None,
    ) -> list[dict]:
        """
        Run the full experiment and return all results.

        Args:
            models: List of model names (from MODEL_REGISTRY)
            channels: Explicit channel list [1,2,3,4,5]. If None, derive from layers.
            layers: Layer IDs to run [1, 2, 3]. Ignored if channels is set.
            domains: Domains to include. None = all.
            max_questions_per_domain: Cap questions per domain (for pilot).
            prompt_variant: "default" (only variant implemented in Brief 03).
            run_id: Unique run identifier. Auto-generated if None.

        Returns:
            List of result dicts (one per question × channel × model).
        """
        if channels is None:
            if layers is None:
                layers = [1, 2]
            from .layers import get_channels_for_layers
            channels = get_channels_for_layers(layers)

        if run_id is None:
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

        return asyncio.run(
            self._run_async(
                models=models,
                channels=channels,
                domains=domains,
                max_questions_per_domain=max_questions_per_domain,
                prompt_variant=prompt_variant,
                run_id=run_id,
            )
        )

    def save_results(self, results: list[dict], path: str) -> None:
        """Save results list to a JSONL file."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"✅ Saved {len(results)} results → {out_path}")

    # ------------------------------------------------------------------
    # Internal async orchestration
    # ------------------------------------------------------------------

    async def _run_async(
        self,
        models: list[str],
        channels: list,
        domains: Optional[list[str]],
        max_questions_per_domain: Optional[int],
        prompt_variant: str,
        run_id: str,
    ) -> list[dict]:
        """Main async orchestration loop."""

        # Load questions
        questions = self._load_questions(domains, max_questions_per_domain)
        print(f"Loaded {len(questions)} questions")

        # Pre-compute Channel 3 pairs
        ch3_pairs = pair_questions_for_difficulty_selection(questions)
        ch3_index = build_channel3_pairs_index(ch3_pairs)
        print(f"Channel 3: {len(ch3_pairs)} easy/hard pairs pre-computed")

        # Load checkpoint
        checkpoint_path = self.results_dir / f"{run_id}_checkpoint.jsonl"
        completed_keys, all_results = self._load_checkpoint(checkpoint_path)
        print(f"Resuming: {len(completed_keys)} measurements already done")

        # Determine layer assignment for each channel
        layer_for_channel = _assign_layers(channels)

        try:
            for model in models:
                print(f"\n{'='*60}")
                print(f"Model: {model}")
                print(f"{'='*60}")

                # Get provider for delay config
                model_config = get_model(model)
                provider = model_config["provider"]
                delay = PROVIDER_DELAYS.get(provider, 0.5)

                questions_done = 0

                for question in questions:
                    qid = question.get("source_id", "unknown")

                    for channel_id in channels:
                        key = (qid, model, str(channel_id))
                        if key in completed_keys:
                            continue  # Already done in previous run

                        # Run this question × channel × model
                        result = await self._run_single(
                            question=question,
                            model=model,
                            channel_id=channel_id,
                            layer=layer_for_channel.get(channel_id, 1),
                            ch3_index=ch3_index,
                            prompt_variant=prompt_variant,
                        )

                        all_results.append(result)
                        completed_keys.add(key)

                        # Rate limiting delay
                        await asyncio.sleep(delay)

                    questions_done += 1

                    # Checkpoint
                    if questions_done % self.checkpoint_interval == 0:
                        self._save_checkpoint(all_results, checkpoint_path)
                        print(
                            f"  Checkpoint: {questions_done}/{len(questions)} questions, "
                            f"{len(all_results)} total measurements"
                        )

                # Final checkpoint after each model
                self._save_checkpoint(all_results, checkpoint_path)
                print(f"  ✅ {model} complete: {questions_done} questions")

        except KeyboardInterrupt:
            print(f"\n⚠️  Interrupted — checkpoint saved to {checkpoint_path}")
            self._save_checkpoint(all_results, checkpoint_path)

        return all_results

    async def _run_single(
        self,
        question: dict,
        model: str,
        channel_id,
        layer: int,
        ch3_index: dict,
        prompt_variant: str,
    ) -> dict:
        """
        Run a single question × channel × model measurement.

        Returns a result dict conforming to the output schema.
        """
        qid = question.get("source_id", "unknown")
        timestamp = datetime.now(timezone.utc).isoformat()
        t0 = time.monotonic()

        # Build prompt
        prompt_text, extra_context = self._build_prompt_for_channel(
            channel_id, question, ch3_index
        )

        if prompt_text is None:
            # Channel 3: no pair available for this question — skip
            return self._make_skipped_result(
                question, model, channel_id, layer, prompt_variant, timestamp,
                reason="no_channel3_pair",
            )

        # Determine max_tokens
        max_tokens = get_max_tokens(model, channel_id)

        # Call API with retries
        raw_response, latency_ms, api_error = await self._call_with_retry(
            model=model,
            prompt=prompt_text,
            max_tokens=max_tokens,
            metadata={
                "task": "experiment",
                "channel": channel_id,
                "question_id": qid,
                "run_variant": prompt_variant,
            },
        )

        elapsed_ms = (time.monotonic() - t0) * 1000

        if api_error or raw_response is None:
            return self._make_error_result(
                question, model, channel_id, layer, prompt_variant,
                timestamp, latency_ms, api_error or "empty_response",
            )

        # Parse response
        parsed, parse_success = self._parse_for_channel(
            channel_id, raw_response, question, extra_context
        )

        # Score answer
        answer_correct = None
        if parsed.get("answer"):
            try:
                answer_correct = match_answer_robust(
                    predicted=parsed["answer"],
                    correct=question["correct_answer"],
                    answer_type=question.get("answer_type", "short_text"),
                    metadata=question.get("metadata", {}),
                )
            except Exception:
                answer_correct = None

        result = {
            "question_id": qid,
            "model": model,
            "channel": channel_id,
            "layer": layer,
            "prompt_variant": prompt_variant,
            "timestamp": timestamp,
            "raw_response": raw_response,
            "parsed": parsed,
            "answer_correct": answer_correct,
            "parse_success": parse_success,
            "latency_ms": round(elapsed_ms, 1),
            "domain": question.get("domain"),
            "difficulty": question.get("difficulty"),
            # Channel 3 extra context
            **({"ch3_pair_id": extra_context.get("pair_id")} if channel_id == 3 and extra_context else {}),
        }

        return result

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_prompt_for_channel(
        self, channel_id, question: dict, ch3_index: dict
    ) -> tuple[Optional[str], dict]:
        """
        Build prompt string and return (prompt_text, extra_context).

        Returns (None, {}) if the channel cannot run for this question
        (e.g., no Channel 3 pair available).
        """
        if channel_id == 3:
            qid = question.get("source_id")
            pairs = ch3_index.get(qid, [])
            if not pairs:
                return None, {}
            pair = pairs[0]  # Use first available pair
            easy_q = pair["easy_question"]
            hard_q = pair["hard_question"]
            prompt = build_prompt(3, question, easy_question=easy_q, hard_question=hard_q)
            return prompt, {"pair_id": pair["pair_id"], "easy_question": easy_q, "hard_question": hard_q}

        elif channel_id == "layer2":
            return build_layer2_prompt(question), {}

        else:
            return build_prompt(channel_id, question), {}

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_for_channel(
        self, channel_id, response_text: str, question: dict, extra_context: dict
    ) -> tuple[dict, bool]:
        """Parse response and return (parsed_dict, parse_success)."""
        try:
            if channel_id == 3:
                parsed = parse_response(
                    3,
                    response_text,
                    easy_question=extra_context["easy_question"],
                    hard_question=extra_context["hard_question"],
                )
            elif channel_id == "layer2":
                parsed = parse_layer2(response_text)
            else:
                parsed = parse_response(channel_id, response_text)

            parse_success = parsed.get("parse_success", True)
            return parsed, parse_success

        except Exception as e:
            return {"parse_error": str(e)}, False

    # ------------------------------------------------------------------
    # API call with retry/backoff
    # ------------------------------------------------------------------

    async def _call_with_retry(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        metadata: dict,
    ) -> tuple[Optional[str], float, Optional[str]]:
        """
        Call API with exponential backoff on failure.

        Returns:
            (response_content, latency_ms, error_message)
        """
        last_error = None

        for attempt in range(MAX_RETRIES):
            if attempt > 0:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                await asyncio.sleep(delay)

            t0 = time.monotonic()
            response = await self.client.complete(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
                metadata=metadata,
            )
            latency_ms = (time.monotonic() - t0) * 1000

            if "error" in response:
                last_error = response["error"]
                continue

            content = response.get("content")
            if content is None:
                last_error = "empty_content"
                continue

            return content, latency_ms, None

        return None, 0.0, last_error

    # ------------------------------------------------------------------
    # Question loading
    # ------------------------------------------------------------------

    def _load_questions(
        self,
        domains: Optional[list[str]],
        max_per_domain: Optional[int],
    ) -> list[dict]:
        """Load and filter questions from questions.jsonl."""
        from collections import defaultdict

        questions = []
        with open(self.questions_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    questions.append(json.loads(line))

        # Filter by domain
        if domains:
            questions = [q for q in questions if q.get("domain") in domains]

        # Cap per domain
        if max_per_domain:
            by_domain: dict[str, list] = defaultdict(list)
            for q in questions:
                by_domain[q.get("domain", "unknown")].append(q)
            questions = []
            for domain_qs in by_domain.values():
                questions.extend(domain_qs[:max_per_domain])

        return questions

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _load_checkpoint(
        self, checkpoint_path: Path
    ) -> tuple[set, list[dict]]:
        """Load existing checkpoint. Returns (completed_keys, results)."""
        completed_keys = set()
        results = []

        if not checkpoint_path.exists():
            return completed_keys, results

        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    results.append(r)
                    key = (
                        r.get("question_id"),
                        r.get("model"),
                        str(r.get("channel")),
                    )
                    completed_keys.add(key)
                except json.JSONDecodeError:
                    pass  # Skip corrupt lines

        return completed_keys, results

    def _save_checkpoint(self, results: list[dict], checkpoint_path: Path) -> None:
        """Overwrite checkpoint file with current results."""
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    # ------------------------------------------------------------------
    # Result constructors for error/skip cases
    # ------------------------------------------------------------------

    def _make_error_result(
        self,
        question: dict,
        model: str,
        channel_id,
        layer: int,
        prompt_variant: str,
        timestamp: str,
        latency_ms: float,
        error: str,
    ) -> dict:
        return {
            "question_id": question.get("source_id", "unknown"),
            "model": model,
            "channel": channel_id,
            "layer": layer,
            "prompt_variant": prompt_variant,
            "timestamp": timestamp,
            "raw_response": None,
            "parsed": {},
            "answer_correct": None,
            "parse_success": False,
            "latency_ms": round(latency_ms, 1),
            "domain": question.get("domain"),
            "difficulty": question.get("difficulty"),
            "error": error,
        }

    def _make_skipped_result(
        self,
        question: dict,
        model: str,
        channel_id,
        layer: int,
        prompt_variant: str,
        timestamp: str,
        reason: str,
    ) -> dict:
        return {
            "question_id": question.get("source_id", "unknown"),
            "model": model,
            "channel": channel_id,
            "layer": layer,
            "prompt_variant": prompt_variant,
            "timestamp": timestamp,
            "raw_response": None,
            "parsed": {},
            "answer_correct": None,
            "parse_success": False,
            "latency_ms": 0.0,
            "domain": question.get("domain"),
            "difficulty": question.get("difficulty"),
            "skipped": True,
            "skip_reason": reason,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assign_layers(channels: list) -> dict:
    """
    Map channel_id → layer_id.

    Convention:
      Channels 1-5: Layer 1
      "layer2": Layer 2
      Channel 5 (also Layer 3): Layer 1 is the primary assignment
    """
    mapping = {}
    for ch in channels:
        if ch == "layer2":
            mapping[ch] = 2
        else:
            mapping[ch] = 1
    return mapping


# ---------------------------------------------------------------------------
# Convenience sync wrapper for scripts
# ---------------------------------------------------------------------------

def run_experiment(
    models: list[str],
    channels: list[int] = None,
    layers: list[int] = None,
    domains: Optional[list[str]] = None,
    max_questions_per_domain: Optional[int] = None,
    questions_path: str = "data/questions.jsonl",
    results_dir: str = "data/results",
    prompt_variant: str = "default",
    run_id: Optional[str] = None,
    experiment_name: str = "experiment",
) -> list[dict]:
    """
    Convenience function for scripts — creates runner and runs experiment.

    Args:
        models: Model names
        channels: Channel IDs
        layers: Layer IDs (used if channels is None)
        domains: Domain filter
        max_questions_per_domain: Cap per domain
        questions_path: Path to questions.jsonl
        results_dir: Results directory
        prompt_variant: Prompt variant string
        run_id: Run identifier
        experiment_name: Name for API logging

    Returns:
        List of result dicts
    """
    from ..api.client import UnifiedClient

    client = UnifiedClient(
        log_dir=str(Path(results_dir) / "api_logs"),
        experiment=experiment_name,
    )
    runner = ExperimentRunner(
        client=client,
        questions_path=questions_path,
        results_dir=results_dir,
    )
    return runner.run(
        models=models,
        channels=channels,
        layers=layers,
        domains=domains,
        max_questions_per_domain=max_questions_per_domain,
        prompt_variant=prompt_variant,
        run_id=run_id,
    )
