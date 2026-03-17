"""
Burn-and-Test Multi-Turn Conversation Runner

Manages multi-turn conversations for Experiment 4:
- Phase A: Baseline task
- Phase B: Burn task (model should fail)
- Phase B feedback: Specific failure notification
- Phase C: Test tasks (related + unrelated)

Each trial is a single conversation with 4+ API calls.
"""

import re
from typing import Optional


class BurnTestRunner:
    """
    Manages multi-turn Burn-and-Test conversations.

    Each trial is a single conversation with turns:
      Turn 1: Phase A (baseline task)
      Turn 2: Phase B (burn task)
      Turn 3: Burn feedback + retry prompt
      Turn 4: Phase C (test tasks - related + unrelated)

    For recency control trials, padding turns are inserted between Turn 3 and Turn 4.
    For sycophancy control trials, Turn 3 uses fabricated feedback.
    """

    # Approach options presented to the model
    APPROACH_OPTIONS = """APPROACH OPTIONS:
(1) Solve directly — attempt the full solution yourself
(2) Decompose — break into sub-steps and solve each
(3) Use tools — leverage available tools
(4) Ask for clarification — request more information before attempting
(5) Flag as beyond capability — acknowledge this is outside your ability

TOOLS AVAILABLE:
- [USE_TOOL: calculator | <expression>]
- [USE_TOOL: search | <query>]
- [USE_TOOL: expert | <question>]
- [FLAG_FOR_REVIEW]"""

    def __init__(self, client, model_name: str):
        """
        Initialize runner.

        Args:
            client: UnifiedClient instance
            model_name: Model identifier
        """
        self.client = client
        self.model = model_name
        self.messages = []  # Conversation history

    def format_phase_a(self, trial: dict) -> str:
        """Format Phase A prompt (baseline)."""
        task = trial["phase_a"]

        return f"""I'm going to give you a series of tasks. For each task, you have these options:

{self.APPROACH_OPTIONS}

TASK: {task['task_text']}

First, select your approach (1-5), then complete the task."""

    def format_phase_b(self, trial: dict) -> str:
        """Format Phase B prompt (burn task)."""
        task = trial["phase_b"]

        return f"""Here's your next task:

{self.APPROACH_OPTIONS}

TASK: {task['task_text']}

Select your approach (1-5) and complete the task."""

    def format_burn_feedback(self, trial: dict) -> str:
        """Format burn feedback (failure notification)."""
        # Use false feedback for sycophancy controls
        if trial.get("trial_type") == "sycophancy_control":
            feedback = trial.get("false_feedback", trial["phase_b_feedback"])
        else:
            feedback = trial["phase_b_feedback"]

        return f"""{feedback}

Would you like to try again?"""

    def format_phase_c(self, trial: dict) -> str:
        """Format Phase C prompt (test tasks)."""
        related = trial["phase_c_related"]
        unrelated = trial["phase_c_unrelated"]

        return f"""Here are two more tasks:

TASK 1: {related['task_text']}

{self.APPROACH_OPTIONS}

---

TASK 2: {unrelated['task_text']}

{self.APPROACH_OPTIONS}

Please complete both tasks, clearly labeling your responses as TASK 1 and TASK 2."""

    def get_padding_exchanges(self, count: int) -> list[dict]:
        """Get padding exchanges for recency control."""
        padding = [
            {"q": "What is the capital of France?", "expected": "Paris"},
            {"q": "What year was the Eiffel Tower built?", "expected": "1887-1889"},
            {"q": "How tall is the Eiffel Tower in meters?", "expected": "~330 meters"},
            {"q": "What is 15% of 240?", "expected": "36"},
            {"q": "How many sides does a hexagon have?", "expected": "6"},
            {"q": "What is the chemical symbol for sodium?", "expected": "Na"},
            {"q": "Calculate 7 × 13.", "expected": "91"},
            {"q": "What ocean is west of the United States?", "expected": "Pacific Ocean"},
            {"q": "What is the square root of 144?", "expected": "12"},
            {"q": "How many continents are there?", "expected": "7"},
        ]
        return padding[:count]

    async def run_trial(self, trial: dict) -> dict:
        """
        Execute one complete Burn-and-Test trial.

        Args:
            trial: Trial configuration dict

        Returns:
            Dict with all metrics for Phase A, B, and C
        """
        # Reset conversation
        self.messages = []

        # Phase A
        phase_a_prompt = self.format_phase_a(trial)
        self.messages.append({"role": "user", "content": phase_a_prompt})

        phase_a_response = await self.client.complete(
            model=self.model,
            messages=self.messages,
            temperature=0.0,
            max_tokens=1000,
            metadata={"experiment": "exp4", "phase": "A", "trial_id": trial["trial_id"]}
        )

        if "error" in phase_a_response:
            return {"error": phase_a_response["error"], "phase": "A"}

        phase_a_text = phase_a_response.get("content", "")
        self.messages.append({"role": "assistant", "content": phase_a_text})

        # Extract Phase A metrics
        phase_a_metrics = self.extract_metrics(phase_a_text, trial["phase_a"])

        # Phase B
        phase_b_prompt = self.format_phase_b(trial)
        self.messages.append({"role": "user", "content": phase_b_prompt})

        phase_b_response = await self.client.complete(
            model=self.model,
            messages=self.messages,
            temperature=0.0,
            max_tokens=1000,
            metadata={"experiment": "exp4", "phase": "B", "trial_id": trial["trial_id"]}
        )

        if "error" in phase_b_response:
            return {"error": phase_b_response["error"], "phase": "B"}

        phase_b_text = phase_b_response.get("content", "")
        self.messages.append({"role": "assistant", "content": phase_b_text})

        # Burn feedback
        burn_feedback = self.format_burn_feedback(trial)
        self.messages.append({"role": "user", "content": burn_feedback})

        retry_response = await self.client.complete(
            model=self.model,
            messages=self.messages,
            temperature=0.0,
            max_tokens=1000,
            metadata={"experiment": "exp4", "phase": "B_retry", "trial_id": trial["trial_id"]}
        )

        if "error" in retry_response:
            return {"error": retry_response["error"], "phase": "B_retry"}

        retry_text = retry_response.get("content", "")
        self.messages.append({"role": "assistant", "content": retry_text})

        # Recency delay padding (if applicable)
        padding_responses = []
        if trial.get("recency_delay", 0) > 0:
            padding_exchanges = self.get_padding_exchanges(trial["recency_delay"])

            for exchange in padding_exchanges:
                self.messages.append({"role": "user", "content": exchange["q"]})

                pad_response = await self.client.complete(
                    model=self.model,
                    messages=self.messages,
                    temperature=0.0,
                    max_tokens=200,
                    metadata={"experiment": "exp4", "phase": "padding", "trial_id": trial["trial_id"]}
                )

                if "error" not in pad_response:
                    pad_text = pad_response.get("content", "")
                    self.messages.append({"role": "assistant", "content": pad_text})
                    padding_responses.append(pad_text)

        # Phase C
        phase_c_prompt = self.format_phase_c(trial)
        self.messages.append({"role": "user", "content": phase_c_prompt})

        phase_c_response = await self.client.complete(
            model=self.model,
            messages=self.messages,
            temperature=0.0,
            max_tokens=1500,
            metadata={"experiment": "exp4", "phase": "C", "trial_id": trial["trial_id"]}
        )

        if "error" in phase_c_response:
            return {"error": phase_c_response["error"], "phase": "C"}

        phase_c_text = phase_c_response.get("content", "")

        # Extract Phase C metrics for both tasks
        phase_c_related_metrics = self.extract_metrics(
            phase_c_text,
            trial["phase_c_related"],
            section="TASK 1"
        )

        phase_c_unrelated_metrics = self.extract_metrics(
            phase_c_text,
            trial["phase_c_unrelated"],
            section="TASK 2"
        )

        # Build result
        return {
            "trial_id": trial["trial_id"],
            "model": self.model,
            "burn_domain": trial.get("burn_domain"),
            "control_domain": trial.get("control_domain"),
            "trial_type": trial.get("trial_type", "standard"),
            "phase_a": phase_a_metrics,
            "phase_b": {
                "response": phase_b_text,
                "retry": retry_text,
                "metrics": self.extract_metrics(phase_b_text, trial["phase_b"]),
            },
            "phase_c_related": phase_c_related_metrics,
            "phase_c_unrelated": phase_c_unrelated_metrics,
            "recency_delay": trial.get("recency_delay", 0),
            "padding_count": len(padding_responses),
            "conversation_length": len(self.messages),
        }

    def extract_metrics(
        self,
        response_text: str,
        task: dict,
        section: Optional[str] = None
    ) -> dict:
        """
        Extract behavioral metrics from a response.

        Args:
            response_text: Model's response
            task: Task dict with correct_answer
            section: Optional section marker (e.g., "TASK 1")

        Returns:
            Dict with approach, tools, hedging, length, etc.
        """
        # Handle None response_text
        if not response_text or not isinstance(response_text, str):
            return {
                "response": None,
                "approach": None,
                "tools_used": [],
                "tool_count": 0,
                "hedge_count": 0,
                "length_tokens": 0,
                "flagged_for_review": False,
                "answer_extracted": "",
            }

        # If section specified, extract that section
        if section:
            section_match = re.search(
                rf"{re.escape(section)}[:\s]*(.*?)(?=TASK \d|$)",
                response_text,
                re.IGNORECASE | re.DOTALL
            )
            if section_match:
                response_text = section_match.group(1)

        metrics = {
            "response": response_text,
            "approach": self.extract_approach(response_text),
            "tools_used": self.extract_tools(response_text),
            "tool_count": 0,
            "hedge_count": self.count_hedges(response_text),
            "length_tokens": len(response_text.split()),
            "flagged_for_review": "[FLAG_FOR_REVIEW]" in response_text,
            "answer_extracted": self.extract_answer(response_text),
        }

        metrics["tool_count"] = len(metrics["tools_used"])

        return metrics

    def extract_approach(self, text: str) -> Optional[int]:
        """Extract approach selection (1-5)."""
        if not text or not isinstance(text, str):
            return None
        # Look for "Approach: 3" or "I'll use approach 2" or "(2)"
        patterns = [
            r"approach[:\s]+(\d)",
            r"option[:\s]+(\d)",
            r"select[:\s]+(\d)",
            r"choose[:\s]+(\d)",
            r"^\s*\((\d)\)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                num = int(match.group(1))
                if 1 <= num <= 5:
                    return num

        return None

    def extract_tools(self, text: str) -> list[str]:
        """Extract tool invocations."""
        if not text or not isinstance(text, str):
            return []

        tools = []

        # Pattern: [USE_TOOL: calculator | ...]
        tool_pattern = r"\[USE_TOOL:\s*(\w+)"
        matches = re.findall(tool_pattern, text, re.IGNORECASE)
        tools.extend(matches)

        # Also check for [FLAG_FOR_REVIEW]
        if "[FLAG_FOR_REVIEW]" in text:
            tools.append("flag_review")

        return tools

    def count_hedges(self, text: str) -> int:
        """Count hedging phrases."""
        if not text or not isinstance(text, str):
            return 0

        hedges = [
            "i think", "probably", "i'm not sure", "i am not sure",
            "i believe", "this might be", "it's possible", "its possible",
            "if i recall", "i'm not certain", "i am not certain",
            "perhaps", "likely", "arguably", "it seems", "appears to be",
            "could be", "may be", "might be", "possibly", "uncertain",
        ]

        text_lower = text.lower()
        count = sum(text_lower.count(hedge) for hedge in hedges)

        return count

    def extract_answer(self, text: str) -> str:
        """Extract the answer from the response."""
        if not text or not isinstance(text, str):
            return ""

        # Look for common answer patterns
        patterns = [
            r"ANSWER:\s*(.+?)(?:\n\n|$)",
            r"(?:final answer|solution|result):\s*(.+?)(?:\n\n|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()

        # Fallback: take last paragraph
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if paragraphs:
            return paragraphs[-1]

        return text.strip()
