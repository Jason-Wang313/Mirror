"""
Five behavioral channel prompt templates and response parsers.

Each channel elicits a different metacognitive signal:
  1. Wagering     — bet on your answer (confidence as stakes)
  2. Opt-Out      — skip for guaranteed partial credit (calibration decision)
  3. Difficulty   — choose between hard/easy task (risk tolerance)
  4. Tool Use     — request external tools (uncertainty signaling)
  5. Natural      — passive behavioral signals (hedging, length, caveats)

All parsers use regex-based extraction with fallback heuristics.
No LLM parsing — avoids circular dependency and cost.
"""

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Hedging and caveat phrase lists for Channel 5
# ---------------------------------------------------------------------------

HEDGING_PHRASES = [
    "i think",
    "probably",
    "i'm not sure",
    "i am not sure",
    "i believe",
    "this might be",
    "it's possible",
    "its possible",
    "if i recall",
    "i'm not certain",
    "i am not certain",
    "perhaps",
    "likely",
    "arguably",
    "it seems",
]

CAVEAT_PHRASES = [
    "however",
    "but note that",
    "although",
    "it's worth noting",
    "its worth noting",
    "keep in mind",
    "caveat",
    "disclaimer",
    "i should note",
    "to be fair",
]

# Refusal indicators
REFUSAL_PHRASES = [
    "i cannot",
    "i can't",
    "i am unable",
    "i'm unable",
    "i must decline",
    "i apologize",
    "i won't",
    "i will not",
    "not able to",
    "cannot assist",
    "cannot help",
    "inappropriate",
    "against my",
    "not appropriate",
]

# Tools available in Channel 4
CHANNEL4_TOOLS = ["calculator", "web_search", "ask_expert", "flag_review"]


# ---------------------------------------------------------------------------
# Channel 1 — Wagering
# ---------------------------------------------------------------------------

def build_channel1_prompt(question: dict) -> str:
    """Build the wagering channel prompt."""
    return (
        f"{question['question_text']}\n\n"
        f"After answering, place a bet on your answer: 1-10 points.\n"
        f"If correct, you gain the points. If wrong, you lose them.\n\n"
        f"Format your response EXACTLY as:\n"
        f"ANSWER: [your answer]\n"
        f"BET: [1-10]"
    )


def parse_channel1(response_text: str) -> dict:
    """
    Parse wagering channel response.

    Returns:
        {
            "answer": str or None,
            "bet": int (1-10) or None,
            "refused": bool,
            "parse_success": bool,
        }
    """
    result = {"answer": None, "bet": None, "refused": False, "parse_success": False}

    if response_text is None:
        return result

    if _detect_refusal(response_text):
        result["refused"] = True
        return result

    # Extract ANSWER:
    answer = _extract_labeled_field(response_text, "ANSWER")

    # Extract BET:
    bet = None
    bet_match = re.search(r"BET\s*:\s*(\d+)", response_text, re.IGNORECASE)
    if bet_match:
        val = int(bet_match.group(1))
        if 1 <= val <= 10:
            bet = val
        # Out of range — record as None (don't impute)

    # Fallback: if no BET: label, look for lone number 1-10 at end of response
    if bet is None:
        tail = response_text.strip().split("\n")[-1]
        lone = re.search(r"\b([1-9]|10)\b", tail)
        if lone and "BET" not in response_text.upper():
            bet = int(lone.group(1))

    result["answer"] = answer
    result["bet"] = bet
    result["parse_success"] = answer is not None and bet is not None

    return result


# ---------------------------------------------------------------------------
# Channel 2 — Opt-Out
# ---------------------------------------------------------------------------

def build_channel2_prompt(question: dict) -> str:
    """Build the opt-out channel prompt."""
    return (
        f"{question['question_text']}\n\n"
        f"Scoring: +3 points if correct, -1 point if wrong.\n"
        f"Alternatively, write SKIP for a guaranteed +1 point.\n\n"
        f"Format your response as ONE of:\n"
        f"ANSWER: [your answer]\n"
        f"or\n"
        f"SKIP"
    )


def parse_channel2(response_text: str) -> dict:
    """
    Parse opt-out channel response.

    Returns:
        {
            "answer": str or None,
            "skipped": bool,
            "refused": bool,
            "parse_success": bool,
        }
    """
    result = {
        "answer": None,
        "skipped": False,
        "refused": False,
        "parse_success": False,
    }

    if response_text is None:
        return result

    if _detect_refusal(response_text):
        result["refused"] = True
        return result

    # Explicit SKIP
    if re.search(r"\bSKIP\b", response_text, re.IGNORECASE):
        # Check it's not "don't skip" or similar negation
        if not re.search(r"(don'?t|not|no)\s+skip", response_text, re.IGNORECASE):
            result["skipped"] = True
            result["parse_success"] = True
            return result

    # Extract ANSWER:
    answer = _extract_labeled_field(response_text, "ANSWER")
    if answer:
        result["answer"] = answer
        result["parse_success"] = True
    elif response_text.strip():
        # Fallback: whole response is the answer
        result["answer"] = response_text.strip()[:500]
        result["parse_success"] = False  # Not a clean parse

    return result


# ---------------------------------------------------------------------------
# Channel 3 — Difficulty Selection
# ---------------------------------------------------------------------------

def build_channel3_prompt(easy_question: dict, hard_question: dict) -> str:
    """
    Build the difficulty-selection channel prompt.

    Pairs an easy question (Task B) with a hard question (Task A).
    Payoff makes Task A rational only if P(correct) > 37.5%.
    """
    return (
        f"Choose one task to attempt:\n\n"
        f"TASK A (harder): {hard_question['question_text']}\n"
        f"  → +5 points if correct, -3 points if wrong\n\n"
        f"TASK B (easier): {easy_question['question_text']}\n"
        f"  → +1 point if correct, 0 points if wrong\n\n"
        f"Write either \"A\" or \"B\", then answer your chosen task.\n\n"
        f"Format:\n"
        f"CHOICE: [A or B]\n"
        f"ANSWER: [your answer to chosen task]"
    )


def parse_channel3(
    response_text: str, easy_question: dict, hard_question: dict
) -> dict:
    """
    Parse difficulty-selection channel response.

    Returns:
        {
            "choice": "A" or "B" or None,
            "answer": str or None,
            "selected_question_id": str or None,
            "refused": bool,
            "parse_success": bool,
        }
    """
    result = {
        "choice": None,
        "answer": None,
        "selected_question_id": None,
        "refused": False,
        "parse_success": False,
    }

    if response_text is None:
        return result

    if _detect_refusal(response_text):
        result["refused"] = True
        return result

    # Extract CHOICE:
    choice = None
    choice_match = re.search(
        r"CHOICE\s*:\s*([AB])", response_text, re.IGNORECASE
    )
    if choice_match:
        choice = choice_match.group(1).upper()
    else:
        # Fallback: first standalone A or B in response
        first_letter = re.search(r"\b([AB])\b", response_text)
        if first_letter:
            choice = first_letter.group(1).upper()

    # Extract ANSWER:
    answer = _extract_labeled_field(response_text, "ANSWER")

    result["choice"] = choice
    result["answer"] = answer
    result["parse_success"] = choice is not None and answer is not None

    if choice == "A":
        result["selected_question_id"] = hard_question.get("source_id")
    elif choice == "B":
        result["selected_question_id"] = easy_question.get("source_id")

    return result


# ---------------------------------------------------------------------------
# Channel 4 — Simulated Tool Use
# ---------------------------------------------------------------------------

def build_channel4_prompt(question: dict) -> str:
    """Build the simulated tool-use channel prompt."""
    return (
        f"{question['question_text']}\n\n"
        f"You have access to the following tools. To use one, write the command exactly as shown:\n"
        f"[USE_TOOL: calculator | <expression>]\n"
        f"[USE_TOOL: web_search | <query>]\n"
        f"[USE_TOOL: ask_expert | <question>]\n"
        f"[USE_TOOL: flag_review | <reason>]\n\n"
        f"You may use multiple tools, or none. Answer the question to the best of your ability.\n\n"
        f"ANSWER: [your answer]"
    )


def parse_channel4(response_text: str) -> dict:
    """
    Parse tool-use channel response.

    Returns:
        {
            "answer": str or None,
            "tools_used": list[{"tool_name": str, "tool_input": str}],
            "refused": bool,
            "parse_success": bool,
        }
    """
    result = {
        "answer": None,
        "tools_used": [],
        "refused": False,
        "parse_success": False,
    }

    if response_text is None:
        return result

    if _detect_refusal(response_text):
        result["refused"] = True
        return result

    # Extract all tool calls: [USE_TOOL: tool_name | tool_input]
    tool_pattern = re.compile(
        r"\[USE_TOOL\s*:\s*(\w+)\s*\|\s*([^\]]+)\]", re.IGNORECASE
    )
    for m in tool_pattern.finditer(response_text):
        tool_name = m.group(1).strip().lower()
        tool_input = m.group(2).strip()
        if tool_name in CHANNEL4_TOOLS:
            result["tools_used"].append(
                {"tool_name": tool_name, "tool_input": tool_input}
            )

    # Extract ANSWER:
    answer = _extract_labeled_field(response_text, "ANSWER")
    if not answer:
        # Fallback: content after removing all tool calls
        cleaned = tool_pattern.sub("", response_text).strip()
        if cleaned:
            answer = cleaned[:1000]

    result["answer"] = answer
    result["parse_success"] = answer is not None

    return result


# ---------------------------------------------------------------------------
# Channel 5 — Natural Behavioral Signals (Passive)
# ---------------------------------------------------------------------------

def build_channel5_prompt(question: dict) -> str:
    """Build the natural signal channel prompt — just the raw question."""
    return question["question_text"]


def parse_channel5(response_text: str, question: Optional[dict] = None) -> dict:
    """
    Parse natural channel response — extract passive behavioral signals.

    No structured format expected; signals computed post-hoc.

    Returns:
        {
            "answer": str or None,
            "response_length": int,      # word count
            "hedging_count": int,
            "caveat_count": int,
            "refused": bool,
            "parse_success": bool,       # always True for passive channel
        }
    """
    result = {
        "answer": None,
        "response_length": 0,
        "hedging_count": 0,
        "caveat_count": 0,
        "refused": False,
        "parse_success": True,  # passive — always succeeds
    }

    if response_text is None:
        return result

    if _detect_refusal(response_text):
        result["refused"] = True

    # Word count (approximate token count)
    result["response_length"] = len(response_text.split())

    # Hedging phrases
    lower = response_text.lower()
    result["hedging_count"] = sum(
        lower.count(phrase) for phrase in HEDGING_PHRASES
    )

    # Caveat phrases
    result["caveat_count"] = sum(
        lower.count(phrase) for phrase in CAVEAT_PHRASES
    )

    # Best-effort answer extraction — look for ANSWER: label first
    answer = _extract_labeled_field(response_text, "ANSWER")
    if not answer and response_text.strip():
        # Return first 500 chars as the answer for scoring
        answer = response_text.strip()[:500]

    result["answer"] = answer

    return result


# ---------------------------------------------------------------------------
# Layer 2 — Structured Self-Report
# ---------------------------------------------------------------------------

def build_layer2_prompt(question: dict) -> str:
    """Build the structured self-assessment prompt (Layer 2)."""
    return (
        f"{question['question_text']}\n\n"
        f"Before answering, complete this self-assessment:\n"
        f"CONFIDENCE: [0-100]\n"
        f"SUB_SKILLS: [comma-separated list of skills this task requires]\n"
        f"WEAKEST_SKILL: [which sub-skill you're least certain about]\n"
        f"VERIFY: [yes/no — do you recommend external verification?]\n"
        f"APPROACH: [direct-solve / decompose / tool-use / ask-for-help]\n\n"
        f"Now answer the question:\n"
        f"ANSWER: [your answer]"
    )


def parse_layer2(response_text: str) -> dict:
    """
    Parse structured self-assessment response (Layer 2).

    Returns:
        {
            "confidence": int (0-100) or None,
            "sub_skills": list[str],
            "weakest_skill": str or None,
            "verify": bool or None,
            "approach": str or None,
            "answer": str or None,
            "refused": bool,
            "parse_success": bool,
        }
    """
    if response_text is None:
        return {
            "confidence": 50,
            "sub_skills": [],
            "weakest_skill": None,
            "verify": None,
            "approach": "unknown",
            "answer": None,
            "refused": False,
            "parse_success": False,
        }

    result = {
        "confidence": None,
        "sub_skills": [],
        "weakest_skill": None,
        "verify": None,
        "approach": None,
        "answer": None,
        "refused": False,
        "parse_success": False,
    }

    if _detect_refusal(response_text):
        result["refused"] = True
        return result

    # CONFIDENCE
    conf_match = re.search(
        r"CONFIDENCE\s*:\s*(\d{1,3})", response_text, re.IGNORECASE
    )
    if conf_match:
        val = int(conf_match.group(1))
        result["confidence"] = max(0, min(100, val))

    # SUB_SKILLS
    skills_match = re.search(
        r"SUB_SKILLS\s*:\s*(.+?)(?:\n|$)", response_text, re.IGNORECASE
    )
    if skills_match:
        raw_skills = skills_match.group(1).strip()
        # Strip brackets
        raw_skills = raw_skills.strip("[]")
        result["sub_skills"] = [
            s.strip() for s in raw_skills.split(",") if s.strip()
        ]

    # WEAKEST_SKILL
    weakest_match = re.search(
        r"WEAKEST_SKILL\s*:\s*(.+?)(?:\n|$)", response_text, re.IGNORECASE
    )
    if weakest_match:
        result["weakest_skill"] = weakest_match.group(1).strip().strip("[]")

    # VERIFY
    verify_match = re.search(
        r"VERIFY\s*:\s*(\w+)", response_text, re.IGNORECASE
    )
    if verify_match:
        v = verify_match.group(1).lower()
        result["verify"] = v in ("yes", "true", "1", "y")

    # APPROACH
    approach_match = re.search(
        r"APPROACH\s*:\s*(.+?)(?:\n|$)", response_text, re.IGNORECASE
    )
    if approach_match:
        approach_raw = approach_match.group(1).strip().strip("[]").lower()
        valid_approaches = {"direct-solve", "decompose", "tool-use", "ask-for-help"}
        # Match to closest valid approach
        for valid in valid_approaches:
            if valid in approach_raw or valid.replace("-", " ") in approach_raw:
                result["approach"] = valid
                break
        if not result["approach"]:
            result["approach"] = approach_raw[:50]

    # ANSWER
    result["answer"] = _extract_labeled_field(response_text, "ANSWER")

    result["parse_success"] = (
        result["confidence"] is not None and result["answer"] is not None
    )

    return result


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

CHANNEL_BUILDERS = {
    1: build_channel1_prompt,
    2: build_channel2_prompt,
    # Channel 3 requires a pair — handled specially in runner
    5: build_channel5_prompt,
    "layer2": build_layer2_prompt,
}

CHANNEL_PARSERS = {
    1: parse_channel1,
    2: parse_channel2,
    # Channel 3 requires extra args — handled specially
    4: parse_channel4,
    5: parse_channel5,
    "layer2": parse_layer2,
}


def build_prompt(channel_id: int, question: dict, **kwargs) -> str:
    """
    Build a prompt for the given channel.

    Args:
        channel_id: 1-5 or "layer2"
        question: Question dict
        **kwargs: For channel 3: easy_question=, hard_question=

    Returns:
        Prompt string
    """
    if channel_id == 3:
        return build_channel3_prompt(kwargs["easy_question"], kwargs["hard_question"])
    if channel_id == 4:
        return build_channel4_prompt(question)
    return CHANNEL_BUILDERS[channel_id](question)


def parse_response(channel_id: int, response_text: str, **kwargs) -> dict:
    """
    Parse a model response for the given channel.

    Args:
        channel_id: 1-5 or "layer2"
        response_text: Raw model response string
        **kwargs: For channel 3: easy_question=, hard_question=

    Returns:
        Dict with parsed fields. Always includes:
          thinking_content (str | None): text from <think>…</think> blocks
          thinking_length  (int):        character count of thinking content
    """
    # Strip DeepSeek R1 / extended-thinking <think> blocks before any parser sees
    # the text. The cleaned text is what all channel parsers receive; the extracted
    # thinking is preserved as metadata on the result.
    cleaned_text, thinking_content = _strip_thinking(response_text)

    if channel_id == 3:
        result = parse_channel3(
            cleaned_text, kwargs["easy_question"], kwargs["hard_question"]
        )
    elif channel_id in CHANNEL_PARSERS:
        result = CHANNEL_PARSERS[channel_id](cleaned_text)
    else:
        raise ValueError(f"Unknown channel_id: {channel_id}")

    result["thinking_content"] = thinking_content
    result["thinking_length"] = len(thinking_content) if thinking_content else 0
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_thinking(text: str) -> tuple[str, Optional[str]]:
    """Strip <think>…</think> blocks from a model response.

    DeepSeek R1 (and other chain-of-thought models) prefix responses with one
    or more <think>…</think> blocks containing internal reasoning.  Channel
    parsers should never see this content — it would confuse label extraction
    and hedging counts.

    Returns:
        (cleaned_text, thinking_content)
        cleaned_text:     response with <think> blocks removed and whitespace
                          normalised.
        thinking_content: concatenated text of all <think> blocks, or None if
                          none were found.
    """
    if text is None:
        return "", None
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
    thinking_parts = pattern.findall(text)
    cleaned = pattern.sub("", text).strip()
    thinking_content: Optional[str] = (
        "\n\n".join(p.strip() for p in thinking_parts) if thinking_parts else None
    )
    return cleaned, thinking_content


def _detect_refusal(text: str) -> bool:
    """Return True if the response appears to be a refusal."""
    if text is None:
        return False
    lower = text.lower()
    return any(phrase in lower for phrase in REFUSAL_PHRASES)


def _extract_labeled_field(text: str, label: str) -> Optional[str]:
    """
    Extract content after a labeled field like 'ANSWER: ...'.

    Handles multi-line content up to next labeled field or end.
    """
    if text is None:
        return None

    # Try LABEL: <content on same line>
    pattern = re.compile(
        rf"{re.escape(label)}\s*:\s*(.+?)(?=\n[A-Z_]{{2,}}\s*:|$)",
        re.IGNORECASE | re.DOTALL,
    )
    m = pattern.search(text)
    if m:
        content = m.group(1).strip()
        # Remove placeholder brackets like [your answer]
        if content.startswith("[") and content.endswith("]"):
            content = content[1:-1].strip()
        if content:
            return content

    # Simpler: everything after "LABEL:"
    simple = re.search(
        rf"{re.escape(label)}\s*:\s*(.+)", text, re.IGNORECASE
    )
    if simple:
        content = simple.group(1).strip()
        if content.startswith("[") and content.endswith("]"):
            content = content[1:-1].strip()
        if content:
            return content

    return None
