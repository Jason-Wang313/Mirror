"""
Enhanced answer matching for experiment scoring.

Extends mirror/data/answer_matcher.py with:
  - Fraction parsing ("3/4" → 0.75)
  - "approximately X" extraction
  - "The answer is X" extraction
  - Substring match for short_text (factual recall)
  - Refusal detection
  - Type inference fallback (numeric → MC → text)
  - Channel-aware answer extraction (strips ANSWER:, BET:, etc.)

Import hierarchy: this module extends the data matcher; both can be used
independently. Use this module in scoring pipelines.
"""

import re
from typing import Optional

# Re-export core functions from data module for convenience
from ..data.answer_matcher import (
    match_answer,
    normalize_boolean,
    normalize_multiple_choice,
    normalize_numeric,
    normalize_text,
)

__all__ = [
    "match_answer",
    "match_answer_robust",
    "extract_answer_from_response",
    "detect_refusal",
    "normalize_numeric",
    "normalize_multiple_choice",
    "normalize_boolean",
    "normalize_text",
]

# ---------------------------------------------------------------------------
# Refusal detection
# ---------------------------------------------------------------------------

_REFUSAL_PHRASES = [
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
    "i refuse",
]


def detect_refusal(text: str) -> bool:
    """Return True if the text appears to be a safety/capability refusal."""
    lower = text.lower()
    return any(phrase in lower for phrase in _REFUSAL_PHRASES)


# ---------------------------------------------------------------------------
# Enhanced answer extraction
# ---------------------------------------------------------------------------

def extract_answer_from_response(
    response_text: str,
    answer_type: Optional[str] = None,
    channel_id=None,
) -> Optional[str]:
    """
    Extract the answer from a model response with multiple fallback strategies.

    Strategy order:
    1. ANSWER: label extraction
    2. Channel-specific extraction (e.g., Channel 2 SKIP)
    3. Explicit final-answer patterns ("the answer is X", "= X", "therefore X")
    4. "approximately X" for numeric
    5. MC letter extraction (if answer_type == multiple_choice)
    6. Last standalone number in response (for numeric types)
    7. Full response as last resort

    Args:
        response_text: Raw model response
        answer_type: Expected answer type (exact_numeric, multiple_choice, etc.)
        channel_id: Channel that generated this response (int or None)

    Returns:
        Extracted answer string, or None
    """
    if not response_text or not response_text.strip():
        return None

    # 1. ANSWER: label
    answer = _extract_after_label(response_text, "ANSWER")
    if answer:
        return answer

    # 2. Channel-specific: Channel 2 SKIP
    if channel_id == 2 and re.search(r"\bSKIP\b", response_text, re.IGNORECASE):
        return "SKIP"

    # 3a. "The answer is X" / "answer: X" / "answer = X"
    the_answer = re.search(
        r"(?:the\s+answer\s+is|answer\s*[:=])\s*([^\n.]+)",
        response_text,
        re.IGNORECASE,
    )
    if the_answer:
        return the_answer.group(1).strip()

    # 3b. Conclusive transition phrases: "Therefore, X", "So, X", "Thus, X",
    #     "Hence, X" — capture what follows on the same line, favouring short values
    conclusive = re.search(
        r"(?:therefore|so|thus|hence)[,\s]+(?:the\s+\w+\s+(?:is|are|was|were)\s+)?"
        r"([^\n.]{1,80})",
        response_text,
        re.IGNORECASE,
    )
    if conclusive:
        candidate = conclusive.group(1).strip()
        # Prefer this only when it looks like a short answer (number or brief phrase)
        if len(candidate) <= 60:
            return candidate

    # 3c. "= X" at the end of the last non-empty line (final calculation step)
    for line in reversed(response_text.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        eq_end = re.search(r"=\s*([^\n=]{1,50})$", line)
        if eq_end:
            return eq_end.group(1).strip()
        break  # only check last non-empty line

    # 3d. "X years old", "X feet tall", "X dollars" etc. — number + unit as
    #     a standalone phrase at sentence end
    unit_phrase = re.search(
        r"\b([\d,.\-\+/]+)\s*"
        r"(?:years?\s+old|feet|foot|meters?|miles?|kg|pounds?|lbs?|"
        r"dollars?|cents?|hours?|days?|weeks?|months?|inches?|cm)\b"
        r"[^a-zA-Z0-9]*$",
        response_text,
        re.IGNORECASE,
    )
    if unit_phrase:
        return unit_phrase.group(1).strip()

    # 4. "approximately X" for numeric
    approx = re.search(
        r"approximately\s+([\d,.\-\+/]+)",
        response_text,
        re.IGNORECASE,
    )
    if approx:
        return approx.group(1).strip()

    # 5. MC letter extraction
    if answer_type == "multiple_choice":
        mc = re.search(r"\b([A-D])\b", response_text)
        if mc:
            return mc.group(1).upper()

    # 6. For numeric types: last standalone number in the response
    #    (heuristic: models state the final answer last)
    if answer_type in ("exact_numeric", None):
        all_numbers = re.findall(r"(?<![.\d])-?\d[\d,]*(?:\.\d+)?(?![.\d])", response_text)
        if all_numbers:
            return all_numbers[-1].replace(",", "")

    # 7. Last resort: first 200 chars (shorter than before to reduce noise)
    return response_text.strip()[:200]


# ---------------------------------------------------------------------------
# Robust answer matching
# ---------------------------------------------------------------------------

def match_answer_robust(
    predicted: str,
    correct: str,
    answer_type: str,
    metadata: Optional[dict] = None,
) -> bool:
    """
    Match predicted against correct answer with enhanced logic.

    Enhancements over base match_answer:
    - Fraction parsing: "3/4" → 0.75
    - "approximately X" → numeric X
    - Large integer tolerance: ±1 for integers > 100
    - Substring match for short_text (factual recall)
    - Type inference if answer_type unrecognized
    - Refusal → always False

    Args:
        predicted: Predicted answer string
        correct: Ground truth answer string
        answer_type: One of: exact_numeric, multiple_choice, boolean, short_text
        metadata: Optional metadata dict (for aliases, etc.)

    Returns:
        True if the answers match under the appropriate matching rules
    """
    if metadata is None:
        metadata = {}

    predicted = str(predicted).strip()
    correct = str(correct).strip()

    if detect_refusal(predicted):
        return False

    # For long prose responses (Channel 5, or any channel without a format
    # instruction), extract the actual answer before type-specific matching.
    # Threshold: 30 chars — short answers are passed through unchanged.
    if len(predicted) > 30:
        extracted = extract_answer_from_response(predicted, answer_type=answer_type)
        if extracted and len(extracted) < len(predicted):
            predicted = extracted.strip()

    # Try the extended logic first, fall back to base matcher
    try:
        if answer_type == "exact_numeric":
            return _match_numeric_robust(predicted, correct)

        elif answer_type == "multiple_choice":
            return _match_mc_robust(predicted, correct)

        elif answer_type == "boolean":
            return _match_boolean_robust(predicted, correct)

        elif answer_type == "short_text":
            return _match_text_robust(predicted, correct, metadata)

        else:
            # Unknown type — try in order: numeric, MC, text
            return _match_with_inference(predicted, correct, metadata)

    except Exception:
        # Final fallback: base matcher
        try:
            return match_answer(predicted, correct, answer_type, metadata)
        except Exception:
            return predicted.strip().lower() == correct.strip().lower()


# ---------------------------------------------------------------------------
# Type-specific matchers
# ---------------------------------------------------------------------------

def _match_numeric_robust(predicted: str, correct: str) -> bool:
    """Numeric matching with fraction support and relative tolerance."""
    try:
        pred_val = _parse_numeric_robust(predicted)
        corr_val = _parse_numeric_robust(correct)

        if pred_val is None or corr_val is None:
            return False

        # Two-regime tolerance:
        #   numbers < 1000  → ±0.01 absolute (handles fractions, floats; rejects
        #                      off-by-one on counting answers like 226 vs 227)
        #   numbers ≥ 1000  → ±1 absolute (large-scale computations may round by 1)
        if abs(corr_val) >= 1000 and corr_val == int(corr_val):
            return abs(pred_val - corr_val) <= 1
        return abs(pred_val - corr_val) <= 0.01

    except Exception:
        return False


def _match_mc_robust(predicted: str, correct: str) -> bool:
    """MC matching — handle 'The answer is A', 'A)', '(A)', etc."""
    try:
        pred_upper = predicted.upper().strip()
        corr_upper = correct.upper().strip()

        # Normalize both sides
        pred_letter = _extract_mc_letter(pred_upper)
        corr_letter = _extract_mc_letter(corr_upper)

        if pred_letter and corr_letter:
            return pred_letter == corr_letter

        # Fallback: only if neither side extracted a letter — no bare [A-D] search
        # (would match 'A' inside "PARIS", "BANANA", etc.)
        return False

    except Exception:
        return False


def _match_boolean_robust(predicted: str, correct: str) -> bool:
    """Boolean matching with word-boundary normalization (avoids 'correct' in 'incorrect' bug)."""
    _TRUE_WORDS = {"true", "yes", "correct", "t", "y", "1", "right"}
    _FALSE_WORDS = {"false", "no", "incorrect", "f", "n", "0", "wrong"}

    def to_bool(text: str):
        words = set(re.sub(r"[^\w]", " ", text.lower()).split())
        if words & _TRUE_WORDS:
            return True
        if words & _FALSE_WORDS:
            return False
        return None

    try:
        pred_bool = to_bool(predicted)
        corr_bool = to_bool(correct)
        if pred_bool is None or corr_bool is None:
            return False
        return pred_bool == corr_bool
    except Exception:
        return False


def _match_text_robust(predicted: str, correct: str, metadata: dict) -> bool:
    """
    Text matching with substring support for factual recall.

    - Case-insensitive exact match
    - Alias match
    - Substring: "Paris" matches "Paris, France"
    """
    pred_norm = normalize_text(predicted)
    corr_norm = normalize_text(correct)

    if pred_norm == corr_norm:
        return True

    # Alias check
    aliases = metadata.get("aliases", [])
    for alias in aliases:
        if normalize_text(alias) == pred_norm:
            return True

    # Substring: predicted is contained in correct, or vice versa
    if pred_norm and (pred_norm in corr_norm or corr_norm in pred_norm):
        return True

    return False


def _match_with_inference(predicted: str, correct: str, metadata: dict) -> bool:
    """Try numeric → MC → text in order when answer_type is unknown."""
    # Try numeric
    pred_num = _parse_numeric_robust(predicted)
    corr_num = _parse_numeric_robust(correct)
    if pred_num is not None and corr_num is not None:
        return abs(pred_num - corr_num) <= 0.01

    # Try MC (only if correct is a single letter A-D)
    if re.match(r"^[A-D]$", correct.strip().upper()):
        letter = _extract_mc_letter(predicted.upper())
        if letter:
            return letter == correct.strip().upper()

    # Try boolean
    try:
        return normalize_boolean(predicted) == normalize_boolean(correct)
    except ValueError:
        pass

    # Text fallback
    return _match_text_robust(predicted, correct, metadata)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_numeric_robust(text: str) -> Optional[float]:
    """
    Parse a number from text, supporting:
    - Plain numbers: "42", "-3.14"
    - Formatted: "$1,234.56", "75%"
    - Fractions: "3/4" → 0.75
    - "approximately 42" → 42
    - LaTeX simple: "$42$" → 42
    """
    if not text:
        return None

    text = text.strip()

    # Strip LaTeX delimiters
    text = re.sub(r"\$", "", text)

    # "approximately X"
    approx = re.search(r"approximately\s+([\d,.\-\+/]+)", text, re.IGNORECASE)
    if approx:
        text = approx.group(1)

    # Strip common measurement/currency unit words so "30 feet" → "30"
    text = re.sub(
        r"\b(feet|foot|meters?|kilometres?|kilometers?|km|miles?|"
        r"kilograms?|kg|grams?|gr?|pounds?|lbs?|ounces?|oz|"
        r"inches?|centimeters?|cm|millimeters?|mm|"
        r"hours?|minutes?|seconds?|days?|weeks?|months?|years?|"
        r"dollars?|cents?|euros?|degrees?|radians?|units?|"
        r"percent|pct)\b",
        " ", text, flags=re.IGNORECASE,
    ).strip()

    # Remove remaining formatting punctuation
    text = text.replace(",", "").replace("%", "").replace("$", "").strip()

    # Fraction: "3/4"
    frac = re.match(r"^(-?\d+)\s*/\s*(-?\d+)$", text)
    if frac:
        num, denom = int(frac.group(1)), int(frac.group(2))
        if denom == 0:
            return None
        return num / denom

    # Plain number
    num_match = re.search(r"[-+]?\d*\.?\d+", text)
    if num_match:
        return float(num_match.group())

    return None


def _extract_mc_letter(text: str) -> Optional[str]:
    """Extract single MC letter from text (A-D)."""
    # Patterns: "A", "(A)", "A)", "A.", "(A) some text"
    patterns = [
        r"^\s*([A-D])\s*$",           # Just the letter
        r"^\s*\(([A-D])\)",           # (A)
        r"^\s*([A-D])[.)\s]",         # A. or A) or A followed by space
        r"\bTHE ANSWER IS ([A-D])\b",  # "The answer is A"
        r"\bANSWER[: ]+([A-D])\b",    # "ANSWER: A"
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    # Fallback: first A-D word boundary
    m = re.search(r"\b([A-D])\b", text)
    if m:
        return m.group(1).upper()
    return None


def _extract_after_label(text: str, label: str) -> Optional[str]:
    """Extract content after 'LABEL: ...' up to next label or end."""
    pattern = re.compile(
        rf"{re.escape(label)}\s*:\s*(.+?)(?=\n[A-Z_]{{2,}}\s*:|$)",
        re.IGNORECASE | re.DOTALL,
    )
    m = pattern.search(text)
    if m:
        content = m.group(1).strip()
        if content.startswith("[") and content.endswith("]"):
            content = content[1:-1].strip()
        if content:
            return content

    # Simpler single-line extraction
    simple = re.search(rf"{re.escape(label)}\s*:\s*(.+)", text, re.IGNORECASE)
    if simple:
        content = simple.group(1).strip()
        if content.startswith("[") and content.endswith("]"):
            content = content[1:-1].strip()
        if content:
            return content

    return None
