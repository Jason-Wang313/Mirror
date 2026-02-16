"""
Answer matcher - defines matching rules for different answer types.

Provides flexible matching logic for scoring model responses.
"""

import re


def normalize_numeric(text: str) -> float:
    """
    Extract and normalize numeric value from text.

    Args:
        text: Text containing a number

    Returns:
        Normalized float value
    """
    # Remove common formatting
    text = text.replace(",", "").replace("$", "").replace("%", "")

    # Extract first number
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if match:
        return float(match.group())

    raise ValueError(f"No number found in: {text}")


def normalize_multiple_choice(text: str) -> str:
    """
    Normalize multiple choice answer to uppercase letter.

    Args:
        text: Text containing answer choice

    Returns:
        Normalized letter (A, B, C, D, etc.)
    """
    text = text.upper().strip()

    # Extract letter
    match = re.search(r"\b([A-Z])\b", text)
    if match:
        return match.group(1)

    # If text is just a letter with parens or period
    text = text.replace("(", "").replace(")", "").replace(".", "").strip()
    if len(text) == 1 and text.isalpha():
        return text

    return text


def normalize_boolean(text: str) -> bool:
    """
    Normalize boolean answer.

    Args:
        text: Text containing boolean value

    Returns:
        Normalized boolean
    """
    text = text.lower().strip()

    true_values = ["true", "yes", "correct", "t", "y", "1"]
    false_values = ["false", "no", "incorrect", "f", "n", "0"]

    if any(val in text for val in true_values):
        return True
    elif any(val in text for val in false_values):
        return False

    raise ValueError(f"Could not parse boolean from: {text}")


def normalize_text(text: str) -> str:
    """
    Normalize short text answer.

    Args:
        text: Text to normalize

    Returns:
        Normalized text
    """
    return text.lower().strip()


def match_answer(
    predicted: str,
    correct: str,
    answer_type: str,
    metadata: dict = None
) -> bool:
    """
    Check if predicted answer matches correct answer.

    Rules by type:
    - exact_numeric: Parse both to float, compare with tolerance (±0.01)
    - multiple_choice: Normalize to uppercase letter
    - short_text: Lowercase, strip, compare. Check aliases if available.
    - boolean: Normalize to True/False

    Args:
        predicted: Predicted answer
        correct: Correct answer
        answer_type: Type of answer
        metadata: Optional metadata (e.g., aliases)

    Returns:
        True if answers match
    """
    if metadata is None:
        metadata = {}

    try:
        if answer_type == "exact_numeric":
            pred_num = normalize_numeric(predicted)
            corr_num = normalize_numeric(correct)
            return abs(pred_num - corr_num) <= 0.01

        elif answer_type == "multiple_choice":
            pred_choice = normalize_multiple_choice(predicted)
            corr_choice = normalize_multiple_choice(correct)
            return pred_choice == corr_choice

        elif answer_type == "boolean":
            pred_bool = normalize_boolean(predicted)
            corr_bool = normalize_boolean(correct)
            return pred_bool == corr_bool

        elif answer_type == "short_text":
            pred_text = normalize_text(predicted)
            corr_text = normalize_text(correct)

            if pred_text == corr_text:
                return True

            # Check aliases
            aliases = metadata.get("aliases", [])
            for alias in aliases:
                if normalize_text(alias) == pred_text:
                    return True

            return False

        else:
            # Unknown type, fall back to exact string match
            return predicted.strip().lower() == correct.strip().lower()

    except Exception:
        # If parsing fails, fall back to string comparison
        return predicted.strip().lower() == correct.strip().lower()


def validate_answer_matching(sample_questions: list[dict], client):
    """
    Validate answer matching logic by comparing auto-scoring to LLM judge.

    Args:
        sample_questions: Sample of questions to validate
        client: UnifiedClient for LLM judge

    Returns:
        Agreement rate between auto-scorer and LLM judge
    """
    import asyncio

    async def judge_answer(question, predicted):
        prompt = f"""Question: {question['question_text']}

Predicted answer: {predicted}
Correct answer: {question['correct_answer']}

Is the predicted answer correct? Answer with ONLY "CORRECT" or "INCORRECT"."""

        response = await client.complete(
            model="llama-3.1-70b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
            metadata={"task": "answer_validation"}
        )

        if "error" in response:
            return None

        return "CORRECT" in response["content"].upper()

    async def run_validation():
        agreements = 0
        total = 0

        for q in sample_questions:
            # Get model response
            response = await client.complete(
                model="llama-3.1-8b",
                messages=[{"role": "user", "content": q["question_text"]}],
                temperature=0.0,
                max_tokens=128
            )

            if "error" in response:
                continue

            predicted = response["content"].strip()

            # Auto-score
            auto_correct = match_answer(
                predicted,
                q["correct_answer"],
                q["answer_type"],
                q.get("metadata", {})
            )

            # LLM judge
            llm_correct = await judge_answer(q, predicted)

            if llm_correct is not None:
                if auto_correct == llm_correct:
                    agreements += 1
                total += 1

        return agreements / total if total > 0 else 0.0

    return asyncio.run(run_validation())
