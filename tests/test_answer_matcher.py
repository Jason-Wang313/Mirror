"""
Tests for answer matching logic.

Covers:
  - Numeric: "42" vs "42.0", fractions, "approximately X", tolerance
  - Multiple choice: "A" vs "The answer is A", case insensitivity
  - Short text: exact, substring, aliases, case
  - Boolean: yes/no/true/false normalization
  - Type inference for unknown answer_type
  - Refusal detection → always False
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mirror.scoring.answer_matcher import (
    match_answer_robust,
    detect_refusal,
    extract_answer_from_response,
    _parse_numeric_robust,
    _extract_mc_letter,
)


# ---------------------------------------------------------------------------
# Numeric matching
# ---------------------------------------------------------------------------

def test_numeric_exact():
    assert match_answer_robust("42", "42", "exact_numeric")


def test_numeric_float_vs_int():
    assert match_answer_robust("42.0", "42", "exact_numeric")


def test_numeric_tolerance():
    # Values within ±0.01 → True
    assert match_answer_robust("3.14", "3.14159", "exact_numeric") is True   # diff ≈ 0.00159
    assert match_answer_robust("42.001", "42", "exact_numeric") is True      # diff = 0.001
    assert match_answer_robust("42.01", "42", "exact_numeric") is True       # diff = 0.01 (boundary)
    # Values outside ±0.01 → False
    assert match_answer_robust("3.16", "3.14159", "exact_numeric") is False  # diff ≈ 0.018
    assert match_answer_robust("42.02", "42", "exact_numeric") is False      # diff = 0.02
    assert match_answer_robust("42.015", "42", "exact_numeric") is False     # diff = 0.015


def test_numeric_fraction():
    assert match_answer_robust("3/4", "0.75", "exact_numeric")
    assert match_answer_robust("1/2", "0.5", "exact_numeric")
    assert match_answer_robust("0.75", "3/4", "exact_numeric")


def test_numeric_formatted():
    assert match_answer_robust("$1,234", "1234", "exact_numeric")
    assert match_answer_robust("75%", "75", "exact_numeric")


def test_numeric_approximately():
    assert match_answer_robust("approximately 42", "42", "exact_numeric")


def test_numeric_negative():
    assert match_answer_robust("-5", "-5", "exact_numeric")
    assert match_answer_robust("-5.0", "-5", "exact_numeric")


def test_numeric_large_int_tolerance():
    # Large integers: ±1 tolerance
    assert match_answer_robust("1000", "1001", "exact_numeric")
    assert match_answer_robust("1000", "1002", "exact_numeric") is False


def test_numeric_mismatch():
    assert match_answer_robust("42", "43", "exact_numeric") is False


# ---------------------------------------------------------------------------
# Multiple choice matching
# ---------------------------------------------------------------------------

def test_mc_exact():
    assert match_answer_robust("A", "A", "multiple_choice")
    assert match_answer_robust("B", "B", "multiple_choice")
    assert match_answer_robust("C", "C", "multiple_choice")


def test_mc_case_insensitive():
    assert match_answer_robust("a", "A", "multiple_choice")
    assert match_answer_robust("A", "a", "multiple_choice")


def test_mc_the_answer_is():
    assert match_answer_robust("The answer is A", "A", "multiple_choice")
    assert match_answer_robust("The answer is B", "A", "multiple_choice") is False


def test_mc_with_paren():
    assert match_answer_robust("(A)", "A", "multiple_choice")
    assert match_answer_robust("(B)", "B", "multiple_choice")


def test_mc_with_period():
    assert match_answer_robust("A.", "A", "multiple_choice")


def test_mc_with_explanation():
    assert match_answer_robust("A) Paris is the capital of France", "A", "multiple_choice")


def test_mc_answer_label():
    assert match_answer_robust("ANSWER: A", "A", "multiple_choice")


def test_mc_wrong():
    assert match_answer_robust("B", "A", "multiple_choice") is False


def test_mc_paris_not_letter():
    # "paris" should not match MC answer "A"
    assert match_answer_robust("paris", "A", "multiple_choice") is False


# ---------------------------------------------------------------------------
# Short text matching
# ---------------------------------------------------------------------------

def test_text_exact():
    assert match_answer_robust("Paris", "Paris", "short_text")


def test_text_case_insensitive():
    assert match_answer_robust("paris", "Paris", "short_text")
    assert match_answer_robust("PARIS", "paris", "short_text")


def test_text_with_extra_words():
    # "Paris, France" contains "Paris" — substring match
    assert match_answer_robust("Paris", "Paris, France", "short_text")
    assert match_answer_robust("Paris, France", "Paris", "short_text")


def test_text_mismatch():
    assert match_answer_robust("London", "Paris", "short_text") is False


def test_text_aliases():
    metadata = {"aliases": ["Paree", "City of Light"]}
    assert match_answer_robust("Paree", "Paris", "short_text", metadata)
    assert match_answer_robust("City of Light", "Paris", "short_text", metadata)
    assert match_answer_robust("London", "Paris", "short_text", metadata) is False


def test_text_whitespace():
    assert match_answer_robust("  Paris  ", "Paris", "short_text")


# ---------------------------------------------------------------------------
# Boolean matching
# ---------------------------------------------------------------------------

def test_bool_yes_true():
    assert match_answer_robust("yes", "true", "boolean")
    assert match_answer_robust("Yes", "True", "boolean")
    assert match_answer_robust("TRUE", "yes", "boolean")


def test_bool_no_false():
    assert match_answer_robust("no", "false", "boolean")
    assert match_answer_robust("No", "False", "boolean")
    assert match_answer_robust("FALSE", "no", "boolean")


def test_bool_correct_incorrect():
    assert match_answer_robust("correct", "true", "boolean")
    assert match_answer_robust("incorrect", "false", "boolean")


def test_bool_mismatch():
    assert match_answer_robust("yes", "false", "boolean") is False
    assert match_answer_robust("no", "true", "boolean") is False


# ---------------------------------------------------------------------------
# Refusal detection
# ---------------------------------------------------------------------------

def test_refusal_cannot():
    assert detect_refusal("I cannot answer this question.")


def test_refusal_wont():
    assert detect_refusal("I won't help with that.")


def test_refusal_unable():
    assert detect_refusal("I am unable to assist.")


def test_refusal_apologize():
    assert detect_refusal("I apologize but I can't do that.")


def test_refusal_against():
    assert detect_refusal("This is against my guidelines.")


def test_no_refusal():
    assert not detect_refusal("The answer is 42.")
    assert not detect_refusal("Paris is the capital of France.")
    assert not detect_refusal("I think the answer is probably 7.")


def test_refusal_always_false():
    # A refused response should never match anything
    refusal = "I cannot answer this question as it may be harmful."
    assert match_answer_robust(refusal, "42", "exact_numeric") is False
    assert match_answer_robust(refusal, "A", "multiple_choice") is False


# ---------------------------------------------------------------------------
# Type inference (unknown answer_type)
# ---------------------------------------------------------------------------

def test_unknown_type_numeric():
    # Should infer numeric
    assert match_answer_robust("42", "42", "unknown_type")


def test_unknown_type_mc():
    # Correct is a single letter → should infer MC
    assert match_answer_robust("A", "A", "unknown_type")


def test_unknown_type_text():
    # Falls through to text matching
    assert match_answer_robust("Paris", "Paris", "unknown_type")


# ---------------------------------------------------------------------------
# Answer extraction from responses
# ---------------------------------------------------------------------------

def test_extract_after_answer_label():
    resp = "ANSWER: 42\nBET: 7"
    extracted = extract_answer_from_response(resp)
    assert extracted == "42"


def test_extract_the_answer_is():
    resp = "After careful thought, the answer is Paris."
    extracted = extract_answer_from_response(resp, answer_type="short_text")
    assert extracted and "Paris" in extracted


def test_extract_mc_from_text():
    resp = "I believe option B is correct."
    extracted = extract_answer_from_response(resp, answer_type="multiple_choice")
    assert extracted == "B"


def test_extract_approximately():
    resp = "The value is approximately 3.14 based on my calculation."
    extracted = extract_answer_from_response(resp, answer_type="exact_numeric")
    assert extracted and "3.14" in extracted


def test_extract_empty_response():
    extracted = extract_answer_from_response("")
    assert extracted is None


# ---------------------------------------------------------------------------
# Numeric parser internals
# ---------------------------------------------------------------------------

def test_parse_fraction():
    assert abs(_parse_numeric_robust("3/4") - 0.75) < 1e-9


def test_parse_negative():
    assert _parse_numeric_robust("-5") == -5.0


def test_parse_formatted():
    assert _parse_numeric_robust("$1,234.56") == 1234.56


def test_parse_none():
    assert _parse_numeric_robust("no number here") is None
    assert _parse_numeric_robust("") is None


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_functions = [
        # Numeric
        test_numeric_exact,
        test_numeric_float_vs_int,
        test_numeric_tolerance,
        test_numeric_fraction,
        test_numeric_formatted,
        test_numeric_approximately,
        test_numeric_negative,
        test_numeric_large_int_tolerance,
        test_numeric_mismatch,
        # MC
        test_mc_exact,
        test_mc_case_insensitive,
        test_mc_the_answer_is,
        test_mc_with_paren,
        test_mc_with_period,
        test_mc_with_explanation,
        test_mc_answer_label,
        test_mc_wrong,
        test_mc_paris_not_letter,
        # Text
        test_text_exact,
        test_text_case_insensitive,
        test_text_with_extra_words,
        test_text_mismatch,
        test_text_aliases,
        test_text_whitespace,
        # Boolean
        test_bool_yes_true,
        test_bool_no_false,
        test_bool_correct_incorrect,
        test_bool_mismatch,
        # Refusal
        test_refusal_cannot,
        test_refusal_wont,
        test_refusal_unable,
        test_refusal_apologize,
        test_refusal_against,
        test_no_refusal,
        test_refusal_always_false,
        # Type inference
        test_unknown_type_numeric,
        test_unknown_type_mc,
        test_unknown_type_text,
        # Extraction
        test_extract_after_answer_label,
        test_extract_the_answer_is,
        test_extract_mc_from_text,
        test_extract_approximately,
        test_extract_empty_response,
        # Numeric parser
        test_parse_fraction,
        test_parse_negative,
        test_parse_formatted,
        test_parse_none,
    ]

    passed = 0
    failed = 0
    for test_fn in test_functions:
        try:
            test_fn()
            print(f"  ✅ {test_fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {test_fn.__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{passed+failed} tests passed")
    sys.exit(0 if failed == 0 else 1)
