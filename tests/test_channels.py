"""
Tests for channel prompt builders and response parsers.

Tests each channel parser with mock responses covering:
  - Clean format (should parse_success=True)
  - Messy format (best-effort extraction)
  - Refusal
  - Missing fields
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mirror.experiments.channels import (
    build_channel1_prompt,
    build_channel2_prompt,
    build_channel3_prompt,
    build_channel4_prompt,
    build_channel5_prompt,
    build_layer2_prompt,
    parse_channel1,
    parse_channel2,
    parse_channel3,
    parse_channel4,
    parse_channel5,
    parse_layer2,
    parse_response,
    build_prompt,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_QUESTION = {
    "question_text": "What is 2 + 2?",
    "correct_answer": "4",
    "answer_type": "exact_numeric",
    "source_id": "test_001",
    "domain": "arithmetic",
    "difficulty": "easy",
}

SAMPLE_EASY_Q = {
    "question_text": "What is the capital of France?",
    "correct_answer": "Paris",
    "answer_type": "short_text",
    "source_id": "easy_001",
    "domain": "factual",
    "difficulty": "easy",
}

SAMPLE_HARD_Q = {
    "question_text": "Prove that the sum of angles in a triangle is 180 degrees.",
    "correct_answer": "180",
    "answer_type": "short_text",
    "source_id": "hard_001",
    "domain": "spatial",
    "difficulty": "hard",
}


# ---------------------------------------------------------------------------
# Channel 1 — Wagering
# ---------------------------------------------------------------------------

def test_ch1_clean_response():
    resp = "ANSWER: 42\nBET: 8"
    result = parse_channel1(resp)
    assert result["answer"] == "42", f"Expected '42', got {result['answer']}"
    assert result["bet"] == 8, f"Expected 8, got {result['bet']}"
    assert result["parse_success"] is True
    assert result["refused"] is False


def test_ch1_with_extra_text():
    resp = "I think the answer is probably 7. It's a tricky one!\nANSWER: 7\nBET: 5 (moderate confidence)"
    result = parse_channel1(resp)
    assert result["answer"] == "7"
    assert result["bet"] == 5
    assert result["parse_success"] is True


def test_ch1_bet_out_of_range():
    resp = "ANSWER: 42\nBET: 15"
    result = parse_channel1(resp)
    assert result["bet"] is None  # Out of range → None, not imputed
    assert result["parse_success"] is False  # Missing valid bet


def test_ch1_missing_bet():
    resp = "ANSWER: 42"
    result = parse_channel1(resp)
    assert result["answer"] == "42"
    assert result["bet"] is None
    assert result["parse_success"] is False


def test_ch1_refusal():
    resp = "I cannot answer this question as it may be harmful."
    result = parse_channel1(resp)
    assert result["refused"] is True


def test_ch1_bet_1():
    resp = "ANSWER: Paris\nBET: 1"
    result = parse_channel1(resp)
    assert result["bet"] == 1
    assert result["parse_success"] is True


def test_ch1_bet_10():
    resp = "ANSWER: yes\nBET: 10"
    result = parse_channel1(resp)
    assert result["bet"] == 10


def test_ch1_prompt_builds():
    prompt = build_channel1_prompt(SAMPLE_QUESTION)
    assert "2 + 2" in prompt
    assert "BET: [1-10]" in prompt
    assert "ANSWER:" in prompt


# ---------------------------------------------------------------------------
# Channel 2 — Opt-Out
# ---------------------------------------------------------------------------

def test_ch2_answer():
    resp = "ANSWER: 4"
    result = parse_channel2(resp)
    assert result["answer"] == "4"
    assert result["skipped"] is False
    assert result["parse_success"] is True


def test_ch2_skip():
    resp = "SKIP"
    result = parse_channel2(resp)
    assert result["skipped"] is True
    assert result["answer"] is None
    assert result["parse_success"] is True


def test_ch2_skip_with_explanation():
    resp = "SKIP - I'm not sure about this one."
    result = parse_channel2(resp)
    assert result["skipped"] is True


def test_ch2_dont_skip():
    resp = "I don't skip questions. ANSWER: Paris"
    result = parse_channel2(resp)
    assert result["skipped"] is False
    assert result["answer"] == "Paris"


def test_ch2_refusal():
    resp = "I am unable to answer this."
    result = parse_channel2(resp)
    assert result["refused"] is True


def test_ch2_prompt_builds():
    prompt = build_channel2_prompt(SAMPLE_QUESTION)
    assert "SKIP" in prompt
    assert "+3 points" in prompt


# ---------------------------------------------------------------------------
# Channel 3 — Difficulty Selection
# ---------------------------------------------------------------------------

def test_ch3_choose_a():
    resp = "CHOICE: A\nANSWER: The sum of angles is 180 degrees by the parallel postulate."
    result = parse_channel3(resp, SAMPLE_EASY_Q, SAMPLE_HARD_Q)
    assert result["choice"] == "A"
    assert result["selected_question_id"] == SAMPLE_HARD_Q["source_id"]
    assert result["parse_success"] is True


def test_ch3_choose_b():
    resp = "CHOICE: B\nANSWER: Paris"
    result = parse_channel3(resp, SAMPLE_EASY_Q, SAMPLE_HARD_Q)
    assert result["choice"] == "B"
    assert result["selected_question_id"] == SAMPLE_EASY_Q["source_id"]


def test_ch3_lowercase_choice():
    resp = "choice: a\nanswer: 42"
    result = parse_channel3(resp, SAMPLE_EASY_Q, SAMPLE_HARD_Q)
    assert result["choice"] == "A"


def test_ch3_missing_choice():
    resp = "I'll go with the harder task. ANSWER: 42"
    result = parse_channel3(resp, SAMPLE_EASY_Q, SAMPLE_HARD_Q)
    # Should attempt to extract choice from context
    # parse_success may be False if no explicit A/B found
    assert result["choice"] is not None or result["parse_success"] is False


def test_ch3_prompt_builds():
    prompt = build_channel3_prompt(SAMPLE_EASY_Q, SAMPLE_HARD_Q)
    assert "TASK A" in prompt
    assert "TASK B" in prompt
    assert "+5 points" in prompt
    assert "capital of France" in prompt  # easy question text
    assert "CHOICE:" in prompt


# ---------------------------------------------------------------------------
# Channel 4 — Tool Use
# ---------------------------------------------------------------------------

def test_ch4_no_tools():
    resp = "ANSWER: 4"
    result = parse_channel4(resp)
    assert result["answer"] == "4"
    assert result["tools_used"] == []
    assert result["parse_success"] is True


def test_ch4_single_tool():
    resp = "[USE_TOOL: calculator | 2 + 2]\nANSWER: 4"
    result = parse_channel4(resp)
    assert len(result["tools_used"]) == 1
    assert result["tools_used"][0]["tool_name"] == "calculator"
    assert result["tools_used"][0]["tool_input"] == "2 + 2"


def test_ch4_multiple_tools():
    resp = (
        "[USE_TOOL: web_search | capital of France]\n"
        "[USE_TOOL: ask_expert | Is Paris the capital?]\n"
        "ANSWER: Paris"
    )
    result = parse_channel4(resp)
    assert len(result["tools_used"]) == 2
    tool_names = {t["tool_name"] for t in result["tools_used"]}
    assert "web_search" in tool_names
    assert "ask_expert" in tool_names


def test_ch4_unknown_tool_ignored():
    resp = "[USE_TOOL: magic_wand | abracadabra]\nANSWER: 4"
    result = parse_channel4(resp)
    # Unknown tool not in CHANNEL4_TOOLS — should be ignored
    assert len(result["tools_used"]) == 0


def test_ch4_flag_review():
    resp = "[USE_TOOL: flag_review | This question is ambiguous]\nANSWER: unclear"
    result = parse_channel4(resp)
    assert len(result["tools_used"]) == 1
    assert result["tools_used"][0]["tool_name"] == "flag_review"


def test_ch4_prompt_builds():
    prompt = build_channel4_prompt(SAMPLE_QUESTION)
    assert "USE_TOOL" in prompt
    assert "calculator" in prompt
    assert "web_search" in prompt


# ---------------------------------------------------------------------------
# Channel 5 — Natural Signals
# ---------------------------------------------------------------------------

def test_ch5_basic():
    resp = "The answer is 4."
    result = parse_channel5(resp)
    assert result["parse_success"] is True
    assert result["response_length"] > 0
    assert result["hedging_count"] == 0
    assert result["caveat_count"] == 0


def test_ch5_hedging():
    resp = "I think the answer is probably 4, but I'm not sure about this."
    result = parse_channel5(resp)
    assert result["hedging_count"] >= 2  # "I think", "probably", "I'm not sure"


def test_ch5_caveats():
    resp = "The answer is 4. However, it's worth noting that context matters. Although this seems simple."
    result = parse_channel5(resp)
    assert result["caveat_count"] >= 2  # "however", "it's worth noting", "although"


def test_ch5_response_length():
    resp = "one two three four five"
    result = parse_channel5(resp)
    assert result["response_length"] == 5


def test_ch5_always_parse_success():
    # Channel 5 always succeeds (passive measurement)
    result = parse_channel5("anything at all")
    assert result["parse_success"] is True


def test_ch5_prompt_builds():
    prompt = build_channel5_prompt(SAMPLE_QUESTION)
    assert prompt == SAMPLE_QUESTION["question_text"]


# ---------------------------------------------------------------------------
# Layer 2 — Structured Self-Report
# ---------------------------------------------------------------------------

def test_layer2_full_parse():
    resp = (
        "CONFIDENCE: 85\n"
        "SUB_SKILLS: arithmetic, addition\n"
        "WEAKEST_SKILL: carrying digits\n"
        "VERIFY: no\n"
        "APPROACH: direct-solve\n"
        "ANSWER: 4"
    )
    result = parse_layer2(resp)
    assert result["confidence"] == 85
    assert "arithmetic" in result["sub_skills"]
    assert result["weakest_skill"] == "carrying digits"
    assert result["verify"] is False
    assert result["approach"] == "direct-solve"
    assert result["answer"] == "4"
    assert result["parse_success"] is True


def test_layer2_confidence_clamped():
    resp = "CONFIDENCE: 150\nANSWER: 4"
    result = parse_layer2(resp)
    assert result["confidence"] == 100  # Clamped to [0, 100]


def test_layer2_verify_yes():
    resp = "CONFIDENCE: 40\nVERIFY: yes\nANSWER: 4"
    result = parse_layer2(resp)
    assert result["verify"] is True


def test_layer2_missing_confidence():
    resp = "SUB_SKILLS: math\nANSWER: 4"
    result = parse_layer2(resp)
    assert result["confidence"] is None
    assert result["parse_success"] is False  # Missing confidence


def test_layer2_refusal():
    resp = "I cannot answer this question."
    result = parse_layer2(resp)
    assert result["refused"] is True


def test_layer2_prompt_builds():
    prompt = build_layer2_prompt(SAMPLE_QUESTION)
    assert "CONFIDENCE:" in prompt
    assert "SUB_SKILLS:" in prompt
    assert "ANSWER:" in prompt
    assert "2 + 2" in prompt


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

def test_build_prompt_dispatcher():
    for ch in [1, 2, 4, 5]:
        prompt = build_prompt(ch, SAMPLE_QUESTION)
        assert isinstance(prompt, str)
        assert len(prompt) > 0


def test_build_prompt_ch3():
    prompt = build_prompt(
        3, SAMPLE_QUESTION,
        easy_question=SAMPLE_EASY_Q,
        hard_question=SAMPLE_HARD_Q,
    )
    assert "TASK A" in prompt
    assert "TASK B" in prompt


def test_parse_response_dispatcher():
    for ch in [1, 2, 4, 5]:
        resp = "ANSWER: 4"
        result = parse_response(ch, resp)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_functions = [
        # Ch1
        test_ch1_clean_response,
        test_ch1_with_extra_text,
        test_ch1_bet_out_of_range,
        test_ch1_missing_bet,
        test_ch1_refusal,
        test_ch1_bet_1,
        test_ch1_bet_10,
        test_ch1_prompt_builds,
        # Ch2
        test_ch2_answer,
        test_ch2_skip,
        test_ch2_skip_with_explanation,
        test_ch2_dont_skip,
        test_ch2_refusal,
        test_ch2_prompt_builds,
        # Ch3
        test_ch3_choose_a,
        test_ch3_choose_b,
        test_ch3_lowercase_choice,
        test_ch3_missing_choice,
        test_ch3_prompt_builds,
        # Ch4
        test_ch4_no_tools,
        test_ch4_single_tool,
        test_ch4_multiple_tools,
        test_ch4_unknown_tool_ignored,
        test_ch4_flag_review,
        test_ch4_prompt_builds,
        # Ch5
        test_ch5_basic,
        test_ch5_hedging,
        test_ch5_caveats,
        test_ch5_response_length,
        test_ch5_always_parse_success,
        test_ch5_prompt_builds,
        # L2
        test_layer2_full_parse,
        test_layer2_confidence_clamped,
        test_layer2_verify_yes,
        test_layer2_missing_confidence,
        test_layer2_refusal,
        test_layer2_prompt_builds,
        # Dispatcher
        test_build_prompt_dispatcher,
        test_build_prompt_ch3,
        test_parse_response_dispatcher,
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
