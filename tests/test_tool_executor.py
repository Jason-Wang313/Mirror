"""Tests for mirror.experiments.tool_executor.ToolExecutor."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mirror.experiments.tool_executor import ToolExecutor


@pytest.fixture
def executor():
    return ToolExecutor()


# ---------------------------------------------------------------------------
# execute_calculator — basic arithmetic
# ---------------------------------------------------------------------------

def test_safe_eval_basic_arithmetic(executor):
    assert executor.execute_calculator("2 + 2") == "4"
    assert executor.execute_calculator("15 * 23") == "345"
    # 100/3 is not a whole number
    result = executor.execute_calculator("100 / 3")
    assert result.startswith("33.33333")


def test_safe_eval_subtraction_and_modulo(executor):
    assert executor.execute_calculator("10 - 3") == "7"
    assert executor.execute_calculator("17 % 5") == "2"
    assert executor.execute_calculator("17 // 5") == "3"


def test_safe_eval_powers(executor):
    assert executor.execute_calculator("2 ** 10") == "1024"
    result = executor.execute_calculator("3 ** 0.5")
    assert result.startswith("1.7320"), f"Expected ~1.732, got {result}"


def test_safe_eval_math_functions(executor):
    result = executor.execute_calculator("sqrt(4)")
    assert result == "2"
    assert executor.execute_calculator("abs(-7)") == "7"
    assert executor.execute_calculator("round(3.14159, 2)") == "3.14"


# ---------------------------------------------------------------------------
# execute_calculator — security rejection
# ---------------------------------------------------------------------------

def test_safe_eval_rejects_imports(executor):
    result = executor.execute_calculator("__import__('os')")
    assert result.startswith("Error:")


def test_safe_eval_rejects_builtins(executor):
    result = executor.execute_calculator("__builtins__")
    assert result.startswith("Error:")


def test_safe_eval_rejects_names(executor):
    # Variable names not in allowed set should be rejected
    result = executor.execute_calculator("x + 5")
    assert result.startswith("Error:")


def test_safe_eval_rejects_attribute_access(executor):
    result = executor.execute_calculator("os.getcwd()")
    assert result.startswith("Error:")


def test_safe_eval_rejects_system_calls(executor):
    # Multi-statement with semicolon (syntax error in expression mode)
    result = executor.execute_calculator("1; import os")
    assert result.startswith("Error:")


# ---------------------------------------------------------------------------
# execute_calculator — edge cases
# ---------------------------------------------------------------------------

def test_safe_eval_division_by_zero(executor):
    result = executor.execute_calculator("1 / 0")
    assert "division by zero" in result.lower()
    assert result.startswith("Error:")


def test_safe_eval_precision(executor):
    result = executor.execute_calculator("1 / 7")
    # Must have at most 6 decimal places
    if "." in result:
        decimals = result.split(".")[1]
        assert len(decimals) <= 6, f"Too many decimal places: {result}"


def test_safe_eval_empty_expression(executor):
    result = executor.execute_calculator("")
    assert result.startswith("Error:")


def test_safe_eval_whole_number_float(executor):
    # sqrt(4) = 2.0 — should be returned as "2" not "2.0"
    assert executor.execute_calculator("sqrt(4)") == "2"
    assert executor.execute_calculator("4.0 * 2.0") == "8"


# ---------------------------------------------------------------------------
# execute() wrapper
# ---------------------------------------------------------------------------

def test_execute_calculator_success(executor):
    r = executor.execute("calculator", "3 + 4")
    assert r["executed"] is True
    assert r["result"] == "7"
    assert r["error"] is None
    assert r["tool_name"] == "calculator"
    assert r["tool_input"] == "3 + 4"


def test_execute_calculator_error(executor):
    r = executor.execute("calculator", "1 / 0")
    assert r["executed"] is True
    assert r["result"] is None
    assert r["error"] is not None


def test_execute_unsupported_tool(executor):
    r = executor.execute("web_search", "population of France")
    assert r["executed"] is False
    assert r["error"] is not None
    assert r["result"] is None


# ---------------------------------------------------------------------------
# process_channel4_response — no tools
# ---------------------------------------------------------------------------

def test_process_no_tools(executor):
    parsed = {
        "answer": "42",
        "tools_used": [],
        "refused": False,
        "parse_success": True,
    }
    result = executor.process_channel4_response(
        parsed_response=parsed,
        model="llama-3.1-8b",
        prompt_base="What is 6 × 7?",
        client=MagicMock(),
    )
    assert result["tool_executed"] is False
    assert result["tool_results"] == []
    assert result["initial_answer"] == "42"
    assert result["final_answer"] == "42"
    assert result["answer_changed"] is False


# ---------------------------------------------------------------------------
# process_channel4_response — calculator tool
# ---------------------------------------------------------------------------

def _make_client(response_content: str) -> MagicMock:
    """Return a mock client whose complete_sync returns the given content."""
    client = MagicMock()
    client.complete_sync.return_value = {"content": response_content, "latency_ms": 100}
    return client


def test_process_calculator_tool(executor):
    parsed = {
        "answer": "let me calculate",
        "tools_used": [{"tool_name": "calculator", "tool_input": "15 * 23"}],
        "refused": False,
        "parse_success": True,
    }
    client = _make_client("The result is 345.\nANSWER: 345")
    result = executor.process_channel4_response(
        parsed_response=parsed,
        model="llama-3.1-8b",
        prompt_base="What is 15 times 23?",
        client=client,
        raw_response="Let me use the calculator.\n[USE_TOOL: calculator | 15*23]",
    )
    assert result["tool_executed"] is True
    assert len(result["tool_results"]) == 1
    assert result["tool_results"][0]["result"] == "345"
    assert result["final_answer"] == "345"
    # The second turn was sent to the API
    assert client.complete_sync.called


def test_second_turn_format(executor):
    """Verify the second-turn message is formatted as specified."""
    parsed = {
        "answer": "...",
        "tools_used": [{"tool_name": "calculator", "tool_input": "7 + 8"}],
        "refused": False,
        "parse_success": True,
    }
    client = _make_client("ANSWER: 15")
    executor.process_channel4_response(
        parsed_response=parsed,
        model="llama-3.1-8b",
        prompt_base="What is 7 + 8?",
        client=client,
        raw_response="[USE_TOOL: calculator | 7+8]",
    )
    call_args = client.complete_sync.call_args
    messages = call_args.kwargs.get("messages") or call_args.args[0]
    # Find the second-turn user message (3rd message)
    assert len(messages) == 3
    second_turn = messages[2]["content"]
    assert "Tool results:" in second_turn
    assert "[TOOL_RESULT: calculator | 7 + 8 = 15]" in second_turn
    assert "Now provide your final answer." in second_turn
    assert "ANSWER:" in second_turn


# ---------------------------------------------------------------------------
# process_channel4_response — unsupported (record-only) tool
# ---------------------------------------------------------------------------

def test_process_unsupported_tool(executor):
    parsed = {
        "answer": "Paris",
        "tools_used": [{"tool_name": "web_search", "tool_input": "capital of France"}],
        "refused": False,
        "parse_success": True,
    }
    client = MagicMock()
    result = executor.process_channel4_response(
        parsed_response=parsed,
        model="llama-3.1-8b",
        prompt_base="What is the capital of France?",
        client=client,
    )
    # web_search is record-only, no second turn
    assert result["tool_executed"] is False
    assert not client.complete_sync.called
    assert result["tool_results"][0]["executed"] is False


# ---------------------------------------------------------------------------
# process_channel4_response — multiple tools
# ---------------------------------------------------------------------------

def test_process_multiple_tools(executor):
    parsed = {
        "answer": "maybe",
        "tools_used": [
            {"tool_name": "calculator", "tool_input": "2 ** 8"},
            {"tool_name": "web_search", "tool_input": "powers of 2"},
        ],
        "refused": False,
        "parse_success": True,
    }
    client = _make_client("ANSWER: 256")
    result = executor.process_channel4_response(
        parsed_response=parsed,
        model="llama-3.1-8b",
        prompt_base="What is 2 to the 8th power?",
        client=client,
        raw_response="[USE_TOOL: calculator | 2**8]\n[USE_TOOL: web_search | powers of 2]",
    )
    assert result["tool_executed"] is True
    assert len(result["tool_results"]) == 2
    calc_result = next(r for r in result["tool_results"] if r["tool_name"] == "calculator")
    web_result = next(r for r in result["tool_results"] if r["tool_name"] == "web_search")
    assert calc_result["executed"] is True
    assert web_result["executed"] is False


# ---------------------------------------------------------------------------
# process_channel4_response — malformed expression
# ---------------------------------------------------------------------------

def test_process_malformed_expression(executor):
    parsed = {
        "answer": "unknown",
        "tools_used": [{"tool_name": "calculator", "tool_input": "abc + xyz"}],
        "refused": False,
        "parse_success": True,
    }
    client = _make_client("I couldn't compute that.\nANSWER: unknown")
    result = executor.process_channel4_response(
        parsed_response=parsed,
        model="llama-3.1-8b",
        prompt_base="...",
        client=client,
    )
    # Tool should be marked executed but with an error
    assert result["tool_executed"] is True
    assert result["tool_results"][0]["error"] is not None
    # Second turn was still sent (with the error in the tool result)
    assert client.complete_sync.called


# ---------------------------------------------------------------------------
# process_channel4_response — answer_changed flag
# ---------------------------------------------------------------------------

def test_answer_changed_flag(executor):
    parsed = {
        "answer": "wrong initial",
        "tools_used": [{"tool_name": "calculator", "tool_input": "3 * 9"}],
        "refused": False,
        "parse_success": True,
    }
    client = _make_client("The answer is 27.\nANSWER: 27")
    result = executor.process_channel4_response(
        parsed_response=parsed,
        model="llama-3.1-8b",
        prompt_base="What is 3 × 9?",
        client=client,
        raw_response="[USE_TOOL: calculator | 3*9]",
    )
    assert result["answer_changed"] is True
    assert result["initial_answer"] == "wrong initial"
    assert result["final_answer"] == "27"


def test_answer_not_changed_flag(executor):
    parsed = {
        "answer": "27",
        "tools_used": [{"tool_name": "calculator", "tool_input": "3 * 9"}],
        "refused": False,
        "parse_success": True,
    }
    client = _make_client("Yes, confirmed.\nANSWER: 27")
    result = executor.process_channel4_response(
        parsed_response=parsed,
        model="llama-3.1-8b",
        prompt_base="What is 3 × 9?",
        client=client,
        raw_response="[USE_TOOL: calculator | 3*9]",
    )
    assert result["answer_changed"] is False
