"""Tool execution for Channel 4 (Tool Use) responses.

Executes the calculator tool safely using AST validation.
All other tools (web_search, ask_expert, flag_review) are record-only.
"""

import ast
import math
import re
from typing import Optional


class ToolExecutor:
    """Executes tool requests from Channel 4 responses."""

    SUPPORTED_TOOLS = {"calculator"}

    # Names allowed in expressions
    _ALLOWED_NAMES = frozenset({"sqrt", "abs", "round", "int", "float", "pi", "e"})

    # AST node types allowed in expressions
    _ALLOWED_NODES = frozenset({
        ast.Expression,
        ast.Constant,
        ast.Num,        # Python < 3.8 numeric literals
        ast.BinOp,
        ast.UnaryOp,
        ast.Add, ast.Sub, ast.Mult, ast.Div,
        ast.Pow, ast.Mod, ast.FloorDiv,
        ast.USub, ast.UAdd,
        ast.Call,
        ast.Name,
        ast.Load,
    })

    def execute(self, tool_name: str, tool_input: str) -> dict:
        """Execute a single tool request.

        Returns:
            {
                "tool_name": str,
                "tool_input": str,
                "executed": bool,
                "result": str | None,
                "error": str | None,
            }
        """
        result = {
            "tool_name": tool_name,
            "tool_input": tool_input,
            "executed": False,
            "result": None,
            "error": None,
        }
        if tool_name == "calculator":
            result["executed"] = True
            calc_result = self.execute_calculator(tool_input)
            if calc_result.startswith("Error:"):
                result["error"] = calc_result
                result["result"] = None
            else:
                result["result"] = calc_result
        else:
            result["executed"] = False
            result["error"] = f"Tool '{tool_name}' is record-only and not executed."
        return result

    def _validate_ast(self, tree: ast.AST) -> bool:
        """Verify the AST contains only allowed node types and names."""
        for node in ast.walk(tree):
            node_type = type(node)
            if node_type not in self._ALLOWED_NODES:
                return False
            if isinstance(node, ast.Name):
                if node.id not in self._ALLOWED_NAMES:
                    return False
            if isinstance(node, ast.Call):
                # Only allow direct function calls by name, not attribute calls
                if not isinstance(node.func, ast.Name):
                    return False
                if node.func.id not in self._ALLOWED_NAMES:
                    return False
        return True

    def execute_calculator(self, expression: str) -> str:
        """Safely evaluate a math expression.

        Supported: +, -, *, /, **, %, //, sqrt, abs, round, int, float
        Unsupported: imports, variable assignment, arbitrary function calls

        Returns string result (rounded to 6 decimal places for floats)
        or an error message starting with "Error:".
        """
        expression = expression.strip()
        if not expression:
            return "Error: empty expression"

        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError:
            return f"Error: invalid expression '{expression}'"

        if not self._validate_ast(tree):
            return "Error: expression contains disallowed operations"

        safe_ns = {
            "sqrt": math.sqrt,
            "abs": abs,
            "round": round,
            "int": int,
            "float": float,
            "pi": math.pi,
            "e": math.e,
        }

        try:
            result = eval(  # noqa: S307  (safe: AST-validated, no builtins)
                compile(tree, "<calculator>", "eval"),
                {"__builtins__": {}},
                safe_ns,
            )
        except ZeroDivisionError:
            return "Error: division by zero"
        except OverflowError:
            return "Error: result too large"
        except Exception as exc:
            return f"Error: {exc}"

        if isinstance(result, float):
            rounded = round(result, 6)
            if rounded == int(rounded) and abs(rounded) < 1e15:
                return str(int(rounded))
            return str(rounded)
        if isinstance(result, int):
            return str(result)
        return str(result)

    def process_channel4_response(
        self,
        parsed_response: dict,
        model: str,
        prompt_base: str,
        client,
        raw_response: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> dict:
        """Process a Channel 4 parsed response, executing calculator tools.

        If calculator tools were requested:
          1. Execute each calculator call safely.
          2. Format results as a second-turn message.
          3. Send second turn to the model via client.complete_sync().
          4. Parse the model's updated answer.
          5. Return augmented result with both initial and final answers.

        If no executable tools were requested, return the original response
        unchanged with tool_executed=False.

        Args:
            parsed_response: Output of parse_response(4, ...) — must contain
                             "tools_used" key.
            model: Model name for the API call.
            prompt_base: The original Channel 4 prompt text (first user turn).
            client: UnifiedClient instance (uses complete_sync).
            raw_response: The model's raw first-turn response text.
            temperature: Temperature for second-turn API call.
            max_tokens: Max tokens for second-turn API call.

        Returns:
            Augmented dict with keys:
                tool_executed (bool), tool_results (list[dict]),
                initial_answer (str|None), final_answer (str|None),
                answer_changed (bool), second_turn_response (str|None).
        """
        result = dict(parsed_response)
        result["tool_executed"] = False
        result["tool_results"] = []
        result["initial_answer"] = parsed_response.get("answer")
        result["final_answer"] = parsed_response.get("answer")
        result["answer_changed"] = False
        result["second_turn_response"] = None

        tools_used = parsed_response.get("tools_used", []) or []
        if not tools_used:
            return result

        # Execute all tools
        tool_results = []
        any_calculator = False
        for tool_req in tools_used:
            tool_name = tool_req.get("tool_name", "")
            tool_input = tool_req.get("tool_input", "")
            exec_result = self.execute(tool_name, tool_input)
            tool_results.append(exec_result)
            if tool_name == "calculator":
                any_calculator = True

        result["tool_results"] = tool_results

        if not any_calculator:
            # All tools are record-only — no second turn needed
            return result

        result["tool_executed"] = True

        # Build second-turn content
        result_lines = []
        for tr in tool_results:
            if tr["executed"]:
                expr = tr["tool_input"]
                val = tr["result"] if tr["result"] is not None else tr["error"]
                result_lines.append(
                    f"[TOOL_RESULT: {tr['tool_name']} | {expr} = {val}]"
                )
            else:
                result_lines.append(
                    f"[TOOL_RESULT: {tr['tool_name']} | not executed (record only)]"
                )

        second_turn_content = (
            "Tool results:\n"
            + "\n".join(result_lines)
            + "\n\nNow provide your final answer.\nANSWER: [your answer]"
        )

        messages = [
            {"role": "user", "content": prompt_base},
            {"role": "assistant", "content": raw_response or ""},
            {"role": "user", "content": second_turn_content},
        ]

        try:
            api_result = client.complete_sync(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            result["second_turn_error"] = str(exc)
            return result

        if "error" in api_result:
            result["second_turn_error"] = api_result["error"]
            return result

        second_response = api_result.get("content", "")
        result["second_turn_response"] = second_response

        final_answer = self._extract_answer_from_second_turn(second_response)
        result["final_answer"] = final_answer
        result["answer"] = final_answer  # Override with final
        result["answer_changed"] = final_answer != result["initial_answer"]

        return result

    def _extract_answer_from_second_turn(self, response: str) -> Optional[str]:
        """Extract answer from the model's second-turn response.

        Looks for ANSWER: label first, then falls back to the last
        non-empty line.
        """
        # ANSWER: [value] or ANSWER: value
        match = re.search(r"ANSWER:\s*\[?([^\n\]]+)\]?", response, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        lines = [ln.strip() for ln in response.strip().split("\n") if ln.strip()]
        if lines:
            return lines[-1]
        return None
