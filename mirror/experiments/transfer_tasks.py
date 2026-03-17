"""
Generate cross-domain transfer tasks for Experiment 2.

For each domain pair (source → surface), generates tasks framed as surface domain
but requiring hidden dependency on source domain skills.
"""

import asyncio
import json
import re
from pathlib import Path

from ..api import UnifiedClient


def _strip_thinking(text: str) -> str:
    """
    Strip <think>...</think> blocks from DeepSeek R1 responses.

    Also handles nested tags and returns clean output.
    """
    if not text:
        return text

    # Remove all <think>...</think> blocks (including nested)
    while "<think>" in text.lower():
        # Find matching pairs
        start = text.lower().find("<think>")
        if start == -1:
            break

        # Find the corresponding closing tag
        depth = 1
        pos = start + 7  # len("<think>")
        while pos < len(text) and depth > 0:
            if text[pos:pos+7].lower() == "<think>":
                depth += 1
                pos += 7
            elif text[pos:pos+8].lower() == "</think>":
                depth -= 1
                pos += 8
            else:
                pos += 1

        # Remove the block
        if depth == 0:
            text = text[:start] + text[pos:]
        else:
            # Unclosed tag, remove from start to end
            text = text[:start]
            break

    return text.strip()


def _extract_json_from_text(text: str) -> str:
    """
    Extract JSON object or array from text.

    Finds the first '{' or '[' and extracts from there.
    """
    text = text.strip()

    # Find first '{' or '['
    json_start = -1
    for i, char in enumerate(text):
        if char in ['{', '[']:
            json_start = i
            break

    if json_start == -1:
        return text  # No JSON found, return as-is

    # Extract from json_start to end
    json_text = text[json_start:]

    # Try to find matching closing bracket
    bracket_type = json_text[0]
    close_bracket = '}' if bracket_type == '{' else ']'

    depth = 0
    for i, char in enumerate(json_text):
        if char == bracket_type:
            depth += 1
        elif char == close_bracket:
            depth -= 1
            if depth == 0:
                return json_text[:i+1]

    # If no matching close found, return from start to end
    return json_text


# Domain pair definitions with hand-crafted seed examples
DOMAIN_PAIRS = [
    {
        "source_domain": "arithmetic",
        "surface_domain": "business_evaluation",
        "hidden_dependency": "financial_calculations",
        "description": "Business decisions requiring hidden numerical reasoning",
        "examples": [
            {
                "task_text": "A supplier offers widgets at $47 per unit with a minimum order of 23 units. Your budget is $1000. Can you afford this order?",
                "correct_answer": "No. 47 × 23 = $1,081, which exceeds the $1000 budget.",
                "requires_from_source": "Multi-digit multiplication",
            },
            {
                "task_text": "Your quarterly revenue is $145,000. Operating costs are $89,000. Marketing is 15% of net profit. What's the marketing budget?",
                "correct_answer": "$8,400. Net profit = $145,000 - $89,000 = $56,000. Marketing = 15% of $56,000 = $8,400.",
                "requires_from_source": "Subtraction and percentage calculation",
            },
            {
                "task_text": "A client pays $3,500 upfront for a 6-month contract. What's the monthly revenue?",
                "correct_answer": "$583.33 per month ($3,500 ÷ 6).",
                "requires_from_source": "Division with decimal result",
            },
        ]
    },
    {
        "source_domain": "spatial",
        "surface_domain": "trip_planning",
        "hidden_dependency": "distance_estimation",
        "description": "Travel itinerary tasks requiring spatial reasoning",
        "examples": [
            {
                "task_text": "You're driving from Denver to Salt Lake City (~530 miles) with a stop in Grand Junction (halfway). You want each segment under 4 hours at 70 mph. Is this feasible?",
                "correct_answer": "Yes. Each segment is ~265 miles. At 70 mph, each takes ~3.8 hours, under the 4-hour limit.",
                "requires_from_source": "Distance estimation and division",
            },
            {
                "task_text": "A rectangular warehouse is 80 feet wide and 120 feet long. You need to place shelving units (each 4×6 feet) with 2-foot aisles. How many fit along one wall?",
                "correct_answer": "10 units along the 120-foot wall (each unit + aisle = 6+2 = 8 feet, 120÷8 = 15, but need aisle space so ~10).",
                "requires_from_source": "2D spatial layout and area calculation",
            },
            {
                "task_text": "A delivery route has 5 stops in a grid: (0,0), (3,4), (7,1), (5,8), (2,6). Which order minimizes backtracking?",
                "correct_answer": "Start (0,0) → (2,6) → (3,4) → (5,8) → (7,1) minimizes total distance using nearest-neighbor heuristic.",
                "requires_from_source": "Spatial positioning and distance comparison",
            },
        ]
    },
    {
        "source_domain": "temporal",
        "surface_domain": "meeting_scheduling",
        "hidden_dependency": "timezone_arithmetic",
        "description": "Multi-timezone coordination requiring time calculations",
        "examples": [
            {
                "task_text": "A 2-hour meeting starts at 9 AM PST. What time does it end in EST?",
                "correct_answer": "2 PM EST (9 AM PST = 12 PM EST, +2 hours = 2 PM EST).",
                "requires_from_source": "Timezone conversion and time addition",
            },
            {
                "task_text": "You need to schedule calls with London (GMT), Tokyo (GMT+9), and San Francisco (GMT-8). Find a 1-hour window where it's 9 AM-5 PM for all.",
                "correct_answer": "No solution. When it's 9 AM in Tokyo, it's midnight in London and 4 PM (previous day) in SF — no overlap.",
                "requires_from_source": "Multi-timezone arithmetic and constraint satisfaction",
            },
            {
                "task_text": "A flight departs at 11:30 PM on Monday and takes 14 hours. Arrival timezone is +7 hours ahead. What day/time local arrival?",
                "correct_answer": "Wednesday 6:30 AM local (11:30 PM + 14h = 1:30 PM Tuesday origin time, +7h = 8:30 PM Tuesday local, wait that's wrong... 11:30 PM Mon + 14h = 1:30 PM Tue, + 7h offset = 8:30 PM Tue local).",
                "requires_from_source": "Multi-day time arithmetic with timezone offset",
            },
        ]
    },
    {
        "source_domain": "linguistic",
        "surface_domain": "document_proofreading",
        "hidden_dependency": "grammar_parsing",
        "description": "Text editing requiring grammatical structure analysis",
        "examples": [
            {
                "task_text": "Does this sentence have a grammatical error? 'Each of the team members have submitted their reports.'",
                "correct_answer": "Yes. 'Each' is singular, so 'have' should be 'has': 'Each...has submitted.'",
                "requires_from_source": "Subject-verb agreement parsing",
            },
            {
                "task_text": "Rewrite this for formal tone: 'We're gonna need you to shoot over those docs ASAP.'",
                "correct_answer": "'We will need you to send those documents at your earliest convenience.'",
                "requires_from_source": "Register/formality transformation",
            },
            {
                "task_text": "Clarify this ambiguous sentence: 'I saw the man with the telescope.' Which interpretation is intended?",
                "correct_answer": "Two interpretations: (1) I used a telescope to see the man, or (2) I saw a man who had a telescope. Need context to disambiguate.",
                "requires_from_source": "Syntactic ambiguity resolution",
            },
        ]
    },
    {
        "source_domain": "logical",
        "surface_domain": "argument_analysis",
        "hidden_dependency": "inference_validity",
        "description": "Evaluating claims requiring logical deduction",
        "examples": [
            {
                "task_text": "Is this argument valid? 'All engineers are analytical. Sam is analytical. Therefore, Sam is an engineer.'",
                "correct_answer": "Invalid. Affirming the consequent fallacy. Being analytical doesn't imply being an engineer.",
                "requires_from_source": "Detecting logical fallacies in syllogisms",
            },
            {
                "task_text": "Given: If sales increase, we hire staff. We didn't hire staff. What can we conclude?",
                "correct_answer": "Sales did not increase (modus tollens: ¬Q → ¬P).",
                "requires_from_source": "Modus tollens inference",
            },
            {
                "task_text": "Premises: All mammals are warm-blooded. No reptiles are warm-blooded. Is this sound: 'Therefore, no mammals are reptiles'?",
                "correct_answer": "Valid and sound. The conclusion follows from the premises, and the premises are true.",
                "requires_from_source": "Categorical logic and set relations",
            },
        ]
    },
    {
        "source_domain": "social",
        "surface_domain": "email_composition",
        "hidden_dependency": "pragmatic_implicature",
        "description": "Diplomatic communication requiring social reasoning",
        "examples": [
            {
                "task_text": "You need to decline a meeting invite without offending. Which is better: (A) 'I can't make it.' (B) 'I'd love to, but I have a conflict.'",
                "correct_answer": "(B). It shows interest while declining, softening the refusal with positive framing.",
                "requires_from_source": "Politeness strategies and face-saving",
            },
            {
                "task_text": "A colleague says 'Interesting approach' about your proposal. Is this positive or negative?",
                "correct_answer": "Ambiguous/likely negative. 'Interesting' often implies skepticism without direct criticism.",
                "requires_from_source": "Conversational implicature and indirect speech acts",
            },
            {
                "task_text": "How do you politely tell someone their presentation had errors? Draft one sentence.",
                "correct_answer": "'I noticed a few data points that might need verification—happy to help review if useful.'",
                "requires_from_source": "Indirect criticism and cooperative tone",
            },
        ]
    },
    {
        "source_domain": "factual",
        "surface_domain": "fact_checking",
        "hidden_dependency": "cross_referencing_claims",
        "description": "Verifying statements requiring factual knowledge",
        "examples": [
            {
                "task_text": "Article claims: 'World War II ended in 1945.' Is this accurate?",
                "correct_answer": "Yes. WWII ended with Japan's surrender on September 2, 1945.",
                "requires_from_source": "Historical fact recall",
            },
            {
                "task_text": "Claim: 'Paris is the capital of France and the largest city in Europe.' Assess accuracy.",
                "correct_answer": "Partially true. Paris is the capital of France, but not the largest city in Europe (Istanbul or Moscow are larger).",
                "requires_from_source": "Geographic fact verification",
            },
            {
                "task_text": "Statement: 'Water boils at 100°C at sea level.' Verify.",
                "correct_answer": "True. Water's boiling point is 100°C (212°F) at standard atmospheric pressure.",
                "requires_from_source": "Scientific fact recall",
            },
        ]
    },
    {
        "source_domain": "procedural",
        "surface_domain": "troubleshooting",
        "hidden_dependency": "multi_step_diagnosis",
        "description": "System debugging requiring procedural reasoning",
        "examples": [
            {
                "task_text": "A web form won't submit. Error: 'Email required.' The email field has text. What's the likely issue?",
                "correct_answer": "Client-side validation likely checking for '@' symbol or valid email format, not just non-empty text.",
                "requires_from_source": "Debugging procedural logic and validation rules",
            },
            {
                "task_text": "Steps: (1) Check power, (2) Check connection, (3) Restart device. Device still fails. What next?",
                "correct_answer": "Check error logs/diagnostic codes, or escalate to hardware inspection after standard protocol exhausted.",
                "requires_from_source": "Following troubleshooting protocol sequentially",
            },
            {
                "task_text": "A backup failed at step 3 of 5. Should you: (A) Retry from step 3, (B) Restart from step 1?",
                "correct_answer": "(B) Restart from step 1. Backup integrity requires complete sequential execution; partial retry risks corruption.",
                "requires_from_source": "Understanding procedural atomicity and error recovery",
            },
        ]
    },
]


async def generate_transfer_task(
    client: UnifiedClient,
    pair: dict,
    task_index: int,
) -> dict:
    """
    Generate one transfer task for a domain pair using few-shot prompting.

    Args:
        client: UnifiedClient instance
        pair: Domain pair specification with examples
        task_index: Index of task (for ID generation)

    Returns:
        Generated task dict
    """
    examples_text = "\n\n".join([
        f"Example {i+1}:\n"
        f"Task: {ex['task_text']}\n"
        f"Correct Answer: {ex['correct_answer']}\n"
        f"Requires: {ex['requires_from_source']}"
        for i, ex in enumerate(pair["examples"])
    ])

    prompt = f"""Generate a task for testing cross-domain skill transfer.

SOURCE DOMAIN: {pair['source_domain']}
SURFACE DOMAIN: {pair['surface_domain']}
HIDDEN DEPENDENCY: {pair['hidden_dependency']}
DESCRIPTION: {pair['description']}

The task should:
1. Be framed as a {pair['surface_domain']} problem
2. Require {pair['source_domain']} skills to solve correctly
3. NOT obviously signal that it's testing {pair['source_domain']}
4. Have ONE definitively correct answer
5. Be solvable in 2-3 sentences

Here are examples of good tasks:

{examples_text}

Generate ONE new task following this pattern. Use a different scenario than the examples.

Respond in this exact JSON format:
{{
  "task_text": "...",
  "correct_answer": "...",
  "requires_from_source": "..."
}}"""

    # Use Llama 70B for task generation
    # Note: DeepSeek R1 was tested but returns reasoning-only responses even when
    # asked for JSON, so Llama 70B is used as it reliably produces structured output
    model = "llama-3.1-70b"

    response = await client.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=800,
        metadata={"task": "transfer_task_generation"},
    )

    if "error" in response:
        print(f"  ⚠️  API error: {response['error']}")
        return None

    # Extract content
    content = response.get("content") or ""

    if not content:
        print(f"  ⚠️  Empty response from {model}")
        return None

    # Parse JSON
    try:
        # Remove markdown fences if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        # Extract JSON from remaining text
        content = _extract_json_from_text(content)

        task_data = json.loads(content.strip())

        # Validate required fields
        if "task_text" not in task_data or "correct_answer" not in task_data:
            raise KeyError("Missing required fields: task_text or correct_answer")

        # Build complete task object
        task_id = f"transfer_{pair['source_domain']}_{pair['surface_domain']}_{task_index:03d}"
        return {
            "task_id": task_id,
            "source_domain": pair["source_domain"],
            "surface_domain": pair["surface_domain"],
            "hidden_dependency": pair["hidden_dependency"],
            "task_text": task_data["task_text"],
            "correct_answer": task_data["correct_answer"],
            "requires_from_source": task_data.get("requires_from_source", ""),
            "difficulty": "medium",
            "is_adversarial_disguise": False,
            "generated_by": model,
        }

    except (json.JSONDecodeError, KeyError) as e:
        print(f"  ⚠️  Parse error with {model}: {e}")
        print(f"  Content (first 200 chars): {content[:200]}")
        return None


async def verify_transfer_task(
    client: UnifiedClient,
    task: dict,
) -> dict:
    """
    Cross-verify task with second model (Llama 70B).

    Checks:
    1. Task genuinely requires source domain skills
    2. Surface framing doesn't obviously signal source domain
    3. Correct answer is unambiguous

    Returns:
        Verification result dict with pass/fail and comments
    """
    prompt = f"""Verify this cross-domain transfer task:

SOURCE DOMAIN: {task['source_domain']}
SURFACE DOMAIN: {task['surface_domain']}

TASK: {task['task_text']}
STATED ANSWER: {task['correct_answer']}

Verify:
1. Does solving this task genuinely require {task['source_domain']} skills?
2. Is the task framed as {task['surface_domain']} without obviously signaling {task['source_domain']}?
3. Is the stated answer correct and unambiguous?

Respond ONLY with:
VALID: YES or NO
REASONING: [brief explanation]
SUGGESTED_FIX: [if NO, suggest how to fix]"""

    response = await client.complete(
        model="llama-3.1-70b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=400,
        metadata={"task": "transfer_task_verification"},
    )

    content = response.get("content") or ""

    # Parse response
    valid = "YES" in content.upper().split("VALID:")[1].split("\n")[0] if "VALID:" in content.upper() else False

    return {
        "task_id": task["task_id"],
        "valid": valid,
        "verification_response": content,
    }


async def generate_transfer_tasks(
    pairs: list[dict] = None,
    n_per_pair: int = 25,
    output_path: str = "data/transfer_tasks.jsonl",
    pilot_mode: bool = False,
) -> dict:
    """
    Generate all transfer tasks.

    Args:
        pairs: Domain pair specifications (defaults to DOMAIN_PAIRS)
        n_per_pair: Number of tasks per domain pair
        output_path: Output file path
        pilot_mode: If True, generate only 5 per pair

    Returns:
        Stats dict
    """
    if pairs is None:
        pairs = DOMAIN_PAIRS

    if pilot_mode:
        n_per_pair = 5

    client = UnifiedClient(experiment="transfer_task_generation")
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    all_tasks = []
    stats = {"total_generated": 0, "total_valid": 0, "by_pair": {}}

    for pair in pairs:
        pair_key = f"{pair['source_domain']}→{pair['surface_domain']}"
        print(f"\nGenerating tasks for {pair_key} ({n_per_pair} tasks)...")

        pair_tasks = []
        attempts = 0
        max_attempts = n_per_pair * 2  # Allow retries

        while len(pair_tasks) < n_per_pair and attempts < max_attempts:
            attempts += 1
            task_index = len(pair_tasks) + 1

            print(f"  Generating task {task_index}/{n_per_pair} (attempt {attempts})...")
            task = await generate_transfer_task(client, pair, task_index)

            if task:
                pair_tasks.append(task)
                stats["total_generated"] += 1

            await asyncio.sleep(0.5)

        print(f"  Generated {len(pair_tasks)} tasks for {pair_key}")
        stats["by_pair"][pair_key] = len(pair_tasks)
        all_tasks.extend(pair_tasks)

    # Write all tasks
    with open(output_file, "w", encoding="utf-8") as f:
        for task in all_tasks:
            f.write(json.dumps(task) + "\n")

    stats["total_valid"] = len(all_tasks)
    print(f"\n✅ Generated {len(all_tasks)} tasks → {output_file}")

    return stats


async def generate_adversarial_disguise_tasks(
    n: int = 30,
    output_path: str = "data/transfer_adversarial.jsonl",
) -> dict:
    """
    Generate adversarial disguise tasks where surface framing misleads.

    Example: Task appears to be "business evaluation" (suggesting arithmetic)
    but actually requires "logical reasoning" to detect flawed arguments.
    """
    client = UnifiedClient(experiment="adversarial_transfer_generation")
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Define adversarial pairs (surface domain suggests X, but actually requires Y)
    adversarial_pairs = [
        {
            "surface_domain": "business_evaluation",
            "actual_dependency": "logical",
            "description": "Business claim with logical fallacy disguised as financial decision",
        },
        {
            "surface_domain": "trip_planning",
            "actual_dependency": "temporal",
            "description": "Itinerary requiring timezone logic disguised as distance calculation",
        },
        # Add more adversarial pairs...
    ]

    # For now, return empty (implementation similar to generate_transfer_tasks)
    print(f"⚠️  Adversarial task generation not yet implemented")
    return {"total_generated": 0}


def load_transfer_tasks(tasks_path: str = "data/transfer_tasks.jsonl") -> list[dict]:
    """Load transfer tasks from JSONL file."""
    tasks = []
    tasks_file = Path(tasks_path)
    if not tasks_file.exists():
        return tasks

    with open(tasks_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))

    return tasks


def run_generate_transfer_tasks(n_per_pair: int = 25, pilot_mode: bool = False) -> dict:
    """Sync wrapper for generate_transfer_tasks."""
    return asyncio.run(generate_transfer_tasks(
        n_per_pair=n_per_pair,
        pilot_mode=pilot_mode,
    ))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate transfer tasks for Experiment 2")
    parser.add_argument("--generate", action="store_true", help="Generate tasks")
    parser.add_argument("--n-per-pair", type=int, default=25, help="Tasks per domain pair")
    parser.add_argument("--pilot", action="store_true", help="Pilot mode (5 per pair)")
    parser.add_argument("--output", default="data/transfer_tasks.jsonl", help="Output path")
    args = parser.parse_args()

    if args.generate:
        stats = run_generate_transfer_tasks(
            n_per_pair=args.n_per_pair,
            pilot_mode=args.pilot,
        )
        print(f"\nStats: {json.dumps(stats, indent=2)}")
    else:
        print("Use --generate to create tasks")
