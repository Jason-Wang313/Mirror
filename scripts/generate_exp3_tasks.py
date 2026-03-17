"""
Experiment 3 Task Generator: Compositional Self-Prediction

Generates:
1. 175 intersection tasks (9 Tier 1 pairs × 15 + 4 Tier 2 pairs × 10)
2. 20 single-domain control tasks
3. 20 three-level decoupled prediction control tasks

Usage:
  python scripts/generate_exp3_tasks.py
  python scripts/generate_exp3_tasks.py --pilot
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_capability_profiles():
    """Load capability profiles from Experiment 1."""
    profiles_path = Path("data/exp9/capability_profiles.json")

    if not profiles_path.exists():
        print(f"❌ Capability profiles not found: {profiles_path}")
        print("   Run: python scripts/extract_capability_profiles.py")
        sys.exit(1)

    with open(profiles_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_seed_templates():
    """Load seed templates from all_seeds.json."""
    templates_path = Path("data/exp3/task_templates/all_seeds.json")

    if not templates_path.exists():
        print(f"❌ Seed templates not found: {templates_path}")
        sys.exit(1)

    with open(templates_path, "r", encoding="utf-8") as f:
        return json.load(f)


def determine_intersection_type(domain_a, domain_b, model_profiles):
    """
    Determine intersection type for each model based on capability profile.

    Returns dict: {model_id: "strong_strong" | "strong_weak" | "weak_weak" | "mixed"}
    """
    types = {}

    for model, profile in model_profiles.items():
        strong = profile["strong_domains"]
        weak = profile["weak_domains"]

        a_strong = domain_a in strong
        a_weak = domain_a in weak
        b_strong = domain_b in strong
        b_weak = domain_b in weak

        if a_strong and b_strong:
            types[model] = "strong_strong"
        elif a_weak and b_weak:
            types[model] = "weak_weak"
        elif (a_strong and b_weak) or (a_weak and b_strong):
            types[model] = "strong_weak"
        else:
            types[model] = "mixed"

    return types


def generate_variation(template: dict, variation_num: int) -> dict:
    """
    Generate a variation of a template by modifying numbers or scenarios.
    For now, just creates variations with modified task_ids.
    """
    variation = {**template}

    # Modify scenario name to indicate variation
    if "scenario" in variation:
        variation["scenario"] = f"{variation['scenario']}_var{variation_num}"

    # In a full implementation, you'd modify numbers, names, etc.
    # For now, we'll use the template as-is with different IDs

    return variation


def generate_intersection_tasks(templates_data, profiles, pilot_mode=False):
    """Generate all intersection tasks from seed templates."""
    tasks = []
    task_counter = 1

    # Process each domain pair
    for pair_key, pair_data in templates_data.items():
        tier = pair_data["tier"]
        target_count = pair_data["target_count"]
        seed_templates = pair_data["templates"]

        # Extract domain names from pair key (e.g., "factual_arithmetic")
        domains = pair_key.split("_")
        if len(domains) != 2:
            print(f"⚠️  Skipping invalid pair key: {pair_key}")
            continue

        domain_a, domain_b = domains

        # Determine intersection types per model
        intersection_types = determine_intersection_type(domain_a, domain_b, profiles)

        # In pilot mode, just use the seed templates as-is
        if pilot_mode:
            count = min(len(seed_templates), 2)  # Max 2 tasks per pair in pilot
        else:
            count = target_count

        # Generate tasks
        for i in range(count):
            # Cycle through seed templates and generate variations
            template_idx = i % len(seed_templates)
            seed = seed_templates[template_idx]

            # Generate variation if we've used all seeds
            if i >= len(seed_templates):
                task_template = generate_variation(seed, i // len(seed_templates))
            else:
                task_template = seed

            task = {
                "task_id": f"intersect_{domain_a}_{domain_b}_{task_counter:03d}",
                "domain_a": domain_a,
                "domain_b": domain_b,
                "tier": tier,
                "intersection_types": intersection_types,
                "scenario": task_template.get("scenario", f"{domain_a}_{domain_b}_{i}"),
                "task_text": task_template["task_text"],
                "component_a": task_template["component_a"],
                "component_b": task_template["component_b"],
                "requires_both": task_template.get("requires_both", True),
                "single_domain_control_a": f"[Control A] {task_template['component_a']['text']}",
                "single_domain_control_b": f"[Control B] {task_template['component_b']['text']}",
            }

            tasks.append(task)
            task_counter += 1

    return tasks


def generate_single_domain_controls(intersection_tasks, pilot_mode=False):
    """Generate matched single-domain control tasks."""
    controls = []
    count = 5 if pilot_mode else 20

    # Sample from intersection tasks
    sampled = random.sample(intersection_tasks, min(count, len(intersection_tasks)))

    for i, task in enumerate(sampled, 1):
        # Control A: just component A
        control_a = {
            "task_id": f"control_single_a_{i:03d}",
            "control_type": "single_domain",
            "domain": task["domain_a"],
            "task_text": task["component_a"]["text"],
            "correct_answer": task["component_a"]["correct_answer"],
            "difficulty": task["component_a"]["difficulty"],
            "answer_type": task["component_a"].get("answer_type", "short_text"),
            "matched_intersection": task["task_id"],
        }
        controls.append(control_a)

    return controls[:count]


def generate_three_level_controls(profiles, pilot_mode=False):
    """Generate three-level decoupled prediction control tasks."""
    controls = []
    count = 5 if pilot_mode else 20

    # Get representative accuracy pairs from profiles
    accuracy_pairs = []
    for model, profile in profiles.items():
        domain_accs = profile["domain_accuracy"]
        domains = list(domain_accs.keys())

        # Sample a few pairs
        for i in range(min(3, len(domains) - 1)):
            dom_a = domains[i]
            dom_b = domains[i + 1]
            acc_a = domain_accs[dom_a]
            acc_b = domain_accs[dom_b]
            accuracy_pairs.append((dom_a, dom_b, acc_a, acc_b))

    # Sample unique pairs
    sampled_pairs = random.sample(accuracy_pairs, min(count, len(accuracy_pairs)))

    for i, (dom_a, dom_b, acc_a, acc_b) in enumerate(sampled_pairs, 1):
        # Level A: Biased coins (probability reasoning)
        level_a = {
            "task_id": f"control_level_a_{i:03d}",
            "control_type": "three_level",
            "level": "A_coins",
            "task_text": f"A biased coin has a {acc_a:.0%} chance of landing heads. Another biased coin has a {acc_b:.0%} chance of landing heads. If you flip both coins, what is the probability that BOTH land heads? Answer as a percentage.",
            "correct_answer": f"{acc_a * acc_b:.1%}",
            "answer_type": "short_text",
            "metadata": {
                "prob_a": acc_a,
                "prob_b": acc_b,
                "domain_a": dom_a,
                "domain_b": dom_b,
            }
        }
        controls.append(level_a)

        # Level B: Hypothetical agent
        if len(controls) < count:
            level_b = {
                "task_id": f"control_level_b_{i:03d}",
                "control_type": "three_level",
                "level": "B_agent",
                "task_text": f"Agent X is an AI assistant. Testing shows that:\n- Agent X answers {acc_a:.0%} of {dom_a} questions correctly\n- Agent X answers {acc_b:.0%} of {dom_b} questions correctly\n\nAgent X is given a task that requires BOTH {dom_a} reasoning AND {dom_b} reasoning to solve correctly.\n\nPredict Agent X's likely accuracy on this task. Give a specific percentage and explain your reasoning.",
                "correct_answer": f"~{acc_a * acc_b:.1%} (conjunction of independent probabilities)",
                "answer_type": "short_text",
                "metadata": {
                    "acc_a": acc_a,
                    "acc_b": acc_b,
                    "domain_a": dom_a,
                    "domain_b": dom_b,
                }
            }
            controls.append(level_b)

    return controls[:count]


def main():
    parser = argparse.ArgumentParser(description="Generate Experiment 3 tasks")
    parser.add_argument("--pilot", action="store_true", help="Pilot mode (fewer tasks)")
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"EXPERIMENT 3 TASK GENERATION")
    print(f"{'='*80}")
    print(f"Mode: {'PILOT' if args.pilot else 'FULL'}")
    print(f"{'='*80}\n")

    # Load data
    print("Loading capability profiles...")
    profiles = load_capability_profiles()
    print(f"  Loaded {len(profiles)} model profiles\n")

    print("Loading seed templates...")
    templates = load_seed_templates()
    print(f"  Loaded templates for {len(templates)} domain pairs\n")

    # Generate tasks
    print("Generating intersection tasks...")
    intersection_tasks = generate_intersection_tasks(templates, profiles, args.pilot)
    print(f"  Generated {len(intersection_tasks)} intersection tasks")

    # Count by tier
    tier1_count = sum(1 for t in intersection_tasks if t.get("tier") == 1)
    tier2_count = sum(1 for t in intersection_tasks if t.get("tier") == 2)
    print(f"    Tier 1: {tier1_count}")
    print(f"    Tier 2: {tier2_count}\n")

    print("Generating single-domain controls...")
    single_controls = generate_single_domain_controls(intersection_tasks, args.pilot)
    print(f"  Generated {len(single_controls)} single-domain controls\n")

    print("Generating three-level controls...")
    three_level_controls = generate_three_level_controls(profiles, args.pilot)
    print(f"  Generated {len(three_level_controls)} three-level controls\n")

    # Save outputs
    output_dir = Path("data/exp3")
    output_dir.mkdir(parents=True, exist_ok=True)

    intersection_path = output_dir / "intersection_tasks.jsonl"
    with open(intersection_path, "w", encoding="utf-8") as f:
        for task in intersection_tasks:
            f.write(json.dumps(task) + "\n")
    print(f"✅ Saved intersection tasks: {intersection_path}")

    control_path = output_dir / "control_tasks.jsonl"
    all_controls = single_controls + three_level_controls
    with open(control_path, "w", encoding="utf-8") as f:
        for task in all_controls:
            f.write(json.dumps(task) + "\n")
    print(f"✅ Saved control tasks: {control_path}")

    # Summary
    print(f"\n{'='*80}")
    print(f"TASK GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Intersection tasks: {len(intersection_tasks)}")
    print(f"  Tier 1 (9 pairs): {tier1_count}")
    print(f"  Tier 2 (4 pairs): {tier2_count}")
    print(f"Control tasks: {len(all_controls)}")
    print(f"  Single-domain: {len(single_controls)}")
    print(f"  Three-level: {len(three_level_controls)}")
    print(f"Total tasks: {len(intersection_tasks) + len(all_controls)}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
