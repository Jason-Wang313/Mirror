"""
Reparse Layer 2 responses from Experiment 3 results with improved parser.
"""
import json
import re
from pathlib import Path


def parse_layer2_flexible(response_text: str) -> dict:
    """Parse layer2 with flexible patterns to handle format variations."""
    parsed = {
        "confidence": None,
        "comparison": None,
        "weak_link": None,
        "prediction": None,
    }

    if not response_text:
        return parsed

    # Extract CONFIDENCE - flexible patterns
    conf_patterns = [
        r'CONFIDENCE[:\s=-]+(\d+)',  # CONFIDENCE: 80, CONFIDENCE = 80
        r'confidence\s+(?:is|at|level)[:\s]+(\d+)',  # confidence is 80
        r'(\d+)%?\s*confidence',  # 80% confidence
        r'rate.*?confidence.*?(\d+)',  # rate my confidence at 80
        r'(\d+)\s*/\s*100',  # 80/100
    ]
    for pattern in conf_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            parsed["confidence"] = int(match.group(1))
            break

    # Extract COMPARISON - flexible patterns
    comp_patterns = [
        r'COMPARISON[:\s=-]+(easier|same|harder)',  # COMPARISON: harder
        r'(?:will be|would be|is|seems)\s+(easier|same|harder)',  # will be harder
        r'(easier|harder|same)\s+than',  # harder than
        r'more\s+(difficult|challenging)',  # more difficult -> harder
        r'less\s+(difficult|challenging)',  # less difficult -> easier
    ]
    for pattern in comp_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            val = match.group(1).lower()
            if val in ['difficult', 'challenging']:
                parsed["comparison"] = 'harder'
            elif 'less' in match.group(0).lower():
                parsed["comparison"] = 'easier'
            else:
                parsed["comparison"] = val
            break

    # Extract WEAK_LINK - flexible patterns
    wl_patterns = [
        r'WEAK[_\s-]?LINK[:\s=-]+(\w+)',  # WEAK_LINK: arithmetic
        r'(?:struggle|difficulty|weaker|weakness)\s+(?:with|at|in)\s+(\w+)',  # struggle with arithmetic
        r'(\w+)\s+(?:is|will be)\s+(?:weaker|harder|more difficult)',  # arithmetic is harder
        r'more\s+likely\s+to\s+struggle\s+with\s+(\w+)',  # struggle with spatial
    ]
    for pattern in wl_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            parsed["weak_link"] = match.group(1).lower()
            break

    # Extract PREDICTION - flexible patterns
    pred_patterns = [
        r'PREDICTION[:\s=-]+(\d+)%?',  # PREDICTION: 85
        r'predict.*?(\d+)%',  # predict 85% accuracy
        r'estimate.*?(\d+)%',  # estimate 85%
        r'(\d+)%\s+(?:accuracy|correct)',  # 85% accuracy
        r'accuracy.*?(\d+)%',  # accuracy of 85%
    ]
    for pattern in pred_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            parsed["prediction"] = int(match.group(1))
            break

    return parsed


def reparse_results(run_id: str):
    """Reparse all layer2 responses in a results file."""
    results_file = Path(f"data/results/{run_id}_results.jsonl")

    if not results_file.exists():
        print(f"Error: {results_file} not found")
        return

    print(f"Reparsing: {results_file}")

    # Load all results
    results = []
    with open(results_file) as f:
        for line in f:
            results.append(json.loads(line))

    print(f"Loaded {len(results)} results")

    # Reparse layer2 for each result
    reparse_count = 0
    for result in results:
        layer2 = result.get("layer2", {})
        answer_text = layer2.get("answer")

        if answer_text:
            # Reparse with flexible patterns
            reparsed = parse_layer2_flexible(answer_text)

            # Update fields if we extracted new values
            if reparsed["confidence"] is not None and layer2.get("confidence") is None:
                layer2["confidence"] = reparsed["confidence"]
                reparse_count += 1
            if reparsed["comparison"] is not None and layer2.get("comparison") is None:
                layer2["comparison"] = reparsed["comparison"]
            if reparsed["weak_link"] is not None and layer2.get("weak_link") is None:
                layer2["weak_link"] = reparsed["weak_link"]
            if reparsed["prediction"] is not None and layer2.get("prediction") is None:
                layer2["prediction"] = reparsed["prediction"]

    print(f"Reparsed {reparse_count} layer2 responses with new values")

    # Write back to file
    with open(results_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"✅ Updated: {results_file}")

    # Print statistics
    print("\nLayer2 parsing statistics by model:")
    from collections import defaultdict
    stats = defaultdict(lambda: {"total": 0, "conf": 0, "comp": 0, "wl": 0, "pred": 0})

    for result in results:
        model = result["model"]
        layer2 = result.get("layer2", {})
        stats[model]["total"] += 1
        if layer2.get("confidence") is not None:
            stats[model]["conf"] += 1
        if layer2.get("comparison") is not None:
            stats[model]["comp"] += 1
        if layer2.get("weak_link") is not None:
            stats[model]["wl"] += 1
        if layer2.get("prediction") is not None:
            stats[model]["pred"] += 1

    for model in sorted(stats.keys()):
        s = stats[model]
        print(f"\n{model}:")
        print(f"  Confidence: {s['conf']}/{s['total']} ({100*s['conf']/s['total']:.1f}%)")
        print(f"  Comparison: {s['comp']}/{s['total']} ({100*s['comp']/s['total']:.1f}%)")
        print(f"  Weak-link:  {s['wl']}/{s['total']} ({100*s['wl']/s['total']:.1f}%)")
        print(f"  Prediction: {s['pred']}/{s['total']} ({100*s['pred']/s['total']:.1f}%)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python scripts/reparse_layer2.py <run_id>")
        sys.exit(1)

    run_id = sys.argv[1]
    reparse_results(run_id)
