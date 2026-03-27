"""
E-Prime Compliance Checker
Detects 'to be' verb violations in E-Prime condition responses.
Also detects 'to have' violations for the no-have condition.

Compliance rate itself is a dependent variable worth reporting.

Usage:
    python scripts/check_compliance.py results/run-pilot-*.jsonl
    python scripts/check_compliance.py results/run-pilot-*.jsonl --verbose
"""

import argparse
import json
import re
import sys
from pathlib import Path

# "To be" forms — the E-Prime banned list
TO_BE_FORMS = {
    "is", "isn't", "isnt",
    "am",
    "are", "aren't", "arent",
    "was", "wasn't", "wasnt",
    "were", "weren't", "werent",
    "be", "been", "being",
}

# Contractions that hide "to be"
TO_BE_CONTRACTIONS = {
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "here's": "here is",
    "what's": "what is",
    "who's": "who is",
    "he's": "he is",
    "she's": "she is",
    "i'm": "i am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
}

# "To have" forms — banned in no-have condition
TO_HAVE_FORMS = {
    "have", "haven't", "havent",
    "has", "hasn't", "hasnt",
    "had", "hadn't", "hadnt",
    "having",
}

TO_HAVE_CONTRACTIONS = {
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "would've": "would have",
    "could've": "could have",
    "should've": "should have",
    "might've": "might have",
    "must've": "must have",
}


def tokenize(text: str) -> list[str]:
    """Split text into word tokens, preserving contractions."""
    return re.findall(r"[a-zA-Z']+", text.lower())


def find_violations(text: str, condition: str) -> list[dict]:
    """Find all constraint violations in a response text.

    Returns list of {word, position, line, context} dicts.
    """
    if condition == "e_prime":
        banned_words = TO_BE_FORMS
        banned_contractions = TO_BE_CONTRACTIONS
    elif condition == "no_have":
        banned_words = TO_HAVE_FORMS
        banned_contractions = TO_HAVE_CONTRACTIONS
    else:
        return []  # No constraints for control condition

    violations = []
    lines = text.split("\n")

    for line_num, line in enumerate(lines, 1):
        tokens = tokenize(line)
        for i, token in enumerate(tokens):
            violated = False
            violation_type = None

            if token in banned_words:
                violated = True
                violation_type = "direct"
            elif token in banned_contractions:
                violated = True
                violation_type = "contraction"

            if violated:
                # Extract surrounding context (5 words each side)
                start = max(0, i - 5)
                end = min(len(tokens), i + 6)
                context_tokens = tokens[start:end]
                context = " ".join(context_tokens)

                violations.append({
                    "word": token,
                    "type": violation_type,
                    "line": line_num,
                    "context": context,
                })

    return violations


def analyze_results(results_file: Path, verbose: bool = False) -> dict:
    """Analyze compliance across all trials in a results file."""
    stats = {}  # condition -> {total_trials, total_violations, trials_with_violations, by_word}

    with open(results_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "error" in record or "response_text" not in record:
                continue

            condition = record["condition"]
            if condition not in stats:
                stats[condition] = {
                    "total_trials": 0,
                    "total_violations": 0,
                    "trials_with_violations": 0,
                    "total_words": 0,
                    "by_word": {},
                    "by_trial": [],
                }

            text = record["response_text"]
            word_count = len(tokenize(text))
            violations = find_violations(text, condition)

            stats[condition]["total_trials"] += 1
            stats[condition]["total_violations"] += len(violations)
            stats[condition]["total_words"] += word_count

            if violations:
                stats[condition]["trials_with_violations"] += 1

            for v in violations:
                word = v["word"]
                stats[condition]["by_word"][word] = stats[condition]["by_word"].get(word, 0) + 1

            stats[condition]["by_trial"].append({
                "trial_id": record["trial_id"],
                "item_id": record["item_id"],
                "task_type": record["task_type"],
                "n_violations": len(violations),
                "word_count": word_count,
                "violations": violations if verbose else [],
            })

    return stats


def print_report(stats: dict, verbose: bool = False):
    """Print a compliance report."""
    print("=" * 70)
    print("COMPLIANCE REPORT")
    print("=" * 70)

    for condition, data in stats.items():
        total = data["total_trials"]
        if total == 0:
            continue

        clean = total - data["trials_with_violations"]
        compliance_rate = clean / total * 100
        violations_per_1k = (data["total_violations"] / data["total_words"] * 1000) if data["total_words"] > 0 else 0

        print(f"\n--- {condition.upper()} ---")
        print(f"  Trials: {total}")
        print(f"  Clean trials (0 violations): {clean}/{total} ({compliance_rate:.1f}%)")
        print(f"  Total violations: {data['total_violations']}")
        print(f"  Violations per 1k words: {violations_per_1k:.2f}")

        if data["by_word"]:
            print(f"  Most frequent violations:")
            sorted_words = sorted(data["by_word"].items(), key=lambda x: -x[1])
            for word, count in sorted_words[:10]:
                print(f"    '{word}': {count}")

        if verbose:
            print(f"\n  Per-trial detail:")
            for trial in data["by_trial"]:
                if trial["n_violations"] > 0:
                    print(f"    {trial['task_type']}/{trial['item_id']}: {trial['n_violations']} violations")
                    for v in trial["violations"]:
                        print(f"      L{v['line']}: '{v['word']}' ({v['type']}) — ...{v['context']}...")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="E-Prime Compliance Checker")
    parser.add_argument("results_file", type=Path, help="JSONL results file to analyze")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-trial violation details")
    args = parser.parse_args()

    if not args.results_file.exists():
        print(f"File not found: {args.results_file}")
        sys.exit(1)

    stats = analyze_results(args.results_file, verbose=args.verbose)
    print_report(stats, verbose=args.verbose)


if __name__ == "__main__":
    main()
