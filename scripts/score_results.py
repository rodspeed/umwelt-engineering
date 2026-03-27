"""
Scoring Pipeline: Extract accuracy, reasoning chain metrics, and epistemic specificity.

Usage:
    python scripts/score_results.py results/run-pilot-*.jsonl
    python scripts/score_results.py results/run-pilot-*.jsonl --export results/scored.csv
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Import compliance checker for integrated analysis
from check_compliance import find_violations, tokenize

ROOT = Path(__file__).resolve().parent.parent


def extract_answer_syllogism(response_text: str) -> str | None:
    """Extract VALID/INVALID answer from a syllogism response."""
    text = response_text.upper()

    # Priority 1: Bolded answer — strongest signal of intentional answer
    bolded = re.findall(r"\*\*(INVALID|VALID)\*\*", text)
    if bolded:
        return bolded[-1]

    # Priority 2: Explicit answer framing (works for control; E-Prime avoids "is")
    for pattern in [
        r"(?:conclusion|answer|verdict|judgment)\s*(?:is|:|remains|stands as|reads as)\s*(INVALID|VALID)",
        r"(?:therefore|thus|so|hence)[,:]?\s*(?:the conclusion (?:is|remains|stands as)\s*)?(INVALID|VALID)",
        r"^(INVALID|VALID)\s*$",
    ]:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            return match.group(1)

    # Priority 3: Last standalone VALID/INVALID, but filter out explanatory uses
    # like "to make the conclusion valid" or "for the argument to be valid"
    matches = list(re.finditer(r"\b(INVALID|VALID)\b", text))
    if matches:
        # Walk backwards, skip matches that appear in explanatory context
        for m in reversed(matches):
            start = max(0, m.start() - 40)
            context_before = text[start:m.start()]
            # Skip if this looks like "make X valid", "become valid", "considered valid"
            if re.search(r"(?:MAKE|BECOME|CONSIDERED|DEEMED|RENDER|ENSURE)\s*(?:\w+\s+)*$", context_before):
                continue
            return m.group(1)
        # If all were filtered, return the last one anyway
        return matches[-1].group(1)

    return None


def extract_answer_causal(response_text: str) -> str | None:
    """Extract A/B/C/D answer from a causal reasoning response."""
    text = response_text.upper()
    raw = response_text  # preserve case for checkmark detection

    # Priority 1: Checkmark/tick next to an option (✓, ✔, or "correct" right after letter)
    check_match = re.search(r"\*\*([A-D])\)\*\*\s*[✓✔]", raw)
    if check_match:
        return check_match.group(1).upper()

    # Priority 2: Explicit "Answer" section header followed by a letter
    answer_section = re.search(r"##\s*(?:ANSWER|FINAL ANSWER|CONCLUSION)\s*\n+\s*\*?\*?\(?([A-D])\)?", text)
    if answer_section:
        return answer_section.group(1)

    # Priority 3: Explicit answer framing
    for pattern in [
        r"(?:best answer|correct answer|answer)\s*(?:is|:|remains|stands as)\s*\*?\*?\(?([A-D])\)?",
        r"(?:select|choose|pick)\s*\*?\*?\(?([A-D])\)?",
    ]:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            return match.group(1)

    # Priority 4: Last bold letter (likely the final answer, not option listing)
    bold_matches = re.findall(r"\*\*\(?([A-D])\)?\*\*", text)
    if bold_matches:
        return bold_matches[-1]

    # Priority 5: Last standalone letter option mentioned
    matches = re.findall(r"\b([A-D])\)", text)
    if matches:
        return matches[-1]

    return None


def score_accuracy(record: dict) -> dict:
    """Score a single trial for accuracy."""
    task_type = record["task_type"]
    response = record["response_text"]
    correct = record["correct_answer"]

    if task_type == "syllogisms":
        extracted = extract_answer_syllogism(response)
    elif task_type in (
        "causal_reasoning", "analogical_reasoning", "classification",
        "epistemic_calibration", "ethical_dilemmas", "math_word_problems",
    ):
        extracted = extract_answer_causal(response)
    else:
        extracted = None

    is_correct = extracted == correct if extracted else None

    return {
        "extracted_answer": extracted,
        "correct_answer": correct,
        "is_correct": is_correct,
        "answer_extracted": extracted is not None,
    }


def measure_reasoning_chain(response_text: str) -> dict:
    """Measure properties of the reasoning chain."""
    tokens = tokenize(response_text)
    sentences = [s.strip() for s in re.split(r'[.!?]+', response_text) if s.strip()]
    lines = [l.strip() for l in response_text.split("\n") if l.strip()]

    # Count reasoning step indicators
    step_markers = len(re.findall(
        r"(?:step \d|first|second|third|next|then|therefore|thus|because|since|so |hence|this means|which means|it follows)",
        response_text.lower()
    ))

    return {
        "word_count": len(tokens),
        "sentence_count": len(sentences),
        "line_count": len(lines),
        "reasoning_steps": step_markers,
        "words_per_sentence": round(len(tokens) / max(len(sentences), 1), 1),
    }


def measure_epistemic_specificity(response_text: str) -> dict:
    """Score epistemic specificity — ratio of grounded vs. bare assertions.

    Grounded markers: "because", "since", "evidence", "suggests", "indicates",
                      "based on", "according to", "the data shows", "we can observe"
    Bare assertion markers: "clearly", "obviously", "certainly", "definitely",
                            "of course", "without doubt", "undeniably"
    """
    text = response_text.lower()

    grounded = len(re.findall(
        r"\b(?:because|since|evidence|suggests?|indicates?|based on|according to|"
        r"the data shows?|we can observe|this shows|which demonstrates?|"
        r"given that|due to|as shown|supported by|confirmed by)\b",
        text
    ))

    bare = len(re.findall(
        r"\b(?:clearly|obviously|certainly|definitely|of course|without doubt|"
        r"undeniably|unquestionably|indisputably|needless to say|it goes without saying)\b",
        text
    ))

    total_assertions = grounded + bare
    specificity_ratio = grounded / max(total_assertions, 1)

    return {
        "grounded_markers": grounded,
        "bare_assertion_markers": bare,
        "specificity_ratio": round(specificity_ratio, 3),
    }


def score_all(results_file: Path) -> list[dict]:
    """Score all trials in a results file."""
    scored = []

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

            accuracy = score_accuracy(record)
            chain = measure_reasoning_chain(record["response_text"])
            epistemic = measure_epistemic_specificity(record["response_text"])
            violations = find_violations(record["response_text"], record["condition"])

            scored.append({
                "trial_id": record["trial_id"],
                "condition": record["condition"],
                "task_type": record["task_type"],
                "item_id": record["item_id"],
                "difficulty": record.get("difficulty"),
                "temperature": record["temperature"],
                **accuracy,
                **chain,
                **epistemic,
                "n_violations": len(violations),
                "latency_seconds": record.get("latency_seconds"),
                "output_tokens": record.get("output_tokens"),
            })

    return scored


def print_summary(scored: list[dict]):
    """Print aggregate summary by condition × task."""
    print("=" * 80)
    print("SCORING SUMMARY")
    print("=" * 80)

    # Group by condition × task
    groups = {}
    for s in scored:
        key = (s["condition"], s["task_type"])
        groups.setdefault(key, []).append(s)

    # Header
    print(f"\n{'Condition':<12} {'Task':<20} {'N':>4} {'Acc':>7} {'Words':>7} {'Steps':>6} {'Spec':>6} {'Viol':>5}")
    print("-" * 80)

    for (condition, task), trials in sorted(groups.items()):
        n = len(trials)
        scored_trials = [t for t in trials if t["is_correct"] is not None]
        accuracy = sum(t["is_correct"] for t in scored_trials) / max(len(scored_trials), 1)
        avg_words = sum(t["word_count"] for t in trials) / n
        avg_steps = sum(t["reasoning_steps"] for t in trials) / n
        avg_spec = sum(t["specificity_ratio"] for t in trials) / n
        avg_violations = sum(t["n_violations"] for t in trials) / n

        print(f"{condition:<12} {task:<20} {n:>4} {accuracy:>6.1%} {avg_words:>7.0f} {avg_steps:>6.1f} {avg_spec:>6.3f} {avg_violations:>5.1f}")

    # Cross-condition comparison per task
    print(f"\n{'='*80}")
    print("CONDITION COMPARISON (accuracy delta: E-Prime - Control)")
    print("=" * 80)

    tasks_seen = set(s["task_type"] for s in scored)
    for task in sorted(tasks_seen):
        control_trials = [s for s in scored if s["condition"] == "control" and s["task_type"] == task and s["is_correct"] is not None]
        eprime_trials = [s for s in scored if s["condition"] == "e_prime" and s["task_type"] == task and s["is_correct"] is not None]

        if control_trials and eprime_trials:
            ctrl_acc = sum(t["is_correct"] for t in control_trials) / len(control_trials)
            ep_acc = sum(t["is_correct"] for t in eprime_trials) / len(eprime_trials)
            delta = ep_acc - ctrl_acc
            direction = "+" if delta > 0 else ""

            ctrl_words = sum(t["word_count"] for t in control_trials) / len(control_trials)
            ep_words = sum(t["word_count"] for t in eprime_trials) / len(eprime_trials)
            word_delta = ep_words - ctrl_words

            print(f"\n  {task}:")
            print(f"    Accuracy: Control={ctrl_acc:.1%}  E-Prime={ep_acc:.1%}  (delta: {direction}{delta:.1%})")
            print(f"    Avg words: Control={ctrl_words:.0f}  E-Prime={ep_words:.0f}  (delta: {direction}{word_delta:.0f})")


def export_csv(scored: list[dict], output_path: Path):
    """Export scored results to CSV for external analysis."""
    import csv
    if not scored:
        return

    keys = scored[0].keys()
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(scored)
    print(f"\nExported {len(scored)} scored trials to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Score experiment results")
    parser.add_argument("results_file", type=Path, help="JSONL results file to score")
    parser.add_argument("--export", type=Path, help="Export scored data to CSV")
    args = parser.parse_args()

    if not args.results_file.exists():
        print(f"File not found: {args.results_file}")
        sys.exit(1)

    scored = score_all(args.results_file)
    print_summary(scored)

    if args.export:
        export_csv(scored, args.export)


if __name__ == "__main__":
    main()
