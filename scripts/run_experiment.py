"""
Experiment Runner: Linguistic Constraints as Cognitive Constraints
Runs all condition × task × trial combinations, logs full outputs.
Resumable — skips completed trials on restart.

Usage:
    python scripts/run_experiment.py                    # full run per config
    python scripts/run_experiment.py --pilot             # pilot: 2 tasks × 2 conditions × temp 0
    python scripts/run_experiment.py --tasks syllogisms  # run specific task(s)
    python scripts/run_experiment.py --conditions control e_prime  # specific conditions
"""

import argparse
import hashlib
import json
import os
import sys
import time
import yaml
from datetime import datetime, timezone
from pathlib import Path

from model_backend import create_backend

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "experiment.yaml"
RESULTS_DIR = ROOT / "results"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_prompt(prompt_file: str) -> str:
    path = ROOT / prompt_file
    return path.read_text(encoding="utf-8").strip()


def load_task(task_file: str) -> dict:
    path = ROOT / task_file
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def trial_id(condition_id: str, task_type: str, item_id: str, trial_num: int, temperature: float, model_id: str = "") -> str:
    """Deterministic trial ID for resumability. Includes model for multi-model runs."""
    raw = f"{condition_id}|{task_type}|{item_id}|{trial_num}|{temperature}|{model_id}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_completed_trials(results_file: Path) -> set:
    """Load IDs of already-completed trials from a JSONL results file."""
    completed = set()
    if results_file.exists():
        with open(results_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        completed.add(record["trial_id"])
                    except (json.JSONDecodeError, KeyError):
                        continue
    return completed


def format_stimulus(task: dict, item: dict) -> str:
    """Build the user message for a given task item."""
    instructions = task["instructions"]

    if task["task_type"] == "syllogisms":
        premises_text = "\n".join(f"  Premise {i+1}: {p}" for i, p in enumerate(item["premises"]))
        return (
            f"{instructions}\n\n"
            f"Premises:\n{premises_text}\n\n"
            f"Conclusion: {item['conclusion']}\n\n"
            f"Is this conclusion VALID or INVALID?"
        )
    elif task["task_type"] == "causal_reasoning":
        options_text = "\n".join(f"  {opt}" for opt in item["options"])
        return (
            f"{instructions}\n\n"
            f"Scenario: {item['scenario']}\n\n"
            f"Question: {item['question']}\n\n"
            f"Options:\n{options_text}\n\n"
            f"Select the best answer and explain your reasoning."
        )
    elif task["task_type"] == "analogical_reasoning":
        return (
            f"{instructions}\n\n"
            f"{item['analogy_stem']}\n\n"
            f"Options:\n" + "\n".join(f"  {opt}" for opt in item["options"]) + "\n\n"
            f"Select the best answer and explain your reasoning."
        )
    elif task["task_type"] == "classification":
        options_text = "\n".join(f"  {opt}" for opt in item["options"])
        return (
            f"{instructions}\n\n"
            f"Description: {item['description']}\n\n"
            f"Question: {item['question']}\n\n"
            f"Options:\n{options_text}\n\n"
            f"Select the best answer and explain your reasoning."
        )
    elif task["task_type"] == "epistemic_calibration":
        options_text = "\n".join(f"  {opt}" for opt in item["options"])
        return (
            f"{instructions}\n\n"
            f"Scenario: {item['scenario']}\n\n"
            f"Question: {item['question']}\n\n"
            f"Options:\n{options_text}\n\n"
            f"Select the best answer and explain your reasoning."
        )
    elif task["task_type"] == "ethical_dilemmas":
        options_text = "\n".join(f"  {opt}" for opt in item["options"])
        return (
            f"{instructions}\n\n"
            f"Scenario: {item['scenario']}\n\n"
            f"Question: {item['question']}\n\n"
            f"Options:\n{options_text}\n\n"
            f"Select the best answer and explain your reasoning."
        )
    elif task["task_type"] == "math_word_problems":
        options_text = "\n".join(f"  {opt}" for opt in item["options"])
        return (
            f"{instructions}\n\n"
            f"Problem: {item['problem']}\n\n"
            f"Options:\n{options_text}\n\n"
            f"Select the correct answer and show your work."
        )
    else:
        # Generic fallback for future task types
        return f"{instructions}\n\n{json.dumps(item, indent=2)}"


def run_trial(
    backend,
    system_prompt: str,
    user_message: str,
    temperature: float,
    max_tokens: int,
) -> dict:
    """Run a single API call and return the response with metadata."""
    result = backend.complete_sync(
        system=system_prompt,
        user=user_message,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return {
        "response_text": result.text,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "latency_seconds": result.latency_seconds,
        "stop_reason": result.stop_reason,
        "model": result.model,
        "provider": result.provider,
    }


def run_experiment(args):
    config = load_config()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Determine which conditions to run
    all_conditions = config["conditions"]
    if args.conditions:
        conditions = {k: v for k, v in all_conditions.items() if k in args.conditions}
    elif args.pilot:
        conditions = {k: v for k, v in all_conditions.items() if k in ("control", "e_prime")}
    else:
        conditions = all_conditions

    # Determine which tasks to run
    all_tasks = config["tasks"]
    if args.tasks:
        tasks = {k: v for k, v in all_tasks.items() if k in args.tasks}
    elif args.pilot:
        tasks = {k: v for k, v in all_tasks.items() if k in ("syllogisms", "causal_reasoning")}
    else:
        tasks = all_tasks

    # Determine models
    model_key = "pilot" if args.pilot else "full"
    if args.models:
        model_ids = args.models
    else:
        model_ids = [m["id"] for m in config["models"][model_key]]

    # Determine temperatures
    if args.pilot:
        temperatures = [config["run_settings"]["temperature_deterministic"]]
    else:
        temperatures = [
            config["run_settings"]["temperature_deterministic"],
            config["run_settings"]["temperature_variance"],
        ]

    trials_per_item = 1 if args.pilot else config["run_settings"]["trials_per_item"]
    max_tokens = config["run_settings"]["max_tokens"]

    # Results file — one JSONL per run
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_label = "pilot" if args.pilot else "full"
    results_file = RESULTS_DIR / f"run-{run_label}-{timestamp}.jsonl"

    # For resumability, also check all existing result files
    completed = set()
    if config["run_settings"].get("resume", True):
        for f in RESULTS_DIR.glob("run-*.jsonl"):
            completed |= load_completed_trials(f)

    # Build run plan — now iterates over models too
    plan = []
    for model_id in model_ids:
        for cond_key, cond_cfg in conditions.items():
            system_prompt = load_prompt(cond_cfg["prompt_file"])
            for task_key, task_cfg in tasks.items():
                task_data = load_task(task_cfg["file"])
                for item in task_data["items"]:
                    for temp in temperatures:
                        n_trials = 1 if temp == 0 else trials_per_item
                        for t in range(n_trials):
                            tid = trial_id(cond_key, task_key, item["id"], t, temp, model_id)
                            if tid not in completed:
                                plan.append({
                                    "trial_id": tid,
                                    "model_id": model_id,
                                    "condition": cond_key,
                                    "task_type": task_key,
                                    "item_id": item["id"],
                                    "trial_num": t,
                                    "temperature": temp,
                                    "system_prompt": system_prompt,
                                    "stimulus": format_stimulus(task_data, item),
                                    "correct_answer": item["correct_answer"],
                                    "difficulty": item.get("difficulty"),
                                    "item_metadata": {
                                        k: v for k, v in item.items()
                                        if k not in ("id", "correct_answer", "difficulty")
                                    },
                                })

    total = len(plan)
    skipped = len(completed)
    if skipped > 0:
        print(f"Resuming: {skipped} trials already completed, {total} remaining")
    print(f"Running {total} trials: {list(conditions.keys())} × {list(tasks.keys())}")
    print(f"Models: {model_ids} | Temperatures: {temperatures} | Trials/item: {trials_per_item}")
    print(f"Results: {results_file}\n")

    if total == 0:
        print("Nothing to run — all trials already completed.")
        return

    # Create backends for each model (reuse across trials)
    backends = {}
    for mid in model_ids:
        try:
            backends[mid] = create_backend(mid)
            print(f"  Backend ready: {mid} ({backends[mid].provider})")
        except Exception as e:
            print(f"  WARNING: Could not create backend for {mid}: {e}")

    with open(results_file, "a", encoding="utf-8") as out:
        for i, trial_plan in enumerate(plan):
            model_id = trial_plan["model_id"]
            if model_id not in backends:
                print(f"  Skipping {trial_plan['item_id']} — no backend for {model_id}")
                continue

            label = f"[{i+1}/{total}] {model_id}/{trial_plan['condition']}/{trial_plan['task_type']}/{trial_plan['item_id']} (t={trial_plan['temperature']})"
            print(f"  {label} ...", end=" ", flush=True)

            try:
                result = run_trial(
                    backend=backends[model_id],
                    system_prompt=trial_plan["system_prompt"],
                    user_message=trial_plan["stimulus"],
                    temperature=trial_plan["temperature"],
                    max_tokens=max_tokens,
                )

                record = {
                    "trial_id": trial_plan["trial_id"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "condition": trial_plan["condition"],
                    "task_type": trial_plan["task_type"],
                    "item_id": trial_plan["item_id"],
                    "trial_num": trial_plan["trial_num"],
                    "temperature": trial_plan["temperature"],
                    "correct_answer": trial_plan["correct_answer"],
                    "difficulty": trial_plan["difficulty"],
                    "stimulus": trial_plan["stimulus"],
                    "response_text": result["response_text"],
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "latency_seconds": result["latency_seconds"],
                    "stop_reason": result["stop_reason"],
                    "model": result["model"],
                    "provider": result["provider"],
                }
                out.write(json.dumps(record) + "\n")
                out.flush()

                print(f"done ({result['output_tokens']} tok, {result['latency_seconds']}s)")

            except Exception as e:
                error_type = type(e).__name__
                if "rate" in error_type.lower() or "rate" in str(e).lower():
                    print(f"rate limited — waiting 60s")
                    time.sleep(60)
                    # Retry once
                    try:
                        result = run_trial(
                            backend=backends[model_id],
                            system_prompt=trial_plan["system_prompt"],
                            user_message=trial_plan["stimulus"],
                            temperature=trial_plan["temperature"],
                            max_tokens=max_tokens,
                        )
                        record = {
                            "trial_id": trial_plan["trial_id"],
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "condition": trial_plan["condition"],
                            "task_type": trial_plan["task_type"],
                            "item_id": trial_plan["item_id"],
                            "trial_num": trial_plan["trial_num"],
                            "temperature": trial_plan["temperature"],
                            "correct_answer": trial_plan["correct_answer"],
                            "difficulty": trial_plan["difficulty"],
                            "stimulus": trial_plan["stimulus"],
                            "response_text": result["response_text"],
                            "input_tokens": result["input_tokens"],
                            "output_tokens": result["output_tokens"],
                            "latency_seconds": result["latency_seconds"],
                            "stop_reason": result["stop_reason"],
                            "model": result["model"],
                            "provider": result["provider"],
                        }
                        out.write(json.dumps(record) + "\n")
                        out.flush()
                        print(f"  retry done ({result['output_tokens']} tok)")
                    except Exception as e2:
                        print(f"  retry failed: {e2}")
                else:
                    print(f"ERROR: {e}")
                    error_record = {
                        "trial_id": trial_plan["trial_id"],
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "condition": trial_plan["condition"],
                        "task_type": trial_plan["task_type"],
                        "item_id": trial_plan["item_id"],
                        "model": model_id,
                        "error": str(e),
                    }
                    out.write(json.dumps(error_record) + "\n")
                    out.flush()

    print(f"\nDone. Results saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(description="E-Prime LLM Experiment Runner")
    parser.add_argument("--pilot", action="store_true", help="Pilot mode: 2 tasks × 2 conditions × temp 0")
    parser.add_argument("--tasks", nargs="+", help="Specific task types to run")
    parser.add_argument("--conditions", nargs="+", help="Specific conditions to run")
    parser.add_argument("--models", nargs="+", help="Specific model IDs to run (e.g. claude-haiku-4-5-20251001 gpt-4o-mini)")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
