"""
Full statistical analysis for the Umwelt Engineering paper.

Produces:
  - Accuracy by model × condition × task (with N)
  - McNemar's test or chi-squared for accuracy differences
  - Cohen's d effect sizes
  - Bootstrap 95% CIs for accuracy deltas
  - Cross-model correlation of constraint effect pattern
  - Compliance-filtered analysis (E-Prime trials with 0 violations only)
  - Summary table for paper

Usage:
    python scripts/analyze_statistics.py results/scored-full.csv
"""

import csv
import sys
import os
import math
import random
from pathlib import Path
from collections import defaultdict
from itertools import combinations

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

random.seed(42)

def load_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))

def accuracy(trials):
    scored = [t for t in trials if t["is_correct"] != ""]
    if not scored:
        return None, 0
    correct = sum(1 for t in scored if t["is_correct"] == "True")
    return correct / len(scored), len(scored)

def cohens_d(group1_binary, group2_binary):
    """Cohen's d for two groups of binary outcomes."""
    n1, n2 = len(group1_binary), len(group2_binary)
    if n1 < 2 or n2 < 2:
        return None
    m1 = sum(group1_binary) / n1
    m2 = sum(group2_binary) / n2
    # Pooled SD for binary data
    var1 = m1 * (1 - m1)
    var2 = m2 * (1 - m2)
    pooled_sd = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return (m2 - m1) / pooled_sd

def bootstrap_ci(control_binary, treatment_binary, n_boot=10000, alpha=0.05):
    """Bootstrap 95% CI for accuracy difference (treatment - control)."""
    deltas = []
    for _ in range(n_boot):
        c_sample = random.choices(control_binary, k=len(control_binary))
        t_sample = random.choices(treatment_binary, k=len(treatment_binary))
        delta = sum(t_sample) / len(t_sample) - sum(c_sample) / len(c_sample)
        deltas.append(delta)
    deltas.sort()
    lo = deltas[int(n_boot * alpha / 2)]
    hi = deltas[int(n_boot * (1 - alpha / 2))]
    return lo, hi

def chi2_2x2(a, b, c, d):
    """Chi-squared test for 2×2 contingency table.
    a=ctrl_correct, b=ctrl_wrong, c=treat_correct, d=treat_wrong.
    Returns chi2 statistic and approximate p-value.
    """
    n = a + b + c + d
    if n == 0:
        return 0, 1.0
    # Expected values
    r1, r2 = a + b, c + d
    c1, c2 = a + c, b + d
    expected = [(r1*c1/n), (r1*c2/n), (r2*c1/n), (r2*c2/n)]
    if any(e == 0 for e in expected):
        return 0, 1.0
    chi2 = sum((o - e)**2 / e for o, e in zip([a, b, c, d], expected))
    # Approximate p-value using chi2 with 1 df
    # Using survival function approximation
    p = chi2_sf(chi2, 1)
    return chi2, p

def chi2_sf(x, df=1):
    """Approximate chi-squared survival function (1-CDF) for df=1."""
    if x <= 0:
        return 1.0
    # Use normal approximation: sqrt(2*chi2) - sqrt(2*df - 1) ~ N(0,1)
    z = math.sqrt(2 * x) - math.sqrt(2 * df - 1)
    # Standard normal survival
    return 0.5 * math.erfc(z / math.sqrt(2))

def stars(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "ns"

def pearson_r(xs, ys):
    """Pearson correlation coefficient."""
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - 1)
    sx = math.sqrt(sum((x - mx)**2 for x in xs) / (n - 1))
    sy = math.sqrt(sum((y - my)**2 for y in ys) / (n - 1))
    if sx == 0 or sy == 0:
        return None
    return cov / (sx * sy)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_statistics.py results/scored-full.csv")
        sys.exit(1)

    data = load_csv(sys.argv[1])
    print(f"Loaded {len(data)} scored trials\n")

    # Parse binary correctness
    for d in data:
        d["correct_bin"] = 1 if d["is_correct"] == "True" else (0 if d["is_correct"] == "False" else None)
        d["n_violations"] = int(d["n_violations"]) if d["n_violations"] else 0

    scoreable = [d for d in data if d["correct_bin"] is not None]

    models = sorted(set(d["model"] for d in scoreable))
    tasks = sorted(set(d["task_type"] for d in scoreable))
    conditions = ["control", "e_prime", "no_have"]

    # =========================================================================
    # 1. Accuracy table: model × condition × task
    # =========================================================================
    print("=" * 100)
    print("TABLE 1: Accuracy by Model × Condition × Task")
    print("=" * 100)

    header = f"{'Model':<30} {'Task':<24} {'Control':>10} {'E-Prime':>10} {'No-Have':>10} {'Δ(EP)':>8} {'Δ(NH)':>8}"
    print(header)
    print("-" * 100)

    # Store deltas for cross-model correlation
    model_deltas_ep = defaultdict(dict)
    model_deltas_nh = defaultdict(dict)

    for model in models:
        for task in tasks:
            accs = {}
            ns = {}
            for cond in conditions:
                trials = [d for d in scoreable if d["model"] == model and d["task_type"] == task and d["condition"] == cond]
                acc, n = accuracy([{"is_correct": "True" if d["correct_bin"] else "False"} for d in trials])
                accs[cond] = acc
                ns[cond] = n

            if accs["control"] is not None and accs["e_prime"] is not None:
                d_ep = accs["e_prime"] - accs["control"]
                model_deltas_ep[model][task] = d_ep
            else:
                d_ep = None

            if accs["control"] is not None and accs["no_have"] is not None:
                d_nh = accs["no_have"] - accs["control"]
                model_deltas_nh[model][task] = d_nh
            else:
                d_nh = None

            ctrl_str = f"{accs['control']:.1%}({ns['control']})" if accs["control"] is not None else "—"
            ep_str = f"{accs['e_prime']:.1%}({ns['e_prime']})" if accs["e_prime"] is not None else "—"
            nh_str = f"{accs['no_have']:.1%}({ns['no_have']})" if accs["no_have"] is not None else "—"
            dep_str = f"{d_ep:+.1%}" if d_ep is not None else "—"
            dnh_str = f"{d_nh:+.1%}" if d_nh is not None else "—"

            print(f"{model:<30} {task:<24} {ctrl_str:>10} {ep_str:>10} {nh_str:>10} {dep_str:>8} {dnh_str:>8}")
        print()

    # =========================================================================
    # 2. Aggregate (all models pooled) with stats
    # =========================================================================
    print("\n" + "=" * 100)
    print("TABLE 2: Aggregate Accuracy (All Models Pooled) with Effect Sizes")
    print("=" * 100)
    print(f"{'Task':<24} {'Ctrl':>7} {'EP':>7} {'NH':>7} {'Δ(EP)':>8} {'d(EP)':>7} {'95%CI(EP)':>16} {'p(EP)':>8} {'Δ(NH)':>8} {'d(NH)':>7} {'95%CI(NH)':>16} {'p(NH)':>8}")
    print("-" * 130)

    for task in tasks:
        ctrl = [d["correct_bin"] for d in scoreable if d["condition"] == "control" and d["task_type"] == task]
        ep = [d["correct_bin"] for d in scoreable if d["condition"] == "e_prime" and d["task_type"] == task]
        nh = [d["correct_bin"] for d in scoreable if d["condition"] == "no_have" and d["task_type"] == task]

        ctrl_acc = sum(ctrl) / len(ctrl) if ctrl else None
        ep_acc = sum(ep) / len(ep) if ep else None
        nh_acc = sum(nh) / len(nh) if nh else None

        # E-Prime vs Control
        if ctrl and ep:
            delta_ep = ep_acc - ctrl_acc
            d_ep = cohens_d(ctrl, ep)
            ci_ep = bootstrap_ci(ctrl, ep)
            a = sum(ep)  # treat correct
            b = len(ep) - a  # treat wrong
            c = sum(ctrl)  # ctrl correct
            d_val = len(ctrl) - c  # ctrl wrong
            chi2_ep, p_ep = chi2_2x2(c, d_val, a, b)
        else:
            delta_ep = d_ep = chi2_ep = p_ep = None
            ci_ep = (None, None)

        # No-Have vs Control
        if ctrl and nh:
            delta_nh = nh_acc - ctrl_acc
            d_nh = cohens_d(ctrl, nh)
            ci_nh = bootstrap_ci(ctrl, nh)
            a = sum(nh)
            b = len(nh) - a
            c = sum(ctrl)
            d_val = len(ctrl) - c
            chi2_nh, p_nh = chi2_2x2(c, d_val, a, b)
        else:
            delta_nh = d_nh = chi2_nh = p_nh = None
            ci_nh = (None, None)

        def fmt(v, pct=False):
            if v is None: return "—"
            return f"{v:.1%}" if pct else f"{v:.3f}"

        ci_ep_str = f"[{ci_ep[0]:+.1%},{ci_ep[1]:+.1%}]" if ci_ep[0] is not None else "—"
        ci_nh_str = f"[{ci_nh[0]:+.1%},{ci_nh[1]:+.1%}]" if ci_nh[0] is not None else "—"
        p_ep_str = f"{p_ep:.4f}{stars(p_ep)}" if p_ep is not None else "—"
        p_nh_str = f"{p_nh:.4f}{stars(p_nh)}" if p_nh is not None else "—"

        print(f"{task:<24} {fmt(ctrl_acc,True):>7} {fmt(ep_acc,True):>7} {fmt(nh_acc,True):>7} "
              f"{fmt(delta_ep,True):>8} {fmt(d_ep):>7} {ci_ep_str:>16} {p_ep_str:>8} "
              f"{fmt(delta_nh,True):>8} {fmt(d_nh):>7} {ci_nh_str:>16} {p_nh_str:>8}")

    # =========================================================================
    # 3. Cross-model correlation of constraint effect
    # =========================================================================
    print("\n" + "=" * 100)
    print("TABLE 3: Cross-Model Correlation of E-Prime Effect (per-task Δ accuracy)")
    print("=" * 100)

    for m1, m2 in combinations(models, 2):
        common_tasks = sorted(set(model_deltas_ep[m1].keys()) & set(model_deltas_ep[m2].keys()))
        if len(common_tasks) >= 3:
            xs = [model_deltas_ep[m1][t] for t in common_tasks]
            ys = [model_deltas_ep[m2][t] for t in common_tasks]
            r = pearson_r(xs, ys)
            r_str = f"{r:.3f}" if r is not None else "—"
            print(f"  {m1} vs {m2}: r = {r_str} (n={len(common_tasks)} tasks)")
            for t, x, y in zip(common_tasks, xs, ys):
                print(f"    {t:<24} {x:+.1%}  {y:+.1%}")

    # =========================================================================
    # 4. Compliance-filtered analysis (E-Prime trials with 0 violations)
    # =========================================================================
    print("\n" + "=" * 100)
    print("TABLE 4: Compliance-Filtered E-Prime (0 violations only) vs Control")
    print("=" * 100)

    ep_compliant = [d for d in scoreable if d["condition"] == "e_prime" and d["n_violations"] == 0]
    ep_noncompliant = [d for d in scoreable if d["condition"] == "e_prime" and d["n_violations"] > 0]

    print(f"\nE-Prime trials: {len([d for d in scoreable if d['condition']=='e_prime'])} total, "
          f"{len(ep_compliant)} compliant (0 violations), {len(ep_noncompliant)} non-compliant")

    print(f"\n{'Task':<24} {'Ctrl':>7} {'EP(all)':>8} {'EP(comp)':>9} {'N(comp)':>8} {'Δ(comp)':>8}")
    print("-" * 70)

    for task in tasks:
        ctrl = [d["correct_bin"] for d in scoreable if d["condition"] == "control" and d["task_type"] == task]
        ep_all = [d["correct_bin"] for d in scoreable if d["condition"] == "e_prime" and d["task_type"] == task]
        ep_comp = [d["correct_bin"] for d in ep_compliant if d["task_type"] == task]

        ctrl_acc = sum(ctrl) / len(ctrl) if ctrl else None
        ep_all_acc = sum(ep_all) / len(ep_all) if ep_all else None
        ep_comp_acc = sum(ep_comp) / len(ep_comp) if ep_comp else None

        delta = (ep_comp_acc - ctrl_acc) if (ep_comp_acc is not None and ctrl_acc is not None) else None

        def fmt(v):
            return f"{v:.1%}" if v is not None else "—"

        print(f"{task:<24} {fmt(ctrl_acc):>7} {fmt(ep_all_acc):>8} {fmt(ep_comp_acc):>9} {len(ep_comp):>8} {fmt(delta):>8}")

    # =========================================================================
    # 5. Word count reduction (conciseness effect)
    # =========================================================================
    print("\n" + "=" * 100)
    print("TABLE 5: Average Word Count by Condition × Task")
    print("=" * 100)
    print(f"{'Task':<24} {'Control':>10} {'E-Prime':>10} {'No-Have':>10} {'Δ%(EP)':>8} {'Δ%(NH)':>8}")
    print("-" * 70)

    for task in tasks:
        for cond in conditions:
            trials = [d for d in data if d["condition"] == cond and d["task_type"] == task and d["word_count"]]
        ctrl_words = [int(d["word_count"]) for d in data if d["condition"] == "control" and d["task_type"] == task]
        ep_words = [int(d["word_count"]) for d in data if d["condition"] == "e_prime" and d["task_type"] == task]
        nh_words = [int(d["word_count"]) for d in data if d["condition"] == "no_have" and d["task_type"] == task]

        c_avg = sum(ctrl_words) / len(ctrl_words) if ctrl_words else 0
        e_avg = sum(ep_words) / len(ep_words) if ep_words else 0
        n_avg = sum(nh_words) / len(nh_words) if nh_words else 0

        ep_pct = ((e_avg - c_avg) / c_avg * 100) if c_avg else 0
        nh_pct = ((n_avg - c_avg) / c_avg * 100) if c_avg else 0

        print(f"{task:<24} {c_avg:>10.0f} {e_avg:>10.0f} {n_avg:>10.0f} {ep_pct:>+7.1f}% {nh_pct:>+7.1f}%")

    # =========================================================================
    # 6. Summary statistics for abstract
    # =========================================================================
    print("\n" + "=" * 100)
    print("SUMMARY FOR ABSTRACT")
    print("=" * 100)

    all_ctrl = [d["correct_bin"] for d in scoreable if d["condition"] == "control"]
    all_ep = [d["correct_bin"] for d in scoreable if d["condition"] == "e_prime"]
    all_nh = [d["correct_bin"] for d in scoreable if d["condition"] == "no_have"]

    print(f"Overall accuracy — Control: {sum(all_ctrl)/len(all_ctrl):.1%} (N={len(all_ctrl)})")
    print(f"Overall accuracy — E-Prime: {sum(all_ep)/len(all_ep):.1%} (N={len(all_ep)})")
    print(f"Overall accuracy — No-Have: {sum(all_nh)/len(all_nh):.1%} (N={len(all_nh)})")

    # Tasks where constraint helped (>2% improvement)
    helped_ep = [t for t in tasks if any(
        d["condition"] == "e_prime" and d["task_type"] == t for d in scoreable
    ) and (sum(d["correct_bin"] for d in scoreable if d["condition"]=="e_prime" and d["task_type"]==t) /
           len([d for d in scoreable if d["condition"]=="e_prime" and d["task_type"]==t]) -
           sum(d["correct_bin"] for d in scoreable if d["condition"]=="control" and d["task_type"]==t) /
           len([d for d in scoreable if d["condition"]=="control" and d["task_type"]==t])) > 0.02]

    hurt_ep = [t for t in tasks if any(
        d["condition"] == "e_prime" and d["task_type"] == t for d in scoreable
    ) and (sum(d["correct_bin"] for d in scoreable if d["condition"]=="e_prime" and d["task_type"]==t) /
           len([d for d in scoreable if d["condition"]=="e_prime" and d["task_type"]==t]) -
           sum(d["correct_bin"] for d in scoreable if d["condition"]=="control" and d["task_type"]==t) /
           len([d for d in scoreable if d["condition"]=="control" and d["task_type"]==t])) < -0.02]

    print(f"\nE-Prime improved (>2%): {', '.join(helped_ep)}")
    print(f"E-Prime hurt (<-2%):    {', '.join(hurt_ep)}")
    print(f"Models tested: {', '.join(models)}")
    print(f"Total trials: {len(data)}")
    print(f"Scoreable trials: {len(scoreable)}")


if __name__ == "__main__":
    main()
