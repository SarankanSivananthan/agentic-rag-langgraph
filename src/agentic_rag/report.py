"""Turn per-question records into the numbers that get reported.

Rates are shown with 95% bootstrap intervals. With 20-30 questions per
category the intervals are wide, and that is the honest size of what this
evaluation can tell apart; they are never dropped to make a table cleaner.
"""
from __future__ import annotations

import math

import numpy as np

from agentic_rag.evaluate import QuestionRecord
from agentic_rag.evalset import CATEGORIES
from agentic_rag.metrics import bootstrap_ci


def _mean(values) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def summarize(records: list[QuestionRecord]) -> list[dict]:
    """One row per category present in the records, then an 'all' row.

    `accuracy` is the share of correct answers; on unanswerable questions that
    is the share of correct abstentions. Read the categories separately: the
    'all' row mixes categories in whatever proportion the set happens to have.
    """
    rows = []
    groups = [(c, [r for r in records if r.category == c]) for c in CATEGORIES] + [("all", records)]
    for name, group in groups:
        if not group:
            continue
        low, high = bootstrap_ci([r.correct for r in group])
        rows.append({
            "category": name,
            "n": len(group),
            "accuracy": _mean(r.correct for r in group),
            "accuracy_low": low,
            "accuracy_high": high,
            "abstention_rate": _mean(r.abstained for r in group),
            "required_tools_rate": _mean(r.required_tools_used for r in group),
            "enough_retrievals_rate": _mean(r.enough_retrievals for r in group),
            "needless_tools_rate": _mean(bool(r.needless_tools) for r in group),
            "mean_tool_calls": _mean(len(r.trace) for r in group),
            "mean_llm_calls": _mean(r.llm_calls for r in group),
            "mean_latency_s": _mean(r.latency_s for r in group),
            "mean_input_tokens": _mean(r.input_tokens for r in group),
            "mean_output_tokens": _mean(r.output_tokens for r in group),
            "errors": sum(r.error is not None for r in group),
            "step_limit_hits": sum(r.step_limit_hit for r in group),
        })
    return rows


def sign_test_p(only_a: int, only_b: int) -> float:
    """Two-sided exact sign test on the questions where the two systems disagree.

    Under "neither system is better" each disagreement is a coin flip. Questions
    both systems got right (or both wrong) carry no information and are ignored.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1))
    return min(1.0, 2 * tail / 2**n)


def compare(records_a: list[QuestionRecord], records_b: list[QuestionRecord]) -> list[dict]:
    """System B against system A on the same questions, paired.

    The difference is taken question by question (B correct minus A correct),
    so the interval reflects how often the two systems actually disagree, which
    is far tighter than comparing two separate intervals. Positive means B is
    better. `only_a` / `only_b` count the questions only that system got right.
    """
    by_id_a = {r.id: r for r in records_a}
    by_id_b = {r.id: r for r in records_b}
    if by_id_a.keys() != by_id_b.keys():
        raise ValueError("the two runs do not cover the same questions")

    rows = []
    groups = [(c, [i for i, r in by_id_a.items() if r.category == c]) for c in CATEGORIES]
    groups.append(("all", list(by_id_a)))
    for name, ids in groups:
        if not ids:
            continue
        a_correct = [by_id_a[i].correct >= 0.5 for i in ids]
        b_correct = [by_id_b[i].correct >= 0.5 for i in ids]
        differences = [float(b) - float(a) for a, b in zip(a_correct, b_correct)]
        low, high = bootstrap_ci(differences)
        only_a = sum(a and not b for a, b in zip(a_correct, b_correct))
        only_b = sum(b and not a for a, b in zip(a_correct, b_correct))
        rows.append({
            "category": name, "n": len(ids),
            "accuracy_a": _mean(a_correct), "accuracy_b": _mean(b_correct),
            "difference": _mean(differences), "difference_low": low, "difference_high": high,
            "only_a": only_a, "only_b": only_b, "sign_test_p": sign_test_p(only_a, only_b),
        })
    return rows


# --- markdown ---------------------------------------------------------------

_CATEGORY_LABELS = {
    "lookup": "lookup", "calculation": "calculation", "multi_step": "multi-step",
    "unanswerable": "unanswerable", "all": "all (mixture)",
}


def _pct(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.0%}"


def summary_markdown(rows: list[dict]) -> str:
    """The per-category table. Accuracy is a share of correct answers; on
    unanswerable questions it is the share of correct abstentions."""
    lines = [
        "| Category | Questions | Correct (95% CI) | Calls to tools / question | Used the tools it needed | LLM calls | Latency (s) | Tokens in / out |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {_CATEGORY_LABELS[r['category']]} | {r['n']} | "
            f"{_pct(r['accuracy'])} ({_pct(r['accuracy_low'])}-{_pct(r['accuracy_high'])}) | "
            f"{r['mean_tool_calls']:.1f} | {_pct(r['required_tools_rate'])} | {r['mean_llm_calls']:.1f} | "
            f"{r['mean_latency_s']:.1f} | {r['mean_input_tokens']:.0f} / {r['mean_output_tokens']:.0f} |"
        )
    return "\n".join(lines) + "\n"


def comparison_markdown(rows: list[dict], name_a: str, name_b: str) -> str:
    """B against A. Difference is in percentage points; 'only X' counts the
    questions only that system got right."""
    lines = [
        f"| Category | Questions | {name_a} | {name_b} | Difference, {name_b} - {name_a} (95% CI) | Only {name_a} right | Only {name_b} right | Sign test p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        diff = f"{r['difference'] * 100:+.0f} ({r['difference_low'] * 100:+.0f} to {r['difference_high'] * 100:+.0f}) pts"
        lines.append(
            f"| {_CATEGORY_LABELS[r['category']]} | {r['n']} | {_pct(r['accuracy_a'])} | {_pct(r['accuracy_b'])} | "
            f"{diff} | {r['only_a']} | {r['only_b']} | {r['sign_test_p']:.3f} |"
        )
    return "\n".join(lines) + "\n"
