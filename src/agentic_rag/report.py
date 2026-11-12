"""Turn per-question records into the numbers that get reported.

Rates are shown with 95% bootstrap intervals. With 20-30 questions per
category the intervals are wide, and that is the honest size of what this
evaluation can tell apart; they are never dropped to make a table cleaner.
"""
from __future__ import annotations

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
