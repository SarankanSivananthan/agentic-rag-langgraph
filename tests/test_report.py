import pytest

from agentic_rag.evaluate import QuestionRecord
from agentic_rag.report import summarize


def record(id, category, correct, **overrides):
    fields = dict(
        id=id, category=category, subtype=None, question="Q?", gold="A", prediction="p", correct=correct,
        exact_match=correct, f1=correct, abstained=False, required_tools_used=True, enough_retrievals=True,
        needless_tools=[], trace=[{"name": "retrieve", "args": {}}], latency_s=1.0, llm_calls=2,
        input_tokens=100, output_tokens=10, error=None, step_limit_hit=False,
    )
    fields.update(overrides)
    return QuestionRecord(**fields)


def test_rows_are_per_category_then_overall():
    records = [record("lookup-1", "lookup", 1.0), record("lookup-2", "lookup", 0.0), record("calculation-1", "calculation", 1.0)]
    rows = summarize(records)
    assert [r["category"] for r in rows] == ["lookup", "calculation", "all"]
    assert [r["n"] for r in rows] == [2, 1, 3]


def test_accuracy_has_an_interval_that_brackets_it():
    records = [record(f"lookup-{i}", "lookup", 1.0 if i < 6 else 0.0) for i in range(10)]
    row = summarize(records)[0]
    assert row["accuracy"] == pytest.approx(0.6)
    assert row["accuracy_low"] < 0.6 < row["accuracy_high"]
    assert row["accuracy_high"] - row["accuracy_low"] > 0.3  # ten questions cannot pin it down


def test_costs_errors_and_tool_use_are_aggregated():
    records = [
        record("lookup-1", "lookup", 1.0, latency_s=2.0, llm_calls=3, input_tokens=300, output_tokens=30,
               trace=[{"name": "retrieve", "args": {}}, {"name": "calculator", "args": {}}], needless_tools=["calculator"]),
        record("lookup-2", "lookup", 0.0, latency_s=4.0, llm_calls=1, input_tokens=100, output_tokens=10,
               error="TimeoutError", step_limit_hit=True, required_tools_used=False, abstained=True),
    ]
    row = summarize(records)[0]
    assert (row["mean_latency_s"], row["mean_llm_calls"]) == (3.0, 2.0)
    assert (row["mean_input_tokens"], row["mean_output_tokens"]) == (200.0, 20.0)
    assert (row["mean_tool_calls"], row["needless_tools_rate"], row["required_tools_rate"]) == (1.5, 0.5, 0.5)
    assert (row["errors"], row["step_limit_hits"], row["abstention_rate"]) == (1, 1, 0.5)


def test_categories_with_no_records_are_left_out():
    assert [r["category"] for r in summarize([record("lookup-1", "lookup", 1.0)])] == ["lookup", "all"]
    assert summarize([]) == []
