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


# --- paired comparison ----------------------------------------------------

from agentic_rag.report import compare, sign_test_p  # noqa: E402


def run(correct_by_id):
    return [record(i, i.rsplit("-", 1)[0].replace("-", "_"), float(c)) for i, c in correct_by_id.items()]


def test_the_sign_test_matches_known_values():
    assert sign_test_p(0, 0) == 1.0
    assert sign_test_p(0, 5) == pytest.approx(2 / 32)
    assert sign_test_p(0, 10) == pytest.approx(2 / 1024)
    assert sign_test_p(5, 5) == 1.0
    assert sign_test_p(3, 7) == pytest.approx(2 * (1 + 10 + 45 + 120) / 1024)


def test_compare_reports_the_paired_difference_and_who_won_where():
    ids = [f"multi-step-{i}" for i in range(8)] + [f"lookup-{i}" for i in range(8)]
    baseline = run({i: (i.startswith("lookup") and int(i.rsplit("-", 1)[1]) < 6) for i in ids})  # right on 6 lookups, no multi-step
    agent = run({i: (i.startswith("multi-step") and int(i.rsplit("-", 1)[1]) < 6) or (i.startswith("lookup") and int(i.rsplit("-", 1)[1]) < 6) for i in ids})

    rows = {r["category"]: r for r in compare(baseline, agent)}

    assert rows["lookup"]["difference"] == 0 and rows["lookup"]["only_a"] == rows["lookup"]["only_b"] == 0
    assert rows["multi_step"]["difference"] == pytest.approx(6 / 8)
    assert (rows["multi_step"]["only_a"], rows["multi_step"]["only_b"]) == (0, 6)
    assert rows["multi_step"]["difference_low"] > 0  # all six disagreements favour the agent
    assert rows["multi_step"]["sign_test_p"] == pytest.approx(2 / 64)
    assert rows["all"]["n"] == 16 and rows["all"]["difference"] == pytest.approx(6 / 16)


def test_a_difference_that_could_be_noise_has_an_interval_spanning_zero():
    ids = [f"lookup-{i}" for i in range(10)]
    a = run({i: n < 5 for n, i in enumerate(ids)})
    b = run({i: n in (0, 1, 2, 3, 5) for n, i in enumerate(ids)})  # one question each way
    row = compare(a, b)[0]
    assert row["difference"] == 0 and row["difference_low"] < 0 < row["difference_high"]
    assert row["sign_test_p"] == 1.0


def test_comparing_runs_over_different_questions_is_refused():
    with pytest.raises(ValueError, match="same questions"):
        compare(run({"lookup-1": 1}), run({"lookup-2": 1}))
