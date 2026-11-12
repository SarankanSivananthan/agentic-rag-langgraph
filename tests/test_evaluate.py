from agentic_rag.evalset import load_questions
from agentic_rag.evaluate import evaluate, evaluate_question, read_records, write_records
from agentic_rag.metrics import ToolCall
from agentic_rag.runs import RunResult

QUESTIONS = {q.id: q for q in load_questions()}
LOOKUP = QUESTIONS["lookup-002"]  # Where was Millard Fillmore born? -> Summerhill, New York
CALCULATION = QUESTIONS["calculation-008"]  # -> 51
UNANSWERABLE = QUESTIONS["unanswerable-001"]  # capital of Ghana


def test_a_correct_answer_with_the_right_tool_is_recorded_as_such():
    system = lambda q: RunResult("Fillmore was born in Summerhill, New York.", [ToolCall("retrieve", {"query": q})], 1.5, 2, 300, 20)
    record = evaluate_question(system, LOOKUP)

    assert record.id == "lookup-002" and record.category == "lookup" and record.gold == "Summerhill, New York"
    assert record.correct == 1.0 and not record.abstained and record.exact_match == 0.0
    assert record.required_tools_used and record.enough_retrievals and record.needless_tools == []
    assert (record.latency_s, record.llm_calls, record.input_tokens, record.output_tokens) == (1.5, 2, 300, 20)
    assert record.trace == [{"name": "retrieve", "args": {"query": LOOKUP.question}, "output": ""}]


def test_a_numeric_question_answered_without_the_calculator_is_flagged():
    system = lambda q: RunResult("About 51.", [ToolCall("retrieve", {})])
    record = evaluate_question(system, CALCULATION)
    assert record.correct == 1.0  # the number is right...
    assert not record.required_tools_used  # ...but the calculator was never called


def test_unanswerable_questions_are_correct_only_when_the_system_abstains():
    abstains = evaluate_question(lambda q: RunResult("I don't know - the knowledge base does not contain this."), UNANSWERABLE)
    answers = evaluate_question(lambda q: RunResult("The capital of Ghana is Accra."), UNANSWERABLE)
    assert (abstains.correct, answers.correct) == (1.0, 0.0)
    assert abstains.gold is None and abstains.subtype == "absent_fact"


def test_a_failed_run_is_a_wrong_answer_that_keeps_its_error():
    record = evaluate_question(lambda q: RunResult("", error="TimeoutError: no reply"), LOOKUP)
    assert record.correct == 0.0 and record.error == "TimeoutError: no reply"


def test_evaluate_scores_every_question_in_order_and_reports_progress():
    questions = [LOOKUP, CALCULATION, UNANSWERABLE]
    seen = []
    records = evaluate(lambda q: RunResult("I don't know."), questions, on_record=seen.append)
    assert [r.id for r in records] == ["lookup-002", "calculation-008", "unanswerable-001"]
    assert seen == records


def test_records_survive_a_round_trip_through_a_file(tmp_path):
    system = lambda q: RunResult("Summerhill, New York", [ToolCall("retrieve", {"query": q})], 0.5, 1, 10, 5)
    records = evaluate(system, [LOOKUP, UNANSWERABLE])
    write_records(records, tmp_path / "nested" / "run.jsonl")
    assert read_records(tmp_path / "nested" / "run.jsonl") == records
