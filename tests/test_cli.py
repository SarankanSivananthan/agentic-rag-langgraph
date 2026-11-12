import pytest
from langchain_core.messages import AIMessage

from agentic_rag.cli import build_system, default_output_path, run_system, select_questions
from agentic_rag.evalset import load_questions
from agentic_rag.evaluate import read_records
from agentic_rag.retriever import Hit
from fakes import ScriptedChatModel, tool_call

QUESTIONS = load_questions()


class StubRetriever:
    def search(self, query, k=4):
        return [Hit("Fillmore was born in a log cabin in Summerhill, New York.", 1, 0.1)]


def test_select_questions_filters_by_category_and_limits_each_category():
    assert {q.category for q in select_questions(QUESTIONS, ("lookup",))} == {"lookup"}
    limited = select_questions(QUESTIONS, None, limit_per_category=2)
    assert len(limited) == 8 and all(sum(q.category == c for q in limited) == 2 for c in {q.category for q in limited})
    assert len(select_questions(QUESTIONS)) == 98


def test_output_path_is_derived_from_system_provider_and_a_safe_model_name():
    path = default_output_path("agent", "ollama", "llama3.1:8b/instruct")
    assert path.name == "agent_ollama_llama3.1-8b-instruct.jsonl" and path.parent.name == "results"


def test_the_baseline_runs_end_to_end_and_saves_each_record(tmp_path):
    lookups = select_questions(QUESTIONS, ("lookup",), 2)
    llm = ScriptedChatModel(responses=[AIMessage("Summerhill, New York."), AIMessage("I don't know.")])
    out = tmp_path / "run.jsonl"
    progress = []

    records = run_system(build_system("baseline", llm, StubRetriever()), lookups, out, progress=progress.append)

    assert [r.id for r in records] == ["lookup-001", "lookup-002"]
    assert read_records(out) == records
    assert len(progress) == 2 and progress[0].startswith("[1/2] lookup-001")


def test_the_agent_runs_end_to_end_through_the_same_loop(tmp_path):
    question = select_questions(QUESTIONS, ("lookup",), 1)
    llm = ScriptedChatModel(responses=[
        AIMessage("", tool_calls=[tool_call("retrieve", {"query": "Uruguay oldest church"})]),
        AIMessage("San Carlos, Maldonado."),
    ])
    records = run_system(build_system("agent", llm, StubRetriever()), question, tmp_path / "run.jsonl", progress=lambda m: None)
    assert records[0].correct == 1.0 and [t["name"] for t in records[0].trace] == ["retrieve"]


def test_an_interrupted_run_resumes_without_redoing_finished_questions(tmp_path):
    questions = select_questions(QUESTIONS, ("lookup",), 3)
    out = tmp_path / "run.jsonl"
    first = ScriptedChatModel(responses=[AIMessage("one"), AIMessage("two")])
    run_system(build_system("baseline", first, StubRetriever()), questions[:2], out, progress=lambda m: None)

    # The second run only has a response for the one question that is still missing.
    second = ScriptedChatModel(responses=[AIMessage("three")])
    messages = []
    records = run_system(build_system("baseline", second, StubRetriever()), questions, out, resume=True, progress=messages.append)

    assert [r.prediction for r in records] == ["one", "two", "three"]
    assert read_records(out) == records
    assert sum("already done" in m for m in messages) == 2


def test_a_run_without_resume_starts_the_file_over(tmp_path):
    out = tmp_path / "run.jsonl"
    out.write_text("stale\n")
    llm = ScriptedChatModel(responses=[AIMessage("x")])
    run_system(build_system("baseline", llm, StubRetriever()), select_questions(QUESTIONS, ("lookup",), 1), out, progress=lambda m: None)
    assert len(read_records(out)) == 1


def test_unknown_systems_are_refused():
    with pytest.raises(ValueError, match="unknown system"):
        build_system("oracle", ScriptedChatModel(responses=[]), StubRetriever())
