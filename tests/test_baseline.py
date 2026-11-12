from langchain_core.messages import AIMessage

from agentic_rag.baseline import run_baseline
from agentic_rag.prompts import ANSWER_RULES
from agentic_rag.retriever import Hit
from fakes import ScriptedChatModel, with_usage


class StubRetriever:
    def __init__(self):
        self.queries = []

    def search(self, query, k=4):
        self.queries.append((query, k))
        return [Hit("Montevideo is the capital of Uruguay.", 1, 0.2), Hit("Uruguay borders Brazil.", 2, 0.4)][:k]


def test_baseline_retrieves_once_with_the_question_and_answers_from_the_passages():
    llm = ScriptedChatModel(responses=[with_usage(AIMessage("Montevideo.", ), 120, 4)])
    retriever = StubRetriever()

    result = run_baseline(llm, retriever, "What is the capital of Uruguay?", k=2)

    assert result.answer == "Montevideo." and result.error is None
    assert retriever.queries == [("What is the capital of Uruguay?", 2)]
    assert [(c.name, c.args) for c in result.trace] == [("retrieve", {"query": "What is the capital of Uruguay?"})]
    assert (result.llm_calls, result.input_tokens, result.output_tokens) == (1, 120, 4)
    assert result.latency_s >= 0


def test_the_model_is_shown_the_rules_the_passages_and_the_question():
    llm = ScriptedChatModel(responses=[AIMessage("Montevideo.")])
    run_baseline(llm, StubRetriever(), "What is the capital of Uruguay?")

    system, human = llm.received[0]
    assert system.content == ANSWER_RULES
    assert "[1] Montevideo is the capital of Uruguay." in human.content
    assert human.content.endswith("Question: What is the capital of Uruguay?")


def test_a_failing_model_call_is_recorded_not_raised():
    llm = ScriptedChatModel(responses=[])  # the script is empty, so the call raises
    result = run_baseline(llm, StubRetriever(), "Anything?")
    assert result.answer == "" and "script ran out" in result.error
    assert len(result.trace) == 1  # the retrieval still happened
