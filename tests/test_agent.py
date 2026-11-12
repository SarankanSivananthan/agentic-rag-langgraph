import pytest
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from agentic_rag.agent import build_agent, run_agent
from agentic_rag.prompts import AGENT_SYSTEM_PROMPT
from agentic_rag.retriever import Hit
from agentic_rag.tools import make_tools
from fakes import ScriptedChatModel, tool_call, with_usage


class StubRetriever:
    def __init__(self):
        self.queries = []

    def search(self, query, k=4):
        self.queries.append(query)
        return [Hit(f"Passage about {query}.", 1, 0.1)]


@pytest.fixture
def retriever():
    return StubRetriever()


def agent_for(responses, retriever, max_steps=8):
    llm = ScriptedChatModel(responses=responses)
    return llm, build_agent(llm, make_tools(retriever), max_steps=max_steps)


def test_a_lookup_is_one_search_then_an_answer(retriever):
    llm, graph = agent_for([
        with_usage(AIMessage("", tool_calls=[tool_call("retrieve", {"query": "Uruguay capital"})]), 100, 8),
        with_usage(AIMessage("Montevideo."), 150, 4),
    ], retriever)

    result = run_agent(graph, "What is the capital of Uruguay?")

    assert result.answer == "Montevideo." and result.error is None and not result.step_limit_hit
    assert [(c.name, c.args) for c in result.trace] == [("retrieve", {"query": "Uruguay capital"})]
    assert (result.llm_calls, result.input_tokens, result.output_tokens) == (2, 250, 12)
    assert retriever.queries == ["Uruguay capital"]


def test_the_tool_result_is_fed_back_to_the_model(retriever):
    llm, graph = agent_for([
        AIMessage("", tool_calls=[tool_call("date_difference", {"start": "February 12, 1809", "end": "November 6, 1860"})]),
        AIMessage("Lincoln was 51."),
    ], retriever)

    run_agent(graph, "How old was Lincoln when elected?")

    second_call = llm.received[1]
    assert isinstance(second_call[-1], ToolMessage) and second_call[-1].content == "51 years"


def test_a_two_step_question_searches_twice(retriever):
    _, graph = agent_for([
        AIMessage("", tool_calls=[tool_call("retrieve", {"query": "thirteenth President"}, "a")]),
        AIMessage("", tool_calls=[tool_call("retrieve", {"query": "Millard Fillmore birthplace"}, "b")]),
        AIMessage("Summerhill, New York."),
    ], retriever)

    result = run_agent(graph, "Where was the thirteenth President born?")

    assert [c.args["query"] for c in result.trace] == ["thirteenth President", "Millard Fillmore birthplace"]
    assert result.llm_calls == 3


def test_several_tool_calls_in_one_turn_are_all_run_and_traced(retriever):
    _, graph = agent_for([
        AIMessage("", tool_calls=[
            tool_call("retrieve", {"query": "one"}, "a"), tool_call("calculator", {"expression": "2 + 2"}, "b"),
        ]),
        AIMessage("Done."),
    ], retriever)
    result = run_agent(graph, "Q?")
    assert [c.name for c in result.trace] == ["retrieve", "calculator"]


def test_an_unknown_tool_or_bad_arguments_become_text_the_model_can_read(retriever):
    llm, graph = agent_for([
        AIMessage("", tool_calls=[tool_call("web_search", {"query": "x"}, "a")]),
        AIMessage("", tool_calls=[tool_call("calculator", {"wrong_argument": "1"}, "b")]),
        AIMessage("I don't know - the knowledge base does not contain this."),
    ], retriever)

    result = run_agent(graph, "Q?")

    assert result.error is None and result.answer.startswith("I don't know")
    tool_messages = [m for m in llm.received[-1] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 2 and all("rror" in m.content for m in tool_messages)


def test_a_model_that_never_stops_calling_tools_hits_the_step_limit(retriever):
    forever = [AIMessage("", tool_calls=[tool_call("retrieve", {"query": "again"}, f"c{i}")]) for i in range(10)]
    _, graph = agent_for(forever, retriever, max_steps=3)

    result = run_agent(graph, "Q?", max_steps=3)

    assert result.step_limit_hit and result.answer == "" and result.error is None
    assert result.llm_calls == 3 and len(result.trace) == 3


def test_the_model_is_given_the_agent_instructions_and_the_question(retriever):
    llm, graph = agent_for([AIMessage("Answer.")], retriever)
    run_agent(graph, "What is X?")

    system, human = llm.received[0]
    assert isinstance(system, SystemMessage) and system.content == AGENT_SYSTEM_PROMPT
    assert human.content == "What is X?"
    assert "retrieve" in AGENT_SYSTEM_PROMPT and "date_difference" in AGENT_SYSTEM_PROMPT


def test_a_failing_model_call_is_recorded_not_raised(retriever):
    _, graph = agent_for([], retriever)  # empty script: the first call raises
    result = run_agent(graph, "Q?")
    assert result.answer == "" and "script ran out" in result.error
