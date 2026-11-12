"""The agent: a LangGraph loop in which the model chooses its own tools.

    start -> agent -> (tool calls?) -> tools -> agent -> ... -> end

The `agent` node asks the model, with the tools bound, what to do next. If it
asks for tools, the `tools` node runs them and the results go back to the
model; if it answers, the run ends. A step limit stops a model that keeps
calling tools forever, and the result says so instead of pretending it answered.
"""
from __future__ import annotations

import time
from typing import Annotated, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agentic_rag.metrics import ToolCall
from agentic_rag.prompts import AGENT_SYSTEM_PROMPT
from agentic_rag.runs import RunResult, count_usage, message_text

DEFAULT_MAX_STEPS = 8  # model calls per question


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    steps: int


def build_agent(llm: BaseChatModel, tools: list[BaseTool], max_steps: int = DEFAULT_MAX_STEPS):
    model = llm.bind_tools(tools)

    def agent(state: AgentState) -> dict:
        reply = model.invoke(state["messages"])
        return {"messages": [reply], "steps": state["steps"] + 1}

    def route(state: AgentState) -> str:
        wants_tools = bool(getattr(state["messages"][-1], "tool_calls", None))
        return "tools" if wants_tools and state["steps"] < max_steps else END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def tool_trace(messages: list[BaseMessage]) -> list[ToolCall]:
    """Every tool call in order, each with the output the tool node returned for it."""
    outputs = {m.tool_call_id: message_text(m) for m in messages if isinstance(m, ToolMessage)}
    return [
        ToolCall(call["name"], call["args"], outputs.get(call["id"], ""))
        for message in messages if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]


def run_agent(graph, question: str, max_steps: int = DEFAULT_MAX_STEPS) -> RunResult:
    started = time.perf_counter()
    try:
        final = graph.invoke(
            {"messages": [SystemMessage(AGENT_SYSTEM_PROMPT), HumanMessage(question)], "steps": 0},
            config={"recursion_limit": 2 * max_steps + 5},
        )
    except Exception as error:  # a failed run is a failed question, not a crashed evaluation
        return RunResult(answer="", latency_s=time.perf_counter() - started, error=f"{type(error).__name__}: {error}")

    messages = final["messages"]
    last = messages[-1]
    step_limit_hit = bool(getattr(last, "tool_calls", None))
    calls, input_tokens, output_tokens = count_usage(messages)
    return RunResult(
        answer="" if step_limit_hit else message_text(last).strip(),
        trace=tool_trace(messages), latency_s=time.perf_counter() - started,
        llm_calls=calls, input_tokens=input_tokens, output_tokens=output_tokens,
        step_limit_hit=step_limit_hit,
    )
