"""The fixed pipeline the agent is compared against: always retrieve once with
the question itself, then answer from what came back. No tool choice, no second
search, no calculator; the same model and the same answering rules as the agent.
"""
from __future__ import annotations

import time

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agentic_rag.metrics import ToolCall
from agentic_rag.prompts import BASELINE_SYSTEM_PROMPT, BASELINE_USER_TEMPLATE
from agentic_rag.retriever import Retriever
from agentic_rag.runs import RunResult, count_usage, message_text
from agentic_rag.tools import DEFAULT_TOP_K


def run_baseline(llm: BaseChatModel, retriever: Retriever, question: str, k: int = DEFAULT_TOP_K) -> RunResult:
    started = time.perf_counter()
    trace = [ToolCall("retrieve", {"query": question})]
    try:
        hits = retriever.search(question, k=k)
        context = "\n\n".join(f"[{i}] {hit.text}" for i, hit in enumerate(hits, start=1))
        trace = [ToolCall("retrieve", {"query": question}, context)]
        reply = llm.invoke([
            SystemMessage(BASELINE_SYSTEM_PROMPT),
            HumanMessage(BASELINE_USER_TEMPLATE.format(context=context, question=question)),
        ])
    except Exception as error:  # a failed call is a failed question, not a crashed evaluation
        return RunResult(answer="", trace=trace, latency_s=time.perf_counter() - started, error=f"{type(error).__name__}: {error}")

    calls, input_tokens, output_tokens = count_usage([reply])
    return RunResult(
        answer=message_text(reply).strip(), trace=trace, latency_s=time.perf_counter() - started,
        llm_calls=calls, input_tokens=input_tokens, output_tokens=output_tokens,
    )
