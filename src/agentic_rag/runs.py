"""What running a system on one question returns, whatever the system is."""
from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage

from agentic_rag.metrics import ToolCall


@dataclass
class RunResult:
    answer: str
    trace: list[ToolCall] = field(default_factory=list)
    latency_s: float = 0.0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None  # set when the run failed; the answer is then empty
    step_limit_hit: bool = False  # the agent was still calling tools when it ran out of steps


def message_text(message: BaseMessage) -> str:
    """The text of a message, whether its content is a string or a list of content blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in content)


def count_usage(messages: list[BaseMessage]) -> tuple[int, int, int]:
    """(LLM calls, input tokens, output tokens) over the AI messages of a run."""
    calls = input_tokens = output_tokens = 0
    for message in messages:
        if isinstance(message, AIMessage):
            calls += 1
            usage = message.usage_metadata or {}
            input_tokens += usage.get("input_tokens", 0)
            output_tokens += usage.get("output_tokens", 0)
    return calls, input_tokens, output_tokens
