"""A chat model that replays a fixed script, so the baseline, the agent and the
evaluation runner can be tested without a network or a real LLM."""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, PrivateAttr


def tool_call(name: str, args: dict, call_id: str = "call_1") -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def with_usage(message: AIMessage, input_tokens: int = 10, output_tokens: int = 5) -> AIMessage:
    message.usage_metadata = {
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    return message


class ScriptedChatModel(BaseChatModel):
    """Returns the scripted messages in order; `received` records what it was shown."""

    responses: list[AIMessage]
    received: list[list[BaseMessage]] = Field(default_factory=list)
    _cursor: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.received.append(list(messages))
        if self._cursor >= len(self.responses):
            raise RuntimeError("the script ran out of responses")
        message = self.responses[self._cursor]
        self._cursor += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools, **kwargs):
        return self
