from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agentic_rag.prompts import ANSWER_RULES
from agentic_rag.runs import count_usage, message_text
from fakes import with_usage


def test_message_text_handles_strings_and_content_blocks():
    assert message_text(AIMessage("plain text")) == "plain text"
    blocks = AIMessage(content=[{"type": "text", "text": "Hello "}, {"type": "text", "text": "world"}])
    assert message_text(blocks) == "Hello world"


def test_usage_is_summed_over_the_ai_messages_only():
    messages = [
        HumanMessage("question"),
        with_usage(AIMessage(""), 100, 10),
        ToolMessage("result", tool_call_id="1"),
        with_usage(AIMessage("answer"), 150, 20),
    ]
    assert count_usage(messages) == (2, 250, 30)


def test_messages_without_usage_metadata_count_as_a_call_with_no_tokens():
    assert count_usage([AIMessage("answer")]) == (1, 0, 0)


def test_the_answering_rules_forbid_memory_and_define_the_refusal():
    assert "Do not use your own memory" in ANSWER_RULES
    assert "I don't know" in ANSWER_RULES  # the refusal the scorer recognises
