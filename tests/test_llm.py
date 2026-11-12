import pytest

from agentic_rag.llm import DEFAULT_MODELS, get_chat_model


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="unknown provider"):
        get_chat_model("anthropic")


def test_openai_needs_an_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        get_chat_model("openai")


def test_openai_model_is_built_with_the_requested_settings(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")  # constructing the client makes no request
    chat = get_chat_model("openai", model="some-model", temperature=0.0)
    assert chat.model_name == "some-model" and chat.temperature == 0.0


def test_ollama_model_is_built_with_the_requested_settings():
    chat = get_chat_model("ollama", model="some-model")
    assert chat.model == "some-model" and chat.temperature == 0.0


def test_defaults_are_used_when_no_model_is_given(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    assert get_chat_model("openai").model_name == DEFAULT_MODELS["openai"]
    assert get_chat_model("ollama").model == DEFAULT_MODELS["ollama"]
