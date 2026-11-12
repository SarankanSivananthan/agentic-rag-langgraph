"""The chat model behind both the baseline and the agent.

They must use the same model, otherwise any difference between them could just
be a difference between models. Two backends are supported:

  openai   a hosted model; needs OPENAI_API_KEY in the environment
  ollama   a model served locally by Ollama (`ollama pull <model>` first)

The default model names are placeholders, not recommendations: pass the model
you actually want (`--model` on the command line, or `model=` here). Temperature
is 0 so a run is as repeatable as the backend allows.
"""
from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel

PROVIDERS = ("openai", "ollama")
DEFAULT_MODELS = {"openai": "gpt-4o-mini", "ollama": "llama3.1"}


def get_chat_model(provider: str, model: str | None = None, temperature: float = 0.0) -> BaseChatModel:
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; expected one of {PROVIDERS}")
    model = model or DEFAULT_MODELS[provider]

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set; export it in your shell before running")
        return ChatOpenAI(model=model, temperature=temperature)

    from langchain_ollama import ChatOllama

    return ChatOllama(model=model, temperature=temperature)
