"""Building the cloud chat model.

The provider is a switch rather than a hard-coded import so the same pipeline
can run against a hosted model or a local one. Nothing here knows about
masking: by the time text reaches this module it is already safe to send.
"""

from __future__ import annotations

import os
from typing import Any

DEFAULT_PROVIDER = "openai"
#: Seconds to wait for a cloud model. The SDK defaults run to minutes, which
#: on stage is indistinguishable from the program having died.
DEFAULT_TIMEOUT = 30.0
#: The SDK retries transient failures itself; two is enough to ride out a
#: blip without leaving the user staring at nothing.
DEFAULT_RETRIES = 2
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "groq": "llama-3.3-70b-versatile",
    "ollama": "llama3.1:8b",
}


def build_llm(
    provider: str | None = None,
    model: str | None = None,
    *,
    temperature: float = 0.2,
    base_url: str | None = None,
) -> Any:
    """Return a LangChain chat model for ``provider``.

    Raises RuntimeError with an actionable message rather than letting a
    missing key surface as an authentication error mid-request.
    """
    name = (provider or os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    chosen_model = model or os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(name)

    if name == "openai":
        _require_key("OPENAI_API_KEY", "https://platform.openai.com/api-keys")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=chosen_model,
            temperature=temperature,
            timeout=DEFAULT_TIMEOUT,
            max_retries=DEFAULT_RETRIES,
        )

    if name == "groq":
        _require_key("GROQ_API_KEY", "https://console.groq.com/keys")
        try:
            from langchain_groq import ChatGroq
        except ImportError as error:
            raise RuntimeError("Run: uv add langchain-groq") from error

        return ChatGroq(
            model=chosen_model,
            temperature=temperature,
            timeout=DEFAULT_TIMEOUT,
            max_retries=DEFAULT_RETRIES,
        )

    if name == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as error:
            raise RuntimeError("Run: uv add langchain-ollama") from error

        return ChatOllama(
            model=chosen_model,
            temperature=temperature,
            base_url=base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    supported = ", ".join(sorted(DEFAULT_MODELS))
    raise RuntimeError(f"Unknown provider {name!r}. Supported: {supported}")


def _require_key(variable: str, where: str) -> None:
    if not os.getenv(variable):
        raise RuntimeError(f"{variable} is not set. Put it in .env -- get one at {where}")
