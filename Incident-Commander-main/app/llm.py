
from __future__ import annotations

from functools import lru_cache
from typing import Any

import os
import streamlit as st

from langchain_groq import ChatGroq

from app.config import settings


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """
    Create and cache the configured Groq LLM.
    """

    provider = settings.llm_provider.lower().strip()

    if provider != "groq":
        raise ValueError(
            f"Unsupported LLM provider: {settings.llm_provider}"
        )

    api_key = settings.groq_api_key or os.getenv("GROQ_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            api_key = None

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured. "
            "Add it to Streamlit Cloud Secrets."
        )

    return ChatGroq(
        model=settings.llm_model,
        groq_api_key=api_key,
        temperature=0,
        max_retries=2,
        timeout=60,
    )


class LazyLLM:
    """
    Lazy wrapper around get_llm().
    """

    def __getattr__(
        self,
        name: str,
    ) -> Any:
        return getattr(
            get_llm(),
            name,
        )

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return get_llm()(
            *args,
            **kwargs,
        )


llm = LazyLLM()
