"""Base class for LLM interfaces."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any


def chunk_batch_requests(
    requests: list[dict[str, Any]],
    max_count: int,
    max_bytes: int,
) -> list[list[dict[str, Any]]]:
    """Split batch requests into chunks bounded by both request count and serialized bytes.

    Batch endpoints cap a submission by payload size as well as request count, and
    long-CoT runs hit the size cap first. `max_bytes` is the caller's budget: the
    providers differ (Anthropic caps the submission, OpenAI caps the input file).
    """
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_bytes = 0
    for req in requests:
        size = len(json.dumps(req).encode())
        if current and (len(current) >= max_count or current_bytes + size > max_bytes):
            chunks.append(current)
            current, current_bytes = [], 0
        current.append(req)
        current_bytes += size
    if current:
        chunks.append(current)
    return chunks


def shutdown_vllm_engine(llm) -> None:
    """Gracefully shut down a vLLM LLM engine instance."""
    llm_engine = getattr(llm, "llm_engine", None)
    engine_core = getattr(llm_engine, "engine_core", None)
    if engine_core is not None and hasattr(engine_core, "shutdown"):
        engine_core.shutdown()
        return
    model_executor = getattr(llm_engine, "model_executor", None)
    if model_executor is not None and hasattr(model_executor, "shutdown"):
        model_executor.shutdown()


class BaseLLM(ABC):
    """Abstract base class for LLM implementations."""

    def __init__(self, model: str, **kwargs):
        """Initialize the LLM.

        Args:
            model: Model identifier.
            **kwargs: Additional model-specific parameters.
        """
        self.model = model
        self.kwargs = kwargs

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> str | dict[str, str]:
        """Generate a single chat response.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            **kwargs: Additional generation parameters.

        Returns:
            Generated response text, or a dict with 'reasoning' and 'content'
            when the backend returns reasoning separately.
        """
        ...

    @abstractmethod
    def chat_batch(
        self,
        messages_list: list[list[dict[str, str]]],
        **kwargs,
    ) -> list[str | dict[str, str]]:
        """Generate responses for multiple conversations.

        Args:
            messages_list: List of message lists.
            **kwargs: Additional generation parameters.

        Returns:
            List of generated responses (string or reasoning+content dict).
        """
        ...

    def close(self) -> None:
        """Release any underlying client resources."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"
