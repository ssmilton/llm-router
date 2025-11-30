from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict

import httpx


class BaseProvider(ABC):
    """Abstract base for provider adapters."""

    def __init__(self, config: Any, client: httpx.AsyncClient):
        self.config = config
        self.client = client

    @abstractmethod
    async def chat_completion(
        self,
        provider_model_id: str,
        messages: list,
        params: dict,
    ) -> Dict[str, Any]:
        """Non-streaming chat completion."""

    @abstractmethod
    async def chat_completion_stream(
        self,
        provider_model_id: str,
        messages: list,
        params: dict,
    ) -> AsyncIterator[str]:
        """Streaming chat completion returning SSE chunks."""
