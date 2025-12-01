from __future__ import annotations

from typing import Tuple

import httpx

from .config import Config
from .providers.anthropic import AnthropicProvider
from .providers.base import BaseProvider
from .providers.llama_cpp import LlamaCppProvider
from .providers.ollama import OllamaProvider
from .providers.openai import OpenAIProvider
from .providers.openrouter import OpenRouterProvider
from .providers.qwen import QwenProvider


class ModelRouter:
    """Core routing logic mapping models to providers."""

    def __init__(self, config: Config):
        self.config = config
        self._model_map = self._build_model_map()
        self._http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=100, max_connections=200)
        )
        self._providers = self._initialize_providers()

    def _build_model_map(self) -> dict:
        model_map = {}

        for provider_name, provider_config in self.config.providers.items():
            if not provider_config.enabled:
                continue

            for model_config in provider_config.models:
                model_map[model_config.name] = (
                    provider_name,
                    model_config.provider_model_id,
                    model_config,
                )

                for alias in model_config.aliases:
                    model_map[alias] = (
                        provider_name,
                        model_config.provider_model_id,
                        model_config,
                    )

        return model_map

    def _initialize_providers(self) -> dict:
        providers: dict[str, BaseProvider] = {}

        for provider_name, provider_config in self.config.providers.items():
            if not provider_config.enabled:
                continue

            provider_type = provider_config.type
            if provider_type == "openai":
                providers[provider_name] = OpenAIProvider(
                    provider_config, self._http_client
                )
            elif provider_type == "anthropic":
                providers[provider_name] = AnthropicProvider(
                    provider_config, self._http_client
                )
            elif provider_type == "ollama":
                providers[provider_name] = OllamaProvider(
                    provider_config, self._http_client
                )
            elif provider_type == "llama_cpp":
                providers[provider_name] = LlamaCppProvider(
                    provider_config, self._http_client
                )
            elif provider_type == "openrouter":
                providers[provider_name] = OpenRouterProvider(
                    provider_config, self._http_client
                )
            elif provider_type == "qwen":
                providers[provider_name] = QwenProvider(provider_config, self._http_client)
            else:
                raise ValueError(f"Unknown provider type: {provider_type}")

        return providers

    def resolve_model(self, model_name: str) -> Tuple[BaseProvider, str]:
        if model_name not in self._model_map:
            available = sorted(self._model_map.keys())
            raise ValueError(
                f"Model '{model_name}' not found. Available models: {available}"
            )

        provider_name, provider_model_id, _ = self._model_map[model_name]
        provider = self._providers[provider_name]

        return provider, provider_model_id

    def list_models(self) -> list:
        models = []
        seen = set()

        for provider_name, provider_config in self.config.providers.items():
            if not provider_config.enabled:
                continue

            for model_config in provider_config.models:
                if model_config.name in seen:
                    continue
                seen.add(model_config.name)

                models.append(
                    {
                        "id": model_config.name,
                        "object": "model",
                        "created": 0,
                        "owned_by": provider_name,
                        "provider": provider_name,
                        "provider_type": provider_config.type,
                    }
                )

        return models

    def list_configured_models(self) -> list:
        """Return all models configured for every enabled provider.

        Unlike :meth:`list_models`, this returns every configured model without
        deduplication and includes provider-specific metadata to aid
        introspection.
        """

        models = []

        for provider_name, provider_config in self.config.providers.items():
            if not provider_config.enabled:
                continue

            for model_config in provider_config.models:
                models.append(
                    {
                        "id": model_config.name,
                        "object": "model",
                        "created": 0,
                        "owned_by": provider_name,
                        "provider": provider_name,
                        "provider_type": provider_config.type,
                        "provider_model_id": model_config.provider_model_id,
                        "aliases": model_config.aliases,
                    }
                )

        return models

    async def close(self) -> None:
        await self._http_client.aclose()
