from __future__ import annotations

from .base import BaseProvider


class OpenRouterProvider(BaseProvider):
    async def chat_completion(self, provider_model_id: str, messages: list, params: dict):
        payload = {
            "model": provider_model_id,
            "messages": messages,
            **params,
        }

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "HTTP-Referer": "https://your-app.com",
            "X-Title": "API Router",
        }

        response = await self.client.post(
            f"{self.config.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        return response.json()

    async def chat_completion_stream(
        self, provider_model_id: str, messages: list, params: dict
    ):
        payload = {
            "model": provider_model_id,
            "messages": messages,
            "stream": True,
            **params,
        }

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "HTTP-Referer": "https://your-app.com",
            "X-Title": "API Router",
        }

        async with self.client.stream(
            "POST",
            f"{self.config.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.config.timeout,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    yield line + "\n\n"
