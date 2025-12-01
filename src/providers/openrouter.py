from __future__ import annotations

from .base import BaseProvider


class OpenRouterProvider(BaseProvider):
    def _clean_messages(self, messages: list) -> list:
        """Remove tool-related messages and content that some providers don't support."""
        cleaned = []
        for msg in messages:
            role = msg.get("role")
            # Skip tool messages entirely
            if role == "tool":
                continue

            # For user/assistant messages, ensure content is a simple string
            if role in ("user", "assistant", "system"):
                content = msg.get("content")
                # If content is a list (multi-part message), extract just text
                if isinstance(content, list):
                    text_parts = [
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    ]
                    content = " ".join(text_parts)

                cleaned.append({
                    "role": role,
                    "content": content
                })

        return cleaned
    async def chat_completion(self, provider_model_id: str, messages: list, params: dict):
        cleaned_messages = self._clean_messages(messages)

        payload = {
            "model": provider_model_id,
            "messages": cleaned_messages,
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
        if response.status_code != 200:
            raise Exception(f"OpenRouter API error: {response.status_code} - {response.text}")
        response.raise_for_status()
        return response.json()

    async def chat_completion_stream(
        self, provider_model_id: str, messages: list, params: dict
    ):
        cleaned_messages = self._clean_messages(messages)

        payload = {
            "model": provider_model_id,
            "messages": cleaned_messages,
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
            if response.status_code != 200:
                error_body = await response.aread()
                raise Exception(f"OpenRouter API error: {response.status_code} - {error_body.decode()}")
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    yield line + "\n\n"
