from __future__ import annotations

import json

from .base import BaseProvider


class AnthropicProvider(BaseProvider):
    def _transform_messages(self, messages: list) -> tuple[str, list]:
        system = ""
        anthropic_messages = []

        for msg in messages:
            if msg.get("role") == "system":
                system = msg.get("content", "")
            else:
                anthropic_messages.append(
                    {
                        "role": msg.get("role"),
                        "content": msg.get("content"),
                    }
                )

        return system, anthropic_messages

    def _to_openai_format(self, anthropic_response: dict) -> dict:
        content = anthropic_response["content"][0]["text"]

        return {
            "id": anthropic_response["id"],
            "object": "chat.completion",
            "created": int(anthropic_response.get("created", 0)),
            "model": anthropic_response["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": anthropic_response.get("stop_reason"),
                }
            ],
            "usage": {
                "prompt_tokens": anthropic_response["usage"]["input_tokens"],
                "completion_tokens": anthropic_response["usage"]["output_tokens"],
                "total_tokens": (
                    anthropic_response["usage"]["input_tokens"]
                    + anthropic_response["usage"]["output_tokens"]
                ),
            },
        }

    async def chat_completion(self, provider_model_id: str, messages: list, params: dict):
        system, anthropic_messages = self._transform_messages(messages)

        payload = {
            "model": provider_model_id,
            "messages": anthropic_messages,
            "max_tokens": params.get("max_tokens", 4096),
        }

        if system:
            payload["system"] = system

        if "temperature" in params:
            payload["temperature"] = params["temperature"]

        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
        }

        response = await self.client.post(
            f"{self.config.base_url}/messages",
            json=payload,
            headers=headers,
            timeout=self.config.timeout,
        )
        response.raise_for_status()

        return self._to_openai_format(response.json())

    async def chat_completion_stream(
        self, provider_model_id: str, messages: list, params: dict
    ):
        system, anthropic_messages = self._transform_messages(messages)

        payload = {
            "model": provider_model_id,
            "messages": anthropic_messages,
            "max_tokens": params.get("max_tokens", 4096),
            "stream": True,
        }

        if system:
            payload["system"] = system

        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
        }

        async with self.client.stream(
            "POST",
            f"{self.config.base_url}/messages",
            json=payload,
            headers=headers,
            timeout=self.config.timeout,
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue

                data = json.loads(line[6:])

                if data.get("type") == "content_block_delta":
                    chunk = {
                        "id": data.get("id", ""),
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": provider_model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": data["delta"].get("text", ""),
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                elif data.get("type") == "message_stop":
                    yield "data: [DONE]\n\n"
