from __future__ import annotations

from .base import BaseProvider


class QwenProvider(BaseProvider):
    async def chat_completion(self, provider_model_id: str, messages: list, params: dict):
        payload = {
            "model": provider_model_id,
            "input": {"messages": messages},
            "parameters": {
                "temperature": params.get("temperature", 1.0),
                "max_tokens": params.get("max_tokens", 2000),
            },
        }

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        response = await self.client.post(
            f"{self.config.base_url}/services/aigc/text-generation/generation",
            json=payload,
            headers=headers,
            timeout=self.config.timeout,
        )
        response.raise_for_status()

        qwen_data = response.json()
        return {
            "id": qwen_data["request_id"],
            "object": "chat.completion",
            "created": 0,
            "model": provider_model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": qwen_data["output"]["text"],
                    },
                    "finish_reason": qwen_data["output"].get("finish_reason"),
                }
            ],
            "usage": qwen_data.get("usage", {}),
        }

    async def chat_completion_stream(
        self, provider_model_id: str, messages: list, params: dict
    ):
        raise NotImplementedError("Qwen streaming is not implemented yet.")
