from __future__ import annotations

import json
import time

from .base import BaseProvider


class OllamaProvider(BaseProvider):
    def _to_openai_format(self, ollama_response: dict, model: str) -> dict:
        return {
            "id": f"ollama-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": ollama_response["message"]["content"],
                    },
                    "finish_reason": "stop" if ollama_response.get("done") else None,
                }
            ],
            "usage": {
                "prompt_tokens": ollama_response.get("prompt_eval_count", 0),
                "completion_tokens": ollama_response.get("eval_count", 0),
                "total_tokens": (
                    ollama_response.get("prompt_eval_count", 0)
                    + ollama_response.get("eval_count", 0)
                ),
            },
        }

    async def chat_completion(self, provider_model_id: str, messages: list, params: dict):
        payload: dict = {
            "model": provider_model_id,
            "messages": messages,
            "stream": False,
        }

        if "temperature" in params or "top_p" in params:
            payload["options"] = {}
            if "temperature" in params:
                payload["options"]["temperature"] = params["temperature"]
            if "top_p" in params:
                payload["options"]["top_p"] = params["top_p"]

        response = await self.client.post(
            f"{self.config.base_url}/api/chat",
            json=payload,
            timeout=self.config.timeout,
        )
        response.raise_for_status()

        return self._to_openai_format(response.json(), provider_model_id)

    async def chat_completion_stream(
        self, provider_model_id: str, messages: list, params: dict
    ):
        payload: dict = {
            "model": provider_model_id,
            "messages": messages,
            "stream": True,
        }

        if "temperature" in params or "top_p" in params:
            payload["options"] = {}
            if "temperature" in params:
                payload["options"]["temperature"] = params["temperature"]
            if "top_p" in params:
                payload["options"]["top_p"] = params["top_p"]

        async with self.client.stream(
            "POST",
            f"{self.config.base_url}/api/chat",
            json=payload,
            timeout=self.config.timeout,
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line:
                    continue

                data = json.loads(line)

                if data.get("message"):
                    chunk = {
                        "id": f"ollama-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": provider_model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": data["message"].get("content", ""),
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                if data.get("done"):
                    yield "data: [DONE]\n\n"
