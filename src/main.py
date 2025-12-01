from __future__ import annotations

import json
import logging

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from .config import load_config
from .models import ChatCompletionRequest
from .router import ModelRouter

config = load_config()
router = ModelRouter(config)
app = FastAPI(title="OpenAI-Compatible API Router")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def verify_api_key(authorization: str | None = Header(default=None)):
    if config.server.api_keys:
        if not authorization:
            raise HTTPException(status_code=401, detail="Missing authorization header")

        token = authorization.replace("Bearer ", "")
        if token not in config.server.api_keys:
            raise HTTPException(status_code=401, detail="Invalid API key")


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest, authorization: str | None = Header(default=None)
):
    if config.server.api_keys:
        await verify_api_key(authorization)

    try:
        provider, provider_model_id = router.resolve_model(request.model)

        params = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            "stop": request.stop,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
        }
        params = {k: v for k, v in params.items() if v is not None}

        messages = [msg.model_dump() for msg in request.messages]

        if request.stream:
            async def generate():
                try:
                    async for chunk in provider.chat_completion_stream(
                        provider_model_id, messages, params
                    ):
                        yield chunk
                except Exception as e:  # pragma: no cover - streaming fallback
                    logger.error(f"Streaming error: {e}")
                    error_chunk = {
                        "error": {
                            "message": str(e),
                            "type": "server_error",
                        }
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            response = await provider.chat_completion(provider_model_id, messages, params)
            return response

    except ValueError as e:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": str(e),
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
        )
    except Exception as e:  # pragma: no cover - top level safety
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "Internal server error",
                    "type": "server_error",
                    "code": "internal_error",
                }
            },
        )


@app.get("/v1/models")
async def list_models():
    models = router.list_models()
    return {"object": "list", "data": models}


@app.get("/v1/models/all")
async def list_all_configured_models():
    models = router.list_configured_models()
    return {"object": "list", "data": models}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "providers": {
            name: "enabled" if cfg.enabled else "disabled"
            for name, cfg in config.providers.items()
        },
    }


@app.on_event("shutdown")
async def shutdown():
    await router.close()
