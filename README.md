# llm-router

OpenAI-compatible router built with FastAPI to consolidate multiple LLM providers behind a single `/v1` API surface.

## Features
- Async routing for chat completions with OpenAI-compatible request/response shapes
- Configuration-driven provider/model mapping via YAML
- Streaming and non-streaming chat completions
- Health and model listing endpoints
- Dockerfile and Compose setup for local deployment

## Getting Started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure providers in `config/providers.yaml` (environment variables are substituted automatically).
3. Run the API:
   ```bash
   uvicorn src.main:app --host 0.0.0.0 --port 8000
   ```
4. Example request:
   ```bash
   curl -X POST http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "gpt-4",
       "messages": [{"role": "user", "content": "Hello!"}]
     }'
   ```

## Docker
Build and run with Docker Compose:
```bash
docker-compose up --build
```
