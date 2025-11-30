from __future__ import annotations

from typing import Dict, List, Optional
import os
import re

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: str
    provider_model_id: str
    aliases: List[str] = Field(default_factory=list)


class ProviderConfig(BaseModel):
    type: str
    base_url: str
    api_key: Optional[str] = None
    enabled: bool = True
    timeout: int = 60
    max_retries: int = 2
    models: List[ModelConfig]


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    api_keys: List[str] = Field(default_factory=list)


class Config(BaseModel):
    server: ServerConfig
    providers: Dict[str, ProviderConfig]


def load_config(path: str = "config/providers.yaml") -> Config:
    """Load configuration from YAML, substituting environment variables."""
    with open(path) as f:
        raw = f.read()

    def replace_env(match: re.Match[str]) -> str:
        var_name = match.group(1)
        return os.getenv(var_name, "")

    raw = re.sub(r"\$\{(\w+)\}", replace_env, raw)
    data = yaml.safe_load(raw)

    config = Config(**data)
    _validate_no_duplicate_models(config)
    return config


def _validate_no_duplicate_models(config: Config) -> None:
    """Ensure no model name conflicts across enabled providers."""
    seen: Dict[str, str] = {}
    for provider_name, provider in config.providers.items():
        if not provider.enabled:
            continue
        for model in provider.models:
            all_names = [model.name] + list(model.aliases)
            for name in all_names:
                if name in seen:
                    raise ValueError(
                        f"Duplicate model name '{name}' in providers "
                        f"'{seen[name]}' and '{provider_name}'"
                    )
                seen[name] = provider_name
