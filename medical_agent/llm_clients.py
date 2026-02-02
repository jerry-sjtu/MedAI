from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any
from urllib import request

from .utils.env import load_dotenv


@dataclass
class LLMConfig:
    model: str
    api_key: str
    base_url: str
    app_name: str
    app_url: str


class BaseLLMClient:
    def chat(self, messages: list[dict[str, str]]) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class BaseEmbeddingClient:
    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - interface
        raise NotImplementedError


class OpenRouterClient(BaseLLMClient):
    def __init__(self, config: LLMConfig, timeout_s: int = 60) -> None:
        self.config = config
        self.timeout_s = timeout_s

    def chat(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0,
        }
        response = _post_json(
            url=f"{self.config.base_url}/chat/completions",
            api_key=self.config.api_key,
            payload=payload,
            timeout_s=self.timeout_s,
            extra_headers={
                "HTTP-Referer": self.config.app_url,
                "X-Title": self.config.app_name,
            },
        )
        return response["choices"][0]["message"]["content"]


class OpenRouterEmbeddingClient(BaseEmbeddingClient):
    def __init__(self, config: LLMConfig, timeout_s: int = 60) -> None:
        self.config = config
        self.timeout_s = timeout_s

    def embed(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "model": self.config.model,
            "input": texts,
        }
        response = _post_json(
            url=f"{self.config.base_url}/embeddings",
            api_key=self.config.api_key,
            payload=payload,
            timeout_s=self.timeout_s,
            extra_headers={
                "HTTP-Referer": self.config.app_url,
                "X-Title": self.config.app_name,
            },
        )
        return [item["embedding"] for item in response["data"]]


def build_llm_client(
    model: str,
    env_path: str = ".env",
    base_url: str | None = None,
    app_name: str = "MedAI",
    app_url: str = "https://example.com",
) -> BaseLLMClient:
    load_dotenv(env_path)
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("Missing API key: OPENROUTER_API_KEY")

    config = LLMConfig(
        model=model,
        api_key=api_key,
        base_url=base_url or "https://openrouter.ai/api/v1",
        app_name=app_name,
        app_url=app_url,
    )
    return OpenRouterClient(config)


def build_embedding_client(
    model: str,
    env_path: str = ".env",
    base_url: str | None = None,
    app_name: str = "MedAI",
    app_url: str = "https://example.com",
) -> BaseEmbeddingClient:
    load_dotenv(env_path)
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("Missing API key: OPENROUTER_API_KEY")

    config = LLMConfig(
        model=model,
        api_key=api_key,
        base_url=base_url or "https://openrouter.ai/api/v1",
        app_name=app_name,
        app_url=app_url,
    )
    return OpenRouterEmbeddingClient(config)


def _post_json(
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout_s: int,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    if extra_headers:
        headers.update(extra_headers)

    req = request.Request(url, data=data, headers=headers, method="POST")
    with request.urlopen(req, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))
