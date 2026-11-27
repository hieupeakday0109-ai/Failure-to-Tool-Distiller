"""External LLM provider HTTP/SDK callers (Ollama, Gemini, OpenAI)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from scar_forge.config import EngineConfig


def call_ollama_chat(
    messages: Sequence[Mapping[str, Any]],
    *,
    model: str,
    config: EngineConfig,
) -> str:
    """Invoke a local Ollama /api/chat endpoint with fast-fail connect timeout."""
    base = config.ollama_base_url.rstrip("/")
    url = f"{base}/api/chat"
    timeout = httpx.Timeout(
        timeout=config.llm_timeout_s,
        connect=min(2.5, config.llm_timeout_s),
    )
    payload = {
        "model": model,
        "messages": [
            {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
            for m in messages
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1},
    }
    resp = httpx.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    message = data.get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"Ollama returned empty content: {data!r}")
    return content.strip()


def call_gemini_chat(
    messages: Sequence[Mapping[str, Any]],
    *,
    model: str,
    config: EngineConfig,
) -> str:
    """Invoke Google Gemini via google-genai SDK."""
    api_key = config.get_gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY is not configured")

    from google import genai  # type: ignore[import-untyped]
    from google.genai import types  # type: ignore[import-untyped]

    system_parts = [
        str(m.get("content", ""))
        for m in messages
        if str(m.get("role", "")).lower() == "system"
    ]
    user_parts = [
        str(m.get("content", ""))
        for m in messages
        if str(m.get("role", "")).lower() != "system"
    ]
    timeout_ms = max(1000, int(config.llm_timeout_s * 1000))
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=timeout_ms),
    )
    cfg_kwargs: dict[str, Any] = {
        "temperature": 0.1,
        "response_mime_type": "application/json",
    }
    if system_parts:
        cfg_kwargs["system_instruction"] = "\n".join(system_parts)

    response = client.models.generate_content(
        model=model,
        contents="\n\n".join(user_parts),
        config=types.GenerateContentConfig(**cfg_kwargs),
    )
    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Gemini returned empty text response")
    return text.strip()


def call_openai_chat(
    messages: Sequence[Mapping[str, Any]],
    *,
    model: str,
    config: EngineConfig,
) -> str:
    """Invoke OpenAI-compatible /chat/completions endpoint."""
    api_key = config.get_openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    base = config.openai_base_url.rstrip("/")
    url = f"{base}/chat/completions"
    timeout = httpx.Timeout(
        timeout=config.llm_timeout_s,
        connect=min(5.0, config.llm_timeout_s),
    )
    payload = {
        "model": model,
        "messages": [
            {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
            for m in messages
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = httpx.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenAI returned no choices: {data!r}")
    content = (choices[0].get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"OpenAI returned empty message content: {data!r}")
    return content.strip()
