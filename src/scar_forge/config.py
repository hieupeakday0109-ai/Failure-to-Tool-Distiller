"""Configuration and console encoding utilities for Scar-Forge."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def configure_utf8_console() -> None:
    """Configure UTF-8 encoding for environment and stdio streams safely."""
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError, ValueError):
                continue


class EngineConfig(BaseModel):
    """Runtime configuration for Scar-Forge storage, retrieval, LLM routing, and sandbox."""

    model_config = ConfigDict(validate_assignment=True)

    # Storage & paths
    data_dir: Path = Field(default=Path(".scar_forge"))

    # Retrieval & Scar token budget (strictly < 15 tokens per axiom, top 3-5 retrieval)
    top_k_scars: int = Field(default=4, ge=1, le=5)
    top_k_skills: int = Field(default=4, ge=1, le=5)
    max_scar_tokens: int = Field(default=14, ge=4, le=14)
    dedup_jaccard_threshold: float = Field(default=0.75, ge=0.1, le=1.0)

    # Dual-tier LLM routing & escalation
    provider: Literal["auto", "local", "ollama", "gemini", "openai"] = "auto"
    escalation_threshold: int = Field(default=2, ge=1, le=10)
    max_attempts: int = Field(default=5, ge=1, le=20)
    ollama_base_url: str = "http://localhost:11434"
    ollama_fast_model: str = "qwen2.5-coder:7b"
    gemini_fast_model: str = "gemini-2.5-flash"
    gemini_frontier_model: str = "gemini-2.5-pro"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_fast_model: str = "gpt-4o-mini"
    openai_frontier_model: str = "gpt-4o"
    llm_timeout_s: float = Field(default=30.0, gt=0.0)

    # Adversarial sandbox limits
    sandbox_timeout_s: float = Field(default=15.0, gt=0.1, le=60.0)
    sandbox_memory_mb: int = Field(default=256, ge=32, le=4096)
    sandbox_max_processes: int = Field(default=2, ge=1, le=16)
    enable_win32_job_object: bool = True

    # Dashboard settings
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = Field(default=8765, ge=1024, le=65535)

    @property
    def scar_db_path(self) -> Path:
        return self.data_dir / "scars.db"

    @property
    def scar_json_path(self) -> Path:
        return self.data_dir / "scars.json"

    @property
    def skills_dir(self) -> Path:
        return self.data_dir / "skills"

    @property
    def skills_manifest_path(self) -> Path:
        return self.skills_dir / "manifest.json"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    def ensure_dirs(self) -> None:
        """Create required persistence directories if they do not exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def get_gemini_api_key(self) -> str | None:
        """Read Gemini API key dynamically without storing secrets in model state."""
        return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    def get_openai_api_key(self) -> str | None:
        """Read OpenAI API key dynamically without storing secrets in model state."""
        return os.getenv("OPENAI_API_KEY")

    @classmethod
    def from_env(
        cls,
        data_dir: Path | str | None = None,
        **overrides: Any,
    ) -> EngineConfig:
        """Construct EngineConfig from environment defaults with explicit overrides."""
        env_values: dict[str, Any] = {}

        env_data_dir = os.getenv("SCAR_FORGE_DATA_DIR")
        if env_data_dir:
            env_values["data_dir"] = Path(env_data_dir)

        env_provider = os.getenv("SCAR_FORGE_PROVIDER")
        if env_provider:
            env_values["provider"] = env_provider.strip().lower()

        env_ollama = os.getenv("OLLAMA_HOST")
        if env_ollama:
            env_values["ollama_base_url"] = env_ollama.strip()

        env_esc = os.getenv("SCAR_FORGE_ESCALATION_THRESHOLD")
        if env_esc:
            env_values["escalation_threshold"] = int(env_esc)

        if data_dir is not None:
            env_values["data_dir"] = Path(data_dir)

        env_values.update(overrides)
        return cls(**env_values)
