"""Dual-tier LLM adapter implementing small-to-large escalation & token telemetry."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

from scar_forge.config import EngineConfig
from scar_forge.llm.local_solver import LocalSolver
from scar_forge.llm.providers import (
    call_gemini_chat,
    call_ollama_chat,
    call_openai_chat,
)
from scar_forge.models import LLMResponse, ModelTier, TaskPurpose, TierUsage
from scar_forge.tokens import count_messages_tokens, count_tokens

VALID_PROVIDERS = frozenset({"auto", "local", "ollama", "gemini", "openai"})


class DualTierLLM:
    """Routes draft/repair loops to FAST_LOCAL and escalates on N failures or distillation."""

    def __init__(
        self,
        config: EngineConfig | None = None,
        *,
        provider: str | None = None,
    ) -> None:
        self.config = config or EngineConfig()
        eff_provider = (
            provider.strip().lower() if provider is not None else self.config.provider
        )
        if eff_provider not in VALID_PROVIDERS:
            raise ValueError(
                f"Unknown LLM provider {eff_provider!r}. Expected one of {sorted(VALID_PROVIDERS)}"
            )
        self.provider = eff_provider
        self._local = LocalSolver()
        self._lock = threading.Lock()
        self._usage: dict[ModelTier, TierUsage] = {
            ModelTier.FAST_LOCAL: TierUsage(tier=ModelTier.FAST_LOCAL),
            ModelTier.FRONTIER: TierUsage(tier=ModelTier.FRONTIER),
        }

    def select_tier(
        self, purpose: TaskPurpose, consecutive_failures: int
    ) -> tuple[ModelTier, bool]:
        """Determine target ModelTier and escalation flag from purpose and failures."""
        if consecutive_failures < 0:
            raise ValueError(
                f"consecutive_failures must be >= 0, got {consecutive_failures}"
            )
        always_frontier = purpose in {
            TaskPurpose.SKILL_DISTILLATION,
            TaskPurpose.REASONING_RECOVERY,
        }
        escalated = always_frontier or (
            consecutive_failures >= self.config.escalation_threshold
        )
        tier = ModelTier.FRONTIER if escalated else ModelTier.FAST_LOCAL
        return tier, escalated

    def _invoke_backend(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tier: ModelTier,
        purpose: TaskPurpose,
        consecutive_failures: int,
    ) -> tuple[str, str, str]:
        """Return (raw_content, provider_used, model_name)."""
        if self.provider == "local":
            content = self._local.solve(
                messages,
                purpose=purpose,
                consecutive_failures=consecutive_failures,
            )
            model_name = (
                self.config.gemini_frontier_model
                if tier == ModelTier.FRONTIER
                else self.config.ollama_fast_model
            )
            return content, "local", model_name

        if self.provider == "ollama":
            model = (
                self.config.ollama_fast_model
                if tier == ModelTier.FAST_LOCAL
                else (self.config.ollama_frontier_model or self.config.ollama_fast_model)
            )
            content = call_ollama_chat(messages, model=model, config=self.config)
            return content, "ollama", model

        if self.provider == "gemini":
            model = (
                self.config.gemini_fast_model
                if tier == ModelTier.FAST_LOCAL
                else self.config.gemini_frontier_model
            )
            content = call_gemini_chat(messages, model=model, config=self.config)
            return content, "gemini", model

        if self.provider == "openai":
            model = (
                self.config.openai_fast_model
                if tier == ModelTier.FAST_LOCAL
                else self.config.openai_frontier_model
            )
            content = call_openai_chat(messages, model=model, config=self.config)
            return content, "openai", model

        # "auto": probe configured local/cloud providers cleanly, falling back to LocalSolver
        if tier == ModelTier.FAST_LOCAL:
            try:
                content = call_ollama_chat(
                    messages, model=self.config.ollama_fast_model, config=self.config
                )
                return content, "ollama", self.config.ollama_fast_model
            except Exception:
                pass
        else:
            if self.config.get_gemini_api_key():
                try:
                    content = call_gemini_chat(
                        messages,
                        model=self.config.gemini_frontier_model,
                        config=self.config,
                    )
                    return content, "gemini", self.config.gemini_frontier_model
                except Exception:
                    pass
            if self.config.get_openai_api_key():
                try:
                    content = call_openai_chat(
                        messages,
                        model=self.config.openai_frontier_model,
                        config=self.config,
                    )
                    return content, "openai", self.config.openai_frontier_model
                except Exception:
                    pass

        content = self._local.solve(
            messages,
            purpose=purpose,
            consecutive_failures=consecutive_failures,
        )
        return content, "local", "local-deterministic"

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        purpose: TaskPurpose = TaskPurpose.INITIAL_DRAFT,
        consecutive_failures: int = 0,
    ) -> LLMResponse:
        """Complete messages using dual-tier routing and record exact token telemetry."""
        if not messages:
            raise ValueError("messages must be non-empty")

        tier, escalated = self.select_tier(purpose, consecutive_failures)
        prompt_tokens = count_messages_tokens(messages)
        t0 = time.perf_counter()

        content, provider_used, model_name = self._invoke_backend(
            messages,
            tier=tier,
            purpose=purpose,
            consecutive_failures=consecutive_failures,
        )
        duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        completion_tokens = count_tokens(content)
        total_tokens = prompt_tokens + completion_tokens

        with self._lock:
            cur = self._usage[tier]
            self._usage[tier] = TierUsage(
                tier=tier,
                calls=cur.calls + 1,
                prompt_tokens=cur.prompt_tokens + prompt_tokens,
                completion_tokens=cur.completion_tokens + completion_tokens,
                total_tokens=cur.total_tokens + total_tokens,
                duration_ms=round(cur.duration_ms + duration_ms, 2),
            )

        return LLMResponse(
            content=content,
            model_name=model_name,
            provider=provider_used,
            tier_used=tier,
            purpose=purpose,
            escalated=escalated,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            duration_ms=duration_ms,
        )

    @property
    def usage(self) -> dict[ModelTier, TierUsage]:
        """Return cumulative per-tier token accounting snapshot."""
        return self.get_usage()

    def get_usage(self) -> dict[ModelTier, TierUsage]:
        """Return a deep copy of cumulative TierUsage per ModelTier."""
        with self._lock:
            return {k: v.model_copy() for k, v in self._usage.items()}

    def reset_usage(self) -> None:
        """Reset cumulative usage counters."""
        with self._lock:
            self._usage = {
                ModelTier.FAST_LOCAL: TierUsage(tier=ModelTier.FAST_LOCAL),
                ModelTier.FRONTIER: TierUsage(tier=ModelTier.FRONTIER),
            }
