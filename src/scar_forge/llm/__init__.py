"""Dual-tier LLM routing, local deterministic solver, and provider backends."""

from __future__ import annotations

from scar_forge.llm.adapter import DualTierLLM
from scar_forge.llm.local_solver import LocalSolver

__all__ = ["DualTierLLM", "LocalSolver"]
