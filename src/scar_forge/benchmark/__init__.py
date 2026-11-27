"""3-Epoch recursive learning benchmark suite and report generator."""

from __future__ import annotations

from scar_forge.benchmark.runner import BenchmarkRunner
from scar_forge.benchmark.tasks import BENCHMARK_TASKS, get_benchmark_tasks

__all__ = ["BENCHMARK_TASKS", "BenchmarkRunner", "get_benchmark_tasks"]
