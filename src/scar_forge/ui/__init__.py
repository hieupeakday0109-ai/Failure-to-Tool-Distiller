"""Rich CLI console renderer and local Flask telemetry dashboard."""

from __future__ import annotations

from scar_forge.ui.console import render_benchmark_summary, render_inspect, render_run_metrics
from scar_forge.ui.dashboard import create_app

__all__ = [
    "create_app",
    "render_benchmark_summary",
    "render_inspect",
    "render_run_metrics",
]
