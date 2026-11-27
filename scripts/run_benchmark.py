"""Standalone 3-Epoch Benchmark runner script for Scar-Forge."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from scar_forge.benchmark.runner import BenchmarkRunner
from scar_forge.config import EngineConfig, configure_utf8_console
from scar_forge.ui.console import render_benchmark_summary


def main() -> int:
    configure_utf8_console()
    parser = argparse.ArgumentParser(description="Run Scar-Forge 3-Epoch Benchmark")
    parser.add_argument(
        "--provider",
        default="local",
        choices=["auto", "local", "ollama", "gemini", "openai"],
    )
    parser.add_argument("--mode", default="variant", choices=["variant", "standard", "ablation"])
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    args = parser.parse_args()

    if args.data_dir is not None:
        cfg = EngineConfig(data_dir=args.data_dir, provider=args.provider)
        runner = BenchmarkRunner(config=cfg, provider=args.provider)
        _, payload = runner.run(mode=args.mode)
    else:
        with tempfile.TemporaryDirectory(prefix="scar_forge_bench_") as tmp_dir:
            cfg = EngineConfig(data_dir=Path(tmp_dir), provider=args.provider)
            runner = BenchmarkRunner(config=cfg, provider=args.provider)
            _, payload = runner.run(mode=args.mode)

    render_benchmark_summary(payload)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(BenchmarkRunner.render_markdown(payload), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
