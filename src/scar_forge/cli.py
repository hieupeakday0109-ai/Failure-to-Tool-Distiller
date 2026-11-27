"""Click CLI entrypoint for Scar-Forge (demo, run, benchmark, inspect, search, dashboard)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
from pydantic import ValidationError

from scar_forge.config import EngineConfig, configure_utf8_console

VALID_BENCHMARK_MODES = ("ablation", "standard", "variant")

_PROVIDER_CHOICES = click.Choice(
    ["auto", "local", "ollama", "gemini", "openai"], case_sensitive=False
)


def _resolve_config(
    data_dir: str | None = None,
    provider: str | None = None,
    **overrides: Any,
) -> EngineConfig:
    kw: dict[str, Any] = dict(overrides)
    if provider is not None:
        kw["provider"] = provider.lower()
    try:
        return EngineConfig.from_env(data_dir=data_dir, **kw)
    except (ValidationError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


def _load_runs(runs_dir: Path, limit: int = 50) -> list[dict[str, Any]]:
    if not runs_dir.is_dir():
        return []
    files = sorted(runs_dir.glob("run_*.json"), key=lambda p: p.name, reverse=True)
    out: list[dict[str, Any]] = []
    for path in files[:limit]:
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return out


@click.group(help="Scar-Forge: self-evolving AI agent with Scar memory and AST-forged Skills.")
def cli() -> None:
    """Configure UTF-8 console output before running any command."""
    configure_utf8_console()


@cli.command("demo")
@click.option("--provider", type=_PROVIDER_CHOICES, default="local", show_default=True)
@click.option("--data-dir", type=click.Path(path_type=Path), default=None)
@click.option("--json", "as_json", is_flag=True, help="Emit JSON output.")
def demo_cmd(provider: str, data_dir: Path | None, as_json: bool) -> None:
    """Run an end-to-end showcase demonstrating cold failure -> Scar -> Skill -> reuse."""
    from scar_forge.agent.engine import ScarForgeEngine
    from scar_forge.ui.console import render_run_metrics

    cfg = _resolve_config(str(data_dir) if data_dir else None, provider=provider)
    engine = ScarForgeEngine(config=cfg, provider=provider.lower())

    cold_1 = engine.run_task("Merge list of record dicts into DataFrame")
    cold_2 = engine.run_task(
        "Coerce webhook JSON payload with string booleans like 'false' and '0'"
    )
    reuse_1 = engine.run_task("Merge batch 2 list of record dicts into DataFrame")

    if as_json:
        click.echo(
            json.dumps(
                {
                    "runs": [
                        cold_1.model_dump(mode="json"),
                        cold_2.model_dump(mode="json"),
                        reuse_1.model_dump(mode="json"),
                    ],
                    "scars_count": len(engine.scar_store.list_scars()),
                    "skills_count": len(engine.skill_registry.list_skills()),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    render_run_metrics(cold_1, title="Demo 1: Cold Start (Pandas)")
    render_run_metrics(cold_2, title="Demo 2: Cold Start (Webhook Bool)")
    render_run_metrics(reuse_1, title="Demo 3: Hot-Reloaded Skill Reuse")


@cli.command("run")
@click.option("--task", required=True, type=str, help="Task prompt to execute.")
@click.option("--provider", type=_PROVIDER_CHOICES, default=None)
@click.option("--data-dir", type=click.Path(path_type=Path), default=None)
@click.option("--max-attempts", type=click.IntRange(1, 20), default=None)
@click.option("--json", "as_json", is_flag=True, help="Emit strict JSON RunMetrics.")
def run_cmd(
    task: str,
    provider: str | None,
    data_dir: Path | None,
    max_attempts: int | None,
    as_json: bool,
) -> None:
    """Execute a single task through Scar-Forge."""
    from scar_forge.agent.engine import ScarForgeEngine
    from scar_forge.ui.console import render_run_metrics

    overrides: dict[str, Any] = {}
    if max_attempts is not None:
        overrides["max_attempts"] = max_attempts
    cfg = _resolve_config(
        str(data_dir) if data_dir else None,
        provider=provider,
        **overrides,
    )
    engine = ScarForgeEngine(config=cfg, provider=cfg.provider)
    try:
        metrics = engine.run_task(task)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if as_json:
        click.echo(metrics.model_dump_json(indent=2))
    else:
        render_run_metrics(metrics)

    if not metrics.success:
        raise SystemExit(1)


@cli.command("benchmark")
@click.option("--provider", type=_PROVIDER_CHOICES, default="local", show_default=True)
@click.option(
    "--mode",
    type=click.Choice(sorted(VALID_BENCHMARK_MODES), case_sensitive=False),
    default="variant",
    show_default=True,
)
@click.option("--out", type=click.Path(path_type=Path), default=None)
@click.option("--md-out", type=click.Path(path_type=Path), default=None)
@click.option("--data-dir", type=click.Path(path_type=Path), default=None)
@click.option("--json", "as_json", is_flag=True, help="Emit JSON report to stdout.")
def benchmark_cmd(
    provider: str,
    mode: str,
    out: Path | None,
    md_out: Path | None,
    data_dir: Path | None,
    as_json: bool,
) -> None:
    """Run the 8-task 3-Epoch benchmark (Cold -> Scar-Guided -> Skill-Reuse)."""
    if out is not None and out.exists() and out.is_dir():
        raise click.ClickException(f"--out points to an existing directory: {out}")
    if md_out is not None and md_out.exists() and md_out.is_dir():
        raise click.ClickException(f"--md-out points to an existing directory: {md_out}")

    from scar_forge.benchmark.runner import BenchmarkRunner
    from scar_forge.ui.console import render_benchmark_summary

    cfg = _resolve_config(str(data_dir) if data_dir else None, provider=provider)
    runner = BenchmarkRunner(config=cfg, provider=provider.lower())
    _, payload = runner.run(mode=mode.lower())
    md_text = BenchmarkRunner.render_markdown(payload)

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if md_out is not None:
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(md_text, encoding="utf-8")

    if as_json:
        click.echo(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        render_benchmark_summary(payload)


@cli.command("inspect")
@click.option(
    "--kind",
    type=click.Choice(["all", "scars", "skills", "runs"], case_sensitive=False),
    default="all",
    show_default=True,
)
@click.option("--data-dir", type=click.Path(path_type=Path), default=None)
@click.option("--json", "as_json", is_flag=True, help="Emit JSON output.")
def inspect_cmd(kind: str, data_dir: Path | None, as_json: bool) -> None:
    """Inspect persisted Scars, forged Skills, and execution history."""
    from scar_forge.forge.registry import SkillRegistry
    from scar_forge.memory.scar_store import ScarStore
    from scar_forge.ui.console import render_inspect

    cfg = _resolve_config(str(data_dir) if data_dir else None)
    norm_kind = kind.lower()
    scars = ScarStore(cfg).list_scars() if norm_kind in {"all", "scars"} else []
    skills = SkillRegistry(cfg).list_skills() if norm_kind in {"all", "skills"} else []
    runs = _load_runs(cfg.runs_dir) if norm_kind in {"all", "runs"} else []

    if as_json:
        if norm_kind == "scars":
            payload: dict[str, Any] = {
                "scars": [s.model_dump(mode="json") for s in scars]
            }
        elif norm_kind == "skills":
            payload = {"skills": [sk.model_dump(mode="json") for sk in skills]}
        elif norm_kind == "runs":
            payload = {"runs": runs}
        else:
            payload = {
                "scars": [s.model_dump(mode="json") for s in scars],
                "skills": [sk.model_dump(mode="json") for sk in skills],
                "runs": runs,
            }
        click.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    render_inspect(scars, skills, runs, kind=norm_kind)


@cli.command("search")
@click.argument("query", type=str)
@click.option("--top-k", type=int, default=4, show_default=True)
@click.option(
    "--kind",
    type=click.Choice(["all", "scars", "skills"], case_sensitive=False),
    default="all",
    show_default=True,
)
@click.option("--data-dir", type=click.Path(path_type=Path), default=None)
@click.option("--json", "as_json", is_flag=True, help="Emit JSON output.")
def search_cmd(
    query: str,
    top_k: int,
    kind: str,
    data_dir: Path | None,
    as_json: bool,
) -> None:
    """Run bounded Top-K (clamped 1..5) BM25 retrieval over Scars and Skills."""
    from scar_forge.forge.registry import SkillRegistry
    from scar_forge.memory.retriever import BM25Retriever
    from scar_forge.memory.scar_store import ScarStore
    from scar_forge.ui.console import render_inspect

    cfg = _resolve_config(str(data_dir) if data_dir else None)
    retriever = BM25Retriever(cfg)
    clamped_k = retriever.clamp_top_k(top_k, cfg.top_k_scars)
    norm_kind = kind.lower()

    matched_scars = (
        retriever.retrieve_scars(query, ScarStore(cfg).list_scars(), top_k=clamped_k)
        if norm_kind in {"all", "scars"}
        else []
    )
    matched_skills = (
        retriever.retrieve_skills(query, SkillRegistry(cfg).list_skills(), top_k=clamped_k)
        if norm_kind in {"all", "skills"}
        else []
    )

    if as_json:
        click.echo(
            json.dumps(
                {
                    "query": query,
                    "top_k": clamped_k,
                    "scars": [s.model_dump(mode="json") for s in matched_scars],
                    "skills": [sk.model_dump(mode="json") for sk in matched_skills],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    render_inspect(matched_scars, matched_skills, [], kind=norm_kind)


@cli.command("dashboard")
@click.option("--host", type=str, default="127.0.0.1", show_default=True)
@click.option("--port", type=int, default=8765, show_default=True)
@click.option("--data-dir", type=click.Path(path_type=Path), default=None)
def dashboard_cmd(host: str, port: int, data_dir: Path | None) -> None:
    """Launch the local Flask web telemetry dashboard."""
    from scar_forge.ui.dashboard import create_app

    cfg = _resolve_config(
        str(data_dir) if data_dir else None,
        dashboard_host=host,
        dashboard_port=port,
    )
    app = create_app(cfg)
    app.run(host=host, port=port, debug=False, use_reloader=False)


def main() -> None:
    """Console script entrypoint."""
    cli()
