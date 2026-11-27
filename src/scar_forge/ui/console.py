"""Rich terminal UI formatters for Scar-Forge CLI commands."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from scar_forge.models import RunMetrics, ScarRecord, SkillRecord


def _console() -> Console:
    return Console(soft_wrap=True)


def render_run_metrics(metrics: RunMetrics, *, title: str = "Scar-Forge Run") -> None:
    """Render a RunMetrics telemetry summary panel to stdout."""
    con = _console()
    status = "[bold green]PASS[/bold green]" if metrics.success else "[bold red]FAIL[/bold red]"
    body = (
        f"Status: {status} | Task: [cyan]{metrics.task_id}[/cyan]\n"
        f"Attempts: {metrics.attempts} | Errors: {metrics.errors_count} "
        f"(Repeated: {metrics.repeated_errors_count}) | Escalated: {metrics.escalated}\n"
        f"Tokens: {metrics.total_tokens} (Prompt: {metrics.prompt_tokens}, "
        f"Completion: {metrics.completion_tokens}) | Duration: {metrics.duration_ms:.1f} ms\n"
        f"Scars Created: {len(metrics.scars_created)} | "
        f"Scars Prevented: {len(metrics.scars_prevented)} | "
        f"Skill Forged: {metrics.skill_forged or '-'} | "
        f"Skill Reused: {metrics.skill_reused or '-'}"
    )
    con.print(Panel(body, title=title, border_style="cyan"))


def render_inspect(
    scars: list[ScarRecord],
    skills: list[SkillRecord],
    runs: list[dict[str, Any]],
    *,
    kind: str = "all",
) -> None:
    """Render Scar Memory, Skill Registry, and Run History tables."""
    con = _console()

    if kind in {"all", "scars"}:
        tbl = Table(title=f"Scar Memory ({len(scars)})", show_lines=False)
        tbl.add_column("ID", style="cyan")
        tbl.add_column("Axiom (<15 tokens)", style="yellow")
        tbl.add_column("Tok", justify="right")
        tbl.add_column("Seen", justify="right")
        tbl.add_column("Prevented", justify="right", style="green")
        tbl.add_column("Tags")
        for s in scars:
            tbl.add_row(
                s.scar_id,
                s.axiom,
                str(s.token_count),
                str(s.occurrence_count),
                str(s.prevented_count),
                ", ".join(s.tags),
            )
        con.print(tbl)

    if kind in {"all", "skills"}:
        tbl = Table(title=f"Forged Skills ({len(skills)})", show_lines=False)
        tbl.add_column("Skill", style="green")
        tbl.add_column("Ver", justify="right")
        tbl.add_column("Reuses", justify="right")
        tbl.add_column("AST/SB")
        tbl.add_column("Docstring")
        for sk in skills:
            gates = f"{'AST' if sk.ast_verified else '-'}/{'SB' if sk.sandbox_verified else '-'}"
            tbl.add_row(
                sk.skill_name,
                f"v{sk.version}",
                str(sk.reuse_count),
                gates,
                sk.docstring,
            )
        con.print(tbl)

    if kind in {"all", "runs"}:
        tbl = Table(title=f"Recent Runs ({len(runs)})", show_lines=False)
        tbl.add_column("Run ID", style="cyan")
        tbl.add_column("Task")
        tbl.add_column("OK")
        tbl.add_column("Tokens", justify="right")
        tbl.add_column("Errors", justify="right")
        tbl.add_column("Duration (ms)", justify="right")
        for r in runs[:20]:
            tbl.add_row(
                str(r.get("run_id", "")),
                str(r.get("task_id", "")),
                "PASS" if r.get("success") else "FAIL",
                str(r.get("total_tokens", 0)),
                str(r.get("errors_count", 0)),
                f"{float(r.get('duration_ms', 0.0)):.1f}",
            )
        con.print(tbl)


def render_benchmark_summary(payload: dict[str, Any]) -> None:
    """Render 3-Epoch Benchmark summary table via Rich."""
    con = _console()
    tbl = Table(
        title=(
            f"Scar-Forge 3-Epoch Benchmark "
            f"(mode={payload.get('mode', 'variant')}, provider={payload.get('provider', 'local')})"
        )
    )
    tbl.add_column("Epoch", style="cyan")
    tbl.add_column("Passed", justify="right")
    tbl.add_column("Accuracy", justify="right")
    tbl.add_column("Total Tokens", justify="right", style="yellow")
    tbl.add_column("Errors", justify="right")
    tbl.add_column("Repeated Err", justify="right")
    tbl.add_column("Duration (ms)", justify="right")
    tbl.add_column("Skill Reuses", justify="right", style="green")

    for label, key in (
        ("Epoch 1: Cold Start", "epoch_1"),
        ("Epoch 2: Scar-Guided", "epoch_2"),
        ("Epoch 3: Skill-Reuse", "epoch_3"),
    ):
        ep = payload[key]
        tbl.add_row(
            label,
            f"{ep['passed_tasks']}/{ep['total_tasks']}",
            f"{ep['accuracy'] * 100:.1f}%",
            f"{ep['total_tokens']:,}",
            str(ep["total_errors"]),
            str(ep["repeated_errors"]),
            f"{ep['total_duration_ms']:.1f}",
            str(ep["skills_reused_count"]),
        )
    con.print(tbl)
    tok_red = float(payload.get("token_reduction_epoch3_vs_epoch1", 0.0)) * 100.0
    err_red = float(payload.get("error_reduction_epoch2_vs_epoch1", 0.0)) * 100.0
    speedup = float(payload.get("speedup_epoch3_vs_epoch1", 1.0))
    con.print(
        f"[bold green]Token Reduction (E3 vs E1): {tok_red:.1f}%[/bold green] | "
        f"[bold cyan]Repeated Error Drop (E2 vs E1): {err_red:.1f}%[/bold cyan] | "
        f"[bold yellow]Speedup (E3 vs E1): {speedup:.2f}x[/bold yellow]"
    )
