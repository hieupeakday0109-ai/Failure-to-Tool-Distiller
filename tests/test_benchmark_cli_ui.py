"""Unit & integration tests for BenchmarkRunner, CLI, and Flask dashboard (M4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from scar_forge.benchmark.runner import BenchmarkRunner
from scar_forge.cli import cli
from scar_forge.config import EngineConfig
from scar_forge.ui.dashboard import create_app


class TestBenchmarkRunnerUnit:
    def test_invalid_mode_raises(self, tmp_path: Path) -> None:
        runner = BenchmarkRunner(EngineConfig(data_dir=tmp_path), provider="local")
        with pytest.raises(ValueError):
            runner.run(mode="nonexistent")

    def test_standard_mode_and_markdown_rendering(self, tmp_path: Path) -> None:
        from scar_forge.benchmark.tasks import get_benchmark_tasks

        tasks = get_benchmark_tasks()[:2]
        runner = BenchmarkRunner(
            EngineConfig(data_dir=tmp_path), provider="local", tasks=tasks
        )
        report, payload = runner.run(mode="standard")
        assert len(report.epochs) == 3
        assert payload["epoch_1"]["total_tasks"] == 2
        assert payload["epoch_2"]["skill_calls"] == 0
        assert payload["epoch_3"]["skills_reused_count"] == 2
        assert payload["token_reduction_epoch3_vs_epoch1"] > 0.5

        md = BenchmarkRunner.render_markdown(payload)
        assert "| Epoch |" in md
        assert "Epoch 1" in md
        assert "Epoch 3" in md


class TestCLIDirect:
    def test_cli_run_inspect_search_and_error_paths(self, tmp_path: Path) -> None:
        runner = CliRunner()
        res = runner.invoke(
            cli,
            [
                "run",
                "--task",
                "Sort semantic version tags numerically with prerelease ordering",
                "--provider",
                "local",
                "--data-dir",
                str(tmp_path),
                "--json",
            ],
        )
        assert res.exit_code == 0, res.output
        data = json.loads(res.output)
        assert data["success"] is True

        insp = runner.invoke(
            cli,
            ["inspect", "--kind", "runs", "--data-dir", str(tmp_path), "--json"],
        )
        assert insp.exit_code == 0
        assert len(json.loads(insp.output)["runs"]) >= 1

        srch = runner.invoke(
            cli,
            ["search", "semver", "--top-k", "10", "--data-dir", str(tmp_path), "--json"],
        )
        assert srch.exit_code == 0
        srch_data = json.loads(srch.output)
        assert srch_data["top_k"] == 5

        bad_empty = runner.invoke(
            cli,
            ["run", "--task", "   ", "--provider", "local", "--data-dir", str(tmp_path)],
        )
        assert bad_empty.exit_code != 0


class TestDashboardUnit:
    def test_dashboard_routes_and_404(self, tmp_path: Path) -> None:
        app = create_app(EngineConfig(data_dir=tmp_path))
        app.config["TESTING"] = True
        with app.test_client() as client:
            r_home = client.get("/")
            assert r_home.status_code == 200
            assert "Scar-Forge" in r_home.get_data(as_text=True)

            r_state = client.get("/api/state")
            assert r_state.status_code == 200
            assert r_state.get_json()["scars_count"] == 0

            r_runs = client.get("/api/runs")
            assert r_runs.status_code == 200
            assert r_runs.get_json()["runs"] == []

            r_bench = client.get("/api/benchmark")
            assert r_bench.status_code == 200
            assert r_bench.get_json()["epoch_1"] is None

            r_404 = client.get("/api/unknown")
            assert r_404.status_code == 404
