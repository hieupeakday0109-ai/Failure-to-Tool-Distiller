"""Flask local observability dashboard for Scar Memory, Forged Skills, and Benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify

from scar_forge.config import EngineConfig
from scar_forge.forge.registry import SkillRegistry
from scar_forge.memory.scar_store import ScarStore

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Scar-Forge | Self-Evolving AI Engine</title>
  <style>
    :root {
      --bg: #0b0f17;
      --panel: #131a26;
      --border: #243044;
      --text: #e6edf7;
      --muted: #8ea0b8;
      --accent: #38bdf8;
      --green: #22c55e;
      --amber: #f59e0b;
      --rose: #f43f5e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }
    header {
      padding: 20px 28px;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: #0e141f;
    }
    header h1 { margin: 0; font-size: 1.25rem; letter-spacing: 0.02em; }
    header span.badge {
      background: rgba(56, 189, 248, 0.14);
      color: var(--accent);
      border: 1px solid rgba(56, 189, 248, 0.35);
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.8rem;
    }
    main { max-width: 1200px; margin: 0 auto; padding: 24px; display: grid; gap: 20px; }
    .kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px 18px;
    }
    .kpi-label { color: var(--muted); font-size: 0.8rem; text-transform: uppercase; }
    .kpi-val { font-size: 1.7rem; font-weight: 700; margin-top: 4px; }
    h2 { margin: 0 0 12px 0; font-size: 1.05rem; color: var(--accent); }
    table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
    th, td { padding: 9px 10px; border-bottom: 1px solid var(--border); text-align: left; }
    th { color: var(--muted); font-weight: 600; font-size: 0.8rem; text-transform: uppercase; }
    code {
      background: #0b1019;
      padding: 2px 6px;
      border-radius: 4px;
      color: #fde68a;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.84rem;
    }
    .pill {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 0.75rem;
      background: rgba(34, 197, 94, 0.16);
      color: var(--green);
      border: 1px solid rgba(34, 197, 94, 0.35);
    }
    .bar-row {
      display: grid;
      grid-template-columns: 180px 1fr 90px;
      gap: 12px;
      align-items: center;
      margin: 8px 0;
    }
    .bar-track {
      background: #0a0f18;
      border-radius: 6px;
      height: 14px;
      overflow: hidden;
      border: 1px solid var(--border);
    }
    .bar-fill { height: 100%; background: linear-gradient(90deg, var(--accent), var(--green)); }
    .muted { color: var(--muted); font-size: 0.85rem; }
  </style>
</head>
<body>
  <header>
    <h1>Scar-Forge Control Plane</h1>
    <span class="badge" id="status-badge">Loading telemetry...</span>
  </header>
  <main>
    <section class="kpis" id="kpi-grid"></section>
    <section class="card">
      <h2>3-Epoch Learning Progression</h2>
      <div id="bench-box" class="muted">Loading benchmark data...</div>
    </section>
    <section class="card">
      <h2>Telegraphic Scar Memory (&lt; 15 Tokens)</h2>
      <table>
        <thead><tr><th>ID</th><th>Axiom</th><th>Tokens</th><th>Seen</th><th>Prevented</th><th>Tags</th></tr></thead>
        <tbody id="scars-body"></tbody>
      </table>
    </section>
    <section class="card">
      <h2>AST-Verified Forged Skills</h2>
      <table>
        <thead><tr><th>Name</th><th>Version</th><th>Reuses</th><th>Gates</th><th>Docstring</th></tr></thead>
        <tbody id="skills-body"></tbody>
      </table>
    </section>
  </main>
  <script>
    async function refresh() {
      const [st, sc, sk, bm] = await Promise.all([
        fetch('/api/state').then(r => r.json()),
        fetch('/api/scars').then(r => r.json()),
        fetch('/api/skills').then(r => r.json()),
        fetch('/api/benchmark').then(r => r.json())
      ]);
      document.getElementById('status-badge').textContent = 'Engine Active | ' + st.data_dir;
      document.getElementById('kpi-grid').innerHTML = [
        ['Active Scars', st.scars_count],
        ['Forged Skills', st.skills_count],
        ['Errors Prevented', st.total_prevented],
        ['Skill Reuses', st.total_skill_reuses],
        ['Recorded Runs', st.runs_count]
      ].map(([k, v]) => (
        `<div class="card"><div class="kpi-label">${k}</div>` +
        `<div class="kpi-val">${v}</div></div>`
      )).join('');

      document.getElementById('scars-body').innerHTML = (sc.scars || []).map(s => (
        `<tr><td><code>${s.scar_id}</code></td>` +
        `<td><code>${s.axiom}</code></td>` +
        `<td>${s.token_count}</td>` +
        `<td>${s.occurrence_count}</td>` +
        `<td><span class="pill">${s.prevented_count}</span></td>` +
        `<td class="muted">${(s.tags || []).join(', ')}</td></tr>`
      )).join('') || '<tr><td colspan="6" class="muted">No Scars recorded yet.</td></tr>';

      document.getElementById('skills-body').innerHTML = (sk.skills || []).map(x => (
        `<tr><td><code>${x.skill_name}</code></td>` +
        `<td>v${x.version}</td>` +
        `<td><span class="pill">${x.reuse_count}</span></td>` +
        `<td>AST+Sandbox</td>` +
        `<td>${x.docstring}</td></tr>`
      )).join('') || '<tr><td colspan="5" class="muted">No Skills forged yet.</td></tr>';

      if (bm && bm.epoch_1 && bm.epoch_3) {
        const maxTok = Math.max(
          bm.epoch_1.total_tokens || 1,
          bm.epoch_2.total_tokens || 1,
          bm.epoch_3.total_tokens || 1
        );
        const rows = [
          ['Epoch 1 (Cold Start)', bm.epoch_1],
          ['Epoch 2 (Scar-Guided)', bm.epoch_2],
          ['Epoch 3 (Skill-Reuse)', bm.epoch_3]
        ].map(([lbl, ep]) => {
          const pct = Math.round((ep.total_tokens / maxTok) * 100);
          return (
            `<div class="bar-row"><div>${lbl}</div>` +
            `<div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>` +
            `<div>${ep.total_tokens} tok</div></div>`
          );
        }).join('');
        const redPct = (Number(bm.token_reduction_epoch3_vs_epoch1 || 0) * 100).toFixed(1);
        document.getElementById('bench-box').innerHTML = (
          rows + `<p class="pill">Epoch 3 Token Reduction: ${redPct}%</p>`
        );
      } else {
        document.getElementById('bench-box').textContent = (
          'Run `scar-forge benchmark` to populate 3-Epoch telemetry.'
        );
      }
    }
    refresh();
  </script>
</body>
</html>
"""


def _read_runs(runs_dir: Path, limit: int = 50) -> list[dict[str, Any]]:
    if not runs_dir.is_dir():
        return []
    files = sorted(runs_dir.glob("run_*.json"), key=lambda p: p.name, reverse=True)
    items: list[dict[str, Any]] = []
    for path in files[:limit]:
        try:
            items.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return items


def create_app(config: EngineConfig | None = None) -> Flask:
    """Create the Flask observability application bound to config.data_dir."""
    cfg = config or EngineConfig()
    cfg.ensure_dirs()
    app = Flask(__name__)

    @app.get("/")
    def index() -> Response:
        return Response(_DASHBOARD_HTML, status=200, mimetype="text/html; charset=utf-8")

    @app.get("/api/state")
    def api_state() -> Response:
        scars = ScarStore(cfg).list_scars()
        skills = SkillRegistry(cfg).list_skills()
        runs = _read_runs(cfg.runs_dir, limit=200)
        bench_path = cfg.data_dir / "latest_benchmark.json"
        return jsonify(
            {
                "status": "ok",
                "data_dir": str(cfg.data_dir),
                "scars_count": len(scars),
                "skills_count": len(skills),
                "runs_count": len(runs),
                "total_prevented": sum(s.prevented_count for s in scars),
                "total_skill_reuses": sum(sk.reuse_count for sk in skills),
                "has_benchmark": bench_path.is_file(),
            }
        )

    @app.get("/api/scars")
    def api_scars() -> Response:
        scars = ScarStore(cfg).list_scars()
        return jsonify({"scars": [s.model_dump(mode="json") for s in scars]})

    @app.get("/api/skills")
    def api_skills() -> Response:
        skills = SkillRegistry(cfg).list_skills()
        return jsonify({"skills": [sk.model_dump(mode="json") for sk in skills]})

    @app.get("/api/runs")
    def api_runs() -> Response:
        return jsonify({"runs": _read_runs(cfg.runs_dir)})

    @app.get("/api/benchmark")
    def api_benchmark() -> Response:
        bench_path = cfg.data_dir / "latest_benchmark.json"
        if bench_path.is_file():
            try:
                data = json.loads(bench_path.read_text(encoding="utf-8"))
                return jsonify(data)
            except (OSError, json.JSONDecodeError):
                pass
        return jsonify(
            {
                "status": "no_benchmark_yet",
                "epochs": [],
                "epoch_1": None,
                "epoch_2": None,
                "epoch_3": None,
            }
        )

    @app.errorhandler(404)
    def not_found(_err: Any) -> tuple[Response, int]:
        return jsonify({"error": "not_found"}), 404

    return app
