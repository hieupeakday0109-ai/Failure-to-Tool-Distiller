"""Opaque-box E2E test infrastructure and contract-resilient fixtures for Scar-Forge.

All ``scar_forge.*`` imports are intentionally lazy (inside fixtures/helpers) so
``pytest --collect-only`` succeeds at every milestone (M1-M5) without top-level
import failures.
"""

from __future__ import annotations

import inspect
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def pytest_configure(config: Any) -> None:
    """Register E2E, tier, milestone, and slow markers without mutating pyproject.toml."""
    markers = [
        "e2e: End-to-end opaque-box test suite",
        "tier1: Tier 1 isolated happy-path feature tests",
        "tier2: Tier 2 boundary, corner, error, and rollback tests",
        "tier3: Tier 3 cross-feature pairwise combination tests",
        "tier4: Tier 4 real-world end-to-end workflow scenarios",
        "m1: Milestone 1 foundation, schemas, tokens, protocol, distiller",
        "m2: Milestone 2 scar memory, BM25 retriever, adversarial sandbox",
        "m3: Milestone 3 AST guard, skill registry hot-reload, dual-tier LLM, engine",
        "m4: Milestone 4 3-epoch benchmark, CLI, Flask dashboard",
        "m5: Milestone 5 README diagrams, release gates",
        "slow: Slow subprocess timeout/memory/benchmark tests",
        "live_llm: Live external LLM API tests (opt-in)",
    ]
    for marker in markers:
        config.addinivalue_line("markers", marker)


def resolve_python_exe() -> Path:
    """Resolve the project ``.venv`` Python executable, falling back to ``sys.executable``."""
    venv_py = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        return venv_py
    return Path(sys.executable)


def resolve_cli_exe() -> Path | None:
    """Resolve the installed ``scar-forge.exe`` script in ``.venv`` if present."""
    venv_cli = REPO_ROOT / ".venv" / "Scripts" / "scar-forge.exe"
    if venv_cli.exists():
        return venv_cli
    return None


@dataclass
class CLIResult:
    """Captured subprocess result from invoking ``scar-forge`` or ``python -m scar_forge``."""

    returncode: int
    stdout: str
    stderr: str
    cmd: list[str]

    def json(self) -> Any:
        """Parse stdout strictly as JSON."""
        return json.loads(self.stdout)


@dataclass
class HTTPResponseAdapter:
    """Unified HTTP response wrapper across Flask test_client and live HTTP."""

    status_code: int
    body: bytes
    headers: dict[str, str]

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.text)


def free_local_port() -> int:
    """Allocate an ephemeral free TCP port on loopback 127.0.0.1."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_get(url: str, timeout: float = 5.0) -> HTTPResponseAdapter:
    """Perform an HTTP GET using stdlib urllib without raising on 4xx/5xx statuses."""
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            headers = {k.lower(): v for k, v in resp.headers.items()}
            return HTTPResponseAdapter(
                status_code=int(resp.status),
                body=resp.read(),
                headers=headers,
            )
    except urllib.error.HTTPError as exc:
        headers = {k.lower(): v for k, v in exc.headers.items()} if exc.headers else {}
        return HTTPResponseAdapter(
            status_code=int(exc.code),
            body=exc.read(),
            headers=headers,
        )


def make_config(data_dir: Path, **overrides: Any) -> Any:
    """Construct ``EngineConfig`` resiliently across minor unpinned field aliases."""
    from scar_forge.config import EngineConfig

    fields = set(getattr(EngineConfig, "model_fields", {}).keys())
    if not fields:
        fields = set(inspect.signature(EngineConfig).parameters.keys())

    kwargs: dict[str, Any] = {}
    for candidate in ("data_dir", "state_dir", "workspace_dir"):
        if candidate in fields:
            kwargs[candidate] = data_dir
            break

    alias_map: dict[str, tuple[str, ...]] = {
        "top_k": ("top_k", "top_k_scars", "top_k_skills"),
        "max_scar_tokens": ("max_scar_tokens", "max_axiom_tokens"),
        "escalation_threshold": ("escalation_threshold", "max_consecutive_failures"),
        "sandbox_timeout_s": ("sandbox_timeout_s", "timeout_s"),
        "sandbox_memory_mb": ("sandbox_memory_mb", "memory_mb"),
        "provider": ("provider", "llm_provider"),
        "ollama_url": ("ollama_url", "ollama_base_url"),
    }

    for key, val in overrides.items():
        if key in fields:
            kwargs[key] = val
        elif key in alias_map:
            matched = False
            for target in alias_map[key]:
                if target in fields:
                    kwargs[target] = val
                    matched = True
            if not matched:
                kwargs[key] = val
        else:
            kwargs[key] = val

    return EngineConfig(**kwargs)


def make_scar_store(data_dir: Path, **config_overrides: Any) -> Any:
    """Instantiate ``ScarStore`` isolated under ``data_dir``."""
    from scar_forge.memory.scar_store import ScarStore

    params = inspect.signature(ScarStore.__init__).parameters
    if "config" in params:
        return ScarStore(config=make_config(data_dir, **config_overrides))
    if "data_dir" in params:
        return ScarStore(data_dir=data_dir)
    if "db_path" in params:
        return ScarStore(db_path=data_dir / "scars.db")
    return ScarStore(data_dir)


def make_retriever(data_dir: Path | None = None, **config_overrides: Any) -> Any:
    """Instantiate ``BM25Retriever``."""
    from scar_forge.memory.retriever import BM25Retriever

    params = inspect.signature(BM25Retriever.__init__).parameters
    if "config" in params and data_dir is not None:
        return BM25Retriever(config=make_config(data_dir, **config_overrides))
    return BM25Retriever()


def make_sandbox_runner(data_dir: Path | None = None, **config_overrides: Any) -> Any:
    """Instantiate ``SandboxRunner``."""
    from scar_forge.sandbox.runner import SandboxRunner

    params = inspect.signature(SandboxRunner.__init__).parameters
    if "config" in params and data_dir is not None:
        return SandboxRunner(config=make_config(data_dir, **config_overrides))
    return SandboxRunner()


def make_ast_guard() -> Any:
    """Instantiate ``ASTGuard``."""
    from scar_forge.forge.sanitizer import ASTGuard

    return ASTGuard()


def make_skill_registry(data_dir: Path, **config_overrides: Any) -> Any:
    """Instantiate ``SkillRegistry`` isolated under ``data_dir``."""
    from scar_forge.forge.registry import SkillRegistry

    params = inspect.signature(SkillRegistry.__init__).parameters
    skills_dir = data_dir / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {}
    if "config" in params:
        kwargs["config"] = make_config(data_dir, **config_overrides)
    elif "skills_dir" in params:
        kwargs["skills_dir"] = skills_dir
    elif "data_dir" in params:
        kwargs["data_dir"] = data_dir

    if "sandbox" in params and params["sandbox"].default is inspect.Parameter.empty:
        kwargs["sandbox"] = make_sandbox_runner(data_dir, **config_overrides)
    if "ast_guard" in params and params["ast_guard"].default is inspect.Parameter.empty:
        kwargs["ast_guard"] = make_ast_guard()

    if kwargs:
        return SkillRegistry(**kwargs)
    return SkillRegistry(data_dir)


def make_llm(
    provider: str = "local",
    data_dir: Path | None = None,
    **config_overrides: Any,
) -> Any:
    """Instantiate ``DualTierLLM`` with explicit provider and config."""
    from scar_forge.llm.adapter import DualTierLLM

    params = inspect.signature(DualTierLLM.__init__).parameters
    kwargs: dict[str, Any] = {}
    if "config" in params:
        cfg_dir = data_dir if data_dir is not None else Path.cwd()
        kwargs["config"] = make_config(cfg_dir, provider=provider, **config_overrides)
    if "provider" in params:
        kwargs["provider"] = provider
    return DualTierLLM(**kwargs)


def make_engine(
    data_dir: Path,
    provider: str = "local",
    **config_overrides: Any,
) -> Any:
    """Instantiate ``ScarForgeEngine`` isolated under ``data_dir``."""
    from scar_forge import ScarForgeEngine

    params = inspect.signature(ScarForgeEngine.__init__).parameters
    if "config" in params:
        return ScarForgeEngine(
            config=make_config(data_dir, provider=provider, **config_overrides)
        )
    kwargs: dict[str, Any] = {}
    if "data_dir" in params:
        kwargs["data_dir"] = data_dir
    if "provider" in params:
        kwargs["provider"] = provider
    return ScarForgeEngine(**kwargs)


def run_engine_task(engine: Any, task: str, **kwargs: Any) -> Any:
    """Invoke the public task execution entrypoint on ``ScarForgeEngine``."""
    for method_name in ("run_task", "run", "solve", "execute"):
        method = getattr(engine, method_name, None)
        if callable(method):
            return method(task, **kwargs)
    raise AttributeError(
        f"ScarForgeEngine exposes none of run_task/run/solve/execute: {dir(engine)}"
    )


def get_ast_valid(result: Any) -> bool:
    """Extract boolean validation status from ``ASTValidationResult``."""
    for attr in ("valid", "passed", "is_valid", "ok"):
        if hasattr(result, attr):
            return bool(getattr(result, attr))
    return bool(result)


def get_ast_errors(result: Any) -> list[str]:
    """Extract error/violation messages from ``ASTValidationResult``."""
    for attr in ("errors", "violations", "reasons", "messages"):
        if hasattr(result, attr):
            val = getattr(result, attr)
            return list(val) if val else []
    return []


def get_scar_id(scar: Any) -> str:
    """Extract identifier from ``ScarRecord``."""
    for attr in ("scar_id", "id"):
        if hasattr(scar, attr):
            return str(getattr(scar, attr))
    raise AttributeError(f"ScarRecord missing id/scar_id: {scar!r}")


def get_prevented_count(scar: Any) -> int:
    """Extract prevention counter from ``ScarRecord``."""
    for attr in ("prevented_count", "hits_prevented", "prevention_count"):
        if hasattr(scar, attr):
            return int(getattr(scar, attr))
    raise AttributeError(f"ScarRecord missing prevented_count: {scar!r}")


def get_skill_name(skill: Any) -> str:
    """Extract skill identifier from ``SkillRecord``."""
    for attr in ("skill_name", "name"):
        if hasattr(skill, attr):
            return str(getattr(skill, attr))
    raise AttributeError(f"SkillRecord missing skill_name/name: {skill!r}")


def get_skill_path(skill: Any) -> Path | None:
    """Extract persisted ``.py`` path from ``SkillRecord`` if present."""
    for attr in ("module_path", "file_path", "path"):
        val = getattr(skill, attr, None)
        if val:
            return Path(str(val))
    return None


def get_llm_content(resp: Any) -> str:
    """Extract text/JSON payload from ``LLMResponse``."""
    for attr in ("content", "text", "raw_output", "output"):
        val = getattr(resp, attr, None)
        if isinstance(val, str):
            return val
    raise AttributeError(f"LLMResponse missing text content field: {resp!r}")


def unwrap_api_list(payload: Any, key: str) -> list[Any]:
    """Normalize either a raw JSON list or a ``{key: [...]}`` envelope."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for candidate in (key, "items", "data", "results"):
            val = payload.get(candidate)
            if isinstance(val, list):
                return val
    raise AssertionError(f"Expected list or dict[{key}] payload, got: {type(payload)} {payload!r}")


def parse_agent_action_text(raw_text: str) -> Any:
    """Invoke the public Zero-Prose parser in ``scar_forge.protocol``."""
    from scar_forge import protocol

    for fn_name in ("parse_agent_action", "parse_action"):
        fn = getattr(protocol, fn_name, None)
        if callable(fn):
            return fn(raw_text)

    parser_cls = getattr(protocol, "ProtocolParser", None)
    if parser_cls is not None:
        parser = parser_cls()
        for method_name in ("parse_agent_action", "parse_action", "parse"):
            method = getattr(parser, method_name, None)
            if callable(method):
                return method(raw_text)

    raise AttributeError("scar_forge.protocol exposes no public AgentAction parser")


def apply_protocol_patch(source: str, patch_text: str) -> str:
    """Invoke the public diff/patch applier in ``scar_forge.protocol``."""
    from scar_forge import protocol

    for fn_name in ("apply_code_patch", "apply_unified_diff", "apply_patch"):
        fn = getattr(protocol, fn_name, None)
        if callable(fn):
            return str(fn(source, patch_text))

    parser_cls = getattr(protocol, "ProtocolParser", None)
    if parser_cls is not None:
        parser = parser_cls()
        for method_name in ("apply_code_patch", "apply_unified_diff", "apply_patch"):
            method = getattr(parser, method_name, None)
            if callable(method):
                return str(method(source, patch_text))

    raise AttributeError("scar_forge.protocol exposes no public patch applier")


def make_distilled_error(
    exception_type: str,
    core_message: str,
    failing_line: str = "",
    line_number: int | None = 1,
    function_name: str | None = "main",
    raw_chars: int = 400,
    distilled_chars: int = 80,
    raw_tokens: int = 100,
    distilled_tokens: int = 20,
) -> Any:
    """Build a ``DistilledError`` instance using pinned schema fields."""
    from scar_forge.models import DistilledError

    ratio = round(1.0 - (distilled_chars / max(raw_chars, 1)), 4)
    return DistilledError(
        exception_type=exception_type,
        core_message=core_message,
        failing_line=failing_line,
        line_number=line_number,
        function_name=function_name,
        raw_chars=raw_chars,
        distilled_chars=distilled_chars,
        raw_tokens=raw_tokens,
        distilled_tokens=distilled_tokens,
        compression_ratio=ratio,
    )


def make_scar_record(
    scar_id: str,
    axiom: str,
    tags: list[str],
    exception_type: str = "ValueError",
    context_cue: str = "",
) -> Any:
    """Construct a valid ``ScarRecord`` populating unpinned defaults if required."""
    from scar_forge.models import ScarRecord
    from scar_forge.tokens import count_tokens

    fields = set(ScarRecord.model_fields.keys())
    now = datetime.now(timezone.utc)
    kwargs: dict[str, Any] = {}
    if "scar_id" in fields:
        kwargs["scar_id"] = scar_id
    elif "id" in fields:
        kwargs["id"] = scar_id
    if "axiom" in fields:
        kwargs["axiom"] = axiom
    if "token_count" in fields:
        kwargs["token_count"] = count_tokens(axiom)
    if "exception_type" in fields:
        kwargs["exception_type"] = exception_type
    if "signature_hash" in fields:
        kwargs["signature_hash"] = f"sig-{scar_id}"
    if "error_signature" in fields:
        kwargs["error_signature"] = f"sig-{scar_id}"
    if "tags" in fields:
        kwargs["tags"] = tags
    if "context_cue" in fields:
        kwargs["context_cue"] = context_cue or axiom[:100]
    if "hit_count" in fields:
        kwargs["hit_count"] = 1
    if "prevented_count" in fields:
        kwargs["prevented_count"] = 0
    if "created_at" in fields:
        kwargs["created_at"] = now
    if "last_seen_at" in fields:
        kwargs["last_seen_at"] = now
    return ScarRecord(**kwargs)


def make_skill_record(
    skill_name: str,
    docstring: str,
    tags: list[str],
    signature: str | None = None,
    module_path: str = "",
) -> Any:
    """Construct a valid ``SkillRecord`` populating unpinned defaults if required."""
    from scar_forge.models import SkillRecord

    fields = set(SkillRecord.model_fields.keys())
    now = datetime.now(timezone.utc)
    kwargs: dict[str, Any] = {}
    if "skill_id" in fields:
        kwargs["skill_id"] = f"skill-{skill_name}"
    if "skill_name" in fields:
        kwargs["skill_name"] = skill_name
    elif "name" in fields:
        kwargs["name"] = skill_name
    if "version" in fields:
        kwargs["version"] = 1
    if "signature" in fields:
        kwargs["signature"] = signature or f"def {skill_name}(*args, **kwargs)"
    if "docstring" in fields:
        kwargs["docstring"] = docstring
    if "tags" in fields:
        kwargs["tags"] = tags
    if "module_path" in fields:
        kwargs["module_path"] = module_path or f"{skill_name}.py"
    elif "file_path" in fields:
        kwargs["file_path"] = module_path or f"{skill_name}.py"
    if "source_hash" in fields:
        kwargs["source_hash"] = "0" * 64
    if "source_sha256" in fields:
        kwargs["source_sha256"] = "0" * 64
    if "ast_verified" in fields:
        kwargs["ast_verified"] = True
    if "sandbox_verified" in fields:
        kwargs["sandbox_verified"] = True
    if "test_cases" in fields:
        kwargs["test_cases"] = []
    if "call_count" in fields:
        kwargs["call_count"] = 0
    elif "reuse_count" in fields:
        kwargs["reuse_count"] = 0
    if "created_at" in fields:
        kwargs["created_at"] = now
    return SkillRecord(**kwargs)


def make_test_case(
    *,
    kwargs: dict[str, Any] | None = None,
    args: list[Any] | None = None,
    expected: Any = None,
) -> dict[str, Any]:
    """Construct a standard sandbox test case dictionary."""
    case: dict[str, Any] = {"expected": expected}
    if kwargs is not None:
        case["kwargs"] = kwargs
    if args is not None:
        case["args"] = args
    if kwargs is None and args is None:
        case["kwargs"] = {}
    return case


@pytest.fixture
def e2e_data_dir(tmp_path: Path) -> Path:
    """Dedicated per-test data directory preventing repo ``.scar_forge/`` pollution."""
    data_dir = tmp_path / "scar_forge_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture
def e2e_workdir(tmp_path: Path) -> Path:
    """Dedicated per-test working directory for subprocess commands."""
    work = tmp_path / "workdir"
    work.mkdir(parents=True, exist_ok=True)
    return work


@pytest.fixture
def e2e_config(e2e_data_dir: Path) -> Any:
    """Isolated ``EngineConfig`` bound to ``e2e_data_dir``."""
    return make_config(e2e_data_dir)


@pytest.fixture
def clean_env(e2e_data_dir: Path) -> dict[str, str]:
    """UTF-8 enforced subprocess environment pointing to isolated ``e2e_data_dir``."""
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["SCAR_FORGE_DATA_DIR"] = str(e2e_data_dir)
    env["SCAR_FORGE_PROVIDER"] = "local"
    return env


@pytest.fixture
def run_cli(e2e_workdir: Path, e2e_data_dir: Path, clean_env: dict[str, str]):
    """Run ``python -m scar_forge`` or ``scar-forge.exe`` in an isolated subprocess."""

    def _run(
        args: list[str],
        *,
        use_console_script: bool = False,
        inject_data_dir: bool = True,
        timeout: float = 45.0,
        extra_env: dict[str, str] | None = None,
        override_env: dict[str, str] | None = None,
    ) -> CLIResult:
        env = clean_env.copy()
        if extra_env:
            env.update(extra_env)
        if override_env is not None:
            env = override_env

        cli_exe = resolve_cli_exe() if use_console_script else None
        base_cmd = (
            [str(cli_exe)] if cli_exe else [str(resolve_python_exe()), "-m", "scar_forge"]
        )

        cmd_args = list(args)
        if (
            inject_data_dir
            and "--data-dir" not in cmd_args
            and cmd_args
            and cmd_args[0] in {"demo", "run", "benchmark", "inspect", "search", "dashboard"}
        ):
            cmd_args.extend(["--data-dir", str(e2e_data_dir)])

        full_cmd = base_cmd + cmd_args
        proc = subprocess.run(
            full_cmd,
            cwd=str(e2e_workdir),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return CLIResult(
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            cmd=full_cmd,
        )

    return _run


@pytest.fixture
def live_dashboard_url(
    e2e_workdir: Path, e2e_data_dir: Path, clean_env: dict[str, str]
) -> Iterator[str]:
    """Launch ``scar-forge dashboard`` on an ephemeral port and clean up in ``finally``."""
    port = free_local_port()
    base_url = f"http://127.0.0.1:{port}"
    cmd = [
        str(resolve_python_exe()),
        "-m",
        "scar_forge",
        "dashboard",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--data-dir",
        str(e2e_data_dir),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(e2e_workdir),
        env=clean_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        deadline = time.monotonic() + 15.0
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                out, err = proc.communicate(timeout=2.0)
                raise RuntimeError(
                    f"Dashboard exited early (code={proc.returncode}):\nSTDOUT:{out}\nSTDERR:{err}"
                )
            try:
                resp = http_get(f"{base_url}/api/state", timeout=1.0)
                if resp.status_code == 200:
                    break
            except Exception as exc:
                last_err = exc
            time.sleep(0.15)
        else:
            raise TimeoutError(f"Dashboard did not become ready at {base_url}: {last_err}")
        yield base_url
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)


@pytest.fixture
def dashboard_get(e2e_workdir: Path, e2e_data_dir: Path, clean_env: dict[str, str]):
    """Provide a unified ``get(path) -> HTTPResponseAdapter`` via ``create_app`` or live CLI.

    Initializes lazily on first call so tests can seed ``e2e_data_dir`` before
    the Flask app starts.
    """
    from scar_forge.ui import dashboard

    state: dict[str, Any] = {"client_ctx": None, "client": None, "proc": None, "base_url": None}

    def _ensure_started() -> None:
        if state["client"] is not None or state["base_url"] is not None:
            return
        create_app = getattr(dashboard, "create_app", None)
        if callable(create_app):
            app = create_app(make_config(e2e_data_dir))
            app.config["TESTING"] = True
            ctx = app.test_client()
            state["client_ctx"] = ctx
            state["client"] = ctx.__enter__()
            return

        port = free_local_port()
        base_url = f"http://127.0.0.1:{port}"
        cmd = [
            str(resolve_python_exe()),
            "-m",
            "scar_forge",
            "dashboard",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--data-dir",
            str(e2e_data_dir),
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=str(e2e_workdir),
            env=clean_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        state["proc"] = proc
        deadline = time.monotonic() + 15.0
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                out, err = proc.communicate(timeout=2.0)
                raise RuntimeError(
                    f"Dashboard exited early (code={proc.returncode}):\nSTDOUT:{out}\nSTDERR:{err}"
                )
            try:
                probe = http_get(f"{base_url}/api/state", timeout=1.0)
                if probe.status_code == 200:
                    state["base_url"] = base_url
                    return
            except Exception as exc:
                last_err = exc
            time.sleep(0.15)
        raise TimeoutError(f"Dashboard did not become ready at {base_url}: {last_err}")

    def _get(path: str) -> HTTPResponseAdapter:
        _ensure_started()
        if state["client"] is not None:
            resp = state["client"].get(path)
            headers = {k.lower(): v for k, v in resp.headers.items()}
            return HTTPResponseAdapter(
                status_code=int(resp.status_code),
                body=resp.get_data(),
                headers=headers,
            )
        return http_get(f"{state['base_url']}{path}", timeout=5.0)

    try:
        yield _get
    finally:
        if state["client_ctx"] is not None:
            state["client_ctx"].__exit__(None, None, None)
        proc = state["proc"]
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)
