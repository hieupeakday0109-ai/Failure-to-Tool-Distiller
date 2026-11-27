"""Isolated sandbox with Win32 Job Objects, RSS/tracemalloc caps, and adversarial tests."""

from __future__ import annotations

import ast
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import psutil

from scar_forge.config import EngineConfig
from scar_forge.distiller import TracebackDistiller
from scar_forge.models import DistilledError, SandboxResult, TestCaseResult

_SAFE_ENV_KEYS = (
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "PATH",
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    "COMMONPROGRAMFILES",
    "LD_LIBRARY_PATH",
    "LC_ALL",
    "LANG",
)

_HARNESS_SCRIPT = r'''
import builtins
import datetime as _dt
import decimal as _dec
import importlib.util
import json
import math
import os
import sys
import time
import traceback
import tracemalloc

_BLOCKED_MODULES = frozenset({
    "subprocess",
    "multiprocessing",
    "ctypes",
    "socket",
    "http",
    "ftplib",
    "smtplib",
    "telnetlib",
    "webbrowser",
    "pty",
    "fcntl",
    "resource",
    "signal",
    "shutil",
    "runpy",
    "importlib",
    "winreg",
    "_winapi",
    "msvcrt",
    "concurrent",
})

_BLOCKED_PREFIXES = (
    "subprocess.",
    "multiprocessing.",
    "ctypes.",
    "socket.",
    "http.",
    "urllib.request",
    "urllib.error",
    "importlib.",
    "concurrent.futures.process",
)


def _decode_val(obj):
    if isinstance(obj, dict) and "__scar_type__" in obj:
        t = obj["__scar_type__"]
        if t == "decimal":
            return _dec.Decimal(obj["value"])
        if t == "datetime":
            return _dt.datetime.fromisoformat(obj["value"])
        if t == "date":
            return _dt.date.fromisoformat(obj["value"])
        if t == "timedelta":
            return _dt.timedelta(seconds=float(obj["value"]))
        if t == "set":
            return {_decode_val(x) for x in obj["items"]}
        if t == "tuple":
            return tuple(_decode_val(x) for x in obj["items"])
        if t == "float_special":
            return float(obj["value"])
    if isinstance(obj, dict):
        return {k: _decode_val(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_val(x) for x in obj]
    return obj


def _encode_val(obj):
    if isinstance(obj, float):
        if math.isnan(obj):
            return {"__scar_type__": "float_special", "value": "nan"}
        if math.isinf(obj):
            return {"__scar_type__": "float_special", "value": "inf" if obj > 0 else "-inf"}
        return obj
    if isinstance(obj, _dec.Decimal):
        return {"__scar_type__": "decimal", "value": str(obj)}
    if isinstance(obj, _dt.datetime):
        return {"__scar_type__": "datetime", "value": obj.isoformat()}
    if isinstance(obj, _dt.date):
        return {"__scar_type__": "date", "value": obj.isoformat()}
    if isinstance(obj, _dt.timedelta):
        return {"__scar_type__": "timedelta", "value": obj.total_seconds()}
    if isinstance(obj, set):
        return {"__scar_type__": "set", "items": [_encode_val(x) for x in sorted(obj, key=repr)]}
    if isinstance(obj, tuple):
        return {"__scar_type__": "tuple", "items": [_encode_val(x) for x in obj]}
    if isinstance(obj, dict):
        return {str(k): _encode_val(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_encode_val(x) for x in obj]
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return repr(obj)


def _values_equal(actual, expected):
    if isinstance(expected, float) and isinstance(actual, (int, float)):
        if math.isnan(expected):
            return isinstance(actual, float) and math.isnan(actual)
        if math.isinf(expected):
            return actual == expected
        return math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(expected, list) and isinstance(actual, list):
        return len(actual) == len(expected) and all(
            _values_equal(a, b) for a, b in zip(actual, expected)
        )
    if isinstance(expected, tuple) and isinstance(actual, tuple):
        return len(actual) == len(expected) and all(
            _values_equal(a, b) for a, b in zip(actual, expected)
        )
    if isinstance(expected, dict) and isinstance(actual, dict):
        return set(actual.keys()) == set(expected.keys()) and all(
            _values_equal(actual[k], expected[k]) for k in expected
        )
    return actual == expected


def _install_guards():
    orig_import = builtins.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        caller_name = globals.get("__name__", "") if isinstance(globals, dict) else ""
        root = name.split(".", 1)[0]
        if caller_name == "scar_sandbox_target" and (
            root in _BLOCKED_MODULES or any(name.startswith(p) for p in _BLOCKED_PREFIXES)
        ):
            raise ImportError(f"Blocked unsafe module '{name}' in sandbox")
        return orig_import(name, globals, locals, fromlist, level)

    builtins.__import__ = _guarded_import

    def _blocked_exec(*_args, **_kwargs):
        raise PermissionError("Process spawning is blocked inside sandbox")

    for attr in ("system", "popen", "fork", "forkpty", "spawnl", "spawnle", "spawnlp", "spawnlpe",
                 "spawnv", "spawnve", "spawnvp", "spawnvpe", "execl", "execle", "execlp",
                 "execlpe", "execv", "execve", "execvp", "execvpe"):
        if hasattr(os, attr):
            setattr(os, attr, _blocked_exec)


def main():
    spec_path = sys.argv[1]
    with open(spec_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    code_path = payload["code_path"]
    entry_fn = payload["entry_fn"]
    cases = _decode_val(payload["cases"])
    memory_limit_bytes = int(payload["memory_mb"]) * 1024 * 1024
    allow_explicit_raises = bool(payload.get("allow_explicit_raises", False))
    result_path = payload["result_path"]

    _install_guards()

    try:
        mod_spec = importlib.util.spec_from_file_location("scar_sandbox_target", code_path)
        if mod_spec is None or mod_spec.loader is None:
            raise RuntimeError("Failed to create module spec for sandbox target")
        mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(mod)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        with open(result_path, "w", encoding="utf-8") as out_fh:
            json.dump(
                {"passed": False, "case_results": [], "last_value": None, "memory_exceeded": False},
                out_fh,
            )
        sys.exit(1)

    tracemalloc.start()

    fn = getattr(mod, entry_fn, None)
    if not callable(fn):
        sys.stderr.write(f"AttributeError: Entry function '{entry_fn}' not found or not callable\n")
        with open(result_path, "w", encoding="utf-8") as out_fh:
            json.dump(
                {"passed": False, "case_results": [], "last_value": None, "memory_exceeded": False},
                out_fh,
            )
        sys.exit(1)

    case_results = []
    all_passed = True
    last_value = None
    mem_exceeded = False
    first_error_printed = False

    for idx, case in enumerate(cases):
        name = str(case.get("name") or f"case_{idx + 1}")
        is_adv = bool(case.get("is_adversarial", False))
        args = case.get("args") or []
        kwargs = case.get("kwargs") or {}
        expected = case.get("expected")
        expected_error = case.get("expected_error")
        has_expected = "expected" in case

        t0 = time.perf_counter()
        try:
            actual = fn(*args, **kwargs)
            dur_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            _, peak_bytes = tracemalloc.get_traced_memory()
            if peak_bytes > memory_limit_bytes:
                mem_exceeded = True
                all_passed = False
                peak_mb = peak_bytes // (1024 * 1024)
                err_msg = (
                    f"MemoryError: Traced allocation {peak_mb}MB "
                    f"exceeded limit {payload['memory_mb']}MB"
                )
                sys.stderr.write(err_msg + "\n")
                case_results.append({
                    "name": name,
                    "passed": False,
                    "is_adversarial": is_adv,
                    "expected": _encode_val(expected) if has_expected else None,
                    "actual": None,
                    "error": err_msg,
                    "duration_ms": dur_ms,
                })
                break

            last_value = actual
            if expected_error:
                passed = False
                err_msg = f"AssertionError: Expected {expected_error} to be raised, got {actual!r}"
                if not first_error_printed:
                    sys.stderr.write(err_msg + "\n")
                    first_error_printed = True
            elif is_adv and not has_expected:
                passed = True
                err_msg = None
            elif has_expected:
                passed = _values_equal(actual, expected)
                err_msg = (
                    None
                    if passed
                    else f"AssertionError: Expected {expected!r}, got {actual!r}"
                )
                if not passed and not first_error_printed:
                    sys.stderr.write(err_msg + "\n")
                    first_error_printed = True
            else:
                passed = True
                err_msg = None

            if not passed:
                all_passed = False

            case_results.append({
                "name": name,
                "passed": passed,
                "is_adversarial": is_adv,
                "expected": _encode_val(expected) if has_expected else None,
                "actual": _encode_val(actual),
                "error": err_msg,
                "duration_ms": dur_ms,
            })
        except MemoryError as exc:
            dur_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            mem_exceeded = True
            all_passed = False
            traceback.print_exc(file=sys.stderr)
            case_results.append({
                "name": name,
                "passed": False,
                "is_adversarial": is_adv,
                "expected": _encode_val(expected) if has_expected else None,
                "actual": None,
                "error": f"MemoryError: {exc}",
                "duration_ms": dur_ms,
            })
            break
        except Exception as exc:
            dur_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            exc_name = type(exc).__name__
            if expected_error and exc_name == expected_error:
                case_results.append({
                    "name": name,
                    "passed": True,
                    "is_adversarial": is_adv,
                    "expected": expected_error,
                    "actual": exc_name,
                    "error": None,
                    "duration_ms": dur_ms,
                })
                continue

            if is_adv and allow_explicit_raises and exc_name in {"ValueError", "TypeError"}:
                case_results.append({
                    "name": name,
                    "passed": True,
                    "is_adversarial": is_adv,
                    "expected": None,
                    "actual": exc_name,
                    "error": None,
                    "duration_ms": dur_ms,
                })
                continue

            all_passed = False
            if not first_error_printed:
                traceback.print_exc(file=sys.stderr)
                first_error_printed = True
            case_results.append({
                "name": name,
                "passed": False,
                "is_adversarial": is_adv,
                "expected": _encode_val(expected) if has_expected else None,
                "actual": None,
                "error": f"{exc_name}: {exc}",
                "duration_ms": dur_ms,
            })

    with open(result_path, "w", encoding="utf-8") as out_fh:
        json.dump(
            {
                "passed": all_passed,
                "case_results": case_results,
                "last_value": _encode_val(last_value),
                "memory_exceeded": mem_exceeded,
            },
            out_fh,
            ensure_ascii=False,
        )
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
'''


def _encode_json_safe(obj: Any) -> Any:
    import datetime as _dt
    import decimal as _dec
    import math

    if isinstance(obj, float):
        if math.isnan(obj):
            return {"__scar_type__": "float_special", "value": "nan"}
        if math.isinf(obj):
            return {"__scar_type__": "float_special", "value": "inf" if obj > 0 else "-inf"}
        return obj
    if isinstance(obj, _dec.Decimal):
        return {"__scar_type__": "decimal", "value": str(obj)}
    if isinstance(obj, _dt.datetime):
        return {"__scar_type__": "datetime", "value": obj.isoformat()}
    if isinstance(obj, _dt.date):
        return {"__scar_type__": "date", "value": obj.isoformat()}
    if isinstance(obj, _dt.timedelta):
        return {"__scar_type__": "timedelta", "value": obj.total_seconds()}
    if isinstance(obj, set):
        return {
            "__scar_type__": "set",
            "items": [_encode_json_safe(x) for x in sorted(obj, key=repr)],
        }
    if isinstance(obj, tuple):
        return {"__scar_type__": "tuple", "items": [_encode_json_safe(x) for x in obj]}
    if isinstance(obj, dict):
        return {str(k): _encode_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_encode_json_safe(x) for x in obj]
    return obj


def _decode_json_safe(obj: Any) -> Any:
    import datetime as _dt
    import decimal as _dec

    if isinstance(obj, dict) and "__scar_type__" in obj:
        t = obj["__scar_type__"]
        if t == "decimal":
            return _dec.Decimal(obj["value"])
        if t == "datetime":
            return _dt.datetime.fromisoformat(obj["value"])
        if t == "date":
            return _dt.date.fromisoformat(obj["value"])
        if t == "timedelta":
            return _dt.timedelta(seconds=float(obj["value"]))
        if t == "set":
            return {_decode_json_safe(x) for x in obj["items"]}
        if t == "tuple":
            return tuple(_decode_json_safe(x) for x in obj["items"])
        if t == "float_special":
            return float(obj["value"])
    if isinstance(obj, dict):
        return {k: _decode_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_json_safe(x) for x in obj]
    return obj


def _build_sanitized_env(work_dir: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for key in _SAFE_ENV_KEYS:
        if key in os.environ:
            env[key] = os.environ[key]
    tmp_str = str(work_dir)
    env["TEMP"] = tmp_str
    env["TMP"] = tmp_str
    env["TMPDIR"] = tmp_str
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    return env


def _kill_process_tree(pid: int) -> None:
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for child in children:
        with contextlib.suppress(psutil.Error):
            child.kill()
    with contextlib.suppress(psutil.Error):
        parent.kill()
    psutil.wait_procs([parent, *children], timeout=2.0)


class SandboxRunner:
    """Isolated subprocess code & adversarial skill execution sandbox."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self.distiller = TracebackDistiller()

    def _create_win32_job(self, memory_mb: int) -> Any | None:
        if sys.platform != "win32" or not self.config.enable_win32_job_object:
            return None
        try:
            import win32job  # type: ignore[import-untyped]

            hjob = win32job.CreateJobObject(None, "")
            info = win32job.QueryInformationJobObject(
                hjob, win32job.JobObjectExtendedLimitInformation
            )
            info["BasicLimitInformation"]["LimitFlags"] = (
                win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                | win32job.JOB_OBJECT_LIMIT_ACTIVE_PROCESS
                | win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY
            )
            # .venv launcher spawns the base python.exe child, requiring >= 2 process slots
            info["BasicLimitInformation"]["ActiveProcessLimit"] = max(
                2, self.config.sandbox_max_processes
            )
            # Virtual commit cap allows NumPy/Pandas DLL maps while catching allocation bombs
            commit_cap_mb = (
                memory_mb + 128 if memory_mb < 128 else max(memory_mb * 2, memory_mb + 256)
            )
            info["ProcessMemoryLimit"] = commit_cap_mb * 1024 * 1024
            win32job.SetInformationJobObject(
                hjob, win32job.JobObjectExtendedLimitInformation, info
            )
            return hjob
        except Exception:
            return None

    def _execute_in_subprocess(
        self,
        code: str,
        cases: list[dict[str, Any]],
        *,
        entry_fn: str,
        timeout_s: float,
        memory_mb: int,
        allow_explicit_raises: bool = False,
    ) -> SandboxResult:
        if not isinstance(code, str) or not code.strip():
            err = DistilledError(
                exception_type="ValueError",
                core_message="Empty code payload passed to sandbox",
                raw_chars=0,
                distilled_chars=45,
                raw_tokens=0,
                distilled_tokens=10,
                compression_ratio=0.0,
            )
            return SandboxResult(
                passed=False,
                exit_code=2,
                stderr="ValueError: Empty code payload passed to sandbox",
                distilled_error=err,
            )

        try:
            ast.parse(code, filename="solution.py")
        except SyntaxError as exc:
            msg = f"{type(exc).__name__}: {exc.msg}"
            raw_tb = (
                f'  File "solution.py", line {exc.lineno or 1}\n'
                f"    {(exc.text or '').rstrip()}\n"
                f"{msg}\n"
            )
            distilled = self.distiller.distill(raw_tb)
            return SandboxResult(
                passed=False,
                exit_code=2,
                stderr=raw_tb,
                distilled_error=distilled,
            )

        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="scar_sandbox_") as tmp_dir:
            work_dir = Path(tmp_dir)
            code_path = work_dir / "solution.py"
            harness_path = work_dir / "_sandbox_runner.py"
            spec_path = work_dir / "spec.json"
            result_path = work_dir / "result.json"

            code_path.write_text(code, encoding="utf-8")
            harness_path.write_text(_HARNESS_SCRIPT, encoding="utf-8")
            spec_payload = {
                "code_path": str(code_path),
                "entry_fn": entry_fn,
                "cases": _encode_json_safe(cases),
                "memory_mb": memory_mb,
                "allow_explicit_raises": allow_explicit_raises,
                "result_path": str(result_path),
            }
            spec_path.write_text(json.dumps(spec_payload), encoding="utf-8")

            env = _build_sanitized_env(work_dir)
            hjob = self._create_win32_job(memory_mb)
            proc: subprocess.Popen[str] | None = None
            timed_out = False
            memory_exceeded = False
            rss_limit_bytes = memory_mb * 1024 * 1024

            try:
                proc = subprocess.Popen(
                    [sys.executable, str(harness_path), str(spec_path)],
                    cwd=str(work_dir),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                if hjob is not None:
                    try:
                        import win32job  # type: ignore[import-untyped]

                        win32job.AssignProcessToJobObject(hjob, proc._handle)
                    except Exception:
                        pass

                deadline = time.monotonic() + timeout_s
                ps_proc: psutil.Process | None = None
                try:
                    ps_proc = psutil.Process(proc.pid)
                except psutil.Error:
                    ps_proc = None

                while proc.poll() is None:
                    if time.monotonic() > deadline:
                        timed_out = True
                        _kill_process_tree(proc.pid)
                        break

                    if ps_proc is not None:
                        try:
                            total_rss = ps_proc.memory_info().rss
                            for child in ps_proc.children(recursive=True):
                                with contextlib.suppress(psutil.Error):
                                    total_rss += child.memory_info().rss
                            if total_rss > rss_limit_bytes:
                                memory_exceeded = True
                                _kill_process_tree(proc.pid)
                                break
                        except psutil.Error:
                            pass

                    time.sleep(0.015)

                try:
                    stdout, stderr = proc.communicate(timeout=2.0)
                except subprocess.TimeoutExpired:
                    _kill_process_tree(proc.pid)
                    stdout, stderr = proc.communicate(timeout=2.0)

                exit_code = proc.returncode if proc.returncode is not None else -1
            finally:
                if proc is not None and proc.poll() is None:
                    _kill_process_tree(proc.pid)
                if hjob is not None:
                    try:
                        import win32api  # type: ignore[import-untyped]

                        win32api.CloseHandle(hjob)
                    except Exception:
                        pass

            duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            if timed_out:
                err_msg = f"TimeoutError: Sandbox execution exceeded {timeout_s:.2f}s limit"
                combined_err = f"{stderr}\n{err_msg}".strip()
                return SandboxResult(
                    passed=False,
                    exit_code=exit_code,
                    stdout=stdout,
                    stderr=combined_err,
                    distilled_error=self.distiller.distill(combined_err, stdout),
                    timed_out=True,
                    duration_ms=duration_ms,
                    total_cases=len(cases),
                    passed_cases=0,
                )

            parsed_res: dict[str, Any] = {}
            if result_path.is_file():
                try:
                    parsed_res = json.loads(result_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    parsed_res = {}

            if parsed_res.get("memory_exceeded"):
                memory_exceeded = True

            if memory_exceeded and not parsed_res:
                err_msg = f"MemoryError: Sandbox RSS exceeded {memory_mb}MB limit"
                combined_err = f"{stderr}\n{err_msg}".strip()
                return SandboxResult(
                    passed=False,
                    exit_code=exit_code,
                    stdout=stdout,
                    stderr=combined_err,
                    distilled_error=self.distiller.distill(combined_err, stdout),
                    memory_exceeded=True,
                    duration_ms=duration_ms,
                    total_cases=len(cases),
                    passed_cases=0,
                )

            raw_cases = parsed_res.get("case_results", [])
            case_results = [
                TestCaseResult(
                    name=str(c["name"]),
                    passed=bool(c["passed"]),
                    is_adversarial=bool(c.get("is_adversarial", False)),
                    expected=_decode_json_safe(c.get("expected")),
                    actual=_decode_json_safe(c.get("actual")),
                    error=c.get("error"),
                    duration_ms=float(c.get("duration_ms", 0.0)),
                )
                for c in raw_cases
            ]
            passed_cases = sum(1 for c in case_results if c.passed)
            passed = (
                bool(parsed_res.get("passed", False))
                and exit_code == 0
                and not memory_exceeded
            )

            distilled: DistilledError | None = None
            if not passed:
                err_text = stderr.strip()
                if not err_text:
                    failing_msgs = [c.error for c in case_results if not c.passed and c.error]
                    err_text = (
                        failing_msgs[0]
                        if failing_msgs
                        else f"ExitCodeError: exit code {exit_code}"
                    )
                distilled = self.distiller.distill(err_text, stdout)

            return SandboxResult(
                passed=passed,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                distilled_error=distilled,
                timed_out=False,
                memory_exceeded=memory_exceeded,
                duration_ms=duration_ms,
                total_cases=len(cases),
                passed_cases=passed_cases,
                case_results=case_results,
                return_value=_decode_json_safe(parsed_res.get("last_value")),
            )

    def execute_code(
        self,
        code: str,
        test_cases: Sequence[dict[str, Any]] | None = None,
        *,
        entry_fn: str = "solve",
        timeout_s: float | None = None,
        memory_mb: int | None = None,
    ) -> SandboxResult:
        """Execute untrusted Python code against explicit test cases in an isolated process."""
        eff_timeout = float(timeout_s if timeout_s is not None else self.config.sandbox_timeout_s)
        eff_mem = int(memory_mb if memory_mb is not None else self.config.sandbox_memory_mb)
        cases = [dict(c) for c in (test_cases or [])]
        if not cases:
            cases = [{"name": "smoke_call", "kwargs": {}}]
        return self._execute_in_subprocess(
            code,
            cases,
            entry_fn=entry_fn,
            timeout_s=eff_timeout,
            memory_mb=eff_mem,
            allow_explicit_raises=False,
        )

    @staticmethod
    def _inspect_function_contract(
        source: str, skill_name: str
    ) -> tuple[list[tuple[str, str, bool]], bool]:
        """Extract [(param_name, type_hint_lower, allows_none)] and explicit_raise flag."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return [], False

        target_fn: ast.FunctionDef | None = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == skill_name:
                target_fn = node
                break
        if target_fn is None:
            return [], False

        has_explicit_raise = any(
            isinstance(n, ast.Raise) and n.exc is not None for n in ast.walk(target_fn)
        )

        params: list[tuple[str, str, bool]] = []
        args_list = list(target_fn.args.posonlyargs) + list(target_fn.args.args)
        num_defaults = len(target_fn.args.defaults)
        default_start = len(args_list) - num_defaults

        for idx, arg in enumerate(args_list):
            if arg.arg in {"self", "cls"}:
                continue
            ann_str = ast.unparse(arg.annotation).lower() if arg.annotation else ""
            allows_none = "none" in ann_str or "optional" in ann_str
            if idx >= default_start:
                def_node = target_fn.args.defaults[idx - default_start]
                if isinstance(def_node, ast.Constant) and def_node.value is None:
                    allows_none = True
            params.append((arg.arg, ann_str, allows_none))

        return params, has_explicit_raise

    @staticmethod
    def _edge_values_for_type(type_hint: str, allows_none: bool) -> list[Any]:
        hint = (type_hint or "").strip().lower()
        values: list[Any] = []
        # Check container types before inner element types (e.g. list[int], dict[str, Any])
        if any(
            hint.startswith(prefix)
            for prefix in ("list", "sequence", "typing.list", "typing.sequence", "tuple")
        ):
            values.append([])
        elif any(
            hint.startswith(prefix)
            for prefix in ("dict", "mapping", "typing.dict", "typing.mapping")
        ):
            values.append({})
        elif hint.startswith(("set", "frozenset")):
            values.append(set())
        elif "str" in hint:
            values.extend(["", "   ", "\ufeff"])
        elif "bool" in hint:
            values.extend([False, True])
        elif "int" in hint:
            values.extend([0, -1])
        elif "float" in hint:
            values.extend([0.0, -1.0])
        else:
            values.extend(["", []])

        if allows_none:
            values.append(None)
        return values

    @staticmethod
    def _default_value_for_type(type_hint: str, allows_none: bool) -> Any:
        hint = (type_hint or "").strip().lower()
        if any(
            hint.startswith(prefix)
            for prefix in ("list", "sequence", "typing.list", "typing.sequence", "tuple")
        ):
            return []
        if any(
            hint.startswith(prefix)
            for prefix in ("dict", "mapping", "typing.dict", "typing.mapping")
        ):
            return {}
        if hint.startswith(("set", "frozenset")):
            return set()
        if "str" in hint:
            return "sample"
        if "bool" in hint:
            return False
        if "int" in hint:
            return 1
        if "float" in hint:
            return 1.0
        if allows_none:
            return None
        return ""

    def verify_skill_adversarial(
        self,
        source: str,
        skill_name: str,
        explicit_cases: Sequence[dict[str, Any]] | None = None,
        param_schema: dict[str, str] | None = None,
        *,
        timeout_s: float | None = None,
        memory_mb: int | None = None,
    ) -> SandboxResult:
        """Verify a forged skill against explicit tests plus synthesized adversarial edge cases."""
        eff_timeout = float(timeout_s if timeout_s is not None else self.config.sandbox_timeout_s)
        eff_mem = int(memory_mb if memory_mb is not None else self.config.sandbox_memory_mb)

        explicit = [dict(c) for c in (explicit_cases or [])]
        for idx, c in enumerate(explicit, start=1):
            c.setdefault("name", f"explicit_{idx}")
            c["is_adversarial"] = False

        params, has_explicit_raise = self._inspect_function_contract(source, skill_name)
        schema_map = {k: str(v).lower() for k, v in (param_schema or {}).items()}

        base_kwargs: dict[str, Any] = {}
        if explicit and isinstance(explicit[0].get("kwargs"), dict):
            base_kwargs = dict(explicit[0]["kwargs"])

        for pname, ptype, allows_none in params:
            eff_type = ptype or schema_map.get(pname, "")
            if pname not in base_kwargs:
                base_kwargs[pname] = self._default_value_for_type(eff_type, allows_none)

        adv_cases: list[dict[str, Any]] = []
        for pname, ptype, allows_none in params:
            eff_type = schema_map.get(pname, "") or ptype
            for idx, edge_val in enumerate(
                self._edge_values_for_type(eff_type, allows_none), start=1
            ):
                adv_kwargs = dict(base_kwargs)
                adv_kwargs[pname] = edge_val
                adv_cases.append(
                    {
                        "name": f"adv_{pname}_{idx}",
                        "kwargs": adv_kwargs,
                        "is_adversarial": True,
                    }
                )

        all_cases = explicit + adv_cases
        if not all_cases:
            all_cases = [{"name": "adv_smoke", "kwargs": base_kwargs, "is_adversarial": True}]

        return self._execute_in_subprocess(
            source,
            all_cases,
            entry_fn=skill_name,
            timeout_s=eff_timeout,
            memory_mb=eff_mem,
            allow_explicit_raises=has_explicit_raise,
        )
