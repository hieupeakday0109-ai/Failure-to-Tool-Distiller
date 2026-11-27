"""Static AST and tokenizer hygiene gate enforcing zero TODO/FIXME/stubs in repo."""

from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_PARTS = frozenset(
    {
        ".venv",
        "venv",
        "__pycache__",
        ".git",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "build",
        "dist",
    }
)
_MARKER_RE = re.compile(r"\b(TODO|FIXME|XXX)\b", re.IGNORECASE)


def collect_python_files(root: Path = ROOT) -> list[Path]:
    """Collect all .py files under src/ and tests/, skipping virtualenvs and caches."""
    files: list[Path] = []
    for folder_name in ("src", "tests"):
        base = root / folder_name
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            files.append(path)
    return files


def find_comment_markers(source: str, filename: str = "<memory>") -> list[str]:
    """Return violations for TODO/FIXME/XXX inside Python comments only."""
    violations: list[str] = []
    reader = io.StringIO(source).readline
    for tok in tokenize.generate_tokens(reader):
        if tok.type == tokenize.COMMENT:
            match = _MARKER_RE.search(tok.string)
            if match:
                violations.append(
                    f"{filename}:{tok.start[0]}: forbidden marker {match.group(1)!r} in comment"
                )
    return violations


def find_not_implemented_raises(source: str, filename: str = "<memory>") -> list[str]:
    """Return violations for any `raise NotImplementedError` statements."""
    tree = ast.parse(source, filename=filename)
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        exc = node.exc
        is_nie = (
            (isinstance(exc, ast.Name) and exc.id == "NotImplementedError")
            or (
                isinstance(exc, ast.Call)
                and isinstance(exc.func, ast.Name)
                and exc.func.id == "NotImplementedError"
            )
            or (
                isinstance(exc, ast.Attribute)
                and exc.attr == "NotImplementedError"
            )
            or (
                isinstance(exc, ast.Call)
                and isinstance(exc.func, ast.Attribute)
                and exc.func.attr == "NotImplementedError"
            )
        )
        if is_nie:
            violations.append(
                f"{filename}:{node.lineno}: forbidden raise NotImplementedError"
            )
    return violations


def _decorator_name(dec: ast.expr) -> str:
    if isinstance(dec, ast.Name):
        return dec.id
    if isinstance(dec, ast.Attribute):
        return dec.attr
    if isinstance(dec, ast.Call):
        return _decorator_name(dec.func)
    return ""


def _base_name(base: ast.expr) -> str:
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    if isinstance(base, ast.Subscript):
        return _base_name(base.value)
    return ""


def _is_ellipsis_expr(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and stmt.value.value is Ellipsis
    )


def _is_docstring_expr(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def find_empty_stub_functions(source: str, filename: str = "<memory>") -> list[str]:
    """Return violations for functions/methods whose body is only pass, ..., or docstring."""
    tree = ast.parse(source, filename=filename)
    violations: list[str] = []

    def _visit(node: ast.AST, class_stack: list[ast.ClassDef]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                _visit(child, [*class_stack, child])
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = list(child.body)
                if body and _is_docstring_expr(body[0]):
                    body = body[1:]

                is_stub = (len(body) == 0) or all(
                    isinstance(stmt, ast.Pass) or _is_ellipsis_expr(stmt)
                    for stmt in body
                )
                if is_stub:
                    decs = {_decorator_name(d) for d in child.decorator_list}
                    enclosing = class_stack[-1] if class_stack else None
                    base_names = (
                        {_base_name(b) for b in enclosing.bases}
                        if enclosing is not None
                        else set()
                    )
                    allowed = (
                        "overload" in decs
                        or "Protocol" in base_names
                        or (
                            "abstractmethod" in decs
                            and bool({"ABC", "ABCMeta"} & base_names)
                        )
                    )
                    if not allowed:
                        violations.append(
                            f"{filename}:{child.lineno}: empty stub function '{child.name}'"
                        )
                _visit(child, class_stack)
            else:
                _visit(child, class_stack)

    _visit(tree, [])
    return violations


def test_repo_has_python_files() -> None:
    files = collect_python_files()
    assert len(files) >= 6


def test_no_todo_or_fixme_comments() -> None:
    violations: list[str] = []
    for path in collect_python_files():
        source = path.read_text(encoding="utf-8")
        violations.extend(find_comment_markers(source, str(path)))
    assert violations == []


def test_no_not_implemented_error_raises() -> None:
    violations: list[str] = []
    for path in collect_python_files():
        source = path.read_text(encoding="utf-8")
        violations.extend(find_not_implemented_raises(source, str(path)))
    assert violations == []


def test_no_empty_stub_functions() -> None:
    violations: list[str] = []
    for path in collect_python_files():
        source = path.read_text(encoding="utf-8")
        violations.extend(find_empty_stub_functions(source, str(path)))
    assert violations == []


def test_hygiene_gate_catches_synthetic_violations() -> None:
    bad_comment = "x = 1  # " + "TO" + "DO: implement later\n"
    bad_fixme = "y = 2  # " + "FIX" + "ME: broken\n"
    assert len(find_comment_markers(bad_comment)) == 1
    assert len(find_comment_markers(bad_fixme)) == 1

    # String literals must NOT trigger false positives
    safe_string = 'pattern = "' + "TO" + 'DO"\n'
    assert find_comment_markers(safe_string) == []

    bad_nie = "def f():\n    raise NotImplementedError('nope')\n"
    assert len(find_not_implemented_raises(bad_nie)) == 1

    bad_pass = "def stub_a():\n    pass\n"
    bad_ellipsis = "def stub_b():\n    ...\n"
    bad_doc_only = 'def stub_c():\n    """Only a docstring."""\n'
    assert len(find_empty_stub_functions(bad_pass)) == 1
    assert len(find_empty_stub_functions(bad_ellipsis)) == 1
    assert len(find_empty_stub_functions(bad_doc_only)) == 1

    # Legitimate ABC abstractmethod and Protocol interfaces pass cleanly
    valid_abc = (
        "from abc import ABC, abstractmethod\n"
        "class Base(ABC):\n"
        "    @abstractmethod\n"
        "    def run(self) -> None: ...\n"
    )
    assert find_empty_stub_functions(valid_abc) == []
