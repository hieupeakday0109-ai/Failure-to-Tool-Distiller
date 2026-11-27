"""Static AST security, typing, docstring, and zero-stub sanitizer for forged Skills."""

from __future__ import annotations

import ast
import keyword
import re
from collections.abc import Iterable

from scar_forge.models import ASTValidationResult

_SNAKE_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]*$")

ALLOWED_MODULES: frozenset[str] = frozenset(
    {
        "typing",
        "collections",
        "collections.abc",
        "math",
        "cmath",
        "statistics",
        "decimal",
        "fractions",
        "random",
        "re",
        "json",
        "csv",
        "io",
        "datetime",
        "time",
        "calendar",
        "itertools",
        "functools",
        "operator",
        "string",
        "textwrap",
        "unicodedata",
        "html",
        "html.parser",
        "urllib.parse",
        "pathlib",
        "dataclasses",
        "enum",
        "copy",
        "bisect",
        "heapq",
        "hashlib",
        "hmac",
        "base64",
        "uuid",
        "pydantic",
        "pandas",
        "numpy",
        "dateutil",
    }
)

BLOCKED_BUILTINS: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "breakpoint",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "dir",
        "__builtins__",
        "__loader__",
        "__spec__",
        "__package__",
        "__name__",
        "memoryview",
    }
)

BLOCKED_PATHLIB_MEMBERS: frozenset[str] = frozenset(
    {
        "Path",
        "PosixPath",
        "WindowsPath",
        "open",
        "read_text",
        "read_bytes",
        "write_text",
        "write_bytes",
        "unlink",
        "mkdir",
        "rmdir",
        "glob",
        "rglob",
        "iterdir",
        "home",
        "cwd",
        "chmod",
        "rename",
        "replace",
        "touch",
        "symlink_to",
        "hardlink_to",
    }
)


def _is_module_allowed(mod_name: str, extra_allowed: frozenset[str]) -> bool:
    if not mod_name:
        return False
    allowed = ALLOWED_MODULES | extra_allowed
    if mod_name in allowed:
        return True
    parts = mod_name.split(".")
    # Allow safe submodules of already-allowlisted packages (e.g. dateutil.parser)
    for i in range(len(parts), 0, -1):
        prefix = ".".join(parts[:i])
        if prefix in allowed and prefix not in {"urllib", "html"}:
            return True
    return False


def _is_ellipsis(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and stmt.value.value is Ellipsis
    )


def _is_docstring(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _is_not_implemented_raise(stmt: ast.stmt) -> bool:
    if not isinstance(stmt, ast.Raise) or stmt.exc is None:
        return False
    exc = stmt.exc
    if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
        return True
    return (
        isinstance(exc, ast.Call)
        and isinstance(exc.func, ast.Name)
        and exc.func.id == "NotImplementedError"
    )


class ASTGuard:
    """Strict AST sanitizer enforcing allowlisted imports, no escapes, typing, and docstrings."""

    def __init__(self, *, extra_allowed_modules: Iterable[str] = ()) -> None:
        self.extra_allowed = frozenset(extra_allowed_modules)

    def validate(
        self, source: str, expected_name: str | None = None
    ) -> ASTValidationResult:
        """Validate forged skill Python source code statically via AST."""
        if not isinstance(source, str) or not source.strip():
            return ASTValidationResult(
                valid=False, violations=["Source code is empty or not a string"]
            )

        try:
            tree = ast.parse(source, filename="<skill>")
        except SyntaxError as exc:
            return ASTValidationResult(
                valid=False,
                violations=[f"SyntaxError at line {exc.lineno}: {exc.msg}"],
            )

        violations: list[str] = []
        imports_used: list[str] = []
        top_level_fns: list[ast.FunctionDef] = []

        # Validate module top-level structure (imports must only appear at module top-level)
        for idx, stmt in enumerate(tree.body):
            if idx == 0 and _is_docstring(stmt):
                continue
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    if not _is_module_allowed(alias.name, self.extra_allowed):
                        violations.append(
                            f"Blocked import '{alias.name}' at line {stmt.lineno}"
                        )
                    else:
                        imports_used.append(alias.name)
            elif isinstance(stmt, ast.ImportFrom):
                if stmt.level and stmt.level > 0:
                    violations.append(
                        f"Relative import not allowed at line {stmt.lineno}"
                    )
                    continue
                mod = stmt.module or ""
                if not _is_module_allowed(mod, self.extra_allowed):
                    violations.append(
                        f"Blocked import from '{mod}' at line {stmt.lineno}"
                    )
                else:
                    if mod == "pathlib":
                        for alias in stmt.names:
                            if alias.name == "*" or alias.name in BLOCKED_PATHLIB_MEMBERS:
                                violations.append(
                                    f"Blocked pathlib member '{alias.name}' at line {stmt.lineno}"
                                )
                    imports_used.append(mod)
            elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                continue
            elif isinstance(stmt, ast.FunctionDef):
                top_level_fns.append(stmt)
            elif isinstance(stmt, ast.ClassDef):
                # Allow pure helper classes (e.g. HTMLParser subclass) if validated below
                continue
            else:
                violations.append(
                    f"Disallowed top-level statement {type(stmt).__name__} at line {stmt.lineno}"
                )

        # Deep walk across entire AST for nested imports, dangerous builtins, dunder escapes, stubs
        _branch_or_scope = (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.If,
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.With,
            ast.AsyncWith,
            ast.Try,
        )
        _blocked_path_attrs = BLOCKED_PATHLIB_MEMBERS - {"open", "rename", "replace", "touch"}
        for parent in ast.walk(tree):
            # Reject nested imports anywhere inside functions, classes, or branches
            if isinstance(parent, _branch_or_scope):
                for child in ast.iter_child_nodes(parent):
                    if isinstance(child, (ast.Import, ast.ImportFrom)):
                        violations.append(
                            f"Nested import disallowed at line {child.lineno}"
                        )
                # Catch deeper nested branches inside non-module scopes
                if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    for sub in ast.walk(parent):
                        if isinstance(sub, (ast.Import, ast.ImportFrom)):
                            violations.append(
                                f"Nested import inside {type(parent).__name__} at line {sub.lineno}"
                            )

            if isinstance(parent, ast.Name):
                if parent.id in BLOCKED_BUILTINS or (
                    parent.id.startswith("__") and parent.id.endswith("__")
                ):
                    violations.append(
                        f"Blocked identifier '{parent.id}' at line {parent.lineno}"
                    )

            elif isinstance(parent, ast.Attribute):
                if parent.attr in BLOCKED_BUILTINS or parent.attr in _blocked_path_attrs:
                    violations.append(
                        f"Blocked attribute '.{parent.attr}' at line {parent.lineno}"
                    )
                if parent.attr.startswith("__") and parent.attr.endswith("__"):
                    violations.append(
                        f"Blocked dunder attribute '.{parent.attr}' at line {parent.lineno}"
                    )

            elif isinstance(parent, ast.Call):
                # Allow super().__init__ only via normal helper methods; block builtins
                if (
                    isinstance(parent.func, ast.Name)
                    and parent.func.id in BLOCKED_BUILTINS
                ):
                    violations.append(
                        f"Blocked builtin call '{parent.func.id}()' at line {parent.lineno}"
                    )

            elif isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn_body = list(parent.body)
                if fn_body and _is_docstring(fn_body[0]):
                    fn_body = fn_body[1:]
                if not fn_body or all(
                    isinstance(s, ast.Pass) or _is_ellipsis(s) for s in fn_body
                ):
                    violations.append(
                        f"Stub function '{parent.name}' at line {parent.lineno}"
                    )
                elif any(_is_not_implemented_raise(s) for s in fn_body):
                    violations.append(
                        f"NotImplementedError stub in '{parent.name}' at line {parent.lineno}"
                    )

        # Target function verification
        target_fn: ast.FunctionDef | None = None
        if expected_name is not None:
            if not _SNAKE_NAME_RE.match(expected_name) or keyword.iskeyword(expected_name):
                violations.append(f"Invalid skill name {expected_name!r}")
            for fn in top_level_fns:
                if fn.name == expected_name:
                    target_fn = fn
                    break
            if target_fn is None:
                violations.append(
                    f"Expected top-level function '{expected_name}' not found"
                )
        elif top_level_fns:
            target_fn = top_level_fns[-1]
        else:
            violations.append("No top-level function definition found")

        param_names: list[str] = []
        has_types = False
        has_doc = False

        if target_fn is not None:
            if not _SNAKE_NAME_RE.match(target_fn.name) or keyword.iskeyword(target_fn.name):
                violations.append(f"Invalid function name {target_fn.name!r}")

            all_args = (
                list(target_fn.args.posonlyargs)
                + list(target_fn.args.args)
                + list(target_fn.args.kwonlyargs)
            )
            param_names = [a.arg for a in all_args]
            missing_ann = [a.arg for a in all_args if a.annotation is None]
            if target_fn.args.vararg and target_fn.args.vararg.annotation is None:
                missing_ann.append(f"*{target_fn.args.vararg.arg}")
            if target_fn.args.kwarg and target_fn.args.kwarg.annotation is None:
                missing_ann.append(f"**{target_fn.args.kwarg.arg}")
            if target_fn.returns is None:
                missing_ann.append("return")

            has_types = len(missing_ann) == 0
            if not has_types:
                violations.append(
                    f"Missing type annotations on '{target_fn.name}': {', '.join(missing_ann)}"
                )

            doc = ast.get_docstring(target_fn)
            has_doc = bool(doc and doc.strip())
            if not has_doc:
                violations.append(f"Missing docstring on '{target_fn.name}'")

        # Deduplicate preserving order
        dedup_violations = list(dict.fromkeys(violations))
        dedup_imports = list(dict.fromkeys(imports_used))
        return ASTValidationResult(
            valid=len(dedup_violations) == 0,
            violations=dedup_violations,
            function_name=target_fn.name if target_fn is not None else None,
            parameters=param_names,
            has_type_annotations=has_types,
            has_docstring=has_doc,
            imports_used=dedup_imports,
        )
