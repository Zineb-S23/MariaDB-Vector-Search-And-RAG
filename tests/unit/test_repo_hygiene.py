"""Repository hygiene tests for the WikiRAG project.

Catch common submission mistakes without needing a running MariaDB server.
These static checks complement, but do not replace, review of Git changes.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
TEST_ROOT = PROJECT_ROOT / "tests"

REQUIRED_README_FIRST_LINE = (
    'A project for area 4, "Vector search and RAG, natively" '
    "for the [MariaDB student database projects, 2026-09]"
    "(https://mariadb.org/bachelor_hackathon_2026-09/)."
)

SECRET_NAME_PATTERN = re.compile(
    r"(?:^|_)(?:password|passwd|api_key|secret|access_token|auth_token)$",
    re.IGNORECASE,
)

CONFIG_SECRET_PATTERN = re.compile(
    r"^\s*(?:-\s*)?([A-Z][A-Z0-9_]*(?:PASSWORD|PASSWD|API_KEY|SECRET|TOKEN))"
    r"\s*[:=]\s*(.*?)\s*$"
)

SQL_START_PATTERN = re.compile(
    r"^\s*(?:SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|REPLACE)\b",
    re.IGNORECASE,
)

PLACEHOLDER_PREFIXES = (
    "replace_",
    "your_",
    "example",
    "placeholder",
    "dummy",
    "fake",
    "test",
    "unused",
    "changeme",
    "<",
    "$",
)


def _python_files() -> list[Path]:
    """Return source and test Python files in stable order.

    Returns:
        list[Path]: Python paths under src/ and tests/.
    """
    return sorted((*SOURCE_ROOT.rglob("*.py"), *TEST_ROOT.rglob("*.py")))


def _parse_python(path: Path) -> ast.Module:
    """Parse Python source without executing it.

    Args:
        path (Path): Source file to examine.

    Returns:
        ast.Module: Parsed syntax tree.
    """
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _is_placeholder(value: str) -> bool:
    """Identify empty or explicitly fake credential values.

    Args:
        value (str): Candidate credential value.

    Returns:
        bool: Whether the value is an acceptable placeholder.
    """
    normalized = value.strip().strip("\"'").lower()
    return not normalized or normalized.startswith(PLACEHOLDER_PREFIXES)


def _literal_string(node: ast.expr) -> str | None:
    """Read a literal string without evaluating arbitrary Python code.

    Args:
        node (ast.expr): Expression to inspect.

    Returns:
        str | None: String value when the expression is a literal.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value

    return None


def _sql_prefix(node: ast.expr) -> str:
    """Extract the literal beginning of a possibly dynamic SQL expression.

    Args:
        node (ast.expr): String expression or concatenation.

    Returns:
        str: Leading literal text, or an empty string if unavailable.
    """
    literal = _literal_string(node)

    if literal is not None:
        return literal

    if isinstance(node, ast.JoinedStr):
        return "".join(
            part.value
            for part in node.values
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
        )

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _sql_prefix(node.left)

    return ""


def test_env_file_is_ignored_by_git() -> None:
    """Require Git to ignore private .env files and not track one."""
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", ".env"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
    )

    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", ".env"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
    )

    assert ignored.returncode == 0, "Git must ignore .env"
    assert tracked.returncode != 0, ".env must not be tracked by Git"


def test_no_literal_credentials() -> None:
    """Reject obvious hardcoded credentials in code and deployment config."""
    violations: list[str] = []

    for path in _python_files():
        for node in ast.walk(_parse_python(path)):
            assignments: list[tuple[str, ast.expr]] = []

            if isinstance(node, ast.Assign):
                assignments.extend(
                    (target.id, node.value)
                    for target in node.targets
                    if isinstance(target, ast.Name)
                )

            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.value is not None
            ):
                assignments.append((node.target.id, node.value))

            for name, expression in assignments:
                value = _literal_string(expression)

                if (
                    SECRET_NAME_PATTERN.search(name)
                    and value is not None
                    and not _is_placeholder(value)
                ):
                    violations.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}: {name}")

    config_paths = (
        PROJECT_ROOT / ".env.example",
        PROJECT_ROOT / "docker-compose.yml",
        PROJECT_ROOT / ".github/workflows/ci.yml",
    )

    for path in config_paths:
        if not path.is_file():
            continue

        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            match = CONFIG_SECRET_PATTERN.match(line)

            if match is None:
                continue

            name, raw_value = match.groups()

            if not _is_placeholder(raw_value):
                violations.append(f"{path.relative_to(PROJECT_ROOT)}:{line_number}: {name}")

    assert not violations, "Potential hardcoded credentials:\n" + "\n".join(violations)


def test_no_sql_interpolation() -> None:
    """Flag SQL f-strings, percent-formatting and format() in application code."""
    violations: set[str] = set()

    for path in SOURCE_ROOT.rglob("*.py"):
        for node in ast.walk(_parse_python(path)):
            sql_text = ""

            if isinstance(node, ast.JoinedStr):
                sql_text = _sql_prefix(node)

            elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mod, ast.Add)):
                sql_text = _sql_prefix(node.left)

            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "format"
            ):
                sql_text = _sql_prefix(node.func.value)

            if sql_text and SQL_START_PATTERN.search(sql_text):
                violations.add(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}")

    assert not violations, "Potential interpolated SQL:\n" + "\n".join(sorted(violations))


def test_no_debug_flags() -> None:
    """Reject enabled debug flags in application source code."""
    violations: set[str] = set()

    for path in SOURCE_ROOT.rglob("*.py"):
        for node in ast.walk(_parse_python(path)):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]

                for target in targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id.upper() == "DEBUG"
                        and isinstance(node.value, ast.Constant)
                        and node.value.value is True
                    ):
                        violations.add(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}")

            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if (
                        keyword.arg in {"debug", "echo"}
                        and isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is True
                    ):
                        violations.add(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}")

    assert not violations, "Debug flags enabled:\n" + "\n".join(sorted(violations))


def test_no_empty_tests() -> None:
    """Reject test files with no tests or tests containing only placeholders."""
    violations: list[str] = []

    for path in sorted(TEST_ROOT.rglob("test_*.py")):
        test_functions = [
            node
            for node in ast.walk(_parse_python(path))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
        ]

        if not test_functions:
            violations.append(f"{path.relative_to(PROJECT_ROOT)}: no tests")
            continue

        for function in test_functions:
            # Exclude the function's docstring before checking its body.
            body = [
                statement
                for statement in function.body
                if not (
                    isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)
                )
            ]

            if not body:
                violations.append(f"{path.name}:{function.lineno}")
                continue

            if len(body) != 1:
                continue

            statement = body[0]

            is_placeholder = (
                isinstance(statement, ast.Pass)
                or (
                    isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Constant)
                    and statement.value.value is Ellipsis
                )
                or (
                    isinstance(statement, ast.Assert)
                    and isinstance(statement.test, ast.Constant)
                    and statement.test.value is True
                )
            )

            if is_placeholder:
                violations.append(f"{path.name}:{function.lineno}")

    assert not violations, "Empty or placeholder tests:\n" + "\n".join(violations)


def test_readme_first_line() -> None:
    """Require the exact Area 4 identifier as README.md's first line."""
    readme_path = PROJECT_ROOT / "README.md"

    assert readme_path.is_file(), "README.md is missing"

    lines = readme_path.read_text(encoding="utf-8").splitlines()

    assert lines, "README.md is empty"

    assert lines[0] == REQUIRED_README_FIRST_LINE, (
        "README.md must start with the exact Area 4 line required by the brief."
    )
