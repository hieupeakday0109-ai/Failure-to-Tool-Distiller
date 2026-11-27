"""Deterministic Zero-Prose local solver backend for offline execution & benchmarks."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from scar_forge.models import TaskPurpose


@dataclass(frozen=True, slots=True)
class _DomainSpec:
    domain_id: str
    skill_name: str
    tags: tuple[str, ...]
    param_schema: dict[str, str]
    docstring: str
    avoid_hint: str
    use_hint: str
    scar_keywords: tuple[str, ...]
    cold_buggy_code: str
    hardened_code: str
    skill_source: str
    default_skill_args: dict[str, Any]
    skill_test_cases: list[dict[str, Any]]


_DOMAIN_SPECS: dict[str, _DomainSpec] = {
    "pandas_concat": _DomainSpec(
        domain_id="pandas_concat",
        skill_name="merge_dataframe_records",
        tags=("pandas", "dataframe", "concat", "merge"),
        param_schema={"records": "list[dict[str, Any]]"},
        docstring="Concatenate record dicts into normalized rows using pd.concat.",
        avoid_hint="DataFrame.append",
        use_hint="pd.concat",
        scar_keywords=("append", "pd.concat", "dataframe"),
        cold_buggy_code=(
            "from typing import Any\n"
            "import pandas as pd\n\n"
            "def solve(records: list[dict[str, Any]]) -> list[dict[str, Any]]:\n"
            "    df = pd.DataFrame()\n"
            "    for row in records:\n"
            "        df = df.append(row, ignore_index=True)\n"
            "    return df.to_dict(orient='records')\n"
        ),
        hardened_code=(
            "from typing import Any\n"
            "import pandas as pd\n\n"
            "def solve(records: list[dict[str, Any]]) -> list[dict[str, Any]]:\n"
            "    if not records:\n"
            "        return []\n"
            "    frames = [pd.DataFrame([row]) for row in records if isinstance(row, dict)]\n"
            "    if not frames:\n"
            "        return []\n"
            "    merged = pd.concat(frames, ignore_index=True)\n"
            "    out: list[dict[str, Any]] = []\n"
            "    for row in merged.to_dict(orient='records'):\n"
            "        out.append({k: (None if pd.isna(v) else v) for k, v in row.items()})\n"
            "    return out\n"
        ),
        skill_source=(
            "from typing import Any\n"
            "import pandas as pd\n\n"
            "def merge_dataframe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:\n"
            '    """Concatenate record dicts into normalized rows using pd.concat."""\n'
            "    if not records:\n"
            "        return []\n"
            "    frames = [pd.DataFrame([row]) for row in records if isinstance(row, dict)]\n"
            "    if not frames:\n"
            "        return []\n"
            "    merged = pd.concat(frames, ignore_index=True)\n"
            "    out: list[dict[str, Any]] = []\n"
            "    for row in merged.to_dict(orient='records'):\n"
            "        out.append({k: (None if pd.isna(v) else v) for k, v in row.items()})\n"
            "    return out\n"
        ),
        default_skill_args={"records": [{"id": 1, "val": 10}, {"id": 2, "val": 20}]},
        skill_test_cases=[
            {
                "name": "merge_nominal",
                "kwargs": {"records": [{"id": 1, "val": 10}, {"id": 2, "val": 20}]},
                "expected": [{"id": 1, "val": 10}, {"id": 2, "val": 20}],
            },
            {"name": "merge_empty", "kwargs": {"records": []}, "expected": []},
        ],
    ),
    "iso_utc_delta": _DomainSpec(
        domain_id="iso_utc_delta",
        skill_name="compute_utc_deltas",
        tags=("datetime", "timezone", "utc", "iso"),
        param_schema={"timestamps": "list[str]"},
        docstring="Normalize mixed ISO timestamps to UTC and return sorted second deltas.",
        avoid_hint="naive datetime",
        use_hint="timezone.utc aware",
        scar_keywords=("naive", "offset-naive", "timezone.utc", "utc aware"),
        cold_buggy_code=(
            "from datetime import datetime\n\n"
            "def solve(timestamps: list[str]) -> list[float]:\n"
            "    parsed = [\n"
            "        datetime.fromisoformat(ts.replace('Z', '+00:00'))\n"
            "        for ts in timestamps\n"
            "    ]\n"
            "    parsed.sort()\n"
            "    return [(b - a).total_seconds() for a, b in zip(parsed, parsed[1:])]\n"
        ),
        hardened_code=(
            "from datetime import datetime, timezone\n\n"
            "def solve(timestamps: list[str]) -> list[float]:\n"
            "    if not timestamps:\n"
            "        return []\n"
            "    parsed: list[datetime] = []\n"
            "    for raw in timestamps:\n"
            "        if not isinstance(raw, str) or not raw.strip():\n"
            "            continue\n"
            "        dt = datetime.fromisoformat(raw.strip().replace('Z', '+00:00'))\n"
            "        if dt.tzinfo is None:\n"
            "            dt = dt.replace(tzinfo=timezone.utc)\n"
            "        else:\n"
            "            dt = dt.astimezone(timezone.utc)\n"
            "        parsed.append(dt)\n"
            "    if len(parsed) < 2:\n"
            "        return []\n"
            "    parsed.sort()\n"
            "    return [(b - a).total_seconds() for a, b in zip(parsed, parsed[1:])]\n"
        ),
        skill_source=(
            "from datetime import datetime, timezone\n\n"
            "def compute_utc_deltas(timestamps: list[str]) -> list[float]:\n"
            '    """Normalize mixed ISO timestamps to UTC and return sorted second deltas."""\n'
            "    if not timestamps:\n"
            "        return []\n"
            "    parsed: list[datetime] = []\n"
            "    for raw in timestamps:\n"
            "        if not isinstance(raw, str) or not raw.strip():\n"
            "            continue\n"
            "        dt = datetime.fromisoformat(raw.strip().replace('Z', '+00:00'))\n"
            "        if dt.tzinfo is None:\n"
            "            dt = dt.replace(tzinfo=timezone.utc)\n"
            "        else:\n"
            "            dt = dt.astimezone(timezone.utc)\n"
            "        parsed.append(dt)\n"
            "    if len(parsed) < 2:\n"
            "        return []\n"
            "    parsed.sort()\n"
            "    return [(b - a).total_seconds() for a, b in zip(parsed, parsed[1:])]\n"
        ),
        default_skill_args={
            "timestamps": [
                "2026-01-01T12:00:00Z",
                "2026-01-01T17:35:00+05:30",
                "2026-01-01T10:00:00",
            ]
        },
        skill_test_cases=[
            {
                "name": "utc_deltas_nominal",
                "kwargs": {
                    "timestamps": [
                        "2026-01-01T12:00:00Z",
                        "2026-01-01T17:35:00+05:30",
                        "2026-01-01T10:00:00",
                    ]
                },
                "expected": [7200.0, 300.0],
            },
            {"name": "utc_deltas_empty", "kwargs": {"timestamps": []}, "expected": []},
        ],
    ),
    "csv_bom_parse": _DomainSpec(
        domain_id="csv_bom_parse",
        skill_name="parse_bom_semicolon_csv",
        tags=("csv", "bom", "utf8", "delimiter"),
        param_schema={"raw_csv": "str"},
        docstring="Strip UTF-8 BOM, detect semicolon/comma delimiter, and normalize numeric cells.",
        avoid_hint="plain csv",
        use_hint="utf8sig strip BOM",
        scar_keywords=("bom", "utf-8-sig", "utf8sig", "semicolon", "csv"),
        cold_buggy_code=(
            "import csv\n"
            "import io\n"
            "from typing import Any\n\n"
            "def solve(raw_csv: str) -> list[dict[str, Any]]:\n"
            "    reader = csv.DictReader(io.StringIO(raw_csv))\n"
            "    rows: list[dict[str, Any]] = []\n"
            "    for row in reader:\n"
            "        rows.append({'id': row['id'], 'amount': float(row['amount'])})\n"
            "    return rows\n"
        ),
        hardened_code=(
            "import csv\n"
            "import io\n"
            "from typing import Any\n\n"
            "def solve(raw_csv: str) -> list[dict[str, Any]]:\n"
            "    cleaned = (raw_csv or '').lstrip('\\ufeff').strip()\n"
            "    if not cleaned:\n"
            "        return []\n"
            "    first_line = cleaned.splitlines()[0]\n"
            "    delim = ';' if first_line.count(';') >= first_line.count(',') else ','\n"
            "    reader = csv.DictReader(io.StringIO(cleaned), delimiter=delim)\n"
            "    num_cols = {'amount', 'price', 'total', 'score', 'qty'}\n"
            "    rows: list[dict[str, Any]] = []\n"
            "    for raw_row in reader:\n"
            "        norm_row: dict[str, Any] = {}\n"
            "        for key, val in raw_row.items():\n"
            "            if key is None:\n"
            "                continue\n"
            "            k = key.lstrip('\\ufeff').strip()\n"
            "            v = (val or '').strip()\n"
            "            if k in num_cols:\n"
            "                norm_row[k] = float(v) if v else None\n"
            "            else:\n"
            "                norm_row[k] = v\n"
            "        rows.append(norm_row)\n"
            "    return rows\n"
        ),
        skill_source=(
            "import csv\n"
            "import io\n"
            "from typing import Any\n\n"
            "def parse_bom_semicolon_csv(raw_csv: str) -> list[dict[str, Any]]:\n"
            '    """Strip UTF-8 BOM, detect delimiter, and normalize numeric cells."""\n'
            "    cleaned = (raw_csv or '').lstrip('\\ufeff').strip()\n"
            "    if not cleaned:\n"
            "        return []\n"
            "    first_line = cleaned.splitlines()[0]\n"
            "    delim = ';' if first_line.count(';') >= first_line.count(',') else ','\n"
            "    reader = csv.DictReader(io.StringIO(cleaned), delimiter=delim)\n"
            "    num_cols = {'amount', 'price', 'total', 'score', 'qty'}\n"
            "    rows: list[dict[str, Any]] = []\n"
            "    for raw_row in reader:\n"
            "        norm_row: dict[str, Any] = {}\n"
            "        for key, val in raw_row.items():\n"
            "            if key is None:\n"
            "                continue\n"
            "            k = key.lstrip('\\ufeff').strip()\n"
            "            v = (val or '').strip()\n"
            "            if k in num_cols:\n"
            "                norm_row[k] = float(v) if v else None\n"
            "            else:\n"
            "                norm_row[k] = v\n"
            "        rows.append(norm_row)\n"
            "    return rows\n"
        ),
        default_skill_args={"raw_csv": "\ufeffid;amount\n1;42.5\n2;\n"},
        skill_test_cases=[
            {
                "name": "csv_bom_nominal",
                "kwargs": {"raw_csv": "\ufeffid;amount\n1;42.5\n2;\n"},
                "expected": [
                    {"id": "1", "amount": 42.5},
                    {"id": "2", "amount": None},
                ],
            },
            {"name": "csv_bom_empty", "kwargs": {"raw_csv": ""}, "expected": []},
        ],
    ),
    "json_strict_rfc8259": _DomainSpec(
        domain_id="json_strict_rfc8259",
        skill_name="serialize_strict_json",
        tags=("json", "rfc8259", "decimal", "nan"),
        param_schema={"payload": "dict[str, Any]"},
        docstring="Serialize Decimal, datetime, and set to strict RFC 8259 JSON rejecting NaN/Inf.",
        avoid_hint="json nan",
        use_hint="strict allow_nan",
        scar_keywords=("allow_nan", "json nan", "rfc8259", "decimal"),
        cold_buggy_code=(
            "import json\n"
            "from typing import Any\n\n"
            "def solve(payload: dict[str, Any]) -> str:\n"
            "    return json.dumps(payload, sort_keys=True)\n"
        ),
        hardened_code=(
            "from datetime import date, datetime\n"
            "from decimal import Decimal\n"
            "import json\n"
            "import math\n"
            "from typing import Any\n\n"
            "def _sanitize_json_val(val: Any) -> Any:\n"
            "    if isinstance(val, float):\n"
            "        if math.isnan(val) or math.isinf(val):\n"
            "            raise ValueError('Non-finite float is not RFC 8259 compliant')\n"
            "        return val\n"
            "    if isinstance(val, Decimal):\n"
            "        if not val.is_finite():\n"
            "            raise ValueError('Non-finite Decimal is not RFC 8259 compliant')\n"
            "        return str(val)\n"
            "    if isinstance(val, (datetime, date)):\n"
            "        return val.isoformat()\n"
            "    if isinstance(val, set):\n"
            "        return [_sanitize_json_val(x) for x in sorted(val, key=repr)]\n"
            "    if isinstance(val, (list, tuple)):\n"
            "        return [_sanitize_json_val(x) for x in val]\n"
            "    if isinstance(val, dict):\n"
            "        return {str(k): _sanitize_json_val(v) for k, v in val.items()}\n"
            "    return val\n\n"
            "def solve(payload: dict[str, Any]) -> str:\n"
            "    clean = _sanitize_json_val(payload)\n"
            "    return json.dumps(clean, allow_nan=False, sort_keys=True)\n"
        ),
        skill_source=(
            "from datetime import date, datetime\n"
            "from decimal import Decimal\n"
            "import json\n"
            "import math\n"
            "from typing import Any\n\n"
            "def _sanitize_json_val(val: Any) -> Any:\n"
            '    """Recursively normalize Python objects for strict RFC 8259 JSON."""\n'
            "    if isinstance(val, float):\n"
            "        if math.isnan(val) or math.isinf(val):\n"
            "            raise ValueError('Non-finite float is not RFC 8259 compliant')\n"
            "        return val\n"
            "    if isinstance(val, Decimal):\n"
            "        if not val.is_finite():\n"
            "            raise ValueError('Non-finite Decimal is not RFC 8259 compliant')\n"
            "        return str(val)\n"
            "    if isinstance(val, (datetime, date)):\n"
            "        return val.isoformat()\n"
            "    if isinstance(val, set):\n"
            "        return [_sanitize_json_val(x) for x in sorted(val, key=repr)]\n"
            "    if isinstance(val, (list, tuple)):\n"
            "        return [_sanitize_json_val(x) for x in val]\n"
            "    if isinstance(val, dict):\n"
            "        return {str(k): _sanitize_json_val(v) for k, v in val.items()}\n"
            "    return val\n\n"
            "def serialize_strict_json(payload: dict[str, Any]) -> str:\n"
            '    """Serialize Decimal, datetime, and set to strict RFC 8259 JSON."""\n'
            "    clean = _sanitize_json_val(payload)\n"
            "    return json.dumps(clean, allow_nan=False, sort_keys=True)\n"
        ),
        default_skill_args={"payload": {"amount": "10.50", "ok": True}},
        skill_test_cases=[
            {
                "name": "strict_json_basic",
                "kwargs": {"payload": {"b": 2, "a": 1}},
                "expected": '{"a": 1, "b": 2}',
            },
            {
                "name": "strict_json_nan",
                "kwargs": {"payload": {"bad": float("nan")}},
                "expected_error": "ValueError",
            },
        ],
    ),
    "webhook_bool_coerce": _DomainSpec(
        domain_id="webhook_bool_coerce",
        skill_name="coerce_webhook_booleans",
        tags=("webhook", "bool", "coerce", "schema"),
        param_schema={"payload": "dict[str, Any]"},
        docstring="Strictly coerce webhook boolean flags avoiding bool('false') trap.",
        avoid_hint="bool(str)",
        use_hint="parse_bool_strict",
        scar_keywords=("bool(str)", "parse_bool_strict", "webhook"),
        cold_buggy_code=(
            "from typing import Any\n\n"
            "def solve(payload: dict[str, Any]) -> dict[str, bool]:\n"
            "    return {str(k): bool(v) for k, v in payload.items()}\n"
        ),
        hardened_code=(
            "from typing import Any\n\n"
            "def solve(payload: dict[str, Any]) -> dict[str, bool]:\n"
            "    if not isinstance(payload, dict):\n"
            "        return {}\n"
            "    truthy = {'true', '1', 'yes', 'on'}\n"
            "    out: dict[str, bool] = {}\n"
            "    for k, v in payload.items():\n"
            "        if isinstance(v, bool):\n"
            "            out[str(k)] = v\n"
            "        elif isinstance(v, (int, float)):\n"
            "            out[str(k)] = v != 0\n"
            "        else:\n"
            "            out[str(k)] = str(v).strip().lower() in truthy\n"
            "    return out\n"
        ),
        skill_source=(
            "from typing import Any\n\n"
            "def coerce_webhook_booleans(payload: dict[str, Any]) -> dict[str, bool]:\n"
            '    """Strictly coerce webhook boolean strings without bool(\'false\') trap."""\n'
            "    if not isinstance(payload, dict):\n"
            "        return {}\n"
            "    truthy = {'true', '1', 'yes', 'on'}\n"
            "    out: dict[str, bool] = {}\n"
            "    for k, v in payload.items():\n"
            "        if isinstance(v, bool):\n"
            "            out[str(k)] = v\n"
            "        elif isinstance(v, (int, float)):\n"
            "            out[str(k)] = v != 0\n"
            "        else:\n"
            "            out[str(k)] = str(v).strip().lower() in truthy\n"
            "    return out\n"
        ),
        default_skill_args={"payload": {"active": "false", "verified": "true"}},
        skill_test_cases=[
            {
                "name": "coerce_false_strings",
                "kwargs": {"payload": {"active": "false", "retry": "0", "ok": "TRUE"}},
                "expected": {"active": False, "retry": False, "ok": True},
            },
            {"name": "coerce_empty", "kwargs": {"payload": {}}, "expected": {}},
        ],
    ),
    "semver_numeric_sort": _DomainSpec(
        domain_id="semver_numeric_sort",
        skill_name="sort_semver_tags",
        tags=("semver", "version", "sort", "release"),
        param_schema={"versions": "list[str]"},
        docstring="Sort semantic version tags numerically with prerelease ordering.",
        avoid_hint="string sort semver",
        use_hint="int tuple key",
        scar_keywords=("semver", "int tuple", "numeric", "tuple key"),
        cold_buggy_code=(
            "def solve(versions: list[str]) -> list[str]:\n"
            "    return sorted(versions)\n"
        ),
        hardened_code=(
            "def _semver_key(tag: str) -> tuple[tuple[int, ...], int, str]:\n"
            "    core, _, pre = str(tag).strip().lstrip('vV').partition('-')\n"
            "    nums = tuple(int(p) if p.isdigit() else 0 for p in core.split('.') if p)\n"
            "    return (nums, 1 if not pre else 0, pre)\n\n"
            "def solve(versions: list[str]) -> list[str]:\n"
            "    if not versions:\n"
            "        return []\n"
            "    return sorted(versions, key=_semver_key)\n"
        ),
        skill_source=(
            "def _semver_key(tag: str) -> tuple[tuple[int, ...], int, str]:\n"
            '    """Compute numeric semver sort tuple with prerelease precedence."""\n'
            "    core, _, pre = str(tag).strip().lstrip('vV').partition('-')\n"
            "    nums = tuple(int(p) if p.isdigit() else 0 for p in core.split('.') if p)\n"
            "    return (nums, 1 if not pre else 0, pre)\n\n"
            "def sort_semver_tags(versions: list[str]) -> list[str]:\n"
            '    """Sort semantic version tags numerically with prerelease ordering."""\n'
            "    if not versions:\n"
            "        return []\n"
            "    return sorted(versions, key=_semver_key)\n"
        ),
        default_skill_args={"versions": ["1.10.0", "1.2.0", "1.2.0-alpha.1", "v2.0.0"]},
        skill_test_cases=[
            {
                "name": "semver_nominal",
                "kwargs": {"versions": ["1.10.0", "1.2.0", "1.2.0-alpha.1", "v2.0.0"]},
                "expected": ["1.2.0-alpha.1", "1.2.0", "1.10.0", "v2.0.0"],
            },
            {"name": "semver_empty", "kwargs": {"versions": []}, "expected": []},
        ],
    ),
    "html_text_extract": _DomainSpec(
        domain_id="html_text_extract",
        skill_name="extract_html_text",
        tags=("html", "parser", "text", "entities"),
        param_schema={"html_fragment": "str"},
        docstring="Extract plain text from HTML fragments using HTMLParser safely.",
        avoid_hint="regex strip html",
        use_hint="HTMLParser",
        scar_keywords=("htmlparser", "regex strip html", "regex html"),
        cold_buggy_code=(
            "import html\n"
            "import re\n\n"
            "def solve(html_fragment: str) -> str:\n"
            "    stripped = re.sub(r'<.*>', '', html_fragment)\n"
            "    return html.unescape(stripped).strip()\n"
        ),
        hardened_code=(
            "import html\n"
            "from html.parser import HTMLParser\n"
            "import re\n\n"
            "class _PlainExtractor(HTMLParser):\n"
            "    def handle_starttag(\n"
            "        self, tag: str, attrs: list[tuple[str, str | None]]\n"
            "    ) -> None:\n"
            "        if tag.lower() in {'script', 'style'}:\n"
            "            self.skip_depth += 1\n\n"
            "    def handle_endtag(self, tag: str) -> None:\n"
            "        if tag.lower() in {'script', 'style'} and self.skip_depth > 0:\n"
            "            self.skip_depth -= 1\n"
            "        else:\n"
            "            self.chunks.append(' ')\n\n"
            "    def handle_data(self, data: str) -> None:\n"
            "        if self.skip_depth == 0:\n"
            "            self.chunks.append(data)\n\n"
            "def solve(html_fragment: str) -> str:\n"
            "    if not isinstance(html_fragment, str) or not html_fragment.strip():\n"
            "        return ''\n"
            "    parser = _PlainExtractor()\n"
            "    parser.chunks = []\n"
            "    parser.skip_depth = 0\n"
            "    parser.feed(html_fragment)\n"
            "    parser.close()\n"
            "    joined = html.unescape(''.join(parser.chunks))\n"
            "    return re.sub(r'\\s+', ' ', joined).strip()\n"
        ),
        skill_source=(
            "import html\n"
            "from html.parser import HTMLParser\n"
            "import re\n\n"
            "class _PlainExtractor(HTMLParser):\n"
            "    def handle_starttag(\n"
            "        self, tag: str, attrs: list[tuple[str, str | None]]\n"
            "    ) -> None:\n"
            '        """Track skipped script/style blocks."""\n'
            "        if tag.lower() in {'script', 'style'}:\n"
            "            self.skip_depth += 1\n\n"
            "    def handle_endtag(self, tag: str) -> None:\n"
            '        """Preserve whitespace boundaries on closed HTML tags."""\n'
            "        if tag.lower() in {'script', 'style'} and self.skip_depth > 0:\n"
            "            self.skip_depth -= 1\n"
            "        else:\n"
            "            self.chunks.append(' ')\n\n"
            "    def handle_data(self, data: str) -> None:\n"
            '        """Collect visible text nodes outside skipped blocks."""\n'
            "        if self.skip_depth == 0:\n"
            "            self.chunks.append(data)\n\n"
            "def extract_html_text(html_fragment: str) -> str:\n"
            '    """Extract plain text from HTML fragments using HTMLParser safely."""\n'
            "    if not isinstance(html_fragment, str) or not html_fragment.strip():\n"
            "        return ''\n"
            "    parser = _PlainExtractor()\n"
            "    parser.chunks = []\n"
            "    parser.skip_depth = 0\n"
            "    parser.feed(html_fragment)\n"
            "    parser.close()\n"
            "    joined = html.unescape(''.join(parser.chunks))\n"
            "    return re.sub(r'\\s+', ' ', joined).strip()\n"
        ),
        default_skill_args={"html_fragment": "<p>Hello</p> <b>World</b>"},
        skill_test_cases=[
            {
                "name": "html_nominal",
                "kwargs": {
                    "html_fragment": "<p>Hello &amp; welcome</p> <b>World</b><script>bad()</script>"
                },
                "expected": "Hello & welcome World",
            },
            {"name": "html_empty", "kwargs": {"html_fragment": ""}, "expected": ""},
        ],
    ),
    "posix_path_guard": _DomainSpec(
        domain_id="posix_path_guard",
        skill_name="resolve_safe_posix_path",
        tags=("path", "posix", "security", "traversal"),
        param_schema={"base_dir": "str", "rel_path": "str"},
        docstring="Canonicalize relative POSIX path under base_dir blocking traversal.",
        avoid_hint="raw os.path.join",
        use_hint="PurePosixPath guard",
        scar_keywords=("pureposixpath", "traversal", "os.path.join", "path.join"),
        cold_buggy_code=(
            "def solve(base_dir: str, rel_path: str) -> str:\n"
            "    return base_dir.rstrip('/') + '/' + rel_path.lstrip('/')\n"
        ),
        hardened_code=(
            "from pathlib import PurePosixPath\n"
            "import re\n\n"
            "def solve(base_dir: str, rel_path: str) -> str:\n"
            "    if '\\x00' in base_dir or '\\x00' in rel_path:\n"
            "        raise ValueError('Null byte disallowed in path')\n"
            "    base_norm = str(PurePosixPath(base_dir.replace('\\\\', '/').strip() or '/'))\n"
            "    rel_norm = rel_path.replace('\\\\', '/').strip()\n"
            "    if not rel_norm:\n"
            "        return base_norm\n"
            "    if rel_norm.startswith('/') or re.match(r'^[A-Za-z]:', rel_norm):\n"
            "        raise ValueError('Absolute path traversal blocked')\n"
            "    stack: list[str] = []\n"
            "    for part in PurePosixPath(rel_norm).parts:\n"
            "        if part in ('', '.'):\n"
            "            continue\n"
            "        if part == '..':\n"
            "            if not stack:\n"
            "                raise ValueError('Path traversal blocked')\n"
            "            stack.pop()\n"
            "        else:\n"
            "            stack.append(part)\n"
            "    if not stack:\n"
            "        return base_norm\n"
            "    return f\"{base_norm.rstrip('/')}/{'/'.join(stack)}\"\n"
        ),
        skill_source=(
            "from pathlib import PurePosixPath\n"
            "import re\n\n"
            "def resolve_safe_posix_path(base_dir: str, rel_path: str) -> str:\n"
            '    """Canonicalize relative POSIX path under base_dir blocking traversal."""\n'
            "    if '\\x00' in base_dir or '\\x00' in rel_path:\n"
            "        raise ValueError('Null byte disallowed in path')\n"
            "    base_norm = str(PurePosixPath(base_dir.replace('\\\\', '/').strip() or '/'))\n"
            "    rel_norm = rel_path.replace('\\\\', '/').strip()\n"
            "    if not rel_norm:\n"
            "        return base_norm\n"
            "    if rel_norm.startswith('/') or re.match(r'^[A-Za-z]:', rel_norm):\n"
            "        raise ValueError('Absolute path traversal blocked')\n"
            "    stack: list[str] = []\n"
            "    for part in PurePosixPath(rel_norm).parts:\n"
            "        if part in ('', '.'):\n"
            "            continue\n"
            "        if part == '..':\n"
            "            if not stack:\n"
            "                raise ValueError('Path traversal blocked')\n"
            "            stack.pop()\n"
            "        else:\n"
            "            stack.append(part)\n"
            "    if not stack:\n"
            "        return base_norm\n"
            "    return f\"{base_norm.rstrip('/')}/{'/'.join(stack)}\"\n"
        ),
        default_skill_args={"base_dir": "/var/app/data", "rel_path": "reports/q1.csv"},
        skill_test_cases=[
            {
                "name": "posix_nested",
                "kwargs": {
                    "base_dir": "/var/app/data",
                    "rel_path": "reports\\2026\\./q1.csv",
                },
                "expected": "/var/app/data/reports/2026/q1.csv",
            },
            {
                "name": "posix_traversal",
                "kwargs": {"base_dir": "/var/app/data", "rel_path": "../../etc/passwd"},
                "expected_error": "ValueError",
            },
        ],
    ),
}


def detect_domain(text: str) -> str | None:
    """Identify one of the 8 known benchmark/application domains from prompt text."""
    low = text.lower()
    if any(k in low for k in ("pandas", "dataframe", "df.append", "pd.concat", "record dict")):
        return "pandas_concat"
    if any(
        k in low
        for k in (
            "timezone",
            "timestamps",
            "utc delta",
            "utc interval",
            "offset-naive",
            "offset-aware",
        )
    ):
        return "iso_utc_delta"
    if any(k in low for k in ("csv", "utf-8 bom", "utf-8-sig", "semicolon")):
        return "csv_bom_parse"
    if any(k in low for k in ("rfc 8259", "rfc8259", "allow_nan", "decimal", "rejecting nan")):
        return "json_strict_rfc8259"
    if any(k in low for k in ("webhook", "boolean", "parse_bool", "bool(", "'false'")):
        return "webhook_bool_coerce"
    if any(k in low for k in ("semver", "semantic version", "prerelease")):
        return "semver_numeric_sort"
    if any(k in low for k in ("html", "markup", "htmlparser", "entities")):
        return "html_text_extract"
    if any(k in low for k in ("posix", "traversal", "pureposixpath", "subpath")):
        return "posix_path_guard"
    return None


def _extract_section(full_text: str, header: str) -> str:
    pattern = re.compile(
        rf"\[{re.escape(header)}\](.*?)(?=\n\[[A-Z_ ]+\]|\Z)", re.DOTALL
    )
    match = pattern.search(full_text)
    return match.group(1).strip() if match else ""


class LocalSolver:
    """Deterministic structured Zero-Prose solver simulating real cold->scar->skill progression."""

    def solve(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        purpose: TaskPurpose,
        consecutive_failures: int = 0,
    ) -> str:
        """Emit strict single-object Zero-Prose JSON for the given prompt & TaskPurpose."""
        joined = "\n".join(str(m.get("content", "")) for m in messages)
        task_section = _extract_section(joined, "TASK") or joined
        scars_section = _extract_section(joined, "ACTIVE SCARS")
        skills_section = _extract_section(joined, "ACTIVE SKILLS")
        error_section = _extract_section(joined, "ERROR")
        args_section = _extract_section(joined, "TASK_ARGS")

        domain_id = detect_domain(task_section) or detect_domain(joined)
        spec = _DOMAIN_SPECS.get(domain_id) if domain_id else None

        if purpose == TaskPurpose.SCAR_DISTILLATION:
            if spec is not None:
                return json.dumps(
                    {
                        "avoid": spec.avoid_hint,
                        "use": spec.use_hint,
                        "tags": list(spec.tags),
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "avoid": "unguarded failure",
                    "use": "validate inputs",
                    "tags": ["general"],
                },
                ensure_ascii=False,
            )

        if purpose == TaskPurpose.SKILL_DISTILLATION:
            if spec is not None:
                return json.dumps(
                    {
                        "skill_name": spec.skill_name,
                        "source": spec.skill_source,
                        "docstring": spec.docstring,
                        "tags": list(spec.tags),
                        "param_schema": spec.param_schema,
                        "test_cases": spec.skill_test_cases,
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "skill_name": "echo_safe_payload",
                    "source": (
                        "def echo_safe_payload(value: str) -> str:\n"
                        '    """Return stripped string payload safely."""\n'
                        "    return str(value or '').strip()\n"
                    ),
                    "docstring": "Return stripped string payload safely.",
                    "tags": ["general"],
                    "param_schema": {"value": "str"},
                    "test_cases": [
                        {"kwargs": {"value": " ok "}, "expected": "ok"},
                        {"kwargs": {"value": ""}, "expected": ""},
                    ],
                },
                ensure_ascii=False,
            )

        if spec is None:
            return json.dumps(
                {
                    "action": "RUN_CODE",
                    "code": (
                        "def solve(*args, **kwargs):\n"
                        "    raise ValueError('Unsupported task domain in local solver')\n"
                    ),
                },
                ensure_ascii=False,
            )

        # Epoch 3 / Skill Reuse: invoke matching active skill directly
        if skills_section and spec.skill_name in skills_section:
            call_args = spec.default_skill_args
            if args_section:
                try:
                    parsed_args = json.loads(args_section)
                    if isinstance(parsed_args, dict):
                        call_args = parsed_args
                except json.JSONDecodeError:
                    pass
            return json.dumps(
                {
                    "action": "CALL_SKILL",
                    "skill_name": spec.skill_name,
                    "skill_args": call_args,
                },
                ensure_ascii=False,
            )

        has_matching_scar = bool(scars_section) and (
            spec.avoid_hint.lower() in scars_section.lower()
            or spec.use_hint.lower() in scars_section.lower()
            or any(kw in scars_section.lower() for kw in spec.scar_keywords)
        )
        has_error_feedback = (
            bool(error_section)
            or "[ERROR]" in joined
            or consecutive_failures > 0
            or purpose
            in {
                TaskPurpose.SYNTAX_REPAIR,
                TaskPurpose.TEST_LOOP,
                TaskPurpose.REASONING_RECOVERY,
            }
        )

        if has_matching_scar or has_error_feedback:
            return json.dumps(
                {"action": "RUN_CODE", "code": spec.hardened_code},
                ensure_ascii=False,
            )

        # Epoch 1 / Cold start: emit realistic naive implementation that triggers the pitfall
        return json.dumps(
            {"action": "RUN_CODE", "code": spec.cold_buggy_code},
            ensure_ascii=False,
        )
