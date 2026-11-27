"""Canonical 8-task benchmark catalog exercising real Python runtime pitfalls."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from scar_forge.models import TaskSpec

BENCHMARK_TASKS: list[TaskSpec] = [
    TaskSpec(
        task_id="pandas_concat",
        title="Pandas DataFrame Record Batch Concatenation",
        prompt="Merge list of record dicts into DataFrame rows safely using pandas",
        domain="pandas",
        tags=["pandas", "dataframe", "concat", "merge"],
        entry_fn="solve",
        param_schema={"records": "list[dict]"},
        test_cases=[
            {
                "name": "nominal_merge",
                "kwargs": {"records": [{"id": 1, "val": 10}, {"id": 2, "val": 20}]},
                "expected": [{"id": 1, "val": 10}, {"id": 2, "val": 20}],
            },
            {
                "name": "empty_records",
                "kwargs": {"records": []},
                "expected": [],
            },
        ],
        variant_prompt="Merge batch 2 list of record dicts into DataFrame rows",
        variant_test_cases=[
            {
                "name": "variant_regions",
                "kwargs": {
                    "records": [
                        {"region": "apac", "score": 9},
                        {"region": "emea", "score": 7},
                    ]
                },
                "expected": [
                    {"region": "apac", "score": 9},
                    {"region": "emea", "score": 7},
                ],
            }
        ],
        reuse_test_cases=[
            {
                "name": "reuse_batch",
                "kwargs": {"records": [{"k": "x", "n": 3}, {"k": "y", "n": 4}]},
                "expected": [{"k": "x", "n": 3}, {"k": "y", "n": 4}],
            }
        ],
    ),
    TaskSpec(
        task_id="iso_utc_delta",
        title="Mixed-Timezone ISO Timestamp UTC Delta Calculator",
        prompt=(
            "Parse mixed timezone timestamps (Z, +05:30, naive) into sorted UTC deltas"
        ),
        domain="datetime",
        tags=["datetime", "timezone", "utc", "iso"],
        entry_fn="solve",
        param_schema={"timestamps": "list[str]"},
        test_cases=[
            {
                "name": "mixed_tz_deltas",
                "kwargs": {
                    "timestamps": [
                        "2026-01-01T12:00:00Z",
                        "2026-01-01T17:35:00+05:30",
                        "2026-01-01T10:00:00",
                    ]
                },
                "expected": [7200.0, 300.0],
            },
            {
                "name": "empty_timestamps",
                "kwargs": {"timestamps": []},
                "expected": [],
            },
        ],
        variant_prompt=(
            "Compute sorted UTC intervals from mixed Z and offset timestamps"
        ),
        variant_test_cases=[
            {
                "name": "variant_tz",
                "kwargs": {
                    "timestamps": [
                        "2026-05-01T01:00:00Z",
                        "2026-05-01T00:00:00",
                        "2026-05-01T03:10:00+02:00",
                    ]
                },
                "expected": [3600.0, 600.0],
            }
        ],
        reuse_test_cases=[
            {
                "name": "reuse_tz",
                "kwargs": {
                    "timestamps": [
                        "2026-08-10T08:00:00-02:00",
                        "2026-08-10T09:00:00Z",
                    ]
                },
                "expected": [3600.0],
            }
        ],
    ),
    TaskSpec(
        task_id="csv_bom_parse",
        title="UTF-8 BOM & Semicolon CSV Normalizer",
        prompt=(
            "Parse CSV payload with UTF-8 BOM, semicolon delimiter, and blank numeric cells"
        ),
        domain="csv",
        tags=["csv", "bom", "utf8", "delimiter"],
        entry_fn="solve",
        param_schema={"raw_csv": "str"},
        test_cases=[
            {
                "name": "bom_semicolon_blank",
                "kwargs": {"raw_csv": "\ufeffid;amount\n1;42.5\n2;\n"},
                "expected": [
                    {"id": "1", "amount": 42.5},
                    {"id": "2", "amount": None},
                ],
            },
            {
                "name": "empty_csv",
                "kwargs": {"raw_csv": ""},
                "expected": [],
            },
        ],
        variant_prompt="Normalize another BOM-prefixed semicolon CSV table with empty cells",
        variant_test_cases=[
            {
                "name": "variant_csv",
                "kwargs": {"raw_csv": "\ufeffname;amount\nalpha;10\nbeta; \n"},
                "expected": [
                    {"name": "alpha", "amount": 10.0},
                    {"name": "beta", "amount": None},
                ],
            }
        ],
        reuse_test_cases=[
            {
                "name": "reuse_csv",
                "kwargs": {"raw_csv": "\ufeffsku;amount\nA1;7.25\nA2;\n"},
                "expected": [
                    {"sku": "A1", "amount": 7.25},
                    {"sku": "A2", "amount": None},
                ],
            }
        ],
    ),
    TaskSpec(
        task_id="json_strict_rfc8259",
        title="Strict RFC 8259 JSON Serializer",
        prompt=(
            "Serialize Decimal, datetime, and set to strict RFC 8259 JSON rejecting NaN"
        ),
        domain="json",
        tags=["json", "rfc8259", "decimal", "nan"],
        entry_fn="solve",
        param_schema={"payload": "dict"},
        test_cases=[
            {
                "name": "decimal_dt_set",
                "kwargs": {
                    "payload": {
                        "price": Decimal("19.95"),
                        "tags": {"b", "a"},
                        "ts": datetime(2026, 1, 1, tzinfo=UTC),
                    }
                },
                "expected": (
                    '{"price": "19.95", "tags": ["a", "b"], '
                    '"ts": "2026-01-01T00:00:00+00:00"}'
                ),
            },
            {
                "name": "reject_nan",
                "kwargs": {"payload": {"bad": float("nan")}},
                "expected_error": "ValueError",
            },
        ],
        variant_prompt="Strict RFC 8259 JSON serialization for Decimal and set objects",
        variant_test_cases=[
            {
                "name": "variant_decimal",
                "kwargs": {
                    "payload": {"total": Decimal("5.50"), "items": {"z", "y"}}
                },
                "expected": '{"items": ["y", "z"], "total": "5.50"}',
            },
            {
                "name": "variant_inf",
                "kwargs": {"payload": {"x": float("inf")}},
                "expected_error": "ValueError",
            },
        ],
        reuse_test_cases=[
            {
                "name": "reuse_strict",
                "kwargs": {"payload": {"amount": Decimal("100.00"), "flags": {"ok"}}},
                "expected": '{"amount": "100.00", "flags": ["ok"]}',
            }
        ],
    ),
    TaskSpec(
        task_id="webhook_bool_coerce",
        title="Strict Webhook Boolean Coercion",
        prompt=(
            "Coerce webhook JSON payload with string booleans like 'false' and '0'"
        ),
        domain="webhook",
        tags=["webhook", "bool", "coerce", "schema"],
        entry_fn="solve",
        param_schema={"payload": "dict"},
        test_cases=[
            {
                "name": "string_bools",
                "kwargs": {
                    "payload": {
                        "active": "false",
                        "retry": "0",
                        "verified": "TRUE",
                        "enabled": True,
                    }
                },
                "expected": {
                    "active": False,
                    "retry": False,
                    "verified": True,
                    "enabled": True,
                },
            },
            {
                "name": "empty_payload",
                "kwargs": {"payload": {}},
                "expected": {},
            },
        ],
        variant_prompt="Coerce second webhook payload containing 'false' boolean flags",
        variant_test_cases=[
            {
                "name": "variant_bools",
                "kwargs": {
                    "payload": {"dry_run": "no", "notify": "yes", "force": "False"}
                },
                "expected": {"dry_run": False, "notify": True, "force": False},
            }
        ],
        reuse_test_cases=[
            {
                "name": "reuse_bools",
                "kwargs": {"payload": {"ok": "1", "skip": "off"}},
                "expected": {"ok": True, "skip": False},
            }
        ],
    ),
    TaskSpec(
        task_id="semver_numeric_sort",
        title="Semantic Version Numeric & Prerelease Sorter",
        prompt="Sort semantic version tags numerically with prerelease ordering",
        domain="semver",
        tags=["semver", "version", "sort", "release"],
        entry_fn="solve",
        param_schema={"versions": "list[str]"},
        test_cases=[
            {
                "name": "semver_prerelease",
                "kwargs": {
                    "versions": ["1.10.0", "1.2.0", "1.2.0-alpha.1", "v2.0.0"]
                },
                "expected": ["1.2.0-alpha.1", "1.2.0", "1.10.0", "v2.0.0"],
            },
            {
                "name": "empty_versions",
                "kwargs": {"versions": []},
                "expected": [],
            },
        ],
        variant_prompt="Order numeric semver release tags including minor >= 10",
        variant_test_cases=[
            {
                "name": "variant_semver",
                "kwargs": {"versions": ["2.10.1", "2.2.0", "1.9.10"]},
                "expected": ["1.9.10", "2.2.0", "2.10.1"],
            }
        ],
        reuse_test_cases=[
            {
                "name": "reuse_semver",
                "kwargs": {"versions": ["0.12.0", "0.2.0", "0.2.0-rc.1"]},
                "expected": ["0.2.0-rc.1", "0.2.0", "0.12.0"],
            }
        ],
    ),
    TaskSpec(
        task_id="html_text_extract",
        title="HTML Fragment Plain-Text Extractor",
        prompt=(
            "Extract clean plain text from HTML fragments while decoding entities safely"
        ),
        domain="html",
        tags=["html", "parser", "text", "entities"],
        entry_fn="solve",
        param_schema={"html_fragment": "str"},
        test_cases=[
            {
                "name": "multi_tag_entities",
                "kwargs": {
                    "html_fragment": (
                        "<p>Hello &amp; welcome</p> <b>World</b><script>bad()</script>"
                    )
                },
                "expected": "Hello & welcome World",
            },
            {
                "name": "empty_html",
                "kwargs": {"html_fragment": ""},
                "expected": "",
            },
        ],
        variant_prompt="Strip tags and decode entities from nested HTML markup",
        variant_test_cases=[
            {
                "name": "variant_html",
                "kwargs": {"html_fragment": "<div>Alpha</div><span>Beta &lt;3&gt;</span>"},
                "expected": "Alpha Beta <3>",
            }
        ],
        reuse_test_cases=[
            {
                "name": "reuse_html",
                "kwargs": {"html_fragment": "<h1>Scar</h1><p>Forge</p>"},
                "expected": "Scar Forge",
            }
        ],
    ),
    TaskSpec(
        task_id="posix_path_guard",
        title="Safe Canonical POSIX Path Traversal Guard",
        prompt=(
            "Resolve relative POSIX paths under a base directory blocking '..' traversal"
        ),
        domain="path",
        tags=["path", "posix", "security", "traversal"],
        entry_fn="solve",
        param_schema={"base_dir": "str", "rel_path": "str"},
        test_cases=[
            {
                "name": "nested_win_sep",
                "kwargs": {
                    "base_dir": "/var/app/data",
                    "rel_path": "reports\\2026\\./q1.csv",
                },
                "expected": "/var/app/data/reports/2026/q1.csv",
            },
            {
                "name": "block_traversal",
                "kwargs": {"base_dir": "/var/app/data", "rel_path": "../../etc/passwd"},
                "expected_error": "ValueError",
            },
            {
                "name": "empty_rel",
                "kwargs": {"base_dir": "/var/app/data", "rel_path": ""},
                "expected": "/var/app/data",
            },
        ],
        variant_prompt="Canonicalize safe relative POSIX subpath while rejecting traversal",
        variant_test_cases=[
            {
                "name": "variant_safe",
                "kwargs": {"base_dir": "/srv/files", "rel_path": "logs/./app.log"},
                "expected": "/srv/files/logs/app.log",
            },
            {
                "name": "variant_escape",
                "kwargs": {"base_dir": "/srv/files", "rel_path": "../secret.key"},
                "expected_error": "ValueError",
            },
        ],
        reuse_test_cases=[
            {
                "name": "reuse_safe",
                "kwargs": {"base_dir": "/opt/bucket", "rel_path": "sub/item.json"},
                "expected": "/opt/bucket/sub/item.json",
            }
        ],
    ),
]


def get_benchmark_tasks() -> list[TaskSpec]:
    """Return a defensive copy of the canonical 8-task benchmark suite."""
    return [t.model_copy(deep=True) for t in BENCHMARK_TASKS]
