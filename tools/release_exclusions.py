"""Read the exact path exclusions shared by release assembly and public checks.

Only literal paths and a trailing ``/**`` subtree rule are supported. Keep the
first matching rule so a refusal names the same exclusion the builder applied.
This module needs only the supplied repository's RELEASE-EXCLUDE.txt; publisher
scaffolding under work/ is deliberately absent from a public source snapshot.
"""
from __future__ import annotations

from pathlib import Path


def read_exclusions(repo_root: str | Path) -> list[str]:
    with (Path(repo_root) / "RELEASE-EXCLUDE.txt").open(encoding="utf-8") as stream:
        return [line for raw in stream if (line := raw.strip())
                and not line.startswith("#")]


def matches(rel: str, rules: list[str]) -> str | None:
    """Return the first rule matching a forward-slash relative path."""
    for rule in rules:
        if rule.endswith("/**"):
            root = rule[:-3]
            if rel == root or rel.startswith(root + "/"):
                return rule
        elif rel == rule:
            return rule
    return None
