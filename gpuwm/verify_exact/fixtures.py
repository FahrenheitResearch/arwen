"""Per-scheme column fixtures: a scheme's oracle inputs and outputs, scored bitwise.

A fixtures root holds one directory per fixture::

    <root>/<scheme>/<fixture-id>/
        fixture.json    what to run and where the oracle came from
        inputs.npz      the arrays handed to the runner, by name
        expected.npz    the oracle's outputs, by name (the words WRF wrote)

``fixture.json``::

    {
      "schema": "gpuwm-verify-exact-fixture-v1",
      "scheme": "wsm6",                       # free text, groups the report
      "runner": "package.module:function",    # WOOF's side, see below
      "options": {...},                       # optional keyword arguments for the runner
      "compare": ["QV", "QC", ...],           # optional; default every array in expected.npz
      "steps": 1,                             # optional, informational
      "reference": {                          # provenance, copied into the report
        "build": "WRF v4.6.1 d66e442, gfortran 13.3 -O2 -fno-fast-math -ffp-contract=off",
        "glibc": "2.39", "source": "tools/<scheme>_oracle/..."
      }
    }

The runner is any importable callable ``runner(inputs, **options)`` that
takes ``{name: numpy array}`` and returns ``{name: numpy array}`` holding at
least every compared name.  A lane exposes its kernel's existing column
entry point under that signature (usually a thin function beside its oracle
validator) and drops fixtures in; the command needs no new code per scheme.

Fixtures run in the same process as the combo replays, so they inherit its
arithmetic (``GPUWM_WRF_EXACT=1`` for the strict build).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from gpuwm.verify_exact.compare import UlpDetail, field_digest, ulp_detail

FIXTURE_SCHEMA = "gpuwm-verify-exact-fixture-v1"


@dataclass(frozen=True)
class Fixture:
    scheme: str
    ident: str
    directory: Path
    runner: str
    options: Mapping[str, object]
    compare: tuple[str, ...]
    reference: Mapping[str, object]

    @property
    def key(self) -> str:
        return f"{self.scheme}/{self.ident}"


@dataclass
class FixtureResult:
    fixture: Fixture
    outcome: str  # PASS, FAIL, ERROR
    differing: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    details: list[UlpDetail] = field(default_factory=list)
    error: str = ""

    def line(self) -> str:
        head = f"fixture {self.fixture.key:<40} {self.outcome}"
        if self.outcome == "PASS":
            return head + f"  {len(self.fixture.compare) or 'all'} fields 0 ULP"
        if self.error:
            return head + f"  {self.error}"
        parts = []
        if self.differing:
            lead = self.details[0] if self.details else None
            text = f"{self.differing[0]}"
            if len(self.differing) > 1:
                text += f" (+{len(self.differing) - 1} more)"
            if lead is not None:
                text += " max " + lead.describe()
            parts.append(text)
        if self.missing:
            parts.append("runner did not return " + ", ".join(self.missing))
        return head + "  " + "; ".join(parts)

    def as_json(self) -> dict:
        return {"fixture": self.fixture.key, "scheme": self.fixture.scheme,
                "outcome": self.outcome, "differing": self.differing,
                "missing": self.missing, "error": self.error,
                "reference": dict(self.fixture.reference),
                "details": [{"field": d.field, "max_ulp": d.max_ulp,
                             "differing_words": d.differing_words, "words": d.words,
                             "first_index": list(d.first[0])} for d in self.details]}


def discover_fixtures(root: Path) -> list[Fixture]:
    """Every ``fixture.json`` under ``root``, in path order.  Malformed ones raise."""

    fixtures = []
    for path in sorted(Path(root).glob("*/*/fixture.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("schema") != FIXTURE_SCHEMA:
            raise ValueError(f"{path}: schema {document.get('schema')!r}, expected {FIXTURE_SCHEMA!r}")
        runner = document.get("runner")
        if not isinstance(runner, str) or ":" not in runner:
            raise ValueError(f"{path}: runner must be 'package.module:function'")
        for name in ("inputs.npz", "expected.npz"):
            if not (path.parent / name).is_file():
                raise ValueError(f"{path.parent}: missing {name}")
        fixtures.append(Fixture(
            scheme=str(document.get("scheme") or path.parent.parent.name),
            ident=path.parent.name, directory=path.parent, runner=runner,
            options=dict(document.get("options") or {}),
            compare=tuple(document.get("compare") or ()),
            reference=dict(document.get("reference") or {})))
    return fixtures


def _resolve(runner: str):
    module, _, name = runner.partition(":")
    target = importlib.import_module(module)
    for part in name.split("."):
        target = getattr(target, part)
    return target


def _host(values) -> np.ndarray:
    getter = getattr(values, "get", None)
    if callable(getter) and not isinstance(values, np.ndarray):
        values = getter()  # a device array
    return np.asarray(values)


def run_fixture(fixture: Fixture) -> FixtureResult:
    """Run one fixture and compare every expected array word for word."""

    with np.load(fixture.directory / "inputs.npz") as stored:
        inputs = {name: stored[name] for name in stored.files}
    with np.load(fixture.directory / "expected.npz") as stored:
        expected = {name: stored[name] for name in stored.files}
    names = fixture.compare or tuple(expected)
    unknown = [n for n in names if n not in expected]
    if unknown:
        return FixtureResult(fixture, "ERROR", error=f"compare names {unknown} are not in expected.npz")
    try:
        produced = _resolve(fixture.runner)(inputs, **fixture.options)
    except Exception as error:  # noqa: BLE001 - one fixture's failure is its own row
        return FixtureResult(fixture, "ERROR", error=f"{type(error).__name__}: {error}"[:400])
    result = FixtureResult(fixture, "PASS")
    for name in names:
        if name not in produced:
            result.missing.append(name)
            continue
        want = expected[name]
        got = _host(produced[name])
        if field_digest(got, want.dtype.str, want.shape) == field_digest(want, want.dtype.str, want.shape):
            continue
        result.differing.append(name)
        if got.shape == want.shape:
            result.details.append(ulp_detail(name, 0, got, want))
    if result.differing or result.missing:
        result.outcome = "FAIL"
    return result
