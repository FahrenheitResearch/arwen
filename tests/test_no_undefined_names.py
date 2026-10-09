"""No name is used that nothing defines: the undefined-name gate.

THE BREAKAGE IT PREVENTS: 2.8.8's candidate crashed every real forecast at
model step 1 with ``NameError: name 'WRF_DIFFUSION' is not defined``
(``gpuwm/core/dycore.py``).  An intake commit used the name and never
defined it, on a GPU-only path that no CPU test reaches, and nobody ran a
forecast before handoff.  The same sweep found older members of the class
that had shipped in 2.8.7: ``gpuwm-wrf-runtime-check`` died on
``NameError: name 'parser'`` instead of printing its usage error, and a
multi-card ``[devices]`` tree that did not fit died on
``NameError: name 'DevicesRefused'`` instead of the named memory refusal.

A static undefined-name check finds the whole class in seconds, on any
box, whether or not a CPU test can execute the line.  This file runs
ruff's three undefined-name rules over every Python tree the repository
owns and fails on any finding:

* F821 -- a name read that no scope binds (the step-1 crash);
* F822 -- an ``__all__`` entry naming nothing, which breaks
  ``from module import *`` and every documentation tool;
* F823 -- a local read before its assignment in the same scope.

Vendored third-party trees (``*/vendor/*``) are excluded: they are other
projects' sources carried byte-for-byte, and editing them would break
their checksums.

An ``__all__`` entry served by a module-level ``__getattr__`` (the lazy
cargo build hints, ``doctor.DOCTOR_SOURCES``) is invisible to a static
scan and carries ``# noqa: F822``.  A suppression is a claim, so
:func:`test_every_f822_suppression_resolves_at_import` imports each such
module and proves every name in its ``__all__`` resolves and that
``from module import *`` works.
"""

from __future__ import annotations

import importlib
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Every Python tree the repository owns.  ``tools`` and ``tilestream`` are
#: checkout material (benches, oracles, harnesses), but the same NameError
#: there wastes a card hour just as surely.
SCANNED_TREES = ("gpuwm", "tools", "tilestream")

RULES = "F821,F822,F823"


def _ruff_binary() -> str:
    try:
        from ruff.__main__ import find_ruff_bin
    except ImportError:  # pragma: no cover - environment defect, not a skip
        pytest.fail(
            "ruff is not installed in this environment; it is a declared "
            "[dev] dependency (pyproject.toml).  Install with "
            "`pip install -e .[dev]`.  This gate is not skippable: the "
            "defect class it catches crashed 2.8.8's candidate at step 1.")
    return str(find_ruff_bin())


def _findings() -> tuple[list[str], str]:
    trees = [name for name in SCANNED_TREES if (REPO_ROOT / name).is_dir()]
    assert "gpuwm" in trees, f"no gpuwm/ under {REPO_ROOT}"
    completed = subprocess.run(
        [_ruff_binary(), "check", "--isolated", "--no-cache",
         "--select", RULES, "--extend-exclude", "vendor",
         "--output-format", "concise", *trees],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    lines = [line.strip() for line in completed.stdout.splitlines()
             if re.search(r"\bF82[123]\b", line)]
    if completed.returncode not in (0, 1) or (completed.returncode == 1
                                              and not lines):
        pytest.fail(f"ruff did not run cleanly (exit {completed.returncode})"
                    f":\n{completed.stdout}\n{completed.stderr}")
    return lines, completed.stdout


def test_no_undefined_names_in_any_owned_tree():
    """Zero F821/F822/F823 findings across gpuwm/, tools/ and tilestream/."""

    lines, _ = _findings()
    assert not lines, (
        f"{len(lines)} undefined-name finding(s).  Each is a NameError "
        "waiting for the first run that reaches the line (2.8.8's candidate "
        "died at model step 1 on one).  Define or import the name; for an "
        "__all__ entry served by a module __getattr__, add "
        "`# noqa: F822 -- module __getattr__` on that line:\n  "
        + "\n  ".join(lines))


def _f822_suppressed_modules() -> list[str]:
    modules = []
    for path in sorted((REPO_ROOT / "gpuwm").rglob("*.py")):
        if "noqa: F822" in path.read_text(encoding="utf-8"):
            relative = path.relative_to(REPO_ROOT).with_suffix("")
            parts = relative.parts
            if parts[-1] == "__init__":
                parts = parts[:-1]
            modules.append(".".join(parts))
    return modules


def test_the_f822_suppressions_are_found():
    """The scan below is not vacuous: the known lazy modules are in it."""

    modules = set(_f822_suppressed_modules())
    for expected in ("gpuwm.bridges", "gpuwm.doctor", "gpuwm.core.physics",
                     "gpuwm.da.enprod", "gpuwm.rustwx"):
        assert expected in modules, sorted(modules)


@pytest.mark.parametrize("module_name", _f822_suppressed_modules())
def test_every_f822_suppression_resolves_at_import(module_name):
    """A ``noqa: F822`` is true only if the module really serves the name."""

    module = importlib.import_module(module_name)
    missing = [name for name in module.__all__
               if not hasattr(module, name)]
    assert not missing, (
        f"{module_name}.__all__ names {missing}, which neither the module "
        "nor its __getattr__ provides; the noqa: F822 on that line is false")
    namespace: dict = {}
    exec(f"from {module_name} import *", namespace)  # noqa: S102
    assert set(module.__all__) <= set(namespace)
