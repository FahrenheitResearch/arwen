"""``-m "not gpu"`` must open no CUDA device.  This proves it, per module.

The exclusion used to depend on authors writing ``@pytest.mark.gpu`` by hand.
Five Noah-MP CUDA modules did not, gating only on
``pytest.importorskip("cupy")``, and an unmarked test is not excluded by
``not gpu`` -- so a run believed to be CPU-only compiled and ran 34 CUDA gates
on a machine whose owner had asked that no GPU work happen there.

``conftest.pytest_collection_modifyitems`` now applies the marker
automatically.  These tests exist so that automation cannot be quietly removed
or narrowed: they assert the property (*no cupy-importing test survives
``-m "not gpu"``*) rather than the mechanism, so they keep working if the
mechanism is rewritten and fail if it is deleted.

Nothing here imports cupy or opens a device; it is all source inspection.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

from conftest import _cupy_scope, _imports_cupy

_TESTS = pathlib.Path(__file__).resolve().parent
_ROOT = _TESTS.parent


def _cupy_modules() -> list[pathlib.Path]:
    return sorted(p for p in _TESTS.glob("test_*.py") if _imports_cupy(str(p)))


def test_some_modules_do_import_cupy():
    """Guard the guard: if this finds nothing, the detector broke."""
    found = _cupy_modules()
    assert found, (
        "no test module appears to import cupy, which cannot be right --"
        " _imports_cupy is probably broken, and every test below would then"
        " pass vacuously"
    )


def _node_ids(selection: str, path: pathlib.Path) -> set[str]:
    """Test ids pytest would select from one file, without running anything."""
    # Exactly one -q: that is the mode that prints one node id per line.
    # Two (-qq) suppresses the listing entirely and this returns an empty set,
    # which would make every assertion below pass vacuously.
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider",
         "--collect-only", "-q", "-m", selection, str(path)],
        capture_output=True, text=True, cwd=str(_ROOT),
        env={**_environ(), "PYTHONPATH": str(_ROOT)},
    )
    ids = set()
    for line in proc.stdout.splitlines():
        line = line.strip()
        if "::" in line and not line.startswith(("<", "=")):
            ids.add(line.split("::", 1)[1].split("[")[0])
    return ids


def test_no_device_touching_test_survives_the_not_gpu_selection():
    """The property that matters, at the granularity that matters.

    Per *test*, not per module.  The coarse form -- "no test in a
    cupy-importing file survives" -- was too blunt: ``test_preflight.py`` has
    one cupy import among ~200 tests that deliberately *stub* cupy to exercise
    CPU paths, and marking the file whole deleted the VRAM preflight's only
    automated evidence.  Over-marking is not a safe failure either; it is
    coverage loss wearing a safety costume.
    """
    leaked = {}
    for path in _cupy_modules():
        whole, functions = _cupy_scope(str(path))
        surviving = _node_ids("not gpu", path)
        bad = surviving if whole else (surviving & set(functions))
        if bad:
            leaked[path.name] = sorted(bad)
    assert not leaked, (
        "these tests open a CUDA device yet survive -m \"not gpu\", so a run"
        f" believed to be CPU-only would use the local card: {leaked}"
    )


def test_every_device_touching_test_is_selected_under_m_gpu():
    """The converse: GPU coverage must not run nowhere.

    A test in neither selection is exercised on no machine, which is how 34
    Noah-MP CUDA gates passed locally while never running on the rented card.
    """
    missing = {}
    for path in _cupy_modules():
        whole, functions = _cupy_scope(str(path))
        selected = _node_ids("gpu", path)
        if whole:
            if not selected:
                missing[path.name] = ["<entire module>"]
            continue
        absent = sorted(set(functions) - selected)
        if absent:
            missing[path.name] = absent
    assert not missing, (
        "these tests open a CUDA device but are selected by neither -m gpu nor"
        f" -m \"not gpu\", so they run on no machine: {missing}"
    )


def test_a_module_that_only_stubs_cupy_keeps_its_cpu_coverage():
    """Guard the granularity itself, with the case that motivated it.

    ``tests/test_preflight.py`` stubs cupy to test CPU failure paths and has a
    single genuine import.  If a future change re-coarsens the rule, its ~46
    CPU tests vanish silently -- and the VRAM preflight is a correctness bar
    on this hardware, not a nicety.
    """
    path = _TESTS / "test_preflight.py"
    if not path.exists():
        pytest.skip("test_preflight.py not present")
    whole, functions = _cupy_scope(str(path))
    assert not whole, (
        "test_preflight.py is marked gpu wholesale; it stubs cupy for CPU"
        " paths and only a couple of its tests really open a device"
    )
    cpu_side = _node_ids("not gpu", path)
    assert len(cpu_side) > 10, (
        f"only {len(cpu_side)} CPU test(s) survive in test_preflight.py --"
        " the VRAM preflight has effectively lost its coverage"
    )


@pytest.mark.parametrize("spelling", [
    "import cupy",
    "import cupy as cp",
    "from cupy import ndarray",
    "import cupy.cuda",
    'pytest.importorskip("cupy")',
])
def test_the_detector_sees_every_spelling(tmp_path, spelling):
    """A detector that misses a spelling is a detector that fails open."""
    module = tmp_path / "test_scratch_probe.py"
    module.write_text(
        f"import pytest\n{spelling}\n\n\ndef test_x():\n    pass\n",
        encoding="utf-8",
    )
    assert _imports_cupy(str(module)), f"missed spelling: {spelling!r}"


def test_the_detector_is_quiet_on_modules_that_only_mention_cupy(tmp_path):
    """A comment or a string is not an import; over-marking hides real tests."""
    module = tmp_path / "test_scratch_mentions.py"
    module.write_text(
        '"""Docstring mentioning cupy."""\n'
        "# cupy is not imported here\n"
        'NOTE = "cupy"\n\n\ndef test_x():\n    pass\n',
        encoding="utf-8",
    )
    assert not _imports_cupy(str(module))


def _environ() -> dict:
    import os
    # Never inherit the ban into the subprocess: these are collection-only
    # runs, and the ban would skip the very items being counted.
    env = {k: v for k, v in os.environ.items() if k != "GPUWM_NO_LOCAL_GPU"}
    return env


def _collected_count(stdout: str) -> int:
    """Parse pytest's collection summary without depending on exit status."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if "test" in line and ("collected" in line or "selected" in line):
            for token in line.replace("/", " ").split():
                if token.isdigit():
                    return int(token)
    return 0
