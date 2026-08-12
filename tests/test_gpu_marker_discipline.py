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

The *detector* checks are pure source inspection.  The *selection* checks are
not, and the docstring used to claim otherwise.  They shell out to ``pytest
--collect-only`` once per module, deliberately with ``GPUWM_NO_LOCAL_GPU``
stripped, and several CUDA modules probe the device at import time -- one
``pytest.importorskip("cupy")`` plus ``getDeviceCount()`` at module scope, and
``pytest.skip(..., allow_module_level=True)`` when it throws.  So these tests
do open a device, at one remove, and their answers depend on this host having
one that answers.

That is why a module which skips itself at import is now reported as
*unanswerable* rather than as a marking failure: with the card saturated,
``getDeviceCount()`` does not return cleanly, the RRTMG modules skip at
collection, and the gate used to read zero-selected as "this coverage runs on
no machine" and go red -- on a busy card, at a release cut, pointing at a merge
it had nothing to do with.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import warnings

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


def _collect(path: pathlib.Path, selection: str | None) -> set[str]:
    """Test ids pytest would select from one file, without running anything.

    ``selection`` is a ``-m`` expression, or None to collect unfiltered.
    """
    # Exactly one -q: that is the mode that prints one node id per line.
    # Two (-qq) suppresses the listing entirely and this returns an empty set,
    # which would make every assertion below pass vacuously.
    command = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider",
               "--collect-only", "-q"]
    if selection is not None:
        command += ["-m", selection]
    command.append(str(path))
    proc = subprocess.run(
        command, capture_output=True, text=True, cwd=str(_ROOT),
        env={**_environ(), "PYTHONPATH": str(_ROOT)},
    )
    # 0 = something was collected, 5 = nothing matched the selection.  Any
    # other code is a crashed or erroring collection, and the empty set it
    # yields is indistinguishable from an honest "no such tests" -- which
    # makes the leak gate below pass VACUOUSLY, the one failure mode this
    # file exists to prevent.  Say so instead of answering nothing.
    if proc.returncode not in (0, 5):
        raise AssertionError(
            f"collection subprocess failed (rc={proc.returncode}) for "
            f"{path.name} with -m {selection!r}; treating that as 'no tests' "
            f"would make this gate pass vacuously.\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    ids = set()
    for line in proc.stdout.splitlines():
        line = line.strip()
        if "::" in line and not line.startswith(("<", "=")):
            ids.add(line.split("::", 1)[1].split("[")[0])
    return ids


def _node_ids(selection: str, path: pathlib.Path) -> set[str]:
    """Test ids pytest would select from one file under one ``-m`` filter."""
    return _collect(path, selection)


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

    Zero-selected has two causes that must not be conflated.  A *marking* gap
    is a real defect and fails here.  A module that skipped itself at import
    -- because its module-scope ``getDeviceCount()`` probe found no device, or
    found one too busy to answer -- collected nothing under any selection, so
    this host has no evidence about its marking either way.  Reporting that as
    "runs on no machine" is a false alarm, and it fired as one: it turned this
    gate red at a release cut on a saturated card, naming two RRTMG modules
    that were byte-identical to the ones that had passed an hour earlier.
    """
    missing = {}
    unanswerable = []
    for path in _cupy_modules():
        whole, functions = _cupy_scope(str(path))
        selected = _node_ids("gpu", path)
        if not selected and not _collect(path, None):
            unanswerable.append(path.name)
            continue
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
    if unanswerable and len(unanswerable) == len(_cupy_modules()):
        # Every module declined to collect: this host answered nothing at all,
        # and a green here would be pure vacuum.
        pytest.skip(
            "no cupy module could be collected on this host (no device, or a"
            f" card too busy to answer): {sorted(unanswerable)}"
        )
    if unanswerable:
        warnings.warn(
            "marking unverified for modules that skip themselves at import on"
            f" this host: {sorted(unanswerable)}",
            stacklevel=2,
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


# --------------------------------------------------------------------------
# the runtime ban: guards no import style can dodge
# --------------------------------------------------------------------------
#
# The AST layer above answers "which tests' own source imports cupy".  It is
# structurally blind to a test that reaches the device through an intermediary
# module, and that blindness was exploited for real: a gpu-marked test whose
# only import was a lazy ``from tilestream import multigpu`` inside the test
# body carried the marker, dodged the AST-driven skip, and ran on the local
# card during a mandated CPU-only invocation.  Two closures, each pinned here
# red-on-revert:
#
# * marker implies skip -- ``pytest_collection_modifyitems`` bans every item
#   CARRYING the gpu marker, not just its own AST hits;
# * the device-visibility backstop -- under GPUWM_NO_LOCAL_GPU=1 the conftest
#   sets ``CUDA_VISIBLE_DEVICES=-1`` before any test runs, so an escaped
#   unmarked test's first device use goes red (cudaErrorNoDevice) instead of
#   silently running on the owner's card, whatever its import style.


def _run_banned(path: pathlib.Path, cwd: pathlib.Path) -> tuple:
    """Run one test file in a subprocess WITH the local-GPU ban set.

    For a scratch file outside tests/, the tests/ conftest is loaded
    explicitly with ``-p conftest`` because directory walking would not find
    it -- and these tests exist precisely to prove what that conftest
    enforces.  A file inside tests/ picks it up normally, and loading it
    twice would double-register the plugin.
    """
    import os
    # CUDA_VISIBLE_DEVICES is deliberately stripped: the conftest under test
    # must plant it itself, and an inherited copy would mask a reverted
    # backstop.
    env = {k: v for k, v in os.environ.items()
           if k != "CUDA_VISIBLE_DEVICES"}
    env.update({"GPUWM_NO_LOCAL_GPU": "1",
                "PYTHONPATH": os.pathsep.join([str(_ROOT), str(_TESTS)])})
    command = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider"]
    if _TESTS not in path.parents:
        command += ["-p", "conftest"]
    command += ["-q", str(path)]
    proc = subprocess.run(command, capture_output=True, text=True,
                          cwd=str(cwd), env=env)
    return proc.returncode, proc.stdout + proc.stderr


def test_the_ban_hides_every_device_from_an_unmarked_escapee(tmp_path):
    """An UNMARKED test reaching CUDA through an INTERMEDIARY sees no card.

    This is the escape the AST automation cannot see: the test file itself
    never mentions cupy -- a helper module does, imported lazily in the test
    body, which is exactly how ``test_multigpu_forced_gpu.py`` reached the
    local card (its intermediary was ``tilestream.multigpu``).  The scratch
    test asserts it CAN see a device, so under the backstop it fails
    red-loud; if the backstop is reverted on a machine with a card it
    passes -- an escaped test would once again run GPU work locally -- and
    THIS test goes red.

    The probe is ``getDeviceCount`` only: under the backstop there is no
    device to see, and on revert enumeration alone opens no context, so
    neither outcome runs work on the owner's card.
    """
    helper = tmp_path / "scratch_gpu_helper.py"
    helper.write_text(
        "import cupy\n\n\n"
        "def visible_devices():\n"
        "    try:\n"
        "        return cupy.cuda.runtime.getDeviceCount()\n"
        "    except Exception:\n"
        "        return 0  # cudaErrorNoDevice: the ban, doing its job\n",
        encoding="utf-8")
    scratch = tmp_path / "test_scratch_lazy_gpu.py"
    scratch.write_text(
        "def test_reaches_for_the_device():\n"
        "    import scratch_gpu_helper  # the intermediary dodge under"
        " test\n"
        "    assert scratch_gpu_helper.visible_devices() > 0\n",
        encoding="utf-8")
    rc, out = _run_banned(scratch, tmp_path)
    assert rc != 0 and "1 failed" in out, (
        "an unmarked test reaching CUDA through an intermediary could still"
        " see a device under GPUWM_NO_LOCAL_GPU=1 -- the CUDA_VISIBLE_DEVICES"
        f" backstop is gone and the local card is reachable again (rc={rc}):"
        f"\n{out}")


def test_a_marked_test_with_clean_source_is_still_skipped(tmp_path):
    """The gpu MARKER alone must trigger the ban skip, without any AST hit.

    A skip, not a run: the marked test never executes at all.  If
    marker-implies-skip is narrowed back to AST-detected items only, the
    test below RUNS (and passes, since importing cupy without touching the
    device is legal) -- turning this "1 skipped" assertion red.
    """
    helper = tmp_path / "scratch_gpu_helper2.py"
    helper.write_text("import cupy\nTOUCHED = cupy.ndarray\n",
                      encoding="utf-8")
    scratch = tmp_path / "test_scratch_marked_gpu.py"
    scratch.write_text(
        "import pytest\n\n"
        "pytestmark = pytest.mark.gpu\n\n\n"
        "def test_transitive_device_use():\n"
        "    # An intermediary, not cupy itself: no AST hit in THIS file,\n"
        "    # so only the marker can trigger the skip.  Never reached when\n"
        "    # the marker skip works.\n"
        "    import scratch_gpu_helper2\n"
        "    assert scratch_gpu_helper2.TOUCHED is not None\n",
        encoding="utf-8")
    rc, out = _run_banned(scratch, tmp_path)
    assert rc == 0 and "1 skipped" in out, (
        "a gpu-marked test with no cupy in its own source was not skipped"
        f" under the ban (rc={rc}) -- marker-implies-skip has regressed:"
        f"\n{out}")


def test_the_original_escapee_is_skipped_under_the_ban():
    """The file that actually ran on the forbidden card, pinned by name."""
    path = _TESTS / "test_multigpu_forced_gpu.py"
    if not path.exists():
        pytest.skip("test_multigpu_forced_gpu.py not present")
    rc, out = _run_banned(path, _ROOT)
    assert rc == 0 and "skipped" in out and "passed" not in out, (
        f"test_multigpu_forced_gpu.py was not skipped under the ban"
        f" (rc={rc}):\n{out}")


def _environ() -> dict:
    import os
    # Never inherit the ban into the subprocess: these are collection-only
    # runs, and the ban would skip the very items being counted.  The
    # device-visibility backstop travels with the ban and is stripped for
    # the same reason -- modules that probe the device at import would
    # otherwise all collect as "unanswerable".
    env = {k: v for k, v in os.environ.items()
           if k not in ("GPUWM_NO_LOCAL_GPU", "CUDA_VISIBLE_DEVICES")}
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
