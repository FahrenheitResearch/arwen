"""Shared fixtures, and the guarantee that ``-m "not gpu"`` touches no device.

The GPU exclusion used to rest on every author remembering
``@pytest.mark.gpu``.  That failed silently and expensively: five Noah-MP CUDA
modules gated only on ``pytest.importorskip("cupy")`` and carried no marker, so
``-m "not gpu"`` *collected* them -- an unmarked test is not excluded by
``not gpu`` -- and 34 CUDA gates compiled and ran on a machine whose owner had
asked that no GPU work run there at all.  A convention that fails open is not a
convention; it is a hope.

Three things close it, in increasing order of strength:

* every test whose module imports cupy is marked ``gpu`` **automatically**, so
  the exclusion no longer depends on anyone remembering;
* ``GPUWM_NO_LOCAL_GPU=1`` skips those tests outright and stops this file
  importing cupy at all, so no device is opened even to ask whether one exists;
* ``tests/test_gpu_marker_discipline.py`` fails if any cupy-importing module
  would survive ``-m "not gpu"``.

Detection is by AST over the module's own source, not by inspecting
``sys.modules``: a module that imports cupy inside a function still needs the
marker, and reading the source cannot itself trigger an import.
"""

import ast
import functools
import os
import pathlib

import pytest

#: Set to 1 to guarantee no local device is opened, whatever is collected.
#: The rented-GPU workflow leaves this set on the user's own machine.
NO_LOCAL_GPU = os.environ.get("GPUWM_NO_LOCAL_GPU", "") not in ("", "0")

if NO_LOCAL_GPU:
    # RUNTIME BACKSTOP, because source inspection is not a guarantee.  The
    # AST detector below marks tests whose *own* source imports cupy, but a
    # test can reach the device through an intermediary --
    # ``test_multigpu_forced_gpu.py`` did exactly that with a lazy
    # ``from tilestream import multigpu`` inside the test body, dodged the
    # marker automation, and RAN ON THE LOCAL CARD during a mandated
    # CPU-only invocation.  No enumeration of intermediaries can close that,
    # so the guarantee is planted where every route converges: the CUDA
    # runtime reads this variable at initialisation, and "-1" is an invalid
    # ordinal that leaves NOTHING visible.  Any escaped test's first device
    # use then fails loudly (cudaErrorNoDevice) instead of silently running
    # on the owner's card, whatever its import style, and subprocesses
    # inherit the ban.
    #
    # An import-level ban was tried first and rejected: merely importing
    # cupy is NOT the crime (module-scope ``import cupy`` sits under swaths
    # of legitimate CPU coverage, and zarr pulls the full package into every
    # pytest process before any conftest runs) -- opening the device is.
    # ``tests/test_gpu_marker_discipline.py`` pins this backstop
    # red-on-revert by asserting the banned process really sees zero
    # devices.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def has_gpu() -> bool:
    """True when a usable CUDA device is present.

    Under ``GPUWM_NO_LOCAL_GPU`` this answers False *without importing cupy*.
    Importing it and calling ``getDeviceCount()`` is itself device contact, and
    it used to happen on every single pytest invocation, including runs that
    had explicitly excluded the GPU.
    """
    if NO_LOCAL_GPU:
        return False
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


HAS_GPU = has_gpu()
requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU / cupy")


def _is_cupy_import(node: ast.AST) -> bool:
    """Whether this single node imports cupy, however it is spelled."""
    if isinstance(node, ast.Import):
        return any(a.name.split(".")[0] == "cupy" for a in node.names)
    if isinstance(node, ast.ImportFrom):
        return (node.module or "").split(".")[0] == "cupy"
    # pytest.importorskip("cupy") is an import in every sense that matters.
    if isinstance(node, ast.Call):
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(
            func, "id", "")
        if name == "importorskip" and node.args:
            first = node.args[0]
            return isinstance(first, ast.Constant) and first.value == "cupy"
    return False


def _is_fixture(node: ast.AST) -> bool:
    for deco in getattr(node, "decorator_list", []):
        target = deco.func if isinstance(deco, ast.Call) else deco
        name = target.attr if isinstance(target, ast.Attribute) else getattr(
            target, "id", "")
        if name == "fixture":
            return True
    return False


@functools.lru_cache(maxsize=None)
def _cupy_scope(path: str) -> tuple[bool, frozenset[str]]:
    """Which parts of a module open a CUDA device.

    Returns ``(whole_module, {function names})``.

    Granularity is the whole point.  A first version answered only "does this
    file mention cupy anywhere", which marked ``tests/test_preflight.py``
    entirely ``gpu`` on the strength of **one** import at ``:1548`` -- and that
    module's other ~200 tests deliberately *stub* cupy to exercise CPU paths
    ("no device touched", says its own comment).  Skipping them to protect one
    test removed the VRAM preflight's only automated evidence, which is a
    correctness bar on this hardware.  So:

    * module-level import -> the whole module, because import happens at
      collection and every test in it pays;
    * inside a fixture -> the whole module, because which tests request that
      fixture is not decidable from the AST alone, and over-marking is the
      safe direction;
    * inside one test function -> that function only.
    """
    try:
        tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return False, frozenset()

    functions: set[str] = set()
    defs = [n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    owned = {id(inner) for fn in defs for inner in ast.walk(fn)}

    for node in ast.walk(tree):
        if not _is_cupy_import(node):
            continue
        if id(node) not in owned:
            return True, frozenset()          # module scope: everything pays

    for fn in defs:
        if not any(_is_cupy_import(n) for n in ast.walk(fn)):
            continue
        # A fixture, or any non-test helper, has callers this cannot see.  Its
        # device use belongs to whoever calls it, so the only safe answer is
        # the whole module.  Narrowing to the helper's own name would be worse
        # than the coarse rule: the helper is not a collected test, so nothing
        # would ever be marked and the leak would reopen silently.
        if _is_fixture(fn) or not fn.name.startswith("test_"):
            return True, frozenset()
        functions.add(fn.name)

    return False, frozenset(functions)


def _imports_cupy(path: str) -> bool:
    """Whether any part of this module opens a CUDA device."""
    whole, functions = _cupy_scope(path)
    return whole or bool(functions)


def _register_silent_deselection_guard(config):
    """Load tools/battery/no_silent_deselection BY PATH, on every run.

    THE BREAKAGE, measured 2026-08-28.  ``pytestmark = pytest.mark.gpu`` at
    the top of tests/test_ruc.py retired all sixty RUC bitwise-oracle tests --
    the suite that detects a ONE-ULP change to Stefan-Boltzmann -- and the leg
    reported rc=0 with 8 passed, 61 deselected.  Nothing saw it: not
    test_gpu_marker_discipline.py, not test_module_skip_placement.py, not
    test_stage1_manifest.py.  One line retires any suite in this repository.

    Registered here rather than left to the battery's command line because a
    guard that runs only when somebody remembers to pass ``-p`` is not a
    default, and the battery is driven from queue scripts that do not live in
    this repository.  It is loaded BY PATH rather than as ``tools.battery.*``
    because that import fails whenever pytest is run from a subdirectory --
    measured: ImportError, and it takes the whole session with it.

    Failure to load is reported and not fatal: this hook must never be the
    reason a test run cannot start.
    """

    import importlib.util
    import sys

    path = (pathlib.Path(__file__).resolve().parents[1]
            / "tools" / "battery" / "no_silent_deselection.py")
    if not path.is_file():
        print()
        print(f"no_silent_deselection guard NOT LOADED: {path} is "
              "missing; "
              "a whole suite can be retired by one marker line and this run "
              "would not see it")
        return
    spec = importlib.util.spec_from_file_location(
        "gpuwm_no_silent_deselection", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["gpuwm_no_silent_deselection"] = module
    try:
        spec.loader.exec_module(module)
        module.pytest_configure(config)
    except Exception as error:                       # pragma: no cover
        print()
        print(f"no_silent_deselection guard NOT LOADED: {error!r}")


def pytest_configure(config):
    if not config.pluginmanager.hasplugin("no_silent_deselection_guard"):
        _register_silent_deselection_guard(config)
    config.addinivalue_line(
        "markers",
        "slow_acceptance: multi-minute end-to-end acceptance runs; excluded "
        "during fix-round iteration (-m 'not slow_acceptance'), REQUIRED in "
        "the full suite before any task's final commit")
    config.addinivalue_line(
        "markers",
        "gpu: opens a CUDA device.  Applied automatically to every test whose "
        "module imports cupy -- do not rely on writing it by hand, and do not "
        "remove the automation to 'clean up' a redundant-looking marker.")


def pytest_collection_modifyitems(config, items):
    """Mark every cupy-importing test ``gpu``, and skip them when banned.

    Marking is unconditional so that ``-m "not gpu"`` is honest on any
    machine, with or without a device.

    The ban-skip applies to every item CARRYING the marker, not only to the
    items this hook marked.  The old form skipped exactly its own AST hits,
    so a hand-marked test whose device use is transitive (no cupy in its own
    source, a lazy ``from tilestream import multigpu`` in the body) was
    marked ``gpu`` yet NOT skipped, and ran on the local card during a
    CPU-only invocation.  Belt: marker implies skip.  Braces: the
    ``CUDA_VISIBLE_DEVICES=-1`` backstop planted at import above, for tests
    carrying no marker at all.
    """
    skip_local = pytest.mark.skip(
        reason="GPUWM_NO_LOCAL_GPU=1: GPU work belongs on the rented device")
    for item in items:
        path = getattr(item, "fspath", None)
        if path is None:
            continue
        whole, functions = _cupy_scope(str(path))
        detected = whole
        if not whole:
            # ``originalname`` is the undecorated name for parametrised items.
            name = getattr(item, "originalname", None) or item.name
            detected = name.split("[")[0] in functions
        if detected:
            item.add_marker(pytest.mark.gpu)
        if NO_LOCAL_GPU and (detected
                             or item.get_closest_marker("gpu") is not None):
            item.add_marker(skip_local)


@pytest.fixture(autouse=True, scope="session")
def _isolated_fetch_lock_root(tmp_path_factory):
    """Keep every test's output locks out of the machine-wide lock root.

    ``gpuwm.fetch_guard`` keys its lock files on the resolved output path
    and keeps them outside the output tree, which means a suite that
    fetches into a hundred ``tmp_path`` directories leaves a hundred
    dead lock files under ``%PROGRAMDATA%/gpuwm/locks``.  Pointing the
    root at the session's own temp directory makes the suite hermetic
    and leaves the user's machine alone.  Tests that need their own root
    (the lock gates themselves) override the variable per test.
    """
    from gpuwm import fetch_guard

    root = tmp_path_factory.mktemp("fetch-locks")
    previous = os.environ.get(fetch_guard.LOCK_ROOT_ENV)
    os.environ[fetch_guard.LOCK_ROOT_ENV] = str(root)
    yield root
    if previous is None:
        os.environ.pop(fetch_guard.LOCK_ROOT_ENV, None)
    else:
        os.environ[fetch_guard.LOCK_ROOT_ENV] = previous


@pytest.fixture(autouse=True)
def _wizard_probe_pinned_to_a_24gib_card(monkeypatch):
    """Pin the ONE number the outside world moves in the domain wizard.

    With neither ``--card`` nor ``--vram-gib``, ``gpuwm domain`` measures
    the local card through its probe seam and refuses when nothing is
    measurable.  Unpinned, every bare-wizard fixture emission in this
    suite (117 of them at the time of writing) would size against
    whatever card the box happens to hold -- or refuse outright on the
    CPU legs -- turning grid dimensions machine-dependent.  The pin is a
    24 GiB card, the tier the old silent default assumed, so every
    historical fixture keeps its exact bytes.

    In-process invocations only; a test that drives the real CLI in a
    subprocess bypasses this and must declare its card (or pin its own
    probe).  The sizing-authority tests that exercise the measure and
    refuse paths re-monkeypatch this same seam with their own answers.
    """

    from gpuwm import domain_wizard

    monkeypatch.setattr(
        domain_wizard, "device_memory_probe_subprocess",
        lambda **_kwargs: {"free_bytes": 20 * 1024 ** 3,
                           "total_bytes": 24 * 1024 ** 3,
                           "profile": None})


def complete_runtime_manifest(payload: dict | None = None,
                              *, platform_name: str = "linux-x86_64",
                              **overrides) -> dict:
    """A sealed-runtime manifest that meets the WHOLE required schema.

    Fixtures used to declare only the keys the assertion under test
    happened to read, which is exactly how a two-key manifest reached a
    field user's preparation and died several minutes in, at a third
    consumer, on a key nobody had validated.  Now that
    :func:`gpuwm.runtime_manifest.validate_manifest` is the one gate,
    a fixture that skips a field is testing a document no consumer will
    ever accept.  Build from here and override deliberately.
    """

    from gpuwm import __version__
    from gpuwm.native_wrf_distribution import distribution_contract
    from gpuwm.runtime_manifest import RUNTIME_SCHEMA

    manifest = {
        "schema": RUNTIME_SCHEMA,
        "status": "READY",
        "artifact": {"name": "gpuwm-native-wrf",
                     "gpuwm_version": __version__},
        "source": {"commit": "0" * 40, "tree": "1" * 40,
                   "worktree_clean": True},
        "contract": distribution_contract(platform_name),
        "payload": payload if payload is not None else {
            "libexec/bridges/placeholder": {"bytes": 0, "sha256": "0" * 64},
        },
    }
    manifest.update(overrides)
    return manifest


def assert_gates(case: str, metrics: dict) -> None:
    """Assert every benchmark gate for ``case`` passes.

    Single-sourcing (Phase 2 Task 12): the gate intervals live in the case
    module's ``GATES`` export, and both the CLI and the benchmark tests
    consume them through ``gpuwm.cli._failing_gate`` -- one table, one
    checker.  The failure message names the failing gate and dumps the
    metrics (list-valued time series dropped for readability).
    """
    from gpuwm.cli import _failing_gate
    bad = _failing_gate(case, metrics)
    assert bad is None, (bad, {k: v for k, v in metrics.items()
                               if not isinstance(v, list)})
