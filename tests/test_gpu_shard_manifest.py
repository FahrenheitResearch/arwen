"""The GPU pytest shard's list, gated the way the stage-1 list is.

``tools/battery/gpu_shard_files.txt`` closes gap G2 of the 2026-08-13
test-estate audit: **there was no GPU pytest leg at all.**
``stage1_files.txt`` is CPU by construction, ``cargo_gates.txt`` is Rust,
``tiles_gates.txt`` names ``python -m tilestream.*`` modules, and no file
anywhere named a GPU pytest shard.  The consequence the audit measured is
that the entire WRF-parity dycore ran on no list -- 6th-order diffusion
including its boundary faces, both Smagorinsky closures, the open lateral
boundary, the acoustic solver, the mapped mass closure, the TKE budget --
and that ``tests/test_wrfout_conformance.py``, the only file that judges a
gpuwm wrfout with a real WRF reader, is 18 of 18 skipped on a CPU box.

This suite is hermetic and runs on the CPU legs, for the same reason
``tests/test_tiles_gate_manifest.py`` does: a gate that stops existing -- a
file renamed, an entry lost in a tidy-up -- must fail on every commit rather
than at a cut, on the node, when there is no time.

FIRST EXECUTION
---------------
Shard 1 was run for the first time in the project's history on 2026-08-13,
on a rented RTX 5080 (sm_120, driver 595.71.05, CUDA 13.2, NVRTC 13.2, cupy
14.1.1): **314 passed, 1 failed, 6 skipped in 121 s**.  The single failure is
``tests/test_pd_advection.py``'s pre-existing CPU-backend defect, which is
why that file is deliberately on the list.  Shard 2 ran 765 passed in 31 s.
"""

from __future__ import annotations

import pathlib

import pytest
from _dycore_oracle_routes import (
    DYCORE_PARITY_FILES, DEVICE_ORACLE_IMPORTS, adapter_path,
)

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = REPOSITORY_ROOT / "tools" / "battery" / "gpu_shard_files.txt"

#: The line that divides the per-cut leg from the weekly one.
SHARD2_MARKER = "SHARD-2-BEGIN"

#: Entries whose device reads live in a test helper, not in their own text:
#: entry -> the helper it uses.  The 2.8.5 ensemble runtime-case witnesses
#: run the ordinary runtime on the card and capture every state and physics
#: array as device words through _RuntimeWords, which reads them with the
#: production forecast gate's CuPy helpers.  conftest's detector reads one
#: file, so it cannot see that; _helper_opens_cuda follows the imports.
DEVICE_HELPERS = {
    "tests/test_ensemble_runtime_forecast_gpu.py": "_RuntimeWords",
    "tests/test_ensemble_simulated_radar_forecast_gpu.py": "_RuntimeWords",
}


def _helper_opens_cuda(module: str, name: str, root: pathlib.Path,
                       seen: set[tuple[str, str]] | None = None) -> bool:
    """Whether ``name`` in ``root/<module>.py`` reaches a CuPy import.

    Follows the definition of ``name`` (or the import that brings it in)
    through every name it loads, across ``from <module> import`` lines of
    sibling test modules, until a definition that imports cupy by
    conftest's own detector.  A helper chain that stops reading the device
    answers False.
    """

    import ast

    from conftest import _is_cupy_import

    seen = set() if seen is None else seen
    if (module, name) in seen:
        return False
    seen.add((module, name))
    path = root / f"{module}.py"
    if not path.is_file():
        return False
    tree = ast.parse(path.read_text(encoding="utf-8"))
    kinds = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    imported = {alias.asname or alias.name: (node.module, alias.name)
                for node in tree.body
                if isinstance(node, ast.ImportFrom) and node.module and not node.level
                for alias in node.names}
    local = {node.name: node for node in tree.body if isinstance(node, kinds)}
    definition = local.get(name)
    if definition is None:
        source = imported.get(name)
        return source is not None and _helper_opens_cuda(*source, root, seen)
    if any(_is_cupy_import(node) for node in ast.walk(definition)):
        return True
    loaded = sorted({node.id for node in ast.walk(definition)
                     if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)})
    for use in loaded:
        if use in imported and _helper_opens_cuda(*imported[use], root, seen):
            return True
        if use in local and _helper_opens_cuda(module, use, root, seen):
            return True
    return False


def _lines() -> list[str]:
    return MANIFEST.read_text(encoding="utf-8").splitlines()


def _entries(which: str = "all") -> list[str]:
    """Entries, parsed exactly as the queue scripts parse the stage-1 list.

    Comment lines are prose about entries, never entries.  The audit's own
    membership check got this wrong on the stage-1 list -- a ``grep -F``
    reported ``tests/test_stage1_manifest.py`` as present when the string
    occurred only inside a comment -- so every consumer of these files parses
    rather than greps.
    """

    shard1: list[str] = []
    shard2: list[str] = []
    seen_marker = False
    for raw in _lines():
        line = raw.strip()
        if SHARD2_MARKER in line:
            seen_marker = True
            continue
        if not line or line.startswith("#"):
            continue
        (shard2 if seen_marker else shard1).append(line)
    if which == "shard1":
        return shard1
    if which == "shard2":
        return shard2
    return shard1 + shard2


def test_the_manifest_exists_and_lists_something() -> None:
    assert MANIFEST.is_file(), f"{MANIFEST} is missing"
    assert _entries("shard1"), (
        "tools/battery/gpu_shard_files.txt parsed to zero shard-1 entries; "
        "the GPU leg would run no tests and still exit 0, which is the "
        "green-on-nothing shape gap G2 was opened about")


def test_the_two_shards_are_both_populated_and_disjoint() -> None:
    """The per-cut leg and the weekly leg must stay distinguishable.

    A runner that could not tell them apart would either run the weekly GF
    block on every cut (which the audit's shard-design finding says is the
    wrong trade) or silently drop it altogether.
    """

    shard1, shard2 = _entries("shard1"), _entries("shard2")
    assert shard1 and shard2, (len(shard1), len(shard2))
    overlap = sorted(set(shard1) & set(shard2))
    assert not overlap, (
        f"{overlap} appear in both shards; the GPU leg would run them twice "
        "and the counts would overstate the coverage")


@pytest.mark.parametrize("entry", _entries())
def test_every_listed_path_exists(entry: str) -> None:
    assert not pathlib.PurePosixPath(entry).is_absolute(), entry
    assert "\\" not in entry, (
        f"{entry} uses a backslash; forward slashes only, so the same list "
        "works on the Windows cut box and the Linux GPU node")
    # tests/ or tilestream/: the tilestream pytest suites joined the shard
    # at the 2.5.0 gating pass (release blocker #5) -- their device halves
    # ran on no list before it.  The GPU-bound property below applies to
    # them unchanged, through the same conftest detector.
    assert entry.startswith(("tests/", "tilestream/")) \
        and entry.endswith(".py"), entry
    assert (REPOSITORY_ROOT / entry).is_file(), (
        f"{entry} is listed in tools/battery/gpu_shard_files.txt but does not "
        "exist; pytest would fail the whole GPU leg on the missing argument")


def test_no_entry_is_listed_twice() -> None:
    entries = _entries()
    duplicates = sorted({e for e in entries if entries.count(e) > 1})
    assert not duplicates, duplicates


@pytest.mark.parametrize("entry", ["tests/test_chem_nest_lifecycle.py",
                                   "tests/test_chem_fork_integration.py"])
def test_chem_integration_runs_on_every_cut_and_not_the_cpu_leg(entry) -> None:
    """Device chemistry and continuation must not disappear under CPU marks."""
    assert entry in _entries("shard1"), (
        "chemistry transport, relocation and restart coverage is absent from "
        "the per-cut GPU leg")
    cpu_list = MANIFEST.with_name("stage1_files.txt").read_text(encoding="utf-8")
    cpu_entries = {line.strip() for line in cpu_list.splitlines()
                   if line.strip() and not line.lstrip().startswith("#")}
    assert entry not in cpu_entries, (
        "the device-only chemistry lifecycle suite contributes no CPU rows")
    from tools.battery.no_silent_deselection import ZERO_COLLECT_ALLOWED
    assert entry not in ZERO_COLLECT_ALLOWED, (
        "the device lifecycle requires an executing GPU leg, not a zero-row allowance")


def test_the_file_is_lf_only() -> None:
    assert b"\r" not in MANIFEST.read_bytes(), (
        "tools/battery/gpu_shard_files.txt contains CR bytes; the repository "
        "commits LF and .gitattributes does no conversion")


_TRANSITIVE_DEVICE_HELPERS = {
    "tests/test_chem_dust_wrf471_parity.py": ("chem_emis_parity_support", "replay"),
    "tests/test_chem_afwa_wrf471_parity.py": ("chem_emis_parity_support", "replay"),
    "tests/test_chem_seasalt_wrf471_parity.py": ("chem_emis_parity_support", "replay"),
    "tests/test_chem_sulfur_wrf471_parity.py": ("chem_gocart_checks", "sulfur_case"),
    "tests/test_chem_ageing_wrf471_parity.py": ("chem_gocart_checks", "aging_case"),
    "tests/test_sfire_moisture_driver_wrf471_parity.py": (
        "tools.sfire_coupled_ideal.moisture_driver.grade", "replay"),
    "tests/test_sfire_open_geopotential_wrf471_parity.py": (
        "tools.sfire_coupled_ideal.rhs_ph_open.grade", "grade"),
    # The ideal initializer graders replay their native words on the card
    # (cupy imported inside the replay helpers).
    "tests/test_sfire_ideal_geometry_wrf471_parity.py": (
        "tools.sfire_coupled_ideal.initialization.grade_geometry", "replay"),
    "tests/test_sfire_ideal_landuse_wrf471_parity.py": (
        "tools.sfire_coupled_ideal.initialization.grade_landuse", "replay"),
}


def _module_fixture_calls_device_helper(entry, fixture_name, helper, name):
    """Follow an imported module returned by a fixture to its CUDA witness."""
    import ast

    tree = ast.parse((REPOSITORY_ROOT / entry).read_text(encoding="utf-8"))
    aliases = {alias.asname or alias.name
               for node in tree.body if isinstance(node, ast.Import)
               for alias in node.names if alias.name == helper}
    assert aliases, (entry, helper)
    fixture = next(node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name == fixture_name)
    assert any(isinstance(node, ast.Return) and isinstance(node.value, ast.Name)
               and node.value.id in aliases for node in ast.walk(fixture)), entry
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and isinstance(node.func.value, ast.Name)
               and node.func.value.id == fixture_name and node.func.attr == name
               for node in ast.walk(tree)), entry
    assert _helper_opens_cuda(helper, name, REPOSITORY_ROOT / "tests"), entry


@pytest.mark.parametrize("entry", _entries())
def test_every_entry_is_actually_gpu_bound(entry: str) -> None:
    """The property that earns a place here, checked rather than asserted.

    A CPU-only file on this list is worse than useless: it makes the shard
    look broader than it is, and it burns node time that is rented by the
    hour.  ``tests/conftest.py`` is the authority on what "GPU-bound" means
    in this tree -- it marks a module ``gpu`` when cupy is imported at module
    scope, and individual functions when the import is inside them -- so this
    asks conftest rather than pattern-matching the filename.
    """

    from conftest import _cupy_scope

    if entry == "tests/test_ruc_fork_gpu.py":
        _module_fixture_calls_device_helper(
            entry, "fork_gpu", "test_ruc_gpu",
            "test_ruc_snow_temperature_cuda_matches_unmodified_wrf_bit_for_bit")
        return
    if entry in _TRANSITIVE_DEVICE_HELPERS:
        import ast
        helper, name = _TRANSITIVE_DEVICE_HELPERS[entry]
        tree = ast.parse((REPOSITORY_ROOT / entry).read_text(encoding="utf-8"))
        assert any(isinstance(node, ast.ImportFrom) and node.module == helper
                   and any(alias.name == name for alias in node.names)
                   for node in ast.walk(tree)), entry
        if "." in helper:
            # A dotted helper is a tools/ module (the SFIRE oracle graders),
            # which imports cupy at module scope; conftest's detector reads it.
            from conftest import _cupy_scope
            helper_whole, helper_functions = _cupy_scope(
                str(REPOSITORY_ROOT / (helper.replace(".", "/") + ".py")))
            assert helper_whole or name in helper_functions, entry
        else:
            assert _helper_opens_cuda(helper, name, REPOSITORY_ROOT / "tests"), entry
        return
    whole, functions = _cupy_scope(str(REPOSITORY_ROOT / entry))
    if entry in DEVICE_ORACLE_IMPORTS:
        # The compiled-WRF suites call device adapters transitively. Keep
        # CPU provenance rows selectable and verify the imported adapter's
        # real CuPy use instead of adding module-scope imports to the tests.
        import ast
        source = (REPOSITORY_ROOT / entry).read_text(encoding="utf-8")
        tree = ast.parse(source)
        module = DEVICE_ORACLE_IMPORTS[entry]
        names = {alias.asname or alias.name
                 for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom) and node.module == module
                 for alias in node.names}
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        imported_call = any(
            isinstance(node.func, ast.Name) and node.func.id in names
            or isinstance(node.func, ast.Subscript)
            and isinstance(node.func.value, ast.Name) and node.func.value.id in names
            for node in calls)
        dynamic_import = any(
            isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"
            and node.args and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == module for node in calls)
        assert imported_call or dynamic_import, (
            f"{entry} no longer calls its device oracle adapter {module}")
        adapter_whole, adapter_functions = _cupy_scope(
            str(REPOSITORY_ROOT / adapter_path(module)))
        assert adapter_whole or adapter_functions, (
            f"{entry}: oracle adapter {module} no longer opens CUDA")
        return
    if entry == "tests/test_streamed_lifecycle_restart.py":
        # The lifecycle witness executes five actual CUDA trajectories through
        # the shared nesting helper. Pin that import and call, and require the
        # helper itself to remain device-bound; this is not a filename exemption.
        import ast
        tree = ast.parse((REPOSITORY_ROOT / entry).read_text(encoding="utf-8"))
        witness = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                       and node.name == "test_actual_lifecycle_tree_checkpoint_preserves_tiled_trajectory")
        assert any(isinstance(node, ast.ImportFrom)
                   and node.module == "test_both_streamed_nesting"
                   and any(name.name == "trajectory" and name.asname is None
                           for name in node.names) for node in ast.walk(witness))
        assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id == "trajectory" for node in ast.walk(witness))
        helper_whole, helpers = _cupy_scope(str(REPOSITORY_ROOT / "tests/test_both_streamed_nesting.py"))
        assert helper_whole or "trajectory" in helpers
        return
    if entry == "tests/test_lbc_time_frame.py":
        # The witness opens CUDA through the production loader. Require the
        # actual loader call and driver attribute read, not just a gpu marker.
        import ast
        tree = ast.parse((REPOSITORY_ROOT / entry).read_text(encoding="utf-8"))
        witness = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                       and node.name == "test_every_boundary_time_entry_has_a_driver_measured_frame")
        assert any(isinstance(node, ast.ImportFrom) and node.module == "gpuwm.core.kernels"
                   and any(name.name == "load_module" and name.asname is None
                           for name in node.names) for node in ast.walk(witness))
        assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id == "load_module" and len(node.args) == 1
                   and isinstance(node.args[0], ast.Constant)
                   and node.args[0].value == "lbc_time" for node in ast.walk(witness))
        assert any(isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute)
                   and node.value.attr == "attributes" and isinstance(node.slice, ast.Constant)
                   and node.slice.value == "local_size_bytes" for node in ast.walk(witness))
        loader = ast.parse((REPOSITORY_ROOT / "gpuwm/core/kernels/__init__.py").read_text(encoding="utf-8"))
        assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr == "RawModule" for node in ast.walk(loader))
        return
    if entry == "tests/test_ensemble_recipe_door_gpu.py":
        # The recipe door's end-to-end test types the real command, so its
        # device is opened by a `python -m gpuwm` subprocess and not by an
        # import this file makes.  Pin the whole route: the helper runs the
        # package entry point, the door is run for real (not only as a dry
        # run), the door hands its members to the prepared forecast runner,
        # and that runner is device-bound by conftest's own detector.
        import ast
        tree = ast.parse((REPOSITORY_ROOT / entry).read_text(encoding="utf-8"))
        functions_by_name = {node.name: node for node in tree.body
                             if isinstance(node, ast.FunctionDef)}
        launcher = functions_by_name["_gpuwm"]
        assert any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run" and node.args
            and isinstance(node.args[0], ast.List)
            and [item.value for item in node.args[0].elts
                 if isinstance(item, ast.Constant)] == ["-m", "gpuwm"]
            for node in ast.walk(launcher)), (
            f"{entry} no longer runs the `python -m gpuwm` entry point")
        door_calls = [
            node for node in ast.walk(functions_by_name["_run_door"])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "_gpuwm" and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "ensemble"]
        real_runs = [
            call for call in door_calls
            if "--dry-run" not in [item.value for item in call.args
                                   if isinstance(item, ast.Constant)]]
        assert real_runs, f"{entry} no longer runs `gpuwm ensemble` for real"
        door = ast.parse((REPOSITORY_ROOT / "gpuwm/ensemble/recipe_door.py")
                         .read_text(encoding="utf-8"))
        runner_names = {
            alias.asname or alias.name for node in ast.walk(door)
            if isinstance(node, ast.ImportFrom) and node.module == "gpuwm"
            for alias in node.names if alias.name == "prepared_single_domain_forecast"}
        assert any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "main" and isinstance(node.func.value, ast.Name)
            and node.func.value.id in runner_names for node in ast.walk(door)), (
            "the recipe door no longer hands its members to the prepared "
            "forecast runner")
        runner_whole, runner_functions = _cupy_scope(
            str(REPOSITORY_ROOT / "gpuwm/prepared_single_domain_forecast.py"))
        assert runner_whole or runner_functions, (
            "the prepared forecast runner no longer opens CUDA")
        return
    if entry == "tests/test_ensemble_go_door_gpu.py":
        # The go door's end-to-end test types `gpuwm go CONFIG --members N`,
        # so, as for the recipe door above, a `python -m gpuwm` subprocess
        # opens the device.  Pin the whole route: the helper runs the
        # package entry point, go is run for real (not only as a dry run),
        # go's member route enters the ensemble door's production scope,
        # that scope builds the production session, and the session's
        # module is device-bound by conftest's own detector.
        import ast
        tree = ast.parse((REPOSITORY_ROOT / entry).read_text(encoding="utf-8"))
        functions_by_name = {node.name: node for node in tree.body
                             if isinstance(node, ast.FunctionDef)}
        assert any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run" and node.args
            and isinstance(node.args[0], ast.List)
            and [item.value for item in node.args[0].elts
                 if isinstance(item, ast.Constant)] == ["-m", "gpuwm"]
            for node in ast.walk(functions_by_name["_gpuwm"])), (
            f"{entry} no longer runs the `python -m gpuwm` entry point")
        go_calls = [
            node for node in ast.walk(functions_by_name["_go"])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "_gpuwm" and node.args
            and isinstance(node.args[0], ast.Constant) and node.args[0].value == "go"]
        assert any("--dry-run" not in [item.value for item in call.args
                                       if isinstance(item, ast.Constant)]
                   for call in go_calls), f"{entry} no longer runs `gpuwm go` for real"
        go = ast.parse((REPOSITORY_ROOT / "gpuwm/go_cli.py").read_text(encoding="utf-8"))
        assert any(
            isinstance(node, ast.ImportFrom) and node.module == "gpuwm.ensemble.door"
            and any(alias.name == "production_run_scope" for alias in node.names)
            for node in ast.walk(go)) and any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "production_run_scope" for node in ast.walk(go)), (
            "gpuwm go no longer runs its members inside the ensemble door's "
            "production scope")
        door = ast.parse((REPOSITORY_ROOT / "gpuwm/ensemble/door.py").read_text(encoding="utf-8"))
        assert any(
            isinstance(node, ast.ImportFrom) and node.module == "gpuwm.ensemble.production"
            and any(alias.name == "PreparedEnsembleSession" for alias in node.names)
            for node in ast.walk(door)), (
            "the ensemble door's production scope no longer builds the "
            "production session")
        session_whole, session_functions = _cupy_scope(
            str(REPOSITORY_ROOT / "gpuwm/ensemble/production.py"))
        assert session_whole or session_functions, (
            "the ensemble production session no longer opens CUDA")
        return
    if entry == "tests/test_ensemble_native_decline_gpu.py":
        # The decline gate runs the production session and the wrfinput
        # door's own runner in this process; it imports neither cupy nor a
        # helper that does, so conftest's detector reads the file as CPU.
        # Pin the route: the gate imports the prepared tree runner and hands
        # it to the session's run_prepared, and that runner is device-bound
        # by conftest's own detector.
        import ast
        tree = ast.parse((REPOSITORY_ROOT / entry).read_text(encoding="utf-8"))
        assert any(
            isinstance(node, ast.ImportFrom)
            and node.module == "gpuwm.prepared_domain_tree_forecast"
            and any(alias.name == "run_prepared_tree" and alias.asname is None
                    for alias in node.names)
            for node in ast.walk(tree)), (
            f"{entry} no longer imports the prepared tree runner")
        assert any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run_prepared" and node.args
            and isinstance(node.args[0], ast.Name) and node.args[0].id == "run_prepared_tree"
            for node in ast.walk(tree)), (
            f"{entry} no longer runs its members through the prepared tree runner")
        runner_whole, runner_functions = _cupy_scope(
            str(REPOSITORY_ROOT / "gpuwm/prepared_domain_tree_forecast.py"))
        assert runner_whole or runner_functions, (
            "the prepared tree runner no longer opens CUDA")
        return
    if entry == "tests/test_urban_ucm_wrf461_parity.py":
        # Its device read belongs to the production replay helper, while
        # this file also owns CPU fixture and native-twin rows.
        import ast
        tree = ast.parse((REPOSITORY_ROOT / entry).read_text(encoding="utf-8"))
        assert any(
            isinstance(node, ast.ImportFrom)
            and node.module == "gpuwm.verify.urban_ucm_oracle"
            and any(alias.name == "replay" and alias.asname is None for alias in node.names)
            for node in ast.walk(tree)), (
            "the original WRF urban oracle no longer imports the device replay")
        assert any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "replay" for node in ast.walk(tree)), (
            "the original WRF urban oracle no longer executes the device replay")
        module_whole, module_functions = _cupy_scope(
            str(REPOSITORY_ROOT / "gpuwm/core/urban_ucm.py"))
        assert module_whole or "run_columns" in module_functions, (
            "the original WRF urban replay no longer reaches CUDA columns")
        return
    if entry in DEVICE_HELPERS:
        # The entry must still use its helper, and the helper must still
        # read device words; either change turns this red.
        import ast
        helper = DEVICE_HELPERS[entry]
        tree = ast.parse((REPOSITORY_ROOT / entry).read_text(encoding="utf-8"))
        assert any(isinstance(node, ast.Name) and node.id == helper
                   and isinstance(node.ctx, ast.Load) for node in ast.walk(tree)), (
            f"{entry} no longer uses {helper}")
        assert _helper_opens_cuda(pathlib.PurePosixPath(entry).stem, helper,
                                  REPOSITORY_ROOT / "tests"), (
            f"{entry}: {helper} no longer reaches a CuPy read of device words")
        return
    assert whole or functions, (
        f"{entry} opens no CUDA device by conftest's own detector, so it is "
        "already covered by a CPU leg and does not belong on the GPU shard.  "
        "If it reaches a device transitively, say so here and pin the reason")


#: Ensemble device files that run their forecasts in `python -m gpuwm`
#: subprocesses, so they leave nothing on the card in the leg's own process.
ENSEMBLE_SUBPROCESS_DOORS = frozenset({
    "tests/test_ensemble_recipe_door_gpu.py",
    "tests/test_ensemble_go_door_gpu.py",
})


def test_the_native_decline_gate_is_the_first_ensemble_device_file() -> None:
    """Shard 1 runs in one pytest process, so its order is card memory.

    tests/test_ensemble_native_decline_gpu.py runs four ordinary members
    after a native pack, and the ordinary runner's memory gate prices the
    card by its free bytes.  Run after the real-input ensemble files in one
    process it was refused with 3.66 GiB free (lane fix/285-ens-runcontrol,
    RTX 5090); alone it passed.  Listed below them, the leg would fail it on
    memory that earlier tests had already released, not on the decline it
    gates.  The two door files run their forecasts in subprocesses and may
    sit anywhere.
    """

    shard1 = _entries("shard1")
    gate = "tests/test_ensemble_native_decline_gpu.py"
    assert gate in shard1, f"{gate} left GPU shard 1"
    earlier = [entry for entry in shard1[:shard1.index(gate)]
               if pathlib.PurePosixPath(entry).name.startswith("test_ensemble_")
               and entry not in ENSEMBLE_SUBPROCESS_DOORS]
    assert not earlier, (
        f"{earlier} run before {gate} in the shard's one process; move the "
        "decline gate above them (SHARD 1w0 in tools/battery/gpu_shard_files.txt)")


@pytest.mark.parametrize("reads_device", [True, False])
def test_the_helper_rule_follows_imports_and_refuses_a_cpu_chain(
        tmp_path: pathlib.Path, reads_device: bool) -> None:
    """The DEVICE_HELPERS exemption holds only while the chain reads CUDA.

    Two hops, as in the ensemble runtime pair: the entry imports a class
    from a sibling module, and that class loads a word reader imported
    from a third.  Take the cupy import out of the reader and the entry
    is refused like any CPU file.
    """

    reader = ("def words(array):\n    import cupy as cp\n    return cp.asnumpy(array)\n"
              if reads_device else
              "def words(array):\n    import numpy as np\n    return np.asarray(array)\n")
    (tmp_path / "test_reader_mod.py").write_text(reader, encoding="utf-8")
    (tmp_path / "test_owner_mod.py").write_text(
        "from test_reader_mod import words\n\n\nclass Capture:\n"
        "    def record(self, value):\n        return words(value)\n",
        encoding="utf-8")
    (tmp_path / "test_entry_mod.py").write_text(
        "from test_owner_mod import Capture\n\n\ndef test_x():\n    Capture().record(1)\n",
        encoding="utf-8")
    assert _helper_opens_cuda("test_entry_mod", "Capture", tmp_path) is reads_device


def test_the_dycore_parity_files_the_audit_named_are_all_present() -> None:
    """Gap G2's own enumeration, pinned so the list cannot quietly shrink.

    These are the files the audit measured as 100% skipped on CPU, by name
    and by test count: openbc 47, diff6_boundary_face 43, diff6 39, smag3d
    19, terrain_dynamics 19, smag2d 18, small_step_lateral_wrap 14,
    mapped_mass_closure 12, acoustic 11, tke_budget 11, tke_km2 10.  Every
    one of those counts was reproduced on the first GPU run.
    """

    required = {
        "tests/test_openbc.py", "tests/test_diff6.py",
        "tests/test_diff6_boundary_face.py", "tests/test_smag2d.py",
        "tests/test_smag3d.py", "tests/test_terrain_dynamics.py",
        "tests/test_small_step_lateral_wrap.py",
        "tests/test_mapped_mass_closure.py", "tests/test_tke_budget.py",
        "tests/test_tke_km2.py", "tests/test_acoustic.py",
        "tests/test_sase_cadence_gpu.py", "tests/test_adaptive_stream_gpu.py",
        "tests/test_rrtm_memory.py", "tests/test_relocation_held_physics_gpu.py",
        "tests/test_attribute_tracking_gpu.py",
        "tests/test_model_stability_gpu.py",
        "tests/test_nssl2_gpu.py",
    }
    missing = sorted(required - set(_entries("shard1")))
    assert not missing, (
        f"{missing} were named by gap G2 as running on no list and have been "
        "dropped from shard 1.  Removing one is allowed, but it is an "
        "argument made in a commit message, not a tidy-up")


def test_the_wrf_reader_conformance_file_is_on_the_per_cut_shard() -> None:
    """The single entry that most justifies the leg existing.

    Its own docstring says every other wrfout test is self-consistency and
    this is the only file that judges a gpuwm wrfout with a WRF reader.  It
    is 18 of 18 skipped on a CPU box, so a CPU-only battery never asks the
    one question a user's tooling will ask.
    """

    assert "tests/test_wrfout_conformance.py" in _entries("shard1")


def test_compiled_wrf_dycore_device_oracles_remain_on_the_per_cut_shard() -> None:
    """A CPU fixture pass cannot replace an executed output-word pin."""
    missing = sorted(DYCORE_PARITY_FILES - set(_entries("shard1")))
    assert not missing, (
        f"{missing} are absent from GPU shard 1; a cut would omit these "
        "compiled WRF device comparisons")


#: The filename prefixes of the cumulus leaf-port families the weekly shard
#: admits, one per byte-frozen WRF v4.6.1 oracle corpus.  Grell-Freitas was
#: the founding family; New Tiedtke joined with cu_physics = 16 (d0c23dad0,
#: 2.6.4), whose 22 stage and pipeline parity suites are the same shape:
#: device rows graded against a frozen oracle, carrying the mass and almost
#: none of the defects.  A third family widens the manifest's header first,
#: then this tuple.
WEEKLY_SHARD_FAMILIES = ("test_gf_", "test_ntiedtke_")


def test_the_weekly_shard_is_the_frozen_cumulus_oracle_families() -> None:
    """Shard 2's membership follows a stated rule, not a mood.

    The audit's shard-design finding is that defect density is inverted
    against test mass: frozen bit-exact WRF leaf ports carry the tests and
    almost none of the defects.  GF cumulus is 376 GPU tests and nine of its
    ten files have never co-moved with a fix, which makes it the right first
    thing to move off the per-cut cadence -- and the wrong thing to delete.
    The New Tiedtke parity suites are the second family on the same terms.

    tests/test_cumulus_momentum_extension.py is deliberately NOT admitted:
    it is the coupling contract every cumulus scheme runs through (the
    inertness gate for the schemes that supply no momentum) rather than an
    oracle parity suite, so it sits on the per-cut shard.  7391fa8db listed
    it here beside the Tiedtke family and this rule caught the placement.
    """

    shard2 = _entries("shard2")
    assert shard2, "shard 2 is empty"
    for entry in shard2:
        name = pathlib.PurePosixPath(entry).name
        assert name.startswith(WEEKLY_SHARD_FAMILIES), (
            f"{entry} is in the weekly shard but is not part of a frozen "
            f"cumulus oracle family ({', '.join(WEEKLY_SHARD_FAMILIES)}), "
            "which is the only rule shard 2 currently states.  Widen the "
            "rule in the file's header and in WEEKLY_SHARD_FAMILIES, or "
            "move the entry")
    # Both families are present, so the rule is measured on each rather
    # than satisfied by whichever one happens to be listed.
    for prefix in WEEKLY_SHARD_FAMILIES:
        assert any(pathlib.PurePosixPath(e).name.startswith(prefix)
                   for e in shard2), prefix
    assert "tests/test_cumulus_momentum_extension.py" in _entries("shard1")


def test_sfire_device_controls_remain_on_per_cut_shard() -> None:
    """Dropping these entries would omit SFIRE compiled WRF comparisons and coupled GPU trajectories."""
    required = {
        "tests/test_sfire_atm_wrf471_parity.py",
        "tests/test_sfire_clock_gpu.py",
        "tests/test_sfire_dated_perimeter_gpu.py",
        "tests/test_sfire_debug_runtime.py",
        "tests/test_sfire_driver_wrf471_parity.py",
        "tests/test_sfire_moisture_driver_wrf471_parity.py",
        "tests/test_sfire_negative_burn_gpu.py",
        "tests/test_sfire_open_geopotential_wrf471_parity.py",
        "tests/test_sfire_ranked_gpu.py",
        "tests/test_sfire_restart_coupled_gpu.py",
        "tests/test_sfire_smoke.py",
        "tests/test_sfire_smoke_tiled_gpu.py",
        "tests/test_sfire_spotting_streamed_gpu.py",
        "tests/test_sfire_spotting_wrf471_parity.py",
        "tests/test_sfire_tilestream_gpu.py",
        "tests/test_sfire_wind_wrf471_parity.py",
        "tests/test_sfire_wrf471_parity.py",
    }
    missing = sorted(required - set(_entries("shard1")))
    assert not missing, (f"{missing} have no required SFIRE battery leg")
