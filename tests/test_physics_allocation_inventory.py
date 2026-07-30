"""Can a physics scheme allocate its working set outside the arena again?

This is the ratchet that ``tests/test_mynn_pbl_scratch.py`` opened, and it
exists because MYNN got all the way to a production-width timing measurement
while allocating **54,595 bytes per column in 484 pool allocations per step**
that no part of the preflight knew about.  At the 360,000 columns of a d04
nest that is 18,743.8 MiB per step against an estimate that did not move at
all.  The failure mode is not slowness.  It is that ``run_alloc_preflight``
returns a comfortable headroom number for a configuration that will run the
card out of memory, on hardware with no ECC, where the observed symptom of
running out is corrupted output rather than an exception.

WHY IT IS A SEPARATE FILE
-------------------------
Every gate here reads source with :mod:`ast` and opens no device.  It used to
live in ``tests/test_mynn_pbl_scratch.py``, whose ``_carried_hash`` helper
imports cupy -- and ``tests/conftest.py`` marks a WHOLE module ``gpu`` when any
non-test function in it imports cupy, deliberately, because which tests call a
helper is not decidable from the AST.  So a purely static, fully deterministic
gate was skipped on every ``GPUWM_NO_LOCAL_GPU=1`` run, which is every routine
run on this machine.

That is not hypothetical.  At ``cb72667`` the ratchet was red in two places at
once and no CPU run could report either:

* four ``noahmp_*_gpu.py`` modules were committed with allocation sites and
  never entered below (``92216d0`` and earlier -- the triage note in
  ``docs/tree_triage_2026-07-26.md`` names them); and
* ``gpuwm/core/ruc_gpu.py`` grew ``ruc_tanhf_glibc`` at ``dd0df4e``, a
  fourteenth site in a module recorded with thirteen -- which is the *count
  increase* this ratchet exists to catch, and which was invisible even to a
  GPU run, because the module-set assertion fired first and stopped the test
  before the counts were compared.

Both are fixed below: the file no longer needs a device, and the two
assertions now report together instead of one masking the other.

WHAT THE INVENTORY MEANS
------------------------
It is a debt register, not a waiver.  ``mynn_pbl_gpu`` is at zero because the
solver draws its whole working set from ``MynnPblScratch``; everything else is
recorded at the count it has today and fails on any increase.  The distinction
that decides whether a recorded row is acceptable is the one YSU's entry
states: **an allocation the preflight knows about is a cost; an allocation it
does not is a hazard** -- and the property that separates MYNN's hazard from
every row below is whether the allocation scales with the nest.  For the
Noah-MP leaf batches that property is gated, not asserted, by
:func:`test_the_noahmp_leaf_batch_transients_are_bounded_by_the_column_batch`.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Raw device-allocation entry points.  ``asarray``/``ascontiguousarray`` are
#: deliberately absent: they are no-ops on an array that is already
#: contiguous float32, which is what every runtime input is, and they are the
#: only sanctioned way to accept a host array from an oracle fixture.  They
#: are covered instead by the churn measurement in the module docstring of
#: gpuwm/core/mynn_pbl_scratch.py, which counts what actually gets allocated.
_ALLOCATORS = frozenset((
    "empty", "zeros", "ones", "full",
    "empty_like", "zeros_like", "ones_like", "full_like",
))

#: The two MYNN modules the arena lane converted.  Their entries in the
#: inventory below are the enumerated exceptions, and each one is a place with
#: no ``DomainState`` to draw from that allocates once rather than per step.
_CONVERTED = ("gpuwm/core/mynn_pbl_gpu.py", "gpuwm/core/mynn_pbl_runtime.py")

#: The four Noah-MP leaf batches on the forecast's own path.  Every one of
#: them is reached from ``noahmp_runtime.DEVICE_LEAF_BATCH``, which is what
#: ``LEAF_BATCH_EVALUATOR`` resolves to by default, so these run per leaf per
#: staging round of every Noah-MP timestep.  (``noahmp_fluxprep_gpu`` and
#: ``noahmp_leaves_gpu``, also recorded below, are reached only from the
#: oracle tests -- nothing in ``gpuwm`` imports them.)
_NOAHMP_LEAF_BATCHES = (
    "gpuwm/core/noahmp_bareflux_gpu.py",
    "gpuwm/core/noahmp_radiation_gpu.py",
    "gpuwm/core/noahmp_sflx_pre_gpu.py",
    "gpuwm/core/noahmp_water_gpu.py",
)

#: The same packing with the column axis vectorised, which is the Noah-MP
#: conversion in flight.  Nothing under ``gpuwm/`` imports either yet -- only
#: their own tests do -- so they are not on the forecast path today, and the
#: ``COLUMN_BATCH`` bound that covers the four above therefore does not cover
#: these.  They are gated for the same shape property anyway, because the day
#: they are wired in is the day the bound has to already be true.
_NOAHMP_SLAB_BATCHES = (
    "gpuwm/core/noahmp_flux_slab.py",
    "gpuwm/core/noahmp_sflx_post_slab.py",
    "gpuwm/core/noahmp_sflx_pre_slab.py",
    "gpuwm/core/noahmp_thermal_slab.py",
    "gpuwm/core/noahmp_water_slab.py",
)

#: Names that ARE a grid dimension.  Used for the slab packers, where the
#: stride constants are still being written -- one per leaf, `THERMOPROP_N_IN`,
#: `TSNOSOI_N_OUT` and six more so far -- so an exhaustive allowlist of them
#: would fire on every commit of the conversion for a reason that is not a
#: defect, and a gate that cries wolf gets deleted.  The load-bearing half of
#: the property is checked either way: the shape must read the batch width.
#: The four modules already on the forecast path keep the strict form, where
#: the name set is settled and an unrecognised name is worth a failure.
_GRID_DIMENSION_NAMES = frozenset((
    "nx", "ny", "nz", "ncol", "ncols", "ncolumn", "ncolumns",
    "ims", "ime", "jms", "jme", "kms", "kme", "its", "ite", "jts", "jte",
))

#: Every name a Noah-MP leaf batch may size a device allocation from.  The
#: whole content of the claim is that ``count``/``n`` is ``len(calls)`` and
#: nothing here is a grid dimension: with the staging loop flushing at
#: ``COLUMN_BATCH``, that makes every allocation below bounded by the batch
#: rather than by the nest.  The rest are per-column strides -- ``nfull`` is
#: ``nsnow + nsoil``, and the three tuples are output-field lists.
_BATCH_WIDTH_NAMES = frozenset(("count", "n"))
_PER_COLUMN_STRIDE_NAMES = frozenset((
    "nfull", "N_OUTPUT", "OUT_STRIDE", "_ATM_OUT", "_PHEN_OUT", "_PRCP_OUT",
    # the slab packers' spellings of the same thing
    "_NFULL", "P_STRIDE", "IN_STRIDE", "INT_STRIDE",
    # ``_batch._ATM_IN``, ``_vege.N_INPUT``, ``_bare.N_OUTPUT``: the slab
    # packers import the per-column packer's layout constants rather than
    # restating them -- deliberately, so the two cannot drift in slot order --
    # so the module handle is what the shape expression reads.
    "_batch", "_vege", "_bare",
))

#: Every ``cp.<allocator>`` call site in every physics GPU module, by the
#: function that owns it.  Written out rather than summarised so that adding
#: one is a visible diff on this line, and so that the modules nobody has
#: converted yet are an itemised debt rather than an unexamined blank.
#:
#: ``mynn_pbl_gpu`` is empty: the solver draws its whole working set from
#: ``MynnPblScratch``.  ``mynn_pbl_runtime`` keeps exactly two, both
#: justified above their call sites -- the six returned tendency fields on
#: the no-``DomainState`` path used by the standalone harnesses, and the
#: 256-byte validity words, built once per process.
_PHYSICS_ALLOCATION_INVENTORY = {
    'gpuwm/core/acoustic.py': {},
    'gpuwm/core/advection.py': {
        '_launch': 1,
        'advect_scalar_rk3_periodic_test': 7,
    },
    'gpuwm/core/diagnostics.py': {},
    'gpuwm/core/diffusion.py': {
        'diffuse_only_test': 1,
    },
    'gpuwm/core/dycore.py': {
        'launch_coriolis_curvature': 2,
        'launch_diff6': 2,
    },
    'gpuwm/core/health.py': {},
    'gpuwm/core/kf.py': {
        'launch_kf': 10,
        'update_trigger_history': 1,
    },
    'gpuwm/core/microphysics.py': {},
    'gpuwm/core/microphysics_transition.py': {},
    'gpuwm/core/moist.py': {
        'launch_pd_fluxes': 1,
        'launch_pd_renorm_apply': 1,
    },
    'gpuwm/core/morrison.py': {
        'launch_morrison': 3,
    },
    'gpuwm/core/mynn_pbl_gpu.py': {},
    'gpuwm/core/mynn_pbl_runtime.py': {
        '_validity_flags': 1,
        'mynn_pbl_step': 1,
    },
    'gpuwm/core/mynn_sfclay.py': {
        '_allocate_result': 1,
        '_surface_array': 1,
    },
    'gpuwm/core/noah.py': {},
    # ---- the four Noah-MP leaf batches -----------------------------------
    # Entered 2026-07-27, having been committed unlisted since at least
    # 92216d0.  They were checked against the question that decides whether a
    # row is a cost or a hazard, and the answer is that they are costs:
    #
    #   * every shape is ``len(calls)`` by a fixed per-column stride, gated by
    #     ``test_the_noahmp_leaf_batch_transients_are_bounded_by_the_
    #     column_batch`` below -- no grid dimension appears in any; and
    #   * ``noahmp_runtime.noahmp_lsm_step`` flushes its staging list at
    #     ``COLUMN_BATCH``, so ``len(calls)`` never exceeds 2,048 whatever the
    #     nest is.
    #
    # Derived at that width, in float32: bare_flux ``count*13``, radiation
    # ``count*19``, water ``count*73 + count`` int32, sflx_pre
    # ``n*(17+1+7+1+1+6+16)`` = ``49n``.  620 B per staged column, so 1.21 MiB
    # for all four together at COLUMN_BATCH -- against MYNN's 18,743.8 MiB,
    # which is the comparison that makes this bookkeeping rather than the
    # defect that caused this file to exist.
    #
    # It is still 1.21 MiB the preflight does not price: ``gpuwm/core/
    # preflight.py`` has no Noah-MP transient term at all (its Noah-MP rows
    # are kernel local-memory frames and the snow stack).  That is recorded
    # here and handed over rather than fixed here, because pricing it moves
    # numbers that tests/test_preflight.py pins.
    'gpuwm/core/noahmp_bareflux_gpu.py': {
        'evaluate_bare_flux_calls': 1,
    },
    'gpuwm/core/noahmp_fluxprep_gpu.py': {
        'evaluate_fluxprep_leaf': 1,
    },
    'gpuwm/core/noahmp_leaves_gpu.py': {
        'evaluate_leaf': 1,
    },
    'gpuwm/core/noahmp_radiation_gpu.py': {
        'evaluate_radiation_calls': 1,
    },
    'gpuwm/core/noahmp_sflx_pre_gpu.py': {
        'evaluate_sflx_pre_calls': 7,
    },
    # ---- and the same packing with the column axis vectorised ------------
    # Entered at the counts they had when they were committed (04a40cc), and
    # the Noah-MP conversion is live: expect this to move, and expect the
    # ratchet to say so, which is the point of it.
    #
    # Nothing under gpuwm/ imports either yet, so neither is on a forecast
    # path and neither inherits the COLUMN_BATCH bound.  WHEN THEY ARE WIRED
    # IN, WHATEVER CALLS THEM MUST CARRY A BOUND.  A slab that takes every
    # land column at once prices, at d04's 360,000 columns and in float32:
    # water pack (26 + 76 + 9) B/col plus evaluate (73 + 1) B/col = 185 B/col
    # = 63.5 MiB, and sflx_pre (17 + 17 + 1 + 7 + 1 + 1 + 6 + 16 + 1) = 268
    # B/col = 92.0 MiB.  155 MiB is survivable against the 29,500 MiB rail
    # and it is 128x what the staged path costs, so it is a number that has
    # to be decided rather than discovered.
    'gpuwm/core/noahmp_flux_slab.py': {
        'evaluate_bare_flux_slab': 2,
        'evaluate_vege_flux_slab': 2,
        'pack_bare_flux_slab': 2,
        'pack_vege_flux_slab': 2,
    },
    'gpuwm/core/noahmp_sflx_post_slab.py': {
        '_error_slab': 5,
        'sflx_post_seg2': 1,
    },
    'gpuwm/core/noahmp_sflx_pre_slab.py': {
        'evaluate_sflx_pre_slab': 9,
    },
    'gpuwm/core/noahmp_thermal_slab.py': {
        '_integers': 1,
        'evaluate_phasechange_slab': 1,
        'evaluate_radiation_slab': 1,
        'evaluate_thermoprop_slab': 1,
        'evaluate_tsnosoi_slab': 1,
        'pack_phasechange_slab': 1,
        'pack_radiation_slab': 1,
        'pack_thermoprop_slab': 1,
        'pack_tsnosoi_slab': 1,
    },
    'gpuwm/core/noahmp_water_gpu.py': {
        'evaluate_water_calls': 2,
    },
    'gpuwm/core/noahmp_water_slab.py': {
        'evaluate_water_slab': 2,
        'pack_water_slab': 3,
    },
    'gpuwm/core/nssl2.py': {},
    'gpuwm/core/nssl2_diagnostics.py': {},
    'gpuwm/core/nssl2_driver_support.py': {
        'gather_initialize_and_sediment': 4,
    },
    'gpuwm/core/nssl2_fused_gs.py': {},
    'gpuwm/core/nssl2_nucond.py': {
        'launch_nucond': 1,
    },
    'gpuwm/core/nssl2_qvexcess.py': {},
    'gpuwm/core/nssl2_radiation.py': {},
    'gpuwm/core/refl.py': {},
    'gpuwm/core/rrtmgp.py': {
        '__call__': 6,
        '_cloud_optics': 1,
        '_finalize_cloud_optics': 2,
        '_gas_optics': 5,
        '_gas_vmr': 1,
        '_interface_temperatures': 1,
        '_interpolation_metadata': 2,
        '_lw_rte': 3,
        '_planck_sources': 3,
        '_sw_rte': 3,
        '_validation_flags_device': 1,
        '_workspace_output': 1,
        'delta_scale': 1,
        'field': 1,
        'hydrometeor_paths': 2,
    },
    'gpuwm/core/ruc_gpu.py': {
        '_soil_phase_partition_cuda': 1,
        'ruc_qsn_cuda': 1,
        'ruc_sea_ice_step_cuda': 3,
        'ruc_snow_preparation_cuda': 3,
        'ruc_snow_sea_ice_step_cuda': 3,
        'ruc_snow_soil_step_cuda': 11,
        'ruc_snow_temperature_step_cuda': 3,
        'ruc_soil_moisture_step_cuda': 2,
        'ruc_soil_properties_cuda': 1,
        'ruc_soil_step_cuda': 6,
        'ruc_soil_temperature_step_cuda': 2,
        'ruc_surface_parameters_cuda': 2,
        # Added at dd0df4e with the whole-column device path, and the one row
        # here that DOES scale with the nest: the argument is the snow-covered
        # subset of the RUC batch, so ``module_sf_ruclsm.F:2087``'s rebuild
        # allocates one float32 output per snow-covered column.  Fully
        # snow-covered d04 is 360,000 columns = 1.37 MiB, once per snow-cover
        # rebuild, and the input copy beside it is the same again.  Recorded
        # rather than removed because the alternative spelling is the Python
        # ``for`` loop over fdlibm's reduction that this replaced.
        'ruc_tanhf_glibc': 1,
        'ruc_transpiration_cuda': 2,
    },
    'gpuwm/core/sfclay.py': {
        '_allocate_result': 1,
        '_surface_array': 1,
    },
    'gpuwm/core/thompson.py': {},
    # UP_HELI_MAX's every-step maximum runs entirely inside the caller's
    # persistent slots -- the scanner finds no bare allocation, and this
    # empty row is the assertion of that, not an unfilled placeholder.
    'gpuwm/core/uh_diag.py': {},
    'gpuwm/core/wsm6.py': {},
    'gpuwm/core/ysu.py': {
        # The precedent this rule follows and the reason it is a ratchet
        # rather than a ban: YSU's nine per-call outputs are raw allocations,
        # but they are priced -- preflight.py's ysu_output_transient_shapes
        # enumerates every one of them by name.  An allocation the preflight
        # knows about is a cost; an allocation it does not is a hazard.
        # MYNN's were the second kind.
        'launch_ysu': 9,
    },
}


def _tracked_core_modules() -> frozenset[str] | None:
    """``gpuwm/core`` files git knows about, or ``None`` if git cannot say.

    The ratchet is a claim about what this repository ships, and an untracked
    file cannot enter a commit.  Scoping to tracked paths is therefore not a
    loosening -- ``git ls-files`` lists staged files too, so a module is
    covered from the moment anyone prepares it for a commit, which is strictly
    before it can be pushed.

    It is here because this tree is shared: a second lane converting Noah-MP
    wrote ``noahmp_sflx_pre_slab.py`` and ``noahmp_water_slab.py`` into
    ``gpuwm/core`` while this file was being written, and pinning per-function
    counts for another lane's half-finished module would record a number that
    is wrong by the time it is read.  When git cannot answer, everything on
    disk is scanned, which is the strict direction.
    """
    try:
        completed = subprocess.run(
            ["git", "ls-files", "--", "gpuwm/core/*.py"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):       # pragma: no cover
        return None
    if completed.returncode != 0:                       # pragma: no cover
        return None
    tracked = frozenset(line.strip() for line in completed.stdout.splitlines()
                        if line.strip())
    return tracked or None


def _physics_gpu_modules() -> tuple[str, ...]:
    """Every tracked ``gpuwm/core`` module that launches a kernel, by AST.

    Discovered rather than listed, so a scheme added tomorrow inherits the
    gate without anyone remembering to add it here.
    """
    tracked = _tracked_core_modules()
    found = []
    for path in sorted((ROOT / "gpuwm" / "core").glob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if tracked is not None and rel not in tracked:
            continue
        source = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "get_kernel"):
                found.append(path.relative_to(ROOT).as_posix())
                break
    return tuple(found)


def _allocation_sites(source: str, rel: str):
    """Every ``cp.<allocator>(...)`` call in ``source``, with its function."""
    found = []

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.stack = ["<module>"]

        def _scoped(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        visit_FunctionDef = _scoped
        visit_AsyncFunctionDef = _scoped

        def visit_ClassDef(self, node):
            self._scoped(node)

        def visit_Call(self, node):
            func = node.func
            if (isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id in ("cp", "cupy")
                    and func.attr in _ALLOCATORS):
                found.append((rel, self.stack[-1], func.attr, node.lineno,
                              node))
            self.generic_visit(node)

    Visitor().visit(ast.parse(source))
    return found


def _inventory(source: str, rel: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for site in _allocation_sites(source, rel):
        function = site[1]
        counts[function] = counts.get(function, 0) + 1
    return counts


def _live_inventory() -> dict[str, dict[str, int]]:
    return {rel: _inventory((ROOT / rel).read_text(encoding="utf-8"), rel)
            for rel in _physics_gpu_modules()}


def test_this_gate_opens_no_device_and_must_never_start():
    """The reason this file is not part of ``test_mynn_pbl_scratch.py``.

    ``conftest._cupy_scope`` is the same function that
    ``pytest_collection_modifyitems`` uses to decide what gets marked
    ``gpu`` and skipped under
    ``GPUWM_NO_LOCAL_GPU=1``.  Asked about this module it must answer "no cupy
    anywhere", because the moment it answers otherwise every assertion below
    stops running on the only machine that routinely runs the suite -- which
    is exactly how four Noah-MP modules and one RUC function stayed missing
    from the inventory.
    """
    from conftest import _cupy_scope

    whole, functions = _cupy_scope(str(Path(__file__).resolve()))
    assert (whole, functions) == (False, frozenset()), (
        "this module now opens a device somewhere, so conftest will mark it "
        "gpu and skip the whole ratchet on every CPU run: "
        f"whole={whole} functions={sorted(functions)}")


def test_the_ast_gate_catches_a_raw_allocation():
    """The failing form, first, in both a converted and an unconverted module.

    Without this the passing tests below could be passing because the
    scanner is broken.  The injected line is the exact shape the old solver
    used -- an ``(ncol, nz)`` work array inside the leaf that needs it -- and
    it is injected twice: into ``mynn_pbl_gpu``, whose recorded count is
    zero, and into ``ysu``, whose recorded count is nine.  A ratchet that
    only notices the first-from-zero case would not be a ratchet.
    """
    converted = ROOT / "gpuwm/core/mynn_pbl_gpu.py"
    source = converted.read_text(encoding="utf-8")
    anchor = "    blocks = (ncol + _TPB - 1) // _TPB\n"
    assert anchor in source
    injected = source.replace(
        anchor, "    sneaky = cp.empty((ncol, nz), dtype=DTYPE)\n" + anchor, 1)
    sites = _allocation_sites(injected, "gpuwm/core/mynn_pbl_gpu.py")
    assert sites, "the scanner did not see an injected cp.empty"
    assert all(site[2] == "empty" for site in sites)
    # And the same scanner over the real file sees nothing.
    assert not _allocation_sites(source, "gpuwm/core/mynn_pbl_gpu.py")

    unconverted = ROOT / "gpuwm/core/ysu.py"
    ysu_source = unconverted.read_text(encoding="utf-8")
    recorded = _PHYSICS_ALLOCATION_INVENTORY["gpuwm/core/ysu.py"]
    assert _inventory(ysu_source, "gpuwm/core/ysu.py") == recorded
    ysu_anchor = "        rthraten = cp.zeros_like(theta)\n"
    assert ysu_anchor in ysu_source
    ysu_injected = ysu_source.replace(
        ysu_anchor, ysu_anchor + "        extra = cp.zeros_like(theta)\n", 1)
    grown = _inventory(ysu_injected, "gpuwm/core/ysu.py")
    assert grown != recorded, "the ratchet did not see an added allocation"


def test_no_physics_gpu_module_allocates_outside_its_workspace():
    """The property, not the instance: scan every physics GPU module.

    MYNN's converted solver must stay at zero, and no other scheme may add
    a site.  The inventory is compared function by function, so moving an
    allocation between functions is a diff too.

    Both halves are collected before anything is asserted.  They used to be
    two statements, and the first one masked the second for a whole session:
    ``ruc_tanhf_glibc`` was a fourteenth site in a module recorded with
    thirteen from ``dd0df4e`` onward, and no run ever reported it, because the
    module-set assertion above it was already failing on four unrelated
    Noah-MP files.
    """
    live = _live_inventory()
    problems = []
    difference = set(live) ^ set(_PHYSICS_ALLOCATION_INVENTORY)
    if difference:
        problems.append(
            f"physics GPU modules appeared or vanished: {sorted(difference)}"
            " -- entering a new module's row belongs in the commit that adds"
            " the allocation, not in a later sweep; state its bound beside it")
    for rel in sorted(set(live) & set(_PHYSICS_ALLOCATION_INVENTORY)):
        recorded = _PHYSICS_ALLOCATION_INVENTORY[rel]
        if live[rel] == recorded:
            continue
        moved = sorted(set(recorded) | set(live[rel]))
        problems.append(f"{rel}: " + ", ".join(
            f"{name} {recorded.get(name, 0)} -> {live[rel].get(name, 0)}"
            for name in moved if recorded.get(name) != live[rel].get(name)))
    assert not problems, "\n".join(problems)
    for rel in _CONVERTED:
        assert rel in live, rel
    assert live["gpuwm/core/mynn_pbl_gpu.py"] == {}


def test_the_allocation_allowlist_has_not_rotted():
    """Every enumerated exception must still be a real site."""
    for rel in _CONVERTED:
        live = _inventory((ROOT / rel).read_text(encoding="utf-8"), rel)
        stale = set(_PHYSICS_ALLOCATION_INVENTORY[rel]) - set(live)
        assert not stale, (
            f"{rel}: enumerated but no longer allocating: {stale}")


def _shape_names(node: ast.Call) -> set[str]:
    """Every bare name the first positional argument of ``node`` reads.

    That argument is the shape.  ``dtype=`` is skipped deliberately: it names
    ``cp``/``np``, which says nothing about how big the array is, and so is
    anything in callee position -- ``len(_ATM_OUT)`` is the field count, and
    ``len`` is not a dimension.
    """
    if not node.args:
        return set()
    shape = node.args[0]
    callees = {id(inner.func) for inner in ast.walk(shape)
               if isinstance(inner, ast.Call)}
    return {inner.id for inner in ast.walk(shape)
            if isinstance(inner, ast.Name) and id(inner) not in callees}


def _grid_sized_sites(source: str, rel: str, *, strict: bool = True):
    """Allocation sites in ``rel`` that are not sized by the batch.

    A shape that reads no name at all is a literal -- ``cp.empty(0)`` on the
    empty-batch early return -- and cannot scale with anything, so it passes.
    Everything else must name the batch width.

    ``strict`` additionally requires every other name to be a recognised
    per-column stride, which is the right rule where the name set is settled.
    Without it the test is that no name is a grid dimension, which is the
    load-bearing half.
    """
    bad = []
    for _rel, function, allocator, lineno, node in _allocation_sites(
            source, rel):
        names = _shape_names(node)
        if not names:
            continue
        if strict:
            wrong = names - _BATCH_WIDTH_NAMES - _PER_COLUMN_STRIDE_NAMES
        else:
            wrong = names & _GRID_DIMENSION_NAMES
        if wrong or not (names & _BATCH_WIDTH_NAMES):
            bad.append(f"{rel}:{lineno} {function} cp.{allocator} "
                       f"shape reads {sorted(names)}")
    return bad


def test_the_noahmp_leaf_batch_transients_are_bounded_by_the_column_batch():
    """What makes the four Noah-MP rows above a cost and not a hazard.

    The four modules are on the forecast's own path -- ``DEVICE_LEAF_BATCH``
    is what ``LEAF_BATCH_EVALUATOR`` resolves to unless
    ``GPUWM_NOAHMP_HOST_LEAVES=1`` is set -- so "it is only a fixture" is not
    available as a defence, the way it is for ``noahmp_leaves_gpu`` and
    ``noahmp_fluxprep_gpu``, which nothing under ``gpuwm/`` imports.

    What IS available is that none of them can grow with the nest.  Two facts
    together give that, and both are checked here rather than believed:

    1. every allocation's shape is ``len(calls)`` times a fixed per-column
       stride, with no grid dimension anywhere in it; and
    2. ``noahmp_runtime.noahmp_lsm_step`` flushes its staging list at
       ``COLUMN_BATCH``, so ``len(calls)`` is bounded by that constant and not
       by ``nx*ny``.

    Both are read out of the source rather than imported.  ``noahmp_runtime``
    imports all eight leaf batches at module scope and several of those import
    cupy, so importing it here would put a device back into a file whose whole
    purpose is not to need one.

    At ``COLUMN_BATCH`` the four together are 620 B per staged column, or
    1.21 MiB.  MYNN's unpriced set, which is why any of this exists, was
    18,743.8 MiB at d04.  That is the whole difference between the two
    findings, and it is the reason the four rows above were entered rather
    than reported as a defect.

    The two ``*_slab.py`` packers are checked for the same shape property and
    are NOT covered by (2): nothing under ``gpuwm/`` imports them yet, so they
    are on no forecast path and inherit no bound.  Whatever eventually calls
    them has to carry one -- see the note beside their inventory rows for what
    an unbounded slab costs at d04.

    The failing form is shown first, in the module with the most sites: a
    grid-shaped allocation must be named, and a batch-shaped one must not.
    """
    # -- the failing form ---------------------------------------------------
    rel = "gpuwm/core/noahmp_sflx_pre_gpu.py"
    source = (ROOT / rel).read_text(encoding="utf-8")
    anchor = "    d_troot = cp.zeros(n, dtype=cp.float32)\n"
    assert anchor in source, "the anchor moved; re-point this negative control"
    injected = source.replace(
        anchor, "    grid = cp.zeros((ny, nx), dtype=cp.float32)\n" + anchor,
        1)
    named = _grid_sized_sites(injected, rel)
    assert len(named) == 1 and "'nx', 'ny'" in named[0].replace('"', "'"), (
        f"the shape check did not name an (ny, nx) allocation: {named}")

    # -- the failing form again, in the form the slab packers are held to ---
    named = _grid_sized_sites(injected, rel, strict=False)
    assert len(named) == 1 and "'nx', 'ny'" in named[0].replace('"', "'"), (
        f"the non-strict shape check did not name it either: {named}")

    # -- and the real files -------------------------------------------------
    for rel in _NOAHMP_LEAF_BATCHES:
        source = (ROOT / rel).read_text(encoding="utf-8")
        bad = _grid_sized_sites(source, rel)
        assert not bad, (
            "a Noah-MP leaf batch now sizes a device allocation from "
            "something that is not its call batch, so the COLUMN_BATCH bound "
            "recorded in _PHYSICS_ALLOCATION_INVENTORY no longer holds.  If "
            "the shape really is batch-sized and only the stride constant is "
            "new, add its name to _PER_COLUMN_STRIDE_NAMES and say so:\n"
            + "\n".join(bad))
    for rel in _NOAHMP_SLAB_BATCHES:
        source = (ROOT / rel).read_text(encoding="utf-8")
        bad = _grid_sized_sites(source, rel, strict=False)
        assert not bad, (
            "a Noah-MP slab packer sizes a device allocation from a grid "
            "dimension, or from nothing that names its batch width:\n"
            + "\n".join(bad))

    # -- and the bound itself ----------------------------------------------
    runtime_source = (ROOT / "gpuwm/core/noahmp_runtime.py").read_text(
        encoding="utf-8")
    assert "if len(staged) >= COLUMN_BATCH:" in runtime_source, (
        "the staging loop no longer flushes at COLUMN_BATCH, so nothing "
        "bounds len(calls) and the four Noah-MP rows above become a hazard "
        "rather than a cost")
    batch = None
    for node in ast.walk(ast.parse(runtime_source)):
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "COLUMN_BATCH"
                        for t in node.targets)
                and isinstance(node.value, ast.Constant)):
            batch = node.value.value
    assert isinstance(batch, int) and batch > 0, (
        "COLUMN_BATCH is no longer a module-level integer constant: "
        f"{batch!r}")
    # 620 B per staged column: 52 + 76 + 296 + 196, derived above.
    assert batch * 620 < 4 * 1024 ** 2, (
        f"COLUMN_BATCH={batch} now prices the Noah-MP leaf transients at "
        f"{batch * 620 / 1024 ** 2:.2f} MiB, above the 4 MiB this row was "
        "entered under; re-derive the per-column figure and say what it costs")
