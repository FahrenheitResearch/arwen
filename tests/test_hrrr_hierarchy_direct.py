"""Fail-closed contracts for the public HRRR hierarchy orchestrator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path

import pytest

from gpuwm import hrrr_hierarchy_direct
from gpuwm.hrrr_forecast import hrrr_forcing_end_hour, hrrr_source_window
from gpuwm.hrrr_hierarchy_direct import (
    _atomic_staging_sibling,
    _compare_stock_experiment,
    _expected_root_cache_identity,
    _require_raw_stock_delta,
    _require_raw_wps_contract,
    _supported_hierarchy_slice,
    _validated_root_preparation_binding,
    verified_root_forcing_inventory,
)
from gpuwm.ingest.hrrr_target import HrrrTargetDomain
from gpuwm.native_wrf_contract import CERTIFIED_ETA_LEVELS


def test_atomic_staging_sibling_keeps_deep_windows_publication_short(tmp_path):
    """The d06 transaction must not repeat a long public output basename."""

    parent = tmp_path
    while len(str(parent)) < 125:
        parent /= "deep-path-budget-segment"
    parent.mkdir(parents=True)
    output = parent / ("hrrr-six-domain-z80-" + "x" * 72)

    staging = _atomic_staging_sibling(output)

    assert staging.parent == output.parent
    assert staging.name.startswith(".d-")
    assert len(staging.name) == len(".d-") + 10
    assert output.name not in staging.name
    # Reproduce the nested hierarchy/domain/prepared-cache publication depth
    # that raised WinError 206 in the genuine six-domain run.
    payload = (
        staging / ".d-0123456789" / "domains" / ".d-0123456789"
        / ".p-012345abcd" / "header.json"
    )
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"path-budget-pass")
    assert payload.read_bytes() == b"path-budget-pass"


@dataclass(frozen=True)
class _Run:
    grid_id: int = 1
    mp_physics: int = 6
    bl_pbl_physics: int = 1
    sf_sfclay_physics: int = 91
    sf_surface_physics: int = 2
    ra_physics: int = 0
    ra_lw_physics: int = 0
    ra_sw_physics: int = 1
    cu_physics: int = 0
    radt: float = 1.0
    radt_minutes: float = 1.0
    cudt_minutes: float = 0.0
    nx: int = 199
    ny: int = 199
    nz: int = 49
    dx: float = 2999.4213047435587
    dy: float = 2999.4213047435587
    dt: float = 15.0
    specified: bool = True
    nested: bool = False
    spec_zone: int = 1
    relax_zone: int = 4
    moist: bool = True
    moist_cq: bool = False
    nest_microphysics_transition: str = "same-scheme-only"
    output_interval_s: float = 3600.0
    restart_interval_s: float = 0.0
    nwp_diagnostics: int = 0
    inflow_perturbation: bool = False
    inflow_perturbation_seed: int = 0
    inflow_perturbation_amplitude_scale: float = 1.0
    inflow_perturbation_faces: str = "inflow"


@dataclass(frozen=True)
class _Domain:
    grid_id: int
    run: _Run = _Run()
    parent_id: int = 0
    i_parent_start: int = 1
    j_parent_start: int = 1
    parent_grid_ratio: int = 1
    parent_time_step_ratio: int = 1
    history_interval_s: float = 300.0


@dataclass(frozen=True)
class _Projection:
    map_proj: str = "lambert"
    ref_lat: float = 35.5028506728143
    ref_lon: float = -98.0021669285660
    truelat1: float = 38.5
    truelat2: float = 38.5
    stand_lon: float = -97.5


@dataclass(frozen=True)
class _Vertical:
    eta_levels: tuple[float, ...] = CERTIFIED_ETA_LEVELS
    p_top: float = 10_000.0
    hybrid_opt: int = 2
    etac: float = 0.2


@dataclass(frozen=True)
class _Experiment:
    name: str
    domains: tuple[_Domain, ...]
    feedback: int = 0
    smooth_option: int = 0
    run_seconds: float = 43_200.0
    projection: _Projection = _Projection()
    vertical: _Vertical = _Vertical()
    spec_bdy_width: int = 5


def _native() -> _Experiment:
    child_run = replace(
        _Run(), grid_id=2, nx=300, ny=300, dx=999.8071015811862,
        dy=999.8071015811862, dt=5.0, specified=False, nested=True,
    )
    return _Experiment(
        name="native",
        domains=(
            _Domain(1),
            _Domain(
                2, child_run, parent_id=1, i_parent_start=50,
                j_parent_start=50, parent_grid_ratio=3,
                parent_time_step_ratio=3),
        ),
    )


def _target(**changes) -> HrrrTargetDomain:
    values = {
        "name": "hrrr_test",
        "map_proj": "lambert",
        "nx": 199,
        "ny": 199,
        "nz": 49,
        "dx_m": 2999.4213047435587,
        "dy_m": 2999.4213047435587,
        "ref_lat": 35.5028506728143,
        "ref_lon": -98.0021669285660,
        "truelat1": 38.5,
        "truelat2": 38.5,
        "stand_lon": -97.5,
        "time_step_seconds": 15,
        "spec_bdy_width": 5,
        "spec_zone": 1,
        "relax_zone": 4,
    }
    values.update(changes)
    return HrrrTargetDomain(**values)


def _raw_runtime_namelist(
        max_dom, *, longwave, theta_m, ghg_input=None, run_hours=12,
        do_radar_ref=None):
    """The certified raw runtime shape.  ``ghg_input`` and
    ``do_radar_ref`` are the two STOCK-ONLY keys: passing either marks
    this text as the stock half of the pair, and the native half must
    omit both."""

    def repeated(value):
        return ", ".join(str(value) for _ in range(max_dom))
    ghg = "" if ghg_input is None else f" ghg_input = {ghg_input},\n"
    radar = ("" if do_radar_ref is None
             else f" do_radar_ref = {do_radar_ref},\n")
    return (
        "&time_control\n"
        f" run_hours = {run_hours},\n"
        " interval_seconds = 3600,\n"
        f" frames_per_outfile = {repeated(1)},\n"
        " restart = .false.,\n"
        " io_form_history = 2,\n io_form_restart = 2,\n"
        " io_form_input = 2,\n io_form_boundary = 2,\n/\n"
        "&domains\n"
        f" max_dom = {max_dom},\n"
        " num_metgrid_levels = 51,\n"
        " num_metgrid_soil_levels = 9,\n"
        " sfcp_to_sfcp = .true.,\n/\n"
        "&physics\n"
        f" ra_lw_physics = {repeated(longwave)},\n"
        f" ra_sw_physics = {repeated(1)},\n"
        " isfflx = 1,\n ifsnow = 1,\n icloud = 1,\n"
        " surface_input_source = 1,\n num_soil_layers = 4,\n"
        f" sf_urban_physics = {repeated(0)},\n"
        " sst_update = 0,\n"
        f"{ghg}{radar}/\n"
        f"&dynamics\n use_theta_m = {theta_m},\n/\n"
    )


def _sealed_forcing(exp) -> tuple[int, ...]:
    """The forcing inventory a root prepared for EXP would carry.

    The duration ceiling is a property of the sealed root preparation, so
    every gate call needs one.  Tests about geometry and physics say "the
    root that was prepared for this experiment" with this; the tests that
    are ABOUT the ceiling pass their own inventory explicitly.  Derived
    from the preparer's own endpoint arithmetic so this helper seals what
    ``tools.prepare_hrrr_wrf`` actually seals, sub-hour runs included.
    """

    return tuple(range(hrrr_forcing_end_hour(exp.run_seconds) + 1))


def _slice(exp, target) -> None:
    _supported_hierarchy_slice(
        exp, target, forcing_hours=_sealed_forcing(exp))


def test_public_gate_accepts_generic_parent_ordered_easy_physics_slice():
    native = _native()
    target = _target()
    _slice(native, target)
    _slice(replace(native, domains=(_Domain(1),)), target)

    child3 = replace(
        native.domains[1], grid_id=3, parent_id=2,
        i_parent_start=100, j_parent_start=100,
        run=replace(
            native.domains[1].run, grid_id=3, nx=240, ny=240,
            dx=333.2690338603954, dy=333.2690338603954,
            dt=5.0 / 3.0))
    sibling4 = replace(
        native.domains[1], grid_id=4, parent_id=1,
        i_parent_start=70, j_parent_start=60,
        run=replace(
            native.domains[1].run, grid_id=4, nx=180, ny=180))
    generic = replace(
        native, domains=(*native.domains, child3, sibling4))
    _slice(generic, target)

    with pytest.raises(ValueError, match="parent-before-child"):
        _slice(replace(
            generic, domains=(generic.domains[0], generic.domains[2],
                              generic.domains[1], generic.domains[3])), target)
    with pytest.raises(ValueError, match="one-way"):
        _slice(replace(_native(), feedback=1), target)
    # A mixed WSM6 -> Morrison edge that names no transition policy.
    #
    # This used to refuse as "unsupported mixed": before v1.3.1 the only
    # admitted mixed edge was Thompson -> NSSL.  The nest-edge lane
    # admits all 20 ordered mixed edges now, and the refusal moved from
    # "the pair is unsupported" to "the pair needs its policy named" --
    # which is the stronger statement, because a silent default is what
    # a cross-scheme translation must never have.  The gate is not
    # weakened: the same hierarchy still fails closed.
    drifted = replace(
        _native(), domains=(
            _native().domains[0],
            replace(_native().domains[1], run=replace(
                _native().domains[1].run, mp_physics=10))))
    with pytest.raises(ValueError,
                       match="requires explicit nest_microphysics_transition"):
        _slice(drifted, target)

    # ...and naming it, with the moist/CQ contract the transition also
    # requires on both domains, admits the same tree.  That is the other
    # half of the contract, and it is what makes the refusal above a
    # named gate rather than a blanket one.
    admitted = replace(
        _native(), domains=(
            replace(_native().domains[0], run=replace(
                _native().domains[0].run, moist=True, moist_cq=True)),
            replace(_native().domains[1], run=replace(
                _native().domains[1].run, mp_physics=10,
                moist=True, moist_cq=True,
                nest_microphysics_transition="mp-edge-mass-diagnosed-v1"))))
    _slice(admitted, target)


def test_public_gate_admits_explicit_thompson_to_nssl_tree_without_consent():
    base = _native()
    root_run = replace(base.domains[0].run, mp_physics=8, moist_cq=True)
    d02_run = replace(
        base.domains[1].run, mp_physics=8, moist_cq=True)
    d03_run = replace(
        d02_run, grid_id=3, nx=180, ny=180,
        dx=d02_run.dx / 3.0, dy=d02_run.dy / 3.0,
        dt=d02_run.dt / 3.0, mp_physics=18,
        nest_microphysics_transition=(
            "mp8-to-mp18-mass-diagnosed-v1"))
    d04_run = replace(
        d03_run, grid_id=4, nx=120, ny=120,
        dx=d03_run.dx / 2.0, dy=d03_run.dy / 2.0,
        dt=d03_run.dt / 2.0,
        nest_microphysics_transition="same-scheme-only")
    mixed = replace(base, domains=(
        replace(base.domains[0], run=root_run),
        replace(base.domains[1], run=d02_run),
        _Domain(
            3, d03_run, parent_id=2, i_parent_start=20,
            j_parent_start=20, parent_grid_ratio=3,
            parent_time_step_ratio=3),
        _Domain(
            4, d04_run, parent_id=3, i_parent_start=20,
            j_parent_start=20, parent_grid_ratio=2,
            parent_time_step_ratio=2),
    ))

    _slice(mixed, _target())

    missing_policy = replace(
        mixed, domains=(*mixed.domains[:2], replace(
            mixed.domains[2], run=replace(
                mixed.domains[2].run,
                nest_microphysics_transition="same-scheme-only")),
            mixed.domains[3]))
    with pytest.raises(ValueError, match="requires explicit"):
        _slice(missing_policy, _target())


def test_public_gate_admits_a_same_scheme_p3_hierarchy():
    """mp=50 rides the same plan leg every other admitted scheme rides.

    The scheme's former exclusion from _SUPPORTED_MICROPHYSICS is
    retired (see the constant's comment); this is the reachability proof
    for the pure-P3 tree -- the shape a stranger's public config takes.
    """
    base = _native()
    pure = replace(base, domains=(
        replace(base.domains[0], run=replace(
            base.domains[0].run, mp_physics=50)),
        replace(base.domains[1], run=replace(
            base.domains[1].run, mp_physics=50)),
    ))
    _slice(pure, _target())


def test_public_gate_admits_mixed_p3_edges_under_the_named_policy():
    """Both directions of the ratified rime-pair closure, on this route.

    A Thompson root carrying a P3 child (entry closure: qi/qs/qg merge,
    rime pair diagnosed) and a P3 root carrying a Thompson child (exit
    closure: split by rime state), each admitted only when the edge
    policy is NAMED -- the same explicit-policy contract every other
    mixed edge on this route carries.
    """
    base = _native()
    for root_mp, child_mp in ((8, 50), (50, 8)):
        root_run = replace(
            base.domains[0].run, mp_physics=root_mp,
            moist=True, moist_cq=True)
        child_run = replace(
            base.domains[1].run, mp_physics=child_mp,
            moist=True, moist_cq=True,
            nest_microphysics_transition="mp-edge-mass-diagnosed-v1")
        mixed = replace(base, domains=(
            replace(base.domains[0], run=root_run),
            replace(base.domains[1], run=child_run)))
        _slice(mixed, _target())

        unnamed = replace(mixed, domains=(mixed.domains[0], replace(
            mixed.domains[1], run=replace(
                mixed.domains[1].run,
                nest_microphysics_transition="same-scheme-only"))))
        with pytest.raises(ValueError, match="requires explicit"):
            _slice(unnamed, _target())


def test_public_gate_accepts_short_ohio_z80_sealed_root():
    target = _target(
        name="hrrr_ohio_z80", nx=192, ny=160, nz=80,
        dx_m=3000.0, dy_m=3000.0, ref_lat=40.0, ref_lon=-83.0,
        truelat1=30.0, truelat2=60.0, stand_lon=-84.0)
    root_run = replace(
        _Run(), nx=192, ny=160, nz=80, dx=3000.0, dy=3000.0)
    child_run = replace(
        root_run, grid_id=2, nx=90, ny=90, dx=1000.0, dy=1000.0,
        dt=5.0, specified=False, nested=True)
    vertical = _Vertical(
        eta_levels=tuple(1.0 - i / 80.0 for i in range(81)))
    short = _Experiment(
        name="ohio-z80", run_seconds=3600.0,
        projection=_Projection(
            ref_lat=40.0, ref_lon=-83.0, truelat1=30.0,
            truelat2=60.0, stand_lon=-84.0),
        vertical=vertical,
        domains=(
            _Domain(1, root_run),
            _Domain(
                2, child_run, parent_id=1, i_parent_start=30,
                j_parent_start=30, parent_grid_ratio=3,
                parent_time_step_ratio=3),
        ),
    )
    _slice(short, target)


def test_public_gate_runs_the_whole_sealed_forcing_window_not_twelve_hours():
    """A retained f00..f24 hierarchy, and the ceiling that replaced 43200.

    The gate carried ``run_seconds <= 43_200`` with the sentence "the
    native f00..f12 forcing horizon", while the root wrapper's
    ``hrrr_source_window``, the fetch stage and the forcing-hour equality
    below it all handle f00..f24.  A user who fetched and prepared a
    24 h root was refused at the last gate before publication, by a
    constant describing neither the data nor the code.  The ceiling is the
    sealed preparation's own inventory now, so this asserts BOTH ends: a
    24 h window runs 24 h, and a 12 h window still refuses 15 h.
    """

    target = _target()
    long_run = replace(_native(), run_seconds=86_400.0)

    _supported_hierarchy_slice(
        long_run, target, forcing_hours=tuple(range(25)))
    # The intermediate case the field report used: > 12 h, < 24 h.
    _supported_hierarchy_slice(
        replace(long_run, run_seconds=54_000.0), target,
        forcing_hours=tuple(range(16)))

    with pytest.raises(ValueError) as refusal:
        _supported_hierarchy_slice(
            replace(long_run, run_seconds=54_000.0), target,
            forcing_hours=tuple(range(13)))
    message = str(refusal.value)
    assert "12 h forcing horizon" in message
    assert "f00..f12" in message
    assert "15 h" in message
    # A zero-length or negative request is still refused, and an
    # inventory that is not a contiguous run of hourly leads from zero is
    # not an authority at all.
    with pytest.raises(ValueError, match="must be positive"):
        _supported_hierarchy_slice(
            replace(long_run, run_seconds=0.0), target,
            forcing_hours=tuple(range(25)))
    with pytest.raises(ValueError, match="contiguous hourly leads"):
        _supported_hierarchy_slice(
            long_run, target, forcing_hours=(0, 1, 3))
    with pytest.raises(ValueError, match="contiguous hourly leads"):
        _supported_hierarchy_slice(long_run, target, forcing_hours=())


def test_sub_hour_run_expects_the_series_the_preparer_actually_seals():
    """The 900 s field failure: a floor endpoint against a ceiling seal.

    With ``run_seconds = 900`` tools.prepare_hrrr_wrf sizes its window
    with ``hrrr_source_window`` -- a ceiling, because the endpoint at
    0.25 h lies BETWEEN forcing hours and boundary temporal
    interpolation needs the bracketing frame above it -- and seals model
    forcing hours (0, 1).  This stage recomputed the endpoint with a
    floor, expected ``(0,)``, and refused every sub-hour root the
    preparer can build ("expected (0,), got (0, 1)"); ``(0,)`` is also a
    series the tree runner downstream rejects outright, because a single
    frame brackets nothing.  Both stages derive the endpoint from
    :func:`gpuwm.hrrr_forecast.hrrr_forcing_end_hour` now.
    """

    # What the preparer actually seals for a 900 s run at lead 0.
    window = hrrr_source_window(
        cycle=datetime(2026, 7, 28, 18), start_hour=0, run_seconds=900.0)
    sealed = tuple(range(len(window)))
    assert sealed == (0, 1)

    # The direct hierarchy accepts exactly that series for the same run,
    # and its duration gate admits the run against it.
    assert verified_root_forcing_inventory(
        sealed, run_seconds=900.0) == sealed
    _supported_hierarchy_slice(
        replace(_native(), run_seconds=900.0), _target(),
        forcing_hours=sealed)

    # Whole-hour runs are unchanged: floor and ceiling agree there.
    assert verified_root_forcing_inventory(
        (0, 1), run_seconds=3600.0) == (0, 1)
    assert verified_root_forcing_inventory(
        tuple(range(13)), run_seconds=43_200.0) == tuple(range(13))

    # Genuinely wrong inventories still refuse with the same sentence:
    # truncated, over-long, and gapped.
    with pytest.raises(ValueError, match="consecutive hourly HRRR forcing"):
        verified_root_forcing_inventory((0,), run_seconds=900.0)
    with pytest.raises(ValueError, match="consecutive hourly HRRR forcing"):
        verified_root_forcing_inventory((0, 1, 2), run_seconds=3600.0)
    with pytest.raises(ValueError, match="consecutive hourly HRRR forcing"):
        verified_root_forcing_inventory((0, 2), run_seconds=7200.0)
    # And in the orchestration a truncated root never even reaches that
    # check: the sealed-horizon gate refuses it first.
    with pytest.raises(ValueError, match="forcing horizon"):
        _supported_hierarchy_slice(
            replace(_native(), run_seconds=900.0), _target(),
            forcing_hours=(0,))


def test_public_gate_accepts_a_per_domain_history_cadence():
    """d01 hourly beside d02 every 15 minutes -- the ladder's whole point.

    Every other stage already supported it: the experiment loader gives
    each domain its own ``history_interval_s`` and divisibility check, the
    clock registers a per-domain history alarm, and the tree runner writes
    per-domain wrfouts on it.  Only this drift check disagreed, and it did
    so after the expensive preparation, with "output_interval_s (900.0,
    3600.0)".  Preparation writes no history frame at all.
    """

    native = _native()
    laddered = replace(native, domains=(
        native.domains[0],
        replace(native.domains[1], history_interval_s=900.0,
                run=replace(native.domains[1].run,
                            output_interval_s=900.0)),
    ))

    _slice(laddered, _target())

    # A control that must still refuse: a genuinely trajectory-bearing
    # per-domain difference is not an output cadence.
    drifted = replace(laddered, domains=(
        laddered.domains[0],
        replace(laddered.domains[1], run=replace(
            laddered.domains[1].run, sf_surface_physics=3)),
    ))
    with pytest.raises(ValueError, match="certified native"):
        _slice(drifted, _target())


def test_public_gate_accepts_a_child_only_inflow_perturbation():
    """The P3 inflow keys on a child, refused after expensive preparation.

    Same finding as the history cadence above, for the third time: the
    generator acts at runtime FORCE on the child-owned rolling NEST
    boundary tables, and preparation never computes those.  The ruling
    that these fields are preparation-inert is already committed as
    ``gpuwm.ingest.prepared_cache.PREPARATION_INERT_RUN_FIELDS``; this
    gate was the second preparation-side comparison and had not been
    given it, so an LES tree with the generator ON died here holding a
    prepared root that was in fact exactly the right prepared root.
    """

    native = _native()
    seeded = replace(native, domains=(
        native.domains[0],
        replace(native.domains[1], run=replace(
            native.domains[1].run,
            inflow_perturbation=True,
            inflow_perturbation_seed=20160524,
            inflow_perturbation_amplitude_scale=1.0,
            inflow_perturbation_faces="inflow")),
    ))

    _slice(seeded, _target())

    # The control: a field preparation DOES read still refuses, so this
    # is an admission of four named keys and not a hole in the check.
    drifted = replace(seeded, domains=(
        seeded.domains[0],
        replace(seeded.domains[1], run=replace(
            seeded.domains[1].run, sf_sfclay_physics=1)),
    ))
    with pytest.raises(ValueError, match="certified native"):
        _slice(drifted, _target())


def test_root_binding_ignores_write_cadence_and_inert_diagnostics():
    """The three switches that made root and hierarchy irreconcilable.

    A sealed root is built from a shipped physics profile, which writes
    ``restart_interval_s = 0`` and no diagnostics; a hierarchy is imported
    from the user's namelist, which may carry ``restart_interval = 60``
    and whose cumulus-off domains inherit RunConfig's live ``cudt_minutes
    = 5.0``.  None of the three changes one model step -- gpuwm/io/
    restart.py already lets all three differ across a checkpoint boundary
    -- so none of them may decide whether a prepared root can be reused.
    """

    native = _native()
    sealed = asdict(native.domains[0])
    sealed["run"]["restart_interval_s"] = 0.0
    sealed["run"]["nwp_diagnostics"] = 0
    sealed["run"]["cudt_minutes"] = 5.0
    identity = {"domain_config": sealed, "namelist_sha256": "b" * 64}

    live = replace(native.domains[0], run=replace(
        native.domains[0].run, restart_interval_s=3600.0,
        nwp_diagnostics=1, cudt_minutes=0.0))
    digest, prepared = _validated_root_preparation_binding(identity, live)

    assert digest == "b" * 64
    # The document RETURNED is still the sealed one, byte for byte: the
    # normalization decides the comparison, never what gets re-hashed.
    assert prepared == sealed

    # And a real trajectory difference in the same document still refuses.
    drifted = replace(live, run=replace(live.run, sf_surface_physics=3))
    with pytest.raises(ValueError, match="run.sf_surface_physics"):
        _validated_root_preparation_binding(identity, drifted)


def test_root_cache_reuse_binds_effective_d01_not_child_topology():
    native = _native()
    identity = {
        "domain_config": asdict(native.domains[0]),
        "namelist_sha256": "a" * 64,
    }
    digest, prepared = _validated_root_preparation_binding(
        identity, native.domains[0])
    assert digest == "a" * 64
    assert prepared == identity["domain_config"]

    deeper = replace(native, domains=(*native.domains, replace(
        native.domains[1], grid_id=3, parent_id=2,
        run=replace(native.domains[1].run, grid_id=3))))
    assert _validated_root_preparation_binding(
        identity, deeper.domains[0])[0] == "a" * 64

    drifted_root = replace(
        native.domains[0], run=replace(native.domains[0].run, mp_physics=10))
    with pytest.raises(ValueError, match="d01 trajectory controls differ"):
        _validated_root_preparation_binding(identity, drifted_root)
    with pytest.raises(ValueError, match="invalid namelist_sha256"):
        _validated_root_preparation_binding(
            {**identity, "namelist_sha256": "editable"}, native.domains[0])


def test_hierarchy_expected_identity_preserves_validated_namelist_invariant():
    native = _native()
    identity = {
        "domain_config": asdict(native.domains[0]),
        "namelist_sha256": "a" * 64,
        "source_identity": {"source": "fixture"},
        "namelist_extension_invariant": {
            "schema": "gpuwm-namelist-extension-invariant-v1",
            "sha256": "e" * 64,
        },
    }
    expected = _expected_root_cache_identity(
        identity, root_domain=native.domains[0],
        bridge_manifest_sha256="b" * 64,
        source_manifest_sha256="c" * 64,
        static_cache_sha256="d" * 64,
        forcing_hours=(0, 1))
    assert expected["namelist_extension_invariant"] == identity[
        "namelist_extension_invariant"]

    identity["namelist_extension_invariant"]["unchecked"] = True
    with pytest.raises(ValueError, match="valid extension invariant"):
        _expected_root_cache_identity(
            identity, root_domain=native.domains[0],
            bridge_manifest_sha256="b" * 64,
            source_manifest_sha256="c" * 64,
            static_cache_sha256="d" * 64,
            forcing_hours=(0, 1))


def test_sealed_legacy_variant_reaches_the_hierarchy_import(tmp_path):
    """The 4/4 IMPLEMENTATION resolves from the sealed root, end to end.

    A WRF namelist spells radiation as selector integers, so the battery's
    legacy-RRTMG profile cannot survive a namelist round trip on its own:
    the importer's default maps 4/4 to the RTE+RRTMGP substitution.  The
    route therefore reads the sealed root preparation's recorded
    ``ra_rrtmg_variant`` and imports under it.  Three legs, one run:
    the profile pins the variant, the unplumbed default is REFUSED by the
    d01 binding (the seam, kept as the negative control), and the plumbed
    import resolves the legacy engine and binds.
    """

    from gpuwm.experiment import load_experiment
    from gpuwm.hrrr_hierarchy_direct import (_native_experiment,
                                             _sealed_root_rrtmg_variant)
    from gpuwm.ingest.prepared_cache import prepared_domain_config_identity
    from gpuwm.physics_compat import (RRTMG_VARIANT_LEGACY,
                                      THOMPSON_LEGACY_RRTMG_PROFILE_ID,
                                      WRF_RRTMG_LEGACY,
                                      single_domain_runtime_switches)
    from tools import battery_wrf_node_plan as node_plan

    # Leg 1: the profile the root preparer materializes pins the variant.
    switches = single_domain_runtime_switches(
        THOMPSON_LEGACY_RRTMG_PROFILE_ID)
    assert switches["ra_rrtmg_variant"] == RRTMG_VARIANT_LEGACY
    assert switches["wrf_rrtmg_compatibility"] == WRF_RRTMG_LEGACY

    repo = Path(__file__).resolve().parents[1]
    config = (repo / "configs" / "battery"
              / "shape_3km_thompson_rrtmg_legacy.toml")
    outdir = tmp_path / "node"
    node_plan.build(config, outdir, ranks=24, repository_root=repo)
    exp = load_experiment(config)
    sealed = {
        "domain_config": prepared_domain_config_identity(exp.domains[0]),
        "namelist_sha256": "a" * 64,
    }
    assert _sealed_root_rrtmg_variant(sealed) == RRTMG_VARIANT_LEGACY

    # Leg 2 (the seam): the importer default resolves the substitution
    # engine, and the d01 binding refuses the mismatch by name.
    default_exp, _resolved, _report = _native_experiment(
        outdir / "namelist.wps", outdir / "namelist.native.input")
    assert default_exp.domains[0].run.ra_rrtmg_variant == "rte-rrtmgp"
    with pytest.raises(ValueError, match="ra_rrtmg_variant"):
        _validated_root_preparation_binding(sealed, default_exp.domains[0])

    # Leg 3 (the plumb): importing under the sealed variant resolves the
    # legacy engine and its compatibility token, and the binding accepts.
    plumbed_exp, _resolved, _report = _native_experiment(
        outdir / "namelist.wps", outdir / "namelist.native.input",
        rrtmg_variant=RRTMG_VARIANT_LEGACY)
    run = plumbed_exp.domains[0].run
    assert run.ra_rrtmg_variant == RRTMG_VARIANT_LEGACY
    assert run.wrf_rrtmg_compatibility == WRF_RRTMG_LEGACY
    _validated_root_preparation_binding(sealed, plumbed_exp.domains[0])

    # Headers written before the field existed fail open to the importer
    # default; the binding above is what still refuses real mismatches.
    assert _sealed_root_rrtmg_variant({"namelist_sha256": "a" * 64}) is None
    assert _sealed_root_rrtmg_variant(
        {"domain_config": {"run": []}}) is None


def test_coupled_legacy_import_is_admitted_by_the_radiation_slice(tmp_path):
    """The B-04 structural refusal, closed at the canonicalization point.

    The WRF namelist importer emits a coupled 4/4 request as the
    historical aggregate spelling -- ra_physics=4 with the split fields
    at their -1 defaults -- so comparing RAW fields made the slice's
    (4, 4) admission unreachable for every namelist-imported experiment:
    the check demanded ra_physics=0 AND an explicit pair at once, and its
    error text advertised a case no import could satisfy.  The slice now
    compares the RESOLVED pair through gpuwm.config.radiation_scheme_ids,
    the production resolver; emission is untouched (sealed bytes stay
    sealed).
    """

    from gpuwm.experiment import load_experiment
    from gpuwm.hrrr_hierarchy_direct import _native_experiment
    from gpuwm.hrrr_route_inputs import target_domain
    from tools import battery_wrf_node_plan as node_plan

    repo = Path(__file__).resolve().parents[1]
    config = (repo / "configs" / "battery"
              / "shape_3km_thompson_rrtmg_legacy.toml")
    outdir = tmp_path / "node"
    node_plan.build(config, outdir, ranks=24, repository_root=repo)
    exp, _resolved, _report = _native_experiment(
        outdir / "namelist.wps", outdir / "namelist.native.input",
        rrtmg_variant="rrtmg_legacy")
    run = exp.domains[0].run
    # The aggregate spelling, exactly as imported.
    assert (run.ra_physics, run.ra_lw_physics, run.ra_sw_physics) \
        == (4, -1, -1)
    target = target_domain(load_experiment(config))

    _supported_hierarchy_slice(exp, target, forcing_hours=tuple(range(25)))

    # CONTROL 1: the aggregate spelling of radiation OFF resolves to
    # (0, 0), which the route does not admit -- the canonicalization is
    # not a wave-through.
    radiation_off = replace(exp, domains=tuple(
        replace(domain, run=replace(domain.run, ra_physics=0))
        for domain in exp.domains))
    with pytest.raises(ValueError, match=r"resolved \(ra_lw_physics"):
        _supported_hierarchy_slice(radiation_off, target,
                                   forcing_hours=tuple(range(25)))

    # CONTROL 2: an incoherent spelling (explicit pair beside a nonzero
    # aggregate) is refused by the resolver itself, by name.
    incoherent = replace(exp, domains=tuple(
        replace(domain, run=replace(domain.run, ra_lw_physics=4,
                                    ra_sw_physics=4))
        for domain in exp.domains))
    with pytest.raises(ValueError, match="require ra_physics=0"):
        _supported_hierarchy_slice(incoherent, target,
                                   forcing_hours=tuple(range(25)))


def test_stock_comparison_allows_only_explicit_longwave_runtime_change():
    native = _native()
    stock = replace(native, name="stock", domains=tuple(
        replace(domain, run=replace(domain.run, ra_lw_physics=1))
        for domain in native.domains))
    _compare_stock_experiment(native, stock)

    physics_drift = replace(stock, domains=(
        stock.domains[0],
        replace(stock.domains[1], run=replace(
            stock.domains[1].run, sf_surface_physics=0)),
    ))
    with pytest.raises(ValueError, match="beyond the allowed"):
        _compare_stock_experiment(native, physics_drift)

    with pytest.raises(ValueError, match="must select ra_lw_physics=1"):
        _compare_stock_experiment(native, replace(stock, domains=(
            stock.domains[0], native.domains[1])))


def test_stock_comparison_rejects_forecast_duration_drift():
    native = _native()
    stock = replace(native, name="stock", run_seconds=15.0, domains=tuple(
        replace(domain, run=replace(domain.run, ra_lw_physics=1))
        for domain in native.domains))
    with pytest.raises(ValueError, match="beyond the allowed"):
        _compare_stock_experiment(native, stock)


@pytest.mark.parametrize("max_dom", range(1, 22))
def test_raw_namelist_gate_allows_only_explicit_runtime_deltas(
        tmp_path, max_dom):
    native = tmp_path / "native.input"
    stock = tmp_path / "stock.input"
    native.write_text(
        _raw_runtime_namelist(max_dom, longwave=0, theta_m=0),
        encoding="ascii")
    stock.write_text(
        _raw_runtime_namelist(
            max_dom, longwave=1, theta_m=1, ghg_input=0, do_radar_ref=1),
        encoding="ascii")
    receipt = _require_raw_stock_delta(native, stock)
    assert set(receipt["allowed_deltas"]) == {
        "physics.ra_lw_physics", "dynamics.use_theta_m",
        "physics.ghg_input", "physics.do_radar_ref"}
    assert receipt["max_dom"] == max_dom

    stock.write_text(
        _raw_runtime_namelist(
            max_dom, longwave=1, theta_m=1, ghg_input=0, do_radar_ref=1,
            run_hours=6),
        encoding="ascii")
    with pytest.raises(ValueError, match="time_control/run_hours"):
        _require_raw_stock_delta(native, stock)

    stock.write_text(
        _raw_runtime_namelist(
            max_dom, longwave=1, theta_m=1, ghg_input=1, do_radar_ref=1),
        encoding="ascii")
    with pytest.raises(ValueError, match="stock-only ghg_input=0"):
        _require_raw_stock_delta(native, stock)

    # do_radar_ref is MANDATORY in the stock half, at 1.  Omitted, the
    # stock arm writes history frames with no REFL_10CM and every
    # reflectivity score on it is unanswerable; at 0 the switch is
    # present and says the wrong thing.  Both refuse.
    stock.write_text(
        _raw_runtime_namelist(
            max_dom, longwave=1, theta_m=1, ghg_input=0),
        encoding="ascii")
    with pytest.raises(ValueError, match="do_radar_ref"):
        _require_raw_stock_delta(native, stock)

    stock.write_text(
        _raw_runtime_namelist(
            max_dom, longwave=1, theta_m=1, ghg_input=0, do_radar_ref=0),
        encoding="ascii")
    with pytest.raises(ValueError, match="stock-only do_radar_ref=1"):
        _require_raw_stock_delta(native, stock)

    # And it stays FORBIDDEN in the native half: gpuwm does not read it,
    # so a native namelist claiming to control REFL_10CM is claiming
    # something the arm that reads the file ignores.
    native.write_text(
        _raw_runtime_namelist(
            max_dom, longwave=0, theta_m=0, do_radar_ref=1),
        encoding="ascii")
    stock.write_text(
        _raw_runtime_namelist(
            max_dom, longwave=1, theta_m=1, ghg_input=0, do_radar_ref=1),
        encoding="ascii")
    with pytest.raises(ValueError, match="do_radar_ref must be omitted"):
        _require_raw_stock_delta(native, stock)


@pytest.mark.parametrize(
    "old, new, key",
    [
        ("interval_seconds = 3600", "interval_seconds = 1800",
         "interval_seconds"),
        ("sfcp_to_sfcp = .true.", "sfcp_to_sfcp = .false.",
         "sfcp_to_sfcp"),
        ("io_form_input = 2", "io_form_input = 3", "io_form_input"),
        ("isfflx = 1", "isfflx = 0", "isfflx"),
        ("num_soil_layers = 4", "num_soil_layers = 9",
         "num_soil_layers"),
        ("interval_seconds = 3600", "interval_seconds = 3600.0",
         "interval_seconds"),
        ("restart = .false.", "restart = 0", "restart"),
        ("sfcp_to_sfcp = .true.", "sfcp_to_sfcp = 1",
         "sfcp_to_sfcp"),
        ("io_form_input = 2", "io_form_input = 2.0", "io_form_input"),
        ("isfflx = 1", "isfflx = 1.0", "isfflx"),
    ],
)
def test_raw_namelist_gate_rejects_dropped_runtime_drift(
        tmp_path, old, new, key):
    native = tmp_path / "native.input"
    stock = tmp_path / "stock.input"
    native_text = _raw_runtime_namelist(4, longwave=0, theta_m=0)
    stock_text = _raw_runtime_namelist(
        4, longwave=1, theta_m=1, ghg_input=0, do_radar_ref=1)
    assert old in native_text and old in stock_text
    native.write_text(native_text.replace(old, new), encoding="ascii")
    stock.write_text(stock_text.replace(old, new), encoding="ascii")
    with pytest.raises(ValueError, match=key):
        _require_raw_stock_delta(native, stock)


def test_the_route_asks_for_an_optional_stock_wrf_export():
    """A missing ORACLE file must not destroy a prepared hierarchy.

    ``native_hierarchy`` offers required/optional/off and defaults to
    ``required``; ``gfs_direct`` asks for ``optional``.  This route
    passed nothing and inherited ``required``, so it was the only one
    that discarded a complete, verified GPU hierarchy when the
    unchanged-WRF file set could not represent the state -- which a
    single sub-freezing soil node anywhere in the domain is enough to
    cause.  Read from the call site itself because the whole
    orchestration below it needs a sealed root preparation to run; the
    behaviour of each mode is pinned in tests/test_native_hierarchy.py.
    """
    import ast
    import inspect
    from gpuwm import hrrr_hierarchy_direct

    tree = ast.parse(inspect.getsource(hrrr_hierarchy_direct))
    modes = [
        keyword.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "initialize_and_export_native_hierarchy"
        for keyword in node.keywords
        if keyword.arg == "stock_wrf_export"
    ]
    assert modes == ["optional"]


def test_raw_wps_gate_binds_hourly_hierarchy_contract(tmp_path):
    wps = tmp_path / "namelist.wps"
    wps.write_text(
        "&share\n max_dom = 4,\n interval_seconds = 3600,\n/\n",
        encoding="ascii")
    assert _require_raw_wps_contract(wps, 4)["status"] == "PASS"

    wps.write_text(
        "&share\n max_dom = 4,\n interval_seconds = 1800,\n/\n",
        encoding="ascii")
    with pytest.raises(ValueError, match="interval_seconds"):
        _require_raw_wps_contract(wps, 4)

    wps.write_text(
        "&share\n max_dom = 4,\n interval_seconds = 3600.0,\n/\n",
        encoding="ascii")
    with pytest.raises(ValueError, match="interval_seconds"):
        _require_raw_wps_contract(wps, 4)


def test_the_hierarchy_prints_the_digest_its_own_chain_promises(
        tmp_path, monkeypatch, capsys):
    """The one placeholder in the emitted chain no stage used to fill.

    `gpuwm domain --source hrrr` closes a multi-domain emission with
    ``--preparation-receipt-sha256 <printed by the hierarchy>``, and the
    forecast runner checks that digest against ``receipt.json``.  Walking
    that printed route end to end found the hierarchy never printed it,
    so the chain could not be completed without hashing a file by hand.
    """

    output_root = tmp_path / "hierarchy"

    def _fake(*, output_root: Path, **_ignored):
        output_root.mkdir(parents=True)
        (output_root / "receipt.json").write_text(
            '{"status": "PASS"}\n', encoding="utf-8")
        return {
            "status": "PASS",
            "workers": 8,
            "timing_seconds": {"total": 1.0},
            "wrf_manifest": {"files": {}},
        }

    monkeypatch.setattr(
        hrrr_hierarchy_direct, "prepare_hrrr_hierarchy", _fake)
    assert hrrr_hierarchy_direct.main([
        "--root-preparation", str(tmp_path / "root"),
        "--root-domain-spec", str(tmp_path / "d01-target.json"),
        "--wps-namelist", str(tmp_path / "namelist.wps"),
        "--namelist-input", str(tmp_path / "namelist.input"),
        "--stock-wrf-namelist-input", str(tmp_path / "stock.namelist.input"),
        "--geog-root", str(tmp_path / "geog"),
        "--source-manifest", str(tmp_path / "SHA256SUMS"),
        "--source-manifest-sha256", "0" * 64,
        "--valid-time", "2020-01-01_00:00:00",
        "--output-root", str(output_root),
    ]) == 0

    printed = json.loads(capsys.readouterr().out)
    expected = hashlib.sha256(
        (output_root / "receipt.json").read_bytes()).hexdigest()
    assert printed["preparation_receipt_sha256"] == expected
