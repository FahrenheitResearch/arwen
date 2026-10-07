"""The root's specified boundary carries the analysed hydrometeors.

WRF v4.7.1 carries the moist species beyond water vapour on a specified
domain only under have_bcs_moist (dyn_em/solve_em.F:2265-2267, :2346,
:4701-4703), off by default, and real.exe writes water vapour only.  A
flow-dependent boundary has zero inflow, so an analysed cloud or snow
field drained out of the domain: a HRRR-forced two-hour winter case kept
1.0 million t of snow over its inner domain where HRRR's own analysis
held 3.1.  These tests hold the table fact (which sources publish the
masses on every frame), the ring's values (the source frame's
hydrometeors, coupled), the number moments seeded beside them, the
end-of-step force-back, the cache admission and the price.
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.boundary_fields import (
    BOUNDARY_HYDROMETEOR_MASSES, COLD_START_SEEDED_NUMBERS,
    admissible_boundary_inventory, boundary_hydrometeor_fields,
    external_scalar_fields, mapping_boundary_species,
    potential_external_scalar_fields, sealed_boundary_species,
    source_boundary_species)
from gpuwm.config import RunConfig

FIVE = ("qc", "qr", "qi", "qs", "qg")


def _cfg(**changes):
    values = dict(nx=13, ny=12, nz=4, dx=12000., dy=12000., ztop=10000.,
                  dt=1., run_seconds=120., moist=True, mp_physics=8,
                  specified=True)
    return RunConfig(**(values | changes))


# ---------------------------------------------------------------- table fact

@pytest.mark.parametrize("source", ["hrrr", "hrrr-prs", "icon-d2", "gdas"])
def test_rows_that_publish_hydrometeors_declare_all_five(source):
    """GDAS/GFS pgrb2.0p25 writes the five masses on 22 of its 33 isobaric
    levels; its packaged mapping declares them on those levels
    (``published_levels``), so its boundary carries them."""
    assert BOUNDARY_HYDROMETEOR_MASSES == FIVE
    assert source_boundary_species(source) == FIVE


@pytest.mark.parametrize("source", [
    "gfs", "era5", "ecmwf", "icon-eu", "icon", "gem", "rap",
    "rrfs", "aifs", "gefs", "not-a-source", "", None])
def test_rows_that_publish_none_keep_water_vapour_only(source):
    assert source_boundary_species(source) == ()


def test_a_mapping_declares_its_own_boundary_hydrometeors():
    """A user's mapping is the same table: declared fields are published."""
    assert mapping_boundary_species({"fields": {}}) == ()
    assert mapping_boundary_species({"fields": {
        "snow_mixing_ratio": {}, "cloud_water_mixing_ratio": {},
        "air_temperature": {}}}) == ("qc", "qs")


def test_a_sealed_inventory_names_the_masses_its_tables_hold():
    """A prepared cache's own interval rows answer for its boundary.

    Numbers and dynamics are not masses; a row set that names no fields
    is unreadable (``None``), which is not the same answer as a boundary
    that carries none (``()``).  The masses it names price like a
    source's own row.
    """
    carried = [{"fields": ["mu", "nr", "phi", "qc", "qr", "qs", "qv",
                           "theta", "u", "v"]}] * 2
    assert sealed_boundary_species(carried) == ("qc", "qr", "qs")
    vapour = [{"fields": ["mu", "phi", "qv", "theta", "u", "v"]}]
    assert sealed_boundary_species(vapour) == ()
    for unreadable in (None, [], [{}], [{"fields": None}], "qc"):
        assert sealed_boundary_species(unreadable) is None
    assert source_boundary_species(("qs", "qc")) == ("qc", "qs")
    assert source_boundary_species(()) == ()


def test_the_cold_start_closure_and_the_ring_seed_the_same_numbers():
    assert COLD_START_SEEDED_NUMBERS == {
        8: (("qr", "nr"), ("qi", "ni")),
        28: (("qc", "nc"), ("qr", "nr"), ("qi", "ni"))}


@pytest.mark.parametrize("mp, expected", [
    (1, ("qc", "qr")),
    (6, FIVE),
    (8, (*FIVE, "nr", "ni")),
    (10, FIVE),
    (28, (*FIVE, "nc", "nr", "ni")),
    (50, ("qc", "qr", "qi")),
])
def test_the_ring_carries_what_the_scheme_holds_and_the_numbers_it_seeds(
        mp, expected):
    cfg = _cfg(mp_physics=mp, mp28_aerosol_source="synthetic")
    assert boundary_hydrometeor_fields(cfg, FIVE) == expected
    assert external_scalar_fields(cfg, boundary_species=FIVE) == (
        "qv", *expected)
    # A source publishing none, or a nest (its parent supplies every
    # moist species through its own tables), keeps water vapour only.
    assert external_scalar_fields(cfg) == ("qv",)
    assert boundary_hydrometeor_fields(
        replace(cfg, specified=False, nested=True), FIVE) == ()


# ------------------------------------------------------------- ring values

def _frame(mp, *, cloud_water=None):
    from test_real_init import _analyzed_hrrr_real_init

    result, cfg = _analyzed_hrrr_real_init(
        mp, shape=(12, 13), cloud_water=cloud_water, specified=True,
        mp28_aerosol_source="synthetic",
        init_kwargs={"boundary_species": source_boundary_species("hrrr-prs")})
    return result, cfg


def _coupled(state, name):
    """The frame's hydrometeor in WRF's coupled boundary units (mu-only
    coupling, module_bc.F:2081), in the state's own float32 arithmetic."""
    mu = np.asarray(state.total_mu())
    chm = (np.asarray(state.c1h)[:, None, None] * mu[None]
           + np.asarray(state.c2h)[:, None, None])
    return np.asarray(chm * np.asarray(getattr(state, name)),
                      dtype=np.float64)


@pytest.mark.parametrize("mp", [8, 28])
def test_the_ring_carries_the_source_frames_hydrometeors(mp):
    from gpuwm.ingest.lateral_bc import StateBoundaryFrames

    first, cfg = _frame(mp)
    second, _ = _frame(mp, cloud_water=[2e-4, 1e-4, 5e-5, 3e-5, 1e-5, 0.])
    carried = boundary_hydrometeor_fields(cfg, FIVE)
    assert first.state._external_scalar_boundary_fields == ("qv", *carried)
    assert first.hydrometeor_initialization["lateral_boundary_species"] == (
        list(carried))
    frames = StateBoundaryFrames(spec_bdy_width=5, spec_zone=1, relax_zone=4)
    frames.add_state(first.state)
    frames.add_state(second.state)
    interval = frames.build([0.0, 3600.0]).intervals[0]
    assert {*FIVE, *carried} <= set(interval.fields)
    for name in carried:
        before = _coupled(first.state, name)
        after = _coupled(second.state, name)
        side = interval.fields[name]
        np.testing.assert_array_equal(side.west.value, before[:, :, :5])
        np.testing.assert_array_equal(
            side.south.value, before[:, :5, :])
        np.testing.assert_array_equal(
            side.west.tendency, (after[:, :, :5] - before[:, :, :5]) / 3600.0)
    # The analysed masses are really on the ring, not zeros standing in.
    assert np.count_nonzero(interval.fields["qs"].west.value) > 0
    assert np.count_nonzero(interval.fields["nr"].east.value) > 0
    # Mass and number from the SAME frame: every cell carrying rain mass
    # above the scheme's gate carries the number the cold start gave it.
    from gpuwm.core.thompson_entry import R1
    qr = np.asarray(first.state.qr)
    nr = np.asarray(first.state.nr)
    assert (nr[qr > R1] > 0).all()


def test_a_source_that_publishes_none_is_byte_identical():
    from test_real_init import _analyzed_hrrr_real_init
    from gpuwm.ingest.lateral_bc import domain_boundary_snapshot

    plain, _ = _analyzed_hrrr_real_init(8, shape=(12, 13), specified=True)
    named, _ = _analyzed_hrrr_real_init(
        8, shape=(12, 13), specified=True,
        init_kwargs={"boundary_species": ()})
    assert plain.state._external_scalar_boundary_fields == ("qv",)
    assert "lateral_boundary_species" not in plain.hydrometeor_initialization
    first, second = (domain_boundary_snapshot(r.state) for r in (plain, named))
    assert set(first) == set(second) == {"u", "v", "theta", "phi", "mu", "qv"}
    for name in first:
        assert first[name].tobytes() == second[name].tobytes()


def test_an_unknown_boundary_species_is_refused():
    from test_real_init import _analyzed_hrrr_real_init

    with pytest.raises(ValueError, match="boundary_species"):
        _analyzed_hrrr_real_init(8, init_kwargs={"boundary_species": ("qh",)})


# ------------------------------------------------------- end-of-step forcing

def test_spec_bdy_final_forces_the_masses_back_and_leaves_the_numbers():
    """Masses are WRF moist-array species (forced back, solve_em.F:4703);
    the seeded numbers are scalar-array species, forced back only on a
    nest, so a specified ring's number moves by its tendency."""
    from gpuwm.ingest import lateral_bc

    fields = dict.fromkeys(
        ("u", "v", "theta", "phi", "mu", "qv", *FIVE, "nr", "ni"), 0)
    forced = []
    for specified in (True, False):
        cfg = SimpleNamespace(specified=specified, nested=not specified,
                              spec_zone=1)
        state = SimpleNamespace(
            qv=object(), lateral_boundaries=object(),
            _lateral_boundary_device=SimpleNamespace(clock=object()))
        calls = []
        interval = SimpleNamespace(fields=fields)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(lateral_bc, "_active_device_interval",
                       lambda *a: (interval, 0., None, None))
            mp.setattr(lateral_bc, "_launch_mu_boundary_values",
                       lambda *a: None)
            mp.setattr(lateral_bc, "_launch_finalize_field",
                       lambda s, name, *a: calls.append(name))
            lateral_bc.apply_state_boundary_values(state, cfg)
        forced.append(calls)
    assert forced[0] == ["u", "v", "theta", "phi", "qv", *FIVE]
    assert forced[1] == ["u", "v", "theta", "phi", "qv", *FIVE, "nr", "ni"]


# ----------------------------------------------------------- cache admission

@pytest.mark.parametrize("scalars, admitted", [
    (("qv",), True),
    (("qv", *FIVE, "nr", "ni"), True),
    (("ni", "nr", "qc", "qg", "qi", "qr", "qs", "qv"), True),
    (("qv", "qc", "qs"), True),
    (("qv", "qr"), False),                 # rain without its seeded number
    (("qv", *FIVE), False),
    (("qv", *FIVE, "nr", "ni", "nc"), False),  # mp=8 has no droplet number
    (("qv", "nwfa", "nifa"), False),       # aerosol off mp=28
    (("qv", "qh"), False),
    ((*FIVE, "nr", "ni"), False),          # no water vapour
])
def test_a_sealed_inventory_is_admitted_by_its_structure(scalars, admitted):
    assert admissible_boundary_inventory(_cfg(), scalars) is admitted


def test_mp28_admits_the_aerosol_pair_beside_the_hydrometeors():
    cfg = _cfg(mp_physics=28)
    assert admissible_boundary_inventory(
        cfg, ("qv", *FIVE, "nc", "nr", "ni", "nwfa", "nifa"))
    assert not admissible_boundary_inventory(cfg, ("qv", "nwfa"))


def _refusal(published, fields, *, mp=28, source="hrrr", analysed=FIVE):
    from gpuwm import prepared_single_domain_forecast as runner

    cfg = _cfg(mp_physics=mp, mp28_aerosol_source="synthetic")
    metadata = {
        "lbc": {"intervals": [{"fields": fields}] * 2},
        "hydrometeor_initialization": {"initialized_state_species": {
            name: {"nonzero_count": 4 if name in analysed else 0}
            for name in FIVE}}}
    try:
        runner._refuse_stale_boundary(
            cfg, metadata, published=published, source=source,
            prepared_root=None)
    except ValueError as error:
        return str(error)
    return None


DYNAMICS = ["mu", "phi", "qv", "theta", "u", "v"]


def test_a_cache_sealed_before_the_hydrometeor_boundary_is_refused():
    """The warning it ran with is retired: a stale boundary is refused.

    Run, a water-vapour-only boundary drains the analysed cloud and snow
    out of the edges, so the refusal names that and the re-preparation.
    """
    message = _refusal(FIVE, DYNAMICS)
    assert message is not None
    assert "sealed before its lateral boundary carried the hydrometeors "         "hrrr publishes" in message
    assert "qc, qr, qi, qs, qg" in message
    assert "drain out of the domain edges" in message
    assert "gpuwm prep --source hrrr" in message and "--output-root" in message
    # One mass short is still stale, and is named alone.
    held = sorted([*DYNAMICS, "qc", "qr", "qi", "qs", "nc", "nr", "ni"])
    assert "(qg)" in _refusal(FIVE, held)
    # Only what the start state holds can drain: two analysed masses.
    assert "(qc, qs)" in _refusal(FIVE, DYNAMICS, analysed=("qc", "qs"))


def test_a_cache_a_fresh_preparation_would_write_is_admitted():
    """A source that publishes none, a cache that carries what it
    publishes, a scheme that carries none of it, and a start state with
    nothing analysed to drain: no refusal and no warning.  The GFS and
    ERA5 caches whose boundaries did not change are not touched."""
    assert _refusal((), DYNAMICS, source="gfs") is None
    assert _refusal(FIVE, sorted(
        [*DYNAMICS, *FIVE, "nc", "nr", "ni"])) is None
    # Kessler carries qc and qr only; a cache with exactly those is fresh.
    assert _refusal(FIVE, sorted([*DYNAMICS, "qc", "qr"]), mp=1) is None
    assert "(qc, qr)" in _refusal(FIVE, DYNAMICS, mp=1)
    assert _refusal(FIVE, DYNAMICS, analysed=()) is None
    # A cache with no initialization receipt names nothing analysed.
    from gpuwm import prepared_single_domain_forecast as runner
    runner._refuse_stale_boundary(
        _cfg(mp_physics=28, mp28_aerosol_source="synthetic"),
        {"lbc": {"intervals": [{"fields": DYNAMICS}]}},
        published=FIVE, source="hrrr", prepared_root=None)


def _gdas_documents():
    import json

    from gpuwm.source_authorities import (packaged_authorities,
                                          packaged_authority_sha256)

    pins = packaged_authority_sha256("gdas-pgrb2-0p25-grib2-v1")
    current = json.loads(packaged_authorities(
        "gdas-pgrb2-0p25-grib2-v1")["mapping"].read_text(encoding="utf-8"))
    earlier = json.loads(json.dumps(current))
    names = ("cloud_water_mixing_ratio", "rain_water_mixing_ratio",
             "cloud_ice_mixing_ratio", "snow_mixing_ratio",
             "graupel_or_hail_mixing_ratio")
    for name in names:
        del earlier["fields"][name]
        earlier["target"]["initialization_policies"][name] = (
            "explicit_zero_with_adapter_validation")
    earlier["target"]["required_fields"] = [
        row for row in earlier["target"]["required_fields"]
        if row["name"] not in names]
    return pins, current, earlier


def test_a_gdas_preparation_from_before_its_mapping_declared_them_is_named(
        tmp_path):
    """The packaged GDAS profile's door names the stale boundary.

    The bound mapping of an earlier preparation no longer matches the
    pin; the door says why that matters rather than only that the
    digests differ, under the profile's own name and when the stage door
    hands the same root in as a caller's ``mapped`` preparation.
    """
    import json

    from gpuwm import prepared_single_domain_forecast as runner

    pins, current, earlier = _gdas_documents()
    stale = json.dumps(earlier).encode()
    for source, profile in (("gdas", "gdas-pgrb2-0p25-grib2-v1"),
                            ("mapped", None)):
        message = runner._stale_profile_mapping_refusal(
            source=source, profile_id=profile, mapping_bytes=stale,
            composition_sha256=pins["composition"], prepared_root=tmp_path)
        assert message is not None, source
        assert "gdas publishes" in message
        assert "qc, qr, qi, qs, qg" in message
        assert "gpuwm prep --source gdas" in message
    fresh = json.dumps(current).encode()
    assert runner._stale_profile_mapping_refusal(
        source="gdas", profile_id="gdas-pgrb2-0p25-grib2-v1",
        mapping_bytes=fresh, composition_sha256=pins["composition"],
        prepared_root=tmp_path) is None
    # A caller's own mapping is its own: another name, or another
    # composition, is not the profile's stale preparation.
    renamed = dict(earlier, name="my-own-gfs-mapping")
    assert runner._stale_profile_mapping_refusal(
        source="mapped", profile_id=None,
        mapping_bytes=json.dumps(renamed).encode(),
        composition_sha256=pins["composition"], prepared_root=tmp_path) is None
    assert runner._stale_profile_mapping_refusal(
        source="mapped", profile_id=None, mapping_bytes=stale,
        composition_sha256="0" * 64, prepared_root=tmp_path) is None


def test_a_mapped_root_publishes_what_its_bound_mapping_declares(tmp_path):
    import json

    from gpuwm import prepared_single_domain_forecast as runner

    _pins, current, earlier = _gdas_documents()
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(current), encoding="utf-8")
    assert runner._boundary_publication(
        "mapped", {"mapped_mapping": path}) == FIVE
    path.write_text(json.dumps(earlier), encoding="utf-8")
    assert runner._boundary_publication(
        "mapped", {"mapped_mapping": path}) == ()
    assert runner._boundary_publication("hrrr", {}) == FIVE
    assert runner._boundary_publication("gfs", {}) == ()


# -------------------------------------------------------------------- price

def test_the_price_carries_the_added_tables_and_no_held_slot():
    from gpuwm.core.preflight import (
        estimate_domain, lbc_interval_values, scratch_slot_registry)
    cfg = _cfg(nz=6)
    carried = boundary_hydrometeor_fields(cfg, FIVE)
    per_field = 2 * (2 * cfg.nz * cfg.ny * cfg.spec_bdy_width
                     + 2 * cfg.nz * cfg.spec_bdy_width * cfg.nx)
    assert (lbc_interval_values(cfg, boundary_species=FIVE)
            - lbc_interval_values(cfg)) == len(carried) * per_field
    assert potential_external_scalar_fields(cfg) == ("qv",)
    slots = scratch_slot_registry(cfg, n_lbc_intervals=3)
    assert not any(f"lbc_{name}_held" in slots for name in carried)
    dc = SimpleNamespace(run=cfg, grid_id=1, parent_id=0)
    plain = estimate_domain(dc, n_lbc_intervals=3)
    carrying = estimate_domain(dc, n_lbc_intervals=3, boundary_species=FIVE)
    table = {item.name: item.nbytes for item in carrying.items}
    base = {item.name: item.nbytes for item in plain.items}
    assert (table["lbc_forcing_tables"] - base["lbc_forcing_tables"]
            == 4 * 3 * len(carried) * per_field)


def test_the_alloc_measurement_builds_the_tables_it_prices():
    """``gpuwm check --alloc`` uploads the tables the run holds, and prices them.

    Its synthetic root boundary carries the masses and seeded numbers the
    source puts on the run's boundary, so the eager upload it measures is
    the same size as the ``lbc_forcing_tables`` its estimate and the
    report's observed envelope price.  Red before the A92 follow-up: the
    synthetic set held water vapour alone while the report's peak
    envelope priced the hydrometeor tables, so ``--alloc`` printed two
    forecast envelopes for one run.
    """
    from gpuwm.core.preflight import (
        _synthetic_root_boundaries, lbc_interval_values)
    from gpuwm.ingest.lateral_bc import boundary_storage_shapes

    cfg = _cfg(nz=6)
    carried = boundary_hydrometeor_fields(cfg, FIVE)
    assert carried
    for species, fields in ((FIVE, carried), ((), ())):
        built = _synthetic_root_boundaries(cfg, 3, boundary_species=species)
        assert len(built.intervals) == 3
        for interval in built.intervals:
            assert set(fields) <= set(interval.fields)
            assert not (set(BOUNDARY_HYDROMETEOR_MASSES) - set(fields)) & set(
                interval.fields)
        assert boundary_storage_shapes(built)["lbc_forcing_tables"] == (
            3 * lbc_interval_values(cfg, boundary_species=species),)


def _hrrr_demo():
    """The shipped native-HRRR 3 km demo: one mp=8 specified root."""
    from pathlib import Path

    from gpuwm.experiment import load_experiment

    exp = load_experiment(Path(__file__).resolve().parents[1] / "configs"
                          / "hrrr_native_3km_demo.toml")
    assert len(exp.domains) == 1 and exp.root.run.specified
    return exp


def _forcing_tables(estimate):
    (tables,) = [item.nbytes for domain in estimate.domains
                 for item in domain.items
                 if item.name == "lbc_forcing_tables"]
    return tables


def _added_values(run):
    from gpuwm.core.preflight import lbc_interval_values

    added = (lbc_interval_values(run, boundary_species=FIVE)
             - lbc_interval_values(run))
    assert added == len(boundary_hydrometeor_fields(run, FIVE)) * (
        4 * run.nz * run.spec_bdy_width * (run.nx + run.ny))
    return added


@pytest.mark.parametrize("source", ["hrrr", "hrrr-prs", "mapping"])
def test_the_admission_prices_the_tables_the_source_publishes(source):
    """Every run door's admission carries the source's hydrometeor tables.

    The tiles admission (cold_tree_streaming_decision,
    cold_single_domain_admission), the resident admission
    (admit_resident_road) and the chained and supervised reservations all
    price from ``admission_estimate``; left without the source it priced a
    HRRR-forced root on water vapour tables alone while the review
    (``estimate_phases``) carried the hydrometeors, so the door weighed a
    smaller forecast than the run holds.
    """
    from gpuwm.core import preflight as pf
    from gpuwm.core import streaming as st
    from gpuwm.ingest import boundary_stream
    from gpuwm.mapped_source import HYDROMETEOR_LEGACY_NAMES
    from gpuwm.supervisor import priced_reservation_bytes

    if source == "mapping":
        # A user's own mapping that declares all five: the document is
        # the table the mapped route hands in.
        source = {"fields": {name: {} for name in HYDROMETEOR_LEGACY_NAMES}}
    exp = _hrrr_demo()
    run = exp.root.run
    intervals = pf.lbc_intervals(exp.run_seconds,
                                 pf.DEFAULT_FORCING_INTERVAL_SECONDS)
    added = 4 * intervals * _added_values(run)
    vapour = pf.admission_estimate(exp)
    carrying = pf.admission_estimate(exp, source=source)
    assert _forcing_tables(carrying) - _forcing_tables(vapour) == added
    assert carrying.resident_bytes - vapour.resident_bytes == added
    assert carrying.peak_envelope_bytes > vapour.peak_envelope_bytes
    # A source that publishes none is priced as it was, byte for byte.
    for plain in (None, "gfs", "era5", "not-a-source", {"fields": {}}):
        assert (pf.admission_estimate(exp, source=plain).resident_bytes
                == vapour.resident_bytes)
    # The review and the doors weigh the same tables.
    review = pf.estimate_phases(exp, source=source if isinstance(
        source, str) else "hrrr").forecast
    assert _forcing_tables(review) == _forcing_tables(carrying)
    single = st.cold_single_domain_admission(exp, source=source)
    assert single.peak_envelope_bytes == carrying.peak_envelope_bytes
    assert (priced_reservation_bytes(exp, source=source)
            == carrying.peak_envelope_bytes)
    chained = boundary_stream.chained_admission(
        experiment=exp, backend="cuda", device_bytes=0,
        card=(1 << 40, 0), source=source)
    assert chained["forecast_bytes"] == carrying.peak_envelope_bytes


def test_the_resident_door_refuses_what_the_tables_push_over_the_card(
        monkeypatch):
    """``admit_resident_road`` on a card between the two envelopes."""
    from gpuwm.core import preflight as pf
    from gpuwm.core import streaming as st
    from gpuwm.core.resident_admission import (
        MEMORY_GATE_OVERRIDE_ENV, resident_forecast_terms)
    from gpuwm.ingest.memory_refusal import ResidentMemoryRefused

    monkeypatch.delenv(MEMORY_GATE_OVERRIDE_ENV, raising=False)
    exp = _hrrr_demo()
    vapour = pf.admission_estimate(exp)
    carrying = pf.admission_estimate(exp, source="hrrr")

    def need(estimate):
        return sum(int(value) for value in
                   dict(resident_forecast_terms(estimate)).values())

    assert need(carrying) - need(vapour) > 0
    card = SimpleNamespace(vram_bytes=(need(vapour) + need(carrying)) // 2,
                           device_profile=None)
    assert st.admit_resident_road(exp, None, machine=card)["fits"] is True
    with pytest.raises(ResidentMemoryRefused) as refused:
        st.admit_resident_road(exp, None, machine=card, source="hrrr")
    assert refused.value.need_bytes == need(carrying)
    # At a prepared cache's retained interval count too.
    retained = 4
    added = 4 * retained * _added_values(exp.root.run)
    at_count = pf.estimate_experiment(
        exp, column_chunk=exp.column_chunk, forcing_intervals=retained,
        boundary_species=FIVE)
    plain = pf.estimate_experiment(
        exp, column_chunk=exp.column_chunk, forcing_intervals=retained)
    assert _forcing_tables(at_count) - _forcing_tables(plain) == added


@pytest.mark.parametrize("source", ["hrrr", "hrrr-prs"])
def test_the_host_series_carries_the_tables_the_source_publishes(source):
    """The streamed host claim holds the float64 series with the masses.

    A streamed root keeps its whole lateral forcing series on the host
    and cuts each tile's edge from it; priced on water vapour alone, a
    HRRR-forced streamed run's host admission was short by the
    hydrometeor tables (``lbc_host_series_bytes``,
    ``streaming._boundary_series_host_bytes``, the streamed envelope's
    ``host/boundary_table_bytes`` and the tree road's).
    """
    from dataclasses import replace as _replace

    from gpuwm.core import preflight as pf
    from gpuwm.core import streaming as st

    exp = _hrrr_demo()
    run = exp.root.run
    retained = 6
    added = 8 * retained * _added_values(run)
    assert (pf.lbc_host_series_bytes(run, retained, source=source)
            - pf.lbc_host_series_bytes(run, retained)) == added
    assert pf.lbc_host_series_bytes(run, retained, source="gfs") == (
        pf.lbc_host_series_bytes(run, retained))
    assert (st._boundary_series_host_bytes(run, 3600.0, retained,
                                           source=source)
            - st._boundary_series_host_bytes(run, 3600.0, retained)) == added
    # A nest's forcing is its parent's rolling frame: no host series.
    nest = _replace(run, specified=False, nested=True)
    assert pf.lbc_host_series_bytes(nest, retained, source=source) == 0

    options = _replace(st.options_for_domain(exp.domains[0], exp.tiles),
                       mode="on", tile_nx=64, tile_ny=64, nbuffers=2)
    vapour = st.streamed_envelope(run, options, forcing_intervals=retained)
    carrying = st.streamed_envelope(run, options, forcing_intervals=retained,
                                    source=source)
    assert carrying.boundary_table_bytes - vapour.boundary_table_bytes == added
    assert carrying.host_bytes - vapour.host_bytes == added
    assert carrying.vram_bytes == vapour.vram_bytes
    tiled = _replace(exp, tiles=options)
    envelope = pf.streamed_forecast_envelope(
        tiled, forcing_intervals=retained, source=source)
    assert envelope.boundary_table_bytes == carrying.boundary_table_bytes


# -------------------------------------------------------------- the forecast

@pytest.mark.gpu
@pytest.mark.parametrize("final", [False, True])
def test_the_forecast_ring_takes_the_supplied_hydrometeors(final):
    """Inflow on the west edge: the specified ring follows the table's mass
    and number (not the zero a flow-dependent inflow gives it), and an
    unsupplied number keeps the flow-dependent zero."""
    cp = pytest.importorskip("cupy")
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import advance_scalars_stage
    from gpuwm.core.state import init_at_rest
    from gpuwm.ingest.lateral_bc import (
        attach_lateral_boundaries, build_lateral_boundaries,
        domain_boundary_snapshot)

    cfg = _cfg(nz=6)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: np.full_like(z, 300.),
                           cfg.p_surf, cfg.ztop)
    state = init_at_rest(cfg, coord, base)
    state._external_scalar_boundary_fields = external_scalar_fields(
        cfg, boundary_species=("qs", "qr"))
    assert state._external_scalar_boundary_fields == ("qv", "qr", "qs", "nr")
    settled = (("qv", .01), ("qr", 2e-4), ("qs", 5e-4), ("nr", 3e3),
               ("ni", 7e4))
    for name, value in settled:
        getattr(state, name)[:] = value
        getattr(state, name + "0")[:] = value
    state.mup0[:] = state.mup
    first = domain_boundary_snapshot(state)
    for name in ("qr", "qs", "nr"):
        getattr(state, name)[:] *= 1.25
    second = domain_boundary_snapshot(state)
    for name in ("qr", "qs", "nr"):
        getattr(state, name)[:] = getattr(state, name + "0")
    attach_lateral_boundaries(
        state, build_lateral_boundaries([first, second], [0., 60.]))
    ru = cp.full_like(state.u, 1.)
    rv = cp.zeros_like(state.v)
    ww = cp.zeros_like(state.w)
    advance_scalars_stage(state, cfg, ru, rv, ww, dt_eff=1., final=final,
                          apply_relax=True)
    for name, value in (("qr", 2e-4), ("qs", 5e-4), ("nr", 3e3)):
        np.testing.assert_allclose(
            cp.asnumpy(getattr(state, name))[:, :, 0],
            value * (1. + .25 / 60.), rtol=3e-6)
    np.testing.assert_array_equal(cp.asnumpy(state.ni)[:, :, 0], 0.)
