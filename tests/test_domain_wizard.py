"""``gpuwm domain`` wizard gates: sizing fit, round trips, edge rejects.

The wizard owns no memory arithmetic and no projection math of its own,
so these tests bind its glue: emitted configs load through the REAL
experiment/case-data loaders, their estimator envelope fits the declared
card budget, the companion namelist.wps agrees with the [projection]
table through the real grid builders, and every documented refusal
(pole containment, bad cycle, missing budget) fails loudly with an
actionable message.  Worldwide contract: the projection is
auto-selected by |lat| (mercator < 25 <= lambert <= 60 < polar),
both hemispheres emit, and antimeridian-crossing footprints produce
wrap-aware (W > E) fetch boxes instead of refusals.
"""
from __future__ import annotations

import os

import tomllib
from pathlib import Path

import numpy as np
import pytest

from gpuwm.case_data import load_experiment_case
from gpuwm.cli import main as cli_main
from gpuwm.core.preflight import (GIB, estimate_experiment,
                                  observed_peak_envelope_bytes)
from gpuwm.domain_wizard import (CARD_VRAM_GIB, LADDER_RATIOS,
                                 _dims_for_scale, experiment_from_text,
                                 vram_reserve_gib)
from gpuwm.experiment import load_experiment
from gpuwm.fetch import validate_fetch_hints
from gpuwm.ingest.grib import parse_vtable
from gpuwm.static.lambert import (grids_from_projection_config,
                                  grids_from_wps_namelist)

BUNDLE = Path(os.environ.get("GPUWM_TEST_WRF74_BUNDLE",
                    "gpuwm-fixture-unset/wrf74-bundle"))
MAY99 = Path(os.environ.get("GPUWM_TEST_MAY99_DATA",
                    "gpuwm-fixture-unset/may99-data"))
requires_staged_real_inputs = pytest.mark.skipif(
    not (MAY99 / "era5_may1999_pl.grib").is_file()
    or not (BUNDLE / "static/WPS_GEOG/topo_gmted2010_30s").is_dir(),
    reason="staged May-1999 ERA5 or the WPS_GEOG tree is absent",
)


def _run_wizard(tmp_path, *extra, point="39.7,-96.6", card="16gb",
                ladder="12-3", source="era5", cycle="1999-05-03T12"):
    out = tmp_path / "area.toml"
    # --point=VALUE form: a leading "-" (southern latitude) must not be
    # parsed as an option flag.
    rc = cli_main([
        "domain", f"--point={point}", "--card", card, "--ladder", ladder,
        "--source", source, "--cycle", cycle, "--out", str(out), *extra])
    return rc, out


# ---------------------------------------------------------------------------
# Input rejection: every documented refusal is a stderr message + exit 2
# through the CLI dispatch boundary -- never a Python traceback.
# ---------------------------------------------------------------------------

def _assert_refused(capsys, needle: str, rc: int) -> None:
    assert rc == 2
    err = capsys.readouterr().err
    assert needle in err
    assert "Traceback" not in err


@pytest.mark.parametrize("point, needle", [
    ("35.3", "lat,lon"),
    ("abc,-97.5", "decimal degrees"),
    ("95.0,-60.0", "[-90, 90]"),
    ("90.0,-60.0", "pole itself"),
    ("-90.0,10.0", "pole itself"),
    ("35.3,-190.0", "[-180, 180]"),
])
def test_point_rejections(tmp_path, capsys, point, needle):
    rc, out = _run_wizard(tmp_path, point=point)
    _assert_refused(capsys, needle, rc)
    assert not out.exists()


def test_card_and_vram_gib_are_mutually_exclusive(tmp_path, capsys):
    rc, _ = _run_wizard(tmp_path, "--vram-gib", "20")
    _assert_refused(capsys, "mutually exclusive", rc)


def test_vram_below_reserve_rejected(tmp_path, capsys):
    out = tmp_path / "area.toml"
    rc = cli_main(["domain", "--point", "39.7,-96.6", "--vram-gib", "2.5",
                   "--cycle", "1999-05-03T12", "--out", str(out)])
    _assert_refused(capsys, "no budget", rc)


def test_bad_cycle_and_gfs_synoptic_hours(tmp_path, capsys):
    rc, _ = _run_wizard(tmp_path, cycle="not-a-time")
    _assert_refused(capsys, "YYYY-MM-DDTHH", rc)
    # GFS cycles are synoptic-only; parse_cycle enforces per source.
    rc, _ = _run_wizard(tmp_path, source="gfs", cycle="2026-07-28T05")
    _assert_refused(capsys, "00/06/12/18", rc)


def test_hours_minimum(tmp_path, capsys):
    rc, _ = _run_wizard(tmp_path, "--hours", "0")
    _assert_refused(capsys, "at least 1", rc)


def test_antimeridian_footprint_emits_wrapping_fetch_box(tmp_path):
    # Worldwide contract: a footprint straddling 180E is supported end
    # to end; the fetch hint wraps (W > E in the signed convention).
    rc, out = _run_wizard(tmp_path, point="52.0,179.5")
    assert rc == 0
    raw = tomllib.loads(out.read_text(encoding="utf-8"))
    s, w, n, e = (float(v) for v in raw["fetch"]["area"].split(","))
    assert w > 0.0 > e, (w, e)  # crossing box: west near 170E, east near -170W
    from gpuwm.fetch import parse_area
    area = parse_area(raw["fetch"]["area"])
    assert area.crosses_antimeridian


def test_pole_containing_domain_refused(tmp_path, capsys):
    # Genuine limit: the fitted footprint may not contain the pole.
    rc, _ = _run_wizard(tmp_path, point="89.0,-100.0", ladder="12",
                        source="gfs", cycle="2026-07-28T06")
    _assert_refused(capsys, "pole", rc)


def test_margined_span_over_180_refused_never_flipped(tmp_path, capsys):
    """Audit reproduction point: auto projection, point 34,0, GFS,
    --vram-gib 64, single-domain ladder.  The fitted 880x704 root's raw
    span (165.1 deg) fits, but the GFS source margin (15 deg per side)
    pushes the EMITTED box to 195.1 deg -- parse_area would read that
    back as the complementary antimeridian crossing.  The wizard must
    refuse on the margined span, not emit the wrong box."""
    out = tmp_path / "area.toml"
    rc = cli_main([
        "domain", "--point=34,0", "--vram-gib", "64", "--ladder", "12",
        "--source", "gfs", "--cycle", "2026-07-28T06", "--out", str(out)])
    _assert_refused(capsys, "boxes wider than 180 degrees", rc)
    assert not out.exists()


def test_fetch_area_just_under_the_limit_round_trips_unflipped():
    """Spans just under the 180-degree refusal must survive the
    emit -> parse_area round trip as the same box (parse_area flips
    only spans OVER 180 into the complementary crossing)."""
    from gpuwm.domain_wizard import _fetch_area, _fetch_margin_deg, \
        _projection_entries
    from gpuwm.fetch import parse_area

    projection = _projection_entries(34.0, 0.0)
    margin = _fetch_margin_deg("gfs")
    area = None
    for nx in range(880, 400, -8):  # widest layout the margined gate admits
        try:
            area = _fetch_area(projection, nx, 704, margin_deg=margin)
        except ValueError:
            continue
        break
    assert area is not None, "no layout fit under the margined gate"
    lat_s, lon_w, lat_n, lon_e = area
    assert lon_w < 0.0 < lon_e  # centered on ref_lon = 0, not crossing
    span = lon_e - lon_w
    assert 160.0 < span <= 180.0, span  # genuinely near the limit
    parsed = parse_area(",".join(f"{v:.2f}" for v in area))
    assert not parsed.crosses_antimeridian
    assert parsed.lon_west == pytest.approx(lon_w, abs=0.01)
    assert parsed.lon_east == pytest.approx(lon_e, abs=0.01)


def test_projection_auto_selection_bands():
    from gpuwm.domain_wizard import _projection_entries, auto_projection

    assert auto_projection(1.3) == "mercator"
    assert auto_projection(-17.8) == "mercator"
    assert auto_projection(-27.5) == "lambert"
    assert auto_projection(39.7) == "lambert"
    assert auto_projection(64.8) == "polar"
    assert auto_projection(-77.85) == "polar"
    # Hemisphere-correct Lambert truelats (both signed with the point).
    sh = _projection_entries(-27.5, 153.0)
    assert sh["map_proj"] == "lambert"
    assert sh["truelat1"] == -17.5 and sh["truelat2"] == -37.5
    # Explicit override wins.
    forced = _projection_entries(-27.5, 153.0, "mercator")
    assert forced["map_proj"] == "mercator"
    assert forced["truelat1"] == -27.5
    with pytest.raises(ValueError, match="--projection"):
        _projection_entries(10.0, 0.0, "cassini")


@pytest.mark.parametrize("point, source, cycle, map_proj, wrf_code", [
    ("-27.5,153.0", "gfs", "2026-07-28T06", "lambert", 1),
    ("1.3,103.8", "gfs", "2026-07-28T06", "mercator", 3),
    ("64.8,-147.7", "gfs", "2026-07-28T06", "polar", 2),
    ("-17.8,178.5", "gfs", "2026-07-28T06", "mercator", 3),
])
def test_worldwide_points_emit_and_round_trip(tmp_path, point, source,
                                              cycle, map_proj, wrf_code):
    """The four worldwide gate sites (Brisbane, Singapore, Fairbanks,
    Fiji) emit, declare the right projection, and round-trip through
    the real loaders and grid builders."""
    rc, out = _run_wizard(tmp_path, "--ladder", "12", point=point,
                          source=source, cycle=cycle)
    assert rc == 0
    raw = tomllib.loads(out.read_text(encoding="utf-8"))
    assert raw["projection"]["map_proj"] == map_proj
    assert raw["shared"]["map_proj"] == wrf_code
    # Single-domain emission: portable-forecast contract.
    assert raw["experiment"]["restart_interval_s"] == 0.0
    exp = load_experiment(out)
    grids = grids_from_projection_config(exp)
    assert len(grids) == 1
    wps = out.parent / f"{out.stem}.namelist.wps"
    wps_text = wps.read_text(encoding="utf-8")
    assert f"map_proj = '{map_proj}'" in wps_text
    wps_grids = grids_from_wps_namelist(wps)
    lat_a, lon_a = grids[0].latlon_mass()
    lat_b, lon_b = wps_grids[0].latlon_mass()
    np.testing.assert_allclose(lat_a, lat_b, rtol=0, atol=1e-9)
    np.testing.assert_allclose(lon_a, lon_b, rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------
# Layout invariants (quantization rules the loader will re-check)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ladder", sorted(LADDER_RATIOS))
@pytest.mark.parametrize("scale", [0.55, 1.0, 2.3, 5.7])
def test_dims_even_divisible_and_clear(ladder, scale):
    ratios = LADDER_RATIOS[ladder]
    dims = _dims_for_scale(scale, ratios)
    assert len(dims) == len(ratios) + 1
    for nx, ny in dims:
        assert nx % 2 == 0 and ny % 2 == 0
    for (pnx, pny), (nx, ny), ratio in zip(dims, dims[1:], ratios):
        assert nx % ratio == 0 and ny % ratio == 0
        # Centered child leaves >= 10 parent rows (Davies + blend zones).
        assert (pnx - nx // ratio) // 2 >= 10
        assert (pny - ny // ratio) // 2 >= 10


# ---------------------------------------------------------------------------
# Sizing fit: emitted dims' estimator envelope fits the card budget
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("card", sorted(CARD_VRAM_GIB))
def test_fit_fills_card_budget(tmp_path, card):
    rc, out = _run_wizard(tmp_path, card=card, ladder="auto")
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    exp = experiment_from_text(text, source=str(out))
    estimate = estimate_experiment(exp, forcing_interval_seconds=21600.0)
    envelope = observed_peak_envelope_bytes(
        estimate.footprint_projection_bytes)
    vram = CARD_VRAM_GIB[card]
    budget = int((vram - vram_reserve_gib(vram)) * GIB)
    assert envelope <= budget
    # The bisection must actually spend the budget, not stop at the floor.
    assert envelope >= 0.8 * budget
    # Certified clock/dx conventions on the emitted chain.
    root = exp.root
    assert root.time_step == 60 and root.run.dx == 12000.0
    for dc in exp.domains:
        assert exp.dx_exact(dc.grid_id) == exp.dx_exact(1) / np.prod(
            [d.parent_grid_ratio for d in exp.domains
             if 1 < d.grid_id <= dc.grid_id], dtype=int)


def test_auto_picks_deepest_ladder_on_32gb(tmp_path):
    rc, out = _run_wizard(tmp_path, card="32gb", ladder="auto")
    assert rc == 0
    exp = experiment_from_text(out.read_text(encoding="utf-8"),
                               source=str(out))
    assert len(exp.domains) == 4  # 12-3-1-0.5
    assert [dc.parent_grid_ratio for dc in exp.domains] == [1, 4, 3, 2]
    assert float(exp.dx_exact(4)) == 500.0


# ---------------------------------------------------------------------------
# The default physics suite, pinned
# ---------------------------------------------------------------------------

def test_wizard_default_suite_is_the_registered_default_template(tmp_path):
    """The wizard emits Thompson mp8 by default (product decision,
    2026-07-29) and its selectors match the registry's declared default
    template (``DEFAULT_TEMPLATE_ID``), so the wizard and the registry
    cannot drift apart.  Morrison (mp10) stays registered and selectable
    at its own maturity label; its Morrison-only ``morr_rimed_ice`` knob
    must not ride along on a Thompson default."""
    from gpuwm.physics_registry import DEFAULT_TEMPLATE_ID, physics_registry

    rc, out = _run_wizard(tmp_path)
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    exp = experiment_from_text(text, source=str(out))
    root = exp.domains[0].run

    registry = physics_registry()
    template = registry["templates"][DEFAULT_TEMPLATE_ID]
    assert template["components"]["microphysics"] == "thompson-mp8"

    selectors: dict[str, int] = {}
    for comp_id, opt_id in template["components"].items():
        selectors.update(
            registry["components"][comp_id]["options"][opt_id]["selectors"])
    # The registry speaks WRF's lw/sw pair; the wizard emits the modern
    # combined selector for the same 4/4 = RTE+RRTMGP route.
    assert selectors.pop("ra_lw_physics") == 4
    assert selectors.pop("ra_sw_physics") == 4
    assert root.ra_physics == 4
    # Kain-Fritsch on the 12 km root only; every nest runs cumulus off.
    assert root.cu_physics == selectors.pop("cu_physics") == 1
    assert all(dc.run.cu_physics == 0 for dc in exp.domains[1:])
    for key, value in selectors.items():
        assert getattr(root, key) == value, key
    assert all(dc.run.mp_physics == 8 for dc in exp.domains)
    assert "morr_rimed_ice" not in text
    # Product decision (STEP17): wizard configs ship the UP_HELI_MAX
    # diagnostic ON -- this audience reads UH products.
    assert all(dc.run.nwp_diagnostics == 1 for dc in exp.domains)


# ---------------------------------------------------------------------------
# Round trips through the real loaders
# ---------------------------------------------------------------------------

def test_emitted_era5_config_round_trips(tmp_path, capsys):
    geog = tmp_path / "GEOG"
    geog.mkdir()
    rc, out = _run_wizard(tmp_path, "--geog-root", str(geog),
                          ladder="12-3-1")
    assert rc == 0
    printed = capsys.readouterr().out
    # Forcing is not on disk yet: the wizard defers the composed check
    # and prints the exact follow-up command + the geog story.
    assert "gpuwm check: deferred" in printed
    assert "gpuwm check" in printed and "--budget-gib" in printed
    assert "WPS_GEOG" in printed

    raw = tomllib.loads(out.read_text(encoding="utf-8"))
    assert raw["fetch"]["source"] == "era5"
    assert raw["case_data"]["wps_namelist"] == "area.namelist.wps"
    # The packaged ERA5 Vtable was copied beside the TOML and parses.
    vtable = out.parent / "Vtable.ERA5_CDO"
    assert vtable.is_file()
    assert len(parse_vtable(vtable)) > 20

    # Create the declared forcing; the full case loader then accepts the
    # emitted file as-is ([fetch] split off and validated, not rejected).
    forcing = out.parent / Path(raw["case_data"]["forcing"][0])
    forcing.parent.mkdir(parents=True, exist_ok=True)
    forcing.write_bytes(b"stub")
    exp, data = load_experiment_case(out)
    assert len(exp.domains) == 3
    assert exp.run_seconds == 6 * 3600.0
    assert data.geog_root == geog
    assert data.source_orography is None  # era5_z_invariant provider
    assert data.forcing_interval_s == 21600.0


def test_wps_namelist_agrees_with_projection_config(tmp_path):
    rc, out = _run_wizard(tmp_path, ladder="12-3-1")
    assert rc == 0
    exp = experiment_from_text(out.read_text(encoding="utf-8"),
                               source=str(out))
    from_toml = grids_from_projection_config(exp)
    from_wps = grids_from_wps_namelist(out.parent / "area.namelist.wps")
    assert len(from_toml) == len(from_wps) == 3
    for a, b in zip(from_toml, from_wps):
        assert (a.e_we, a.e_sn, a.dx) == (b.e_we, b.e_sn, b.dx)
        for attr in ("ref_lat", "ref_lon", "truelat1", "truelat2",
                     "stand_lon", "known_x", "known_y"):
            assert getattr(a, attr) == pytest.approx(
                getattr(b, attr), abs=1e-9), attr
        lat_a, lon_a = a.latlon_mass()
        lat_b, lon_b = b.latlon_mass()
        np.testing.assert_allclose(lat_a, lat_b, atol=1e-9)
        np.testing.assert_allclose(lon_a, lon_b, atol=1e-9)


def test_children_are_centered_on_the_point(tmp_path):
    rc, out = _run_wizard(tmp_path, ladder="12-3-1", point="39.7,-96.6")
    assert rc == 0
    exp = experiment_from_text(out.read_text(encoding="utf-8"),
                               source=str(out))
    for grid in grids_from_projection_config(exp):
        # The wizard's centering arithmetic makes every domain's grid
        # center coincide with the parent's exactly (child center in
        # parent coordinates = P/2 + 0.5 = the parent center), so each
        # projected center maps back to the requested point.
        lat, lon = grid.ij_to_latlon(grid.e_we / 2.0, grid.e_sn / 2.0)
        assert float(lat) == pytest.approx(39.7, abs=1e-9)
        assert float(lon) == pytest.approx(-96.6, abs=1e-9)


# ---------------------------------------------------------------------------
# GFS/HRRR honesty: no [case_data], actionable front-door messages
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("source", ["gfs", "hrrr"])
def test_native_sources_omit_case_data(tmp_path, capsys, source):
    cycle = "2026-07-28T06" if source == "gfs" else "2026-07-28T05"
    rc, out = _run_wizard(tmp_path, source=source, cycle=cycle)
    assert rc == 0
    printed = capsys.readouterr().out
    assert "rw-wps" in printed  # the honest front-door note
    raw = tomllib.loads(out.read_text(encoding="utf-8"))
    assert "case_data" not in raw
    assert raw["fetch"]["source"] == source
    # load_experiment (the native front doors' loader) accepts the file.
    exp = load_experiment(out)
    assert len(exp.domains) == 2


def test_check_without_case_data_runs_memory_preflight(tmp_path, capsys):
    """No [case_data] (GFS/HRRR emissions): `gpuwm check` says the
    input preflight is not applicable, then certifies the memory
    preflight -- rc 0 with the honest note, not a refusal."""
    rc, out = _run_wizard(tmp_path, source="gfs", cycle="2026-07-28T06")
    assert rc == 0
    capsys.readouterr()
    rc = cli_main(["check", str(out), "--budget-gib", "20"])
    printed = capsys.readouterr().out
    assert rc == 0
    assert "not applicable" in printed
    assert "rw-wps" in printed
    assert "memory preflight" in printed


def test_existing_divergent_vtable_is_never_overwritten(tmp_path, capsys):
    (tmp_path / "Vtable.ERA5_CDO").write_text("not the packaged table")
    rc, _ = _run_wizard(tmp_path)
    _assert_refused(capsys, "refusing to overwrite", rc)


# ---------------------------------------------------------------------------
# [fetch] hints schema
# ---------------------------------------------------------------------------

def test_fetch_hints_validation():
    good = {"source": "era5", "cycle": "1999-05-03T12", "hours": 6,
            "area": "25,-112,45,-83", "out": "data/x", "cadence": 6}
    validate_fetch_hints(good, source="unit")
    with pytest.raises(ValueError, match="unknown key"):
        validate_fetch_hints({"source": "era5", "extra": 1}, source="unit")
    with pytest.raises(ValueError, match="must carry source"):
        validate_fetch_hints({"hours": 6}, source="unit")
    with pytest.raises(ValueError, match="not one of"):
        validate_fetch_hints({"source": "cfs"}, source="unit")
    with pytest.raises(ValueError, match="scalar"):
        validate_fetch_hints({"source": "era5", "hours": [1, 2]},
                             source="unit")


def test_loaders_reject_bad_fetch_table(tmp_path):
    rc, out = _run_wizard(tmp_path, source="gfs", cycle="2026-07-28T06")
    assert rc == 0
    text = out.read_text(encoding="utf-8").replace(
        'source = "gfs"', 'source = "gfs"\nbogus_key = 1')
    bad = tmp_path / "bad.toml"
    bad.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="bogus_key"):
        load_experiment(bad)


# ---------------------------------------------------------------------------
# Full composed-check integration on staged real inputs (gated)
# ---------------------------------------------------------------------------

@requires_staged_real_inputs
@pytest.mark.slow
def test_wizard_full_check_passes_on_staged_inputs(tmp_path, capsys):
    """point -> emitted config -> composed `gpuwm check` rc 0, in-process.

    A genuinely new area (central Oklahoma) against the staged May-1999
    ERA5 GRIB1 pair and the standard WPS_GEOG tree: the wizard's final
    step runs the real composed check (input preflight decode + geog tile
    coverage + memory estimate vs budget) and must report PASS.
    """
    out = tmp_path / "okc.toml"
    rc = cli_main([
        "domain", "--point", "35.3,-97.5", "--card", "24gb",
        "--cycle", "1999-05-03T12", "--hours", "6",
        "--out", str(out),
        "--forcing", str(MAY99 / "era5_may1999_pl.grib"),
        str(MAY99 / "era5_may1999_sl.grib"),
        "--geog-root", str(BUNDLE / "static/WPS_GEOG")])
    printed = capsys.readouterr().out
    assert rc == 0
    assert "gpuwm input preflight: PASS" in printed
    assert "gpuwm check: PASS (rc 0)" in printed
    assert "WARNING" not in printed  # envelope fit keeps check warning-free


def test_gfs_fetch_hint_margin_is_the_front_door_coverage_margin():
    """The wizard's suggested GFS --area must pass the front door's own
    donor-coverage proof: its margin comes from the one shared function
    (gpuwm.fetch.gfs_suggested_fetch_margin_deg) instead of a private
    too-small constant the coverage check then rejects."""
    from gpuwm.domain_wizard import _FETCH_MARGIN_DEG, _fetch_margin_deg
    from gpuwm.fetch import (GFS_LAKE_DONOR_MARGIN_DEG,
                             gfs_suggested_fetch_margin_deg)

    assert _fetch_margin_deg("gfs") == gfs_suggested_fetch_margin_deg()
    assert gfs_suggested_fetch_margin_deg() == GFS_LAKE_DONOR_MARGIN_DEG
    # The acceptance lane measured +8..15 deg beyond the old 2-deg hint
    # as the empirical requirement for an interior-CONUS domain.
    assert gfs_suggested_fetch_margin_deg() >= 8.0
    # ERA5 needs only interpolation halo; HRRR must stay inside its own
    # CONUS coverage box, so both keep the small margin.
    assert _fetch_margin_deg("era5") == _FETCH_MARGIN_DEG
    assert _fetch_margin_deg("hrrr") == _FETCH_MARGIN_DEG


def test_fetch_area_applies_and_clamps_the_margin():
    from gpuwm.domain_wizard import _fetch_area, _projection_entries

    projection = _projection_entries(35.0, -97.5)
    small = _fetch_area(projection, 60, 48, margin_deg=2.0)
    wide = _fetch_area(projection, 60, 48, margin_deg=15.0)
    # 13 more degrees on every side (S grows down, N up, W down, E up).
    assert wide[0] == pytest.approx(small[0] - 13.0)
    assert wide[1] == pytest.approx(small[1] - 13.0)
    assert wide[2] == pytest.approx(small[2] + 13.0)
    assert wide[3] == pytest.approx(small[3] + 13.0)
    # Near the dateline the margin wraps across the seam instead of
    # truncating: the donor margin is honoured on both sides, and the
    # resulting box is the W > E crossing form the fetch layer serves.
    west = _projection_entries(52.0, -170.0)
    wrapped = _fetch_area(west, 60, 48, margin_deg=15.0)
    assert wrapped[1] > 0.0 > wrapped[3]
    from gpuwm.fetch import Area
    assert Area(wrapped[0], wrapped[1], wrapped[2],
                wrapped[3]).crosses_antimeridian
