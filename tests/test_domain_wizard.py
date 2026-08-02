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

from fractions import Fraction
import os

import tomllib
from pathlib import Path

import numpy as np
import pytest

from gpuwm.case_data import load_experiment_case
from gpuwm.cli import main as cli_main
from gpuwm.core.preflight import (GIB, estimate_experiment,
                                  estimate_phases,
                                  observed_peak_envelope_bytes)
from gpuwm.domain_wizard import (CARD_VRAM_GIB, LADDER_RATIOS,
                                 DomainFitError, _dims_for_scale,
                                 card_assumed_free_gib,
                                 experiment_from_text, fit_headroom_bytes,
                                 sizing_budget_bytes, vram_reserve_gib)
from gpuwm.experiment import load_experiment
from gpuwm.fetch import validate_fetch_hints
from gpuwm.hrrr_route_inputs import HrrrRouteInputError, route_input_paths
from gpuwm.physics_compat import MORRISON_PROFILE_ID
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
    # parsed as an option flag.  card=None omits --card entirely, for the
    # cases that pass --vram-gib instead (the two are exclusive).
    rc = cli_main([
        "domain", f"--point={point}",
        *(() if card is None else ("--card", card)), "--ladder", ladder,
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
    ("35.3,-400.0", "[-180, 180]"),
])
def test_point_rejections(tmp_path, capsys, point, needle):
    rc, out = _run_wizard(tmp_path, point=point)
    _assert_refused(capsys, needle, rc)
    assert not out.exists()


def test_out_of_convention_longitude_wraps_with_a_warning(tmp_path, capsys):
    """Warn-not-block: 170E spelled as -190 is a real longitude; the
    wizard wraps it to the [-180, 180] convention, says so in one line,
    and proceeds.

    The artifacts are checked, not just the sentence.  This test used to
    assert the warning alone, and passed for two releases while the
    wrapped value reached the emitted TOML as a quoted STRING and the
    emitted namelist.wps as ``array(170.)``.
    """

    rc, out = _run_wizard(tmp_path, point="35.3,-190.0")
    captured = capsys.readouterr()
    assert rc == 0
    assert out.exists()
    assert "warning:" in captured.err
    assert "wrapped to 170" in captured.err

    projection = tomllib.loads(out.read_text())["projection"]
    for key in ("ref_lon", "stand_lon", "ref_lat", "truelat1", "truelat2"):
        assert isinstance(projection[key], float), (key, projection[key])
    assert projection["ref_lon"] == pytest.approx(170.0)

    namelist = (out.parent / f"{out.stem}.namelist.wps").read_text()
    for line in namelist.splitlines():
        key, _, value = line.partition("=")
        if key.strip() in ("ref_lat", "ref_lon", "truelat1", "truelat2",
                           "stand_lon"):
            # Fortran has to be able to read it.
            assert float(value.strip().rstrip(",")) == pytest.approx(
                projection[key.strip()])


def test_a_wrapped_longitude_matches_its_unwrapped_twin_byte_for_byte(
        tmp_path, capsys):
    """--point 35.3,-190 and --point 35.3,170 name the same meridian, so
    the two emissions may not differ in type or in text."""

    rc_a, out_a = _run_wizard(tmp_path / "a", point="35.3,-190.0")
    rc_b, out_b = _run_wizard(tmp_path / "b", point="35.3,170.0")
    capsys.readouterr()
    assert (rc_a, rc_b) == (0, 0)
    assert tomllib.loads(out_a.read_text())["projection"] == \
        tomllib.loads(out_b.read_text())["projection"]
    assert (out_a.parent / f"{out_a.stem}.namelist.wps").read_text() == \
        (out_b.parent / f"{out_b.stem}.namelist.wps").read_text()


def test_toml_emitter_refuses_to_quote_a_value_it_cannot_type(tmp_path):
    """A number emitted as a string is valid TOML under the right key
    and the wrong type, so it survives review.  The emitter renders
    scalars it recognises and refuses the rest rather than quoting."""
    from gpuwm.domain_wizard import _toml_value

    assert _toml_value(np.float32(-160.0)) == repr(-160.0)
    assert _toml_value(np.asarray(-160.0)) == repr(-160.0)
    assert _toml_value(np.int64(7)) == "7"
    assert _toml_value("lambert") == '"lambert"'
    with pytest.raises(TypeError, match="cannot render"):
        _toml_value(np.asarray([1.0, 2.0]))
    with pytest.raises(TypeError, match="cannot render"):
        _toml_value(object())


def test_point_longitude_refusal_names_the_range_it_enforces(
        tmp_path, capsys):
    """One wrap is accepted, so a message claiming [-180, 180] alone
    described a refusal that does not happen."""
    rc, out = _run_wizard(tmp_path, point="35.3,-400.0")
    _assert_refused(capsys, "[-360, 360]", rc)
    assert not out.exists()
    # and the accepted-with-a-wrap case really is accepted
    rc_ok, out_ok = _run_wizard(tmp_path / "ok", point="35.3,270.0")
    capsys.readouterr()
    assert rc_ok == 0 and out_ok.exists()


def test_card_and_vram_gib_are_mutually_exclusive(tmp_path, capsys):
    rc, _ = _run_wizard(tmp_path, "--vram-gib", "20")
    _assert_refused(capsys, "mutually exclusive", rc)


def test_vram_below_reserve_rejected(tmp_path, capsys):
    """A card too small to size is refused, and says which wall it hit.

    Two walls now, not one.  Below one CUDA context plus the external
    margin there is nothing to size against at all; above that, the fit
    loop refuses and names the layout, the arithmetic and the share of it
    that no smaller grid can move.  The flat 4 GiB reserve used to draw
    the line for both, which refused cards the suite-priced reserve would
    have sized.
    """
    out = tmp_path / "area.toml"
    rc = cli_main(["domain", "--point", "39.7,-96.6", "--vram-gib", "2.5",
                   "--cycle", "1999-05-03T12", "--out", str(out)])
    _assert_refused(capsys, "smallest layout exceeds the budget", rc)
    assert not out.exists()

    rc = cli_main(["domain", "--point", "39.7,-96.6", "--vram-gib", "0.9",
                   "--cycle", "1999-05-03T12", "--out", str(out)])
    _assert_refused(capsys, "leaves no budget", rc)
    assert not out.exists()


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


def test_negative_coordinates_parse_in_both_forms(tmp_path, capsys):
    """`--point -33.87,151.21` must work, not just `--point=-33.87,...`.

    argparse reads a leading `-` as an option prefix unless the token
    matches its negative-number regex, which a `lat,lon` pair never
    does.  Every documented example was a positive CONUS latitude, so
    the whole southern hemisphere failed with "expected one argument"
    -- on the release whose headline claim is worldwide forecasts.
    """
    from gpuwm.cli import _join_negative_coordinates

    assert _join_negative_coordinates(
        ["domain", "--point", "-33.87,151.21"]
    ) == ["domain", "--point=-33.87,151.21"]
    assert _join_negative_coordinates(
        ["fetch", "--area", "-58.58,119.65,-8.38,-177.23"]
    ) == ["fetch", "--area=-58.58,119.65,-8.38,-177.23"]
    # Positive values and non-coordinate flags are left exactly alone.
    assert _join_negative_coordinates(
        ["domain", "--point", "35.3,-97.5", "--hours", "6"]
    ) == ["domain", "--point", "35.3,-97.5", "--hours", "6"]
    # A following token that is not all-numeric stays an option string,
    # so `--point --help` still errors the way it should.
    assert _join_negative_coordinates(
        ["domain", "--point", "--help"]
    ) == ["domain", "--point", "--help"]

    sydney = tmp_path / "spaced"
    sydney.mkdir()
    out = sydney / "area.toml"
    rc = cli_main(["domain", "--point", "-33.87,151.21", "--card", "24gb",
                   "--ladder", "12", "--source", "gfs",
                   "--cycle", "2026-07-29T18", "--out", str(out)])
    assert rc == 0, capsys.readouterr()
    raw = tomllib.loads(out.read_text(encoding="utf-8"))
    assert raw["projection"]["ref_lat"] == pytest.approx(-33.87)
    assert raw["projection"]["map_proj"] == "lambert"
    # Hemisphere-correct: southern standard parallels.
    assert raw["projection"]["truelat1"] < 0.0
    assert raw["projection"]["truelat2"] < 0.0
    # And the printed next: line is pasteable -- negative area in = form.
    printed = capsys.readouterr().out
    assert "--area=-" in printed


def test_point_refusal_names_the_equals_form(tmp_path, capsys):
    rc, _ = _run_wizard(tmp_path, point="35.3")
    assert rc == 2
    err = capsys.readouterr().err
    assert "--point=-33.87,151.21" in err


def test_cycle_latest_resolves_instead_of_contradicting_itself(
        tmp_path, capsys, monkeypatch):
    """v1.0.0: "--cycle 'latest' must be YYYY-MM-DDTHH (UTC) or 'latest'".

    Worse than self-contradictory: the documented order is
    wizard-then-fetch, so nothing told a user which cycle was current
    and they had to run a throwaway fetch to find out.  The resolver
    already existed.
    """
    import gpuwm.fetch as fetch_module
    from datetime import datetime

    resolved = datetime(2026, 7, 29, 18)
    calls = []

    def fake_resolve(source, last_hour, **kwargs):
        calls.append((source, last_hour))
        return resolved

    monkeypatch.setattr(fetch_module, "resolve_latest_cycle", fake_resolve)
    rc, out = _run_wizard(tmp_path, source="gfs", cycle="latest",
                          ladder="12")
    printed = capsys.readouterr().out
    assert rc == 0, printed
    assert calls == [("gfs", 6)]
    assert "--cycle latest resolved to 2026-07-29T18Z" in printed
    raw = tomllib.loads(out.read_text(encoding="utf-8"))
    # The emitted config records the resolved time, never the query.
    assert raw["experiment"]["start_time"] == resolved
    assert raw["fetch"]["cycle"] == "2026-07-29T18"
    assert "--cycle 2026-07-29T18" in printed


def test_cycle_latest_is_refused_for_era5_with_a_reason(tmp_path, capsys):
    rc, _ = _run_wizard(tmp_path, source="era5", cycle="latest")
    _assert_refused(capsys, "reanalysis with weeks of latency", rc)


def test_tropical_points_get_the_halved_root_clock(tmp_path, capsys):
    """|lat| < 25 emits 2.5 s/km, with the reason in the file.

    A 12 km Mercator domain at Manila on the wizard's own 60 s clock
    destabilised at +1 h; the same domain at a shorter step completed
    6 h.  The emitted rationale names the co-located v1.1 CFL gate.
    """
    from gpuwm.domain_wizard import (ROOT_TIME_STEP_S,
                                     TROPICAL_ROOT_TIME_STEP_S,
                                     root_time_step_s)

    assert TROPICAL_ROOT_TIME_STEP_S * 2 == ROOT_TIME_STEP_S
    assert root_time_step_s(14.6) == TROPICAL_ROOT_TIME_STEP_S
    assert root_time_step_s(-14.6) == TROPICAL_ROOT_TIME_STEP_S
    assert root_time_step_s(24.99) == TROPICAL_ROOT_TIME_STEP_S
    assert root_time_step_s(25.0) == ROOT_TIME_STEP_S
    assert root_time_step_s(-33.87) == ROOT_TIME_STEP_S

    rc, out = _run_wizard(tmp_path / "manila", point="14.6,120.98",
                          source="gfs", cycle="2026-07-29T18")
    assert rc == 0, capsys.readouterr()
    text = out.read_text(encoding="utf-8")
    raw = tomllib.loads(text)
    assert raw["domain"][0]["time_step"] == TROPICAL_ROOT_TIME_STEP_S
    assert "TROPICAL CLOCK" in text
    assert "co-located vertical" in text
    # The chain still derives exactly, and the config still loads.
    exp = experiment_from_text(text, source=str(out))
    assert exp.root.time_step == TROPICAL_ROOT_TIME_STEP_S
    assert float(exp.dt_exact(2)) == TROPICAL_ROOT_TIME_STEP_S / 4

    # A mid-latitude point is untouched by all of this.
    rc, out = _run_wizard(tmp_path / "kansas", point="39.7,-96.6",
                          source="gfs", cycle="2026-07-29T18")
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert tomllib.loads(text)["domain"][0]["time_step"] == ROOT_TIME_STEP_S
    assert "TROPICAL CLOCK" not in text


def test_polar_fetch_box_stays_clear_of_the_pole(tmp_path, capsys):
    """PP-11: the wizard suggested `--area ...,90.00,...` for Tromso.

    The README refuses domains touching a pole; suggesting a forcing box
    whose top edge IS the pole, with no comment, contradicts it -- and
    `gpuwm fetch` accepted the box and downloaded 89 MB.
    """
    from gpuwm.domain_wizard import MAX_FETCH_ABS_LAT, POLE_CLEARANCE_DEG

    assert 0.0 < POLE_CLEARANCE_DEG < 1.0
    assert MAX_FETCH_ABS_LAT == pytest.approx(90.0 - POLE_CLEARANCE_DEG)

    # A 32 GiB card, so the footprint is large enough to reach the pole
    # in the first place -- on the small tiers the fitted domain now stops
    # short of it and the clamp never fires, which proves nothing.
    rc, out = _run_wizard(tmp_path, point="69.65,18.96", source="gfs",
                          cycle="2026-07-29T18", card="32gb")
    captured = capsys.readouterr()
    printed = captured.out
    assert rc == 0, printed
    raw = tomllib.loads(out.read_text(encoding="utf-8"))
    south, west, north, east = (
        float(v) for v in raw["fetch"]["area"].split(","))
    assert abs(north) <= MAX_FETCH_ABS_LAT + 1e-9
    assert abs(south) <= MAX_FETCH_ABS_LAT + 1e-9
    assert north < 90.0 and south > -90.0
    # This particular point clamps, so it must say so rather than
    # silently handing over a box it refuses elsewhere.  The clamp is
    # a one-line warning now (stderr), with the mechanism on --explain.
    assert "clear of the pole" in captured.err
    from gpuwm.fetch import parse_area
    parse_area(raw["fetch"]["area"])  # still a valid fetch box


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
    vram = CARD_VRAM_GIB[card]
    estimate = estimate_experiment(exp, forcing_interval_seconds=21600.0,
                                   vram_gib=vram)
    envelope = estimate.peak_envelope_bytes
    # The budget is the CANDIDATE's own -- its suite's reserve out of the
    # free VRAM a card that size really presents, not the nameplate.
    free_bytes = int(card_assumed_free_gib(vram) * GIB)
    budget = sizing_budget_bytes(exp, free_bytes=free_bytes, vram_gib=vram,
                                 forcing_interval_seconds=21600.0)
    assert envelope <= budget
    # ...and it must stop SHORT of it.  Every ladder v1.4.0 emitted landed
    # 0.01-0.19 GiB from the wall, which is a rounding error away from a
    # refusal on the machine that then runs it.
    assert envelope <= budget - fit_headroom_bytes(budget), (
        "the fit loop must leave headroom, not touch the budget")
    # The bisection must still actually spend the budget, not stop at the
    # floor: an envelope model is not licence to size timidly.
    assert envelope >= 0.7 * budget
    # Certified clock/dx conventions on the emitted chain.
    root = exp.root
    assert root.time_step == 60 and root.run.dx == 12000.0
    for dc in exp.domains:
        assert exp.dx_exact(dc.grid_id) == exp.dx_exact(1) / np.prod(
            [d.parent_grid_ratio for d in exp.domains
             if 1 < d.grid_id <= dc.grid_id], dtype=int)


def test_the_wizard_budgets_with_the_platform_envelope_factor(
        tmp_path, capsys, monkeypatch):
    """Same card, same ladder: Linux gets a bigger grid than Windows.

    The 1.75 envelope models WDDM.  Two instrumented Linux 4090 pilots
    (node 1: 19.80 GiB predicted vs 8.32 GiB actual; node 2: 19.94 vs
    9.0) showed the peak landing at or below the raw footprint there, so
    the wizard must price Linux at the measured-preliminary 1.15 and say
    which factor it applied.
    """
    import gpuwm.core.preflight as pf

    monkeypatch.setattr(pf.sys, "platform", "win32")
    rc, out = _run_wizard(tmp_path / "win", "--explain", card="24gb")
    windows_out = capsys.readouterr().out
    assert rc == 0
    windows_exp = experiment_from_text(
        out.read_text(encoding="utf-8"), source=str(out))
    assert "peak envelope" in windows_out
    assert "envelope basis: windows;" in windows_out

    monkeypatch.setattr(pf.sys, "platform", "linux")
    rc, out = _run_wizard(tmp_path / "lin", "--explain", card="24gb")
    linux_out = capsys.readouterr().out
    assert rc == 0
    linux_exp = experiment_from_text(
        out.read_text(encoding="utf-8"), source=str(out))
    assert "local-memory backing store" in linux_out
    assert "envelope basis: linux;" in linux_out

    windows_cells = windows_exp.root.run.nx * windows_exp.root.run.ny
    linux_cells = linux_exp.root.run.nx * linux_exp.root.run.ny
    assert linux_cells > windows_cells

    # Each still fits its own platform's budget, measured by the
    # estimator -- which now itemizes differently per platform too, so
    # each config has to be re-priced under the platform that built it.
    vram = CARD_VRAM_GIB["24gb"]
    free_bytes = int(card_assumed_free_gib(vram) * GIB)
    for exp, platform in ((windows_exp, "win32"), (linux_exp, "linux")):
        monkeypatch.setattr(pf.sys, "platform", platform)
        estimate = estimate_experiment(exp, forcing_interval_seconds=21600.0,
                                       vram_gib=vram)
        budget = sizing_budget_bytes(
            exp, free_bytes=free_bytes, vram_gib=vram,
            forcing_interval_seconds=21600.0)
        assert estimate.peak_envelope_bytes <= budget, platform
        assert estimate.peak_envelope_bytes >= 0.7 * budget, platform


@pytest.mark.parametrize("ladder", sorted(LADDER_RATIOS))
def test_windows_12gib_is_an_experimental_tier_not_a_refusal(
        tmp_path, capsys, monkeypatch, ladder):
    """A 12 GiB Windows card sizes, and says exactly what it is doing.

    The refusal it replaces was not "your card is too small": at the
    minimum layout, 4.12 GiB of the 5.4 GiB projection was calibration
    constants measured on a 32 GiB 5090 running campaign-scale work.  A
    reduced fixed reserve and the Linux envelope let the card be sized;
    the pioneer warning is the price, and the calibration ask is how it
    stops being experimental.
    """
    import gpuwm.core.preflight as pf

    monkeypatch.setattr(pf.sys, "platform", "win32")
    rc, out = _run_wizard(tmp_path / ladder, "--explain", card="12gb",
                          ladder=ladder)
    assert rc == 0, ladder
    printed = capsys.readouterr().out

    assert "peak envelope" in printed
    assert "envelope basis: windows-small;" in printed
    # The honest pioneer warning, in full.
    assert "EXPERIMENTAL: 12 GiB is at or below the 12 GiB Windows " \
           "small-card threshold" in printed
    assert "calibrated from ONE much larger machine" in printed
    assert "reduced 1.5 GiB reserve" in printed
    assert "Worst case is paging (slow) or a clean out-of-memory failure" \
           in printed
    assert "Neither corrupts a forecast" in printed
    assert "Please report your measured peak" in printed
    assert "gpuwm check <config>" in printed

    # And the emitted config really fits the experimental accounting.
    exp = experiment_from_text(out.read_text(encoding="utf-8"),
                               source=str(out))
    estimate = estimate_experiment(exp, forcing_interval_seconds=21600.0,
                                   vram_gib=12.0)
    budget = sizing_budget_bytes(
        exp, free_bytes=int(card_assumed_free_gib(12.0) * GIB),
        vram_gib=12.0, forcing_interval_seconds=21600.0)
    assert estimate.peak_envelope_bytes <= budget


def test_windows_16gib_and_up_keep_the_conservative_accounting(
        tmp_path, capsys, monkeypatch):
    """The experimental tier is bounded to small cards, by size only."""
    import gpuwm.core.preflight as pf

    monkeypatch.setattr(pf.sys, "platform", "win32")
    rc, _ = _run_wizard(tmp_path / "sixteen", "--explain", card="16gb")
    assert rc == 0
    printed = capsys.readouterr().out
    assert "peak envelope" in printed
    assert "envelope basis: windows;" in printed
    assert "EXPERIMENTAL" not in printed

    # The accounting seam itself, without going through the wizard: the
    # 5090-derived pool constants come back the moment the card is big
    # enough to afford them, and stay for a caller that names no card
    # (`gpuwm check`, which measures free VRAM, not capacity).
    assert pf.envelope_platform("win32", 12.0) == "windows-small"
    assert pf.envelope_platform("win32", 11.0) == "windows-small"
    assert pf.envelope_platform("win32", 16.0) == "windows"
    assert pf.envelope_platform("win32", None) == "windows"
    assert pf.envelope_platform("linux", 12.0) == "linux"
    assert pf.peak_envelope_factor("win32", 12.0) == 1.45
    assert pf.peak_envelope_factor("win32", 16.0) == 1.75
    assert pf.platform_projection_constants("win32", 12.0) == (
        0, pf.WINDOWS_SMALL_CARD_RESERVE_BYTES)
    assert pf.platform_projection_constants("win32", 16.0) == (
        pf.pool_retention_residual_bytes(), pf.PROBE_DEVICE_OVERHEAD_BYTES)
    assert pf.platform_projection_constants("win32", None) == (
        pf.pool_retention_residual_bytes(), pf.PROBE_DEVICE_OVERHEAD_BYTES)


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

    rc, out = _run_wizard(tmp_path, "--explain")
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
    # The wizard now emits the registry's OWN lw/sw pair rather than the
    # legacy combined selector.  Both resolve to (4, 4) = RTE+RRTMGP, but
    # only the pair form can compare equal to the shipped runner
    # profiles, which are written the same way -- v1.0.0's combined form
    # made every wizard config fail the prepared-forecast profile guard.
    from gpuwm.config import radiation_scheme_ids
    assert selectors.pop("ra_lw_physics") == root.ra_lw_physics == 4
    assert selectors.pop("ra_sw_physics") == root.ra_sw_physics == 4
    assert root.ra_physics == 0
    assert radiation_scheme_ids(root) == (4, 4)
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
    # Forcing is not on disk yet.  By default the deferral is not a
    # stanza of its own -- it is step 2 of the next-steps block, with
    # the exact follow-up command and the note that it waits on step 1.
    assert "gpuwm check: deferred" not in printed
    assert "gpuwm check" in printed and "--budget-gib" in printed
    assert "after the fetch lands" in printed

    # --explain restores the inventory and the geog story, verbatim.
    rc, _ = _run_wizard(tmp_path, "--explain", "--geog-root", str(geog),
                        ladder="12-3-1")
    assert rc == 0
    explained = capsys.readouterr().out
    assert "gpuwm check: deferred" in explained
    assert "WPS_GEOG" in explained

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
    """Warn-not-block: the user's Vtable is kept (never overwritten),
    the wizard says so in one line, and the emission succeeds."""

    marker = "not the packaged table"
    (tmp_path / "Vtable.ERA5_CDO").write_text(marker)
    rc, out = _run_wizard(tmp_path)
    captured = capsys.readouterr()
    assert rc == 0
    assert out.exists()
    assert "warning:" in captured.err
    assert "kept your existing Vtable.ERA5_CDO" in captured.err
    # The refusal is gone; the protection is not.
    assert (tmp_path / "Vtable.ERA5_CDO").read_text() == marker


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


# ---------------------------------------------------------------------------
# Custom ladders: --root-dx / --chain alongside the presets.
# ---------------------------------------------------------------------------

def test_chain_and_root_dx_parsing_and_refusals(capsys):
    from gpuwm.domain_wizard import (MAX_CHAIN_DEPTH, MAX_CHAIN_RATIO,
                                     MAX_ROOT_DX_KM, MIN_CHAIN_RATIO,
                                     MIN_ROOT_DX_KM, ROOT_DX_M,
                                     parse_chain, parse_custom_ladder)

    assert parse_chain("4,3,3") == (4, 3, 3)
    assert parse_chain(" 4 , 3 ") == (4, 3)
    assert parse_chain("") == ()
    # KEEP-HARD: an unparseable ratio and a non-refinement stay refusals.
    with pytest.raises(ValueError, match="not an integer"):
        parse_chain("4,3.5")
    with pytest.raises(ValueError, match="not a refinement"):
        parse_chain(str(MIN_CHAIN_RATIO - 1))
    # Warn-not-block: the conservative upper bounds report and continue.
    capsys.readouterr()
    assert parse_chain(str(MAX_CHAIN_RATIO + 1)) == (MAX_CHAIN_RATIO + 1,)
    err = capsys.readouterr().err
    assert "warning:" in err and "exceeds the blessed maximum" in err
    deep = tuple([2] * (MAX_CHAIN_DEPTH + 1))
    assert parse_chain(",".join(map(str, deep))) == deep
    err = capsys.readouterr().err
    assert "warning:" in err and "nests" in err

    # A preset run stays a preset run.
    assert parse_custom_ladder(
        root_dx_km=None, chain=None, ladder="auto") is None
    assert parse_custom_ladder(
        root_dx_km=None, chain=None, ladder="12-3") is None
    # Either flag alone switches to the custom form.
    assert parse_custom_ladder(
        root_dx_km=3.0, chain=None, ladder="auto") == (3000.0, ())
    assert parse_custom_ladder(
        root_dx_km=None, chain="4", ladder="auto") == (ROOT_DX_M, (4,))
    with pytest.raises(ValueError, match="cannot be combined"):
        parse_custom_ladder(root_dx_km=3.0, chain="4", ladder="12-3")
    # Warn-not-block: the km-typo window warns and continues; a
    # non-positive spacing stays a refusal.
    capsys.readouterr()
    for odd in (MIN_ROOT_DX_KM / 2, MAX_ROOT_DX_KM * 2):
        root_m, _ = parse_custom_ladder(
            root_dx_km=odd, chain="4", ladder="auto")
        assert root_m == odd * 1000.0
        err = capsys.readouterr().err
        assert "warning:" in err and "--root-dx" in err
    with pytest.raises(ValueError, match="positive spacing"):
        parse_custom_ladder(root_dx_km=-1.0, chain="4", ladder="auto")


def test_custom_ladder_3km_to_750m_emits_and_checks(tmp_path, capsys):
    """Drew's r4 case: an arbitrary root dx with an integer ratio.

    Validated by the same estimator fit loop, the same experiment
    loader, and the same `gpuwm check` the presets go through.
    """
    out = tmp_path / "r4.toml"
    rc = cli_main(["domain", "--point=35.3,-97.5", "--card", "24gb",
                   "--root-dx", "3", "--chain", "4", "--source", "gfs",
                   "--cycle", "2026-07-29T18", "--hours", "6",
                   "--out", str(out)])
    printed = capsys.readouterr().out
    assert rc == 0, printed
    assert "gpuwm check: PASS (rc 0)" in printed

    text = out.read_text(encoding="utf-8")
    exp = experiment_from_text(text, source=str(out))
    assert [float(exp.dx_exact(d.grid_id)) for d in exp.domains] == [
        3000.0, 750.0]
    # 5 s/km at 35 N: 15 s root, exactly quartered on the nest.
    assert exp.root.time_step == 15
    assert float(exp.dt_exact(2)) == 15 / 4
    # The companion namelist.wps agrees through the real grid builders.
    wps = out.parent / "r4.namelist.wps"
    assert " dx = 3000," in wps.read_text(encoding="utf-8")
    from_wps = grids_from_wps_namelist(wps)
    from_toml = grids_from_projection_config(load_experiment(out))
    assert len(from_wps) == len(from_toml) == 2
    for a, b in zip(from_wps, from_toml):
        lat_a, lon_a = a.latlon_mass()
        lat_b, lon_b = b.latlon_mass()
        assert a.dx == b.dx
        np.testing.assert_allclose(lat_a, lat_b, rtol=0, atol=1e-9)
        np.testing.assert_allclose(lon_a, lon_b, rtol=0, atol=1e-9)


@pytest.mark.parametrize("ladder_flags", [
    ("--root-dx", "3", "--chain", "4"),
    ("--ladder", "12-3-1"),
])
def test_every_domain_inherits_the_profiles_epssm(tmp_path, capsys,
                                                  ladder_flags):
    """The nest gets the profile's epssm, not WRF's Registry default.

    The regression this pins killed a reported nested forecast: the
    wizard wrote ``epssm = 0.1`` on every nest while the root took 0.5
    from the physics profile, stripping the vertical-acoustic
    off-centering exactly where nest terrain is steepest.  A 3 km ->
    750 m ladder over the Cascades grew w to non-finite in seven
    acoustic substeps at the child's steepest cell; the same geometry
    with the nest at the profile's 0.5 ran clean.

    Asserted against ``profile_switches`` rather than the number 0.5, so
    a profile that ships a different epssm still propagates to depth.
    """
    from gpuwm.domain_wizard import profile_switches

    out = tmp_path / "epssm.toml"
    rc = cli_main(["domain", "--point=46.9,-121.8", "--card", "24gb",
                   *ladder_flags, "--source", "gfs",
                   "--cycle", "2026-07-29T18", "--hours", "6",
                   "--out", str(out)])
    printed = capsys.readouterr().out
    assert rc == 0, printed

    exp = load_experiment(out)
    expected = float(profile_switches(None)["epssm"])
    assert len(exp.domains) > 1
    assert [float(d.run.epssm) for d in exp.domains] == (
        [expected] * len(exp.domains))
    # Stated per domain in the file the reader opens, so the value that
    # matters is visible where it is set -- and overridable there.
    text = out.read_text(encoding="utf-8")
    assert text.count("epssm = ") == len(exp.domains)
    assert "epssm = 0.1" not in text


def test_a_named_profiles_epssm_reaches_the_nest(tmp_path, capsys):
    """Same contract when --physics-profile names the suite."""
    from gpuwm.domain_wizard import profile_switches

    out = tmp_path / "epssm-profile.toml"
    assert cli_main([
        "domain", "--point=46.9,-121.8", "--card", "24gb",
        "--ladder", "12-3", "--source", "gfs", "--cycle",
        "2026-07-29T18", "--hours", "6", "--physics-profile",
        MORRISON_PROFILE_ID, "--out", str(out)]) == 0
    capsys.readouterr()
    exp = load_experiment(out)
    expected = float(profile_switches(MORRISON_PROFILE_ID)["epssm"])
    assert [float(d.run.epssm) for d in exp.domains] == [expected] * 2


def test_custom_ladder_deep_chain_reaches_the_hundred_metre_scale(
        tmp_path, capsys):
    out = tmp_path / "deep.toml"
    rc = cli_main(["domain", "--point=35.3,-97.5", "--card", "32gb",
                   "--root-dx", "3", "--chain", "3,3,3", "--source", "gfs",
                   "--cycle", "2026-07-29T18", "--hours", "3",
                   "--out", str(out)])
    printed = capsys.readouterr().out
    assert rc == 0, printed
    assert "gpuwm check: PASS (rc 0)" in printed

    exp = experiment_from_text(out.read_text(encoding="utf-8"),
                               source=str(out))
    dxs = [float(exp.dx_exact(d.grid_id)) for d in exp.domains]
    assert len(dxs) == 4
    assert dxs[0] == 3000.0
    assert dxs[-1] == pytest.approx(3000.0 / 27, rel=1e-12)
    assert 110.0 < dxs[-1] < 112.0
    # Exact rational clock all the way down: 15 s / 27.
    assert exp.root.time_step == 15
    assert exp.dt_exact(4) == Fraction(15, 27)


def test_the_gray_zone_advisory_warns_and_never_refuses(tmp_path, capsys):
    from gpuwm.domain_wizard import (GRAY_ZONE_DX_KM, _SHARED_CERTIFIED,
                                     gray_zone_advisory)

    # Above the gray zone: silent.
    assert gray_zone_advisory([12.0, 3.0, 1.0], _SHARED_CERTIFIED) == []
    # PBL scheme off: the overlap it warns about does not exist.
    assert gray_zone_advisory(
        [3.0, 0.75], {**_SHARED_CERTIFIED, "bl_pbl_physics": 0}) == []

    lines = gray_zone_advisory([3.0, 0.75, 0.25], _SHARED_CERTIFIED)
    assert len(lines) == 1, "one honest sentence, not a lecture"
    assert "GRAY ZONE" in lines[0]
    assert "2 domain(s)" in lines[0]
    assert "finest 250 m" in lines[0]
    assert "SASE" in lines[0]
    assert f"below {GRAY_ZONE_DX_KM:g} km" in lines[0]

    # End to end it is an advisory: rc 0, in the file and on stdout.
    out = tmp_path / "gray.toml"
    rc = cli_main(["domain", "--point=35.3,-97.5", "--card", "24gb",
                   "--root-dx", "3", "--chain", "4", "--source", "gfs",
                   "--cycle", "2026-07-29T18", "--hours", "6",
                   "--out", str(out)])
    printed = capsys.readouterr().out
    assert rc == 0
    assert "advisory: GRAY ZONE" in printed
    assert "finest 750 m" in printed
    assert "GRAY ZONE" in out.read_text(encoding="utf-8")


def test_the_deepest_preset_also_declares_its_gray_zone(tmp_path, capsys):
    """12-3-1-0.5 lands at 500 m; the advisory is not custom-only."""
    out = tmp_path / "preset.toml"
    rc = cli_main(["domain", "--point=35.3,-97.5", "--card", "32gb",
                   "--ladder", "12-3-1-0.5", "--source", "gfs",
                   "--cycle", "2026-07-29T18", "--hours", "3",
                   "--out", str(out)])
    printed = capsys.readouterr().out
    assert rc == 0, printed
    assert "advisory: GRAY ZONE" in printed
    assert "finest 500 m" in printed


def test_custom_root_dx_in_the_tropics_keeps_an_exact_half_second(
        tmp_path, capsys):
    out = tmp_path / "trop.toml"
    rc = cli_main(["domain", "--point=14.6,120.98", "--card", "24gb",
                   "--root-dx", "3", "--chain", "4", "--source", "gfs",
                   "--cycle", "2026-07-29T18", "--hours", "6",
                   "--out", str(out)])
    printed = capsys.readouterr().out
    assert rc == 0, printed
    text = out.read_text(encoding="utf-8")
    raw = tomllib.loads(text)
    # 2.5 s/km at 3 km = 7.5 s -> WRF's exact rational clock keys.
    assert raw["domain"][0]["time_step"] == 7
    assert raw["domain"][0]["time_step_fract_num"] == 1
    assert raw["domain"][0]["time_step_fract_den"] == 2
    exp = experiment_from_text(text, source=str(out))
    assert exp.dt_exact(1) == Fraction(15, 2)
    assert exp.dt_exact(2) == Fraction(15, 8)
    assert "TROPICAL CLOCK" in text


# ---------------------------------------------------------------------------
# The documented GFS -> GPU route: physics representation and honesty.
# ---------------------------------------------------------------------------

def test_emitted_radiation_uses_the_representation_the_guard_compares(
        tmp_path, capsys):
    """v1.0.0's wizard config could never pass the runner's guard.

    Every shipped profile writes radiation as the split pair
    (`ra_physics = 0` + `ra_lw_physics`/`ra_sw_physics`); the wizard
    wrote the legacy combined `ra_physics = 4`.  Both resolve to (4, 4),
    the guard even printed that both sides resolved to (4, 4), and it
    rejected them anyway because it compares the raw switch dicts.
    """
    from gpuwm.config import radiation_scheme_ids

    rc, out = _run_wizard(tmp_path, "--explain", ladder="12", source="gfs",
                          cycle="2026-07-29T18")
    capsys.readouterr()
    assert rc == 0
    raw = tomllib.loads(out.read_text(encoding="utf-8"))
    assert raw["shared"]["ra_physics"] == 0
    assert raw["shared"]["ra_lw_physics"] == 4
    assert raw["shared"]["ra_sw_physics"] == 4
    exp = experiment_from_text(out.read_text(encoding="utf-8"),
                               source=str(out))
    assert radiation_scheme_ids(exp.root.run) == (4, 4)


@pytest.mark.parametrize("profile", [
    "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1",
    "thompson-mp8-ysu-mm5-noah-validation-v1",
    "wsm6-ysu-mm5-noah-no-radiation-v1",
])
def test_physics_profile_configs_pass_the_runner_guard_as_emitted(
        tmp_path, capsys, profile):
    """--physics-profile emits a config the prepared runner accepts.

    Not "nearly accepts": this is the runner's own validator, run over
    the exact bytes the wizard wrote, with no hand edits -- the loop
    that cost node 2 three full 200 s front-door cycles to escape.
    """
    import tools.prepared_single_domain_forecast as runner

    out = tmp_path / f"{profile[:12]}.toml"
    rc = cli_main(["domain", "--point=35.3,-97.5", "--card", "24gb",
                   "--ladder", "12", "--source", "gfs",
                   "--physics-profile", profile, "--explain",
                   "--cycle", "2026-07-29T18", "--hours", "6",
                   "--out", str(out)])
    printed = capsys.readouterr().out
    assert rc == 0, printed
    assert "every runner enforces it switch for switch" in printed

    exp = load_experiment(out)
    validation = runner._validate_profile_switches(
        exp, source="gfs", profile=profile)
    assert validation["profile"] == profile
    # And the whole physics gate, not just the switch comparison.
    runner._validate_physics(exp, profile, exp.run_seconds,
                             float(exp.root.history_interval_s),
                             source="gfs")
    # The file states, in words, what it will actually run.
    text = out.read_text(encoding="utf-8")
    assert "# PHYSICS:" in text
    assert profile in text


def test_the_default_suite_states_its_verification_status_and_runs(
        tmp_path, capsys):
    """Converted (owner ruling 2026-07-31): status is stated, never a gate.

    The old contract here was "say out loud that the single door refuses
    this file".  The door no longer refuses any suite the engine
    implements, so what must be said out loud is the verification
    status -- one sentence on the default screen -- with the shipped
    profiles and what each ACTUALLY runs behind --explain.  A pilot once
    read the profiles' `ra_physics: 0` as "radiation off", so the words,
    never the raw switch, still carry the resolved behaviour.
    """
    from gpuwm.domain_wizard import (WIZARD_PHYSICS_PROFILES,
                                     physics_summary,
                                     prepared_route_physics_notice)

    rc, _ = _run_wizard(tmp_path, "--explain", ladder="12", source="gfs",
                        cycle="2026-07-29T18")
    printed = capsys.readouterr().out
    assert rc == 0
    # One clear sentence of status on the physics line...
    assert "supported, not yet WRF-verified" in printed
    # ...no refusal talk anywhere...
    assert "will refuse" not in printed
    assert "refuses it as emitted" not in printed
    # ...and the runs-as-written statement plus the profile catalog
    # under --explain.
    assert "runs as written on the prepared single-domain route" in printed
    for profile in WIZARD_PHYSICS_PROFILES:
        assert profile in printed

    # Words, never the raw switch: full-physics profiles must not read
    # as "off", and reduced ones must not read as full.
    full = physics_summary("morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1")
    assert "longwave RTE+RRTMGP, shortwave RTE+RRTMGP" in full
    assert "Kain-Fritsch cumulus" in full
    reduced = physics_summary("thompson-mp8-ysu-mm5-noah-validation-v1")
    assert "longwave OFF, shortwave Dudhia" in reduced
    assert "NO cumulus parameterization" in reduced

    # ERA5 does not go through that door, so it gets no such notice.
    assert prepared_route_physics_notice(None, "era5") == []


# ---------------------------------------------------------------------------
# Output layering: the wizard's default is one screen ending in the
# next-steps block, and --explain restores every word of the long form.
#
# The field exhibit this pins: a first-run ERA5 wizard printed 20 lines
# whose correct `gpuwm fetch` command sat at line 15, under a gray-zone
# advisory and above a nine-name dataset inventory, and the user's
# public verdict was "still can't get it working".  The commands were
# right; they were not findable.
# ---------------------------------------------------------------------------

#: What a first run may print before the reader has to scroll.  Not a
#: style preference: the exhibit's wall was 20 lines and the block that
#: matters is the last four, so the cap is what keeps the whole thing on
#: one screen alongside a shell prompt.
WIZARD_DEFAULT_LINE_CAP = 14


@pytest.mark.parametrize("explain", [False, True])
def test_wizard_output_is_layered_and_the_default_fits_a_screen(
        tmp_path, capsys, explain):
    """Terse by default, complete under --explain -- both, every time."""

    extra = ("--explain",) if explain else ()
    rc, out = _run_wizard(tmp_path, *extra, ladder="12-3-1-0.5",
                          card="24gb", source="era5")
    printed = capsys.readouterr().out
    assert rc == 0
    lines = printed.splitlines()

    if explain:
        # Every word that moved is back, in its original wording.
        assert "sizing (itemized preflight estimator, in-process):" in printed
        assert "peak envelope" in printed
        assert "envelope basis:" in printed
        assert "gpuwm check: deferred" in printed
        assert "static geography:" in printed
        assert "treat sub-kilometre PBL structure as indicative" in printed
    else:
        assert len(lines) <= WIZARD_DEFAULT_LINE_CAP, printed
        # The advisory still fires -- it is shortened, not dropped.
        assert "GRAY ZONE" in printed
        assert "treat sub-kilometre PBL structure as indicative" \
            not in printed
        # One sizing line, carrying the numbers that decide whether it runs.
        assert "sizing (itemized" not in printed
        assert "peak envelope" in printed and "headroom" in printed
        # And the way back to everything above.
        assert "--explain" in printed


def test_wizard_ends_with_three_numbered_commands_and_nothing_after(
        tmp_path, capsys):
    """The last thing on screen is the only thing asking for an action.

    Nothing prints after step 3.  The exhibit's failure was a correct
    next command with more output beneath it, which reads as "and then
    this happened" rather than "do this".
    """

    rc, out = _run_wizard(tmp_path, ladder="12-3", source="era5")
    printed = capsys.readouterr().out
    assert rc == 0
    lines = [line for line in printed.splitlines() if line.strip()]

    block = lines[lines.index("next:"):]
    # Exactly three numbered steps, in order, and nothing numbered four.
    numbered = [line for line in block if line.lstrip()[:2] in
                ("1.", "2.", "3.", "4.")]
    assert len(numbered) == 3
    assert numbered[0].startswith("  1. gpuwm fetch ")
    assert numbered[1].startswith("  2. gpuwm check ")
    assert numbered[2].startswith("  3. gpuwm run ")
    # Step 3 is the last line of output.  Anything after it competes
    # with the one thing the reader is being asked to do.
    assert lines[-1] == numbered[2]
    # Step 2 carries the deferral instead of a stanza of its own.
    assert "after the fetch lands" in numbered[1]


def test_the_era5_next_block_names_the_missing_cds_key_only_when_missing(
        tmp_path, capsys, monkeypatch):
    """A first-run pointer that is a pointer, and only when it is true.

    Presence of ``~/.cdsapirc`` is the one prerequisite of the ERA5
    route that lives entirely outside this project, and without it the
    failure arrives several commands later as a cdsapi exception.  A
    line that printed whether or not the key was there would be noise
    on every subsequent run, so it is gated on the file.
    """

    from gpuwm import fetch

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(fetch.Path, "home", staticmethod(lambda: home))

    rc, _ = _run_wizard(tmp_path / "no-key", source="era5")
    absent = capsys.readouterr().out
    assert rc == 0
    assert "Copernicus CDS key" in absent
    assert str(home / fetch.CDSAPIRC_NAME) in absent

    (home / fetch.CDSAPIRC_NAME).write_text("url: x\nkey: y\n")
    rc, _ = _run_wizard(tmp_path / "with-key", source="era5")
    present = capsys.readouterr().out
    assert rc == 0
    assert "Copernicus CDS key" not in present
    # The fetch step itself is unchanged either way.
    assert "1. gpuwm fetch --source era5" in present


def test_a_gfs_wizard_run_gets_no_era5_credential_line(tmp_path, capsys):
    """The pointer belongs to the route that needs it, and to no other."""

    rc, _ = _run_wizard(tmp_path, ladder="12", source="gfs",
                        cycle="2026-07-29T18")
    printed = capsys.readouterr().out
    assert rc == 0
    assert "Copernicus CDS key" not in printed
    assert "1. gpuwm fetch --source gfs" in printed


# ---------------------------------------------------------------------------
# The closing block names the route the emitted file is actually on
# ---------------------------------------------------------------------------

def _emit(tmp_path, capsys, *extra, source="gfs", name="area"):
    """Emit one config through the real CLI; return its printed output."""

    out = tmp_path / f"{name}.toml"
    rc = cli_main(["domain", "--point=35.3,-97.5", "--source", source,
                   "--cycle", "2026-07-29T18", "--hours", "6",
                   "--card", "12gb", "--out", str(out), *extra])
    assert rc == 0
    return out, capsys.readouterr().out


def test_a_gfs_emission_never_points_at_gpuwm_run(tmp_path, capsys):
    """The bug an owner hit on 1.3.0, in the shape it hit him.

    ``gpuwm run`` executes the ``[case_data]`` config-driven route,
    which is ERA5's; it refuses a GFS config by design and says so.  The
    closing block used to print it for every source, so following the
    numbered list to the end produced a refusal -- the tool telling its
    own user it was broken.  The block branches on the source now.
    """

    out, printed = _emit(
        tmp_path, capsys, "--ladder", "12", "--physics-profile",
        MORRISON_PROFILE_ID)
    block = printed.split("next:")[-1]
    assert "gpuwm run " not in block
    assert f"gpuwm go {_posix(out)}" in block


def test_the_gfs_emission_the_block_names_passes_gos_plan_reader(
        tmp_path, capsys):
    """What it names is not merely different -- it is accepted."""

    from gpuwm.go_cli import plan_from_config

    out, printed = _emit(
        tmp_path, capsys, "--ladder", "12", "--physics-profile",
        MORRISON_PROFILE_ID)
    assert "gpuwm go " in printed.split("next:")[-1]
    plan = plan_from_config(out)
    assert plan["profile"] == MORRISON_PROFILE_ID
    assert plan["source"] == "gfs"


def test_an_era5_emission_still_points_at_gpuwm_run(tmp_path, capsys):
    """ERA5 is the route `gpuwm run` exists for; nothing changed there."""

    out, printed = _emit(tmp_path, capsys, "--ladder", "12", source="era5")
    assert f"gpuwm run {_posix(out)}" in printed.split("next:")[-1]


def test_an_hrrr_emission_names_the_front_door_not_a_refusing_command(
        tmp_path, capsys):
    """HRRR reaches neither `gpuwm run` nor `gpuwm go`; name its own chain.

    It used to name a documentation section instead of a command, which
    left the reader to author four input files the wizard could have
    written.  It writes them now, so it names them.
    """

    out, printed = _emit(tmp_path, capsys, "--ladder", "12", source="hrrr",
                         name="hrrr-area")
    block = printed.split("next:")[-1]
    assert "gpuwm run " not in block
    assert "gpuwm go " not in block
    # Not rw-wps either: that is the GFS front door.
    assert "rw-wps \\" not in block
    assert "tools.prepare_hrrr_wrf" in block
    # And it must NOT name a runner that refuses HRRR by design:
    # prepared_single_domain_forecast's --source takes gfs/era5/20crv3,
    # and the tree runner requires two domains.  Printing either for a
    # single HRRR domain is the failure this whole block exists to end.
    assert "prepared_single_domain_forecast" not in block
    assert not [line for line in block.splitlines()
                if "prepared_domain_tree_forecast" in line
                and "#" not in line]
    assert "hrrr_single_domain_benchmark.py" in block
    paths = route_input_paths(out)
    assert all(path.exists() for path in paths.values())
    # The root preparation takes the target-domain document and the
    # native namelist; namelist.wps is a hierarchy input, so it is
    # written but not named by a single-domain chain.
    for key in ("target_domain", "namelist_input"):
        assert _posix(paths[key]) in block


def test_a_nested_hrrr_emission_drives_the_route_with_no_hand_edits(
        tmp_path, capsys):
    """The acceptance this lane exists for, run through the ROUTE's gates.

    ``gpuwm domain --source hrrr`` used to emit one file of the five the
    nested HRRR route consumes, and the namelist.wps it did emit was
    missing ``&share/interval_seconds`` -- the single key that route's
    first gate demands.  The gate run that proved the route worked at
    all had to author the rest with a lane's proof harness.

    So this test does not check that four files exist.  It runs the
    route's own validators, imported from the route, over the exact
    bytes the wizard wrote: the raw-WPS contract, the native/stock
    delta, the namelist import (which carries the Lambert contract
    check), the public hierarchy slice, and the root preparer's profile
    binding and vertical grid.  Every one of them is the thing that
    refused wizard output before.
    """
    from gpuwm.hrrr_hierarchy_direct import (
        _native_experiment, _require_raw_stock_delta,
        _require_raw_wps_contract, _supported_hierarchy_slice)
    from gpuwm.ingest.hrrr_target import (load_hrrr_target_domain,
                                          required_hrrr_source_window)
    from gpuwm.physics_compat import WSM6_PROFILE_ID
    from gpuwm.vertical_contract import explicit_vertical_from_wrf_namelist
    from tools.hrrr_single_domain_benchmark import (
        _validate_native_hrrr_physics_profile)

    out = tmp_path / "wind.toml"
    rc = cli_main(["domain", "--point=46.4,-118.3", "--card", "24gb",
                   "--root-dx", "3", "--chain", "4", "--source", "hrrr",
                   "--cycle", "2026-07-29T18", "--hours", "3",
                   "--out", str(out)])
    printed = capsys.readouterr().out
    assert rc == 0, printed

    paths = route_input_paths(out)
    exp = load_experiment(out)
    assert len(exp.domains) == 2

    # The route's raw contract gates, on the emitted bytes.
    assert _require_raw_wps_contract(
        paths["wps_namelist"], len(exp.domains))["status"] == "PASS"
    assert _require_raw_stock_delta(
        paths["namelist_input"],
        paths["stock_namelist_input"])["status"] == "PASS"

    # The target-domain document loads, and its HRRR source window fits.
    target = load_hrrr_target_domain(paths["target_domain"])
    assert (target.nx, target.ny, target.nz) == (
        exp.root.run.nx, exp.root.run.ny, exp.root.run.nz)
    required_hrrr_source_window(target)

    # The importer + Lambert contract check the hierarchy runs, and the
    # public slice gate, for this run's own forcing inventory.
    native_exp, _resolved, _report = _native_experiment(
        paths["wps_namelist"], paths["namelist_input"])
    hours = tuple(range(int(exp.run_seconds // 3600) + 1))
    _supported_hierarchy_slice(native_exp, target, forcing_hours=hours)

    # The namelist describes the same tree as the TOML beside it --
    # including the epssm that made this lane necessary.
    assert [(d.run.nx, d.run.ny, float(d.run.epssm))
            for d in native_exp.domains] == [
        (d.run.nx, d.run.ny, float(d.run.epssm)) for d in exp.domains]
    assert {float(d.run.epssm) for d in native_exp.domains} == {0.5}

    # And the ROOT preparation's own two gates over the same file.
    binding = _validate_native_hrrr_physics_profile(
        paths["namelist_input"], WSM6_PROFILE_ID)
    assert binding["profile"] == WSM6_PROFILE_ID
    vertical = explicit_vertical_from_wrf_namelist(
        paths["namelist_input"], expected_nz=target.nz,
        context="native HRRR initializer")
    assert vertical.eta_levels == exp.vertical.eta_levels
    assert vertical.p_top == exp.vertical.p_top


def test_a_nested_hrrr_next_block_names_hrrr_commands_not_the_gfs_door(
        tmp_path, capsys):
    """The multi-domain branch used to answer first, and said rw-wps.

    ``final_step_command`` tested ``domain_count > 1`` before it tested
    the source, so a multi-domain HRRR emission -- the only shape the
    nested route takes -- was told to prepare with the GFS front door.
    """
    out = tmp_path / "tree.toml"
    assert cli_main([
        "domain", "--point=46.4,-118.3", "--card", "24gb",
        "--root-dx", "3", "--chain", "4", "--source", "hrrr",
        "--cycle", "2026-07-29T18", "--hours", "3",
        "--out", str(out)]) == 0
    block = capsys.readouterr().out.split("next:")[-1]

    # rw-wps may be NAMED (the block says it is the wrong door); it must
    # not be the thing the reader is asked to run.
    assert not [line for line in block.splitlines()
                if "rw-wps" in line and "#" not in line]
    assert "gpuwm run " not in block and "gpuwm go " not in block
    assert "tools.prepare_hrrr_wrf" in block
    assert "gpuwm.hrrr_hierarchy_direct" in block
    assert "gpuwm.prepared_domain_tree_forecast" in block
    # Every file the chain names was written, and every value the
    # wizard knows is bound rather than left as a placeholder.
    for path in route_input_paths(out).values():
        assert path.exists()
        assert _posix(path) in block
    # Both stages get the CYCLE under one name.  They used to get one
    # `--valid-time` string that the preparer reads as the cycle and the
    # hierarchy reads as model time zero -- the same instant only at lead
    # zero, which is the only lead this door used to allow.
    assert block.count("--cycle 2026-07-29_18:00:00") == 2
    assert "--valid-time" not in block
    assert "--run-seconds 10800" in block


def test_a_nested_hrrr_chain_at_a_lead_hands_both_stages_the_same_two_values(
        tmp_path, capsys):
    """The lead is printed beside the cycle, on every stage of the chain.

    At lead 0 the cycle and model time zero are the same instant, so one
    string served both stages for four releases.  At lead 6 the printed
    chain has to say which is which, and both stages have to be able to
    derive the other.
    """
    out = tmp_path / "lead-tree.toml"
    assert cli_main([
        "domain", "--point=46.4,-118.3", "--card", "24gb",
        "--root-dx", "3", "--chain", "4", "--source", "hrrr",
        "--cycle", "2026-07-29T18", "--hours", "3",
        "--forecast-start-hour", "6", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    block = printed.split("next:")[-1]

    # The fetch downloads f06..f09, not f00..f03, and both preparation
    # stages carry the same cycle and the same lead.
    assert "gpuwm fetch --source hrrr --cycle 2026-07-29T18" in block
    assert block.count("--forecast-start-hour 6") == 3  # fetch + both stages
    assert block.count("--cycle 2026-07-29_18:00:00") == 2
    assert "--valid-time" not in block
    # The emitted config, and therefore the namelist the hierarchy reads
    # and compares against model time zero, start at cycle + 6 h.  Before
    # the lead was reachable here the namelist could only say the cycle
    # hour, which the nested route refuses at any nonzero lead.
    text = out.read_text(encoding="utf-8")
    assert "start_time = 2026-07-30T00:00:00" in text
    assert "forecast_start_hour = 6" in text
    namelist = route_input_paths(out)["namelist_input"].read_text(
        encoding="utf-8")
    start = {line.split("=")[0].strip(): line.split("=")[1].strip()
             for line in namelist.splitlines()
             if line.strip().startswith("start_")}
    assert start["start_day"] == "30, 30,"
    assert start["start_hour"] == "00, 00,"


def test_hrrr_sizing_respects_hrrrs_own_grid_not_only_the_card(
        tmp_path, capsys):
    """VRAM is not the only bound on how large an HRRR domain may be.

    HRRR's native grid is 1799 x 1059, and the interpolation stencil
    plus the surface-fallback halo need real source cells outside the
    target on every side.  A ladder sized purely against VRAM is a
    legal, well-sized experiment that no HRRR fetch can force: on a
    24 GiB Linux card, a 3 km root near the Washington/Oregon border
    ran its halo nine rows off the top of the HRRR grid, and the root
    preparation found out after the download.

    Sized against the SOURCE's own window function -- the one the root
    preparer calls -- the fit loop stops where HRRR does, and says so.
    """
    from gpuwm.hrrr_route_inputs import coverage_refusal, target_domain
    from gpuwm.ingest.hrrr_target import (HRRR_SOURCE_NY,
                                          required_hrrr_source_window)

    out = tmp_path / "edge.toml"
    rc = cli_main(["domain", "--point=46.35,-118.10", "--card", "32gb",
                   "--root-dx", "3", "--chain", "4", "--source", "hrrr",
                   "--cycle", "2026-07-29T18", "--hours", "1",
                   "--out", str(out)])
    printed = capsys.readouterr().out
    assert rc == 0, printed

    exp = load_experiment(out)
    assert coverage_refusal(exp) is None
    window = required_hrrr_source_window(
        target_domain(exp)).to_dict()["zero_based_inclusive"]
    assert window["j"][1] <= HRRR_SOURCE_NY - 1

    # It stopped AT the grid, with card headroom still unspent, and the
    # printed advisory says which of the two bounds it hit.
    assert window["j"][1] == HRRR_SOURCE_NY - 1
    assert "bounded by HRRR's own grid, not by your card" in printed


def test_a_point_hrrr_cannot_force_at_all_is_refused_by_the_fit_loop(
        tmp_path, capsys):
    """Watched firing: outside HRRR's coverage, nothing fits.

    Not a silent shrink to zero and not a refusal about memory -- the
    smallest layout this ladder has still needs source cells HRRR does
    not have, and the message says so and names the flag.
    """
    out = tmp_path / "atlantic.toml"
    rc = cli_main(["domain", "--point=25.0,-40.0", "--card", "12gb",
                   "--ladder", "12", "--source", "hrrr",
                   "--cycle", "2026-07-29T18", "--hours", "1",
                   "--out", str(out)])
    captured = capsys.readouterr()
    assert rc != 0
    message = captured.err + captured.out
    assert "cannot be forced by hrrr" in message
    assert "leaves HRRR coverage" in message
    assert "Move --point" in message
    assert not out.exists()


def test_a_route_incompatible_profile_is_refused_at_emission(
        tmp_path, capsys):
    """Named at emission, not after a fetch and a root preparation.

    The HRRR routes admit one physics slice.  A profile outside it used
    to be emitted happily and refused three stages later, by a gate
    naming a switch the wizard had already chosen.
    """
    out = tmp_path / "bad.toml"
    rc = cli_main(["domain", "--point=46.4,-118.3", "--card", "24gb",
                   "--ladder", "12-3", "--source", "hrrr",
                   "--cycle", "2026-07-29T18", "--hours", "3",
                   "--physics-profile", MORRISON_PROFILE_ID,
                   "--out", str(out)])
    captured = capsys.readouterr()
    assert rc != 0
    message = captured.err + captured.out
    assert "cannot drive the nested HRRR route" in message
    assert "cu_physics=1" in message
    assert "ra_sw_physics=4" in message
    assert "--physics-profile" in message
    # Watched firing: the wizard's own HRRR default is compatible, so
    # the refusal above is about the profile, not about HRRR.
    good = tmp_path / "good.toml"
    assert cli_main([
        "domain", "--point=46.4,-118.3", "--card", "24gb",
        "--ladder", "12-3", "--source", "hrrr", "--cycle",
        "2026-07-29T18", "--hours", "3", "--out", str(good)]) == 0
    capsys.readouterr()
    assert route_input_paths(good)["namelist_input"].exists()


def test_the_emitted_namelists_are_refused_if_they_drift_from_the_config(
        tmp_path):
    """The round trip that makes 'zero hand edits' mechanical.

    The route reads the namelists, not the TOML.  A set that describes a
    different tree than the config beside it is a defect that surfaces
    only after a fetch and a root preparation, so the writer re-imports
    what it wrote through the REAL importer and refuses on any
    difference.  Watched firing: one edited geometry key, and it fires.
    """
    from gpuwm.hrrr_route_inputs import verify_round_trip

    out = tmp_path / "drift.toml"
    assert cli_main([
        "domain", "--point=46.4,-118.3", "--card", "24gb",
        "--ladder", "12-3", "--source", "hrrr", "--cycle",
        "2026-07-29T18", "--hours", "3", "--out", str(out)]) == 0
    paths = route_input_paths(out)
    exp = load_experiment(out)

    # Unedited, the round trip is silent.
    verify_round_trip(exp, paths["wps_namelist"], paths["namelist_input"])

    # The edit is this lane's own bug, written into the namelist: the
    # nest back on WRF's Registry default while the config says 0.5.
    namelist = paths["namelist_input"]
    text = namelist.read_text(encoding="utf-8")
    edited = text.replace("0.5, 0.5,", "0.5, 0.1,", 1)
    assert edited != text
    namelist.write_text(edited, encoding="utf-8")
    with pytest.raises(HrrrRouteInputError, match="epssm"):
        verify_round_trip(exp, paths["wps_namelist"], namelist)


def test_a_tree_emission_names_the_tree_runner(tmp_path, capsys):
    """A ladder `gpuwm go` will not drive says so, with the runner.

    B-03: the runner ships as a console script and no message named it,
    so the only invocation offered was a `python -m` form carrying two
    bare `...` placeholders, followed by a docs path a pip install does
    not contain.  All three are pinned here.
    """

    _, printed = _emit(tmp_path, capsys, "--ladder", "12-3", "--name",
                       "treecase")
    block = printed.split("next:")[-1]
    assert "gpuwm run " not in block
    assert "gpuwm go " not in block
    # the installed console script, and the module form as an alternative
    assert "gpuwm-prepared-tree-forecast" in block
    assert "prepared_domain_tree_forecast" in block
    # no bare ellipsis standing in for a value the reader must supply
    assert " ... " not in block and block.count("...") == 0
    # every placeholder says what to put there
    assert "<the directory rw-wps wrote>" in block
    # and the pointer resolves without a checkout
    assert "https://" in block


def test_the_manual_chain_pointer_is_reachable_without_a_checkout():
    """A-10.  The wheel ships no docs tree, so a repo-relative path was
    the whole of an instruction the reader provably could not follow."""
    from gpuwm.go_cli import MANUAL_CHAIN

    assert "docs/public/FIRST-LIGHT.md" in MANUAL_CHAIN
    assert MANUAL_CHAIN.count("https://") == 1
    assert "FahrenheitResearch/arwen" in MANUAL_CHAIN


def test_a_profileless_gfs_emission_points_at_gpuwm_go(tmp_path, capsys):
    """Converted (owner ruling 2026-07-31): the chain runs the default
    suite as written, so a profileless single-domain GFS emission gets
    the same one-command next step a bound one does."""

    _, printed = _emit(tmp_path, capsys, "--ladder", "12")
    block = printed.split("next:")[-1]
    assert "gpuwm run " not in block
    assert "gpuwm go " in block


def _posix(path) -> str:
    return str(path).replace("\\", "/")


# ---------------------------------------------------------------------------
# The advisory for a domain sized to the card rather than to the weather
# ---------------------------------------------------------------------------

def test_a_card_filling_footprint_gets_an_advisory_not_a_refusal():
    """An owner's 32 GiB emission spanned 152 degrees of longitude.

    Legal arithmetic -- the sizer's job is to use the card it was given
    -- and an absurd first run.  This says so once, names the flag that
    makes it smaller, and refuses nothing.
    """

    from gpuwm.domain_wizard import oversized_footprint_advisory

    wide = oversized_footprint_advisory("-6.39,-159.63,73.19,-35.37")
    assert len(wide) == 1
    assert "--vram-gib" in wide[0]
    assert "124 x" in wide[0]
    # It names the flag that CAUSED the box as well as the one that
    # shrinks it, and says why narrowing the download alone is wrong.
    assert "--area" in wide[0]
    assert "starve" in wide[0]

    # A continental domain is not remarkable and gets no line.
    assert oversized_footprint_advisory("6.24,-135.55,63.02,-59.45") == []
    # Nor is a shape this function cannot read a reason to say anything.
    assert oversized_footprint_advisory("not-a-box") == []

    # A box that is merely TALL used to pass unremarked, because only
    # longitude was measured: the wheel user's Linux --card 24gb
    # --ladder 12 emitted 88 degrees of latitude.  This one is 84 tall
    # and 70 wide -- under the longitude bar, over the latitude one.
    tall = oversized_footprint_advisory("-8.00,-120.00,76.00,-50.00")
    assert len(tall) == 1
    assert "70 x 84" in tall[0]


def test_the_advisory_reaches_the_terminal_on_a_large_card(tmp_path,
                                                           capsys):
    """Emitted for real, at the size that provoked it."""

    out = tmp_path / "wide.toml"
    assert cli_main(["domain", "--point=35.3,-97.5", "--source", "gfs",
                     "--cycle", "2026-07-29T18", "--hours", "6",
                     "--ladder", "12", "--vram-gib", "32.00",
                     "--physics-profile", MORRISON_PROFILE_ID,
                     "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "advisory: this domain is sized to fill your card" in printed
    assert "--vram-gib N" in printed
    assert "gpuwm domain: FAIL" not in printed


# ---------------------------------------------------------------------------
# 2026-08-01 sizing calibration: the tier, the reserve and the two checks
# ---------------------------------------------------------------------------

#: Free VRAM real cards of each tier hand to a fresh CUDA context.
#:
#: The 16 GiB row is MEASURED: an idle headless RTX 4080 (16,376 MiB
#: physical) presents 15.33 GiB.  The others are the same 0.66 GiB
#: driver/nameplate gap applied to the tier's nominal size, which is the
#: assumption the tier has to be conservative against.
REAL_CARD_FREE_GIB = {"12gb": 11.34, "16gb": 15.33, "24gb": 23.33,
                      "32gb": 30.27}


@pytest.mark.parametrize("card", sorted(CARD_VRAM_GIB))
def test_the_card_tier_is_conservative_against_a_real_card(card):
    """A tier may never assume more free VRAM than its class delivers.

    The 16 GiB tier assumed the card would hand over its whole nominal
    size.  It does not -- a real RTX 4080 presents 15.33 GiB of a 15.99
    GiB card -- so every ladder the tier emitted was sized against VRAM
    that does not exist, landed 0.13-0.32 GiB over the real budget, and
    failed the product's own `gpuwm check` minutes after the wizard
    printed PASS.
    """
    assumed = card_assumed_free_gib(CARD_VRAM_GIB[card])
    assert assumed <= REAL_CARD_FREE_GIB[card], card
    # ...and not so conservative that the tier stops being useful.
    assert assumed >= REAL_CARD_FREE_GIB[card] - 1.0, card


@pytest.mark.parametrize("card", sorted(CARD_VRAM_GIB))
@pytest.mark.parametrize("ladder", ["12", "12-3", "12-3-1-0.5"])
def test_an_emitted_config_fits_the_card_it_was_sized_for(
        tmp_path, card, ladder):
    """The A-0 regression, as an inequality rather than a subprocess.

    Every `--card 16gb` ladder v1.4.0 emitted exceeded the budget a real
    16 GB card leaves.  Re-priced here against that card's real free
    VRAM and its own suite's reserve, with nothing declared and nothing
    added back.
    """
    rc, out = _run_wizard(tmp_path, card=card, ladder=ladder, source="gfs",
                          cycle="2026-07-28T00")
    assert rc == 0
    exp = experiment_from_text(out.read_text(encoding="utf-8"),
                               source=str(out))
    vram = CARD_VRAM_GIB[card]
    interval = 10800.0
    estimate = estimate_experiment(exp, forcing_interval_seconds=interval,
                                   vram_gib=vram)
    phases = estimate_phases(exp, source="gfs",
                             forcing_interval_seconds=interval,
                             vram_gib=vram)
    real_free = int(REAL_CARD_FREE_GIB[card] * GIB)
    budget = sizing_budget_bytes(exp, free_bytes=real_free, vram_gib=vram,
                                 forcing_interval_seconds=interval)
    assert phases.peak_envelope_bytes <= budget, (
        f"{card} {ladder}: emitted envelope "
        f"{phases.peak_envelope_bytes / GIB:.2f} GiB over a real budget of "
        f"{budget / GIB:.2f} GiB")
    assert estimate.alloc_estimate_bytes <= budget


NSSL2_PROFILES = (
    "nssl2-mp18-ysu-mm5-noah-kf-rte-rrtmgp-validation-candidate-v1",
    "nssl2-mp18-ysu-mm5-noah-kf-rrtmg-legacy-validation-candidate-v1",
)


@pytest.mark.parametrize("physics_profile", NSSL2_PROFILES)
@pytest.mark.parametrize("vram", [12.0, 15.0, 24.0, 32.0])
def test_a_suite_with_a_large_backing_store_still_sizes(
        tmp_path, physics_profile, vram):
    """Defect 4: the reserve's overhead term is SUITE-dependent.

    It tracks the local-memory backing store of the selected kernel set
    -- 1.93 GiB for WSM6+MYNN, 3.94 for NSSL2 double-moment -- while the
    fit loop assumed the documented flat 4.0.  Any suite whose overhead
    pushed the reserve past that flat figure was sized against one budget
    and verified against a smaller one, so BOTH NSSL2 profiles emitted a
    config that failed their own check at EVERY card size.  The loop
    prices the reserve from the candidate now, which is the same call
    check makes.
    """
    out = tmp_path / "n.toml"
    rc = cli_main([
        "domain", "--point=35.22,-97.44", "--vram-gib", str(vram),
        "--root-dx", "12", "--hours", "2", "--source", "gfs",
        "--cycle", "2026-07-28T00", "--physics-profile", physics_profile,
        "--out", str(out)])
    assert rc == 0, f"{physics_profile} at {vram} GiB emitted rc {rc}"

    exp = experiment_from_text(out.read_text(encoding="utf-8"),
                               source=str(out))
    interval = 10800.0
    phases = estimate_phases(exp, source="gfs",
                             forcing_interval_seconds=interval,
                             vram_gib=vram)
    # The budget the VERIFIER would use, on a card that really is this
    # size -- the number the fit loop has to have targeted.
    free_bytes = int(card_assumed_free_gib(vram) * GIB)
    budget = sizing_budget_bytes(exp, free_bytes=free_bytes, vram_gib=vram,
                                 forcing_interval_seconds=interval)
    assert phases.peak_envelope_bytes <= budget


def test_the_reserve_the_loop_targets_is_the_one_the_verifier_uses(tmp_path):
    """Two budgets in one command's output was the mechanism.

    The wizard printed "peak envelope 10.97 GiB of a 11.00 GiB budget"
    and then, four lines later, "EXCEEDS the 10.33 GiB budget by 0.64".
    One file, one invocation, two budgets.
    """
    from gpuwm.core.preflight import ReservePolicy, card_local_memory_profile

    rc, out = _run_wizard(tmp_path, card="16gb", ladder="12-3",
                          source="gfs", cycle="2026-07-28T00")
    assert rc == 0
    exp = experiment_from_text(out.read_text(encoding="utf-8"),
                               source=str(out))
    interval = 10800.0
    estimate = estimate_experiment(exp, forcing_interval_seconds=interval,
                                   vram_gib=16.0)
    verifier = ReservePolicy.n0_alloc(
        exp, profile=card_local_memory_profile(16.0),
        estimate_bytes=estimate.alloc_estimate_bytes)
    free_bytes = int(card_assumed_free_gib(16.0) * GIB)
    assert sizing_budget_bytes(
        exp, free_bytes=free_bytes, vram_gib=16.0,
        forcing_interval_seconds=interval) == (
            free_bytes - verifier.reserve_bytes)


def test_the_wizard_prints_the_bare_check_as_the_next_step(tmp_path, capsys):
    """Two documented commands, one file, one machine, opposite verdicts.

    `gpuwm check CONFIG` returned 4 while the wizard's own printed
    `gpuwm check CONFIG --budget-gib 12 --vram-gib 16` returned 0, because
    --budget-gib re-declared the free figure the tier had invented.  The
    bare form is the next step now, and the declared form is printed
    beside it saying what it is for.
    """
    rc, out = _run_wizard(tmp_path, card="16gb", ladder="12", source="gfs",
                          cycle="2026-07-28T00")
    assert rc == 0
    printed = capsys.readouterr().out
    assert f"2. gpuwm check {out}" in printed.replace("\\", "/") or (
        "2. gpuwm check" in printed)
    assert "that measures THIS machine's free VRAM" in printed
    assert "--budget-gib" in printed, "the declared form is still offered"


def test_the_minimum_layout_refusal_does_not_contradict_itself(
        tmp_path, capsys):
    """"the other 0.00 GiB (0% of the projection) is grid-independent
    calibration constants, so a smaller grid cannot help" -- if 0% is
    grid-independent then the grid is exactly what would help."""

    out = tmp_path / "tiny.toml"
    rc = cli_main(["domain", "--point=35.22,-97.44", "--vram-gib", "5",
                   "--ladder", "12-3-1-0.5", "--source", "gfs",
                   "--cycle", "2026-07-28T00", "--out", str(out)])
    assert rc == 2
    message = capsys.readouterr().err
    assert "minimum layout" in message
    assert "there is no smaller grid on this ladder" in message
    # The self-contradiction: a share and a claim that disagree.
    share = float(message.split("% of the envelope")[0].split("(")[-1])
    if share < 25.0:
        assert "so a smaller grid cannot help" not in message
    assert not out.exists(), "a refusal writes no file"


@pytest.mark.parametrize("spelling", ["nan", "inf", "-inf"])
def test_non_finite_vram_is_refused_without_inventing_a_capacity(
        tmp_path, capsys, spelling):
    """E-10.  The finite CHECK existed; the MESSAGE did not respect it.

    It shared a sentence with the too-small-card branch, which describes
    the card it was given -- and card_assumed_free_gib launders a
    non-finite value through max(), which returns the finite operand.  So
    `--vram-gib nan` was reported as a card presenting "about 0.00 GiB
    free": a specific, false, plausible number invented for an input that
    names no quantity.
    """
    # =VALUE form: a leading "-" (-inf) must not be read as an option.
    rc, out = _run_wizard(tmp_path, f"--vram-gib={spelling}", card=None)
    assert rc == 2
    err = capsys.readouterr().err
    assert "is not a size" in err
    assert "0.00 GiB" not in err
    assert "Traceback" not in err
    assert not out.exists()


def test_help_names_the_profiles_a_route_cannot_prepare(capsys):
    """Advertised-and-impossible is worse than not advertised.

    `--help` listed eight `--physics-profile` values with no marker
    while two were refused unconditionally on `--source gfs` -- the
    DEFAULT source -- so a reader choosing from the list had a 1-in-4
    chance of picking one that could never work, and found out from the
    refusal.  Measured on the 1.4.1 build, RTX 4080: those two are rc 2
    at every one of the four card tiers.

    The refusal is not what is wrong with that and is not touched here.
    """
    from gpuwm.domain_wizard import (WIZARD_PHYSICS_PROFILES,
                                     _profile_help_route_note,
                                     profile_route_blocker,
                                     profiles_blocked_on_source)

    blocked = profiles_blocked_on_source("gfs")
    # Non-vacuous, and the reason is a real registry refusal rather than
    # this test's opinion of it.
    assert blocked, "nothing to advertise a caveat about"
    for profile in blocked:
        assert "ruc" in profile
        assert "ruc-lsm" in profile_route_blocker(profile, "gfs")

    note = _profile_help_route_note()
    for profile in blocked:
        assert profile in note
    # And nothing that DOES run is listed as if it did not.
    for profile in WIZARD_PHYSICS_PROFILES:
        if profile not in blocked:
            assert profile not in note

    with pytest.raises(SystemExit):
        cli_main(["domain", "--help"])
    printed = capsys.readouterr().out
    # argparse rewraps, so compare on the unwrapped text.
    flat = " ".join(printed.split())
    assert "NOT every profile runs on every route" in flat
    for profile in blocked:
        assert profile in flat


def test_the_help_caveat_is_derived_not_listed(monkeypatch):
    """A hard-coded pair would go stale in the direction that lies.

    If a route regains the component, the caveat has to disappear on its
    own -- otherwise `--help` starts refusing a pairing that works.
    """
    from gpuwm import domain_wizard

    monkeypatch.setattr(domain_wizard, "profile_route_blocker",
                        lambda profile, source: None)
    assert domain_wizard.profiles_blocked_on_source("gfs") == ()
    assert domain_wizard._profile_help_route_note() == ""
