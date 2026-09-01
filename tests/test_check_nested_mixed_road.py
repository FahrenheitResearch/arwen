"""``gpuwm check`` prices the road the run door actually takes on a tree.

THE DEFECT, reproduced on the published 2.5.8 and 2.6.0 wheels and again
on this line's pre-merge tip.  On a NESTED tree with ``[tiles]``, ``check``
priced only the fully-resident tree and refused it, while the run door's
own per-domain walk (:func:`gpuwm.core.streaming.steppers_for_tree`) took a
MIXED road -- child streamed, parent resident -- that fits the same card
and completes.  Measured on the tree below at ``--budget-gib 13``: the
report said "the forecast is the memory-binding phase at 17.76 GiB peak
envelope; that EXCEEDS the 15.87 GiB budget by 1.89 GiB", failed
``alloc_estimate_le_wddm_budget`` and exited 1 -- beside an advisory that
admitted in the same breath that "a refusal or a fit below describes the
resident tree, not the mixed-road one the run will take".

A report whose own advisory says it priced a run nobody asked for, and
whose exit code refuses that run anyway, teaches the reader that streaming
has no point.  The same walk now prices the report: 15.04 GiB against the
15.87 GiB budget, exit 0, with the per-domain plan printed.

THE CONTROL IS THE OTHER HALF.  A tree that fits NO road still refuses --
the fix is a different question being asked, not a gate being loosened.
"""
from __future__ import annotations

import json
import textwrap

import pytest

from gpuwm import cli


GIB = 1024 ** 3

#: The declared card where the resident tree does not fit and the mixed
#: road does.  ``--budget-gib`` names an allocation budget; the envelope
#: budget the verdict compares is that plus the reserve less the
#: other-process margin, which is 15.87 GiB here.  Resident 17.76 GiB
#: refuses against it; the mixed road's 15.04 GiB radiation peak fits.
_FITS_MIXED_ROAD_GIB = "13"

#: One GiB down, where the mixed road does not fit either.  The tree is
#: unchanged: only the card moved, so the pair isolates the verdict.
_FITS_NO_ROAD_GIB = "12"


def _nested_auto_tiles(tmp_path, name="tree"):
    """A two-domain tree whose child cannot be held resident on that card.

    ``mode = "auto"`` over the tree rather than a pinned tiling, and that
    is load-bearing twice over.  ``mode = "on"`` over a nest is refused by
    the loader (every coupling edge would have both ends streamed), and a
    tiling PINNED on every enabled domain consults no planner and no card,
    so its claims carry neither the once-per-process floor nor the
    radiation reservation and cannot stand in the forecast term.  ``auto``
    is the shape a user writes and the one the walk prices.

    Written as a TOML and loaded through the product's own loader, for the
    reason ``tests/test_streamed_admission.py`` states: the finding is
    about the config a user actually types.
    """
    path = tmp_path / f"{name}.toml"
    path.write_text(textwrap.dedent("""\
        [experiment]
        name = "synth"
        start_time = 2024-05-03T12:00:00
        run_seconds = 3600.0
        restart_interval_s = 0.0

        [fetch]
        source = "gfs"
        cycle = "2024-05-03T12"
        hours = 1

        [shared]
        nz = 49
        ztop = 20000.0
        moist = true
        moist_cq = true
        mp_physics = 10
        ra_lw_physics = 4
        ra_sw_physics = 4
        sf_sfclay_physics = 91
        sf_surface_physics = 2
        bl_pbl_physics = 1
        cu_physics = 1
        nwp_diagnostics = 1

        [tiles]
        mode = "auto"

        [[domain]]
        grid_id = 1
        parent_id = 0
        i_parent_start = 1
        j_parent_start = 1
        parent_grid_ratio = 1
        parent_time_step_ratio = 1
        nx = 300
        ny = 300
        time_step = 20
        dx = 9000.0
        history_interval_s = 3600.0

        [[domain]]
        grid_id = 2
        parent_id = 1
        i_parent_start = 12
        j_parent_start = 12
        parent_grid_ratio = 3
        parent_time_step_ratio = 3
        nx = 600
        ny = 600
        history_interval_s = 3600.0
        """), encoding="utf-8")
    return path


def _check(capsys, config, *flags):
    code = cli.main(["check", str(config), *flags])
    captured = capsys.readouterr()
    return code, captured.out


def _plan_rows(out):
    """The per-domain rows of the printed plan, and nothing else.

    Taken as the BLOCK under the heading rather than by matching ``d0``
    anywhere: the resident itemization further down the report opens its
    lines the same way, and a looser filter reads those as plan rows.
    """
    rows = []
    inside = False
    for line in out.splitlines():
        if "MIXED-ROAD PLAN" in line:
            inside = True
            continue
        if inside:
            stripped = line.strip()
            if not stripped.startswith(("d0", "tree budget", "REFUSED:")):
                break
            if stripped.startswith("d0"):
                rows.append(stripped)
    return rows


def test_a_tree_that_only_fits_the_mixed_road_is_admitted(tmp_path, capsys):
    """THE REGRESSION.  Exit 0, on the road the run door takes.

    Asserted as a RELATION, not a pinned figure: the mixed road has to be
    genuinely cheaper than the resident tree it replaced, and the verdict
    has to be about the mixed road.  A gate that had merely stopped
    refusing everything would pass an exit-code-only assertion and still
    price the wrong run.
    """
    config = _nested_auto_tiles(tmp_path)
    code, out = _check(capsys, config, "--budget-gib", _FITS_MIXED_ROAD_GIB)

    assert code == 0, out
    # The verdict names the road it priced, and the resident tree it did
    # not: the two figures beside each other are what say streaming
    # bought something.
    assert "mixed-road forecast" in out
    assert "with the whole tree resident" in out
    # The advisory no longer disowns the numbers under it.
    assert "not the mixed-road one the run will take" not in out


def test_the_plan_names_each_domains_road_and_claim(tmp_path, capsys):
    """The report SHOWS the walk, per domain, default-on and unflagged.

    The exit code alone is not the fix.  A user whose tree was refused for
    a reason the report would not print had no way to see which domain
    claimed what, which is where the lever actually is.
    """
    config = _nested_auto_tiles(tmp_path)
    _code, out = _check(capsys, config, "--budget-gib", _FITS_MIXED_ROAD_GIB)

    assert "MIXED-ROAD PLAN" in out
    plan = _plan_rows(out)
    # Parent resident, child streamed -- the shape the coupler supports
    # and the one the resident-only pricing could not describe.
    assert any(line.startswith("d01 resident:") for line in plan), plan
    assert any(line.startswith("d02 streams (") for line in plan), plan
    # Every row carries its claim, and the streamed row its tiling.
    assert all("claim" in line for line in plan), plan
    assert any("buffer(s) of tile" in line for line in plan), plan
    # ...and the budget the claims were priced against, so a reader can do
    # the arithmetic the walk did.
    assert "tree budget" in out
    # The gate leg explains itself in terms that are TRUE of a mixed road.
    # "the resident domain is never allocated" is the single-domain
    # reason, and it is false about a tree whose parent the plan above
    # just showed the reader sitting resident on the card.
    assert "ALLOC GATE, MIXED ROAD" in out
    assert "the resident domain is never allocated" not in out


def test_a_tree_that_fits_no_road_still_refuses(tmp_path, capsys):
    """THE CONTROL, and the reason the regression test is not vacuous.

    One GiB less card, the same tree: the mixed road does not fit either
    and the door still refuses.  Without this pair, a check that had
    simply stopped refusing nested trees would pass the test above.
    """
    config = _nested_auto_tiles(tmp_path)
    code, out = _check(capsys, config, "--budget-gib", _FITS_NO_ROAD_GIB)

    assert code == 1, out
    assert "EXCEEDS" in out
    # Refused ON THE MIXED ROAD, not on the resident tree: a refusal that
    # priced the wrong run is the defect, whichever way it lands.
    assert "OVER BUDGET, MIXED ROAD" in out
    # And the plan is printed even in refusal -- that is the half the
    # reader needs in order to act on it.
    assert "MIXED-ROAD PLAN" in out


def test_the_refusal_and_the_admission_price_the_same_road(tmp_path, capsys):
    """One tree, two cards, one arithmetic.

    The peak the fitting card admits and the peak the smaller card
    refuses are the SAME number: only the budget moved.  This is what
    stops the fix from being "streamed when it suits us".
    """
    config = _nested_auto_tiles(tmp_path)
    _fit, fit_out = _check(capsys, config, "--json",
                           "--budget-gib", _FITS_MIXED_ROAD_GIB)
    _refuse, refuse_out = _check(capsys, config, "--json",
                                 "--budget-gib", _FITS_NO_ROAD_GIB)

    fits = json.loads(fit_out)
    refuses = json.loads(refuse_out)
    assert (fits["peak_envelope_bytes"]
            == refuses["peak_envelope_bytes"]), (
        "the road priced must not depend on whether it fits")
    assert fits["observed_peak_envelope_exceeds_budget"] is False
    assert refuses["observed_peak_envelope_exceeds_budget"] is True


def test_the_json_report_publishes_the_walk_as_data(tmp_path, capsys):
    """A machine reader gets the roads as fields, not as prose.

    ``gpuwm check --json`` is how another door asks this question, and
    parsing the plan back out of the advisory sentence is not an
    interface.
    """
    config = _nested_auto_tiles(tmp_path)
    _code, out = _check(capsys, config, "--json",
                        "--budget-gib", _FITS_MIXED_ROAD_GIB)
    payload = json.loads(out)

    road = payload["tree_road"]
    assert road is not None
    assert road["replaces_forecast_term"] is True
    assert road["priced"] is True and road["streams_any"] is True
    assert road["refusal"] is None
    roads = {int(row["grid_id"]): row["road"] for row in road["rows"]}
    assert roads == {1: "resident", 2: "streamed"}
    # The tile belongs to the DOMAIN, never to the tree: a mixed road has
    # no single tiling, and publishing one at the top level is how a
    # reader takes the child's tile for the whole run's.
    assert payload["streamed"]["road"] == "mixed (nested tree)"
    assert "tile_nx" not in payload["streamed"]
    assert payload["tree_road"]["rows"][1]["tile"]["tile_nx"] > 0
    # The figure every gate weighed is the walk's own peak.
    assert payload["peak_envelope_bytes"] == road["peak_vram_bytes"]
    # ...and it is genuinely cheaper than the tree it replaced.
    assert (road["peak_vram_bytes"]
            < payload["streamed"]["resident_forecast_envelope_bytes"])


def test_a_single_domain_config_is_not_given_a_tree_road(tmp_path, capsys):
    """THE NEGATIVE CONTROL.  The question does not arise, so it is not asked.

    A one-domain config has no mixed road, and the single-domain streamed
    envelope -- with its one real tiling -- is what its report must keep
    carrying.  This pins that the tree walk did not become a second answer
    for configs that already had one.
    """
    path = tmp_path / "single.toml"
    text = _nested_auto_tiles(tmp_path, name="src").read_text(encoding="utf-8")
    path.write_text(text.split("[[domain]]")[0]
                    + "[[domain]]" + text.split("[[domain]]")[1],
                    encoding="utf-8")

    _code, out = _check(capsys, path, "--json", "--budget-gib", "13")
    payload = json.loads(out)

    assert payload["tree_road"] is None
    assert "MIXED-ROAD PLAN" not in out
    if payload["streamed_forecast"]:
        # It streams as a single domain, so it has exactly one tiling and
        # says so under the name every existing reader uses.
        assert payload["streamed"]["road"] == "streamed (single domain)"
        assert payload["streamed"]["tile_nx"] > 0
