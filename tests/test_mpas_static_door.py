"""The static half of what ``gpuwm mesh`` delivers.

A grid file alone reaches no dycore. The mesh registry that admits a mesh
to a run pins BOTH the grid and a matching static by byte count and
sha256, so a door that emits only a grid emits something nothing accepts.
These tests hold the door to emitting the pair, hold the refusals to
naming that breakage, and bind the Python-side ABI literal to the one the
Rust binary actually prints.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import pytest

from gpuwm import bridge_assets, bridges, mpas_mesh, rustwx_static

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- wiring


def test_the_static_builder_has_a_bundled_artifact_row():
    """Without a row the wheel ships the generator and not its partner."""

    names = {a.name for a in bridge_assets.BUNDLED_ARTIFACTS}
    assert "rw_mpas_static" in names, (
        "rw_mpas_mesh ships but rw_mpas_static does not, so a user gets a "
        "grid file nothing will run")


def test_the_bundled_row_names_the_same_environment_variable_the_bridge_reads():
    row = next(a for a in bridge_assets.BUNDLED_ARTIFACTS
               if a.name == "rw_mpas_static")
    assert row.env_var == rustwx_static.STATIC_ENV


def test_the_static_builder_has_an_abi_marker():
    """A stale build must fail the handshake before the geography pass."""

    assert "rw_mpas_static" in bridges.BRIDGE_ABI_MARKERS


def test_the_abi_marker_is_a_prefix_of_the_literal_the_wrapper_expects():
    """The two spellings are one contract; a drift between them would let
    a binary pass the byte scan and then fail the launch probe."""

    marker = bridges.BRIDGE_ABI_MARKERS["rw_mpas_static"].decode()
    assert rustwx_static.STATIC_ABI_MARKER.startswith(marker), (
        f"bridges marker {marker!r} is not the head of the wrapper's "
        f"expected --abi output")


def test_every_mpas_binary_the_pair_needs_is_bundled():
    names = {a.name for a in bridge_assets.BUNDLED_ARTIFACTS}
    assert {"rw_mpas_mesh", "rw_mpas_static"} <= names


# --------------------------------------------------------------- naming


@pytest.mark.parametrize("grid,expected", [
    ("kansas.grid.nc", "kansas.static.nc"),
    ("mesh.nc", "mesh.static.nc"),
    ("plain", "plain.static.nc"),
    ("/tmp/a/b.grid.nc", "b.static.nc"),
])
def test_the_static_lands_beside_the_grid_under_a_matching_name(grid, expected):
    """They are admitted as a pair, so they are written as a pair."""

    assert mpas_mesh.default_static_path(Path(grid)).name == expected


def test_an_explicit_static_out_wins():
    """`--static-out` is taken as given; the default only fills a gap."""

    args = _mesh_args(static_out="/elsewhere/z.nc")
    assert args.static_out == Path("/elsewhere/z.nc")
    assert (mpas_mesh.default_static_path(Path("a.grid.nc"))
            != Path("/elsewhere/z.nc"))


# ------------------------------------------------------------- refusals


def test_a_missing_geography_archive_is_refused_by_naming_the_breakage():
    text = rustwx_static.geog_refusal(None)
    assert text is not None
    assert "reaches no dycore" in text
    assert "byte count and sha256" in text
    # A refusal that does not say what DOES work is a dead end.
    assert "--geog" in text and "--no-static" in text


def test_an_incomplete_geography_archive_names_the_missing_datasets(tmp_path):
    root = tmp_path / "WPS_GEOG"
    (root / "topo_gmted2010_30s").mkdir(parents=True)
    (root / "topo_gmted2010_30s" / "index").write_text("x", encoding="utf-8")
    text = rustwx_static.geog_refusal(root)
    assert text is not None
    assert "topo_gmted2010_30s" not in text.split("what does work")[0], (
        "the dataset that IS present is listed as missing")
    assert "soiltype_top_30s" in text
    assert "field of zeros" in text


def test_the_refusal_names_the_plain_fetch_now_that_it_covers_the_soil(
        tmp_path):
    """The remedy has to be the command that actually closes the gap.

    While the soil archive was an opt-in, closing this gap meant naming
    it: `--datasets soilgrids`.  It is a sanctioned default download now,
    so the bare command stages it, and a remedy that still leads with the
    flag teaches an operator a narrower command than the one that works.
    """

    root = tmp_path / "WPS_GEOG"
    root.mkdir(parents=True)
    text = rustwx_static.geog_refusal(root)
    assert text is not None
    works = text.split("what does work")[1]
    assert "gpuwm fetch-geog\n" in works or works.count(
        "gpuwm fetch-geog,") == 1
    assert "--datasets soilgrids" not in works, (
        "the bare fetch stages it; the flag is a narrower command")


def test_a_complete_geography_archive_is_not_refused(tmp_path):
    root = tmp_path / "WPS_GEOG"
    for name in rustwx_static.REQUIRED_GEOG_DATASETS:
        (root / name).mkdir(parents=True)
        (root / name / "index").write_text("x", encoding="utf-8")
    assert rustwx_static.geog_refusal(root) is None
    assert rustwx_static.missing_geog_datasets(root) == []


def test_an_environment_override_naming_a_missing_file_is_a_hard_error(
        tmp_path, monkeypatch):
    """Falling through to a different binary would build the static from
    geography the operator did not choose, and nothing downstream would
    say so."""

    monkeypatch.setenv(rustwx_static.STATIC_ENV,
                       str(tmp_path / "nope" / "rw_mpas_static"))
    with pytest.raises(FileNotFoundError) as caught:
        rustwx_static.find_static_bin()
    assert rustwx_static.STATIC_ENV in str(caught.value)


# ------------------------------------------------------------- the door


def _mesh_args(**overrides):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    mpas_mesh.register_cli(sub)
    argv = ["mesh", "--background-km", "120", "--cells", "1000",
            "--out", "x.grid.nc"]
    for flag, value in overrides.items():
        argv.append(f"--{flag.replace('_', '-')}")
        if value is not True:
            argv.append(str(value))
    return parser.parse_args(argv)


def test_the_door_builds_the_static_by_default():
    """FIXED MEANS DEFAULT: the defect is a grid nothing can run, so a
    bare invocation must stop producing one."""

    assert _mesh_args().static is True


def test_the_door_offers_a_named_opt_out():
    """And it is a WORKAROUND, so its help text has to say so."""

    assert _mesh_args(no_static=True).static is False
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    mesh = mpas_mesh.register_cli(sub)
    help_text = mesh.format_help()
    assert "WORKAROUND: write only the grid" in help_text
    assert "not runnable" in help_text or "NOT" in help_text


def test_the_door_accepts_a_geography_root_and_a_declared_spacing():
    args = _mesh_args(geog="/some/where", nominal_dx_m=15000.0)
    assert args.geog == Path("/some/where")
    assert args.nominal_dx_m == pytest.approx(15000.0)


def test_a_pair_missing_the_advection_tables_is_not_called_runnable():
    """The door must not promise a run the port refuses.

    MEASURED: with `deriv_two` absent the MPAS GPU port stops at
    `AttributeError: mesh does not expose required array 'deriv_two'` inside
    `transport.build_advection_coefficients`, before any device memory is
    allocated.  The read-out said "this pair is what a run needs" anyway.  A
    sentence that survives its own disproof is how engine-proven gets
    reported as shipped.
    """

    text = mpas_mesh.runnability_verdict(["deriv_two", "defc_a"])
    assert "NOT YET RUNNABLE" in text
    assert "deriv_two" in text
    assert "build_advection_coefficients" in text
    # And it names the way through, not just the wall.
    assert "init_atmosphere" in text
    assert "grid and static together" not in text


#: A static whose FP32-stored edge lengths the consumer accepts.
#: MEASURED on the published x1.40962.static.nc, whose shortest dual edge is
#: 45,016.7 m: its own vertices reproduce dvEdge to 1.19e-05, inside the
#: port's 2.0e-05.
HEALTHY_FP32 = {
    "max_dv_edge_relative": 1.19e-5,
    "consumer_metric_rtol": 2.0e-5,
    "min_dv_edge_m": 45_016.7,
    "edges_past_consumer_tolerance": 0,
}

#: MEASURED on a generated 127,051-cell mesh at a 120 km background.
DEGENERATE_FP32 = {
    "max_dv_edge_relative": 8.52e-3,
    "consumer_metric_rtol": 2.0e-5,
    "min_dv_edge_m": 3.45,
    "edges_past_consumer_tolerance": 147,
}


def test_a_pair_with_the_advection_tables_is_called_runnable():
    """The other direction: the claim comes back when the gap is closed.

    Without this the refusal could be hard-coded and nobody would notice.
    `defc_a` and `defc_b` are absent from the published x4 static too, so
    their absence alone must not block the claim.
    """

    text = mpas_mesh.runnability_verdict(["defc_a", "defc_b"], HEALTHY_FP32)
    assert "this pair is what a run needs: grid and static together" in text
    assert "NOT YET RUNNABLE" not in text


def test_an_unread_absent_list_is_not_reported_as_runnable():
    """The optimistic branch must not be the safe default.

    The door reads the absent list out of the static engine's receipt.
    When it first shipped it did not ASK for that receipt, so the list
    came back empty and every pair -- including one missing `deriv_two` --
    was announced as "what a run needs".  Absent evidence has to read as
    absent evidence.
    """

    text = mpas_mesh.runnability_verdict(None)
    assert "NOT MEASURED" in text
    assert "grid and static together" not in text
    assert "--receipt" in text


def test_every_blocking_field_is_one_the_engine_has_an_answer_for():
    """A hand-kept blocking list would drift the day the engine changes.

    This used to require every blocking name to appear in the crate's
    UNWRITTEN_OPERATOR_FIELDS, which read as "the verdict must describe a
    file the engine produces" and quietly meant something narrower: that
    the field must stay MISSING.  The day ``deriv_two`` was implemented
    the test failed on the fix.

    The invariant it was reaching for survives the fix: a blocking name
    must be one the engine has an answer for -- it either WRITES the
    variable or DECLARES it absent.  A name that is neither describes a
    file nobody produces, which is the drift worth catching.
    """

    crate = REPO_ROOT / "tools" / "rustwx" / "crates" / "rw-mpas" / "src"
    module = crate / "staticfile" / "schema.rs"
    if not module.is_file():
        pytest.skip("NOT MEASURED: the crate source is not in this install")
    pinned = module.read_text(encoding="utf-8").split(
        "PINNED_STATIC_VARIABLES", 1)[1].split("];", 1)[0]
    for name in mpas_mesh.PORT_BLOCKING_STATIC_FIELDS:
        assert f'"{name}"' in pinned, (
            f"{name} is treated as blocking but the static engine's pinned "
            "manifest does not carry it, so the verdict describes a file "
            "nobody produces")


def test_the_one_blocking_field_is_the_one_the_port_actually_reads():
    """MEASURED, not inferred from what a published static carries.

    The list held three names.  Searching the MPAS GPU port's whole tree
    finds ONE read of `deriv_two` (`transport.py:469`) and NO read of
    `cell_gradient_coef_x` or `cell_gradient_coef_y` anywhere.  Those two
    were on the list because the published static carries them, which is
    a different question from whether anything reads them, and a door
    that reports a pair unrunnable over a field nothing reads is refusing
    a run that would have worked.
    """

    assert mpas_mesh.PORT_BLOCKING_STATIC_FIELDS == ("deriv_two",)
    for unread in ("cell_gradient_coef_x", "cell_gradient_coef_y"):
        assert unread not in mpas_mesh.PORT_BLOCKING_STATIC_FIELDS


def test_a_pair_carrying_deriv_two_is_reported_runnable():
    """The verdict this fix exists to change.

    With `deriv_two` absent the port stops in
    `transport.build_advection_coefficients` before allocating any device
    memory, and the door said so.  The engine writes it now, so the same
    absent list -- which still names the fields genuinely missing --
    must produce the runnable verdict rather than the refusal.
    """

    text = mpas_mesh.runnability_verdict(
        ["cell_gradient_coef_x", "cell_gradient_coef_y", "defc_a", "defc_b"],
        HEALTHY_FP32)
    assert "NOT YET RUNNABLE" not in text
    assert "deriv_two" in text


def test_a_mesh_whose_dual_edges_cannot_be_stored_is_not_called_runnable():
    """A complete pair the consumer still refuses, for a second reason.

    MEASURED: generated meshes at a 120 km background carry a shortest dual
    edge of 7,337.6 m at 2,000 cells but 3.4 m at 127,051, while the
    PUBLISHED x1.40962 has nothing under 45 km at 40,962.  An edge that
    short cannot survive FP32 storage at earth radius, where one ULP is
    half a metre, so the port recomputes dvEdge from the stored vertices
    and refuses the whole pair with `dvEdge disagrees with spherical
    vertex arc length`.  The door must not call such a pair runnable just
    because every field is present.
    """

    text = mpas_mesh.runnability_verdict(["defc_a"], DEGENERATE_FP32)
    assert "NOT RUNNABLE" in text
    assert "dvEdge disagrees with spherical vertex arc length" in text
    assert "3.4 m" in text or "3.5 m" in text
    # The remedy has to be one that was tried.  This refusal first said
    # "more relaxation sweeps", which MEASURED false: --sweeps 200, 600
    # and 2000 on the same request all deliver the same 75.04 m shortest
    # edge.  A refusal naming a remedy nobody checked is a dead end that
    # looks like help.
    assert "fewer cells" in text
    assert "what does NOT work, measured" in text


def test_a_missing_fp32_reading_is_not_reported_as_runnable():
    """The optimistic branch is not the safe default, here either.

    The same trap the absent-field list already carries: an engine that
    did not take the reading must not be quoted as having taken a good
    one.  A static built before this measurement existed is UNKNOWN, not
    healthy.
    """

    text = mpas_mesh.runnability_verdict(["defc_a"], None)
    assert text.startswith("NOT MEASURED")
    assert "fp32_metric_agreement" in text


# ------------------------------------------- against the artifact itself


def _built_static_binary() -> Path | None:
    for candidate in rustwx_static.static_candidates():
        if candidate.is_file():
            return candidate
    return None


@pytest.mark.skipif(_built_static_binary() is None,
                    reason="NOT MEASURED: rw_mpas_static is not built here")
def test_the_binary_prints_exactly_the_abi_this_wrapper_expects():
    """The handshake, run against the real executable rather than a
    fixture: a marker that agrees with itself and disagrees with the
    binary protects nothing."""

    binary = _built_static_binary()
    assert binary is not None
    probe = subprocess.run([str(binary), "--abi"], capture_output=True,
                           text=True, timeout=30)
    assert probe.returncode == 0, probe.stderr
    observed = probe.stdout.strip().replace("\r\n", "\n")
    assert observed == rustwx_static.STATIC_ABI_MARKER


@pytest.mark.skipif(_built_static_binary() is None,
                    reason="NOT MEASURED: rw_mpas_static is not built here")
def test_the_binary_answers_the_version_probe_the_resolver_uses():
    binary = _built_static_binary()
    assert binary is not None
    ok, evidence = rustwx_static.probe_static_bin(binary)
    assert ok, evidence


@pytest.mark.skipif(_built_static_binary() is None,
                    reason="NOT MEASURED: rw_mpas_static is not built here")
def test_the_binary_names_the_fields_it_writes_as_placeholders():
    """A reader must be able to learn from the tool which slots carry a
    bitwise +0 placeholder and which of the old absences are closed, not
    by opening a static and noticing something."""

    binary = _built_static_binary()
    assert binary is not None
    probe = subprocess.run([str(binary), "--help"], capture_output=True,
                           text=True, timeout=30)
    assert probe.returncode == 0, probe.stderr
    for field in rustwx_static.PLACEHOLDER_FIELDS:
        assert field in probe.stdout, f"{field} is not named in --help"
    # And the fields that USED to be absent are named as written, so a
    # reader who learnt the old list is not left looking for a gap.
    for field in rustwx_static.FORMERLY_UNWRITTEN_FIELDS:
        assert field in probe.stdout, f"{field} is not named in --help"
