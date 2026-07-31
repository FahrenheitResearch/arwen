"""Where five separately-developed lanes touch, and could disagree.

Each lane in the v1.3.1 wave was green on its own branch.  Green on a
branch means "my rules hold"; it does not mean "my rules and yours hold
at the same time", and the three seams below are the ones where two
lanes both have an opinion about the same configuration:

* **combination law x nest edges.**  The nest-edge lane admits all 20
  mixed parent->child microphysics edges.  The combination lane makes
  WRF v4.6.1's own compatibility table the admission authority for each
  domain.  A tree the edge matrix likes is still made of domains, and
  every one of them still has to be a legal WRF suite -- an edge policy
  must not become a way to smuggle a WRF-fatal PBL/surface-layer pairing
  into a nest.

* **projections x wizard.**  The projection lane landed polar
  stereographic and Mercator end to end; the wizard lane added the
  prompt session.  The session never mentions projections, so the only
  thing that can go wrong is silent: the short front door authoring a
  domain whose projection the rest of the pipeline refuses.

* **knobs x combination law.**  Six knobs were wired end to end, and one
  of them (``isfflx``) is consumed by the very surface-layer components
  the compatibility matrix rules on.  A knob must not change a verdict
  the matrix owns, in either direction.
"""

from __future__ import annotations

import pytest

from gpuwm.config import RunConfig, validate_run_config
from gpuwm.physics_compat import UnsupportedPhysicsSuiteError
from gpuwm.physics_registry import (
    PLAN_SCHEMA,
    registry_sha256,
    validate_physics_plan,
)
from gpuwm.wrf461_compatibility import (
    WRFVerdict,
    pbl_surface_layer_verdict,
)

MORRISON_TEMPLATE_ID = "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1"
NSSL2_TEMPLATE_ID = (
    "nssl2-mp18-ysu-mm5-noah-kf-rte-rrtmgp-validation-candidate-v1")
MP_EDGE_POLICY = "mp-edge-mass-diagnosed-v1"


def _mixed_tree(parent: str, child_component: str, policy: str) -> dict:
    """A two-domain tree whose child overrides only its microphysics."""

    return {
        "schema": PLAN_SCHEMA,
        "plan_id": "cross-lane-mixed-tree-v1",
        "registry_sha256": registry_sha256(),
        "context": {
            "source_id": "hrrr",
            "runner_id": "tools.prepared_domain_tree_forecast",
            "topology_id": "one-way-nested-v1",
        },
        "domains": [
            {"domain_id": "d01", "template_id": parent},
            {"domain_id": "d02", "template_id": parent,
             "components": {"microphysics": child_component},
             "parameters": {"nest_microphysics_transition": policy}},
        ],
        "edges": [{"parent_domain_id": "d01", "child_domain_id": "d02"}],
    }


def _run_config(**overrides) -> RunConfig:
    values = {
        "nx": 4, "ny": 3, "nz": 49,
        "dx": 1000.0, "dy": 1000.0, "ztop": 15000.0,
        "dt": 5.0, "run_seconds": 0.0, "moist": True,
        "mp_physics": 10,
        "sf_sfclay_physics": 1,
        "bl_pbl_physics": 1,
        "sf_surface_physics": 2,
        "num_soil_layers": 4,
        "cu_physics": 1,
        "cudt_minutes": 5.0,
        "ra_lw_physics": 0,
        "ra_sw_physics": 0,
    }
    values.update(overrides)
    return RunConfig(**values)


# ---------------------------------------------------------------------------
# combination law x nest edges
# ---------------------------------------------------------------------------

def test_a_matrix_admitted_mixed_edge_tree_is_launchable():
    """The agreeable case, so the refusals below mean something."""

    report = validate_physics_plan(
        _mixed_tree(MORRISON_TEMPLATE_ID, "nssl2-mp18", MP_EDGE_POLICY))
    assert report["launchable"] is True, report["errors"]

    # ...and both of that tree's domains are WRF-legal suites on their
    # own, which is the property the next test says an edge cannot buy.
    for mp in (10, 18):
        assert validate_run_config(_run_config(mp_physics=mp)) is not None


@pytest.mark.parametrize("pbl, sfclay", [(1, 5), (1, 0), (5, 0)])
def test_an_edge_policy_cannot_admit_a_wrf_fatal_domain(pbl, sfclay):
    """Per-domain admission outranks the edge matrix, both ways round.

    The nest-edge lane's 20 admitted edges are a statement about
    microphysics translation between two domains.  They say nothing
    about whether either domain is a suite WRF would run, and the
    combination lane's answer to that question is fatal for these three
    pairings.  A tree is not a loophole: the same RunConfig admission
    refuses the domain whether it is standing alone or nested.
    """

    verdict, citation = pbl_surface_layer_verdict(pbl, sfclay)
    assert verdict is WRFVerdict.FATAL

    for mp in (10, 18):  # the parent and the child of the mixed edge
        with pytest.raises(UnsupportedPhysicsSuiteError) as caught:
            validate_run_config(_run_config(
                mp_physics=mp, bl_pbl_physics=pbl,
                sf_sfclay_physics=sfclay))
        blocker = caught.value.blockers[0]
        assert blocker.component == (
            "WRF v4.6.1 PBL/surface-layer compatibility")
        assert citation.path in str(caught.value)


def test_a_mixed_edge_still_needs_its_named_policy():
    """The edge lane's own gate is not weakened by the matrix landing."""

    report = validate_physics_plan(
        _mixed_tree(MORRISON_TEMPLATE_ID, "nssl2-mp18", "same-scheme-only"))
    assert report["launchable"] is False
    assert "transition-required-setting" in {
        error["code"] for error in report["errors"]}


# ---------------------------------------------------------------------------
# projections x wizard
# ---------------------------------------------------------------------------

def test_the_interactive_wizard_can_author_a_polar_stereographic_domain(
        tmp_path, monkeypatch, capsys):
    """A high-latitude point through the short front door.

    The prompt session asks for a point and nothing about projections:
    the wizard picks one from |lat|, and above 60 degrees that is the
    polar stereographic support the projection lane landed.  This is the
    seam where a session could silently emit a domain the rest of the
    pipeline refuses, so it is checked by emitting one and reading the
    projection back out of the file the two lanes jointly produced.
    """

    import tomllib

    from gpuwm import domain_interactive as interactive
    from gpuwm.cli import main as cli_main
    from gpuwm.static.projection import WRF_MAP_PROJ_CODES

    out = tmp_path / "fairbanks.toml"
    replies = iter(["64.8,-147.7", "gfs", "2026-07-29T18", "6", str(out)])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(replies))
    monkeypatch.setattr(interactive, "detected_vram_gib", lambda: 12.0)
    monkeypatch.setattr(interactive, "is_interactive",
                        lambda argv, **k: argv == ["domain"])
    assert cli_main(["domain"]) == 0
    capsys.readouterr()

    payload = tomllib.loads(out.read_text(encoding="utf-8"))
    assert payload["projection"]["map_proj"] == "polar"
    assert WRF_MAP_PROJ_CODES["polar"] == 2

    # The companion namelist carries the same WRF convention, which is
    # what rw-wps and stock WPS read.
    namelist = (out.parent / f"{out.stem}.namelist.wps").read_text(
        encoding="utf-8")
    assert "map_proj = 'polar'" in namelist or "map_proj" in namelist


def test_the_wizard_emits_mercator_in_the_tropics_too(tmp_path, capsys):
    """The other projection this wave has to keep reachable."""

    import tomllib

    from gpuwm.cli import main as cli_main

    out = tmp_path / "manila.toml"
    assert cli_main(["domain", "--point=14.6,120.98", "--source", "gfs",
                     "--cycle", "2026-07-29T18", "--hours", "6",
                     "--ladder", "12", "--card", "12gb",
                     "--out", str(out)]) == 0
    capsys.readouterr()
    payload = tomllib.loads(out.read_text(encoding="utf-8"))
    assert payload["projection"]["map_proj"] == "mercator"


# ---------------------------------------------------------------------------
# knobs x combination law
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("isfflx", [0, 1])
@pytest.mark.parametrize("pbl, sfclay, legal", [
    (1, 1, True), (5, 5, True), (0, 0, True),
    (1, 5, False), (1, 0, False), (5, 0, False),
])
def test_isfflx_never_moves_a_matrix_verdict(isfflx, pbl, sfclay, legal):
    """A wired knob is a runtime choice, not an admission argument.

    ``isfflx`` selects how the ported MM5 and MYNN surface layers compute
    their fluxes.  The compatibility matrix rules on whether that surface
    layer may be paired with that PBL scheme at all.  If the knob could
    move the verdict, one of the two lanes would be deciding something
    that belongs to the other -- so both values are swept against both
    halves of the matrix's own table.
    """

    verdict, _ = pbl_surface_layer_verdict(pbl, sfclay)
    assert (verdict is not WRFVerdict.FATAL) is legal

    cfg_values = {"bl_pbl_physics": pbl, "sf_sfclay_physics": sfclay,
                  "isfflx": isfflx}
    if not sfclay:
        # An active LSM with no surface layer is ArWen's own structural
        # refusal (no writer for the UST/CHS/... exchange seam), and not
        # what this test is about, so the LSM comes off with it.
        cfg_values["sf_surface_physics"] = 0
    if not legal:
        # The matrix's verdict, unchanged by either value of the knob.
        with pytest.raises(UnsupportedPhysicsSuiteError):
            validate_run_config(_run_config(**cfg_values))
    elif not sfclay and not isfflx:
        # The knob lane's OWN gate, for its own named reason: with no
        # surface layer there is nothing that reads isfflx.  It refuses
        # as a plain ValueError naming the missing consumer -- never as
        # a compatibility verdict, which is the distinction this test
        # exists to keep.
        with pytest.raises(ValueError, match="no consumer") as caught:
            validate_run_config(_run_config(**cfg_values))
        assert not isinstance(caught.value, UnsupportedPhysicsSuiteError)
    else:
        cfg = _run_config(**cfg_values)
        assert validate_run_config(cfg) is cfg
        assert cfg.isfflx == isfflx


def test_the_other_wired_knobs_are_admission_neutral():
    """The rest of the retired ledger, at both ends of its envelope.

    Each value is exercised inside the radiation variant that implements
    it -- ``o3input`` and ``use_mp_re`` are legacy-RRTMG branches, and
    the knob lane's own gate says so -- because the point is that a knob
    does not move a PBL/surface-layer verdict, not that every knob is
    legal everywhere.
    """

    from gpuwm.physics_compat import WRF_RRTMG_LEGACY

    legacy = {"ra_lw_physics": 4, "ra_sw_physics": 4,
              "wrf_rrtmg_compatibility": WRF_RRTMG_LEGACY,
              "ra_rrtmg_variant": "rrtmg_legacy"}
    cases = (
        ("o3input", (0, 2), legacy),
        ("use_mp_re", (0, 1), legacy),
        ("rdmaxalb", (False, True), {}),
        ("seaice_albedo_default", (0.0, 0.65, 1.0),
         {"sf_surface_physics": 3, "num_soil_layers": 9}),
    )
    for name, values, extra in cases:
        for value in values:
            cfg = _run_config(**{name: value}, **extra)
            assert validate_run_config(cfg) is cfg
            assert getattr(cfg, name) == value
            # ...and the matrix still says what it said.
            verdict, _ = pbl_surface_layer_verdict(
                cfg.bl_pbl_physics, cfg.sf_sfclay_physics)
            assert verdict is not WRFVerdict.FATAL
