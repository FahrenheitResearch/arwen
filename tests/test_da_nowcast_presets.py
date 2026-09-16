"""--da presets: full / vr / custom, and --without subtraction.

The preset is product surface: 'full' must reproduce the certified
full-stack configuration (evidence/da-demo/full-stack/run.sh) exactly,
'vr' must stay the bare base rung, and 'custom' must leave the legacy
flag behaviour untouched so every existing invocation keeps meaning
what it meant.
"""
from __future__ import annotations

import pytest

from tools.da_nowcast import (
    DA_FULL_CWP_VLOC_M, DA_SUBTRACTABLE, FrontDoorError, build_parser,
    resolve_da_preset,
)

BASE = ["run", "--site", "KTLX",
        "--window-end", "2026-08-01T11:30:00Z",
        "--out", "case"]


def parse(*extra):
    return build_parser().parse_args(BASE + list(extra))


class TestFullPreset:
    def test_full_reproduces_the_certified_configuration(self, tmp_path):
        sfc = tmp_path / "sfc.v1"
        cwp = tmp_path / "cwp.nc"
        args = parse("--da", "full", "--surface-obs", str(sfc),
                     "--sfc-t2-sigma-k", "2.0", "--goes-cwp", str(cwp))
        resolve_da_preset(args)
        assert args.hydrometeors is True
        assert args.positivity_policy == "clip"
        assert args.reflectivity_analysis is True
        assert args.clear_air_analysis is True
        assert args.dealias is True
        assert args.cwp_vertical_loc_m == DA_FULL_CWP_VLOC_M

    def test_full_keeps_the_callers_stated_values(self, tmp_path):
        args = parse("--da", "full", "--without", "surface",
                     "--goes-cwp", str(tmp_path / "cwp.nc"),
                     "--positivity-policy", "reject",
                     "--cwp-vertical-loc-m", "8000")
        resolve_da_preset(args)
        assert args.positivity_policy == "reject"
        assert args.cwp_vertical_loc_m == 8000.0

    def test_full_minus_the_file_backed_streams_needs_no_files(self):
        args = parse("--da", "full", "--without", "surface",
                     "--without", "cwp")
        resolve_da_preset(args)
        assert args.reflectivity_analysis is True
        assert args.clear_air_analysis is True
        assert args.dealias is True
        assert args.surface_obs is None
        assert args.goes_cwp == []
        assert args.cwp_vertical_loc_m is None

    def test_full_without_a_surface_file_refuses_and_names_the_exit(self):
        args = parse("--da", "full")
        with pytest.raises(FrontDoorError, match="--without surface"):
            resolve_da_preset(args)

    def test_full_without_goes_files_refuses_and_names_the_exit(self, tmp_path):
        args = parse("--da", "full", "--surface-obs",
                     str(tmp_path / "sfc.v1"), "--sfc-t2-sigma-k", "2.0")
        with pytest.raises(FrontDoorError, match="--without cwp"):
            resolve_da_preset(args)

    def test_subtracting_an_explicitly_enabled_stream_is_refused(self):
        # Re-anchored off --dealias, which no longer exists as an opt-in:
        # dealiasing is on for every run, so its being true states nothing
        # and --without dealias simply drops it.  Reflectivity is still a
        # stream a caller turns on, so it still contradicts.
        args = parse("--da", "full", "--without", "reflectivity",
                     "--reflectivity-analysis")
        with pytest.raises(FrontDoorError, match="state one intention"):
            resolve_da_preset(args)

    def test_full_without_dealias_drops_it_without_contradicting(self):
        args = parse("--da", "full", "--without", "surface",
                     "--without", "cwp", "--without", "dealias")
        resolve_da_preset(args)
        assert args.dealias is False


class TestVrPreset:
    def test_vr_is_bare(self):
        args = parse("--da", "vr")
        resolve_da_preset(args)
        assert args.reflectivity_analysis is False
        assert args.clear_air_analysis is False
        assert args.hydrometeors is False
        # Dealiasing is not a stream, it is how the velocity stream is
        # prepared, and "radial velocity alone" must not quietly mean
        # "radial velocity with a coherent fold left in it".
        assert args.dealias is True

    def test_vr_takes_the_off_switch_without_contradicting(self):
        args = parse("--da", "vr", "--no-dealias")
        resolve_da_preset(args)
        assert args.dealias is False

    def test_vr_refuses_stream_flags(self):
        args = parse("--da", "vr", "--reflectivity-analysis")
        with pytest.raises(FrontDoorError, match="--reflectivity-analysis"):
            resolve_da_preset(args)

    def test_vr_refuses_without(self):
        args = parse("--da", "vr", "--without", "cwp")
        with pytest.raises(FrontDoorError, match="--without cwp"):
            resolve_da_preset(args)


class TestCustomStaysLegacy:
    def test_custom_is_the_default_and_changes_nothing(self):
        args = parse("--reflectivity-analysis", "--hydrometeors",
                     "--positivity-policy", "clip")
        before = vars(args).copy()
        resolve_da_preset(args)
        assert vars(args) == before

    def test_custom_refuses_without(self):
        args = parse("--without", "cwp")
        with pytest.raises(FrontDoorError, match="every stream is already"):
            resolve_da_preset(args)


def test_vr_is_not_subtractable():
    assert "vr" not in DA_SUBTRACTABLE


class TestTheOldSpellingStillParses:
    """`--dealias` names the default instead of dying on it.

    Unfolding is on for every run now, so the flag enables nothing.  It
    stays because invocations that spell it out live outside this door
    (evidence/da-demo/full-stack/run.sh passes it), and `unrecognized
    arguments` is a worse answer than "you asked for what already
    happens".  Only asking for it AND subtracting it is a contradiction.
    """

    def test_the_old_spelling_is_accepted_and_states_the_default(self):
        assert parse("--dealias").dealias is True

    def test_whichever_spelling_comes_last_wins(self):
        assert parse("--dealias", "--no-dealias").dealias is False
        assert parse("--no-dealias", "--dealias").dealias is True

    def test_stating_it_and_subtracting_it_is_refused(self):
        args = parse("--da", "full", "--without", "surface",
                     "--without", "cwp", "--without", "dealias", "--dealias")
        with pytest.raises(FrontDoorError, match="state one intention"):
            resolve_da_preset(args)

    def test_the_default_alone_never_contradicts_the_subtraction(self):
        args = parse("--da", "full", "--without", "surface",
                     "--without", "cwp", "--without", "dealias")
        resolve_da_preset(args)
        assert args.dealias is False
