"""Regression witnesses that import only entry points present in v2.7.2."""
import argparse
from datetime import datetime
from types import SimpleNamespace
import tomllib

import pytest


def test_regression_source_selected_configuration():
    from gpuwm.cyclone_setup import configuration_text
    text, _ = configuration_text(cycle="2026090900",point=(18.,-65.),forcing_source="aigfs")
    assert tomllib.loads(text)["fetch"]["source"] == "aigfs"


def test_regression_member_is_not_an_era_only_fetch_hint():
    from gpuwm.fetch import validate_fetch_hints
    hints = {"source":"gefs","cycle":"2026-09-09T00","hours":6,"member":"p03"}
    validate_fetch_hints(hints,source="fixture")
    assert hints["member"] == "p03"


def test_regression_wps_interval_is_an_explicit_contract():
    from gpuwm.cyclone_setup import configuration_text
    from gpuwm.hrrr_prepared_bundle import render_wps_namelist
    _, exp = configuration_text(cycle="2026090900",point=(18.,-65.))
    assert " interval_seconds = 21600," in render_wps_namelist(exp,interval_seconds=21600)


def test_regression_cli_accepts_source_and_member():
    from gpuwm.cyclone_setup import register_cli
    parser=argparse.ArgumentParser()
    register_cli(parser.add_subparsers())
    args=parser.parse_args(["cyclone-setup","--source","gefs","--member","p03"])
    assert args.source == "gefs" and args.member == "p03"


def test_regression_member_source_has_staged_chain():
    from gpuwm.runplan import intent_drivability
    assert intent_drivability()["gefs"]["chain"] == "prepared:staged"


def test_regression_empty_inventory_never_certifies_a_member():
    from gpuwm.member_grammar import load_member_grammar, MemberIdentityRefusal
    from gpuwm.member_prep import verify_member_rows
    from gpuwm.source_authorities import packaged_member_grammar
    grammar=load_member_grammar(packaged_member_grammar("gefs-ensemble-grib2-members-v1"))
    with pytest.raises(MemberIdentityRefusal,match="no GRIB messages"):
        verify_member_rows(grammar,grammar.member("p03"),[],source_label="empty")


def test_regression_domain_exit_ends_track_instead_of_nan_forever(tmp_path):
    from gpuwm.core.storm_track_writer import TrackConfig,TrackWriter
    writer=TrackWriter(TrackConfig("track.csv"),initial_time=datetime(2026,9,9),outdir=tmp_path)
    fix=SimpleNamespace(found=None,evidence={"track_end_reason":"parent-domain boundary"})
    result=writer.emit(fix,t=0,parent_state=None)
    receipt=writer.close()
    assert result.get("status") == "ended"
    assert receipt["termination"]["reason"] == "parent-domain boundary"


def test_regression_follower_can_own_track_output():
    from gpuwm.core.nest_lifecycle import build_domain_follow_config
    from gpuwm.companion_domains import VORTEX_PRESET
    cfg=build_domain_follow_config({**VORTEX_PRESET,"track":{"path":"track.d02.csv"}},"fixture",grid_id=2)
    assert cfg.track.path == "track.d02.csv"
