"""Declared inactive fire domains skip products; corrupt active fields refuse."""
from types import SimpleNamespace
from datetime import datetime
import subprocess
import pytest


def test_native_catalog_inactive_skip_is_distinct_from_missing_active_fields():
    from gpuwm.rustwx import catalog_verdict
    from gpuwm.render import inactive_domain_skips, skip_notice
    detail = "inactive-fire-domain: IFIRE=0 has no coupled fire outputs"
    rows = [("fire_ros","direct","excluded",detail,"inactive-fire-domain")]
    available, skipped = catalog_verdict(rows,"fire_ros")
    assert not available and inactive_domain_skips(skipped)
    assert "successful skip" in str(skip_notice(skipped,wrote_any=False))
    corrupt = [("fire_ros","direct","excluded","active fire lacks ROS_FRONT","invalid-active-fire-domain")]
    available, skipped = catalog_verdict(corrupt,"fire_ros")
    assert available == "fire_ros" and not skipped
    assert not inactive_domain_skips([])
    assert not inactive_domain_skips([("fire_ros","missing-fire-grid-fields")])
    assert not inactive_domain_skips([("fire_ros",detail),("t2m","missing T2")])


def test_native_history_declares_fire_ownership_for_inactive_parent():
    from gpuwm.io.wrfout import wrf_global_attrs
    from gpuwm.config import RunConfig
    grid=SimpleNamespace(ref_lat=40.,ref_lon=-120.,truelat1=30.,truelat2=60.,stand_lon=-120.)
    cfg=RunConfig(nx=6,ny=4,nz=5,dx=100.,dy=100.,ztop=1000.,dt=1.,run_seconds=1.)
    assert wrf_global_attrs(grid,datetime(2026,1,1),run=cfg)["IFIRE"] == 0


def test_native_product_path_already_filed_by_renderer_stays_quiet(tmp_path, capsys):
    from gpuwm.render import _place_engine_output
    from gpuwm.render_layout import NESTED
    path=tmp_path/"d03"/"native_flux"/"20260101"/"native_flux_20260101T000000.png"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"native artifact path control")
    assert _place_engine_output(path,tmp_path,"d03",NESTED) == path
    assert "left flat" not in capsys.readouterr().err


@pytest.mark.parametrize("early", [False, True])
@pytest.mark.parametrize("inactive", [False, True])
def test_landing_observer_distinguishes_successful_skip_from_active_missing_input(
        tmp_path, monkeypatch, early, inactive):
    from pathlib import Path
    from gpuwm.first_products import FirstProducts
    from gpuwm.live_products import LiveProducts
    from gpuwm.render_receipts import publish_invocation, read_summary
    import gpuwm.render as render
    monkeypatch.setattr(render, "announce_missing_basemap", lambda *args, **kwargs: None)
    frame = tmp_path / "wrfout_d01_20260101_00_00_00"
    frame.write_bytes(b"owned frame identity for observer contract")
    reason = ("inactive-fire-domain: IFIRE=0 has no coupled fire outputs"
              if inactive else "active fire lacks ROS_FRONT")

    def runner(command):
        scratch = Path(command[command.index("--out") + 1])
        publish_invocation(root=scratch, engine="rust", requested_spec="fire_ros",
                           written=[], failures=[], skipped=[("fire_ros", reason)],
                           layout="nested", inputs=[frame])
        return subprocess.CompletedProcess(command, 0 if inactive else 1, "", "")

    warnings = []
    reports = []
    plan = dict(run=tmp_path, render=tmp_path / "png", render_products="fire_ros")
    cls = FirstProducts if early else LiveProducts
    observer = cls(plan, report=reports.append, warn=lambda code, *args, **kwargs: warnings.append(code),
                   runner=runner)
    observer._render(domain=1, valid_time=datetime(2026, 1, 1), frame=frame)
    assert reports == []
    if inactive:
        assert warnings == []
        assert read_summary(plan["render"])["skipped_count"] == 1
    else:
        assert warnings == ["first_products_empty" if early else "live_products_empty"]
