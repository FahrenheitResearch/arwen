"""Prepared multi-run reservations price verified urban workspace counts."""

from types import SimpleNamespace

import pytest

from gpuwm import multi_run, stage_cli
from gpuwm.core import preflight as pf
from test_urban_workspace_price import _city_tree


@pytest.mark.parametrize("module", [
    "gpuwm.prepared_domain_tree_forecast",
    "gpuwm.prepared_single_domain_forecast",
])
def test_worker_reservation_uses_verified_prepared_urban_counts(
        tmp_path, monkeypatch, module):
    exp = _city_tree(tmp_path)
    config = tmp_path / "urban-price.toml"
    root = tmp_path / "prepared"
    wps = tmp_path / "namelist.wps"
    counts = {1: 500, 2: 750}
    arguments = ["--prepared-root", str(root),
                 "--experiment-config", str(config)]
    if module.endswith("single_domain_forecast"):
        arguments += ["--wps-namelist", str(wps)]

    def verified(args):
        assert args.config == config
        assert args.prepared_root == root
        assert args.wps_namelist == (
            wps if module.endswith("single_domain_forecast") else None)
        return SimpleNamespace(urban_columns=counts)

    monkeypatch.setattr(pf, "_prepared_check_inputs", verified)
    monkeypatch.setattr(stage_cli, "boundary_pricing_source",
                        lambda *_: "gfs")
    actual = multi_run._worker_reservation_bytes(module, arguments)
    expected = pf.admission_estimate(
        exp, source="gfs", urban_columns=counts).peak_envelope_bytes
    bound = pf.admission_estimate(exp, source="gfs").peak_envelope_bytes
    assert actual == expected
    assert expected < bound


@pytest.mark.parametrize("failure", [OSError, ValueError, RuntimeError, ImportError])
def test_unverified_worker_land_cover_retains_the_configuration_bound(
        tmp_path, monkeypatch, failure):
    exp = _city_tree(tmp_path)
    config = tmp_path / "urban-price.toml"

    def unreadable(_args):
        raise failure("prepared inputs cannot be verified")

    monkeypatch.setattr(pf, "_prepared_check_inputs", unreadable)
    monkeypatch.setattr(stage_cli, "boundary_pricing_source",
                        lambda *_: "gfs")
    actual = multi_run._worker_reservation_bytes(
        "gpuwm.prepared_domain_tree_forecast",
        ("--prepared-root", str(tmp_path / "prepared"),
         "--experiment-config", str(config)))
    assert actual == pf.admission_estimate(
        exp, source="gfs").peak_envelope_bytes
    assert actual > pf.admission_estimate(
        exp, source="gfs", urban_columns={1: 500, 2: 750}).peak_envelope_bytes


@pytest.mark.parametrize("live_head", [False, True])
def test_worker_reservation_reads_sealed_and_live_prepared_authority(
        tmp_path, monkeypatch, live_head):
    from test_prepared_single_domain_forecast import (
        _bind_synthetic_preflight_geometry, _prepared_fixture)

    fixture = _prepared_fixture(tmp_path, "gfs", physics_profile=None)
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)
    if live_head:
        from test_prepared_head_binding import _unseal

        _unseal(fixture)
    original = pf._prepared_check_inputs
    verified = []

    def read(args):
        result = original(args)
        verified.append(result)
        return result

    monkeypatch.setattr(pf, "_prepared_check_inputs", read)
    multi_run._worker_reservation_bytes(
        "gpuwm.prepared_single_domain_forecast",
        ("--prepared-root", str(fixture.prepared),
         "--experiment-config", str(fixture.experiment),
         "--wps-namelist", str(fixture.wps)))
    assert len(verified) == 1
    assert verified[0].prepared_root == fixture.prepared
    assert verified[0].source == "gfs"
    assert bool(verified[0].prepared_head_sha256) is live_head
