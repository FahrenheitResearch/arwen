"""One prepared analysis serves stability policies, while restarts bind them."""
from copy import deepcopy
from dataclasses import asdict, replace

import pytest

from gpuwm.ingest.prepared_cache import (
    PreparedCacheCorruptError, PreparedCacheMismatchError,
    PreparedCacheReader, write_prepared_cache)
from gpuwm.io.restart import RestartMismatchError, _require_config_match
from gpuwm.io.restart import _configuration_fingerprint
from test_prepared_cache import _fixture
from test_restart import _cfg

CHANGES = {"epssm": .5, "diff_6th_opt": 2,
           "diff_6th_factor": .3, "diff_6th_slopeopt": 1}


def _bundle(tmp_path):
    initial, met, boundaries = _fixture()
    old = _cfg(epssm=.1, diff_6th_opt=0, diff_6th_factor=.12,
               diff_6th_slopeopt=0)
    identity = {"source": "unchanged-input", "domain_config": {
        "grid_id": 1, "run": asdict(old)}}
    path = tmp_path / "prepared"
    write_prepared_cache(path, identity=identity, initial_result=initial,
                         met=met, boundaries=boundaries)
    return path, identity, old


def test_complete_stability_policy_reuses_the_same_verified_preparation(tmp_path):
    path, identity, _old = _bundle(tmp_path)
    before = {file.name: file.read_bytes() for file in path.iterdir() if file.is_file()}
    new_identity = deepcopy(identity)
    new_identity["domain_config"]["run"].update(CHANGES)
    reader = PreparedCacheReader(path, expected_identity=new_identity)
    assert reader.verify_all()["status"] == "PASS"
    assert {file.name: file.read_bytes() for file in path.iterdir()
            if file.is_file()} == before
    # Retained provenance still states which policy originally prepared it.
    assert reader.header["identity"]["domain_config"]["run"]["epssm"] == .1


@pytest.mark.parametrize("name,value", CHANGES.items())
def test_each_preparation_inert_setting_still_binds_restart_identity(name, value):
    old = _cfg(epssm=.1, diff_6th_opt=0, diff_6th_factor=.12,
               diff_6th_slopeopt=0)
    new = replace(old, **{name: value})
    with pytest.raises(RestartMismatchError, match=name):
        _require_config_match(asdict(old), new, "original-checkpoint")
    assert _configuration_fingerprint(old) != _configuration_fingerprint(new)


def test_stability_reuse_keeps_geometry_and_array_integrity_strict(tmp_path):
    path, identity, _old = _bundle(tmp_path)
    desired = deepcopy(identity)
    desired["domain_config"]["run"].update(CHANGES)
    changed_geometry = deepcopy(desired)
    changed_geometry["domain_config"]["run"]["dx"] += 1
    with pytest.raises(PreparedCacheMismatchError, match=r"run\.dx"):
        PreparedCacheReader(path, expected_identity=changed_geometry)
    reader = PreparedCacheReader(path, expected_identity=desired)
    payload = path / reader.arrays["state/u"]["file"]
    raw = bytearray(payload.read_bytes())
    raw[-1] ^= 1
    payload.write_bytes(raw)
    with pytest.raises(PreparedCacheCorruptError, match="fails its manifest"):
        PreparedCacheReader(path, expected_identity=desired).read_array("state/u")
