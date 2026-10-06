"""Packaged contributor evidence preserves distinct, pinned role identities."""
import shutil

import pytest

from gpuwm import source_authorities as authorities


@pytest.mark.parametrize("profile_id", ["hrrr-native-grib2-v1", "hrrr-prs-grib2-v1"])
def test_surface_contributor_has_distinct_pinned_evidence(profile_id):
    files = authorities.packaged_provenance_files(profile_id)
    assert "vegetation_surface_provenance" in files
    identities = {(path.stat().st_dev, path.stat().st_ino) for path in files.values()}
    assert len(identities) == len(files)
    assert files["vegetation_surface_provenance"].name == (
        "rw-wps-hrrr-surface-vegetation-grib2.provenance.json")


def test_other_packaged_sources_preserve_primary_provenance():
    for profile_id in authorities.packaged_profile_ids():
        profile = authorities.packaged_profile(profile_id)
        if profile.get("composition_state") != "composed" or profile.get("contributing_provenances"):
            continue
        composition = authorities.packaged_composition(profile_id)
        if composition.get("field_sources"):
            continue
        files = authorities.packaged_provenance_files(profile_id)
        assert dict(files) == {
            profile["provenance_role"]: authorities.packaged_authorities(profile_id)["provenance"]}


def test_contributor_provenance_corruption_is_refused(tmp_path, monkeypatch):
    profile_id = "hrrr-native-grib2-v1"
    profile = authorities.packaged_profile(profile_id)
    for name in profile["files"].values():
        shutil.copy2(authorities._AUTHORITY_ROOT / name, tmp_path / name)
    name = profile["contributing_provenances"]["vegetation_surface_provenance"]["file"]
    shutil.copy2(authorities._AUTHORITY_ROOT / name, tmp_path / name)
    data = bytearray((tmp_path / name).read_bytes())
    data[-1] ^= 1
    (tmp_path / name).write_bytes(data)
    monkeypatch.setattr(authorities, "_AUTHORITY_ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="provenance.*hash differs"):
        authorities.packaged_provenance_files(profile_id)
