"""Packaged forecast tables reproduce the pinned Fortran initialiser."""
from pathlib import Path
import hashlib,json
import numpy as np
import pytest
from tools.fast_sbm_host import host as H, replay as RP

ROOT=H.REPO_ROOT
DATA=ROOT/"gpuwm/data/fast_sbm"

@pytest.mark.skipif(not H.GLIBC_LIBM,reason=H.NOT_GLIBC_REASON)
@pytest.mark.parametrize("precision,build",(("float32","o32"),("float64","o64")))
def test_packed_tables_match_fortran(precision,build):
    h=RP.host_for(build,H.flags_corrected())
    with np.load(DATA/f"tables-{precision}.npz") as data:
        for name,a in zip(H.TABLE_ORDER,h._tables):
            assert np.array_equal(data[name],a),name
        for name,a in h.derived().items():
            assert np.array_equal(data[name],a),name


def test_packed_provenance_matches_source():
    manifest=json.loads((DATA/"packed-manifest.json").read_text())
    for name,sha in manifest["source_files"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
    for name,sha in manifest["outputs"].items():
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest()==sha,name
