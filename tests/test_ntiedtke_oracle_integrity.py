"""The shipped New Tiedtke oracle must match its own recorded digests.

``tools/ntiedtke_wrf461_oracle/build.sh`` WRITES ``oracle-sha256sums.txt``
but does not check it -- it cannot, because the CSVs are that build's
outputs and pinning an output against itself is circular.  The real byte
pin is on the WRF SOURCE, and build.sh does verify that (three files, by
sha256, refusing to run otherwise).

What was missing is an integrity check on the SHIPPED artefact.  The CSVs
travel from the build directory into ``gpuwm/data/ntiedtke/oracle/`` by
hand, and that copy is exactly where a fixture regeneration can go half
done -- MEASURED once already: after a case-table retune, 7 of 12 recorded
digests did not match the files beside them, because the copy took
``nt-*.csv`` and left the digest file behind.  Every parity test in the
suite would still have passed, because they grade the mirror and the
kernel against whatever CSVs are present.

So this is the gate that notices.  It is CPU-only and reads nothing but
the data directory.
"""
from __future__ import annotations

import hashlib

import pytest

from gpuwm.verify.ntiedtke_oracle import ORACLE_DIR

_SUMS = "oracle-sha256sums.txt"


def _recorded():
    path = ORACLE_DIR / _SUMS
    if not path.exists():
        pytest.skip(f"{_SUMS} not shipped")
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(None, 1)
        out[name.strip().lstrip("*")] = digest
    return out


def test_every_recorded_digest_matches_the_shipped_file():
    bad = []
    for name, want in sorted(_recorded().items()):
        f = ORACLE_DIR / name
        if not f.exists():
            bad.append(f"{name}: recorded but not shipped")
            continue
        got = hashlib.sha256(f.read_bytes()).hexdigest()
        if got != want:
            bad.append(f"{name}: {got[:12]} != recorded {want[:12]}")
    assert not bad, (
        "the shipped oracle does not match its own receipt -- most likely a "
        "fixture regeneration whose copy went half done:\n  "
        + "\n  ".join(bad))


def test_no_shipped_csv_is_missing_from_the_receipt():
    """A new fixture file must be recorded, not silently ride along."""
    recorded = set(_recorded())
    shipped = {p.name for p in ORACLE_DIR.glob("*.csv")}
    missing = sorted(shipped - recorded)
    assert not missing, (
        f"shipped but absent from {_SUMS}: {missing}.  Regenerate it with "
        "build.sh rather than adding the row by hand.")
