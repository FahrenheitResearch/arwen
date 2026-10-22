"""A per-frame running ledger is not series identity.

Breakage it prevents (2.9.4, box D smoke-2card run of the packaged HRRR v4
recipe, 2026-10-21): GPUWM_CHEM_LEDGER, the smoke budget the history writer
rewrites at every frame, was read as series identity, so every frame of a
smoke run became its own series and the go render's windowed pass exited 1
on every whole hour ("'windowed' names no product these frames can draw").
"""

from gpuwm.render import _history_identity_attribute


def test_a_running_ledger_is_not_identity():
    assert not _history_identity_attribute("GPUWM_CHEM_LEDGER")
    assert not _history_identity_attribute("GPUWM_CARRIER_SOURCE")


def test_run_identity_is_still_identity():
    for name in ("START_DATE", "GRID_ID", "DX", "MP_PHYSICS", "GPUWM_RUN_ID"):
        assert _history_identity_attribute(name), name
