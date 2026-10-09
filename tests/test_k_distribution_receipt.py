"""Planning reads the RRTMGP tables' sizes, shapes and gas names from the packaged receipt.

Breakage this prevents: since 2.8.8 the tables decode through rw_netcdf, and
planning decoded them just to read these facts, so the domain wizard,
run-plan, check, tile-memory pricing and composition checks refused on any
install without the native decoder (the native-free Windows cpu job failed
them with NetcdfBridgeMissing).
"""

from __future__ import annotations

from gpuwm.core import preflight, rrtmgp

GAS_NAMES = ("h2o", "co2", "o3", "n2o", "co", "ch4", "o2", "n2", "ccl4",
             "cfc11", "cfc12", "cfc22", "hfc143a", "hfc125", "hfc23", "hfc32",
             "hfc134a", "cf4", "no2")


def _clear():
    rrtmgp.packaged_table_facts.cache_clear()
    rrtmgp.coefficient_gas_names.cache_clear()
    preflight.k_distribution_bytes.cache_clear()
    preflight._gas_table_meta.cache_clear()


def test_planning_needs_no_decoder_when_the_tables_match_the_receipt(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("planning opened an RRTMGP table")

    monkeypatch.setattr(rrtmgp, "load_gas_tables", refuse)
    monkeypatch.setattr(rrtmgp, "load_cloud_tables", refuse)
    monkeypatch.setattr(rrtmgp, "_open_table", refuse)
    _clear()
    try:
        assert rrtmgp.packaged_table_facts() is not None, (
            "the installed tables do not match the packaged receipt; "
            "re-measure gpuwm/data/rrtmgp_k_distribution_bytes.json")
        assert preflight.k_distribution_bytes() == 23_777_296
        assert preflight._gas_table_meta() == {
            "ngpt_lw": 256, "ngpt_sw": 224, "ngas_lw": 19, "ngas_sw": 19,
            "nband_lw": 16, "nband_sw": 14}
        lw = rrtmgp.gas_table_shape("lw")
        assert (lw.ngpt, lw.ngas, lw.nband) == (256, 19, 16)
        assert rrtmgp.coefficient_gas_names("lw") == GAS_NAMES
        assert rrtmgp.coefficient_gas_names("sw") == GAS_NAMES
    finally:
        _clear()


def test_tables_that_differ_from_the_receipt_are_not_trusted(monkeypatch, tmp_path):
    other = tmp_path / "other.nc"
    other.write_bytes(b"not the measured table")
    monkeypatch.setattr(rrtmgp, "_table", lambda _name: other)
    _clear()
    try:
        assert rrtmgp.packaged_table_facts() is None
    finally:
        _clear()
