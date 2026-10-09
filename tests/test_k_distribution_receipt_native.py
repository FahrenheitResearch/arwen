"""The packaged RRTMGP receipt equals what decoding the installed tables gives.

Breakage this prevents: a stale receipt would size every plan's radiation
tables, and name its gases, wrongly; this decodes them (rw_netcdf) and compares.
"""

from __future__ import annotations

from gpuwm.core import preflight, rrtmgp


def test_the_receipt_is_the_decoded_measurement():
    rrtmgp.packaged_table_facts.cache_clear()
    facts = rrtmgp.packaged_table_facts()
    assert facts is not None
    assert int(facts["bytes"]) == preflight._decoded_k_distribution_bytes()
    for kind in ("lw", "sw"):
        tables = rrtmgp.load_gas_tables(kind)
        assert int(facts["meta"][f"ngpt_{kind}"]) == tables.ngpt
        assert int(facts["meta"][f"ngas_{kind}"]) == tables.ngas
        assert int(facts["meta"][f"nband_{kind}"]) == tables.nband
        with rrtmgp._open_table(rrtmgp._table(
                "rrtmgp-gas-lw-g256.nc" if kind == "lw" else "rrtmgp-gas-sw-g224.nc")) as nc:
            assert list(rrtmgp._strings(nc.variables["gas_names"])) == facts["gas_names"][kind]
