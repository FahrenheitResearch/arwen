"""Import path kept for the native GFS callers of the physical field contract.

The contract itself is the packaged document named below, pinned in
:mod:`gpuwm.source_authorities` and read by
:mod:`gpuwm.ensemble.physical_fields`. This module holds no table
and no rule. It binds that one document to the generic functions under the
names the GFS preparation and its tests already import.
"""
from __future__ import annotations

from gpuwm.ensemble import physical_fields as _contracts

CONTRACT_ID = "gfs-pgrb2-0p25-physical-fields-v1"


def native_gfs_field_contract(grid_identity, *, evidence):
    """Describe the unchanged native GFS RH/pressure-level preparation door."""
    return _contracts.native_field_contract(CONTRACT_ID, grid_identity, evidence=evidence)


def require_native_gfs_field_contract(contract, grid_identity):
    """Refuse units or coordinates the unchanged GFS initializer cannot read."""
    return _contracts.require_native_field_contract(CONTRACT_ID, contract, grid_identity)


def contract_sha256():
    """The digest of the document that defines this contract, as evidence."""
    return _contracts.native_contract_sha256(CONTRACT_ID)
