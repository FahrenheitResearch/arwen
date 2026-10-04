"""Import path kept for the native HRRR callers of the physical field contract.

The contract itself is the packaged document named below, pinned in
:mod:`gpuwm.source_authorities` and read by
:mod:`gpuwm.ensemble.physical_fields`. This module holds no table
and no rule. It binds that one document to the generic functions under the
names the HRRR preparation and its tests already import.
"""
from __future__ import annotations

from gpuwm.ensemble import physical_fields as _contracts

CONTRACT_ID = "hrrr-f00-f12-physical-fields-v1"


def hrrr_physical_field_contract(grid_identity, *, evidence):
    """Describe the verified native mapper, with its actual source receipts.

    ``evidence`` must bind the raw input manifest, or the ordinary posted
    source input plan before that manifest exists, and the mapper source,
    together with either the running native decoder or a sealed native
    bridge manifest. A legacy qualification supplies its original evidence,
    not newly invented numeric labels or a new weather-array generation.
    """
    return _contracts.native_field_contract(CONTRACT_ID, grid_identity, evidence=evidence)


def validate_hrrr_physical_field_contract(contract, grid_identity):
    """Reject renamed units, vertical kinds and wind bases before real setup."""
    return _contracts.require_native_field_contract(CONTRACT_ID, contract, grid_identity)


def validate_hrrr_physical_snapshot(snapshot):
    """The coordinate values must describe the native 50-level HRRR axis."""
    return _contracts.validate_native_snapshot(CONTRACT_ID, snapshot)
