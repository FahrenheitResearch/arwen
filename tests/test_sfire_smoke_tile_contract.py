"""Closed global carriers and output lifecycles for the tiled bulk profile."""
import numpy as np
import pytest


def test_global_ledger_inventory_is_complete_and_unknown_vectors_are_refused():
    from tilestream.global_inventory import global_keys
    from tilestream.sfire_smoke import GLOBAL_KEYS
    assert global_keys(GLOBAL_KEYS) == tuple(sorted(GLOBAL_KEYS))
    with pytest.raises(ValueError, match='unknown chemistry ledger'):
        global_keys((*GLOBAL_KEYS, 'state/chemdiag_ledger_unclassified'))


def test_native_global_shapes_do_not_expand_with_domain_extent():
    from tilestream.hoststore import manifest_from_arrays
    from tilestream.sfire_smoke import GLOBAL_KEYS
    values = {'state/thp': np.empty((16, 12, 17), np.float32),
              **{name: np.empty(1, np.int32 if name.endswith('started') else np.float64)
                 for name in GLOBAL_KEYS}}
    manifest = {entry.name: entry for entry in manifest_from_arrays(values, 16, 12, 17)}
    for name in GLOBAL_KEYS:
        assert manifest[name].shape(59, 640, 360) == (1,)
        assert manifest[name].dtype == values[name].dtype


def test_output_and_rk_operands_rebuild_but_native_source_clocks_persist():
    from tilestream.physics_inventory import is_checkpointed
    from tilestream.sfire_smoke import OLD_MU_KEY
    assert not is_checkpointed(OLD_MU_KEY)
    assert not is_checkpointed('transport/sfire_smoke/chem0_fire_smoke')
    assert not is_checkpointed('diag/chem/fire_smoke')
    assert is_checkpointed('state/chemdiag_sfire_source_lasttime')
    assert is_checkpointed('state/chem_fire_smoke')
    with pytest.raises(ValueError, match='unknown transport operand'):
        is_checkpointed('transport/unclassified')


def test_only_the_native_bulk_source_profile_has_a_partitioned_operator():
    from gpuwm.config import RunConfig
    from tilestream.sfire_smoke import accepts
    dimensions = dict(nx=17, ny=12, nz=16, dx=90., dy=120., ztop=2000., dt=.25, run_seconds=2.)
    assert accepts(RunConfig(**dimensions, fire_smoke=True))
    assert not accepts(RunConfig(**dimensions, chem_sets='smoke'))
    assert not accepts(RunConfig(**dimensions, chem_sets='sfire_smoke_mixed'))
