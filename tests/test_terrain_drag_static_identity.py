"""Prepared drag fields bind the arithmetic that selected their source stencils."""
import copy

import pytest

from gpuwm.config import RunConfig
from gpuwm.experiment import DomainConfig, domain_config_document
from gpuwm.ingest.prepared_cache import (
    compare_prepared_domain_config, prepared_domain_config_identity,
)
from gpuwm.static.rust_bridge import OROGRAPHIC_MARKER


def _domain(**options):
    run=RunConfig(nx=10,ny=10,nz=10,dx=3000.0,dy=3000.0,ztop=20000.0,
                  dt=12.0,run_seconds=3600.0,**options)
    return DomainConfig(grid_id=1,parent_id=0,i_parent_start=1,j_parent_start=1,
                        parent_grid_ratio=1,parent_time_step_ratio=1,
                        history_interval_s=3600.0,run=run)


def test_default_prepared_identity_has_no_orographic_revision():
    domain=_domain()
    assert prepared_domain_config_identity(domain)==domain_config_document(domain)


@pytest.mark.parametrize("options",[
    {"topo_wind":1},{"topo_wind":2},{"gwd_opt":1},{"gwd_opt":3},
])
def test_old_orographic_arithmetic_cannot_restore_as_corrected_arithmetic(options):
    live=prepared_domain_config_identity(_domain(**options))
    assert live["orographic_sampling_contract"]==OROGRAPHIC_MARKER
    assert compare_prepared_domain_config(live,live)==([],[])
    for revision in (None,"gpuwm_static_orographic_v1"):
        old=copy.deepcopy(live)
        if revision is None:
            old.pop("orographic_sampling_contract")
        else:
            old["orographic_sampling_contract"]=revision
        assert compare_prepared_domain_config(old,live)==(
            [],["orographic_sampling_contract"])
