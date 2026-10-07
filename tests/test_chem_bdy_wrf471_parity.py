"""WRF module_input_chem_data.F:1531-2031 boundary identity cases."""
import os

import numpy as np
import pytest

from gpuwm.core.chem_bdy import apply_chem_flow_boundaries
from gpuwm.verify.chem_bdy_ref import apply_chem_flow_boundaries_cpu
from gpuwm.verify.chem_oracle import ORACLE_ROOT, load, ulp_table

CASES = load(ORACLE_ROOT / "core" / "flowdep_bdy")
BOUNDARIES = ("bxs", "btxs", "bxe", "btxe", "bys", "btys", "bye", "btye")


def inputs(case, xp):
    # WRF (i,k,j,row) -> row sequence of contiguous (k,j,i).
    initial = case["initial"].transpose(3, 1, 2, 0)
    fields = [xp.asarray(np.ascontiguousarray(f)).copy() for f in initial]
    u = xp.asarray(np.ascontiguousarray(case["u"].transpose(1, 2, 0)))
    v = xp.asarray(np.ascontiguousarray(case["v"].transpose(1, 2, 0)))
    bounds = [xp.asarray(np.ascontiguousarray(case[b].transpose(2, 1, 0)))
              for b in BOUNDARIES]
    return (fields, u, v, int(case["spec_zone"]), case["has_bc"],
            case["default_inflow"], *bounds, case["dt"])


def assert_identity(fields, case, name, to_host=np.asarray):
    expected = case["chem"].transpose(3, 1, 2, 0)
    for row, (field, ref) in enumerate(zip(fields, expected)):
        table = ulp_table(to_host(field), ref)
        print(f"{name}/chem/row{row}: {table}")
        assert table == {"max_ulp": 0, "n_nonzero": 0, "n": ref.size}


@pytest.mark.parametrize("name", sorted(CASES))
def test_cpu_wrf471(name):
    args = inputs(CASES[name], np)
    apply_chem_flow_boundaries_cpu(*args)
    assert_identity(args[0], CASES[name], name)


@pytest.mark.gpu
@pytest.mark.parametrize("name", sorted(CASES))
def test_gpu_wrf471(name):
    if os.environ.get("GPUWM_NO_LOCAL_GPU") == "1":
        pytest.skip("GPUWM_NO_LOCAL_GPU prevents desktop device access")
    cp = pytest.importorskip("cupy")
    args = inputs(CASES[name], cp)
    apply_chem_flow_boundaries(*args)
    assert_identity(args[0], CASES[name], name, cp.asnumpy)


@pytest.mark.parametrize("name", [n for n in sorted(CASES) if n.startswith(("mode2", "mode3"))])
def test_no_boundary_source_arrays(name):
    args = list(inputs(CASES[name], np))
    args[6:14] = [None]*8
    apply_chem_flow_boundaries_cpu(*args)
    assert_identity(args[0], CASES[name], name)


def test_arbitrary_row_defaults():
    case = CASES["mode2_w3_dt2"]
    args = list(inputs(case, np))
    args[5] = np.arange(len(args[0]), dtype=np.float32) + np.float32(0.123)
    apply_chem_flow_boundaries_cpu(*args)
    # Exact zero on the south face is inflow, using the supplied row value.
    zero = np.argwhere(args[2][:, 0, :] == 0)[0]
    k, i = zero
    for row, field in enumerate(args[0]):
        assert field[k, 0, i] == args[5][row]


def test_empty_batch_is_inert():
    apply_chem_flow_boundaries([], None, None, 1, [], [], *([None]*8), 0.)
    apply_chem_flow_boundaries_cpu([], None, None, 1, [], [], *([None]*8), 0.)


def test_production_boundary_rejects_host_fields_before_mutation():
    args = inputs(CASES["mode2_w3_dt2"], np)
    before = [field.tobytes() for field in args[0]]
    with pytest.raises(TypeError, match="host pointers"):
        apply_chem_flow_boundaries(*args)
    assert [field.tobytes() for field in args[0]] == before
