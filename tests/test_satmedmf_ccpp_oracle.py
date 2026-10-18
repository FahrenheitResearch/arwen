"""The full-column qualification is intentionally red until a real port exists."""
import importlib,os
from pathlib import Path
import numpy as np
import pytest
from tools.satmedmf_ccpp_oracle.check import Oracle,columns,compare,advance,FIELDS,REGIMES

@pytest.fixture
def oracle():
    path=os.environ.get("SATMEDMF_ORACLE_LIBRARY")
    if not path:
        pytest.skip("set SATMEDMF_ORACLE_LIBRARY to the pinned sprint-box build")
    return Oracle(path)

def test_column_inputs_cover_required_regimes():
    x=columns()
    assert x["u"].shape==(64,48)
    assert len(REGIMES)==8
    assert x["sfc"][:,13].min()==250**2
    assert x["sfc"][:,13].max()==25000**2

def test_oracle_multistep_is_finite_and_prognostic(oracle):
    x=columns()
    for _ in range(3):
        y=oracle(x)
        assert set(y)==set(FIELDS)
        assert all(np.all(np.isfinite(a)) for a in y.values())
        assert np.any(y["rtg"][:,:,-1]!=0)
        assert np.all((y["kpbl"]>=1)&(y["kpbl"]<=48))
        x=advance(x,y,60.0)

@pytest.mark.parametrize("field",FIELDS)
def test_every_output_one_bit_change_is_caught(oracle,field):
    y=oracle(columns())
    changed={k:v.copy() for k,v in y.items()}
    a=changed[field]
    words=a.view(np.uint64 if a.dtype==np.float64 else np.uint32)
    words.flat[0]^=np.array(1,dtype=words.dtype)
    receipt=compare(y,changed)
    assert receipt[field]["pass"] is False
    assert receipt[field]["mismatches"]==1
    assert receipt[field]["max_ulp"]==1

def test_nonfinite_is_not_a_declared_divergence():
    with pytest.raises(ValueError,match="non-finite"):
        compare({"tdt":np.array([np.nan])},{"tdt":np.array([np.nan])})

def test_default_requires_per_field_tolerance():
    x={"tdt":np.array([1.0])}
    with pytest.raises(ValueError,match="no declared default tolerance"):
        compare(x,x,strict=False)

@pytest.mark.gpu
def test_full_column_strict_port_matches_reference(oracle):
    port=importlib.import_module("gpuwm.core.satmedmf")
    x=columns()
    for step in range(3):
        expected=oracle(x)
        actual=port.run_columns(x,dt=60.0,strict=True)
        receipt=compare(expected,actual)
        assert all(v["pass"] for v in receipt.values()),(step,receipt)
        x=advance(x,expected,60.0)

@pytest.mark.gpu
def test_extended_tracer_layout_is_word_identical(oracle):
    port=importlib.import_module("gpuwm.core.satmedmf")
    x=columns(ntrac=8)
    x["q"][:,:,4]=1.e-5
    x["q"][:,:,5]=2.e-5
    x["q"][:,:,6]=0.25
    for _ in range(3):
        reference=oracle(x)
        assert all(v["pass"] for v in compare(reference,port.run_columns(x)).values())
        x=advance(x,reference,60.)
