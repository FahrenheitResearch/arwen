"""Transcription refuses unknown operations and preserves default-real width."""
import pytest
from tools.satmedmf_ccpp_oracle.transcribe import Expr,split

SYMBOLS={"x":{"type":"d","dims":[]},"n":{"type":"i","dims":[]}}
def test_real_literal_retains_float32_rounding():
    assert Expr("0.13",SYMBOLS).all()=="0x1.0a3d700000000p-3f"
def test_kind_phys_literal_retains_binary64():
    assert Expr("0.13_kind_phys",SYMBOLS).all()=="0x1.0a3d70a3d70a4p-3"
def test_separate_rounding_sites_for_multiply_add():
    code=Expr("x*x+x",SYMBOLS).all()
    assert code.startswith("sm_add(")
    assert "sm_mul(" in code
def test_unknown_intrinsic_refused():
    with pytest.raises(ValueError,match="unknown intrinsic"):
        Expr("invented(x)",SYMBOLS).all()
def test_unknown_symbol_refused():
    with pytest.raises(ValueError,match="undeclared symbol"):
        Expr("bad+x",SYMBOLS).all()
def test_nested_argument_split():
    assert split("min(a,b),c(1,2),d")==["min(a,b)","c(1,2)","d"]
def test_integer_and_real_powers_follow_distinct_routes():
    assert Expr("x**3",SYMBOLS).all().startswith("sm_powi(")
    assert Expr("x**0.33333333",SYMBOLS).all().startswith("sm_pow(")
    assert Expr("x**2.0",SYMBOLS).all().startswith("sm_mul(")

def test_negative_real_one_folds_to_reciprocal():
    assert Expr("x**(-1.0)",SYMBOLS).all().startswith("sm_div(")
