"""Shape constants retain ABI, conditional signatures and generic source."""
import pytest
from gpuwm.core.kernel_shape import FUNCTIONS, shape_defines, specialize_source
from gpuwm.core.kernels import module_source, module_source_int_defines


def test_generic_sources_are_untouched():
    for name in FUNCTIONS:
        source = module_source(name)
        assert specialize_source(name, source, ()) == source
        assert specialize_source(name, source, (("EXISTING_TIER", 128),)) == source
    assert shape_defines("advection", "advect_u", (44, 70, 120)) == ()


def test_conditional_signature_and_comments_keep_abi():
    source = '''// void advance_nested_phi_w(int nz) { misleading comment
void advance_nested_phi_w(
#if GPUWM_WRF_EXACT
 int nz, int ny, int nx, const float* muts)
#else
 int nz, int ny, int nx)
#endif
{ float x = 0; for (int k=0; k<nz; ++k) x += k; }
void advance_w_phi_msf(int nz,int ny,int nx) { return; }
void advance_mu_th_msf(int nz,int ny,int nx) { return; }
'''
    defs = shape_defines("acoustic", "advance_nested_phi_w", (44, 70, 120))
    shaped = specialize_source("acoustic", source, defs)
    assert shaped.count("int gpuwm_runtime_nz") == 4
    assert shaped.count("constexpr int nz = 44;") == 3
    assert "// void advance_nested_phi_w(int nz) { misleading comment" in shaped
    assert "for (int k=0; k<nz; ++k) x += k;" in shaped
    assert "const float* muts)" in shaped


def test_all_registered_units_have_shape_parameters():
    for name, funcs in FUNCTIONS.items():
        defs = shape_defines(name, funcs[0], (44, 117, 267))
        source = module_source_int_defines(name, defs)
        assert "constexpr int" in source
        assert "gpuwm_runtime_" in source
    a = module_source_int_defines("acoustic", shape_defines("acoustic", "advance_mu_th_msf", (44,70,120)))
    b = module_source_int_defines("acoustic", shape_defines("acoustic", "advance_mu_th_msf", (44,117,267)))
    assert a != b


@pytest.mark.parametrize("shape", [(0,70,120), (44,-1,120), (True,70,120), (44,70), (44,2**30,3)])
def test_invalid_shape_refused_before_compile(shape):
    with pytest.raises(ValueError):
        shape_defines("acoustic", "advance_mu_th_msf", shape)


def test_partial_shape_refused():
    with pytest.raises(ValueError, match="incomplete"):
        specialize_source("acoustic", "", (("GPUWM_SHAPE_NZ",44),))
