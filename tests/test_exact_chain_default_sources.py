"""Independent 2.8.9 source identities for mode-selected default physics."""
import hashlib
from pathlib import Path
import pytest
from gpuwm.core import kernels

BASE_SOURCES = {
    'nssl2_nucond_default.cu': 'b4819a1e912924fe2e30c33897585e3a46c32e89dcb3ab43eb1ee32b212b6db1',
    'nssl2_default.cu': '37173f1bcc9ec0fd6c6f4afe060c0d984bcd14abc5d4841d05aa62e6ac85d3ea',
    'nssl2_fused_gs_default.cu': '07dff4bf34509e89b75683a83d356c4691b6eab1aa96140af3e6fa5b00fe429b',
    'nssl2_driver_support_default.cu': '12f193c6918056b995577930304867cd8369a46ab1facfcde589d244cbd6eca8',
    'nssl2_diagnostics_default.cu': 'a95ae9e0bc3dd20a13865cfa6d1148d2a78ee5d7c17c9c1bca9a0c8dbdf19868',
    # ysu_default.cu since b00a0b9ecb1: ae709149b5a reworded a measured-on
    # comment (an export hit) and the calm-column early exit was removed
    # (bl_ysu_run has none: a calm column runs the whole scheme in WRF).
    "ysu_default.cu": "f56e3ec7646a111d07c5447383ae136b2959b5f4545441a598b51be16ccdc768",
    "ysu_topo_default.cuh": "031efb139b827dd94fa9407455c3392e85cb29f4af6b954ef18d931e735890b0",
    # Kessler is ONE unit in both modes since the 2.9.1 exact-chain merge
    # (d5e98e3ba59: "kessler_default.cu ... not carried"): kessler.cu with
    # the glibc_flt32 header is bitwise WRF 4.6.1 (9ccf3e922b5), so no
    # mode-selected default blob exists for it.
}

@pytest.mark.parametrize("filename", sorted(BASE_SOURCES))
def test_selected_default_source_is_the_independent_base_blob(filename):
    root = Path(kernels.__file__).parent
    if filename == "nssl2_default.cu":
        source = (root / "nssl2.cu").read_text()
        for old, new in (("gfk_pow(", "powf("), ("gfk_exp(", "expf("), ("gfk_log(", "logf(")):
            source = source.replace(old, new)
        data = source.encode()
    else:
        data = (root / filename).read_bytes()
    assert hashlib.sha256(data).hexdigest() == BASE_SOURCES[filename]


def test_kessler_is_one_unit_in_both_modes(monkeypatch):
    import gpuwm.wrf_exact as exact
    root = Path(kernels.__file__).parent
    assert not (root / "kessler_default.cu").exists()
    assert "kessler" not in kernels._DEFAULT_SOURCE_MODULES
    for enabled in (False, True):
        monkeypatch.setattr(exact, "ENABLED", enabled)
        assert kernels._unit_text("kessler") == (root / "kessler.cu").read_text()

@pytest.mark.parametrize("name", ("ysu",))
def test_ysu_module_keeps_the_default_source_in_both_modes(name, monkeypatch):
    import gpuwm.wrf_exact as exact
    root = Path(kernels.__file__).parent
    monkeypatch.setattr(exact, "ENABLED", False)
    assert kernels._unit_text(name) == (root / (name + "_default.cu")).read_text()
    expected = ("glibc_flt32.cuh", "ysu_topo_default.cuh") if name == "ysu" else ()
    assert kernels._extra_header_text(name) == "".join((root / h).read_text() for h in expected)
    assert kernels.module_options(name) == (("-std=c++17", "--fmad=false") if name == "ysu" else ("-std=c++17",))
    monkeypatch.setattr(exact, "ENABLED", True)
    assert kernels._unit_text(name) == (root / (name + "_default.cu")).read_text()
    assert "--fmad=false" in kernels.module_options(name)
    strict = kernels.module_source(name)
    if name == "ysu":
        assert "real ysu_max(" in strict and "real ysu_get_pblh(" in strict


def test_strict_ysu_header_has_one_divisor_helper():
    header = (Path(kernels.__file__).parent / "ysu_480_topo.cuh").read_text()
    assert header.count("real ysu_pos(") == 1

@pytest.mark.parametrize("name", ('nssl2_nucond', 'nssl2', 'nssl2_fused_gs', 'nssl2_driver_support', 'nssl2_diagnostics'))
def test_nssl_default_keeps_base_source_and_strict_receives_its_own_headers(name, monkeypatch):
    import gpuwm.wrf_exact as exact
    root = Path(kernels.__file__).parent
    monkeypatch.setattr(exact, "ENABLED", False)
    filename = name + ("_default.cu" if name in kernels._DEFAULT_SOURCE_MODULES else ".cu")
    assert kernels._unit_text(name) == (root / filename).read_text()
    expected = "#define gfk_pow powf\n#define gfk_exp expf\n#define gfk_log logf\n" if name == "nssl2" else ""
    assert kernels._extra_header_text(name) == expected
    monkeypatch.setattr(exact, "ENABLED", True)
    assert kernels._unit_text(name) == (root / (name + ".cu")).read_text()
    assert kernels._extra_header_text(name) == "".join((root / h).read_text() for h in ("glibc_flt32.cuh", "nssl2_wrf_math.cuh"))
