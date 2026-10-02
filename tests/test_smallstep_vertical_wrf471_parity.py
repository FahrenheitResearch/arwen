"""Native acoustic columns against byte-unmodified compiled WRF v4.7.1.

The exact-word receipt is a measured baseline, asserted for equality.
Changing a result in either direction requires a new explained receipt.
The compiled reference must be built by the pinned small-step oracle tool.
"""
from __future__ import annotations

import json
import hashlib

import numpy as np

import pytest

from conftest import requires_gpu
from gpuwm.core.kernels import module_source
from gpuwm.verify.smallstep_oracle import ORACLE_DIR
from gpuwm.verify.smallstep_vertical_oracle import _measure, vertical_port_outputs, vertical_cases


def test_vertical_workspace_witness_only_stores_existing_register_words():
    from tools.smallstep_wrf471_oracle.vertical_workspace import workspace_source
    from gpuwm.verify.default_kernel_source import default_source
    original = default_source(module_source("acoustic"))
    source = workspace_source(original)
    restored = source
    for declaration, store in (
            ("real t2_dn =", "oracle_t2[c] = t2_dn;"),
            ("real t2_up =", "oracle_t2[h] = t2_up;"),
            ("real muave =", "oracle_mu[c] = muts; oracle_mu[st + c] = muave;")):
        assert source.count(store) == original.count(declaration)
        restored = restored.replace(store, "")
    parameters = "real* __restrict__ oracle_t2, real* __restrict__ oracle_mu,"
    prototype = "real cf1, real cf2, real cf3, real rdx, real rdy,"
    assert source.count(parameters) == original.count(prototype) == 2
    restored = restored.replace(parameters, "")
    assert " ".join(restored.split()) == " ".join(original.split())
    # It instruments the default compile: no opt-in WRF-exact branch survives.
    assert "GPUWM_WRF_EXACT" not in source


def test_vertical_workspace_default_view_resolves_or_refuses_each_selector():
    from gpuwm.verify.default_kernel_source import default_source
    text = ("a\n#if GPUWM_WRF_EXACT_X\nb\n#else\nc\n#endif\n#if !GPUWM_WRF_EXACT\nd\n#endif\n"
            "#ifdef GPUWM_WRF_EXACT\ne\n#endif\n#ifndef OTHER\n#if GPUWM_WRF_EXACT\nf\n#endif\ng\n#endif\n")
    assert default_source(text) == "a\nc\nd\n#ifndef OTHER\ng\n#endif\n"
    selector_or = ("GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP || "
                   "GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT_C_DIFFUSION || "
                   "GPUWM_WRF_EXACT_D_DIAGNOSTICS")
    assert default_source(f"#if {selector_or}\nstrict\n#else\ndefault\n#endif\n") == "default\n"
    assert default_source("#if GPUWM_WRF_EXACT_C_ADVECTION || GPUWM_WRF_EXACT\nstrict\n#endif\n") == ""
    for unreadable in ("#if GPUWM_WRF_EXACT && X\n#endif\n", "#if GPUWM_WRF_EXACT\n#elif X\n#endif\n",
                       "#if GPUWM_WRF_EXACT || X\n#endif\n",
                       "#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_X\n#endif\n",
                       "#if GPUWM_WRF_EXACT && GPUWM_WRF_EXACT_C_BIGSTEP\n#endif\n",
                       "#if (GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP)\n#endif\n",
                       "#if GPUWM_WRF_EXACT || GPUWM_WRF_EXACT_C_BIGSTEP\n#elif X\n#endif\n",
                       "#if GPUWM_WRF_EXACT\n", "#endif\n"):
        with pytest.raises(ValueError):
             default_source(unreadable)


def test_vertical_selector_invokes_native_callable_once(monkeypatch):
    """A nested native get_kernel lookup must not retain a recorder."""
    from gpuwm.core import acoustic
    from gpuwm.verify.smallstep_vertical_oracle import selected_vertical_launch
    calls = []

    def native_get(module, name):
        return lambda grid, block, args: calls.append((name, grid, block, args))

    def native_w(name, nz):
        return acoustic.get_kernel("acoustic", name)

    def prepare(state, cfg, dtau, coefficients):
        uv = acoustic.get_kernel("acoustic", "advance_uv")
        w = acoustic._w_phi_kernel("advance_w_phi", 49)

        def launch(*, first):
            uv((1,), (256,), ("uv",))
            w((2,), (128,), ("vertical",))
        return launch

    monkeypatch.setattr(acoustic, "get_kernel", native_get)
    monkeypatch.setattr(acoustic, "_w_phi_kernel", native_w)
    monkeypatch.setattr(acoustic, "prepare_acoustic_substep_launch", prepare)
    name = selected_vertical_launch(None, None, 2.0, ())
    assert name == "advance_w_phi"
    assert calls == [("advance_w_phi", (2,), (128,), ("vertical",))]


@pytest.fixture(scope="module")
def vertical_measurements():
    baseline_path = ORACLE_DIR / "vertical-native.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    measured, native_word_identity = {}, {}
    with np.load(ORACLE_DIR / "vertical-reference.npz", allow_pickle=False) as refs:
        for name, raw, metadata in vertical_cases():
            got = vertical_port_outputs(raw, metadata,
                  reference_inputs={key: refs[name + "__" + key + "_input"]
                                    for key in ("history_p", "history_old")})
            native_word_identity[name] = {field: np.array_equal(value.view(np.uint32),
                refs[name + "__" + field + "_native"].view(np.uint32))
                for field, value in got.items() if field != "ph_diag"}
            ref = lambda field: refs[name + "__" + field + "_wrf"]
            measured[name] = {
                "calc_coef_w": {field: _measure(got[field], ref(field))
                                for field in ("a", "alpha", "gamma")},
                "advance_w": {label: _measure(got[field], ref(field)) for label, field in
                              (("w", "w"), ("ph", "ph"), ("t_2ave", "t2"),
                               ("muts_workspace", "muts"), ("muave_workspace", "muave"))},
                "calc_p_rho": {field: _measure(got[field], ref(field)) for field in ("al", "p")}}
            measured[name]["calc_p_rho"]["ph_unchanged"] = _measure(got["ph_diag"], ref("ph_diag"))
            measured[name]["calc_p_rho"]["reference_pm1_copy"] = _measure(ref("p"), ref("pm1"))
            measured[name]["calc_p_rho"]["history_weighted_p"] = _measure(got["history"], ref("history"))
            measured[name]["calc_p_rho"]["reference_pm1_history_copy"] = _measure(ref("p"), ref("pm1_history"))
    assert set(measured) == set(baseline["cases"])
    return measured, baseline["cases"], native_word_identity


def test_vertical_compiled_reference_arrays_are_pinned():
    baseline = json.loads((ORACLE_DIR / "vertical-native.json").read_text(encoding="utf-8"))
    reference = ORACLE_DIR / "vertical-reference.npz"
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == baseline["arrays_sha256"]
    with np.load(reference, allow_pickle=False) as refs:
        for name in baseline["cases"]:
            assert np.array_equal(refs[name + "__p_wrf"].view(np.uint32),
                                  refs[name + "__pm1_wrf"].view(np.uint32)), name


@pytest.mark.gpu
@requires_gpu
def test_calc_coef_w_matches_compiled_wrf471_measured_words(vertical_measurements):
    measured, baseline, _ = vertical_measurements
    for name in measured:
        assert measured[name]["calc_coef_w"] == baseline[name]["calc_coef_w"], name


@pytest.mark.gpu
@requires_gpu
def test_advance_w_matches_compiled_wrf471_measured_words(vertical_measurements):
    measured, baseline, _ = vertical_measurements
    for name in measured:
        assert measured[name]["advance_w"] == baseline[name]["advance_w"], name


@pytest.mark.gpu
@requires_gpu
def test_calc_p_rho_matches_compiled_wrf471_measured_words(vertical_measurements):
    measured, baseline, _ = vertical_measurements
    for name in measured:
        assert measured[name]["calc_p_rho"] == baseline[name]["calc_p_rho"], name


@pytest.mark.gpu
@requires_gpu
def test_native_vertical_output_words_equal_frozen_receipt(vertical_measurements):
    _, _, identities = vertical_measurements
    for name, outputs in identities.items():
        assert all(outputs.values()), (name, outputs)


def test_implicit_w_damper_uses_wrf_rounded_half_pi():
    source = module_source("acoustic")
    assert source.count("sinf(1.5707963267948966f *") == 2
    assert "sinf(1.5707963f *" not in source
    red = json.loads((ORACLE_DIR / "vertical-pi-red.json").read_text(encoding="utf-8"))
    assert red["old_half_pi_word"] == 0x3FC90FDA
    assert red["corrected_half_pi_word"] == 0x3FC90FDB
    assert red["evidence"]["native"]["repaired_words"] == 3


@pytest.mark.gpu
@requires_gpu
def test_implicit_w_damper_corrected_words_match_compiled_wrf471(vertical_measurements):
    red = json.loads((ORACLE_DIR / "vertical-pi-red.json").read_text(encoding="utf-8"))
    assert vertical_measurements[2]["implicit_damper"]["w"]
    with np.load(ORACLE_DIR / "vertical-reference.npz", allow_pickle=False) as refs:
        native = refs["implicit_damper__w_native"]
        for repaired in red["evidence"]["native"]["repaired"]:
            ix = tuple(repaired["index"])
            assert repaired["old_word"] != repaired["fortran_word"]
            assert int(native[ix].view(np.uint32)) == repaired["fortran_word"]
