"""Fail-closed contracts for GPUWM mixed-domain microphysics."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.config import RunConfig, validate_run_config
from gpuwm.core.microphysics_transition import (
    MP8_TO_MP18_POLICY,
    SAME_SCHEME_POLICY,
    launch_mp8_to_mp18_parent_field,
    resolve_microphysics_transition,
)


def _run(mp_physics: int, *, nested: bool = False,
         transition: str = SAME_SCHEME_POLICY) -> RunConfig:
    return RunConfig(
        nx=16, ny=16, nz=4, dx=1000.0, dy=1000.0,
        ztop=10000.0, dt=3.0, run_seconds=9.0,
        nested=nested, specified=not nested,
        grid_id=2 if nested else 1, moist=True, moist_cq=True,
        mp_physics=mp_physics,
        nest_microphysics_transition=transition,
    )


def test_same_scheme_transition_preserves_existing_contract():
    contract = resolve_microphysics_transition(_run(8), _run(8, nested=True))
    assert not contract.mixed
    assert contract.policy_id == SAME_SCHEME_POLICY
    assert contract.receipt() == {
        "policy_id": SAME_SCHEME_POLICY,
        "source_mp_physics": 8,
        "target_mp_physics": 8,
        "mixed": False,
        "stock_wrf_equivalent": True,
    }


def test_mp8_to_mp18_requires_exact_explicit_policy_and_cq():
    parent = _run(8)
    child = _run(18, nested=True)
    with pytest.raises(ValueError, match="requires explicit"):
        resolve_microphysics_transition(parent, child)

    child = replace(child, nest_microphysics_transition=MP8_TO_MP18_POLICY)
    contract = resolve_microphysics_transition(parent, child)
    receipt = contract.receipt()
    assert contract.mixed
    assert receipt["stock_wrf_equivalent"] is False
    assert receipt["ignored_source_fields"] == {
        "nr": "ignored_due_to_scheme_closure_mismatch",
        "ni": "ignored_due_to_scheme_closure_mismatch",
    }
    assert receipt["implementation"]["kernel_sha256"]
    assert receipt["implementation"]["driver_sha256"]
    assert receipt["implementation"]["contract_sha256"]

    with pytest.raises(ValueError, match="parent.moist_cq=true"):
        resolve_microphysics_transition(replace(parent, moist_cq=False), child)


def test_transition_identity_binds_field_map_and_normalizes_newlines(
        tmp_path, monkeypatch):
    import gpuwm.core.microphysics_transition as transition

    before = transition.transition_implementation_identity()
    monkeypatch.setitem(transition._FIELD_CODES, "qv", 99)
    after = transition.transition_implementation_identity()
    assert after["kernel_sha256"] == before["kernel_sha256"]
    assert after["driver_sha256"] == before["driver_sha256"]
    assert after["contract_sha256"] != before["contract_sha256"]

    lf = tmp_path / "lf.cu"
    crlf = tmp_path / "crlf.cu"
    lf.write_bytes(b"one\ntwo\n")
    crlf.write_bytes(b"one\r\ntwo\r\n")
    assert transition._canonical_source_sha256(lf) == \
        transition._canonical_source_sha256(crlf)


def test_unvalidated_and_reverse_mixed_edges_fail_closed():
    with pytest.raises(ValueError, match="unsupported mixed"):
        resolve_microphysics_transition(
            _run(18),
            _run(8, nested=True, transition=MP8_TO_MP18_POLICY),
        )
    with pytest.raises(ValueError, match="unsupported mixed"):
        resolve_microphysics_transition(_run(6), _run(18, nested=True))


def test_transition_selector_validation_is_domain_scoped():
    with pytest.raises(ValueError, match="only be selected on a nested"):
        validate_run_config(_run(18, transition=MP8_TO_MP18_POLICY))
    with pytest.raises(ValueError, match="must be 'same-scheme-only'"):
        validate_run_config(replace(
            _run(18, nested=True), nest_microphysics_transition="unknown"))


@pytest.mark.gpu
def test_mp8_mass_translation_matches_admitted_nssl_initial_state_bitwise():
    import cupy as cp

    from gpuwm.core.nssl2 import launch_initial_state
    from gpuwm.ingest.lateral_bc import couple_nest_field

    shape = (2, 3, 4)
    cell = cp.arange(np.prod(shape), dtype=cp.float32).reshape(shape)
    rho = cp.float32(0.65) + cell * cp.float32(0.025)
    alt = cp.float32(1.0) / rho
    # The production NSSL adapter constructs rho from this exact FP32
    # reciprocal, so use that effective density for the independent call.
    effective_rho = cp.float32(1.0) / alt
    masses = {
        "qv": cp.float32(0.006) + cell * cp.float32(1.0e-5),
        "qc": (cell % 4) * cp.float32(5.0e-9),
        "qr": (cell % 5) * cp.float32(4.0e-9),
        "qi": (cell % 6) * cp.float32(3.0e-9),
        "qs": (cell % 7) * cp.float32(2.0e-9),
        "qg": (cell % 8) * cp.float32(2.5e-9),
    }
    state = SimpleNamespace(
        alt=alt,
        mub2d=cp.full(shape[1:], cp.float32(90000.0)),
        mup=cp.full(shape[1:], cp.float32(500.0)),
        c1h=cp.asarray([0.8, 0.6], dtype=cp.float32),
        c2h=cp.asarray([1.0, 2.0], dtype=cp.float32),
        # Deliberately incompatible Thompson moments: the translator has no
        # API path by which these values can affect MP18 canonicalization.
        nr=cp.full(shape, cp.float32(1.0e30)),
        ni=cp.full(shape, cp.float32(-1.0e30)),
        **{name: value.copy() for name, value in masses.items()},
    )

    expected = {
        **{name: value.copy() for name, value in masses.items()},
        "qh": cp.zeros(shape, dtype=cp.float32),
        "qndrop": cp.zeros(shape, dtype=cp.float32),
        "qnr": cp.zeros(shape, dtype=cp.float32),
        "qni": cp.zeros(shape, dtype=cp.float32),
        "qns": cp.zeros(shape, dtype=cp.float32),
        "qng": cp.zeros(shape, dtype=cp.float32),
        "qnh": cp.zeros(shape, dtype=cp.float32),
        "qnn": cp.full(shape, cp.float32(408163264.0)),
        "qvolg": cp.zeros(shape, dtype=cp.float32),
        "qvolh": cp.zeros(shape, dtype=cp.float32),
    }
    ordered = (
        "qv", "qc", "qr", "qi", "qs", "qg", "qh", "qndrop",
        "qnr", "qni", "qns", "qng", "qnh", "qnn", "qvolg", "qvolh",
    )
    launch_initial_state(effective_rho, *(expected[name] for name in ordered))
    expected_state = SimpleNamespace(
        **expected,
        mub2d=state.mub2d,
        mup=state.mup,
        c1h=state.c1h,
        c2h=state.c2h,
        c1f=cp.asarray([1.0, 0.7, 0.4], dtype=cp.float32),
        c2f=cp.asarray([0.0, 1.5, 3.0], dtype=cp.float32),
        thb=cp.asarray([300.0, 302.0], dtype=cp.float32),
        msft=cp.ones(shape[1:], dtype=cp.float32),
        msfu=cp.ones((shape[1], shape[2] + 1), dtype=cp.float32),
        msfv=cp.ones((shape[1] + 1, shape[2]), dtype=cp.float32),
        has_msf=False,
    )

    for name in ordered:
        actual = cp.empty(shape, dtype=cp.float32)
        launch_mp8_to_mp18_parent_field(
            state, name, out=actual, coupled=False)
        cp.testing.assert_array_equal(actual, expected[name])

        coupled = cp.empty(shape, dtype=cp.float32)
        launch_mp8_to_mp18_parent_field(
            state, name, out=coupled, coupled=True)
        expected_coupled = cp.empty(shape, dtype=cp.float32)
        couple_nest_field(expected_state, name, out=expected_coupled)
        cp.testing.assert_array_equal(coupled, expected_coupled)

    cp.cuda.Stream.null.synchronize()
