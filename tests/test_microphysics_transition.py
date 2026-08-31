"""Fail-closed contracts for GPUWM mixed-domain microphysics."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.config import RunConfig, validate_run_config
from gpuwm.core.microphysics_transition import (
    EDGE_MATRIX_POLICY,
    MP8_TO_MP18_POLICY,
    PORTED_MP_PHYSICS,
    SAME_SCHEME_POLICY,
    launch_microphysics_edge_parent_field,
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


def test_all_thirty_mixed_edges_require_their_explicit_policy():
    resolved = []
    for source in PORTED_MP_PHYSICS:
        for target in PORTED_MP_PHYSICS:
            policy = (
                SAME_SCHEME_POLICY if source == target
                else MP8_TO_MP18_POLICY if (source, target) == (8, 18)
                else EDGE_MATRIX_POLICY
            )
            contract = resolve_microphysics_transition(
                _run(source),
                _run(target, nested=True, transition=policy),
            )
            assert contract.mixed is (source != target)
            resolved.append((source, target))
    # Six ported selectors (mp=50 joined with its rime-pair closure) give
    # 30 ordered mixed edges plus 6 same-scheme edges.
    assert len(resolved) == 36

    with pytest.raises(ValueError, match="requires explicit"):
        resolve_microphysics_transition(
            _run(18),
            _run(8, nested=True, transition=MP8_TO_MP18_POLICY),
        )
    with pytest.raises(ValueError, match="ported selectors"):
        resolve_microphysics_transition(
            _run(18),
            _run(55, nested=True, transition=EDGE_MATRIX_POLICY),
        )


def test_species_receipts_cover_mapped_defaulted_diagnosed_and_dropped():
    wsm_to_morr = resolve_microphysics_transition(
        _run(6), _run(10, nested=True, transition=EDGE_MATRIX_POLICY))
    rows = [dict(row) for row in wsm_to_morr.species_actions()]
    assert {
        "mapped": 5, "defaulted": 1, "diagnosed": 4, "dropped": 1,
    } == wsm_to_morr.receipt()["species_action_counts"]
    assert any(row["source_field"] == "qg"
               and row["action"] == "dropped" for row in rows)
    assert any(row["target_field"] == "qg"
               and row["action"] == "defaulted" for row in rows)

    nssl_to_morr = resolve_microphysics_transition(
        _run(18), _run(10, nested=True, transition=EDGE_MATRIX_POLICY))
    assert nssl_to_morr.mass_source("qg") == "qh"
    nssl_rows = [dict(row) for row in nssl_to_morr.species_actions()]
    assert any(row["source_field"] == "qg"
               and row["action"] == "dropped" for row in nssl_rows)
    assert sum(row["action"] == "dropped" for row in nssl_rows) == 10

    graupel_morr_parent = replace(_run(10), morr_rimed_ice=0)
    graupel_morr_child = replace(
        _run(6, nested=True, transition=EDGE_MATRIX_POLICY),
        wsm6_hail_opt=0)
    compatible = resolve_microphysics_transition(
        graupel_morr_parent, graupel_morr_child)
    assert compatible.mass_source("qg") == "qg"


def test_transition_selector_validation_is_domain_scoped():
    with pytest.raises(ValueError, match="only be selected on a nested"):
        validate_run_config(_run(18, transition=MP8_TO_MP18_POLICY))
    with pytest.raises(ValueError, match="must be 'same-scheme-only'"):
        validate_run_config(replace(
            _run(18, nested=True), nest_microphysics_transition="unknown"))


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("source_mp", "target_mp", "moment_fields"),
    [
        (6, 10, ("nr", "ni", "ns")),
        (10, 8, ("nr", "ni")),
    ],
)
def test_generic_two_moment_targets_diagnose_positive_finite_numbers(
        source_mp, target_mp, moment_fields):
    import cupy as cp

    shape = (2, 3, 4)
    parent = SimpleNamespace(
        alt=cp.full(shape, cp.float32(0.8)),
        mub2d=cp.full(shape[1:], cp.float32(90000.0)),
        mup=cp.full(shape[1:], cp.float32(100.0)),
        c1h=cp.asarray([0.8, 0.6], dtype=cp.float32),
        c2h=cp.asarray([1.0, 2.0], dtype=cp.float32),
        qv=cp.full(shape, cp.float32(0.008)),
        qc=cp.full(shape, cp.float32(2.0e-5)),
        qr=cp.full(shape, cp.float32(3.0e-5)),
        qi=cp.full(shape, cp.float32(4.0e-5)),
        qs=cp.full(shape, cp.float32(5.0e-5)),
        qg=cp.full(shape, cp.float32(6.0e-5)),
    )
    contract = resolve_microphysics_transition(
        _run(source_mp),
        _run(target_mp, nested=True, transition=EDGE_MATRIX_POLICY),
    )
    for field in moment_fields:
        actual = cp.empty(shape, dtype=cp.float32)
        launch_microphysics_edge_parent_field(
            contract, parent, field, out=actual, coupled=False)
        assert bool(cp.isfinite(actual).all())
        assert bool((actual > 0.0).all())
    cp.cuda.Stream.null.synchronize()


@pytest.mark.gpu
def test_morrison_hail_to_nssl_diagnoses_hail_number_and_volume():
    import cupy as cp

    shape = (2, 2, 3)
    parent = SimpleNamespace(
        alt=cp.full(shape, cp.float32(0.8)),
        mub2d=cp.full(shape[1:], cp.float32(90000.0)),
        mup=cp.zeros(shape[1:], dtype=cp.float32),
        c1h=cp.asarray([0.8, 0.6], dtype=cp.float32),
        c2h=cp.asarray([1.0, 2.0], dtype=cp.float32),
        qv=cp.full(shape, cp.float32(0.008)),
        qc=cp.full(shape, cp.float32(2.0e-5)),
        qr=cp.full(shape, cp.float32(3.0e-5)),
        qi=cp.full(shape, cp.float32(4.0e-5)),
        qs=cp.full(shape, cp.float32(5.0e-5)),
        qg=cp.full(shape, cp.float32(9.0e-5)),
    )
    contract = resolve_microphysics_transition(
        _run(10),
        _run(18, nested=True, transition=EDGE_MATRIX_POLICY),
    )
    outputs = {}
    for field in ("qg", "qh", "qng", "qnh", "qvolg", "qvolh"):
        outputs[field] = cp.empty(shape, dtype=cp.float32)
        launch_microphysics_edge_parent_field(
            contract, parent, field, out=outputs[field], coupled=False)
    cp.testing.assert_array_equal(outputs["qg"], 0.0)
    cp.testing.assert_array_equal(outputs["qng"], 0.0)
    cp.testing.assert_array_equal(outputs["qvolg"], 0.0)
    cp.testing.assert_array_equal(outputs["qh"], parent.qg)
    assert bool((outputs["qnh"] > 0.0).all())
    cp.testing.assert_array_equal(
        outputs["qvolh"], parent.qg / cp.float32(900.0))
    cp.cuda.Stream.null.synchronize()


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


# ---------------------------------------------------------------------------
# The P3 mixed nest edge: a defined, documented, mass-conserving closure.
# The named refusal mp=50 used to carry is retired with the defect that
# installed it (its closure gap); these tests are the closure's contract.
# ---------------------------------------------------------------------------

def test_a_mixed_p3_edge_resolves_both_directions_with_the_matrix_policy():
    """The refusal is retired: every P3 mixed pair resolves and receipts.

    Both directions against every other ported selector, each carrying the
    ``p3_edge`` receipt block that records the exact closure constants --
    the same way the ratified MP8->MP18 pair records its own.
    """
    from gpuwm.core import microphysics_transition as mt

    partners = [mp for mp in mt.PORTED_MP_PHYSICS if mp != 50]
    assert partners == [1, 6, 8, 10, 18]
    for other in partners:
        for source, target in ((other, 50), (50, other)):
            contract = resolve_microphysics_transition(
                _run(source),
                _run(target, nested=True, transition=EDGE_MATRIX_POLICY))
            assert contract.mixed is True
            receipt = contract.receipt()
            block = receipt["p3_edge"]
            assert block["direction"] == (
                "enter" if target == 50 else "leave")
            assert block["fresh_snow_rime_density_kg_m3"] == 100.0
            assert block["dense_rime_density_kg_m3"] == 400.0
            assert block["rime_density_bounds_kg_m3"] == [50.0, 900.0]
            assert block["mass_conserving"] is True
            # The policy is still explicit, never implied.
            with pytest.raises(ValueError, match="requires explicit"):
                resolve_microphysics_transition(
                    _run(source), _run(target, nested=True))

    # mp=16 stays refused, and a WDM6/P3 pair gets WDM6's OWN refusal --
    # the lowest refused selector owns the message, and 50 is no longer
    # refused at all.
    parent = SimpleNamespace(mp_physics=16)
    child = SimpleNamespace(mp_physics=50,
                            nest_microphysics_transition=EDGE_MATRIX_POLICY)
    with pytest.raises(ValueError) as caught:
        resolve_microphysics_transition(parent, child)
    assert "MP16 (WDM6" in str(caught.value)
    assert "MP50" not in str(caught.value).replace("MP16->MP50", "")


def test_p3_entry_receipt_merges_every_frozen_species():
    """NSSL -> P3: qi/qs/qg/qh all merge into the single ice category."""
    contract = resolve_microphysics_transition(
        _run(18), _run(50, nested=True, transition=EDGE_MATRIX_POLICY))
    rows = [dict(row) for row in contract.species_actions()]
    merged = [row["source_field"] for row in rows
              if row["reason"] == "merged_into_p3_single_ice_category"]
    assert merged == ["qi", "qs", "qg", "qh"]
    diagnosed = {row["target_field"] for row in rows
                 if row["action"] == "diagnosed"}
    assert diagnosed == {"nr", "ni", "qir", "qib"}
    rime_rows = [row for row in rows if row["target_field"] in
                 ("qir", "qib") and row["action"] == "diagnosed"]
    assert all(row["reason"]
               == "p3_rime_state_diagnosed_from_source_frozen_species"
               for row in rime_rows)
    # No source MASS is dropped entering P3 -- the merge conserves it all;
    # NSSL's nine moments drop as on every other edge.
    dropped = [row["source_field"] for row in rows
               if row["action"] == "dropped"]
    assert dropped == [
        "qndrop", "qnr", "qni", "qns", "qng", "qnh", "qnn",
        "qvolg", "qvolh"]
    counts = contract.receipt()["species_action_counts"]
    assert counts == {
        "mapped": 7, "defaulted": 0, "diagnosed": 4, "dropped": 9}

    # A Kessler parent has no frozen species: qi defaults to zero and the
    # degenerate diagnosis is exact zeros (proven on the reference below).
    kessler = resolve_microphysics_transition(
        _run(1), _run(50, nested=True, transition=EDGE_MATRIX_POLICY))
    kessler_rows = [dict(row) for row in kessler.species_actions()]
    assert any(row["action"] == "defaulted"
               and row["target_field"] == "qi" for row in kessler_rows)


def test_p3_exit_receipt_splits_ice_and_consumes_the_rime_pair():
    """P3 -> Morrison: qi/qs/qg are cut from ONE category by rime state."""
    contract = resolve_microphysics_transition(
        _run(50), _run(10, nested=True, transition=EDGE_MATRIX_POLICY))
    rows = [dict(row) for row in contract.species_actions()]
    split = {row["target_field"]: row["reason"] for row in rows
             if row["action"] == "mapped" and row["source_field"] == "qi"}
    assert split == {
        "qi": "p3_unrimed_ice_after_rime_split",
        "qs": "p3_rime_split_fresh_snow_fraction",
        "qg": "p3_rime_split_dense_rime_fraction",
    }
    # The rime pair is CONSUMED by the split, never dropped: dropping it
    # would drop the only record of where the snow/graupel boundary lies.
    pair = {row["source_field"]: row["action"] for row in rows
            if row["source_field"] in ("qir", "qib")}
    assert pair == {"qir": "mapped", "qib": "mapped"}
    dropped = [row["source_field"] for row in rows
               if row["action"] == "dropped"]
    assert dropped == ["nr", "ni"]

    # A Kessler child has no ice inventory: frozen mass drops exactly as
    # it does from any other ice-bearing parent, split reasons absent.
    kessler = resolve_microphysics_transition(
        _run(50), _run(1, nested=True, transition=EDGE_MATRIX_POLICY))
    kessler_rows = [dict(row) for row in kessler.species_actions()]
    assert not any("p3_" in str(row["reason"]) for row in kessler_rows
                   if row["action"] == "mapped")
    assert sorted(row["source_field"] for row in kessler_rows
                  if row["action"] == "dropped") == [
        "ni", "nr", "qi", "qib", "qir"]


def test_p3_edge_constants_are_the_p3_ports_own():
    """Every constant in the closure is P3's or a WRF bulk density.

    The port (gpuwm/core/p3.py) is the transcription authority for
    module_mp_p3.F, so the edge constants are asserted against it rather
    than against re-typed decimals.  The rain-number closed form must be
    what P3's OWN ``get_rain_dsd2`` settles on for rain mass with no
    number, and the diagnosed ice number must be a fixed point of P3's own
    ``impose_max_total_Ni`` cap.
    """
    import numpy as np

    from gpuwm.core import microphysics_transition as mt
    from gpuwm.core import p3

    assert mt.P3_RIME_DENSITY_BOUNDS_KG_M3 == (
        float(p3.RHO_RIMEMIN), float(p3.RHO_RIMEMAX))
    assert np.float32(mt.P3_EDGE_QSMALL_KG_PER_KG) == p3.QSMALL
    assert np.float32(mt.P3_EDGE_ICE_NUCLEATION_MASS_KG) == p3.MI0
    assert mt.P3_EDGE_MAX_TOTAL_NI_PER_M3 == float(p3.MAX_TOTAL_NI)
    # Both densities inside P3's admitted rime range.
    low, high = mt.P3_RIME_DENSITY_BOUNDS_KG_M3
    assert low < mt.P3_EDGE_FRESH_SNOW_RIME_DENSITY < high
    assert low < mt.P3_EDGE_DENSE_RIME_DENSITY < high

    # get_rain_dsd2 with nr=0 floors at nsmall, hits lammin and
    # reconstructs -- the closed form is that reconstruction.
    for qr in (1.0e-6, 3.0e-5, 2.0e-3):
        settled, _mu, _lam, _cd, _ln = p3.get_rain_dsd2(
            np.float32(qr), np.float32(0.0), np.float32(1.0))
        closed = np.float32(qr) * np.float32(
            mt.P3_EDGE_RAIN_NUMBER_PER_RAIN_MASS)
        assert abs(float(settled) - float(closed)) <= 1e-5 * float(closed)

    # The diagnosed ni is inside P3's own cap for any density.
    for qitot, inv_rho in ((1.0e-6, 0.8), (5.0e-3, 1.6), (1.0e-13, 1.0)):
        entry = mt.p3_edge_entry_reference(
            0.0, 0.0, 0.0, qitot, 0.0, 0.0, 0.0, inv_rho)
        capped = p3.impose_max_total_Ni(entry["ni"], np.float32(inv_rho))
        assert capped == entry["ni"]


def test_p3_entry_reference_conserves_mass_and_holds_the_bounds():
    """qirim <= qitot and the density bound hold BY CONSTRUCTION."""
    import numpy as np

    from gpuwm.core import microphysics_transition as mt

    f = np.float32
    rng = np.random.default_rng(50)
    cases = [
        # (qv, qc, qr, qi, qs, qg, qh)
        (0.008, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5),
        (0.001, 0.0, 0.0, 0.0, 1e-3, 0.0, 0.0),      # pure snow rime
        (0.001, 0.0, 0.0, 0.0, 0.0, 2e-3, 0.0),      # pure dense rime
        (0.001, 0.0, 0.0, 5e-4, 0.0, 0.0, 0.0),      # unrimed ice only
        (0.002, 1e-9, 1e-9, 0.0, 0.0, 0.0, 0.0),     # no ice at all
        (0.002, 0.0, 0.0, 3e-15, 4e-15, 0.0, 0.0),   # sub-qsmall ice
        (0.01, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2),  # adversarial: huge
    ] + [tuple(rng.random(7) * 1e-3) for _ in range(32)]
    for qv, qc, qr, qi, qs, qg, qh in cases:
        out = mt.p3_edge_entry_reference(qv, qc, qr, qi, qs, qg, qh, 0.9)
        # Mass conservation: total water in equals total water out, in the
        # reference's own float32 chain.
        total_in = f(f(f(qv) + f(f(qi) + f(qs))) + f(f(qg) + f(qh)))
        total_out = f(f(out["qv"]) + f(out["qi"]))
        assert abs(float(total_in) - float(total_out)) <= max(
            4.0 * np.finfo(np.float32).eps * float(total_in), 1e-18)
        # The contract bounds, by construction on EVERY output.
        assert out["qir"] <= out["qi"]
        assert out["qir"] >= 0.0 and out["qib"] >= 0.0
        if out["qib"] > 0.0:
            rho_rime = float(out["qir"]) / float(out["qib"])
            assert 50.0 <= rho_rime <= 900.0
            # Entry diagnoses inside [100, 400] specifically.
            assert 100.0 * (1.0 - 1e-5) <= rho_rime <= 400.0 * (1.0 + 1e-5)
        assert out["nr"] >= 0.0 and out["ni"] >= 0.0
        assert np.isfinite(
            [float(out[name]) for name in
             ("qv", "qc", "qr", "qi", "nr", "ni", "qir", "qib")]).all()

    # Degenerate cases are EXACT.
    zero = mt.p3_edge_entry_reference(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      1.0)
    assert all(float(zero[name]) == 0.0 for name in
               ("qv", "qc", "qr", "qi", "nr", "ni", "qir", "qib"))
    unrimed = mt.p3_edge_entry_reference(
        0.0, 0.0, 0.0, 5e-4, 0.0, 0.0, 0.0, 1.0)
    assert float(unrimed["qir"]) == 0.0 and float(unrimed["qib"]) == 0.0
    assert float(unrimed["qi"]) == float(np.float32(5e-4))


def test_p3_exit_reference_splits_by_rime_state_exactly():
    """Zero ice and zero rime exact; endpoints pure; round trip inverts."""
    import numpy as np

    from gpuwm.core import microphysics_transition as mt

    f = np.float32
    # Zero ice.
    assert mt.p3_edge_exit_reference(0.0, 0.0, 0.0) == (0.0, 0.0, 0.0)
    # Zero rime: ALL mass to qi, exactly.
    ice, snow, graupel = mt.p3_edge_exit_reference(1e-3, 0.0, 0.0)
    assert (float(ice), float(snow), float(graupel)) == (
        float(f(1e-3)), 0.0, 0.0)
    # Pure fresh-snow rime density: all snow (up to fp32 rounding of the
    # diagnosed volume), none unrimed.
    entry = mt.p3_edge_entry_reference(0.0, 0.0, 0.0, 0.0, 1e-3, 0.0, 0.0,
                                       1.0)
    ice, snow, graupel = mt.p3_edge_exit_reference(
        entry["qi"], entry["qir"], entry["qib"])
    assert float(ice) == 0.0
    assert abs(float(graupel)) <= 4.0 * np.finfo(np.float32).eps * 1e-3
    assert abs(float(snow) - 1e-3) <= 4.0 * np.finfo(np.float32).eps * 1e-3
    # Pure dense rime: all graupel.
    entry = mt.p3_edge_entry_reference(0.0, 0.0, 0.0, 0.0, 0.0, 2e-3, 0.0,
                                       1.0)
    ice, snow, graupel = mt.p3_edge_exit_reference(
        entry["qi"], entry["qir"], entry["qib"])
    assert float(ice) == 0.0
    assert abs(float(snow)) <= 4.0 * np.finfo(np.float32).eps * 2e-3
    assert abs(float(graupel) - 2e-3) <= 4.0 * np.finfo(np.float32).eps * 2e-3
    # A genuine mix inverts the entry diagnosis (mass AND volume conserved
    # against the same two densities).
    qi0, qs0, qg0 = 4e-5, 5e-5, 6e-5
    entry = mt.p3_edge_entry_reference(0.0, 0.0, 0.0, qi0, qs0, qg0, 0.0,
                                       1.0)
    ice, snow, graupel = mt.p3_edge_exit_reference(
        entry["qi"], entry["qir"], entry["qib"])
    for got, want in ((ice, qi0), (snow, qs0), (graupel, qg0)):
        assert abs(float(got) - want) <= 8.0 * np.finfo(np.float32).eps * want
    # Conservation telescopes: qi + qs + qg == qitot in float32.
    total = f(f(ice + snow) + graupel)
    assert abs(float(total) - float(entry["qi"])) <= 2.0 * np.finfo(
        np.float32).eps * float(entry["qi"])


def test_p3_exit_reference_is_safe_on_adversarial_inputs():
    """The invariants hold even on state P3 itself would never produce."""
    import numpy as np

    from gpuwm.core import microphysics_transition as mt

    cases = [
        (1e-3, 2e-3, 5e-6),     # qirim > qitot: clamps to qitot
        (1e-3, 5e-4, 0.0),      # rime mass with NO volume: zero-rime rule
        (1e-3, 5e-4, 1.0),      # absurd volume: density clamps at 50
        (1e-3, 5e-4, 1e-12),    # tiny volume: density clamps at 900
        (-1e-3, -5e-4, -1e-6),  # negatives floor at zero
        (1e-3, 1e-15, 1e-17),   # sub-qsmall rime zeroes exactly
    ]
    for qitot, qirim, birim in cases:
        ice, snow, graupel = mt.p3_edge_exit_reference(qitot, qirim, birim)
        for value in (ice, snow, graupel):
            assert np.isfinite(float(value)) and float(value) >= 0.0
        clamped_total = max(float(np.float32(qitot)), 0.0)
        assert float(ice) + float(snow) + float(graupel) <= (
            clamped_total * (1.0 + 1e-5) + 1e-30)
    # qirim > qitot specifically: everything is rime, nothing negative.
    ice, snow, graupel = mt.p3_edge_exit_reference(1e-3, 2e-3, 5e-6)
    assert float(ice) == 0.0
    assert abs(float(snow) + float(graupel) - 1e-3) <= 1e-8


def test_same_scheme_p3_nesting_still_resolves_same_scheme():
    """A pure-P3 nest is untouched by the mixed-edge ratification."""
    from gpuwm.core import microphysics_transition as mt

    cfg = SimpleNamespace(
        mp_physics=50,
        nest_microphysics_transition=mt.SAME_SCHEME_POLICY)
    contract = mt.resolve_microphysics_transition(cfg, cfg)
    assert contract.mixed is False
    assert contract.source_mp_physics == contract.target_mp_physics == 50


def test_the_offline_cross_scheme_mirror_no_longer_refuses_p3():
    """The offline mirror is DERIVED, so retiring 50 online retired it
    offline in the same change -- the mirror invariant the WDM6 tests pin.

    mp=50 IS offline-readable (same-scheme P3 downscaling landed its
    qir/qib field-map rows), so the retired closure mirror cannot be
    covered by an "unreadable" gate either.  What refuses a P3
    cross-scheme edge offline is the lane's own NAMED gate
    (``_P3_OFFLINE_EDGE_UNBUILT_MP_PHYSICS``): the ratified merge/split
    maps have no offline leg wired (follow-up offline-p3-edge-closure).
    ``PARENT_SCHEME_CONTRACT`` therefore still excludes 50, now through
    that named subtraction instead of the closure mirror.
    """
    from gpuwm import offline_child as oc
    from gpuwm.core import microphysics_transition as mt

    assert (set(oc._CROSS_SCHEME_REFUSED_MP_PHYSICS)
            == set(mt.UNVALIDATED_MIXED_EDGE_SELECTORS) == {16, 28})
    assert 50 not in oc._CROSS_SCHEME_REFUSED_MP_PHYSICS
    assert 50 in oc.OFFLINE_CHILD_MP_PHYSICS
    assert oc._P3_OFFLINE_EDGE_UNBUILT_MP_PHYSICS == frozenset({50})
    assert 50 not in oc.PARENT_SCHEME_CONTRACT
    assert oc.PARENT_SCHEME_CONTRACT == frozenset({6, 8, 10, 18})


@pytest.mark.gpu
def test_p3_entry_kernel_matches_the_cpu_reference_bitwise():
    """The target_mp==50 kernel arm against p3_edge_entry_reference.

    Both sides compute in float32 with round-to-nearest per operation and
    identical operation order, so the comparison is exact, the same
    standard the MP8->MP18 translation holds itself to.
    """
    import cupy as cp

    from gpuwm.core import microphysics_transition as mt

    shape = (2, 3, 4)
    cell = cp.arange(np.prod(shape), dtype=cp.float32).reshape(shape)
    parent = SimpleNamespace(
        alt=cp.float32(1.0) / (cp.float32(0.65) + cell * cp.float32(0.02)),
        mub2d=cp.full(shape[1:], cp.float32(90000.0)),
        mup=cp.full(shape[1:], cp.float32(100.0)),
        c1h=cp.asarray([0.8, 0.6], dtype=cp.float32),
        c2h=cp.asarray([1.0, 2.0], dtype=cp.float32),
        qv=cp.float32(0.006) + cell * cp.float32(1.0e-5),
        qc=(cell % 4) * cp.float32(5.0e-6),
        qr=(cell % 5) * cp.float32(4.0e-6),
        qi=(cell % 6) * cp.float32(3.0e-6),
        qs=(cell % 7) * cp.float32(2.0e-6),
        qg=(cell % 8) * cp.float32(2.5e-6),
        qh=(cell % 9) * cp.float32(1.5e-6),
    )
    contract = mt.resolve_microphysics_transition(
        _run(18), _run(50, nested=True, transition=EDGE_MATRIX_POLICY))
    host = {name: cp.asnumpy(getattr(parent, name))
            for name in ("qv", "qc", "qr", "qi", "qs", "qg", "qh")}
    inv_rho = cp.asnumpy(parent.alt)
    for field in ("qv", "qc", "qr", "qi", "nr", "ni", "qir", "qib"):
        actual = cp.empty(shape, dtype=cp.float32)
        launch_microphysics_edge_parent_field(
            contract, parent, field, out=actual, coupled=False)
        expected = np.empty(shape, dtype=np.float32)
        for index in np.ndindex(shape):
            expected[index] = mt.p3_edge_entry_reference(
                host["qv"][index], host["qc"][index], host["qr"][index],
                host["qi"][index], host["qs"][index], host["qg"][index],
                host["qh"][index], inv_rho[index])[field]
        cp.testing.assert_array_equal(actual, expected)
    cp.cuda.Stream.null.synchronize()


@pytest.mark.gpu
def test_p3_exit_kernel_matches_the_cpu_reference_bitwise():
    """The source_mp==50 split against p3_edge_exit_reference, plus the
    target scheme's own moment closure running on the split masses."""
    import cupy as cp

    from gpuwm.core import microphysics_transition as mt

    shape = (2, 3, 4)
    cell = cp.arange(np.prod(shape), dtype=cp.float32).reshape(shape)
    parent = SimpleNamespace(
        alt=cp.full(shape, cp.float32(0.8)),
        mub2d=cp.full(shape[1:], cp.float32(90000.0)),
        mup=cp.full(shape[1:], cp.float32(100.0)),
        c1h=cp.asarray([0.8, 0.6], dtype=cp.float32),
        c2h=cp.asarray([1.0, 2.0], dtype=cp.float32),
        qv=cp.full(shape, cp.float32(0.008)),
        qc=cp.full(shape, cp.float32(2.0e-5)),
        qr=cp.full(shape, cp.float32(3.0e-5)),
        qi=cp.float32(1.0e-4) + (cell % 5) * cp.float32(5.0e-5),
        qir=(cell % 3) * cp.float32(4.0e-5),
        qib=(cell % 3) * cp.float32(4.0e-5) / cp.float32(250.0),
    )
    contract = mt.resolve_microphysics_transition(
        _run(50), _run(10, nested=True, transition=EDGE_MATRIX_POLICY))
    host = {name: cp.asnumpy(getattr(parent, name))
            for name in ("qi", "qir", "qib")}
    expected_split = {
        name: np.empty(shape, dtype=np.float32)
        for name in ("qi", "qs", "qg")}
    for index in np.ndindex(shape):
        ice, snow, graupel = mt.p3_edge_exit_reference(
            host["qi"][index], host["qir"][index], host["qib"][index])
        expected_split["qi"][index] = ice
        expected_split["qs"][index] = snow
        expected_split["qg"][index] = graupel
    for field in ("qi", "qs", "qg"):
        actual = cp.empty(shape, dtype=cp.float32)
        launch_microphysics_edge_parent_field(
            contract, parent, field, out=actual, coupled=False)
        cp.testing.assert_array_equal(actual, expected_split[field])
    # Morrison's own closure diagnoses finite non-negative numbers from
    # the split masses; positive wherever the split mass is positive.
    for field, mass in (("ns", "qs"), ("ng", "qg"), ("ni", "qi")):
        actual = cp.empty(shape, dtype=cp.float32)
        launch_microphysics_edge_parent_field(
            contract, parent, field, out=actual, coupled=False)
        values = cp.asnumpy(actual)
        assert np.isfinite(values).all()
        assert (values >= 0.0).all()
        assert (values[expected_split[mass] > 1.0e-13] > 0.0).all()
    cp.cuda.Stream.null.synchronize()

