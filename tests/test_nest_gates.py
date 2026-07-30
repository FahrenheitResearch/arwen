"""Pins for the pre-registered Phase-5 nest gate tables (Task 8 stage (a)).

GATES-FIRST discipline (design D7, architecture section F): the tables in
``gpuwm/verify/nest_gates.py`` land before any nesting implementation lane
merges, and a change to ANY threshold, bound kind, or table row must fail
here.  That is the point -- gates may only move by controller plan
amendment with the N1.5 attribution run in hand, and these pins force the
amendment to be visible in the diff.
"""

from dataclasses import FrozenInstanceError, fields

import pytest

from gpuwm.verify import nest_gates as ng


# ---------------------------------------------------------------------------
# The exhaustive ledger pin: every (milestone, metric, kind, threshold)
# row, in registration order.  Adding, removing, renaming, re-kinding, or
# re-numbering ANY gate fails this test.
# ---------------------------------------------------------------------------
PRE_AMENDMENT_LEDGER = (
    # N0 -- memory preflight
    ("N0", "alloc_fits_wddm_budget", "measured_bound", None),
    ("N0", "alloc_measured_le_estimate", "measured_bound", None),
    ("N0", "alloc_estimate_le_wddm_budget", "measured_bound", None),
    # N1 -- unit oracles
    ("N1", "sint_vs_fp64_mirror", "fp32_floor", None),
    ("N1", "bdy_interp1_vs_fp64_mirror", "fp32_floor", None),
    ("N1", "blend_terrain_vs_fp64_mirror", "fp32_floor", None),
    ("N1", "adjust_tempqv_vs_fp64_mirror", "fp32_floor", None),
    ("N1", "copy_fcn_vs_fp64_mirror", "fp32_floor", None),
    ("N1", "sint_quadratic_reproduction", "exact", None),
    ("N1", "sint_parent_coincident_odd_ratio", "exact", None),
    ("N1", "schedule_flat_vs_recursion", "exact", None),
    ("N1", "stepra_pin", "exact", None),
    ("N1", "chained_fp32_dt_pin", "exact", None),
    ("N1", "diff_6th_coefficient_pin", "exact", None),
    ("N1", "child_spec_exp_rejection", "exact", None),
    ("N1", "d02_hgt_blend_max_abs_diff_m", "max", 0.5),
    ("N1", "d02_mub_blend_max_abs_diff_pa", "max", 10.0),
    ("N1", "d03_hgt_blend_max_abs_diff_m", "max", 0.5),
    ("N1", "d03_mub_blend_max_abs_diff_pa", "max", 10.0),
    ("N1", "d04_hgt_blend_max_abs_diff_m", "max", 0.5),
    ("N1", "d04_mub_blend_max_abs_diff_pa", "max", 10.0),
    # N1.5 -- instrumented-WRF table oracle
    ("N1.5", "bdy_value_tables_fp32_relative", "max", 1.0e-6),
    ("N1.5", "bdy_tendency_tables_fp32_relative", "max", 1.0e-6),
    # N2 -- identity and idealized nests
    ("N2", "null_nest_r1_dry_restriction", "fp32_floor", None),
    ("N2", "identity_nest_r1_moist_wk82", "fp32_floor", None),
    ("N2", "wk_r3_boundary_zone_blowup", "max", 0.5),
    ("N2", "wk_r3_interface_crossing", "structural", None),
    ("N2", "wk_r3_interior_vs_uniform_highres", "structural", None),
    # N3 -- real d01+d02 to 13:15
    ("N3", "d01_bitwise_vs_phase4_13z", "bitwise", None),
    ("N3", "d02_mslp_pattern_correlation", "min", 0.95),
    ("N3", "d02_t500_rmse_k", "max", 2.0),
    ("N3", "d02_t850_rmse_k", "max", 2.5),
    ("N3", "d02_boundary_zone_blowup", "max", 0.5),
    ("N3", "d02_refl_10cm_structure", "structural", None),
    ("N3", "d02_hgt_blend_recheck_m", "max", 0.5),
    ("N3", "d02_mub_blend_recheck_pa", "max", 10.0),
    ("N3", "d02_blend_zone_t2_tsk_bias", "diagnostic", None),
    ("N3", "restart_split_bit_identity", "bitwise", None),
    ("N3", "two_domain_alloc_check", "measured_bound", None),
    # N4 -- +d03 ratchets
    ("N4", "d01_bitwise_vs_phase4_13z", "bitwise", None),
    ("N4", "d02_bitwise_vs_n3", "bitwise", None),
    ("N4", "d03_mslp_pattern_correlation", "min", 0.95),
    ("N4", "d03_t500_rmse_k", "max", 2.0),
    ("N4", "d03_t850_rmse_k", "max", 2.5),
    ("N4", "d03_boundary_zone_blowup", "max", 0.5),
    ("N4", "d03_refl_10cm_structure", "structural", None),
    ("N4", "d03_w_cfl_health", "structural", None),
    # N5 -- full chain
    ("N5", "d03_bitwise_vs_n4", "bitwise", None),
    ("N5", "d04_mslp_pattern_correlation", "min", 0.95),
    ("N5", "d04_t500_rmse_k", "max", 2.0),
    ("N5", "d04_t850_rmse_k", "max", 2.5),
    ("N5", "d04_boundary_zone_blowup", "max", 0.5),
    ("N5", "d04_refl_10cm_structure", "structural", None),
    ("N5", "d04_domain_mean_profiles", "structural", None),
    ("N5", "d04_updraft_intensity_distribution", "structural", None),
    ("N5", "full_tree_restart_bit_identity", "bitwise", None),
    ("N5", "tick_exact_sync_ledger", "exact", None),
    ("N5", "memory_peak_le_estimate", "measured_bound", None),
    ("N5", "estimate_le_wddm_budget", "measured_bound", None),
    ("N5", "host_overhead_fraction", "strict_max", 0.10),
    # N6 -- Super Outbreak 12 h flagship
    ("N6", "scientific_reasonability", "structural", None),
    ("N6", "d01_13z_gates_unchanged", "bitwise", None),
    ("N6", "completion_zero_validator_fires", "exact", None),
    ("N6", "output_inventory_complete", "exact", None),
    ("N6", "memory_peak_le_estimate", "measured_bound", None),
    ("N6", "mempool_abort_threshold_compliance", "measured_bound", None),
    ("N6", "mempool_soak_telemetry", "diagnostic", None),
    ("N6", "restart_3h_bitwise_resume", "bitwise", None),
    ("N6", "wall_clock_vs_goal_h", "diagnostic", None),
    # CLOSING -- Phase-5 two-way dormant closure (Task 16, executed
    # this phase; F13 amendment moved it out of P5B)
    ("CLOSING", "feedback0_bitwise_inert", "bitwise", None),
    # Phase-5b dormant two-way activation set (registered, executed 5b)
    ("P5B", "feedback1_child_region_dry_mass", "structural", None),
    ("P5B", "ht_coarse_bookkeeping", "exact", None),
    ("P5B", "even_ratio_d02_branch_in_scope", "structural", None),
)


# F17 (2026-07-16, controller amendment WITH the N1.5 attribution run in
# hand -- the register's sanctioned loosening path): the ONLY threshold
# moves ever ratified.  Evidence: chain isolation bit-exact on bundle
# inputs (0.0, full grid, all children; out/n1-measure/attribution.json),
# miss 100% the ratified Phase-3 static terrain residual (worst 1.185 m
# d03), MUB hydrostatically inherited (-11.1..-11.5 Pa/m).  Paired with
# three ADDITIVE chain_isolation_bit_exact records (strictly tightening).
THRESHOLD_AMENDMENTS = {
    ("N1", "d02_hgt_blend_max_abs_diff_m"): (0.5, 1.5),
    ("N1", "d02_mub_blend_max_abs_diff_pa"): (10.0, 17.0),
    ("N1", "d03_hgt_blend_max_abs_diff_m"): (0.5, 1.5),
    ("N1", "d03_mub_blend_max_abs_diff_pa"): (10.0, 17.0),
    ("N1", "d04_hgt_blend_max_abs_diff_m"): (0.5, 1.5),
    ("N1", "d04_mub_blend_max_abs_diff_pa"): (10.0, 17.0),
    ("N3", "d02_hgt_blend_recheck_m"): (0.5, 1.5),
    ("N3", "d02_mub_blend_recheck_pa"): (10.0, 17.0),
}

# External-review adjudication: these are the ONLY allowed transformations
# of pre-amendment rows.  Numeric thresholds remain byte-for-byte equal.
KIND_AMENDMENTS = {
    ("N1", "sint_vs_fp64_mirror"): "fp64_mirror_floor",
    ("N1", "bdy_interp1_vs_fp64_mirror"): "fp64_mirror_floor",
    ("N1", "blend_terrain_vs_fp64_mirror"): "fp64_mirror_floor",
    ("N1", "adjust_tempqv_vs_fp64_mirror"): "fp64_mirror_floor",
    ("N1", "copy_fcn_vs_fp64_mirror"): "fp64_mirror_floor",
    # F19 (p5n2attr attribution): the identity rungs re-kind to the
    # composite oracle -- the pre-registered full-state comparison set
    # two intentionally different equations (periodic parent vs Davies
    # child) against each other; the replacement is strictly sharper
    # where it binds (bit-exact tables, 2-field-scale-ULP successor,
    # Davies-formula oracle, w-substep bit identity, raw caps).
    ("N2", "null_nest_r1_dry_restriction"): "identity_oracle",
    ("N2", "identity_nest_r1_moist_wk82"): "identity_oracle",
}

# Additive rows are pinned to an exact predecessor, which makes the derived
# EXPECTED_LEDGER just as order-sensitive as a flat 84-row literal while the
# old 73-row ledger remains directly auditable.
ADDITIONS_AFTER = {
    # F17: the chain-error detector, pinned at bit-exactness.
    ("N1", "d04_mub_blend_max_abs_diff_pa"): (
        ("N1", "d02_chain_isolation_bit_exact", "exact", None),
        ("N1", "d03_chain_isolation_bit_exact", "exact", None),
        ("N1", "d04_chain_isolation_bit_exact", "exact", None),
    ),
    # F18: the successor comparison's recurrence-identity guard.
    ("N1.5", "bdy_tendency_tables_fp32_relative"): (
        ("N1.5", "producer_stored_recurrence_fp32_relative", "max", 1.0e-6),),
    ("N3", "d02_refl_10cm_structure"): (
        ("N3", "d02_refl_10cm_fss", "measured_bound", None),),
    ("N4", "d03_refl_10cm_structure"): (
        ("N4", "d03_refl_10cm_fss", "measured_bound", None),),
    ("N5", "d04_refl_10cm_structure"): (
        ("N5", "d04_refl_10cm_fss", "measured_bound", None),),
    ("N5", "d04_updraft_intensity_distribution"): (
        ("N5", "d04_updraft_intensity_percentile_band", "measured_bound",
         None),
        ("N5", "N5S_matched_physics_wrf_shadow", "measured_bound", None),
        ("N5", "N5B_d04_boundary_location_invariance", "measured_bound",
         None),
        ("N5", "ancestor_inertness_every_output", "bitwise", None),
    ),
    ("N6", "d01_13z_gates_unchanged"): (
        ("N6", "ancestor_inertness_every_output", "bitwise", None),),
    ("N6", "completion_zero_validator_fires"): (
        ("N6", "health_validator_invocations_exact", "exact", None),),
    ("N6", "output_inventory_complete"): (
        ("N6", "n6_high_frequency_products", "exact", None),
        ("N6", "d04_budget_records", "measured_bound", None),
    ),
}


def _expected_amended_ledger():
    rows = []
    for milestone, metric, kind, threshold in PRE_AMENDMENT_LEDGER:
        key = (milestone, metric)
        if key in THRESHOLD_AMENDMENTS:
            old, new = THRESHOLD_AMENDMENTS[key]
            assert threshold == old  # the pre-amendment value, verbatim
            threshold = new
        rows.append((milestone, metric, KIND_AMENDMENTS.get(key, kind),
                     threshold))
        rows.extend(ADDITIONS_AFTER.get(key, ()))
    return tuple(rows)


EXPECTED_LEDGER = _expected_amended_ledger()


def test_exhaustive_ledger():
    """Every row, in order -- any threshold/kind/inventory edit fails."""
    got = tuple((g.milestone, g.metric, g.kind, g.threshold)
                for g in ng.NEST_GATES)
    assert got == EXPECTED_LEDGER
    assert len(ng.NEST_GATES) == 88


def test_pre_amendment_rows_survive_without_threshold_loosening():
    """Every old row survives in order; the ONLY threshold moves are the
    F17 rows, each ratified with the N1.5 attribution run in hand and
    pinned old->new in THRESHOLD_AMENDMENTS."""
    got = {(g.milestone, g.metric): (i, g) for i, g in
           enumerate(ng.NEST_GATES)}
    prior_index = -1
    for milestone, metric, old_kind, old_threshold in PRE_AMENDMENT_LEDGER:
        key = (milestone, metric)
        index, rec = got[key]
        assert index > prior_index
        prior_index = index
        if key in THRESHOLD_AMENDMENTS:
            old, new = THRESHOLD_AMENDMENTS[key]
            assert old_threshold == old
            assert rec.threshold == new  # exactly the ratified F17 value
            assert "F17" in rec.convention  # the amendment is on the record
        else:
            assert rec.threshold == old_threshold  # no other threshold moved
        assert rec.kind == KIND_AMENDMENTS.get(key, old_kind)
        if key in KIND_AMENDMENTS:
            if milestone == "N1":
                assert rec.kind == "fp64_mirror_floor"
                assert "SEMANTICS-PRESERVING" in rec.convention
            else:
                assert rec.kind == "identity_oracle"
                assert "F19" in rec.convention

    # N1 retained the old 8-ULP predicate.  N2 retained that ceiling and
    # added two independent distribution guards, hence can only reject more.
    assert ng.FP64_MIRROR_MAX_ULPS == 8
    assert ng.FP32_STATE_EQUIV_MAX_ULPS == 8
    assert ng.FP32_STATE_EQUIV_P99_MAX_ULPS == 2
    assert ng.FP32_STATE_EQUIV_MEAN_SIGNED_ULP_MAX_ABS == 0.25


def test_milestone_inventory():
    assert ng.MILESTONES == ("N0", "N1", "N1.5", "N2", "N3", "N4", "N5",
                             "N6", "CLOSING", "P5B")
    counts = {m: len(ng.gates_for(m)) for m in ng.MILESTONES}
    assert counts == {"N0": 3, "N1": 21, "N1.5": 3, "N2": 5, "N3": 12,
                      "N4": 9, "N5": 18, "N6": 13, "CLOSING": 1,
                      "P5B": 3}


def test_records_frozen_and_typed():
    g = ng.NEST_GATES[0]
    with pytest.raises(FrozenInstanceError):
        g.threshold = 0.0
    assert isinstance(ng.NEST_GATES, tuple)
    assert [f.name for f in fields(ng.NestGate)] == [
        "milestone", "metric", "kind", "threshold", "convention", "anchor",
        "evaluator_commit", "mask_hash", "baseline_hash", "cadence",
        "expected_samples"]
    for rec in ng.NEST_GATES:
        assert rec.kind in ng.BOUND_KINDS
        # Numeric kinds carry a float; every other kind carries None.
        if rec.kind in ng.NUMERIC_KINDS:
            assert isinstance(rec.threshold, float)
        else:
            assert rec.threshold is None
        assert rec.convention
        assert "section F" in rec.anchor
        pins = (rec.evaluator_commit, rec.mask_hash, rec.baseline_hash,
                rec.cadence, rec.expected_samples)
        if rec.milestone in {"N5", "N6"} and rec.kind != "diagnostic":
            assert all(pin is not None for pin in pins)
        for name, pin in zip(("evaluator_commit", "mask_hash", "baseline_hash",
                              "cadence", "expected_samples"), pins):
            if name == "expected_samples":
                assert (pin is None or isinstance(pin, str) and pin
                        or isinstance(pin, int) and not isinstance(pin, bool)
                        and pin >= 0)
            else:
                assert pin is None or isinstance(pin, str) and pin
    # No duplicate (milestone, metric) rows.
    keys = [(r.milestone, r.metric) for r in ng.NEST_GATES]
    assert len(keys) == len(set(keys))


def test_named_threshold_constants():
    """The module constants the tables are built from, pinned verbatim."""
    # F17 amended values (0.5 / 10.0 pre-amendment; see
    # THRESHOLD_AMENDMENTS for the ratification evidence).
    assert ng.HGT_BLEND_MAX_ABS_DIFF_M == 1.5
    assert ng.MUB_BLEND_MAX_ABS_DIFF_PA == 17.0
    assert ng.NEST_FORCE_FP32_RELATIVE_BAND == 1.0e-6
    assert ng.MSLP_PATTERN_CORRELATION_MIN == 0.95
    assert ng.T500_RMSE_MAX_K == 2.0
    assert ng.T850_RMSE_MAX_K == 2.5
    assert ng.BOUNDARY_ZONE_BLOWUP_MAX == 0.5
    assert ng.HOST_OVERHEAD_MAX_FRACTION == 0.10
    # Comparator split: no old ceiling widened; state equivalence tightened.
    assert ng.FP32_FLOOR_MAX_ULPS == 8
    assert ng.FP64_MIRROR_MAX_ULPS == 8
    assert ng.REAL_EMULATION_MIRROR_MAX_ULPS == 8
    assert ng.FP32_STATE_EQUIV_MAX_ULPS == 8
    assert ng.FP32_STATE_EQUIV_P99_MAX_ULPS == 2
    assert ng.FP32_STATE_EQUIV_MEAN_SIGNED_ULP_MAX_ABS == 0.25
    assert ng.REFL_10CM_FSS_FAMILY == (
        (20.0, 5.0, 0.90), (30.0, 5.0, 0.80), (40.0, 5.0, 0.70))
    assert ng.F27_DOCUMENTED_DEFICIENCY_ROWS == {
        ("d03", 20.0): 0.8558,
        ("d04", 20.0): 0.8558,
    }
    assert ng.FSS_DEGENERATE_EVENT_FLOOR == 1.0e-4
    assert ng.UPDRAFT_INTENSITY_PERCENTILES == (50, 90, 99)
    assert ng.UPDRAFT_INTENSITY_RATIO_MIN == 0.80
    assert ng.UPDRAFT_INTENSITY_RATIO_MAX == 1.25


def test_comparator_semantics_per_kind():
    """F10 amendment: every bound kind carries executable comparator
    semantics; diagnostic is the ONLY non-blocking kind."""
    assert set(ng.COMPARATORS) == ng.BOUND_KINDS
    for kind, semantics in ng.COMPARATORS.items():
        assert semantics  # never empty
    # The numeric and mirror comparators define a NaN rule.
    for kind in ("max", "min", "strict_max", "fp64_mirror_floor",
                 "fp32_state_equiv", "real_emulation_mirror"):
        assert "NaN" in ng.COMPARATORS[kind]
    # Every split FP32 comparator names its executable ULP predicate.
    fp64 = ng.COMPARATORS["fp64_mirror_floor"]
    assert "ULP" in fp64 and "FP64_MIRROR_MAX_ULPS" in fp64
    assert "MAX" in fp64
    state = ng.COMPARATORS["fp32_state_equiv"]
    for phrase in ("MAX(ULP)", "P99(ULP)", "mean(signed ULP)", "ceil(0.99*N)"):
        assert phrase in state
    real = ng.COMPARATORS["real_emulation_mirror"]
    assert "dtype=np.float32" in real
    assert "REAL_EMULATION_MIRROR_MAX_ULPS" in real
    # bitwise/exact are identity comparators.
    assert "byte identity" in ng.COMPARATORS["bitwise"]
    assert "exact discrete equality" in ng.COMPARATORS["exact"]
    # structural gates BLOCK: a missing verdict is a fail, never a skip.
    assert "BLOCKING" in ng.COMPARATORS["structural"]
    assert "missing verdict" in ng.COMPARATORS["structural"]
    # diagnostic is the only kind describing itself as non-blocking.
    nonblocking = [k for k, s in ng.COMPARATORS.items()
                   if "non-blocking" in s]
    assert nonblocking == ["diagnostic"]


def test_n1_static_oracle_per_child():
    """Section F N1 as F17-amended: HGT <= 1.5 m AND MUB <= 17 Pa in the
    blend zone per child (0.5 m / 10 Pa pre-amendment; loosened with the
    attribution run in hand), PAIRED with the additive bit-exact
    chain-isolation records that now carry the chain-error detection."""
    for dom in ("d02", "d03", "d04"):
        hgt = ng.gate("N1", f"{dom}_hgt_blend_max_abs_diff_m")
        mub = ng.gate("N1", f"{dom}_mub_blend_max_abs_diff_pa")
        assert (hgt.kind, hgt.threshold) == ("max", 1.5)
        assert (mub.kind, mub.threshold) == ("max", 17.0)
        assert "blend zone" in hgt.convention
        assert "blend zone" in mub.convention
        assert f"wrfout_{dom}_1974-04-03_13_15_00" in hgt.convention
        chain = ng.gate("N1", f"{dom}_chain_isolation_bit_exact")
        assert (chain.kind, chain.threshold) == ("exact", None)
        assert "0.0" in chain.convention
        assert "F17" in chain.convention
    assert "geo_em HGT_M" in ng.gate(
        "N1", "d02_hgt_blend_max_abs_diff_m").convention


def test_n15_band_and_field_list():
    """Section F N1.5 as amended (F9/F10): FP32-relative ~1e-6 band,
    values AND tendencies, u/v/w/t/ph/mu/moist AND all active scalars
    (Morrison numbers); executable metric with per-table absolute
    fallback and NaN rule; mandatory before any threshold loosening."""
    for metric in ("bdy_value_tables_fp32_relative",
                   "bdy_tendency_tables_fp32_relative"):
        g = ng.gate("N1.5", metric)
        assert (g.kind, g.threshold) == ("max", 1.0e-6)
        assert "u/v/w/t/ph/mu/moist" in g.convention
        assert "FP32-relative" in g.convention
        # F9 amendment: scalar tables (Morrison numbers) in the oracle.
        assert "scalar" in g.convention
        assert "qni/qns/qnr/qng" in g.convention
        # F10 amendment: NaN rule present in the metric definition.
        assert "NaN" in g.convention
    value = ng.gate("N1.5", "bdy_value_tables_fp32_relative")
    assert "~1e-6" in value.convention        # the registered wording
    assert "MANDATORY" in value.convention    # ...before any loosening
    # F10: the executable metric (relative + per-table absolute fallback).
    assert "max|wrf|" in value.convention


@pytest.mark.parametrize("milestone,dom", [
    ("N3", "d02"), ("N4", "d03"), ("N5", "d04")])
def test_statistical_family_at_n3_values(milestone, dom):
    """The N3(ii) family, re-registered verbatim for d03 (N4) and d04
    (N5): 'thresholds pre-registered at the N3 values'."""
    corr = ng.gate(milestone, f"{dom}_mslp_pattern_correlation")
    t500 = ng.gate(milestone, f"{dom}_t500_rmse_k")
    t850 = ng.gate(milestone, f"{dom}_t850_rmse_k")
    bzb = ng.gate(milestone, f"{dom}_boundary_zone_blowup")
    refl = ng.gate(milestone, f"{dom}_refl_10cm_structure")
    refl_fss = ng.gate(milestone, f"{dom}_refl_10cm_fss")
    assert (corr.kind, corr.threshold) == ("min", 0.95)
    assert (t500.kind, t500.threshold) == ("max", 2.0)
    assert (t850.kind, t850.threshold) == ("max", 2.5)
    assert (bzb.kind, bzb.threshold) == ("max", 0.5)
    assert (refl.kind, refl.threshold) == ("structural", None)
    assert (refl_fss.kind, refl_fss.threshold) == ("measured_bound", None)
    fss20 = "FSS20 >= 0.90" if dom == "d02" else "FSS20 >= 0.8558"
    for phrase in (fss20, "FSS30 >= 0.80", "FSS40 >= 0.70",
                   "supplementary"):
        assert phrase in refl_fss.convention
    if dom == "d02":
        assert "F27" not in refl_fss.convention
        assert "DOCUMENTED-DEFICIENCY" not in refl_fss.convention
    else:
        for phrase in (
                "F27", "DOCUMENTED-DEFICIENCY", "below 0.8558",
                "self-revokes", "blocking at 0.8558"):
            assert phrase in refl_fss.convention
    assert "interior convention" in t850.convention
    assert f"wrfout_{dom}_1974-04-03_13_15_00" in corr.convention


def test_bitwise_ratchet_definitions():
    """Plan Task 8(a): d01 bitwise vs the anchor epoch (historic Phase-4
    metric name); d02 bitwise vs N3; d03 bitwise vs N4 -- GPUWM's
    invariance protects its registered deviation.  Davies clock bind
    (2026-07-28): the byte authority is the seam-closure anchor epoch
    real74-t7-final-r3; the retired F20 'must NOT bind' adjudication is
    gone from the convention text."""
    n3_d01 = ng.gate("N3", "d01_bitwise_vs_phase4_13z")
    assert n3_d01.kind == "bitwise"
    assert "Phase-4" in n3_d01.convention
    assert "13:00" in n3_d01.convention
    assert "real74-t7-final-r3" in n3_d01.convention
    assert "Davies clock bind" in n3_d01.convention
    assert "must NOT bind" not in n3_d01.convention
    n4_d01 = ng.gate("N4", "d01_bitwise_vs_phase4_13z")
    assert n4_d01.kind == "bitwise" and "Phase 4" in n4_d01.convention
    assert "real74-t7-final-r3" in n4_d01.convention
    n4_d02 = ng.gate("N4", "d02_bitwise_vs_n3")
    assert n4_d02.kind == "bitwise" and "N3" in n4_d02.convention
    n5_d03 = ng.gate("N5", "d03_bitwise_vs_n4")
    assert n5_d03.kind == "bitwise" and "N4" in n5_d03.convention
    n6_d01 = ng.gate("N6", "d01_13z_gates_unchanged")
    assert "real74-t7-final-r3" in n6_d01.convention
    for row in (n3_d01, n4_d01, n4_d02, n5_d03):
        assert "protects the registered non-mutating-coupling deviation" in (
            row.convention)


def test_n5_host_overhead_strict():
    g = ng.gate("N5", "host_overhead_fraction")
    assert (g.kind, g.threshold) == ("strict_max", 0.10)
    assert "25,920" in g.convention


def test_memory_contracts_are_measured_bounds():
    """N0/N5/N6: the full enforced chain measured <= estimate <= budget
    (F11/F12 amendments); the WDDM budget is measured, so no number is
    registered."""
    for milestone, metric in (
            ("N0", "alloc_fits_wddm_budget"),
            ("N0", "alloc_measured_le_estimate"),
            ("N0", "alloc_estimate_le_wddm_budget"),
            ("N3", "two_domain_alloc_check"),
            ("N5", "memory_peak_le_estimate"),
            ("N5", "estimate_le_wddm_budget"),
            ("N6", "memory_peak_le_estimate"),
            ("N6", "mempool_abort_threshold_compliance")):
        g = ng.gate(milestone, metric)
        assert (g.kind, g.threshold) == ("measured_bound", None)
    # F11: the estimate-vs-budget leg states the full chain and is
    # explicitly blocking at N0 with an N5 re-check.
    est = ng.gate("N0", "alloc_estimate_le_wddm_budget")
    assert "estimate <= measured budget" in est.convention
    assert "measured <= estimate <= budget" in est.convention
    assert "BLOCKING" in est.convention and "N5" in est.convention


def test_n2_identity_and_reflection_gates():
    null = ng.gate("N2", "null_nest_r1_dry_restriction")
    moist = ng.gate("N2", "identity_nest_r1_moist_wk82")
    # F19: composite identity oracle replaces the full-state comparison
    # that set a periodic parent against a Davies child.
    assert null.kind == "identity_oracle" and "physics off" in null.convention
    assert moist.kind == "identity_oracle"
    assert "Weisman-Klemp" in moist.convention
    for rec in (null, moist):
        assert "F19" in rec.convention
    oracle = ng.COMPARATORS["identity_oracle"]
    assert "t=0 full-state BIT identity" in oracle
    assert "2 parent-field-scale ULP" in oracle
    assert "five-point relaxation formula" in oracle
    assert "fl(w + fl(dtau*rw_t))" in oracle
    assert "h_diabatic" in oracle
    bzb = ng.gate("N2", "wk_r3_boundary_zone_blowup")
    assert (bzb.kind, bzb.threshold) == ("max", 0.5)
    # F15 amendment: the N3-family adoption is now RATIFIED in the
    # architecture/plan by controller amendment, not an implementation
    # improvisation.
    assert "N3 family value 0.5" in bzb.convention
    assert "RATIFIED" in bzb.convention
    # F10 amendment: fixture identities are pre-registered.
    for metric in ("wk_r3_boundary_zone_blowup", "wk_r3_interface_crossing",
                   "wk_r3_interior_vs_uniform_highres"):
        assert "nest_ideal_r3.py" in ng.gate("N2", metric).convention
    assert "nest_null_r1.py" in ng.gate(
        "N2", "null_nest_r1_dry_restriction").convention
    assert "nest_ideal_r1_moist.py" in ng.gate(
        "N2", "identity_nest_r1_moist_wk82").convention


def test_n1_schedule_and_clock_pins():
    sched = ng.gate("N1", "schedule_flat_vs_recursion")
    for chain in ("(1,4,3,3)", "(1,3,3)", "(1,5,3)", "(1,4,4,2)"):
        assert chain in sched.convention
    # F8 amendment: WRF's terminal stop-time feedback guard is
    # represented, and both interior AND final periods are pinned.
    assert "module_integrate.F:439-445" in sched.convention
    assert "final period" in sched.convention
    assert "12/12/12/36" in ng.gate("N1", "stepra_pin").convention
    assert "set_timekeeping.F:368" in ng.gate(
        "N1", "chained_fp32_dt_pin").convention
    assert "spec_exp=0" in ng.gate(
        "N1", "child_spec_exp_rejection").convention
    mirror = ng.gate("N1", "bdy_interp1_vs_fp64_mirror")
    assert "rdt=1.D0/cdt" in mirror.convention


def test_n6_flagship_gates():
    assert ng.gate("N6", "scientific_reasonability").kind == "structural"
    assert ng.gate("N6", "d01_13z_gates_unchanged").kind == "bitwise"
    assert ng.gate("N6", "restart_3h_bitwise_resume").kind == "bitwise"
    assert ng.gate("N6", "mempool_soak_telemetry").kind == "diagnostic"
    assert ng.gate("N6", "wall_clock_vs_goal_h").kind == "diagnostic"
    reason = ng.gate("N6", "scientific_reasonability")
    for phrase in ("supercells", "right-movers", "hook/UH", "cold pools"):
        assert phrase in reason.convention
    assert "COMPOUND" in reason.convention
    assert "d02/d03/d04 refl_10cm_fss" in reason.convention
    assert "updraft_intensity_percentile_band" in reason.convention
    assert "blinded" in reason.convention
    # F12 amendment: the Task-17 hard gates are BLOCKING ledger records.
    comp = ng.gate("N6", "completion_zero_validator_fires")
    assert comp.kind == "exact"
    assert "129,600" in comp.convention and "== 0" in comp.convention
    inv = ng.gate("N6", "output_inventory_complete")
    assert inv.kind == "exact"
    for phrase in ("REFL_10CM", "hook/UH", "d01 3600 s"):
        assert phrase in inv.convention
    peak = ng.gate("N6", "memory_peak_le_estimate")
    assert peak.kind == "measured_bound"
    assert "BLOCKING" in peak.convention
    abort = ng.gate("N6", "mempool_abort_threshold_compliance")
    assert abort.kind == "measured_bound"
    assert "BLOCKING" in abort.convention


def test_external_review_n5_blockers_are_executable():
    shadow = ng.gate("N5", "N5S_matched_physics_wrf_shadow")
    assert shadow.kind == "measured_bound"
    for phrase in ("WRF v4.6.1", "Morrison+YSU", ">= 30 min", "M >= 3",
                   "1-ULP", "low-pass state RMSE", "boundary-increment",
                   "reflectivity FSS", "storm-object timing", "E95",
                   "BLOCKING N6", "WSL"):
        assert phrase in shadow.convention
    for phrase in (
            "F28", "DOCUMENTED-EVIDENCE", "E95 == 0",
            "non-degenerate rows remain binding", "row-by-row",
            "Phase-6 convective-window extension"):
        assert phrase in shadow.convention
    boundary = ng.gate("N5", "N5B_d04_boundary_location_invariance")
    assert boundary.kind == "measured_bound"
    for phrase in ("600x600", "498x498", "central 400x400", ">= 0.90",
                   "<= 3 km", "<= 5 min", "count == 0", "[0.80, 1.25]",
                   ">=40 dBZ", "within 10 km", ">=20 min", ">=25 km^2",
                   "PRE-LOOK ENVELOPE PROCEDURE", "same-geometry 1-ULP",
                   "allowed=max", "post-look change FAILS"):
        assert phrase in boundary.convention


def test_quantitative_updraft_companion():
    structural = ng.gate("N5", "d04_updraft_intensity_distribution")
    quantitative = ng.gate(
        "N5", "d04_updraft_intensity_percentile_band")
    assert structural.kind == "structural"
    assert quantitative.kind == "measured_bound"
    for phrase in ("P50/P90/P99", "[0.80, 1.25]", "nearest-rank",
                   "wrfout_d04_1974-04-03_13_15_00", "independently blocking"):
        assert phrase in quantitative.convention


@pytest.mark.parametrize("milestone", ["N5", "N6"])
def test_ancestor_inertness_at_every_output(milestone):
    rec = ng.gate(milestone, "ancestor_inertness_every_output")
    assert rec.kind == "bitwise"
    assert "EVERY scheduled output frame" in rec.convention
    assert "no-younger-child" in rec.convention
    assert "expected_samples" in rec.convention
    if milestone == "N6":
        assert "FULL 12 h" in rec.convention


def test_n6_health_budget_and_high_frequency_blockers():
    health = ng.gate("N6", "health_validator_invocations_exact")
    assert health.kind == "exact"
    for phrase in ("validator_fire_count", "validator_invocation_count",
                   "diagnostic_frames_processed", "expected_count",
                   "expected_frames"):
        assert phrase in health.convention
    products = ng.gate("N6", "n6_high_frequency_products")
    assert products.kind == "exact"
    for phrase in ("EVERY d04 STEP", "900 s", "UH 2-5 km", "column w",
                   "10 m wind", "2 m theta-e", "finite", "540"):
        assert phrase in products.convention
    assert products.cadence == "every-d04-step accumulation; 900-s emission"
    budgets = ng.gate("N6", "d04_budget_records")
    assert budgets.kind == "measured_bound"
    for phrase in ("EACH 15-min window", "full 12 h", "1e-5",
                   "0.005", "cumulative_clipped_condensate_fraction",
                   "1e-8", "per-species", "boundary-vs-interior"):
        assert phrase in budgets.convention


def test_n5_n6_blocking_evidence_manifest_pins():
    fields_ = ("evaluator_commit", "mask_hash", "baseline_hash", "cadence",
               "expected_samples")
    for rec in ng.NEST_GATES:
        if rec.milestone not in {"N5", "N6"} or rec.kind == "diagnostic":
            continue
        for name in fields_:
            assert getattr(rec, name) is not None, (rec.metric, name)
        # Run-bound hashes/revisions are explicitly deferred, never blank.
        for name in ("evaluator_commit", "mask_hash", "baseline_hash"):
            assert getattr(rec, name) == ng.CONTROLLER_EXECUTION_SENTINEL


def test_closing_milestone():
    """F13 amendment: the Phase-5 closing gate lives in CLOSING (executed
    this phase), distinct from P5B, and defines a real control -- the
    runtime FEEDBACK-call-path bypass A/B on the same N5 state."""
    fb0 = ng.gate("CLOSING", "feedback0_bitwise_inert")
    assert fb0.kind == "bitwise"
    assert "feedback=0" in fb0.convention
    assert "bypass A/B" in fb0.convention
    assert "NOT a control" in fb0.convention   # pre/post-merge diff
    assert "Task 16" in fb0.convention
    # ...and it no longer sits in the never-executed-this-phase P5B set.
    with pytest.raises(KeyError):
        ng.gate("P5B", "feedback0_bitwise_inert")


def test_phase5b_dormant_set():
    fb1 = ng.gate("P5B", "feedback1_child_region_dry_mass")
    assert fb1.kind == "structural" and "dry mass" in fb1.convention
    assert ng.gate("P5B", "ht_coarse_bookkeeping").kind == "exact"
    even = ng.gate("P5B", "even_ratio_d02_branch_in_scope")
    assert "nri=4" in even.convention
    assert "1/(nri*nrj)" in even.convention


def test_lookup_helpers():
    assert len(ng.gates_for("N3")) == 12
    with pytest.raises(KeyError):
        ng.gates_for("N7")
    with pytest.raises(KeyError):
        ng.gate("N3", "no_such_metric")
    # Reference-frame table pins the three one-frame child references.
    assert ng.CHILD_REFERENCE_FRAMES == {
        "d02": "wrfout_d02_1974-04-03_13_15_00",
        "d03": "wrfout_d03_1974-04-03_13_15_00",
        "d04": "wrfout_d04_1974-04-03_13_15_00",
    }
