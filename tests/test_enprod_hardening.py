"""The ensemble-product failure modes, each pinned by the probe that found it.

Two silent ones (a NaN counted as a non-exceedance, and PMM moving
missingness onto the strongest feature), one loud one (a radius wider than
the domain), and one policy hole (``--accept-status`` admitting bytes
rather than members).  Every test here fails on the code as it stood.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from gpuwm.da import enprod
from gpuwm.ensemble import wrfout_inventory

_THRESHOLD = 40.0


# ------------------------------------------------------------ E-01 NMEP NaN


def test_a_nan_member_is_not_counted_as_a_non_exceedance():
    """The probe: one member, ``[10, NaN, 0]``, threshold 5.

    ``NaN > 5`` is ``False``, so the invalid member was scored a MISS
    against the full ensemble denominator -- probability understated at
    every point the bad member could reach.
    """

    stack = np.array([[[10.0, np.nan, 0.0]]])
    got = enprod.exceedance_probability(stack, 5.0, radius_cells=0.0)
    assert got[0, 0] == 1.0
    assert np.isnan(got[0, 1]), "no finite member here, so no probability"
    assert got[0, 2] == 0.0


def test_probability_does_not_fall_when_the_radius_grows_over_a_nan():
    """The probe's headline: 1 at radius 0 became 0 at radius 1.

    ``np.maximum`` propagated the NaN across the neighbourhood and the
    comparison turned every affected point into a miss, so the advertised
    structural monotonicity broke on exactly the data that most needed it.
    """

    stack = np.array([[[10.0, np.nan, 0.0]]])
    zero = enprod.exceedance_probability(stack, 5.0, radius_cells=0.0)
    one = enprod.exceedance_probability(stack, 5.0, radius_cells=1.0)
    assert_array_equal(zero[:, [0, 2]], np.array([[1.0, 0.0]]))
    assert_array_equal(one[:, [0, 2]], np.array([[1.0, 0.0]]))
    # The point whose only member has no forecast there stays blank at
    # every radius: the voting roster is a property of the point.
    assert np.isnan(zero[0, 1]) and np.isnan(one[0, 1])


def test_monotonicity_is_exact_over_a_stack_that_is_partly_missing():
    """The denominator is radius-independent, so the promise is exact.

    Masking against a neighborhood-derived roster would let the
    denominator grow with the radius and the probability fall anyway; the
    voting roster is fixed by the point's own data precisely so this
    assertion can be ``>=`` everywhere rather than ``>=`` where defined.
    """

    rng = np.random.default_rng(20260730)
    stack = rng.normal(30.0, 15.0, (6, 12, 15))
    stack[2, 4:7, 5:9] = np.nan
    stack[5, 0, :] = np.nan
    stack[:, 11, 14] = np.nan
    previous = enprod.exceedance_probability(stack, _THRESHOLD,
                                             radius_cells=0.0)
    blank = np.isnan(previous)
    assert blank[11, 14] and blank.sum() == 1
    for radius in (0.5, 1.0, 2.0, 3.5, 6.0):
        current = enprod.exceedance_probability(stack, _THRESHOLD,
                                                radius_cells=radius)
        assert_array_equal(np.isnan(current), blank), radius
        defined = ~blank
        assert np.all(current[defined] >= previous[defined]), radius
        previous = current


def test_the_refuse_policy_names_the_members_and_draws_nothing():
    stack = np.zeros((3, 2, 2))
    stack[1, 0, 0] = np.nan
    stack[2, 1, 1] = np.inf
    with pytest.raises(enprod.EnsembleRefusal) as excinfo:
        enprod.exceedance_probability(stack, 1.0, nan_policy="refuse",
                                      member_numbers=[7, 8, 9])
    message = str(excinfo.value)
    assert "member 8: 1 point(s)" in message
    assert "member 9: 1 point(s)" in message
    assert "member 7" not in message


def test_the_masked_denominator_is_the_finite_member_count():
    stack = np.array([[[10.0]], [[10.0]], [[np.nan]], [[0.0]]])
    # Three finite members, two above the threshold.
    assert enprod.exceedance_probability(stack, 5.0)[0, 0] == \
        pytest.approx(2.0 / 3.0)
    # Against the roster of four it would have been 0.5, which is the
    # understatement the NaN-as-miss policy produced.
    assert enprod.exceedance_probability(stack, 5.0)[0, 0] != 0.5


# ----------------------------------------------- NEW-01 infinity under mask

#: The probe that found it: two ordinary members and one infinite one at
#: a single point.  ``mask`` counts the infinity as missing -- the
#: coverage stamp says so -- and every reduction must agree with that
#: stamp.  ``np.nansum`` ignores NaN and PROPAGATES infinity, so mean and
#: spread both came back ``+Inf`` and PMM came back NaN at a point where
#: two finite members existed.
_INF_STACK = np.array([[[1.0]], [[2.0]], [[np.inf]]])


def test_the_coverage_report_counts_an_infinity_as_missing():
    report = enprod.missingness_report(_INF_STACK, member_numbers=[4, 5, 6])
    assert report["nonfinite_values"] == 1
    assert report["min_finite_members"] == 2
    assert report["members_affected"] == [6]
    assert report["fully_missing_points"] == 0
    assert "MASKED" in enprod.coverage_caption(report)


def test_an_infinite_member_does_not_poison_the_mean():
    got = enprod.ensemble_mean(_INF_STACK)
    assert got[0, 0] == pytest.approx(1.5), (
        "the mean of the finite members is 1.5; np.nansum let the "
        "infinity through and returned +Inf")
    assert np.all(np.isfinite(got))


def test_an_infinite_member_does_not_poison_the_spread():
    got = enprod.ensemble_spread(_INF_STACK)
    # Sample spread of {1, 2}: ddof=1, so exactly sqrt(0.5).
    assert got[0, 0] == pytest.approx(np.sqrt(0.5))
    assert np.all(np.isfinite(got))


def test_an_infinite_member_leaves_the_pmm_finite():
    got = enprod.probability_matched_mean(_INF_STACK)
    assert np.all(np.isfinite(got)), (
        "two finite members exist at this point, so the PMM has an "
        "intensity to assign; it used to be NaN")
    assert got[0, 0] in (1.0, 2.0)


def test_an_infinite_member_votes_as_missing_in_the_probability():
    got = enprod.exceedance_probability(_INF_STACK, 1.5)
    assert got[0, 0] == pytest.approx(0.5), (
        "one of the two finite members exceeds 1.5; the infinite member "
        "does not vote at all")


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_every_non_finite_value_is_masked_the_same_way(bad):
    """One policy, whichever flavour of non-finite arrives.

    Keyed on the kind of non-finite value because that is exactly the
    dimension the bug lived on: NaN was excluded and infinity was not,
    from the same array, under the same stated policy.
    """
    stack = np.array([[[1.0]], [[2.0]], [[bad]]])
    assert enprod.ensemble_mean(stack)[0, 0] == pytest.approx(1.5)
    assert enprod.ensemble_spread(stack)[0, 0] == pytest.approx(
        np.sqrt(0.5))
    assert np.isfinite(enprod.probability_matched_mean(stack)).all()
    assert enprod.exceedance_probability(stack, 1.5)[0, 0] == \
        pytest.approx(0.5)


def test_an_infinity_still_refuses_under_the_refuse_policy():
    stack = np.array([[[1.0]], [[np.inf]]])
    with pytest.raises(enprod.EnsembleRefusal, match="member 5: 1 point"):
        enprod.ensemble_mean(stack, nan_policy="refuse",
                             member_numbers=[4, 5])


def test_a_point_where_only_infinities_live_stays_blank():
    stack = np.array([[[1.0, np.inf]], [[2.0, -np.inf]]])
    mean = enprod.ensemble_mean(stack)
    assert mean[0, 0] == pytest.approx(1.5)
    assert np.isnan(mean[0, 1]), "no finite member here, so nothing to say"
    assert np.isnan(enprod.probability_matched_mean(stack)[0, 1])
    assert np.isnan(enprod.exceedance_probability(stack, 0.0)[0, 1])


def test_a_panel_drawn_over_an_infinity_is_finite_and_says_so(tmp_path):
    """The rendered artifact, not only the array.

    A mean panel whose colour limits come from an infinite field either
    fails to draw or draws a single flat colour; either way the stamp on
    it would claim two-thirds coverage of something that is not there.
    """

    pytest.importorskip("matplotlib")

    rng = np.random.default_rng(20260730)
    stack = rng.normal(30.0, 10.0, (3, 6, 7))
    stack[2, 3, 4] = np.inf
    report = enprod.missingness_report(stack, member_numbers=[0, 1, 2])
    assert report["nonfinite_values"] == 1
    caption = enprod.coverage_caption(report)
    assert caption is not None

    lat = np.tile(np.linspace(38.0, 40.0, 6)[:, None], (1, 7))
    lon = np.tile(np.linspace(-98.0, -95.0, 7)[None, :], (6, 1))
    context = enprod.ensemble_plot_context(
        "d02", 3000.0, "1970-01-01_18:00:00", "src", 3, notes=(caption,))
    plt = enprod._pyplot()
    for product, render in (("mean", enprod.render_mean),
                            ("spread", enprod.render_spread),
                            ("pmm", enprod.render_pmm)):
        out_png = tmp_path / f"{product}.png"
        render(stack, enprod.FIELDS["refl"], lat=lat, lon=lon, plt=plt,
               context=context, out_png=out_png, dpi=60)
        assert out_png.is_file() and out_png.stat().st_size > 0
    assert np.all(np.isfinite(enprod.ensemble_mean(stack)))
    assert np.all(np.isfinite(enprod.ensemble_spread(stack)))
    assert np.all(np.isfinite(enprod.probability_matched_mean(stack)))


# ------------------------------------------------------------- E-02 PMM NaN


def test_pmm_leaves_missingness_where_it_is():
    """The probe: the panel's strongest feature was replaced by the NaN.

    Reversing the pooled sort put NaNs at the front; the stable descending
    sort of the mean put its NaN ranks at the back; so ``out[order] =
    picked`` assigned the pooled NaN to the HIGHEST finite mean point.
    """

    members = np.array([[[100.0, 1.0, np.nan]],
                        [[90.0, 2.0, 0.0]]])
    got = enprod.probability_matched_mean(members)
    assert got[0, 0] == 100.0, (
        "the strongest mean point must carry the pooled maximum; it used "
        "to be handed the pooled NaN")
    assert np.all(np.isfinite(got)), (
        "every point here has at least one finite member, so masking "
        "leaves nothing blank")
    # The pooled draw is over the finite values only: 100, 90, 2, 1, 0.
    assert sorted(got.ravel(), reverse=True) == [100.0, 90.0, 1.0]


def test_pmm_leaves_a_fully_missing_point_blank():
    members = np.array([[[100.0, 1.0, np.nan]],
                        [[90.0, 2.0, np.nan]]])
    got = enprod.probability_matched_mean(members)
    assert np.isnan(got[0, 2]), "no finite member here, so no intensity"
    assert got[0, 0] == 100.0
    assert np.isfinite(got[0, 1])


def test_pmm_is_ebert_exactly_when_nothing_is_missing():
    """The masked path must reduce to the every-M-th stride."""

    rng = np.random.default_rng(4)
    stack = rng.gamma(2.0, 8.0, (6, 9, 11))
    n_members = stack.shape[0]
    pooled = np.sort(stack.reshape(-1))[::-1]
    picked = pooled[::n_members][:stack[0].size]
    order = np.argsort(-stack.mean(axis=0).reshape(-1), kind="stable")
    reference = np.empty(stack[0].size)
    reference[order] = picked
    assert_array_equal(enprod.probability_matched_mean(stack),
                       reference.reshape(stack.shape[1:]))


def test_pmm_pools_only_finite_values():
    stack = np.array([[[5.0, 4.0, np.nan]],
                      [[3.0, 2.0, 1.0]]])
    got = enprod.probability_matched_mean(stack)
    finite = got[np.isfinite(got)]
    assert set(finite) <= {5.0, 4.0, 3.0, 2.0, 1.0}
    assert finite.max() == 5.0, "the pooled maximum is retained"
    assert not np.isnan(got).any()


# ------------------------------------------------------- E-05 PMM tie rule


def test_the_flat_index_tie_rule_is_named_and_pinned():
    """A four-cell plateau gets [3, 2, 1, 0]: deterministic, and an artifact.

    Ebert prescribes no tie policy, and this one keeps the defining
    property -- the output's value distribution IS the pooled members' one.
    It is pinned so it cannot change quietly, and reported so a reader can
    see how much of a panel is plateau.
    """

    stack = np.array([[[3.0, 2.0], [1.0, 0.0]],
                      [[0.0, 1.0], [2.0, 3.0]]])
    assert_array_equal(stack.mean(axis=0), np.full((2, 2), 1.5))
    got = enprod.probability_matched_mean(stack)
    assert_array_equal(got, np.array([[3.0, 2.0], [1.0, 0.0]]))
    report = enprod.pmm_tie_report(stack)
    assert report["tied_points"] == 4
    assert report["largest_tie_group"] == 4
    assert report["tied_fraction"] == 1.0


def test_the_average_tie_rule_removes_the_artificial_gradient():
    stack = np.array([[[3.0, 2.0], [1.0, 0.0]],
                      [[0.0, 1.0], [2.0, 3.0]]])
    got = enprod.probability_matched_mean(stack, tie_rule="average")
    assert_array_equal(got, np.full((2, 2), 1.5))
    # And it is opt-in precisely because it gives up the pooled
    # distribution the flat-index rule preserves exactly.
    assert set(np.unique(got)) != set(np.unique(stack))


def test_an_unknown_tie_rule_is_refused():
    with pytest.raises(ValueError, match="unknown pmm tie rule"):
        enprod.probability_matched_mean(np.zeros((2, 2, 2)),
                                        tie_rule="whatever")


# -------------------------------------------------- E-04 radius past domain


def test_a_radius_wider_than_the_domain_clips_instead_of_crashing():
    """The probe: radius 4 on a (members, 3, 5) stack raised a broadcast
    error, because the source and destination slices had different empty
    shapes."""

    stack = np.zeros((2, 3, 5))
    stack[0, 1, 2] = 100.0
    got = enprod.exceedance_probability(stack, 50.0, radius_cells=4.0)
    # A domain-spanning radius sees the one hit from every point.
    assert_array_equal(got, np.full((3, 5), 0.5))
    # And still monotone through the crossing radius.
    previous = enprod.exceedance_probability(stack, 50.0, radius_cells=0.0)
    for radius in (1.0, 2.0, 3.0, 3.1, 4.0, 9.0, 40.0):
        current = enprod.exceedance_probability(stack, 50.0,
                                                radius_cells=radius)
        assert np.all(current >= previous), radius
        previous = current


def test_the_footprint_saturates_at_the_whole_domain():
    footprint = enprod.neighborhood_footprint((3, 5), 40.0)
    assert np.all(footprint == 15)


# ----------------------------------------------------- E-03 status override


_STAMPS = ("1970-01-01_18:00:00", "1970-01-01_19:00:00")


def _write_manifest(root, records, *, status="COMPLETE"):
    (root / enprod.MANIFEST_FILENAME).write_text(json.dumps({
        "schema": enprod.MANIFEST_SCHEMA,
        "status": status,
        "n_members": len(records),
        "members": records,
    }, indent=2), encoding="utf-8")


def _write_member_file(root, number, *, name="wrfout_d02_1970-01-01_18-00-00",
                       body=b"member bytes"):
    """A file under the member directory, standing in for a wrfout.

    Deliberately not a real wrfout: what
    :func:`enprod.verify_override_inventory` does with the file is hash
    it and compare its size, and a 200 kB netCDF here would test netCDF4
    rather than the binding.  The frames it is supposed to carry come
    from the indexer, which these tests supply directly.
    """
    directory = root / f"member_{number:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(body)
    return path


def _inventory_entry(path, root, number, *, stamps=_STAMPS[:1]):
    """The manifest inventory entry the writer would have recorded."""
    return {
        "contract": wrfout_inventory.WRFOUT_INVENTORY_CONTRACT,
        "path": wrfout_inventory.canonical_relative_path(
            path, root / f"member_{number:03d}"),
        "domain": wrfout_inventory.domain_token(path),
        "frames": [{"index": index, "valid_time": stamp}
                   for index, stamp in enumerate(stamps)],
        "size_bytes": path.stat().st_size,
        "sha256": wrfout_inventory.file_sha256(path),
    }


def _member_record(number, *, status="DONE", inventory=None,
                   wrfout_count="from-inventory"):
    record = {"index": number, "member_dir": f"member_{number:03d}",
              "status": status}
    if inventory is not None:
        record[wrfout_inventory.WRFOUT_INVENTORY_KEY] = inventory
    if wrfout_count == "from-inventory":
        wrfout_count = None if inventory is None else len(inventory)
    if wrfout_count is not None:
        record["wrfout_count"] = wrfout_count
    return record


def _bound_ensemble(root, *, failed_body=b"member bytes",
                    declared_body=None, status="FAILED"):
    """Two members, member 1 FAILED, both with declared inventories.

    ``declared_body`` is what the manifest's inventory was computed over;
    ``failed_body`` is what is actually on disk for member 1.  Passing
    two different values is the stale-bytes probe.
    """
    records = []
    for number in range(2):
        body = failed_body if number == 1 else b"member zero bytes"
        path = _write_member_file(root, number, body=body)
        if number == 1 and declared_body is not None:
            # Compute the declared inventory over the bytes the writer
            # saw, then put the stale bytes back on disk.
            path.write_bytes(declared_body)
            entry = _inventory_entry(path, root, number)
            path.write_bytes(body)
        else:
            entry = _inventory_entry(path, root, number)
        records.append(_member_record(
            number, status="FAILED" if number == 1 else "DONE",
            inventory=[entry]))
    _write_manifest(root, records, status=status)
    manifest = enprod.load_manifest(root, accept_status=("DONE", "FAILED"))
    indexed = [
        _FakeFrames(member,
                    {_STAMPS[0]: (root / member.directory.name
                                  / "wrfout_d02_1970-01-01_18-00-00", 0)})
        for member in manifest.members]
    return manifest, indexed


def test_an_overridden_member_is_flagged_as_an_override(tmp_path):
    for number in range(2):
        (tmp_path / f"member_{number:03d}").mkdir(parents=True)
    _write_manifest(tmp_path, [_member_record(0),
                               _member_record(1, status="FAILED")],
                    status="FAILED")
    manifest = enprod.load_manifest(tmp_path,
                                    accept_status=("DONE", "FAILED"))
    assert [m.number for m in manifest.overridden] == [1]
    assert manifest.status == "FAILED"
    assert manifest.members[1].overridden is True
    assert manifest.members[0].overridden is False


def test_a_same_count_stale_replacement_is_refused(tmp_path):
    """THE probe that reopened E-03, promoted to a permanent test.

    One declared file, one file present: the count check the first fix
    installed passed, and ``verify_override_inventory`` reported
    ``overridden=true, verified=[1], unverifiable=[]`` over arbitrary
    stale bytes.  The bytes are what the inventory binds now, so the same
    substitution is a refusal that names the member, the file, and both
    hashes.
    """

    manifest, indexed = _bound_ensemble(
        tmp_path, declared_body=b"the bytes the run wrote",
        failed_body=b"stale bytes from an earlier attempt")
    with pytest.raises(enprod.EnsembleRefusal) as excinfo:
        enprod.verify_override_inventory(manifest, indexed)
    message = str(excinfo.value)
    assert "member 1" in message
    assert "not the same bytes" in message
    assert "wrfout_d02" in message


def test_a_stale_replacement_of_the_same_length_is_still_refused(tmp_path):
    """Same count AND same size: only the hash can tell these apart."""

    manifest, indexed = _bound_ensemble(
        tmp_path, declared_body=b"AAAAAAAAAAAAAAAA",
        failed_body=b"BBBBBBBBBBBBBBBB")
    with pytest.raises(enprod.EnsembleRefusal, match="not the same bytes"):
        enprod.verify_override_inventory(manifest, indexed)


def test_an_admitted_member_with_a_matching_inventory_is_stamped(tmp_path):
    manifest, indexed = _bound_ensemble(tmp_path)
    report = enprod.verify_override_inventory(manifest, indexed)
    assert report["overridden"] is True
    assert report["verified"] == [1]
    assert report["contract"] == wrfout_inventory.WRFOUT_INVENTORY_CONTRACT
    assert report["detail"][0]["bound_frames"] == 1
    caption = enprod.override_caption(report)
    assert "ROSTER OVERRIDE" in caption
    assert "1:FAILED" in caption
    assert "1 frame(s) bound" in caption
    assert "sha256" in caption
    assert "ensemble status FAILED" in caption
    # And the caption reaches the panel.
    context = enprod.ensemble_plot_context(
        "d02", 3000.0, "1970-01-01_18:00:00", "src", 2,
        notes=(None, caption))
    assert caption in context.splitlines()


def test_a_member_with_no_declared_inventory_is_refused_not_admitted(
        tmp_path):
    """"Unverifiable" used to be a state that still rendered.

    A FAILED member the writer never finished declares no inventory.  The
    previous behaviour admitted it, stamped "inventory unverifiable" on
    the panel, and put its bytes into the mean -- which is exactly the
    unverified-bytes outcome the whole check exists to prevent.
    """

    for number in range(2):
        _write_member_file(tmp_path, number)
    _write_manifest(tmp_path, [
        _member_record(0),
        {"index": 1, "member_dir": "member_001", "status": "FAILED",
         "error": {"type": "RuntimeError", "message": "device fell over"}},
    ], status="FAILED")
    manifest = enprod.load_manifest(tmp_path,
                                    accept_status=("DONE", "FAILED"))
    indexed = [
        _FakeFrames(member, {_STAMPS[0]: (
            tmp_path / member.directory.name
            / "wrfout_d02_1970-01-01_18-00-00", 0)})
        for member in manifest.members]
    with pytest.raises(enprod.EnsembleRefusal) as excinfo:
        enprod.verify_override_inventory(manifest, indexed)
    message = str(excinfo.value)
    assert "member 1" in message
    assert "declares no" in message
    assert wrfout_inventory.WRFOUT_INVENTORY_KEY in message


def test_an_explicitly_empty_inventory_cannot_verify_vacuously(tmp_path):
    """RV4-02: zero checked files and frames is no verification evidence.

    This is the exact shape that re-verification #4 found: the inventory is
    present but explicitly empty, its declared count is zero, and the
    indexer supplied no ``MemberFrames`` for the overridden member.  Every
    old loop was empty, so the member appeared in ``verified`` with zero
    bound frames.
    """

    for number in range(2):
        (tmp_path / f"member_{number:03d}").mkdir(parents=True)
    _write_manifest(tmp_path, [
        _member_record(0),
        _member_record(1, status="FAILED", inventory=[],
                       wrfout_count=0),
    ], status="FAILED")
    manifest = enprod.load_manifest(
        tmp_path, accept_status=("DONE", "FAILED"))

    with pytest.raises(enprod.EnsembleRefusal) as excinfo:
        enprod.verify_override_inventory(manifest, [])
    message = str(excinfo.value)
    assert "member 1" in message
    assert "bound 0 file(s) and 0 frame(s)" in message
    assert "no evidence" in message


def test_an_undeclared_extra_wrfout_is_refused(tmp_path):
    """The stale run's OTHER shape: a file the writer never declared.

    ``index_member_frames`` globs ``wrfout*`` recursively, so a leftover
    file from an earlier attempt supplies real frames at real valid
    times.  Binding the indexed frames to the declared inventory is what
    keeps it out; a count would have grown to match it.
    """

    manifest, indexed = _bound_ensemble(tmp_path)
    extra = _write_member_file(tmp_path, 1,
                               name="wrfout_d02_1970-01-01_19-00-00",
                               body=b"an earlier attempt")
    indexed[1].frames[_STAMPS[1]] = (extra, 0)
    with pytest.raises(enprod.EnsembleRefusal,
                       match="does not.*declare"):
        enprod.verify_override_inventory(manifest, indexed)


def test_a_frame_at_the_wrong_index_is_refused(tmp_path):
    """The inventory binds (valid time -> frame), not just the file."""

    manifest, indexed = _bound_ensemble(tmp_path)
    path, _ = indexed[1].frames[_STAMPS[0]]
    indexed[1].frames[_STAMPS[0]] = (path, 3)
    with pytest.raises(enprod.EnsembleRefusal, match="at frame 3"):
        enprod.verify_override_inventory(manifest, indexed)


@pytest.mark.parametrize("escape", [
    "../member_000/wrfout_d02_1970-01-01_18-00-00",
    "/etc/passwd",
    "\\\\server\\share\\wrfout",
])
def test_an_inventory_path_that_leaves_the_member_directory_is_refused(
        tmp_path, escape):
    """The inventory names THIS member's files, relative, and nothing else.

    ``WindowsPath('/x').is_absolute()`` is False, so a leading separator
    has to be tested on its own; without that a manifest could point the
    verifier at a file outside the ensemble entirely and have it hash
    that instead.
    """

    manifest, indexed = _bound_ensemble(tmp_path)
    document = json.loads(
        (tmp_path / enprod.MANIFEST_FILENAME).read_text(encoding="utf-8"))
    document["members"][1][
        wrfout_inventory.WRFOUT_INVENTORY_KEY][0]["path"] = escape
    (tmp_path / enprod.MANIFEST_FILENAME).write_text(
        json.dumps(document, indent=2), encoding="utf-8")
    manifest = enprod.load_manifest(tmp_path,
                                    accept_status=("DONE", "FAILED"))
    with pytest.raises(enprod.EnsembleRefusal, match="not a relative path"):
        enprod.verify_override_inventory(manifest, indexed)


def test_a_missing_declared_file_is_refused(tmp_path):
    manifest, indexed = _bound_ensemble(tmp_path)
    (tmp_path / "member_001" / "wrfout_d02_1970-01-01_18-00-00").unlink()
    indexed[1].frames.clear()
    with pytest.raises(enprod.EnsembleRefusal, match="not present"):
        enprod.verify_override_inventory(manifest, indexed)


def test_a_count_that_disagrees_with_the_inventory_warns(tmp_path,
                                                         capsys):
    """Warn-not-block: the per-file sha256 binding is the identity
    proof; the redundant scalar's drift is named and the binding
    continues to decide."""

    manifest, indexed = _bound_ensemble(tmp_path)
    document = json.loads(
        (tmp_path / enprod.MANIFEST_FILENAME).read_text(encoding="utf-8"))
    document["members"][1]["wrfout_count"] = 3
    (tmp_path / enprod.MANIFEST_FILENAME).write_text(
        json.dumps(document, indent=2), encoding="utf-8")
    manifest = enprod.load_manifest(tmp_path,
                                    accept_status=("DONE", "FAILED"))
    report = enprod.verify_override_inventory(manifest, indexed)
    err = capsys.readouterr().err
    assert "warning:" in err and "wrfout_count=3" in err
    assert report["overridden"] is True


def test_a_default_roster_carries_no_override_stamp(tmp_path):
    for number in range(2):
        (tmp_path / f"member_{number:03d}").mkdir(parents=True)
    _write_manifest(tmp_path, [_member_record(0), _member_record(1)])
    manifest = enprod.load_manifest(tmp_path)
    report = enprod.verify_override_inventory(manifest, [])
    assert report == {"overridden": False}
    assert enprod.override_caption(report) is None


class _FakeFrames:
    """Just enough of ``MemberFrames`` for the inventory check."""

    def __init__(self, member, frames):
        self.member = member
        self.frames = dict(frames)


# ------------------------------------------------------------- end to end


def _fixture_with_failed_member(tmp_path, *, drop_inventory=False,
                                stale=False):
    """A real fixture ensemble with member 1 marked FAILED.

    The manifest is the one ``write_synthetic_ensemble`` writes, which
    records the frame inventory through the SAME
    ``gpuwm.ensemble.wrfout_inventory.member_inventory`` the engine calls
    when a real member finishes -- so this exercises the contract the
    writer lane produces, not a shape invented here.
    """
    root = tmp_path / "ens"
    enprod.write_synthetic_ensemble(root, n_members=3)
    path = root / enprod.MANIFEST_FILENAME
    document = json.loads(path.read_text(encoding="utf-8"))
    document["status"] = "FAILED"
    document["members"][1]["status"] = "FAILED"
    document["members"][1]["error"] = {"type": "RuntimeError",
                                       "message": "device fell over"}
    if drop_inventory:
        document["members"][1].pop(wrfout_inventory.WRFOUT_INVENTORY_KEY,
                                   None)
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    if stale:
        # Append to the member's wrfout: still one file, still readable,
        # different bytes.  This is the stale-replacement case in the
        # shape an operator actually meets it.
        declared = document["members"][1][
            wrfout_inventory.WRFOUT_INVENTORY_KEY][0]
        target = root / "member_001" / declared["path"]
        with open(target, "ab") as stream:
            stream.write(b"\0" * 64)
    return root


def test_a_failed_member_with_stale_bytes_renders_nothing(tmp_path):
    """The end-to-end probe: --accept-status used to admit stale bytes.

    Default is still fail-closed; the override binds the admitted
    member's files to the manifest's sha256 inventory BEFORE anything is
    drawn, so a stale or partial run refuses instead of quietly entering
    the mean.  The count check this replaced could not see it: one file
    was declared and one file was there.
    """

    pytest.importorskip("netCDF4")
    pytest.importorskip("wrf")
    pytest.importorskip("matplotlib")

    root = _fixture_with_failed_member(tmp_path, stale=True)
    outdir = tmp_path / "out"
    with pytest.raises(enprod.EnsembleRefusal, match="not the same bytes"):
        enprod.run_suite(root, fields=("refl",), products=("mean",),
                         thresholds=(40.0,), radii=(0.0,), domain=None,
                         timeidx=0, outdir=outdir, dpi=60,
                         accept_status=("DONE", "FAILED"))
    assert not list(outdir.glob("*.png")) if outdir.exists() else True


def test_a_failed_member_with_no_inventory_renders_nothing(tmp_path):
    pytest.importorskip("netCDF4")
    pytest.importorskip("wrf")
    pytest.importorskip("matplotlib")

    root = _fixture_with_failed_member(tmp_path, drop_inventory=True)
    outdir = tmp_path / "out"
    with pytest.raises(enprod.EnsembleRefusal, match="declares no"):
        enprod.run_suite(root, fields=("refl",), products=("mean",),
                         thresholds=(40.0,), radii=(0.0,), domain=None,
                         timeidx=0, outdir=outdir, dpi=60,
                         accept_status=("DONE", "FAILED"))
    assert not list(outdir.glob("*.png")) if outdir.exists() else True


def test_a_failed_member_that_is_admitted_is_stamped_on_the_panel(tmp_path):
    pytest.importorskip("netCDF4")
    pytest.importorskip("wrf")
    pytest.importorskip("matplotlib")

    root = _fixture_with_failed_member(tmp_path)
    provenance: dict = {}
    written, failures = enprod.run_suite(
        root, fields=("refl",), products=("mean",), thresholds=(40.0,),
        radii=(0.0,), domain=None, timeidx=0, outdir=tmp_path / "out",
        dpi=60, accept_status=("DONE", "FAILED"), provenance=provenance)
    assert written and not failures
    override = provenance["roster_override"]
    assert override["overridden"] is True
    assert override["members"] == [1]
    assert override["verified"] == [1]
    assert override["ensemble_status"] == "FAILED"
    # Both frames of the fixture's wrfout are bound, not merely the file.
    assert override["detail"][0]["bound_frames"] == 2
    assert "ROSTER OVERRIDE" in enprod.override_caption(override)


def test_the_default_roster_skips_a_failed_member_with_a_warning(
        tmp_path, capsys):
    pytest.importorskip("netCDF4")
    pytest.importorskip("wrf")
    pytest.importorskip("matplotlib")

    root = _fixture_with_failed_member(tmp_path)
    written, failures = enprod.run_suite(
        root, fields=("refl",), products=("mean",),
        thresholds=(40.0,), radii=(0.0,), domain=None,
        timeidx=0, outdir=tmp_path / "out", dpi=60)
    err = capsys.readouterr().err
    assert written and not failures
    assert "warning:" in err and "'FAILED'" in err
    assert "--accept-status" in err


def test_the_suite_stamps_coverage_and_records_it(tmp_path):
    pytest.importorskip("netCDF4")
    pytest.importorskip("wrf")
    pytest.importorskip("matplotlib")

    root = tmp_path / "ens"
    enprod.write_synthetic_ensemble(root, n_members=3)
    provenance: dict = {}
    written, failures = enprod.run_suite(
        root, fields=("refl",), products=("mean", "prob"), thresholds=(40.0,),
        radii=(0.0,), domain=None, timeidx=0, outdir=tmp_path / "out",
        dpi=60, provenance=provenance)
    assert written and not failures
    assert provenance["nan_policy"] == "mask"
    assert provenance["pmm_tie_rule"] == "flat-index"
    assert provenance["roster_override"] == {"overridden": False}
    report = next(iter(provenance["missingness"].values()))["refl"]
    assert report["nonfinite_values"] == 0
    assert report["coverage"] == 1.0
