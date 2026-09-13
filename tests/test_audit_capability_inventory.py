"""C-019: supported prepared sources must not disappear from capability rows."""
from gpuwm import prepared_single_domain_forecast as runner


def test_every_supported_source_has_a_reported_capability_row():
    report = runner.runner_capabilities()
    assert set(report["source_profiles"]) == set(report["supported_sources"])
    assert set(report["authority_materialization"]["source_physics_profile_ids"]) == set(runner.SUPPORTED_SOURCES)


def test_a_source_with_nothing_measured_is_priced_not_emptied():
    """C-019's other half: an empty evidence row is not the safe reading.

    An empty list here is copied into the route's ``source_template_ids``
    by tools/build_registry.py, which is the REACHABILITY declaration a
    front end reads, and
    ``gpuwm.physics_registry.expert_template_ids_for_source`` then
    withholds the route-wide expert list from a source whose normal list
    is empty.  So an empty row published "this source reaches no named
    suite at all" about two sources this runner runs: every composition
    suite and every Noah-MP expert suite vanished from aigfs and
    era5-l137 alone.  Both reach the runner through the same generic
    mapped route as their siblings, so they report what that route
    reports, with the limitation saying no source-specific verification is
    claimed.
    """

    report = runner.runner_capabilities()
    generic = report["source_profiles"]["aifs"]["physics_profile_ids"]
    assert generic, "the generic mapped basis is what the rest are priced from"
    for source in ("aigfs", "era5-l137"):
        row = report["source_profiles"][source]
        assert row["single_d01_gpu_execution"] is True
        assert row["prepared_layouts"] == ["mapped-direct-d01-v1", "mapped-hierarchy-d01-v1"]
        assert row["physics_profile_ids"] == generic
        assert "NOT an admission list" in row["physics_profile_ids_semantics"]
        assert "no source-specific WRF verification is claimed" in " ".join(row["limitations"])
        # Priced from the route, not inherited from a source: no row another
        # source's own verification minted reaches these two.
        for source_specific in (
                runner.RUC_PHYSICS_PROFILE,
                runner.MYNN_PHYSICS_PROFILE,
                runner.MYNN_RUC_PHYSICS_PROFILE,
                runner.TWENTYCRV3_WSM6_PHYSICS_PROFILE):
            assert source_specific not in row["physics_profile_ids"]


def test_the_caller_supplied_composition_is_the_only_source_naming_no_suite():
    """The one empty row left is empty because the caller states the physics.

    Kept as a row rather than a default so that "this source names no
    suite" stays something the inventory SAYS, which is what makes the
    emptiness readable instead of a source that was simply forgotten.
    """

    report = runner.runner_capabilities()
    empty = sorted(source for source, row
                   in report["source_profiles"].items()
                   if not row["physics_profile_ids"])
    assert empty == ["mapped"]
    assert "no model-specific verification is claimed" in " ".join(
        report["source_profiles"]["mapped"]["limitations"])
