"""ERA5's native model-level atmosphere is a registry ROW, not a route.

The hybrid closure taught both engines to read a GRIB2 Section-4 pv
coordinate ladder, materialize ``p = A + B*ps`` from it, and integrate
geopotential height hydrostatically up the resulting half levels.  That
is engine work and it is done.  What was still missing is the thing a
user actually needs: a NAME.  Until this row existed, initializing from
ERA5's 137 native model levels meant hand-authoring a mapping, a
composition, a donor mapping and a sealed manifest outside the product
and driving ``gpuwm.mapped_direct`` by hand -- engine-proven, not
shipped.

These tests hold the arbitrary acceptance test at the place it is
easiest to break: the difference between "gpuwm can decode ERA5 model
levels" and "``era5-l137`` is selectable like any other source" must be
table data -- three pinned JSON authorities plus one ``_adapter(...)``
row -- and never a runner, a module, or a branch that names this model.
"""

from __future__ import annotations

import pytest

from gpuwm.source_adapters import (AdapterStatus, SourceKind,
                                   get_source_adapter,
                                   packaged_profile_sources, source_adapters)
from gpuwm.source_authorities import packaged_profile, packaged_profile_ids


ROUTE = "era5-l137"
PROFILE = "era5-model-level-l137-grib2-v1"


def test_the_model_level_route_is_a_registered_source():
    """A name a user can type, resolvable through the same door as any
    other source id or alias."""

    adapter = get_source_adapter(ROUTE)
    assert adapter.source_id == ROUTE
    assert adapter.file_family == "GRIB2"
    assert adapter.source_kind is SourceKind.DETERMINISTIC_STATE
    # Reanalysis: every valid time is an analysis, no forecast leads.
    assert adapter.max_forecast_hour == 0
    for alias in ("era5-model-level", "era5-ml"):
        assert get_source_adapter(alias).source_id == ROUTE


def test_the_route_runs_on_a_packaged_profile_and_no_runner_of_its_own():
    """The arbitrary seam: a shipped mapping, not a code path."""

    adapter = get_source_adapter(ROUTE)
    assert adapter.runnable is True
    assert adapter.runner == "mapped_composition_v1"
    assert adapter.packaged_profile == PROFILE
    assert PROFILE in packaged_profile_ids()
    assert packaged_profile_sources()[ROUTE] == PROFILE
    profile = packaged_profile(PROFILE)
    assert profile["source_format"] == "grib2"
    # The land surface ERA5's model-level product does not publish is
    # BORROWED from the same-hour pressure-level analysis, so the profile
    # is a real cross-source composition, never a pending declaration.
    assert profile["composition_state"] != "pending_cross_source"
    assert profile["data_role"] == "physical_analysis_surface_data"


def test_the_row_declares_the_cadence_and_the_coverage():
    """The two facts `gpuwm domain` needs before any JSON is opened.

    ERA5 is a global reanalysis published hourly: the cadence is the
    hourly spacing of its analyses, and the coverage window is ``None``
    because a global product has no corner to refuse.
    """

    adapter = get_source_adapter(ROUTE)
    assert adapter.forcing_interval_seconds == 3600.0
    assert adapter.coverage_window is None
    assert adapter.status is AdapterStatus.RUNNABLE_NOT_CERTIFIED


def test_the_row_states_that_its_land_surface_is_borrowed():
    """A user who reads the registry learns the second file is required
    BEFORE paying for an acquisition, not out of a prep refusal."""

    adapter = get_source_adapter(ROUTE)
    assert adapter.composition_requirement
    assert "pressure-level" in adapter.composition_requirement


def test_the_pressure_level_row_and_the_model_level_row_are_distinct():
    """The existing ``era5`` row keeps its certified pressure-level
    identity; the model-level route does not silently take it over."""

    pressure = get_source_adapter("era5")
    model_level = get_source_adapter(ROUTE)
    assert pressure.source_id != model_level.source_id
    assert pressure.status is AdapterStatus.CERTIFIED
    assert pressure.packaged_profile is None
    assert model_level.credentials == pressure.credentials
    assert model_level.credentials, (
        "ERA5 model levels come from the same CDS account as every other "
        "ERA5 product; a row that declares no credential sends a user to "
        "an authentication failure instead of a setup step")


def test_no_module_names_this_route():
    """The row is the whole difference.  If any importable module grows
    an ``era5-l137`` branch, the arbitrary acceptance test has been lost
    and this test is where that shows up."""

    from pathlib import Path

    package = Path(__file__).parents[1] / "gpuwm"
    offenders = []
    for path in package.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if ROUTE in text and path.name != "source_adapters.py":
            offenders.append(str(path.relative_to(package)))
    assert offenders == [], (
        f"{ROUTE} reached code outside the registry table: {offenders}")


@pytest.mark.parametrize("row", source_adapters())
def test_every_runnable_row_still_declares_its_cadence(row):
    """The row added here must not be the one that breaks the property
    the wizard depends on."""

    if not row.runnable:
        return
    if row.source_id == "mapped":
        # The generic route reads cadence from the caller's own mapping.
        assert row.forcing_interval_seconds is None
        return
    assert row.forcing_interval_seconds is not None
    assert row.forcing_interval_seconds > 0.0


# ------------------------------------------------------------------
# The fetch door's answer for a source whose bytes gpuwm does not broker
# ------------------------------------------------------------------

def test_the_fetch_refusal_does_not_call_a_runnable_row_unrunnable():
    """A refusal must be TRUE before it can be useful.

    ``era5-l137`` is ``runnable=True`` -- the packaged profile decodes
    its bytes and a preparation reaches rc 0 -- but its bytes come from
    a Copernicus CDS request gpuwm does not broker.  The fetch door's
    fallback answered that with "the registry row is not runnable ...
    nothing in this ArWen could read the bytes a download produced",
    which is a false statement of cause: this ArWen reads them fine, it
    just cannot go and get them.  A reader who believes it stops.
    """

    from gpuwm import fetch_routes

    with pytest.raises(ValueError) as refusal:
        fetch_routes.route_for(ROUTE)
    text = str(refusal.value)
    assert "the registry row is not runnable" not in text
    # It must name the real breakage and a way forward.
    assert "--source-root" in text


def test_the_row_declares_why_its_bytes_are_not_downloadable():
    """Table data, like every other undownloadable source: the reason
    lives in the fetch-route authority beside 20crv3's, not in a
    fallback sentence generated from the wrong field."""

    from gpuwm import fetch_routes

    assert ROUTE in fetch_routes.refusal_ids()
    with pytest.raises(ValueError) as refusal:
        fetch_routes.route_for(ROUTE)
    text = str(refusal.value)
    assert "Copernicus" in text or "CDS" in text


# ------------------------------------------------------------------
# The prepared-forecast stage's per-source tables
# ------------------------------------------------------------------

def test_the_prepared_stage_tables_cover_every_packaged_profile():
    """A new packaged row must not raise KeyError one stage later.

    ``gpuwm/prepared_single_domain_forecast.py`` carries six per-source
    lookups that every composed mapped profile answers IDENTICALLY --
    the mapped input-manifest schema, the mapped direct and hierarchy
    proof schemas, their legacy companions and the mapped adapter id.
    Spelled as literal id lists, they turned "add a registry row" into
    "add a registry row and remember six more places", and the miss
    surfaced as a bare ``KeyError`` in the stage that runs a prepared
    bundle rather than as a refusal.  Derived from the packaged-profile
    table, they cost a new source nothing -- which is what the arbitrary
    acceptance test asks for.
    """

    from gpuwm.prepared_single_domain_forecast import (
        _HIERARCHY_PROOF_SCHEMA, _LEGACY_HIERARCHY_PROOF_SCHEMAS,
        _LEGACY_PROOF_SCHEMAS, _MAPPED_PACKAGED_PROFILE, _PROOF_SCHEMA,
        _SOURCE_ADAPTER, _SOURCE_SCHEMA)

    tables = {
        "_SOURCE_SCHEMA": _SOURCE_SCHEMA,
        "_PROOF_SCHEMA": _PROOF_SCHEMA,
        "_HIERARCHY_PROOF_SCHEMA": _HIERARCHY_PROOF_SCHEMA,
        "_SOURCE_ADAPTER": _SOURCE_ADAPTER,
        "_LEGACY_PROOF_SCHEMAS": _LEGACY_PROOF_SCHEMAS,
        "_LEGACY_HIERARCHY_PROOF_SCHEMAS": _LEGACY_HIERARCHY_PROOF_SCHEMAS,
    }
    for name, table in tables.items():
        missing = sorted(set(_MAPPED_PACKAGED_PROFILE) - set(table))
        assert missing == [], f"{name} does not answer for {missing}"


def test_the_stage_keeps_the_member_route_told_apart_from_the_rest():
    """Deriving the common rows must not flatten the one that differs:
    the 20CRv3 member route writes its OWN input-manifest schema, and
    that difference is how the stage knows to demand a member manifest.
    """

    from gpuwm.prepared_single_domain_forecast import _SOURCE_SCHEMA

    assert _SOURCE_SCHEMA["20crv3"] == "gpuwm-20crv3-grib2-inputs-v1"
    assert _SOURCE_SCHEMA[ROUTE] == "gpuwm-mapped-composition-inputs-v1"
    assert _SOURCE_SCHEMA["20crv3"] != _SOURCE_SCHEMA[ROUTE]


def test_the_forecast_stage_accepts_the_new_source_by_name():
    """`--source era5-l137` must be a name the forecast stage takes.

    ``SUPPORTED_SOURCES`` was a seventh literal id list, and the prep
    door prints a ready-to-run forecast command for any packaged source
    -- so a row missing from it would have printed a command the very
    next stage refuses.  Derived from the packaged-profile table, the
    two cannot disagree.
    """

    from gpuwm.prepared_single_domain_forecast import (
        SUPPORTED_SOURCES, _MAPPED_PACKAGED_PROFILE)

    assert ROUTE in SUPPORTED_SOURCES
    assert set(_MAPPED_PACKAGED_PROFILE) <= SUPPORTED_SOURCES
    # The generic caller-supplied-mapping id stays OUT: it has no
    # packaged certificate for this stage to check.
    assert "mapped" not in SUPPORTED_SOURCES
