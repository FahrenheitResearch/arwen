"""Resolution-level proofs for the packaged acquisition route table.

Nothing here moves a byte.  These tests hold the table (which host, which
key, which lead ladder, which file-set composition) against the shapes
measured on the live services on 2026-08-17, and hold the front door's
coverage against the registry: every runnable source either resolves a
route or refuses by name.

The live smoke -- one small real object per route -- is
``tests/test_fetch_routes_live.py``.
"""

from __future__ import annotations

from datetime import datetime
import json

import pytest

from gpuwm import fetch_routes, source_adapters


# --------------------------------------------------------------------------
# Coverage: the registry and the route table agree about every source
# --------------------------------------------------------------------------

def test_every_runnable_source_either_routes_or_refuses_by_name():
    """No runnable source may fall off the front door silently.

    The breakage this prevents is the one the 6 h battery reported: a
    source with a runnable profile, a passing sim, and no way to get its
    bytes -- engine-proven, not shipped.
    """

    handled = set(fetch_routes.route_ids()) | set(fetch_routes.refusal_ids())
    # gfs/hrrr/era5 keep their own hand-written routes in gpuwm.fetch.
    handled |= set(fetch_routes.LEGACY_ROUTE_SOURCES)
    runnable = {row.source_id for row in source_adapters.source_adapters()
                if row.runnable}
    assert runnable - handled == set()


def test_a_registered_source_with_no_route_refuses_naming_the_registry():
    """A non-runnable row is refused for the reason it is not runnable."""

    with pytest.raises(ValueError) as error:
        fetch_routes.route_for("nam")
    message = str(error.value)
    assert "nam" in message
    # Names WHY, not merely that it is unknown.
    assert "runnable" in message
    assert "gpuwm sources" in message


def test_an_unregistered_name_is_refused_with_the_registry_listing():
    with pytest.raises(ValueError) as error:
        fetch_routes.route_for("not-a-model")
    message = str(error.value)
    assert "not-a-model" in message
    assert "rap" in message and "icon-eu" in message


def test_aliases_resolve_to_their_registry_row():
    assert fetch_routes.route_for("gdps").source_id == "gem-gdps"
    assert fetch_routes.route_for("ifs").source_id == "ecmwf-open-data"
    assert fetch_routes.route_for("hrrr-wrfprs").source_id == "hrrr-prs"


# --------------------------------------------------------------------------
# Table integrity
# --------------------------------------------------------------------------

def test_the_packaged_route_table_matches_its_pin():
    """The wheel's copy is the copy this code was written against."""

    assert (fetch_routes.packaged_route_table_sha256()
            == fetch_routes.ROUTE_TABLE_SHA256)


def test_every_route_declares_an_endpoint_ladder_and_known_tokens():
    """A route's hosts are an ORDERED ladder, and every rung says why.

    The `default: true` marker used to pick one host out of a set.  It
    now marks the ladder's HEAD, which the loader binds to position 0,
    so the table cannot declare a default it would never ask first.
    """

    for source_id in fetch_routes.route_ids():
        route = fetch_routes.route_for(source_id)
        assert route.hosts, source_id
        assert all(host.why.strip() for host in route.hosts), source_id
        assert route.host(None) is route.hosts[0], source_id
        for row in route.files:
            unknown = fetch_routes.unknown_tokens(row.path)
            assert unknown == (), (source_id, row.role, unknown)


def test_every_route_row_names_why_it_does_not_record_subset():
    """Full files are the default; the opt-out has to say what it costs."""

    for source_id in fetch_routes.route_ids():
        route = fetch_routes.route_for(source_id)
        if not route.record_subset_supported:
            assert route.record_subset_why.strip()


# --------------------------------------------------------------------------
# Cycle grammar
# --------------------------------------------------------------------------

def test_a_cycle_hour_the_publisher_does_not_run_is_refused_by_name():
    with pytest.raises(ValueError) as error:
        fetch_routes.resolve_request(
            "gem-gdps", cycle=datetime(2026, 8, 17, 6), hours=6)
    message = str(error.value)
    assert "06Z" in message
    assert "00, 12" in message


def test_icon_eu_runs_eight_cycles_a_day():
    plan = fetch_routes.resolve_request(
        "icon-eu", cycle=datetime(2026, 8, 17, 3), hours=2)
    assert plan.leads == (0, 1, 2)


# --------------------------------------------------------------------------
# Lead ladders (all measured against the live listings 2026-08-17)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cycle_hour,last_lead", [(0, 21), (3, 51)])
def test_rap_extends_to_f51_on_the_off_synoptic_cycles(cycle_hour, last_lead):
    route = fetch_routes.route_for("rap")
    ladder = fetch_routes.ladder_for(
        route, datetime(2026, 8, 17, cycle_hour))
    assert ladder[-1] == last_lead


@pytest.mark.parametrize("cycle_hour,count,last_lead", [
    (0, 93, 120),    # hourly to 78, then three-hourly to 120
    (3, 34, 48),     # hourly to 30, then six-hourly to 48
])
def test_icon_eu_ladders_match_the_measured_dwd_listing(
        cycle_hour, count, last_lead):
    route = fetch_routes.route_for("icon-eu")
    ladder = fetch_routes.ladder_for(
        route, datetime(2026, 8, 17, cycle_hour))
    assert len(ladder) == count
    assert ladder[-1] == last_lead


def test_gefs_reaches_840_hours_only_on_the_00z_cycle():
    route = fetch_routes.route_for("gefs")
    assert fetch_routes.ladder_for(route, datetime(2026, 8, 17, 0))[-1] == 840
    assert fetch_routes.ladder_for(route, datetime(2026, 8, 17, 6))[-1] == 384


def test_a_lead_past_the_ladder_refuses_naming_the_horizon():
    with pytest.raises(ValueError) as error:
        fetch_routes.resolve_request(
            "rap", cycle=datetime(2026, 8, 17, 0), hours=30)
    message = str(error.value)
    assert "f021" in message or "f21" in message
    assert "03, 09, 15, 21" in message


def test_a_cadence_off_the_ladder_refuses_naming_the_ladder():
    with pytest.raises(ValueError) as error:
        fetch_routes.resolve_request(
            "gefs", cycle=datetime(2026, 8, 17, 0), hours=6, cadence=1)
    assert "cadence" in str(error.value)
    assert "3" in str(error.value)


def test_a_forecast_start_hour_shifts_the_window_not_its_length():
    plan = fetch_routes.resolve_request(
        "rap", cycle=datetime(2026, 8, 17, 3), hours=3, start_hour=12)
    assert plan.leads == (12, 13, 14, 15)


# --------------------------------------------------------------------------
# File-set composition, per source, against the measured key shapes
# --------------------------------------------------------------------------

def test_rap_resolves_the_measured_awip32_keys():
    plan = fetch_routes.resolve_request(
        "rap", cycle=datetime(2026, 8, 16, 0), hours=1)
    assert [obj.url for obj in plan.objects] == [
        "https://noaa-rap-pds.s3.amazonaws.com/rap.20260816/"
        "rap.t00z.awip32f00.grib2",
        "https://noaa-rap-pds.s3.amazonaws.com/rap.20260816/"
        "rap.t00z.awip32f01.grib2",
    ]
    # Flat layout: the key's basename is the file on disk.
    assert [obj.relpath for obj in plan.objects] == [
        "rap.t00z.awip32f00.grib2", "rap.t00z.awip32f01.grib2"]


def test_hrrr_prs_resolves_the_conus_wrfprs_keys():
    plan = fetch_routes.resolve_request(
        "hrrr-prs", cycle=datetime(2026, 8, 17, 0), hours=1)
    assert plan.objects[0].url == (
        "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20260817/conus/"
        "hrrr.t00z.wrfprsf00.grib2")


def test_rrfs_pairs_prslev_with_the_2dfld_supplement():
    plan = fetch_routes.resolve_request(
        "rrfs", cycle=datetime(2026, 8, 17, 0), hours=1)
    roles = [obj.role for obj in plan.objects]
    assert roles == ["prslev", "2dfld", "prslev", "2dfld"]
    assert plan.objects[0].url.endswith(
        "rrfs.20260817/00/rrfs.t00z.prslev.3km.f000.conus.grib2")
    # The 2dfld files are the composition's declared surface supplement,
    # not extra primaries.
    assert [path.name for path in plan.supplement_files] == [
        "rrfs.t00z.2dfld.3km.f000.conus.grib2",
        "rrfs.t00z.2dfld.3km.f001.conus.grib2",
    ]
    assert plan.supplement_role == "rrfs_prslev_2dfld_in_band_surface"


def test_gefs_keeps_member_identity_in_the_path_and_pairs_a_with_b():
    plan = fetch_routes.resolve_request(
        "gefs", cycle=datetime(2026, 8, 17, 0), hours=3)
    assert plan.leads == (0, 3)
    assert plan.objects[0].url == (
        "https://noaa-gefs-pds.s3.amazonaws.com/gefs.20260817/00/atmos/"
        "pgrb2ap5/gec00.t00z.pgrb2a.0p50.f000")
    # upstream layout: the whole key becomes the on-disk path, because the
    # member grammar addresses files by their upstream-relative path.
    assert plan.objects[0].relpath == (
        "upstream/gefs.20260817/00/atmos/pgrb2ap5/gec00.t00z.pgrb2a.0p50.f000")
    assert [step.name for step in plan.compose] == [
        "pairs/gec00.t00z.pair.0p50.f000.grib2",
        "pairs/gec00.t00z.pair.0p50.f003.grib2",
    ]
    # The concatenation order is a+b, and the pair is what prep consumes.
    assert [part.role for part in plan.compose[0].parts] == [
        "pgrb2a", "pgrb2b"]
    assert [path.name for path in plan.primary_files] == [
        "gec00.t00z.pair.0p50.f000.grib2",
        "gec00.t00z.pair.0p50.f003.grib2",
    ]


def test_gefs_perturbed_member_selects_its_own_token():
    plan = fetch_routes.resolve_request(
        "gefs", cycle=datetime(2026, 8, 17, 0), hours=3, member="p07")
    assert "gep07.t00z.pgrb2a.0p50.f000" in plan.objects[0].url


def test_an_unknown_member_refuses_naming_the_declared_set():
    with pytest.raises(ValueError) as error:
        fetch_routes.resolve_request(
            "gefs", cycle=datetime(2026, 8, 17, 0), hours=3, member="p99")
    assert "p99" in str(error.value)
    assert "c00" in str(error.value)


def test_aigefs_member_identity_is_only_in_the_path():
    plan = fetch_routes.resolve_request(
        "aigefs", cycle=datetime(2026, 8, 17, 0), hours=6)
    assert plan.objects[0].relpath == (
        "upstream/aigefs.20260817/00/mem000/model/atmos/grib2/"
        "aigefs.t00z.pres.f000.grib2")
    assert plan.member_set == "aigefs-ensemble-grib2-members-v1"


def test_aigfs_is_nomads_only_and_says_why_the_s3_copy_is_not_a_mirror():
    route = fetch_routes.route_for("aigfs")
    assert [host.name for host in route.hosts] == ["nomads"]
    with pytest.raises(ValueError) as error:
        fetch_routes.resolve_request(
            "aigfs", cycle=datetime(2026, 8, 17, 0), hours=6, host="aws")
    message = str(error.value)
    assert "aws" in message and "nomads" in message
    assert "subCentre" in route.host_note


def test_aigfs_plans_its_declared_same_cycle_gdas_donor():
    plan = fetch_routes.resolve_request(
        "aigfs", cycle=datetime(2026, 8, 17, 0), hours=6)
    assert len(plan.donors) == 1
    donor = plan.donors[0]
    assert donor.source == "gdas"
    assert donor.leads == (0,)
    assert donor.cycle == datetime(2026, 8, 17, 0)
    assert donor.role == "physical_analysis_surface_data"


def test_ecmwf_open_data_stamps_the_full_cycle_and_an_unpadded_lead():
    plan = fetch_routes.resolve_request(
        "ecmwf-open-data", cycle=datetime(2026, 8, 16, 0), hours=3)
    assert [obj.url for obj in plan.objects] == [
        "https://data.ecmwf.int/forecasts/20260816/00z/ifs/0p25/oper/"
        "20260816000000-0h-oper-fc.grib2",
        "https://data.ecmwf.int/forecasts/20260816/00z/ifs/0p25/oper/"
        "20260816000000-3h-oper-fc.grib2",
    ]


def test_aifs_is_six_hourly_and_supplements_from_the_first_file_only():
    plan = fetch_routes.resolve_request(
        "aifs", cycle=datetime(2026, 8, 17, 0), hours=6)
    assert plan.leads == (0, 6)
    assert plan.objects[0].url.endswith(
        "20260817/00z/aifs-single/0p25/oper/"
        "20260817000000-0h-oper-fc.grib2")
    assert len(plan.supplement_files) == 1
    assert plan.supplement_role == "aifs_single_in_band_surface"


def test_icon_eu_expands_125_field_objects_a_lead_plus_two_invariants():
    plan = fetch_routes.resolve_request(
        "icon-eu", cycle=datetime(2026, 8, 17, 0), hours=6)
    # The exact census the 6 h battery downloaded by hand: 125 x 7 + 2.
    assert len(plan.objects) == 877
    urls = {obj.url for obj in plan.objects}
    base = "https://opendata.dwd.de/weather/nwp/icon-eu/grib/00"
    assert (f"{base}/t/icon-eu_europe_regular-lat-lon_pressure-level_"
            "2026081700_000_850_T.grib2.bz2") in urls
    assert (f"{base}/t_2m/icon-eu_europe_regular-lat-lon_single-level_"
            "2026081700_003_T_2M.grib2.bz2") in urls
    assert (f"{base}/w_so/icon-eu_europe_regular-lat-lon_soil-level_"
            "2026081700_006_243_W_SO.grib2.bz2") in urls
    assert (f"{base}/hsurf/icon-eu_europe_regular-lat-lon_time-invariant_"
            "2026081700_HSURF.grib2.bz2") in urls
    # HSURF is the terrain supplement; FR_LAND rides in with the state.
    assert [path.name for path in plan.supplement_files] == [
        "icon-eu_europe_regular-lat-lon_time-invariant_2026081700_"
        "HSURF.grib2.bz2"]
    assert plan.supplement_role == "icon_eu_invariant_surface"


def test_gem_gdps_expands_174_a_lead_with_analysis_only_invariants():
    plan = fetch_routes.resolve_request(
        "gem-gdps", cycle=datetime(2026, 8, 16, 0), hours=3)
    # 174 state files at each of two leads, plus the two analysis-only
    # invariants and the terrain supplement, all at PT000H: 351 -- the
    # exact count the GEM staging lane fetched.
    assert len(plan.objects) == 351
    urls = {obj.url for obj in plan.objects}
    stem = "https://dd.weather.gc.ca/20260816/WXO-DD/model_gdps/15km/00"
    assert (f"{stem}/000/20260816T00Z_MSC_GDPS_AirTemp_IsbL-0850_"
            "LatLon0.15_PT000H.grib2") in urls
    assert (f"{stem}/003/20260816T00Z_MSC_GDPS_WindV_AGL-10m_"
            "LatLon0.15_PT003H.grib2") in urls
    assert (f"{stem}/000/20260816T00Z_MSC_GDPS_SeaIceFraction_Sfc_"
            "LatLon0.15_PT000H.grib2") in urls
    # The analysis-only invariants never appear at a later lead: their
    # /003/ URLs are 404s, which is the once-per-cycle publication shape.
    assert (f"{stem}/003/20260816T00Z_MSC_GDPS_SeaIceFraction_Sfc_"
            "LatLon0.15_PT003H.grib2") not in urls
    # Terrain is held OUT of the composed primary and passed once.
    assert [step.name for step in plan.compose] == [
        "composed/gdps_2026081600_PT000H.grib2",
        "composed/gdps_2026081600_PT003H.grib2",
    ]
    assert [path.name for path in plan.supplement_files] == [
        "20260816T00Z_MSC_GDPS_GeopotentialHeight_Sfc_"
        "LatLon0.15_PT000H.grib2"]
    assert plan.supplement_role == "gdps_analysis_invariant_surface"


# --------------------------------------------------------------------------
# Transport policy
# --------------------------------------------------------------------------

def test_full_file_is_the_default_mode_on_every_table_route():
    for source_id in fetch_routes.route_ids():
        assert fetch_routes.resolve_mode(source_id, None) == "full-file"


def test_idx_subset_refuses_with_the_row_s_own_reason():
    with pytest.raises(ValueError) as error:
        fetch_routes.resolve_mode("gefs", "idx-subset")
    message = str(error.value)
    assert "idx-subset" in message
    assert "disjoint" in message


def test_auto_mode_refuses_because_there_is_nothing_to_probe():
    with pytest.raises(ValueError) as error:
        fetch_routes.resolve_mode("rap", "auto")
    assert "auto" in str(error.value)


def test_an_area_crop_refuses_naming_where_the_crop_actually_happens():
    with pytest.raises(ValueError) as error:
        fetch_routes.resolve_request(
            "rap", cycle=datetime(2026, 8, 17, 0), hours=1,
            area="30,-110,45,-90")
    message = str(error.value)
    assert "--area" in message
    assert "gpuwm prep" in message


# --------------------------------------------------------------------------
# Named refusals for sources with no public bytes
# --------------------------------------------------------------------------

@pytest.mark.parametrize("source_id,fragment", [
    ("20crv3", "every-member"),
    ("20crv3-cf", "no cycle"),
    ("mapped", "generic declarative adapter"),
])
def test_a_source_without_fetchable_bytes_points_at_source_root(
        source_id, fragment):
    with pytest.raises(ValueError) as error:
        fetch_routes.route_for(source_id)
    message = str(error.value)
    assert fragment in message
    assert "--source-root" in message


def test_the_20crv3_refusal_reaches_through_the_alias():
    with pytest.raises(ValueError) as error:
        fetch_routes.route_for("20cr")
    assert "--source-root" in str(error.value)


# --------------------------------------------------------------------------
# Transfer, composition and the handoff, over a fake transport
# --------------------------------------------------------------------------

def _grib(payload: bytes = b"body") -> bytes:
    return b"GRIB" + payload + b"7777"


def _fake_downloader(record):
    def download(url, dest, *, magic, opener=None):
        record.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        body = _grib(url.encode()) if magic == "GRIB" else b"BZh" + url.encode()
        dest.write_bytes(body)
        return {"name": dest.name, "bytes": len(body),
                "sha256": __import__("hashlib").sha256(body).hexdigest(),
                "url": url}
    return download


def test_a_table_route_moves_every_object_and_publishes_its_receipts(tmp_path):
    plan = fetch_routes.resolve_request(
        "rap", cycle=datetime(2026, 8, 16, 0), hours=1)
    seen: list[str] = []
    payload = fetch_routes.run_plan(
        plan, out=tmp_path, downloader=_fake_downloader(seen),
        progress=lambda *_: None)

    assert len(seen) == 2
    assert payload["schema"] == fetch_routes.ROUTE_MANIFEST_SCHEMA
    assert payload["request"]["cycle"] == "2026-08-16T00Z"
    assert payload["mode"] == "full-file"
    # The pool receipt rides in the manifest, so a slow fetch is
    # explicable after the fact.
    assert payload["concurrency"]["files"] == 2
    assert payload["concurrency"]["workers_requested"] >= 1
    sums = (tmp_path / fetch_routes.SHA256SUMS_NAME).read_text().splitlines()
    assert len(sums) == 2
    assert all(len(line.split("  ")[0]) == 64 for line in sums)


def test_the_pool_is_the_default_transport_not_a_flag(tmp_path):
    plan = fetch_routes.resolve_request(
        "rap", cycle=datetime(2026, 8, 16, 0), hours=5)
    payload = fetch_routes.run_plan(
        plan, out=tmp_path, downloader=_fake_downloader([]),
        progress=lambda *_: None)
    assert (payload["concurrency"]["workers_requested"]
            == fetch_routes.fetch_pool.DEFAULT_FILE_WORKERS)


def test_a_second_run_reuses_the_bytes_already_on_disk(tmp_path):
    plan = fetch_routes.resolve_request(
        "rap", cycle=datetime(2026, 8, 16, 0), hours=1)
    fetch_routes.run_plan(plan, out=tmp_path,
                          downloader=_fake_downloader([]),
                          progress=lambda *_: None)
    second: list[str] = []
    payload = fetch_routes.run_plan(
        plan, out=tmp_path, downloader=_fake_downloader(second),
        progress=lambda *_: None)
    assert second == []
    assert all(entry["reused"] for entry in payload["files"])


def test_a_different_request_into_the_same_directory_refuses(tmp_path):
    first = fetch_routes.resolve_request(
        "rap", cycle=datetime(2026, 8, 16, 0), hours=1)
    fetch_routes.run_plan(first, out=tmp_path,
                          downloader=_fake_downloader([]),
                          progress=lambda *_: None)
    second = fetch_routes.resolve_request(
        "rap", cycle=datetime(2026, 8, 16, 12), hours=1)
    with pytest.raises(ValueError) as error:
        fetch_routes.run_plan(second, out=tmp_path,
                              downloader=_fake_downloader([]),
                              progress=lambda *_: None)
    assert "--force-refetch" in str(error.value)


def test_a_truncated_object_refuses_by_name(tmp_path):
    plan = fetch_routes.resolve_request(
        "rap", cycle=datetime(2026, 8, 16, 0), hours=1)

    def truncating(url, dest, *, magic, opener=None):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"GRIB" + b"half a message")
        fetch_routes._verify_payload(dest, magic=magic, label=dest.name)
        return {}

    with pytest.raises(ValueError) as error:
        fetch_routes.run_plan(plan, out=tmp_path, downloader=truncating,
                              progress=lambda *_: None)
    assert "end marker" in str(error.value)
    assert "rap.t00z.awip32f00.grib2" in str(error.value)


def test_gefs_concatenates_its_pair_in_the_declared_order(tmp_path):
    plan = fetch_routes.resolve_request(
        "gefs", cycle=datetime(2026, 8, 17, 0), hours=3)
    payload = fetch_routes.run_plan(
        plan, out=tmp_path, downloader=_fake_downloader([]),
        progress=lambda *_: None)
    assert len(payload["composed"]) == 2
    pair = tmp_path / "pairs" / "gec00.t00z.pair.0p50.f000.grib2"
    assert pair.is_file()
    body = pair.read_bytes()
    assert body.index(b"pgrb2ap5") < body.index(b"pgrb2bp5")


def test_the_handoff_binds_the_input_list_and_the_supplement_role(tmp_path):
    plan = fetch_routes.resolve_request(
        "rrfs", cycle=datetime(2026, 8, 17, 0), hours=1)
    fetch_routes.run_plan(plan, out=tmp_path,
                          downloader=_fake_downloader([]),
                          progress=lambda *_: None)
    inputs, command = fetch_routes.write_handoff(plan, tmp_path)
    listed = inputs.read_text().splitlines()
    assert len(listed) == 2
    assert all(line.endswith(".grib2") and "prslev" in line
               for line in listed)
    text = command.read_text()
    assert "gpuwm prep \\" in text
    assert "--source rrfs" in text
    assert "--input-list" in text
    # Every continuation but the last carries its backslash, and no
    # comment ever rides on a continued line: the file runs as written.
    body = [line for line in text.splitlines()
            if line and not line.lstrip().startswith("#")]
    assert all(line.endswith("\\") for line in body[:-1])
    assert not body[-1].endswith("\\")
    assert "rrfs_prslev_2dfld_in_band_surface=" in text
    assert text.count("--supplement") == 2


def test_the_handoff_publishes_machine_readable_prep_arguments(tmp_path):
    """`prep-arguments.json` carries the SAME binding as the text file.

    The staged run-plan chain composes its preparation from this
    document, so it must hold argv TOKENS (no quoting convention to
    round-trip), the caller-owned flags by name, and an explicit list
    of any supplement role this fetch left unbound.
    """

    plan = fetch_routes.resolve_request(
        "rrfs", cycle=datetime(2026, 8, 17, 0), hours=1)
    fetch_routes.run_plan(plan, out=tmp_path,
                          downloader=_fake_downloader([]),
                          progress=lambda *_: None)
    inputs, command = fetch_routes.write_handoff(plan, tmp_path)
    document = json.loads(
        (tmp_path / fetch_routes.PREP_ARGUMENTS_NAME).read_text())
    assert document["schema"] == fetch_routes.PREP_ARGUMENTS_SCHEMA
    argv = document["argv"]
    assert argv[:4] == ["--source", "rrfs", "--input-list",
                        str(inputs.resolve())]
    assert argv.count("--supplement") == 2
    assert "--author-input-manifest" in argv
    assert document["unbound_supplement_roles"] == []
    assert document["caller_supplies"] == [
        "--wps-namelist", "--experiment-config", "--geog-root",
        "--output-root"]
    # Token for token, the text command is these argv pairs rendered.
    text = command.read_text()
    for flag, value in zip(argv[::2], argv[1::2]):
        assert flag in text
        assert value.split("=")[0] in text


def test_the_prep_arguments_name_an_unbound_donor_role(tmp_path):
    """A donor this fetch did not bring is a named hole, not a surprise."""

    plan = fetch_routes.resolve_request(
        "aigfs", cycle=datetime(2026, 8, 17, 0), hours=6)
    fetch_routes.run_plan(plan, out=tmp_path,
                          downloader=_fake_downloader([]),
                          progress=lambda *_: None)
    fetch_routes.write_handoff(plan, tmp_path)
    document = json.loads(
        (tmp_path / fetch_routes.PREP_ARGUMENTS_NAME).read_text())
    assert document["unbound_supplement_roles"] == [
        "physical_analysis_surface_data"]


def test_the_handoff_names_member_prep_for_an_ensemble_route(tmp_path):
    plan = fetch_routes.resolve_request(
        "aigefs", cycle=datetime(2026, 8, 17, 0), hours=6)
    fetch_routes.run_plan(plan, out=tmp_path,
                          downloader=_fake_downloader([]),
                          progress=lambda *_: None)
    _, command = fetch_routes.write_handoff(plan, tmp_path)
    text = command.read_text()
    assert "gpuwm-member-prep" in text
    assert "aigefs-ensemble-grib2-members-v1" in text
    # The donor is declared even when this run did not fetch it.
    assert "physical_analysis_surface_data" in text


def test_the_handoff_binds_a_donor_that_was_fetched(tmp_path):
    plan = fetch_routes.resolve_request(
        "aigfs", cycle=datetime(2026, 8, 17, 0), hours=6)
    fetch_routes.run_plan(plan, out=tmp_path,
                          downloader=_fake_downloader([]),
                          progress=lambda *_: None)
    donor = tmp_path / "donor-gdas" / "gdas.t00z.pgrb2.0p25.f000"
    donor.parent.mkdir(parents=True, exist_ok=True)
    donor.write_bytes(_grib())
    _, command = fetch_routes.write_handoff(
        plan, tmp_path, donor_files={"physical_analysis_surface_data": donor})
    text = command.read_text()
    assert "--supplement" in text
    assert "physical_analysis_surface_data=" in text
    assert "not fetched" not in text
