"""AI-GEFS files served by NOMADS verify and prepare.

The operational door serves the same members as the AWS mirror, stamped
differently: typeOfEnsembleForecast 6 where the mirror's copy says 3, and
a sfc product with no surface-pressure record, which the mirror's copy
appends.  The breakage these tests prevent, measured on the 2026-09-27
12Z cycle while the mirror had not caught up: member verification
refused every NOMADS file, and a preparation that skipped it stopped on
``lacks required fields ['surface_pressure']`` for the control and every
member.

Both doors serve mean-sea-level pressure, and the one profile derives
surface pressure from it at every lead, so a request prepares the same
way whichever door served it.  The mirror's appended record is not read:
it sits on the AI model's own orography, 5 hPa RMS over CONUS off the
analysis surface pressure on the terrain the composition pairs it with,
where the reduction is 0.3 hPa off.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import pytest

from gpuwm import download_budget, fetch_routes
from gpuwm.member_grammar import (MemberIdentityRefusal, MemberGrammar,
                                  MemberGrammarError, load_member_grammar)
from gpuwm.member_prep import verify_member_file, verify_member_rows
from gpuwm.source_adapters import get_source_adapter
from gpuwm.source_authorities import (packaged_authorities,
                                      packaged_member_grammar)

PROFILE = "aigefs-member-hybrid-grib2-v1"
CYCLE = datetime(2026, 9, 27, 12)
#: Late enough that the cycle is inside both endpoints' windows, so the
#: unpinned ladder asks NOMADS first and the mirror second.
NOW = datetime(2026, 9, 27, 21)
FIXTURE = (Path(__file__).resolve().parent / "fixtures"
           / "ensemble-member-identity" / "aigefs.20260927" / "12" / "mem000"
           / "model" / "atmos" / "grib2" / "aigefs.t12z.sfc.f000.grib2")


def _aigefs() -> MemberGrammar:
    return load_member_grammar(
        packaged_member_grammar("aigefs-ensemble-grib2-members-v1"))


def _row(index: int, *, member: str, ensemble_type: str) -> dict:
    return {
        "index": str(index), "pdt": "1", "member": member,
        "ensemble_type": ensemble_type, "ensemble_size": "31",
        "derived_forecast": "-", "generating_process": "4",
        "forecast_generating_process_id": "138",
    }


# ---------------------------------------------------------------------------
# Member verification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("member_id,ordinal", [("mem000", "0"), ("mem001", "1")])
def test_a_nomads_stamped_member_verifies_and_the_receipt_says_what_it_carries(
        member_id, ordinal):
    grammar = _aigefs()
    evidence = verify_member_rows(
        grammar, grammar.member(member_id),
        [_row(0, member=ordinal, ensemble_type="6"),
         _row(1, member=ordinal, ensemble_type="6")],
        source_label="nomads")
    assert evidence.type_of_ensemble_forecast == 6
    mirror = verify_member_rows(
        grammar, grammar.member(member_id),
        [_row(0, member=ordinal, ensemble_type="3")], source_label="aws")
    assert mirror.type_of_ensemble_forecast == 3


def test_an_undeclared_ensemble_type_still_refuses_naming_every_declared_one():
    grammar = _aigefs()
    with pytest.raises(MemberIdentityRefusal) as caught:
        verify_member_rows(grammar, grammar.member("mem000"),
                           [_row(0, member="0", ensemble_type="2")],
                           source_label="foreign")
    message = str(caught.value)
    assert "declared typeOfEnsembleForecast 3 or 6" in message
    assert "carries typeOfEnsembleForecast 2" in message


def test_the_nomads_control_verifies_on_real_bytes():
    evidence = verify_member_file(_aigefs(), "mem000", FIXTURE)
    assert evidence.type_of_ensemble_forecast == 6
    assert evidence.perturbation_number == 0
    assert evidence.ensemble_size == (31,)


@pytest.mark.parametrize("declared", [[], [3, 3], ["3"], "3", True])
def test_a_malformed_ensemble_type_list_refuses(declared):
    document = json.loads(packaged_member_grammar(
        "aigefs-ensemble-grib2-members-v1").read_text(encoding="utf-8"))
    document["classes"]["control"]["verification"][
        "type_of_ensemble_forecast"] = declared
    with pytest.raises(MemberGrammarError, match="type_of_ensemble_forecast"):
        MemberGrammar(document, source="synthetic")


# ---------------------------------------------------------------------------
# One profile reads what both doors serve
# ---------------------------------------------------------------------------

def _mapping():
    return json.loads(packaged_authorities(PROFILE)["mapping"].read_text(
        encoding="utf-8"))


def test_surface_pressure_is_derived_from_the_members_own_mslp():
    mapping = _mapping()
    surface = mapping["fields"]["surface_pressure"]
    assert surface["selectors"] == [] and "provider" not in surface
    derivation = {item["name"]: item for item in mapping["derivations"]}[
        surface["derivation"]]
    assert derivation == {
        "name": "surface-pressure-from-sea-level",
        "operation": "surface_pressure_from_sea_level",
        "sea_level_pressure": "air_pressure_at_mean_sea_level",
        "level_height": "geopotential_height",
        "pressure": "air_pressure",
        "surface_height": "terrain_height",
    }
    # The member's own record, pinned to the individual-member template
    # like every other selector, then the same record as the mirror's
    # re-encoded archive carries it (PDT 0 under master table 4 and local
    # table 0, which no deterministic product stamps), so a deterministic
    # file still refuses.
    record = {"format": "grib2", "discipline": 0, "category": 3,
              "parameter": 1, "level_type": 101}
    assert mapping["fields"]["air_pressure_at_mean_sea_level"]["selectors"] == [
        {**record, "pdt": 1},
        {**record, "pdt": 0, "master_table_version": 4,
         "local_table_version": 0}]


def test_no_selector_reads_the_surface_pressure_record_only_one_door_serves():
    for name, field in _mapping()["fields"].items():
        for selector in field.get("selectors") or ():
            assert (selector.get("category"), selector.get("parameter"),
                    selector.get("level_type")) != (3, 0, 1), name


def test_surface_pressure_is_not_borrowed_from_the_analysis():
    composition = json.loads(packaged_authorities(PROFILE)[
        "composition"].read_text(encoding="utf-8"))
    assert "surface_pressure" not in composition["field_sources"][
        "physical_analysis_surface"]["fields"]


def test_the_source_prepares_through_one_profile():
    assert get_source_adapter("aigefs").packaged_profile == PROFILE
    route = fetch_routes.route_for("aigefs")
    assert "profile_by_endpoint" not in route.prep


# ---------------------------------------------------------------------------
# Either door's files hand the same preparation over
# ---------------------------------------------------------------------------

def _grib(payload: bytes) -> bytes:
    return b"GRIB" + payload + b"7777"


def _downloader(url, dest, *, magic, opener=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    body = _grib(url.encode())
    dest.write_bytes(body)
    return {"name": dest.name, "bytes": len(body),
            "sha256": __import__("hashlib").sha256(body).hexdigest(),
            "url": url}


def _fetch(tmp_path, *, host=None, probe=lambda url: False, **kwargs):
    plan = fetch_routes.resolve_request(
        "aigefs", cycle=CYCLE, hours=6, host=host, now=NOW, **kwargs)
    receipt = fetch_routes.run_plan(plan, out=tmp_path, downloader=_downloader,
                                    probe=probe, progress=lambda *_: None)
    donor = tmp_path / "donor-gdas" / "gdas.t12z.pgrb2.0p25.f000"
    donor.parent.mkdir(parents=True, exist_ok=True)
    donor.write_bytes(_grib(b"donor"))
    fetch_routes.write_handoff(
        plan, tmp_path, donor_files={"physical_analysis_surface_data": donor})
    arguments = json.loads(
        (tmp_path / fetch_routes.PREP_ARGUMENTS_NAME).read_text())
    return receipt, arguments["argv"]


@pytest.mark.parametrize("host", ["nomads", "aws"])
def test_either_door_hands_the_source_profile_its_donor(tmp_path, host):
    receipt, argv = _fetch(tmp_path, host=host)
    assert receipt["endpoints"]["served"] == [host]
    assert argv[:2] == ["--source", "aigefs"]
    assert "--source-profile" not in argv
    # The donor the profile reads its land surface and terrain from is
    # the one the route fetches.
    assert any(token.startswith("physical_analysis_surface_data=")
               for token in argv)


def test_a_mixed_request_hands_over_the_same_preparation(tmp_path):
    # The mirror has only the analysis-time surface file; NOMADS serves
    # the rest, and the whole series decodes through the one profile.
    receipt, argv = _fetch(tmp_path, probe=lambda url: "sfc.f000" in url)
    assert set(receipt["endpoints"]["served"]) == {"nomads", "aws"}
    assert argv[:2] == ["--source", "aigefs"]
    assert "--source-profile" not in argv


def test_the_download_estimate_prices_the_donor_the_profile_reads():
    estimate = download_budget.download_estimate(
        {"source": "aigefs", "cycle": "2026-09-27T12", "hours": 6})
    assert estimate["bytes"] is not None
    assert "gdas donor" in estimate["basis"]
