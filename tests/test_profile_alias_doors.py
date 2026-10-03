"""An old physics profile ID behaves exactly as its current ID at every door.

2.8.4 renamed three profile IDs and promised the old ones stay accepted as
aliases (``gpuwm.physics_registry.TEMPLATE_ID_ALIASES``).  WOOF and saved
2.8.3 plans, catalogs, drafts and receipts still send the old IDs, and the
promise was broken at the doors below, each measured on the merged tree:

* HRRR preparation through ``gpuwm run-plan`` or ``gpuwm stream`` ran the
  whole preparation and then refused its own receipt, because the benchmark
  records the current ID while the wrapper and the stream compared the raw
  old one;
* ``gpuwm case-catalog`` refused the old ID from its flag and from custom
  catalogs;
* a run-plan manifest recorded no physics components for an old ID;
* ``gpuwm physics-catalog --check`` lost the component choices of an old
  suite ID;
* New forecast dropped an old ID, refused it with 409 against the check's
  current ID, and skipped the day-only pre-check.

Each test drives the real door with the old ID and asserts the same answer
as the current ID.  CPU only.
"""

from __future__ import annotations

from argparse import Namespace
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm.physics_registry import TEMPLATE_ID_ALIASES, physics_registry

PAIRS = sorted(TEMPLATE_ID_ALIASES.items())
IDS = [old for old, _new in PAIRS]


def test_the_alias_table_is_the_registry_s_own():
    """The doors read the registry's table; the constant and the JSON agree."""
    assert physics_registry()["template_aliases"] == TEMPLATE_ID_ALIASES
    for old, new in PAIRS:
        assert old not in physics_registry()["templates"]
        assert new in physics_registry()["templates"]


# ---------------------------------------------------------------------------
# HRRR preparation: tools.prepare_hrrr_wrf
# ---------------------------------------------------------------------------

def _receipt(profile):
    return {"physics": {"schema": "gpuwm-prepared-physics-profile-v1",
                        "profile": profile}}


@pytest.mark.parametrize("old,new", PAIRS, ids=IDS)
def test_prepare_hrrr_wrf_reads_an_old_id_as_the_current_one(old, new):
    from tools.prepare_hrrr_wrf import _parser

    def parsed(profile):
        return _parser().parse_args([
            "--source-root", "src", "--geog-root", "geog",
            "--namelist-input", "namelist.input", "--output-root", "out",
            "--physics-profile", profile]).physics_profile

    assert parsed(old) == parsed(new) == new


@pytest.mark.parametrize("old,new", PAIRS, ids=IDS)
def test_preparation_receipt_check_matches_either_spelling(old, new):
    """The benchmark records the current ID; a chain document or a 2.8.3
    report may carry the old one.  Every pairing is the same physics: the
    identity check passes and the receipt meets its next check (the
    cold-start evidence this minimal receipt does not carry)."""
    from tools.prepare_hrrr_wrf import _validated_physics_receipt

    for recorded, requested in ((new, old), (old, new), (new, new), (old, old)):
        with pytest.raises(RuntimeError, match="cold-start evidence"):
            _validated_physics_receipt(
                _receipt(recorded), requested_profile=requested)
    other = next(n for _o, n in PAIRS if n != new)
    with pytest.raises(RuntimeError, match="differs from the request"):
        _validated_physics_receipt(_receipt(new), requested_profile=other)


def _sealed_predecessor(tmp_path: Path, profile: str) -> Path:
    prior = tmp_path / "prior"
    for name in ("native/prepared-cache", "native/preparation-report",
                 "native/native-bridge"):
        (prior / name).mkdir(parents=True, exist_ok=True)
    for name in ("native/preparation-report/report.json",
                 "native/source-manifest.snapshot", "native-static.npz",
                 "native-static-receipt.json", "native-geometry-receipt.json"):
        (prior / name).write_text("{}", encoding="utf-8")
    (prior / "public-wrapper-result.json").write_text(json.dumps({
        "status": "PASS",
        "prepared_cache_contract": {"mode": "sealed-prefix-v1"},
        "source_forecast_hours": [0, 1],
        "source_cycle": "2026-09-01T12:00:00",
        "physics": {"profile": profile},
        "history_interval_seconds": 3600.0,
    }), encoding="utf-8")
    return prior


@pytest.mark.parametrize("old,new", PAIRS, ids=IDS)
def test_a_predecessor_sealed_under_an_old_id_still_extends(tmp_path, old, new):
    """The run-contract check passes for either spelling on either side and
    the extension reaches its next check (the source manifest digest)."""
    from tools.prepare_hrrr_wrf import _sealed_extension

    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text("", encoding="utf-8")

    def extend(sealed, requested):
        prior = _sealed_predecessor(tmp_path / f"{sealed}-{requested}", sealed)
        args = Namespace(
            extend_root_preparation=prior, forecast_start_hour=0,
            run_seconds=2 * 3600, physics_profile=requested,
            history_interval_seconds=3600.0, source_manifest=manifest,
            source_manifest_sha256="0" * 64)
        _sealed_extension(
            args, valid_time=datetime(2026, 9, 1, 12), source_forecast_hours=[0, 1, 2],
            output=tmp_path / "out", env={}, decoder=tmp_path, started=0.0,
            namelist_invariant={})

    for sealed, requested in ((old, new), (new, old), (old, old)):
        with pytest.raises(ValueError, match="source manifest digest differs"):
            extend(sealed, requested)
    other = next(n for _o, n in PAIRS if n != new)
    with pytest.raises(ValueError, match="another run contract"):
        extend(old, other)


# ---------------------------------------------------------------------------
# gpuwm stream
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("old,new", PAIRS, ids=IDS)
def test_a_stream_plan_naming_an_old_id_carries_the_current_one(tmp_path, old, new):
    from gpuwm import stream
    from test_stream import _make_plan

    plan = _make_plan(tmp_path)
    text = plan.path.read_text(encoding="utf-8")
    assert 'physics_profile = "generic-profile-v1"' in text
    plan.path.write_text(text.replace(
        'physics_profile = "generic-profile-v1"', f'physics_profile = "{old}"'),
        encoding="utf-8")
    assert stream.load_stream_plan(plan.path).physics_profile == new


@pytest.mark.parametrize("old,new", PAIRS, ids=IDS)
def test_the_stream_root_check_matches_either_spelling(tmp_path, old, new):
    """Past the identity check for every pairing: the next refusal is the
    missing sealed-operation contract, not an identity mismatch."""
    from gpuwm import stream

    cycle = datetime(2026, 9, 1, 12, tzinfo=timezone.utc)

    def root(recorded, planned):
        path = tmp_path / f"{recorded}-{planned}"
        path.mkdir()
        (path / "public-wrapper-result.json").write_text(json.dumps({
            "status": "PASS", "source_cycle": cycle.isoformat(),
            "model_start_time": cycle.isoformat(),
            "source_forecast_hours": [0, 1], "model_forcing_hours": [0, 1],
            "forcing_hours": [0, 1], "history_interval_seconds": 3600.0,
            "physics": {"profile": recorded}}), encoding="utf-8")
        plan = SimpleNamespace(
            physics_profile=planned,
            experiment=SimpleNamespace(root=SimpleNamespace(history_interval_s=3600)))
        stream._valid_root(path, plan=plan, cycle=cycle, lead=1,
                           source_sums=tmp_path / "SHA256SUMS")

    for recorded, planned in ((new, old), (old, new), (old, old)):
        with pytest.raises(ValueError, match="not the requested sealed operation"):
            root(recorded, planned)
    other = next(n for _o, n in PAIRS if n != new)
    with pytest.raises(ValueError, match="identity/status mismatch"):
        root(other, old)


# ---------------------------------------------------------------------------
# gpuwm run-plan
# ---------------------------------------------------------------------------

def _plan(tmp_path, *, run_options=None, intent_profile=None):
    from gpuwm.runplan import build_plan

    intent = {"point": "37.62,-122.2", "source": "gfs", "cycle": "2026-09-20T18",
              "hours": 1, "card": "16gb", "root_dx_km": 3}
    if intent_profile is not None:
        intent["physics_profile"] = intent_profile
    document = {"schema": "gpuwm.run-plan.v1", "name": "p", "route": "prepared",
                "config": {"intent": intent}, "output_root": str(tmp_path / "out")}
    if run_options is not None:
        document["run_options"] = run_options
    return build_plan(document, source="test", base_dir=tmp_path, sha256="0" * 64)


@pytest.mark.parametrize("old,new", PAIRS, ids=IDS)
def test_a_run_plan_naming_an_old_id_records_the_current_suite(tmp_path, old, new):
    from gpuwm.runplan import _asserted_profile, manifest_physics

    stated = _plan(tmp_path, run_options={"physics_profile": old})
    current = _plan(tmp_path, run_options={"physics_profile": new})
    assert stated.run_options["physics_profile"] == new
    # The value the HRRR chain hands `python -m tools.prepare_hrrr_wrf`.
    assert _asserted_profile(stated, config_path=tmp_path / "unused.toml") == new
    recorded = manifest_physics(stated)
    assert recorded == manifest_physics(current)
    assert recorded["components"] == dict(physics_registry()["templates"][new]["components"])

    intent = _plan(tmp_path, intent_profile=old)
    assert intent.config_intent["physics_profile"] == new
    assert manifest_physics(intent) == manifest_physics(_plan(tmp_path, intent_profile=new))


# ---------------------------------------------------------------------------
# gpuwm case-catalog
# ---------------------------------------------------------------------------

CASE_NOW = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)


def _catalog_path():
    from gpuwm import case_catalog

    return Path(case_catalog.__file__).parent / "data" / "case-catalog" / "example.json"


@pytest.mark.parametrize("old,new", PAIRS, ids=IDS)
def test_a_custom_catalog_naming_an_old_id_validates_and_previews_as_the_current(
        tmp_path, old, new):
    from gpuwm import case_catalog

    def preview(profile):
        document = json.loads(_catalog_path().read_text(encoding="utf-8"))
        case = next(c for c in document["cases"] if c["id"] == "synthetic-overrides-example")
        case["physics_profile"] = profile
        for row in case["tiers"].values():
            row["physics_profile"] = profile
        case_catalog.validate_catalog(document)
        path = tmp_path / f"{profile}.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        result = case_catalog.preview_case(path, "synthetic-overrides-example",
                                           tier="lower", now=CASE_NOW)
        return {key: value for key, value in result.items() if key != "provenance"}

    stated = preview(old)
    assert stated["physics_profile"] == new
    assert stated == preview(new)


@pytest.mark.parametrize("old,new", PAIRS, ids=IDS)
def test_case_catalog_create_flag_takes_an_old_id(tmp_path, old, new, monkeypatch, capsys):
    """The CLI door, as WOOF drives it: the same outcome as the current ID."""
    from gpuwm import domain_wizard as wizard
    from gpuwm.cli import main

    monkeypatch.setattr(wizard, "resolve_sizing_budget",
                        lambda *a, **k: pytest.fail("GPU probe on Open Case"))

    def create(profile):
        folder = tmp_path / ("old" if profile == old else "new")
        folder.mkdir()
        out = folder / "case.toml"
        code = main(["case-catalog", "create", "synthetic-overrides-example",
                     "--catalog", str(_catalog_path()), "--tier", "lower",
                     "--physics-profile", profile, "--out", str(out),
                     "--geometry-only", "--json"])
        document = json.loads(capsys.readouterr().out)

        def local(text):
            return (None if text is None else
                    str(text).replace(str(folder), "<dir>").replace(folder.as_posix(), "<dir>"))

        written = local(out.read_text(encoding="utf-8")) if out.exists() else None
        return code, document.get("schema"), local(document.get("error")), written

    stated = create(old)
    assert stated == create(new)
    assert "unknown native physics profile" not in str(stated[2])


# ---------------------------------------------------------------------------
# gpuwm physics-catalog --check
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("old,new", PAIRS, ids=IDS)
def test_the_physics_check_reads_an_old_suite_id_as_the_current_one(old, new):
    from gpuwm import physics_catalog

    request = {"source": "gfs", "dx_km": 3}
    stated = physics_catalog.check({**request, "suite": old})
    assert stated["base_suite"] == new
    assert stated == physics_catalog.check({**request, "suite": new})


# ---------------------------------------------------------------------------
# New forecast (gpuwm gui)
# ---------------------------------------------------------------------------

class _Drafts:
    """The GUI's create mixin with the engine's answers stubbed."""

    def __new__(cls, *, verdict=None, suites=()):
        from gpuwm.gui.api import CreateMixin

        class Drafts(CreateMixin):
            def _offered(self):
                return {"sources": [{"id": "gfs", "route": "prepared",
                                     "profiles": [{"id": new} for _old, new in PAIRS]}]}

            def availability_of(self, *args, **kwargs):
                return {"starts": "yes"}

            def _cached(self, key, argv, ttl):
                return {"suites": list(suites), "default_suite": None}

        drafts = Drafts()
        drafts.runner = SimpleNamespace(query=lambda argv: dict(verdict or {}))
        return drafts


def _payload(**extra):
    return {"name": "alias-run", "source": "gfs", "cycle": "2026-09-24T00",
            "card": "8gb", "lat": 35, "lon": -100, **extra}


@pytest.mark.parametrize("old,new", PAIRS, ids=IDS)
def test_new_forecast_keeps_an_old_id_as_the_current_set(old, new):
    stated = _Drafts().draft(_payload(profile=old), need_name=True)
    assert stated["profile"] == new
    assert stated == _Drafts().draft(_payload(profile=new), need_name=True)


@pytest.mark.parametrize("old,new", PAIRS, ids=IDS)
def test_new_forecast_composed_physics_agrees_with_the_check_for_an_old_id(old, new):
    """The 409 compared the draft's old ID with the check's current one."""
    drafts = _Drafts(verdict={"valid": True, "named_suite": new, "plan_intent": {}})
    draft = drafts.draft(_payload(profile=old), need_name=True)
    drafts.composed_physics({"physics_choices": {"microphysics": "x"}}, draft)
    assert draft["profile"] == new


@pytest.mark.parametrize("old,new", PAIRS, ids=IDS)
def test_new_forecast_day_only_check_runs_for_an_old_id(old, new):
    """The night-window refusal comes on the click, not at config load."""
    from gpuwm.gui.api import ApiError

    refusal = {"valid": False, "words": "This set is day-only.",
               "refusal": {"door": "nocturnal-radiation", "message": "night"}}
    drafts = _Drafts(verdict=refusal, suites=[{"id": new, "day_only": True}])
    draft = drafts.draft(_payload(profile=old), need_name=True)
    with pytest.raises(ApiError) as raised:
        drafts.night_refusal(draft)
    assert raised.value.status == 422


def test_new_forecast_sources_carry_the_alias_table_for_kept_drafts():
    from gpuwm.gui.api import SystemMixin

    class Offered(SystemMixin):
        def _cached(self, key, argv, ttl):
            return {"sources": []}

    assert Offered()._offered()["profile_aliases"] == TEMPLATE_ID_ALIASES


@pytest.mark.parametrize("old,new", PAIRS, ids=IDS)
def test_an_event_recipe_recorded_with_an_old_id_offers_the_current_set(old, new):
    from gpuwm.gui.wiki import recipe_of

    best = {"cards": [{"fits": True, "card_gb": 24,
                       "recipe": {"source": "gfs", "hours": 6},
                       "intent": {"physics_profile": old},
                       "physics": {"profile": old, "why": "event"}}]}
    store = SimpleNamespace(data=lambda: {
        "events": {"e": {"title": "Event", "recipe": {"source": "gfs", "profile": old}}},
        "recipes": {"e": best}})
    assert recipe_of(store, "e", {"gfs"})["profile"] == new
    assert recipe_of(store, "e", {"gfs"})["layout"]["physics"]["profile"] == new
    fallback = SimpleNamespace(data=lambda: {
        "events": {"e": {"title": "Event", "recipe": {"source": "gfs", "profile": old}}},
        "recipes": {}})
    assert recipe_of(fallback, "e", {"gfs"})["profile"] == new


def test_the_page_maps_a_kept_old_id_before_it_checks_the_offered_list():
    """create.js clears a set the source does not offer; it maps first."""
    script = (Path(__file__).resolve().parents[1] / "gpuwm" / "gui" / "static"
              / "js" / "create.js").read_text(encoding="utf-8")
    assert "profileAliases = data.profile_aliases || {};" in script
    fill = script[script.index("function fillProfiles()"):]
    fill = fill[:fill.index("\n  }\n")]
    assert fill.index("draft.profile = currentProfile(draft.profile);") < fill.index(
        'draft.profile = "";')
