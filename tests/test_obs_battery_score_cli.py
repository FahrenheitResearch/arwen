"""The scoring command's own layer: its defaults, and where they come from.

``tools/obs_battery_score.py`` wires the battery's engine, its registration
discipline and its readers together, and the first attempt to run it on a
machine other than the one that fetched the observations found four things
wrong at exactly that layer -- none of them in the engine, all of them in what
the command chose on the operator's behalf:

* a hardcoded reported-lead default that asked for forecast hours past the end
  of the observation archive;
* lead 0 in that default, where the history frame is the initial condition
  written back out and carries no reflectivity at all;
* one accumulation window where the registration ratifies two, which the
  engine correctly refused to score;
* a re-hash bound to the absolute paths of the box that pulled the archive.

So the pins below are about the *command*: the leads and windows come off the
registration document a score is bound to, and the re-hash reaches every
source through a relocation root.  The registration used here is a minimal
fixture built by the same constructors the campaign document was, because a
test that reads the ratified file measures the file rather than the rule.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest

from gpuwm.obs import sources
from gpuwm.verify.obs import registration as reg_mod
from gpuwm.verify.obs.contracts import ObsProvenance

from test_obs_battery_ingest import geo_pack_bytes, grid_pack_bytes

COMMIT = "2" * 40
INIT = "2026-08-03T12:00:00"
FETCHED_AT = "2026-08-03T09:00:00"


def score_cli():
    """The command under test, imported the way the repository does it."""
    return importlib.import_module("tools.obs_battery_score")


def _registration(*, leads, windows=(1, 6)):
    """A minimal document; only its lead and window pins matter here."""
    return reg_mod.make_registration(
        evaluator_commit=COMMIT,
        reflectivity=reg_mod.reflectivity_parameters(),
        surface=reg_mod.surface_parameters(),
        precipitation=reg_mod.precipitation_parameters(
            window_hours=tuple(windows),
            guardrail_window_hours=max(windows)),
        promotion=reg_mod.promotion_parameters(),
        cases=[{"case_id": "case-fixture", "init_time": INIT}],
        arms=[{"arm_id": "faithful"}],
        twin={"rung": 1},
        scored_lead_hours_=tuple(leads))


def _reflectivity_tree(tmp_path, *, uri, sha256):
    directory = tmp_path / "reflectivity"
    packs = directory / "packs"
    packs.mkdir(parents=True)
    (packs / "geometry.obspack").write_bytes(geo_pack_bytes())
    (packs / "frame.obspack").write_bytes(
        grid_pack_bytes(valid_time="2026-08-03T14:00:00", uri=uri,
                        sha256=sha256))
    return directory


def _stage4_tree(tmp_path, windows):
    directory = tmp_path / "stage4"
    packs = directory / "packs"
    packs.mkdir(parents=True)
    (packs / "geometry.obspack").write_bytes(geo_pack_bytes())
    for window in windows:
        (packs / f"accumulation_{window:02d}h_grib.obspack").write_bytes(
            grid_pack_bytes(
                valid_time="2026-08-03T14:00:00",
                quantity=sources.QUANTITY_PRECIPITATION_ACCUMULATION,
                extra={"accumulation_hours": int(window)}))
    return directory


def _provenance(*, archive, uri, sha256):
    return ObsProvenance(source=archive, product=f"{archive}-product",
                         uri=str(uri), sha256=sha256, fetched_at=FETCHED_AT)


# ------------------------------------------------- the registered lead set

def test_the_reported_leads_come_from_the_registration_and_exclude_lead_zero():
    cli = score_cli()
    document = _registration(leads=tuple(range(2, 19)))

    reported = cli.registered_reported_lead_hours(document)

    # The document's spin-up policy reports the leads BEFORE the first scored
    # hour: with scoring from hour 2, that is hour 1 and nothing else.
    assert reported == [1]
    assert 0 not in reported
    scored = [int(hour) for hour in document["parameters"]["scored_lead_hours"]]
    assert not set(reported) & set(scored)
    # and nothing beyond the scored window, which is what the old hardcoded
    # default asked for and the archive does not carry.
    assert max(reported) < min(scored)


def test_a_registration_that_scores_from_hour_one_reports_no_spinup_lead():
    cli = score_cli()
    assert cli.registered_reported_lead_hours(
        _registration(leads=(1, 2, 3))) == []


def test_an_explicit_lead_zero_is_refused_by_the_spinup_policy():
    cli = score_cli()
    with pytest.raises(ValueError) as refusal:
        cli.parse_reported_lead_hours("0,1,2")
    message = str(refusal.value)
    assert "spin-up policy" in message
    assert "initial condition" in message
    assert cli.parse_reported_lead_hours("1") == [1]


def test_the_command_refuses_lead_zero_before_it_opens_anything(
        monkeypatch, tmp_path, capsys):
    cli = score_cli()
    monkeypatch.setattr(
        "sys.argv",
        ["obs_battery_score.py",
         "--run-directory", str(tmp_path / "no-such-run"),
         "--reported-lead-hours", "0"])

    with pytest.raises(SystemExit) as exit_code:
        cli.main()

    assert exit_code.value.code == 2
    assert "spin-up policy" in capsys.readouterr().err


# ------------------------------------------ the registered window set

def test_every_registered_accumulation_window_gets_its_own_source(tmp_path):
    cli = score_cli()
    document = _registration(leads=(2, 3, 4), windows=(1, 6))

    windows = cli.registered_precipitation_window_hours(document)
    assert windows == [1, 6]

    built = cli.precipitation_sources(_stage4_tree(tmp_path, windows), windows)

    assert sorted(built) == [1, 6]
    assert [built[window].accumulation_hours for window in sorted(built)] == [1, 6]
    assert all(source.quantity() == sources.QUANTITY_PRECIPITATION_ACCUMULATION
               for source in built.values())


def test_the_coverage_floor_comes_from_the_registration_or_its_amendment():
    cli = score_cli()
    document = _registration(leads=(2, 3, 4))
    assert cli.registered_frame_coverage_floor(document) == \
        reg_mod.DEFAULT_FRAME_COVERAGE_FLOOR

    # A document written before amendment v2.1 does not carry the pin, and
    # the amendment's own floor applies -- that is what makes it an amendment
    # rather than a parameter somebody may have forgotten to set.
    document["parameters"]["reflectivity"].pop("frame_coverage_floor")
    assert cli.registered_frame_coverage_floor(document) == \
        reg_mod.DEFAULT_FRAME_COVERAGE_FLOOR

    document["parameters"]["reflectivity"]["frame_coverage_floor"] = 0.75
    assert cli.registered_frame_coverage_floor(document) == 0.75


def test_an_explicit_window_list_parses_in_the_cli_list_style():
    cli = score_cli()
    assert cli.parse_precipitation_window_hours("1,6") == [1, 6]
    assert cli.parse_precipitation_window_hours("6") == [6]
    with pytest.raises(ValueError, match="positive number of hours"):
        cli.parse_precipitation_window_hours("0")
    with pytest.raises(ValueError, match="whole hours"):
        cli.parse_precipitation_window_hours("1,six")


# ------------------------------------------------- the relocated archive

def test_a_relocated_archive_re_hashes_only_through_the_root_override(tmp_path):
    cli = score_cli()
    fetched = tmp_path / "fetched"
    fetched.mkdir()
    artifact = fetched / "obs.grib2.gz"
    artifact.write_bytes(b"the archive object as it arrived")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    directory = _reflectivity_tree(tmp_path, uri=str(artifact), sha256=digest)
    packs = sorted((directory / "packs").glob("frame*.obspack"))
    source = sources.MrmsCompositeSource(packs,
                                         directory / "packs" / "geometry.obspack")
    provenance = _provenance(archive="mrms", uri=artifact, sha256=digest)

    # On the box that fetched the bytes, a root changes nothing.
    assert cli.ObservationRehash({"mrms": source})(provenance) is True

    moved = tmp_path / "relocated"
    moved.mkdir()
    artifact.rename(moved / artifact.name)

    # Without a root the behaviour is the old one exactly: the recorded path,
    # and a wiring fault raised rather than a corruption reported.
    with pytest.raises(FileNotFoundError, match="not on disk"):
        cli.ObservationRehash({"mrms": source})(provenance)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="obs-archive-root"):
        cli.ObservationRehash({"mrms": source}, roots=[empty])(provenance)
    assert cli.ObservationRehash({"mrms": source}, roots=[moved])(provenance) \
        is True


def test_a_same_named_object_under_a_root_fails_rather_than_passing(tmp_path):
    cli = score_cli()
    fetched = tmp_path / "fetched"
    fetched.mkdir()
    artifact = fetched / "obs.grib2.gz"
    artifact.write_bytes(b"the archive object as it arrived")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    directory = _reflectivity_tree(tmp_path, uri=str(artifact), sha256=digest)
    source = sources.MrmsCompositeSource(
        sorted((directory / "packs").glob("frame*.obspack")),
        directory / "packs" / "geometry.obspack")
    provenance = _provenance(archive="mrms", uri=artifact, sha256=digest)
    artifact.unlink()

    decoy = tmp_path / "decoy"
    decoy.mkdir()
    (decoy / "obs.grib2.gz").write_bytes(b"a different file of the same name")
    truth = tmp_path / "relocated"
    truth.mkdir()
    (truth / "obs.grib2.gz").write_bytes(b"the archive object as it arrived")

    # A root holding only the impostor reports a mismatch: the digest decides,
    # so relocation cannot launder a changed object.
    assert cli.ObservationRehash({"mrms": source}, roots=[decoy])(provenance) \
        is False
    # And the real object still verifies when its own directory is named too.
    assert cli.ObservationRehash({"mrms": source},
                                 roots=[decoy, truth])(provenance) is True


def test_the_relocation_root_reaches_every_source_not_only_the_radar(tmp_path):
    cli = score_cli()

    class Recorder:
        """A loud stand-in: it records the routing, it hashes nothing."""

        def __init__(self, name):
            self.name = name
            self.calls = []

        def verify(self, provenance, *, root=None):
            self.calls.append((provenance.source, None if root is None
                               else Path(root)))
            return True

    radar, rain, surface = (Recorder("mrms"), Recorder("stage4"),
                            Recorder("asos"))
    root = tmp_path / "relocated"
    root.mkdir()
    rehash = cli.ObservationRehash(
        {cli.MRMS_ARCHIVE_ID: radar, cli.STAGE4_ARCHIVE_ID: rain,
         cli.ASOS_ARCHIVE_ID: surface},
        roots=[root])

    digest = "c" * 64
    for archive, name in (("mrms", "radar.grib2.gz"),
                          ("stage4", "rain.grib"),
                          ("asos", "observations.csv")):
        assert rehash(_provenance(archive=archive, uri=tmp_path / name,
                                  sha256=digest)) is True

    assert radar.calls == [("mrms", root)]
    assert rain.calls == [("stage4", root)]
    assert surface.calls == [("asos", root)]


# ------------------------------------------------------ the command itself

def test_the_command_takes_its_leads_windows_and_root_from_the_registration(
        monkeypatch, tmp_path, capsys):
    """One run of ``main`` with the readers stood in for, and the four
    command-layer choices read back off what it handed the engine."""
    cli = score_cli()
    document = _registration(leads=tuple(range(2, 19)), windows=(1, 6))
    document_path = tmp_path / "registration.json"
    document_path.write_text(json.dumps(document), encoding="utf-8")
    packs = tmp_path / "packs"
    (packs / "packs").mkdir(parents=True)
    root = tmp_path / "relocated"
    root.mkdir()

    class Arm:
        def record(self):
            return {"reader": "stand-in arm"}

    captured: dict[str, object] = {}

    def fake_score(**kwargs):
        captured.update(kwargs)
        return {"reflectivity": {"primary_scalar": 0.4242}}

    monkeypatch.setattr(cli.model_source, "WrfHistorySource",
                        lambda *a, **k: Arm())
    monkeypatch.setattr(cli, "_packs",
                        lambda directory, pattern="*.obspack": (["frame"],
                                                                "geometry"))
    monkeypatch.setattr(cli, "MrmsCompositeSource",
                        lambda frames, geometry, **kwargs: captured.setdefault(
                            "radar_source_kwargs", kwargs) or "radar-source")
    monkeypatch.setattr(cli, "precipitation_sources",
                        lambda directory, windows: {int(window): f"rain-{window}"
                                                    for window in windows})
    monkeypatch.setattr(cli.battery, "score_case_arm", fake_score)
    monkeypatch.setattr(
        "sys.argv",
        ["obs_battery_score.py",
         "--run-directory", str(tmp_path / "run"),
         "--case-id", "case-fixture", "--arm-id", "faithful",
         "--init-time", INIT,
         "--reflectivity-packs", str(packs),
         "--precipitation-packs", str(packs),
         "--boundary-width-cells", "5",
         "--registration", str(document_path),
         "--obs-archive-root", str(root)])

    assert cli.main() == 0

    assert captured["reported_lead_hours"] == [1]
    assert sorted(captured["precipitation_obs"]) == [1, 6]
    assert captured["radar_source_kwargs"] == {
        "minimum_observed_fraction": reg_mod.DEFAULT_FRAME_COVERAGE_FLOOR}
    rehash = captured["rehash"]
    assert isinstance(rehash, cli.ObservationRehash)
    assert rehash.roots == [root]
    assert "0.4242" in capsys.readouterr().out
