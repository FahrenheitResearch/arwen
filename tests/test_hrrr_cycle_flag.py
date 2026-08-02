"""One `--valid-time`, two meanings: the collision and its resolution.

``--valid-time`` shipped on the two HRRR entry points the wizard prints,
and they read it differently:

* ``tools/prepare_hrrr_wrf.py`` treats it as the CYCLE -- it opens
  ``hrrr.tHHz.wrfnatfNN.grib2`` with that hour and adds the lead to get
  model time zero;
* ``gpuwm/hrrr_hierarchy_direct.py`` treats it as MODEL TIME ZERO -- it
  is compared to the namelist's ``start_*`` keys and to the sealed
  cache's ``initial_valid_time``.

At lead 0 those are the same instant, which is why four releases never
noticed.  At lead K the wizard handed the same string to both and one of
them was wrong by K hours.

The resolution: **the typed time is always the cycle** (``--cycle``), and
model time zero is always derived from it and ``--forecast-start-hour``.
``--valid-time`` stays accepted on each command with exactly the meaning
it had in v1.4.0 THERE, so a v1.4.0 script keeps working; passing both is
refused rather than ranked.
"""

from datetime import datetime

import pytest

from gpuwm.hrrr_forecast import parse_hrrr_cycle, resolve_cycle_flags


CYCLE = "2026-07-18_05:00:00"


# ---------------------------------------------------------------------------
# The shared resolver
# ---------------------------------------------------------------------------

def test_cycle_wins_valid_time_warns_and_both_together_are_refused():
    said = []
    assert resolve_cycle_flags(
        CYCLE, None, tool="t", legacy_means="the HRRR cycle",
        warn=said.append) == (datetime(2026, 7, 18, 5), False)
    assert said == []

    instant, legacy = resolve_cycle_flags(
        None, CYCLE, tool="t", legacy_means="the HRRR cycle",
        warn=said.append)
    assert (instant, legacy) == (datetime(2026, 7, 18, 5), True)
    assert len(said) == 1
    assert "--valid-time is deprecated" in said[0]
    assert "the HRRR cycle" in said[0]

    with pytest.raises(ValueError, match="not both"):
        resolve_cycle_flags(CYCLE, CYCLE, tool="t", legacy_means="x")
    with pytest.raises(ValueError, match="--cycle is required"):
        resolve_cycle_flags(None, None, tool="t", legacy_means="x")


def test_a_cycle_must_be_an_exact_hour_in_wrf_form():
    assert parse_hrrr_cycle(CYCLE) == datetime(2026, 7, 18, 5)
    with pytest.raises(ValueError, match="YYYY-MM-DD_HH:MM:SS"):
        parse_hrrr_cycle("2026-07-18T05")
    with pytest.raises(ValueError, match="exact hourly HRRR cycle"):
        parse_hrrr_cycle("2026-07-18_05:30:00")


# ---------------------------------------------------------------------------
# The preparer: --valid-time always meant the cycle here
# ---------------------------------------------------------------------------

def _prepared(tmp_path, monkeypatch, *time_flags):
    """Run the preparer far enough to capture the commands it composes."""

    import tools.prepare_hrrr_wrf as prepare

    source = tmp_path / "src"
    source.mkdir(parents=True)
    for hour in (12, 13):
        (source / f"hrrr.t05z.wrfnatf{hour}.grib2").write_bytes(b"g")
        (source / f"hrrr.t05z.soilf{hour}.grib2").write_bytes(b"g")
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text("", encoding="utf-8")
    namelist = tmp_path / "namelist.input"
    namelist.write_text("", encoding="utf-8")
    static_cache = tmp_path / "static.npz"
    static_cache.write_bytes(b"s")
    static_receipt = tmp_path / "static.json"
    static_receipt.write_text("{}", encoding="utf-8")

    commands = []

    def fake_run(command, env):
        commands.append(list(command))
        if any(str(value).endswith("hrrr_single_domain_benchmark.py")
               for value in command):
            # Enough of a report for the wrapper's own validation to pass.
            raise _Captured(commands)

    class _Captured(Exception):
        pass

    monkeypatch.setattr(prepare, "_run", fake_run)
    with pytest.raises(_Captured):
        prepare.main([
            "--source-root", str(source),
            "--source-manifest", str(manifest),
            "--source-manifest-sha256", "0" * 64,
            "--static-cache", str(static_cache),
            "--static-receipt", str(static_receipt),
            "--namelist-input", str(namelist),
            *time_flags,
            "--forecast-start-hour", "12",
            "--forecast-end-hour", "13",
            "--run-seconds", "3600",
            "--output-root", str(tmp_path / "out"),
        ])
    return commands


def test_the_preparer_reads_cycle_and_the_deprecated_valid_time_identically(
        tmp_path, monkeypatch):
    """Backward compatibility, proved by comparing the composed commands."""

    named = _prepared(tmp_path / "a", monkeypatch, "--cycle", CYCLE)
    legacy = _prepared(tmp_path / "b", monkeypatch, "--valid-time", CYCLE)

    def normalized(commands):
        benchmark = next(
            command for command in commands
            if any(str(value).endswith("hrrr_single_domain_benchmark.py")
                   for value in command))
        return [item for item in benchmark
                if "\\" not in str(item) and "/" not in str(item)]

    assert normalized(named) == normalized(legacy)
    # And the benchmark is told the CYCLE plus the lead, never a
    # pre-computed model start.
    benchmark = next(
        command for command in named
        if any(str(value).endswith("hrrr_single_domain_benchmark.py")
               for value in command))
    assert benchmark[benchmark.index("--cycle") + 1] == CYCLE
    assert benchmark[benchmark.index("--forecast-start-hour") + 1] == "12"
    assert "--valid-time" not in benchmark


# ---------------------------------------------------------------------------
# The hierarchy: --valid-time always meant MODEL TIME ZERO here
# ---------------------------------------------------------------------------

def _hierarchy_instant(monkeypatch, tmp_path, *time_flags) -> datetime:
    """The instant the hierarchy will check the namelist and cache against."""

    import gpuwm.hrrr_hierarchy_direct as hierarchy

    seen = {}

    def fake_prepare(**kwargs):
        seen["valid_time"] = kwargs["valid_time"]
        raise _Stop()

    class _Stop(Exception):
        pass

    monkeypatch.setattr(hierarchy, "prepare_hrrr_hierarchy", fake_prepare)
    with pytest.raises(_Stop):
        hierarchy.main([
            "--root-preparation", str(tmp_path),
            "--root-domain-spec", str(tmp_path / "d01.json"),
            "--wps-namelist", str(tmp_path / "namelist.wps"),
            "--namelist-input", str(tmp_path / "namelist.input"),
            "--stock-wrf-namelist-input", str(tmp_path / "stock.input"),
            "--geog-root", str(tmp_path),
            "--source-manifest", str(tmp_path / "SHA256SUMS"),
            "--source-manifest-sha256", "0" * 64,
            *time_flags,
            "--output-root", str(tmp_path / "tree"),
        ])
    return seen["valid_time"]


def test_the_hierarchy_derives_model_time_zero_from_the_cycle_and_the_lead(
        tmp_path, monkeypatch):
    # New spelling: cycle + lead.
    assert _hierarchy_instant(
        monkeypatch, tmp_path, "--cycle", CYCLE,
        "--forecast-start-hour", "12") == datetime(2026, 7, 18, 17)
    # Lead 0: the cycle IS model time zero, which is the only case that
    # could ever be expressed before.
    assert _hierarchy_instant(
        monkeypatch, tmp_path, "--cycle", CYCLE) == datetime(2026, 7, 18, 5)
    # Deprecated spelling: unchanged from v1.4.0 -- the string IS model
    # time zero and nothing is added to it.
    assert _hierarchy_instant(
        monkeypatch, tmp_path, "--valid-time",
        "2026-07-18_17:00:00") == datetime(2026, 7, 18, 17)


def test_the_hierarchy_refuses_a_lead_added_to_the_deprecated_flag(
        tmp_path, monkeypatch):
    """Negative control: the one combination that would move the clock twice."""

    with pytest.raises(ValueError, match="--forecast-start-hour needs --cycle"):
        _hierarchy_instant(
            monkeypatch, tmp_path, "--valid-time", CYCLE,
            "--forecast-start-hour", "12")


def test_the_refusals_name_both_instants_and_the_pair_that_fixes_them():
    """The exact shape of the collision, and what each refusal says about it.

    A root prepared from cycle 05Z at lead f12 has model time zero 17Z.
    Handing this stage the cycle -- which is what the wizard's chain used
    to do -- is off by 12 hours in both checks.
    """

    from gpuwm.hrrr_hierarchy_direct import (
        namelist_start_refusal, sealed_start_refusal)

    sealed = sealed_start_refusal(
        requested=datetime(2026, 7, 18, 5),
        sealed_start=datetime(2026, 7, 18, 17),
        sealed_cycle="2026-07-18T05:00:00")
    assert "2026-07-18 17:00:00" in sealed and "2026-07-18 05:00:00" in sealed
    assert "at lead f12" in sealed
    assert "--cycle CYCLE --forecast-start-hour K" in sealed

    # A cache with no recorded cycle still names both instants; it just
    # cannot name the lead.
    silent = sealed_start_refusal(
        requested=datetime(2026, 7, 18, 5),
        sealed_start=datetime(2026, 7, 18, 17), sealed_cycle=None)
    assert "at lead" not in silent
    assert "2026-07-18 17:00:00" in silent

    namelist = namelist_start_refusal(
        requested=datetime(2026, 7, 18, 17),
        observed=datetime(2026, 7, 18, 5),
        namelist_name="case.namelist.input")
    assert "case.namelist.input starts at 2026-07-18 05:00:00" in namelist
    assert "gpuwm domain --forecast-start-hour K" in namelist


# ---------------------------------------------------------------------------
# The other place the two hour numberings meet
# ---------------------------------------------------------------------------

def test_the_hierarchy_reads_the_bridge_by_absolute_lead_not_model_hour():
    """The nested route's own version of the same confusion.

    A prepared cache's ``forcing_hours`` are MODEL-relative -- 0, 1, 2 --
    because that is what a forecast clock counts.  The decoder's published
    tree is keyed by NOAA's absolute lead: ``atmosphere-f06``, gate
    ``(6, 7, 8)``.  Equal at lead 0; at lead 6 the hierarchy asked that
    tree for hour 0 and a root prepared from f06 refused with
    "forecast_hour must be one of (6, 7, 8), got 0" -- after every check
    above it had passed.
    """

    from gpuwm.hrrr_hierarchy_direct import sealed_source_leads

    sealed = {"source_identity": {"source_forecast_hours": [6, 7, 8]}}
    assert sealed_source_leads(sealed, (0, 1, 2)) == (6, 7, 8)
    # Lead 0: the two numberings coincide, which is the only case the
    # releases before this one could express.
    assert sealed_source_leads(
        {"source_identity": {"source_forecast_hours": [0, 1, 2]}},
        (0, 1, 2)) == (0, 1, 2)
    # A cache sealed before the field existed carries only the
    # model-relative inventory, and at the lead 0 it must have been, that
    # IS the absolute one.  Exact, not a guess.
    assert sealed_source_leads({"source_identity": {}}, (0, 1, 2)) == (0, 1, 2)
    assert sealed_source_leads({}, (0, 1, 2)) == (0, 1, 2)


def test_a_sealed_lead_inventory_that_cannot_be_true_is_refused():
    """Negative control: neither length nor contiguity is assumed."""

    from gpuwm.hrrr_hierarchy_direct import sealed_source_leads

    with pytest.raises(ValueError, match="model forcing hour"):
        sealed_source_leads(
            {"source_identity": {"source_forecast_hours": [6, 7]}}, (0, 1, 2))
    with pytest.raises(ValueError, match="contiguous"):
        sealed_source_leads(
            {"source_identity": {"source_forecast_hours": [6, 8, 9]}},
            (0, 1, 2))
