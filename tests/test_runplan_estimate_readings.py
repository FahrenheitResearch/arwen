"""One plan, one card, one figure: the estimate document is the same
number on every reading and on every surface that prices it.

THE DEFECT.  ``gpuwm run-plan --estimate`` quoted one resident plan
three different figures on one idle RTX 4090 -- 1,162,304,164,
1,163,352,740 and 1,166,498,468 bytes -- while the direct
``estimate_experiment`` call answered 1,353,931,428 every time.  Two
things were wrong, at two lines.

The readings moved because the device probe SAMPLED the card's bare
CUDA context as an NVML ``memory.used`` delta either side of its own
context, a card-wide figure in whole MiB: one probe printed 395 MiB
with every other field of the probe identical across readings, and two
earlier receipts of the same plan sit exactly 3 and 4 MiB below it.
That reading went into ``non_pool_device_bytes`` and from there into
``peak_envelope_bytes``, so the document was a sample.  The context is
now priced from the card's shader census, which is what the run door's
own ``Machine`` has always priced.

The fence disagreed because the document prices the card in the
machine and the direct call it was compared with priced the reference
card: the two were about different devices.  The document now states
every device term it priced on (``device_profile``,
``device_total_bytes``) and the cadence, and the fence prices the
direct call on exactly those.

A THIRD SURFACE, found by the same fence: ``gpuwm check`` on a machine
whose card answers the probe but whose kernels do not compile (a CuPy
wheel without CUDA headers) took its CPU-only route and priced the
reference card, so one file on one RTX 4090 read 1,353,931,428 bytes
from ``gpuwm check --json`` and 1,205,295,780 from ``run-plan
--estimate``.  That route now reads the card through the same probe
and prices its census, saying beside the figure that the kernels were
not compiled here; a machine with no readable card keeps the reference
figure and says so.

CPU-only.  The probe is pinned at its subprocess seam or run in a bare
interpreter against a shadow ``cupy``; no test here touches a card.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from gpuwm import domain_wizard
from gpuwm.core import preflight as pf
from gpuwm.experiment import load_experiment
from test_runplan_tiles import (_config, _estimate,
                                resident_estimate_on_the_documents_device)

GIB = 1024 ** 3

#: What this build's probe ships for a card: its census, no sample.
_CARD = {"name": "fixture 68-SM card", "multiprocessor_count": 68,
         "max_threads_per_multiprocessor": 1536,
         "default_stack_limit_bytes": 1024, "compile_platform": None}


def _payload(free_bytes: int, *, total_bytes: int = 10 * GIB,
             profile: dict | None = None) -> dict:
    return {"free_bytes": int(free_bytes), "total_bytes": int(total_bytes),
            "profile": dict(_CARD if profile is None else profile)}


def _six_hour_config(tmp_path):
    """The tiles fixture's HRRR configuration, run for six hours, so the
    boundary cadence the file is priced at moves the LBC term."""

    config = _config(tmp_path)
    text = config.read_text(encoding="utf-8")
    assert "run_seconds = 3600.0" in text
    config.write_text(text.replace("run_seconds = 3600.0",
                                   "run_seconds = 21600.0"),
                      encoding="utf-8")
    return config


# ---------------------------------------------------------------------------
# the document, read repeatedly, is one figure
# ---------------------------------------------------------------------------


def test_the_document_figure_is_one_value_across_readings(tmp_path, monkeypatch):
    """Ten readings of one resident plan on one card: one figure, equal to
    the direct call on the device the document names.  The free-memory
    sample moves between readings, as a shared card's does, and a
    resident plan's figure does not follow it."""

    frees = [8 * GIB - index * 37 * 1024 ** 2 for index in range(10)]
    served = []

    def probe(**_kwargs):
        served.append(frees[len(served)])
        return _payload(served[-1])

    monkeypatch.setattr(pf, "device_memory_probe_subprocess", probe)
    config = _six_hour_config(tmp_path)
    documents = [_estimate(tmp_path, config) for _ in range(10)]

    assert served == frees, "every reading went through the probe"
    figures = {d["vram"]["peak_envelope_bytes"] for d in documents}
    assert len(figures) == 1, figures
    (figure,) = figures
    for document in documents:
        vram = document["vram"]
        assert vram["envelope_basis"] == "resident"
        assert vram["device_basis"] == "measured local device"
        assert vram["device_profile"] == {**_CARD, "bare_context_bytes": None}
        assert vram["device_total_bytes"] == 10 * GIB
        direct = resident_estimate_on_the_documents_device(document, config)
        assert direct.peak_envelope_bytes == figure
        assert vram["estimate_bytes"] == direct.alloc_estimate_bytes
    # The receipts of two readings differ in exactly the term that moved.
    assert [d["vram"]["device_free_bytes"] for d in documents] == frees


def test_a_stated_bare_context_is_priced_as_stated_and_reads_the_same(tmp_path, monkeypatch):
    """A payload that STATES a bare context (a target-hardware sizing
    document) is priced on it, and ten readings of it are one figure: a
    stated number is not a sample."""

    stated = {**_CARD, "bare_context_bytes": 174 * 1024 ** 2}
    monkeypatch.setattr(pf, "device_memory_probe_subprocess",
                        lambda **_: _payload(8 * GIB, profile=stated))
    config = _config(tmp_path)
    documents = [_estimate(tmp_path, config) for _ in range(10)]
    figures = {d["vram"]["peak_envelope_bytes"] for d in documents}
    assert len(figures) == 1
    document = documents[0]
    assert document["vram"]["device_profile"]["bare_context_bytes"] == 174 * 1024 ** 2
    direct = resident_estimate_on_the_documents_device(document, config)
    assert direct.peak_envelope_bytes == document["vram"]["peak_envelope_bytes"]
    assert direct.local_memory_profile.context_is_measured
    # ...and it is a different figure from the census-priced one, which
    # is the whole reason the document has to say which it used.
    monkeypatch.setattr(pf, "device_memory_probe_subprocess",
                        lambda **_: _payload(8 * GIB))
    census = _estimate(tmp_path, config)
    assert census["vram"]["peak_envelope_bytes"] != document["vram"]["peak_envelope_bytes"]


def test_the_document_prices_the_cadence_the_check_prices(tmp_path, monkeypatch):
    """A producer whose fetch takes no cadence flag is priced at the
    cadence it publishes, on the document as on the check.  The document
    used to read only the declared key and price such a file at the
    21,600 s default."""

    from gpuwm.source_adapters import source_forcing_interval_seconds

    monkeypatch.setattr(pf, "device_memory_probe_subprocess", lambda **_: None)
    config = _six_hour_config(tmp_path)
    document = _estimate(tmp_path, config)
    published = source_forcing_interval_seconds("hrrr")
    assert published != pf.DEFAULT_FORCING_INTERVAL_SECONDS
    assert document["vram"]["forcing_interval_seconds"] == published
    exp = load_experiment(config)
    at_published = pf.estimate_experiment(exp, forcing_interval_seconds=published)
    at_default = pf.estimate_experiment(exp)
    assert at_published.peak_envelope_bytes != at_default.peak_envelope_bytes
    assert document["vram"]["peak_envelope_bytes"] == at_published.peak_envelope_bytes


@pytest.mark.parametrize("raw, expected", [
    ({"fetch": {"source": "hrrr", "cadence": 2}}, 7200.0),
    ({"fetch": {"source": "hrrr"}}, 3600.0),
    ({"fetch": {"source": "no-such-producer"}}, None),
    ({}, None),
])
def test_the_recorded_cadence_is_one_read(raw, expected):
    assert pf.recorded_forcing_interval_seconds(raw) == expected


# ---------------------------------------------------------------------------
# `gpuwm check` reports the document's number
# ---------------------------------------------------------------------------


def _check_json(capsys, config, *, free_bytes, total_bytes, profile):
    """``gpuwm check --json`` on the same card, through the wizard's own
    in-process handoff (``_check_emitted_config``): a measured sizing
    sample carrying the card's profile, and the declared free and total
    figures that sample read."""

    from gpuwm.cli import build_parser

    args = build_parser().parse_args([
        "check", str(config), "--json",
        "--free-gib", f"{free_bytes / GIB:.17g}",
        "--vram-gib", f"{total_bytes / GIB:.17g}"])
    args._shared_sizing_budget = domain_wizard.SizingBudget(
        total_bytes / GIB, int(free_bytes), profile, "fixture", measured=True)
    code = args.func(args)
    out = capsys.readouterr().out
    return code, json.loads(out)


@pytest.fixture
def _cpu_check(monkeypatch):
    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    monkeypatch.setattr(pf, "device_physical_total_bytes", lambda **_: None)
    monkeypatch.setattr(pf, "host_available_bytes", lambda: 128 * GIB)


def test_check_reports_the_documents_figure_on_the_same_card(tmp_path, monkeypatch, _cpu_check, capsys):
    """The number ``gpuwm check`` reports for a config and the number
    ``run-plan --estimate`` writes for the same config on the same card
    are one number, to the byte, with the cadence term exercised."""

    payload = _payload(8 * GIB)
    monkeypatch.setattr(pf, "device_memory_probe_subprocess", lambda **_: payload)
    config = _six_hour_config(tmp_path)
    document = _estimate(tmp_path, config)
    profile = pf.profile_from_device_probe(payload)
    assert profile is not None and not profile.context_is_measured
    code, report = _check_json(capsys, config, free_bytes=payload["free_bytes"],
                               total_bytes=payload["total_bytes"], profile=profile)
    assert code in (0, 4), report.get("memory_verdict")
    assert report["observed_peak_envelope_bytes"] == document["vram"]["peak_envelope_bytes"]
    assert report["alloc_estimate_bytes"] == document["vram"]["estimate_bytes"]
    assert report["non_pool_device_bytes"] == pf.non_pool_device_bytes(
        load_experiment(config), profile=profile)
    assert report["local_memory_profile"] == _CARD["name"]


def test_check_reports_the_documents_figure_with_no_card_read(tmp_path, monkeypatch, _cpu_check, capsys):
    """No card readable: both surfaces price the reference profile, and
    the document is the bare direct call."""

    monkeypatch.setattr(pf, "device_memory_probe_subprocess", lambda **_: None)
    config = _six_hour_config(tmp_path)
    document = _estimate(tmp_path, config)
    assert document["vram"]["device_profile"] is None
    assert document["vram"]["device_total_bytes"] is None
    code, report = _check_json(capsys, config, free_bytes=8 * GIB,
                               total_bytes=10 * GIB, profile=None)
    assert code in (0, 4), report.get("memory_verdict")
    assert report["observed_peak_envelope_bytes"] == document["vram"]["peak_envelope_bytes"]
    direct = resident_estimate_on_the_documents_device(document, config)
    assert direct.peak_envelope_bytes == document["vram"]["peak_envelope_bytes"]


# ---------------------------------------------------------------------------
# `gpuwm check` on its CPU-only route reports the document's number
# ---------------------------------------------------------------------------


def _unmet_readiness(monkeypatch):
    """The verdict a CuPy wheel without CUDA headers gets: the compile
    probe cannot run, readiness is ``missing``, and ``check_main`` takes
    the CPU-only route."""

    from gpuwm import doctor

    monkeypatch.setattr(doctor, "_cuda_headers_check", lambda: doctor.Check(
        "CUDA kernel headers", "missing",
        "the compile probe could not be run: no CUDA headers",
        "gpuwm doctor --explain", action="gpuwm doctor --explain",
        brief="CUDA probe unavailable", blocking=True))


def _check_cpu_only(capsys, config, *flags):
    """``gpuwm check CONFIG`` with no declared budget: the readiness gate
    decides the route, and with it unmet the CPU-only estimate answers."""

    from gpuwm.cli import build_parser

    args = build_parser().parse_args(["check", str(config), *flags])
    code = args.func(args)
    return code, capsys.readouterr().out


def test_the_cpu_only_check_route_reports_the_documents_figure_on_the_same_card(
        tmp_path, monkeypatch, _cpu_check, capsys):
    """Kernels not compiled here, a card answering the probe: ``gpuwm
    check`` prices that card's census, the same profile and capacity
    the estimate document prices, and the two figures are one figure
    to the byte.  The basis beside it names the card, says the kernels
    were not compiled on it, and says why."""

    payload = _payload(8 * GIB)
    monkeypatch.setattr(pf, "device_memory_probe_subprocess", lambda **_: payload)
    _unmet_readiness(monkeypatch)
    config = _six_hour_config(tmp_path)
    document = _estimate(tmp_path, config)
    assert document["vram"]["device_profile"]["name"] == _CARD["name"]
    code, out = _check_cpu_only(capsys, config, "--json")
    report = json.loads(out)
    assert code == 1
    assert report["gpu_readiness"]["status"] == "missing"
    required = report["required_memory"]
    assert required["status"] == "estimated"
    assert required["resident_forecast_peak_envelope_bytes"] == document["vram"]["peak_envelope_bytes"]
    assert required["alloc_estimate_bytes"] == document["vram"]["estimate_bytes"]
    assert required["device_read"] is True
    assert required["local_memory_profile"] == _CARD["name"]
    assert required["device_total_bytes"] == payload["total_bytes"]
    assert required["device_basis"] == "measured local device; kernels not compiled here"
    profile = pf.profile_from_device_probe(payload)
    assert required["non_pool_device_bytes"] == pf.non_pool_device_bytes(
        load_experiment(config), profile=profile)
    assert required["non_pool_basis"] == pf.non_pool_basis(profile, load_experiment(config))
    assert required["basis"].startswith(
        "CPU-only metadata; resident alternative priced on the card read in "
        "this machine, kernels not compiled here (this machine's GPU "
        "readiness is missing (CUDA probe unavailable)): ")
    assert f"read on this card ({_CARD['name']}, 68 SMs x 1536 threads)" in required["basis"]
    # The direct call on the device the document names is the same figure.
    direct = resident_estimate_on_the_documents_device(document, config)
    assert direct.peak_envelope_bytes == required["resident_forecast_peak_envelope_bytes"]
    # The text form says the same beside the same figure.
    code, out = _check_cpu_only(capsys, config)
    assert code == 1
    envelope = required["resident_forecast_peak_envelope_bytes"] / GIB
    assert (f"{envelope:.2f} GiB forecast envelope; on {_CARD['name']} as "
            "read here, kernels not compiled.") in out
    assert "  Basis: CPU-only metadata; resident alternative priced on the card read in this machine" in out


def test_the_cpu_only_check_route_keeps_the_reference_figure_with_no_card_and_says_so(
        tmp_path, monkeypatch, _cpu_check, capsys):
    """No card answers the probe: both surfaces price the reference
    profile, the figures are one figure, and the report says the card
    was not read and why."""

    monkeypatch.setattr(pf, "device_memory_probe_subprocess", lambda **_: None)
    monkeypatch.setattr(pf, "device_memory_probe_reason",
                        lambda **_: "no CUDA device answered")
    _unmet_readiness(monkeypatch)
    config = _six_hour_config(tmp_path)
    document = _estimate(tmp_path, config)
    assert document["vram"]["device_profile"] is None
    code, out = _check_cpu_only(capsys, config, "--json")
    report = json.loads(out)
    assert code == 1
    required = report["required_memory"]
    assert required["status"] == "estimated"
    assert required["resident_forecast_peak_envelope_bytes"] == document["vram"]["peak_envelope_bytes"]
    assert required["alloc_estimate_bytes"] == document["vram"]["estimate_bytes"]
    assert required["device_read"] is False
    assert required["local_memory_profile"] == pf.MEASURED_LOCAL_MEMORY_PROFILE.name
    assert required["device_total_bytes"] is None
    assert required["device_basis"] == (
        "conservative reference; local device unmeasured (no CUDA device answered)")
    assert required["basis"] == ("CPU-only metadata; resident alternative with "
                                 "conservative reference GPU overhead")
    direct = resident_estimate_on_the_documents_device(document, config)
    assert direct.peak_envelope_bytes == required["resident_forecast_peak_envelope_bytes"]
    code, out = _check_cpu_only(capsys, config)
    assert code == 1
    assert "GiB forecast envelope; on the reference card, none read here." in out
    assert "  Basis:" not in out


def test_the_cpu_only_check_route_prices_a_declared_card_that_is_elsewhere_on_the_reference(
        tmp_path, monkeypatch, _cpu_check, capsys):
    """``--vram-gib`` naming a capacity that is not the read card's is a
    machine that is elsewhere, priced on the reference profile as the
    measured route prices it; naming the read card's own capacity is
    the read card."""

    payload = _payload(8 * GIB, total_bytes=10 * GIB)
    monkeypatch.setattr(pf, "device_memory_probe_subprocess", lambda **_: payload)
    _unmet_readiness(monkeypatch)
    config = _six_hour_config(tmp_path)
    document = _estimate(tmp_path, config)
    code, out = _check_cpu_only(capsys, config, "--json", "--vram-gib", "48")
    elsewhere = json.loads(out)["required_memory"]
    assert elsewhere["device_read"] is False
    assert elsewhere["local_memory_profile"] == pf.MEASURED_LOCAL_MEMORY_PROFILE.name
    assert elsewhere["device_basis"].startswith(
        "conservative reference; local device unmeasured (--vram-gib 48 names "
        f"a card that is not the one read in this machine ({_CARD['name']})")
    assert elsewhere["resident_forecast_peak_envelope_bytes"] == pf.estimate_experiment(
        load_experiment(config), vram_gib=48.0,
        forcing_interval_seconds=document["vram"]["forcing_interval_seconds"]).peak_envelope_bytes
    code, out = _check_cpu_only(capsys, config, "--json", "--vram-gib", "10")
    same = json.loads(out)["required_memory"]
    assert same["device_read"] is True
    assert same["local_memory_profile"] == _CARD["name"]
    assert same["resident_forecast_peak_envelope_bytes"] == document["vram"]["peak_envelope_bytes"]


@pytest.mark.parametrize("declared,total,expected", [
    (None, 10 * GIB, False),
    (10.0, None, False),
    (10.0, 10 * GIB, True),
    (10.0, 10 * GIB + 40 * 1024 ** 2, True),
    (9.9, 10 * GIB, False),
    (48.0, 10 * GIB, False),
])
def test_a_declared_capacity_names_the_card_whose_total_it_is(declared, total, expected):
    assert pf.declares_this_card(declared, total) is expected


# ---------------------------------------------------------------------------
# the instrument: a profile is the card's census, never a sample
# ---------------------------------------------------------------------------


class _Runtime:
    @staticmethod
    def memGetInfo():
        return 8 * GIB, 10 * GIB

    @staticmethod
    def getDeviceProperties(device):
        return {"name": b"fixture 68-SM card", "multiProcessorCount": 68,
                "maxThreadsPerMultiProcessor": 1536}

    @staticmethod
    def deviceGetLimit(limit):
        return 1024


class _FakeCupy:
    class cuda:
        runtime = _Runtime


def test_the_live_profile_is_the_census_and_takes_no_reading(monkeypatch):
    """``local_memory_profile_from_device`` reads constants of the card
    and nothing else: no ``nvidia-smi`` either side of the first CUDA
    call, no bare-context delta, and two reads are one profile."""

    from gpuwm import supervisor

    def never(*_args, **_kwargs):
        raise AssertionError("the live profile read nvidia-smi")

    monkeypatch.setattr(supervisor, "_run_nvidia_smi", never)
    monkeypatch.setattr(pf, "read_compile_platform", lambda: ("99", "0.0.1"))
    first = pf.local_memory_profile_from_device(_FakeCupy)
    second = pf.local_memory_profile_from_device(_FakeCupy)
    assert first == second
    assert first.bare_context_bytes is None and not first.context_is_measured
    assert first.multiprocessor_count == 68
    assert first.cuda_context_bytes == (
        pf.MODELLED_BARE_CONTEXT_BYTES_PER_RESIDENT_THREAD * 68 * 1536
        + pf.CONTEXT_RUNTIME_GROWTH_BYTES)
    basis = pf.non_pool_basis(first)
    assert basis.startswith("read on this card (fixture 68-SM card, 68 SMs x 1536 threads)")
    assert "modelled from its shader census" in basis
    assert "sm_99 / NVRTC 0.0.1" in basis


def test_the_probe_ships_the_census_and_no_sample(tmp_path):
    """The probe's own source, run twice in a bare interpreter against a
    shadow ``cupy`` and no ``nvidia-smi``: the profile half is the card's
    census with no bare-context reading, and the two runs are identical."""

    script = tmp_path / "probe.py"
    script.write_text(pf._DEVICE_MEMORY_PROBE_SOURCE, encoding="utf-8")
    shadow = tmp_path / "shadow"
    (shadow / "cupy").mkdir(parents=True)
    (shadow / "cupy" / "__init__.py").write_text(textwrap.dedent("""\
        class _Runtime:
            @staticmethod
            def deviceGetPCIBusId(device):
                return "0000:01:00.0"

            @staticmethod
            def memGetInfo():
                return 8 * 1024 ** 3, 10 * 1024 ** 3

            @staticmethod
            def getDeviceProperties(device):
                return {"name": b"fixture 68-SM card",
                        "multiProcessorCount": 68,
                        "maxThreadsPerMultiProcessor": 1536}

            @staticmethod
            def deviceGetLimit(limit):
                return 1024


        class _Device:
            def __init__(self, index):
                self.compute_capability = "99"


        class cuda:
            runtime = _Runtime
            Device = _Device
        """), encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(shadow)
    env["PATH"] = str(shadow)          # no nvidia-smi anywhere on it
    # Devices stay visible, so on Linux the probe does look for nvidia-smi
    # and finds none: the absent-utility path is the one measured here.
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("PYTHONSAFEPATH", None)
    payloads = []
    for _ in range(2):
        done = subprocess.run([sys.executable, str(script)],
                              capture_output=True, text=True, env=env,
                              cwd=str(tmp_path), timeout=300)
        assert done.returncode == 0, done.stderr
        payloads.append(json.loads(done.stdout.strip().splitlines()[-1]))
    payload = payloads[0]
    assert payload["profile"] == _CARD
    assert payloads[1]["profile"] == _CARD
    assert "bare_context_bytes" not in payload["profile"]
    assert payload["free_bytes_memgetinfo"] == 8 * GIB
    assert payload["total_bytes"] == 10 * GIB == payloads[1]["total_bytes"]
    if sys.platform != "win32":
        # PATH is the whole search on Linux, so no nvidia-smi answers and
        # the free figure is the shadow runtime's.  Windows resolves the
        # bare name from System32 ahead of PATH, so there the probe reads
        # the real card's NVML and its free half is that card's sample.
        assert payloads[0] == payloads[1]
        assert payload["free_bytes"] == 8 * GIB
        assert payload["free_bytes_nvml"] is None
    profile = pf.profile_from_device_probe(payload)
    assert profile == pf.DeviceLocalMemoryProfile(
        "fixture 68-SM card", 68, 1536, 1024)
    assert not profile.context_is_measured
