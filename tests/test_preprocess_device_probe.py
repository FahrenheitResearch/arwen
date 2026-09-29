"""Live load comes from the CUDA device selected for preparation."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gpuwm.core import preflight


GIB = 1024 ** 3


def _raise_or_return(value):
    if isinstance(value, BaseException):
        raise value
    return value


def _exec_probe(monkeypatch, capsys, *, device, rows, devices,
                visible_devices="1,0"):
    """Run the probe source in this interpreter; ``(exit code, stdout, calls)``.

    ``rows`` are nvidia-smi rows keyed by query field (a field a row does
    not name prints ``[N/A]``, as nvidia-smi does).  A device's ``pci`` or
    ``free`` may be an exception, raised when the probe asks for it, and
    its ``properties`` replace the device properties the probe reads.
    """

    active = [0]
    calls = []

    def run(argv, **_kwargs):
        assert argv[0] == "nvidia-smi"
        calls.append("nvml")
        query = next(value for value in argv if value.startswith("--query-gpu="))
        fields = query.split("=", 1)[1].split(",")
        output = "\n".join(
            ", ".join(str(row.get(field, "[N/A]")) for field in fields)
            for row in rows)
        return SimpleNamespace(returncode=0, stdout=output)

    def set_device(value):
        calls.append(("set_device", value))
        active[0] = value

    def mem_get_info():
        calls.append(("memory", active[0]))
        return _raise_or_return(devices[active[0]]["free"]), 32 * GIB

    def device_properties(value):
        calls.append(("properties", value))
        return devices[value].get("properties", {
            "name": f"device-{value}", "multiProcessorCount": 64,
            "maxThreadsPerMultiProcessor": 1536})

    cp = SimpleNamespace(cuda=SimpleNamespace(
        runtime=SimpleNamespace(
            getDevice=lambda: active[0],
            setDevice=set_device,
            deviceGetPCIBusId=lambda value: _raise_or_return(
                devices[value]["pci"]),
            memGetInfo=mem_get_info,
            getDeviceProperties=device_properties,
            deviceGetLimit=lambda _limit: 1024),
        Device=lambda _value: SimpleNamespace(compute_capability="120")))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_devices)
    monkeypatch.setitem(sys.modules, "cupy", cp)
    monkeypatch.setattr(sys, "argv", ["probe", str(device)])
    monkeypatch.setattr(subprocess, "run", run)
    code = None
    try:
        exec(compile(preflight._DEVICE_MEMORY_PROBE_SOURCE, "device_probe",
                     "exec"), {})
    except SystemExit as stop:
        code = stop.code
    if visible_devices not in ("", "-1"):
        assert calls[0] == "nvml", "the memory sample must precede CUDA context creation"
    return code, capsys.readouterr().out, calls


def _run_probe(monkeypatch, capsys, **kwargs):
    code, out, calls = _exec_probe(monkeypatch, capsys, **kwargs)
    assert code is None, f"the probe exited {code} instead of measuring"
    return json.loads(out), calls


@pytest.mark.parametrize(("used_mib", "utilization"), [
    (2 * 1024, 7),
    (29 * 1024, 93),
])
def test_probe_matches_visible_device_to_physical_pci_identity(
        monkeypatch, capsys, used_mib, utilization):
    rows = [
        {"pci.bus_id": "00000000:01:00.0", "memory.used": 31 * 1024,
         "utilization.gpu": 99},
        {"pci.bus_id": "00000000:AF:00.0", "memory.used": used_mib,
         "utilization.gpu": utilization},
    ]
    payload, _ = _run_probe(monkeypatch, capsys, device=0, rows=rows, devices={
        0: {"pci": b"0000:af:00.0", "free": 31 * GIB}})
    assert payload["free_bytes_nvml"] == 32 * GIB - used_mib * 1024 ** 2
    assert payload["free_bytes"] == payload["free_bytes_nvml"]
    assert payload["utilization_gpu_percent"] == utilization


def test_probe_selects_the_current_nonzero_cuda_device(monkeypatch, capsys):
    rows = [
        {"pci.bus_id": "00000000:01:00.0", "memory.used": 2 * 1024,
         "utilization.gpu": 1},
        {"pci.bus_id": "00000000:02:00.0", "memory.used": 28 * 1024,
         "utilization.gpu": 87},
    ]
    payload, calls = _run_probe(monkeypatch, capsys, device=1, rows=rows, devices={
        0: {"pci": "0000:01:00.0", "free": 30 * GIB},
        1: {"pci": "0000:02:00.0", "free": 3 * GIB},
    })
    assert payload["free_bytes_memgetinfo"] == 3 * GIB
    assert payload["free_bytes_nvml"] == 4 * GIB
    assert payload["free_bytes"] == 3 * GIB
    assert payload["utilization_gpu_percent"] == 87
    assert payload["profile"]["name"] == "device-1"
    assert ("set_device", 1) in calls
    assert ("properties", 1) in calls


def test_probe_keeps_memory_when_utilization_is_unsupported(monkeypatch, capsys):
    rows = [
        {"pci.bus_id": "00000000:01:00.0", "memory.used": 31 * 1024,
         "utilization.gpu": 99},
        {"pci.bus_id": "00000000:02:00.0", "memory.used": 27 * 1024,
         "utilization.gpu": "[N/A]"},
    ]
    payload, _ = _run_probe(monkeypatch, capsys, device=0, rows=rows, devices={
        0: {"pci": "0000:02:00.0", "free": 30 * GIB}})
    assert payload["free_bytes_nvml"] == 5 * GIB
    assert payload["free_bytes"] == 5 * GIB
    assert payload["utilization_gpu_percent"] is None


def test_probe_uses_runtime_memory_when_nvml_has_no_matching_device(
        monkeypatch, capsys):
    # Two cards and neither bus ID is the selected device's: no row can be
    # attributed to it, so none is (the one-card case is below).
    rows = [
        {"pci.bus_id": "00000000:01:00.0", "memory.used": 31 * 1024,
         "utilization.gpu": 99},
        {"pci.bus_id": "00000000:03:00.0", "memory.used": 30 * 1024,
         "utilization.gpu": 98},
    ]
    payload, _ = _run_probe(monkeypatch, capsys, device=0, rows=rows, devices={
        0: {"pci": "0000:02:00.0", "free": 30 * GIB}})
    assert payload["free_bytes"] == 30 * GIB
    assert payload["free_bytes_nvml"] is None
    assert payload["utilization_gpu_percent"] is None


@pytest.mark.parametrize("pci", [
    "GPU-5e1f:65:00.0",
    RuntimeError("cudaErrorNotSupported: operation not supported"),
])
def test_a_one_card_box_keeps_its_nvml_row_when_the_bus_id_does_not_match(
        monkeypatch, capsys, pci):
    """nvidia-smi lists every card the machine has, so one row is the
    selected device whatever the bus-ID read says.  Dropping it drops the
    WDDM ceiling: memGetInfo over-stated a loaded RTX 3080 desktop's free
    memory by 5.7 GiB on 2026-08-20, and an error in the bus-ID read threw
    away the whole fit measurement."""

    rows = [{"pci.bus_id": "00000000:01:00.0", "memory.used": 27 * 1024,
             "utilization.gpu": 12, "memory.total": 32 * 1024}]
    payload, _ = _run_probe(monkeypatch, capsys, device=0, rows=rows, devices={
        0: {"pci": pci, "free": 30 * GIB}})
    assert payload["free_bytes_memgetinfo"] == 30 * GIB
    assert payload["free_bytes_nvml"] == 5 * GIB
    assert payload["free_bytes"] == 5 * GIB
    assert payload["utilization_gpu_percent"] == 12
    assert payload["profile"]["name"] == "device-0"


def test_a_bus_id_error_on_a_multi_card_box_keeps_the_fit_without_guessing(
        monkeypatch, capsys):
    rows = [
        {"pci.bus_id": "00000000:01:00.0", "memory.used": 2 * 1024,
         "utilization.gpu": 1},
        {"pci.bus_id": "00000000:02:00.0", "memory.used": 28 * 1024,
         "utilization.gpu": 87},
    ]
    payload, _ = _run_probe(monkeypatch, capsys, device=0, rows=rows, devices={
        0: {"pci": RuntimeError("cudaErrorNotSupported"), "free": 30 * GIB}})
    assert payload["free_bytes"] == 30 * GIB
    assert payload["free_bytes_nvml"] is None
    assert payload["utilization_gpu_percent"] is None
    assert payload["profile"]["name"] == "device-0"


@pytest.mark.parametrize("failing", ["free", "pci_and_free"])
def test_probe_prints_the_nvml_sample_when_cuda_cannot_open_the_card(
        monkeypatch, capsys, failing):
    """A card too full for a context, or held in exclusive-process mode,
    fails inside CUDA; the NVML sample taken before it is still printed so
    automatic backend selection can see how full and busy the card is."""

    rows = [{"pci.bus_id": "00000000:01:00.0", "memory.used": 31 * 1024,
             "utilization.gpu": 97, "memory.total": 32 * 1024}]
    device = {"pci": "0000:01:00.0",
              "free": RuntimeError("cudaErrorMemoryAllocation: out of memory")}
    if failing == "pci_and_free":
        device["pci"] = RuntimeError("cudaErrorDevicesUnavailable")
    code, out, _ = _exec_probe(monkeypatch, capsys, device=0, rows=rows,
                               devices={0: device})
    assert code == 3
    report = json.loads(out.strip().splitlines()[-1])
    assert "cudaErrorMemoryAllocation: out of memory" in report["cuda_error"]
    assert report["nvml"] == {"used_bytes": 31 * GIB, "total_bytes": 32 * GIB,
                              "utilization_gpu_percent": 97}


#: Two cards listed, and the second row did not parse (nvidia-smi prints
#: ``[N/A]`` for a field a card cannot report).
_ONE_ROW_UNPARSED = [
    {"pci.bus_id": "00000000:01:00.0", "memory.used": 31 * 1024,
     "utilization.gpu": 97, "memory.total": 32 * 1024},
    {"pci.bus_id": "00000000:02:00.0", "memory.used": "[N/A]",
     "utilization.gpu": "[N/A]"},
]


@pytest.mark.parametrize("pci", [
    "0000:02:00.0",
    RuntimeError("cudaErrorNotSupported: operation not supported"),
])
def test_a_row_that_did_not_parse_still_counts_as_a_listed_card(
        monkeypatch, capsys, pci):
    """The one-card rule is about the cards nvidia-smi LISTED, not the rows
    that parsed.  Counting parsed rows handed the selected card (the one
    whose row read ``[N/A]``) the other card's 31 GiB used and 97% busy,
    which put a full card's free-memory ceiling on an idle one."""

    payload, _ = _run_probe(monkeypatch, capsys, device=0,
                            rows=_ONE_ROW_UNPARSED,
                            devices={0: {"pci": pci, "free": 30 * GIB}})
    assert payload["free_bytes_memgetinfo"] == 30 * GIB
    assert payload["free_bytes_nvml"] is None
    assert payload["free_bytes"] == 30 * GIB
    assert payload["utilization_gpu_percent"] is None


def test_a_cuda_failure_on_a_card_whose_row_did_not_parse_prints_no_sample(
        monkeypatch, capsys):
    code, out, _ = _exec_probe(monkeypatch, capsys, device=0,
                               rows=_ONE_ROW_UNPARSED,
                               devices={0: {
                                   "pci": "0000:02:00.0",
                                   "free": RuntimeError("cudaErrorUnknown")}})
    assert code == 3
    report = json.loads(out.strip().splitlines()[-1])
    assert "cudaErrorUnknown" in report["cuda_error"]
    assert report["nvml"] is None


class CUDARuntimeError(RuntimeError):
    """Named as CuPy names its runtime error class."""


def test_a_cuda_runtime_error_is_known_by_its_class_as_well_as_its_code(
        monkeypatch, capsys):
    rows = [{"pci.bus_id": "00000000:01:00.0", "memory.used": 31 * 1024,
             "utilization.gpu": 97, "memory.total": 32 * 1024}]
    code, out, _ = _exec_probe(monkeypatch, capsys, device=0, rows=rows,
                               devices={0: {
                                   "pci": "0000:01:00.0",
                                   "free": CUDARuntimeError("out of memory")}})
    assert code == 3
    report = preflight._probe_failure_report(out)
    assert report["cuda_error"] == "CUDARuntimeError: out of memory"
    assert report["nvml"]["used_bytes"] == 31 * GIB


def test_a_probe_error_that_is_not_cuda_s_does_not_send_auto_to_the_cpu(
        monkeypatch, capsys):
    """The card opened, and then the probe itself failed (here a device
    property it reads under another name).  That is not CUDA refusing the
    card, so it prints no CUDA failure report: automatic backend selection
    used to read any exception as "CUDA could not open the card" and move
    a preparation off a card its own context would have opened."""

    from gpuwm.ingest import preprocess_backend as backend
    from gpuwm.local_gpu import NO_LOCAL_GPU_ENV

    rows = [{"pci.bus_id": "00000000:01:00.0", "memory.used": 2 * 1024,
             "utilization.gpu": 0, "memory.total": 32 * 1024}]
    code, out, _ = _exec_probe(monkeypatch, capsys, device=0, rows=rows,
                               devices={0: {
                                   "pci": "0000:01:00.0", "free": 30 * GIB,
                                   "properties": {
                                       "device_name": "device-0",
                                       "multiProcessorCount": 64,
                                       "maxThreadsPerMultiProcessor": 1536}}})
    assert code == 3
    assert preflight._probe_failure_report(out) is None
    monkeypatch.delenv(NO_LOCAL_GPU_ENV, raising=False)
    run = _completed(code, out)
    assert preflight.device_memory_probe_subprocess(
        run=run, report_failure=True) is None
    assert preflight.device_memory_probe_reason(run=run) == (
        "the probe could not read the card through CUDA (KeyError: 'name')")
    # The same answer through automatic backend selection's own call:
    # missing telemetry, so the prior choice stands and no reason is made up.
    monkeypatch.setattr(subprocess, "run", run)
    cp = SimpleNamespace(cuda=SimpleNamespace(
        runtime=SimpleNamespace(getDevice=lambda: 0)))
    assert backend._auto_device_load(cp) == (None, None)


def test_a_cuda_failure_without_an_attributable_row_prints_no_sample(
        monkeypatch, capsys):
    rows = [
        {"pci.bus_id": "00000000:01:00.0", "memory.used": 2 * 1024,
         "utilization.gpu": 1, "memory.total": 32 * 1024},
        {"pci.bus_id": "00000000:02:00.0", "memory.used": 31 * 1024,
         "utilization.gpu": 97, "memory.total": 32 * 1024},
    ]
    code, out, _ = _exec_probe(monkeypatch, capsys, device=0, rows=rows,
                               devices={0: {
                                   "pci": RuntimeError("cudaErrorUnknown"),
                                   "free": RuntimeError("cudaErrorUnknown")}})
    assert code == 3
    report = json.loads(out.strip().splitlines()[-1])
    assert "cudaErrorUnknown" in report["cuda_error"]
    assert report["nvml"] is None


def _completed(code, stdout):
    return lambda *_a, **_k: SimpleNamespace(
        returncode=code, stdout=stdout, stderr="")


def test_the_failure_report_reaches_only_a_caller_that_asks_for_it(
        monkeypatch):
    """The fit keeps reading a card CUDA cannot open as "no card answered";
    automatic backend selection asks for the report and gets the sample."""

    from gpuwm.local_gpu import NO_LOCAL_GPU_ENV

    monkeypatch.delenv(NO_LOCAL_GPU_ENV, raising=False)
    report = {"cuda_error": "CUDARuntimeError: cudaErrorDevicesUnavailable",
              "nvml": {"used_bytes": 3 * GIB, "total_bytes": 32 * GIB,
                       "utilization_gpu_percent": 0}}
    run = _completed(3, json.dumps(report) + "\n")
    assert preflight.device_memory_probe_subprocess(run=run) is None
    assert preflight.device_memory_probe_reason(run=run) == (
        "no CUDA device answered")
    assert preflight.device_memory_probe_subprocess(
        run=run, report_failure=True) == report
    # Nothing printed, or not the report: still no numbers.
    for stdout in ("", "not json\n", json.dumps({"free_bytes": 1}) + "\n"):
        assert preflight.device_memory_probe_subprocess(
            run=_completed(3, stdout), report_failure=True) is None
    # A measured card is the payload either way.
    payload = {"free_bytes": 1024, "total_bytes": 2048}
    assert preflight.device_memory_probe_subprocess(
        run=_completed(0, json.dumps(payload)), report_failure=True) == payload


@pytest.mark.skipif(sys.platform == "win32", reason=(
    "Windows resolves nvidia-smi from System32 ahead of PATH, so a fixture "
    "nvidia-smi cannot stand in for the real one there"))
def test_the_probe_subprocess_hands_a_full_card_sample_to_its_caller(
        tmp_path, monkeypatch):
    """The probe's own source in a real interpreter: a shadow cupy whose
    context fails and a fixture nvidia-smi reporting a full, busy card."""

    import os
    import stat
    import textwrap

    from gpuwm.local_gpu import NO_LOCAL_GPU_ENV

    monkeypatch.delenv(NO_LOCAL_GPU_ENV, raising=False)
    shadow = tmp_path / "shadow"
    (shadow / "cupy").mkdir(parents=True)
    (shadow / "cupy" / "__init__.py").write_text(textwrap.dedent("""\
        class _Runtime:
            @staticmethod
            def deviceGetPCIBusId(device):
                return "0000:01:00.0"

            @staticmethod
            def memGetInfo():
                raise RuntimeError("cudaErrorMemoryAllocation: out of memory")


        class cuda:
            runtime = _Runtime
        """), encoding="utf-8")
    smi = shadow / "nvidia-smi"
    smi.write_text("#!/bin/sh\necho '00000000:01:00.0, 31744, 97, 32768'\n",
                   encoding="utf-8")
    smi.chmod(smi.stat().st_mode | stat.S_IXUSR)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(shadow)
    env["PATH"] = str(shadow)
    env.pop("PYTHONSAFEPATH", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)

    def run(argv, **kwargs):
        return subprocess.run(argv, env=env, cwd=str(tmp_path), **kwargs)

    assert preflight.device_memory_probe_subprocess(run=run) is None
    report = preflight.device_memory_probe_subprocess(
        run=run, report_failure=True)
    assert "cudaErrorMemoryAllocation" in report["cuda_error"]
    assert report["nvml"] == {"used_bytes": 31 * GIB, "total_bytes": 32 * GIB,
                              "utilization_gpu_percent": 97}


@pytest.mark.parametrize("visible_devices", ["", "-1"])
def test_probe_does_not_query_nvml_when_devices_are_hidden(
        monkeypatch, capsys, visible_devices):
    payload, calls = _run_probe(
        monkeypatch, capsys, device=0, rows=[],
        devices={0: {"pci": "0000:02:00.0", "free": 30 * GIB}},
        visible_devices=visible_devices)
    assert "nvml" not in calls
    assert payload["free_bytes_nvml"] is None
    assert payload["utilization_gpu_percent"] is None
