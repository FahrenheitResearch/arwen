"""The kernel plan compiles a run's recorded kernel set ahead, on a pool, with identical arguments.

Breakage these tests prevent: a cold-cache forecast holding ONE core for minutes
before step 1 while its card idles (x1.40962 on an RTX 4090: 111.5 s of step 1
at 1.1 cores, 2026-10-16), and any change to what NVRTC is asked to compile.
"""
from __future__ import annotations

import json
import threading
import time

import pytest

from gpuwm import kernel_plan


class _StubCompiler:
    """The NVRTC funnel's signature; returns a deterministic fake image."""

    def __init__(self, arch="89"):
        self.calls = []
        self.arch = arch
        self.gate = None
        self.fail_on = None
        self._lock = threading.Lock()

    def _compile_using_nvrtc_no_warning(self, source, options=(), arch=None, filename="kern.cu",
                                        name_expressions=None, log_stream=None, cache_in_memory=False,
                                        jitify=None, method=None):
        if self.gate is not None and source == "slow":
            self.gate.wait(5)
        if source == self.fail_on:
            raise RuntimeError("nvrtc refused")
        resolved = self.arch if arch is None else str(arch)
        with self._lock:
            self.calls.append((source, tuple(options), resolved, filename, cache_in_memory,
                               threading.current_thread().name))
        return (f"image:{source}:{options}:{resolved}:{filename}:{cache_in_memory}".encode(),
                {"k": "mangled"} if name_expressions else None)

    def _get_arch(self):
        return self.arch

    @staticmethod
    def _get_nvrtc_version():
        return (13, 3)

    @staticmethod
    def _get_cupy_cache_key():
        return "stub-key"


def _fresh(monkeypatch, compiler):
    monkeypatch.setattr(kernel_plan, "_compiler", lambda: compiler)
    monkeypatch.setattr(kernel_plan, "compiler_identity", lambda: {"stub": 1})
    monkeypatch.setattr(kernel_plan, "_current_device", lambda: None)
    monkeypatch.setattr(kernel_plan.atexit, "register", lambda function: None)
    monkeypatch.setattr(kernel_plan, "_flight", {})
    monkeypatch.setattr(kernel_plan, "_requested", {})
    monkeypatch.setattr(kernel_plan, "_state", {"installed": False, "plan": None, "tag": None, "tag_text": None,
                                                "device_arch": None, "replay": None, "counts": {}})


@pytest.fixture
def plan_dir(tmp_path, monkeypatch):
    monkeypatch.delenv(kernel_plan.ENVIRONMENT, raising=False)
    monkeypatch.delenv(kernel_plan.WORKERS_ENVIRONMENT, raising=False)
    return tmp_path / "plans"


def _record_run(monkeypatch, plan_dir, compiler, requests, tag="forecast --mesh"):
    _fresh(monkeypatch, compiler)
    assert kernel_plan.install(plan_dir=plan_dir, tag_text=tag)
    for args, kwargs in requests:
        compiler._compile_using_nvrtc_no_warning(*args, **kwargs)
    kernel_plan.save()


def _wait(predicate):
    for _ in range(500):
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("pool did not finish")


def test_next_run_compiles_the_plan_ahead_and_serves_identical_results(monkeypatch, plan_dir):
    requests = [(("A", ("--std=c++17", "--fmad=false"), None, "a.cu"), {}),
                (("B", ("--std=c++17",), "89", "b.cu", ["f<float>"]), {}),
                (("C", ("--std=c++17",)), {"cache_in_memory": True})]
    first = _StubCompiler()
    _record_run(monkeypatch, plan_dir, first, requests)
    serial = [(call[0], call[1], call[2]) for call in first.calls]
    document = json.loads(next(plan_dir.rglob("*.json")).read_text())
    assert len(document["entries"]) == 3 and all(row["arguments"]["arch"] == "device" for row in document["entries"])

    second = _StubCompiler()
    _fresh(monkeypatch, second)
    assert kernel_plan.install(plan_dir=plan_dir, tag_text="forecast --mesh")
    record = kernel_plan.start()
    kernel_plan.close()
    assert record["entries"] == 3 and record["plan_source"] == "tag"
    _wait(lambda: record["pool_compiled"] == 3)
    assert sorted((c[0], c[1], c[2]) for c in second.calls) == sorted(serial)
    assert all(c[5].startswith(kernel_plan.POOL_PREFIX) for c in second.calls)
    results = [second._compile_using_nvrtc_no_warning(*args, **kwargs) for args, kwargs in requests]
    assert len(second.calls) == 3  # served from the pool, not compiled again
    expected = [first.__class__()._compile_using_nvrtc_no_warning(*args, **kwargs) for args, kwargs in requests]
    assert results == expected
    assert kernel_plan.receipt()["calls"]["served"] == 3


def test_a_plan_recorded_on_another_architecture_compiles_for_this_card(monkeypatch, plan_dir):
    old, current = _cupy_program("A-120"), _cupy_program("A-89")
    _record_run(monkeypatch, plan_dir, _StubCompiler(arch="120"), [(("A", ("-x",), "120", old), {})])
    monkeypatch.setattr(kernel_plan, "_cupy_entry_elsewhere", lambda arguments, entries, arch: current[:-3])
    here = _StubCompiler(arch="89")
    _fresh(monkeypatch, here)
    kernel_plan.install(plan_dir=plan_dir, tag_text="forecast --mesh")
    record = kernel_plan.start()
    kernel_plan.close()
    _wait(lambda: record["pool_compiled"] == 1)
    assert here.calls[0][2] == "89"
    # CuPy's architecture-bound hash is part of the image too. Replay must use this card's name.
    assert here.calls[0][3] == current
    got = here._compile_using_nvrtc_no_warning("A", ("-x",), "89", current)
    assert got == _StubCompiler()._compile_using_nvrtc_no_warning("A", ("-x",), "89", current)
    assert len(here.calls) == 1


def test_program_name_is_matched_without_asserts_or_debug_options(monkeypatch, plan_dir):
    source = 'static __device__ float table[] = {1.0f};'
    _record_run(monkeypatch, plan_dir, _StubCompiler(), [((source, ("-x",), None, "one.cu"), {})])
    here = _StubCompiler()
    _fresh(monkeypatch, here)
    kernel_plan.install(plan_dir=plan_dir, tag_text="forecast --mesh")
    record = kernel_plan.start()
    kernel_plan.close()
    _wait(lambda: record["pool_compiled"] == 1)
    here._compile_using_nvrtc_no_warning(source, ("-x",), None, "two.cu")
    assert len(here.calls) == 2  # NVRTC embeds the program name in internal symbols too.
    here._compile_using_nvrtc_no_warning(source, ("-x",), None, "one.cu")
    assert len(here.calls) == 2


def test_the_run_takes_back_a_compile_the_pool_has_not_started(monkeypatch, plan_dir):
    _record_run(monkeypatch, plan_dir, _StubCompiler(), [(("slow", ()), {}), (("queued", ()), {})])
    here = _StubCompiler()
    here.gate = threading.Event()
    _fresh(monkeypatch, here)
    monkeypatch.setenv(kernel_plan.WORKERS_ENVIRONMENT, "1")
    kernel_plan.install(plan_dir=plan_dir, tag_text="forecast --mesh")
    kernel_plan._state["device_arch"] = None
    record = kernel_plan.start()
    kernel_plan.close()
    here._compile_using_nvrtc_no_warning("queued", ())  # reached before the one worker is free
    here.gate.set()
    _wait(lambda: record["pool_compiled"] == 1)
    assert [c[0] for c in here.calls].count("queued") == 1
    assert kernel_plan.receipt()["calls"]["taken_back"] == 1


def test_a_failed_pool_compile_is_compiled_again_by_the_run(monkeypatch, plan_dir):
    _record_run(monkeypatch, plan_dir, _StubCompiler(), [(("bad", ()), {})])
    here = _StubCompiler()
    here.fail_on = "bad"
    _fresh(monkeypatch, here)
    kernel_plan.install(plan_dir=plan_dir, tag_text="forecast --mesh")
    record = kernel_plan.start()
    kernel_plan.close()
    _wait(lambda: record["pool_failed"] == 1)
    with pytest.raises(RuntimeError, match="nvrtc refused"):
        here._compile_using_nvrtc_no_warning("bad", ())


def test_an_unknown_configuration_compiles_the_union_of_known_plans(monkeypatch, plan_dir):
    _record_run(monkeypatch, plan_dir, _StubCompiler(), [(("A", ()), {})], tag="one")
    _record_run(monkeypatch, plan_dir, _StubCompiler(), [(("B", ()), {})], tag="two")
    here = _StubCompiler()
    _fresh(monkeypatch, here)
    kernel_plan.install(plan_dir=plan_dir, tag_text="three")
    record = kernel_plan.start()
    kernel_plan.close()
    assert record["plan_source"] == "union" and record["entries"] == 2


def test_off_switch_leaves_the_compiler_untouched(monkeypatch, plan_dir):
    here = _StubCompiler()
    _fresh(monkeypatch, here)
    monkeypatch.setenv(kernel_plan.ENVIRONMENT, "off")
    original = here._compile_using_nvrtc_no_warning
    assert not kernel_plan.install(plan_dir=plan_dir)
    assert here._compile_using_nvrtc_no_warning == original
    assert kernel_plan.start() is None


def test_damaged_plan_reads_as_empty(monkeypatch, plan_dir):
    here = _StubCompiler()
    _fresh(monkeypatch, here)
    kernel_plan.install(plan_dir=plan_dir, tag_text="x")
    kernel_plan._state["plan"].parent.mkdir(parents=True)
    kernel_plan._state["plan"].write_text("{not json\n")
    record = kernel_plan.start()
    kernel_plan.close()
    assert record["entries"] == 0


def test_tag_ignores_paths_and_numbers():
    a = kernel_plan.default_tag_text(["forecast", "--init", "/a/x1.40962.init.nc", "--hours", "24", "--no-products"])
    b = kernel_plan.default_tag_text(["forecast", "--init", "/b/other.nc", "--hours", "3", "--no-products"])
    assert a == b and "--no-products" in a


def _cupy_program(tag):
    """CuPy's cached route names its program after the disk entry: <sha1>.cubin.cu."""
    import hashlib
    return hashlib.sha1(tag.encode()).hexdigest() + ".cubin.cu"


def test_a_row_its_route_finds_on_disk_is_not_compiled_ahead(monkeypatch, plan_dir, tmp_path):
    # Breakage: with a warm cache and plans present, warm-kernels took 122.5 s and 78 NVRTC
    # compiles against 2.5 s and 0 without the plans (RTX 4090, 2026-10-17).
    cache = tmp_path / "cupy-cache"
    cache.mkdir()
    recorder = _StubCompiler()
    recorder.get_cache_dir = lambda: str(cache)
    program = _cupy_program("A-89")
    _record_run(monkeypatch, plan_dir, recorder, [(("A", ("-x",), None, program), {}),
                                                  (("M", ("-x",)), {"cache_in_memory": True})])
    rows = json.loads(next(plan_dir.rglob("*.json")).read_text())["entries"]
    entries = {row["arguments"]["source"]: row.get("cache_entries") for row in rows}
    assert entries == {"A": {"89": program[:-3]}, "M": None}
    (cache / program[:-3]).write_bytes(b"cubin")  # what CuPy stored after the compile

    here = _StubCompiler()
    here.get_cache_dir = lambda: str(cache)
    _fresh(monkeypatch, here)
    kernel_plan.install(plan_dir=plan_dir, tag_text="forecast --mesh")
    record = kernel_plan.start()
    kernel_plan.close()
    _wait(lambda: record["pool_compiled"] == 1)
    assert record["pool_skipped_cached"] == 1
    assert [call[0] for call in here.calls] == ["M"]  # the in-memory route still compiles ahead

    (cache / program[:-3]).unlink()  # a cold cache: the plan compiles the set again
    cold = _StubCompiler()
    cold.get_cache_dir = lambda: str(cache)
    _fresh(monkeypatch, cold)
    kernel_plan.install(plan_dir=plan_dir, tag_text="forecast --mesh")
    record = kernel_plan.start()
    kernel_plan.close()
    _wait(lambda: record["pool_compiled"] == 2)
    assert record["pool_skipped_cached"] == 0


def probe_for_test(arguments, arch):
    return f"probe/{arguments['source']}-{arch}.image"


def test_a_named_entry_and_its_probe_cover_another_architecture(monkeypatch, plan_dir, tmp_path):
    cache = tmp_path / "cupy-cache"
    (cache / "probe").mkdir(parents=True)
    recorder = _StubCompiler(arch="89")
    recorder.get_cache_dir = lambda: str(cache)
    _fresh(monkeypatch, recorder)
    kernel_plan.install(plan_dir=plan_dir, tag_text="t")
    with kernel_plan.cache_entry(f"{__name__}:probe_for_test", cache / "probe" / "D-89.image"):
        recorder._compile_using_nvrtc_no_warning("D", ("-x",), None, "d.cu")
    kernel_plan.save()
    row = json.loads(next(plan_dir.rglob("*.json")).read_text())["entries"][0]
    assert row["cache_entries"] == {"89": "probe/D-89.image"}  # stored relative to the cache root
    (cache / "probe" / "D-120.image").write_bytes(b"x")  # baked for sm_120, never recorded there
    here = _StubCompiler(arch="120")
    here.get_cache_dir = lambda: str(cache)
    _fresh(monkeypatch, here)
    kernel_plan.install(plan_dir=plan_dir, tag_text="t")
    record = kernel_plan.start()
    kernel_plan.close()
    assert record["pool_skipped_cached"] == 1 and here.calls == []


def test_rebind_gives_each_configuration_its_own_plan(monkeypatch, plan_dir):
    # Breakage: a warm-kernels worker wrote every profile it warmed under its first profile's tag.
    here = _StubCompiler()
    _fresh(monkeypatch, here)
    kernel_plan.install(plan_dir=plan_dir, tag_text="one")
    here._compile_using_nvrtc_no_warning("A", ())
    assert kernel_plan.rebind("two")
    here._compile_using_nvrtc_no_warning("B", ())
    kernel_plan.save()
    plans = {json.loads(path.read_text())["tag_text"]: [row["arguments"]["source"] for row in
             json.loads(path.read_text())["entries"]] for path in plan_dir.rglob("*.json")}
    assert plans == {"one": ["A"], "two": ["B"]}


def _loader(compiler, source, records):
    """A loader's uncached body: its NVRTC request, then the records a real load writes."""

    def load(name):
        image = compiler._compile_using_nvrtc_no_warning(source + name, ("--std=c++17", "--ftz=false"), None,
                                                         f"{name}.cu")
        kernel_plan.refuse_while_capturing()
        records.append(name)
        return image
    return load


def test_a_first_run_compiles_its_static_plan_ahead_and_serves_identical_results(monkeypatch, plan_dir):
    """No recorded plan (a fresh box): the static plan's captures compile on the pool, the run is served."""
    here = _StubCompiler()
    _fresh(monkeypatch, here)
    assert kernel_plan.install(plan_dir=plan_dir, tag_text="first run")
    record = kernel_plan.start()
    assert record["entries"] == 0 and record["plan_source"] == "none"
    records = []
    load = _loader(here, "src:", records)
    futures = [kernel_plan.submit(kernel_plan.ahead, load, name) for name in ("noah", "smag2d", "ysu_480")]
    kernel_plan.close()
    assert sorted(f.result(5) for f in futures) == ["compiled"] * 3
    assert all(c[5].startswith(kernel_plan.POOL_PREFIX) for c in here.calls) and len(here.calls) == 3
    assert records == []  # the capture wrote nothing a real load writes
    results = [load(name) for name in ("noah", "smag2d", "ysu_480")]
    assert len(here.calls) == 3  # served from the table, not compiled again
    assert results == [_StubCompiler()._compile_using_nvrtc_no_warning(
        "src:" + name, ("--std=c++17", "--ftz=false"), None, f"{name}.cu") for name in ("noah", "smag2d", "ysu_480")]
    assert records == ["noah", "smag2d", "ysu_480"]
    assert kernel_plan.receipt()["replay"]["static_compiled"] == 3


def test_a_static_capture_whose_image_is_on_disk_compiles_and_records_nothing(monkeypatch, plan_dir):
    here = _StubCompiler()
    _fresh(monkeypatch, here)
    kernel_plan.install(plan_dir=plan_dir, tag_text="warm")
    kernel_plan.start()
    records = []

    def cached_load(name):  # the route read its image from disk: no NVRTC request
        kernel_plan.refuse_while_capturing()
        records.append(name)

    assert kernel_plan.submit(kernel_plan.ahead, cached_load, "noah").result(5) == "cached"
    kernel_plan.close()
    assert here.calls == [] and records == []
    cached_load("noah")
    assert records == ["noah"]  # outside a capture the guard is inert


def test_a_static_capture_the_run_requests_differently_is_compiled_by_the_run(monkeypatch, plan_dir):
    """A parameter set declared after the capture changes the bytes: the run misses and compiles inline."""
    here = _StubCompiler()
    _fresh(monkeypatch, here)
    kernel_plan.install(plan_dir=plan_dir, tag_text="declared later")
    kernel_plan.start()
    records = []
    assert kernel_plan.submit(kernel_plan.ahead, _loader(here, "default:", records), "mynn").result(5) == "compiled"
    kernel_plan.close()
    _loader(here, "declared:", records)("mynn")
    assert [c[0] for c in here.calls] == ["default:mynn", "declared:mynn"]
    assert here.calls[1][5] == threading.current_thread().name


def test_a_static_capture_never_compiles_twice_what_the_run_already_compiles(monkeypatch, plan_dir):
    here = _StubCompiler()
    _fresh(monkeypatch, here)
    kernel_plan.install(plan_dir=plan_dir, tag_text="raced")
    kernel_plan.start()
    records = []
    load = _loader(here, "src:", records)
    load("noah")  # the run got there first
    assert kernel_plan.submit(kernel_plan.ahead, load, "noah").result(5) == "known"
    kernel_plan.close()
    assert len(here.calls) == 1 and records == ["noah"]


def test_a_static_capture_that_fails_leaves_the_compile_to_the_run(monkeypatch, plan_dir):
    here = _StubCompiler()
    here.fail_on = "src:bad"
    _fresh(monkeypatch, here)
    kernel_plan.install(plan_dir=plan_dir, tag_text="failing")
    kernel_plan.start()
    records = []
    load = _loader(here, "src:", records)
    assert kernel_plan.submit(kernel_plan.ahead, load, "bad").result(5) == "failed"
    kernel_plan.close()
    with pytest.raises(RuntimeError, match="nvrtc refused"):
        load("bad")
    assert records == []
