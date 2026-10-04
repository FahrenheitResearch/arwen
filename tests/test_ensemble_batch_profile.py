"""CPU checks for non-invasive, hierarchical launch instrumentation."""
from types import SimpleNamespace

import pytest

from tools.ensemble_batch_profile import (
    EventRecorder, TimedCallable, arguments, instrument, kernel_family, profile_configuration)


class FakeCuda:
    def __init__(self):
        self.clock = 0.0
        self.stream = SimpleNamespace(ptr=7, synchronize=lambda: None)
        self.ranges = []
        self.nvtx = SimpleNamespace(RangePush=self.ranges.append, RangePop=self.ranges.pop)

    def Event(self):
        cuda = self

        class Event:
            def record(self, stream):
                assert stream is cuda.stream
                self.time = cuda.clock

        return Event()

    def get_current_stream(self):
        return self.stream

    @staticmethod
    def get_elapsed_time(start, end):
        return end.time - start.time


def test_raw_wrapper_forwards_original_arguments_return_and_attributes():
    cuda = FakeCuda()
    recorder = EventRecorder(SimpleNamespace(cuda=cuda), capacity=4, nvtx=True)
    observed = []
    original_argument = object()

    def kernel(*args, **kwargs):
        observed.append((args, kwargs))
        cuda.clock += 4.5
        return original_argument

    kernel.attributes = {"num_regs": 38, "local_size_bytes": 512, "shared_size_bytes": 0}
    kernel.operation = "exact original operation text"
    timed = recorder.wrap(kernel, "acoustic:advance_w_phi", "acoustic_column_solve")
    assert recorder.wrap(timed, timed.label, timed.family) is timed
    assert timed.operation == kernel.operation
    recorder.enabled = True
    grid, block, inputs = (8,), (128,), (original_argument,)
    assert timed(grid, block, inputs, shared_mem=16) is original_argument
    assert observed == [((grid, block, inputs), {"shared_mem": 16})]
    assert cuda.ranges == []
    row = recorder.summarize()["rows"][0]
    assert row["mean_ms"] == 4.5
    assert row["kernel_attributes"] == kernel.attributes
    assert row["grids"] == [grid] and row["blocks"] == [block]


def test_nested_family_spans_subtract_only_direct_children():
    cuda = FakeCuda()
    recorder = EventRecorder(SimpleNamespace(cuda=cuda), capacity=8)

    def raw():
        cuda.clock += 10

    child = recorder.wrap(raw, "flux", "stage_fluxes", kind="elementwise")

    def family():
        cuda.clock += 2
        child()
        cuda.clock += 3

    middle = recorder.wrap(family, "family:flux", "stage_fluxes", kind="family")

    def driver():
        cuda.clock += 1
        middle()
        cuda.clock += 4

    recorder.enabled = True
    recorder.wrap(driver, "driver:step", "driver", kind="driver")()
    report = recorder.summarize()
    rows = {row["label"]: row for row in report["rows"]}
    assert rows["flux"]["exclusive_ms"] == 10
    assert rows["family:flux"]["inclusive_ms"] == 15
    assert rows["family:flux"]["exclusive_ms"] == 5
    assert rows["driver:step"]["exclusive_ms"] == 5
    assert report["driver_span_ms"] == 20
    assert report["raw_launch_span_ms"] == 10
    assert report["uncategorized_driver_span_ms"] == 5


def test_disabled_recorder_has_no_event_or_range_activity():
    cuda = FakeCuda()
    recorder = EventRecorder(SimpleNamespace(cuda=cuda), capacity=1, nvtx=True)
    recorder.wrap(lambda: 123, "disabled", "other", kind="family")()
    assert not recorder.records and not cuda.ranges


def test_capacity_refuses_before_unrecorded_work_and_failed_work_closes_span():
    cuda = FakeCuda()
    recorder = EventRecorder(SimpleNamespace(cuda=cuda), capacity=1)
    called = []
    recorder.enabled = True
    fn = recorder.wrap(lambda: called.append(1), "one", "other", kind="family")
    fn()
    with pytest.raises(RuntimeError, match="capacity exhausted"):
        fn()
    assert called == [1]
    assert recorder.summarize()["events_used"] == 1

    recorder = EventRecorder(SimpleNamespace(cuda=cuda), capacity=1, nvtx=True)
    recorder.enabled = True

    def fail():
        raise ValueError("original failure")

    with pytest.raises(ValueError, match="original failure"):
        recorder.wrap(fail, "failed", "other", kind="family")()
    assert recorder.stack == [] and cuda.ranges == []
    assert recorder.records[0]["failure_type"] == "ValueError"


def test_exact_source_text_and_hashes_are_retained_once():
    import hashlib
    cuda = FakeCuda()
    recorder = EventRecorder(SimpleNamespace(cuda=cuda), capacity=1)
    source = "float unchanged;\r\n// exact source bytes\n"
    digest = recorder.source(source, "first")
    assert digest == hashlib.sha256(source.encode()).hexdigest()
    assert recorder.source(source, "second") == digest
    assert len(recorder.sources) == 1
    assert recorder.sources[digest] == {"source": source, "origins": ["first", "second"]}


def test_provider_and_family_aliases_are_restored_after_an_exception(monkeypatch):
    from gpuwm.core import kernels
    from gpuwm.core.state import DomainState

    result = object()

    def provider(module, entry):
        return lambda *args, **kwargs: result

    monkeypatch.setattr(kernels, "get_kernel", provider)
    total_mu = DomainState.total_mu
    recorder = EventRecorder(SimpleNamespace(cuda=FakeCuda()), capacity=1)
    with pytest.raises(ValueError, match="body failure"):
        with instrument(recorder):
            assert kernels.get_kernel is not provider
            assert DomainState.total_mu is not total_mu
            function = kernels.get_kernel("face_mass", "average_mass_faces")
            assert isinstance(function, TimedCallable)
            assert function((1,), (128,), ()) is result
            raise ValueError("body failure")
    assert kernels.get_kernel is provider
    assert DomainState.total_mu is total_mu


def test_missing_experimental_provider_does_not_block_committed_profiler(monkeypatch):
    from tools import ensemble_batch_profile as profile
    original = profile.importlib.import_module
    qualified = "gpuwm.ensemble.batch_openbc"

    def missing(name):
        if name == qualified:
            raise ModuleNotFoundError("optional provider absent", name=qualified)
        return original(name)

    monkeypatch.setattr(profile.importlib, "import_module", missing)
    test_provider_and_family_aliases_are_restored_after_an_exception(monkeypatch)


def test_optional_provider_does_not_hide_a_missing_dependency(monkeypatch):
    from tools import ensemble_batch_profile as profile

    def broken(name):
        raise ModuleNotFoundError("required dependency absent", name="required_dependency")

    monkeypatch.setattr(profile.importlib, "import_module", broken)
    with pytest.raises(ModuleNotFoundError, match="required dependency"):
        profile._import_optional_provider("batch_openbc")


@pytest.mark.parametrize("entry,family", [
    ("calc_coefs", "acoustic_coefficients"),
    ("advance_w_phi_msf", "acoustic_column_solve"),
    ("advance_mu_th", "acoustic_mass_theta"),
    ("advance_uv", "acoustic_horizontal"),
])
def test_column_and_horizontal_families_are_separate(entry, family):
    assert kernel_family("acoustic", entry) == family


@pytest.mark.parametrize("entry,family", [
    ("slow_pgf", "big_step"), ("slow_buoyancy", "big_step"),
    ("slow_geopotential", "big_step"), ("small_step_init_uv", "bookkeeping"),
    ("small_step_finish_column", "bookkeeping"),
])
def test_dycore_raw_entries_have_their_actual_family(entry, family):
    assert kernel_family("dycore", entry) == family


def test_profile_clock_interval_includes_control_steps():
    args = arguments(["--nx", "11", "--ny", "7", "--dx", "3000", "--dt", "3",
                      "--warmup", "2", "--control-steps", "4", "--steps", "5",
                      "--receipt", "receipt.json"])
    assert profile_configuration(args).run_seconds == 33.0


@pytest.mark.parametrize("extra", [["--members", "1"], ["--steps", "0"], ["--dt", "nan"]])
def test_profile_arguments_refuse_non_comparable_or_invalid_workloads(extra):
    with pytest.raises(SystemExit):
        arguments(["--nx", "11", "--ny", "7", "--dx", "3000", "--dt", "3",
                   "--receipt", "receipt.json"] + extra)
