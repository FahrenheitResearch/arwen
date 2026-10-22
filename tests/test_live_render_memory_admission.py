"""Live renders are admitted against memory read at every admission.

THE BREAKAGE: production NYC 500 m 12Z on 2026-10-05 died at forecast hour
46.7 of 48 on a 32 GiB, 8 vCPU host.  The live frame slots read available
memory once, at the start of the forecast, and admitted up to three
renders against half of that for the rest of the run, however much the
model had grown since.  These tests hold the slots to the rules that
replace that: memory re-read at every admission, peaks learned from the
renders that ran, the model's floor kept free, the oldest render always
able to run, queue order kept, and a kill switch that restores the
one-shot budget exactly.

The host is the site worker budget's fixture: 32 GiB,
a 6 GiB model.
"""

import os
import subprocess
import sys
import threading
import time

import pytest

from gpuwm import first_products, live_products
from test_live_products import _Renderer, _frame, _plan

GIB = 1 << 30
MIB = 1 << 20


class _Host:
    """A fake host: MemAvailable, the model's RSS and each render tree's
    (rss, peak) are set by the test and read by the slots."""

    def __init__(self, available=30 * GIB, model=6 * GIB, total=32 * GIB):
        self.available_bytes = available
        self.model = model
        self.total_bytes = total
        self.trees = {}
        self.reads = 0

    def available(self):
        self.reads += 1
        return self.available_bytes

    def total(self):
        return self.total_bytes

    def sample(self, pids):
        return self.model, {pid: self.trees[pid] for pid in pids if pid in self.trees}


class _Refusing(_Host):
    """A probe the one-shot path must never consult."""

    def available(self):
        raise AssertionError("the kill switch read memory at an admission")

    def sample(self, pids):
        raise AssertionError("the kill switch measured a render")


def _settings(**overrides):
    values = {"poll_seconds": 0.01, "sample_seconds": 0.01, **overrides}
    return live_products.LiveRenderMemorySettings(**values)


class _Pids:
    """Maps the thread drawing a render to a fake render pid."""

    def __init__(self):
        self.by_thread = {}

    def __call__(self, ident):
        return self.by_thread.get(ident)


def _slots(host, width=3, budget=None, pids=None, **settings):
    return live_products._LiveRenderSlots(
        width, budget, adaptive=True, settings=_settings(**settings), probe=host,
        render_pid=pids or _Pids())


class _Draw(threading.Thread):
    """One render: takes ``turn`` (or none), is admitted, holds until released."""

    def __init__(self, slots, name, price=0, turn=None, pid=None, pids=None, log=None):
        super().__init__(daemon=True, name=f"draw-{name}")
        self.slots, self.label, self.price, self.turn = slots, name, price, turn
        self.pid, self.pids, self.log = pid, pids, log
        self.started = threading.Event()
        self.release = threading.Event()

    def run(self):
        if self.pids is not None and self.pid is not None:
            self.pids.by_thread[threading.get_ident()] = self.pid
        with self.slots.admit(self.price, self.turn):
            if self.log is not None:
                self.log.append(self.label)
            self.started.set()
            assert self.release.wait(20)


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


# -- the thresholds -------------------------------------------------------


def test_the_model_floor_matches_the_worker_budget_numbers():
    settings = live_products.LiveRenderMemorySettings()
    floor = live_products.live_render_model_floor
    assert floor(6 * GIB, 32 * GIB, settings) == pytest.approx(12.8 * GIB)
    assert floor(6 * GIB, 256 * GIB, settings) == pytest.approx(8.5 * GIB)
    assert floor(20 * GIB, 32 * GIB, settings) == pytest.approx(26 * GIB)
    # Only the floor's unspent part is kept free (the model's own RSS is
    # already outside MemAvailable), plus the system reserve.
    keep = live_products.live_render_keep_free(6 * GIB, 32 * GIB, settings)
    assert keep == pytest.approx((12.8 - 6 + 2) * GIB)


def test_every_threshold_is_settable_by_environment_and_bad_values_keep_defaults():
    settings = live_products.LiveRenderMemorySettings.from_env({
        "GPUWM_LIVE_RENDER_PEAK_DEFAULT_BYTES": "3G",
        "GPUWM_LIVE_RENDER_PEAKS_KEPT": "4",
        "GPUWM_LIVE_RENDER_MODEL_MARGIN_FRACTION": "0.5",
        "GPUWM_LIVE_RENDER_MODEL_MARGIN_BYTES": "512MiB",
        "GPUWM_LIVE_RENDER_SMALL_HOST_BYTES": "128GiB",
        "GPUWM_LIVE_RENDER_SMALL_HOST_FRACTION": "0.3",
        "GPUWM_LIVE_RENDER_SYSTEM_RESERVE_BYTES": "1073741824",
        "GPUWM_LIVE_RENDER_HYSTERESIS_BYTES": "0",
        "GPUWM_LIVE_RENDER_POLL_SECONDS": "0.25",
        "GPUWM_LIVE_RENDER_SAMPLE_SECONDS": "2",
    })
    assert settings.as_dict() == {
        "peak_default": 3 * GIB, "peaks_kept": 4, "model_margin_fraction": 0.5,
        "model_margin_bytes": 512 * MIB, "small_host_bytes": 128 * GIB,
        "small_host_fraction": 0.3, "system_reserve": GIB, "hysteresis": 0,
        "poll_seconds": 0.25, "sample_seconds": 2.0}
    bad = live_products.LiveRenderMemorySettings.from_env({
        "GPUWM_LIVE_RENDER_PEAK_DEFAULT_BYTES": "lots",
        "GPUWM_LIVE_RENDER_PEAKS_KEPT": "0",
        "GPUWM_LIVE_RENDER_SMALL_HOST_FRACTION": "1.5",
        "GPUWM_LIVE_RENDER_POLL_SECONDS": "-1"})
    assert bad.as_dict() == live_products.LiveRenderMemorySettings().as_dict()


# -- admission under a shrinking MemAvailable ------------------------------


def test_admission_follows_memory_as_it_shrinks_and_frees():
    """Free memory read at each admission, not once: renders that fitted at
    the start of the run are held when the model has taken the host, and
    run again when memory frees.  The model is never touched."""

    host = _Host(available=30 * GIB)
    slots = _slots(host)
    # keep_free(6 GiB model, 32 GiB host) = 6.8 + 2 = 8.8 GiB; a render is
    # reckoned at 4 GiB until one has been measured.
    a, b, c = (_Draw(slots, name) for name in "abc")
    for draw in (a, b, c):
        draw.start()
        assert draw.started.wait(5)        # 30 - 4 - 4 - 4 = 18 >= 8.8
    for draw in (a, b, c):
        draw.release.set()
        draw.join(5)
    reads = host.reads
    # Late in the run: the model has grown and the host has filled.
    host.model = 9 * GIB
    host.available_bytes = int(13.5 * GIB)  # keep_free = 12.8 - 9 + 2 = 5.8
    d, e = _Draw(slots, "d"), _Draw(slots, "e")
    d.start()
    assert d.started.wait(5)               # it would run alone
    e.start()
    assert not e.started.wait(0.3)         # 13.5 - 4 - 4 = 5.5 < 5.8: held
    assert host.reads > reads              # read again at this admission
    host.available_bytes = 20 * GIB        # memory frees
    assert e.started.wait(5)
    record = slots.record()
    assert record["renders_held"] == 1
    assert record["longest_hold_seconds"] > 0
    assert record["model_rss_bytes"] == 9 * GIB
    for draw in (d, e):
        draw.release.set()
        draw.join(5)


def test_a_held_render_needs_the_hysteresis_margin_to_start():
    host = _Host(available=30 * GIB)
    slots = _slots(host)
    a = _Draw(slots, "a")
    a.start()
    assert a.started.wait(5)
    host.available_bytes = 16 * GIB        # 16 - 4 - 4 = 8.0 < 8.8
    b = _Draw(slots, "b")
    b.start()
    assert not b.started.wait(0.2)
    host.available_bytes = 17 * GIB        # 9.0 >= 8.8 but < 8.8 + 0.5
    assert not b.started.wait(0.2)
    host.available_bytes = 17.5 * GIB      # 9.5 >= 9.3
    assert b.started.wait(5)
    for draw in (a, b):
        draw.release.set()
        draw.join(5)


def test_the_model_floor_holds_while_the_model_grows():
    """With two renders running the model grows to 14 GiB and free drops to
    8 GiB: the next render is not admitted; the ones running continue."""

    host = _Host(available=30 * GIB)
    slots = _slots(host)
    a, b = _Draw(slots, "a"), _Draw(slots, "b")
    for draw in (a, b):
        draw.start()
        assert draw.started.wait(5)
    host.model, host.available_bytes = 14 * GIB, 8 * GIB
    c = _Draw(slots, "c")
    c.start()
    assert not c.started.wait(0.3)
    a.release.set()
    a.join(5)
    assert not c.started.wait(0.3)         # b still drawing, still short
    b.release.set()
    b.join(5)
    assert c.started.wait(5)               # now the oldest, alone
    c.release.set()
    c.join(5)


def test_no_memory_reading_imposes_no_cap():
    host = _Host(available=None)
    slots = _slots(host)
    draws = [_Draw(slots, str(i)) for i in range(3)]
    for draw in draws:
        draw.start()
    assert all(draw.started.wait(5) for draw in draws)
    for draw in draws:
        draw.release.set()
        draw.join(5)


# -- never a deadlock: the oldest always runs --------------------------------


def test_the_oldest_render_runs_when_it_would_run_alone_however_short_memory_is():
    host = _Host(available=1 * GIB)
    slots = _slots(host)
    a, b = _Draw(slots, "a"), _Draw(slots, "b")
    a.start()
    assert a.started.wait(5)
    b.start()
    assert not b.started.wait(0.3)
    a.release.set()
    a.join(5)
    assert b.started.wait(5)               # memory is still short
    b.release.set()
    b.join(5)


def test_one_slot_and_no_memory_still_draws_every_frame_without_deadlock():
    host = _Host(available=0)
    slots = _slots(host, width=1)
    done = []
    lock = threading.Lock()

    def worker(count):
        for _ in range(count):
            turn = slots.ticket()
            with slots.admit(MIB, turn):
                with lock:
                    done.append(1)
                time.sleep(0.002)

    threads = [threading.Thread(target=worker, args=(10,), daemon=True) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert not any(thread.is_alive() for thread in threads)
    assert len(done) == 40


def test_a_cancelled_ticket_never_blocks_the_frames_behind_it():
    host = _Host(available=30 * GIB)
    slots = _slots(host, width=1)
    skipped = slots.ticket()
    later = slots.ticket()
    draw = _Draw(slots, "later", turn=later)
    draw.start()
    assert not draw.started.wait(0.2)      # behind the frame still deciding
    slots.cancel(skipped)                  # (a windows-only skip, or a halt)
    assert draw.started.wait(5)
    draw.release.set()
    draw.join(5)


def test_live_products_with_one_slot_and_no_memory_publish_every_frame(tmp_path, monkeypatch):
    monkeypatch.setattr(live_products, "_available_memory_bytes", lambda: 0)
    host = _Host(available=0)
    slots = _slots(host, width=1)
    renderer = _Renderer(delay=0.01)
    live = live_products.LiveProducts(_plan(tmp_path), report=lambda _: None,
                                     warn=lambda *a, **k: None, runner=renderer,
                                     slot=slots, windowed_slugs=frozenset)
    try:
        for hour in range(8):
            frame, valid = _frame(tmp_path, 1, hour)
            live.frame_committed(domain=1, valid_time=valid, path=frame)
        summary = live.stop(timeout=60)
    finally:
        live.halt()
    assert summary["published"] == 8 and summary["dropped"] == 0
    assert renderer.most == 1
    assert summary["memory_admission"]["renders_admitted"] == 8
    # The run's own record says how renders were admitted.
    receipt = live_products.read_receipt(tmp_path / "png")
    assert receipt["memory_admission"]["adaptive"] is True
    assert receipt["memory_admission"]["renders_admitted"] >= 7


# -- order ----------------------------------------------------------------


def test_frames_are_admitted_in_ticket_order_whichever_thread_arrives_first():
    host = _Host(available=None)
    slots = _slots(host, width=1)
    turns = [slots.ticket() for _ in range(6)]
    log = []
    draws = [_Draw(slots, i, turn=turn, log=log) for i, turn in enumerate(turns)]
    for draw in reversed(draws):           # the last ticket's thread asks first
        draw.start()
        time.sleep(0.01)
    for draw in draws:
        assert draw.started.wait(5)
        draw.release.set()
        draw.join(5)
    assert log == list(range(6))


def test_live_frames_are_drawn_in_commit_order_when_memory_holds_them_to_one(tmp_path):
    host = _Host(available=9 * GIB)        # 9 - 4 - 4 < 8.8: one at a time
    slots = _slots(host, width=3)
    renderer = _Renderer(delay=0.02)
    live = live_products.LiveProducts(_plan(tmp_path), report=lambda _: None,
                                     warn=lambda *a, **k: None, runner=renderer,
                                     slot=slots, windowed_slugs=frozenset)
    frames = []
    try:
        for hour in range(9):
            frame, valid = _frame(tmp_path, 1, hour)
            frames.append(frame)
            live.frame_committed(domain=1, valid_time=valid, path=frame)
        assert live.stop(timeout=60)["published"] == 9
    finally:
        live.halt()
    drawn = [call[call.index("--series") - 1] for call in renderer.calls]
    # With every product asked for, a frame is drawn twice in a row: the
    # headline set first (product_publish.PRIORITY_PRODUCTS, published as soon
    # as it lands), then the full set. The order is the order of first draw.
    assert all(drawn.count(path) in (1, 2) for path in drawn)
    assert list(dict.fromkeys(drawn)) == [str(frame) for frame in frames]
    assert renderer.most == 1


# -- learned peaks ----------------------------------------------------------


def test_the_render_peak_is_learned_from_the_renders_that_ran():
    """Reckoned at 4 GiB at first; after a 2 GiB render finishes, two new
    renders fit at 13 GiB free where the default allowed one."""

    host = _Host(available=13 * GIB)
    pids = _Pids()
    slots = _slots(host, pids=pids)
    assert slots.peak_estimate() == 4 * GIB
    a = _Draw(slots, "a", pid=101, pids=pids)
    b = _Draw(slots, "b", pid=102, pids=pids)
    a.start()
    assert a.started.wait(5)
    b.start()
    assert not b.started.wait(0.3)         # 13 - 4 - 4 = 5 < 8.8
    host.trees[101] = (2 * GIB, 2 * GIB)   # a measured at its peak
    assert _wait_for(lambda: any(flight["rss"] == 2 * GIB
                                 for flight in list(slots._flights.values())))
    assert not b.started.wait(0.2)         # 13 - (4 - 2) - 4 = 7 < 9.3
    a.release.set()
    a.join(5)
    assert b.started.wait(5)
    b.release.set()
    b.join(5)
    assert slots.peak_estimate() == 2 * GIB
    c = _Draw(slots, "c", pid=103, pids=pids)
    d = _Draw(slots, "d", pid=104, pids=pids)
    c.start()
    assert c.started.wait(5)
    d.start()
    assert d.started.wait(5)               # 13 - 2 - 2 = 9 >= 8.8
    for draw in (c, d):
        draw.release.set()
        draw.join(5)


def test_a_frame_priced_above_the_learned_peak_keeps_its_price():
    host = _Host()
    slots = _slots(host)
    slots._peaks.append(1 * GIB)
    assert slots.peak_estimate(7 * GIB) == 7 * GIB
    assert slots.peak_estimate(MIB) == 1 * GIB


def test_a_running_render_counts_only_the_growth_it_has_left():
    """A render already at its learned peak needs nothing more; one not yet
    measured is reckoned at its whole estimate."""

    host = _Host(available=int(11.5 * GIB))
    pids = _Pids()
    slots = _slots(host, pids=pids)
    slots._peaks.append(2 * GIB)
    a = _Draw(slots, "a", pid=201, pids=pids)
    a.start()
    assert a.started.wait(5)
    b = _Draw(slots, "b", pid=202, pids=pids)
    b.start()
    assert not b.started.wait(0.3)         # 11.5 - 2 (a unmeasured) - 2 = 7.5 < 8.8
    host.trees[201] = (2 * GIB, 2 * GIB)   # a holds its peak: no growth left
    assert b.started.wait(5)               # 11.5 - 0 - 2 = 9.5 >= 8.8 + 0.5
    assert not a.release.is_set()          # beside a, not after it
    for draw in (a, b):
        draw.release.set()
        draw.join(5)


@pytest.mark.skipif(not os.path.isdir("/proc/self/task"), reason="Linux /proc only")
def test_a_real_render_tree_is_measured_from_proc():
    """A real child that holds 300 MiB, below a parent: the tree's peak is
    learned, and the model is this process without the render."""

    slots = live_products._LiveRenderSlots(
        3, None, adaptive=True, settings=_settings(peak_default=64 * MIB))
    # Written bytes, so the pages are resident (a zeroed buffer may not be).
    held = "b = bytes([1]) * (300 * 1024 * 1024); import time; time.sleep(1.5)"
    script = ("import subprocess, sys\n"
              f"child = subprocess.Popen([sys.executable, '-c', {held!r}])\n"
              "child.wait()\n")
    with slots.admit(MIB):
        process = subprocess.Popen([sys.executable, "-c", script])
        with first_products._RUNNING_LOCK:
            first_products._RUNNING[threading.get_ident()] = process
        try:
            process.wait(30)
        finally:
            with first_products._RUNNING_LOCK:
                first_products._RUNNING.pop(threading.get_ident(), None)
    assert slots.peak_estimate() >= 300 * MIB
    model, trees = live_products._HostMemoryProbe().sample([])
    assert model is not None and model > 0 and trees == {}


# -- the kill switch ---------------------------------------------------------


@pytest.mark.parametrize("word", ["0", "off", "false", "no"])
def test_the_kill_switch_restores_the_one_shot_budget(monkeypatch, word):
    monkeypatch.setenv("GPUWM_LIVE_RENDER_ADAPTIVE", word)
    monkeypatch.setattr(live_products, "_available_memory_bytes", lambda: 64 * GIB)
    slots, _early = live_products.shared_render_slots()
    assert slots.adaptive is False
    assert slots.ticket() is None
    # Memory is never read at an admission: three renders run together
    # within the one-shot budget even though the probe would refuse.
    legacy = live_products._LiveRenderSlots(3, 64 * GIB, probe=_Refusing())
    assert legacy.adaptive is False
    draws = [_Draw(legacy, str(i), price=MIB) for i in range(3)]
    for draw in draws:
        draw.start()
    assert all(draw.started.wait(5) for draw in draws)
    for draw in draws:
        draw.release.set()
        draw.join(5)


def test_the_kill_switch_keeps_the_old_summary_and_admission_path(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUWM_LIVE_RENDER_ADAPTIVE", "0")
    monkeypatch.setattr(live_products, "_available_memory_bytes", lambda: 64 * GIB)
    slots = live_products._LiveRenderSlots(3, 32 * GIB, probe=_Refusing())
    renderer = _Renderer(delay=0.01)
    live = live_products.LiveProducts(_plan(tmp_path), report=lambda _: None,
                                     warn=lambda *a, **k: None, runner=renderer,
                                     slot=slots, windowed_slugs=frozenset)
    try:
        for hour in range(4):
            frame, valid = _frame(tmp_path, 1, hour)
            live.frame_committed(domain=1, valid_time=valid, path=frame)
        summary = live.stop(timeout=60)
    finally:
        live.halt()
    assert set(summary) == {"published", "dropped", "render_seconds_total",
                            "render_seconds_max"}
    assert summary["published"] == 4
    receipt = live_products.read_receipt(tmp_path / "png")
    assert set(receipt) == {"schema", "render_products", "render_section", "frames"}


def test_the_admission_is_on_by_default(monkeypatch):
    monkeypatch.delenv("GPUWM_LIVE_RENDER_ADAPTIVE", raising=False)
    slots, _early = live_products.shared_render_slots()
    assert slots.adaptive is True
    assert live_products.live_render_adaptive({"GPUWM_LIVE_RENDER_ADAPTIVE": "1"})
