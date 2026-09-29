"""Live hosts share the same slots, native budget and lifecycle."""
import subprocess
import threading
import time

import pytest

from gpuwm import first_products, live_products
from test_live_products import _Renderer, _frame, _plan


def _cgroups(monkeypatch, tmp_path, membership, files):
    """A cgroup mount holding ``files`` and a ``/proc/self/cgroup`` naming ``membership``.

    ``files`` maps a path under the mount to its text.
    """

    root = tmp_path / "sys-fs-cgroup"
    root.mkdir(exist_ok=True)
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="ascii")
    proc = tmp_path / "proc-self-cgroup"
    proc.write_text(membership + "\n", encoding="ascii")
    monkeypatch.setattr(live_products, "_CGROUP_ROOT", str(root))
    monkeypatch.setattr(live_products, "_PROC_SELF_CGROUP", str(proc))


def _cpu_max(monkeypatch, tmp_path, text):
    """A container's cgroup v2 ``cpu.max`` holding ``text``, at its own root."""

    _cgroups(monkeypatch, tmp_path, "0::/", {"cpu.max": text})


@pytest.fixture
def eight_cpus(monkeypatch, tmp_path):
    monkeypatch.setattr(live_products.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(live_products.os, "sched_getaffinity", lambda _: set(range(8)), raising=False)
    monkeypatch.delenv("RAYON_NUM_THREADS", raising=False)
    # No container quota and free memory for every frame, so the processors
    # alone set the budget whichever box runs the test.
    _cpu_max(monkeypatch, tmp_path, "max 100000")
    monkeypatch.setattr(live_products, "_available_memory_bytes", lambda: 64 << 30)


@pytest.mark.parametrize("host", ["observer", "child"])
def test_host_concurrency_matches_each_render_resource_budget(tmp_path, monkeypatch, eight_cpus, host):
    from gpuwm.offline_child_run import _ChildProgress
    from gpuwm.runplan import EventStream, RunObserver

    barrier = threading.Barrier(3, timeout=10)
    renderer = _Renderer()
    lock = threading.Lock()
    active = maximum = 0
    budgets = []
    early_budgets = []

    def render(command, **options):
        nonlocal active, maximum
        if "wrfout_d01_" in command[command.index("--series") - 1]:
            # The early render, alone on the host while it holds every slot.
            early_budgets.append(options.get("env_overrides"))
            return renderer(command)
        with lock:
            active += 1
            maximum = max(maximum, active)
            budgets.append(options["env_overrides"])
        try:
            barrier.wait()
            return renderer(command)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(first_products, "_run_render", render)
    events = EventStream(tmp_path / "events.jsonl", mirror=None)
    if host == "observer":
        owner = RunObserver(events, root_domain=1)
        owner.arm_first_products(_plan(tmp_path, "all"))
    else:
        owner = _ChildProgress()
        owner.arm_render(outdir=tmp_path, render_products="all")
    live = owner.live_products
    early = owner.first_products
    # A late first frame takes every shared slot before it publishes.
    with early._slot:
        assert not live._slot.acquire(blocking=False)
    try:
        # The guard is taken again by the render itself, which holds the
        # whole live budget on every route, as `gpuwm go`'s runner does.
        root, valid = _frame(tmp_path, 1, 0)
        assert early.frame_committed(domain=1, valid_time=valid, path=root)
        assert early.wait(30) is not None
        assert early_budgets == [{"RUSTWX_LIVE_RENDER_SLOTS": "1", "RAYON_NUM_THREADS": "7"}]
        for hour in range(3):
            frame, valid = _frame(tmp_path, 2, hour)
            live.frame_committed(domain=2, valid_time=valid, path=frame)
        summary = live.stop(timeout=30)
        assert summary["published"] == 3
        assert maximum == 3
        assert all(int(row["RUSTWX_LIVE_RENDER_SLOTS"]) == maximum for row in budgets)
        assert all(row["RAYON_NUM_THREADS"] == "2" for row in budgets)
    finally:
        live.halt()
        events.close()


def test_external_lock_gets_one_slot_and_the_matching_native_budget(tmp_path, monkeypatch, eight_cpus):
    calls = []

    def render(command, **options):
        calls.append(options["env_overrides"])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(first_products, "_run_render", render)
    live = live_products.LiveProducts(_plan(tmp_path), report=lambda _: None,
                                     warn=lambda *a, **k: None, slot=threading.Lock())
    live._run_render(["render"])
    assert calls == [{"RUSTWX_LIVE_RENDER_SLOTS": "1", "RAYON_NUM_THREADS": "7"}]


def test_shared_scratch_survives_frames_until_worker_shutdown(tmp_path):
    live = live_products.LiveProducts(_plan(tmp_path), report=lambda _: None,
                                     warn=lambda *a, **k: None, runner=_Renderer())
    frame, valid = _frame(tmp_path, 2, 0)
    live._render(domain=2, valid_time=valid, frame=frame)
    root = live.render_dir / live_products._SCRATCH_NAME
    assert root.is_dir()
    assert list(root.iterdir()) == []
    live.stop()
    assert not root.exists()


def test_request_catalog_is_initialized_once_without_blocking_the_queue(tmp_path, monkeypatch, eight_cpus):
    entered, release = threading.Event(), threading.Event()
    calls = []
    window_calls = []

    def classify(*args):
        calls.append(args)
        entered.set()
        assert release.wait(10)
        return False

    monkeypatch.setattr(live_products, "windows_only", classify)
    def request_windows(*args):
        window_calls.append(args)
        return True

    monkeypatch.setattr(live_products, "requests_windows", request_windows)
    live = live_products.LiveProducts(_plan(tmp_path), report=lambda _: None,
                                     warn=lambda *a, **k: None)
    live._closed = True
    loop = threading.Thread(target=live._loop)
    loop.start()
    try:
        assert entered.wait(5)
        assert live._cond.acquire(blocking=False)
        live._cond.release()
    finally:
        release.set()
        loop.join(10)
    assert not loop.is_alive()
    assert len(calls) == 1
    assert live._draws_windows_only() is False
    assert len(calls) == 1
    assert live._requests_windows() is True
    assert len(window_calls) == 1


def test_the_early_render_guard_can_be_entered_again():
    """Each render of the early frame takes the whole host, every time."""

    slots, early = live_products.shared_render_slots()
    for _ in range(2):
        with early:
            assert not slots.acquire(blocking=False)
    for _ in range(slots.width):
        assert slots.acquire(blocking=False)


def test_free_memory_caps_concurrent_frames_below_the_processor_budget(monkeypatch, eight_cpus):
    """Eight processors afford three frames; the memory has to as well."""

    assert live_products._render_concurrency() == 3
    monkeypatch.setattr(live_products, "_available_memory_bytes", lambda: 1 << 30)
    assert live_products._render_concurrency() == 1
    slots, _ = live_products.shared_render_slots()
    assert slots.width == 1
    # Half the free memory is the frames' share, as it is the renderer's.
    monkeypatch.setattr(live_products, "_available_memory_bytes",
                        lambda: 4 * live_products.LIVE_FRAME_PEAK_BYTES)
    assert live_products._render_concurrency() == 2
    # A platform that will not report free memory imposes no cap.
    monkeypatch.setattr(live_products, "_available_memory_bytes", lambda: None)
    assert live_products._render_concurrency() == 3


def test_free_memory_is_the_engine_host_probe(monkeypatch):
    from gpuwm.core import preflight

    monkeypatch.setattr(preflight, "host_available_bytes", lambda: 123 << 20)
    assert live_products._available_memory_bytes() == 123 << 20


@pytest.mark.parametrize("cpu_max,processors,frames,native", [
    ("400000 100000", 4, 1, "3"),
    ("150000 100000", 2, 1, "1"),
    ("max 100000", 64, 3, "21"),
])
def test_a_container_cpu_quota_bounds_the_processors_frames_are_budgeted_on(
        tmp_path, monkeypatch, cpu_max, processors, frames, native):
    monkeypatch.setattr(live_products.os, "sched_getaffinity", lambda _: set(range(64)), raising=False)
    monkeypatch.setattr(live_products, "_available_memory_bytes", lambda: 64 << 30)
    monkeypatch.delenv("RAYON_NUM_THREADS", raising=False)
    _cpu_max(monkeypatch, tmp_path, cpu_max)
    calls = []

    def render(command, **options):
        calls.append(options["env_overrides"])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(first_products, "_run_render", render)
    assert live_products._available_cpus() == processors
    assert live_products._render_concurrency() == frames
    live_products._run_render(["render"])
    assert calls == [{"RUSTWX_LIVE_RENDER_SLOTS": str(frames), "RAYON_NUM_THREADS": native}]


@pytest.mark.parametrize("membership,files,processors", [
    # A systemd slice two levels below a cgroup v2 root that has no
    # cpu.max of its own (a host's root never does).
    ("0::/system.slice/arwen.service",
     {"system.slice/cpu.max": "200000 100000",
      "system.slice/arwen.service/cpu.max": "max 100000"}, 2),
    # The tightest level binds, wherever it is.
    ("0::/user.slice/user-1000.slice/session-3.scope",
     {"user.slice/cpu.max": "600000 100000",
      "user.slice/user-1000.slice/session-3.scope/cpu.max": "250000 100000"}, 3),
    # A cgroup v1 cpu controller, the quota on the process's own group.
    ("4:cpu,cpuacct:/docker/abc\n2:memory:/docker/abc",
     {"cpu,cpuacct/docker/abc/cpu.cfs_quota_us": "300000",
      "cpu,cpuacct/docker/abc/cpu.cfs_period_us": "100000"}, 3),
    # A v1 container whose mount is its own group while /proc names the
    # host path: the mount's own quota is still read.
    ("4:cpu,cpuacct:/docker/abc",
     {"cpu/cpu.cfs_quota_us": "150000", "cpu/cpu.cfs_period_us": "100000"}, 2),
    # No quota anywhere on either hierarchy.
    ("4:cpu,cpuacct:/\n0::/init.scope",
     {"cpu,cpuacct/cpu.cfs_quota_us": "-1",
      "cpu,cpuacct/cpu.cfs_period_us": "100000",
      "init.scope/cpu.max": "max 100000"}, 64),
])
def test_a_cpu_quota_anywhere_up_the_process_cgroup_bounds_live_frames(
        tmp_path, monkeypatch, membership, files, processors):
    """THE BREAKAGE: only the root ``cpu.max`` was read, so a quota on a
    systemd slice, a nested group or a cgroup v1 host left 64 processors
    budgeted where the process could use two or three."""

    monkeypatch.setattr(live_products.os, "sched_getaffinity", lambda _: set(range(64)), raising=False)
    _cgroups(monkeypatch, tmp_path, membership, files)
    assert live_products._available_cpus() == processors


def _classic_frame(path, west_east, south_north, bottom_top):
    """A classic (CDF-2) history header with a WRF grid's dimensions and no data."""

    import netCDF4

    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(str(path), "w", format="NETCDF3_64BIT_OFFSET") as dataset:
        dataset.createDimension("Time", None)
        dataset.createDimension("DateStrLen", 19)
        dataset.createDimension("west_east", west_east)
        dataset.createDimension("south_north", south_north)
        dataset.createDimension("bottom_top", bottom_top)
        dataset.createDimension("bottom_top_stag", bottom_top + 1)
        dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))
    return path


#: Peak resident memory of one live render (the front door plus
#: ``rw_wrfbatch``, from the kernel's resource usage) against the frame's
#: grid and the grids of the baselines imported beside it.  The first
#: three rows are the 49-level GFS nest the single price was set on; the
#: next two a 3 km 880x704x55 GFS frame on the same eight-processor
#: worker; the last a 3 km 1154x922x55 GFS frame, about the largest
#: streamed domain an RTX PRO 4500 worker with 30 GiB of memory admits,
#: drawn with its pressure-level charts.  (A 1132x906x55 frame measured
#: 10,434 MiB while those charts were left out of frames that size.)
_MEASURED_PEAKS_MIB = [
    ((72, 58, 49), [], 482),
    ((72, 58, 49), [(72, 58, 49)], 346),
    ((144, 112, 49), [(144, 112, 49)] * 4, 616),
    ((880, 704, 55), [], 7181),
    ((880, 704, 55), [(880, 704, 55)] * 2, 7863),
    ((1154, 922, 55), [], 12001),
]


@pytest.mark.parametrize("grid,baselines,measured", _MEASURED_PEAKS_MIB)
def test_a_live_frame_is_priced_at_or_above_its_measured_peak(tmp_path, grid, baselines, measured):
    """A price below the measured peak is the failure the cap exists to prevent."""

    frame = _classic_frame(tmp_path / "wrfout_d01_2024-05-20_13_00_00", *grid)
    context = [_classic_frame(tmp_path / f"wrfout_d01_2024-05-20_12_{index:02d}_00", *shape)
               for index, shape in enumerate(baselines)]
    assert live_products._frame_cells(frame) == grid[0] * grid[1] * grid[2]
    price = live_products.live_frame_peak_bytes(frame, context)
    assert price >= measured << 20
    # And not the whole host for a small frame: at most a fifth over the peak.
    assert price <= (measured << 20) * 1.2 + live_products.LIVE_FRAME_PEAK_BYTES


def test_a_frame_whose_header_does_not_say_is_priced_from_its_size(tmp_path):
    frame = tmp_path / "wrfout_d01_2024-05-20_13_00_00"
    frame.write_bytes(b"\x89HDF\r\n\x1a\n" + bytes(4088))
    assert live_products._frame_cells(frame) == 1024
    assert live_products._frame_cells(tmp_path / "gone") == 0


def test_large_frames_are_drawn_one_at_a_time_where_small_ones_overlap(tmp_path, monkeypatch, eight_cpus):
    """THE BREAKAGE: one 640 MiB price for every grid.  With 16 GiB free an
    eight-processor host draws three frames at once, and three 880x704x55
    frames each held 7 GiB there: 21 GiB of renderers beside the forecast,
    against the 8 GiB half the frames may plan on."""

    monkeypatch.setattr(live_products, "_available_memory_bytes", lambda: 16 << 30)
    renderer = _Renderer(delay=0.3)
    live = live_products.LiveProducts(_plan(tmp_path), report=lambda _: None,
                                     warn=lambda *a, **k: None, runner=renderer,
                                     windowed_slugs=frozenset)
    assert live._concurrency == 3
    try:
        for domain in (1, 2, 3):
            frame, valid = _frame(tmp_path, domain, 0)
            _classic_frame(frame, 880, 704, 55)
            live.frame_committed(domain=domain, valid_time=valid, path=frame)
        assert live.stop(timeout=60)["published"] == 3
    finally:
        live.halt()
    assert renderer.most == 1


def test_frames_are_admitted_in_order_within_the_memory_budget():
    """A frame waits for room beside the frames drawing, in the order the
    frames asked; one priced past the whole budget still runs, alone."""

    unit = live_products.LIVE_FRAME_PEAK_BYTES
    slots = live_products._LiveRenderSlots(3, budget=10 * unit)
    order = []
    started = {name: threading.Event() for name in ("big", "small", "huge")}
    release = {name: threading.Event() for name in ("first", *started)}

    def draw(name, price):
        with slots.admit(price):
            order.append(name)
            if name in started:
                started[name].set()
            assert release[name].wait(10)

    first = threading.Thread(target=draw, args=("first", 6 * unit))
    first.start()
    while not order:
        time.sleep(0.01)
    big = threading.Thread(target=draw, args=("big", 6 * unit))
    big.start()
    time.sleep(0.1)
    small = threading.Thread(target=draw, args=("small", 3 * unit))
    small.start()
    # 6 + 3 fits, but the 6 that asked first is still waiting.
    assert not started["small"].wait(0.3)
    assert not started["big"].is_set()
    release["first"].set()
    assert started["big"].wait(5) and started["small"].wait(5)
    huge = threading.Thread(target=draw, args=("huge", 50 * unit))
    huge.start()
    assert not started["huge"].wait(0.3)
    release["big"].set()
    release["small"].set()
    assert started["huge"].wait(5)
    release["huge"].set()
    for thread in (first, big, small, huge):
        thread.join(10)
    assert order[0] == "first" and order[3] == "huge"
    assert sorted(order[1:3]) == ["big", "small"]
