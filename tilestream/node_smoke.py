"""Fresh-node acceptance smoke: full physics, streaming, multi-GPU.

Six stages, each of which fails a DIFFERENT real bring-up failure that
has actually happened on a rented box.  Run it as the last step of the
bootstrap; a node that passes all six can be given work.

  1  cuBLAS load        the wrong CuPy wheel imports cleanly, compiles
                        kernels, and dies at the first cuBLAS load.  An
                        import probe cannot see it; a matmul can.
  2  Thompson tables    freezeH2O.dat is not in the wheel and not in the
                        repo.  Missing, it blocks mp28, therefore every
                        moist rung, therefore cumulus -- and it surfaces
                        as "cumulus physics requires a moist DomainState",
                        which reads like a config mistake and is not one.
  3  full physics fires builds the real rung (full+MYNN+Noah-MP, nz=49,
                        ztop=20000) and steps from clock 0 so radiation
                        AND cumulus are DUE inside the window.  Asserts
                        the fire counts are non-zero by wrapping the
                        driver's own due-predicates -- a rung that
                        silently fell back to dry passes every other
                        check and every number taken on it is wrong.
                        Also records peak-on-fire VRAM, which is what a
                        sizing formula must be built from: RRTMGP and KF
                        allocate scratch only on the steps they fire, so
                        a ceiling that never crossed a radiation boundary
                        is not a ceiling.
  4  streaming exact    tiled == monolithic, carrier by carrier, at the
                        same rung.  Streaming buys CAPACITY, never speed.
  5  multi-GPU exact    P=2 halo exchange bit-exact against the 1-GPU
                        monolithic reference, with rad/cu fired on both
                        sides.  Skipped with a stated reason on 1 card.
  6  sweep returns      runs several DIFFERENT configurations back to
                        back in one process and checks device occupancy
                        returns to baseline each time.  Decides whether
                        this node may be driven by a sweep at all.

Every stage prints what it proved, not that it passed.  Exit status is
the number of failed stages.

USAGE
    cd <checkout>            # tilestream is NOT an installed package
    python -m tilestream.node_smoke [--skip-mgpu] [--json OUT]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

GIB = 1 << 30

#: The rung that matters.  Copied from tilestream.test_gate at run time,
#: never re-typed: ztop=20000.0 is load-bearing (RRTMGP refuses nz>=41
#: under the 8 km harness default) and a locally-written rung dict is how
#: a run silently becomes a different numerical setup.
RUNG = "full+MYNN+Noah-MP"

#: The streaming stage's domain.  Small on purpose: it is a CORRECTNESS
#: stage and 3x3-ragged over 96x80 exercises the ragged edge in seconds.
NX, NY, NZ = 96, 80, 49

#: PEAK-ON-FIRE sizing, from the 8x4090 gate.  Steady-state is 268.6
#: B/cell; RRTMGP and Kain-Fritsch allocate scratch ONLY on the steps they
#: fire, so a ceiling measured in a window that never crossed a radiation
#: boundary is not a ceiling -- that error is what OOM'd a 60.2 Mcell
#: reference inside _prepare_atmosphere.  Size on these two numbers.
PEAK_BYTES_PER_CELL = 869.5
PEAK_FIXED_BYTES = int(3.412 * GIB)

#: Domains stage 3 will try, largest first.  512x384x49 = 9.6 Mcell is the
#: gate's domain, so its ns/cell is directly comparable to the recorded
#: 49.3 ns/cell (RTX 4090, full+MYNN+Noah-MP).
PHYSICS_LADDER = ((512, 384), (384, 288), (256, 192), (128, 96), (96, 80))

#: Fraction of FREE VRAM the smoke is allowed to take.  Deliberately under
#: half: the smoke must not be the reason a node's real job OOMs, and
#: cudaMemGetInfo is device-wide so "free" already accounts for whatever
#: else is resident.
VRAM_BUDGET_FRACTION = 0.45


def peak_on_fire_bytes(cells: int) -> int:
    return int(cells * PEAK_BYTES_PER_CELL) + PEAK_FIXED_BYTES


def pick_physics_domain() -> tuple[int, int, str]:
    """Largest ladder domain whose PEAK-ON-FIRE fits the budget."""
    import cupy as cp
    free, total = cp.cuda.runtime.memGetInfo()
    budget = int(free * VRAM_BUDGET_FRACTION)
    for nx, ny in PHYSICS_LADDER:
        need = peak_on_fire_bytes(nx * ny * NZ)
        if need <= budget:
            return nx, ny, (
                f"{free / GIB:.1f} GiB free of {total / GIB:.1f}; budget "
                f"{budget / GIB:.1f} GiB at {VRAM_BUDGET_FRACTION:.0%}; "
                f"{nx}x{ny}x{NZ} projects {need / GIB:.1f} GiB peak-on-fire")
    nx, ny = PHYSICS_LADDER[-1]
    return nx, ny, (f"{free / GIB:.1f} GiB free: even {nx}x{ny}x{NZ} exceeds "
                    f"the budget; running it anyway as the floor")


class Stage:
    def __init__(self, key: str, title: str):
        self.key, self.title = key, title
        self.ok: bool | None = None
        self.detail = ""
        self.seconds = 0.0
        self.data: dict = {}


def _banner(text: str) -> None:
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}", flush=True)


# ---------------------------------------------------------------- stage 0

def stage_topology(stage: Stage) -> None:
    """Everything a bootstrap must DETECT rather than hardcode.

    Never fails the run on its own -- a single-NUMA-node box and a
    dual-socket box are both legitimate.  It fails only when the answer
    would silently make a later stage wrong: no CUDA driver at all, or a
    reported GPU affinity the bootstrap would have to guess around.
    """
    from tilestream import node_probe as probe

    major, how = probe.driver_cuda_major()
    mem, mem_how = probe.cgroup_memory_limit()
    nodes = probe.numa_nodes()
    gpus = probe.gpu_affinity()
    plan = probe.core_plan(gpus)
    host = probe.meminfo_total()

    stage.data = dict(cuda_major=major, cuda_how=how,
                      cupy_wheel=probe.cupy_wheel_for(major),
                      cgroup_bytes=mem, cgroup_how=mem_how,
                      meminfo_bytes=host,
                      numa_nodes={str(k): len(v) for k, v in nodes.items()},
                      gpus=gpus, rank_core_plan={str(k): v
                                                 for k, v in plan.items()})

    if major is None:
        stage.ok = False
        stage.detail = f"no CUDA driver: {how}"
        return

    lines = [f"CUDA driver major {major} ({how}) -> "
             f"{probe.cupy_wheel_for(major)}"]

    if mem is not None:
        lines.append(f"host RAM budget {mem / GIB:.1f} GiB from {mem_how}"
                     + (f"  (/proc/meminfo claims {host / GIB:.1f} GiB and is "
                        f"LYING -- do not size from it)" if host and
                        host > mem * 1.05 else ""))
    else:
        lines.append(f"host RAM: {mem_how}"
                     + (f"; /proc/meminfo says {host / GIB:.1f} GiB"
                        if host else ""))

    binding = [g for g in gpus if g["binding_required"]]
    if len(nodes) <= 1:
        lines.append(f"NUMA: 1 node, {len(nodes.get(0, []))} cpus -- no "
                     f"binding required")
    elif not binding:
        lines.append(
            f"NUMA: {len(nodes)} nodes but NO gpu reports an affinity "
            f"(numa_node=-1 or sysfs absent) -- do NOT assume node 0; run "
            f"unbound")
    else:
        lines.append(f"NUMA: {len(nodes)} nodes; binding IS required")
        for gpu in gpus:
            cpus = gpu["local_cpus"] or []
            span = f"{cpus[0]}-{cpus[-1]}" if cpus else "-"
            lines.append(f"  gpu{gpu['device']} {gpu['pci']} "
                         f"numa_node={gpu['numa_node']} local {span} "
                         f"({len(cpus)} cpus)")
        for rank in sorted(plan):
            cores = plan[rank]
            lines.append(f"  rank {rank} -> cores {cores[0]}-{cores[-1]} "
                         f"({len(cores)})")
        remote = {n for n in nodes if n not in
                  {g["numa_node"] for g in binding}}
        if remote:
            lines.append(f"  node(s) {sorted(remote)} are REMOTE to every "
                         f"card: binding a rank there is the bug, not the "
                         f"optimisation")

    stage.ok = True
    stage.detail = "\n        ".join(lines)


# ---------------------------------------------------------------- stage 1

#: Each CUDA library is loaded SEPARATELY, because "cuBLAS failed" has two
#: completely different causes with two different fixes and the existing
#: tooling conflates them:
#:
#:   wrong wheel      cupy-cuda12x on a CUDA-13-only box.  Imports cleanly,
#:                    compiles kernels, dies at the first cuBLAS load.
#:                    Fix: install the wheel matching the DRIVER major.
#:   missing libs     the RIGHT wheel, but no CUDA math libraries on the
#:                    box at all.  `pip install cupy-cuda13x` pulls ONLY
#:                    numpy + cuda-pathfinder -- it does NOT pull cuBLAS,
#:                    cuFFT or cuSOLVER.  The 8x4090 box got them from its
#:                    image's system toolkit (nvcc 13.0.88), which a bare
#:                    node does not have.  MEASURED on this dev box: NVRTC
#:                    and cuRAND load, cuBLAS and cuFFT do not.
#:                    Fix: pip install 'cupy-cuda13x[ctk]'.
#:
#: NVRTC is separated out because it is the one that is actually fatal to
#: a forecast: every gpuwm kernel is JIT-compiled.  Full physics runs with
#: cuBLAS absent (measured); the DA eigensolver and `gpuwm doctor`'s own
#: pairing probe do not.
_LIB_PROBE = r"""
import json, sys
out = {"libs": {}}
try:
    import cupy as cp
except Exception as e:
    print(json.dumps({"fatal": f"import cupy: {e!r}", "libs": {}}))
    raise SystemExit(0)
out["cupy"] = cp.__version__
out["runtime"] = int(cp.cuda.runtime.runtimeGetVersion())
out["driver"] = int(cp.cuda.runtime.driverGetVersion())
try:
    out["devices"] = int(cp.cuda.runtime.getDeviceCount())
except Exception as e:
    out["devices"] = 0
    out["fatal"] = f"getDeviceCount: {e!r}"
    print(json.dumps(out)); raise SystemExit(0)
a = cp.arange(64, dtype=cp.float32)
probes = (
    ("nvrtc",   lambda: float((a * 2 + 1).sum())),
    ("curand",  lambda: float(cp.random.random(16).sum())),
    ("cublas",  lambda: float((cp.ones((64, 64), cp.float32)
                               @ cp.ones((64, 64), cp.float32)).sum())),
    ("cufft",   lambda: float(abs(cp.fft.fft(a.astype(cp.complex64))[0]))),
    ("cusolver", lambda: float(cp.linalg.solve(
        cp.eye(4, dtype=cp.float32), cp.ones(4, cp.float32)).sum())),
)
for name, fn in probes:
    try:
        fn()
        out["libs"][name] = True
    except Exception as e:
        out["libs"][name] = f"{type(e).__name__}: {e}"
print(json.dumps(out))
"""

#: Absent, these stop a forecast.  cuBLAS/cuFFT/cuSOLVER are reported but
#: do not fail the stage: full physics was measured running without them.
_FATAL_LIBS = ("nvrtc", "curand")


def stage_cublas(stage: Stage) -> None:
    proc = subprocess.run([sys.executable, "-c", _LIB_PROBE],
                          capture_output=True, text=True, timeout=300)
    try:
        info = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        stage.ok = False
        stage.detail = f"probe produced no JSON: {proc.stderr.strip()[-400:]}"
        return
    stage.data = info
    if info.get("fatal"):
        stage.ok = False
        stage.detail = info["fatal"]
        return

    libs = info.get("libs", {})
    wheel_major = (info.get("runtime") or 0) // 1000
    box_major = (info.get("driver") or 0) // 1000
    broken = [k for k, v in libs.items() if v is not True]
    fatal = [k for k in _FATAL_LIBS if libs.get(k) is not True]

    head = (f"cupy {info.get('cupy')}, wheel serves CUDA {wheel_major}, "
            f"driver serves CUDA {box_major}, {info.get('devices')} device(s)")

    if fatal:
        stage.ok = False
        stage.detail = (f"{head}\n        FATAL: "
                        + "; ".join(f"{k}: {libs[k]}" for k in fatal))
        return

    if broken:
        wrong_wheel = bool(wheel_major and box_major
                           and wheel_major != box_major)
        # Reported, not fatal, unless the wheel major itself is wrong: a
        # forecast was measured running with cuBLAS/cuFFT/cuSOLVER absent.
        stage.ok = not wrong_wheel
        if wrong_wheel:
            remedy = (f"WRONG WHEEL: pip uninstall -y cupy-cuda{wheel_major}x "
                      f"&& pip install cupy-cuda{box_major}x")
        else:
            remedy = (f"MISSING CUDA MATH LIBS (the wheel major is right): "
                      f"pip install 'cupy-cuda{box_major}x[ctk]'\n"
                      f"        -- or install a system CUDA {box_major} "
                      f"toolkit.  Full physics runs without these; DA, the "
                      f"eigensolver\n           and `gpuwm doctor`'s pairing "
                      f"probe do not.")
        stage.detail = (f"{head}\n        loaded: "
                        + ", ".join(k for k, v in libs.items() if v is True)
                        + "\n        NOT loaded: "
                        + "; ".join(f"{k} ({libs[k].split(':')[-1].strip()})"
                                    for k in broken)
                        + f"\n        {remedy}")
        return

    stage.ok = True
    stage.detail = (f"{head}\n        all of "
                    + ", ".join(libs) + " loaded and executed")


# ---------------------------------------------------------------- stage 2

def stage_tables(stage: Stage) -> None:
    from gpuwm.core.thompson_contract import (CLASSIC_TABLE_ASSETS,
                                              validate_table_assets)
    from gpuwm.physics_compat import (packaged_thompson_table_root,
                                      thompson_table_root,
                                      user_thompson_table_root)
    root = thompson_table_root()
    stage.data["root"] = str(root)
    stage.data["packaged"] = str(packaged_thompson_table_root())
    stage.data["staged"] = str(user_thompson_table_root())
    try:
        assets = validate_table_assets(root)
    except Exception as error:
        missing = [a.filename for a in CLASSIC_TABLE_ASSETS
                   if not (Path(root) / a.filename).is_file()]
        stage.ok = False
        stage.detail = (
            f"{error}\n"
            f"        resolved root: {root}\n"
            f"        absent: {', '.join(missing) or '(present but wrong bytes)'}\n"
            f"        remedy: gpuwm fetch-tables      (315 MiB, SHA-256 pinned)\n"
            f"                gpuwm fetch-tables --from DIR   (offline)\n"
            f"        destination: {user_thompson_table_root()}")
        return
    stage.ok = True
    stage.detail = (f"{len(assets)} assets byte-validated at {root} "
                    f"({sum(a.bytes for a in assets):,} B, exact size + "
                    f"SHA-256, the same check every mp8 run runs at load)")


# ------------------------------------------------------------ stages 3 & 6

def _install_fire_counters():
    """Wrap the driver's OWN due-predicates so firings are counted."""
    from gpuwm.core import physics as physmod
    existing = getattr(physmod, "_smoke_fire_counts", None)
    if existing is not None:
        return existing
    counts = {"rad": 0, "cu": 0}
    rad0, cu0 = physmod._radiation_step_due, physmod._cumulus_step_due

    def rad(*a, **k):
        v = rad0(*a, **k)
        counts["rad"] += bool(v)
        return v

    def cu(*a, **k):
        v = cu0(*a, **k)
        counts["cu"] += bool(v)
        return v

    physmod._radiation_step_due, physmod._cumulus_step_due = rad, cu
    physmod._smoke_fire_counts = counts
    return counts


def _build_and_fire(nx: int, ny: int, nsteps: int = 3) -> dict:
    import cupy as cp
    from tilestream import harness, physics_inventory as physinv, test_gate

    kwargs = dict(test_gate.PHYSICS_RUNGS[RUNG])
    counts = _install_fire_counters()
    counts["rad"] = counts["cu"] = 0

    pool = cp.get_default_memory_pool()
    t0 = time.perf_counter()
    cfg = harness.make_config(nx, ny, NZ, **kwargs)
    state, drv = physinv.default_builder(cfg)
    cp.cuda.runtime.deviceSynchronize()
    build_s = time.perf_counter() - t0

    # Carrier count BEFORE and AFTER the first step, because they differ
    # and a host store sized from the first number is short a field:
    # Kain-Fritsch allocates cumulus/w0avg on its first call, so a state
    # that has never stepped has a SHORTER manifest than one that has.
    carriers_fresh = len(physinv.carrier_inventory(state))

    # STEP 1 IS THE FIRE STEP and it is timed on its own.  At dt=3 s with
    # radt=12 min radiation is due every 240 steps and cumulus every 100,
    # so the ONLY step in a short window that fires is the one taken from
    # clock 0.  Rolling it into a warmup -- which an earlier draft of this
    # file did -- leaves the timed window crossing no cadence boundary at
    # all, and the fire assertion below is what caught that.
    counts["rad"] = counts["cu"] = 0
    t1 = time.perf_counter()
    harness.run_steps(state, cfg, 1)
    cp.cuda.runtime.deviceSynchronize()
    fire_s = time.perf_counter() - t1
    fire_rad, fire_cu = counts["rad"], counts["cu"]
    carriers_stepped = len(physinv.carrier_inventory(state))
    steady = pool.used_bytes()
    # PEAK-ON-FIRE, measured DEVICE-WIDE, because that is how the
    # 869.5 B/cell + 3.412 GiB sizing constants were derived.  The pool
    # counter cannot see the CUDA context, the JIT-compiled modules or any
    # non-pool allocation, and it is the device-wide number that decides
    # whether a domain OOMs.  The trap that comes with it: cudaMemGetInfo
    # is device-wide in the other direction too, so any FOREIGN process on
    # this card silently inflates it -- hence foreign_ctx below.
    free_now, total_dev = cp.cuda.runtime.memGetInfo()
    peak = pool.total_bytes()          # the fire step is where the peak is
    device_peak = total_dev - free_now

    # The remaining steps cross no boundary: this is the PLAIN rate, the
    # one a forecast spends almost all of its wall clock at.
    counts["rad"] = counts["cu"] = 0
    t2 = time.perf_counter()
    harness.run_steps(state, cfg, nsteps)
    cp.cuda.runtime.deviceSynchronize()
    plain_s = time.perf_counter() - t2

    cells = nx * ny * NZ
    out = dict(kwargs=kwargs, nx=nx, ny=ny, nz=NZ,
               carriers_fresh=carriers_fresh,
               carriers_stepped=carriers_stepped, cells=cells,
               build_seconds=build_s, nsteps=nsteps,
               fire_seconds=fire_s, plain_seconds=plain_s,
               rad=fire_rad, cu=fire_cu,
               plain_rad=counts["rad"], plain_cu=counts["cu"],
               steady_bytes=steady, peak_bytes=peak,
               device_peak_bytes=device_peak, device_total_bytes=total_dev,
               projected_peak_bytes=peak_on_fire_bytes(cells),
               ns_per_cell=plain_s / nsteps / cells * 1e9,
               mcell_per_s=cells / (plain_s / nsteps) / 1e6,
               fire_ratio=fire_s / max(plain_s / nsteps, 1e-9),
               dt=float(cfg.dt), radt=float(cfg.radt_minutes),
               cudt=float(cfg.cudt_minutes), halo=harness.halo_radius(cfg))
    # gc.collect() BEFORE free_all_blocks(), and it is load-bearing.
    # DomainState and its PhysicsDriver reference each other, so dropping
    # the last name does NOT run __del__: the arrays survive until the
    # CYCLIC collector runs, and free_all_blocks() called before that
    # returns nothing.  A sweep harness that omits it retains the whole
    # previous configuration -- a domain-sized amount of VRAM -- and the
    # next configuration OOMs on a domain that fits fine.  MEASURED here:
    # 4.69 GiB retained without this line, 0.00 with it.
    del state, drv, cfg
    gc.collect()
    pool.free_all_blocks()
    return out


def stage_physics(stage: Stage) -> None:
    nx, ny, why = pick_physics_domain()
    info = _build_and_fire(nx, ny)
    info["sizing"] = why
    stage.data = info
    if info["rad"] < 1 or info["cu"] < 1:
        stage.ok = False
        stage.detail = (
            f"rung built but the step from clock 0 fired radiation "
            f"{info['rad']}x and cumulus {info['cu']}x.\n"
            f"        A rung that does not fire is not the rung you think "
            f"you measured, and every ns/cell taken on it is wrong.\n"
            f"        dt={info['dt']}s radt={info['radt']}min "
            f"cudt={info['cudt']}min -> radiation is due every "
            f"{int(info['radt'] * 60 / info['dt'])} steps, cumulus every "
            f"{int(info['cudt'] * 60 / info['dt'])}")
        return
    ratio = info["device_peak_bytes"] / max(info["projected_peak_bytes"], 1)
    stage.ok = True
    stage.detail = (
        f"{RUNG}: {nx}x{ny}x{NZ} = {info['cells'] / 1e6:.2f} Mcell, "
        f"halo={info['halo']}\n"
        f"        sizing: {why}\n"
        f"        carriers {info['carriers_fresh']} fresh -> "
        f"{info['carriers_stepped']} after one step -- KF allocates "
        f"cumulus/w0avg on its first call,\n"
        f"                so a host store sized from the fresh manifest is "
        f"short a field that later appears\n"
        f"        FIRE step (from clock 0): radiation {info['rad']}x, "
        f"cumulus {info['cu']}x, {info['fire_seconds'] * 1e3:.0f} ms = "
        f"{info['fire_ratio']:.1f}x plain\n"
        f"                counted off the driver's own due-predicates, not "
        f"read off the config\n"
        f"        PLAIN {info['plain_seconds'] / info['nsteps'] * 1e3:.0f} "
        f"ms/step, {info['mcell_per_s']:.1f} Mcell/s, "
        f"{info['ns_per_cell']:.0f} ns/cell "
        f"(rad={info['plain_rad']} cu={info['plain_cu']} in that window)\n"
        f"        build {info['build_seconds']:.1f}s; pool steady "
        f"{info['steady_bytes'] / GIB:.2f} GiB, pool peak "
        f"{info['peak_bytes'] / GIB:.2f} GiB,\n"
        f"                DEVICE-WIDE PEAK-ON-FIRE "
        f"{info['device_peak_bytes'] / GIB:.2f} GiB of "
        f"{info['device_total_bytes'] / GIB:.1f} = {ratio:.2f}x the "
        f"869.5 B/cell + 3.412 GiB projection\n"
        f"        size domains from the DEVICE-WIDE PEAK: the pool counter "
        f"cannot see the context, the JIT\n"
        f"                modules or any non-pool allocation, and the "
        f"steady value misses the scheme scratch\n"
        f"                that RRTMGP and KF take only on the steps they "
        f"fire")


def stage_leak(stage: Stage, first: dict) -> None:
    """Does a SWEEP return the card, or does configuration N+1 inherit N?

    This is the stage that decides whether a node may be driven by a
    sweep at all, so it runs the thing at issue: several DIFFERENT
    configurations, back to back, in one process, reading the DEVICE-WIDE
    figure between them.  The pool counter cannot answer it -- what OOMs
    the next configuration is device occupancy, not pool bookkeeping.

    MEASURED, and it corrects the received rule.  With references dropped
    and the CYCLIC collector run before free_all_blocks(), a four-
    configuration sweep returns to the same device baseline every time
    (3.73 GiB here, flat).  Without gc.collect(), the previous state
    survives -- DomainState and its PhysicsDriver reference each other, so
    the last `del` does not free anything -- and the process retains a
    whole domain (4.69 GiB measured at 9.6 Mcell).  That is the reported
    failure, and its fix is one line, not a policy.
    """
    import cupy as cp
    pool = cp.get_default_memory_pool()

    def device_used() -> int:
        free, total = cp.cuda.runtime.memGetInfo()
        return total - free

    baseline = device_used()
    ladder = [(nx, ny) for nx, ny in PHYSICS_LADDER
              if nx * ny <= first["nx"] * first["ny"]][:3] or [
        (first["nx"], first["ny"])]

    marks = []
    for nx, ny in ladder:
        _build_and_fire(nx, ny, nsteps=1)
        marks.append((nx, ny, device_used()))

    residual = marks[-1][2] - baseline
    # One domain's worth of steady state is the scale that matters: if the
    # process is holding that much extra, it is holding a whole state.
    one_domain = int(first["nx"] * first["ny"] * NZ * 268.6)
    stage.data = dict(baseline=baseline, marks=marks, residual=residual,
                      one_domain=one_domain, pool_retained=pool.total_bytes())

    trail = "  ".join(f"{nx}x{ny}:{used / GIB:.2f}" for nx, ny, used in marks)
    if residual > 0.5 * one_domain:
        stage.ok = False
        stage.detail = (
            f"device occupancy did NOT return across a "
            f"{len(ladder)}-configuration sweep:\n"
            f"        baseline {baseline / GIB:.2f} GiB -> {trail} GiB "
            f"(residual {residual / GIB:.2f} GiB,\n"
            f"        vs {one_domain / GIB:.2f} GiB for one domain's steady "
            f"state)\n"
            f"        Drive this node ONE CONFIGURATION PER PROCESS, and "
            f"check the sweep harness drops its\n"
            f"        references and calls gc.collect() BEFORE "
            f"free_all_blocks().")
        return
    stage.ok = True
    stage.detail = (
        f"{len(ladder)} DIFFERENT configurations back to back returned the "
        f"card every time:\n"
        f"        baseline {baseline / GIB:.2f} GiB -> {trail} GiB "
        f"(residual {residual / GIB:.2f} GiB)\n"
        f"        The received 'sweeps leak' rule is a missing gc.collect(): "
        f"DomainState and its\n"
        f"        PhysicsDriver reference each other, so `del state` alone "
        f"frees nothing and\n"
        f"        free_all_blocks() called before the cyclic collector "
        f"returns nothing.")


# ---------------------------------------------------------------- stage 4

def stage_streaming(stage: Stage) -> None:
    from tilestream import test_gate
    t0 = time.perf_counter()
    # 3x3 ragged over the 96x80 domain: tiles do not divide the domain, so
    # the ragged edge is exercised, and nbuffers=2 exercises the cadence
    # warmup that a single buffer hides.
    case = test_gate.physics_case(RUNG, tile_nx=40, tile_ny=30, nsteps=3,
                                  nbuffers=2, write_mode="ring")
    stage.data = {k: case.get(k) for k in
                  ("bitexact", "carriers", "tiles", "differing", "seconds")}
    stage.seconds = time.perf_counter() - t0
    if not case.get("bitexact"):
        stage.ok = False
        stage.detail = (
            f"tiled != monolithic: {len(case.get('differing', []))} of "
            f"{case.get('carriers')} carriers differ\n"
            f"        first few: {case.get('differing', [])[:6]}")
        return
    stage.ok = True
    stage.detail = (
        f"tiled == monolithic on ALL {case['carriers']} carriers, "
        f"{case['tiles']} tiles (3x3 ragged), 3 steps, nbuffers=2\n"
        f"        streaming buys CAPACITY at ~2x the wall clock, never speed")


# ---------------------------------------------------------------- stage 5

def stage_multigpu(stage: Stage, skip: bool) -> None:
    import cupy as cp
    try:
        ndev = cp.cuda.runtime.getDeviceCount()
    except Exception as error:
        stage.ok = None
        stage.detail = f"device count unavailable: {error}"
        return
    stage.data["devices"] = ndev
    if skip:
        stage.ok = None
        stage.detail = "skipped on request (--skip-mgpu)"
        return
    if ndev < 2:
        stage.ok = None
        stage.detail = (f"only {ndev} device: multi-GPU cannot be gated here. "
                        f"This node is single-card-usable ONLY.")
        return
    here = Path(__file__).resolve().parent
    gate = here / "mgpu_gate.py"
    if not gate.is_file():
        have_decomp = (here / "decomp.py").is_file()
        stage.ok = False
        stage.detail = (
            f"{ndev} devices present but {gate.name} does not exist, so this "
            f"node's multi-GPU claim is UNGATED.\n"
            f"        Present: tilestream/decomp.py + test_decomp_gate.py "
            f"gate the initial-condition SLICE\n"
            f"        (every point owned once, slice bit-identical to "
            f"monolithic, seams continuous) -- {'yes' if have_decomp else 'NO'}.\n"
            f"        Missing: the RUNNING gate -- ranks on separate GPUs, "
            f"halo exchanged every step,\n"
            f"        digest over all carriers equal to the 1-GPU monolithic "
            f"run, rad/cu fired both sides.\n"
            f"        The working implementation is rescued but uncommitted at "
            f"rescued/box-8x4090/\n"
            f"        {{mgpu_phys2.py, gate_phys.py}}; it needs "
            f"os.sched_setaffinity from node_probe.core_plan\n"
            f"        rather than its hardcoded NODE0 list, and $ARWEN_SHM "
            f"staging so two lanes on one\n"
            f"        box cannot delete each other's planes.")
        return
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(gate), "--nx", "256", "--ny", "192",
         "--nz", str(NZ), "--p", "2", "--steps", "3", "--json", "-"],
        capture_output=True, text=True, timeout=900, cwd=str(here.parent))
    stage.seconds = time.perf_counter() - t0
    try:
        info = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        stage.ok = False
        stage.detail = f"gate produced no JSON: {proc.stderr.strip()[-500:]}"
        return
    stage.data.update(info)
    if not info.get("bitexact") or info.get("rad", 0) < 1 or info.get("cu", 0) < 1:
        stage.ok = False
        stage.detail = (f"P=2 gate did not pass: bitexact="
                        f"{info.get('bitexact')} rad={info.get('rad')} "
                        f"cu={info.get('cu')}")
        return
    stage.ok = True
    stage.detail = (
        f"P=2 halo exchange bit-exact vs the 1-GPU monolithic reference "
        f"over all {info.get('carriers')} carriers,\n"
        f"        rad={info['rad']} cu={info['cu']} fired on BOTH sides, "
        f"exchange {info.get('exch_fraction', float('nan')) * 100:.0f}% of "
        f"the step")


# -------------------------------------------------------------------- main

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-mgpu", action="store_true")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    print(f"node smoke  --  {RUNG}  nz={NZ}  "
          f"python {sys.version.split()[0]}  cwd={os.getcwd()}", flush=True)

    stages = [
        Stage("topology", "0  node topology: CUDA major, cgroup RAM, NUMA"),
        Stage("cublas", "1  cuBLAS actually loads (the wrong-wheel killer)"),
        Stage("tables", "2  Thompson tables staged and byte-valid"),
        Stage("physics", "3  full physics builds AND fires"),
        Stage("streaming", "4  streaming is bit-exact"),
        Stage("mgpu", "5  multi-GPU is bit-exact"),
        Stage("leak", "6  a sweep returns the card between configurations"),
    ]
    by_key = {s.key: s for s in stages}

    order = ["topology", "cublas", "tables", "physics", "streaming", "mgpu",
             "leak"]
    for key in order:
        stage = by_key[key]
        _banner(stage.title)
        t0 = time.perf_counter()
        try:
            if key == "topology":
                stage_topology(stage)
            elif key == "cublas":
                stage_cublas(stage)
            elif key == "tables":
                stage_tables(stage)
            elif key == "physics":
                stage_physics(stage)
            elif key == "streaming":
                stage_streaming(stage)
            elif key == "mgpu":
                stage_multigpu(stage, args.skip_mgpu)
            elif key == "leak":
                if by_key["physics"].ok:
                    stage_leak(stage, by_key["physics"].data)
                else:
                    stage.ok = None
                    stage.detail = "skipped: stage 3 did not pass"
        except Exception as error:            # noqa: BLE001
            import traceback
            stage.ok = False
            stage.detail = f"{type(error).__name__}: {error}"
            traceback.print_exc()
        stage.seconds = stage.seconds or (time.perf_counter() - t0)
        mark = {True: "PASS", False: "FAIL", None: "SKIP"}[stage.ok]
        print(f"  {mark}  ({stage.seconds:.1f}s)  {stage.detail}", flush=True)
        # A node with no working cuBLAS or no tables cannot answer anything
        # downstream; stopping is honest, continuing produces noise.
        if stage.ok is False and key in ("cublas", "tables"):
            break

    _banner("SUMMARY")
    failed = 0
    total = 0.0
    for stage in stages:
        mark = {True: "PASS", False: "FAIL", None: "SKIP"}[stage.ok]
        failed += stage.ok is False
        total += stage.seconds
        print(f"  {mark}  {stage.seconds:6.1f}s  {stage.title}")
    print(f"\n  total {total:.1f}s   {failed} stage(s) failed")

    if args.json:
        payload = {s.key: dict(ok=s.ok, seconds=s.seconds, detail=s.detail,
                               data=s.data) for s in stages}
        if args.json == "-":
            print(json.dumps(payload, default=str))
        else:
            Path(args.json).write_text(json.dumps(payload, indent=2,
                                                  default=str))
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
