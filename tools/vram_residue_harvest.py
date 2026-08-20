"""Modelled-vs-measured device peak over a tree of finished runs.

Reads each run's ``report.json`` -- the runtime's own
:class:`gpuwm.core.gpu_mem_watch.GpuPeakMemoryWatcher` receipt, sampled
at 20 Hz through the whole forecast -- and prices the same experiment
with the estimator, on the LIVE device profile.  Prints one JSON object.

Per run it reports the three numbers a reserve is built from:

* ``measured_non_pool_bytes`` = device footprint peak - pool-held peak.
  The CUDA context plus the launch-time local-memory backing store, as
  the run actually paid for them.
* ``implied_launched_frame_bytes`` -- that backing store read BACKWARDS
  through the reservation law, i.e. the widest per-thread frame the run
  really launched.  The estimator prices the widest frame a scheme CAN
  launch (a module maximum), which is an upper bound; this says by how
  much.
* ``residual_bytes`` = measured device peak - modelled envelope.  Must
  be <= 0 for the envelope to be an envelope.

Usage::

    python tools/vram_residue_harvest.py RUNROOT CFGROOT   # paired layout
    python tools/vram_residue_harvest.py --scan DIR [DIR...]

``--scan`` walks for ``report.json`` files carrying a peak-sampling
receipt and finds each one's experiment TOML by searching the run
directory and its parents.  A report whose config cannot be found is
reported as a skip, never silently dropped: a residue fit that quietly
lost its inconvenient points is not a measurement.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GIB = 1024 ** 3


def _peaks(report: dict) -> dict:
    mem = report.get("memory") or {}
    probes = ((mem.get("gpu_peak_sampling") or {}).get("probes") or {})

    def peak(name):
        row = probes.get(name) or {}
        value = row.get("peak_bytes")
        return int(value) if isinstance(value, int) else None

    return {
        "device_peak_bytes": peak("cuda_device_used"),
        "pool_total_peak_bytes": peak("cupy_pool_total"),
        "pool_used_peak_bytes": peak("cupy_pool_used"),
        "samples": ((probes.get("cuda_device_used") or {}).get("samples")),
    }


def _pairs_paired(run_root: Path, cfg_root: Path):
    for report_path in sorted(run_root.glob("*/run/report.json")):
        model = report_path.parents[1].name
        cfgs = sorted((cfg_root / model).glob("*.toml"))
        yield model, report_path, (cfgs[0] if cfgs else None)


def _find_config(report_path: Path) -> Path | None:
    """The experiment TOML for a run, searched outward from its report."""
    for up in (1, 2, 3, 4):
        if len(report_path.parents) <= up:
            break
        base = report_path.parents[up]
        seen: list[Path] = []
        for pattern in ("exp.toml", "authority/experiment.toml",
                        "experiment.toml", "*.toml", "cfg/*/*.toml"):
            seen.extend(sorted(base.glob(pattern)))
        # A namelist is not an experiment config; nor is a run receipt's
        # own sidecar.  Take the first real candidate only.
        seen = [p for p in seen if "namelist" not in p.name]
        if seen:
            return seen[0]
    return None


def _pairs_scan(roots: list[Path]):
    for root in roots:
        for report_path in sorted(root.rglob("report.json")):
            try:
                head = report_path.read_text()[:200000]
            except Exception:  # noqa: BLE001
                continue
            if "gpu_peak_sampling" not in head:
                continue
            label = "/".join(report_path.parts[-4:-1])
            yield label, report_path, _find_config(report_path)


def main(argv: list[str]) -> int:
    # A declared profile keeps this off the device entirely: harvesting
    # finished receipts needs no CUDA context, and a card another lane is
    # holding must not be touched to read a JSON file.
    declared = None
    family = None
    argv = list(argv)
    if "--profile" in argv:
        i = argv.index("--profile")
        declared = argv[i + 1]
        del argv[i:i + 2]
    if "--family" in argv:
        # Harvesting another machine's receipts on this one: the envelope
        # family belongs to the machine that RAN them, not to this shell.
        i = argv.index("--family")
        family = argv[i + 1]
        del argv[i:i + 2]

    if argv[1] == "--scan":
        pairs = list(_pairs_scan([Path(p) for p in argv[2:]]))
    else:
        pairs = list(_pairs_paired(Path(argv[1]), Path(argv[2])))

    from gpuwm.core import preflight as pf
    from gpuwm.domain_wizard import experiment_from_text

    if declared is None:
        import cupy as cp

        profile = pf.local_memory_profile_from_device(cp)
    else:
        name, sms, threads, stack = declared.split(",")
        profile = pf.DeviceLocalMemoryProfile(
            name=name, multiprocessor_count=int(sms),
            max_threads_per_multiprocessor=int(threads),
            default_stack_limit_bytes=int(stack))
    capacity = profile.resident_thread_capacity
    rows = []
    skipped = []
    for model, report_path, cfg in pairs:
        if cfg is None:
            skipped.append({"model": model, "why": "no experiment TOML found"})
            continue
        cfgs = [cfg]
        try:
            exp = experiment_from_text(cfgs[0].read_text(),
                                       source=f"<{model}>")
        except Exception as exc:  # noqa: BLE001
            skipped.append({"model": model, "why": f"config: {exc}"})
            continue
        report = json.loads(report_path.read_text())
        peaks = _peaks(report)
        if peaks["device_peak_bytes"] is None:
            skipped.append({"model": model, "why": "no device peak"})
            continue
        if peaks["pool_total_peak_bytes"] is None:
            skipped.append({"model": model, "why": "no pool peak"})
            continue

        est = pf.estimate_experiment(exp, profile=profile)
        if family is not None:
            import dataclasses as _dc
            est = _dc.replace(est, envelope_family=family)
        frames = pf.kernel_local_frame_bytes(exp)
        priced_frame = max(frames.values(), default=0)
        modelled_backing = pf.kernel_local_memory_bytes(exp, profile=profile)
        modelled_non_pool = pf.CUDA_CONTEXT_BYTES + modelled_backing
        measured_non_pool = (peaks["device_peak_bytes"]
                             - peaks["pool_total_peak_bytes"])
        # Invert the reservation law: what frame does that backing store
        # correspond to?  Only meaningful once the context is removed.
        measured_backing = measured_non_pool - pf.CUDA_CONTEXT_BYTES
        implied_frame = (None if measured_backing <= 0 else
                         int(measured_backing / capacity) + 1024)
        rows.append({
            "model": model,
            "config": str(cfg),
            "report": str(report_path),
            "grid": [[d.run.ny, d.run.nx, d.run.nz] for d in exp.domains],
            "domains": len(exp.domains),
            "run_seconds": exp.run_seconds,
            **peaks,
            "alloc_estimate_bytes": est.alloc_estimate_bytes,
            "modelled_non_pool_bytes": modelled_non_pool,
            "modelled_backing_bytes": modelled_backing,
            "priced_widest_frame_bytes": priced_frame,
            "measured_non_pool_bytes": measured_non_pool,
            "measured_backing_bytes": measured_backing,
            "implied_launched_frame_bytes": implied_frame,
            "modelled_envelope_bytes": est.peak_envelope_bytes,
            "residual_bytes": (peaks["device_peak_bytes"]
                               - est.peak_envelope_bytes),
            "pool_over_estimate": (peaks["pool_total_peak_bytes"]
                                   / est.alloc_estimate_bytes),
        })
    print(json.dumps({
        "profile": {
            "name": profile.name,
            "multiprocessor_count": profile.multiprocessor_count,
            "max_threads_per_multiprocessor":
                profile.max_threads_per_multiprocessor,
            "default_stack_limit_bytes": profile.default_stack_limit_bytes,
            "resident_thread_capacity": capacity,
        },
        "cuda_context_bytes_priced": pf.CUDA_CONTEXT_BYTES,
        "envelope_family": family or pf.envelope_platform(),
        "runs": rows,
        "skipped": skipped,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
