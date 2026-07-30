"""Two-arm solo-GPU bitwise/performance A/B for Morrison step segments.

The parent process launches each checkout from a neutral temporary directory
with ``PYTHONPATH`` pinned to that arm.  Each arm runs the same radiation-off
real74 d01 segment and the same Morrison WK82 moist segment, synchronizes the
timed region, and hashes the final FP32 prognostics plus Morrison diagnostics.
The parent exits nonzero unless every named output is bit-identical.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np


_DRY_PROGNOSTICS = ("u", "v", "w", "thp", "php", "mup")
_MORRISON_OUTPUTS = (
    "qv", "qc", "qr", "qi", "qs", "qg",
    "nc", "nr", "ni", "ns", "ng", "h_diabatic",
    "effc", "effr", "effi", "effs",
)
_SURFACE_OUTPUTS = (
    "rainnc", "rainncv", "snownc", "snowncv",
    "graupelnc", "graupelncv", "sr",
)
_MORRISON_SURFACE_SCRATCH = {
    name: f"mp_{name}" for name in _SURFACE_OUTPUTS
}
_RESULT_PREFIX = "AB_MP_RESULT "


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--real74-steps", type=int, default=8)
    parser.add_argument("--wk82-steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument(
        "--case", choices=("all", "real74", "wk82"), default="all",
        help="parent-mode case selection (default: both cases)",
    )
    parser.add_argument("--_arm-case", choices=("real74", "wk82"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--_arm-root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("real74_steps", "wk82_steps"):
        if getattr(args, name) < 2:
            parser.error(f"--{name.replace('_', '-')} must be >= 2")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be >= 0")
    if args._arm_case:
        if args._arm_root is None:
            parser.error("internal arm execution requires --_arm-root")
    elif args.baseline is None or args.candidate is None:
        parser.error("--baseline and --candidate are required")
    return args


def _git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "--short=12", "HEAD"],
        text=True,
    ).strip()


def _build_case(case: str, total_steps: int):
    if case == "real74":
        from gpuwm.verify.cases.real74_d01 import (
            phase3_config,
            phase3_integration_config,
            prepare_phase3_case,
        )

        # Match the profiled full-physics-minus-radiation configuration.  The
        # configured 60 s clock and every other real74 physics choice remain.
        configured = replace(
            phase3_config(), ra_physics=0,
            run_seconds=total_steps * phase3_config().dt,
        )
        prepared = prepare_phase3_case(configured)
        cfg = phase3_integration_config(prepared.cfg)
        state = prepared.initial_result.state
        state.physics.bldt_seconds = cfg.dt
        state.physics.stepbl = 1
        return state, cfg

    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.verify.cases import wk82

    base_cfg = wk82.default_config()
    cfg = replace(
        base_cfg, mp_physics=10,
        run_seconds=total_steps * base_cfg.dt,
    )
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(
        coord, lambda z: wk82.wk82_sounding(z)[0],
        p_surf=cfg.p_surf, ztop=cfg.ztop,
    )
    return wk82.build(cfg, coord, base), cfg


def _digest(array) -> tuple[str, bytes]:
    import cupy as cp

    host = np.ascontiguousarray(cp.asnumpy(array))
    if host.dtype != np.float32:
        raise TypeError(f"expected FP32 output, got {host.dtype}")
    payload = (
        host.dtype.str.encode("ascii")
        + np.asarray(host.shape, dtype=np.int64).tobytes()
        + host.tobytes(order="C")
    )
    return hashlib.sha256(payload).hexdigest(), payload


def _hash_outputs(state) -> tuple[str, dict[str, str]]:
    import cupy as cp

    arrays = {}
    for name in _DRY_PROGNOSTICS + _MORRISON_OUTPUTS:
        value = getattr(state, name, None)
        if isinstance(value, cp.ndarray):
            arrays[f"state.{name}"] = value
    # The canonical idealized WK82 run deliberately has no non-timesplit
    # PhysicsDriver.  Morrison owns its surface accumulators in DomainState's
    # named scratch pool (wk82.run reads mp_rainnc the same way).  Require the
    # Morrison-only snow/graupel slots to exist rather than allocating them
    # here, so a skipped/wrong microphysics path cannot pass with zero hashes.
    diagnostics = (state.physics.microphysics
                   if state.physics is not None else None)
    for name in _SURFACE_OUTPUTS:
        if diagnostics is None:
            value = state._scratch.get(_MORRISON_SURFACE_SCRATCH[name])
        else:
            value = getattr(diagnostics, name, None)
        if isinstance(value, cp.ndarray):
            arrays[f"microphysics.{name}"] = value

    missing = [f"state.{name}" for name in _MORRISON_OUTPUTS
               if f"state.{name}" not in arrays]
    missing += [f"microphysics.{name}" for name in _SURFACE_OUTPUTS
                if f"microphysics.{name}" not in arrays]
    if missing:
        raise RuntimeError("missing Morrison outputs: " + ", ".join(missing))

    combined = hashlib.sha256()
    hashes = {}
    for name, array in sorted(arrays.items()):
        digest, payload = _digest(array)
        hashes[name] = digest
        combined.update(name.encode("ascii"))
        combined.update(payload)
    return combined.hexdigest(), hashes


def _run_arm(args: argparse.Namespace) -> None:
    root = args._arm_root.resolve()
    import gpuwm

    imported = Path(gpuwm.__file__).resolve()
    try:
        imported.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            f"arm imported gpuwm from {imported}, outside pinned root {root}"
        ) from exc

    import cupy as cp
    from gpuwm.core.dycore import step

    timed_steps = (args.real74_steps if args._arm_case == "real74"
                   else args.wk82_steps)
    state, cfg = _build_case(
        args._arm_case, args.warmup_steps + timed_steps)
    for _ in range(args.warmup_steps):
        step(state, cfg)
    cp.cuda.runtime.deviceSynchronize()

    started = time.perf_counter()
    for _ in range(timed_steps):
        step(state, cfg)
    cp.cuda.runtime.deviceSynchronize()
    seconds = time.perf_counter() - started
    combined, hashes = _hash_outputs(state)
    result = {
        "case": args._arm_case,
        "root": str(root),
        "head": _git_head(root),
        "warmup_steps": args.warmup_steps,
        "timed_steps": timed_steps,
        "dt_seconds": float(cfg.dt),
        "segment_seconds": seconds,
        "seconds_per_step": seconds / timed_steps,
        "combined_sha256": combined,
        "hashes": hashes,
    }
    print(_RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)


def _validate_root(root: Path, label: str) -> Path:
    root = root.resolve()
    if not (root / "gpuwm").is_dir() or not (root / ".git").exists():
        raise SystemExit(f"{label} is not a gpuwm checkout: {root}")
    return root


def _launch_arm(root: Path, case: str, args: argparse.Namespace,
                neutral_cwd: Path) -> dict[str, object]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    # numpy/cupy are installed in user site-packages on the controller.
    # Checkout isolation comes from the pinned PYTHONPATH plus _run_arm's
    # imported-path verification, not from disabling the user site.
    env.pop("PYTHONNOUSERSITE", None)
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--_arm-case", case,
        "--_arm-root", str(root),
        "--real74-steps", str(args.real74_steps),
        "--wk82-steps", str(args.wk82_steps),
        "--warmup-steps", str(args.warmup_steps),
    ]
    completed = subprocess.run(
        command, cwd=neutral_cwd, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if completed.returncode:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise SystemExit(
            f"{case} arm {root} failed with exit {completed.returncode}")
    lines = [line for line in completed.stdout.splitlines()
             if line.startswith(_RESULT_PREFIX)]
    if len(lines) != 1:
        sys.stderr.write(completed.stdout)
        raise SystemExit(f"{case} arm {root} emitted {len(lines)} results")
    return json.loads(lines[0][len(_RESULT_PREFIX):])


def _assert_equal(case: str, baseline: dict[str, object],
                  candidate: dict[str, object]) -> None:
    left = baseline["hashes"]
    right = candidate["hashes"]
    if left != right:
        names = sorted(set(left) | set(right))
        mismatches = [name for name in names
                      if left.get(name) != right.get(name)]
        for name in mismatches:
            print(
                f"MISMATCH case={case} field={name} "
                f"baseline={left.get(name)} candidate={right.get(name)}",
                file=sys.stderr,
            )
        raise SystemExit(
            f"BITWISE_MISMATCH case={case} fields={len(mismatches)}")
    if baseline["combined_sha256"] != candidate["combined_sha256"]:
        raise SystemExit(f"combined hash mismatch despite equal fields: {case}")


def _print_arm(label: str, result: dict[str, object]) -> None:
    print(
        f"ARM case={result['case']} label={label} head={result['head']} "
        f"warmup={result['warmup_steps']} steps={result['timed_steps']} "
        f"dt={result['dt_seconds']:.9g} "
        f"segment_seconds={result['segment_seconds']:.9f} "
        f"seconds_per_step={result['seconds_per_step']:.9f} "
        f"combined_sha256={result['combined_sha256']}"
    )


def _run_parent(args: argparse.Namespace) -> None:
    baseline_root = _validate_root(args.baseline, "baseline")
    candidate_root = _validate_root(args.candidate, "candidate")
    if baseline_root == candidate_root:
        raise SystemExit("baseline and candidate must be different checkouts")

    with tempfile.TemporaryDirectory(prefix="gpuwm-ab-mp-") as temp:
        neutral = Path(temp)
        cases = (("real74", "wk82") if args.case == "all"
                 else (args.case,))
        for case in cases:
            baseline = _launch_arm(baseline_root, case, args, neutral)
            candidate = _launch_arm(candidate_root, case, args, neutral)
            _print_arm("baseline", baseline)
            _print_arm("candidate", candidate)
            _assert_equal(case, baseline, candidate)
            speedup = (baseline["seconds_per_step"]
                       / candidate["seconds_per_step"])
            print(
                f"BITWISE_EQUAL case={case} fields={len(baseline['hashes'])} "
                f"combined_sha256={baseline['combined_sha256']}"
            )
            print(f"SPEEDUP case={case} baseline_over_candidate={speedup:.6f}")


def main() -> None:
    args = _arguments()
    if args._arm_case:
        _run_arm(args)
    else:
        _run_parent(args)


if __name__ == "__main__":
    main()
