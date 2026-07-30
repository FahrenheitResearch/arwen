"""Two-arm exact dynamics A/B for real74, Straka, and WK82.

The controller launches each checkout in a neutral temporary directory with
``PYTHONPATH`` replaced by that checkout.  Each arm builds identical cases,
runs short RK/acoustic segments, and hashes every prognostic plus the
diagnostics that feed the next step.  Any field mismatch is a hard failure.

Example (from any directory)::

    python C:\\path\\to\\candidate\\tools\\ab_dyn_step2.py \
      --baseline C:\\path\\to\\baseline --candidate C:\\path\\to\\candidate \
      --real74-steps 20 --straka-steps 20 --wk82-steps 5
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


_FIELDS = (
    "u", "v", "w", "thp", "php", "mup", "p", "al", "alt",
    "qv", "qc", "qr", "qi", "qs", "qg", "nc", "nr", "ni", "ns",
    "ng", "h_diabatic",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--real74-steps", type=int, default=2)
    parser.add_argument("--straka-steps", type=int, default=4)
    parser.add_argument("--wk82-steps", type=int, default=2)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repo", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.real74_steps, args.straka_steps, args.wk82_steps) < 1:
        parser.error("all step counts must be positive")
    if args.child:
        if args.repo is None:
            parser.error("--child requires --repo")
    elif args.baseline is None or args.candidate is None:
        parser.error("controller mode requires --baseline and --candidate")
    return args


def _array_digest(cp, array) -> str:
    import numpy as np

    host = np.ascontiguousarray(cp.asnumpy(array))
    if host.dtype != np.float32:
        raise TypeError(f"expected FP32 output, got {host.dtype}")
    digest = hashlib.sha256()
    digest.update(host.dtype.str.encode("ascii"))
    digest.update(np.asarray(host.shape, dtype=np.int64).tobytes())
    digest.update(host.tobytes(order="C"))
    return digest.hexdigest()


def _hash_state(cp, state) -> dict[str, str]:
    hashes = {}
    seen = set()
    for name in _FIELDS:
        array = getattr(state, name, None)
        if array is None or id(array) in seen:
            continue
        seen.add(id(array))
        hashes[f"state.{name}"] = _array_digest(cp, array)

    # Accumulated precipitation is an externally visible model output.
    rain = getattr(state, "_scratch", {}).get("mp_rainnc")
    if rain is not None and id(rain) not in seen:
        seen.add(id(rain))
        hashes["state.mp_rainnc"] = _array_digest(cp, rain)

    driver = getattr(state, "physics", None)
    if driver is not None:
        for owner_name, owner in (("physics", vars(driver)),
                                  ("physics.fields", driver.fields)):
            for name, array in sorted(owner.items()):
                if (not isinstance(array, cp.ndarray) or id(array) in seen
                        or array.dtype != cp.float32):
                    continue
                seen.add(id(array))
                hashes[f"{owner_name}.{name}"] = _array_digest(cp, array)
    return hashes


def _timed_steps(cp, state, cfg, steps: int) -> float:
    from gpuwm.core.dycore import step

    cp.cuda.runtime.deviceSynchronize()
    started = time.perf_counter()
    for _ in range(steps):
        step(state, cfg)
    cp.cuda.runtime.deviceSynchronize()
    return time.perf_counter() - started


def _run_real74(cp, steps: int):
    from gpuwm.verify.cases.real74_d01 import (
        phase3_config,
        phase3_integration_config,
        prepare_phase3_case,
    )

    prepared = prepare_phase3_case(phase3_config())
    cfg = phase3_integration_config(prepared.cfg)
    state = prepared.initial_result.state
    state.physics.bldt_seconds = cfg.dt
    state.physics.stepbl = 1
    elapsed = _timed_steps(cp, state, cfg, steps)
    return cfg, state, elapsed


def _run_straka(cp, steps: int):
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.verify.cases import straka

    cfg = replace(straka.default_config(), run_seconds=steps * 0.5)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(
        coord, straka.sounding, p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = straka.build(cfg, coord, base)
    elapsed = _timed_steps(cp, state, cfg, steps)
    return cfg, state, elapsed


def _run_wk82(cp, steps: int):
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.verify.cases import wk82

    cfg = replace(wk82.default_config(), run_seconds=steps * 6.0)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(
        coord, wk82.wk82_theta, p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = wk82.build(cfg, coord, base)
    elapsed = _timed_steps(cp, state, cfg, steps)
    return cfg, state, elapsed


def _child(args: argparse.Namespace) -> int:
    repo = args.repo.resolve()
    import cupy as cp
    import gpuwm

    imported = Path(gpuwm.__file__).resolve()
    if repo not in imported.parents:
        raise RuntimeError(f"gpuwm imported from {imported}, not {repo}")

    results = {}
    runners = (
        ("real74", _run_real74, args.real74_steps),
        ("straka", _run_straka, args.straka_steps),
        ("wk82", _run_wk82, args.wk82_steps),
    )
    for case, runner, steps in runners:
        cfg, state, wall = runner(cp, steps)
        results[case] = {
            "steps": steps,
            "dt": cfg.dt,
            "elapsed_seconds": state.elapsed_seconds,
            "wall_seconds": wall,
            "hashes": _hash_state(cp, state),
        }
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    print(json.dumps({"repo": str(repo), "head": head, "cases": results},
                     sort_keys=True))
    return 0


def _launch_arm(label: str, repo: Path, args: argparse.Namespace,
                neutral: Path) -> dict:
    repo = repo.resolve()
    if not (repo / "gpuwm").is_dir():
        raise ValueError(f"{label} is not a gpuwm checkout: {repo}")
    cwd = neutral / label
    cwd.mkdir()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    # numpy/cupy are installed in user site-packages on the controller.
    # Checkout isolation comes from the pinned PYTHONPATH plus _child's
    # imported-path verification, not from disabling the user site.
    env.pop("PYTHONNOUSERSITE", None)
    command = [
        sys.executable, str(Path(__file__).resolve()), "--child",
        "--repo", str(repo), "--real74-steps", str(args.real74_steps),
        "--straka-steps", str(args.straka_steps),
        "--wk82-steps", str(args.wk82_steps),
    ]
    completed = subprocess.run(
        command, cwd=cwd, env=env, text=True, capture_output=True)
    if completed.returncode:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise SystemExit(f"{label} arm failed with {completed.returncode}")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{label} arm produced no JSON")
    return json.loads(lines[-1])


def _controller(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="gpuwm-ab-dyn-step2-") as tmp:
        neutral = Path(tmp)
        baseline = _launch_arm("baseline", args.baseline, args, neutral)
        candidate = _launch_arm("candidate", args.candidate, args, neutral)

    mismatches = []
    for case in ("real74", "straka", "wk82"):
        left = baseline["cases"][case]
        right = candidate["cases"][case]
        for key in ("steps", "dt", "elapsed_seconds"):
            if left[key] != right[key]:
                mismatches.append(f"{case}.{key}: {left[key]} != {right[key]}")
        names = sorted(set(left["hashes"]) | set(right["hashes"]))
        for name in names:
            if left["hashes"].get(name) != right["hashes"].get(name):
                mismatches.append(
                    f"{case}.{name}: {left['hashes'].get(name)} != "
                    f"{right['hashes'].get(name)}")
        print(
            f"{case}: baseline={left['wall_seconds']:.6f}s "
            f"candidate={right['wall_seconds']:.6f}s "
            f"fields={len(names)}")

    print(f"baseline_head={baseline['head']}")
    print(f"candidate_head={candidate['head']}")
    if mismatches:
        print("EXACT_EQUALITY=FAIL")
        for mismatch in mismatches:
            print(mismatch)
        return 1
    print("EXACT_EQUALITY=PASS")
    return 0


def main() -> int:
    args = _arguments()
    return _child(args) if args.child else _controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
