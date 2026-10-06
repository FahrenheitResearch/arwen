"""State hash of a short run with no radar forcing attached.

Run the same command against two trees (the integration tip and this
branch) and compare the digests: with no forcing the microphysics entry
point must leave every byte as it was.

    PYTHONPATH=<tree> python default_path_hash.py --out <json> [--mp 1 8]

The run is the WK82 supercell start of ``gpuwm.verify.cases.wk82`` on a
small grid, open boundaries, ``--steps`` steps per microphysics scheme, and
then the same start on a specified-boundary configuration so the
microphysics ring guard is on the path as well.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

FIELDS = ("u", "v", "w", "ph", "php", "thp", "mup", "p", "alt", "qv", "qc",
          "qr", "qi", "qs", "qg", "ni", "nr", "h_diabatic")


def _run(mp, steps, *, specified):
    import cupy as cp
    from gpuwm.config import validate_run_config
    from gpuwm.core.dycore import step
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.verify.cases import wk82

    changes = dict(nx=48, ny=48, nz=40, run_seconds=steps * 6.0,
                   mp_physics=mp)
    cfg = validate_run_config(dataclasses.replace(wk82.default_config(),
                                                  **changes))
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: wk82.wk82_sounding(z)[0],
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = wk82.build(cfg, coord, base)
    run_cfg = cfg
    if specified:
        # The ring guard reads cfg.specified and cfg.spec_zone; the dycore
        # still integrates the open configuration.
        run_cfg = dataclasses.replace(cfg, specified=False)
    from gpuwm.core import microphysics

    digest = hashlib.sha256()
    for _ in range(steps):
        step(state, run_cfg)
        if specified:
            ring_cfg = dataclasses.replace(cfg, specified=True, spec_zone=1,
                                           open_x=False, open_y=False)
            microphysics.apply(state, ring_cfg, cfg.dt)
    cp.cuda.Stream.null.synchronize()
    per_field = {}
    for name in FIELDS:
        value = getattr(state, name, None)
        if value is None:
            continue
        data = cp.asnumpy(value).tobytes()
        per_field[name] = hashlib.sha256(data).hexdigest()
        digest.update(name.encode())
        digest.update(data)
    for slot in ("mp_rainnc", "mp_rainncv"):
        value = state.existing_scratch(slot)
        if value is not None:
            data = cp.asnumpy(value).tobytes()
            per_field[slot] = hashlib.sha256(data).hexdigest()
            digest.update(slot.encode())
            digest.update(data)
    return {"mp_physics": mp, "steps": steps, "specified_ring": specified,
            "max_w": float(state.w.max()), "max_qr": float(state.qr.max()),
            "sha256": digest.hexdigest(), "fields": per_field}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mp", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()
    import gpuwm

    runs = []
    for mp in args.mp:
        for specified in (False, True):
            try:
                runs.append(_run(mp, args.steps, specified=specified))
            except Exception as exc:          # recorded, not hidden
                runs.append({"mp_physics": mp, "specified_ring": specified,
                             "error": f"{type(exc).__name__}: {exc}"})
            print(json.dumps({k: v for k, v in runs[-1].items()
                              if k != "fields"}), flush=True)
    args.out.write_text(json.dumps({"tree": str(Path(gpuwm.__file__).parents[1]),
                                    "runs": runs}, indent=1))


if __name__ == "__main__":
    main()
