"""Exact merged metric receipts and source-isolated oracle-change controls.

Controls replace only the named default-on oracle changes. They are diagnostic
source assemblies, not alternate supported model modes. Original metric words
and checkpoint payloads stay intact.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from tools.wrf_diffopt1_oracle.model_case import configuration, model_state, word_arrays


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install_controls(source_dir, level):
    """Restore successive default source groups without changing inputs."""
    import gpuwm.core.dycore as dycore
    import gpuwm.core.kernels as kernels

    selected = {}
    functions = []
    if level >= 1:
        # Open, unforced lateral boundaries route rhs_ph through the compiled
        # slow_geopotential_open kernel (WRF's outer-row upwinding); the
        # control restores the launchers that did not route there.
        functions.extend(("_launch_slow_geopotential", "_launch_slow_geopotential_faces"))
    if level >= 2:
        selected.update({name: source_dir / (name + ".cu")
                         for name in ("smag2d", "diff6", "diff6_seam")})
        functions.extend(("_compute_wrf_smag_tendencies", "launch_wrf_smag2d_vertical",
                          "launch_diff6", "_launch_diff6_seam", "prepare_fixed_tendencies",
                          "_couple_dry_mixing_map_factor"))
    if level >= 3:
        selected["dycore"] = source_dir / "dycore.cu"
    if level >= 4:
        selected.update({name: source_dir / (name + ".cu")
                         for name in ("advection", "openbc")})
        functions.append("apply_open_radiative_bc")
    if level >= 5:
        selected["acoustic"] = source_dir / "acoustic.cu"

    original = kernels.module_source
    original_defined = kernels.module_source_int_defines

    def module_source(name):
        if name not in selected:
            return original(name)
        return (kernels._preamble() + kernels._extra_header_text(name)
                + selected[name].read_text(encoding="utf-8"))

    def defined_source(name, defines, *, prefix=None):
        if name not in selected:
            return original_defined(name, defines, prefix=prefix)
        if prefix is None:
            prefix = "\n".join(f"#define {key} {value}" for key, value in defines)
        return (kernels._preamble() + kernels._extra_header_text(name) + prefix + "\n"
                + selected[name].read_text(encoding="utf-8"))

    kernels.module_source = module_source
    kernels.module_source_int_defines = defined_source
    if functions:
        text = (source_dir / "dycore.py").read_text(encoding="utf-8")
        tree = ast.parse(text)
        body = [node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in functions]
        assert {node.name for node in body} == set(functions)
        exec(compile(ast.Module(body=body, type_ignores=[]),
                     "pre-oracle-diffusion-control", "exec"), dycore.__dict__)
    receipt={name: sha256(path) for name, path in selected.items()}
    if functions:
        receipt["dycore.py"] = sha256(source_dir / "dycore.py")
    return receipt


def checkpoint_payload_state(path, cfg):
    """Seed old state bytes directly, independently of restore validation."""
    import cupy as cp
    from gpuwm.io.restart import read_restart_header
    state = model_state(cfg)
    header = read_restart_header(path)
    with np.load(path) as stored:
        for key in stored.files:
            if key.startswith(("state/", "acoustic/")):
                name = key.split("/", 1)[1]
                getattr(state, name)[...] = cp.asarray(stored[key])
            elif key.startswith("scratch/"):
                slot = key.split("/", 1)[1]
                state.scratch(stored[key].shape, slot)[...] = cp.asarray(stored[key])
    state.elapsed_seconds = float(header["elapsed_seconds"])
    return state


#: The metric checkpoints the resume regression continues, written by the
#: build under test.  The a4177ebbf checkpoints (diff2-legacy-k*.npz) stay
#: byte for byte as the 2.8.7 attribution's inputs; from 2.8.8 a restore
#: refuses them before restore because they record no dycore mixing
#: identity, so the exact-resume claim needs checkpoints this build wrote.
CURRENT_CHECKPOINT = "diff2-288-legacy-k{km}.npz"
CURRENT_CHECKPOINT_RECEIPT = "diff2-288-legacy-checkpoints.json"
CHECKPOINT_STEPS = 2


def checkpoint_configuration(km):
    """The default-on metric form (diff_opt=2, mix_full_fields) as shipped."""
    return configuration(km=km, diff=2, mix=True)


def record_checkpoints(data, engine_commit):
    """Integrate and write the current-build metric checkpoints into ``data``.

    The same synthetic case and step count as the a4177ebbf originals
    (2 steps of 0.5 s), under this build's defaults, then the ordinary
    production writer.  The receipt seals each file and names the build
    and card that integrated it.
    """
    import cupy as cp
    from gpuwm.checkpoint_identity import DYCORE_MIXING_ALGORITHM_IDENTITY
    from gpuwm.core.dycore import step
    from gpuwm.io.restart import write_restart
    import gpuwm.wrf_exact as exact
    assert not exact.ENABLED, "Record the default production arithmetic"
    checkpoints = {}
    for km in (2, 4):
        cfg = checkpoint_configuration(km)
        state = model_state(cfg)
        for _ in range(CHECKPOINT_STEPS):
            step(state, cfg)
        assert state.elapsed_seconds == CHECKPOINT_STEPS * cfg.dt
        path = write_restart(data / CURRENT_CHECKPOINT.format(km=km), state, cfg)
        checkpoints[f"k{km}"] = sha256(path)
    receipt = dict(
        what=("Metric (diff_opt=2, mix_full_fields) checkpoints integrated "
              f"{CHECKPOINT_STEPS} steps and written by the build under test, "
              "for the exact-resume regression"),
        why=("2.8.8 records the dycore mixing identity in every checkpoint of "
             "a mixing run and refuses, before restore, one that lacks it. "
             "The a4177ebbf checkpoints lack it and stay unchanged as the "
             "2.8.7 attribution's inputs and as the refusal witness."),
        recipe=("python -m tools.wrf_diffopt1_oracle.merged_metric_capture "
                "FIXTURE_DIRECTORY --record-checkpoints ENGINE_COMMIT"),
        engine_commit=engine_commit,
        device=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        steps=CHECKPOINT_STEPS,
        dycore_mixing=DYCORE_MIXING_ALGORITHM_IDENTITY,
        checkpoints=checkpoints)
    (data / CURRENT_CHECKPOINT_RECEIPT).write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checkpoints))


def capture(data, output, level=0, source_dir=None):
    import cupy as cp
    from gpuwm.core.dycore import step
    from gpuwm.io.restart import restore_restart
    import gpuwm.wrf_exact as exact
    assert not exact.ENABLED, "Capture the default production arithmetic"
    controls = install_controls(source_dir, level) if level else {}
    arrays = {}
    for km in (2, 4):
        for boundary in (False, True):
            for moist in (False, True):
                # Preserve the sealed pre-CQ trajectory, as the merged
                # baseline test runs it; the default CQ coupling is a later
                # change this receipt does not attribute.
                cfg = replace(configuration(km=km, diff=2, mix=True,
                                            boundary=boundary, moist=moist), moist_cq=False)
                state = model_state(cfg)
                for _ in range(3):
                    step(state, cfg)
                name = f"diff2_k{km}_b{int(boundary)}_m{int(moist)}"
                arrays.update({name + "_" + field: value
                               for field, value in word_arrays(state).items()})
        cfg = replace(configuration(km=km, diff=2, mix=True), moist_cq=False)
        checkpoint = data / f"diff2-legacy-k{km}.npz"
        straight = checkpoint_payload_state(checkpoint, cfg)
        resumed = model_state(cfg)
        restore_restart(checkpoint, resumed, cfg)
        for _ in range(2):
            step(straight, cfg)
            step(resumed, cfg)
        resumed_words = word_arrays(resumed)
        for field, value in word_arrays(straight).items():
            np.testing.assert_array_equal(value.view("u4"),
                                          resumed_words[field].view("u4"), err_msg=field)
            arrays[f"legacy_k{km}_" + field] = value
    np.savez_compressed(output.with_suffix(".npz"), **arrays)
    sources = {
        str(path.relative_to(Path(__file__).parents[2])): sha256(path)
        for path in (Path(__file__).parents[2] / "gpuwm/core").rglob("*")
        if path.suffix in (".py", ".cu", ".cuh")
    }
    receipt = dict(control_level=level, control_sources=controls,
                   arrays=len(arrays), words=sum(value.size for value in arrays.values()),
                   device=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                   archive_sha256=sha256(output.with_suffix(".npz")),
                   legacy_checkpoints={f"k{km}": sha256(data / f"diff2-legacy-k{km}.npz")
                                       for km in (2, 4)}, runtime_sources=sources)
    output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n",
                                           encoding="utf-8")
    print(json.dumps({key: receipt[key] for key in ("control_level", "arrays", "words")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--control-level", type=int, choices=range(6), default=0)
    parser.add_argument("--control-source", type=Path)
    parser.add_argument("--record-checkpoints", metavar="ENGINE_COMMIT",
                        help="write the current-build metric checkpoints into DATA")
    args = parser.parse_args()
    if args.record_checkpoints:
        record_checkpoints(args.data, args.record_checkpoints)
    elif args.output is None:
        parser.error("OUTPUT is required unless --record-checkpoints is given")
    else:
        capture(args.data, args.output, args.control_level, args.control_source)
