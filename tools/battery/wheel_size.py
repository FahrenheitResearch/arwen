"""Measure platform wheels before release gates, using the release ZIP profile."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

MAX_BYTES = 100_000_000


def measure(paths, limit=MAX_BYTES):
    rows = []
    for path in paths:
        path = Path(path)
        size = path.stat().st_size
        rows.append({"path": str(path), "bytes": size})
        if size > limit:
            raise ValueError(f"{path.name} is {size} bytes, above the {limit}-byte per-file limit")
    if not rows:
        raise ValueError("wheel size check found no wheels to measure")
    return rows


def pre_gate(root, output, *, build_natives=False):
    """Build a measurement wheel from this source's own stamped artifacts.

    This does not qualify pins for publication. The cut generates final pins
    after the gate; this earlier build uses the documented development override
    and records that fact. Native stamps must still name this exact revision.
    """
    from gpuwm import bridge_assets
    from tools import stage_wheel_bridges
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    platform = bridge_assets.host_platform()
    if platform is None:
        raise ValueError("wheel size pre-gate requires a supported native platform")
    environment = dict(os.environ, GPUWM_BRIDGE_SOURCE_REV=revision,
                       CARGO_PROFILE_RELEASE_LTO="fat", GPUWM_NO_LOCAL_GPU="1",
                       CUDA_VISIBLE_DEVICES="-1", GPUWM_ALLOW_UNPINNED_WHEEL="1")
    # Do not inherit the two rejected optimizations or another profile.
    for key in list(environment):
        if key.startswith("CARGO_PROFILE_") and key != "CARGO_PROFILE_RELEASE_LTO":
            environment.pop(key)
    if build_natives:
        environment.pop("CARGO_TARGET_DIR", None)
        for crate in dict.fromkeys(a.crate for a in bridge_assets.wheel_artifacts()):
            command = ["cargo", "build", "--release", "--locked", "--offline"]
            with (output / (Path(crate).name + ".log")).open("w") as log:
                subprocess.run(command, cwd=root / crate, env=environment,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
    # Staging and the revision checks use the selected checkout, never a cached
    # installed distribution or another worktree's target directory.
    stage_wheel_bridges._REPO_ROOT = root
    for artifact in bridge_assets.wheel_artifacts():
        path = stage_wheel_bridges.source_path(artifact, platform)
        if not path.is_file():
            raise ValueError(f"wheel size pre-gate: {artifact.name} is not built at {path}; build this source's release artifacts first")
        if not artifact.vendored:
            bridge_assets.verify_source_revision(path.read_bytes(), expected=revision, label=artifact.name)
        bridge_assets.verify_contract_marker(artifact.name, path)
    stage_wheel_bridges.stage(platform, destination=root / "gpuwm/libexec/bridges")
    with (output / "wheel-build.log").open("w") as log:
        subprocess.run([sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(output)],
                       cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    rows = measure(sorted(output.glob("*.whl")))
    (output / "wheel-size.json").write_text(json.dumps({
        "source_revision": revision, "platform": platform, "limit_bytes": MAX_BYTES,
        "publication_pins_qualified": False, "wheels": rows}, indent=2) + "\n")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", nargs="*", type=Path)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--build-natives", action="store_true")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.build and (args.output is None or args.wheels):
        parser.error("--build requires --output and takes no wheel filenames")
    try:
        rows = pre_gate(args.root, args.output, build_natives=args.build_natives) if args.build else measure(args.wheels)
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"wheel size pre-gate refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
