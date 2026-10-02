"""Run the declared release Stage 1 and public publication controls.

The public snapshot excludes private campaign inputs. Collecting directories
there therefore runs suites whose inputs are absent. The release's maintained
Stage 1 manifest defines the public CPU leg, with its existing census guards.
Linux runs that complete list with its native artifacts. Windows partitions
only the annotated native-dependent files into Linux, while retaining the
publication workflow's parsed files and these selection controls. The separate
Windows native Rust job still runs. No scientific suite list is copied into
the workflow, and no native-dependent file loses its Linux selection.

Paths are passed directly to pytest, avoiding Windows CRLF shell translation.
A missing or duplicate entry refuses before collection, naming lost coverage.
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
MANIFEST = "tools/battery/stage1_files.txt"
NATIVE_MANIFEST = "tools/battery/stage1_native_files.txt"
# These controls keep the public job coupled to the maintained release list.
CONTRACT_FILES = ("tests/test_ci_workflow.py", "tests/test_public_ci_contract.py",
                  # CPU-only SASE budgeting must never import the CuPy launcher.
                  "tests/test_sase_preflight_no_cupy.py")


def listed_files(root: pathlib.Path = ROOT, manifest: str = MANIFEST) -> list[str]:
    manifest_path = pathlib.PurePosixPath(manifest)
    if manifest_path.is_absolute() or ".." in manifest_path.parts:
        raise ValueError("Stage 1 manifest must be inside the selected checkout: " + manifest)
    text = (root / manifest).read_text(encoding="utf-8")
    files = [line.strip() for line in text.splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    duplicates = sorted({name for name in files if files.count(name) > 1})
    if duplicates:
        raise ValueError("Stage 1 duplicates coverage entries: " + ", ".join(duplicates))
    for name in files:
        path = pathlib.PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or not name.endswith(".py"):
            raise ValueError("Stage 1 names a non-repository test path: " + name)
    return files


def publication_files(root: pathlib.Path = ROOT) -> list[str]:
    # Import only the existing workflow parser. Its replay entry point, which
    # provisions a checkout, is never called by this selection runner.
    import importlib.util
    spec = importlib.util.spec_from_file_location("stage1_publication_parser", ROOT / "tools/ci_test_replay.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _marker, files = module.parse_test_job((root / ".github/workflows/publish.yml").read_text(encoding="utf-8"))
    return files


def native_files(root: pathlib.Path = ROOT, manifest: str = NATIVE_MANIFEST) -> dict[str, str]:
    """The explicit platform partition and each native artifact dependency."""
    path = pathlib.PurePosixPath(manifest)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("native partition must be inside the selected checkout: " + manifest)
    entries = {}
    for raw in (root / manifest).read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        filename, separator, reason = raw.partition("#")
        filename, reason = filename.strip(), reason.strip()
        path = pathlib.PurePosixPath(filename)
        if not separator or not reason:
            raise ValueError("native partition entry names no required artifact: " + filename)
        if path.is_absolute() or ".." in path.parts or not filename.endswith(".py"):
            raise ValueError("native partition names a non-repository test path: " + filename)
        if filename in entries:
            raise ValueError("native partition duplicates coverage entry: " + filename)
        entries[filename] = reason
    return entries


def selected_files(root: pathlib.Path = ROOT, manifest: str = MANIFEST,
                   native_manifest: str = NATIVE_MANIFEST, platform: str = "linux") -> list[str]:
    stage1 = listed_files(root, manifest)
    native = native_files(root, native_manifest)
    outside = set(native) - set(stage1)
    if outside:
        raise ValueError("native partition removes files that have no maintained Linux Stage 1 coverage: "
                         + ", ".join(sorted(outside)))
    if platform not in ("linux", "windows"):
        raise ValueError("unknown Stage 1 platform: " + platform)
    selected = stage1 if platform == "linux" else [name for name in stage1 if name not in native]
    return list(dict.fromkeys([*selected, *publication_files(root), *CONTRACT_FILES]))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough = []
    if "--" in argv:
        split = argv.index("--")
        argv, passthrough = argv[:split], argv[split + 1:]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--manifest", default=MANIFEST, help="repository-relative release Stage 1 manifest")
    parser.add_argument("--native-manifest", default=NATIVE_MANIFEST,
                        help="annotated Stage 1 native dependency partition")
    parser.add_argument("--platform", choices=("linux", "windows"), default="linux", type=str.lower)
    parser.add_argument("--minimum", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        stage1 = listed_files(args.root, args.manifest)
        files = selected_files(args.root, args.manifest, args.native_manifest, args.platform)
        if len(stage1) < args.minimum:
            raise ValueError(f"Stage 1 lists {len(stage1)} files, fewer than the {args.minimum} this leg requires")
        missing = [name for name in files if not (args.root / name).is_file()]
        if missing:
            raise ValueError("Stage 1/publication coverage would run zero tests because files are missing: "
                             + ", ".join(missing))
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(f"Stage 1: {len(stage1)} declared files; {len(files)} selected on {args.platform} "
          "with publication and CI controls", flush=True)
    return subprocess.call([sys.executable, "-m", "pytest", *passthrough, *files], cwd=args.root)


if __name__ == "__main__":
    raise SystemExit(main())
