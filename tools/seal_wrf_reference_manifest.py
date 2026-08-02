"""Seal a WRF reference manifest from a measurement taken on the compute node.

    python tools/seal_wrf_reference_manifest.py \
        --capture evidence/<lane>/wrf-reference-measurement-<date>.txt \
        --config configs/<the configuration>.toml

A manifest is a receipt, so nothing here invents a digest.  Every hash the
manifest carries either comes out of the capture -- a file written by hashing
bytes on the machine that holds them -- or is taken from a document committed
beside the manifest.  Re-running this command on an unchanged tree rewrites the
manifest byte for byte, which is what makes a hand-edited manifest detectable
by the tool that made it.

The tool refuses rather than guesses:

* if the committed namelists do not hash to what the node measured, the
  committed text is not the text that ran, and that is a corruption of the
  receipt, not a formatting difference;
* if any of the four required hash groups would come out absent, it refuses --
  the same condition ``gpuwm certify`` refuses on, caught before the manifest
  is written rather than after it is committed.

## The capture format

A capture is plain text so that the person who ran it can read what they
signed.  Lines outside a section, and lines beginning with ``#``, are prose.
A ``[section]`` header opens a section; inside one, ``key=value`` lines carry
the measurements.  Sections read here:

``[wrf_exe]``
    ``wrf.exe=<digest>`` -- the executable that produced the reference stream.

``[namelists]``
    ``<file name>=<digest>`` for every namelist the reference run consumed.
    A key carrying a dot-suffixed qualifier (``namelist.input.config_copy``)
    is a corroborating second reading of another entry, not a namelist of its
    own, and is checked against its base entry rather than published.

``[reference_wrfout_frames]``
    ``<frame name>=<digest>  bytes=<size>`` for every reference frame the
    comparison scored.  The size is recorded for the reader; the manifest
    publishes the digest.

Top-level ``key=value`` lines carry ``wrf_version`` and ``wrf_commit``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpuwm.certify.band import (  # noqa: E402
    dumps_certification_json, sha256_file, write_certification_json)
from gpuwm.certify.wrf_reference import (  # noqa: E402
    MANIFEST_DIR_NAME, MANIFEST_SCHEMA_ID, WrfReferenceError,
    absent_reference_hashes, validate_wrf_reference_manifest)

#: Suffixes of the documents a sealed manifest commits beside itself.
BUILD_RECIPE_SUFFIX = ".build-recipe.md"
NAMELIST_SUFFIX_TEMPLATE = ".{name}"

#: Capture sections this tool reads.
SECTION_EXE = "wrf_exe"
SECTION_NAMELISTS = "namelists"
SECTION_FRAMES = "reference_wrfout_frames"

#: Namelist keys that are a second reading of another key, not a namelist.
#: The suffix names what the corroborating copy was, so it is checked against
#: the base key and then dropped.
CORROBORATION_SUFFIXES = ("config_copy",)


class CaptureError(ValueError):
    """A measurement capture does not carry what the manifest needs."""


def parse_capture(path: str | Path) -> dict[str, Any]:
    """Read a capture into ``{'top': {...}, 'sections': {name: {...}}}``."""
    top: dict[str, str] = {}
    sections: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = sections.setdefault(line[1:-1], {})
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        target = top if current is None else current
        target[key.strip()] = value.strip()
    return {"top": top, "sections": sections}


def _digest_only(value: str) -> str:
    """The digest at the head of a capture value, dropping trailing fields."""
    return value.split()[0] if value.split() else ""


def _fold_corroborations(entries: dict[str, str]) -> dict[str, str]:
    """Check every corroborating reading against its base, then drop it."""
    folded = {key: _digest_only(value) for key, value in entries.items()
              if not key.endswith(CORROBORATION_SUFFIXES)}
    for key, value in entries.items():
        for suffix in CORROBORATION_SUFFIXES:
            if not key.endswith("." + suffix):
                continue
            base = key[: -len("." + suffix)]
            if base not in folded:
                raise CaptureError(
                    f"capture carries {key} but not the {base} it corroborates")
            if folded[base] != _digest_only(value):
                raise CaptureError(
                    f"capture disagrees with itself on {base}: "
                    f"{folded[base]} vs {_digest_only(value)} from {key}")
    return folded


def manifest_path_for_config(config_sha256: str, *,
                             directory: str | Path | None = None) -> Path:
    """Where the manifest for a configuration lives.  Identity, never a name."""
    root = (Path(directory) if directory is not None
            else REPO_ROOT / MANIFEST_DIR_NAME)
    return root / f"{config_sha256}.manifest.json"


def seal(capture_path: Path, config_sha256: str, *,
         manifest_dir: Path | None = None) -> tuple[Path, dict[str, Any]]:
    """Build the manifest for one configuration and write it."""
    capture = parse_capture(capture_path)
    top, sections = capture["top"], capture["sections"]
    directory = (Path(manifest_dir) if manifest_dir is not None
                 else REPO_ROOT / MANIFEST_DIR_NAME)

    for required in (SECTION_EXE, SECTION_NAMELISTS, SECTION_FRAMES):
        if required not in sections:
            raise CaptureError(
                f"{capture_path} carries no [{required}] section")

    exe = _digest_only(sections[SECTION_EXE].get("wrf.exe", ""))
    if not exe:
        raise CaptureError(
            f"{capture_path} [{SECTION_EXE}] carries no wrf.exe digest")

    measured_namelists = _fold_corroborations(sections[SECTION_NAMELISTS])
    if not measured_namelists:
        raise CaptureError(f"{capture_path} measured no namelists")

    # The committed namelist must BE the measured namelist.  Anything else --
    # a newline translated on the way into the repository, a stray edit -- and
    # the receipt would describe text that did not run.
    namelists: dict[str, str] = {}
    for name in sorted(measured_namelists):
        committed = directory / f"{config_sha256}{NAMELIST_SUFFIX_TEMPLATE.format(name=name)}"
        if not committed.is_file():
            raise WrfReferenceError(
                f"the capture measured {name} but no copy is committed at "
                f"{committed.relative_to(REPO_ROOT).as_posix()}")
        committed_digest = sha256_file(committed)
        if committed_digest != measured_namelists[name]:
            raise WrfReferenceError(
                f"committed {committed.name} hashes {committed_digest}, but "
                f"the node measured {measured_namelists[name]} for {name}; "
                "the committed text is not the text the reference run "
                "consumed")
        namelists[name] = committed_digest

    recipe = directory / f"{config_sha256}{BUILD_RECIPE_SUFFIX}"
    if not recipe.is_file():
        raise WrfReferenceError(
            f"no build recipe committed at "
            f"{recipe.relative_to(REPO_ROOT).as_posix()}")

    frames = {name: _digest_only(value)
              for name, value in sections[SECTION_FRAMES].items()}
    if not frames:
        raise CaptureError(f"{capture_path} measured no reference frames")

    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA_ID,
        "wrf_version": top.get("wrf_version", ""),
        "wrf_commit": top.get("wrf_commit") or None,
        "config_sha256": config_sha256,
        "wrf_exe_sha256": exe,
        "build_recipe_sha256": sha256_file(recipe),
        "namelist_sha256": namelists,
        "reference_wrfout_sha256": frames,
        "build_recipe": recipe.name,
        "namelists": {name: (directory / f"{config_sha256}"
                             f"{NAMELIST_SUFFIX_TEMPLATE.format(name=name)}"
                             ).name
                      for name in namelists},
    }
    validate_wrf_reference_manifest(manifest)
    absent = absent_reference_hashes(manifest)
    if absent:
        raise WrfReferenceError(
            "a sealed manifest would still be refused by certify; absent "
            f"hash groups: {', '.join(absent)}")

    path = manifest_path_for_config(config_sha256, directory=directory)
    write_certification_json(path, manifest)
    return path, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True, metavar="TXT",
                        help="measurement capture written on the node that "
                             "holds the reference bytes")
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--config", type=Path, metavar="TOML",
                          help="the configuration this reference is the "
                               "counterpart of; its SHA-256 is the identity")
    identity.add_argument("--config-sha256", metavar="SHA256",
                          help="that identity directly, when the "
                               "configuration file is not to hand")
    parser.add_argument("--manifest-dir", type=Path, default=None,
                        metavar="DIR")
    args = parser.parse_args(argv)

    config_sha256 = (args.config_sha256 if args.config_sha256
                     else sha256_file(args.config))
    path, manifest = seal(args.capture, config_sha256,
                          manifest_dir=args.manifest_dir)
    print(f"manifest: {path}")
    print(f"exe:      {manifest['wrf_exe_sha256']}")
    print(f"recipe:   {manifest['build_recipe_sha256']}")
    print(f"namelists:{len(manifest['namelist_sha256'])}  "
          f"frames: {len(manifest['reference_wrfout_sha256'])}")
    print(f"bytes:    {len(dumps_certification_json(manifest))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
