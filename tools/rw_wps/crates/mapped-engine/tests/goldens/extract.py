"""Extract mapped-engine goldens by running the REAL Python engine.

The goldens beside this script are not hand-written expectations: each is
the output of `gpuwm.mapped_source.inspect_mapped_source` on REAL staged
agency bytes, reduced to the identities the Rust engine must reproduce
(per-field array sha256, shape, axes, missing count, source references,
the grid fingerprint, and the frame-header digests).  A golden that was
authored rather than measured would only prove the Rust engine agrees
with whoever wrote it.

Usage (from the repository root, with gpuwm importable):

    python tools/rw_wps/crates/mapped-engine/tests/goldens/extract.py \
        --case icon-eu

Each case names a mapping in the repository and an input set under the
staging tree.  Re-running is how a golden is refreshed; the file records
the exact command and the input digests so a stale golden is visible.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys


REPOSITORY = pathlib.Path(__file__).resolve().parents[6]
#: Composed from this account's home rather than spelled out, and for a
#: reason that is not style: a written-out default is one developer's
#: absolute path, and the release snapshot's machine-path gate refuses
#: to build a tree that ships one.  Same default as `goldens.rs`.
STAGING = pathlib.Path(
    os.environ.get("GPUWM_MODEL_GAUNTLET_STAGING")
    or (pathlib.Path.home() / "gpuwm-model-gauntlet-staging"))

#: Case name -> (mapping path, input directory, glob, cap on file count).
#:
#: The caps keep a golden small enough to read in a diff while still
#: exercising the whole path: every case below decodes real published
#: bytes, and the cap only limits HOW MANY of a producer's objects are
#: read, never which code runs.
CASES = {
    # DWD regular-lat-lon GRIB2, bzip2-wrapped: the acquisition codec, the
    # regular GDT-0 frame export, and the declared-level ladder.
    "icon-eu": {
        "mapping": "gpuwm/authorities/rw-wps-icon-eu-regular-grib2.mapping.json",
        "directory": STAGING / "icon" / "eu-regular-latlon",
        "suffix": ".grib2.bz2",
        "limit": None,
    },
    # NCEP AWIPS-grid GRIB2 on a declared Lambert grid: the projected-axis
    # export, the declared-grid octet cross-check, and the grid-relative
    # wind rotation.
    # NCEP AWIPS-grid GRIB2 on a declared Lambert grid: the projected-axis
    # export, the declared-grid octet cross-check, the JPEG2000 unpack, and
    # the grid-relative wind rotation.  Three consecutive leads, because the
    # mapping declares hourly lateral boundaries and a single lead can never
    # materialize — so a one-file golden would leave the derivation catalog
    # and the frame header uncompared.
    "lambert-awips": {
        "mapping": "gpuwm/authorities/rw-wps-rap-awip32-grib2.mapping.json",
        "directory": STAGING / "rap",
        "suffix": ".grib2",
        "prefix": "rap.t00z.awip32f0",
        "limit": 3,
    },
    # NetCDF4 through gpuwm's own checked-in NetCDF mapping authority.  The
    # bytes are written by `make_netcdf_sample.py` (real netCDF4/HDF5
    # writer, real CF attributes) because this box's staging tree holds no
    # mapped NetCDF FORCING set; see that script for what the case does and
    # does not prove.  It is the only case whose inputs live beside the
    # golden rather than under the staging tree.
    "netcdf-pressure-level": {
        "mapping": "configs/rw-wps-era5-netcdf.mapping.json",
        "directory": pathlib.Path(__file__).resolve().parent,
        "suffix": "netcdf-pressure-level-sample.nc",
        "limit": None,
    },
}


def _inputs(case: dict[str, object]) -> list[pathlib.Path]:
    directory = pathlib.Path(str(case["directory"]))
    if not directory.is_dir():
        raise SystemExit(f"staging directory is absent: {directory}")
    prefix = str(case.get("prefix") or "")
    files = sorted(
        path for path in directory.iterdir()
        if path.name.endswith(str(case["suffix"])) and path.name.startswith(prefix)
    )
    limit = case.get("limit")
    if limit is not None:
        files = files[: int(limit)]
    if not files:
        raise SystemExit(f"no inputs matched in {directory}")
    return files


def _mask_input_paths(value, files: list[pathlib.Path]):
    """Reduce absolute input paths to file names, anywhere in a document.

    The canonical frame header quotes each field's source references, and
    those are absolute paths, so a raw header digest identifies the
    machine that measured it rather than the decode.  This is the same
    mask ``tests/test_mapped_engine_parity.py`` applies for the same
    reason: inputs are keyed by basename plus content hash.  Nothing else
    is masked.
    """

    if isinstance(value, str):
        for path in files:
            value = value.replace(str(path), path.name)
        return value
    if isinstance(value, (list, tuple)):
        # tuple as well as list: ``SourceFrameHeader.to_dict`` returns the
        # field descriptors as a TUPLE, and a list-only walk slid straight
        # past them and returned the document unmasked -- a mask that
        # silently does nothing is worse than no mask, because the digest
        # still looks reproducible.  json.dumps writes both as arrays, so
        # returning a list here does not change a byte of the digest.
        return [_mask_input_paths(item, files) for item in value]
    if isinstance(value, dict):
        return {key: _mask_input_paths(item, files) for key, item in value.items()}
    return value


def _masked_header_digests(
    mapping: pathlib.Path, files: list[pathlib.Path]
) -> list[str]:
    """The Python engine's frame headers, digested under the mask."""

    import hashlib

    from gpuwm.mapped_source import decode_mapped_source

    resolved = [path.resolve() for path in files]
    return [
        hashlib.sha256(
            json.dumps(
                _mask_input_paths(frame.header.to_dict(), resolved),
                sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        for frame in decode_mapped_source(mapping, files)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=sorted(CASES))
    arguments = parser.parse_args(argv)

    sys.path.insert(0, str(REPOSITORY))
    from gpuwm.mapped_source import inspect_mapped_source

    case = CASES[arguments.case]
    mapping = REPOSITORY / str(case["mapping"])
    files = _inputs(case)
    document = inspect_mapped_source(mapping, files)

    golden = {
        "case": arguments.case,
        "produced_by": "gpuwm.mapped_source.inspect_mapped_source",
        "mapping": str(pathlib.Path(str(case["mapping"])).as_posix()),
        "mapping_sha256": document["mapping"]["sha256"],
        "input_names": [path.name for path in files],
        "input_sha256": [entry["sha256"] for entry in document["inputs"]],
        "source_format": document["source_format"],
        "grid": document["grid"],
        "status": document["status"],
        "frames": [
            {
                "valid_time": frame["valid_time"],
                "source_cycle": frame["source_cycle"],
                "member": frame["member"],
                "decoded_direct_fields": frame["decoded_direct_fields"],
                "unresolved_direct_fields": frame["unresolved_direct_fields"],
                "fields": {
                    name: {
                        "axes": value["axes"],
                        "shape": value["shape"],
                        "missing": value["missing"],
                        "sha256": value["sha256"],
                        "source_reference_count": len(value["source_references"]),
                    }
                    for name, value in frame["fields"].items()
                },
            }
            for frame in document["frames"]
        ],
        "materialization": {
            key: value
            for key, value in document["materialization"].items()
            if key in {"verdict", "frame_count", "frame_header_sha256"}
        },
    }
    if golden["materialization"].get("verdict") == "PASS":
        # Recorded BESIDE the raw digest, not instead of it: the raw one
        # says what this machine measured, the masked one is what another
        # checkout can reproduce.  The crate test compares the masked one.
        golden["materialization"]["frame_header_sha256_masked"] = (
            _masked_header_digests(mapping, files)
        )
    target = pathlib.Path(__file__).resolve().parent / f"{arguments.case}.json"
    target.write_text(
        json.dumps(golden, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # The repository stores text as LF on every platform; Python's text
    # layer would have written CRLF here on Windows.
    from normalize import normalize

    normalize(target)
    print(f"wrote {target} ({len(files)} inputs, {len(golden['frames'])} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
