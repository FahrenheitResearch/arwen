"""Hand a WRF input file back to an in-place analysis, unchanged where
nothing changed.

The exchange this serves
------------------------
A variational or ensemble analysis that runs beside the model on CPU (the
kind that opens a WRF file, rewrites the fields it analysed and leaves
every other byte alone) owns the file's shape: its header, the alignment
its I/O library chose for the data section, and every variable it carries
that this model has no state for.  The model side of the exchange has two
duties:

* read the analysed file with every variable either mapped to model state
  or named as passed through (:func:`gpuwm.ingest.wrfinput.read_wrfinput`
  and its ``ANALYSIS_PASSTHROUGH_WRFINPUT`` table), and
* write the file back so the next analysis reads the same file it would
  have read from its own model: same header bytes, same layout, the
  variables the model advanced carrying the model's values and every
  other variable carrying the bytes it arrived with.

:func:`write_back` is the second duty.  It never builds a file from a
schema.  It patches a copy of the file that was read
(:class:`gpuwm.io.nc_writer_bridge.ClassicPatch`), so a variable nobody
rewrites cannot be dropped, reordered or retyped, and a write-back of the
values that were read is byte-identical to the file that was read.  That
identity is the test of the contract: it fails if the reader loses a bit
of any mapped variable, if a mapped variable has no way back, or if any
variable in the file is neither mapped nor named.

What it does not do
-------------------
It does not turn model state into WRF variables.  ``updates`` takes arrays
that are already WRF-named and WRF-shaped; producing them from a forecast
is the history writer's mapping and is wired by the caller.  Without
``updates`` the values written are the ones the reader restored, which is
the zero-step exchange.

Who moves the bytes: the Rust classic writer.  Python here decides which
variable takes which array and writes the receipt.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Mapping

import numpy as np

#: Receipt schema of :func:`write_back`.
RECEIPT_SCHEMA = "gpuwm-analysis-exchange-v1"

#: Dispositions a variable of the file can have in the receipt.
REWRITTEN = "rewritten"
PASSED_THROUGH = "passed-through"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def variable_dispositions(restored, names) -> dict[str, tuple[str, str]]:
    """Say what happens to each variable of the file on the way back.

    Returns ``name -> (disposition, reason)`` for every name in ``names``
    (the file's own variable list, in its order).  A name that is neither
    restored by the reader nor named in one of its pass-through
    inventories is refused: writing such a file back would carry a
    variable nobody has accounted for, stale after a forecast and silent
    about it.
    """

    from gpuwm.ingest.wrfinput import (
        ANALYSIS_PASSTHROUGH_WRFINPUT, IGNORED_WRFINPUT)

    transformed = dict.fromkeys(restored.soil_unit_conversions, (
        "the reader converted its units, so the restored values are not "
        "the file's values"))
    if restored.soil_recovery:
        # The recovery replaces soil water and, when the source carries
        # it, liquid soil water; its receipt does not say which, so both
        # keep the file's bytes.
        transformed.update(dict.fromkeys(("SMOIS", "SH2O"), (
            "the reader recovered soil water from the original source "
            "layers, so the restored values are not the file's values")))
    result: dict[str, tuple[str, str]] = {}
    unknown = []
    for name in names:
        if name in transformed:
            result[name] = (PASSED_THROUGH, transformed[name])
        elif name in restored.raw:
            result[name] = (REWRITTEN, "restored by the reader")
        elif name == "Times":
            result[name] = (PASSED_THROUGH, "the file's time stamp")
        elif name in ANALYSIS_PASSTHROUGH_WRFINPUT:
            result[name] = (PASSED_THROUGH,
                            ANALYSIS_PASSTHROUGH_WRFINPUT[name])
        elif name in restored.surface_input_dispositions:
            result[name] = (PASSED_THROUGH,
                            str(restored.surface_input_dispositions[name]))
        elif name in IGNORED_WRFINPUT:
            result[name] = (PASSED_THROUGH,
                            "a WRF input variable this model does not "
                            "consume")
        else:
            unknown.append(name)
    if unknown:
        raise ValueError(
            f"{restored.path} carries variable(s) the reader neither "
            f"restored nor named as passed through: {sorted(unknown)}.  "
            "The file changed after it was read, or it was read by a "
            "reader with a different inventory; read it again before "
            "writing it back.")
    return result


def write_back(restored, target, *,
               updates: Mapping[str, np.ndarray] | None = None,
               hash_files: bool = True) -> dict:
    """Write the file ``restored`` was read from back to ``target``.

    ``restored`` is the :class:`gpuwm.ingest.wrfinput.RestoredDomain` of
    the analysed file.  ``target`` must not exist; it appears, complete,
    only when every variable has been written.

    Every variable the reader restored is rewritten: from ``updates`` when
    the caller supplies an array under its WRF name, otherwise from the
    restored values, which reproduces the file's own bytes.  Every other
    variable keeps the bytes it arrived with.  Values cross to the Rust
    writer as float64 and are narrowed to the stored type only when that
    is exact; a value that would have to be rounded is refused by name,
    because a rounded write would put a number in the analysis file that
    no analysis and no forecast chose.

    Returns the receipt: per-variable disposition and reason, the two
    file hashes, and whether they are equal.
    """

    from gpuwm.io.nc_writer_bridge import ClassicPatch

    template = Path(restored.path)
    target = Path(os.fspath(target))
    updates = dict(updates or {})
    patch = ClassicPatch(template, target)
    try:
        if patch.num_records != 1:
            raise ValueError(
                f"{template} holds {patch.num_records} Time record(s); an "
                "analysis exchange file holds exactly one, the analysis "
                "time")
        dispositions = variable_dispositions(restored, patch.variables)
        stray = sorted(name for name in updates
                       if dispositions.get(name, ("", ""))[0] != REWRITTEN)
        if stray:
            raise ValueError(
                f"update(s) {stray} name no variable of {template} that "
                "the reader restores; the exchange rewrites the variables "
                "the model carries and cannot add one or overwrite one it "
                "passes through")
        variables = {}
        for name, variable in patch.variables.items():
            disposition, reason = dispositions[name]
            entry = {"disposition": disposition, "reason": reason,
                     "stored_type": variable.dtype.str}
            if disposition == REWRITTEN:
                restored_values = np.asarray(restored.raw[name])
                values = restored_values
                if name in updates:
                    values = np.asarray(updates[name])
                    if values.shape != restored_values.shape:
                        raise ValueError(
                            f"update {name} has shape {values.shape}; the "
                            f"file's {name} has shape "
                            f"{restored_values.shape}")
                    entry["source"] = "update"
                else:
                    entry["source"] = "restored"
                if values.size != variable.elements:
                    raise ValueError(
                        f"{name}: {values.size} value(s) for a variable "
                        f"of {variable.elements} element(s) in {template}")
                patch.put(name, values,
                          record=0 if variable.is_record else None)
            variables[name] = entry
        patch.finish()
    except BaseException:
        patch.abort()
        raise
    counts = {
        REWRITTEN: sum(1 for entry in variables.values()
                       if entry["disposition"] == REWRITTEN),
        PASSED_THROUGH: sum(1 for entry in variables.values()
                            if entry["disposition"] == PASSED_THROUGH),
    }
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "template": str(template),
        "target": str(target),
        "updated": sorted(updates),
        "counts": counts,
        "variables": variables,
    }
    if hash_files:
        receipt["template_sha256"] = _sha256(template)
        receipt["target_sha256"] = _sha256(target)
        receipt["byte_identical"] = (
            receipt["template_sha256"] == receipt["target_sha256"])
    return receipt


#: Inventory classes of :func:`inventory`, in the order the reader tests
#: them.
RESTORED = "restored"
AUXILIARY = "auxiliary"
ANALYSIS_CARRIED = "analysis-carried"
NOT_CONSUMED = "not-consumed"
UNNAMED = "unnamed"


def inventory(path) -> dict:
    """Sort every variable of a WRF input file into the reader's inventories.

    Reads the header only.  It answers one question before any field is
    decoded: is every variable of this file either restored by the reader
    or named as passed through?  ``unnamed`` lists the ones that are
    neither, which the reader refuses and which need a consumer or a
    pass-through reason before the file can be exchanged.

    The scheme-specific inventories (which hydrometeors a microphysics
    package activates, which land-surface records a scheme initialises
    itself) are not applied here, because they need the run's
    configuration; a name they would refuse is listed as ``restored``.
    """

    from gpuwm import netcdf_bridge
    from gpuwm.ingest.wrfinput import (
        ALLOWED_WRFINPUT, ANALYSIS_PASSTHROUGH_WRFINPUT,
        EXPLICIT_AUXILIARY_WRFINPUT, IGNORED_WRFINPUT)

    path = Path(os.fspath(path))
    with netcdf_bridge.open_dataset(path) as dataset:
        names = list(dataset.variables)
        dimensions = {name: len(dim)
                      for name, dim in dataset.dimensions.items()}
    classes: dict[str, list[str]] = {
        RESTORED: [], AUXILIARY: [], ANALYSIS_CARRIED: [],
        NOT_CONSUMED: [], UNNAMED: []}
    for name in names:
        if name in EXPLICIT_AUXILIARY_WRFINPUT:
            classes[AUXILIARY].append(name)
        elif name in ALLOWED_WRFINPUT:
            classes[RESTORED].append(name)
        elif name in ANALYSIS_PASSTHROUGH_WRFINPUT:
            classes[ANALYSIS_CARRIED].append(name)
        elif name in IGNORED_WRFINPUT:
            classes[NOT_CONSUMED].append(name)
        else:
            classes[UNNAMED].append(name)
    return {
        "schema": "gpuwm-analysis-exchange-inventory-v1",
        "file": str(path),
        "dimensions": dimensions,
        "variable_count": len(names),
        "counts": {name: len(members) for name, members in classes.items()},
        "classes": classes,
    }


def _file_physics(metadata, *, bl_mynn_tkeadvect: bool, wif_input_opt: int):
    """The physics selection a file's own global attributes declare."""

    from types import SimpleNamespace

    from gpuwm.ingest.wrfinput import _integral_attribute

    attributes = metadata.global_attributes
    boundary_layer = _integral_attribute(attributes.get("BL_PBL_PHYSICS"))
    return SimpleNamespace(
        moist=True,
        mp_physics=int(metadata.mp_physics),
        sf_surface_physics=int(metadata.sf_surface_physics),
        bl_pbl_physics=boundary_layer,
        bl_mynn_tkeadvect=bool(bl_mynn_tkeadvect),
        wif_input_opt=int(wif_input_opt))


def round_trip(path, target, *, bl_mynn_tkeadvect: bool = False,
               wif_input_opt: int = 1) -> dict:
    """Read a WRF input file through the reader and write it straight back.

    The zero-step exchange on a real file.  Geometry and physics packages
    are taken from the file's own dimensions and global attributes, so
    this checks the exchange itself: that the reader names every variable
    and that the way back reproduces the file.  It does not check that the
    file is the one a run expects; that is the forecast door's check
    against its namelist.  The two selectors a WRF file does not record
    are arguments.

    Whether a forecast could START from the file is a separate question:
    the receipt's ``forecast_start_missing`` lists the names the forecast
    door requires and the file lacks, which that door refuses by name.
    """

    from gpuwm.ingest.wrfinput import (
        missing_required_wrfinput, read_wrfinput, read_wrfinput_metadata)

    metadata = read_wrfinput_metadata(path)
    cfg = _file_physics(metadata, bl_mynn_tkeadvect=bl_mynn_tkeadvect,
                        wif_input_opt=wif_input_opt)
    restored = read_wrfinput(
        path, expected_dimensions=metadata.pinned_dimensions(), cfg=cfg,
        require_complete=False)
    receipt = write_back(restored, target)
    receipt["forecast_start_missing"] = missing_required_wrfinput(
        restored.raw, cfg)
    return receipt


def main(argv=None) -> int:
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m gpuwm.io.analysis_exchange",
        description="Check a WRF input file against the analysis exchange "
                    "contract: every variable restored or named as passed "
                    "through, and a write-back that reproduces the file.")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser(
        "inventory", help="sort the file's variables into the reader's "
                          "inventories; exit 1 when any is unnamed")
    listing.add_argument("file", type=Path)
    trip = commands.add_parser(
        "round-trip", help="read the file and write it straight back; "
                           "exit 1 unless the copy is byte-identical")
    trip.add_argument("file", type=Path)
    trip.add_argument("target", type=Path)
    trip.add_argument(
        "--bl-mynn-tkeadvect", action="store_true",
        help="the run advects MYNN TKE (a WRF file does not record it)")
    trip.add_argument(
        "--wif-input-opt", type=int, default=1,
        help="the run's aerosol input selector (a WRF file does not record "
             "it); default 1")
    trip.add_argument(
        "--receipt", type=Path, default=None,
        help="write the full per-variable receipt here as JSON")
    args = parser.parse_args(argv)
    try:
        if args.command == "inventory":
            document = inventory(args.file)
            print(json.dumps(document, indent=1))
            return 1 if document["classes"][UNNAMED] else 0
        receipt = round_trip(args.file, args.target,
                             bl_mynn_tkeadvect=args.bl_mynn_tkeadvect,
                             wif_input_opt=args.wif_input_opt)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"analysis exchange: {error}", file=sys.stderr)
        return 2
    if args.receipt is not None:
        args.receipt.write_text(json.dumps(receipt, indent=1) + "\n",
                                encoding="utf-8")
    summary = {key: receipt[key] for key in (
        "schema", "template", "target", "counts", "template_sha256",
        "target_sha256", "byte_identical", "forecast_start_missing")}
    print(json.dumps(summary, indent=1))
    return 0 if receipt["byte_identical"] else 1


__all__ = [
    "ANALYSIS_CARRIED",
    "AUXILIARY",
    "NOT_CONSUMED",
    "PASSED_THROUGH",
    "RECEIPT_SCHEMA",
    "RESTORED",
    "REWRITTEN",
    "UNNAMED",
    "inventory",
    "main",
    "round_trip",
    "variable_dispositions",
    "write_back",
]


if __name__ == "__main__":
    raise SystemExit(main())
