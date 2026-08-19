"""Native WRF-input preparation for hash-bound 20CRv3 member GRIB2 series."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Sequence

from gpuwm.ingest.source_coverage import owns_source_coverage_refusal
from gpuwm.mapped_composition import load_composition
from gpuwm.mapped_direct import prepare_mapped_wrf
from gpuwm.mapped_source import (
    _build_grib2_tools,
    _grib2_inventory,
    _mapped_engine_choice,
    _sha256,
    load_mapping,
)
from gpuwm.twentycrv3_direct import _manifest, _verify_archive_inventory


#: The adapter identity every 20CRv3 preparation records.
SOURCE_ADAPTER = "rw-wps-20crv3-member-grib2-v1"


def prepare_20crv3_wrf(
    *,
    mapping: str | Path,
    composition: str | Path,
    provenance: str | Path,
    manifest: str | Path,
    manifest_sha256: str,
    grib2_inventory: str | Path | None = None,
    grib2_dump: str | Path | None = None,
    wps_namelist: str | Path,
    geog_root: str | Path,
    experiment_config: str | Path,
    output_root: str | Path,
    preprocess_backend: str = "cuda",
    preprocess_workers: int | None = None,
    cpu_preprocess_bridge: str | Path | None = None,
    hierarchy_workers: int | None = None,
) -> dict[str, object]:
    """Decode one filename-authoritative member and prepare a WRF hierarchy.

    The route's user-facing authority stays its OWN sealed member
    manifest -- the filename-bound member identity, the pairing contract,
    the per-file hashes -- and the two gates that authority funds run
    here, before a byte is composed:

    * PRODUCT IDENTITY.  :func:`_verify_archive_inventory` binds the
      exact every-member GRIB2 product contract per file.  Its
      measurement instrument on the bare default is the engine's raw
      record-inventory surface; on the documented Python-engine
      workaround (``GPUWM_MAPPED_ENGINE=python`` or an explicit tool
      pin) it is the subprocess ``grib2_inventory``.  Same contract,
      same refusal wording, either instrument.
    * ENSEMBLE IDENTITY.  The verified filename member becomes an
      EXPLICIT binding in the bridged composition-inputs manifest, and
      the composition -- whichever engine runs it -- stamps it onto
      every canonical frame and into the sealed alignment receipt.

    The composition itself then runs through
    :func:`gpuwm.mapped_direct.prepare_mapped_wrf` and
    ``decode_composed_source`` like every other composed source: the
    member manifest is bridged into a generic
    ``gpuwm-mapped-composition-inputs-v1`` document (deterministic
    authoring -- two authorings of the same inputs are byte-identical),
    which the composition receipt seals, while the prepared tree's
    evidence copy and identity chain stay bound to the member manifest
    the caller pinned.
    """

    mapping = Path(mapping).resolve()
    composition = Path(composition).resolve()
    provenance = Path(provenance).resolve()
    manifest = Path(manifest).resolve()
    for path in (mapping, composition, provenance, manifest):
        if not path.is_file():
            raise FileNotFoundError(path)

    document, primary = _manifest(manifest, manifest_sha256)
    contract = load_composition(composition, mapping)
    terrain_spec = contract["supplements"]["terrain_height"]
    surface = tuple(
        Path(row["path"]).resolve()
        for row in document["files"]
        if row["role"] == "sfc"
    )
    if len(surface) * 2 != len(primary):
        raise ValueError("20CRv3 preparation lost a pressure/surface pair")
    mapping_document = load_mapping(mapping)
    if mapping_document["format"] != "grib2":
        raise ValueError("20CRv3 named adapter requires a GRIB2 mapping")
    if mapping_document["target"]["boundary_interval_seconds"] \
            != document["cadence_seconds"]:
        raise ValueError(
            "20CRv3 manifest cadence differs from the mapping target contract"
        )

    # WHICH instrument answers the product-identity gate is the same
    # question as which engine composes: an explicit tool pin or the
    # documented workaround selects the Python engine and its subprocess
    # inventory; the bare default reads both in process.
    inventory = None if grib2_inventory is None \
        else Path(grib2_inventory).resolve()
    dump = None if grib2_dump is None else Path(grib2_dump).resolve()
    if (inventory is None) != (dump is None):
        raise ValueError(
            "grib2 inventory and dump tools are an atomic pair on this route"
        )
    from gpuwm.mapped_engine_bridge import ENGINE_RUST

    on_rust_engine = _mapped_engine_choice(
        grib1_bridge=None,
        grib2_inventory=inventory,
        grib2_dump=dump,
        subcommand="compose",
        source_format="grib2",
    ) == ENGINE_RUST
    if on_rust_engine:
        from gpuwm.mapped_engine_bridge import engine_record_inventory

        inventories = engine_record_inventory(primary)

        def inventory_rows(source: Path):
            return inventories[source]
    else:
        if inventory is None:
            inventory, dump = _build_grib2_tools()

        def inventory_rows(source: Path):
            return _grib2_inventory(source, inventory)

    _verify_archive_inventory(
        document,
        primary,
        inventory_rows,
        mapping_document["coordinates"]["vertical"]["levels"],
    )

    from gpuwm.mapped_authoring import author_input_manifest

    with tempfile.TemporaryDirectory(prefix="gpuwm-20crv3-bridge-") as work:
        bridged = Path(work) / "composition-inputs.json"
        author_input_manifest(
            bridged,
            mapping_path=mapping,
            composition_path=composition,
            primary_files=primary,
            supplement_files={str(terrain_spec["data_role"]): surface},
            provenance_files={
                str(terrain_spec["provenance_role"]): provenance,
            },
            grib2_inventory=inventory,
            grib2_dump=dump,
            expected_format="grib2",
            member=str(document["member"]),
            member_identity=str(document["member_identity"]),
        )
        return prepare_mapped_wrf(
            composition=composition,
            mapping=mapping,
            primary_files=primary,
            supplement_files={str(terrain_spec["data_role"]): surface},
            provenance_files={
                str(terrain_spec["provenance_role"]): provenance,
            },
            input_manifest=bridged,
            input_manifest_sha256=_sha256(bridged),
            grib2_inventory=inventory,
            grib2_dump=dump,
            wps_namelist=wps_namelist,
            geog_root=geog_root,
            experiment_config=experiment_config,
            output_root=output_root,
            preprocess_backend=preprocess_backend,
            preprocess_workers=preprocess_workers,
            cpu_preprocess_bridge=cpu_preprocess_bridge,
            hierarchy_workers=hierarchy_workers,
            _source_manifest=manifest,
            _source_manifest_sha256=manifest_sha256,
            _source_adapter=SOURCE_ADAPTER,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gpuwm.twentycrv3_wrf",
        description=(
            "Prepare native WRF initial/boundary files from one hash-bound "
            "20CRv3 member series without WPS or real.exe."
        ),
    )
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--composition", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    # NOT required: a bare run composes in the engine and reads the
    # record inventory in process.  Naming the tools is the explicit
    # Python-engine pin, exactly as on the generic mapped route.
    parser.add_argument("--grib2-inventory", type=Path)
    parser.add_argument("--grib2-dump", type=Path)
    from gpuwm.mapped_engine_bridge import ENGINES

    parser.add_argument(
        "--mapped-engine", choices=ENGINES, default=None,
        help="decode engine override; python is the documented workaround",
    )
    parser.add_argument("--wps-namelist", type=Path, required=True)
    parser.add_argument("--geog-root", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--preprocess-backend", choices=("cpu", "cuda"), default="cuda",
    )
    parser.add_argument("--preprocess-workers", type=int)
    parser.add_argument("--cpu-preprocess-bridge", type=Path)
    parser.add_argument("--hierarchy-workers", type=int)
    return parser


#: The tables the native front doors consume.  A config carrying
#: anything else is not a 20CRv3 config, and saying so in one sentence is
#: the whole difference between a refusal and a crash.
_NATIVE_CONFIG_TABLES = ("experiment", "shared", "projection", "domain")


def _refuse_incompatible_config(path: Path, error: Exception) -> str:
    """One sentence: what is incompatible, and which config route works.

    A node-8 pilot pointed this front door at the wizard's default TOML
    and got a raw traceback out of `experiment.build_experiment`
    ("unknown table(s)/top-level key(s) ['case_data']").  The gate is
    right -- `[case_data]` declares the ERA5 config-driven run path, and
    this route does not read it -- but a traceback does not tell anyone
    how to produce a config that works.
    """

    detail = str(error)
    if "case_data" in detail:
        why = ("its [case_data] table declares the ERA5 config-driven run "
               "path, which this native front door does not read")
    else:
        why = detail
    return (
        f"rw-wps --source 20crv3: {path} is not a config this front door "
        f"can consume: {why}.  It reads "
        f"[{']/['.join(_NATIVE_CONFIG_TABLES)}] only.  Emit a compatible "
        f"config with `gpuwm domain --source gfs` -- that route emits no "
        f"[case_data] table, and only the geometry and physics tables it "
        f"writes are consumed here -- then pass it as --experiment-config."
    )


@owns_source_coverage_refusal
def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mapped_engine is not None:
        # Published, not merely remembered, exactly as the generic mapped
        # runner publishes it: the flag and the documented workaround
        # spelling must mean one thing to everything downstream.
        from gpuwm.mapped_engine_bridge import ENGINE_ENV

        os.environ[ENGINE_ENV] = args.mapped_engine
    # Config compatibility first, before any GRIB2 is decoded: a refusal
    # the user can act on beats a traceback twenty minutes in.
    from gpuwm.experiment import load_experiment

    try:
        load_experiment(args.experiment_config)
    except ValueError as error:
        print(_refuse_incompatible_config(Path(args.experiment_config),
                                          error), file=sys.stderr)
        return 2
    proof = prepare_20crv3_wrf(
        mapping=args.mapping,
        composition=args.composition,
        provenance=args.provenance,
        manifest=args.manifest,
        manifest_sha256=args.manifest_sha256,
        grib2_inventory=args.grib2_inventory,
        grib2_dump=args.grib2_dump,
        wps_namelist=args.wps_namelist,
        geog_root=args.geog_root,
        experiment_config=args.experiment_config,
        output_root=args.output_root,
        preprocess_backend=args.preprocess_backend,
        preprocess_workers=args.preprocess_workers,
        cpu_preprocess_bridge=args.cpu_preprocess_bridge,
        hierarchy_workers=args.hierarchy_workers,
    )
    print(json.dumps(proof, indent=2, sort_keys=True, allow_nan=False))
    # Parity with the GFS front door: a complete, hash-bound run command,
    # on stderr so the proof document on stdout stays machine-readable.
    # The 20CRv3 route ran end to end for a node-8 pilot and then said
    # nothing at all about how to run the forecast.
    from gpuwm.gfs_direct import prepared_forecast_next_command

    for line in prepared_forecast_next_command(
            proof, output_root=args.output_root,
            experiment_config=args.experiment_config,
            wps_namelist=args.wps_namelist, source="20crv3"):
        print(line, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SOURCE_ADAPTER", "prepare_20crv3_wrf"]
