"""Native WRF-input preparation for hash-bound 20CRv3 member GRIB2 series."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Sequence

from gpuwm.mapped_composition import MappedSourceBundle, load_composition
from gpuwm.mapped_direct import prepare_mapped_wrf
from gpuwm.mapped_source import _sha256
from gpuwm.twentycrv3_direct import _manifest, decode_20crv3_grib2


def prepare_20crv3_wrf(
    *,
    mapping: str | Path,
    composition: str | Path,
    provenance: str | Path,
    manifest: str | Path,
    manifest_sha256: str,
    grib2_inventory: str | Path,
    grib2_dump: str | Path,
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

    The adapter keeps the custom 20CRv3 manifest as the input authority; it
    does not translate it into the generic manifest schema or erase the member
    label that is absent from the supplied GRIB2 PDT. Terrain and exact Noah
    soil layers are carried directly by the paired surface files.
    """

    mapping = Path(mapping).resolve()
    composition = Path(composition).resolve()
    provenance = Path(provenance).resolve()
    manifest = Path(manifest).resolve()
    inventory = Path(grib2_inventory).resolve()
    dump = Path(grib2_dump).resolve()
    for path in (mapping, composition, provenance, manifest, inventory, dump):
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

    decode_started = time.perf_counter()
    frames, canonical_receipt = decode_20crv3_grib2(
        mapping=mapping,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        grib2_inventory=inventory,
        grib2_dump=dump,
    )
    decode_seconds = time.perf_counter() - decode_started
    hashes = {
        Path(path).resolve(): digest
        for path, digest in canonical_receipt["input_sha256"].items()
    }
    bundle = MappedSourceBundle(
        frames=frames,
        mapping_path=mapping,
        mapping_sha256=_sha256(mapping),
        composition_path=composition,
        composition_sha256=_sha256(composition),
        input_manifest_path=manifest,
        input_manifest_sha256=_sha256(manifest),
        decoder_paths={
            "grib2_inventory": inventory,
            "grib2_dump": dump,
        },
        decoder_sha256={
            "grib2_inventory": _sha256(inventory),
            "grib2_dump": _sha256(dump),
        },
        terrain_data_paths=surface,
        terrain_data_sha256=tuple(hashes[path] for path in surface),
        terrain_provenance_path=provenance,
        terrain_provenance_sha256=_sha256(provenance),
        soil_layer_contract=contract["soil_layers"],
        alignment_receipt={
            "strategy": "20crv3_in_band_surface_same_grid",
            "terrain_external_supplement": False,
            "member": document["member"],
            "member_identity": document["member_identity"],
            "surface_file_count": len(surface),
            "valid_time_count": len(frames),
            "canonical_receipt_content_sha256": canonical_receipt[
                "receipt_content_sha256"
            ],
            "coordinate_match": "same_decoded_grib2_grid_fingerprint",
            "terrain_invariant_across_all_times": True,
        },
    )
    return prepare_mapped_wrf(
        composition=composition,
        mapping=mapping,
        primary_files=primary,
        supplement_files={str(terrain_spec["data_role"]): surface},
        provenance_files={
            str(terrain_spec["provenance_role"]): provenance,
        },
        input_manifest=manifest,
        input_manifest_sha256=manifest_sha256,
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
        _predecoded_bundle=bundle,
        _predecoded_seconds=decode_seconds,
        _source_adapter="rw-wps-20crv3-member-grib2-v1",
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
    parser.add_argument("--grib2-inventory", type=Path, required=True)
    parser.add_argument("--grib2-dump", type=Path, required=True)
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


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["prepare_20crv3_wrf"]
