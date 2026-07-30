"""Read and validate gpuwm N1.5 instrumented-WRF boundary-table dumps.

The binary format is emitted by ``instrument-med-force.patch``.  WRF stores
each table in Fortran order as ``(along_boundary, vertical, bdy_width)``;
this reader returns the gpuwm-facing canonical order
``(vertical, along_boundary, bdy_width)``.  Optional ``PROG`` and ``DBDY``
extension records carry force-bracketing prognostic samples and solve-side
boundary-clock samples.  Optional ``INPT`` records carry the complete
pre-coupling arrays needed by the independent CPU force producer; legacy
table-only and diagnostic-extension dumps remain valid.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field as dataclass_field
import json
from pathlib import Path
import struct
from typing import BinaryIO, Mapping

import numpy as np


MAGIC = b"GPUWM-N1P5-DUMP1"
SCHEMA_VERSION = 1
ENDIAN_MARKER = 0x01020304
SIDES = ("xs", "xe", "ys", "ye")
KINDS = ("value", "tendency")
STATE_FIELDS = ("u", "v", "w", "t", "ph", "mu")
MORRISON_SCALARS = frozenset({"qni", "qns", "qnr", "qng"})

_WRF_NAMES = {
    "qvapor": "qv",
    "qcloud": "qc",
    "qrain": "qr",
    "qice": "qi",
    "qice2": "qi2",
    "qice3": "qi3",
    "qsnow": "qs",
    "qgraup": "qg",
    "qhail": "qh",
    "qndrop": "qndrop",
    "qnice": "qni",
    "qnice2": "qni2",
    "qnice3": "qni3",
    "qnsnow": "qns",
    "qnrain": "qnr",
    "qngraupel": "qng",
    "qnhail": "qnh",
    "qnccn": "qnn",
    "qncloud": "qnc",
}


class DumpFormatError(ValueError):
    """The dump is truncated, inconsistent, or violates the N1.5 contract."""


@dataclass(frozen=True, order=True)
class TableKey:
    """Stable NPZ key for one child boundary table."""

    family: str
    field: str
    side: str
    kind: str

    def __str__(self) -> str:
        return f"{self.family}.{self.field}.{self.side}.{self.kind}"

    @classmethod
    def parse(cls, value: str) -> TableKey:
        parts = value.split(".")
        if len(parts) != 4:
            raise DumpFormatError(
                f"invalid table key {value!r}; expected family.field.side.kind")
        return cls(*parts)


@dataclass(frozen=True)
class DumpMetadata:
    schema_version: int
    domain_id: int
    parent_id: int
    force_ordinal: int
    mp_physics: int
    spec_bdy_width: int
    active_moist: int
    active_scalar: int
    parent_dt: float
    child_dt: float
    nids: int
    nide: int
    njds: int
    njde: int
    nkds: int
    nkde: int


@dataclass(frozen=True)
class PrognosticSample:
    """Small parent/child prognostic sample before or after force."""

    phase: str
    domain_id: int
    values: np.ndarray


@dataclass(frozen=True)
class BoundaryClockSample:
    """One pre-application child dtbc and bdy+dtbc*bdy_t sample."""

    domain_id: int
    step_ordinal: int
    dtbc: float
    values: np.ndarray


@dataclass(frozen=True, order=True)
class ForceInputKey:
    """Identity of one full-array force input snapshot."""

    phase: str
    domain_id: int
    family: str
    field: str

    def __str__(self) -> str:
        return f"{self.phase}.d{self.domain_id:02d}.{self.family}.{self.field}"


@dataclass(frozen=True)
class NestForceDump:
    metadata: DumpMetadata
    tables: Mapping[TableKey, np.ndarray]
    prognostic_samples: tuple[PrognosticSample, ...] = ()
    boundary_clock_samples: tuple[BoundaryClockSample, ...] = ()
    force_inputs: Mapping[ForceInputKey, np.ndarray] = dataclass_field(
        default_factory=dict)


def _read_exact(stream: BinaryIO, size: int, what: str) -> bytes:
    payload = stream.read(size)
    if len(payload) != size:
        raise DumpFormatError(
            f"truncated {what}: wanted {size} bytes, got {len(payload)}")
    return payload


def _decode_fixed(payload: bytes, what: str) -> str:
    try:
        return payload.decode("ascii").rstrip(" \x00").lower()
    except UnicodeDecodeError as exc:
        raise DumpFormatError(f"non-ASCII {what}") from exc


def _canonical_field(family: str, field: str) -> str:
    normalized = field.strip().lower()
    return _WRF_NAMES.get(normalized, normalized) if family != "state" else normalized


def read_dump(path: str | Path, *, validate: bool = True) -> NestForceDump:
    """Read one complete dump, optionally enforcing all F9 structure checks."""
    dump_path = Path(path)
    with dump_path.open("rb") as stream:
        magic = _read_exact(stream, len(MAGIC), "magic")
        if magic != MAGIC:
            raise DumpFormatError(
                f"bad magic {magic!r}; expected {MAGIC!r}")

        marker_bytes = _read_exact(stream, 4, "endian marker")
        if struct.unpack("<i", marker_bytes)[0] == ENDIAN_MARKER:
            endian = "<"
        elif struct.unpack(">i", marker_bytes)[0] == ENDIAN_MARKER:
            endian = ">"
        else:
            raise DumpFormatError("invalid endian marker")

        integers = struct.unpack(
            f"{endian}8i",
            _read_exact(stream, 8 * 4, "fixed integer metadata"),
        )
        (schema, domain_id, parent_id, force_ordinal, mp_physics,
         width, active_moist, active_scalar) = integers
        parent_dt, child_dt = struct.unpack(
            f"{endian}2f", _read_exact(stream, 8, "time-step metadata"))
        nids, nide, njds, njde, nkds, nkde = struct.unpack(
            f"{endian}6i", _read_exact(stream, 24, "domain bounds"))
        metadata = DumpMetadata(
            schema, domain_id, parent_id, force_ordinal, mp_physics,
            width, active_moist, active_scalar, parent_dt, child_dt,
            nids, nide, njds, njde, nkds, nkde,
        )

        tables: dict[TableKey, np.ndarray] = {}
        prognostic_samples: list[PrognosticSample] = []
        boundary_clock_samples: list[BoundaryClockSample] = []
        force_inputs: dict[ForceInputKey, np.ndarray] = {}
        done = False
        while True:
            tag = stream.read(4)
            if not tag:
                if not done:
                    raise DumpFormatError("truncated dump before DONE record")
                break
            if len(tag) != 4:
                raise DumpFormatError(
                    f"trailing truncated record tag: wanted 4 bytes, got {len(tag)}")
            if tag == b"DONE":
                if done:
                    raise DumpFormatError("duplicate DONE record")
                (declared_count,) = struct.unpack(
                    f"{endian}i", _read_exact(stream, 4, "record count"))
                if declared_count != len(tables):
                    raise DumpFormatError(
                        f"trailer declares {declared_count} tables, "
                        f"but {len(tables)} were read")
                done = True
                continue
            if tag == b"PROG":
                phase = _decode_fixed(
                    _read_exact(stream, 8, "prognostic phase"),
                    "prognostic phase",
                )
                domain_id, count = struct.unpack(
                    f"{endian}2i",
                    _read_exact(stream, 8, "prognostic sample metadata"),
                )
                if count <= 0:
                    raise DumpFormatError(
                        f"non-positive prognostic sample count {count}")
                dtype = np.dtype(f"{endian}f4")
                values = np.frombuffer(
                    _read_exact(stream, count * 4, "prognostic samples"),
                    dtype=dtype,
                    count=count,
                ).astype(np.float32, copy=False)
                values.setflags(write=False)
                prognostic_samples.append(
                    PrognosticSample(phase, domain_id, values))
                continue
            if tag == b"DBDY":
                domain_id, step_ordinal = struct.unpack(
                    f"{endian}2i",
                    _read_exact(stream, 8, "boundary-clock identifiers"),
                )
                (dtbc,) = struct.unpack(
                    f"{endian}f", _read_exact(stream, 4, "boundary dtbc"))
                (count,) = struct.unpack(
                    f"{endian}i",
                    _read_exact(stream, 4, "boundary sample count"),
                )
                if count <= 0:
                    raise DumpFormatError(
                        f"non-positive boundary sample count {count}")
                dtype = np.dtype(f"{endian}f4")
                values = np.frombuffer(
                    _read_exact(stream, count * 4, "boundary samples"),
                    dtype=dtype,
                    count=count,
                ).astype(np.float32, copy=False)
                values.setflags(write=False)
                boundary_clock_samples.append(
                    BoundaryClockSample(domain_id, step_ordinal, dtbc, values))
                continue
            if tag == b"INPT":
                if done:
                    raise DumpFormatError("force input appears after DONE record")
                phase = _decode_fixed(
                    _read_exact(stream, 8, "force-input phase"),
                    "force-input phase",
                )
                (domain_id,) = struct.unpack(
                    f"{endian}i", _read_exact(stream, 4, "force-input domain"))
                family = _decode_fixed(
                    _read_exact(stream, 8, "force-input family"),
                    "force-input family",
                )
                field = _decode_fixed(
                    _read_exact(stream, 32, "force-input field"),
                    "force-input field",
                )
                (rank,) = struct.unpack(
                    f"{endian}i", _read_exact(stream, 4, "force-input rank"))
                dims = struct.unpack(
                    f"{endian}3i", _read_exact(stream, 12, "force-input shape"))
                if rank not in (1, 2, 3):
                    raise DumpFormatError(
                        f"invalid force-input rank {rank} for {family}.{field}")
                shape_f = dims[:rank]
                if any(size <= 0 for size in shape_f) or any(dims[rank:]):
                    raise DumpFormatError(
                        f"invalid rank-{rank} force-input shape {dims} "
                        f"for {family}.{field}")
                count = int(np.prod(shape_f, dtype=np.int64))
                dtype = np.dtype(f"{endian}f4")
                fortran = np.frombuffer(
                    _read_exact(stream, count * 4, f"force input {family}.{field}"),
                    dtype=dtype,
                    count=count,
                ).astype(np.float32, copy=False).reshape(shape_f, order="F")
                if rank == 3:
                    canonical = np.ascontiguousarray(fortran.transpose(1, 2, 0))
                elif rank == 2:
                    canonical = np.ascontiguousarray(fortran.transpose(1, 0))
                else:
                    canonical = np.ascontiguousarray(fortran)
                key = ForceInputKey(
                    phase, domain_id, family,
                    _canonical_field(family, field),
                )
                if key in force_inputs:
                    raise DumpFormatError(f"duplicate force input {key}")
                canonical.setflags(write=False)
                force_inputs[key] = canonical
                continue
            if tag != b"TABL":
                raise DumpFormatError(
                    f"trailing/unknown record tag {tag!r}")
            if done:
                raise DumpFormatError("boundary table appears after DONE record")

            family = _decode_fixed(_read_exact(stream, 8, "family"), "family")
            field = _decode_fixed(_read_exact(stream, 32, "field"), "field")
            side = _decode_fixed(_read_exact(stream, 2, "side"), "side")
            kind = _decode_fixed(_read_exact(stream, 8, "kind"), "kind")
            shape_f = struct.unpack(
                f"{endian}3i", _read_exact(stream, 12, "table shape"))
            if any(size <= 0 for size in shape_f):
                raise DumpFormatError(
                    f"non-positive Fortran shape {shape_f} for {field}")
            count = int(np.prod(shape_f, dtype=np.int64))
            dtype = np.dtype(f"{endian}f4")
            payload = _read_exact(stream, count * 4, f"payload for {field}")
            fortran = np.frombuffer(payload, dtype=dtype, count=count)
            fortran = fortran.astype(np.float32, copy=False).reshape(
                shape_f, order="F")
            canonical = np.ascontiguousarray(fortran.transpose(1, 0, 2))
            key = TableKey(family, _canonical_field(family, field), side, kind)
            if key in tables:
                raise DumpFormatError(f"duplicate table {key}")
            canonical.setflags(write=False)
            tables[key] = canonical

    result = NestForceDump(
        metadata,
        tables,
        tuple(prognostic_samples),
        tuple(boundary_clock_samples),
        force_inputs,
    )
    if validate:
        validate_dump(result)
    return result


def _expected_shape(metadata: DumpMetadata, field: str, side: str) -> tuple[int, int, int]:
    nz_mass = metadata.nkde - metadata.nkds
    nz = nz_mass + 1 if field in {"w", "ph"} else (1 if field == "mu" else nz_mass)
    if side in {"xs", "xe"}:
        along = metadata.njde - metadata.njds
        if field == "v":
            along += 1
    else:
        along = metadata.nide - metadata.nids
        if field == "u":
            along += 1
    return nz, along, metadata.spec_bdy_width


def validate_dump(dump: NestForceDump) -> None:
    """Enforce completeness, shape, finiteness, and Morrison F9 checks."""
    md = dump.metadata
    if md.schema_version != SCHEMA_VERSION:
        raise DumpFormatError(
            f"unsupported schema {md.schema_version}; expected {SCHEMA_VERSION}")
    if md.domain_id <= 1 or md.parent_id <= 0:
        raise DumpFormatError(
            f"expected child/parent ids, got d{md.domain_id:02d}/d{md.parent_id:02d}")
    if md.force_ordinal != 1:
        raise DumpFormatError(
            f"expected pinned force ordinal 1, got {md.force_ordinal}")
    if md.spec_bdy_width <= 0:
        raise DumpFormatError("spec_bdy_width must be positive")
    if not np.isfinite([md.parent_dt, md.child_dt]).all():
        raise DumpFormatError("non-finite parent or child dt")
    if md.parent_dt <= 0 or md.child_dt <= 0:
        raise DumpFormatError("parent and child dt must be positive")

    by_family: dict[str, set[str]] = {"state": set(), "moist": set(), "scalar": set()}
    coverage: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for key, values in dump.tables.items():
        if key.family not in by_family:
            raise DumpFormatError(f"unexpected family {key.family!r} in {key}")
        if key.side not in SIDES or key.kind not in KINDS:
            raise DumpFormatError(f"invalid side/kind in {key}")
        if values.dtype != np.float32:
            raise DumpFormatError(f"{key} is {values.dtype}, expected float32")
        if not np.isfinite(values).all():
            bad = np.argwhere(~np.isfinite(values))[0]
            raise DumpFormatError(
                f"non-finite value in {key} at canonical index {tuple(bad)}")
        expected = _expected_shape(md, key.field, key.side)
        if values.shape != expected:
            raise DumpFormatError(
                f"shape mismatch for {key}: {values.shape}, expected {expected}")
        by_family[key.family].add(key.field)
        coverage.setdefault((key.family, key.field), set()).add((key.side, key.kind))

    if by_family["state"] != set(STATE_FIELDS):
        raise DumpFormatError(
            f"state fields are {sorted(by_family['state'])}; expected {list(STATE_FIELDS)}")
    if len(by_family["moist"]) != md.active_moist:
        raise DumpFormatError(
            f"dump has {len(by_family['moist'])} moist species, "
            f"header declares {md.active_moist}")
    if len(by_family["scalar"]) != md.active_scalar:
        raise DumpFormatError(
            f"dump has {len(by_family['scalar'])} scalar species, "
            f"header declares {md.active_scalar}")
    complete = {(side, kind) for side in SIDES for kind in KINDS}
    incomplete = {
        f"{family}.{field}": sorted(complete - seen)
        for (family, field), seen in coverage.items() if seen != complete
    }
    if incomplete:
        raise DumpFormatError(f"incomplete value/tendency side coverage: {incomplete}")
    if md.mp_physics == 10 and not MORRISON_SCALARS <= by_family["scalar"]:
        missing = sorted(MORRISON_SCALARS - by_family["scalar"])
        raise DumpFormatError(
            f"Morrison mp_physics=10 dump is missing mandatory scalars {missing}")

    if dump.prognostic_samples:
        observed = [(sample.phase, sample.domain_id)
                    for sample in dump.prognostic_samples]
        expected = [
            ("before", md.parent_id),
            ("before", md.domain_id),
            ("after", md.parent_id),
            ("after", md.domain_id),
        ]
        if observed != expected:
            raise DumpFormatError(
                f"prognostic sample sequence is {observed}; expected {expected}")
        for sample in dump.prognostic_samples:
            if sample.values.dtype != np.float32 or sample.values.shape != (12,):
                raise DumpFormatError(
                    f"invalid prognostic sample for d{sample.domain_id:02d}: "
                    f"dtype={sample.values.dtype}, shape={sample.values.shape}")
            if not np.isfinite(sample.values).all():
                raise DumpFormatError("non-finite prognostic sample")

    if dump.boundary_clock_samples:
        ratio = int(round(md.parent_dt / md.child_dt))
        if ratio <= 0 or not np.isclose(
            np.float32(md.child_dt * ratio),
            np.float32(md.parent_dt),
        ):
            raise DumpFormatError("parent/child dt do not form an integer ratio")
        ordinals = [sample.step_ordinal
                    for sample in dump.boundary_clock_samples]
        if ordinals != list(range(1, ratio + 1)):
            raise DumpFormatError(
                f"boundary-clock ordinals are {ordinals}; expected 1..{ratio}")
        expected_dtbc = np.float32(0.0)
        for sample in dump.boundary_clock_samples:
            if sample.domain_id != md.domain_id:
                raise DumpFormatError(
                    f"boundary-clock sample belongs to d{sample.domain_id:02d}, "
                    f"expected d{md.domain_id:02d}")
            expected_dtbc = np.float32(expected_dtbc + np.float32(md.child_dt))
            if np.float32(sample.dtbc).tobytes() != expected_dtbc.tobytes():
                raise DumpFormatError(
                    f"boundary dtbc at ordinal {sample.step_ordinal} is "
                    f"{sample.dtbc}, expected FP32 {float(expected_dtbc)}")
            if sample.values.dtype != np.float32 or sample.values.shape != (6,):
                raise DumpFormatError(
                    f"invalid boundary sample at ordinal {sample.step_ordinal}: "
                    f"dtype={sample.values.dtype}, shape={sample.values.shape}")
            if not np.isfinite(sample.values).all():
                raise DumpFormatError("non-finite boundary sample")

    for key, values in dump.force_inputs.items():
        if key.phase != "before":
            raise DumpFormatError(
                f"force input {key} is not a pre-coupling 'before' snapshot")
        if key.domain_id not in {md.parent_id, md.domain_id}:
            raise DumpFormatError(
                f"force input {key} belongs to an unrelated domain")
        if key.family not in {"state", "moist", "scalar", "aux"}:
            raise DumpFormatError(
                f"force input {key} has unexpected family {key.family!r}")
        if values.dtype != np.float32 or not values.flags.c_contiguous:
            raise DumpFormatError(
                f"force input {key} must be C-contiguous float32")
        if not np.isfinite(values).all():
            raise DumpFormatError(f"non-finite value in force input {key}")


def export_npz(dump: NestForceDump, path: str | Path) -> None:
    """Export canonical tables using the comparison-harness NPZ contract."""
    np.savez(
        Path(path),
        **{str(key): values for key, values in sorted(dump.tables.items())},
    )


def summary(dump: NestForceDump) -> dict[str, object]:
    families: dict[str, list[str]] = {}
    for key in dump.tables:
        families.setdefault(key.family, []).append(key.field)
    return {
        "schema_version": dump.metadata.schema_version,
        "domain_id": dump.metadata.domain_id,
        "parent_id": dump.metadata.parent_id,
        "force_ordinal": dump.metadata.force_ordinal,
        "mp_physics": dump.metadata.mp_physics,
        "parent_dt": dump.metadata.parent_dt,
        "child_dt": dump.metadata.child_dt,
        "spec_bdy_width": dump.metadata.spec_bdy_width,
        "table_count": len(dump.tables),
        "prognostic_samples": [
            {
                "phase": sample.phase,
                "domain_id": sample.domain_id,
                "values": sample.values.tolist(),
            }
            for sample in dump.prognostic_samples
        ],
        "boundary_clock_samples": [
            {
                "domain_id": sample.domain_id,
                "step_ordinal": sample.step_ordinal,
                "dtbc": sample.dtbc,
                "values": sample.values.tolist(),
            }
            for sample in dump.boundary_clock_samples
        ],
        "force_input_count": len(dump.force_inputs),
        "force_inputs": [str(key) for key in sorted(dump.force_inputs)],
        "families": {
            family: sorted(set(fields)) for family, fields in sorted(families.items())
        },
        "all_finite": True,
        "shapes": {
            str(key): list(values.shape)
            for key, values in sorted(dump.tables.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path)
    parser.add_argument(
        "--export-npz", type=Path,
        help="write canonical comparison-input arrays to this NPZ")
    parser.add_argument(
        "--compact", action="store_true", help="omit the per-table shape map")
    args = parser.parse_args()

    dump = read_dump(args.dump)
    report = summary(dump)
    if args.compact:
        report.pop("shapes")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.export_npz is not None:
        export_npz(dump, args.export_npz)


if __name__ == "__main__":
    main()
