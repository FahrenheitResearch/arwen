"""CPU-only contract pins for the instrumented-WRF N1.5 harness."""

from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]
TOOLS = ROOT / "tools" / "wrf_instrumented"
FIXTURE = TOOLS / "fixtures" / "single-table-dump.bin.b64"
INSTRUMENT_PATCH = TOOLS / "instrument-med-force.patch"
ORACLE_NAMELIST = TOOLS / "namelist.input.n1p5"
PRISTINE_SOURCE_SHA256 = {
    Path("share/mediation_force_domain.F"):
        "aaef43f69eb810809eb890688b840ce894aea9ae1ae99f8266e4fa8c3b9f5518",
    Path("dyn_em/solve_em.F"):
        "e42df5d7db4b6ec4a3b8e2f228a8ec8f9a4c426656093bfcebe58a8de6c3e8f4",
}

# The harness files are executable scripts rather than an installed package.
sys.path.insert(0, str(TOOLS))

from compare_nest_force import BAND, compare, load_candidate  # noqa: E402
from dump_reader import (  # noqa: E402
    DumpFormatError,
    DumpMetadata,
    TableKey,
    export_npz,
    read_dump,
)


_RAW_FIELDS = (
    ("state", "u", "u"),
    ("state", "v", "v"),
    ("state", "w", "w"),
    ("state", "t", "t"),
    ("state", "ph", "ph"),
    ("state", "mu", "mu"),
    ("moist", "QVAPOR", "qv"),
)


def test_oracle_namelist_pins_gpuwm_thermodynamic_scope() -> None:
    """The N1.5 case owns the two scope selectors; no Registry defaults."""
    text = ORACLE_NAMELIST.read_text(encoding="ascii")
    time_control = text[text.index("&time_control"):text.index("\n/", 1)]
    dynamics_start = text.index("&dynamics")
    dynamics = text[dynamics_start:text.index("\n/", dynamics_start)]

    nwp_line = " nwp_diagnostics                     = 0,"
    theta_line = " use_theta_m                         = 0,"
    assert text.count(nwp_line) == 1
    assert text.count(theta_line) == 1
    assert nwp_line in time_control
    assert theta_line in dynamics


_MORRISON_FIELDS = (
    ("QNICE", "qni"),
    ("QNSNOW", "qns"),
    ("QNRAIN", "qnr"),
    ("QNGRAUPEL", "qng"),
)


def _fixed(value: str, size: int) -> bytes:
    return value.encode("ascii").ljust(size, b" ")


def _expected_shape(
    metadata: DumpMetadata, field: str, side: str,
) -> tuple[int, int, int]:
    nz_mass = metadata.nkde - metadata.nkds
    nz = nz_mass + 1 if field in {"w", "ph"} else (1 if field == "mu" else nz_mass)
    if side in {"xs", "xe"}:
        along = metadata.njde - metadata.njds + (field == "v")
    else:
        along = metadata.nide - metadata.nids + (field == "u")
    return nz, along, metadata.spec_bdy_width


def _synthetic_dump_bytes(
    scalar_fields: tuple[tuple[str, str], ...] = _MORRISON_FIELDS,
    *,
    extensions: bool = False,
) -> tuple[bytes, DumpMetadata, dict[TableKey, np.ndarray]]:
    metadata = DumpMetadata(
        schema_version=1,
        domain_id=2,
        parent_id=1,
        force_ordinal=1,
        mp_physics=10,
        spec_bdy_width=2,
        active_moist=1,
        active_scalar=len(scalar_fields),
        parent_dt=60.0,
        child_dt=15.0,
        nids=1,
        nide=4,
        njds=1,
        njde=3,
        nkds=1,
        nkde=4,
    )
    payload = bytearray(b"GPUWM-N1P5-DUMP1")
    payload += struct.pack("<i", 0x01020304)
    payload += struct.pack(
        "<8i",
        metadata.schema_version,
        metadata.domain_id,
        metadata.parent_id,
        metadata.force_ordinal,
        metadata.mp_physics,
        metadata.spec_bdy_width,
        metadata.active_moist,
        metadata.active_scalar,
    )
    payload += struct.pack("<2f", metadata.parent_dt, metadata.child_dt)
    payload += struct.pack(
        "<6i",
        metadata.nids,
        metadata.nide,
        metadata.njds,
        metadata.njde,
        metadata.nkds,
        metadata.nkde,
    )

    expected: dict[TableKey, np.ndarray] = {}
    fields = _RAW_FIELDS + tuple(
        ("scalar", raw_name, canonical_name)
        for raw_name, canonical_name in scalar_fields
    )
    record_number = 0
    for family, raw_field, canonical_field in fields:
        for side in ("xs", "xe", "ys", "ye"):
            for kind in ("value", "tendency"):
                record_number += 1
                shape = _expected_shape(metadata, canonical_field, side)
                count = int(np.prod(shape))
                values = np.arange(count, dtype=np.float32).reshape(shape)
                values = values + np.float32(record_number * 100 + 1)
                fortran = np.asfortranarray(values.transpose(1, 0, 2))

                payload += b"TABL"
                payload += _fixed(family, 8)
                payload += _fixed(raw_field, 32)
                payload += _fixed(side, 2)
                payload += _fixed(kind, 8)
                payload += struct.pack("<3i", *fortran.shape)
                payload += fortran.astype("<f4", copy=False).tobytes(order="F")
                expected[TableKey(family, canonical_field, side, kind)] = values

    payload += b"DONE" + struct.pack("<i", record_number)
    if extensions:
        for phase, domain_id in (
            ("before", metadata.parent_id),
            ("before", metadata.domain_id),
            ("after", metadata.parent_id),
            ("after", metadata.domain_id),
        ):
            values = np.arange(12, dtype="<f4") + np.float32(domain_id)
            payload += b"PROG" + _fixed(phase, 8)
            payload += struct.pack("<2i", domain_id, values.size)
            payload += values.tobytes()
        dtbc = np.float32(0.0)
        ratio = int(metadata.parent_dt / metadata.child_dt)
        for ordinal in range(1, ratio + 1):
            dtbc = np.float32(dtbc + np.float32(metadata.child_dt))
            values = np.arange(6, dtype="<f4") + dtbc
            payload += b"DBDY"
            payload += struct.pack(
                "<2ifi", metadata.domain_id, ordinal, dtbc, values.size)
            payload += values.tobytes()
    return bytes(payload), metadata, expected


def _read_synthetic(
    tmp_path: Path,
) -> tuple[object, dict[TableKey, np.ndarray]]:
    payload, _, expected = _synthetic_dump_bytes()
    path = tmp_path / "synthetic-force001.bin"
    path.write_bytes(payload)
    return read_dump(path), expected


def _candidate(reference: object) -> dict[TableKey, np.ndarray]:
    return {key: values.copy() for key, values in reference.tables.items()}


def test_reader_parses_committed_fixture_record(tmp_path: Path) -> None:
    encoded = FIXTURE.read_text(encoding="ascii").strip()
    raw = base64.b64decode(encoded, validate=True)
    assert len(raw) == 182
    assert hashlib.sha256(raw).hexdigest() == (
        "b78743580169161b5b0ef9543c0eccccf1611ed805e9f5e44b50ccbcda9ffbc5"
    )
    path = tmp_path / "fixture.bin"
    path.write_bytes(raw)

    dump = read_dump(path, validate=False)
    assert dump.metadata == DumpMetadata(
        1, 2, 1, 1, 10, 1, 0, 1, 60.0, 15.0, 1, 3, 1, 3, 1, 4,
    )
    key = TableKey("scalar", "qni", "xs", "value")
    assert list(dump.tables) == [key]
    np.testing.assert_array_equal(
        dump.tables[key],
        np.arange(1, 7, dtype=np.float32).reshape(3, 2, 1),
    )
    assert dump.tables[key].dtype == np.float32
    assert not dump.tables[key].flags.writeable


def test_reader_strict_round_trip_and_stagger_shapes(tmp_path: Path) -> None:
    reference, expected = _read_synthetic(tmp_path)
    assert len(reference.tables) == 88
    assert set(reference.tables) == set(expected)
    for key, values in expected.items():
        np.testing.assert_array_equal(reference.tables[key], values)

    assert reference.tables[TableKey("state", "u", "ys", "value")].shape == (3, 4, 2)
    assert reference.tables[TableKey("state", "v", "xs", "value")].shape == (3, 3, 2)
    assert reference.tables[TableKey("state", "w", "xe", "tendency")].shape == (4, 2, 2)
    assert reference.tables[TableKey("state", "ph", "ye", "value")].shape == (4, 3, 2)
    assert reference.tables[TableKey("state", "mu", "xs", "value")].shape == (1, 2, 2)

    npz = tmp_path / "round-trip.npz"
    export_npz(reference, npz)
    loaded = load_candidate(npz)
    assert compare(reference, loaded)["pass"] is True
    for key in reference.tables:
        np.testing.assert_array_equal(loaded[key], reference.tables[key])


def test_reader_accepts_extension_records_without_changing_comparator(
    tmp_path: Path,
) -> None:
    payload, _, _ = _synthetic_dump_bytes(extensions=True)
    path = tmp_path / "extended-force001.bin"
    path.write_bytes(payload)
    reference = read_dump(path)

    assert [(sample.phase, sample.domain_id)
            for sample in reference.prognostic_samples] == [
                ("before", 1), ("before", 2),
                ("after", 1), ("after", 2),
            ]
    assert [sample.step_ordinal
            for sample in reference.boundary_clock_samples] == [1, 2, 3, 4]
    assert [sample.dtbc
            for sample in reference.boundary_clock_samples] == [
                15.0, 30.0, 45.0, 60.0]
    assert compare(reference, _candidate(reference))["pass"] is True


def test_reader_requires_all_morrison_number_scalars(tmp_path: Path) -> None:
    payload, _, _ = _synthetic_dump_bytes(_MORRISON_FIELDS[:-1])
    path = tmp_path / "missing-qng.bin"
    path.write_bytes(payload)
    with pytest.raises(DumpFormatError, match="missing mandatory scalars.*qng"):
        read_dump(path)


@pytest.mark.parametrize("corruption", ["truncated", "trailing"])
def test_reader_rejects_truncation_and_trailing_bytes(
    tmp_path: Path, corruption: str,
) -> None:
    payload, _, _ = _synthetic_dump_bytes()
    damaged = payload[:-1] if corruption == "truncated" else payload + b"x"
    path = tmp_path / f"{corruption}.bin"
    path.write_bytes(damaged)
    with pytest.raises(DumpFormatError, match=corruption):
        read_dump(path)


def test_comparator_accepts_in_band_and_rejects_out_of_band(tmp_path: Path) -> None:
    reference, _ = _read_synthetic(tmp_path)
    key = TableKey("state", "t", "xs", "value")
    index = (0, 0, 0)
    wrf = reference.tables[key]
    denominator = abs(float(wrf[index])) + float(np.max(np.abs(wrf)))

    in_band = _candidate(reference)
    in_band[key][index] = np.float32(
        float(wrf[index]) + 0.25 * BAND * denominator,
    )
    accepted = compare(reference, in_band)
    assert accepted["pass"] is True
    assert accepted["value_max_metric"] <= BAND

    out_of_band = _candidate(reference)
    out_of_band[key][index] = np.float32(
        float(wrf[index]) + 4.0 * BAND * denominator,
    )
    rejected = compare(reference, out_of_band)
    assert rejected["pass"] is False
    record = next(item for item in rejected["tables"] if item["key"] == str(key))
    assert record["pass"] is False
    assert record["max_metric"] > BAND


def test_comparator_rejects_missing_and_extra_tables(tmp_path: Path) -> None:
    reference, _ = _read_synthetic(tmp_path)
    missing_key = next(iter(reference.tables))
    missing = _candidate(reference)
    del missing[missing_key]
    missing_report = compare(reference, missing)
    assert missing_report["pass"] is False
    assert missing_report["missing"] == [str(missing_key)]
    assert missing_report["extra"] == []

    extra_key = TableKey("state", "unexpected", "xs", "value")
    extra = _candidate(reference)
    extra[extra_key] = np.ones((1, 1, 1), dtype=np.float32)
    extra_report = compare(reference, extra)
    assert extra_report["pass"] is False
    assert extra_report["missing"] == []
    assert extra_report["extra"] == [str(extra_key)]


def test_candidate_loader_rejects_non_fp32_dtype(tmp_path: Path) -> None:
    path = tmp_path / "float64.npz"
    np.savez(path, **{"state.u.xs.value": np.ones((1, 1, 1), dtype=np.float64)})
    with pytest.raises(DumpFormatError, match="float64.*must be float32"):
        load_candidate(path)


def test_comparator_rejects_nan_and_shape_mismatch(tmp_path: Path) -> None:
    reference, _ = _read_synthetic(tmp_path)
    key = TableKey("state", "u", "xs", "value")

    nan_candidate = _candidate(reference)
    nan_candidate[key][0, 0, 0] = np.nan
    nan_report = compare(reference, nan_candidate)
    assert nan_report["pass"] is False
    nan_record = next(item for item in nan_report["tables"] if item["key"] == str(key))
    assert "non-finite candidate" in nan_record["error"]

    wrong_shape = _candidate(reference)
    wrong_shape[key] = wrong_shape[key][:-1]
    shape_report = compare(reference, wrong_shape)
    assert shape_report["pass"] is False
    shape_record = next(item for item in shape_report["tables"] if item["key"] == str(key))
    assert "candidate shape" in shape_record["error"]


def _available_pristine_wrf_source() -> Path | None:
    roots: list[Path] = []
    if configured := os.environ.get("WRF_V461_SOURCE"):
        roots.append(Path(configured))
    if bundle := os.environ.get("GPUWM_TEST_WRF74_BUNDLE"):
        roots.append(Path(bundle) / "WRF_source_v4.6.1_group")
    roots.append(
        Path.home()
        / "Downloads"
        / "WRF_1974_MP55_reference_bundle"
        / "WRF_source_v4.6.1_group",
    )
    for root in roots:
        matches = True
        for relative, expected in PRISTINE_SOURCE_SHA256.items():
            source = root / relative
            if not source.is_file():
                matches = False
                break
            normalized = source.read_bytes().replace(b"\r\n", b"\n")
            if hashlib.sha256(normalized).hexdigest() != expected:
                matches = False
                break
        if matches:
            return root
    return None


def test_instrument_patch_applies_without_fuzz_when_wrf_is_available(
    tmp_path: Path,
) -> None:
    source_root = _available_pristine_wrf_source()
    if source_root is None:
        pytest.skip("pristine WRF v4.6.1 source tree is not available")
    patch_executable = shutil.which("patch")
    if patch_executable is None:
        pytest.skip("patch executable is not available")

    wrf = tmp_path / "wrf"
    for relative, expected in PRISTINE_SOURCE_SHA256.items():
        normalized = (source_root / relative).read_bytes().replace(
            b"\r\n", b"\n")
        assert hashlib.sha256(normalized).hexdigest() == expected
        destination = wrf / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(normalized)
    result = subprocess.run(
        [patch_executable, "--batch", "--dry-run", "--fuzz=0", "-p1"],
        cwd=wrf,
        input=INSTRUMENT_PATCH.read_text(encoding="ascii"),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "fuzz" not in (result.stdout + result.stderr).lower()
    applied = subprocess.run(
        [patch_executable, "--batch", "--fuzz=0", "-p1"],
        cwd=wrf,
        input=INSTRUMENT_PATCH.read_text(encoding="ascii"),
        text=True,
        capture_output=True,
        check=False,
    )
    assert applied.returncode == 0, applied.stdout + applied.stderr
    force_source = (wrf / "share/mediation_force_domain.F").read_text(
        encoding="ascii")
    assert "SUBROUTINE gpuwm_dump_force_inputs" in force_source
    assert "'INPT', phase, domain_i" in force_source
    assert force_source.rstrip().endswith("END SUBROUTINE med_force_domain")


def test_f18_theta_successor_treatment_is_t_family_only() -> None:
    """F18: state.t.*.tendency compares at the FP32 successor scale
    E = fl32(V + fl32(cdt*T)) under the unchanged band; every other
    field keeps the raw metric.  The perturbation below FAILS raw
    (metric ~1e-2) and PASSES at successor scale (~3e-7), so the test
    pins that the treatment (a) changed the theta outcome and (b) did
    NOT leak to u."""
    from dump_reader import DumpMetadata, NestForceDump

    metadata = DumpMetadata(2, 2, 1, 1, 10, 5, 6, 4, 60.0, 15.0,
                            1, 4, 1, 4, 1, 4)
    value = np.full((2, 3, 1), 1.0e5, dtype=np.float32)
    tendency = np.full((2, 3, 1), 0.05, dtype=np.float32)
    tables = {
        TableKey("state", "t", "xs", "value"): value.copy(),
        TableKey("state", "t", "xs", "tendency"): tendency.copy(),
        TableKey("state", "u", "xs", "value"): value.copy(),
        TableKey("state", "u", "xs", "tendency"): tendency.copy(),
    }
    reference = NestForceDump(metadata=metadata, tables=tables)
    delta = np.float32(1.0e-3)

    theta_cand = {key: values.copy() for key, values in tables.items()}
    theta_cand[TableKey("state", "t", "xs", "tendency")][0, 0, 0] += delta
    report = compare(reference, theta_cand)
    by_key = {rec["key"]: rec for rec in report["tables"]}
    assert by_key["state.t.xs.tendency"]["treatment"] == "f18_theta_successor"
    assert by_key["state.t.xs.tendency"]["parent_dt"] == 60.0
    assert by_key["state.t.xs.tendency"]["pass"] is True
    assert 0.0 < by_key["state.t.xs.tendency"]["max_metric"] < BAND
    assert report["pass"] is True

    u_cand = {key: values.copy() for key, values in tables.items()}
    u_cand[TableKey("state", "u", "xs", "tendency")][0, 0, 0] += delta
    report_u = compare(reference, u_cand)
    by_key_u = {rec["key"]: rec for rec in report_u["tables"]}
    assert "treatment" not in by_key_u["state.u.xs.tendency"]
    assert by_key_u["state.u.xs.tendency"]["pass"] is False
    assert report_u["pass"] is False
