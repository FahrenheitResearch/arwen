"""Synthetic byte-level integration tests for the generic stack frontend."""
from dataclasses import FrozenInstanceError
from pathlib import Path
import hashlib
import json
import struct
import subprocess
import numpy as np
import pytest
from gpuwm.grib2_stack import stack_grib2, read_stack
from gpuwm.ingest.cpu_backend import CpuPreprocessBackend, CPU_BRIDGE_ENV, _library_names
from types import SimpleNamespace
import ctypes
import os
import os
from gpuwm import bridges

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = Path(os.environ.get(CPU_BRIDGE_ENV, ROOT / "tools/grib1_bridge/target/release" / _library_names()[0]))

@pytest.fixture(autouse=True)
def use_this_export_library(monkeypatch):
    monkeypatch.setenv(CPU_BRIDGE_ENV, str(LIBRARY))

def section(number, length):
    value = bytearray(length)
    struct.pack_into(">I", value, 0, length)
    value[4] = number
    return value

def message(*, pdt=40, level=1, discipline=0, category=None, parameter=None, constituent=5, bitmap=False, step=3, scan=0, lat1=40, lat2=39, lon1=0, lon2=1, gdt=0, coeff=(1000.0, .5)):
    # Envelope layout follows grib-core/src/grib2/unpack.rs:3131.
    s1 = section(1, 21)
    struct.pack_into(">H", s1, 12, 2026)
    s1[14:16] = bytes([9, 30])
    s3 = section(3, 72)
    struct.pack_into(">I", s3, 6, 4)
    struct.pack_into(">H", s3, 12, gdt)
    s3[14] = 6
    struct.pack_into(">II", s3, 30, 2, 2)
    for offset, value in [(46,lat1), (50,lon1), (55,lat2), (59,lon2)]:
        integer = round(value*1.e6)
        struct.pack_into(">I", s3, offset, abs(integer) | (0x80000000 if integer<0 else 0))
    struct.pack_into(">II", s3, 63, 1000000, 1000000)
    s3[71] = scan
    shift = 2 if pdt == 40 else 0
    s4 = section(4, 34+shift+4*len(coeff))
    struct.pack_into(">H", s4, 5, len(coeff))
    struct.pack_into(">H", s4, 7, pdt)
    s4[9:11] = bytes([category if category is not None else (210 if discipline == 192 else 20),
                         parameter if parameter is not None else (1 if discipline == 192 else 2)])
    if shift:
        struct.pack_into(">H", s4, 11, constituent)
    s4[17+shift] = 1
    struct.pack_into(">I", s4, 18+shift, step)
    s4[22+shift] = 105
    struct.pack_into(">I", s4, 24+shift, level)
    s4[28+shift] = 255
    for i,c in enumerate(coeff):
        struct.pack_into(">f", s4, 34+shift+4*i, c)
    s5 = section(5,12)
    count = 3 if bitmap else 4
    struct.pack_into(">I", s5, 5, count)
    struct.pack_into(">H", s5, 9, 4)
    s5[11] = 2
    s6 = section(6, 7 if bitmap else 6)
    s6[5] = 0 if bitmap else 255
    if bitmap:
        s6[6] = 0b10110000
    s7 = section(7, 5+8*count)
    values = [level+.123456789+i for i in range(4)]
    if bitmap:
        values.pop(1)
    for i,value in enumerate(values):
        struct.pack_into(">d", s7, 5+8*i, value)
    body = b"".join([s1,s3,s4,s5,s6,s7])
    return b"GRIB" + bytes([0,0,discipline,2]) + struct.pack(">Q",20+len(body)) + body + b"7777"

def selector(key="gas", discipline=0, levels=None, constituent_type=None):
    return dict(key=key, discipline=discipline, category=210 if discipline == 192 else 20, parameter=1 if discipline == 192 else 2,
                constituent_type=constituent_type, aerosol_type=None, level_type=105, levels=levels)

def decode(tmp_path, records, select=None, name="stacks"):
    source = tmp_path / f"{name}.grib2"
    source.write_bytes(b"".join(records))
    return stack_grib2(source, select or [selector()], tmp_path/name)

def test_interleaved_stacks_order_metadata_axes_and_lazy_mapping(tmp_path):
    fields = decode(tmp_path, [message(level=137,bitmap=True), message(pdt=0,discipline=192,level=2),
                               message(level=1), message(pdt=0,discipline=192,level=1)],
                    [selector(levels=[137,1], constituent_type=5),selector("local",192)])
    assert [f.key for f in fields] == ["gas", "local"]
    gas,local = fields
    assert gas.levels == (1.0,137.0) and local.levels == (1.0,2.0)
    assert gas.valid_time.isoformat() == "2026-09-30T03:00:00+00:00"
    assert gas.coordinate_values == (1000.0,.5)
    assert (gas.pdt,local.pdt) == (40,0)
    np.testing.assert_array_equal(gas.latitude,[40,39])
    np.testing.assert_array_equal(gas.longitude,[0,1])
    assert not gas.latitude.flags.writeable and not gas.longitude.flags.writeable
    assert isinstance(gas.array,np.memmap) and not gas.array.flags.writeable
    expected = np.array([[[1.123456789,2.123456789],[3.123456789,4.123456789]],
                         [[137.123456789,np.nan],[139.123456789,140.123456789]]],dtype=np.float32)
    np.testing.assert_array_equal(gas.array.view(np.uint32),expected.view(np.uint32))
    with pytest.raises(FrozenInstanceError):
        gas.key = "other"
    manifest = json.loads((tmp_path/"stacks/stack.json").read_text())
    row = manifest["outputs"][0]
    assert row["sha256"] == hashlib.sha256(gas.file.read_bytes()).hexdigest()
    assert row["selector_values"] == [dict(discipline=0,category=20,parameter=2,constituent_type=5,aerosol_type=None,level_type=105)]
    assert row["reference_time"] == "2026-09-30T00:00:00Z"
    assert (row["forecast_step"],row["forecast_unit"]) == (3,1)
    assert row["grid"] == dict(template=0,nx=2,ny=2,lat1=40,lat2=39,lon1=0,lon2=1,dlat=1,dlon=1,scan_mode=0)

@pytest.mark.parametrize("scan,lat1,lat2,lon1,lon2", [(64,39,40,0,1),(128,40,39,1,0),(0,40,39,359,0)])
def test_scan_order_is_preserved(tmp_path,scan,lat1,lat2,lon1,lon2):
    field, = decode(tmp_path,[message(scan=scan,lat1=lat1,lat2=lat2,lon1=lon1,lon2=lon2)])
    assert field.scan_mode == scan
    np.testing.assert_array_equal(field.latitude,[lat1,lat2])
    np.testing.assert_array_equal(field.longitude,[lon1,lon1+(-1 if scan&128 else 1)])
    np.testing.assert_array_equal(field.array.ravel(),np.array([1.123456789,2.123456789,3.123456789,4.123456789],dtype=np.float32))

@pytest.mark.parametrize("records,select,error", [
    ([message()], [selector(constituent_type=8)], "the stack would be empty"),
    ([message(),message()], [selector()], "duplicate.*level"),
    ([message()], [selector(levels=[1,2])], "requested level 2 missing"),
    ([message(),message(pdt=0,level=2)], [selector()], "mixed PDTs"),
    ([message(),message(level=2,coeff=(2,.5))], [selector()], "coordinate values"),
    ([message(),message(level=2,lat1=41)], [selector()], "disagree on grid"),
    ([message(gdt=99)], [selector()], "grid template 99"),
])
def test_named_refusals_create_no_output(tmp_path, records, select, error):
    with pytest.raises(ValueError,match=error):
        decode(tmp_path,records,select)
    assert not (tmp_path/"stacks").exists()

def test_multiple_forecast_times(tmp_path):
    fields = decode(tmp_path,[message(step=6),message(step=3)])
    assert [f.valid_time.hour for f in fields] == [3,6]
    assert fields[0].file.name == "gas__2026-09-30T030000Z.f32le"

def test_existing_output_refuses_without_overwrite(tmp_path):
    decode(tmp_path,[message()])
    with pytest.raises(ValueError,match="overwrite"):
        decode(tmp_path,[message(level=2)])

def test_reader_checks_hash_and_size(tmp_path):
    field, = decode(tmp_path,[message()])
    data = field.file.read_bytes()
    field.file.write_bytes(b"\x00"*len(data))
    with pytest.raises(ValueError,match="sha256"):
        read_stack(tmp_path/"stacks")
    field.file.write_bytes(data[:-1])
    with pytest.raises(ValueError,match="byte count"):
        read_stack(tmp_path/"stacks")

def test_stack_uses_the_loaded_cpu_library(monkeypatch, tmp_path):
    calls = []
    def native(self, source, spec, output):
        calls.append((self.path, source, json.loads(spec.read_text()), output))
    monkeypatch.setattr(CpuPreprocessBackend, "grib2_stack", native)
    monkeypatch.setattr("gpuwm.grib2_stack.read_stack", lambda path: ())
    source = tmp_path / "input.grib2"
    assert stack_grib2(source, [selector()], tmp_path / "output") == ()
    assert len(calls) == 1
    assert calls[0][0] == LIBRARY.resolve()
    assert calls[0][2] == {"schema": "gpuwm-grib2-stack-v1", "select": [selector()]}


def test_old_library_names_the_missing_stack_symbol(tmp_path):
    backend = CpuPreprocessBackend.__new__(CpuPreprocessBackend)
    backend._library = SimpleNamespace()
    with pytest.raises(RuntimeError, match="missing gpuwm_grib2_stack; rebuild tools/grib1_bridge"):
        backend.grib2_stack(tmp_path / "input", tmp_path / "spec", tmp_path / "output")


def test_c_export_reports_null_and_invalid_utf8_paths(tmp_path):
    backend = CpuPreprocessBackend(LIBRARY)
    function = backend._chem_symbol("gpuwm_grib2_stack", [ctypes.c_char_p] * 3)
    error = backend._library.gpuwm_bridge_last_error
    error.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    error.restype = ctypes.c_size_t
    for path, expected in [(None, "must not be null"), (b"\xff", "not valid UTF-8"), (b"", "empty")]:
        assert function(path, b"spec.json", b"output") != 0
        buffer = ctypes.create_string_buffer(8192)
        count = error(buffer, len(buffer))
        assert expected in buffer.raw[:count].decode("utf-8")


def test_unicode_paths_are_passed_as_utf8(tmp_path):
    directory = tmp_path / "\u00e9\u6c14"
    directory.mkdir()
    field, = decode(directory, [message()])
    assert field.file.parent.parent == directory


def test_inventory_header_append_is_compatible_with_named_consumers(tmp_path):
    source = tmp_path/"inventory.grib2"
    source.write_bytes(message())
    binary = LIBRARY.with_name(bridges.executable_name("grib2_inventory"))
    result = subprocess.run([str(binary),str(source)],text=True,capture_output=True,check=True)
    lines = [line for line in result.stdout.splitlines() if not line.startswith("#")]
    columns = lines[0].split("\t")
    assert columns[-5:] == ["pv","constituent_type","aerosol_type","aerosol_size","optical_wavelength"]
    row = dict(zip(columns,lines[1].split("\t"),strict=True))
    assert row["constituent_type"] == "5" and row["aerosol_size"] == "-"
    assert (row["level_type"],row["level_value"],row["pv"]) == ("105","1","1000,0.5")
    from gpuwm.mapped_source import _parse_grib2_tsv
    parsed = _parse_grib2_tsv(lines, required=frozenset(["index","level_type","pv"]),label="test")
    assert parsed[0]["constituent_type"] == "5"

def test_eccodes_committed_oracle_identifiers_coefficients_and_stack_bits(tmp_path):
    """Check every record against the independent ecCodes 2.48 fixture oracle."""
    import csv
    fixture = ROOT / "tests/data/chem_cams"
    for line in (fixture / "SHA256SUMS").read_text().splitlines():
        digest, filename = line.split("  ", 1)
        assert hashlib.sha256((fixture / filename).read_bytes()).hexdigest() == digest
    manifest = json.loads((fixture / "SYNTHETIC-cams-20250730T00.manifest.json").read_text())
    assert manifest["encoder"] == "ecCodes 2.48.0"
    coefficients = json.loads((fixture / "ifs_l137_half_levels.json").read_text())
    expected_pv = np.asarray([row[1] for row in coefficients] +
                             [row[2] for row in coefficients], dtype=np.float32).astype(np.float64)
    inventory = LIBRARY.with_name(bridges.executable_name("grib2_inventory"))
    total_records = 0
    total_points = 0
    mismatches = []
    for kind in ["ml", "sl"]:
        source = fixture / f"SYNTHETIC-cams-{kind}-20250730T00.grib2"
        assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["files"][source.name]
        metadata = json.loads(source.with_suffix(".eccodes-meta.json").read_text())
        oracle = np.load(source.with_suffix(".eccodes-values.npy"), mmap_mode="r")
        completed = subprocess.run([str(inventory), str(source)], capture_output=True,
                                   text=True, check=True)
        rows = list(csv.DictReader([line for line in completed.stdout.splitlines()
                                   if not line.startswith("#")], delimiter="\t"))
        assert len(rows) == len(metadata) == oracle.shape[0]
        for row, expected in zip(rows, metadata, strict=True):
            for column, key in [("index", "index"), ("discipline", "discipline"),
                                ("category", "category"), ("parameter", "parameter"),
                                ("pdt", "pdt"), ("level_type", "level_type"),
                                ("level_value", "level"), ("forecast_time", "step")]:
                actual = float(row[column])
                if actual != expected[key]:
                    mismatches.append((source.name, expected["index"], column, actual, expected[key]))
            actual_type = None if row["constituent_type"] == "-" else int(row["constituent_type"])
            if actual_type != expected["constituent_type"]:
                mismatches.append((source.name, expected["index"], "constituent_type",
                                   actual_type, expected["constituent_type"]))
            pv = [] if row["pv"] == "-" else [float(value) for value in row["pv"].split(",")]
            assert len(pv) == expected["nv"], (source.name, expected["index"], "NV")
            np.testing.assert_array_equal(pv, expected_pv if kind == "ml" else [],
                                          err_msg=f"{source.name} record {expected['index']} coefficients")
            assert int(row["forecast_unit"]) == 1
            assert int(row["gdt"]) == 0
            assert row["scan_mode"] == "0x00"
            assert float(row["dx"]) == float(row["dy"]) == .4
            assert int(row["nx"]) == int(row["ny"]) == 3
        identities = sorted(set((row["discipline"], row["category"], row["parameter"],
                                 row["constituent_type"]) for row in metadata), key=str)
        selectors = [dict(key=f"field_{i}", discipline=d, category=c, parameter=p,
                          constituent_type=t, aerosol_type=None,
                          level_type=105 if kind == "ml" else 1, levels=None)
                     for i, (d, c, p, t) in enumerate(identities)]
        fields = stack_grib2(source, selectors, tmp_path / kind)
        stacks = json.loads((tmp_path / kind / "stack.json").read_text())["outputs"]
        assert len(fields) == 2 * len(selectors)
        for field, stack in zip(fields, stacks, strict=True):
            selected = selectors[int(field.key.split("_")[1])]
            matching = [row for row in metadata if
                        all(row[key] == selected[key] for key in
                            ["discipline", "category", "parameter", "constituent_type", "level_type"])
                        and row["step"] == field.valid_time.hour]
            matching.sort(key=lambda row: row["level"])
            expected = oracle[[row["index"] for row in matching]].astype(np.float32).reshape(field.array.shape)
            actual_bits = field.array.view(np.uint32)
            expected_bits = expected.view(np.uint32)
            different = np.argwhere(actual_bits != expected_bits)
            assert not different.size, (source.name, field.key, field.valid_time,
                                        "bit mismatches", different.tolist())
            assert field.levels == tuple(row["level"] for row in matching)
            assert field.pdt == matching[0]["pdt"]
            assert stack["selector_values"] == [{key: selected[key] for key in
                ["discipline", "category", "parameter", "constituent_type", "aerosol_type", "level_type"]}]
            assert stack["forecast_step"] == matching[0]["step"]
            assert stack["forecast_unit"] == 1
            np.testing.assert_array_equal(field.coordinate_values, expected_pv if kind == "ml" else [])
            assert field.latitude[0] > field.latitude[-1]
            assert len(field.levels) == (137 if kind == "ml" else 1)
            total_records += len(matching)
            total_points += expected.size
    assert mismatches == [], mismatches
    # The committed fixture: the four PDT 4.40 gases and specific humidity
    # on 137 levels at two steps, plus two single-level surface pressures,
    # on a 3 x 3 grid (tests/data/chem_cams/README.md).
    assert total_records == 1372
    assert total_points == 12348


def test_inventory_dump_surface_level_metadata_agree(tmp_path):
    import csv
    source = ROOT / "tests/data/chem_cams/SYNTHETIC-cams-sl-20250730T00.grib2"
    binary = LIBRARY.with_name(bridges.executable_name("grib2_dump"))
    output = tmp_path / "dump"
    subprocess.run([str(binary), str(source), str(output), "0", "1"], check=True,
                   capture_output=True, text=True)
    with (output / "metadata.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert [row["level_value"] for row in rows] == ["0", "0"]
    oracle = np.load(source.with_suffix(".eccodes-values.npy"))
    for row in rows:
        values = np.fromfile(output / row["filename"], dtype="<f8")
        np.testing.assert_array_equal(values.view(np.uint64),
                                      oracle[int(row["index"])].view(np.uint64))
