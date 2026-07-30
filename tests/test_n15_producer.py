"""CPU-only plumbing tests for the independent N1.5 candidate producer."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]
TOOLS = ROOT / "tools" / "wrf_instrumented"
sys.path.insert(0, str(TOOLS))

from compare_nest_force import compare, load_candidate  # noqa: E402
from dump_reader import (  # noqa: E402
    DumpMetadata,
    ForceInputKey,
    TableKey,
    read_dump,
)
from produce_nest_force_candidate import (  # noqa: E402
    ProducerError,
    _clock_values,
    _repo_root,
    _validate_repository_layout,
    build_candidate,
    produce,
    read_pinned_case,
)
from gpuwm.core.nest_interp import register_nest  # noqa: E402
from gpuwm.verify.npref import np_nest_force  # noqa: E402


_INVENTORY = (
    ("state", "u"), ("state", "v"), ("state", "w"),
    ("state", "t"), ("state", "ph"), ("state", "mu"),
    ("moist", "qv"),
    ("scalar", "qni"), ("scalar", "qns"),
    ("scalar", "qnr"), ("scalar", "qng"),
)
_AUX = ("mub", "t_init", "c1h", "c2h", "c1f", "c2f",
        "msft", "msfu", "msfv")
_SIDE_SUFFIX = {"west": "xs", "east": "xe",
                "south": "ys", "north": "ye"}


def _fixed(value: str, size: int) -> bytes:
    return value.encode("ascii").ljust(size, b" ")


def _state(nx: int, ny: int, nz: int, offset: float):
    def values(shape, scale, shift=0.0):
        return (np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
                * np.float32(scale) + np.float32(offset + shift))

    mub = values((ny, nx), 0.003, 8.0)
    mup = values((ny, nx), 0.0002, 0.2)
    t_init = values((nz, ny, nx), 0.0001, 296.0)
    t_wrf = values((nz, ny, nx), 0.0007, -2.0)
    thp = np.asarray(
        np.asarray(t_wrf + np.float32(300.0), dtype=np.float32) - t_init,
        dtype=np.float32,
    )
    state = SimpleNamespace(
        mup=mup,
        u=values((nz, ny, nx + 1), 0.0005, 1.0),
        v=values((nz, ny + 1, nx), -0.0004, -1.0),
        w=values((nz + 1, ny, nx), 0.00003, 0.1),
        thp=thp,
        php=values((nz + 1, ny, nx), 0.02, 3.0),
        qv=values((nz, ny, nx), 1.0e-7, 0.01),
        qni=values((nz, ny, nx), 2.0e-5, 1.0),
        qns=values((nz, ny, nx), 3.0e-5, 2.0),
        qnr=values((nz, ny, nx), 4.0e-5, 3.0),
        qng=values((nz, ny, nx), 5.0e-5, 4.0),
        mub2d=mub,
        thb=t_init,
        c1h=np.asarray([0.8, 0.6, 0.4], dtype=np.float32),
        c2h=np.asarray([1.0, 1.5, 2.0], dtype=np.float32),
        c1f=np.asarray([1.0, 0.75, 0.5, 0.25], dtype=np.float32),
        c2f=np.asarray([0.0, 0.5, 1.0, 1.5], dtype=np.float32),
        msft=values((ny, nx), 0.00001, 1.1),
        msfu=values((ny, nx + 1), 0.00001, 1.2),
        msfv=values((ny + 1, nx), 0.00001, 1.3),
        has_msf=True,
    )
    raw = {
        ("state", "mu"): state.mup,
        ("state", "u"): state.u,
        ("state", "v"): state.v,
        ("state", "w"): state.w,
        ("state", "t"): t_wrf,
        ("state", "ph"): state.php,
        ("moist", "qv"): state.qv,
        ("scalar", "qni"): state.qni,
        ("scalar", "qns"): state.qns,
        ("scalar", "qnr"): state.qnr,
        ("scalar", "qng"): state.qng,
        ("aux", "mub"): state.mub2d,
        ("aux", "t_init"): state.thb,
        ("aux", "c1h"): state.c1h,
        ("aux", "c2h"): state.c2h,
        ("aux", "c1f"): state.c1f,
        ("aux", "c2f"): state.c2f,
        ("aux", "msft"): state.msft,
        ("aux", "msfu"): state.msfu,
        ("aux", "msfv"): state.msfv,
    }
    return state, raw


def _tables(parent, child) -> dict[TableKey, np.ndarray]:
    registrations = {
        stagger: register_nest(
            nri=3, nrj=3, i_parent_start=4, j_parent_start=4,
            child_nx=30, child_ny=30, parent_nx=16, parent_ny=16,
            stagger="" if stagger == "m" else stagger, wrapper="bdy")
        for stagger in ("m", "x", "y")
    }
    mirror = np_nest_force(
        parent, child, registrations,
        field_names=tuple(field for _, field in _INVENTORY),
        parent_dt_fp32=np.float32(9.0), parent_interval_ticks=3,
        spec_zone=1, relax_zone=1, spec_bdy_width=2,
        dtype=np.float32)
    family = {field: family for family, field in _INVENTORY}
    result = {}
    for field, sides in mirror.items():
        for side, suffix in _SIDE_SUFFIX.items():
            for index, kind in enumerate(("value", "tendency")):
                values = sides[side][index]
                if side in {"south", "north"}:
                    values = np.swapaxes(values, 1, 2)
                result[TableKey(family[field], field, suffix, kind)] = \
                    np.ascontiguousarray(values, dtype=np.float32)
    return result


def _fortran_payload(values: np.ndarray) -> tuple[tuple[int, ...], bytes]:
    if values.ndim == 3:
        fortran = np.asfortranarray(values.transpose(2, 0, 1))
    elif values.ndim == 2:
        fortran = np.asfortranarray(values.transpose(1, 0))
    else:
        fortran = np.asfortranarray(values)
    return fortran.shape, fortran.astype("<f4", copy=False).tobytes(order="F")


def _prog(raw, nx: int, ny: int, nz: int) -> np.ndarray:
    mu = raw[("state", "mu")]
    t = raw[("state", "t")]
    u = raw[("state", "u")]
    v = raw[("state", "v")]
    w = raw[("state", "w")]
    ph = raw[("state", "ph")]
    return np.asarray([
        mu[0, 0], t[0, 0, 0], u[0, 0, 0], v[0, 0, 0],
        w[0, 0, 0], ph[0, 0, 0],
        mu[ny - 1, nx - 1], t[nz - 1, ny - 1, nx - 1],
        u[nz - 1, ny - 1, nx - 1],
        v[nz - 1, ny - 1, nx - 1],
        w[nz - 1, ny - 1, nx - 1],
        ph[nz - 1, ny - 1, nx - 1],
    ], dtype=np.float32)


def _dump_bytes(*, missing: ForceInputKey | None = None) -> bytes:
    parent, parent_raw = _state(16, 16, 3, 0.5)
    child, child_raw = _state(30, 30, 3, 1.5)
    tables = _tables(parent, child)
    metadata = DumpMetadata(
        1, 2, 1, 1, 10, 2, 1, 4, 9.0, 3.0,
        1, 31, 1, 31, 1, 4)
    payload = bytearray(b"GPUWM-N1P5-DUMP1")
    payload += struct.pack("<i8i2f6i", 0x01020304,
                           metadata.schema_version, metadata.domain_id,
                           metadata.parent_id, metadata.force_ordinal,
                           metadata.mp_physics, metadata.spec_bdy_width,
                           metadata.active_moist, metadata.active_scalar,
                           metadata.parent_dt, metadata.child_dt,
                           metadata.nids, metadata.nide,
                           metadata.njds, metadata.njde,
                           metadata.nkds, metadata.nkde)

    for domain_id, raw, shape in (
            (1, parent_raw, (16, 16, 3)),
            (2, child_raw, (30, 30, 3))):
        values = _prog(raw, *shape)
        payload += b"PROG" + _fixed("before", 8)
        payload += struct.pack("<2i", domain_id, values.size)
        payload += values.astype("<f4", copy=False).tobytes()

    for domain_id, raw in ((1, parent_raw), (2, child_raw)):
        for (family, field), values in raw.items():
            key = ForceInputKey("before", domain_id, family, field)
            if key == missing:
                continue
            dims, data = _fortran_payload(values)
            padded = dims + (0,) * (3 - len(dims))
            payload += b"INPT" + _fixed("before", 8)
            payload += struct.pack("<i", domain_id)
            payload += _fixed(family, 8) + _fixed(field, 32)
            payload += struct.pack("<i3i", values.ndim, *padded)
            payload += data

    for key, values in sorted(tables.items()):
        fortran = np.asfortranarray(values.transpose(1, 0, 2))
        payload += b"TABL" + _fixed(key.family, 8) + _fixed(key.field, 32)
        payload += _fixed(key.side, 2) + _fixed(key.kind, 8)
        payload += struct.pack("<3i", *fortran.shape)
        payload += fortran.astype("<f4", copy=False).tobytes(order="F")
    payload += b"DONE" + struct.pack("<i", len(tables))

    for domain_id, raw, shape in (
            (1, parent_raw, (16, 16, 3)),
            (2, child_raw, (30, 30, 3))):
        values = _prog(raw, *shape)
        payload += b"PROG" + _fixed("after", 8)
        payload += struct.pack("<2i", domain_id, values.size)
        payload += values.astype("<f4", copy=False).tobytes()

    dtbc = np.float32(0.0)
    for ordinal in range(1, 4):
        dtbc = np.float32(dtbc + np.float32(3.0))
        values = _clock_values(tables, dtbc)
        payload += b"DBDY" + struct.pack(
            "<2ifi", 2, ordinal, dtbc, values.size)
        payload += values.astype("<f4", copy=False).tobytes()
    return bytes(payload)


def _namelist(path: Path, *, child_e_we: int = 31) -> Path:
    path.write_text(f"""&domains
 time_step = 9,
 max_dom = 2,
 e_we = 17, {child_e_we},
 e_sn = 17, 31,
 e_vert = 4, 4,
 grid_id = 1, 2,
 parent_id = 0, 1,
 i_parent_start = 1, 4,
 j_parent_start = 1, 4,
 parent_grid_ratio = 1, 3,
 parent_time_step_ratio = 1, 3,
/
&physics
 mp_physics = 10, 10,
/
&dynamics
 use_theta_m = 0,
/
&bdy_control
 spec_bdy_width = 2,
 spec_zone = 1,
 relax_zone = 1,
/
""", encoding="ascii")
    return path


def _flip_fp32_lsb(values: np.ndarray, index: int = 0) -> np.ndarray:
    damaged = np.array(values, dtype=np.float32, copy=True, order="C")
    damaged.view(np.uint32).reshape(-1)[index] ^= np.uint32(1)
    return damaged


def test_producer_repo_root_is_module_anchored_and_layout_error_names_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    misleading_cwd = tmp_path / "caller"
    (misleading_cwd / "gpuwm").mkdir(parents=True)
    monkeypatch.chdir(misleading_cwd)
    assert _repo_root() == ROOT.resolve()

    fake_script = tmp_path / "foreign" / "produce_nest_force_candidate.py"
    with pytest.raises(RuntimeError) as caught:
        _validate_repository_layout(tmp_path / "repo", fake_script)
    message = str(caught.value)
    assert str(tmp_path / "repo" / "gpuwm" / "__init__.py") in message
    assert str(fake_script.with_name("dump_reader.py")) in message


def test_synthetic_mirror_dump_produces_passing_canonical_npz(
    tmp_path: Path,
) -> None:
    dump_path = tmp_path / "synthetic-force.bin"
    dump_path.write_bytes(_dump_bytes())
    namelist = _namelist(tmp_path / "namelist.input")
    candidate_path = tmp_path / "candidate.npz"

    report = produce(
        dump_path,
        candidate_path,
        namelist_path=namelist,
        successor_metrics=True,
    )
    reference = read_dump(dump_path)
    candidate = load_candidate(candidate_path)

    assert report["table_count"] == len(reference.tables) == 88
    assert report["registered_comparator"]["pass"] is True
    assert report["successor_metrics"]["successor_pass"] is True
    assert report["successor_metrics"]["stored_table_recurrence_pass"] is True
    assert set(report["successor_metrics"]["tables"]) == {
        f"state.t.{side}.tendency" for side in ("xs", "xe", "ys", "ye")
    }
    assert compare(reference, candidate)["pass"] is True
    assert set(candidate) == set(reference.tables)
    assert all(values.dtype == np.float32 for values in candidate.values())
    with np.load(candidate_path, allow_pickle=False) as archive:
        assert all(not key.startswith("__") for key in archive.files)
        assert all(str(TableKey.parse(key)) == key for key in archive.files)


def test_producer_rejects_pinned_geometry_mismatch(tmp_path: Path) -> None:
    dump_path = tmp_path / "synthetic-force.bin"
    dump_path.write_bytes(_dump_bytes())
    namelist = _namelist(tmp_path / "namelist.input", child_e_we=32)
    with pytest.raises(ProducerError, match="pinned geometry mismatch.*nide"):
        produce(dump_path, tmp_path / "candidate.npz", namelist_path=namelist)


def test_producer_names_missing_full_input_field(tmp_path: Path) -> None:
    missing = ForceInputKey("before", 2, "scalar", "qng")
    dump_path = tmp_path / "missing-input.bin"
    dump_path.write_bytes(_dump_bytes(missing=missing))
    namelist = _namelist(tmp_path / "namelist.input")
    with pytest.raises(
        ProducerError,
        match=r"missing force input field\(s\): before\.d02\.scalar\.qng",
    ):
        produce(dump_path, tmp_path / "candidate.npz", namelist_path=namelist)


def test_producer_rejects_prog_bit_mismatch_by_name(tmp_path: Path) -> None:
    dump_path = tmp_path / "prog-corruption.bin"
    dump_path.write_bytes(_dump_bytes())
    reference = read_dump(dump_path)
    samples = list(reference.prognostic_samples)
    samples[0] = replace(samples[0], values=_flip_fp32_lsb(samples[0].values))
    damaged = replace(reference, prognostic_samples=tuple(samples))
    case = read_pinned_case(_namelist(tmp_path / "namelist.input"))

    with pytest.raises(
        ProducerError,
        match=r"before d01 prognostic sample mismatch.*PROG",
    ):
        build_candidate(damaged, case)


def test_producer_rejects_boundary_clock_bit_mismatch_by_name(
    tmp_path: Path,
) -> None:
    dump_path = tmp_path / "clock-corruption.bin"
    dump_path.write_bytes(_dump_bytes())
    reference = read_dump(dump_path)
    samples = list(reference.boundary_clock_samples)
    samples[1] = replace(samples[1], values=_flip_fp32_lsb(
        samples[1].values, index=2))
    damaged = replace(reference, boundary_clock_samples=tuple(samples))
    case = read_pinned_case(_namelist(tmp_path / "namelist.input"))

    with pytest.raises(
        ProducerError,
        match=r"boundary-clock oracle mismatch at ordinal 2.*DBDY",
    ):
        build_candidate(damaged, case)


def test_producer_rejects_extra_force_input_by_full_name(tmp_path: Path) -> None:
    dump_path = tmp_path / "extra-input.bin"
    dump_path.write_bytes(_dump_bytes())
    reference = read_dump(dump_path)
    inputs = dict(reference.force_inputs)
    extra = ForceInputKey("before", 2, "aux", "unexpected")
    inputs[extra] = np.ones((1,), dtype=np.float32)
    damaged = replace(reference, force_inputs=inputs)
    case = read_pinned_case(_namelist(tmp_path / "namelist.input"))

    with pytest.raises(
        ProducerError,
        match=r"unexpected force input field\(s\): before\.d02\.aux\.unexpected",
    ):
        build_candidate(damaged, case)


def test_producer_rejects_incomplete_candidate_inventory_by_table_name(
    tmp_path: Path,
) -> None:
    dump_path = tmp_path / "inventory-corruption.bin"
    dump_path.write_bytes(_dump_bytes())
    reference = read_dump(dump_path)
    tables = dict(reference.tables)
    missing = TableKey("state", "u", "xs", "value")
    del tables[missing]
    damaged = replace(reference, tables=tables)
    case = read_pinned_case(_namelist(tmp_path / "namelist.input"))

    with pytest.raises(
        ProducerError,
        match=r"candidate table inventory mismatch.*state\.u\.xs\.value",
    ):
        build_candidate(damaged, case)
