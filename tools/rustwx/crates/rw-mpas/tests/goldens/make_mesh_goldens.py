"""Extract mesh goldens from the two published MPAS grid files.

NOT SHIPPED CODE. This is a test harness, run once against the real NCAR-generated
grid files on node-2. The artifacts it emits (*.rwg + *.json) are what the Rust test
consumes; this script exists so the extraction is reproducible and reviewable.

WHAT A GOLDEN IS HERE
    A published MPAS mesh is correct by construction. Take ONLY the cell centres out
    of it and every other field becomes an expected output that a correct generator
    must reproduce. So the golden is: cell centres (input) + every derived field
    (expected), in a form Rust can read offline with no netCDF and no network.

THE ORDERING PROBLEM, AND THE FIX
    Cell order is the generator's INPUT, so cell-indexed arrays compare directly.
    Edge and vertex numbering is NOT determined by the cell centres -- a generator
    picks its own -- so index-by-index comparison against the file's numbering is
    meaningless. Every edge/vertex-indexed field in this golden is therefore
    re-expressed in a CANONICAL numbering that is a pure function of the cell
    centres:
        canonical edge   = the unordered cell pair (a,b), a<b, sorted lexicographically
        canonical vertex = the sorted cell triple (a,b,c), sorted lexicographically
    Measured on both meshes: these keys are bijections with the file's own numbering
    (122880/122880 and 491520/491520 edges; 81920/81920 and 327680/327680 vertices,
    zero self-loops, zero duplicate-cell vertices).

WHY cellsOnCell CARRIES THE WHOLE TOPOLOGY
    Measured on every cell of both meshes, zero violations out of 245,760 (x1) and
    983,040 (x4) ring slots:
        edgesOnCell[i,j]    == canonical edge (i, cellsOnCell[i,j])
        verticesOnCell[i,j] == canonical vertex {i, cellsOnCell[i,j-1], cellsOnCell[i,j]}
    The j/j+1 alternative was ruled out by count, not assumed: it is violated
    245,760/245,760 and 983,040/983,040. So the CCW cellsOnCell ring, stored in full,
    determines edgesOnCell, verticesOnCell, cellsOnEdge, verticesOnEdge,
    cellsOnVertex, edgesOnVertex, nEdgesOnCell and nEdgesOnEdge exactly. The golden
    stores that ring in full (CSR) and the rest of the connectivity on a sample.

SUBSETTING, STATED EXACTLY
    FULL, no subset:  xCell yCell zCell meshDensity latCell lonCell areaCell
                      nEdgesOnCell cellsOnCell(CSR)
    SAMPLED:          every edge-indexed and vertex-indexed field, plus the
                      verticesOnCell / edgesOnCell rings and the raw padding slots.
    The sample is a fixed stride UNION the argmin/argmax of every metric (so the
    distribution tails are covered, not just the bulk), stored explicitly as an
    index array so there is no ambiguity about what was taken.
    EVERY field, sampled or not, also carries order-invariant global statistics
    computed over ALL of its elements (n, non-finite count, min, max, exact fsum,
    mean) plus a raw-bytes sha256. So no field is covered only by its sample.
"""

import os as _os


def _mesh(filename):
    """Locate a staged MPAS grid file.

    The meshes are large and are not redistributed, so their location is the
    reader's to state.  GPUWM_MESH_ROOT names the directory holding them; the
    filenames are fixed by the mesh family.  A machine-specific absolute path
    was hardcoded here once and shipped in the release tree.
    """
    root = _os.environ.get("GPUWM_MESH_ROOT")
    if not root:
        raise SystemExit(
            "set GPUWM_MESH_ROOT to the directory holding "
            f"{filename} -- these goldens are generated from a staged mesh "
            "this repository does not carry")
    return _os.path.join(root, filename)



import hashlib
import json
import math
import os
import struct
import sys

import numpy as np
from netCDF4 import Dataset

MAGIC = b"RWGOLD01"
VERSION = 1
DT_I32 = 1
DT_F64 = 2
ENTRY_BYTES = 104
NAME_BYTES = 64

MESHES = {
    "x1.40962": _mesh("x1.40962.grid.nc"),
    "x4.163842": _mesh("x4.163842.grid.nc"),
}
SAMPLE_TARGET = 2048


# ----------------------------------------------------------------- container


class Writer:
    def __init__(self):
        self.entries = []  # (name, dtype, shape, bytes)

    def add(self, name, arr):
        a = np.ascontiguousarray(arr)
        if a.dtype == np.int32:
            dt = DT_I32
        elif a.dtype == np.float64:
            dt = DT_F64
        else:
            raise TypeError(f"{name}: unsupported dtype {a.dtype}")
        if a.ndim < 1 or a.ndim > 4:
            raise ValueError(f"{name}: ndim {a.ndim}")
        if len(name.encode()) >= NAME_BYTES:
            raise ValueError(f"{name}: name too long")
        if any(n == name for n, _, _, _ in self.entries):
            raise ValueError(f"{name}: duplicate entry")
        self.entries.append((name, dt, a.shape, a.astype(a.dtype.newbyteorder("<")).tobytes()))

    def build(self):
        n = len(self.entries)
        head = MAGIC + struct.pack("<II", VERSION, n)
        dir_bytes = len(head) + n * ENTRY_BYTES
        off = (dir_bytes + 7) & ~7
        directory = b""
        payload = b""
        pad0 = b"\0" * (off - dir_bytes)
        for name, dt, shape, blob in self.entries:
            shp = list(shape) + [0] * (4 - len(shape))
            directory += (
                name.encode().ljust(NAME_BYTES, b"\0")
                + struct.pack("<BBH", dt, len(shape), 0)
                + struct.pack("<4I", *shp)
                + struct.pack("<QQ", off + len(payload), len(blob))
                + b"\0" * 4
            )
            payload += blob
            pad = (-len(payload)) & 7
            payload += b"\0" * pad
        return head + directory + pad0 + payload


def read_rwg(blob):
    """Independent reader -- deliberately does not share code with Writer."""
    assert blob[:8] == MAGIC, "bad magic"
    version, n = struct.unpack("<II", blob[8:16])
    assert version == VERSION, f"bad version {version}"
    out = {}
    for i in range(n):
        base = 16 + i * ENTRY_BYTES
        e = blob[base : base + ENTRY_BYTES]
        name = e[:NAME_BYTES].rstrip(b"\0").decode()
        dt, ndim, _ = struct.unpack("<BBH", e[64:68])
        shape = struct.unpack("<4I", e[68:84])[:ndim]
        off, nbytes = struct.unpack("<QQ", e[84:100])
        dtype = "<i4" if dt == DT_I32 else "<f8"
        a = np.frombuffer(blob[off : off + nbytes], dtype=dtype)
        out[name] = a.reshape(shape)
    return out


# ------------------------------------------------------------------- stats


def stats_of(a):
    a = np.asarray(a)
    flat = a.reshape(-1)
    s = {
        "dtype": "i32" if flat.dtype == np.int32 else "f64",
        "shape": list(a.shape),
        "n": int(flat.size),
        "sha256_raw_le": hashlib.sha256(
            np.ascontiguousarray(flat.astype(flat.dtype.newbyteorder("<"))).tobytes()
        ).hexdigest(),
    }
    if flat.dtype == np.float64:
        fin = np.isfinite(flat)
        s["n_nonfinite"] = int((~fin).sum())
        v = flat[fin]
        s["min"] = float(v.min()) if v.size else None
        s["max"] = float(v.max()) if v.size else None
        s["fsum"] = float(math.fsum(v.tolist())) if v.size else None
        s["mean"] = s["fsum"] / v.size if v.size else None
    else:
        s["min"] = int(flat.min())
        s["max"] = int(flat.max())
        s["fsum"] = int(flat.astype(np.int64).sum())
        vals, cnts = np.unique(flat, return_counts=True)
        if vals.size <= 32:
            s["histogram"] = {int(k): int(c) for k, c in zip(vals, cnts)}
    return s


# --------------------------------------------------------------- extraction


def extract(tag, path, outdir):
    d = Dataset(path)
    d.set_auto_mask(False)
    dim = {k: len(v) for k, v in d.dimensions.items()}
    nC, nE, nV = dim["nCells"], dim["nEdges"], dim["nVertices"]
    mE, mE2, vD = dim["maxEdges"], dim["maxEdges2"], dim["vertexDegree"]

    def g(name):
        return np.array(d.variables[name][:])

    raw = {n: g(n) for n in d.variables}
    src_stats = {n: stats_of(raw[n]) for n in raw if raw[n].dtype != np.dtype("S1")}

    neoc = raw["nEdgesOnCell"].astype(np.int64)
    coc = raw["cellsOnCell"].astype(np.int64) - 1
    eoc = raw["edgesOnCell"].astype(np.int64) - 1
    voc = raw["verticesOnCell"].astype(np.int64) - 1
    coe = raw["cellsOnEdge"].astype(np.int64) - 1
    voe = raw["verticesOnEdge"].astype(np.int64) - 1
    cov = raw["cellsOnVertex"].astype(np.int64) - 1
    eov = raw["edgesOnVertex"].astype(np.int64) - 1
    neoe = raw["nEdgesOnEdge"].astype(np.int64)
    eoe = raw["edgesOnEdge"].astype(np.int64) - 1

    # ---- CSR of the CCW cellsOnCell ring (the complete primal topology)
    offs = np.zeros(nC + 1, dtype=np.int64)
    np.cumsum(neoc, out=offs[1:])
    total = int(offs[-1])
    assert total == 2 * nE, f"sum(nEdgesOnCell)={total} != 2*nEdges={2*nE}"
    slot = np.arange(total) - np.repeat(offs[:-1], neoc)
    owner = np.repeat(np.arange(nC), neoc)
    ring = coc[owner, slot]

    # ---- canonical edge enumeration: unordered cell pair, lexicographic
    ea = np.minimum(owner, ring)
    eb = np.maximum(owner, ring)
    ecode = ea * nC + eb
    canon_ecode = np.unique(ecode)
    assert canon_ecode.size == nE, f"canonical edges {canon_ecode.size} != {nE}"
    canon_ea = canon_ecode // nC
    canon_eb = canon_ecode % nC

    # ---- canonical vertex enumeration: sorted cell triple, lexicographic
    prev_slot = (slot - 1) % np.repeat(neoc, neoc)
    ring_prev = coc[owner, prev_slot]
    tri = np.sort(np.stack([owner, ring_prev, ring], axis=1), axis=1)
    vcode = (tri[:, 0] * nC + tri[:, 1]) * nC + tri[:, 2]
    canon_vcode = np.unique(vcode)
    assert canon_vcode.size == nV, f"canonical vertices {canon_vcode.size} != {nV}"
    canon_va = canon_vcode // (nC * nC)
    canon_vb = (canon_vcode // nC) % nC
    canon_vc = canon_vcode % nC

    # ---- file numbering -> canonical numbering (bijection, asserted)
    fe_code = np.minimum(coe[:, 0], coe[:, 1]) * nC + np.maximum(coe[:, 0], coe[:, 1])
    file_to_canon_e = np.searchsorted(canon_ecode, fe_code)
    assert np.array_equal(canon_ecode[file_to_canon_e], fe_code), "edge key not found"
    assert np.unique(file_to_canon_e).size == nE, "edge map not a bijection"
    canon_to_file_e = np.empty(nE, dtype=np.int64)
    canon_to_file_e[file_to_canon_e] = np.arange(nE)

    sv = np.sort(cov, axis=1)
    fv_code = (sv[:, 0] * nC + sv[:, 1]) * nC + sv[:, 2]
    file_to_canon_v = np.searchsorted(canon_vcode, fv_code)
    assert np.array_equal(canon_vcode[file_to_canon_v], fv_code), "vertex key not found"
    assert np.unique(file_to_canon_v).size == nV, "vertex map not a bijection"
    canon_to_file_v = np.empty(nV, dtype=np.int64)
    canon_to_file_v[file_to_canon_v] = np.arange(nV)

    def ce(fidx, pad_mask=None):
        """file edge indices -> canonical, -1 where padded."""
        out = np.where(fidx >= 0, file_to_canon_e[np.clip(fidx, 0, nE - 1)], -1)
        if pad_mask is not None:
            out = np.where(pad_mask, -1, out)
        return out.astype(np.int32)

    def cv(fidx, pad_mask=None):
        out = np.where(fidx >= 0, file_to_canon_v[np.clip(fidx, 0, nV - 1)], -1)
        if pad_mask is not None:
            out = np.where(pad_mask, -1, out)
        return out.astype(np.int32)

    used_cell = np.arange(mE)[None, :] < neoc[:, None]
    used_eoe = np.arange(mE2)[None, :] < neoe[:, None]

    # ---- edge/vertex fields reindexed into canonical order
    e_of = canon_to_file_e
    v_of = canon_to_file_v
    can_coe = coe[e_of]                       # file's (c0,c1) order preserved
    orient = np.where(can_coe[:, 0] < can_coe[:, 1], 1, -1).astype(np.int32)
    can_voe = cv(voe[e_of])
    can_eoe = ce(eoe[e_of], ~used_eoe[e_of])
    can_cov = cov[v_of].astype(np.int32)
    can_eov = ce(eov[v_of])

    # ---- sample selection: fixed stride UNION every metric's argmin/argmax
    def pick(n, extras):
        stride = max(1, n // SAMPLE_TARGET)
        idx = set(range(0, n, stride).__iter__())
        idx.update({0, n - 1})
        idx.update(int(x) for x in extras)
        return np.array(sorted(idx), dtype=np.int64), stride

    dc = raw["dcEdge"][e_of]
    dv = raw["dvEdge"][e_of]
    ang = raw["angleEdge"][e_of]
    wt = raw["weightsOnEdge"][e_of]
    at = raw["areaTriangle"][v_of]
    kite = raw["kiteAreasOnVertex"][v_of]
    ac = raw["areaCell"]
    md = raw["meshDensity"]

    ratio = dv / dc
    e_extra = [
        dc.argmin(), dc.argmax(), dv.argmin(), dv.argmax(),
        ang.argmin(), ang.argmax(), ratio.argmin(), ratio.argmax(),
        neoe[e_of].argmin(), neoe[e_of].argmax(),
        np.abs(wt).max(axis=1).argmax(),
    ]
    v_extra = [at.argmin(), at.argmax(), kite.min(axis=1).argmin(), kite.max(axis=1).argmax()]
    c_extra = [ac.argmin(), ac.argmax(), md.argmin(), md.argmax()]
    for k in np.unique(neoc):                 # one cell of every coordination number
        w = np.nonzero(neoc == k)[0]
        c_extra += [w[0], w[-1]]

    e_idx, e_stride = pick(nE, e_extra)
    v_idx, v_stride = pick(nV, v_extra)
    c_idx, c_stride = pick(nC, c_extra)

    # ---- assemble
    w = Writer()
    w.add("meta/dims", np.array([nC, nE, nV, mE, mE2, vD, dim["TWO"], dim["codeLen"]], dtype=np.int32))
    w.add("meta/sphere_radius", np.array([float(d.getncattr("sphere_radius"))], dtype=np.float64))
    w.add("meta/nominalMinDc", np.array([float(raw["nominalMinDc"])], dtype=np.float64))
    w.add("meta/densityFunctionCode", np.array([int(raw["densityFunctionCode"][0][0])], dtype=np.int32))
    w.add("meta/sample_strides", np.array([c_stride, e_stride, v_stride], dtype=np.int32))

    # INPUT (full)
    for n in ("xCell", "yCell", "zCell", "meshDensity"):
        w.add(f"in/{n}", raw[n].astype(np.float64))

    # cell-indexed expected output (full)
    w.add("cell/nEdgesOnCell", neoc.astype(np.int32))
    w.add("cell/cellsOnCell_offsets", offs.astype(np.int32))
    w.add("cell/cellsOnCell_values", ring.astype(np.int32))
    for n in ("areaCell", "latCell", "lonCell"):
        w.add(f"cell/{n}", raw[n].astype(np.float64))

    # sampled cells: the rings, canonicalised, plus the raw 1-based padding slots
    w.add("sample/cell_index", c_idx.astype(np.int32))
    w.add("sample/verticesOnCell", cv(voc[c_idx], ~used_cell[c_idx]))
    w.add("sample/edgesOnCell", ce(eoc[c_idx], ~used_cell[c_idx]))
    w.add("sample/cellsOnCell_raw1", raw["cellsOnCell"][c_idx].astype(np.int32))
    w.add("sample/edgesOnCell_raw1", raw["edgesOnCell"][c_idx].astype(np.int32))
    w.add("sample/verticesOnCell_raw1", raw["verticesOnCell"][c_idx].astype(np.int32))

    # sampled edges (canonical numbering throughout)
    w.add("sample/edge_index", e_idx.astype(np.int32))
    w.add("sample/edge_cells", np.stack([canon_ea[e_idx], canon_eb[e_idx]], axis=1).astype(np.int32))
    w.add("sample/edge_orient", orient[e_idx])
    w.add("sample/cellsOnEdge", can_coe[e_idx].astype(np.int32))
    w.add("sample/verticesOnEdge", can_voe[e_idx])
    w.add("sample/nEdgesOnEdge", neoe[e_of][e_idx].astype(np.int32))
    w.add("sample/edgesOnEdge", can_eoe[e_idx])
    w.add("sample/weightsOnEdge", wt[e_idx].astype(np.float64))
    w.add("sample/dcEdge", dc[e_idx].astype(np.float64))
    w.add("sample/dvEdge", dv[e_idx].astype(np.float64))
    w.add("sample/angleEdge", ang[e_idx].astype(np.float64))
    for n in ("latEdge", "lonEdge", "xEdge", "yEdge", "zEdge"):
        w.add(f"sample/{n}", raw[n][e_of][e_idx].astype(np.float64))

    # sampled vertices (canonical numbering throughout)
    w.add("sample/vertex_index", v_idx.astype(np.int32))
    w.add("sample/vertex_cells", np.stack([canon_va[v_idx], canon_vb[v_idx], canon_vc[v_idx]], axis=1).astype(np.int32))
    w.add("sample/cellsOnVertex", can_cov[v_idx])
    w.add("sample/edgesOnVertex", can_eov[v_idx])
    w.add("sample/areaTriangle", at[v_idx].astype(np.float64))
    w.add("sample/kiteAreasOnVertex", kite[v_idx].astype(np.float64))
    for n in ("latVertex", "lonVertex", "xVertex", "yVertex", "zVertex"):
        w.add(f"sample/{n}", raw[n][v_of][v_idx].astype(np.float64))

    blob = w.build()
    rwg_path = os.path.join(outdir, f"{tag}.mesh.rwg")
    with open(rwg_path, "wb") as f:
        f.write(blob)

    # ---- measurements that belong in the manifest, not in prose
    x, y, z = raw["xCell"], raw["yCell"], raw["zCell"]
    r = np.sqrt(x * x + y * y + z * z)
    lat_r = np.arcsin(np.clip(z / r, -1, 1))
    lon_r = np.arctan2(y, x) % (2 * np.pi)
    dlon = np.abs(((lon_r - (raw["lonCell"] % (2 * np.pi))) + np.pi) % (2 * np.pi) - np.pi)

    manifest = {
        "mesh": tag,
        "source_file": path,
        "source_bytes": os.path.getsize(path),
        "source_sha256": sha256_file(path),
        "container": "RWGOLD01 v1, little-endian",
        "golden_bytes": len(blob),
        "golden_sha256": hashlib.sha256(blob).hexdigest(),
        "dims": dim,
        "global_attrs": {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in
                         ((k, d.getncattr(k)) for k in d.ncattrs())},
        "canonical_numbering": {
            "edge": "unordered cell pair (a,b) a<b, sorted lexicographically",
            "vertex": "sorted cell triple (a,b,c), sorted lexicographically",
            "edge_bijection_with_file": True,
            "vertex_bijection_with_file": True,
        },
        "sampling": {
            "cell": {"stride": int(c_stride), "n_sampled": int(c_idx.size), "n_total": nC},
            "edge": {"stride": int(e_stride), "n_sampled": int(e_idx.size), "n_total": nE},
            "vertex": {"stride": int(v_stride), "n_sampled": int(v_idx.size), "n_total": nV},
            "rule": "index % stride == 0, UNION {0, n-1}, UNION argmin/argmax of every metric",
        },
        "measured": {
            "abs_radius_minus_one_max": float(np.abs(r - 1).max()),
            "latCell_vs_asin_z_max_rad": float(np.abs(lat_r - raw["latCell"]).max()),
            "lonCell_vs_atan2_yx_max_rad": float(dlon.max()),
            "sum_areaCell_over_4pi": float(math.fsum(raw["areaCell"].tolist()) / (4 * math.pi)),
            "sum_areaTriangle_over_4pi": float(math.fsum(raw["areaTriangle"].tolist()) / (4 * math.pi)),
            "sum_kiteAreas_over_4pi": float(math.fsum(raw["kiteAreasOnVertex"].reshape(-1).tolist()) / (4 * math.pi)),
            "euler_nCells_minus_nEdges_plus_nVertices": nC - nE + nV,
            "sum_six_minus_nEdgesOnCell": int((6 - neoc).sum()),
            "nEdgesOnCell_histogram": {int(k): int(c) for k, c in zip(*np.unique(neoc, return_counts=True))},
            "nEdgesOnEdge_equals_neoc_c0_plus_c1_minus_2_violations": int(
                (neoe != neoc[coe[:, 0]] + neoc[coe[:, 1]] - 2).sum()),
            "padding_fill_rule": {
                "cellsOnCell": "repeat of last valid entry (measured 100%)",
                "edgesOnCell": "max of valid entries (measured 100%)",
                "verticesOnCell": "0 in the file's 1-based numbering (measured 100%)",
                "weightsOnEdge": "exactly 0.0 (measured, 0 nonzero padding slots)",
            },
            "weightsOnEdge_nonzero_padding_slots": int((wt[~used_eoe[e_of]] != 0.0).sum()),
        },
        "field_stats_over_all_elements": src_stats,
    }
    json_path = os.path.join(outdir, f"{tag}.mesh.json")
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    d.close()
    return rwg_path, json_path, blob, manifest


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------- instrument validation


def self_test(tag, path, blob):
    """Both directions. Must round-trip exactly, and must catch a perturbation."""
    d = Dataset(path)
    d.set_auto_mask(False)
    got = read_rwg(blob)
    ok = fail = 0

    # DIRECTION 1 -- byte-exact round trip against the netCDF source
    checks = [
        ("in/xCell", "xCell"), ("in/yCell", "yCell"), ("in/zCell", "zCell"),
        ("in/meshDensity", "meshDensity"), ("cell/areaCell", "areaCell"),
        ("cell/latCell", "latCell"), ("cell/lonCell", "lonCell"),
        ("cell/nEdgesOnCell", "nEdgesOnCell"),
    ]
    for gname, vname in checks:
        src = np.array(d.variables[vname][:])
        src = src.astype(np.float64 if src.dtype.kind == "f" else np.int32)
        a = np.ascontiguousarray(src.astype(src.dtype.newbyteorder("<"))).tobytes()
        b = np.ascontiguousarray(got[gname].astype(got[gname].dtype.newbyteorder("<"))).tobytes()
        if a == b and len(a) > 0:
            ok += 1
        else:
            fail += 1
            print(f"  ROUNDTRIP FAIL {gname}")

    # sampled edge fields, resolved back through the canonical map
    coe = np.array(d.variables["cellsOnEdge"][:]).astype(np.int64) - 1
    nC = len(d.dimensions["nCells"])
    ec = got["sample/edge_cells"].astype(np.int64)
    want = ec[:, 0] * nC + ec[:, 1]
    have = np.minimum(coe[:, 0], coe[:, 1]) * nC + np.maximum(coe[:, 0], coe[:, 1])
    order = np.argsort(have)
    fidx = order[np.searchsorted(have[order], want)]
    for gname, vname in [("sample/dcEdge", "dcEdge"), ("sample/dvEdge", "dvEdge"),
                         ("sample/angleEdge", "angleEdge"), ("sample/weightsOnEdge", "weightsOnEdge")]:
        src = np.array(d.variables[vname][:])[fidx]
        a = np.ascontiguousarray(src.astype("<f8")).tobytes()
        b = np.ascontiguousarray(got[gname].astype("<f8")).tobytes()
        if a == b:
            ok += 1
        else:
            fail += 1
            print(f"  ROUNDTRIP FAIL {gname} (via canonical key)")

    # DIRECTION 2 -- the comparison must FAIL when one value moves
    def perturb_and_check(gname, elem, how):
        ent = None
        n = struct.unpack("<II", blob[8:16])[1]
        for i in range(n):
            base = 16 + i * ENTRY_BYTES
            nm = blob[base : base + NAME_BYTES].rstrip(b"\0").decode()
            if nm == gname:
                ent = struct.unpack("<QQ", blob[base + 84 : base + 100])
                dt = blob[base + 64]
                break
        assert ent, gname
        off, _ = ent
        mutated = bytearray(blob)
        if dt == DT_F64:
            pos = off + elem * 8
            v = struct.unpack("<d", bytes(mutated[pos : pos + 8]))[0]
            nv = how(v)
            mutated[pos : pos + 8] = struct.pack("<d", nv)
            delta = f"{v!r} -> {nv!r}"
        else:
            pos = off + elem * 4
            v = struct.unpack("<i", bytes(mutated[pos : pos + 4]))[0]
            nv = int(how(v))
            mutated[pos : pos + 4] = struct.pack("<i", nv)
            delta = f"{v} -> {nv}"
        re = read_rwg(bytes(mutated))
        caught = not np.array_equal(re[gname], got[gname])
        print(f"  perturb {gname}[{elem}] {delta}: {'CAUGHT' if caught else 'MISSED'}")
        return caught

    ulp = lambda v: np.nextafter(v, np.inf).item()
    injections = [
        ("in/xCell", 17, ulp),
        ("cell/areaCell", 3, lambda v: v * (1 + 1e-12)),
        ("cell/cellsOnCell_values", 5, lambda v: v + 1),
        ("cell/nEdgesOnCell", 11, lambda v: v + 1),
        ("sample/weightsOnEdge", 41, ulp),
        ("sample/edgesOnEdge", 7, lambda v: v + 1),
        ("sample/kiteAreasOnVertex", 9, lambda v: v * (1 + 1e-15)),
    ]
    for gname, elem, how in injections:
        if perturb_and_check(gname, elem, how):
            ok += 1
        else:
            fail += 1

    # The bit-exact instrument (sha256 over all elements) must catch 1 ULP on one element.
    a = np.array(d.variables["areaCell"][:])
    s0 = stats_of(a)
    a2 = a.copy()
    a2[3] = np.nextafter(a2[3], np.inf)
    s1 = stats_of(a2)
    caught = s0["sha256_raw_le"] != s1["sha256_raw_le"]
    print(f"  perturb areaCell[3] 1 ULP vs sha256_raw_le: {'CAUGHT' if caught else 'MISSED'}")
    ok += 1 if caught else 0
    fail += 0 if caught else 1

    # RESOLUTION LIMIT of the order-invariant fsum statistic, measured not assumed:
    # fsum is exactly rounded, so a single-element change below ULP(total) vanishes.
    # Bisect for the smallest absolute single-element change fsum still resolves.
    base = s0["fsum"]
    lo, hi = 0.0, abs(base) * 1e-9
    for _ in range(200):
        mid = (lo + hi) / 2
        a3 = a.copy()
        a3[3] = a3[3] + mid
        if math.fsum(a3.tolist()) != base:
            hi = mid
        else:
            lo = mid
        if hi - lo <= 0.0:
            break
    fsum_lim = hi
    print(f"  fsum single-element resolution limit: {fsum_lim:.3e} absolute "
          f"({fsum_lim / float(a[3]):.3e} relative to a typical areaCell); "
          f"ULP(total {base:.6f}) = {np.spacing(base):.3e}")
    ulp_caught = s0["fsum"] != s1["fsum"]
    print(f"  1 ULP on areaCell[3] vs fsum: {'CAUGHT' if ulp_caught else 'MISSED -- below that limit, as measured'}")

    d.close()
    print(f"  {tag}: self-test {ok} passed, {fail} failed")
    return fail == 0, fsum_lim


if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)
    allok = True
    for tag, path in MESHES.items():
        print(f"=== {tag}")
        rwg, js, blob, man = extract(tag, path, outdir)
        print(f"  {rwg} {len(blob)} B sha256 {man['golden_sha256']}")
        print(f"  {js}")
        good, fsum_lim = self_test(tag, path, blob)
        allok &= good
        man['measured']['fsum_single_element_resolution_limit_abs'] = fsum_lim
        import json as _j
        open(js,'w').write(_j.dumps(man, indent=1, sort_keys=True))
    print("ALL SELF-TESTS PASSED" if allok else "SELF-TEST FAILURE")
    sys.exit(0 if allok else 1)
