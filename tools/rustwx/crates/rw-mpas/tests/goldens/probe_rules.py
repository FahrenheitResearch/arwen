"""Probe: establish the reconstruction rules the golden format depends on.

Throwaway measurement harness. Answers three questions on BOTH published meshes:
  Q1  Is edgesOnCell[i,j] the edge between cell i and cellsOnCell[i,j]?
  Q2  Is verticesOnCell[i,j] the vertex {i, cellsOnCell[i,j-1], cellsOnCell[i,j]}?
      (and the j/j+1 alternative, so the off-by-one is ruled out by count, not assumed)
  Q3  How far do the file's latCell/lonCell sit from asin(z)/atan2(y,x)?
If Q1 and Q2 hold with zero violations, the full cellsOnCell ring determines every
other connectivity array, and the golden can carry complete topology in CSR form.
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


import sys
import numpy as np
from netCDF4 import Dataset

MESHES = {
    "x1.40962": _mesh("x1.40962.grid.nc"),
    "x4.163842": _mesh("x4.163842.grid.nc"),
}


def zb(a):
    return a.astype(np.int64) - 1


for tag, path in MESHES.items():
    d = Dataset(path)
    nC = len(d.dimensions["nCells"])
    nE = len(d.dimensions["nEdges"])
    nV = len(d.dimensions["nVertices"])
    neoc = d.variables["nEdgesOnCell"][:].astype(np.int64)
    coc = zb(d.variables["cellsOnCell"][:])
    eoc = zb(d.variables["edgesOnCell"][:])
    voc = zb(d.variables["verticesOnCell"][:])
    coe = zb(d.variables["cellsOnEdge"][:])
    cov = zb(d.variables["cellsOnVertex"][:])
    print(f"=== {tag}  nCells={nC} nEdges={nE} nVertices={nV}")

    # map canonical keys -> file index
    ekey = {}
    for e in range(nE):
        a, b = coe[e]
        ekey[(min(a, b), max(a, b))] = e
    vkey = {}
    for v in range(nV):
        t = tuple(sorted(cov[v].tolist()))
        vkey[t] = v
    print(f"  key maps: edges {len(ekey)}/{nE}  vertices {len(vkey)}/{nV}")

    # Q1 / Q2 over EVERY cell, counted (not sampled, not fitted)
    q1_bad = 0
    q2_bad_prev = 0   # vertex j == {i, coc[j-1], coc[j]}
    q2_bad_next = 0   # vertex j == {i, coc[j], coc[j+1]}
    missing_e = 0
    missing_v = 0
    for i in range(nC):
        n = int(neoc[i])
        ring = coc[i, :n]
        for j in range(n):
            k = ekey.get((min(i, ring[j]), max(i, ring[j])))
            if k is None:
                missing_e += 1
            elif k != eoc[i, j]:
                q1_bad += 1
            tprev = tuple(sorted((i, int(ring[(j - 1) % n]), int(ring[j]))))
            tnext = tuple(sorted((i, int(ring[j]), int(ring[(j + 1) % n]))))
            kp = vkey.get(tprev)
            kn = vkey.get(tnext)
            if kp is None or kn is None:
                missing_v += 1
                continue
            if kp != voc[i, j]:
                q2_bad_prev += 1
            if kn != voc[i, j]:
                q2_bad_next += 1
    tot = int(neoc.sum())
    print(f"  Q1 edgesOnCell[i,j] == edge(i, cellsOnCell[i,j]) : violations {q1_bad}/{tot}, unresolved {missing_e}")
    print(f"  Q2a verticesOnCell[i,j] == vtx(i,coc[j-1],coc[j]) : violations {q2_bad_prev}/{tot}")
    print(f"  Q2b verticesOnCell[i,j] == vtx(i,coc[j],coc[j+1]) : violations {q2_bad_next}/{tot}, unresolved {missing_v}")

    # Q3 lat/lon vs xyz
    x = d.variables["xCell"][:]
    y = d.variables["yCell"][:]
    z = d.variables["zCell"][:]
    lat = d.variables["latCell"][:]
    lon = d.variables["lonCell"][:]
    r = np.sqrt(x * x + y * y + z * z)
    latr = np.arcsin(np.clip(z / r, -1, 1))
    lonr = np.arctan2(y, x) % (2 * np.pi)
    dlat = np.abs(latr - lat).max()
    dlon = np.abs(((lonr - (lon % (2 * np.pi))) + np.pi) % (2 * np.pi) - np.pi).max()
    print(f"  Q3 |r|-1 max {np.abs(r-1).max():.3e}   max|dlat| {dlat:.3e} rad   max|dlon| {dlon:.3e} rad")

    # padding fill rule, counted
    for name, arr in (("cellsOnCell", coc), ("edgesOnCell", eoc), ("verticesOnCell", voc)):
        me = arr.shape[1]
        zero = last = mx = 0
        slots = 0
        for i in range(0, nC, 7):
            n = int(neoc[i])
            for j in range(n, me):
                slots += 1
                v = int(arr[i, j])
                if v == -1:
                    zero += 1
                if v == int(arr[i, n - 1]):
                    last += 1
                if v == int(arr[i, :n].max()):
                    mx += 1
        print(f"  padding {name}: slots {slots} zero(1-based) {zero} repeat-last {last} max-of-valid {mx}")
    d.close()
    sys.stdout.flush()
