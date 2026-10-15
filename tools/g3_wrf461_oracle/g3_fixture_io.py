"""Read the Grell 3D oracle's binary input/output files.

One reader for the build script's packer and for the parity tests, so the
byte layout written by gen_g3_cases.py / run_g3.F90 is spelled once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import json

ENSDIM = 144

IN_LEV = ("u", "v", "t", "q", "p", "pi", "rho", "dz8w", "rthften",
          "rqvften", "rthblten", "rqvblten")
IN_IFACE = ("w", "p8w")
IN_SFC = ("ht", "xland", "gsw")
IN_APR = ("apr_gr", "apr_w", "apr_mc", "apr_st", "apr_as", "apr_capma",
          "apr_capme", "apr_capmi")

#: run_g3.F90's stage-1 (G3DRV) volume outputs, in write order.
OUT1_LEV = ("cugd_tten", "cugd_qvten", "cugd_qcten", "cugd_ttens",
            "cugd_qvtens", "rqccuten_g3", "rqicuten_g3", "gdc", "gdc2")
OUT1_SFC = ("raincv_g3", "pratec_g3", "htop", "hbot", "mass_flux",
            "edt_out", "xmb_shallow") + IN_APR
OUT1_ISFC = ("k22_shallow", "kbcon_shallow", "ktop_shallow", "ktop_deep",
             "cu_act_flag")
#: stage 2 (conv_grell_spread3d)
OUT2_LEV = ("rthcuten", "rqvcuten", "rqccuten", "rqicuten")
OUT2_SFC = ("raincv", "pratec")


def read_input(path: Path) -> dict:
    raw = Path(path).read_bytes()
    hdr = np.frombuffer(raw, "<i4", 8)
    nx, ny, nz, ishallow, avedx, ichoice, fqi, itimestep = (int(x) for x in hdr)
    dt, dx = (float(x) for x in np.frombuffer(raw, "<f4", 2, 32))
    off = 40
    out = dict(nx=nx, ny=ny, nz=nz, ishallow=ishallow, avedx=avedx,
               ichoice=ichoice, fqi=fqi, itimestep=itimestep,
               dt=np.float32(dt), dx=np.float32(dx))

    def take(shape, dtype="<f4"):
        nonlocal off
        n = int(np.prod(shape))
        a = np.frombuffer(raw, dtype, n, off).reshape(shape).copy()
        off += 4 * n
        return a

    for k in IN_LEV:
        out[k] = take((ny, nz, nx))
    for k in IN_IFACE:
        out[k] = take((ny, nz + 1, nx))
    for k in IN_SFC:
        out[k] = take((ny, nx))
    out["kpbl"] = take((ny, nx), "<i4")
    for k in IN_APR:
        out[k + "_in"] = take((ny, nx))
    if off != len(raw):
        raise ValueError(f"{path}: {len(raw) - off} trailing bytes")
    metadata = Path(path).parent / "configurations.json"
    defaults = dict(maxiens=1, maxens=3, maxens2=3, maxens3=16, ensdim=144,
                    periodic_x=0, periodic_y=0)
    if metadata.exists():
        cfg = json.loads(metadata.read_text())[Path(path).stem.removeprefix("g3-in-")]
        defaults.update({k: cfg[k] for k in defaults if k in cfg})
    out.update(defaults)
    return out


def read_output(path: Path, nx: int, ny: int, nz: int, ensdim: int = ENSDIM) -> dict:
    raw = Path(path).read_bytes()
    off = 0
    out = {}

    def take(shape, dtype="<f4"):
        nonlocal off
        n = int(np.prod(shape))
        a = np.frombuffer(raw, dtype, n, off).reshape(shape).copy()
        off += 4 * n
        return a

    for k in OUT1_LEV:
        out[k] = take((ny, nz, nx))
    for k in OUT1_SFC:
        out[k] = take((ny, nx))
    for k in OUT1_ISFC:
        out[k] = take((ny, nx), "<i4")
    out["xf_ens"] = take((ensdim, ny, nx))
    out["pr_ens"] = take((ensdim, ny, nx))
    for k in OUT2_LEV:
        out[k] = take((ny, nz, nx))
    for k in OUT2_SFC:
        out[k] = take((ny, nx))
    if off != len(raw):
        raise ValueError(f"{path}: {len(raw) - off} trailing bytes")
    return out


def band(nx: int, ny: int) -> tuple[slice, slice]:
    """WRF's written band for a non-periodic single-tile domain, 0-based."""
    return slice(4, ny - 4), slice(4, nx - 4)
