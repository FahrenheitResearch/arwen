"""Run compiled WRF rhs_ph and phy_prep over the pinned real-state corpus."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from prep_build import FIELDS1, FIELDS2, FIELDS3, OUTPUTS

CASES = ("real-periodic2", "real-specified2", "real-periodic5", "real-specified5",
         "steep-terrain", "map-extremes", "zero-motion", "near-zero-motion")


def make_case(raw, name, omega):
    s = {k: v.copy() for k, v in raw.items()}
    nz, ny, nx = s["T"].shape
    s["ALT"] = (s["AL"] + s["ALB"]).astype("f4")
    s["MUT"] = (s["MU"] + s["MUB"]).astype("f4")
    # Use actual compiled calc_ww_cp output from the same real wind state.
    s["WW"] = omega.copy()
    if name in ("steep-terrain", "map-extremes"):
        height = np.arange(nz + 1, dtype="f4")[:, None, None]
        s["W"] += (np.float32(0.5) * np.sin(np.float32(np.pi) * height / np.float32(nz))).astype("f4")
    if name == "steep-terrain":
        ridge = (np.arange(nx, dtype="f4")[None, :] % 3 - 1) * np.float32(9.81 * 1000)
        s["PHB"] += ridge[None]
    if name == "map-extremes":
        for field in ("MAPFAC_M", "MAPFAC_U", "MAPFAC_V", "MAPFAC_MX", "MAPFAC_MY",
                      "MAPFAC_UX", "MAPFAC_UY", "MAPFAC_VX", "MAPFAC_VY"):
            s[field][...] = np.linspace(0.3, 3.0, s[field].size, dtype="f4").reshape(s[field].shape)
        # The engine represents isotropic map factors. Use that same input
        # on both routines, including the reciprocal V map stored by WRF.
        s["MAPFAC_MX"] = s["MAPFAC_MY"] = s["MAPFAC_M"].copy()
        s["MAPFAC_UX"] = s["MAPFAC_UY"] = s["MAPFAC_U"].copy()
        s["MAPFAC_VX"] = s["MAPFAC_VY"] = s["MAPFAC_V"].copy()
        s["MF_VX_INV"] = np.float32(1) / s["MAPFAC_V"]
    if name == "zero-motion":
        for field in ("U", "V", "W", "WW"):
            s[field][...] = 0
    if name == "near-zero-motion":
        for field in ("U", "V", "W", "WW"):
            s[field] *= np.float32(1e-25)
    if "specified" not in name:
        for field in ("U", "MAPFAC_U", "MAPFAC_UX", "MAPFAC_UY"):
            s[field][..., -1] = s[field][..., 0]
        for field in ("V", "MAPFAC_V", "MAPFAC_VX", "MAPFAC_VY", "MF_VX_INV"):
            s[field][..., -1, :] = s[field][..., 0, :]
    # Explicitly preserve ArWen's documented full-mass averaging convention.
    s["MUU"] = np.concatenate((np.float32(0.5) * (s["MUT"] + np.roll(s["MUT"], 1, axis=1)),
                               (np.float32(0.5) * (s["MUT"] + np.roll(s["MUT"], 1, axis=1)))[:, :1]), axis=1)
    s["MUV"] = np.concatenate((np.float32(0.5) * (s["MUT"] + np.roll(s["MUT"], 1, axis=0)),
                               (np.float32(0.5) * (s["MUT"] + np.roll(s["MUT"], 1, axis=0)))[:1]), axis=0)
    return s


def pad3(a, nz, ny, nx):
    if a.ndim == 2:
        a = a[None]
    # Shared WRF memory includes the full halo rectangle and vertical top.
    out = np.empty((nz + 1, ny + 8, nx + 8), dtype="f4")
    for k in range(nz + 1):
        for j in range(ny + 8):
            for i in range(nx + 8):
                out[k, j, i] = a[min(k, a.shape[0]-1), (j-4) % ny, (i-4) % nx]
    # Preserve the distinct east/north physical face supplied by WRF.
    if a.shape[2] == nx+1:
        out[:, 4:ny+4, nx+4] = a[np.minimum(np.arange(nz+1), a.shape[0]-1), :, nx]
    if a.shape[1] == ny+1:
        out[:, ny+4, 4:nx+4] = a[np.minimum(np.arange(nz+1), a.shape[0]-1), ny, :nx]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state", type=Path)
    ap.add_argument("executable", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    raw = dict(np.load(args.state))
    coupling_path = args.state.parent / "coupling.npz"
    with np.load(coupling_path) as coupling:
        omega = {"periodic":coupling["real_periodic/ww"], "specified":coupling["real_boundary/ww"]}
    args.output.mkdir(parents=True, exist_ok=True)
    nz, ny, nx = raw["T"].shape
    mapping = {"ph_old":"PH", "msfux":"MAPFAC_UX", "msfuy":"MAPFAC_UY",
               "msfvx":"MAPFAC_VX", "msfvx_inv":"MF_VX_INV", "msfvy":"MAPFAC_VY",
               "msftx":"MAPFAC_MX", "msfty":"MAPFAC_MY"}
    for name in CASES:
        state = make_case(raw,name,omega["specified" if "specified" in name else "periodic"])
        folder = args.output / name
        folder.mkdir(exist_ok=True)
        with (folder / "inputs.bin").open("wb") as f:
            for field in FIELDS3:
                a = pad3(state[mapping.get(field, field.upper())], nz, ny, nx)
                f.write(a.transpose(1, 0, 2).tobytes())
            for field in FIELDS2:
                a = pad3(state[mapping.get(field, field.upper())], nz, ny, nx)[0]
                f.write(a.tobytes())
            for field in FIELDS1:
                a = state[field.upper()]
                f.write(np.pad(a, (0, nz+1-len(a)), mode="edge").astype("f4").tobytes())
            for field in (None, "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP"):
                a = np.zeros((nz+1, ny+8, nx+8), "f4") if field is None else pad3(state[field], nz, ny, nx)
                f.write(a.transpose(1, 0, 2).tobytes())
            f.write(np.asarray([state["CFN"][0], state["CFN1"][0], 1/3000, 1/3000, 5000], "f4").tobytes())
        order = 5 if name.endswith("5") else 2
        specified = int("specified" in name)
        subprocess.run([str(args.executable.resolve()), str(nx), str(ny), str(nz), str(order), str(specified), "1"], cwd=folder, check=True)
        words = np.frombuffer((folder / "outputs.bin").read_bytes(), dtype="<f4")
        n = (nx+8)*(ny+8)*(nz+1)
        expected = {field: words[i*n:(i+1)*n].reshape(ny+8,nz+1,nx+8).transpose(1,0,2)[:,4:ny+4,4:nx+4].copy()
                    for i, field in enumerate(OUTPUTS)}
        # Causal control: call the same unchanged routine with identical
        # total geopotential but a zero base/total perturbation decomposition.
        # This isolates rounding from subtracting before or after adding PHB.
        input_bytes = bytearray((folder / "inputs.bin").read_bytes())
        ph_offset = FIELDS3.index("ph") * n * 4
        phb_offset = FIELDS3.index("phb") * n * 4
        ph_words = np.frombuffer(input_bytes[ph_offset:ph_offset+n*4],dtype="<f4")
        phb_words = np.frombuffer(input_bytes[phb_offset:phb_offset+n*4],dtype="<f4")
        input_bytes[ph_offset:ph_offset+n*4] = (ph_words + phb_words).astype("f4").tobytes()
        input_bytes[phb_offset:phb_offset+n*4] = bytes(n*4)
        (folder / "inputs-total-phi.bin").write_bytes(input_bytes)
        original_bytes = (folder / "inputs.bin").read_bytes()
        (folder / "inputs.bin").write_bytes(input_bytes)
        subprocess.run([str(args.executable.resolve()),str(nx),str(ny),str(nz),str(order),str(specified),"1"],cwd=folder,check=True)
        control = np.frombuffer((folder / "outputs.bin").read_bytes(),dtype="<f4")[:n]
        expected["ph_tend_total_phi"] = control.reshape(ny+8,nz+1,nx+8).transpose(1,0,2)[:,4:ny+4,4:nx+4].copy()
        (folder / "inputs.bin").write_bytes(original_bytes)
        expected.update({f"input_{k}":v for k,v in state.items()})
        expected.update(order=np.int32(order), specified=np.int32(specified))
        np.savez_compressed(args.output / f"prep-{name}.npz", **expected)
    (args.output / "prep-cases.json").write_text(json.dumps({"cases":list(CASES), "nx":nx,"ny":ny,"nz":nz,
        "input_basis":"Stored WRF real initialization; edge cases add a smooth 0.5 m/s vertical-velocity perturbation",
        "omega":"Actual compiled WRF calc_ww_cp real-state periodic or specified output",
        "outputs":OUTPUTS}, indent=2)+"\n")


if __name__ == "__main__":
    main()
