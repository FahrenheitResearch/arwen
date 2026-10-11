"""Read benchmark columns through Rust and compare full GPU/Fortran words."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
class oracle:
    SPECIES_MP8 = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr")
    WRF_OUT3 = (*SPECIES_MP8, "th", "refl", "re_cloud", "re_ice", "re_snow")
    WRF_OUT2 = ("rainnc", "rainncv", "snownc", "snowncv", "graupelnc", "graupelncv", "sr")

    @staticmethod
    def prepare(cols, mp=8):
        from gpuwm.core import constants as C
        z = (np.zeros_like(cols["geop"]) + cols["geop"]) / np.float32(C.G)
        return dict(cols, pii=np.power(cols["p"] / np.float32(C.P0), np.float32(C.RCP)), dz=z[:, 1:] - z[:, :-1], hgt=z[:, :-1])

    @staticmethod
    def write_wrf_input(path, inp, dt, mp=8):
        ncol, nz = inp["p"].shape
        with path.open("wb") as stream:
            np.asarray([ncol, nz], "<i4").tofile(stream)
            np.asarray([dt], "<f4").tofile(stream)
            for name in ("th", "pii", "p", "w", "dz", "hgt", *oracle.SPECIES_MP8):
                np.ascontiguousarray(inp[name][:, :nz].T, dtype="<f4").tofile(stream)

    @staticmethod
    def read_wrf_output(path, ncol, nz, mp=8):
        words = np.fromfile(path, "<f4")
        if words.size != len(oracle.WRF_OUT3) * ncol * nz + len(oracle.WRF_OUT2) * ncol:
            raise ValueError((path, words.size))
        output, off = {}, 0
        for name in oracle.WRF_OUT3:
            output[name] = words[off:off+ncol*nz].reshape(nz, ncol).T.copy()
            off += ncol*nz
        for name in oracle.WRF_OUT2:
            output[name] = words[off:off+ncol].copy()
            off += ncol
        return output


def extract(folder, out, reader=None):
    from gpuwm.netcdf_bridge import open_dataset
    names = {"qv": "QVAPOR", "qc": "QCLOUD", "qi": "QICE", "qr": "QRAIN", "qs": "QSNOW", "qg": "QGRAUP", "ni": "QNICE", "nr": "QNRAIN"}
    receipts = []
    for path in (folder / "wrfinput_d01", folder / "wrfrst_d01_2019-11-26_23:00:00.gnu"):
        dataset = open_dataset(path)
        ny, nx = len(dataset.dimensions["south_north"]), len(dataset.dimensions["west_east"])
        packs = []
        positions = []
        restart = path.name.startswith("wrfrst")
        if reader:
            import tempfile
            with tempfile.TemporaryDirectory(prefix="native-columns-",dir=out) as temporary:
                raw = Path(temporary)
                subprocess.run([str(reader),str(path),str(raw),"restart" if restart else "input"],check=True)
                metadata = json.loads((raw/"metadata.json").read_text())
                decoded = {r["name"]:np.fromfile(raw/(r["name"]+".f32"),"<f4").reshape(256,r["levels"]) for r in metadata}
                manifest = out.parent / "MANIFEST.txt"
                if manifest.is_file():
                    with manifest.open("a") as log:
                        log.write(str(raw)+"\n")
                        for file in raw.iterdir():
                            log.write(str(file)+"\n")
            columns = {key:decoded[name] for key,name in names.items()}
            columns["p"] = decoded["P"] + decoded["PB"]
            columns["th"] = decoded["T"] + np.float32(300)
            columns["w"] = decoded["W_2" if restart else "W"]
            columns["geop"] = decoded["PH_2" if restart else "PH"] + decoded["PHB"]
        for fy in (0.15, 0.38, 0.62, 0.85):
            for fx in (0.15, 0.38, 0.62, 0.85):
                j, i = int(ny * fy), int(nx * fx)
                positions.extend([[j + dj, i + di] for dj in range(4) for di in range(4)])
                if reader:
                    continue
                requested = [*names.values(), "P", "PB", "T", "W_2" if restart else "W", "PH_2" if restart else "PH", "PHB"]
                decoded = dataset._decode(requested, raw=True, scale=False, window=(i,4,j,4))
                def field(name):
                    a = decoded[name][0]
                    if a.ndim == 4:
                        a = a[0]
                    return np.asarray(a, np.float32).reshape(a.shape[-3], 16).T.copy()
                pack = {key: field(name) for key, name in names.items()}
                pack["p"] = field("P") + field("PB")
                pack["th"] = field("T") + np.float32(300)
                pack["w"] = field("W_2" if restart else "W")
                pack["geop"] = field("PH_2" if restart else "PH") + field("PHB")
                packs.append(pack)
        if not reader:
            columns = {k: np.concatenate([p[k] for p in packs], axis=0) for k in packs[0]}
        columns["positions"] = np.asarray(positions, np.int32)
        columns["dt"] = np.float32(dataset.global_attributes["DT"])
        destination = out / (folder.name + ("-restart" if restart else "-input") + ".npz")
        np.savez(destination, **columns)
        receipts.append({"source": str(path), "TITLE": dataset.global_attributes.get("TITLE"), "MP_PHYSICS": dataset.global_attributes.get("MP_PHYSICS"), "DT": float(columns["dt"]), "columns": len(positions), "levels": columns["p"].shape[1], "positions": positions, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "active_words": {k: int((v > 1.e-12).sum()) for k, v in columns.items() if k in names}})
    (out / (folder.name + "-inputs.json")).write_text(json.dumps(receipts, indent=2) + "\n")


class State:
    def __init__(self, cols, cp):
        self.cp = cp
        ncol, nz = cols["p"].shape
        def vol(a):
            return cp.asarray(np.ascontiguousarray(a.T.reshape(a.shape[1], 1, ncol), dtype=np.float32))
        self.p, self.thp, self.php, self.w = (vol(cols[k]) for k in ("p", "th", "geop", "w"))
        self.thb, self.phb = cp.zeros(nz, cp.float32), cp.zeros(nz + 1, cp.float32)
        for name in oracle.SPECIES_MP8:
            setattr(self, name, vol(cols[name]))
        for name in ("effc", "effi", "effs", "h_diabatic"):
            setattr(self, name, cp.zeros_like(self.p))
        self.physics = SimpleNamespace(refl_10cm=None, state=self)
        self._scratch = {}

    def scratch(self, shape, slot, dtype=None):
        if slot not in self._scratch:
            self._scratch[slot] = self.cp.zeros(shape, np.float32 if dtype is None else dtype)
        return self._scratch[slot]

    def existing_scratch(self, slot):
        return self._scratch.get(slot)


def compare(columns, lane, out, version):
    import cupy as cp
    from gpuwm.core.microphysics import _apply_thompson
    cols = {k: v for k, v in np.load(columns).items()}
    state = State(cols, cp)
    cfg = SimpleNamespace(mp_physics=8, no_mp_heating=0, mp_tend_lim=10.0, thompson_version=version)
    if version == "wrf_44":
        from gpuwm.core.thompson_column import apply_generated
        diag = apply_generated(state, cfg, float(cols["dt"]), refl_10cm_due=True)
    else:
        diag = _apply_thompson(state, cfg, float(cols["dt"]), refl_10cm_due=True)
    cp.cuda.Device().synchronize()
    def volume(a):
        return cp.asnumpy(a).reshape(a.shape[0], -1).T.copy()
    gpu = {s: volume(getattr(state, s)) for s in oracle.SPECIES_MP8}
    gpu["th"] = volume(state._scratch["mp_th"])
    gpu["refl"] = volume(state.physics.refl_10cm)
    for source, name in (("effc", "re_cloud"), ("effi", "re_ice"), ("effs", "re_snow")):
        gpu[name] = volume(getattr(state, source))
    for name in oracle.WRF_OUT2:
        gpu[name] = cp.asnumpy(getattr(diag, name)).reshape(-1)
    inp = oracle.prepare(cols, mp=8)
    inp["pii"], inp["dz"] = volume(state._scratch["mp_pii"]), volume(state._scratch["mp_dz8w"])
    input_path = lane / "tmp" / (columns.stem + "-input.bin")
    oracle.write_wrf_input(input_path, inp, float(cols["dt"]), mp=8)
    reports = []
    for generation in ("44", "461"):
        output_path = lane / "tmp" / (columns.stem + "-" + generation + ".bin")
        build = lane / ("oracle" + generation)
        env = dict(os.environ)
        if generation == "44":
            env["GFORTRAN_CONVERT_UNIT"] = "big_endian:63,64,65"
        else:
            env.pop("GFORTRAN_CONVERT_UNIT", None)
        proc = subprocess.run([str(build / "run_columns_classic"), str(input_path), str(output_path)], cwd=build / "run", env=env, capture_output=True, text=True)
        (out / (columns.stem + "-" + generation + ".log")).write_text(proc.stdout + proc.stderr)
        if proc.returncode:
            raise RuntimeError((generation, proc.returncode, proc.stderr))
        wrf = oracle.read_wrf_output(output_path, len(cols["p"]), cols["p"].shape[1], mp=8)
        for name in ("re_cloud", "re_ice", "re_snow"):
            wrf[name] *= np.float32(1.e6)
        fields = {}
        for name, a in gpu.items():
            b = wrf[name]
            mismatch = a.view(np.uint32) != b.view(np.uint32)
            examples = []
            for index in np.argwhere(mismatch)[:8]:
                key = tuple(index)
                examples.append({"index": index.tolist(), "gpu": float(a[key]), "wrf": float(b[key]), "gpu_word": hex(int(a.view(np.uint32)[key])), "wrf_word": hex(int(b.view(np.uint32)[key])), "input": {k: float(v[key]) for k,v in cols.items() if v.ndim == 2 and v.shape == a.shape and k != "positions"}})
            fields[name] = {"words": a.size, "different": int(mismatch.sum()), "max_absolute": float(np.nanmax(np.abs(a.astype(np.float64) - b))), "gpu_sha256": hashlib.sha256(a.tobytes()).hexdigest(), "wrf_sha256": hashlib.sha256(b.tobytes()).hexdigest(), "examples": examples}
        from gpuwm.core import kernels
        code = kernels.module_source("thompson_column" if version == "wrf_44" else "thompson")
        reports.append({"gpu_generation": version,"git_tip":subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),"assembled_source_sha256":hashlib.sha256(code.encode()).hexdigest(), "backend":"host" if getattr(cp,"__gpuwm_host_backend__",False) else "GPU", "wrf_generation": generation, "strict": os.environ.get("GPUWM_WRF_EXACT") == "1", "columns": len(cols["p"]), "levels": cols["p"].shape[1], "words": sum(f["words"] for f in fields.values()), "different": sum(f["different"] for f in fields.values()), "fields": fields})
    (out / (columns.stem + ".json")).write_text(json.dumps(reports, indent=2) + "\n")
    for r in reports:
        print(columns.stem, version, r["wrf_generation"], r["different"], "of", r["words"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("extract", "compare"))
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--lane", type=Path)
    parser.add_argument("--version", default="wrf_461")
    parser.add_argument("--host", action="store_true", help="measure the same generated statements through the existing C++ host harness")
    parser.add_argument("--reader",type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.action == "extract":
        extract(args.source, args.out,args.reader)
    else:
        if args.host:
            sys.path.insert(0,str(ROOT/"tools/thompson_real_column_parity"))
            import host_backend
            host_backend.install()
        compare(args.source, args.lane, args.out, args.version)
