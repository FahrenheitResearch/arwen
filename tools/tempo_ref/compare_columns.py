"""Compare TEMPO port column outputs (out_cpu.bin, out_gpu.bin) with the
TEMPO Fortran reference (tempo_ref_columns.bin).  Writes a JSON summary.

Metric per field: max over levels and steps of |a-b| / max(|b|, floor),
with floors sized to each field's physical noise level, plus the count of
bitwise-identical values.
"""
import json
import sys

import numpy as np

NFLD = 17
FIELDS = ["t", "p", "w", "dz", "qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc",
          "nwfa", "nifa", "ng", "qb"]
SCAL = ["rain_precip", "ice_precip", "snow_precip", "graupel_precip", "frz_rain_precip",
        "max_hail_sfc", "max_hail_col"]
DIAG = ["refl10cm", "re_cloud", "re_ice", "re_snow"]
FLOOR = {"t": 1.0, "p": 1.0, "w": 1e-3, "dz": 1.0, "qv": 1e-8, "qc": 1e-9, "qr": 1e-9,
         "qi": 1e-9, "qs": 1e-9, "qg": 1e-9, "ni": 1e-2, "nr": 1e-2, "nc": 1e2,
         "nwfa": 1e3, "nifa": 1e1, "ng": 1e-2, "qb": 1e-9, "rain_precip": 1e-9,
         "ice_precip": 1e-9, "snow_precip": 1e-9, "graupel_precip": 1e-9,
         "frz_rain_precip": 1e-9, "max_hail_sfc": 1e-5, "max_hail_col": 1e-5,
         "refl10cm": 1.0, "re_cloud": 1e-7, "re_ice": 1e-7, "re_snow": 1e-7}


def read_ref(path):
    raw = np.fromfile(path, dtype="<f8")
    i = 0
    scen = []
    while i < raw.size:
        hdr = raw[i:i + 5].view("<i8")
        sid, nz, nsteps = int(hdr[0]), int(hdr[1]), int(hdr[2])
        dt, semi = raw[i + 3], raw[i + 4]
        i += 5
        i += NFLD * nz  # initial state
        step = NFLD * nz + 7 + 4 * nz
        steps = raw[i:i + step * nsteps].reshape(nsteps, step)
        i += step * nsteps
        scen.append(dict(sid=sid, nz=nz, nsteps=nsteps, dt=float(dt), semi=float(semi), steps=steps))
    return scen


def split(steps, nz):
    out = {}
    o = 0
    for f in FIELDS:
        out[f] = steps[:, o:o + nz]
        o += nz
    for s in SCAL:
        out[s] = steps[:, o:o + 1]
        o += 1
    for d in DIAG:
        out[d] = steps[:, o:o + nz]
        o += nz
    return out


def compare(a, b):
    res = {}
    for name in FIELDS[:] + SCAL + DIAG:
        x, y = a[name], b[name]
        floor = FLOOR[name]
        rel = np.abs(x - y) / np.maximum(np.abs(y), floor)
        res[name] = dict(max_rel=float(np.max(rel)),
                         bitwise_frac=float(np.mean(x.view("<i8") == y.view("<i8"))))
    return res


def main():
    ref_path, cpu_path, gpu_path, out_json = sys.argv[1:5]
    scen = read_ref(ref_path)
    cpu = np.fromfile(cpu_path, dtype="<f8")
    gpu = np.fromfile(gpu_path, dtype="<f8")
    summary = {"scenarios": []}
    o = 0
    for s in scen:
        nz, n = s["nz"], s["nsteps"]
        step = NFLD * nz + 7 + 4 * nz
        c = cpu[o:o + step * n].reshape(n, step)
        g = gpu[o:o + step * n].reshape(n, step)
        o += step * n
        R, C, G = split(s["steps"], nz), split(c, nz), split(g, nz)
        entry = dict(sid=s["sid"], dt=s["dt"], nsteps=n, semi=s["semi"],
                     cpu_vs_fortran=compare(C, R), gpu_vs_fortran=compare(G, R),
                     gpu_vs_cpu=compare(G, C))
        for key in ("cpu_vs_fortran", "gpu_vs_fortran", "gpu_vs_cpu"):
            entry[key + "_worst"] = max(((v["max_rel"], k) for k, v in entry[key].items()))
        # physical sanity: column totals at the end
        entry["final_ref_qg_max"] = float(R["qg"][-1].max())
        entry["final_ref_qr_max"] = float(R["qr"][-1].max())
        entry["ref_rain_precip_sum"] = float(R["rain_precip"].sum())
        entry["ref_graupel_precip_sum"] = float(R["graupel_precip"].sum())
        summary["scenarios"].append(entry)
        print(f"scenario {s['sid']} dt={s['dt']} n={n} semi={s['semi']}: "
              f"cpu-vs-F worst {entry['cpu_vs_fortran_worst']}, "
              f"gpu-vs-F worst {entry['gpu_vs_fortran_worst']}, "
              f"gpu-vs-cpu worst {entry['gpu_vs_cpu_worst']}")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
