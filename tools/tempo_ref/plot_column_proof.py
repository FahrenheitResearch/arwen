"""Analysis charts for the TEMPO column proof (not weather-field maps).

1. tempo-column-profiles.png: scenario 2 after 30 min, TEMPO Fortran vs
   WOOF GPU port, six hydrometeor fields by model level.
2. tempo-column-error-growth.png: worst relative difference per step,
   GPU port vs Fortran for scenarios 1, 2, 4 and, for scenario 3
   (semi-Lagrangian), GPU vs Fortran beside the Fortran-equivalent CPU port
   run from an initial graupel field changed by one part in 1e15.
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import compare_columns as cc


def per_step_worst(A, B, fields):
    n = A[fields[0]].shape[0]
    out = np.zeros(n)
    for f in fields:
        rel = np.abs(A[f] - B[f]) / np.maximum(np.abs(B[f]), cc.FLOOR[f])
        out = np.maximum(out, rel.max(axis=1))
    return out


def main():
    ref, gpu_p, pert_p, outdir = sys.argv[1:5]
    scen = cc.read_ref(ref)
    gpu = np.fromfile(gpu_p, dtype="<f8")
    pert = np.fromfile(pert_p, dtype="<f8")
    fields = ["t", "qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc", "nwfa", "nifa", "ng", "qb"]
    o = 0
    curves = {}
    prof = None
    for s in scen:
        nz, n = s["nz"], s["nsteps"]
        step = cc.NFLD * nz + 7 + 4 * nz
        G = cc.split(gpu[o:o + step * n].reshape(n, step), nz)
        X = cc.split(pert[o:o + step * n].reshape(n, step), nz)
        o += step * n
        R = cc.split(s["steps"], nz)
        curves[s["sid"]] = (s, per_step_worst(G, R, fields), per_step_worst(X, R, fields))
        if s["sid"] == 2:
            prof = (R, G, nz)

    R, G, nz = prof
    lev = np.arange(1, nz + 1)
    fig, axes = plt.subplots(1, 6, figsize=(16, 6), sharey=True)
    for ax, (f, unit, scale) in zip(axes, [("qc", "g/kg", 1e3), ("qr", "g/kg", 1e3),
                                          ("qs", "g/kg", 1e3), ("qg", "g/kg", 1e3),
                                          ("ng", "1/kg", 1.0), ("qb", "L/kg x1000", 1e3)]):
        ax.plot(R[f][-1] * scale, lev, color="#1f4e79", lw=3, label="TEMPO Fortran")
        ax.plot(G[f][-1] * scale, lev, color="#e07b00", lw=1.2, ls="--", label="WOOF GPU port")
        ax.set_title(f"{f} ({unit})")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("model level")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("TEMPO v3.0.3 column, 30 min at dt 20 s (scenario 2): Fortran vs GPU port")
    fig.tight_layout()
    fig.savefig(f"{outdir}/tempo-column-profiles.png", dpi=110)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {1: "#1f4e79", 2: "#2e8b57", 4: "#7b3fa0", 5: "#d4a017", 6: "#17a2b8"}
    for sid, (s, g, x) in curves.items():
        t = np.arange(1, len(g) + 1) * s["dt"] / 60.0
        if sid == 3:
            ax.plot(t, np.maximum(g, 1e-17), color="#c0392b", lw=1.5,
                    label="semi-Lagrangian: GPU port vs Fortran")
            ax.plot(t, np.maximum(x, 1e-17), color="#c0392b", lw=1.5, ls=":",
                    label="semi-Lagrangian: CPU port, graupel x(1+1e-15), vs Fortran")
        else:
            ax.plot(t, np.maximum(g, 1e-17), color=colors[sid], lw=1.5,
                    label=f"scenario {sid} (dt {s['dt']:.0f} s{', +8 K' if sid == 5 else (', -18 K' if sid == 6 else '')}): GPU port vs Fortran")
    ax.set_yscale("log")
    ax.set_xlabel("minutes")
    ax.set_ylabel("worst relative difference over all fields and levels")
    ax.axhline(1e-6, color="gray", lw=0.8, ls="--")
    ax.text(0.5, 1.4e-6, "stated tolerance 1e-6", color="gray", fontsize=8)
    ax.set_title("TEMPO column: GPU port agreement with TEMPO Fortran, per step")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(f"{outdir}/tempo-column-error-growth.png", dpi=110)


if __name__ == "__main__":
    main()
