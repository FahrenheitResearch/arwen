#!/usr/bin/env python3
"""Composite-reflectivity comparison figures for matched wrfout pairs.

For each requested domain/valid-time with both a gpuwm frame and a CPU
WRF reference frame present, renders one three-panel figure: CPU, GPU,
and GPU-minus-CPU composite reflectivity (column max of the model-native
REFL_10CM on both sides -- a direct field comparison, no derived
formula).  Reflectivity panels use the standard NWS dBZ scale (domain
convention); the difference panel uses a two-hue diverging map with a
neutral midpoint at 0 dBZ.  CPU-only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

# Standard NWS 5-dBZ-step reflectivity scale (5..75).
_NWS_COLORS = (
    "#04e9e7", "#019ff4", "#0300f4", "#02fd02", "#01c501", "#008e00",
    "#fdf802", "#e5bc00", "#fd9500", "#fd0000", "#d40000", "#bc0000",
    "#f800fd", "#9854c6",
)
_NWS_LEVELS = np.arange(5.0, 76.0, 5.0)


def _composite(path: Path) -> tuple[np.ndarray, str]:
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        times = ds.variables["Times"][:]
        stamp = b"".join(times[0].data).decode() if hasattr(
            times[0], "data") else "".join(
                x.decode() for x in times[0].astype(str))
        refl = np.asarray(ds.variables["REFL_10CM"][0], dtype=np.float32)
    return refl.max(axis=0), stamp


def render_pair(gpu_path: Path, cpu_path: Path, out_png: Path,
                *, title: str) -> dict:
    gpu, gpu_stamp = _composite(gpu_path)
    cpu, cpu_stamp = _composite(cpu_path)
    if gpu_stamp != cpu_stamp:
        raise ValueError(f"Times mismatch {gpu_stamp!r} vs {cpu_stamp!r}")
    diff = gpu.astype(np.float64) - cpu.astype(np.float64)

    cmap = ListedColormap(_NWS_COLORS)
    cmap.set_under("none")
    norm = BoundaryNorm(_NWS_LEVELS, cmap.N)

    fig, axes = plt.subplots(1, 3, figsize=(19.5, 6.4), dpi=150)
    for axis, (field, label) in zip(
            axes[:2], ((cpu, "CPU WRF v4.6.1"), (gpu, "gpuwm"))):
        im = axis.imshow(field, origin="lower", cmap=cmap, norm=norm,
                         interpolation="nearest")
        axis.set_title(label, fontsize=12)
        fig.colorbar(im, ax=axis, shrink=0.85, label="composite dBZ",
                     ticks=_NWS_LEVELS[::2])
    vmax = max(10.0, float(np.percentile(np.abs(diff), 99.5)))
    im = axes[2].imshow(diff, origin="lower", cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax, interpolation="nearest")
    axes[2].set_title("gpuwm - CPU", fontsize=12)
    fig.colorbar(im, ax=axes[2], shrink=0.85, label="delta dBZ")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#999999")

    interior = (slice(5, -5), slice(5, -5))
    a = gpu[interior].ravel().astype(np.float64)
    b = cpu[interior].ravel().astype(np.float64)
    corr = float(np.corrcoef(a, b)[0, 1])
    mae = float(np.abs(a - b).mean())
    fig.suptitle(
        f"{title}\ncomposite REFL_10CM -- interior corr {corr:.3f}, "
        f"MAE {mae:.2f} dBZ", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return {"corr": corr, "mae": mae, "stamp": gpu_stamp}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-dir", type=Path, required=True)
    parser.add_argument("--cpu-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stamps", nargs="+", required=True,
                        help="valid-time stamps, e.g. 1974-04-03_15_00_00")
    parser.add_argument("--domains", nargs="+",
                        default=("d01", "d02", "d03", "d04"))
    parser.add_argument("--title-prefix", default="matched comparison")
    args = parser.parse_args()

    rendered = []
    for stamp in args.stamps:
        for domain in args.domains:
            gpu = args.gpu_dir / f"wrfout_{domain}_{stamp}"
            cpu = args.cpu_dir / f"wrfout_{domain}_{stamp}"
            if not gpu.is_file() or not cpu.is_file():
                print(f"skip {domain} {stamp}: missing "
                      f"{'gpu' if not gpu.is_file() else 'cpu'} frame")
                continue
            out = args.out_dir / f"refl_{domain}_{stamp}.png"
            stats = render_pair(
                gpu, cpu, out,
                title=f"{args.title_prefix} {domain} {stamp}")
            print(f"{domain} {stamp}: corr {stats['corr']:.3f} "
                  f"MAE {stats['mae']:.2f} dBZ -> {out}")
            rendered.append(out)
    return 0 if rendered else 1


if __name__ == "__main__":
    raise SystemExit(main())
