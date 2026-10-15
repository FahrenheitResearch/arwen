"""Analysis chart of column statistics, not a weather-field map."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tools.noahmp_snow_age_oracle.full_hashes import receipt

root, output = map(Path, sys.argv[1:])
curves = {generation: receipt(root / "full-input.bin", root / ("wrf" + generation + ".bin"))["diagnostics"]
          for generation in ("461", "470")}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
for generation, label, color in (("470", "Corrected", "#17684a"),
                                 ("461", "4.6.1 reproduction", "#9c4e25")):
    rows = curves[generation]
    x = [r["step"] for r in rows]
    axes[0].plot(x, [r["tauss_max"] for r in rows], "o-", label=label, color=color)
    axes[1].plot(x, [r["shortwave_absorbed_max"] for r in rows], "o-", label=label, color=color)
for ax in axes:
    ax.axvspan(0.5, 3.5, color="#dfe6eb", alpha=0.65)
    ax.set_xlabel("Step (300 seconds)")
    ax.grid(alpha=0.2)
    ax.set_xticks(range(7))
axes[0].set_ylabel("Maximum TAUSS (dimensionless)")
axes[1].set_ylabel("Maximum absorbed shortwave (W/m²)")
axes[0].set_title("Snow age advances at night")
axes[1].set_title("Nighttime solar flux remains zero")
axes[0].legend(frameon=False, fontsize=9)
fig.suptitle("30 Noah-MP columns, CLASS albedo, pinned release Fortran oracles", fontsize=12)
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=170)
print(output)
