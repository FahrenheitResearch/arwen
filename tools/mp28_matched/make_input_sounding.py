"""Write the matched idealized case's ``input_sounding`` for WRF.

The two models must be given the SAME sounding.  gpuwm evaluates the
Weisman-Klemp (1982) analytic profile directly
(``gpuwm.verify.cases.wk82.wk82_sounding``); WRF's idealized initializer has
no analytic branch and reads a tabulated ``input_sounding`` through
``dyn_em/module_initialize_ideal.F::get_sounding``.  This script evaluates
gpuwm's own analytic profile on a fine height grid and writes it in WRF's
file format, so the tabulation WRF reads IS gpuwm's profile.

Winds are identically zero: the matched case is deliberately unsheared so no
storm-relative translation carries the convection around the periodic domain.

Format (WRF ``get_sounding``, dyn_em/module_initialize_ideal.F):

    p_sfc(hPa)  theta_sfc(K)  qv_sfc(g/kg)
    z(m)  theta(K)  qv(g/kg)  u(m/s)  v(m/s)
    ...

Usage:  python make_input_sounding.py OUTPUT_PATH [ZTOP_M] [DZ_M]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Import gpuwm's own analytic sounding -- the point of the exercise.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gpuwm.verify.cases.wk82 import wk82_sounding  # noqa: E402

#: Surface pressure of the WK82/em_quarter_ss environment (Pa).
P_SURF = 1.0e5


def write_sounding(path: Path, z_max: float = 25000.0,
                   dz: float = 100.0) -> np.ndarray:
    """Write the sounding and return the ``(n, 5)`` table that was written."""
    z = np.arange(0.0, z_max + 0.5 * dz, dz)
    theta, qv = wk82_sounding(z, p_surf=P_SURF)
    theta_sfc, qv_sfc = float(theta[0]), float(qv[0])

    rows = np.column_stack([z[1:], theta[1:], qv[1:] * 1000.0,
                            np.zeros(z.size - 1), np.zeros(z.size - 1)])

    lines = [f"{P_SURF / 100.0:12.4f} {theta_sfc:12.6f} "
             f"{qv_sfc * 1000.0:12.8f}"]
    for zz, th, q, u, v in rows:
        lines.append(f"{zz:12.4f} {th:12.6f} {q:12.8f} {u:10.6f} {v:10.6f}")
    path.write_text("\n".join(lines) + "\n")
    return rows


if __name__ == "__main__":
    out = Path(sys.argv[1])
    zmax = float(sys.argv[2]) if len(sys.argv) > 2 else 25000.0
    step = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0
    table = write_sounding(out, zmax, step)
    print(f"wrote {out} : {table.shape[0] + 1} levels, "
          f"0 to {table[-1, 0]:.0f} m")
