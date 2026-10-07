"""The control's radial-velocity innovation, evaluated on each radar's window.

CPU only.  THE BREAKAGE THIS PREVENTS: the driver expanded five whole-domain
planes per radar to compute it, about 25 s of every 9 km CONUS analysis (some
140 radars) with seven cards idle.  On the window it must be the same residual
array, byte for byte, as the whole-domain evaluation.
"""
from __future__ import annotations

import numpy as np

from gpuwm.da.obs_radar import beam_unit_vectors, simulated_radial_velocity
from gpuwm.obs.radar_grid import radar_plane
from tools.da_cycle_prepared import control_vr_innovations


def _document(windowed=True):
    rng = np.random.default_rng(3)
    nz, ny, nx = 4, 9, 11
    # Stored planes are the largest window (7 x 9), smaller than the 9 x 11
    # domain, as a v2 writer stores them.
    windows = [(1, 5, 2, 8), (0, 6, 0, 8), (4, 8, 6, 10)]
    wy, wx = 7, 9
    variables = {}
    for name in ("vr_mask", "vr_obs", "vr_beam_east", "vr_beam_north",
                 "vr_beam_up"):
        planes = np.zeros((len(windows), nz, wy, wx), np.float32)
        for r, (j0, j1, i0, i1) in enumerate(windows):
            nj, ni = j1 - j0 + 1, i1 - i0 + 1
            values = rng.standard_normal((nz, nj, ni)).astype(np.float32)
            if name == "vr_mask":
                values = (values > 0.2).astype(np.float32)
            planes[r, :, :nj, :ni] = values
        variables[name] = planes.astype(np.int8) if name == "vr_mask" else planes
    doc = {"dims": {"level": nz, "south_north": ny, "west_east": nx},
           "radars": [{"id": f"K{r:03d}"} for r in range(len(windows))],
           "variables": variables}
    if windowed:
        doc["radar_windows"] = windows
    else:
        full = {}
        for name, planes in variables.items():
            full[name] = np.stack([radar_plane(dict(doc, radar_windows=windows), name, r)
                                   for r in range(len(windows))])
        doc["variables"] = full
    winds = tuple(rng.standard_normal((nz, ny, nx)) for _ in range(3))
    return doc, winds


def _reference(document, u, v, w):
    rows, pooled = [], []
    for index, radar in enumerate(document["radars"]):
        mask = np.asarray(radar_plane(document, "vr_mask", index)).astype(bool)
        row = {"radar": radar["id"], "points": int(mask.sum())}
        if mask.any():
            obs = np.asarray(radar_plane(document, "vr_obs", index), np.float64)
            sim = simulated_radial_velocity(u, v, w, beam_unit_vectors(document, index))
            d = obs[mask] - sim[mask]
            pooled.append(d)
            row["innovation_mean_ms"] = float(d.mean())
            row["innovation_rms_ms"] = float(np.sqrt(np.mean(d ** 2)))
        rows.append(row)
    return rows, pooled


def test_window_innovation_is_the_whole_domain_one_byte_for_byte():
    for windowed in (True, False):
        doc, (u, v, w) = _document(windowed)
        rows, pooled = control_vr_innovations(doc, u, v, w)
        ref_rows, ref_pooled = _reference(doc, u, v, w)
        assert rows == ref_rows
        assert len(pooled) == len(ref_pooled) == 3
        for a, b in zip(pooled, ref_pooled):
            assert a.tobytes() == b.tobytes()
