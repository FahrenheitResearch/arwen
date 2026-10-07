"""Mean absolute surface pressure tendency (MASPT) over a forecast's start.

The standard measure of insertion noise (Lynch and Huang 1992; the HRRR
and RAP digital-filter and IAU studies): the domain mean of
``|d p_sfc / dt|``.  A balanced start sits near the synoptic value, a
shocked one carries the acoustic and gravity-wave ring on top of it.  The
DA design logs it for every forecast (E2) and gates promotion on it (G4:
MASPT over the first 30 minutes no higher than the no-DA start).

The recorder wraps the domain's step function (the ``steppers`` seam of
:func:`gpuwm.core.model.execute_experiment`).  After each step it differences
the lowest model level's diagnosed pressure against the previous step's,
over the interior (the specified and relaxation rows excluded: the
boundary forcing is not insertion noise), and keeps the mean on the card;
the series is read back once, by :meth:`MasptRecorder.receipt`, so a forecast
pays no per-step host synchronization.  It reads the state and writes
nothing to it.

The lowest level's full pressure (``state.p[0]``, non-hydrostatic part
included) is used rather than a hydrostatic surface pressure on purpose:
the vertical acoustic ring the one-shot insertion launches is exactly what
a hydrostatic diagnostic would hide.  The column dry-mass tendency
``|d mu / dt|`` is recorded beside it (the external mode alone).
"""

from __future__ import annotations

import numpy as np

SCHEMA = "gpuwm-da.maspt.v1"

#: Pa/s to hPa/h.
_HPA_PER_HOUR = 3600.0 / 100.0


class MasptRecorder:
    """Per-step MASPT over ``[start, start + horizon]`` seconds."""

    def __init__(self, *, start_seconds: float, horizon_seconds: float = 3600.0,
                 margin: int = 0):
        if not horizon_seconds > 0:
            raise ValueError("the MASPT horizon must be positive")
        self.start = float(start_seconds)
        self.horizon = float(horizon_seconds)
        self.margin = int(margin)
        self._times = []
        self._device = []
        self._prev_p = None
        self._prev_mu = None
        self._prev_t = None
        self._before_insertion = None
        self._insertion = None

    @classmethod
    def for_config(cls, cfg, **kwargs):
        """The interior margin from the domain's boundary rows."""
        margin = 0
        if getattr(cfg, "specified", False) or getattr(cfg, "nested", False):
            margin = int(getattr(cfg, "spec_bdy_width", 0) or
                         (getattr(cfg, "spec_zone", 0)
                          + getattr(cfg, "relax_zone", 0)))
        return cls(margin=margin, **kwargs)

    def _interior(self, field):
        m = self.margin
        if m <= 0:
            return field
        return field[m:-m, m:-m]

    def _surface(self, state):
        """Interior lowest-level pressure, column mass and temperature."""
        from gpuwm.core import constants as c

        p = self._interior(state.p[0]).astype("float64")
        mu = self._interior(state.mup).astype("float64")
        thb = state.thb[0] if state.thb.ndim == 3 else state.thb[0]
        theta = self._interior(state.thp[0]).astype("float64") + (
            self._interior(thb).astype("float64")
            if getattr(thb, "ndim", 0) == 2 else float(thb))
        t = theta * (p / c.P0) ** c.RCP
        return p, mu, t

    def before_insertion(self, state) -> None:
        """Snapshot the background's surface before the analysis goes in,
        so the jump the insertion itself makes (S3: the f00 2 m T jump;
        the one-shot shock happens before any step) is measured too."""
        self._before_insertion = self._surface(state)

    def stepper(self, step):
        """``step`` wrapped: the step, then this step's tendency."""

        def maspt_step(state, run, **kwargs):
            before = float(state.elapsed_seconds)
            if before >= self.start + self.horizon:
                return step(state, run, **kwargs)
            if self._prev_p is None:
                self._prev_p, self._prev_mu, self._prev_t =                     self._surface(state)
                if self._before_insertion is not None:
                    p0, mu0, t0 = self._before_insertion
                    xp0 = np
                    if hasattr(p0, "__cuda_array_interface__"):
                        import cupy as xp0  # noqa: N813
                    self._insertion = xp0.stack([
                        xp0.abs(self._prev_p - p0).mean(),
                        xp0.abs(self._prev_mu - mu0).mean(),
                        xp0.abs(self._prev_t - t0).mean(),
                        xp0.abs(self._prev_t - t0).max()])
                    self._before_insertion = None
            result = step(state, run, **kwargs)
            after = float(state.elapsed_seconds)
            if after <= before:
                after = before + float(getattr(run, "dt"))
            p_now, mu_now, t_now = self._surface(state)
            dt = after - before
            xp = np
            if hasattr(p_now, "__cuda_array_interface__"):
                import cupy as xp  # noqa: N813
            self._device.append(xp.stack([
                xp.abs(p_now - self._prev_p).mean() / dt,
                xp.abs(mu_now - self._prev_mu).mean() / dt,
                xp.abs(t_now - self._prev_t).mean() / dt]))
            self._times.append(after)
            self._prev_p, self._prev_mu, self._prev_t = p_now, mu_now, t_now
            return result

        return maspt_step

    def series(self):
        """``(times_s, maspt_hpa_per_h, mu_tendency_hpa_per_h,
        lowest_level_t_tendency_k_per_h)``, host."""
        if not self._device:
            return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
        stacked = [np.asarray(getattr(v, "get", lambda: v)(), dtype=np.float64)
                   for v in self._device]
        values = np.stack(stacked)
        return (np.asarray(self._times), values[:, 0] * _HPA_PER_HOUR,
                values[:, 1] * _HPA_PER_HOUR, values[:, 2] * 3600.0)

    def mean_over(self, seconds: float):
        """Time-weighted MASPT over ``[start, start + seconds]``."""
        times, maspt, mu, _t = self.series()
        if times.size == 0:
            return None, None
        edges = np.concatenate([[self.start], times])
        width = np.clip(np.minimum(edges[1:], self.start + seconds)
                        - edges[:-1], 0.0, None)
        if width.sum() <= 0:
            return None, None
        return (float((maspt * width).sum() / width.sum()),
                float((mu * width).sum() / width.sum()))

    def receipt(self) -> dict:
        times, maspt, mu, tlow = self.series()
        m30, mu30 = self.mean_over(1800.0)
        m60, mu60 = self.mean_over(3600.0)
        insertion = None
        if self._insertion is not None:
            jump = np.asarray(getattr(self._insertion, "get",
                                      lambda: self._insertion)(), np.float64)
            insertion = {"lowest_level_p_mean_abs_hpa": float(jump[0]) / 100.0,
                         "mu_mean_abs_hpa": float(jump[1]) / 100.0,
                         "lowest_level_t_mean_abs_k": float(jump[2]),
                         "lowest_level_t_max_abs_k": float(jump[3])}
        return {
            "insertion_jump": insertion,
            "schema": SCHEMA,
            "units": "hPa/h",
            "start_seconds": self.start,
            "interior_margin_cells": self.margin,
            "maspt_first_30min": m30,
            "maspt_first_60min": m60,
            "mu_tendency_first_30min": mu30,
            "mu_tendency_first_60min": mu60,
            "series": {
                "minutes": [round((t - self.start) / 60.0, 4) for t in times],
                "maspt": [float(v) for v in maspt],
                "mu_tendency": [float(v) for v in mu],
                "lowest_level_t_tendency_k_per_h": [float(v) for v in tlow],
            },
            "rule": ("mean over the interior of |p(k=0) step difference| "
                     "/ step length, per step; the 30 and 60 min values "
                     "are step-length weighted means"),
        }
