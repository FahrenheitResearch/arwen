"""Where the urban models run inside the surface step, and what they read.

WRF runs the single-layer UCM INSIDE Noah's column loop
(module_sf_noahdrv.F:1317-1600) and BEP / BEP_BEM after it (:1603-1790);
Noah-MP calls ``noahmp_urban`` from the surface driver after ``noahmplsm``
(module_surface_driver.F:3184).  Columns are independent, so gpuwm runs every
arm as a post-LSM kernel, which is equivalent PROVIDED the land-surface
scheme hands over the rural values it computed and nobody re-derives them.
:class:`UrbanCoupler` is that handover and the two call points:

1. the LSM runner calls :meth:`UrbanCoupler.noah_kernel_args` (Noah) or
   :meth:`UrbanCoupler.noahmp_vegtype_remap` (Noah-MP) to get the urban
   inputs of its own solve (NATURAL vegetation on urban columns, the rural
   skin temperature);
2. :meth:`UrbanCoupler.after_lsm` -- before the runner zeroes ``rainbl``,
   because the UCM and BEM read RAINBL -- snapshots the rural values into
   ``UrbanState.rural`` and calls the model module's ``after_lsm``;
3. :meth:`UrbanCoupler.after_surface_diagnostics` -- after Noah's SFCDIAGS
   and before the PBL -- calls the module's 2 m overrides
   (module_surface_driver.F:3001-3035 Noah, :3383-3423 Noah-MP).

The model modules (``URBAN_MODEL_MODULES`` in :mod:`gpuwm.config`) are loaded
by NAME; each exports ``STATE_SPEC``, ``DIMENSIONS``, ``init_state``,
``after_lsm`` and ``after_surface_diagnostics`` (DESIGN.md section 3.4).

PBL contract for options 2/3 (the bep lane implements both ends):

* YSU: ``gpuwm.core.ysu.launch_ysu(..., bep=pbl_terms, frc_urb2d=...)``;
  without ``bep`` the non-BEP path is untouched.
* MYJ: ``gpuwm.core.myjurb.launch_myjurb(columns, surface, state, tke, *,
  dtturbl, bep, frc_urb2d, ht)`` with the same ``columns``/``surface``/
  ``state`` mappings :func:`gpuwm.core.myjpbl.myj_pbl_step` takes and the
  same output keys; ``myjurb`` has no QI/QS/QR/QG tendencies
  (module_bl_myjurb.F) and seeds ZINT with the terrain height HT.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Mapping

import numpy as np

from gpuwm.config import URBAN_MODEL_MODULES
from gpuwm.core.urban_state import (NOAH_RURAL_FIELD_SOURCES,
                                    NOAH_RURAL_KERNEL_FIELDS,
                                    NOAHMP_RURAL_FIELDS, UrbanSolar,
                                    UrbanState, load_model_module)

#: The model module per option, as DESIGN.md names it.
URBAN_MODELS = URBAN_MODEL_MODULES


class UrbanModelMissing(RuntimeError):
    """An urban option reached the driver without its model module."""


def solar_geometry(start_time: datetime, elapsed_seconds: float,
                   latitude_deg, longitude_deg, *,
                   radt_seconds: float) -> UrbanSolar:
    """WRF's radconst + calc_coszen at a radiation call.

    module_radiation_driver.F:1188-1208: DECLIN from the call-time Julian
    day, COSZEN and HRANG at ``xtime + radt/2``.  The same arithmetic as the
    COSZEN carrier's analytic provider (:func:`gpuwm.core.dudhia.
    wrf_solar_geometry`), so the urban hour angle and the LSM's COSZEN are
    one sun.
    """
    valid = start_time + timedelta(seconds=float(elapsed_seconds))
    hour = (valid.hour + valid.minute / 60.0 + valid.second / 3600.0
            + valid.microsecond / 3.6e9)
    julian = valid.timetuple().tm_yday - 1.0 + hour / 24.0
    degrad = np.pi / 180.0
    dpd = 360.0 / 365.0
    slong = dpd * (julian - 80.0 if julian >= 80.0 else julian + 285.0)
    declin = float(np.arcsin(np.sin(23.5 * degrad) * np.sin(slong * degrad)))
    da = 2.0 * np.pi * (julian - 1.0) / 365.0
    eot = 229.18 * (0.000075 + 0.001868 * np.cos(da) - 0.032077 * np.sin(da)
                    - 0.014615 * np.cos(2.0 * da)
                    - 0.04089 * np.sin(2.0 * da))
    lat = np.asarray(latitude_deg, dtype=np.float64)
    lon = np.asarray(longitude_deg, dtype=np.float64)
    offset = 0.5 * float(radt_seconds)
    minutes = 60.0 * (hour + offset / 3600.0) + eot + 4.0 * lon
    hrang = np.deg2rad(minutes / 4.0 - 180.0)
    xx = np.deg2rad(lat)
    coszen = np.clip(np.sin(xx) * np.sin(declin)
                     + np.cos(xx) * np.cos(declin) * np.cos(hrang), -1.0, 1.0)
    # GMT and JULDAY are the RUN's start hour and start day of year, held
    # for the whole run (grid%gmt, grid%julday); JULIAN moves with XTIME.
    return UrbanSolar(
        declin=declin, coszen=coszen.astype(np.float32),
        hrang=hrang.astype(np.float32), xlat=lat.astype(np.float32),
        xlong=lon.astype(np.float32),
        gmt=float(start_time.hour + start_time.minute / 60.0
                  + start_time.second / 3600.0),
        julday=int(start_time.timetuple().tm_yday), julian=float(julian),
        julyr=int(valid.year), model_time=float(elapsed_seconds),
        # module_radiation_driver.F radconst (Paltridge & Platt orbit), at
        # the call-time Julian day, as gpuwm.core.dudhia computes it.
        solcon=float(1370.0 * (
            1.000110 + 0.034221 * np.cos(2.0 * np.pi * julian / 365.0)
            + 0.001280 * np.sin(2.0 * np.pi * julian / 365.0)
            + 0.000719 * np.cos(4.0 * np.pi * julian / 365.0)
            + 0.000077 * np.sin(4.0 * np.pi * julian / 365.0))))


class UrbanCoupler:
    """The urban model's two call points and the LSM handover."""

    def __init__(self, state: UrbanState, *, lsm: int,
                 start_time: datetime | None, latitude_deg=None,
                 longitude_deg=None, module=None) -> None:
        self.state = state
        self.lsm = int(lsm)
        self.module = module if module is not None else load_model_module(
            state.option)
        if self.module is None:
            raise UrbanModelMissing(
                f"sf_urban_physics={state.option}: option {state.option}'s "
                f"kernel is not in this build ({URBAN_MODELS[state.option]} "
                "is absent); validate_run_config refuses this at the door")
        if start_time is None or latitude_deg is None or longitude_deg is None:
            raise ValueError(
                "an urban model needs the domain's start time, latitude and "
                "longitude: the UCM reads DECLIN, COSZEN, HRANG and XLAT and "
                "BEP/BEM the same plus GMT/JULDAY/XLONG, and a silent zero "
                "would put every city on the equator at New Year")
        self.start_time = start_time
        self.latitude_deg = np.asarray(latitude_deg, dtype=np.float32)
        self.longitude_deg = np.asarray(longitude_deg, dtype=np.float32)
        self.solar_updates = 0

    # -- solar ------------------------------------------------------------
    def update_solar(self, elapsed_seconds: float, radt_seconds: float) -> None:
        """Refresh the held solar geometry on the radiation cadence."""
        import cupy as cp

        host = solar_geometry(self.start_time, elapsed_seconds,
                              self.latitude_deg, self.longitude_deg,
                              radt_seconds=radt_seconds)
        solar = self.state.solar
        solar.declin = host.declin
        solar.coszen = cp.asarray(host.coszen)
        solar.hrang = cp.asarray(host.hrang)
        solar.xlat = cp.asarray(host.xlat)
        solar.xlong = cp.asarray(host.xlong)
        solar.gmt = host.gmt
        solar.julday = host.julday
        solar.julian = host.julian
        solar.julyr = host.julyr
        solar.solcon = host.solcon
        solar.model_time = host.model_time
        self.solar_updates += 1

    def split_shortwave(self, fields: dict, *, ht) -> None:
        """The radiation driver's direct/diffuse split for a shortwave
        scheme that does not compute one (module_radiation_driver.F:
        2882-2914, Ruiz-Arias et al. 2010 with the Perez air-mass
        correction), on this radiation call's sun.  Zero where the sun is
        down, as the driver zeroes both before every call (:1724-1726).
        """
        import cupy as cp

        f4 = cp.float32
        solar = self.state.solar
        cosz = cp.asarray(solar.coszen, dtype=f4)
        swdown = fields["swdown"]
        day = cosz > f4(1.0e-3)
        c = cp.where(day, cosz, f4(1.0))
        ioh = f4(solar.solcon) * c
        kt = swdown / cp.maximum(ioh, f4(1.0e-3))
        airmass = cp.exp(-cp.asarray(ht, dtype=f4) / f4(8434.5)) / (
            c + f4(0.50572) * cp.power(
                cp.arcsin(c) * f4(57.295779513082323) + f4(6.07995),
                f4(-1.6364)))
        kt = kt / (f4(0.1) + f4(1.031) * cp.exp(
            f4(-1.4) / (f4(0.9) + f4(9.4) / cp.maximum(airmass, f4(1.0e-3)))))
        kd = f4(0.952) - f4(1.041) * cp.exp(-cp.exp(f4(2.300) - f4(4.702) * kt))
        fields["swddif"][...] = cp.where(day, kd * swdown, f4(0.0))
        fields["swddir"][...] = cp.where(day, (f4(1.0) - kd) * swdown,
                                         f4(0.0))

    # -- the LSM handover --------------------------------------------------
    def noah_kernel_args(self) -> dict:
        """What ``launch_noah(urban=...)`` needs for its pre-SFLX remap."""
        import cupy as cp

        s = self.state
        f = s.fields
        rural = s.rural
        shape = f["tsk"].shape
        for name in NOAH_RURAL_KERNEL_FIELDS:
            if name not in rural or rural[name].shape != shape:
                rural[name] = cp.zeros(shape, dtype=cp.float32)
        return {
            "option": s.option,
            "category_mask": s.category_lookup,
            "natural": int(s.categories.natural),
            "frc_urb2d": f["frc_urb2d"],
            "ts_urb2d": f["ts_urb2d"],
            "tsk_rural_bep": f.get("tsk_rural_bep"),
            "rural_q1": rural["q1"], "rural_q2k": rural["q2k"],
            "rural_zlvl": rural["zlvl"],
        }

    def noahmp_vegtype_remap(self) -> dict:
        """noahmpdrv.F:916-925 under ``sf_urban_physics > 0``."""
        return {"option": self.state.option,
                "natural": int(self.state.categories.natural)}

    def _snapshot_rural(self, fields: dict) -> None:
        rural = self.state.rural
        if self.lsm == 2:
            sources = NOAH_RURAL_FIELD_SOURCES
        else:
            sources = {name: name for name in NOAHMP_RURAL_FIELDS}
        for local, source in sources.items():
            if source not in fields:
                continue
            held = rural.get(local)
            if held is None or held.shape != fields[source].shape:
                rural[local] = fields[source].copy()
            else:
                held[...] = fields[source]
        # TSK_RURAL (Noah) is T1 after SFLX on every land column
        # (module_sf_noahdrv.F:1242); water keeps TSK (:869-876).
        self.state.fields["tsk_rural"][...] = fields["tsk"]

    def after_lsm(self, fields: dict, atmosphere: Mapping, cfg, *,
                  dt: float, itimestep: int) -> None:
        """Snapshot the rural values, then run the option's model."""
        if self.state.option == 1 and int(getattr(cfg, "sf_surface_mosaic", 0)) == 1:
            # noahdrv.F:3733-4057 runs UCM inside the reverse tile loop.
            # Calling the grid blend here would integrate urban state twice.
            return
        self._snapshot_rural(fields)
        self.module.after_lsm(
            self.state, self.state.params, lsm=self.lsm, fields=fields,
            atmosphere=atmosphere, dt=float(dt), itimestep=int(itimestep),
            solar=self.state.solar, cfg=cfg)

    def after_surface_diagnostics(self, fields: dict, atmosphere: Mapping,
                                  cfg) -> None:
        self.module.after_surface_diagnostics(
            self.state, lsm=self.lsm, fields=fields, atmosphere=atmosphere,
            cfg=cfg)


__all__ = ["URBAN_MODELS", "UrbanCoupler", "UrbanModelMissing",
           "solar_geometry"]
