"""``mp_tend_radar = 1`` on the WRF-input door: the radar tendency the WRF
input carries, applied as the operational WRF fork applies it.

What WRF does
-------------
NOAA's operational WRF (NOAA-EMC/HRRR v4.1.21, WRFV3.9,
``dyn_em/module_big_step_utilities_em.F:5901-6005``) reads four
radar-derived temperature-tendency slots, ``RAD_TTEN_DFI_1..4``, and their
end times ``TTEN_TIMES`` (minutes) out of the WRF input the cloud analysis
wrote.  On every microphysics step, with ``mp_tend_radar = 1``, it picks the
first slot whose time the clock (``xtime``, minutes at the beginning of the
step) has not passed; the loop stops at slot 4, so the last slot holds for
the rest of the run.  Below the model top, where the slot value is in
``[-1, 1]`` K/s (radar coverage) the tendency times ``dt`` REPLACES the
clamped microphysics heating; elsewhere the clamped microphysics heating is
kept.  The clamp is the namelist's ``mp_tend_lim``.

What this module does
---------------------
:func:`read_radar_tendency` reads those five variables from the same file
the state is restored from; :class:`gpuwm.wrfinput_radar_forcing.WrfinputRadarTendency`
(kept out of this preparation package, which the standalone preparation
wheel ships without the forecast runtime) attaches one
:class:`gpuwm.da.radar_tten.RadarTtenForcing` per forced domain (one slab
per card on a multi-card domain) after the run's steppers are known.  The
application is the device kernel ``rtt_apply`` of
:mod:`gpuwm.da.radar_tten`, a line-for-line port of the lines above.  The
slot clock is the domain's own model clock, read at each microphysics call,
so a rewind (health recovery) or a resumed checkpoint picks the slot WRF
would pick at that time.  The case's ``mp_tend_lim`` is kept, because that
is what the namelist that asked for the forcing says.

The importer (:func:`gpuwm.namelist_import._consume_unrun_physics_controls`)
admits ``mp_tend_radar = 1`` only on domains whose input carries all five
variables; on any other door it is refused by name.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

#: Receipt schema of :meth:`WrfinputRadarTendency.receipt`.
SCHEMA = "gpuwm.wrfinput-radar-tendency.v1"


class WrfinputRadarTendencyRefused(ValueError):
    """The WRF input's radar tendency cannot be applied as WRF applies it."""


@dataclass(frozen=True)
class RadarTendency:
    """One domain's slots (host float32 ``(nz, ny, nx)``) and end times."""

    path: Path
    slots: tuple[np.ndarray, ...]
    slot_minutes: tuple[float, ...]

    def receipt(self) -> dict:
        return {
            "file": str(self.path),
            "slot_minutes": list(self.slot_minutes),
            "covered_points_by_slot": [
                int(np.count_nonzero((slot[:-1] >= -1.0) & (slot[:-1] <= 1.0)))
                for slot in self.slots],
            "max_heating_k_per_s_by_slot": [
                float(np.max(np.where((slot[:-1] >= -1.0) & (slot[:-1] <= 1.0),
                                      slot[:-1], -np.inf), initial=-np.inf))
                for slot in self.slots],
        }


def forced_domains(namelist_input, grid_ids) -> frozenset[int]:
    """Every grid id when ``&physics mp_tend_radar`` is 1, else none.

    The fork declares the switch with one entry for every domain
    (``Registry.EM_COMMON:2441``).  The importer has already validated it
    (one value, 0 or 1, and 1 only when every domain's input carries the
    tendency); this reads the same file again so the forecast and the
    import report cannot disagree.
    """
    from gpuwm.fortran_namelist import parse_namelist

    column = parse_namelist(Path(namelist_input)).get(
        "physics", {}).get("mp_tend_radar") or [0]
    if int(column[0]) != 1:
        return frozenset()
    return frozenset(int(gid) for gid in grid_ids)


def read_radar_tendency(path, *, nz: int, ny: int, nx: int) -> RadarTendency:
    """``RAD_TTEN_DFI_1..4`` and ``TTEN_TIMES`` from one WRF input."""
    from gpuwm import netcdf_bridge
    from gpuwm.ingest.wrfinput import RADAR_TENDENCY_VARIABLES

    path = Path(path)
    with netcdf_bridge.open_dataset(path) as dataset:
        missing = [name for name in RADAR_TENDENCY_VARIABLES
                   if name not in dataset.variables]
        if missing:
            raise WrfinputRadarTendencyRefused(
                f"{path} has no {', '.join(missing)}: mp_tend_radar = 1 "
                "applies the cloud analysis's radar tendency, and without it "
                "wrf.exe reads zero-filled slots and removes the "
                "microphysics heating everywhere")
        slots = []
        for name in RADAR_TENDENCY_VARIABLES[:4]:
            value = np.asarray(dataset.variables[name][...], dtype=np.float32)
            if value.ndim == 4:
                if value.shape[0] != 1:
                    raise WrfinputRadarTendencyRefused(
                        f"{path}: {name} has {value.shape[0]} Time records; "
                        "a WRF input carries one")
                value = value[0]
            if value.shape != (nz, ny, nx):
                raise WrfinputRadarTendencyRefused(
                    f"{path}: {name} is {value.shape} and the domain is "
                    f"{(nz, ny, nx)}; read at these indices it would heat "
                    "other columns")
            if not np.all(np.isfinite(value)):
                raise WrfinputRadarTendencyRefused(
                    f"{path}: {name} holds non-finite values")
            slots.append(np.ascontiguousarray(value))
        times = np.asarray(dataset.variables["TTEN_TIMES"][...],
                           dtype=np.float32).reshape(-1)
    if times.size != 4 or not np.all(np.isfinite(times)):
        raise WrfinputRadarTendencyRefused(
            f"{path}: TTEN_TIMES is {times.tolist()}; WRF reads four finite "
            "slot end times in minutes")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise WrfinputRadarTendencyRefused(
            f"{path}: TTEN_TIMES {times.tolist()} do not increase.  WRF walks "
            "them in order and stops at the first time the clock has not "
            "passed, so a slot after a later time is never read; this input "
            "was not written by the cloud analysis's four-window driver")
    return RadarTendency(path, tuple(slots),
                         tuple(float(value) for value in times))


def prepare(run, experiment) -> dict[int, RadarTendency]:
    """Read every forced domain's tendency on CPU, before a card is taken.

    ``run`` is :class:`gpuwm.wrfinput_door.WrfinputRun`.  The state file
    (``wrfinput_d0N`` on a cold start, ``wrfrst_d0N`` on a warm start) is
    the one the tendency is read from, as WRF reads it from the file it
    starts from.
    """
    forced = forced_domains(run.namelist_input,
                            [domain.grid_id for domain in experiment.domains])
    if not forced:
        return {}
    relocation = getattr(experiment, "relocation", None)
    moving = {int(dc.grid_id) for dc in experiment.domains
              if getattr(dc, "follow", None) is not None}
    if relocation is not None and getattr(relocation, "enabled", False):
        moving.add(int(relocation.grid_id))
    if forced & moving:
        raise WrfinputRadarTendencyRefused(
            "mp_tend_radar = 1 on a moving nest ("
            + ", ".join(f"d{gid:02d}" for gid in sorted(forced & moving))
            + "): the tendency is a field on the input's fixed placement, "
            "and after a move it would heat the columns the nest left")
    files = (run.restart_paths if getattr(run, "restart_paths", None) is not None
             else run.wrfinput_paths)
    tendencies = {}
    for domain in experiment.domains:
        gid = int(domain.grid_id)
        if gid not in forced:
            continue
        cfg = domain.run
        if int(getattr(cfg, "mp_physics", 0)) == 0:
            raise WrfinputRadarTendencyRefused(
                f"d{gid:02d}: mp_tend_radar = 1 replaces the microphysics "
                "heating, and mp_physics = 0 runs no microphysics step, so "
                "the tendency would never be applied")
        tendencies[gid] = read_radar_tendency(
            files[gid], nz=int(cfg.nz), ny=int(cfg.ny), nx=int(cfg.nx))
    return tendencies


__all__ = ["RadarTendency", "SCHEMA",
           "WrfinputRadarTendencyRefused", "forced_domains", "prepare",
           "read_radar_tendency"]
