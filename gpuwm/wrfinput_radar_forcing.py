"""The forecast-side half of ``mp_tend_radar = 1`` on the WRF-input door.

:mod:`gpuwm.ingest.wrfinput_radar` reads the radar tendency the WRF input
carries; this module attaches it to the running domains.  It imports the
dycore and the device kernel, so it lives outside the preparation package
(the standalone preparation wheel ships without the forecast runtime).
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from gpuwm.ingest.wrfinput_radar import (SCHEMA, RadarTendency,
                                         WrfinputRadarTendencyRefused)


def _clocked_forcing_class():
    from gpuwm.da.radar_tten import RadarTtenForcing

    class ModelClockForcing(RadarTtenForcing):
        """A forcing whose slot clock is the domain's model clock.

        WRF picks the slot from ``xtime`` at the beginning of the step; the
        domain clock at the microphysics call is that time.  Reading it at
        every call keeps the slot right across a health-recovery rewind or a
        resumed checkpoint, where a counter that only adds ``dt`` would not.
        """

        def __init__(self, *args, clock=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._clock = clock

        def before_microphysics(self, state, cfg, dt):
            if self._clock is not None:
                self.elapsed_seconds = float(self._clock())
            super().before_microphysics(state, cfg, dt)

    return ModelClockForcing


class WrfinputRadarTendency:
    """The run's attached forcings, their detach, and the receipt."""

    def __init__(self, tendencies: Mapping[int, RadarTendency]):
        self.tendencies = dict(tendencies)
        self._attached: list = []
        self._receipts: list = []
        self.roads: dict[int, str] = {}

    def attach(self, nodes: Mapping[int, object], steppers) -> None:
        from gpuwm.core import dycore
        from gpuwm.da import radar_tten

        forcing_class = _clocked_forcing_class()
        for gid, tendency in sorted(self.tendencies.items()):
            node = nodes[gid]
            clock = (lambda node=node: node.clock.elapsed_seconds)
            stepper = (steppers or {}).get(int(gid))
            provenance = {"source": "the WRF input's RAD_TTEN_DFI_1..4 and "
                                    "TTEN_TIMES", "file": str(tendency.path)}
            if stepper is None or stepper is dycore.step:
                if getattr(node.state, "_streamed_domain", None) is not None:
                    raise WrfinputRadarTendencyRefused(
                        f"d{gid:02d} is integrated out of a host store; its "
                        "tiles carry no forcing, so mp_tend_radar = 1 would "
                        "be skipped while the record says forced")
                import cupy as cp

                with cp.cuda.Device(node.state.thp.device.id):
                    slots = [cp.asarray(slot) for slot in tendency.slots]
                forcing = forcing_class(
                    slots, tendency.slot_minutes,
                    provenance=provenance, mp_tend_lim=None, clock=clock)
                radar_tten.attach(node.state, forcing, node.cfg.run)
                self._attached.append((gid, node.state, forcing, None))
                self.roads[gid] = "resident"
            elif getattr(stepper, "ranked", False):
                self._attach_ranked(gid, node, stepper, tendency,
                                    forcing_class, clock, provenance)
            else:
                raise WrfinputRadarTendencyRefused(
                    f"d{gid:02d} is tile-streamed ([tiles]); a tile buffer "
                    "serves a different tile every sweep and carries no "
                    "forcing, so mp_tend_radar = 1 would be skipped or read "
                    "at the wrong columns.  Run it resident or on [devices] "
                    "ranks")

    def _attach_ranked(self, gid, node, stepper, tendency, forcing_class,
                       clock, provenance) -> None:
        import cupy as cp

        from gpuwm.da import radar_tten

        run = getattr(stepper, "tiled_run", None)
        if run is None or not getattr(run, "ranked", False):
            raise WrfinputRadarTendencyRefused(
                f"d{gid:02d}: the multi-card stepper exposes no resident "
                "ranks to attach the radar tendency to")
        cfg = node.cfg.run
        ny, nx = int(cfg.ny), int(cfg.nx)
        specs, devices, tiles = list(run.specs), list(run.devices), list(run.tiles)
        for spec in specs:
            if (spec.periodic_x or spec.periodic_y or spec.ci0 < 0 or spec.cj0 < 0
                    or spec.ci0 + spec.cnx > nx or spec.cj0 + spec.cny > ny):
                raise WrfinputRadarTendencyRefused(
                    f"d{gid:02d}: rank {spec.index} has a wrapped compute "
                    "window; a slab of a periodic axis is not a rectangle of "
                    "the domain and would be heated at the wrong columns")
        sub_cfgs = list(getattr(run, "sub_cfgs", [cfg] * len(tiles)))
        for rank, (tile, spec, dev) in enumerate(zip(tiles, specs, devices)):
            with cp.cuda.Device(dev):
                slots = [cp.asarray(np.ascontiguousarray(
                    slot[:, spec.cj0:spec.cj0 + spec.cny,
                         spec.ci0:spec.ci0 + spec.cnx]))
                    for slot in tendency.slots]
                forcing = forcing_class(
                    slots, tendency.slot_minutes, provenance=provenance,
                    mp_tend_lim=None, extent=(spec.cj0, spec.ci0, ny, nx),
                    clock=clock)
                setattr(tile, radar_tten.SLAB_ATTRIBUTE, (spec.cj0, spec.ci0))
                radar_tten.attach(tile, forcing, sub_cfgs[rank])
                cp.cuda.runtime.deviceSynchronize()
            self._attached.append((gid, tile, forcing, rank))
        self.roads[gid] = "ranks"

    def detach(self) -> None:
        from gpuwm.da import radar_tten

        for gid, state, forcing, rank in self._attached:
            self._receipts.append((gid, rank, forcing.receipt()))
            radar_tten.detach(state)
            if radar_tten.SLAB_ATTRIBUTE in vars(state):
                delattr(state, radar_tten.SLAB_ATTRIBUTE)
        self._attached = []

    def _rows(self):
        return self._receipts or [
            (gid, rank, forcing.receipt())
            for gid, _, forcing, rank in self._attached]

    def require_applied(self) -> None:
        """A forced domain whose forcing no microphysics step read is an
        error: the record would say forced while no step was."""
        for gid, rank, row in self._rows():
            calls = sum(row.get("calls_by_slot", ())) + int(
                row.get("calls_skipped_no_mp_heating", 0))
            if calls == 0:
                where = f"d{gid:02d}" + ("" if rank is None else f" rank {rank}")
                raise WrfinputRadarTendencyRefused(
                    f"mp_tend_radar = 1 was attached on {where} and no "
                    "microphysics step read it")

    def receipt(self) -> dict:
        domains = {}
        for gid, rank, row in self._rows():
            entry = domains.setdefault(f"d{gid:02d}", {
                "road": self.roads.get(gid),
                "input": self.tendencies[gid].receipt(),
                "forcing": [],
            })
            entry["forcing"].append({"rank": rank, **row})
        return {
            "schema": SCHEMA,
            "rule": ("mp_tend_radar = 1: the slot whose end time the clock "
                     "has not passed (the last slot after that) replaces the "
                     "clamped microphysics heating where its value is in "
                     "[-1, 1] K/s below the model top; NOAA-EMC/HRRR v4.1.21 "
                     "dyn_em/module_big_step_utilities_em.F:5901-6005"),
            "domains": domains,
        }
