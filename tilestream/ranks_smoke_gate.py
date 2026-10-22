"""Offline resident-rank smoke proof using real Rust emission decode/remapping."""
from __future__ import annotations
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np


class SyntheticFire:
    """A tiny RAVE-schema tape, with a burning cell on each side of a seam."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.path = None
        self.frames = []
        self.nx = None

    def boundary(self, state, other, cfg):
        import cupy as cp
        from gpuwm.chem_source_init import chem_boundary_fields
        # Nonzero initial and time-varying incoming smoke exercise rank-local
        # chemistry boundary windows as well as the interior source seam.
        for source, scale in ((state, .02), (other, .08)):
            source._external_scalar_boundary_fields = chem_boundary_fields(source.chem.table, cfg)
            for row in source.chem.transported:
                value = getattr(source, row.state_attr)
                value[...] = cp.arange(cfg.nz, dtype=cp.float32)[:, None, None] * np.float32(.001) + np.float32(scale)

    def store_bundle(self, state, cfg, boundaries, geo):
        """The forecast's disk-cache -> slab initializer -> pinned-store road."""
        import cupy as cp
        from types import SimpleNamespace
        from gpuwm.core.grid import make_base_state, make_vertical_coord
        from gpuwm.verify.cases.wk82 import wk82_sounding
        from gpuwm.ingest.prepared_cache import write_prepared_cache
        from gpuwm.ingest.prepared_store import store_from_prepared_cache
        from gpuwm.ingest.hrrr_physics import initialize_prepared_physics
        from gpuwm.state_serialization_contract import chem_state_attrs
        from gpuwm.core import streaming

        coord = make_vertical_coord(cfg.nz, hybrid_opt=cfg.hybrid_opt, etac=cfg.etac)
        base = make_base_state(coord, lambda z: wk82_sounding(z)[0],
            p_surf=cfg.p_surf, ztop=cfg.ztop, terrain_z=geo.terrain)
        plane = np.ones((cfg.ny, cfg.nx), np.float32)
        static = dict(HGT_M=geo.terrain, MAPFAC_M=geo.msft,
            MAPFAC_U=geo.msfu, MAPFAC_V=geo.msfv, F=geo.f, E=geo.e,
            SINALPHA=geo.sina, COSALPHA=geo.cosa,
            LU_INDEX=np.full(plane.shape, 7, np.int32),
            SCT_DOM=np.full(plane.shape, 6, np.int32), LANDMASK=plane,
            GREENFRAC=np.full((12, *plane.shape), .5),
            ALBEDO12M=np.full((12, *plane.shape), 20.),
            LAI12M=np.full((12, *plane.shape), 2.), SNOALB=60.*plane)
        met = SimpleNamespace(fields=dict(LANDSEA=plane, SKINTEMP=286.*plane,
            T2=286.*plane, U10=np.full((cfg.ny, cfg.nx+1), 25., np.float32),
            V10=np.zeros((cfg.ny+1, cfg.nx), np.float32)))
        surface = dict(TSK=286.*plane,
            TSLB=np.full((cfg.num_soil_layers, *plane.shape), 285., np.float32),
            SMOIS=np.full((cfg.num_soil_layers, *plane.shape), .25, np.float32),
            SH2O=np.full((cfg.num_soil_layers, *plane.shape), .25, np.float32),
            TMN=284.*plane, SEAICE=0.*plane, XLAND=plane, LANDMASK=plane,
            SNOW=0.*plane, SNOWH=0.*plane)
        initial = SimpleNamespace(state=state, coord=coord, base=base,
            surface_pressure=cp.asnumpy(state.physics.fields['psfc']),
            surface_qv=cp.asnumpy(state.qv[0]))
        identity = {'purpose': 'ranked-smoke-prepared-store-gate'}
        path = self.directory / 'prepared-cache'
        write_prepared_cache(path, identity=identity, initial_result=initial,
            met=met, surface=surface, boundaries=boundaries)
        attrs = dict(MMINLU='MODIFIED_IGBP_MODIS_NOAH', ISWATER=17, ISLAKE=21, ISICE=15)
        valid = state.chem.start_time
        # Both sides initialize from the same source-neutral prepared inputs.
        initialize_prepared_physics(initial, cfg, met, SimpleNamespace(fields=surface),
            static, attrs, geo.grid, valid)
        streaming.prime_lazy_carriers(state, cfg)
        bundle = store_from_prepared_cache(path, expected_identity=identity,
            cfg=cfg, static=static, landuse_attrs=attrs, grid=geo.grid,
            valid_time=valid, rows_per_slab=24, log=print)
        expected = streaming.streamed_store_inventory()(state)
        assert set(bundle.store) == set(expected)
        for name, value in expected.items():
            np.testing.assert_array_equal(bundle.store[name], cp.asnumpy(value), err_msg=name)
        names = chem_state_attrs(state)
        for name in names:
            assert 'state/' + name in bundle.store, name
        assert bundle.store['state/chemdiag_fire_ebu'].shape == (1, cfg.nz, cfg.ny, cfg.nx)
        self.prepared_store = {**dict(bundle.receipt), 'chem_carriers': list(names),
            'initial_carriers_word_identical': True}
        return bundle

    def verify(self, state, streamed):
        import cupy as cp
        from gpuwm.core.chem_driver import ledger_report
        from gpuwm.core.radiation_composition import transported_smoke_attr
        report = ledger_report(state)
        for row in report["rows"].values():
            assert row["emitted_kg"] > 0, report
            assert row["deposited_kg"] < 0, report
            assert row["scavenged_kg"] < 0, report
        emitted = cp.asnumpy(state.chemdiag_fire_emitted_mass)
        seam = emitted[..., self.nx//2-1:self.nx//2+1]
        assert np.all(np.sum(seam, axis=tuple(range(seam.ndim-1))) > 0), seam
        assert float(cp.max(state.chemdiag_plume_top)) > 0
        attr = transported_smoke_attr(streamed.tiled_run.cfg)
        assert attr is not None
        for tile in streamed.tiled_run.tiles:
            assert getattr(tile, attr) is not None
            adapter = tile.physics.radiation_callable
            adapter = getattr(adapter, 'shortwave_adapter', adapter)
            assert adapter._transported_smoke == attr
        # Write and reopen the actual default history tape, including row
        # diagnostics, integer caches and the domain-owned float64 vectors.
        import json
        from netCDF4 import Dataset
        from gpuwm.io.wrfout import WrfoutWriter, streamed_carrier_provenance_attrs
        attrs = streamed_carrier_provenance_attrs(streamed)
        ranked_ledger = json.loads(attrs["GPUWM_CHEM_LEDGER"])
        for name, entry in report["rows"].items():
            for bucket, value in entry.items():
                np.testing.assert_allclose(ranked_ledger["rows"][name][bucket], value, rtol=2e-12, atol=1e-7)
        fields = streamed.history_fields().materialize()
        cfg = streamed.tiled_run.cfg
        path = self.directory / "smoke-history.nc"
        with WrfoutWriter(path, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, dx=cfg.dx, dy=cfg.dy,
                global_attrs=attrs, field_schema=fields, soil_layers=cfg.num_soil_layers) as writer:
            writer.write_frame("2026-01-01_00:00:20", fields)
        with Dataset(path) as tape:
            assert tape.getncattr("GPUWM_CHEM_LEDGER") == attrs["GPUWM_CHEM_LEDGER"]
            for name, value in fields.items():
                disk = np.asarray(tape[name][0])
                if disk.ndim and disk.ndim == value.ndim and disk.shape[0] == value.shape[0] + 1:
                    assert np.all(disk[-1] == 0), name
                    disk = disk[:-1]
                np.testing.assert_array_equal(disk.reshape(value.shape), value, err_msg=name)
        return {"ledger": report, "seam_emission_ug_m2": seam.sum(axis=tuple(range(seam.ndim-1))).tolist(),
                "plume_top_m": float(cp.max(state.chemdiag_plume_top)),
                "transported_smoke_attr": attr, "history_disk_roundtrip": True}

    def __call__(self, state, cfg):
        import cupy as cp
        from functools import partial
        from unittest.mock import patch
        from gpuwm.chem_emission_frames import EmissionFrames, frames_for_context
        from gpuwm.core.chem_driver import _context
        lat = cp.asnumpy(state.physics.fields["xlat"])
        lon = cp.asnumpy(state.physics.fields["xlong"])
        start = state.chem.start_time
        if self.path is None:
            self.nx = cfg.nx
            from gpuwm.io.nc_writer_bridge import ClassicSchema
            self.hour = (start - timedelta(hours=3)).replace(minute=0, second=0, microsecond=0)
            schema = ClassicSchema()
            # RAVE is a regular geographic source grid, distinct from the
            # curvilinear model grid. Decode and remap it as the live reader does.
            slat, slon = np.meshgrid(
                np.arange(float(lat.min())-.03, float(lat.max())+.06, .03),
                np.arange(float(lon.min())-.03, float(lon.max())+.06, .03), indexing="ij")
            dims = [schema.def_dim("y", slat.shape[0]), schema.def_dim("x", slat.shape[1])]
            mass = np.zeros(slat.shape, np.float64)
            frp = np.zeros_like(mass)
            for j, i, kg, mw in ((cfg.ny//2, cfg.nx//2-1, 3600., 400.),
                                  (cfg.ny//2, cfg.nx//2, 1800., 150.),
                                  (cfg.ny//3, cfg.nx//3, 900., 20.)):
                cell = np.unravel_index(np.argmin((slat-lat[j,i])**2 +
                                                  (slon-lon[j,i])**2), slat.shape)
                mass[cell] += kg
                frp[cell] += mw
            values = {}
            for name, array, units in (("grid_latt", slat, "degrees_north"),
                    ("grid_lont", slon, "degrees_east"), ("PM25", mass, "kg"),
                    ("FRP_MEAN", frp, "MW"), ("QA", np.full(slat.shape, 3.), "1")):
                var = schema.def_var(name, "f8", dims)
                schema.put_var_attr(var, "units", units)
                values[var] = array
            tdim = schema.def_dim("time", 1)
            time = schema.def_var("time", "f8", [tdim])
            schema.put_var_attr(time, "units", "seconds since " + self.hour.strftime("%Y-%m-%d %H:%M:%S"))
            values[time] = np.array([0.])
            self.path = self.directory / "synthetic-fire.nc"
            with schema.create(self.path) as tape:
                for var, array in values.items():
                    tape.write_var(var, np.asarray(array, np.float64))
        factory = partial(EmissionFrames,
            fetch=lambda row, hour, cache: (self.path, {}),
            listing=lambda row, month, cache: ("synthetic", {self.hour: self.path.name}),
            prefetch=False)
        # Only the network endpoints are replaced. The production lazy factory
        # must discover each rank's own geometry from its chemistry context.
        with patch('gpuwm.chem_emission_frames.EmissionFrames', factory):
            frames = frames_for_context(_context(state, cfg, 1))
        np.testing.assert_array_equal(frames.latitude, lat)
        np.testing.assert_array_equal(frames.longitude, lon)
        np.testing.assert_array_equal(frames.map_factors, cp.asnumpy(state.msft))
        self.frames.append(frames)


def run(*, devices=None, grid=(1, 2), mode="threads", nsteps=20, directory=None):
    from tilestream.ranks_gate import config, integrate
    cfg = replace(config(nx=96, ny=80, nz=20, rung="mynn"),
        mp_physics=28, chem_sets="smoke", chem_sources="rave-3km,rap-native-smoke,hrrr-native-smoke", chem_adv_opt=2,
        chem_mix6_off=True, wetscav_onoff=-1, vertmix_onoff=1,
        aer_opt=3, simple_dir_fdb=1, biomass_burn_opt=4, enh_vermix=True,
        flam_part=.75, aer_drydep_opt=111)
    with TemporaryDirectory(prefix="rank-smoke-") as temp:
        out = Path(directory or temp)
        out.mkdir(parents=True, exist_ok=True)
        fire = SyntheticFire(out)
        result = integrate(cfg, grid=grid, mode=mode, nsteps=nsteps, devices=devices,
            change_live=False, prepare=fire, restart_path=out / "ranked-mid.npz",
            ledger_tolerance=2e-12, wind=25.)
        result["source_receipts"] = [list(frame.receipts.values()) for frame in fire.frames]
        result['prepared_store'] = fire.prepared_store
        return result


if __name__ == "__main__":
    run()
