"""WOOF's fork Thompson (mp_physics = 28, thompson_version =
thompson_fork_snow_fall = "wrf_39_noaa") against the operational HRRR
fork's own Fortran, word for word, on the GPU.

tests/data/thompson_fork_column_oracle.npz holds the inputs and what the
unmodified NOAA-EMC/HRRR v4.1.21 module_mp_thompson.F returned for them
(tools/thompson_fork_oracle: build.sh, gpu_column_oracle.py,
make_column_fixture.py): 42 saved real-data columns at dt = 5 s and those
plus 42 synthetic regime and edge columns at dt = 20 s, at dt = 150 s (the
fork's dt > 120 s riming hand-off) and on the first step (no melting-snow
echo).  The adapter (gpuwm.core.microphysics_aerosol._apply_thompson_aerosol)
runs the column kernel gpuwm/core/kernels/thompson_fork.cu on the same
float32 words, and every output cell must carry the fork's bits: the eleven
species, theta, the 10 cm reflectivity, the three effective radii (the
fork's metres through WOOF's ``* 1.0e6f`` micron contract) and the seven
surface accumulations.

The fork's process tables come from the staged fork root (gpuwm fetch-tables
--thompson-fork, or GPUWM_THOMPSON_FORK_TABLE_ROOT); without them the test
skips naming that. These word proofs explicitly select GPUWM_WRF_EXACT=1
in a fresh process; the default adapter uses the legacy aerosol network.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wrf_exact_process import wrf_exact_process

cp = pytest.importorskip("cupy")
pytestmark = pytest.mark.gpu

FIXTURE = (Path(__file__).resolve().parent / "data"
           / "thompson_fork_column_oracle.npz")
SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc", "nwfa",
           "nifa")
FIELDS3 = SPECIES + ("th", "refl", "re_cloud", "re_ice", "re_snow")
FIELDS2 = ("rainnc", "rainncv", "snownc", "snowncv", "graupelnc",
           "graupelncv", "sr")


def _require_device_and_tables():
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:  # pragma: no cover
        pytest.skip("no CUDA runtime")
    from gpuwm.physics_compat import thompson_fork_table_root
    from gpuwm.core.thompson_contract import FORK_TABLE_ASSETS
    root = Path(thompson_fork_table_root())
    missing = [a.filename for a in FORK_TABLE_ASSETS
               if not (root / a.filename).is_file()]
    if missing:
        pytest.skip(f"fork Thompson tables not staged under {root}: "
                    f"{', '.join(missing)} (gpuwm fetch-tables "
                    "--thompson-fork)")


class _Columns:
    """The mp=28 DomainState surface over (nz, 1, ncol) device arrays."""

    def __init__(self, cols, ncol, itimestep, dt):
        def vol(a):
            a = np.ascontiguousarray(np.asarray(a[:ncol], np.float32).T)
            return cp.asarray(a.reshape(a.shape[0], 1, ncol))

        nz = cols["p"].shape[1]
        self.p = vol(cols["p"])
        self.thb = cp.zeros((nz,), cp.float32)
        self.thp = vol(cols["th"])
        self.phb = cp.zeros((nz + 1,), cp.float32)
        self.php = vol(cols["geop"])
        self.w = vol(cols["w"])
        for s in SPECIES:
            setattr(self, s, vol(cols[s]))
        for name in ("effc", "effi", "effs", "h_diabatic"):
            setattr(self, name, cp.zeros((nz, 1, ncol), cp.float32))
        self.nwfa2d = cp.asarray(np.asarray(cols["nwfa2d"][:ncol],
                                            np.float32).reshape(1, ncol))
        self.nifa2d = cp.asarray(np.asarray(cols["nifa2d"][:ncol],
                                            np.float32).reshape(1, ncol))
        self.physics = SimpleNamespace(refl_10cm=None, state=self)
        self.elapsed_seconds = float(itimestep - 1) * float(dt)
        self._scratch = {}

    def scratch(self, shape, slot, dtype=None):
        want = np.dtype(np.float32 if dtype is None else dtype)
        value = self._scratch.get(slot)
        if value is None:
            value = cp.zeros(tuple(shape), dtype=want)
            self._scratch[slot] = value
        return value

    def existing_scratch(self, slot):
        return self._scratch.get(slot)


def _cols(a):
    a = cp.asnumpy(a)
    return np.ascontiguousarray(a.reshape(a.shape[0], -1).T, np.float32)


def _bits(a):
    return np.ascontiguousarray(a, np.float32).view(np.uint32)


def _batches():
    with np.load(FIXTURE) as z:
        meta = json.loads(str(z["meta"]))
    return meta["batches"]


@pytest.mark.parametrize("batch", _batches(),
                         ids=lambda b: f"ncol{b['ncol']}-dt{b['dt']:g}-"
                                       f"step{b['itimestep']}")
@wrf_exact_process
def test_every_output_cell_is_the_forks_word(batch):
    _require_device_and_tables()
    from gpuwm.core.microphysics_aerosol import _apply_thompson_aerosol

    z = np.load(FIXTURE)
    cols = {k[3:]: z[k] for k in z.files if k.startswith("in_")}
    ncol, dt, step = batch["ncol"], batch["dt"], batch["itimestep"]
    state = _Columns(cols, ncol, step, dt)
    cfg = SimpleNamespace(mp_physics=28, no_mp_heating=0, mp_tend_lim=10.0,
                          thompson_version="wrf_39_noaa",
                          thompson_fork_snow_fall="wrf_39_noaa")
    diag = _apply_thompson_aerosol(state, cfg, dt, refl_10cm_due=True)
    cp.cuda.Device().synchronize()
    woof = {s: _cols(getattr(state, s)) for s in SPECIES}
    woof["th"] = _cols(state._scratch["mp_th"])
    woof["refl"] = _cols(state.physics.refl_10cm)
    woof["re_cloud"] = _cols(state.effc)
    woof["re_ice"] = _cols(state.effi)
    woof["re_snow"] = _cols(state.effs)
    for name in FIELDS2:
        woof[name] = np.ascontiguousarray(
            cp.asnumpy(getattr(diag, name)).ravel(), np.float32)

    for field, slot in (("pii", "mp_pii"), ("dz", "mp_dz8w")):
        np.testing.assert_array_equal(
            _bits(_cols(state._scratch[slot])), _bits(cols[field][:ncol]),
            err_msg=f"fork oracle {field} preparation changed")

    prefix = f"b{batch['index']}_fork_"
    differing = {}
    for name in FIELDS3 + FIELDS2:
        if name.startswith("re_"):
            fork = (z[prefix + name + "_m"] * np.float32(1.0e6)).astype(
                np.float32)
        else:
            fork = z[prefix + name]
        bad = int((_bits(woof[name]) != _bits(fork)).sum())
        if bad:
            differing[name] = bad
    assert not differing, (
        f"cells off the fork's words (ncol {ncol}, dt {dt:g}, ITIMESTEP "
        f"{step}): {differing}")


@wrf_exact_process
def test_the_blend_snow_fall_moves_only_melting_snow():
    """``thompson_fork_snow_fall = "blend"`` replaces the fork's live melting
    snow fall speed with WRF v4.6.1's rain-share blend and nothing else, so
    against the fork's words only the snow, its echo, its radius and the
    surface snow and rain sums may move, and only in columns that carry
    snow at or above 0 C."""
    _require_device_and_tables()
    from gpuwm.core.microphysics_aerosol import _apply_thompson_aerosol

    batch = next(b for b in _batches()
                 if b["dt"] == 20.0 and b["itimestep"] == 2)
    z = np.load(FIXTURE)
    cols = {k[3:]: z[k] for k in z.files if k.startswith("in_")}
    ncol = batch["ncol"]
    state = _Columns(cols, ncol, 2, batch["dt"])
    cfg = SimpleNamespace(mp_physics=28, no_mp_heating=0, mp_tend_lim=10.0,
                          thompson_version="wrf_39_noaa",
                          thompson_fork_snow_fall="blend")
    diag = _apply_thompson_aerosol(state, cfg, batch["dt"],
                                   refl_10cm_due=True)
    cp.cuda.Device().synchronize()
    woof = {s: _cols(getattr(state, s)) for s in SPECIES}
    woof["th"] = _cols(state._scratch["mp_th"])
    woof["refl"] = _cols(state.physics.refl_10cm)
    woof["re_snow"] = _cols(state.effs)
    for name in FIELDS2:
        woof[name] = np.ascontiguousarray(
            cp.asnumpy(getattr(diag, name)).ravel(), np.float32)
    prefix = f"b{batch['index']}_fork_"
    moved, columns = set(), set()
    for name in woof:
        fork = ((z[prefix + "re_snow_m"] * np.float32(1.0e6)).astype(
            np.float32) if name == "re_snow" else z[prefix + name])
        bad = _bits(woof[name]) != _bits(fork)
        if bad.any():
            moved.add(name)
            columns |= set(np.nonzero(bad.reshape(ncol, -1).any(1))[0])
    assert moved, "the blend changed nothing"
    assert moved <= {"qs", "re_snow", "refl", "rainnc", "rainncv",
                     "snownc", "snowncv", "sr"}, moved
    p = np.asarray(cols["p"][:ncol], np.float64)
    t = np.asarray(cols["th"][:ncol], np.float64) * (p / 1.0e5) ** (
        287.0 / 1004.5)
    warm_snow = ((np.asarray(cols["qs"][:ncol]) > 1.0e-12)
                 & (t > 273.0)).any(axis=1)
    assert all(warm_snow[c] for c in columns)



def test_the_column_unit_compiles_to_the_frame_it_is_priced_at():
    """The platform check no ``.cu`` enumeration reaches for this unit.

    The fork unit launches only through gpuwm/core/thompson_fork.py's own
    compile, so preflight prices it from its CHAINED_TRANSLATION_UNIT_FRAMES
    row and ``under_priced_kernel_frames`` cannot see it drift.  THE
    BREAKAGE THIS PREVENTS: with its column arrays in the local frame the
    unit read 25,136 B per thread at 50 levels, a 6.5 GiB reservation on a
    188-SM card that the fit gate never charged; the arrays now sit in the
    column arena, and a compiler that spills wider than the priced frame
    would bring an unpriced reservation back.
    """
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:  # pragma: no cover
        pytest.skip("no CUDA runtime")
    from cupy.cuda import driver
    from gpuwm.core import preflight as pf
    from gpuwm.core import thompson_fork as tf

    function = tf._kernel()
    local_size_bytes = 3  # CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
    frame = int(driver.funcGetAttribute(local_size_bytes, function.ptr))
    row = pf.CHAINED_TRANSLATION_UNIT_FRAMES["thompson_fork_unit"]
    profile = pf.local_memory_profile_from_device(cp)
    unpriced = (frame - row.max_local_size_bytes) * profile.resident_thread_capacity
    assert frame <= row.max_local_size_bytes, (
        f"thompson_fork_column compiles to {frame} B per thread on this "
        f"platform ({profile.name}, NVRTC {cp.cuda.nvrtc.getVersion()}) "
        f"against the {row.max_local_size_bytes} B gpuwm/core/preflight.py "
        "CHAINED_TRANSLATION_UNIT_FRAMES prices it at, so every fork run "
        f"under-charges the local-memory reservation by up to "
        f"{unpriced / 1024 ** 2:.1f} MiB.  Remedy: move the row to this "
        "reading and record it in CHAINED_UNITS_WITHOUT_A_PER_PLATFORM_ROW "
        "(gpuwm/core/kernel_frame_recordings.py)")


def test_the_arena_tile_never_exceeds_the_priced_tile():
    """The launcher's tile is what preflight prices the arena at, or less:
    a device whose occupancy allows more blocks per multiprocessor than
    TILE_BLOCKS_PER_SM still launches the priced tile."""
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:  # pragma: no cover
        pytest.skip("no CUDA runtime")
    from gpuwm.core import thompson_fork as tf

    sms = cp.cuda.Device().attributes["MultiProcessorCount"]
    for ncol in (1, 63, 64, 65, 10_000, 10_000_000):
        threads = tf._tile(tf._kernel(), ncol)
        assert threads % tf.BLOCK == 0
        assert threads <= tf.tile_threads(sms, ncol)
