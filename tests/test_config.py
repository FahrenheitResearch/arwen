import textwrap

import pytest

from gpuwm.config import RunConfig, load_config

# Phase 2 fields and their defaults (plan Task 1); defaults must preserve
# Phase 1 behavior exactly.
PHASE2_DEFAULTS = {
    "hybrid_opt": 0,
    "etac": 0.2,
    "moist": False,
    "mp_physics": 0,
    "moist_adv_opt": 1,
    "diff_6th_opt": 0,
    "diff_6th_factor": 0.12,
    "km_opt": 1,
    "c_s": 0.25,
    "w_damping": 0,
    "open_x": False,
    "open_y": False,
    "terrain_opt": 0,
    "hill_height": 100.0,
    "hill_halfwidth": 10000.0,
}


def _write_toml(tmp_path, *, nz=64, time_step_sound=4, grid="", dynamics=""):
    """Minimal valid TOML; ``grid``/``dynamics`` append extra key lines
    (TOML tolerates the ragged indentation of interpolated lines)."""
    f = tmp_path / "run.toml"
    f.write_text(textwrap.dedent(f"""
        [grid]
        nx = 512
        ny = 1
        nz = {nz}
        dx = 100.0
        dy = 100.0
        ztop = 6400.0
        {grid}
        [dynamics]
        dt = 0.5
        time_step_sound = {time_step_sound}
        {dynamics}
        [run]
        run_seconds = 900.0
    """))
    return f

def test_load_config(tmp_path):
    f = tmp_path / "run.toml"
    f.write_text(textwrap.dedent("""
        [grid]
        nx = 512
        ny = 1
        nz = 64
        dx = 100.0
        dy = 100.0
        ztop = 6400.0
        [dynamics]
        dt = 0.5
        time_step_sound = 4
        khdif = 75.0
        kvdif = 75.0
        [run]
        run_seconds = 900.0
        case = "straka"
    """))
    cfg = load_config(f)
    assert cfg.nx == 512 and cfg.nz == 64
    assert cfg.dt == 0.5
    assert cfg.khdif == 75.0
    assert cfg.epssm == 0.1          # default applied
    assert cfg.case == "straka"

def test_rejects_missing_required(tmp_path):
    f = tmp_path / "bad.toml"
    f.write_text("[grid]\nnx = 10\n")
    with pytest.raises(TypeError):   # RunConfig(**merged) with missing required args
        load_config(f)

def test_rejects_odd_time_step_sound(tmp_path):
    with pytest.raises(ValueError, match="time_step_sound must be even"):
        load_config(_write_toml(tmp_path, time_step_sound=3))
    # smallest valid even value still loads
    cfg = load_config(_write_toml(tmp_path, time_step_sound=2))
    assert cfg.time_step_sound == 2

def test_rejects_too_few_vertical_levels(tmp_path):
    with pytest.raises(ValueError, match="nz must be >= 4"):
        load_config(_write_toml(tmp_path, nz=2))
    # smallest valid nz still loads
    cfg = load_config(_write_toml(tmp_path, nz=4))
    assert cfg.nz == 4


def test_phase2_fields_roundtrip(tmp_path):
    """TOMLs setting every Phase 2 field to a non-default value round-trip.

    Two files, because terrain + open lateral boundaries is a rejected
    combination (Task 12 final review: fail-loud guards): the open/moist
    file keeps flat terrain, the terrain file stays periodic.
    """
    f = tmp_path / "run.toml"
    f.write_text(textwrap.dedent("""
        [grid]
        nx = 512
        ny = 1
        nz = 64
        dx = 100.0
        dy = 100.0
        ztop = 6400.0
        [dynamics]
        dt = 0.5
        time_step_sound = 4
        hybrid_opt = 2
        etac = 0.15
        moist = true
        mp_physics = 1
        moist_adv_opt = 2
        diff_6th_opt = 2
        diff_6th_factor = 0.25
        km_opt = 4
        c_s = 0.18
        w_damping = 1
        open_x = true
        open_y = true
        bl_pbl_physics = 1
        # A PBL scheme without a surface layer is refused by
        # validate_run_config and by gpuwm/core/physics.py
        # initialize_physics alike: YSU consumes UST/HFX/QFX/WSPD/RMOL and
        # nothing writes them at sf_sfclay_physics=0.
        sf_sfclay_physics = 91
        [run]
        run_seconds = 900.0
    """))
    cfg = load_config(f)
    assert cfg.hybrid_opt == 2
    assert cfg.etac == 0.15
    assert cfg.moist is True
    assert cfg.mp_physics == 1
    assert cfg.moist_adv_opt == 2
    assert cfg.diff_6th_opt == 2
    assert cfg.diff_6th_factor == 0.25
    assert cfg.km_opt == 4
    assert cfg.c_s == 0.18
    assert cfg.w_damping == 1
    assert cfg.open_x is True
    assert cfg.open_y is True
    g = tmp_path / "terrain.toml"
    g.write_text(textwrap.dedent("""
        [grid]
        nx = 512
        ny = 1
        nz = 64
        dx = 100.0
        dy = 100.0
        ztop = 6400.0
        terrain_opt = 1
        hill_height = 500.0
        hill_halfwidth = 5000.0
        [dynamics]
        dt = 0.5
        time_step_sound = 4
        [run]
        run_seconds = 900.0
    """))
    tcfg = load_config(g)
    assert tcfg.terrain_opt == 1
    assert tcfg.hill_height == 500.0
    assert tcfg.hill_halfwidth == 5000.0


def test_phase2_defaults_preserve_phase1(tmp_path):
    """RunConfig built with only Phase 1 args carries the Phase 2 defaults,
    and a Phase 1 style TOML still loads with those defaults."""
    cfg = RunConfig(nx=512, ny=1, nz=64, dx=100.0, dy=100.0, ztop=6400.0,
                    dt=0.5, run_seconds=900.0)
    for name, default in PHASE2_DEFAULTS.items():
        assert getattr(cfg, name) == default, name
    loaded = load_config(_write_toml(tmp_path))
    for name, default in PHASE2_DEFAULTS.items():
        assert getattr(loaded, name) == default, name


# ---- fail-loud guards for silently-wrapping open-BC combinations ----------
# (Task 12 final review: these combinations would wrap stencils across the
# open boundaries or bypass the PD limiter; load_config and dycore.step both
# reject them.)

def test_rejects_open_with_terrain(tmp_path):
    with pytest.raises(NotImplementedError, match="terrain_opt"):
        load_config(_write_toml(tmp_path, grid="terrain_opt = 1",
                                dynamics="open_x = true"))
    with pytest.raises(NotImplementedError, match="terrain_opt"):
        load_config(_write_toml(tmp_path, grid="terrain_opt = 1",
                                dynamics="open_y = true"))
    # flat + open and terrain + periodic both stay legal
    assert load_config(_write_toml(tmp_path, dynamics="open_x = true")).open_x
    assert load_config(_write_toml(tmp_path,
                                   grid="terrain_opt = 1")).terrain_opt == 1


def test_rejects_open_with_constant_k(tmp_path):
    with pytest.raises(NotImplementedError, match="khdif"):
        load_config(_write_toml(tmp_path,
                                dynamics="open_x = true\nkhdif = 75.0"))
    with pytest.raises(NotImplementedError, match="khdif"):
        load_config(_write_toml(tmp_path,
                                dynamics="open_y = true\nkvdif = 75.0"))
    with pytest.raises(NotImplementedError, match="khdif"):
        load_config(_write_toml(tmp_path,
                                dynamics="specified = true\nkhdif = 75.0"))
    # constant K stays legal on periodic domains (the Straka setup)
    cfg = load_config(_write_toml(tmp_path, dynamics="khdif = 75.0"))
    assert cfg.khdif == 75.0


def test_km_opt_selects_exactly_one_diffusion_scheme(tmp_path):
    with pytest.raises(ValueError, match="km_opt=4.*khdif/kvdif"):
        load_config(_write_toml(
            tmp_path, dynamics="km_opt = 4\nkhdif = 75.0"))
    cfg = load_config(_write_toml(
        tmp_path, dynamics="km_opt = 1\nkhdif = 75.0\nkvdif = 25.0"))
    assert (cfg.km_opt, cfg.khdif, cfg.kvdif) == (1, 75.0, 25.0)


def test_km_opt4_requires_pbl_until_wrf_vertical_diffusion_is_wired(
        tmp_path):
    with pytest.raises(NotImplementedError,
                       match=r"km_opt=4.*bl_pbl_physics=0.*vertical"):
        load_config(_write_toml(tmp_path, dynamics="km_opt = 4"))
    # The surface layer is not decoration here: validate_run_config refuses a
    # PBL scheme without one (it consumes UST/HFX/QFX/WSPD/RMOL from it), and
    # gpuwm/core/physics.py initialize_physics has always refused the same
    # pair at driver construction.  91 is classic MM5, the production choice.
    cfg = load_config(_write_toml(
        tmp_path,
        dynamics="km_opt = 4\nbl_pbl_physics = 1\nsf_sfclay_physics = 91"))
    assert cfg.km_opt == 4 and cfg.bl_pbl_physics == 1


def test_rejects_nonmonotonic_diff6_with_moist(tmp_path):
    with pytest.raises(ValueError, match="diff_6th_opt"):
        load_config(_write_toml(tmp_path,
                                dynamics="moist = true\ndiff_6th_opt = 1"))
    # the monotonic option + moist is the WK82 production combination
    cfg = load_config(_write_toml(tmp_path,
                                  dynamics="moist = true\ndiff_6th_opt = 2"))
    assert cfg.diff_6th_opt == 2 and cfg.moist
    # non-monotonic + dry stays legal
    assert load_config(_write_toml(
        tmp_path, dynamics="diff_6th_opt = 1")).diff_6th_opt == 1


def test_step_guards_unsupported_combinations_cpu():
    """dycore.step's twin guards fire before any device work, so directly
    constructed RunConfigs (which bypass load_config) fail loudly too --
    exercised CPU-only with a stub state (numpy ht/qv, nothing launches)."""
    pytest.importorskip("cupy")        # dycore imports cupy at module scope
    import numpy as np

    from gpuwm.core.dycore import step

    class _Stub:
        def __init__(self, moist=False):
            self.ht = np.zeros((4, 8), np.float32)
            self.qv = np.zeros((6, 4, 8), np.float32) if moist else None

    base = dict(nx=8, ny=4, nz=6, dx=100.0, dy=100.0, ztop=1000.0,
                dt=0.5, run_seconds=0.0)
    with pytest.raises(NotImplementedError, match="terrain"):
        step(_Stub(), RunConfig(**base, open_x=True, terrain_opt=1))
    bumpy = _Stub()
    bumpy.ht[2, 3] = 25.0              # nonzero ht without terrain_opt
    with pytest.raises(NotImplementedError, match="terrain"):
        step(bumpy, RunConfig(**base, open_y=True))
    with pytest.raises(NotImplementedError, match="khdif"):
        step(_Stub(), RunConfig(**base, open_x=True, khdif=75.0))
    with pytest.raises(NotImplementedError, match="khdif"):
        step(_Stub(), RunConfig(**base, open_y=True, kvdif=75.0))
    with pytest.raises(NotImplementedError, match="khdif"):
        step(_Stub(), RunConfig(**base, specified=True, khdif=75.0))
    with pytest.raises(ValueError, match="km_opt=4.*khdif/kvdif"):
        step(_Stub(), RunConfig(**base, km_opt=4, khdif=75.0))
    with pytest.raises(NotImplementedError,
                       match=r"km_opt=4.*bl_pbl_physics=0.*vertical"):
        step(_Stub(), RunConfig(**base, km_opt=4))
    with pytest.raises(ValueError, match="diff_6th_opt"):
        step(_Stub(moist=True), RunConfig(**base, moist=True,
                                          diff_6th_opt=1))


def test_rejects_unknown_table(tmp_path):
    f = tmp_path / "run.toml"
    f.write_text(textwrap.dedent("""
        [grid]
        nx = 512
        ny = 1
        nz = 64
        dx = 100.0
        dy = 100.0
        ztop = 6400.0
        [dynamics]
        dt = 0.5
        [run]
        run_seconds = 900.0
        [physics]
        mp_physics = 1
    """))
    with pytest.raises(ValueError) as excinfo:
        load_config(f)
    msg = str(excinfo.value)
    assert str(f) in msg
    assert "physics" in msg


# ---- Phase 2 final-review carry-overs (Phase 3 Task 1: config hygiene) ----

def test_rejects_duplicate_key_across_tables(tmp_path):
    # T1-1: a key present in two known tables must fail loudly naming both
    # tables (merged.update would otherwise let the later table silently win).
    f = _write_toml(tmp_path, grid="dt = 0.25")   # [dynamics] also sets dt
    with pytest.raises(ValueError) as excinfo:
        load_config(f)
    msg = str(excinfo.value)
    assert str(f) in msg
    assert "dt" in msg and "grid" in msg and "dynamics" in msg

    # a second table pair, and identical values are still duplicates
    g = _write_toml(tmp_path, dynamics="run_seconds = 900.0")
    with pytest.raises(ValueError) as excinfo:
        load_config(g)
    msg = str(excinfo.value)
    assert "run_seconds" in msg and "dynamics" in msg and "run" in msg


def test_rejects_invalid_hybrid_opt(tmp_path):
    # hybrid_opt is validated at the config layer: 0/1 (B=eta) or 2 (WRF v4
    # cubic-B hybrid) only, instead of failing later in make_vertical_coord.
    for bad in (3, -1):
        with pytest.raises(ValueError, match="hybrid_opt"):
            load_config(_write_toml(tmp_path,
                                    dynamics=f"hybrid_opt = {bad}"))
    for ok in (0, 1, 2):
        cfg = load_config(_write_toml(tmp_path,
                                      dynamics=f"hybrid_opt = {ok}"))
        assert cfg.hybrid_opt == ok


def test_rejects_unknown_key(tmp_path):
    f = tmp_path / "run.toml"
    f.write_text(textwrap.dedent("""
        [grid]
        nx = 512
        ny = 1
        nz = 64
        dx = 100.0
        dy = 100.0
        ztop = 6400.0
        [dynamics]
        dt = 0.5
        not_a_real_option = 1
        [run]
        run_seconds = 900.0
    """))
    with pytest.raises(ValueError) as excinfo:
        load_config(f)
    msg = str(excinfo.value)
    assert str(f) in msg
    assert "not_a_real_option" in msg
