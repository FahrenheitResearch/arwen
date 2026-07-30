"""Host algebra gates for WRF v4.6.1 ``diff_opt=2, km_opt=4``.

These are deliberately independent of the CUDA mirror tests in
``test_smag2d.py``.  The expected numbers below are direct reductions of
``dyn_em/module_diffusion_em.F`` rather than a second implementation of the
legacy coordinate-surface, fieldwise Laplacian.
"""

from pathlib import Path

import numpy as np


def test_slope_corrected_deformation_and_k_are_not_flat_reduction():
    """WRF cal_deform_and_div:176-215 + smag2d_km:2004-2038.

    Unit grid/map factors, horizontally uniform u, du/dz=1, zx=.5 and
    v=zy=0 give D11=-1.  With c_s=.25 the resulting K is .03125.  The old
    same-level reduction sees no horizontal difference and returns zero.
    """
    from gpuwm.core.smag2d import wrf_d11_algebra, wrf_smag2d_km_algebra

    d11 = wrf_d11_algebra(du_dx_hat=0.0, du_deta_hat=1.0,
                          zx=0.5, rdzw=1.0)
    assert d11 == -1.0
    d22 = 0.0
    d12_four_corner_mean = 0.0
    got = wrf_smag2d_km_algebra(
        d11=d11, d22=d22, d12=d12_four_corner_mean,
        dx=1.0, dy=1.0, msftx=1.0, msfty=1.0, c_s=0.25,
        slope_alpha=1.0,
    )
    assert got == 0.03125
    assert got != 0.0


def test_flat_cross_component_u_stress_is_nonzero():
    """WRF horizontal_diffusion_u_2:3233-3313 cross-stress gate.

    On a flat unit grid, u=0 and v=sin(x)sin(y) produce tau12 from dv/dx.
    Therefore d(tau12)/dy drives u even though a fieldwise Laplacian of u
    is identically zero.
    """
    from gpuwm.core.smag2d import wrf_flat_u_cross_stress_tendency

    nx = ny = 32
    x = (np.arange(nx) + 0.5) * (2.0 * np.pi / nx)
    y_face = np.arange(ny + 1) * (2.0 * np.pi / ny)
    v = np.sin(y_face[:, None]) * np.sin(x[None, :])
    # Positive nonuniform K keeps this a generic variable-stress example.
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    km = 1.0 + 0.1 * np.cos(2.0 * np.pi * xx / nx)
    rho = np.ones((ny, nx))

    got = wrf_flat_u_cross_stress_tendency(
        v=v, km=km, rho=rho,
        dx=2.0 * np.pi / nx, dy=2.0 * np.pi / ny,
    )
    legacy_fieldwise_u_laplacian = np.zeros_like(got)
    assert np.max(np.abs(got)) > 1.0e-3
    assert not np.allclose(got, legacy_fieldwise_u_laplacian)


def test_production_smag_source_carries_metric_and_stress_inputs():
    """Guard against routing production back through the legacy kernels."""
    source = (Path(__file__).parents[1] / "gpuwm" / "core" / "dycore.py").read_text()
    assert "launch_wrf_smag2d_km" in source
    assert "launch_wrf_smag2d_hd" in source
    cuda = (Path(__file__).parents[1] / "gpuwm" / "core" / "kernels" /
            "smag2d.cu").read_text()
    for token in ("wrf_smag2d_km", "wrf_smag_hd_u", "wrf_smag_hd_v",
                  "wrf_smag_hd_w", "wrf_smag_hd_s", "wrf_defor12"):
        assert token in cuda


def test_theta_row_reconstructs_wrf_t2_and_boundary_k_stays_zero():
    """Source gates for the two active-real74 M16 review corrections."""
    repo = Path(__file__).parents[1]
    cuda = (repo / "gpuwm" / "core" / "kernels" / "smag2d.cu").read_text()
    assert "__fadd_rn(s.thb[h], value), WRF_T0" in cuda
    assert "static constexpr real WRF_T0 = 300.0f" in cuda
    boundary = cuda[cuda.index("void wrf_smag_km_bc"):
                    cuda.index("real wrf_tau11")]
    assert "xkmh[IDX3(k, j, i)] = 0.0f" in boundary
    assert "xkhh[IDX3(k, j, i)] = 0.0f" in boundary
    assert "xkmh[IDX3(k, sj, si)]" not in boundary

    dycore = (repo / "gpuwm" / "core" / "dycore.py").read_text()
    assert 'full_theta=(slot == "smag_rth")' in dycore
    assert "state.thb.ndim == 3" in dycore


def test_wrf_smag_density_uses_vapor_loading_with_exact_dry_branch():
    """Pin WRF's ``rho=(1+qv)/alt`` and the no-qv dry dispatch."""
    repo = Path(__file__).parents[1]
    cuda = (repo / "gpuwm" / "core" / "kernels" / "smag2d.cu").read_text()
    rho_start = cuda.index("real wrf_rho")
    rho_end = cuda.index("real wrf_uhat")
    rho_source = cuda[rho_start:rho_end]
    assert "if (q.moist) return (1.0f + q.qv[h]) / q.alt[h]" in rho_source
    assert "return 1.0f / q.alt[h]" in rho_source

    dycore = (repo / "gpuwm" / "core" / "dycore.py").read_text()
    assert 'qv = getattr(state, "qv" + suffix) if moist else state.alt' in dycore
    assert "np.int32(moist)" in dycore


def test_full_theta_diffusion_is_not_perturbation_only_on_sloping_levels():
    """WRF t_2 carries the base-theta gradient through metric correction."""
    from gpuwm.core.constants import G
    from gpuwm.core.smag2d import wrf_periodic_scalar_metric_tendency

    nz, ny, nx = 4, 6, 8
    thp = np.zeros((nz, ny, nx), dtype=np.float64)
    thb = np.array([300.0, 301.0, 304.0, 309.0])[:, None, None]
    theta_minus_t0 = thp + thb - 300.0
    kh = np.ones_like(thp) * 30.0
    rho = np.ones_like(thp)
    x = np.arange(nx, dtype=np.float64)
    y = np.arange(ny, dtype=np.float64)
    terrain = (80.0 * np.sin(2.0 * np.pi * x / nx)[None, :]
               + 50.0 * np.cos(2.0 * np.pi * y / ny)[:, None])
    sigma_w = 1.0 - np.arange(nz + 1, dtype=np.float64) / nz
    height = (1000.0 * np.arange(nz + 1)[:, None, None]
              + sigma_w[:, None, None] * terrain[None, :, :])
    phi = G * height
    dnw = np.full(nz, -1.0 / nz)
    dn = np.r_[0.0, np.full(nz - 1, -1.0 / nz)]
    fnm = np.r_[0.0, np.full(nz - 1, 0.5)]
    fnp = fnm.copy()
    args = dict(kh=kh, rho=rho, phi=phi, dnw=dnw, dn=dn,
                fnm=fnm, fnp=fnp, cf1=1.0, cf2=0.0, cf3=0.0,
                dx=1000.0, dy=1200.0)
    full = wrf_periodic_scalar_metric_tendency(
        field=np.broadcast_to(theta_minus_t0, thp.shape), **args)
    perturbation = wrf_periodic_scalar_metric_tendency(field=thp, **args)
    np.testing.assert_array_equal(perturbation, 0.0)
    assert np.max(np.abs(full)) > 0.0


def test_step_refreshes_density_before_first_smag_evaluation():
    """A fresh dry state leaves ``alt`` zero until the first EOS refresh."""
    source = (Path(__file__).parents[1] / "gpuwm" / "core" / "dycore.py").read_text()
    step_source = source[source.index("def step("):]
    refresh = "update_diagnostics(state, cfg.hypsometric_opt)\n    prepare_fixed_tendencies"
    assert refresh in step_source


def test_smag_memory_manifest_includes_shared_face_workspaces():
    """The WRF stress/scalar-flux route needs both horizontal face buffers."""
    from gpuwm.config import RunConfig
    from gpuwm.core.preflight import scratch_slot_registry

    cfg = RunConfig(nx=8, ny=6, nz=4, dx=1000.0, dy=1000.0,
                    ztop=10000.0, dt=1.0, run_seconds=10.0, km_opt=4)
    slots = scratch_slot_registry(cfg)
    assert slots["diff6_x"] == (4, 6, 9)
    assert slots["diff6_y"] == (4, 7, 8)
    assert "diff6_z" not in slots and "diff6_m" not in slots


def test_flat_periodic_wrf_scalar_operator_telescopes():
    """With zx=zy=0 the WRF scalar source has zero global mass budget."""
    from gpuwm.core.constants import G
    from gpuwm.core.smag2d import wrf_periodic_scalar_metric_tendency

    nz, ny, nx = 4, 7, 9
    rng = np.random.default_rng(812)
    field = rng.uniform(0.0, 0.02, (nz, ny, nx))
    kh = rng.uniform(1.0, 40.0, field.shape)
    rho = rng.uniform(0.8, 1.2, field.shape)
    phi = (G * np.arange(nz + 1, dtype=np.float64)[:, None, None]
           * np.ones((1, ny, nx)))
    dnw = np.full(nz, -1.0 / nz)
    dn = np.r_[0.0, np.full(nz - 1, -1.0 / nz)]
    fnm = np.r_[0.0, np.full(nz - 1, 0.5)]
    fnp = fnm.copy()
    total, horizontal, metric = wrf_periodic_scalar_metric_tendency(
        field=field, kh=kh, rho=rho, phi=phi, dnw=dnw, dn=dn,
        fnm=fnm, fnp=fnp, cf1=1.0, cf2=0.0, cf3=0.0,
        dx=1000.0, dy=1200.0, return_components=True,
    )
    np.testing.assert_array_equal(metric, 0.0)
    budget = np.sum(-dnw[:, None, None] * total, dtype=np.float64)
    scale = np.sum(np.abs(dnw[:, None, None] * horizontal), dtype=np.float64)
    assert abs(budget) <= 1.0e-14 * scale


def test_flat_periodic_scalar_operator_carries_face_and_mass_maps():
    """A uniform map factor enters WRF's flux and divergence once each."""
    from gpuwm.core.constants import G
    from gpuwm.core.smag2d import wrf_periodic_scalar_metric_tendency

    nz, ny, nx = 4, 5, 7
    rng = np.random.default_rng(713)
    field = rng.normal(size=(nz, ny, nx))
    kh = rng.uniform(2.0, 15.0, field.shape)
    rho = rng.uniform(0.7, 1.3, field.shape)
    phi = (G * np.arange(nz + 1, dtype=np.float64)[:, None, None]
           * np.ones((1, ny, nx)))
    dnw = np.full(nz, -1.0 / nz)
    dn = np.r_[0.0, np.full(nz - 1, -1.0 / nz)]
    fnm = np.r_[0.0, np.full(nz - 1, 0.5)]
    fnp = fnm.copy()
    args = dict(field=field, kh=kh, rho=rho, phi=phi, dnw=dnw, dn=dn,
                fnm=fnm, fnp=fnp, cf1=1.0, cf2=0.0, cf3=0.0,
                dx=900.0, dy=1100.0)
    identity = wrf_periodic_scalar_metric_tendency(**args)
    map_factor = 1.37
    mapped = wrf_periodic_scalar_metric_tendency(
        **args, msft=np.full((ny, nx), map_factor),
        msfu=np.full((ny, nx + 1), map_factor),
        msfv=np.full((ny + 1, nx), map_factor))
    np.testing.assert_allclose(mapped, map_factor ** 2 * identity,
                               rtol=2.0e-15, atol=2.0e-15)
