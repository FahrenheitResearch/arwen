//! wrf-core's `avo` against wrf-python's definition on a field whose
//! answer is known: solid-body rotation has relative vorticity 2 Omega
//! everywhere.  The vendored body had (v, u) swapped and returned zero
//! here, which erased every closed low from the absolute-vorticity maps
//! (2026-10-21 WOOF vs HRRR ABSV, 500 hPa corr 0.23).

use wrf_core::diag::vorticity::absolute_vorticity_staggered;

#[test]
fn solid_rotation_has_vorticity_two_omega_on_the_staggered_grid() {
    let (nx, ny, nz) = (7usize, 5usize, 2usize);
    let (dx, dy) = (3000.0, 3000.0);
    let omega = 1.0e-4;
    // Mass point (i, j) sits at x = i dx, y = j dy; u faces at x = (i - 1/2) dx,
    // v faces at y = (j - 1/2) dy.
    let mut u = Vec::new();
    let mut v = Vec::new();
    for _ in 0..nz {
        for j in 0..ny {
            for _i in 0..=nx {
                u.push(-omega * (j as f64 * dy));
            }
        }
        for _j in 0..=ny {
            for i in 0..nx {
                v.push(omega * (i as f64 * dx));
            }
        }
    }
    let msfu = vec![1.0; ny * (nx + 1)];
    let msfv = vec![1.0; (ny + 1) * nx];
    let msfm = vec![1.0; nx * ny];
    let cor = vec![1.0e-4; nx * ny];
    let avo = absolute_vorticity_staggered(&u, &v, &msfu, &msfv, &msfm, &cor, nx, ny, nz, dx, dy).unwrap();
    for (index, value) in avo.iter().enumerate() {
        assert!((value - (2.0 * omega + 1.0e-4)).abs() < 1e-12, "cell {index}: {value}");
    }
    // A cyclone and an anticyclone differ in sign: reversing the rotation
    // reverses the relative part.
    let u_anti: Vec<f64> = u.iter().map(|x| -x).collect();
    let v_anti: Vec<f64> = v.iter().map(|x| -x).collect();
    let anti = absolute_vorticity_staggered(&u_anti, &v_anti, &msfu, &msfv, &msfm, &cor, nx, ny, nz, dx, dy).unwrap();
    assert!(anti.iter().all(|value| (value - (-2.0 * omega + 1.0e-4)).abs() < 1e-12));
}

#[test]
fn map_factors_enter_as_wrf_python_applies_them() {
    // Uniform map factor m: winds divided by m, differences times m^2,
    // so the relative vorticity scales by m.
    let (nx, ny, nz) = (4usize, 4usize, 1usize);
    let omega = 2.0e-4;
    let m = 1.05;
    let mut u = Vec::new();
    let mut v = Vec::new();
    for j in 0..ny { for _ in 0..=nx { u.push(-omega * j as f64 * 1000.0); } }
    for _ in 0..=ny { for i in 0..nx { v.push(omega * i as f64 * 1000.0); } }
    let avo = absolute_vorticity_staggered(
        &u, &v, &vec![m; ny * (nx + 1)], &vec![m; (ny + 1) * nx], &vec![m; nx * ny],
        &vec![0.0; nx * ny], nx, ny, nz, 1000.0, 1000.0).unwrap();
    assert!(avo.iter().all(|value| (value - 2.0 * omega * m).abs() < 1e-12));
}
