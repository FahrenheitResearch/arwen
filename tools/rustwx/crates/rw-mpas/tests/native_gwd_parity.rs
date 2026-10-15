//! Word comparison against pristine MPAS-A v8.4.2, not a Rust golden answer.
//! See native_gwd/build_reference.py and driver.f90 for the strict oracle.
use rw_mpas::native_gwd::{RE, box_size, centre, diameter, reduce_box, smooth, wrap};

#[test]
#[ignore = "requires MPAS_GWD_ORACLE (build pristine source with the strict GNU flags)"]
fn native_gwd_matches_pristine_fortran_words() {
    let path =
        std::env::var("MPAS_GWD_ORACLE").expect("MPAS_GWD_ORACLE must name the Fortran output");
    let bytes = std::fs::read(path).unwrap();
    let mut words = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()));
    assert_eq!(words.next(), Some(100));
    assert_eq!(words.next(), Some(96));
    for k in 0..100 {
        let lat = (k as f32 - 50.0) * 0.0314159f32;
        let lon = (k as f32 - 50.0) * 0.0628318f32;
        let dx = 3700.0f32 + k as f32 * 5377.0f32;
        let (nx, ny) = box_size(lat, dx);
        let (ic, jc) = centre(lat, lon);
        let (ii, jj) = wrap(1 - nx / 2 + ic, 1 - ny / 2 + jc);
        for v in [nx, ny, ic, jc, ii, jj] {
            assert_eq!(v as i32 as u32, words.next().unwrap(), "geometry case {k}");
        }
    }
    let mut compared = 0;
    for k in 0..96 {
        let nx = 4 + (k * 13) % 134;
        let ny = 4 + (k * 17) % 92;
        assert_eq!(nx as u32, words.next().unwrap());
        assert_eq!(ny as u32, words.next().unwrap());
        let mut topo = Vec::new();
        let mut land = Vec::new();
        for j in 1..=ny {
            for i in 1..=nx {
                let v = match k % 12 {
                    0 => 200.0,
                    1 => ((i + j) % 2) as f32 * 0.25,
                    2 => ((i + j) % 2) as f32 * 2.0,
                    3 => 1116.2,
                    4 => ((i + j) * 10) as f32,
                    _ => ((i * 37 + j * 101 + k * 53) % 7000 - 1000) as f32,
                };
                topo.push(v);
                land.push(if (i * 3 + j * 5 + k) % 8 < k % 9 {
                    16
                } else {
                    1
                });
            }
        }
        let s = reduce_box(&topo, &land, nx as i64, ny as i64).unwrap();
        for (field, value) in [
            s.var2d, s.con, s.oa[0], s.oa[1], s.oa[2], s.oa[3], s.ol[0], s.ol[1], s.ol[2], s.ol[3],
        ]
        .iter()
        .enumerate()
        {
            let expected = words.next().unwrap();
            assert_eq!(
                value.to_bits(),
                expected,
                "box {k}, field {field}: Rust={value} native={}",
                f32::from_bits(expected)
            );
            compared += 1;
        }
        assert_eq!(
            s.land as u32,
            words.next().unwrap(),
            "dominant landmask {k}"
        );
    }
    let mut var = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
    assert_eq!(
        smooth(
            &mut var,
            &[0, 1, 0, 1, 1, 0, 1, 0],
            &[1, 1, 1, 1, 1, 1, 2, 2],
            &[
                1,
                usize::MAX,
                0,
                usize::MAX,
                3,
                usize::MAX,
                2,
                usize::MAX,
                4,
                usize::MAX,
                5,
                usize::MAX,
                4,
                5,
                4,
                5
            ],
            2
        )
        .unwrap(),
        4
    );
    for v in var {
        assert_eq!(v.to_bits(), words.next().unwrap(), "in-place smoothing");
    }
    for k in 0..100 {
        let dc = diameter(
            &[0.0001f32 + k as f32 * 0.0000123f32, 0.03, 0.004],
            RE,
            true,
            0.93,
        );
        assert_eq!(dc.to_bits(), words.next().unwrap(), "diameter {k}");
    }
    assert!(words.next().is_none());
    println!(
        "NATIVE_GWD_WORDS fields={compared} geometry=600 landmask=96 smoothing=8 diameter=100 mismatches=0"
    );
}

#[test]
fn native_gwd_validates_storage_and_dormant_metrics() {
    assert!(reduce_box(&[0.0; 8], &[1; 9], 3, 3).is_err());
    assert!(smooth(&mut [1.0], &[], &[1], &[usize::MAX], 1).is_err());
    assert!(rw_mpas::native_gwd::dormant_diagnostics(&[0.0; 9], &[0.0; 9], 3, 3).is_err());
}

#[test]
#[ignore = "requires MPAS_GWD_ORACLE (build pristine source with the strict GNU flags)"]
fn complete_native_module_diagnostics_match_fortran_words() {
    use rw_mpas::native_gwd::{SG_DELTA, dormant_diagnostics, pixel_dx};
    let path = std::env::var("MPAS_GWD_ORACLE").unwrap() + ".diagnostics";
    let bytes = std::fs::read(path).unwrap();
    let mut words = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()));
    assert_eq!(words.next(), Some(64));
    assert_eq!(words.next(), Some(SG_DELTA.to_bits()));
    let mut metrics = 0;
    for k in 0..64 {
        let nx = 4 + (k * 13) % 31;
        let ny = 4 + (k * 17) % 27;
        let jc = 1 + (k * 347) % 21600;
        assert_eq!(words.next(), Some(nx as u32));
        assert_eq!(words.next(), Some(ny as u32));
        let mut topo = Vec::new();
        let mut dxm = Vec::new();
        for j in 1..=ny {
            let (_, jj) = wrap(1, j as i64 - (ny / 2) as i64 + jc as i64);
            let dx = pixel_dx(jj);
            assert_eq!(words.next(), Some(dx.to_bits()), "metric case {k} row {j}");
            metrics += 1;
            for i in 1..=nx {
                topo.push(match k % 8 {
                    0 => 200.0,
                    1 => (i * 10) as f32,
                    2 => (j * 10) as f32,
                    3 => ((i + j) * 10) as f32,
                    4 => ((i as i32 - j as i32) * 10) as f32,
                    _ => ((i * 37 + j * 101 + k * 53) % 7000) as f32 - 1000.0,
                });
                dxm.push(dx);
            }
        }
        let d = dormant_diagnostics(&topo, &dxm, nx, ny).unwrap();
        for (field, v) in [d.elvmax, d.htheta, d.hgamma, d.hsigma].iter().enumerate() {
            assert_eq!(
                words.next(),
                Some(v.to_bits()),
                "diagnostic case {k} field {field} Rust={v}"
            );
        }
    }
    assert!(words.next().is_none());
    println!(
        "NATIVE_GWD_DIAGNOSTICS fields=256 metric_words={metrics} mismatches=0 disabled_in_native_output=true"
    );
}
