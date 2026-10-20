//! Shared vertical diagnostics, with GRIB-facing analytic checks.
pub use rw_mlexport::diagnostics::{compute, Output};

#[cfg(test)]
mod tests {
    use super::*;
    use netcdf_writer::{NcFormat, NcType, NcWriter, Schema, VarData};

    /// Two mass levels with independent hand formulas. Nonzero terrain and
    /// rotation expose using MSL heights or leaving winds grid-relative.
    #[test]
    fn analytic_moist_column_and_height_winds() {
        let path = std::env::temp_dir().join(format!("grib2-column-{}.nc", std::process::id()));
        let mut s = Schema::new(NcFormat::Offset64);
        let t = s.def_dim("Time", 0, true).unwrap();
        let x = s.def_dim("west_east", 2, false).unwrap();
        let y = s.def_dim("south_north", 2, false).unwrap();
        let k = s.def_dim("bottom_top", 2, false).unwrap();
        let xs = s.def_dim("west_east_stag", 3, false).unwrap();
        let ys = s.def_dim("south_north_stag", 3, false).unwrap();
        let ks = s.def_dim("bottom_top_stag", 3, false).unwrap();
        let mut fields = Vec::new();
        for (name, dims, values) in [
            ("P", vec![t, k, y, x], vec![0.; 8]),
            (
                "PB",
                vec![t, k, y, x],
                [vec![90000.; 4], vec![60000.; 4]].concat(),
            ),
            (
                "T",
                vec![t, k, y, x],
                [
                    vec![(280. / 0.9_f64.powf(0.2857142857) - 300.) as f32; 4],
                    vec![(260. / 0.6_f64.powf(0.2857142857) - 300.) as f32; 4],
                ]
                .concat(),
            ),
            (
                "QVAPOR",
                vec![t, k, y, x],
                [vec![0.01; 4], vec![0.02; 4]].concat(),
            ),
            (
                "QCLOUD",
                vec![t, k, y, x],
                [vec![0.001; 4], vec![0.002; 4]].concat(),
            ),
            ("U", vec![t, k, y, xs], [vec![3.; 6], vec![9.; 6]].concat()),
            ("V", vec![t, k, ys, x], [vec![6.; 6], vec![12.; 6]].concat()),
            (
                "W",
                vec![t, ks, y, x],
                [vec![0.2; 4], vec![0.4; 4], vec![0.8; 4]].concat(),
            ),
            ("PH", vec![t, ks, y, x], vec![0.; 12]),
            (
                "PHB",
                vec![t, ks, y, x],
                [
                    vec![(10. * 9.80665) as f32; 4],
                    vec![(110. * 9.80665) as f32; 4],
                    vec![(310. * 9.80665) as f32; 4],
                ]
                .concat(),
            ),
            ("HGT", vec![t, y, x], vec![10.; 4]),
            ("PSFC", vec![t, y, x], vec![100000.; 4]),
            ("SINALPHA", vec![t, y, x], vec![0.6; 4]),
            ("COSALPHA", vec![t, y, x], vec![0.8; 4]),
        ] {
            let id = s.def_var(name, NcType::Float, &dims).unwrap();
            fields.push((id, values));
        }
        let mut writer = NcWriter::create(&path, s).unwrap();
        for (id, values) in fields {
            writer.write_record(0, id, VarData::F32(&values)).unwrap();
        }
        writer.finish().unwrap();
        let output = compute(&path, &[750], &[100], true).unwrap();
        let f = (75000_f64 / 90000.).ln() / (60000_f64 / 90000.).ln();
        let lerp = |a: f64, b: f64| a + f * (b - a);
        let q = lerp(0.01 / 1.011, 0.02 / 1.022);
        let omega = lerp(
            -90000. * 9.80665 * 0.3 / (287.058 * 280. * (1. + 0.61 * 0.01)),
            -60000. * 9.80665 * 0.6 / (287.058 * 260. * (1. + 0.61 * 0.02)),
        );
        for (plane, expected) in output.pressure.iter().zip([q, omega, lerp(0.3, 0.6)]) {
            for &actual in plane {
                assert!(
                    (f64::from(actual) - expected).abs() < 2e-6,
                    "{actual} vs {expected}"
                );
            }
        }
        // At 100 m, z=(50,200) gives 1/3: grid wind=(5,8), earth=(-0.8,9.4).
        for &u in &output.wind[0].1 {
            assert!((u + 0.8).abs() < 2e-6, "u={u}");
        }
        for &v in &output.wind[0].2 {
            assert!((v - 9.4).abs() < 2e-6, "v={v}");
        }
        std::fs::remove_file(path).unwrap();
    }
}
