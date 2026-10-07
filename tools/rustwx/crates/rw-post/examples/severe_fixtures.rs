//
// Group C against the staged black-box fixtures (SPEC 9.2): storm motion
// (graded USTM/VSTM; shear and SRH reported), best-180 CAPE/CIN (words 41
// and 42), the lowest-layer LCL height (first-band plane 5, centre) and the
// LCL edge records (centre height; centre pressure reported).  The fixtures
// are outputs of the old code on given inputs: a distance, not a definition.
//
//   severe_fixtures FIXTURE_DIR

use rw_post::severe::{parcel, thermo, wind};
use serde_json::json;

fn load(dir: &str, name: &str) -> Vec<f32> {
    let b = std::fs::read(format!("{dir}/{name}")).unwrap_or_else(|e| panic!("{name}: {e}"));
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn hypsometric_heights(p_int: &[f64], tk: &[f64], r: &[f64], zsfc: f64) -> Vec<f64> {
    let mut z = vec![zsfc];
    for k in 0..tk.len() {
        let tv = thermo::virtual_temperature(tk[k], r[k].max(0.0));
        let last = *z.last().unwrap();
        z.push(last + thermo::RD * tv / thermo::G * (p_int[k] / p_int[k + 1]).ln());
    }
    z
}

fn main() {
    let dir = std::env::args().nth(1).expect("usage: severe_fixtures FIXTURE_DIR");
    let mut report = serde_json::Map::new();

    // Storm motion: 12 cases, 12 levels; staggered winds are face pairs per level.
    let si = load(&dir, "storm-inputs.f32");
    let so = load(&dir, "storm-outputs.f32");
    let mut rows = Vec::new();
    for case in 0..12 {
        let a = &si[case * 64..(case + 1) * 64];
        let o = &so[case * 32..(case + 1) * 32];
        let hgt = a[0] as f64;
        let col = wind::WindColumn {
            z_agl: (0..12).map(|k| a[1 + k] as f64 - hgt).collect(),
            u: (0..12).map(|k| 0.5 * (a[13 + 2 * k] as f64 + a[14 + 2 * k] as f64)).collect(),
            v: (0..12).map(|k| 0.5 * (a[37 + 2 * k] as f64 + a[38 + 2 * k] as f64)).collect(),
            u10: a[62] as f64,
            v10: a[63] as f64,
        };
        let w = wind::winds(&col);
        rows.push(json!({
            "case": case,
            "latitude": a[61],
            "ustm": [w[0], o[24]], "vstm": [w[1], o[25]],
            "shear06_uv": [[w[6], w[7]], [o[26], o[27]]],
            "shear01_uv": [[w[4], w[5]], [o[28], o[29]]],
            "srh03": [w[3], o[30]], "srh01": [w[2], o[31]],
        }));
    }
    report.insert("storm_motion [new, old]".into(), json!(rows));

    // Best-parcel inputs: 12 cases of 3 x 3 x 56, level-major, bottom-up.
    let bi = load(&dir, "best-parcel-inputs.f32");
    let bo = load(&dir, "best-parcel-outputs.f32");
    let lo = load(&dir, "first-band-lcl-outputs.f32");
    let (nz, nc, centre) = (56usize, 9usize, 4usize);
    let mut rows = Vec::new();
    for case in 0..12 {
        let a = &bi[case * 1531..(case + 1) * 1531];
        let p_top = a[0] as f64;
        let ph = &a[1..1 + nz * nc];
        let tp = &a[1 + nz * nc..1 + 2 * nz * nc];
        let qv = &a[1 + 2 * nz * nc..1 + 3 * nz * nc];
        let psfc = a[1 + 3 * nz * nc + centre] as f64;
        let hgt = a[1 + 3 * nz * nc + nc + centre] as f64;
        let p: Vec<f64> = (0..nz).map(|k| ph[k * nc + centre] as f64).collect();
        let tk: Vec<f64> = (0..nz).map(|k| (tp[k * nc + centre] as f64 + 300.0) * (p[k] / thermo::P0).powf(thermo::KAPPA)).collect();
        let r: Vec<f64> = (0..nz).map(|k| qv[k * nc + centre] as f64).collect();
        let mut p_int = vec![psfc];
        for k in 1..nz {
            p_int.push(0.5 * (p[k - 1] + p[k]));
        }
        p_int.push(p_top);
        let z_int = hypsometric_heights(&p_int, &tk, &r, hgt);
        let col = parcel::Column { p: p.clone(), tk: tk.clone(), r: r.clone(), p_int, z_int, zsfc: hgt, psfc, t2: tk[0], q2: r[0], p2: psfc };
        let v = parcel::parcels(&col);
        let o = &bo[case * 46..(case + 1) * 46];
        let l = &lo[case * 45..(case + 1) * 45];
        rows.push(json!({
            "case": case,
            "cape_best180": [v[6], o[40]], "cin_best180": [v[7], o[41]],
            "lcl_height": [v[8], l[36 + centre]],
            "mucape": v[4], "mlcape": v[2],
        }));
    }
    report.insert("best_parcel [new, old]".into(), json!(rows));

    // LCL edge records.
    let ei = load(&dir, "lcl-edge-inputs.f32");
    let eo = load(&dir, "lcl-edge-outputs.f32");
    let mut rows = Vec::new();
    for rec in 0..18 {
        let a = &ei[rec * 118..(rec + 1) * 118];
        let o = &eo[rec * 18..(rec + 1) * 18];
        let (po, to, q, hgt) = (a[0] as f64, a[1] as f64, a[2] as f64, a[3] as f64);
        let r = q / (1.0 - q);
        let p_int: Vec<f64> = a[4..61].iter().map(|&x| x as f64).collect();
        let z_int: Vec<f64> = a[61..118].iter().map(|&x| x as f64).collect();
        let p: Vec<f64> = (0..56).map(|k| 0.5 * (p_int[k] + p_int[k + 1])).collect();
        let col = parcel::Column { tk: vec![to; 56], r: vec![r; 56], p, p_int, z_int, zsfc: hgt, psfc: po, t2: to, q2: r, p2: po };
        let h = parcel::lcl_height_agl(&col, po, to, r);
        let (_, pl) = parcel::lcl(po, to, r.max(thermo::R_FLOOR));
        rows.push(json!({"record": rec, "inputs": [po, to, q, hgt], "lcl_height": [h, o[13]], "lcl_pressure": [pl, o[4]]}));
    }
    report.insert("lcl_edge [new, old]".into(), json!(rows));
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}
