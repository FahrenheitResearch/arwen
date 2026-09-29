#[path = "../tests/lane1_goldens.rs"]
mod authority;
use authority::{Goldens, lambert_chain, twin_state_map, adopted_state_map};
use std::{path::PathBuf, collections::BTreeMap};
use static_fields::projection::{ProjectedGrid, wps32::{twin_for, sampling_surface}};

fn main() {
    use serde_json::json;
    use sha2::{Digest, Sha256};
    let output = PathBuf::from(std::env::var_os("GPUWM_STATIC_WPS32_OUTPUT")
        .expect("set GPUWM_STATIC_WPS32_OUTPUT to a separate output directory"));
    assert!(!output.exists(), "re-pin output must be a new directory");
    std::fs::create_dir_all(&output).unwrap();
    let g = Goldens::load_reference();
    let (_, _, d03) = lambert_chain(&g);
    let parent = ProjectedGrid::new(g.spec("lam_parent")).unwrap();
    let translated = parent.translated(3, -2, None, None).unwrap();
    let cases = vec![
        ("lam_parent", "lambert", parent), ("lam_d03", "lambert", d03),
        ("lam_sh", "lambert", ProjectedGrid::new(g.spec("lam_sh")).unwrap()),
        ("merc", "mercator", ProjectedGrid::new(g.spec("merc")).unwrap()),
        ("merc_subkm", "mercator", ProjectedGrid::new(g.spec("merc_subkm")).unwrap()),
        ("polar", "polar", ProjectedGrid::new(g.spec("polar")).unwrap()),
        ("polar_sh", "polar", ProjectedGrid::new(g.spec("polar_sh")).unwrap()),
        ("polar_subkm", "polar", ProjectedGrid::new(g.spec("polar_subkm")).unwrap()),
        ("translated", "lambert", translated),
    ];
    let mut manifest = json!({"arithmetic_backend": "portable-libm-0.2.16",
        "reason": "Identical WPS sampling on preparation and run machines",
        "cases": {}});
    for (case, kind, grid) in cases {
        let mut row = json!({"arrays": {}});
        for (key, state) in [("twin_state", twin_state_map(kind, &grid)),
                            ("twin_state_adopted", adopted_state_map(kind, &grid))] {
            row[key] = serde_json::to_value(state.into_iter().map(|(k, v)|
                (k, format!("{:08x}", v.to_bits()))).collect::<BTreeMap<_, _>>()).unwrap();
        }
        let mut write = |key: &str, bytes: Vec<u8>| {
            let entry = g.entry(case, key);
            let file = format!("{case}.{key}.{}", entry["dtype"].as_str().unwrap());
            std::fs::write(output.join(&file), &bytes).unwrap();
            row["arrays"][key] = json!({"dtype": entry["dtype"], "shape": entry["shape"],
                "file": file, "sha256": format!("{:x}", Sha256::digest(&bytes))});
        };
        let mut twin = twin_for(&grid).unwrap();
        if grid.spec.dx < 1000.0 { twin.adopt_public_pole(&grid); }
        let mut lat = Vec::new(); let mut lon = Vec::new();
        let mut xs = Vec::new(); let mut ys = Vec::new();
        for j in 0..grid.spec.e_sn + 5 {
            for i in 0..grid.spec.e_we + 5 {
                let (la, lo) = twin.ij_to_latlon32(i as f32 - 2.0, j as f32 - 2.0);
                let (x, y) = twin.latlon_to_ij32(la, lo);
                lat.push(la); lon.push(lo); xs.push(x); ys.push(y);
            }
        }
        for (key, values) in [("twin_lat", lat), ("twin_lon", lon),
                              ("twin_llij_x", xs), ("twin_llij_y", ys)] {
            write(key, values.iter().flat_map(|v| v.to_le_bytes()).collect());
        }
        if !g.case(case)["arrays"]["surface_lat_e"].is_null() {
            let s = sampling_surface(&grid, 3).unwrap();
            for (key, values) in [("surface_lat_e", &s.lat_e),
                ("surface_lat_lower_e", &s.lat_lower_e), ("surface_lat_c", &s.lat_c)] {
                write(key, values.iter().flat_map(|v| v.to_le_bytes()).collect());
            }
            if s.lon_e_is_f32 {
                write("surface_lon_e", s.lon_e.iter().flat_map(|v| (*v as f32).to_le_bytes()).collect());
            }
            if !s.lon_e_is_f32 {
                write("surface_lon_e", s.lon_e.iter().flat_map(|v| v.to_le_bytes()).collect());
            }
            write("surface_lon_c", s.lon_c.iter().flat_map(|v| v.to_le_bytes()).collect());
            write("surface_lon_boundary_band", s.lon_boundary_band.iter().map(|v| u8::from(*v)).collect());
            write("surface_lat_integer_band", s.lat_integer_band.iter().map(|v| u8::from(*v)).collect());
        }
        manifest["cases"][case] = row;
    }
    std::fs::write(output.join("manifest.json"),
        serde_json::to_string_pretty(&manifest).unwrap() + "\n").unwrap();
}
