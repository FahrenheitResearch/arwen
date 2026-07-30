//! Export one rw-store hour file + its grid to a self-contained
//! **NetCDF classic 64-bit-offset (CDF-2)** file.
//!
//! The entry point is [`export_hour_to_netcdf3`].  It maps the hour's
//! variables to NetCDF dimensions and variables following the contract in
//! `docs/FORMAT.md §8` and the CF-1.6 conventions.
//!
//! ## Mapping overview
//!
//! ### Dimensions
//! - `y` (= `ny`), `x` (= `nx`) — always present.
//! - For each **distinct** `levels_hpa` vector among the exported 3D
//!   variables (first-seen order) a pressure-level dimension is created:
//!   `level`, `level2`, `level3`, … Each gets a same-named `NC_FLOAT`
//!   coordinate variable holding the hPa values with attrs
//!   `units="hPa"`, `long_name="pressure level"`, `positive="down"`.
//!
//! ### Coordinate variables (always written first, before data vars)
//! - `lat(y, x)` — from [`GridFile`]; attrs `units="degrees_north"`,
//!   `long_name="latitude"`.
//! - `lon(y, x)` — from [`GridFile`]; attrs `units="degrees_east"`,
//!   `long_name="longitude"`.
//! - `levelN(levelN)` — one per distinct level set (see above).
//!
//! ### Data variables
//! - 2D (`surface2d`): shape `(y, x)`, decoded with `read_full_2d`.
//! - 3D (`pressure3d`): shape `(levelN, y, x)`, decoded with `read_full_3d`
//!   (level-major `[level][y][x]` order — matches the NetCDF dim order
//!   directly with no transpose).
//! - All data vars carry: `units` (from meta), `long_name` = variable name,
//!   `coordinates = "lat lon"`, `_FillValue = f32::NAN`.
//! - 3D vars additionally carry `rw_quantization` (scientific-honesty note).
//!
//! ### Global attributes
//! `Conventions`, `title`, `model`, `run`, `grid_hash`, `source`, `comment`,
//! plus either v1 `forecast_hour` or v2 `storage_slot`, `lead_seconds`, and
//! `valid_unix`. Exact integer times use decimal text because NetCDF-3 has no
//! i64 attribute type and an f32 timestamp would be lossy.
//!
//! ### Name sanitation
//! rw-store names are `[a-z0-9_]` — the only chars that could violate
//! NC name rules are a leading digit.  If a name is not NC-safe the export
//! **errors** (`RwStoreError::Format`) rather than silently renaming it.
//!
//! ### Memory discipline
//! All variable definitions are built up front; data is written one variable
//! at a time — read → write → drop.  Peak memory is roughly one decoded 3D
//! volume.

use std::path::Path;

use crate::error::{RwResult, RwStoreError};
use crate::grid::GridFile;
use crate::netcdf3::{Nc3Attr, Nc3Dim, Nc3VarDef, Nc3Writer, name_is_valid};
use crate::reader::HourReader;

// ── quantization disclosure string (matches FORMAT.md §8 / §3.2) ──────────
const RW_QUANTIZATION: &str = "affine-i16 per 16x16-column chunk at ingest; \
     ~ (chunk max-min)/65534 absolute step";

// ── ExportSummary ───────────────────────────────────────────────────────────

/// Summary returned by [`export_hour_to_netcdf3`].
///
/// `variables` counts only **data variables** (the variables that carry
/// weather field values).  Coordinate variables (`lat`, `lon`, and the
/// `levelN` pressure-coordinate variables) are always written but are *not*
/// included in this count — they are infrastructure, not data products.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportSummary {
    /// Number of data variables written (coordinate vars excluded).
    pub variables: usize,
    /// Final size of the output `.nc` file in bytes (measured after `fsync`).
    pub bytes_written: u64,
}

// ── main export function ────────────────────────────────────────────────────

/// Export one hour file + its grid as a CDF-2 NetCDF file at `out`.
///
/// `vars` selects which variables to export; `None` exports all variables in
/// meta order.  Unknown names in the filter produce
/// [`RwStoreError::UnknownVariable`] listing all available names.
///
/// On error a partial `.nc` may exist at `out`; the caller is responsible for
/// removing it (matches standard NetCDF tooling behaviour — see `netcdf3.rs`
/// module docs for the rationale).
pub fn export_hour_to_netcdf3(
    hour: &HourReader,
    grid: &GridFile,
    vars: Option<&[String]>,
    out: &Path,
) -> RwResult<ExportSummary> {
    let meta = hour.meta();
    let (nx, ny) = (meta.nx, meta.ny);

    // ── 1. Resolve the variable list ──────────────────────────────────────
    let all_vars: Vec<&crate::format::RwsVariableMeta> = meta.variables.iter().collect();
    let export_vars: Vec<&crate::format::RwsVariableMeta> = match vars {
        None => all_vars.clone(),
        Some(names) => {
            let available: Vec<&str> = all_vars.iter().map(|v| v.name.as_str()).collect();
            for name in names {
                if !available.contains(&name.as_str()) {
                    return Err(RwStoreError::UnknownVariable(format!(
                        "{name} (available: {})",
                        available.join(", ")
                    )));
                }
            }
            // Preserve meta order, filtered to the requested names.
            all_vars
                .iter()
                .filter(|v| names.iter().any(|n| n == &v.name))
                .copied()
                .collect()
        }
    };

    // ── 2. Validate NC names ──────────────────────────────────────────────
    // Shares `netcdf3::name_is_valid` (the exact rule the writer enforces) so
    // there is no second copy to drift; we wrap it here only to emit a clearer,
    // variable-naming `Format` error before any writer allocation.
    for v in &export_vars {
        if !name_is_valid(&v.name) {
            return Err(RwStoreError::Format(format!(
                "export: variable name '{}' is not NC-safe \
                 (must start [A-Za-z_], rest [A-Za-z0-9_+.@-])",
                v.name
            )));
        }
    }

    // ── 3. Collect distinct level sets (first-seen order) ─────────────────
    // level_sets[i] = the distinct levels_hpa vector; dim name = "level" (i==0)
    // or "level{i+1}" (i>0).
    let mut level_sets: Vec<Vec<u16>> = Vec::new();
    for v in &export_vars {
        if v.kind == "pressure3d"
            && !v.levels_hpa.is_empty()
            && !level_sets.iter().any(|set| set == &v.levels_hpa)
        {
            level_sets.push(v.levels_hpa.clone());
        }
    }

    // ── 4. Build NetCDF dimension list ────────────────────────────────────
    // Order: y, x, then level/level2/level3…
    // dim-index 0 = y, 1 = x, 2+ = one per distinct level set.
    let dim_y = 0usize;
    let dim_x = 1usize;
    let dim_level_base = 2usize;

    let mut dims: Vec<Nc3Dim> = Vec::new();
    dims.push(Nc3Dim {
        name: "y".to_string(),
        len: ny,
    });
    dims.push(Nc3Dim {
        name: "x".to_string(),
        len: nx,
    });
    for (i, level_set) in level_sets.iter().enumerate() {
        let dim_name = if i == 0 {
            "level".to_string()
        } else {
            format!("level{}", i + 1)
        };
        dims.push(Nc3Dim {
            name: dim_name,
            len: level_set.len(),
        });
    }

    // ── 5. Global attributes ──────────────────────────────────────────────
    let source = format!(
        "rw-store {} via rws export, writer {} {} {}",
        meta.schema, meta.writer.name, meta.writer.version, meta.writer.build
    );
    let mut gattrs = vec![
        Nc3Attr::text("Conventions", "CF-1.6"),
        Nc3Attr::text("title", "rusty-weather rw-store export"),
        Nc3Attr::text("model", &meta.model),
        Nc3Attr::text("run", &meta.run),
    ];
    if let Some(exact_time) = meta.exact_time() {
        // NetCDF-3 has no i64 attribute type. Decimal text preserves both
        // values exactly; using f32 would silently lose timestamp precision.
        gattrs.push(Nc3Attr::floats(
            "storage_slot",
            vec![meta.forecast_hour as f32],
        ));
        gattrs.push(Nc3Attr::text(
            "lead_seconds",
            exact_time.lead_seconds.to_string(),
        ));
        gattrs.push(Nc3Attr::text(
            "valid_unix",
            exact_time.valid_unix.to_string(),
        ));
    } else {
        gattrs.push(Nc3Attr::floats(
            "forecast_hour",
            vec![meta.forecast_hour as f32],
        ));
    }
    gattrs.extend([
        Nc3Attr::text("grid_hash", &meta.grid_hash),
        Nc3Attr::text("source", &source),
        Nc3Attr::text(
            "comment",
            "Format spec: https://github.com/FahrenheitResearch/rusty-weather/blob/main/docs/FORMAT.md",
        ),
    ]);

    // ── 6. Variable definitions (order: lat, lon, level coords, data vars) ─
    let mut var_defs: Vec<Nc3VarDef> = Vec::new();

    // lat(y, x)
    var_defs.push(Nc3VarDef {
        name: "lat".to_string(),
        dimids: vec![dim_y, dim_x],
        attrs: vec![
            Nc3Attr::text("units", "degrees_north"),
            Nc3Attr::text("long_name", "latitude"),
        ],
    });

    // lon(y, x)
    var_defs.push(Nc3VarDef {
        name: "lon".to_string(),
        dimids: vec![dim_y, dim_x],
        attrs: vec![
            Nc3Attr::text("units", "degrees_east"),
            Nc3Attr::text("long_name", "longitude"),
        ],
    });

    // level/level2/… coordinate variables
    for (i, _) in level_sets.iter().enumerate() {
        let dim_name = if i == 0 {
            "level".to_string()
        } else {
            format!("level{}", i + 1)
        };
        let dim_id = dim_level_base + i;
        var_defs.push(Nc3VarDef {
            name: dim_name.clone(),
            dimids: vec![dim_id],
            attrs: vec![
                Nc3Attr::text("units", "hPa"),
                Nc3Attr::text("long_name", "pressure level"),
                Nc3Attr::text("positive", "down"),
            ],
        });
    }

    // data variables
    for v in &export_vars {
        let dimids = if v.kind == "pressure3d" {
            // Find which level-set index this variable uses.
            let ls_idx = level_sets
                .iter()
                .position(|set| set == &v.levels_hpa)
                .expect("level set was registered above");
            vec![dim_level_base + ls_idx, dim_y, dim_x]
        } else {
            vec![dim_y, dim_x]
        };

        let mut attrs = vec![
            Nc3Attr::text("units", &v.units),
            Nc3Attr::text("long_name", &v.name),
            Nc3Attr::text("coordinates", "lat lon"),
            Nc3Attr::floats("_FillValue", vec![f32::NAN]),
        ];
        if v.kind == "pressure3d" {
            attrs.push(Nc3Attr::text("rw_quantization", RW_QUANTIZATION));
        }

        var_defs.push(Nc3VarDef {
            name: v.name.clone(),
            dimids,
            attrs,
        });
    }

    // ── 7. Create the writer (validates defs, writes header) ─────────────
    let mut writer = Nc3Writer::create(out, dims, gattrs, var_defs)?;

    // ── 8. Write coordinate data ──────────────────────────────────────────

    // lat and lon are flat row-major arrays from GridFile — already the right
    // shape and order for (y, x).
    writer.write_var(&grid.lat)?;
    writer.write_var(&grid.lon)?;

    // level coordinate variables: hPa values as f32.
    for level_set in &level_sets {
        let hpa: Vec<f32> = level_set.iter().map(|&l| l as f32).collect();
        writer.write_var(&hpa)?;
    }

    // ── 9. Write data variables (one at a time) ───────────────────────────
    for v in &export_vars {
        if v.kind == "pressure3d" {
            let data = hour.read_full_3d(&v.name)?;
            writer.write_var(&data)?;
            // data is dropped here — peak = one 3D volume
        } else {
            let data = hour.read_full_2d(&v.name)?;
            writer.write_var(&data)?;
        }
    }

    // ── 10. Finish (flush + fsync) ────────────────────────────────────────
    writer.finish()?;

    let bytes_written = std::fs::metadata(out)?.len();

    Ok(ExportSummary {
        variables: export_vars.len(),
        bytes_written,
    })
}

// ── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::RwsExactTime;
    use crate::netcdf3::NC_FLOAT;
    use crate::netcdf3::test_parser::{ParsedAttrValue, ParsedNc};
    use rustwx_core::{GridShape, LatLonGrid};

    // ── test helpers ────────────────────────────────────────────────────────

    const NX: usize = 20;
    const NY: usize = 300;

    fn test_dir(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("rw-store-export-{}-{}", std::process::id(), name));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Build the canonical test lat/lon grid (matches the golden fixture shape).
    fn build_latlongrid() -> LatLonGrid {
        let mut lat = Vec::with_capacity(NX * NY);
        let mut lon = Vec::with_capacity(NX * NY);
        for gy in 0..NY {
            for gx in 0..NX {
                lat.push(30.0_f32 + 0.01 * gy as f32);
                lon.push(-100.0_f32 + 0.05 * gx as f32);
            }
        }
        LatLonGrid::new(GridShape::new(NX, NY).unwrap(), lat, lon).unwrap()
    }

    /// t2m field: smooth analytic surface.
    fn t2m_values() -> Vec<f32> {
        (0..NY)
            .flat_map(|gy| {
                (0..NX).map(move |gx| 280.0 + (0.1 * gx as f32).sin() * 5.0 + 0.02 * gy as f32)
            })
            .collect()
    }

    /// mask_demo: NaN for gy >= 256 and (gx == 3 && gy < 10), else same formula.
    fn mask_values() -> Vec<f32> {
        (0..NY)
            .flat_map(|gy| {
                (0..NX).map(move |gx| {
                    if gy >= 256 || (gx == 3 && gy < 10) {
                        f32::NAN
                    } else {
                        280.0 + (0.1 * gx as f32).sin() * 5.0 + 0.02 * gy as f32
                    }
                })
            })
            .collect()
    }

    /// const_demo: all 101325.0.
    fn const_values() -> Vec<f32> {
        vec![101325.0f32; NY * NX]
    }

    /// Build a 3-level 3D volume at levels [850, 700, 500] hPa.
    /// Column at (gx==5, gy==5) is all NaN across all levels.
    fn temp_iso_planes(levels: &[u16]) -> Vec<Vec<f32>> {
        levels
            .iter()
            .enumerate()
            .map(|(level_idx, _)| {
                (0..NY)
                    .flat_map(|gy| {
                        (0..NX).map(move |gx| {
                            if gx == 5 && gy == 5 {
                                f32::NAN
                            } else {
                                270.0 - (level_idx as f32) * 10.0
                                    + (0.05 * gx as f32).cos()
                                    + 0.01 * gy as f32
                            }
                        })
                    })
                    .collect()
            })
            .collect()
    }

    /// Write the standard test store (t2m, mask_demo, const_demo, temp_iso)
    /// and return (hour_path, grid_file).
    fn write_test_store(store_root: &std::path::Path) -> (std::path::PathBuf, GridFile) {
        use crate::ingest::HourIngestWriter;

        let grid = build_latlongrid();
        let mut w = HourIngestWriter::begin(
            store_root,
            "test_model",
            "20260101_00z",
            0,
            &grid,
            None,
            "export-test-build",
        )
        .unwrap();

        w.add_field_2d(
            "t2m",
            "K",
            serde_json::json!({"var":"TMP","level":"2 m above ground"}),
            &t2m_values(),
        )
        .unwrap();
        w.add_field_2d(
            "mask_demo",
            "1",
            serde_json::json!({"var":"MASK"}),
            &mask_values(),
        )
        .unwrap();
        w.add_field_2d(
            "const_demo",
            "Pa",
            serde_json::json!({"var":"CONST"}),
            &const_values(),
        )
        .unwrap();

        let levels: [u16; 3] = [850, 700, 500];
        let planes = temp_iso_planes(&levels);
        let level_pairs: Vec<(u16, &[f32])> = levels
            .iter()
            .zip(planes.iter())
            .map(|(&l, p)| (l, p.as_slice()))
            .collect();
        w.add_volume(
            "temp_iso",
            "K",
            serde_json::json!({"var":"TMP","level":"{level} mb"}),
            &level_pairs,
        )
        .unwrap();

        let written = w.finish(1_770_000_000).unwrap();

        let grid_file = GridFile::open(
            &store_root
                .join("test_model")
                .join("20260101_00z")
                .join("grid.rwg"),
        )
        .unwrap();

        (written.path, grid_file)
    }

    // ── NaN-safe bit equality helper ────────────────────────────────────────

    fn assert_f32_bits_eq(got: f32, want: f32, ctx: &str) {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{ctx}: got {got} vs want {want}"
        );
    }

    // ── Test 1: export_full_hour_round_trips ────────────────────────────────

    #[test]
    fn export_full_hour_round_trips() {
        let dir = test_dir("full");
        let (hour_path, grid_file) = write_test_store(&dir);
        let out_nc = dir.join("export.nc");

        let hour = HourReader::open(&hour_path).unwrap();
        let summary = export_hour_to_netcdf3(&hour, &grid_file, None, &out_nc).unwrap();

        // ExportSummary: 4 data variables.
        assert_eq!(
            summary.variables, 4,
            "4 data vars: t2m, mask_demo, const_demo, temp_iso"
        );
        assert!(summary.bytes_written > 0, "output file non-empty");

        let bytes = std::fs::read(&out_nc).unwrap();
        let parsed = ParsedNc::parse(&bytes).expect("output must parse as CDF-2");

        // ── Dims ──
        // Dims: y=300, x=20, level=3
        let dim_map: std::collections::HashMap<&str, u32> =
            parsed.dims.iter().map(|(n, l)| (n.as_str(), *l)).collect();
        assert_eq!(dim_map["y"], NY as u32, "dim y");
        assert_eq!(dim_map["x"], NX as u32, "dim x");
        assert_eq!(dim_map["level"], 3, "dim level");
        // No level2 dim (only one distinct level set).
        assert!(!dim_map.contains_key("level2"), "no level2 dim expected");

        // ── Global attrs ──
        let conv = parsed.gattr("Conventions").unwrap();
        assert_eq!(*conv, ParsedAttrValue::Text("CF-1.6".to_string()));
        let title = parsed.gattr("title").unwrap();
        assert_eq!(
            *title,
            ParsedAttrValue::Text("rusty-weather rw-store export".to_string())
        );
        assert!(parsed.gattr("model").is_some());
        assert!(parsed.gattr("run").is_some());
        assert!(parsed.gattr("grid_hash").is_some());
        assert!(parsed.gattr("source").is_some());
        assert!(parsed.gattr("comment").is_some());
        // forecast_hour is a Floats attr with one value.
        match parsed.gattr("forecast_hour").unwrap() {
            ParsedAttrValue::Floats(fs) => {
                assert_eq!(fs.len(), 1);
                assert_eq!(fs[0], 0.0f32);
            }
            other => panic!("forecast_hour must be Floats, got {other:?}"),
        }

        // ── Variable presence and order: lat, lon, level, t2m, mask_demo, const_demo, temp_iso ──
        let var_names: Vec<&str> = parsed.vars.iter().map(|v| v.name.as_str()).collect();
        assert_eq!(
            var_names,
            vec![
                "lat",
                "lon",
                "level",
                "t2m",
                "mask_demo",
                "const_demo",
                "temp_iso"
            ],
            "variable order"
        );

        // Dim-id lookups for assertions below.
        let id_y = parsed.dims.iter().position(|(n, _)| n == "y").unwrap();
        let id_x = parsed.dims.iter().position(|(n, _)| n == "x").unwrap();
        let id_level = parsed.dims.iter().position(|(n, _)| n == "level").unwrap();

        // ── lat(y,x) ──
        {
            let v = parsed.vars.iter().find(|v| v.name == "lat").unwrap();
            assert_eq!(v.dimids, vec![id_y, id_x]);
            assert_eq!(v.nc_type, NC_FLOAT);
            assert_eq!(
                v.attr("units").unwrap(),
                &ParsedAttrValue::Text("degrees_north".to_string())
            );
            // Bit-exact round-trip vs GridFile.
            let elems = NY * NX;
            let nc_lat = parsed.read_var_floats(
                &bytes,
                parsed.vars.iter().position(|v| v.name == "lat").unwrap(),
                elems,
            );
            for (i, (&got, &want)) in nc_lat.iter().zip(grid_file.lat.iter()).enumerate() {
                assert_f32_bits_eq(got, want, &format!("lat[{i}]"));
            }
        }

        // ── lon(y,x) ──
        {
            let elems = NY * NX;
            let v_idx = parsed.vars.iter().position(|v| v.name == "lon").unwrap();
            let nc_lon = parsed.read_var_floats(&bytes, v_idx, elems);
            for (i, (&got, &want)) in nc_lon.iter().zip(grid_file.lon.iter()).enumerate() {
                assert_f32_bits_eq(got, want, &format!("lon[{i}]"));
            }
        }

        // ── level coordinate var ──
        {
            let v_idx = parsed.vars.iter().position(|v| v.name == "level").unwrap();
            let v = &parsed.vars[v_idx];
            assert_eq!(v.dimids, vec![id_level]);
            assert_eq!(
                v.attr("units").unwrap(),
                &ParsedAttrValue::Text("hPa".to_string())
            );
            assert_eq!(
                v.attr("positive").unwrap(),
                &ParsedAttrValue::Text("down".to_string())
            );
            let nc_levels = parsed.read_var_floats(&bytes, v_idx, 3);
            // levels_hpa stored descending: [850, 700, 500].
            assert_eq!(nc_levels, vec![850.0f32, 700.0, 500.0]);
        }

        // ── t2m: 2D, bit-exact vs read_full_2d ──
        {
            let v_idx = parsed.vars.iter().position(|v| v.name == "t2m").unwrap();
            let v = &parsed.vars[v_idx];
            assert_eq!(v.dimids, vec![id_y, id_x]);
            assert_eq!(
                v.attr("coordinates").unwrap(),
                &ParsedAttrValue::Text("lat lon".to_string())
            );
            match v.attr("_FillValue").unwrap() {
                ParsedAttrValue::Floats(fs) => {
                    assert_eq!(fs.len(), 1);
                    assert!(fs[0].is_nan());
                }
                other => panic!("_FillValue must be Floats NaN, got {other:?}"),
            }
            // rw_quantization must NOT be present on 2D var.
            assert!(
                v.attr("rw_quantization").is_none(),
                "no rw_quantization on 2D var"
            );

            let nc_t2m = parsed.read_var_floats(&bytes, v_idx, NY * NX);
            let store_t2m = hour.read_full_2d("t2m").unwrap();
            for i in 0..NY * NX {
                assert_f32_bits_eq(nc_t2m[i], store_t2m[i], &format!("t2m[{i}]"));
            }
        }

        // ── mask_demo: NaN holes preserved ──
        {
            let v_idx = parsed
                .vars
                .iter()
                .position(|v| v.name == "mask_demo")
                .unwrap();
            let nc_mask = parsed.read_var_floats(&bytes, v_idx, NY * NX);
            let store_mask = hour.read_full_2d("mask_demo").unwrap();
            // spot-check NaN at gy>=256 and at gx==3,gy<10.
            assert!(nc_mask[256 * NX].is_nan(), "mask_demo NaN at gy=256");
            assert!(nc_mask[3].is_nan(), "mask_demo NaN at gx=3,gy=0");
            // Bit-exact overall.
            for i in 0..NY * NX {
                assert_f32_bits_eq(nc_mask[i], store_mask[i], &format!("mask_demo[{i}]"));
            }
        }

        // ── temp_iso: 3D, bit-exact vs read_full_3d, rw_quantization attr ──
        {
            let v_idx = parsed
                .vars
                .iter()
                .position(|v| v.name == "temp_iso")
                .unwrap();
            let v = &parsed.vars[v_idx];
            assert_eq!(v.dimids, vec![id_level, id_y, id_x]);
            match v.attr("rw_quantization").unwrap() {
                ParsedAttrValue::Text(s) => {
                    assert!(
                        s.contains("affine-i16"),
                        "rw_quantization text must mention affine-i16, got: {s}"
                    );
                }
                other => panic!("rw_quantization must be Text, got {other:?}"),
            }

            let nc_temp = parsed.read_var_floats(&bytes, v_idx, 3 * NY * NX);
            let store_temp = hour.read_full_3d("temp_iso").unwrap();
            for i in 0..3 * NY * NX {
                assert_f32_bits_eq(nc_temp[i], store_temp[i], &format!("temp_iso[{i}]"));
            }
            // NaN column at gx==5, gy==5 must survive.
            for lvl in 0..3usize {
                let idx = lvl * NY * NX + 5 * NX + 5;
                assert!(nc_temp[idx].is_nan(), "temp_iso NaN column lvl={lvl}");
            }
        }
    }

    // ── Exact-time export attributes ────────────────────────────────────────

    #[test]
    fn exact_export_uses_lossless_time_attributes_not_forecast_hour() {
        use crate::ingest::HourIngestWriter;

        let dir = test_dir("exact-time");
        let store_root = dir.join("store");
        let grid = build_latlongrid();
        let exact = RwsExactTime {
            lead_seconds: 2_700,
            valid_unix: 1_700_002_700,
        };
        let mut writer = HourIngestWriter::begin_exact(
            &store_root,
            "test_model",
            "exact_run",
            7,
            exact,
            &grid,
            None,
            "test-build",
        )
        .unwrap();
        writer
            .add_field_2d("t2m", "K", serde_json::json!({"var":"TMP"}), &t2m_values())
            .unwrap();
        let written = writer.finish(1_770_000_000).unwrap();
        let grid_file = GridFile::open(
            &store_root
                .join("test_model")
                .join("exact_run")
                .join("grid.rwg"),
        )
        .unwrap();
        let hour = HourReader::open(&written.path).unwrap();
        let out = dir.join("exact.nc");
        export_hour_to_netcdf3(&hour, &grid_file, None, &out).unwrap();
        let bytes = std::fs::read(&out).unwrap();
        let parsed = ParsedNc::parse(&bytes).unwrap();

        assert!(parsed.gattr("forecast_hour").is_none());
        assert_eq!(
            parsed.gattr("storage_slot"),
            Some(&ParsedAttrValue::Floats(vec![7.0]))
        );
        assert_eq!(
            parsed.gattr("lead_seconds"),
            Some(&ParsedAttrValue::Text("2700".to_string()))
        );
        assert_eq!(
            parsed.gattr("valid_unix"),
            Some(&ParsedAttrValue::Text("1700002700".to_string()))
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn export_vars_filter() {
        let dir = test_dir("filter");
        let (hour_path, grid_file) = write_test_store(&dir);
        let out_nc = dir.join("filter.nc");

        let hour = HourReader::open(&hour_path).unwrap();
        let filter = vec!["t2m".to_string()];
        let summary = export_hour_to_netcdf3(&hour, &grid_file, Some(&filter), &out_nc).unwrap();

        // ExportSummary.variables counts only data vars.
        assert_eq!(summary.variables, 1, "only t2m exported as data var");

        let bytes = std::fs::read(&out_nc).unwrap();
        let parsed = ParsedNc::parse(&bytes).expect("filter output must parse");

        // Variables: lat, lon, t2m — no level dim/coord (temp_iso excluded).
        let var_names: Vec<&str> = parsed.vars.iter().map(|v| v.name.as_str()).collect();
        assert_eq!(
            var_names,
            vec!["lat", "lon", "t2m"],
            "vars with t2m-only filter"
        );

        // No level dim at all.
        let dim_names: Vec<&str> = parsed.dims.iter().map(|(n, _)| n.as_str()).collect();
        assert!(
            !dim_names.contains(&"level"),
            "no level dim when 3D var excluded"
        );

        // t2m values present.
        let v_idx = parsed.vars.iter().position(|v| v.name == "t2m").unwrap();
        let nc_t2m = parsed.read_var_floats(&bytes, v_idx, NY * NX);
        let store_t2m = hour.read_full_2d("t2m").unwrap();
        for i in 0..NY * NX {
            assert_f32_bits_eq(nc_t2m[i], store_t2m[i], &format!("t2m_filter[{i}]"));
        }
    }

    // ── Test 3: export_unknown_var_errors ───────────────────────────────────

    #[test]
    fn export_unknown_var_errors() {
        let dir = test_dir("unknownvar");
        let (hour_path, grid_file) = write_test_store(&dir);
        let out_nc = dir.join("unknown.nc");

        let hour = HourReader::open(&hour_path).unwrap();
        let filter = vec!["no_such_var".to_string()];
        let err = export_hour_to_netcdf3(&hour, &grid_file, Some(&filter), &out_nc).unwrap_err();

        match &err {
            RwStoreError::UnknownVariable(msg) => {
                assert!(
                    msg.contains("no_such_var"),
                    "error must name the unknown var, got: {msg}"
                );
                // Must list available names in the message.
                assert!(
                    msg.contains("t2m"),
                    "error must list available names, got: {msg}"
                );
            }
            other => panic!("expected UnknownVariable, got {other:?}"),
        }
    }

    // ── Test 4: export_two_level_sets ───────────────────────────────────────

    #[test]
    fn export_two_level_sets() {
        // Build a store with two 3D vars that have DIFFERENT level sets.
        let dir = test_dir("twolevelsets");

        // Grid
        let grid = build_latlongrid();

        // Two distinct level sets.
        let levels_a: [u16; 3] = [850, 700, 500];
        let levels_b: [u16; 2] = [925, 850]; // different vector even though 850 shared

        let planes_a = temp_iso_planes(&levels_a);
        let planes_b = temp_iso_planes(&[925, 850]); // reuse helper, 2 levels

        let store_root = dir.join("store");
        std::fs::create_dir_all(&store_root).unwrap();

        use crate::ingest::HourIngestWriter;
        let mut w = HourIngestWriter::begin(
            &store_root,
            "test_model",
            "20260101_00z",
            0,
            &grid,
            None,
            "export-test-build",
        )
        .unwrap();

        // Need at least one 2D field via add_field_2d to provide the grid.
        // Actually HourIngestWriter::begin already sets the grid; we still
        // need a 2D field in the hour file. add_derived_2d is fine.
        w.add_derived_2d("t2m_dummy", "K", &t2m_values()).unwrap();

        let level_pairs_a: Vec<(u16, &[f32])> = levels_a
            .iter()
            .zip(planes_a.iter())
            .map(|(&l, p)| (l, p.as_slice()))
            .collect();
        w.add_volume("temp_a", "K", serde_json::Value::Null, &level_pairs_a)
            .unwrap();

        let level_pairs_b: Vec<(u16, &[f32])> = levels_b
            .iter()
            .zip(planes_b.iter())
            .map(|(&l, p)| (l, p.as_slice()))
            .collect();
        w.add_volume("temp_b", "K", serde_json::Value::Null, &level_pairs_b)
            .unwrap();

        let written = w.finish(1_770_000_000).unwrap();

        let grid_file = GridFile::open(
            &store_root
                .join("test_model")
                .join("20260101_00z")
                .join("grid.rwg"),
        )
        .unwrap();

        let out_nc = dir.join("two_levels.nc");
        let hour = HourReader::open(&written.path).unwrap();
        let summary = export_hour_to_netcdf3(&hour, &grid_file, None, &out_nc).unwrap();

        // 3 data vars: t2m_dummy, temp_a, temp_b.
        assert_eq!(summary.variables, 3);

        let bytes = std::fs::read(&out_nc).unwrap();
        let parsed = ParsedNc::parse(&bytes).expect("two-level-set output must parse");

        // Expect dims: y, x, level (len=3 for temp_a), level2 (len=2 for temp_b).
        let dim_map: std::collections::HashMap<&str, u32> =
            parsed.dims.iter().map(|(n, l)| (n.as_str(), *l)).collect();
        assert_eq!(dim_map["level"], 3, "first level dim has 3 levels");
        assert_eq!(dim_map["level2"], 2, "second level dim has 2 levels");

        // Vars: lat, lon, level, level2, t2m_dummy, temp_a, temp_b.
        let var_names: Vec<&str> = parsed.vars.iter().map(|v| v.name.as_str()).collect();
        assert_eq!(
            var_names,
            vec![
                "lat",
                "lon",
                "level",
                "level2",
                "t2m_dummy",
                "temp_a",
                "temp_b"
            ]
        );

        // Level coord values.
        let id_level = parsed.dims.iter().position(|(n, _)| n == "level").unwrap();
        let id_level2 = parsed.dims.iter().position(|(n, _)| n == "level2").unwrap();
        let level_v_idx = parsed.vars.iter().position(|v| v.name == "level").unwrap();
        let level2_v_idx = parsed.vars.iter().position(|v| v.name == "level2").unwrap();
        let nc_level = parsed.read_var_floats(&bytes, level_v_idx, 3);
        let nc_level2 = parsed.read_var_floats(&bytes, level2_v_idx, 2);
        // temp_a levels [850,700,500] stored descending.
        assert_eq!(nc_level, vec![850.0f32, 700.0, 500.0], "level coord values");
        // temp_b levels [925,850] stored descending.
        assert_eq!(nc_level2, vec![925.0f32, 850.0], "level2 coord values");

        // temp_a dimids reference level (id_level), temp_b reference level2 (id_level2).
        let id_y = parsed.dims.iter().position(|(n, _)| n == "y").unwrap();
        let id_x = parsed.dims.iter().position(|(n, _)| n == "x").unwrap();
        let temp_a_v = parsed.vars.iter().find(|v| v.name == "temp_a").unwrap();
        let temp_b_v = parsed.vars.iter().find(|v| v.name == "temp_b").unwrap();
        assert_eq!(temp_a_v.dimids, vec![id_level, id_y, id_x], "temp_a dims");
        assert_eq!(temp_b_v.dimids, vec![id_level2, id_y, id_x], "temp_b dims");
    }
}
