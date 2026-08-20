//! Gather one MPAS history frame onto a render window and write it as a
//! wrfout-shaped classic netCDF frame.
//!
//! The renderer has no unstructured path. Its real named products -- the ones
//! with operational colortables, titles and map furniture -- come from the
//! real-WRF import route, which needs a structured grid, `Times`,
//! `XLAT`/`XLONG` and a rank-4 `T`. This module produces exactly that, and
//! nothing more: no field is invented, and every emitted file carries the
//! source history digest, the mesh digest, the weights digest, the resample
//! method and the engine that wrote it.

use std::path::{Path, PathBuf};

use rw_store::netcdf_classic::{
    NcAttr, NcClassicWriter, NcData, NcDim, NcFormat, NcType, NcVarDef,
};

use crate::error::{MpasError, MpasResult};
use crate::fieldmap::{field_set_allows, field_set_is_known, FieldSource, Rank, FIELD_MAP};
use crate::history::{expected_levels, MpasFrame, Timestamp};
use crate::weights::NearestCellWeights;
use crate::window::ProjAttr;

pub const BRIDGE_SCHEMA: &str = "mpas-port.history-to-wrfout-render-frame/v1";

/// WRF's `T` is a perturbation about this reference potential temperature,
/// and wrf-core reconstructs full theta as `T + 300`.
pub const WRF_THETA_REFERENCE_K: f32 = 300.0;
/// wrf-core divides full geopotential by exactly this to get height.
pub const WRF_GRAVITY: f32 = 9.80665;

const MMINLU: &str = "MODIFIED_IGBP_MODIS_NOAH";

/// What one conversion produced.
#[derive(Debug, Clone)]
pub struct EmittedFrame {
    pub path: PathBuf,
    pub window: String,
    pub shape: (usize, usize),
    pub levels: usize,
    pub written: Vec<String>,
    pub absent: Vec<String>,
    pub seconds: f64,
    pub bytes: u64,
    pub valid_time: Timestamp,
    pub simulation_start: Timestamp,
    pub lead_seconds: i64,
}

/// Knobs the caller sets; everything else is fixed by the schema.
#[derive(Debug, Clone)]
pub struct ConvertOptions {
    pub simulation_start: Option<Timestamp>,
    pub field_set: String,
    pub grid_id: i32,
    pub title: String,
    pub format: NcFormat,
    pub clobber: bool,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        ConvertOptions {
            simulation_start: None,
            field_set: "full".to_string(),
            grid_id: 1,
            title: "MPAS-A v8.4.1 CUDA port history resampled for rw_wrfbatch".to_string(),
            format: NcFormat::Offset64,
            clobber: false,
        }
    }
}

/// Cell-centre values to x-staggered faces, WRF `U` layout.
///
/// The renderer destaggers with a two-point average, so the round trip is a
/// 1/4-1/2-1/4 smoother in x rather than the identity. That is a real and
/// deliberate divergence: WRF's `U` genuinely lives on faces and there is no
/// local face field whose two-point average reproduces an arbitrary
/// cell-centre field. It touches the three-dimensional wind only; the surface
/// wind products read the unstaggered `U10`/`V10` and are untouched.
fn restagger_x(values: &[f32], levels: usize, ny: usize, nx: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; levels * ny * (nx + 1)];
    for level in 0..levels {
        for row in 0..ny {
            let src = &values[(level * ny + row) * nx..(level * ny + row + 1) * nx];
            let dst = &mut out[(level * ny + row) * (nx + 1)..(level * ny + row + 1) * (nx + 1)];
            dst[0] = src[0];
            for i in 1..nx {
                dst[i] = (src[i - 1] + src[i]) * 0.5;
            }
            dst[nx] = src[nx - 1];
        }
    }
    out
}

/// Cell-centre values to y-staggered faces, WRF `V` layout.
fn restagger_y(values: &[f32], levels: usize, ny: usize, nx: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; levels * (ny + 1) * nx];
    for level in 0..levels {
        let src = &values[level * ny * nx..(level + 1) * ny * nx];
        let dst = &mut out[level * (ny + 1) * nx..(level + 1) * (ny + 1) * nx];
        dst[..nx].copy_from_slice(&src[..nx]);
        for row in 1..ny {
            for i in 0..nx {
                dst[row * nx + i] = (src[(row - 1) * nx + i] + src[row * nx + i]) * 0.5;
            }
        }
        dst[ny * nx..].copy_from_slice(&src[(ny - 1) * nx..ny * nx]);
    }
    out
}

/// The per-variable attributes every emitted field carries.
fn field_attrs(rank_is_3d: bool, description: &str, units: &str, stagger: &str) -> Vec<NcAttr> {
    vec![
        NcAttr::int("FieldType", 104),
        NcAttr::text("MemoryOrder", if rank_is_3d { "XYZ" } else { "XY " }),
        NcAttr::text("description", description),
        NcAttr::text("units", units),
        NcAttr::text("stagger", stagger),
        NcAttr::text("coordinates", "XLONG XLAT XTIME"),
    ]
}

/// Write one wrfout-shaped frame. Returns the emission record.
pub fn convert_frame(
    frame: &MpasFrame,
    weights: &NearestCellWeights,
    destination: &Path,
    options: &ConvertOptions,
) -> MpasResult<EmittedFrame> {
    let started = std::time::Instant::now();
    if !field_set_is_known(&options.field_set) {
        return Err(MpasError::Refusal(format!(
            "unknown field set '{}' (known: full, surface)",
            options.field_set
        )));
    }
    if destination.exists() && !options.clobber {
        return Err(MpasError::Refusal(format!(
            "{} exists; pass --clobber to replace it",
            destination.display()
        )));
    }
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let (ny, nx) = weights.shape();
    let nz = frame.n_levels;
    let origin = options.simulation_start.unwrap_or(frame.valid_time);
    let lead_seconds = frame.valid_time.seconds_since(&origin);
    if lead_seconds < 0 {
        return Err(MpasError::Refusal(
            "simulation start follows the frame valid time".to_string(),
        ));
    }

    // --- dimensions ---------------------------------------------------------
    let mut dims = vec![
        NcDim::record("Time"),
        NcDim::fixed("DateStrLen", 19),
        NcDim::fixed("west_east", nx),
        NcDim::fixed("south_north", ny),
        NcDim::fixed("bottom_top", nz),
        NcDim::fixed("west_east_stag", nx + 1),
        NcDim::fixed("south_north_stag", ny + 1),
        NcDim::fixed("bottom_top_stag", nz + 1),
    ];
    if frame.n_soil_levels > 0 {
        dims.push(NcDim::fixed("soil_layers_stag", frame.n_soil_levels));
    }
    let dimid = |name: &str| -> usize {
        dims.iter()
            .position(|d| d.name == name)
            .expect("dimension declared above")
    };

    // --- global attributes --------------------------------------------------
    let projection = weights.window.projection_attributes()?;
    let mut gattrs: Vec<NcAttr> = vec![
        NcAttr::text("TITLE", options.title.clone()),
        NcAttr::text("SIMULATION_START_DATE", origin.label()),
        NcAttr::text("START_DATE", origin.label()),
        NcAttr::int("WEST-EAST_GRID_DIMENSION", nx as i32 + 1),
        NcAttr::int("SOUTH-NORTH_GRID_DIMENSION", ny as i32 + 1),
        NcAttr::int("BOTTOM-TOP_GRID_DIMENSION", nz as i32 + 1),
        NcAttr::text("GRIDTYPE", "C"),
        NcAttr::text("MMINLU", MMINLU),
        NcAttr::int("ISWATER", 17),
        NcAttr::int("ISLAKE", 21),
        NcAttr::int("ISICE", 15),
        NcAttr::int("ISURBAN", 13),
        NcAttr::int("GRID_ID", options.grid_id),
        NcAttr::int("PARENT_ID", 0),
        NcAttr::int("I_PARENT_START", 1),
        NcAttr::int("J_PARENT_START", 1),
        NcAttr::int("PARENT_GRID_RATIO", 1),
    ];
    for (name, value) in &projection {
        gattrs.push(match value {
            ProjAttr::Int(v) => NcAttr::int(name.clone(), *v),
            ProjAttr::Float(v) => NcAttr::float(name.clone(), *v as f32),
            ProjAttr::Text(v) => NcAttr::text(name.clone(), v.clone()),
        });
    }
    let file_name = |p: Option<&PathBuf>| -> String {
        p.and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string()
    };
    gattrs.extend([
        NcAttr::text("MPAS_BRIDGE_SCHEMA", BRIDGE_SCHEMA),
        NcAttr::text("MPAS_SOURCE", "MPAS-A v8.4.1 full-physics CUDA port"),
        NcAttr::text(
            "MPAS_HISTORY_FILE",
            file_name(Some(&frame.history_path)),
        ),
        NcAttr::text("MPAS_HISTORY_SHA256", frame.history_sha256.clone()),
        NcAttr::text("MPAS_MESH_SHA256", weights.mesh_sha256.clone()),
        NcAttr::text(
            "MPAS_MESH_FILE",
            file_name(Some(&PathBuf::from(&weights.mesh_path))),
        ),
        NcAttr::text(
            "MPAS_INIT_SHA256",
            frame.init_sha256.clone().unwrap_or_default(),
        ),
        NcAttr::text("MPAS_INIT_FILE", file_name(frame.init_path.as_ref())),
        NcAttr::text("MPAS_WEIGHTS_SHA256", weights.weights_sha256.clone()),
        NcAttr::text("MPAS_WEIGHTS_SCHEMA", crate::weights::WEIGHTS_SCHEMA),
        NcAttr::text("MPAS_RESAMPLE_METHOD", crate::weights::RESAMPLE_METHOD),
        NcAttr::text("MPAS_MESH_ANGLE_UNITS", weights.mesh_angle_units.label()),
        NcAttr::text("MPAS_RENDER_WINDOW", weights.window_name.clone()),
        NcAttr::text("MPAS_RENDER_WINDOW_SPEC", weights.window.spec_json()),
        NcAttr::int("MPAS_NATIVE_NCELLS", frame.n_cells as i32),
        NcAttr::float("MPAS_MAX_NEAREST_CELL_KM", weights.max_distance_km()),
        NcAttr::text("MPAS_FIELD_SET", options.field_set.clone()),
        NcAttr::text(
            "MPAS_WIND_FRAME",
            "U, V, U10 and V10 are earth-relative; SINALPHA=0 and COSALPHA=1 declare no grid rotation so the renderer's grid-to-earth rotation is the identity",
        ),
        NcAttr::text(
            "MPAS_NONCLAIM",
            "nearest-cell visualization resample; not a conservative remap and not a forecast-skill claim",
        ),
        NcAttr::text(
            "MPAS_BRIDGE_ENGINE",
            concat!("rw-mpas ", env!("CARGO_PKG_VERSION"), " (rust)"),
        ),
    ]);

    // --- variable schedule --------------------------------------------------
    // Resolved once so the writer gets the whole schema up front, then walked
    // a second time to push the data. `plan` holds what each variable is; the
    // values are produced on the second pass so no frame is held twice.
    struct Planned {
        name: &'static str,
        dims: Vec<usize>,
        mapping: Option<usize>,
        stagger: &'static str,
        rank_is_3d: bool,
        units: &'static str,
        description: &'static str,
    }

    let mut plan: Vec<Planned> = Vec::new();
    let mut absent: Vec<String> = frame.absent.clone();

    for (index, mapping) in FIELD_MAP.iter().enumerate() {
        if !field_set_allows(&options.field_set, mapping) {
            continue;
        }
        if mapping.source != FieldSource::Constant && !frame.fields.contains_key(mapping.wrf_name) {
            if !absent.iter().any(|n| n == mapping.wrf_name) {
                absent.push(mapping.wrf_name.to_string());
            }
            continue;
        }
        if mapping.rank == Rank::Soil && frame.n_soil_levels == 0 {
            if !absent.iter().any(|n| n == mapping.wrf_name) {
                absent.push(mapping.wrf_name.to_string());
            }
            continue;
        }
        let (dim_names, stagger, rank_is_3d): (Vec<&str>, &'static str, bool) = match mapping.rank {
            Rank::Mass3d => (
                vec!["Time", "bottom_top", "south_north", "west_east"],
                "",
                true,
            ),
            Rank::W3d => (
                vec!["Time", "bottom_top_stag", "south_north", "west_east"],
                "Z",
                true,
            ),
            Rank::U3d => (
                vec!["Time", "bottom_top", "south_north", "west_east_stag"],
                "X",
                true,
            ),
            Rank::V3d => (
                vec!["Time", "bottom_top", "south_north_stag", "west_east"],
                "Y",
                true,
            ),
            Rank::Soil => (
                vec!["Time", "soil_layers_stag", "south_north", "west_east"],
                "Z",
                true,
            ),
            Rank::Surface => (vec!["Time", "south_north", "west_east"], "", false),
        };
        plan.push(Planned {
            name: mapping.wrf_name,
            dims: dim_names.iter().map(|n| dimid(n)).collect(),
            mapping: Some(index),
            stagger,
            rank_is_3d,
            units: mapping.units,
            description: mapping.description,
        });
    }

    let mut vars: Vec<NcVarDef> = vec![
        NcVarDef::new(
            "Times",
            NcType::Char,
            vec![dimid("Time"), dimid("DateStrLen")],
        ),
        NcVarDef::new(
            "XLAT",
            NcType::Float,
            vec![dimid("Time"), dimid("south_north"), dimid("west_east")],
        )
        .with_attrs(field_attrs(
            false,
            "LATITUDE, SOUTH IS NEGATIVE",
            "degree_north",
            "",
        )),
        NcVarDef::new(
            "XLONG",
            NcType::Float,
            vec![dimid("Time"), dimid("south_north"), dimid("west_east")],
        )
        .with_attrs(field_attrs(
            false,
            "LONGITUDE, WEST IS NEGATIVE",
            "degree_east",
            "",
        )),
    ];
    for planned in &plan {
        vars.push(
            NcVarDef::new(planned.name, NcType::Float, planned.dims.clone()).with_attrs(
                field_attrs(
                    planned.rank_is_3d,
                    planned.description,
                    planned.units,
                    planned.stagger,
                ),
            ),
        );
    }
    let since = format!(
        "minutes since {:04}-{:02}-{:02} {:02}:{:02}:{:02}",
        origin.year, origin.month, origin.day, origin.hour, origin.minute, origin.second
    );
    vars.push(
        NcVarDef::new("XTIME", NcType::Float, vec![dimid("Time")]).with_attrs(vec![
            NcAttr::int("FieldType", 104),
            NcAttr::text("MemoryOrder", "0  "),
            NcAttr::text("description", since.clone()),
            NcAttr::text("units", since),
            NcAttr::text("stagger", ""),
        ]),
    );
    vars.push(
        NcVarDef::new("ITIMESTEP", NcType::Int, vec![dimid("Time")]).with_attrs(vec![
            NcAttr::int("FieldType", 106),
            NcAttr::text("MemoryOrder", "0  "),
            NcAttr::text("description", ""),
            NcAttr::text("units", ""),
            NcAttr::text("stagger", ""),
        ]),
    );

    // --- write --------------------------------------------------------------
    let temporary = destination.with_file_name(format!(
        ".{}.tmp",
        destination
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("frame")
    ));
    if temporary.exists() {
        std::fs::remove_file(&temporary)?;
    }
    let mut written: Vec<String> = Vec::new();
    let result = (|| -> MpasResult<()> {
        let mut writer =
            NcClassicWriter::create(&temporary, options.format, dims.clone(), gattrs, vars, 1)?;
        writer.put_record_text("Times", 0, &frame.valid_time.label())?;
        let lat32: Vec<f32> = weights.target_latitude.iter().map(|&v| v as f32).collect();
        let lon32: Vec<f32> = weights.target_longitude.iter().map(|&v| v as f32).collect();
        writer.put_record("XLAT", 0, NcData::Floats(&lat32))?;
        writer.put_record("XLONG", 0, NcData::Floats(&lon32))?;
        written.push("XLAT".into());
        written.push("XLONG".into());

        for planned in &plan {
            let mapping = &FIELD_MAP[planned.mapping.expect("every planned entry maps")];
            let levels = expected_levels(mapping, frame).ok_or_else(|| {
                MpasError::Refusal(format!(
                    "{} has no level count on this frame",
                    mapping.wrf_name
                ))
            })?;
            let mut values: Vec<f32> = match mapping.source {
                FieldSource::Constant => {
                    let points = ny * nx;
                    match mapping.wrf_name {
                        "COSALPHA" => vec![1.0f32; points],
                        "SINALPHA" => vec![0.0f32; points],
                        _ => vec![0.0f32; levels * points],
                    }
                }
                _ => {
                    let (source, source_levels) = frame
                        .fields
                        .get(mapping.wrf_name)
                        .expect("planned entries have their source field");
                    if *source_levels != levels {
                        return Err(MpasError::Refusal(format!(
                            "{} carries {source_levels} level(s); the frame declares {levels}",
                            mapping.wrf_name
                        )));
                    }
                    weights.gather(source, levels)?
                }
            };
            match mapping.wrf_name {
                "T" => values.iter_mut().for_each(|v| *v -= WRF_THETA_REFERENCE_K),
                "PHB" => values.iter_mut().for_each(|v| *v *= WRF_GRAVITY),
                _ => {}
            }
            let staggered = match mapping.rank {
                Rank::U3d => restagger_x(&values, levels, ny, nx),
                Rank::V3d => restagger_y(&values, levels, ny, nx),
                _ => values,
            };
            writer.put_record(mapping.wrf_name, 0, NcData::Floats(&staggered))?;
            written.push(mapping.wrf_name.to_string());
        }

        writer.put_record(
            "XTIME",
            0,
            NcData::Floats(&[(lead_seconds as f64 / 60.0) as f32]),
        )?;
        writer.put_record("ITIMESTEP", 0, NcData::Ints(&[0]))?;
        writer.finish()?;
        Ok(())
    })();
    if let Err(err) = result {
        let _ = std::fs::remove_file(&temporary);
        return Err(err);
    }
    std::fs::rename(&temporary, destination)?;

    let bytes = std::fs::metadata(destination)?.len();
    written.sort();
    written.dedup();
    absent.sort();
    absent.dedup();
    Ok(EmittedFrame {
        path: destination.to_path_buf(),
        window: weights.window_name.clone(),
        shape: (ny, nx),
        levels: nz,
        written,
        absent,
        seconds: started.elapsed().as_secs_f64(),
        bytes,
        valid_time: frame.valid_time,
        simulation_start: origin,
        lead_seconds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn x_restaggering_copies_the_edges_and_averages_the_interior() {
        let values = vec![1.0f32, 3.0, 7.0];
        let out = restagger_x(&values, 1, 1, 3);
        assert_eq!(out, vec![1.0, 2.0, 5.0, 7.0]);
    }

    #[test]
    fn y_restaggering_copies_the_edges_and_averages_the_interior() {
        // two rows, two columns
        let values = vec![1.0f32, 2.0, 5.0, 8.0];
        let out = restagger_y(&values, 1, 2, 2);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 5.0, 5.0, 8.0]);
    }

    #[test]
    fn restaggering_averages_in_f32_not_f64() {
        // 0.1 + 0.2 differs between an f32 add and an f64 add rounded to f32,
        // which is exactly the kind of drift a port introduces by accident.
        let values = vec![0.1f32, 0.2f32];
        let out = restagger_x(&values, 1, 1, 2);
        assert_eq!(out[1].to_bits(), ((0.1f32 + 0.2f32) * 0.5f32).to_bits());
    }

    #[test]
    fn staggered_levels_survive_independently() {
        let values = vec![1.0f32, 3.0, 10.0, 30.0];
        let out = restagger_x(&values, 2, 1, 2);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0]);
    }
}
