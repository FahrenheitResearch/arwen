//! Writing the CDF-5 `lbc` stream file.
//!
//! ## Schema by table, not by donor
//! Unlike the init emitter, which re-emits a native capsule's container, the
//! lbc container is small and fixed by the v8.4.1 registry (`lbc` stream,
//! `lbcs` package: the three `moist` scalars, `lbc_u`, `lbc_w`, `lbc_rho`,
//! `lbc_theta`, `xtime`, `Time`), so its dimensions, variables and attributes
//! are declared here as a table.  There is no donor file to make the
//! container trivially right; what this writer emits is checkable against a
//! native file only because the table was transcribed from the registry and
//! measured against the reference files.
//!
//! ## Header identity
//! Global attributes reproduce the native header field-for-field, from three
//! sources:
//! * **carried** — mesh and build lineage (`on_a_sphere` .. `mesh_spec`,
//!   `git_version`, the `model_name`/`core_name`/`version`/`source`/
//!   `Conventions` block) read from the grid/init file this run consumed,
//!   the same precedent `init::emit` set;
//! * **stamped** — `config_*` values: the producer's own switches where a
//!   switch exists, the v8.4.1 registry defaults otherwise, overridable per
//!   name by caller-supplied table data (`--config-attrs`);
//! * **owned** — `history` names this producer truthfully (never the native
//!   command line), `parent_id` is the consumed grid file's `file_id`, and
//!   `file_id` is minted deterministically from the run's inputs.
//!
//! `history`, `file_id` and the trailing `gpuwm_provenance` are therefore the
//! only header fields expected to differ from a native file produced from the
//! same inputs; the compare module reports exactly which fields differ so
//! that claim is measured, not asserted.
//!
//! ## Layout identity
//! The header is zero-padded so the data section starts at the same offset
//! SMIOL reserves ([`SMIOL_HEADER_RESERVE_BYTES`]), making the data sections
//! of a produced file and a native file comparable byte-for-byte at equal
//! offsets.

use std::collections::BTreeMap;
use std::path::Path;

use rw_store::netcdf_classic::{
    NcAttr, NcAttrValue, NcClassicWriter, NcData, NcDim, NcFormat, NcType, NcVarDef,
};

use crate::error::{MpasError, MpasResult};
use crate::lbc::LbcConfig;

/// The header space SMIOL reserves ahead of the data section, measured on the
/// native v8.4.1 `init_atmosphere_model` lbc and init writes (data begins at
/// 0x20000 in both).  Reproduced so data offsets line up file-to-file.
pub const SMIOL_HEADER_RESERVE_BYTES: u64 = 131_072;

/// `lbc.$Y-$M-$D_$h.$m.$s.nc` from a `YYYY-MM-DD_HH:MM:SS` stamp — the
/// stream's own filename template.
pub fn lbc_file_name(valid_time: &str) -> String {
    format!("lbc.{}.nc", valid_time.replace(':', "."))
}

/// Identity carried out of the grid/init file: the lineage attributes and the
/// parent's `file_id`.
#[derive(Debug)]
pub struct HeaderSource {
    carried: Vec<NcAttr>,
    pub parent_file_id: String,
}

/// The carried attribute names, in the order the native header emits them.
const CARRIED_LEADING: [&str; 6] = [
    "model_name",
    "core_name",
    "version",
    "source",
    "Conventions",
    "git_version",
];
const CARRIED_MESH: [&str; 5] = [
    "on_a_sphere",
    "sphere_radius",
    "is_periodic",
    "x_period",
    "y_period",
];

fn carry_attr(file: &netcrust::File, name: &str, from: &Path) -> MpasResult<NcAttr> {
    let attr = file.attribute(name).ok_or_else(|| {
        MpasError::Refusal(format!(
            "{} carries no global attribute {name}; the lbc header carries the grid file's \
             lineage, and inventing a value here would stamp the stream with an identity its \
             mesh never had",
            from.display()
        ))
    })?;
    let value = match attr.value() {
        netcrust::AttributeValue::Chars(s) => NcAttrValue::Text(s.clone()),
        netcrust::AttributeValue::Strings(v) if v.len() == 1 => NcAttrValue::Text(v[0].clone()),
        netcrust::AttributeValue::Ints(v) => NcAttrValue::Ints(v.clone()),
        netcrust::AttributeValue::Floats(v) => NcAttrValue::Floats(v.clone()),
        netcrust::AttributeValue::Doubles(v) => NcAttrValue::Doubles(v.clone()),
        other => {
            return Err(MpasError::Refusal(format!(
                "global attribute {name} in {} has a value shape this writer cannot carry: {other:?}",
                from.display()
            )))
        }
    };
    Ok(NcAttr {
        name: name.to_string(),
        value,
    })
}

impl HeaderSource {
    pub fn from_grid(grid_path: &Path) -> MpasResult<HeaderSource> {
        let file = netcrust::File::open(grid_path)?;
        let mut carried = Vec::new();
        for name in CARRIED_LEADING.iter().chain(CARRIED_MESH.iter()) {
            carried.push(carry_attr(&file, name, grid_path)?);
        }
        // mesh_spec sits after parent_id in the native order; carried
        // separately so the assembly below can interleave the owned fields.
        carried.push(carry_attr(&file, "mesh_spec", grid_path)?);
        let parent_file_id = file
            .attribute("file_id")
            .and_then(|a| a.as_string().map(str::to_string))
            .ok_or_else(|| {
                MpasError::Refusal(format!(
                    "{} carries no file_id; parent_id cannot be recorded, and an lbc stream \
                     with no parentage cannot be tied to the init it belongs to",
                    grid_path.display()
                ))
            })?;
        Ok(HeaderSource {
            carried,
            parent_file_id,
        })
    }
}

/// One row of the config-attribute table.
enum ConfigValue {
    /// Producer switch: the value comes from [`LbcConfig`], never from an
    /// override.
    Knob(NcAttrValue),
    /// Namelist metadata: the v8.4.1 registry default, overridable by name.
    Default(NcAttrValue),
}

fn text(v: &str) -> NcAttrValue {
    NcAttrValue::Text(v.to_string())
}
fn int(v: i32) -> NcAttrValue {
    NcAttrValue::Ints(vec![v])
}
fn float(v: f32) -> NcAttrValue {
    NcAttrValue::Floats(vec![v])
}
fn logical(v: bool) -> NcAttrValue {
    text(if v { "YES" } else { "NO" })
}

/// The `config_*` block, in the registry's declaration order — which is the
/// order the native writer stamps it in.  Defaults are the v8.4.1
/// `core_init_atmosphere` Registry.xml `default_value`s, transcribed; rows
/// marked `Knob` take the producer's actual switch instead and refuse an
/// override, because the header must not claim a configuration the run did
/// not use.
pub fn config_attributes(
    cfg: &LbcConfig,
    n_vert_levels: usize,
) -> MpasResult<Vec<NcAttr>> {
    use ConfigValue::{Default, Knob};
    let rows: Vec<(&str, ConfigValue)> = vec![
        ("config_init_case", Knob(int(9))),
        ("config_calendar_type", Default(text("gregorian"))),
        ("config_start_time", Knob(text(&cfg.start_time))),
        ("config_stop_time", Knob(text(&cfg.stop_time))),
        ("config_theta_adv_order", Knob(int(cfg.theta_adv_order))),
        ("config_coef_3rd_order", Knob(float(cfg.coef_3rd_order))),
        ("config_num_halos", Default(int(2))),
        (
            "config_interface_projection",
            Default(text("linear_interpolation")),
        ),
        ("config_nvertlevels", Knob(int(n_vert_levels as i32))),
        ("config_nsoillevels", Default(int(4))),
        ("config_nfglevels", Knob(int(cfg.n_fg_levels as i32))),
        ("config_nfgsoillevels", Default(int(4))),
        ("config_gocartlevels", Default(int(30))),
        ("config_months", Default(int(12))),
        (
            "config_geog_data_path",
            Default(text("/glade/work/wrfhelp/WPS_GEOG/")),
        ),
        ("config_met_prefix", Default(text("CFSR"))),
        ("config_sfc_prefix", Default(text("SST"))),
        (
            "config_fg_interval",
            Knob(int(cfg.fg_interval_seconds as i32)),
        ),
        (
            "config_landuse_data",
            Default(text("MODIFIED_IGBP_MODIS_NOAH")),
        ),
        ("config_soilcat_data", Default(text("STATSGO"))),
        ("config_topo_data", Default(text("GMTED2010"))),
        ("config_vegfrac_data", Default(text("MODIS"))),
        ("config_albedo_data", Default(text("MODIS"))),
        ("config_maxsnowalbedo_data", Default(text("MODIS"))),
        ("config_supersample_factor", Default(int(3))),
        ("config_lu_supersample_factor", Default(int(1))),
        ("config_30s_supersample_factor", Default(int(1))),
        ("config_noahmp_static", Default(logical(true))),
        ("config_use_spechumd", Knob(logical(cfg.use_spechumd))),
        ("config_ztop", Default(float(30000.0))),
        ("config_hybrid_coordinate", Default(logical(true))),
        ("config_hybrid_top_z", Default(float(30000.0))),
        ("config_nsmterrain", Default(int(1))),
        ("config_smooth_surfaces", Default(logical(true))),
        ("config_dzmin", Default(float(0.3))),
        ("config_nsm", Default(int(30))),
        ("config_tc_vertical_grid", Default(logical(true))),
        ("config_specified_zeta_levels", Default(text(""))),
        ("config_blend_bdy_terrain", Default(logical(false))),
        (
            "config_extrap_airtemp",
            Knob(text(cfg.extrap_airtemp.label())),
        ),
        ("config_static_interp", Default(logical(true))),
        ("config_native_gwd_static", Default(logical(true))),
        ("config_native_gwd_gsl_static", Default(logical(false))),
        ("config_gwd_cell_scaling", Default(float(1.0))),
        ("config_vertical_grid", Default(logical(true))),
        ("config_met_interp", Default(logical(true))),
        ("config_input_sst", Default(logical(false))),
        ("config_frac_seaice", Default(logical(true))),
        ("config_tsk_seaice_threshold", Default(float(100.0))),
        ("config_pio_num_iotasks", Default(int(0))),
        ("config_pio_stride", Default(int(1))),
        // The registry's own default string; overridden per run like any
        // other namelist value.
        (
            "config_block_decomp_file_prefix",
            Default(text("x1.40962.graph.info.part.")),
        ),
        ("config_number_of_blocks", Default(int(0))),
        ("config_explicit_proc_decomp", Default(logical(false))),
        (
            "config_proc_decomp_file_prefix",
            Default(text("graph.info.part.")),
        ),
    ];

    let known: BTreeMap<&str, bool> = rows
        .iter()
        .map(|(n, v)| (*n, matches!(v, ConfigValue::Knob(_))))
        .collect();
    for (name, _) in cfg.attr_overrides.iter() {
        match known.get(name.as_str()) {
            None => {
                return Err(MpasError::Refusal(format!(
                    "--config-attrs names {name}, which is not a v8.4.1 init_atmosphere \
                     namelist attribute this table declares; a misspelled name would silently \
                     leave the default in the header"
                )))
            }
            Some(true) => {
                return Err(MpasError::Refusal(format!(
                    "--config-attrs tries to override {name}, which is a producer switch; the \
                     header must record the value the run actually used, so set the switch \
                     itself instead"
                )))
            }
            Some(false) => {}
        }
    }

    let mut out = Vec::with_capacity(rows.len());
    for (name, value) in rows {
        let resolved = match value {
            ConfigValue::Knob(v) => v,
            ConfigValue::Default(v) => match cfg.attr_overrides.get(name) {
                None => v,
                Some(json) => override_value(name, &v, json)?,
            },
        };
        out.push(NcAttr {
            name: name.to_string(),
            value: resolved,
        });
    }
    Ok(out)
}

/// Apply one override, typed by the default it replaces.
fn override_value(
    name: &str,
    default: &NcAttrValue,
    json: &serde_json::Value,
) -> MpasResult<NcAttrValue> {
    let refuse = |wanted: &str| {
        MpasError::Refusal(format!(
            "--config-attrs value for {name} is {json}; this attribute takes {wanted}"
        ))
    };
    Ok(match default {
        NcAttrValue::Text(current) => {
            // Logical attributes are stored as YES/NO text; accept JSON
            // booleans for them and plain strings for the rest.
            if let Some(b) = json.as_bool() {
                if current == "YES" || current == "NO" {
                    NcAttrValue::Text(if b { "YES" } else { "NO" }.to_string())
                } else {
                    return Err(refuse("a string"));
                }
            } else {
                NcAttrValue::Text(
                    json.as_str().ok_or_else(|| refuse("a string"))?.to_string(),
                )
            }
        }
        NcAttrValue::Ints(_) => NcAttrValue::Ints(vec![json
            .as_i64()
            .ok_or_else(|| refuse("an integer"))? as i32]),
        NcAttrValue::Floats(_) => NcAttrValue::Floats(vec![json
            .as_f64()
            .ok_or_else(|| refuse("a number"))?
            as f32]),
        other => {
            return Err(MpasError::Refusal(format!(
                "attribute {name} has an unoverridable default shape {other:?}"
            )))
        }
    })
}

/// The seven field slabs of one boundary time, flat in file order (point
/// major, vertical fastest).
pub struct LbcFields<'a> {
    pub qv: &'a [f32],
    pub qc: &'a [f32],
    pub qr: &'a [f32],
    pub u: &'a [f32],
    pub w: &'a [f32],
    pub rho: &'a [f32],
    pub theta: &'a [f32],
}

/// Mint the deterministic per-file id, same alphabet and length as the init
/// emitter's.
fn mint_file_id(seed: &str) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(seed.as_bytes());
    const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyz";
    digest
        .iter()
        .take(10)
        .map(|b| ALPHABET[(*b as usize) % ALPHABET.len()] as char)
        .collect()
}

/// Write one boundary time's file.
#[allow(clippy::too_many_arguments)]
pub fn write_lbc_file(
    out_path: &Path,
    cfg: &LbcConfig,
    header: &HeaderSource,
    n_cells: usize,
    n_edges: usize,
    nz: usize,
    valid_time: &str,
    time_seconds: f32,
    fields: &LbcFields<'_>,
) -> MpasResult<()> {
    let nzp1 = nz + 1;
    let check = |name: &str, got: usize, want: usize| -> MpasResult<()> {
        if got != want {
            return Err(MpasError::Refusal(format!(
                "computed {name} holds {got} value(s); the file's slab is {want}.  A mismatched \
                 count here is the cell-major/level-major transpose, which produces a readable \
                 file full of wrong numbers"
            )));
        }
        Ok(())
    };
    check("lbc_qv", fields.qv.len(), n_cells * nz)?;
    check("lbc_qc", fields.qc.len(), n_cells * nz)?;
    check("lbc_qr", fields.qr.len(), n_cells * nz)?;
    check("lbc_u", fields.u.len(), n_edges * nz)?;
    check("lbc_w", fields.w.len(), n_cells * nzp1)?;
    check("lbc_rho", fields.rho.len(), n_cells * nz)?;
    check("lbc_theta", fields.theta.len(), n_cells * nz)?;

    // Dimensions in the native file's order.
    let dims = vec![
        NcDim::fixed("nVertLevels", nz),
        NcDim::fixed("nCells", n_cells),
        NcDim::record("Time"),
        NcDim::fixed("nEdges", n_edges),
        NcDim::fixed("nVertLevelsP1", nzp1),
        NcDim::fixed("StrLen", 64),
    ];
    const D_NZ: usize = 0;
    const D_CELLS: usize = 1;
    const D_TIME: usize = 2;
    const D_EDGES: usize = 3;
    const D_NZP1: usize = 4;
    const D_STRLEN: usize = 5;

    let cell_var = |name: &str, units: &str, long_name: &str| -> NcVarDef {
        NcVarDef::new(name, NcType::Float, vec![D_TIME, D_CELLS, D_NZ]).with_attrs(vec![
            NcAttr::text("units", units),
            NcAttr::text("long_name", long_name),
        ])
    };
    let vars = vec![
        cell_var(
            "lbc_qv",
            "kg kg^{-1}",
            "Water vapor mixing ratio on lateral boundary cells",
        ),
        cell_var(
            "lbc_qc",
            "kg kg^{-1}",
            "Cloud water mixing ratio on lateral boundary cells",
        ),
        cell_var(
            "lbc_qr",
            "kg kg^{-1}",
            "Rain water mixing ratio on lateral boundary cells",
        ),
        NcVarDef::new("lbc_u", NcType::Float, vec![D_TIME, D_EDGES, D_NZ]).with_attrs(vec![
            NcAttr::text("units", "m s^{-1}"),
            NcAttr::text(
                "long_name",
                "Horizontal normal velocity on domain lateral boundary edges",
            ),
        ]),
        NcVarDef::new("lbc_w", NcType::Float, vec![D_TIME, D_CELLS, D_NZP1]).with_attrs(vec![
            NcAttr::text("units", "m s^{-1}"),
            NcAttr::text(
                "long_name",
                "Vertical velocity on domain lateral boundary vertical cell faces",
            ),
        ]),
        cell_var(
            "lbc_rho",
            "kg m^{-3}",
            "Dry air density on lateral boundary cells",
        ),
        cell_var(
            "lbc_theta",
            "K",
            "Potential temperature on lateral boundary cells",
        ),
        NcVarDef::new("xtime", NcType::Char, vec![D_TIME, D_STRLEN]).with_attrs(vec![
            NcAttr::text("units", "YYYY-MM-DD_hh:mm:ss"),
            NcAttr::text("long_name", "Model valid time"),
        ]),
        NcVarDef::new("Time", NcType::Float, vec![D_TIME]).with_attrs(vec![
            NcAttr::text(
                "units",
                format!("seconds since {}", space_separated(&cfg.start_time)?),
            ),
            NcAttr::text("long_name", "CF-compliant valid time"),
            NcAttr::text("standard_name", "time"),
        ]),
    ];

    // Global attributes: carried lineage, owned identity, stamped config.
    let mut gattrs: Vec<NcAttr> = Vec::new();
    // model_name .. y_period.
    for a in header.carried.iter().take(CARRIED_LEADING.len() + CARRIED_MESH.len()) {
        gattrs.push(a.clone());
    }
    gattrs.push(NcAttr::text("history", "rw_mpas_lbc"));
    gattrs.push(NcAttr::text(
        "parent_id",
        format!("{}\n", header.parent_file_id),
    ));
    // mesh_spec, carried last.
    gattrs.push(
        header
            .carried
            .last()
            .expect("HeaderSource always carries mesh_spec")
            .clone(),
    );
    gattrs.extend(config_attributes(cfg, nz)?);
    let seed = format!(
        "{}|{}|{}",
        cfg.grid_path.display(),
        cfg.start_time,
        valid_time
    );
    gattrs.push(NcAttr::text("file_id", mint_file_id(&seed)));
    gattrs.push(NcAttr::text("gpuwm_provenance", cfg.provenance.clone()));

    let mut writer = NcClassicWriter::create_with_min_header(
        out_path,
        NcFormat::Data64,
        dims,
        gattrs,
        vars,
        1,
        SMIOL_HEADER_RESERVE_BYTES,
    )
    .map_err(|e| MpasError::Refusal(format!("cannot lay out {}: {e}", out_path.display())))?;

    let put = |writer: &mut NcClassicWriter, name: &str, data: &[f32]| -> MpasResult<()> {
        writer
            .put_record(name, 0, NcData::Floats(data))
            .map_err(|e| MpasError::Refusal(format!("cannot write {name}: {e}")))
    };
    put(&mut writer, "lbc_qv", fields.qv)?;
    put(&mut writer, "lbc_qc", fields.qc)?;
    put(&mut writer, "lbc_qr", fields.qr)?;
    put(&mut writer, "lbc_u", fields.u)?;
    put(&mut writer, "lbc_w", fields.w)?;
    put(&mut writer, "lbc_rho", fields.rho)?;
    put(&mut writer, "lbc_theta", fields.theta)?;

    // xtime is space-padded to StrLen, the way the native stream writes it.
    let mut xtime = valid_time.as_bytes().to_vec();
    if xtime.len() > 64 {
        return Err(MpasError::Refusal(format!(
            "valid time '{valid_time}' is {} byte(s); xtime holds 64",
            xtime.len()
        )));
    }
    xtime.resize(64, b' ');
    writer
        .put_record("xtime", 0, NcData::Chars(&xtime))
        .map_err(|e| MpasError::Refusal(format!("cannot write xtime: {e}")))?;
    writer
        .put_record("Time", 0, NcData::Floats(&[time_seconds]))
        .map_err(|e| MpasError::Refusal(format!("cannot write Time: {e}")))?;

    writer
        .finish()
        .map_err(|e| MpasError::Refusal(format!("cannot finish {}: {e}", out_path.display())))?;
    Ok(())
}

/// `YYYY-MM-DD_HH:MM:SS` to the `YYYY-MM-DD HH:MM:SS` form the `Time`
/// units string uses.
fn space_separated(stamp: &str) -> MpasResult<String> {
    if stamp.len() < 19 || stamp.as_bytes().get(10) != Some(&b'_') {
        return Err(MpasError::Refusal(format!(
            "start time '{stamp}' is not YYYY-MM-DD_HH:MM:SS"
        )));
    }
    let mut s = stamp.to_string();
    s.replace_range(10..11, " ");
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_file_name_follows_the_stream_template() {
        assert_eq!(
            lbc_file_name("2026-08-12_06:00:00"),
            "lbc.2026-08-12_06.00.00.nc"
        );
    }

    #[test]
    fn the_time_units_epoch_swaps_the_underscore_for_a_space() {
        assert_eq!(
            space_separated("2026-08-12_06:00:00").unwrap(),
            "2026-08-12 06:00:00"
        );
        assert!(space_separated("junk").is_err());
    }

    #[test]
    fn a_file_id_is_deterministic_in_the_inputs() {
        let a = mint_file_id("g|2026-08-12_06:00:00|t");
        let b = mint_file_id("g|2026-08-12_06:00:00|t");
        let c = mint_file_id("g|2026-08-12_09:00:00|t");
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(a.len(), 10);
    }

    fn config_for_test() -> LbcConfig {
        use crate::init::hinterp::Underflow;
        use crate::init::vinterp::Extrap;
        LbcConfig {
            grid_path: "grid.nc".into(),
            out_dir: ".".into(),
            start_time: "2026-08-12_06:00:00".to_string(),
            stop_time: "2026-08-12_12:00:00".to_string(),
            intervals: vec![],
            n_fg_levels: 34,
            extrap_airtemp: Extrap::LapseRate,
            use_spechumd: false,
            theta_adv_order: 3,
            coef_3rd_order: 0.25,
            fg_interval_seconds: 10800,
            oned_underflow: Underflow::ReproduceIfxFtz,
            attr_overrides: Default::default(),
            provenance: "test".to_string(),
        }
    }

    #[test]
    fn config_attributes_take_switches_from_the_run_not_from_overrides() {
        let cfg = config_for_test();
        let attrs = config_attributes(&cfg, 55).unwrap();
        let by_name: BTreeMap<&str, &NcAttrValue> =
            attrs.iter().map(|a| (a.name.as_str(), &a.value)).collect();
        assert!(matches!(by_name["config_init_case"], NcAttrValue::Ints(v) if v == &vec![9]));
        assert!(
            matches!(by_name["config_extrap_airtemp"], NcAttrValue::Text(t) if t == "lapse-rate")
        );
        assert!(matches!(by_name["config_use_spechumd"], NcAttrValue::Text(t) if t == "NO"));
        assert!(matches!(by_name["config_nvertlevels"], NcAttrValue::Ints(v) if v == &vec![55]));
        // Registry order is preserved: init_case first, proc_decomp last.
        assert_eq!(attrs.first().unwrap().name, "config_init_case");
        assert_eq!(attrs.last().unwrap().name, "config_proc_decomp_file_prefix");
    }

    #[test]
    fn an_override_may_not_shadow_a_switch_and_must_name_a_real_attribute() {
        let mut cfg = config_for_test();
        cfg.attr_overrides
            .insert("config_extrap_airtemp".to_string(), "linear".into());
        let err = config_attributes(&cfg, 55).unwrap_err().to_string();
        assert!(err.contains("producer switch"), "{err}");

        let mut cfg = config_for_test();
        cfg.attr_overrides
            .insert("config_extrap_temp".to_string(), "linear".into());
        let err = config_attributes(&cfg, 55).unwrap_err().to_string();
        assert!(err.contains("config_extrap_temp"), "{err}");
    }

    #[test]
    fn an_override_lands_with_the_declared_type() {
        let mut cfg = config_for_test();
        cfg.attr_overrides
            .insert("config_met_prefix".to_string(), "MET".into());
        cfg.attr_overrides
            .insert("config_supersample_factor".to_string(), 1i32.into());
        cfg.attr_overrides
            .insert("config_blend_bdy_terrain".to_string(), true.into());
        let attrs = config_attributes(&cfg, 55).unwrap();
        let by_name: BTreeMap<&str, &NcAttrValue> =
            attrs.iter().map(|a| (a.name.as_str(), &a.value)).collect();
        assert!(matches!(by_name["config_met_prefix"], NcAttrValue::Text(t) if t == "MET"));
        assert!(
            matches!(by_name["config_supersample_factor"], NcAttrValue::Ints(v) if v == &vec![1])
        );
        assert!(matches!(by_name["config_blend_bdy_terrain"], NcAttrValue::Text(t) if t == "YES"));

        let mut cfg = config_for_test();
        cfg.attr_overrides
            .insert("config_supersample_factor".to_string(), "one".into());
        assert!(config_attributes(&cfg, 55).is_err());
    }
}
