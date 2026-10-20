// SPDX-License-Identifier: Apache-2.0
// NOAA-OWP/cfe f9182dfb3d81c407c66c5b782114173893c30cf2 initialization,
// adapted from bmi_cfe.c; full grant and NOAA TERMS: NOTICE-CFE.txt.
// Deterministic defined zero state replaces upstream uninitialized allocation.
// Config reads, initialization and output I/O/hashing are parallel Rust.
use nextgen_replay::{file_sha256, sealed_bytes, Fallible, Seal};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};
const PIN: &str = "f9182dfb3d81c407c66c5b782114173893c30cf2";
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Column {
    id: String,
    config: Seal,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema: String,
    time_origin: String,
    columns: Vec<Column>,
    output: PathBuf,
}
#[derive(Deserialize)]
struct Layout {
    fields: Vec<String>,
    options: Vec<String>,
    offsets: Vec<String>,
}
struct Config(BTreeMap<String, String>);
impl Config {
    fn number(&self, keys: &[&str], default: Option<f64>) -> Fallible<f64> {
        let value = keys.iter().find_map(|k| self.0.get(*k));
        let Some(text) = value else {
            return default.ok_or(format!("missing parameter {}", keys[0]));
        };
        let n: f64 = text
            .parse()
            .map_err(|_| format!("{}: invalid number {text}", keys[0]))?;
        if !n.is_finite() {
            return Err(format!("{}: nonfinite parameter", keys[0]));
        }
        Ok(n)
    }
    fn flag(&self, key: &str) -> Fallible<bool> {
        match self.0.get(key).map(String::as_str) {
            None | Some("false" | "False" | "0") => Ok(false),
            Some("true" | "True" | "1") => Ok(true),
            _ => Err(format!("invalid boolean {key}")),
        }
    }
    fn vector(&self, key: &str) -> Fallible<Option<Vec<f64>>> {
        self.0
            .get(key)
            .map(|v| {
                v.split(',')
                    .map(|x| {
                        let y = x
                            .trim()
                            .parse::<f64>()
                            .map_err(|_| format!("bad vector {key}"))?;
                        if y.is_finite() {
                            Ok(y)
                        } else {
                            Err(format!("nonfinite {key}"))
                        }
                    })
                    .collect()
            })
            .transpose()
    }
    fn count(&self, key: &str, default: usize) -> Fallible<usize> {
        let n = self.number(&[key], Some(default as f64))?;
        if n < 0.0 || n > i32::MAX as f64 || n.fract() != 0.0 {
            Err(format!("invalid count {key}"))
        } else {
            Ok(n as usize)
        }
    }
    fn scheme(&self, keys: &[&str], default: Option<i32>, names: &[(&str, i32)]) -> Fallible<i32> {
        let Some(v) = keys.iter().find_map(|k| self.0.get(*k)) else {
            return default.ok_or(format!("missing {}", keys[0]));
        };
        names
            .iter()
            .find_map(|(name, n)| v.eq_ignore_ascii_case(name).then_some(*n))
            .ok_or(format!("invalid {}: {v}", keys[0]))
    }
}
struct State {
    f: BTreeMap<String, f64>,
    i: BTreeMap<String, i32>,
    vectors: BTreeMap<String, Vec<f64>>,
    raw_initial: bool,
}
impl State {
    fn f(&mut self, key: &str, n: f64) {
        self.f.insert(key.into(), n);
    }
    fn i(&mut self, key: &str, n: i32) {
        self.i.insert(key.into(), n);
    }
}
// Fresh initialization, not restart: preserve supplied physical stores without
// the lossy storage/max -> fraction -> storage round trip. These keys belong
// to this Rust preparer, not to the native CFE INI parser.
const RAW_INITIAL_KEYS: [&str; 5] = [
    "initial_soil_storage_m",
    "initial_gw_storage_m",
    "initial_nash_subsurface_m",
    "initial_giuh_queue_m",
    "initial_flux_perc_m",
];
fn apply_raw_initial(c: &Config, s: &mut State) -> Fallible<bool> {
    let supplied = RAW_INITIAL_KEYS
        .iter()
        .filter(|k| c.0.contains_key(**k))
        .count();
    if supplied == 0 {
        return Ok(false);
    }
    if supplied != RAW_INITIAL_KEYS.len() {
        return Err("raw initial water requires all five initial_* keys; partial state would lose carried water".into());
    }
    if s.i["surface_runoff_scheme"] != 1
        || s.i["soil_reservoir.is_sft_coupled"] != 0
        || s.i["soil_reservoir.is_aet_rootzone"] != 0
    {
        return Err("raw initial water lacks surface-Nash, frozen-water and rootzone-profile state; supply a complete checkpoint for those formulations".into());
    }
    let soil = c.number(&[RAW_INITIAL_KEYS[0]], None)?;
    let gw = c.number(&[RAW_INITIAL_KEYS[1]], None)?;
    let perc = c.number(&[RAW_INITIAL_KEYS[4]], None)?;
    let nash = c.vector(RAW_INITIAL_KEYS[2])?.unwrap();
    let mut queue = c.vector(RAW_INITIAL_KEYS[3])?.unwrap();
    if !(0.0..=s.f["soil_reservoir.storage_max_m"]).contains(&soil)
        || !(0.0..=s.f["gw_reservoir.storage_max_m"]).contains(&gw)
        || perc < 0.0
        || nash.iter().chain(&queue).any(|v| *v < 0.0)
    {
        return Err("raw initial water is negative or exceeds reservoir capacity".into());
    }
    if nash.len() != s.i["N_nash_subsurface"] as usize
        || queue.len() != s.i["num_giuh_ordinates"] as usize
    {
        return Err("raw initial ragged counts differ from configured Nash/GIUH counts (GIUH excludes sentinel)".into());
    }
    if !(((gw + soil) + nash.iter().sum::<f64>()) + queue.iter().sum::<f64>()).is_finite() {
        return Err("raw initial water inventory overflows f64".into());
    }
    queue.push(0.0); // Native extra slot is not an additional water store.
    s.f("soil_reservoir.storage_m", soil);
    s.f("gw_reservoir.storage_m", gw);
    s.f("flux_perc_m", perc);
    s.vectors.insert("nash_subsurface".into(), nash);
    s.vectors.insert("runoff_queue".into(), queue);
    Ok(true)
}
fn initial_inventory(s: &State) -> Value {
    let soil = s.f["soil_reservoir.storage_m"];
    let gw = s.f["gw_reservoir.storage_m"];
    let nash: f64 = s.vectors["nash_subsurface"].iter().sum();
    let surface: f64 = if s.i["surface_runoff_scheme"] == 1 {
        s.vectors["runoff_queue"][..s.i["num_giuh_ordinates"] as usize]
            .iter()
            .sum()
    } else {
        s.vectors["nash_surface"].iter().sum()
    };
    json!({"soil_m":soil,"groundwater_m":gw,"nash_subsurface_m":nash,
        "storage_source":if s.raw_initial {"raw initial_* keys"} else {"native fractional configuration"},
        "surface_held_m":surface,"total_held_m":((gw+soil)+nash)+surface,
        "carried_percolation_m":s.f.get("flux_perc_m").copied().unwrap_or(0.0),
        "percolation_is_additional_storage":false,
        "nash_count":s.i["N_nash_subsurface"],"giuh_count":s.i["num_giuh_ordinates"],
        "giuh_extra_sentinel_m":0.0,
        "native_budget_start_m":gw+soil,
        "native_budget_domain":"soil + groundwater; lateral outflow counted before Nash",
        "surface_budget_initial_offset_m":surface,"nash_budget_initial_offset_m":nash,
        "diagnostics":"native fresh counters reset; initial surface/Nash offsets supplied separately; not restart",
        "initial_step":0,"current_time_step":0})
}
fn initialize(column: &Column) -> Fallible<State> {
    let bytes = sealed_bytes(&column.config)?;
    let text = std::str::from_utf8(&bytes).map_err(|e| e.to_string())?;
    let mut kv = BTreeMap::new();
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (key, value) = line
            .split_once('=')
            .ok_or(format!("{}: config line lacks '='", column.id))?;
        // Check before discarding units: e.g. 0.5[%] must never become a
        // 0.5 storage fraction. The pinned native percent parser is defective.
        if value.contains('%') {
            return Err(format!(
                "{}: {}: use an explicit fractional value, not percent",
                column.id,
                key.trim()
            ));
        }
        let value = value.split('[').next().unwrap().trim().to_owned();
        if kv.insert(key.trim().to_owned(), value).is_some() {
            return Err(format!("{}: duplicate config key {key}", column.id));
        }
    }
    let c = Config(kv);
    if c.0.get("forcing_file").map(String::as_str) != Some("BMI") {
        return Err("CFE uses resident forcing; forcing_file must be BMI".into());
    }
    let mut s = State {
        f: BTreeMap::new(),
        i: BTreeMap::new(),
        vectors: BTreeMap::new(),
        raw_initial: false,
    };
    let p = [
        ("D", vec!["soil_params.D", "soil_params.depth"]),
        ("bb", vec!["soil_params.bb", "soil_params.b"]),
        ("satdk", vec!["soil_params.satdk"]),
        ("satpsi", vec!["soil_params.satpsi"]),
        ("slop", vec!["soil_params.slop", "soil_params.slope"]),
        ("smcmax", vec!["soil_params.smcmax", "soil_params.maxsmc"]),
        ("wltsmc", vec!["soil_params.wltsmc"]),
        ("alpha_fc", vec!["soil_params.alpha_fc", "alpha_fc"]),
        ("soil_storage", vec!["soil_storage"]),
    ];
    for (name, keys) in &p {
        s.f(&format!("NWM_soil_params.{name}"), c.number(keys, None)?);
    }
    s.f(
        "NWM_soil_params.refkdt",
        c.number(&["soil_params.refkdt", "refkdt"], Some(3.0))?,
    );
    // Native CFE never reads soil_params.mult, despite that example setting.
    let d = s.f["NWM_soil_params.D"];
    let b = s.f["NWM_soil_params.bb"];
    let satdk = s.f["NWM_soil_params.satdk"];
    let psi = s.f["NWM_soil_params.satpsi"];
    let sm = s.f["NWM_soil_params.smcmax"];
    let wl = s.f["NWM_soil_params.wltsmc"];
    let a = s.f["NWM_soil_params.alpha_fc"];
    let soil_fraction = s.f["NWM_soil_params.soil_storage"].max(0.0);
    if !(d > 0.0
        && b > 0.0
        && b != 1.0
        && psi > 0.0
        && sm > 0.0
        && wl >= 0.0
        && wl < sm
        && soil_fraction <= 1.0)
    {
        return Err(format!("{}: invalid soil parameters/fraction", column.id));
    }
    s.f("NWM_soil_params.soil_storage", soil_fraction);
    s.f("time_step_size", 3600.0);
    s.f("timestep_h", 1.0);
    s.f("NWM_soil_params.wilting_point_m", wl * d);
    let omega = (a * 101325.0 / 9810.0) - 0.5;
    if omega <= 0.0 {
        return Err("field capacity parameter gives nonpositive Omega".into());
    }
    let power = 1.0 - 1.0 / b;
    let lower = omega.powf(power) / power;
    let upper = (omega + d).powf(power) / power;
    let threshold = sm * (1.0 / psi).powf(-1.0 / b) * (upper - lower);
    s.f("soil_reservoir.storage_max_m", sm * d);
    s.f("soil_reservoir.storage_m", soil_fraction * sm * d);
    s.f(
        "soil_reservoir.coeff_primary",
        satdk * s.f["NWM_soil_params.slop"] * 3600.0,
    );
    s.f("soil_reservoir.coeff_secondary", c.number(&["K_lf"], None)?);
    s.f(
        "soil_reservoir.exponent_primary",
        c.number(&["soil_params.expon"], Some(1.0))?,
    );
    s.f(
        "soil_reservoir.exponent_secondary",
        c.number(&["soil_params.expon_secondary"], Some(1.0))?,
    );
    s.f("soil_reservoir.storage_threshold_primary_m", threshold);
    s.f("soil_reservoir.storage_threshold_secondary_m", threshold);
    let gwmax = c.number(&["max_gw_storage"], None)?;
    let gwfrac = c.number(&["gw_storage"], None)?;
    if gwmax <= 0.0 || !(0.0..=1.0).contains(&gwfrac) {
        return Err("invalid groundwater maximum/fraction".into());
    }
    s.f("gw_reservoir.gw_storage", gwfrac);
    s.f("gw_reservoir.storage_max_m", gwmax);
    s.f("gw_reservoir.storage_m", gwfrac * gwmax);
    s.f("gw_reservoir.coeff_primary", c.number(&["Cgw"], None)?);
    s.f("gw_reservoir.exponent_primary", c.number(&["expon"], None)?);
    s.f("gw_reservoir.exponent_secondary", 1.0);
    s.i("gw_reservoir.is_exponential", 1);
    s.f(
        "K_nash_subsurface",
        c.number(&["K_nash_subsurface", "K_nash"], None)?,
    );
    let mut nn = c.count("N_nash_subsurface", 2)?;
    let mut nash = c.vector("nash_storage_subsurface")?.unwrap_or_default();
    if nash.len() < 2 {
        nash = vec![0.0; nn];
    } else {
        nn = nash.len();
    }
    if nn < 1 {
        return Err("Nash subsurface count must be positive".into());
    }
    s.i("N_nash_subsurface", nn as i32);
    s.vectors.insert("nash_subsurface".into(), nash);
    let partition = c.scheme(
        &[
            "surface_water_partitioning_scheme",
            "surface_partitioning_scheme",
        ],
        None,
        &[("Schaake", 1), ("1", 1), ("Xinanjiang", 2), ("2", 2)],
    )?;
    s.i(
        "infiltration_excess_params_struct.surface_water_partitioning_scheme",
        partition,
    );
    if partition == 1 {
        s.f(
            "infiltration_excess_params_struct.Schaake_adjusted_magic_constant_by_soil_type",
            s.f["NWM_soil_params.refkdt"] * satdk / 0.000002,
        );
    } else {
        for key in [
            "a_Xinanjiang_inflection_point_parameter",
            "b_Xinanjiang_shape_parameter",
            "x_Xinanjiang_shape_parameter",
            "urban_decimal_fraction",
        ] {
            s.f(
                &format!("infiltration_excess_params_struct.{key}"),
                c.number(&[key], None)?,
            );
        }
    }
    let sft = c.flag("is_sft_coupled")?;
    s.i("soil_reservoir.is_sft_coupled", sft as i32);
    if sft && partition == 1 {
        s.f(
            "infiltration_excess_params_struct.ice_content_threshold",
            c.number(&["ice_content_threshold"], None)?,
        );
    }
    let root = c.flag("is_aet_rootzone")?;
    s.i("soil_reservoir.is_aet_rootzone", root as i32);
    let mut profile = vec![0.0];
    let mut depths = vec![0.0];
    let mut deltas = vec![0.0];
    if root {
        depths = c
            .vector("soil_layer_depths")?
            .ok_or("rootzone requires soil_layer_depths")?;
        if depths.is_empty() || *depths.last().unwrap() != d {
            return Err("soil depths must end at soil depth".into());
        }
        let layer = c.count("max_rootzone_layer", usize::MAX)?;
        if layer >= depths.len() {
            return Err("max_rootzone_layer outside soil profile".into());
        }
        s.i("soil_reservoir.max_rootzone_layer", layer as i32);
        let mut last = 0.0;
        deltas = vec![];
        for &z in &depths {
            if z <= last {
                return Err("soil depths must increase".into());
            }
            deltas.push(z - last);
            last = z;
        }
        profile = vec![0.0; depths.len()];
        s.f(
            "soil_reservoir.soil_water_content_field_capacity",
            sm * ((a * 101325.0) / (9810.0 * psi)).powf(-1.0 / b),
        );
    }
    s.i("soil_reservoir.n_soil_layers", profile.len() as i32);
    s.vectors.insert("smc_profile".into(), profile);
    s.vectors.insert("soil_layer_depths".into(), depths);
    s.vectors.insert("delta_soil_layer_depths".into(), deltas);
    let surface = c.scheme(
        &["surface_runoff_scheme"],
        Some(1),
        &[("GIUH", 1), ("1", 1), ("NASH_CASCADE", 2), ("2", 2)],
    )?;
    s.i("surface_runoff_scheme", surface);
    let mut giuh = vec![];
    let mut nsurface = vec![];
    if surface == 1 {
        giuh = c
            .vector("giuh_ordinates")?
            .ok_or("GIUH requires ordinates")?;
        if giuh.is_empty() {
            return Err("empty GIUH".into());
        }
    } else {
        let count = c.count("N_nash_surface", 0)?;
        let supplied = c
            .vector("nash_storage_surface")?
            .ok_or("Nash surface requires storage vector")?;
        if count == 0 || supplied.len() != count {
            return Err("Nash surface count differs from storage".into());
        }
        // Pinned native uses provided storage only for count>2.
        nsurface = if count > 2 {
            supplied
        } else {
            vec![0.0; count]
        };
        s.i("nash_surface_params.N_nash", count as i32);
        s.i("nash_surface_params.has_storage", 1);
        s.i(
            "nash_surface_params.nsubsteps",
            c.count("nsubsteps_nash_surface", 10)? as i32,
        );
        s.f(
            "nash_surface_params.K_nash",
            c.number(&["K_nash_surface"], None)?,
        );
        s.f(
            "nash_surface_params.K_infiltration",
            c.number(&["Kinf_nash_surface"], Some(0.001))?,
        );
        s.f(
            "nash_surface_params.retention_depth",
            c.number(&["retention_depth_nash_surface"], Some(0.001))?,
        );
    }
    s.i("num_giuh_ordinates", giuh.len() as i32);
    s.vectors
        .insert("runoff_queue".into(), vec![0.0; giuh.len() + 1]);
    s.vectors.insert("giuh_ordinates".into(), giuh);
    s.vectors.insert("nash_surface".into(), nsurface);
    s.raw_initial = apply_raw_initial(&c, &mut s)?;
    let gw = s.f["gw_reservoir.storage_m"];
    let soil = s.f["soil_reservoir.storage_m"];
    let start = (0.0 + gw) + soil;
    for key in ["volstart", "cumulative_vol", "volume_in_domain"] {
        s.f(&format!("vol_struct.{key}"), start);
    }
    s.f("vol_struct.vol_in_gw_start", gw);
    s.f("vol_struct.vol_soil_start", soil);
    Ok(s)
}
fn write_new(path: &Path, bytes: &[u8]) -> Fallible<()> {
    let mut f = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    f.write_all(bytes).map_err(|e| e.to_string())
}
fn run() -> Fallible<()> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 2 {
        return Err("usage: nextgen-cfe-prepare REQUEST.json".into());
    }
    let start = Instant::now();
    let request_path = Path::new(&args[1]);
    let r: Request = serde_json::from_slice(&fs::read(request_path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if r.schema != "arwen.cfe.prepare/v1" || r.time_origin.is_empty() || r.columns.is_empty() {
        return Err("invalid CFE preparation schema/time/columns".into());
    }
    let mut ids = BTreeSet::new();
    if r.columns
        .iter()
        .any(|c| !ids.insert(&c.id) || c.id.is_empty())
    {
        return Err("empty or duplicate catchment ID".into());
    }
    let layout: Layout =
        serde_json::from_str(include_str!("../layout.json")).map_err(|e| e.to_string())?;
    let initialize_start = Instant::now();
    let states: Vec<_> = r
        .columns
        .par_iter()
        .map(initialize)
        .collect::<Fallible<_>>()?;
    let known_fields: BTreeSet<_> = layout.fields.iter().map(String::as_str).collect();
    let known_options: BTreeSet<_> = layout.options.iter().map(String::as_str).collect();
    for s in &states {
        if s.f
            .iter()
            .any(|(k, v)| !known_fields.contains(k.as_str()) || !v.is_finite())
            || s.i.keys().any(|k| !known_options.contains(k.as_str()))
        {
            return Err("initializer produced unknown or nonfinite state".into());
        }
    }
    let initialize_ms = initialize_start.elapsed().as_secs_f64() * 1000.0;
    let pack_start = Instant::now();
    let n = states.len();
    let mut buffers: BTreeMap<String, (&str, Vec<usize>, Vec<u8>)> = BTreeMap::new();
    let mut fields = vec![];
    for name in &layout.fields {
        for s in &states {
            fields.extend_from_slice(&s.f.get(name).copied().unwrap_or(0.0).to_le_bytes());
        }
    }
    let mut options = vec![];
    for name in &layout.options {
        for s in &states {
            options.extend_from_slice(&s.i.get(name).copied().unwrap_or(0).to_le_bytes());
        }
    }
    buffers.insert(
        "fields".into(),
        ("f64", vec![layout.fields.len(), n], fields),
    );
    buffers.insert(
        "options".into(),
        ("i32", vec![layout.options.len(), n], options),
    );
    let vnames = [
        "giuh_ordinates",
        "runoff_queue",
        "nash_subsurface",
        "nash_surface",
        "smc_profile",
        "soil_layer_depths",
        "delta_soil_layer_depths",
    ];
    let mut offsets: BTreeMap<&str, Vec<i32>> = BTreeMap::new();
    for key in vnames {
        let mut raw = vec![];
        let mut off = vec![];
        for state in &states {
            off.push(i32::try_from(raw.len() / 8).map_err(|_| "ragged buffer exceeds int32")?);
            for x in &state.vectors[key] {
                raw.extend_from_slice(&x.to_le_bytes());
            }
        }
        let len = raw.len() / 8;
        buffers.insert(key.into(), ("f64", vec![len], raw));
        offsets.insert(key, off);
    }
    let mut packed_offsets = vec![];
    for key in &layout.offsets {
        let src = if key == "soil_profile" {
            "smc_profile"
        } else {
            key.as_str()
        };
        for x in &offsets[src] {
            packed_offsets.extend_from_slice(&x.to_le_bytes());
        }
    }
    buffers.insert("offsets".into(), ("i32", vec![5, n], packed_offsets));
    let pack_ms = pack_start.elapsed().as_secs_f64() * 1000.0;
    let write_start = Instant::now();
    fs::create_dir(&r.output).map_err(|e| e.to_string())?;
    let output = r.output.canonicalize().map_err(|e| e.to_string())?;
    let entries:Vec<Value>=buffers.par_iter().map(|(name,(dtype,shape,bytes))|{let path=output.join(format!("{name}.bin"));write_new(&path,bytes)?;Ok(json!({"name":name,"dtype":dtype,"shape":shape,"bytes":bytes.len(),"path":path,"sha256":file_sha256(&path)?}))}).collect::<Fallible<_>>()?;
    let inventory: Vec<_> = r
        .columns
        .iter()
        .zip(&states)
        .map(|(c, s)| json!({"id":c.id,"water":initial_inventory(s)}))
        .collect();
    let manifest = json!({"schema":"cfe.initial.pack/v1","source_pin":PIN,"period_seconds":3600.0,"field_names":layout.fields,"option_names":layout.options,"catchments":r.columns.iter().map(|c|&c.id).collect::<Vec<_>>(),"time_origin":r.time_origin,"buffers":entries,"request_sha256":file_sha256(request_path)?,"configurations":r.columns.iter().map(|c|json!({"id":c.id,"config":c.config})).collect::<Vec<_>>(),"initial_water_inventory":inventory,"initialization":"native equations; defined zero state; explicit fractions or complete raw initial water keys; fresh diagnostic clock; BMI inputs bound on GPU","stages_ms":{"config_read_verify_initialize":initialize_ms,"ordered_typed_pack":pack_ms,"write_hash":write_start.elapsed().as_secs_f64()*1000.0},"preparation_wall_ms":start.elapsed().as_secs_f64()*1000.0,"parallelism":format!("Rust Rayon {} workers; catchments and output buffers",rayon::current_num_threads())});
    write_new(
        &output.join("manifest.json"),
        &serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?,
    )?;
    println!("{}", output.join("manifest.json").display());
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("nextgen-cfe-prepare: {e}");
        std::process::exit(2)
    }
}

#[cfg(test)]
mod tests;
