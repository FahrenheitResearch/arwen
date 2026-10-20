// SPDX-License-Identifier: Apache-2.0
// Offline packing of referee-owned CFE captures; no reference physics here.
use nextgen_replay::{file_sha256, sealed_bytes, Fallible, Seal};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};
const PIN: &str = "f9182dfb3d81c407c66c5b782114173893c30cf2";

#[derive(Deserialize)]
struct Descriptor {
    name: String,
    r#type: String,
    offset: usize,
    count: usize,
    itemsize: usize,
    nbytes: usize,
}
struct Snapshot {
    desc: BTreeMap<String, Descriptor>,
    bytes: Vec<u8>,
}
impl Snapshot {
    fn item(&self, name: &str) -> Fallible<Option<&[u8]>> {
        let Some(d) = self.desc.get(&format!("state.{name}")) else {
            return Ok(None);
        };
        if d.count.checked_mul(d.itemsize) != Some(d.nbytes) {
            return Err(format!("{name}: descriptor size mismatch"));
        }
        Ok(Some(
            self.bytes
                .get(d.offset..d.offset.checked_add(d.nbytes).ok_or("offset overflow")?)
                .ok_or("descriptor outside payload")?,
        ))
    }
    fn scalar(&self, name: &str, dtype: &str) -> Fallible<Vec<u8>> {
        let d = self
            .desc
            .get(&format!("state.{name}"))
            .ok_or(format!("missing scalar {name}"))?;
        let raw = self.item(name)?.unwrap();
        if d.count != 1 {
            return Err(format!("{name}: not scalar"));
        }
        if dtype == "f64" && name == "time_step_size" && d.r#type == "int" && raw.len() == 4 {
            return Ok((i32::from_le_bytes(raw.try_into().unwrap()) as f64)
                .to_le_bytes()
                .to_vec());
        }
        let (native, width) = if dtype == "f64" {
            ("double", 8)
        } else {
            ("int", 4)
        };
        if d.r#type != native || raw.len() != width {
            return Err(format!("{name}: incompatible native type {}", d.r#type));
        }
        Ok(raw.to_vec())
    }
    fn int(&self, name: &str) -> Fallible<i32> {
        Ok(i32::from_le_bytes(
            self.scalar(name, "i32")?.try_into().unwrap(),
        ))
    }
    fn vector(&self, name: &str) -> Fallible<Vec<u8>> {
        let Some(d) = self.desc.get(&format!("state.{name}")) else {
            return Ok(vec![]);
        };
        if d.r#type != "double" || d.itemsize != 8 {
            return Err(format!("{name}: non-double vector"));
        }
        Ok(self.item(name)?.unwrap().to_vec())
    }
}
struct Captures {
    root: PathBuf,
    prefix: String,
    seals: BTreeMap<String, Seal>,
}
impl Captures {
    fn read(&self, relative: &str) -> Fallible<Vec<u8>> {
        sealed_bytes(
            self.seals
                .get(relative)
                .ok_or(format!("not sealed in fixture index: {relative}"))?,
        )
    }
    fn snapshot(&self, model: usize, step: usize, boundary: &str) -> Fallible<Snapshot> {
        let stem = format!("{}model-{model}/step-{step}-{boundary}", self.prefix);
        let descriptor_bytes = self.read(&format!("{stem}.jsonl"))?;
        let mut desc = BTreeMap::new();
        for line in std::str::from_utf8(&descriptor_bytes)
            .map_err(|e| e.to_string())?
            .lines()
        {
            let d: Descriptor = serde_json::from_str(line).map_err(|e| e.to_string())?;
            if desc.insert(d.name.clone(), d).is_some() {
                return Err("duplicate capture descriptor".into());
            }
        }
        Ok(Snapshot {
            desc,
            bytes: self.read(&format!("{stem}.bin"))?,
        })
    }
}
#[derive(Deserialize)]
struct Layout {
    fields: Vec<String>,
    options: Vec<String>,
    offsets: Vec<String>,
}

#[derive(Deserialize)]
struct PublicDescriptor {
    name: String,
    r#type: String,
    units: String,
    offset: usize,
    itemsize: usize,
    nbytes: usize,
}

// Read actual public Noah output words, never invert a converted CFE input.
fn noah_rain(
    index_path: &Path,
    primary_hash: &str,
    captures: &Captures,
    entry: &[Snapshot],
    step: usize,
) -> Fallible<(Vec<u8>, Value, Seal)> {
    let index_path = index_path.canonicalize().map_err(|e| e.to_string())?;
    let index: Value = serde_json::from_slice(&fs::read(&index_path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if index["schema"] != "nextgen.native.capture/v1"
        || index["byte_order"] != "little"
        || index["recipe"]["original_primary_index_sha256"] != primary_hash
        || index["recipe"]["exchange_audit"]["valid"] != true
    {
        return Err("framework witness does not match the selected primary capture".into());
    }
    let audit = &index["recipe"]["exchange_audit"];
    if audit["native_conversion_scale"].as_f64() != Some(3600000.0000000005)
        || audit["native_conversion_zero"].as_f64() != Some(0.)
    {
        return Err("framework witness differs from the published UDUNITS conversion".into());
    }
    let root = PathBuf::from(
        index["capture_directory"]
            .as_str()
            .ok_or("missing framework root")?,
    );
    let mut seals = BTreeMap::new();
    for row in index["files"].as_array().ok_or("missing framework seals")? {
        let path = PathBuf::from(row["path"].as_str().ok_or("missing framework path")?);
        seals.insert(
            path.clone(),
            Seal {
                path,
                sha256: row["sha256"]
                    .as_str()
                    .ok_or("missing framework seal")?
                    .into(),
            },
        );
    }
    let field =
        |stem: &Path, name: &str, dtype: &str, units: &str, width: usize| -> Fallible<Vec<u8>> {
            let desc_path = stem.with_extension("jsonl");
            let raw_path = stem.with_extension("bin");
            let desc = sealed_bytes(seals.get(&desc_path).ok_or("unsealed public descriptors")?)?;
            let bytes = sealed_bytes(seals.get(&raw_path).ok_or("unsealed public payload")?)?;
            let rows: Vec<PublicDescriptor> = std::str::from_utf8(&desc)
                .map_err(|e| e.to_string())?
                .lines()
                .map(serde_json::from_str)
                .collect::<Result<_, _>>()
                .map_err(|e| e.to_string())?;
            let selected: Vec<_> = rows.iter().filter(|r| r.name == name).collect();
            if selected.len() != 1 {
                return Err(format!("ambiguous public field {name}"));
            }
            let d = selected[0];
            if d.r#type != dtype || d.units != units || d.itemsize != width || d.nbytes != width {
                return Err(format!("public scalar type/unit differs: {name}"));
            }
            Ok(bytes
                .get(
                    d.offset
                        ..d.offset
                            .checked_add(width)
                            .ok_or("public offset overflow")?,
                )
                .ok_or("public field outside payload")?
                .to_vec())
        };
    let mut rain = vec![];
    let mut roster = vec![];
    for (model, state) in entry.iter().enumerate() {
        let path: String = serde_json::from_slice(&captures.read(&format!(
            "{}model-{model}/config-path.json",
            captures.prefix
        ))?)
        .map_err(|e| e.to_string())?;
        let filename = Path::new(&path)
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or("invalid config path")?;
        let catchment = filename
            .strip_suffix("_bmi_config.ini")
            .ok_or("unrecognized native config naming")?;
        let q = field(
            &root
                .join(catchment)
                .join("bmi_fortran_noahowp")
                .join(format!("step-{step}-outputs")),
            "QINSUR",
            "real",
            "m/s",
            4,
        )?;
        let cfe_stem = root
            .join(catchment)
            .join("bmi_c_cfe")
            .join(format!("step-{step}-inputs"));
        let amount = field(
            &cfe_stem,
            "atmosphere_water__liquid_equivalent_precipitation_rate",
            "double",
            "mm h-1",
            8,
        )?;
        let pet = field(
            &cfe_stem,
            "water_potential_evaporation_flux",
            "double",
            "m s-1",
            8,
        )?;
        // Reject mismatched runs before combining their two different boundaries.
        if amount != state.scalar("aorc.precip_kg_per_m2", "f64")?
            || pet != state.scalar("et_struct.potential_et_m_per_s", "f64")?
        {
            return Err(format!(
                "framework and private CFE entry disagree for {catchment}"
            ));
        }
        rain.extend(q);
        roster.push(catchment.to_string());
    }
    let seal = Seal {
        path: index_path.clone(),
        sha256: file_sha256(&index_path)?,
    };
    let metadata = json!({"index":index_path,"sha256":seal.sha256,
        "catchment_ids":roster,"native_exchange_audit":audit,
        "rain_source":"sealed public Noah QINSUR f32 m/s, original words; no inverse conversion",
        "scope":"matched captured native upstream input; not live GPU Noah/PET composition"});
    Ok((rain, metadata, seal))
}
type Buffers = BTreeMap<String, (String, Vec<usize>, Vec<u8>)>;
fn pack(snapshots: &[Snapshot], layout: &Layout) -> Fallible<Buffers> {
    let n = snapshots.len();
    let mut result = BTreeMap::new();
    let rows: Vec<Vec<u8>> = layout
        .fields
        .par_iter()
        .map(|name| {
            let mut row = vec![];
            for s in snapshots {
                row.extend(s.scalar(name, "f64")?);
            }
            Ok(row)
        })
        .collect::<Fallible<_>>()?;
    let fields = rows.into_iter().flatten().collect();
    result.insert(
        "fields".into(),
        ("f64".into(), vec![layout.fields.len(), n], fields),
    );
    let mut options = vec![];
    for name in &layout.options {
        for s in snapshots {
            options.extend(if name == "nash_surface_params.has_storage" {
                let present = s
                    .desc
                    .contains_key("state.nash_surface_params.nash_storage");
                // f918 src/bmi_cfe.c:1123-1124 explicitly sets NULL for GIUH.
                // The observer omits zero-count arrays, so active Nash requires
                // a positive count and a complete descriptor to prove presence.
                let scheme = s.int("surface_runoff_scheme")?;
                if (scheme == 1 && present)
                    || (scheme == 2 && (!present || s.int("nash_surface_params.N_nash")? <= 0))
                {
                    return Err(
                        "surface-storage pointer presence is not witnessed by this capture".into(),
                    );
                }
                i32::from(present).to_le_bytes().to_vec()
            } else {
                s.scalar(name, "i32")?
            });
        }
    }
    result.insert(
        "options".into(),
        ("i32".into(), vec![layout.options.len(), n], options),
    );
    let specs = [
        ("giuh_ordinates", "giuh_ordinates"),
        ("runoff_queue", "runoff_queue_m_per_timestep"),
        ("nash_subsurface", "nash_storage_subsurface"),
        ("nash_surface", "nash_surface_params.nash_storage"),
        ("smc_profile", "soil_reservoir.smc_profile"),
        ("soil_layer_depths", "soil_reservoir.soil_layer_depths_m"),
        (
            "delta_soil_layer_depths",
            "soil_reservoir.delta_soil_layer_depth_m",
        ),
    ];
    let mut offsets = vec![];
    let mut soil_counts = vec![];
    for (which, (buffer, native)) in specs.iter().enumerate() {
        let mut values = vec![];
        for (catch, s) in snapshots.iter().enumerate() {
            let mut raw = s.vector(native)?;
            let native_count = raw.len() / 8;
            if which == 0
                && native_count
                    != usize::try_from(s.int("num_giuh_ordinates")?).map_err(|e| e.to_string())?
            {
                return Err("GIUH count differs".into());
            }
            if which == 1
                && native_count
                    != usize::try_from(s.int("num_giuh_ordinates")?).map_err(|e| e.to_string())? + 1
            {
                return Err("GIUH queue lacks native sentinel".into());
            }
            if which == 2
                && native_count
                    != usize::try_from(s.int("N_nash_subsurface")?).map_err(|e| e.to_string())?
            {
                return Err("Nash count differs".into());
            }
            if which < 5 {
                offsets.extend(
                    i32::try_from(values.len() / 8)
                        .map_err(|e| e.to_string())?
                        .to_le_bytes(),
                );
            }
            if which == 4 {
                soil_counts.push(native_count);
            }
            if which >= 5 && native_count == 0 && s.int("soil_reservoir.is_aet_rootzone")? == 0 {
                // Binding ABI needs equal extents. Native NULL inactive depths are explicitly tool slots.
                raw.resize(soil_counts[catch] * 8, 0);
            }
            if which >= 5 && raw.len() / 8 != soil_counts[catch] {
                return Err("active soil vectors differ in extent".into());
            }
            values.extend(raw);
        }
        result.insert(
            buffer.to_string(),
            ("f64".into(), vec![values.len() / 8], values),
        );
    }
    result.insert("offsets".into(), ("i32".into(), vec![5, n], offsets));
    Ok(result)
}
fn new_file(path: &Path, bytes: &[u8]) -> Fallible<()> {
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| e.to_string())?
        .write_all(bytes)
        .map_err(|e| e.to_string())
}

// Native expected histories only. Resident capture/export belongs to the driver
// and output owners; this deliberately defines no GPU history or writer ABI.
fn history(
    captures: &Captures,
    index: &Value,
    index_path: &Path,
    output: &Path,
    count: usize,
    extractor_revision: &str,
    start: Instant,
) -> Fallible<()> {
    if count == 0 || count > 48 {
        return Err("history count must be within the sealed 48-hour case".into());
    }
    let mut catchments = vec![];
    for model in 0..3 {
        let config: String = serde_json::from_slice(&captures.read(&format!(
            "{}model-{model}/config-path.json",
            captures.prefix
        ))?)
        .map_err(|e| e.to_string())?;
        let id = Path::new(&config)
            .file_name()
            .and_then(|s| s.to_str())
            .and_then(|s| s.strip_suffix("_bmi_config.ini"))
            .ok_or("unrecognized native catchment config name")?
            .to_string();
        if catchments.contains(&id) {
            return Err("duplicate native catchment".into());
        }
        catchments.push(id);
    }
    let frames: Vec<_> = (0..count)
        .into_par_iter()
        .map(|step| {
            let mut frame = vec![];
            for model in 0..3 {
                let entry = captures.snapshot(model, step, "entry")?;
                let exit = captures.snapshot(model, step, "exit")?;
                if entry.int("current_time_step")? != step as i32
                    || exit.int("current_time_step")? != step as i32 + 1
                    || entry.int("time_step_size")? != 3600
                    || exit.int("time_step_size")? != 3600
                    || entry.scalar("time_step_fraction", "f64")? != 1f64.to_le_bytes()
                    || exit.scalar("time_step_fraction", "f64")? != 1f64.to_le_bytes()
                {
                    return Err("history clock/period/fraction differs from hourly case".into());
                }
                frame.push(exit);
            }
            Ok(frame)
        })
        .collect::<Fallible<_>>()?;
    let specs = [
        ("flux_Qout_m", "m/step"),
        ("soil_reservoir.storage_m", "m"),
        ("gw_reservoir.storage_m", "m"),
        ("timestep_rainfall_input_m", "m/step"),
        ("et_struct.potential_et_m_per_s", "m/s"),
        ("et_struct.actual_et_m_per_timestep", "m/step"),
    ];
    // Extract bytes only, with no recalculation, normalization or conversion.
    let fields = specs
        .par_iter()
        .map(|(name, units)| {
            let mut raw = vec![];
            for frame in &frames {
                for snapshot in frame {
                    raw.extend(snapshot.scalar(name, "f64")?);
                }
            }
            Ok((*name, *units, raw))
        })
        .collect::<Fallible<Vec<_>>>()?;
    fs::create_dir(output).map_err(|e| e.to_string())?;
    let output = output.canonicalize().map_err(|e| e.to_string())?;
    let mut manifest = vec![];
    for (name, units, raw) in fields {
        let path = output.join(format!("{name}.f64le"));
        new_file(&path, &raw)?;
        manifest.push(json!({"name":name,"units":units,"dtype":"f64",
            "shape":[count,3],"order":"c","axes":["time","catchment"],
            "path":path,"sha256":file_sha256(&path)?,"bytes":raw.len()}));
    }
    let endpoint_raw: Vec<_> = (1..=count)
        .flat_map(|s| ((s as i64) * 3600).to_le_bytes())
        .collect();
    let endpoint_path = output.join("endpoint_seconds.i64le");
    new_file(&endpoint_path, &endpoint_raw)?;
    let executable = std::env::current_exe().map_err(|e| e.to_string())?;
    let receipt = json!({"schema":"cfe.native.history/v1","source_pin":PIN,
        "extractor_revision":extractor_revision,"command":std::env::args().collect::<Vec<_>>(),
        "executable":{"path":executable,"sha256":file_sha256(&executable)?},
        "native_capture_index":{"path":index_path,"sha256":file_sha256(index_path)?},
        "native_recipe":index["recipe"],"catchment_ids":catchments,
        "fields":manifest,"clock":{"semantics":"post-BMI-Update endpoints, relative to native initial clock; no UTC origin inferred",
            "period_seconds":3600,"initial_step":0,"steps":count,
            "endpoint_seconds":{"dtype":"i64","shape":[count],"path":endpoint_path,"sha256":file_sha256(&endpoint_path)?}},
        "scope":"sealed native expectations only; no GPU run, comparison result, resident writer contract or discharge conversion",
        "workers":rayon::current_num_threads(),"wall_ms":start.elapsed().as_secs_f64()*1000.});
    new_file(
        &output.join("manifest.json"),
        &serde_json::to_vec_pretty(&receipt).map_err(|e| e.to_string())?,
    )?;
    println!(
        "{}",
        json!({"output":output,"steps":count,"catchments":3,"fields":6,
        "words":count*3*6,"wall_ms":start.elapsed().as_secs_f64()*1000.,"workers":rayon::current_num_threads()})
    );
    Ok(())
}
fn run() -> Fallible<()> {
    let start = Instant::now();
    let mut args: Vec<_> = std::env::args().collect();
    let framework_index = if let Some(i) = args.iter().position(|a| a == "--framework-index") {
        if i + 2 != args.len() {
            return Err("--framework-index INDEX must be last".into());
        }
        let path = PathBuf::from(&args[i + 1]);
        args.truncate(i);
        Some(path)
    } else {
        None
    };
    if !matches!(args.len(), 6 | 7) {
        return Err(
            "usage: cfe-capture-pack INDEX LAYOUT STEP|wet|bmi:STEP|bmi-mm-h:STEP|bmi-noah:STEP|history:COUNT NEW_OUTPUT CANDIDATE_REVISION [CAPTURE_ROOT] [--framework-index INDEX]".into(),
        );
    }
    let index_path = Path::new(&args[1])
        .canonicalize()
        .map_err(|e| e.to_string())?;
    let index: Value = serde_json::from_slice(&fs::read(&index_path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if index["schema"] != "nextgen.native.capture/v1" || index["byte_order"] != "little" {
        return Err("unsupported native capture index".into());
    }
    // The original observer captured unused, uninitialized native members.
    // A sealed payload alone does not make those words a numerical referee.
    let audit = index["recipe"]["initialized_state_audit"]
        .as_str()
        .filter(|s| !s.trim().is_empty())
        .ok_or("missing native initialized-state audit")?;
    if audit.trim_start().to_ascii_lowercase().starts_with("pending") {
        return Err(
            "native initialized-state audit pending; cannot pack numerical replay or history".into(),
        );
    }
    if !index["recipe"]["sources"]
        .as_array()
        .ok_or("missing source pins")?
        .iter()
        .any(|s| s["repository"] == "NOAA-OWP/cfe" && s["revision"] == PIN)
    {
        return Err("CFE pin absent".into());
    }
    let root = PathBuf::from(
        index["capture_directory"]
            .as_str()
            .ok_or("missing capture directory")?,
    );
    let relocated_root = if args.len() == 7 {
        Path::new(&args[6])
            .canonicalize()
            .map_err(|e| e.to_string())?
    } else {
        root.clone()
    };
    let mut seals = BTreeMap::new();
    for item in index["files"].as_array().ok_or("missing file seals")? {
        let path = Path::new(item["path"].as_str().ok_or("missing path")?);
        // The referee index also seals recipe/config files outside its capture root.
        // Those remain evidence in the sealed index, never capture payloads here.
        let rel = if path.is_absolute() {
            match path.strip_prefix(&root) {
                Ok(p) => p,
                Err(_) => continue,
            }
        } else {
            path
        };
        if rel
            .components()
            .any(|c| !matches!(c, std::path::Component::Normal(_)))
        {
            return Err("invalid index relative path".into());
        }
        seals.insert(
            rel.to_str().ok_or("non-UTF8 capture path")?.to_string(),
            Seal {
                path: relocated_root.join(rel),
                sha256: item["sha256"].as_str().ok_or("missing sha256")?.into(),
            },
        );
    }
    let prefix = if seals.contains_key("cfe/model-0/step-0-core-entry.jsonl") {
        "cfe/"
    } else if seals.contains_key("model-0/step-0-core-entry.jsonl") {
        ""
    } else {
        return Err("index has no recognized CFE module capture namespace".into());
    };
    let captures = Captures {
        root: relocated_root,
        prefix: prefix.into(),
        seals,
    };
    if let Some(count) = args[3].strip_prefix("history:") {
        if framework_index.is_some() {
            return Err("native history uses the primary index only".into());
        }
        return history(
            &captures,
            &index,
            &index_path,
            Path::new(&args[4]),
            count.parse::<usize>().map_err(|e| e.to_string())?,
            &args[5],
            start,
        );
    }
    let layout: Layout = serde_json::from_slice(&fs::read(&args[2]).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if layout.fields.len() != 97
        || layout.options.len() != 18
        || layout.offsets
            != [
                "giuh_ordinates",
                "runoff_queue",
                "nash_subsurface",
                "nash_surface",
                "soil_profile",
            ]
    {
        return Err("layout differs from published CFE ABI".into());
    }
    let noah = args[3].starts_with("bmi-noah:");
    if noah != framework_index.is_some() {
        return Err("bmi-noah:STEP requires exactly one --framework-index INDEX witness".into());
    }
    let native_rain = args[3].starts_with("bmi-mm-h:") || noah;
    let bmi = args[3].starts_with("bmi:") || native_rain;
    let mut selection = vec![];
    let step = if args[3] == "wet" {
        let scores: Vec<_> = (0usize..48)
            .into_par_iter()
            .map(|step| -> Fallible<(usize, f64)> {
                let mut rain = 0.;
                for model in 0..3 {
                    let s = captures.snapshot(model, step, "core-entry")?;
                    rain += f64::from_le_bytes(
                        s.scalar("timestep_rainfall_input_m", "f64")?
                            .try_into()
                            .unwrap(),
                    );
                }
                if !rain.is_finite() {
                    return Err("nonfinite captured rainfall".into());
                }
                Ok((step, rain))
            })
            .collect::<Fallible<_>>()?;
        let chosen = scores
            .iter()
            .max_by(|a, b| a.1.total_cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
            .ok_or("empty native case")?
            .0;
        selection = scores
            .into_iter()
            .map(|(s, r)| json!({"step":s,"sum_rainfall_m":r}))
            .collect();
        chosen
    } else {
        args[3]
            .strip_prefix("bmi:")
            .or_else(|| args[3].strip_prefix("bmi-mm-h:"))
            .or_else(|| args[3].strip_prefix("bmi-noah:"))
            .unwrap_or(&args[3])
            .parse::<usize>()
            .map_err(|e| e.to_string())?
    };
    let pairs: Vec<_> = (0usize..3)
        .into_par_iter()
        .map(|m| {
            Ok((
                captures.snapshot(m, step, if bmi { "entry" } else { "core-entry" })?,
                captures.snapshot(m, step, if bmi { "exit" } else { "core-exit" })?,
            ))
        })
        .collect::<Fallible<_>>()?;
    let (entry, exit): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
    let period: Vec<_> = entry
        .iter()
        .map(|s| s.int("time_step_size"))
        .collect::<Fallible<_>>()?;
    if period.iter().any(|p| *p <= 0 || *p != period[0]) {
        return Err("mixed/invalid native periods".into());
    }
    let mut input = pack(&entry, &layout)?;
    let mut expected = pack(&exit, &layout)?;
    let mut bmi_metadata = Value::Null;
    if bmi {
        let clocks: Vec<_> = entry
            .iter()
            .map(|s| s.int("current_time_step"))
            .collect::<Fallible<_>>()?;
        let fractions: Vec<_> = entry
            .iter()
            .map(|s| s.scalar("time_step_fraction", "f64"))
            .collect::<Fallible<_>>()?;
        if clocks.iter().any(|c| *c != clocks[0]) || fractions.iter().any(|f| *f != fractions[0]) {
            return Err("current API requires common native initial clock/fraction".into());
        }
        let fraction = f64::from_le_bytes(fractions[0].as_slice().try_into().unwrap());
        bmi_metadata = json!({"initial_step":clocks[0],"time_step_fraction":fraction,
            "precipitation_buffer":if native_rain { "precipitation_mm_h" } else { "rainrate_mm_s" },
            "precipitation_units":if native_rain { "mm/h" } else { "mm/s" },
            "rain_conversion":if native_rain { "captured native aorc.precip_kg_per_m2 preserved bitwise in native BMI mm/h units; GPU native /1000 then fraction" } else { "captured native aorc.precip_kg_per_m2 divided by native time_step_size in Rust to canonical mm/s; GPU multiplies period before native /1000. No native input is treated as already mm/s." }});
        for (target, native, dtype) in [
            (
                if native_rain {
                    "precipitation_mm_h"
                } else {
                    "rainrate_mm_s"
                },
                "aorc.precip_kg_per_m2",
                "f64",
            ),
            ("potential_et_m_s", "et_struct.potential_et_m_per_s", "f64"),
            ("current_time_step", "current_time_step", "i32"),
            ("time_step_fraction", "time_step_fraction", "f64"),
            ("nwm_ponded_depth_m", "nwm_ponded_depth_m", "f64"),
        ] {
            for (snapshots, buffers) in [(&entry, &mut input), (&exit, &mut expected)] {
                let mut raw = vec![];
                for s in snapshots {
                    if s.int("is_forcing_from_bmi")? != 1 {
                        return Err("BMI replay requires native BMI forcing".into());
                    }
                    let word = s.scalar(native, dtype)?;
                    if target == "rainrate_mm_s" {
                        let amount = f64::from_le_bytes(word.as_slice().try_into().unwrap());
                        raw.extend((amount / f64::from(period[0])).to_le_bytes());
                    } else {
                        raw.extend(word);
                    }
                }
                buffers.insert(target.into(), (dtype.into(), vec![3], raw));
            }
        }
    }
    let index_hash = file_sha256(&index_path)?;
    let mut framework_seal = None;
    if let Some(framework_index) = framework_index {
        let (rain, metadata, seal) =
            noah_rain(&framework_index, &index_hash, &captures, &entry, step)?;
        for buffers in [&mut input, &mut expected] {
            buffers.remove("precipitation_mm_h");
            buffers.insert(
                "precipitation_m_s".into(),
                ("f32".into(), vec![3], rain.clone()),
            );
        }
        bmi_metadata["precipitation_buffer"] = json!("precipitation_m_s");
        bmi_metadata["precipitation_units"] = json!("m/s");
        bmi_metadata["rain_conversion"] = json!("original captured Noah f32 m/s; GPU native UDUNITS 2.2.28 scale3600000.0000000005 then native /1000 and fraction");
        bmi_metadata["framework_witness"] = metadata;
        framework_seal = Some(seal);
    }
    expected.insert("status".into(), ("i32".into(), vec![3], vec![0; 12]));
    for (name, (_, shape, _)) in &input {
        if expected[name].1 != *shape {
            return Err(format!("native output changed extent {name}"));
        }
    }
    let output = PathBuf::from(&args[4]);
    fs::create_dir(&output).map_err(|e| e.to_string())?;
    let output = output.canonicalize().map_err(|e| e.to_string())?;
    for sub in ["input", "expected", "actual"] {
        fs::create_dir(output.join(sub)).map_err(|e| e.to_string())?;
    }
    let mut manifest = vec![];
    let mut compare = vec![];
    for (name, (dtype, shape, raw)) in &input {
        let path = output.join("input").join(format!("{name}.bin"));
        new_file(&path, raw)?;
        manifest.push(json!({"name":name,"dtype":dtype,"shape":shape,"path":path,"sha256":file_sha256(&path)?,"bytes":raw.len()}));
    }
    for (name, (dtype, shape, raw)) in &expected {
        let path = output.join("expected").join(format!("{name}.bin"));
        new_file(&path, raw)?;
        compare.push(json!({"name":name,"dtype":dtype,"shape":shape,"order":"c","expected":{"path":path,"sha256":file_sha256(&path)?},"actual":output.join("actual").join(format!("{name}.bin"))}));
    }
    let layout_path = Path::new(&args[2])
        .canonicalize()
        .map_err(|e| e.to_string())?;
    let native_case = index["recipe"]["case"]
        .as_str()
        .ok_or("missing native case")?;
    let boundary = if bmi { "bmi-update" } else { "core" };
    let mut request = json!({"schema":"nextgen.raw.compare/v1","unit":"cfe","case":format!("{native_case}-{boundary}-step-{step}-3catchments"),"numerics":"strict",
        "candidate_revision":args[5],"command":["python3","tools/cfe_replay/run_gpu.py",output.join("manifest.json")],
        "provenance":{"origin":"native-bmi","sources":[{"name":"NOAA-OWP/cfe","revision":PIN,"licence":"Apache-2.0"}],"evidence":[{"path":index_path,"sha256":index_hash},{"path":layout_path,"sha256":file_sha256(&layout_path)?}]},"fields":compare});
    if let Some(seal) = framework_seal {
        request["provenance"]["evidence"]
            .as_array_mut()
            .unwrap()
            .push(json!({"path":seal.path,"sha256":seal.sha256}));
    }
    let manifest = json!({"schema":"cfe.capture.pack/v1","source_pin":PIN,"candidate_revision":args[5],"native_capture_index":index_path,"native_capture_index_sha256":index_hash,
        "step":step,"catchments":["model-0","model-1","model-2"],"capture_root":captures.root,"capture_prefix":captures.prefix,"period_seconds":period[0],"buffers":manifest,
        "native_recipe":index["recipe"],
        "bmi":bmi_metadata,
        "field_names":layout.fields,"option_names":layout.options,"selection_rainfall":selection,
        "scope":format!("{}; initialization/adaptations stated verbatim in native_recipe; patched reference is not untouched-upstream identity.",if bmi { "BMI entry to exit including canonical rain conversion, native clock, fraction and ponded depth" } else { "core-entry to core-exit; BMI clock/rainfall conversion/ponded depth excluded" }),
        "inactive_tool_slots":"Only missing native soil-depth vectors with rootzone disabled receive zero binding slots. No native captured words are masked.",
        "parallelism":format!("Rust Rayon {} workers; sealed captures and offline packing",rayon::current_num_threads()),"wall_ms":start.elapsed().as_secs_f64()*1000.});
    for (name, value) in [("manifest.json", manifest), ("compare.json", request)] {
        new_file(
            &output.join(name),
            &serde_json::to_vec_pretty(&value).map_err(|e| e.to_string())?,
        )?;
    }
    println!(
        "{}",
        json!({"output":output,"step":step,"wall_ms":start.elapsed().as_secs_f64()*1000.,"workers":rayon::current_num_threads()})
    );
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(2);
    }
}
