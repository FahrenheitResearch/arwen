//! Read the optional surface member archive through its pinned native contract.
//! Its numeric identities never pass through floating point, and its original
//! diagnostic planes are not reconstructed from prognostic history.
use super::*;
use chrono::{DateTime, NaiveDateTime, Utc};
use netcdf_reader::{NcFile, NcFormat, NcType};
use std::path::Component;

pub const CONTRACT: &str = "gpuwm-ensemble-member-diagnostics.v1";

#[derive(Deserialize)]
pub struct DiagnosticMatch {
    plan: PathBuf,
    quantity: String,
    valid_time: String,
    #[serde(default)]
    precipitation_observation: Option<PathBuf>,
    archive: ArchiveRequest,
    members: Vec<ExpectedMember>,
}
#[derive(Deserialize)]
struct ArchiveRequest {
    root: PathBuf,
    manifest: Reference,
    grid_id: u64,
    episode: u64,
}
#[derive(Deserialize)]
struct ExpectedMember {
    id: String,
    member_id: u64,
    seed: u64,
}
#[derive(Deserialize)]
struct Manifest {
    schema: String,
    member_order: Vec<u64>,
    member_metadata: Vec<Value>,
    fields: BTreeMap<String, String>,
    precipitation_accumulation_start: String,
    forecast_volume_fields: bool,
    files: Vec<Entry>,
    unavailable: Vec<Unavailable>,
}
#[derive(Clone, Serialize, Deserialize)]
struct Entry {
    path: String,
    bytes: u64,
    sha256: String,
    member_id: u64,
    seed: u64,
    grid_id: u64,
    episode: u64,
    valid_time: String,
    geometry_sha256: String,
}
#[derive(Deserialize)]
struct Unavailable {
    member_id: u64,
    grid_id: u64,
    episode: u64,
    valid_time: String,
}
struct Selected {
    label: String,
    path: PathBuf,
    entry: Entry,
    provenance: Value,
}
pub struct Archive {
    manifest: Reference,
    start: String,
    end: String,
    selected: Vec<Selected>,
}

fn time(raw: &str) -> Result<DateTime<Utc>> {
    if let Ok(value) = DateTime::parse_from_rfc3339(raw) {
        return Ok(value.with_timezone(&Utc));
    }
    // datetime.isoformat() on the engine's naive UTC clock carries no suffix.
    // Assign UTC explicitly; never consult the host's local timezone.
    Ok(NaiveDateTime::parse_from_str(raw, "%Y-%m-%dT%H:%M:%S%.f")?.and_utc())
}
fn digest(raw: &str) -> bool {
    raw.len() == 64
        && raw
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}
fn archive_path(root: &Path, raw: &str) -> Result<PathBuf> {
    let path = Path::new(raw);
    if raw.is_empty()
        || raw.contains('\\')
        || path
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err(err(
            "diagnostic path must be relative to the archive root without traversal",
        ));
    }
    let result = root.join(path).canonicalize()?;
    if !result.starts_with(root) || !result.is_file() {
        return Err(err(
            "diagnostic file leaves the archive root or is not a file",
        ));
    }
    Ok(result)
}
fn fields() -> BTreeMap<String, String> {
    [
        ("T2", "K"),
        ("U10", "m s-1"),
        ("V10", "m s-1"),
        ("RAIN_TOTAL", "mm"),
    ]
    .into_iter()
    .map(|(name, units)| (name.into(), units.into()))
    .collect()
}
pub fn resolve(request: &DiagnosticMatch) -> Result<(Match, Archive)> {
    let body = std::fs::read(&request.archive.manifest.path)?;
    if !digest(&request.archive.manifest.sha256)
        || hex_sha256(&body) != request.archive.manifest.sha256
    {
        return Err(err(
            "diagnostic manifest digest differs from the request pin",
        ));
    }
    let manifest: Manifest = serde_json::from_slice(&body)?;
    let start = time(&manifest.precipitation_accumulation_start)?;
    let end = time(&request.valid_time)?;
    let elapsed = (end - start)
        .num_microseconds()
        .ok_or_else(|| err("diagnostic time overflow"))?;
    if manifest.schema != CONTRACT
        || manifest.forecast_volume_fields
        || manifest.fields != fields()
        || elapsed < 0
        || elapsed % 3_600_000_000 != 0
        || end.timestamp_subsec_nanos() != 0
        || start.timestamp_subsec_nanos() != 0
    {
        return Err(err(
            "diagnostic schema, fields, or exact hourly timeline differs",
        ));
    }
    let order = request
        .members
        .iter()
        .map(|m| m.member_id)
        .collect::<Vec<_>>();
    if order.is_empty()
        || order != manifest.member_order
        || order.iter().collect::<BTreeSet<_>>().len() != order.len()
        || manifest.member_metadata.len() != order.len()
    {
        return Err(err(
            "diagnostic request must preserve the exact complete archive member order",
        ));
    }
    let mut metadata = BTreeMap::new();
    for row in &manifest.member_metadata {
        let id = row
            .get("member_id")
            .and_then(Value::as_u64)
            .ok_or_else(|| err("diagnostic provenance has no uint64 member_id"))?;
        if row.get("seed").and_then(Value::as_u64).is_none() || metadata.insert(id, row).is_some() {
            return Err(err(
                "diagnostic provenance seeds and member identities must be exact and unique",
            ));
        }
    }
    let mut entry_keys = BTreeSet::new();
    let mut paths = BTreeSet::new();
    for entry in &manifest.files {
        let entry_time = time(&entry.valid_time)?;
        let entry_elapsed = (entry_time - start)
            .num_microseconds()
            .ok_or_else(|| err("diagnostic time overflow"))?;
        if !metadata.contains_key(&entry.member_id)
            || !digest(&entry.sha256)
            || !digest(&entry.geometry_sha256)
            || entry.bytes == 0
            || metadata[&entry.member_id]["seed"].as_u64() != Some(entry.seed)
            || entry_elapsed < 0
            || entry_elapsed % 3_600_000_000 != 0
            || entry_time.timestamp_subsec_nanos() != 0
            || !entry_keys.insert((entry.member_id, entry.grid_id, entry.episode, entry_time))
            || !paths.insert(&entry.path)
        {
            return Err(err(
                "diagnostic manifest contains an invalid or duplicate member/grid/time entry",
            ));
        }
    }
    for item in &manifest.unavailable {
        if item.grid_id == request.archive.grid_id
            && item.episode == request.archive.episode
            && time(&item.valid_time)? == end
            && order.contains(&item.member_id)
        {
            return Err(err(
                "requested diagnostic member/grid/time is explicitly unavailable",
            ));
        }
    }
    let root = request.archive.root.canonicalize()?;
    if !request
        .archive
        .manifest
        .path
        .canonicalize()?
        .starts_with(&root)
    {
        return Err(err("diagnostic manifest leaves the archive root"));
    }
    let mut selected = Vec::new();
    let mut members = Vec::new();
    for member in &request.members {
        let provenance = metadata
            .get(&member.member_id)
            .ok_or_else(|| err("requested diagnostic member is absent"))?;
        if provenance["seed"].as_u64() != Some(member.seed) {
            return Err(err("diagnostic member seed differs from the request"));
        }
        let entry = manifest
            .files
            .iter()
            .find(|entry| {
                entry.member_id == member.member_id
                    && entry.grid_id == request.archive.grid_id
                    && entry.episode == request.archive.episode
                    && time(&entry.valid_time).is_ok_and(|t| t == end)
            })
            .ok_or_else(|| {
                err(format!(
                    "missing diagnostic member {} at requested grid, episode and time",
                    member.member_id
                ))
            })?;
        let path = archive_path(&root, &entry.path)?;
        members.push(Member {
            id: member.id.clone(),
            file: path.clone(),
            initial_file: None,
        });
        selected.push(Selected {
            label: member.id.clone(),
            path,
            entry: entry.clone(),
            provenance: (*provenance).clone(),
        });
    }
    let start = seam_time(start);
    let end = seam_time(end);
    Ok((
        Match {
            plan: request.plan.clone(),
            quantity: request.quantity.clone(),
            valid_time: end.clone(),
            precipitation_observation: request.precipitation_observation.clone(),
            members,
        },
        Archive {
            manifest: request.archive.manifest.clone(),
            start,
            end,
            selected,
        },
    ))
}
fn attribute(file: &NcFile, name: &str) -> Result<String> {
    file.global_attribute(name)?
        .value
        .as_string()
        .ok_or_else(|| err(format!("diagnostic attribute {name} must be text")))
}
fn variable(
    file: &NcFile,
    name: &str,
    dtype: NcType,
    dims: &[(&str, usize)],
    units: Option<&str>,
) -> Result<()> {
    let var = file.variable(name)?;
    if var.dtype != dtype
        || var.dimensions.len() != dims.len()
        || var
            .dimensions
            .iter()
            .zip(dims)
            .any(|(d, (name, size))| d.name != *name || d.size != *size as u64 || d.is_unlimited)
        || units.is_some_and(|units| {
            var.attribute("units")
                .and_then(|a| a.value.as_string())
                .as_deref()
                != Some(units)
        })
        || ["scale_factor", "add_offset"]
            .iter()
            .any(|name| var.attribute(name).is_some())
    {
        return Err(err(format!("diagnostic variable {name} has wrong dtype, dimensions, units or packed-value transform")));
    }
    Ok(())
}
fn f32_values(file: &NcFile, name: &str) -> Result<Vec<f32>> {
    Ok(file.read_variable::<f32>(name)?.into_raw_vec_and_offset().0)
}
fn geometry_f32(lat: &[f32], lon: &[f32]) -> String {
    hex_sha256(
        &lat.iter()
            .chain(lon)
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}
fn flush_subnormal(value: f32) -> f32 {
    if value.is_subnormal() {
        f32::from_bits(value.to_bits() & 0x8000_0000)
    } else {
        value
    }
}
fn wind_f32(u: f32, v: f32) -> f32 {
    // The archive's production RawModule compiles with -ftz=true. Preserve
    // its signed-zero input/output flushing at every intrinsic operation,
    // with separate round-to-nearest f32 operations and no FMA contraction.
    let u = flush_subnormal(u);
    let v = flush_subnormal(v);
    let uu = flush_subnormal(u * u);
    let vv = flush_subnormal(v * v);
    let sum = flush_subnormal(flush_subnormal(uu) + flush_subnormal(vv));
    flush_subnormal(flush_subnormal(sum).sqrt())
}

#[cfg(test)]
mod arithmetic_tests {
    use super::*;

    #[test]
    fn production_ftz_preserves_signs_and_normal_boundary() {
        for bits in [1, 0x007f_ffff, 0x8000_0001, 0x807f_ffff] {
            assert_eq!(
                flush_subnormal(f32::from_bits(bits)).to_bits(),
                bits & 0x8000_0000
            );
        }
        for bits in [
            0,
            0x8000_0000,
            0x0080_0000,
            0x8080_0000,
            0x7f80_0000,
            0xff80_0000,
        ] {
            assert_eq!(flush_subnormal(f32::from_bits(bits)).to_bits(), bits);
        }
        // 2^-63 squares to the minimum normal f32. Its lower neighbor's
        // square is subnormal and the production intrinsic flushes it.
        assert_eq!(wind_f32(f32::from_bits(0x1fff_ffff), 0.0).to_bits(), 0);
        assert_eq!(
            wind_f32(f32::from_bits(0x2000_0000), 0.0).to_bits(),
            0x2000_0000
        );
        assert_eq!(
            wind_f32(f32::from_bits(0x2000_0001), 0.0).to_bits(),
            0x2000_0001
        );
        assert_eq!(wind_f32(0.0, f32::from_bits(0x1a80_0000)).to_bits(), 0);
        assert_eq!(wind_f32(-0.0, -0.0).to_bits(), 0);
    }

    #[test]
    fn production_wind_normal_thresholds_and_nonfinite_classification() {
        assert_eq!(wind_f32(3.0, 4.0).to_bits(), 5.0f32.to_bits());
        assert_eq!(wind_f32(1.0, 1.0).to_bits(), 0x3fb5_04f3);
        for bits in [
            10.0f32.to_bits() - 1,
            10.0f32.to_bits(),
            10.0f32.to_bits() + 1,
        ] {
            assert_eq!(wind_f32(f32::from_bits(bits), 0.0).to_bits(), bits);
        }
        assert_eq!(wind_f32(f32::MAX, 0.0), f32::INFINITY);
        assert_eq!(wind_f32(f32::NEG_INFINITY, 1.0), f32::INFINITY);
        assert!(wind_f32(f32::NAN, 1.0).is_nan());
        assert!(wind_f32(f32::INFINITY, f32::NAN).is_nan());
    }
}
impl Archive {
    pub fn field(
        &self,
        member: usize,
        plan: &Plan,
        quantity: &str,
        precipitation_start: Option<&str>,
    ) -> Result<(Vec<f64>, Value)> {
        let item = &self.selected[member];
        let bytes = std::fs::read(&item.path)?;
        if bytes.len() as u64 != item.entry.bytes || hex_sha256(&bytes) != item.entry.sha256 {
            return Err(err(
                "diagnostic file bytes or digest differ from the pinned manifest",
            ));
        }
        // Parse the bytes just verified rather than reopening a mutable path.
        let file = NcFile::from_bytes(&bytes)?;
        if file.format() != NcFormat::Cdf5
            || attribute(&file, "member_diagnostic_contract")? != CONTRACT
            || time(&attribute(&file, "valid_time")?)? != time(&self.end)?
            || time(&attribute(&file, "precipitation_accumulation_start")?)? != time(&self.start)?
        {
            return Err(err(
                "diagnostic file format, contract or time differs from its manifest",
            ));
        }
        for (name, expected) in [
            ("member_id", item.entry.member_id),
            ("member_seed", item.entry.seed),
        ] {
            variable(&file, name, NcType::UInt64, &[("member", 1)], None)?;
            let words = file.read_variable::<u64>(name)?.into_raw_vec_and_offset().0;
            if words != vec![expected] {
                return Err(err(format!(
                    "diagnostic native uint64 {name} differs from its manifest"
                )));
            }
        }
        if serde_json::from_str::<Value>(&attribute(&file, "member_provenance")?)?
            != item.provenance
        {
            return Err(err("diagnostic file provenance differs from its manifest"));
        }
        let plane = [("south_north", plan.ny), ("west_east", plan.nx)];
        variable(&file, "XLAT", NcType::Float, &plane, Some("degrees_north"))?;
        variable(&file, "XLONG", NcType::Float, &plane, Some("degrees_east"))?;
        let lat = f32_values(&file, "XLAT")?;
        let lon = f32_values(&file, "XLONG")?;
        let geometry = geometry_f32(&lat, &lon);
        let lat64 = lat.iter().map(|v| f64::from(*v)).collect::<Vec<_>>();
        let lon64 = lon.iter().map(|v| f64::from(*v)).collect::<Vec<_>>();
        if geometry != item.entry.geometry_sha256
            || attribute(&file, "geometry_sha256")? != geometry
            || geometry_hash(&lat64, &lon64) != plan.grid_geometry_sha256
        {
            return Err(err(
                "diagnostic coordinates differ from the frozen reference grid or manifest",
            ));
        }
        let dims = [
            ("member", 1),
            ("south_north", plan.ny),
            ("west_east", plan.nx),
        ];
        for (name, units) in fields() {
            variable(&file, &name, NcType::Float, &dims, Some(&units))?;
        }
        let (values, arithmetic) = match quantity {
            "temperature_2m" => (f32_values(&file, "T2")?, "original T2 float32 words"),
            "wind_speed_10m" => {
                let u = f32_values(&file, "U10")?;
                let v = f32_values(&file, "V10")?;
                (
                    u.into_iter().zip(v).map(|(u, v)| wind_f32(u, v)).collect(),
                    "f32 multiply/add/sqrt round-to-nearest with signed-zero FTZ at each operation before interpolation",
                )
            }
            "precipitation_accumulation" => {
                if precipitation_start.map(time).transpose()?.as_ref() != Some(&time(&self.start)?)
                {
                    return Err(err("RAIN_TOTAL requires the exact forecast-start observation accumulation window"));
                }
                (
                    f32_values(&file, "RAIN_TOTAL")?,
                    "original forecast-start RAIN_TOTAL float32 words, no endpoint subtraction",
                )
            }
            _ => return Err(err("unsupported diagnostic quantity")),
        };
        Ok((
            values.into_iter().map(f64::from).collect(),
            json!({"member_id":item.label,
            "native_member_id":item.entry.member_id,"seed":item.entry.seed,"manifest":self.manifest,
            "entry":item.entry,"frame":{"path":item.path,"sha256":item.entry.sha256},
                "member_provenance":item.provenance,"diagnostic_arithmetic":arithmetic,
                "wind_compile_contract":if quantity == "wind_speed_10m" {json!({"raw_module_options":["--std=c++17"],
                    "effective_nvrtc_option":"-ftz=true","rounding":"round-to-nearest, ties-to-even",
                    "fma_contraction":false,"nonfinite":"any nonfinite wind becomes an unavailable matched member value; NaN payloads are not compared"})} else {Value::Null}}),
        ))
    }
}
