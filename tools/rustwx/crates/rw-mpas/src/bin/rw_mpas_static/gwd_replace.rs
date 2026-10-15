//! Forecast-test composition: copy an existing complete init and replace only
//! the ten fields produced by MPAS-A v8.4.2 mpas_init_atm_gwd.F:102-333.
//! This is a native CPU stand-in for the other init lanes, not a full-init port.
use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
};
const FIELDS: [&str; 10] = [
    "var2d", "con", "oa1", "oa2", "oa3", "oa4", "ol1", "ol2", "ol3", "ol4",
];

// Classic fixed arrays are stored as contiguous big-endian words. Preserve the
// complete native header and unrelated payload while replacing only these arrays.
struct Header {
    file: File,
    cdf5: bool,
    version: u8,
    length: u64,
}
impl Header {
    fn u32(&mut self) -> Result<u32, String> {
        let mut b = [0; 4];
        self.file.read_exact(&mut b).map_err(|e| e.to_string())?;
        Ok(u32::from_be_bytes(b))
    }
    fn u64(&mut self) -> Result<u64, String> {
        let mut b = [0; 8];
        self.file.read_exact(&mut b).map_err(|e| e.to_string())?;
        Ok(u64::from_be_bytes(b))
    }
    fn count(&mut self) -> Result<u64, String> {
        if self.cdf5 {
            self.u64()
        } else {
            self.u32().map(u64::from)
        }
    }
    fn skip(&mut self, n: u64) -> Result<(), String> {
        let pos = self.file.stream_position().map_err(|e| e.to_string())?;
        let end = pos
            .checked_add(n)
            .filter(|&e| e <= self.length)
            .ok_or("truncated classic header")?;
        self.file
            .seek(SeekFrom::Start(end))
            .map_err(|e| e.to_string())?;
        Ok(())
    }
    fn name(&mut self) -> Result<String, String> {
        let n = self.count()?;
        if n > self.length {
            return Err("truncated classic name".into());
        }
        let n = usize::try_from(n).map_err(|e| e.to_string())?;
        let mut b = vec![0; n];
        self.file.read_exact(&mut b).map_err(|e| e.to_string())?;
        self.skip(((4 - n % 4) % 4) as u64)?;
        String::from_utf8(b).map_err(|e| e.to_string())
    }
    fn list(&mut self, tag: u32) -> Result<u64, String> {
        let t = self.u32()?;
        let n = self.count()?;
        if (t != tag && !(t == 0 && n == 0)) || n > self.length {
            return Err("invalid classic header list".into());
        }
        Ok(n)
    }
    fn attrs(&mut self) -> Result<(), String> {
        for _ in 0..self.list(12)? {
            self.name()?;
            let ty = self.u32()?;
            let n = self.count()?;
            let width = match ty {
                1 | 2 | 7 => 1,
                3 | 8 => 2,
                4 | 5 | 9 => 4,
                6 | 10 | 11 => 8,
                _ => return Err("invalid classic attribute type".into()),
            };
            let size = n
                .checked_mul(width)
                .ok_or("classic attribute length overflow")?;
            self.skip(
                size.checked_add((4 - size % 4) % 4)
                    .ok_or("classic attribute padding overflow")?,
            )?;
        }
        Ok(())
    }
}
fn offsets(path: &Path) -> Result<HashMap<String, u64>, String> {
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    let length = file.metadata().map_err(|e| e.to_string())?.len();
    let mut magic = [0; 4];
    file.read_exact(&mut magic).map_err(|e| e.to_string())?;
    if &magic[..3] != b"CDF" || ![1, 2, 5].contains(&magic[3]) {
        return Err("composition needs classic CDF1/2/5 fixed-array storage".into());
    }
    let mut h = Header {
        file,
        cdf5: magic[3] == 5,
        version: magic[3],
        length,
    };
    h.count()?;
    for _ in 0..h.list(10)? {
        h.name()?;
        h.count()?;
    }
    h.attrs()?;
    let mut result = HashMap::new();
    for _ in 0..h.list(11)? {
        let name = h.name()?;
        let n = h.count()?;
        h.skip(
            n.checked_mul(if h.cdf5 { 8 } else { 4 })
                .ok_or("classic dimension count overflow")?,
        )?;
        h.attrs()?;
        h.u32()?;
        h.count()?;
        let begin = if h.version == 1 {
            u64::from(h.u32()?)
        } else {
            h.u64()?
        };
        if begin > length {
            return Err("classic variable begins beyond file".into());
        }
        result.insert(name, begin);
    }
    Ok(result)
}
fn field(file: &netcrust::File, name: &str) -> Result<Vec<f32>, String> {
    let v = file
        .variable(name)
        .ok_or_else(|| format!("missing GWD variable: {name}"))?;
    if v.dtype() != &netcrust::DataType::F32
        || v.dimensions().len() != 1
        || v.dimensions()[0].name() != "nCells"
        || v.dimensions()[0].is_unlimited()
    {
        return Err(format!(
            "{name}: composition requires fixed NC_FLOAT(nCells) storage"
        ));
    }
    Ok(file
        .read_array::<f32>(name)
        .map_err(|e| e.to_string())?
        .iter()
        .copied()
        .collect())
}
fn compose(base: &Path, overlay: &Path, out: &Path) -> Result<serde_json::Value, String> {
    if out.exists() {
        return Err("output already exists".into());
    }
    let native = netcrust::File::open(base).map_err(|e| e.to_string())?;
    let positions = offsets(base)?;
    let port = netcrust::File::open(overlay).map_err(|e| e.to_string())?;
    let mut fields = Vec::new();
    let mut replacements = Vec::new();
    // Validate every array before creating an output. No computed word is excluded.
    for name in FIELDS {
        let before = field(&native, name)?;
        let values = field(&port, name)?;
        if before.len() != values.len() {
            return Err(format!("{name}: different nCells lengths"));
        }
        let differences = before
            .iter()
            .zip(&values)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        fields.push(serde_json::json!({"field":name,"words":values.len(),"differences_from_native":differences}));
        replacements.push((name, values));
    }
    // Published reference inputs are read-only. Create an owned writable output
    // instead of inheriting their permissions through fs::copy.
    let mut output = File::create_new(out).map_err(|e| e.to_string())?;
    std::io::copy(
        &mut File::open(base).map_err(|e| e.to_string())?,
        &mut output,
    )
    .map_err(|e| e.to_string())?;
    for (name, values) in replacements {
        let begin = *positions
            .get(name)
            .ok_or_else(|| format!("missing classic offset: {name}"))?;
        let end = begin
            .checked_add((values.len() as u64) * 4)
            .ok_or("classic field length overflow")?;
        if end > output.metadata().map_err(|e| e.to_string())?.len() {
            return Err("classic field exceeds file length".into());
        }
        output
            .seek(SeekFrom::Start(begin))
            .map_err(|e| e.to_string())?;
        for value in values {
            output
                .write_all(&value.to_bits().to_be_bytes())
                .map_err(|e| e.to_string())?;
        }
    }
    drop(output);
    let hash = |p: &Path| rw_mpas::sha256_file(p).map_err(|e| e.to_string());
    Ok(
        serde_json::json!({"schema":"native-gwd-forecast-composition/v1",
        "role":"ten Rust GWD fields; other init fields supplied by pristine native stand-in",
        "complete_init_port":false,"process_count":1,"base":base,"overlay":overlay,"output":out,
        "base_sha256":hash(base)?,"overlay_sha256":hash(overlay)?,"output_sha256":hash(out)?,"fields":fields}),
    )
}
fn run(args: Vec<std::ffi::OsString>) -> Result<(), String> {
    if args.len() != 4 {
        return Err("usage: rw_mpas_static gwd-replace BASE.nc GWD.nc OUTPUT.nc RECEIPT.json".into());
    }
    let receipt = compose(
        Path::new(&args[0]),
        Path::new(&args[1]),
        Path::new(&args[2]),
    )?;
    std::fs::write(&args[3], serde_json::to_vec_pretty(&receipt).unwrap())
        .map_err(|e| e.to_string())
}
/// `rw_mpas_static gwd-replace ...`, or this executable invoked as
/// `rw_mpas_gwd_replace`.
pub fn main_with(args: Vec<std::ffi::OsString>) -> std::process::ExitCode {
    if let Err(error) = run(args) {
        eprintln!("{error}");
        return std::process::ExitCode::FAILURE;
    }
    std::process::ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;
    use rw_store::netcdf_classic::{
        NcAttr, NcClassicWriter, NcData, NcDim, NcFormat, NcType, NcVarDef,
    };
    #[test]
    fn replaces_every_gwd_word_and_preserves_other_fields() {
        // TMPDIR when set (a lane points it at its data disk), else the system
        // temporary directory, so the battery's cargo leg runs these too.
        let root = std::env::temp_dir();
        let dir =
            std::path::PathBuf::from(root).join(format!("gwd-replace-test-{}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        for format in [NcFormat::Classic, NcFormat::Offset64, NcFormat::Data64] {
            let base = dir.join("base.nc");
            let overlay = dir.join("overlay.nc");
            let out = dir.join("out.nc");
            for (path, replacement) in [(&base, false), (&overlay, true)] {
                let mut vars: Vec<_> = FIELDS
                    .iter()
                    .map(|&n| NcVarDef::new(n, NcType::Float, vec![0]))
                    .collect();
                vars.push(NcVarDef::new("unchanged", NcType::Int, vec![0]));
                let mut w = NcClassicWriter::create(
                    path,
                    format,
                    vec![NcDim::fixed("nCells", 4)],
                    vec![NcAttr::text("provenance", "kept verbatim")],
                    vars,
                    0,
                )
                .unwrap();
                let values = if replacement {
                    [f32::from_bits(1), -0.0, 3.5, -2.0]
                } else {
                    [0.0; 4]
                };
                for name in FIELDS {
                    w.put(name, NcData::Floats(&values)).unwrap();
                }
                w.put("unchanged", NcData::Ints(&[7, 9, -2, 11])).unwrap();
                w.finish().unwrap();
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&base, std::fs::Permissions::from_mode(0o444))
                    .unwrap();
            }
            let r = compose(&base, &overlay, &out).unwrap();
            assert_eq!(r["complete_init_port"], false);
            let f = netcrust::File::open(&out).unwrap();
            for name in FIELDS {
                let v = f.read_array::<f32>(name).unwrap();
                assert_eq!(
                    v.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    vec![
                        1,
                        (-0.0f32).to_bits(),
                        3.5f32.to_bits(),
                        (-2.0f32).to_bits()
                    ]
                );
            }
            assert_eq!(f.read_f64("unchanged").unwrap(), vec![7.0, 9.0, -2.0, 11.0]);
            assert!(compose(&base, &overlay, &out).is_err());
            drop(f);
            for path in [&base, &overlay, &out] {
                std::fs::remove_file(path).unwrap();
            }
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
}
