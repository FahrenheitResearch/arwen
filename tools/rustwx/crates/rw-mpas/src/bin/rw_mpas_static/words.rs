//! Proof-only raw stored-word comparison of complete classic NetCDF artifacts.
//! No floating conversion or tolerance. --gwd explicitly proves the complete
//! ten-field i3 overlay; the default still requires every variable and dimension.
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

fn width(ty: u32) -> Result<u64, String> {
    match ty {
        1 | 2 | 7 => Ok(1),
        3 | 8 => Ok(2),
        4 | 5 | 9 => Ok(4),
        6 | 10 | 11 => Ok(8),
        _ => Err(format!("invalid classic type {ty}")),
    }
}
fn add(a: u64, b: u64) -> Result<u64, String> {
    a.checked_add(b).ok_or("classic size overflow".into())
}
fn mul(a: u64, b: u64) -> Result<u64, String> {
    a.checked_mul(b).ok_or("classic size overflow".into())
}
struct Reader {
    file: File,
    version: u8,
    length: u64,
}
impl Reader {
    fn integer(&mut self, bytes: usize) -> Result<u64, String> {
        let mut b = [0; 8];
        self.file
            .read_exact(&mut b[8 - bytes..])
            .map_err(|e| e.to_string())?;
        Ok(u64::from_be_bytes(b))
    }
    fn u32(&mut self) -> Result<u32, String> {
        self.integer(4).map(|x| x as u32)
    }
    fn count(&mut self) -> Result<u64, String> {
        self.integer(if self.version == 5 { 8 } else { 4 })
    }
    fn skip(&mut self, n: u64) -> Result<(), String> {
        let end = add(self.file.stream_position().map_err(|e| e.to_string())?, n)?;
        if end > self.length {
            return Err("truncated classic header".into());
        }
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
        let mut b = vec![0; usize::try_from(n).map_err(|e| e.to_string())?];
        self.file.read_exact(&mut b).map_err(|e| e.to_string())?;
        self.skip((4 - n % 4) % 4)?;
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
            let size = mul(width(ty)?, n)?;
            self.skip(add(size, (4 - size % 4) % 4)?)?;
        }
        Ok(())
    }
}
#[derive(Debug, PartialEq, Eq, Clone)]
struct Dim {
    name: String,
    length: u64,
    unlimited: bool,
}
#[derive(Debug)]
struct Var {
    ty: u32,
    dims: Vec<Dim>,
    begin: u64,
    slab: u64,
    padded: u64,
    unpadded_vsize: bool,
    records: u64,
    stride: u64,
}
struct Classic {
    file: File,
    dimensions: BTreeMap<String, (u64, bool)>,
    vars: BTreeMap<String, Var>,
}
fn open(path: &Path) -> Result<Classic, String> {
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    let length = file.metadata().map_err(|e| e.to_string())?.len();
    let mut magic = [0; 4];
    file.read_exact(&mut magic).map_err(|e| e.to_string())?;
    if &magic[..3] != b"CDF" || ![1, 2, 5].contains(&magic[3]) {
        return Err("requires CDF1/2/5 storage".into());
    }
    let mut r = Reader {
        file,
        version: magic[3],
        length,
    };
    let records = r.count()?;
    if records
        == if r.version == 5 {
            u64::MAX
        } else {
            u32::MAX as u64
        }
    {
        return Err("streaming record count must be finalized before proof".into());
    }
    let mut dimensions = BTreeMap::new();
    let mut dims = Vec::new();
    for _ in 0..r.list(10)? {
        let name = r.name()?;
        let n = r.count()?;
        let dim = Dim {
            name: name.clone(),
            length: if n == 0 { records } else { n },
            unlimited: n == 0,
        };
        if dimensions
            .insert(name, (dim.length, dim.unlimited))
            .is_some()
        {
            return Err("duplicate dimension".into());
        }
        dims.push(dim);
    }
    if dims.iter().filter(|d| d.unlimited).count() > 1 {
        return Err("multiple classic record dimensions".into());
    }
    r.attrs()?;
    let mut vars = BTreeMap::new();
    for _ in 0..r.list(11)? {
        let name = r.name()?;
        let n = r.count()?;
        if n > length / if r.version == 5 { 8 } else { 4 } {
            return Err("invalid variable dimension count".into());
        }
        let mut vd = Vec::new();
        for _ in 0..n {
            let id = r.count()?;
            vd.push(
                dims.get(usize::try_from(id).map_err(|e| e.to_string())?)
                    .ok_or("invalid dimension ID")?
                    .clone(),
            );
        }
        if vd.iter().skip(1).any(|d| d.unlimited) {
            return Err("record dimension must be first".into());
        }
        r.attrs()?;
        let ty = r.u32()?;
        let stored = r.count()?;
        let begin = r.integer(if r.version == 1 { 4 } else { 8 })?;
        let record = vd.first().is_some_and(|d| d.unlimited);
        let slab = vd
            .iter()
            .skip(usize::from(record))
            .try_fold(width(ty)?, |n, d| mul(n, d.length))?;
        let padded = add(slab, (4 - slab % 4) % 4)?;
        // CDF1/2 encode large vsize with a 32-bit all-ones sentinel.
        if stored != padded
            && !(record && width(ty)? <= 2 && stored == slab)
            && !(r.version != 5 && stored == u32::MAX as u64 && padded >= u32::MAX as u64)
        {
            return Err(format!("{name}: invalid stored vsize"));
        }
        if vars
            .insert(
                name,
                Var {
                    ty,
                    dims: vd,
                    begin,
                    slab,
                    padded,
                    unpadded_vsize: stored == slab && slab != padded,
                    records: if record { records } else { 1 },
                    stride: 0,
                },
            )
            .is_some()
        {
            return Err("duplicate variable".into());
        }
    }
    let header_end = r.file.stream_position().map_err(|e| e.to_string())?;
    let record_vars: Vec<_> = vars
        .values()
        .filter(|v| v.dims.first().is_some_and(|d| d.unlimited))
        .collect();
    let stride = if record_vars.len() == 1 && width(record_vars[0].ty)? <= 2 {
        record_vars[0].slab
    } else {
        record_vars.iter().try_fold(0, |n, v| add(n, v.padded))?
    };
    if record_vars.len() != 1 && record_vars.iter().any(|v| v.unpadded_vsize) {
        return Err("unpadded vsize requires a sole sub-word record variable".into());
    }
    for (name, v) in &mut vars {
        if v.dims.first().is_some_and(|d| d.unlimited) {
            v.stride = stride;
        }
        if v.begin < header_end {
            return Err(format!("{name}: payload overlaps header"));
        }
        let end = if v.records == 0 {
            v.begin
        } else {
            add(add(v.begin, mul(v.records - 1, v.stride)?)?, v.slab)?
        };
        if end > length {
            return Err(format!("{name}: truncated payload"));
        }
    }
    Ok(Classic {
        file: r.file,
        dimensions,
        vars,
    })
}
const GWD_FIELDS: [&str; 10] = ["var2d", "con", "oa1", "oa2", "oa3", "oa4", "ol1", "ol2", "ol3", "ol4"];

fn compare(reference: &Path, candidate: &Path) -> Result<Value, String> {
    compare_scope(reference, candidate, false)
}
fn compare_scope(reference: &Path, candidate: &Path, gwd: bool) -> Result<Value, String> {
    compare_scopes(reference,candidate,gwd,false)
}
fn compare_scopes(reference: &Path, candidate: &Path, gwd: bool, complete_static: bool) -> Result<Value, String> {
    let mut a = open(reference)?;
    let mut b = open(candidate)?;
    if gwd && ((!complete_static && b.vars.len() != GWD_FIELDS.len()) || GWD_FIELDS.iter().any(|name| !b.vars.contains_key(*name) || !a.vars.contains_key(*name))) {
        return Err("GWD proof requires exactly all ten GWD variables in the candidate and all ten in native".into());
    }
    let mut fields = Vec::new();
    let mut differences = 0u64;
    let mut words = 0u64;
    let mut only_reference = Vec::new();
    let mut only_candidate = Vec::new();
    let mut schema_mismatches = Vec::new();
    for (name, av) in &a.vars {
        if gwd && !GWD_FIELDS.contains(&name.as_str()) { continue; }
        let Some(bv) = b.vars.get(name) else {
            only_reference.push(name.clone());
            continue;
        };
        if av.ty != bv.ty || av.dims != bv.dims {
            schema_mismatches.push(name.clone());
            continue;
        }
        let w = width(av.ty)? as usize;
        let mut count = 0;
        let mut different = 0;
        let mut ab = [0u8; 65536];
        let mut bb = [0u8; 65536];
        for record in 0..av.records {
            a.file
                .seek(SeekFrom::Start(add(av.begin, mul(record, av.stride)?)?))
                .map_err(|e| e.to_string())?;
            b.file
                .seek(SeekFrom::Start(add(bv.begin, mul(record, bv.stride)?)?))
                .map_err(|e| e.to_string())?;
            let mut remaining = av.slab;
            while remaining > 0 {
                let n = remaining.min(ab.len() as u64) as usize;
                a.file.read_exact(&mut ab[..n]).map_err(|e| e.to_string())?;
                b.file.read_exact(&mut bb[..n]).map_err(|e| e.to_string())?;
                different += ab[..n]
                    .chunks_exact(w)
                    .zip(bb[..n].chunks_exact(w))
                    .filter(|(x, y)| x != y)
                    .count() as u64;
                count += (n / w) as u64;
                remaining -= n as u64;
            }
        }
        words = add(words, count)?;
        differences = add(differences, different)?;
        fields.push(json!({"name":name,"nc_type":av.ty,"shape":av.dims.iter().map(|d|d.length).collect::<Vec<_>>(),"words":count,"differing_words":different}));
    }
    for name in b.vars.keys() {
        if gwd && complete_static && !GWD_FIELDS.contains(&name.as_str()) {continue;}
        if !a.vars.contains_key(name) {
            only_candidate.push(name.clone());
        }
    }
    let dimensions_agree = if gwd { a.dimensions.get("nCells") == b.dimensions.get("nCells") } else { a.dimensions == b.dimensions };
    let passed = dimensions_agree
        && only_reference.is_empty()
        && only_candidate.is_empty()
        && schema_mismatches.is_empty()
        && differences == 0
        && !fields.is_empty();
    Ok(
        json!({"schema":"i3-complete-static-stored-words/v1","passed":passed,
        "comparison":"raw stored type-width words; attributes not graded",
        "scope":if gwd {"all ten i3 GWD static fields"} else {"all variables and records"},
        "exclusions":[],"native_other_fields_are_outside_i3_scope":gwd,
        "reference":reference,"candidate":candidate,"reference_sha256":rw_mpas::sha256_file(reference).map_err(|e|e.to_string())?,
        "candidate_sha256":rw_mpas::sha256_file(candidate).map_err(|e|e.to_string())?,
        "dimensions_agree":dimensions_agree,"only_in_reference":only_reference,"only_in_candidate":only_candidate,
        "schema_mismatches":schema_mismatches,"words":words,"differing_words":differences,"fields":fields}),
    )
}
fn run(mut args: Vec<std::ffi::OsString>) -> Result<bool, String> {
    let complete_static=args.first().is_some_and(|arg|arg=="--gwd-static");
    let gwd = complete_static || args.first().is_some_and(|arg| arg == "--gwd");
    if gwd { args.remove(0); }
    if args.len() != 3 {
        return Err("usage: rw_mpas_static words [--gwd|--gwd-static] REFERENCE.nc CANDIDATE.nc RECEIPT.json".into());
    }
    let receipt = compare_scopes(Path::new(&args[0]), Path::new(&args[1]), gwd,complete_static)?;
    std::fs::write(&args[2], serde_json::to_vec_pretty(&receipt).unwrap())
        .map_err(|e| e.to_string())?;
    Ok(receipt["passed"] == true)
}
/// `rw_mpas_static words ...`, or this executable invoked as
/// `rw_mpas_static_words`.
pub fn main_with(args: Vec<std::ffi::OsString>) -> std::process::ExitCode {
    match run(args) {
        Ok(true) => std::process::ExitCode::SUCCESS,
        Ok(false) => std::process::ExitCode::FAILURE,
        Err(e) => {
            eprintln!("{e}");
            std::process::ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rw_store::netcdf_classic::{NcClassicWriter, NcData, NcDim, NcFormat, NcType, NcVarDef};
    fn write(path: &Path, format: NcFormat, changed: bool, sole: bool) {
        let mut vars = vec![NcVarDef::new("text", NcType::Char, vec![0, 1])];
        if !sole {
            vars.extend([
                NcVarDef::new("float", NcType::Float, vec![0, 1]),
                NcVarDef::new("double", NcType::Double, vec![1]),
                NcVarDef::new("integer", NcType::Int, vec![1]),
                NcVarDef::new("short", NcType::Short, vec![0, 1]),
            ]);
        }
        let mut w = NcClassicWriter::create(
            path,
            format,
            vec![NcDim::record("Time"), NcDim::fixed("nCells", 3)],
            vec![],
            vars,
            2,
        )
        .unwrap();
        for i in 0..2 {
            w.put_record(
                "text",
                i,
                NcData::Chars(if changed && i == 1 { b"aXd" } else { b"abc" }),
            )
            .unwrap();
            if !sole {
                w.put_record(
                    "float",
                    i,
                    NcData::Floats(&[
                        f32::from_bits(if changed { 2 } else { 1 }),
                        -0.,
                        f32::from_bits(if changed { 0x7fc00002 } else { 0x7fc00001 }),
                    ]),
                )
                .unwrap();
                w.put_record("short", i, NcData::Shorts(&[-7, 3, 11]))
                    .unwrap();
            }
        }
        if !sole {
            w.put(
                "double",
                NcData::Doubles(&[
                    f64::from_bits(1),
                    if changed { 0. } else { -0. },
                    f64::from_bits(0x7ff8000000000001),
                ]),
            )
            .unwrap();
            w.put(
                "integer",
                NcData::Ints(&[i32::MIN, if changed { 13 } else { 12 }, i32::MAX]),
            )
            .unwrap();
        }
        w.finish().unwrap();
    }
    #[test]
    fn all_formats_records_types_and_raw_payloads() {
        let dir = std::env::temp_dir()
            .join(format!("static-word-test-{}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        let a = dir.join("a.nc");
        let b = dir.join("b.nc");
        for format in [NcFormat::Classic, NcFormat::Offset64, NcFormat::Data64] {
            for sole in [false, true] {
                write(&a, format, false, sole);
                write(&b, format, false, sole);
                let r = compare(&a, &b).unwrap();
                assert_eq!(r["passed"], true);
                assert_eq!(r["words"], if sole { 6 } else { 24 });
                write(&b, format, true, sole);
                let r = compare(&a, &b).unwrap();
                assert_eq!(r["passed"], false);
                assert_eq!(r["differing_words"], if sole { 2 } else { 8 });
            }
        }
        write(&a, NcFormat::Classic, false, false);
        write(&b, NcFormat::Data64, false, false);
        assert_eq!(compare(&a, &b).unwrap()["passed"], true);
        write(&b, NcFormat::Data64, false, true);
        assert_eq!(
            compare(&a, &b).unwrap()["only_in_reference"]
                .as_array()
                .unwrap()
                .len(),
            4
        );
        File::options()
            .write(true)
            .open(&b)
            .unwrap()
            .set_len(20)
            .unwrap();
        assert!(compare(&a, &b).is_err());
        for (dim, ty) in [
            ("nCells", NcType::Int),
            ("otherCells", NcType::Int),
            ("nCells", NcType::Float),
        ] {
            let path = if dim == "nCells" && ty == NcType::Int {
                &a
            } else {
                &b
            };
            let mut w = NcClassicWriter::create(
                path,
                NcFormat::Classic,
                vec![NcDim::fixed(dim, 3)],
                vec![],
                vec![NcVarDef::new("x", ty, vec![0])],
                0,
            )
            .unwrap();
            if ty == NcType::Int {
                w.put("x", NcData::Ints(&[0, 0, 0])).unwrap();
            } else {
                w.put("x", NcData::Floats(&[0., 0., 0.])).unwrap();
            }
            w.finish().unwrap();
            if path == &b {
                assert_eq!(compare(&a, &b).unwrap()["schema_mismatches"], json!(["x"]));
            }
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn gwd_inventory_and_signed_zero_are_required() {
        let dir = std::env::temp_dir()
            .join(format!("gwd-word-scope-test-{}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        let a=dir.join("a.nc");let b=dir.join("b.nc");
        let emit=|path:&Path, extra:bool, missing:bool, changed:bool| {
            let mut vars:Vec<_>=GWD_FIELDS.iter().filter(|n| !(missing && **n=="ol4"))
                .map(|n| NcVarDef::new(*n,NcType::Float,vec![0])).collect();
            if extra {vars.push(NcVarDef::new("other_static",NcType::Float,vec![0]));}
            let names:Vec<_>=vars.iter().map(|v|v.name.clone()).collect();
            let mut w=NcClassicWriter::create(path,NcFormat::Classic,vec![NcDim::fixed("nCells",2)],vec![],vars,0).unwrap();
            for name in names {w.put(&name,NcData::Floats(&[0.0,if changed && name=="ol4" {0.0} else {-0.0}])).unwrap();}
            w.finish().unwrap();
        };
        emit(&a,true,false,false);emit(&b,false,false,false);
        assert_eq!(compare_scope(&a,&b,true).unwrap()["passed"],true);
        assert_eq!(compare_scope(&a,&b,true).unwrap()["words"],20);
        assert_eq!(compare(&a,&b).unwrap()["passed"],false);
        emit(&b,false,false,true);
        assert_eq!(compare_scope(&a,&b,true).unwrap()["differing_words"],1);
        assert_eq!(compare_scope(&a,&b,true).unwrap()["passed"],false);
        emit(&b,false,true,false);assert!(compare_scope(&a,&b,true).is_err());
        emit(&b,true,false,false);assert!(compare_scope(&a,&b,true).is_err());
        assert_eq!(compare_scopes(&a,&b,true,true).unwrap()["passed"],true);
        emit(&b,true,false,true);
        assert_eq!(compare_scopes(&a,&b,true,true).unwrap()["differing_words"],1);
        assert_eq!(compare_scopes(&a,&b,true,true).unwrap()["passed"],false);
        emit(&b,true,true,false);assert!(compare_scopes(&a,&b,true,true).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

}
