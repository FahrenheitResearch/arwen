//! Registry pool transport. MPAS: framework/mpas_io_streams.F:925-1023,
//! framework/mpas_pool_routines.F:2110-3143. No scheme arithmetic.
use serde::{Deserialize, Serialize};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::{
    ffi::{CStr, CString},
    fs,
    io::Write,
    os::raw::{c_char, c_int},
    path::Path,
};
// Built only with the `system-netcdf` feature (Cargo.toml required-features),
// which links the system netCDF C library; a default `cargo test -p rw-mpas`
// builds neither this binary nor the link.
#[cfg_attr(feature = "system-netcdf", link(name = "netcdf"))]
unsafe extern "C" {
    fn nc_open(p: *const c_char, m: c_int, id: *mut c_int) -> c_int;
    fn nc_close(id: c_int) -> c_int;
    fn nc_strerror(c: c_int) -> *const c_char;
    fn nc_inq_varid(id: c_int, n: *const c_char, v: *mut c_int) -> c_int;
    fn nc_inq_ndims(id: c_int, r: *mut c_int) -> c_int;
    fn nc_inq_varndims(id: c_int, v: c_int, r: *mut c_int) -> c_int;
    fn nc_inq_var(
        id: c_int,
        v: c_int,
        n: *mut c_char,
        k: *mut c_int,
        r: *mut c_int,
        d: *mut c_int,
        a: *mut c_int,
    ) -> c_int;
    fn nc_inq_dim(id: c_int, d: c_int, n: *mut c_char, l: *mut usize) -> c_int;
    fn nc_get_vara_float(
        id: c_int,
        v: c_int,
        s: *const usize,
        c: *const usize,
        p: *mut f32,
    ) -> c_int;
    fn nc_get_vara_int(id: c_int, v: c_int, s: *const usize, c: *const usize, p: *mut i32)
    -> c_int;
    fn nc_get_vara_text(
        id: c_int,
        v: c_int,
        s: *const usize,
        c: *const usize,
        p: *mut c_char,
    ) -> c_int;
}
fn check(c: c_int) -> Result<(), String> {
    if c == 0 {
        Ok(())
    } else {
        Err(unsafe { CStr::from_ptr(nc_strerror(c)) }
            .to_string_lossy()
            .into_owned())
    }
}
struct File {
    id: c_int,
    path: String,
}
impl File {
    fn open(path: &str) -> Result<Self, String> {
        let mut id = 0;
        check(unsafe {
            nc_open(
                CString::new(path).map_err(|e| e.to_string())?.as_ptr(),
                0,
                &mut id,
            )
        })?;
        Ok(Self {
            id,
            path: path.into(),
        })
    }
}
impl Drop for File {
    fn drop(&mut self) {
        unsafe {
            nc_close(self.id);
        }
    }
}
#[derive(Deserialize)]
struct Request {
    schema: String,
    sources: Vec<String>,
    fields: Vec<Field>,
}
#[derive(Deserialize)]
struct Field {
    pool: String,
    name: String,
    kind: String,
    dimensions: Vec<String>,
    shape: Vec<usize>,
    #[serde(default)]
    components: Vec<String>,
    #[serde(default)]
    initial: String,
    #[serde(default)]
    required: bool,
    #[serde(default)]
    read_input: bool,
}
#[derive(Serialize)]
struct Receipt {
    schema: &'static str,
    fields: Vec<Record>,
}
#[derive(Serialize)]
struct Record {
    pool: String,
    name: String,
    kind: String,
    dimensions: Vec<String>,
    shape: Vec<usize>,
    payload: Option<String>,
    sha256: Option<String>,
    text: Option<String>,
    origins: Vec<String>,
}
fn read(
    files: &[File],
    name: &str,
    kind: &str,
) -> Result<Option<(Vec<u8>, Vec<String>, Vec<usize>, String)>, String> {
    for f in files.iter().rev() {
        let mut v = 0;
        let n = CString::new(name).map_err(|e| e.to_string())?;
        if unsafe { nc_inq_varid(f.id, n.as_ptr(), &mut v) } != 0 {
            continue;
        }
        let (mut k, mut rank) = (0, 0);
        check(unsafe { nc_inq_varndims(f.id, v, &mut rank) })?;
        if !(0..=16).contains(&rank) {
            return Err(format!("{name}: invalid rank {rank}"));
        }
        let mut ds = vec![0; rank as usize];
        check(unsafe {
            nc_inq_var(
                f.id,
                v,
                std::ptr::null_mut(),
                &mut k,
                &mut rank,
                ds.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })?;
        let mut dims = Vec::new();
        let mut counts = Vec::new();
        for d in &ds[..rank as usize] {
            let mut label = [0 as c_char; 257];
            let mut len = 0;
            check(unsafe { nc_inq_dim(f.id, *d, label.as_mut_ptr(), &mut len) })?;
            dims.push(
                unsafe { CStr::from_ptr(label.as_ptr()) }
                    .to_string_lossy()
                    .into_owned(),
            );
            counts.push(len);
        }
        let starts = vec![0; counts.len()];
        if dims.first().map(String::as_str) == Some("Time") {
            if counts[0] == 0 {
                return Err(format!("{name}: empty Time"));
            }
            counts[0] = 1;
        }
        let len: usize = counts.iter().product();
        let mut bytes = Vec::new();
        match kind {
            "real" => {
                if k != 5 && k != 6 {
                    return Err(format!("{name}: expected NC_FLOAT/DOUBLE, got {k}"));
                }
                let mut b = vec![0f32; len];
                check(unsafe {
                    nc_get_vara_float(f.id, v, starts.as_ptr(), counts.as_ptr(), b.as_mut_ptr())
                })?;
                // Fixed-width stores the compiler turns into a plain copy on
                // little-endian hosts: this thread is the reader for every
                // worker, so it does no per-word appends.
                bytes.resize(len * 4, 0);
                for (word, x) in bytes.chunks_exact_mut(4).zip(&b) {
                    word.copy_from_slice(&x.to_bits().to_le_bytes());
                }
            }
            "integer" => {
                if k != 4 {
                    return Err(format!("{name}: expected NC_INT, got {k}"));
                }
                let mut b = vec![0i32; len];
                check(unsafe {
                    nc_get_vara_int(f.id, v, starts.as_ptr(), counts.as_ptr(), b.as_mut_ptr())
                })?;
                bytes.resize(len * 4, 0);
                for (word, x) in bytes.chunks_exact_mut(4).zip(&b) {
                    word.copy_from_slice(&x.to_le_bytes());
                }
            }
            "character" => {
                if k != 2 {
                    return Err(format!("{name}: expected NC_CHAR, got {k}"));
                }
                bytes.resize(len, 0);
                check(unsafe {
                    nc_get_vara_text(
                        f.id,
                        v,
                        starts.as_ptr(),
                        counts.as_ptr(),
                        bytes.as_mut_ptr().cast(),
                    )
                })?;
            }
            _ => return Err(format!("unsupported kind {kind}")),
        }
        if dims.first().map(String::as_str) == Some("Time") {
            dims.remove(0);
            counts.remove(0);
        }
        return Ok(Some((bytes, dims, counts, format!("{}:{name}", f.path))));
    }
    Ok(None)
}
// Permute named axes and insert the native trailing dummy entity, byte for byte.
fn pack(
    input: &[u8],
    dims: &[String],
    counts: &[usize],
    field: &Field,
    component: Option<usize>,
    out: &mut [u8],
) -> Result<(), String> {
    let offset = usize::from(component.is_some());
    if dims.len() + offset != field.dimensions.len() {
        return Err(format!("{}: dimension rank differs", field.name));
    }
    let axes: Vec<usize> = dims
        .iter()
        .map(|d| {
            field
                .dimensions
                .iter()
                .position(|x| x == d)
                .ok_or_else(|| format!("{}: unexpected axis {d}", field.name))
        })
        .collect::<Result<_, _>>()?;
    for (a, n) in axes.iter().zip(counts) {
        let expected = field.shape[*a];
        let padded = matches!(
            field.dimensions[*a].as_str(),
            "nCells" | "nEdges" | "nVertices"
        );
        if *n != expected - usize::from(padded) {
            return Err(format!(
                "{}: axis {} length {n} != {}",
                field.name,
                field.dimensions[*a],
                expected - usize::from(padded)
            ));
        }
    }
    if axes.iter().collect::<std::collections::HashSet<_>>().len() != axes.len() {
        return Err(format!("{}: repeated source axis", field.name));
    }
    if component.is_some_and(|c| c >= field.shape[0]) {
        return Err("component exceeds selected scalar count".into());
    }
    let len: usize = counts.iter().product();
    if input.len() != len * 4 {
        return Err("invalid payload word count".into());
    }
    // C source axes reversed into Fortran destination axes already have the
    // same word order. Only the slowest destination axis may have a trailing
    // dummy: padding an interior axis requires the general permutation below.
    // Scalar components stay interleaved through the original path.
    if component.is_none()
        && axes.iter().copied().eq((0..field.shape.len()).rev())
        && axes.iter().zip(counts).skip(1).all(|(a, n)| *n == field.shape[*a])
    {
        let target = &mut out[..input.len()];
        if input.len() < (1 << 20) {
            target.copy_from_slice(input);
        } else {
            target.par_chunks_mut(1 << 20)
                .zip(input.par_chunks(1 << 20))
                .for_each(|(dst, src)| dst.copy_from_slice(src));
        }
        return Ok(());
    }
    let mut strides = vec![1; field.shape.len()];
    for a in 1..strides.len() {
        strides[a] = strides[a - 1] * field.shape[a - 1];
    }
    for i in 0..len {
        let mut r = i;
        let mut dst = component.unwrap_or(0);
        for a in (0..counts.len()).rev() {
            let c = r % counts[a];
            r /= counts[a];
            dst += c * strides[axes[a]];
        }
        out[dst * 4..dst * 4 + 4].copy_from_slice(&input[i * 4..i * 4 + 4]);
    }
    Ok(())
}
/// One component read for a numeric field: `Ok(None)` when no source carries
/// it, `Err` when reading it failed (later components are then not read).
type Component = Result<Option<(Vec<u8>, Vec<String>, Vec<usize>, String)>, String>;
/// A numeric field whose inputs this thread has read, for a worker to lay out,
/// write and hash.
struct Job {
    index: usize,
    field: Field,
    names: Vec<String>,
    components: Vec<Component>,
}
// Lay out, write and hash one numeric field; the checks run in the order the
// serial reader made them, so a field refuses with the same message.
fn finish(job: Job, out: &Path) -> Result<Record, String> {
    let Job { index, field: f, names, components } = job;
    let word = if f.kind == "integer" {
        if f.initial.is_empty() {
            0
        } else {
            f.initial
                .parse::<i32>()
                .map_err(|e| format!("{}: invalid integer initial value: {e}", f.name))?
        }
        .to_le_bytes()
    } else {
        if f.initial.is_empty() {
            0.
        } else {
            f.initial
                .replace(['d', 'D'], "e")
                .parse::<f32>()
                .map_err(|e| format!("{}: invalid real initial value: {e}", f.name))?
        }
        .to_bits()
        .to_le_bytes()
    };
    let len: usize = f.shape.iter().product();
    let mut bytes = word.repeat(len);
    let mut origins = Vec::new();
    for (c, (n, component)) in names.iter().zip(components).enumerate() {
        if let Some((b, d, s, src)) = component? {
            pack(
                &b,
                &d,
                &s,
                &f,
                if f.components.is_empty() { None } else { Some(c) },
                &mut bytes,
            )?;
            origins.push(src);
        } else {
            if f.required {
                return Err(format!("required input {n} absent"));
            }
            origins.push(format!("registry-initial-value:{n}"));
        }
    }
    // Retain the allocation default in unread dummy storage.
    let name = format!("field-{index:04}.bin");
    let mut target = fs::File::create(out.join(&name)).map_err(|e| e.to_string())?;
    target.write_all(&bytes).map_err(|e| e.to_string())?;
    Ok(Record {
        pool: f.pool,
        name: f.name,
        kind: f.kind,
        dimensions: f.dimensions,
        shape: f.shape,
        payload: Some(name),
        sha256: Some(format!("{:x}", Sha256::digest(&bytes))),
        text: None,
        origins,
    })
}
fn run(request: &Path, out: &Path) -> Result<(), String> {
    let r: Request = serde_json::from_slice(&fs::read(request).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if r.schema != "mpas-native-pool-request-v1" {
        return Err("unsupported request".into());
    }
    let files: Vec<File> = r
        .sources
        .iter()
        .map(|p| File::open(p))
        .collect::<Result<_, _>>()?;
    fs::create_dir_all(out).map_err(|e| e.to_string())?;
    // netCDF-C is not thread-safe, so this thread does every read, in field
    // order; layout, write and SHA-256 of each numeric field run on worker
    // threads meanwhile. Results are taken in field order, so the manifest,
    // the payload files and the first refusal are the serial reader's.
    let count = r.fields.len();
    let results: Vec<std::sync::Mutex<Option<Result<Record, String>>>> =
        (0..count).map(|_| std::sync::Mutex::new(None)).collect();
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 16)
        .min(count.max(1));
    let (sender, receiver) = std::sync::mpsc::sync_channel::<Job>(workers);
    let receiver = std::sync::Mutex::new(receiver);
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let job = match receiver.lock() {
                    Ok(guard) => guard.recv(),
                    Err(_) => return,
                };
                let Ok(job) = job else { return };
                let index = job.index;
                let result = finish(job, out);
                *results[index].lock().unwrap() = Some(result);
            });
        }
        for (index, f) in r.fields.into_iter().enumerate() {
            // MPAS-Model-rrfs/src/framework/mpas_field_routines.F:261-299:
            // zero-sized Registry dimensions allocate associated empty arrays.
            if f.shape.len() != f.dimensions.len() {
                *results[index].lock().unwrap() =
                    Some(Err(format!("{}: invalid destination shape", f.name)));
                break;
            }
            if f.kind == "character" {
                let outcome = (|| -> Result<Record, String> {
                    let mut origins = Vec::new();
                    let b = if f.read_input {
                        read(&files, &f.name, &f.kind)?
                    } else {
                        None
                    };
                    let text = Some(if let Some((b, _, _, src)) = b {
                        origins.push(src);
                        String::from_utf8(b)
                            .map_err(|e| e.to_string())?
                            .trim_end_matches(['\0', ' '])
                            .to_string()
                    } else {
                        if f.required {
                            return Err(format!("required input {} absent", f.name));
                        }
                        origins.push("registry-initial-value".into());
                        f.initial.clone()
                    });
                    Ok(Record {
                        pool: f.pool.clone(),
                        name: f.name.clone(),
                        kind: f.kind.clone(),
                        dimensions: f.dimensions.clone(),
                        shape: f.shape.clone(),
                        payload: None,
                        sha256: None,
                        text,
                        origins,
                    })
                })();
                let failed = outcome.is_err();
                *results[index].lock().unwrap() = Some(outcome);
                if failed {
                    break;
                }
                continue;
            }
            let names = if f.components.is_empty() {
                vec![f.name.clone()]
            } else {
                f.components.clone()
            };
            let mut components = Vec::with_capacity(names.len());
            let mut failed = false;
            for n in &names {
                let component = if f.read_input { read(&files, n, &f.kind) } else { Ok(None) };
                failed = component.is_err();
                components.push(component);
                if failed {
                    break;
                }
            }
            if sender.send(Job { index, field: f, names, components }).is_err() || failed {
                break;
            }
        }
        drop(sender);
    });
    let mut records = Vec::with_capacity(count);
    for slot in results {
        match slot.into_inner().unwrap() {
            Some(Ok(record)) => records.push(record),
            Some(Err(e)) => return Err(e),
            None => return Err("field transport stopped before every field was read".into()),
        }
    }
    let receipt = Receipt {
        schema: "mpas-native-pool-payload-v1",
        fields: records,
    };
    fs::write(
        out.join("manifest.json"),
        serde_json::to_vec_pretty(&receipt).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}
// File metadata only; Registry allocation remains in the orchestration layer.
fn inspect(path: &str) -> Result<(), String> {
    let f = File::open(path)?;
    let mut rank = 0;
    check(unsafe { nc_inq_ndims(f.id, &mut rank) })?;
    let mut dims = std::collections::BTreeMap::new();
    for d in 0..rank {
        let mut label = [0 as c_char; 257];
        let mut len = 0;
        check(unsafe { nc_inq_dim(f.id, d, label.as_mut_ptr(), &mut len) })?;
        dims.insert(
            unsafe { CStr::from_ptr(label.as_ptr()) }
                .to_string_lossy()
                .into_owned(),
            len,
        );
    }
    println!(
        "{}",
        serde_json::to_string(&dims).map_err(|e| e.to_string())?
    );
    Ok(())
}
fn main() {
    let a: Vec<_> = std::env::args().collect();
    if a.len() == 2 && a[1] == "--abi" {
        println!("gpuwm-rw-mpas-physics-pool-v1");
        return;
    }
    if a.len() == 3 && a[1] == "--inspect" {
        if let Err(e) = inspect(&a[2]) {
            eprintln!("physics pool: {e}");
            std::process::exit(1);
        }
        return;
    }
    if a.len() != 3 {
        eprintln!("usage: rw_mpas_physics_pool REQUEST.json OUTPUT_DIR");
        std::process::exit(2);
    }
    if let Err(e) = run(Path::new(&a[1]), Path::new(&a[2])) {
        eprintln!("physics pool: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn field() -> Field {
        Field {
            pool: "state".into(),
            name: "scalars".into(),
            kind: "real".into(),
            dimensions: vec!["num_scalars".into(), "nVertLevels".into(), "nCells".into()],
            shape: vec![2, 2, 4],
            components: vec!["qv".into(), "qc".into()],
            initial: String::new(),
            required: true,
            read_input: true,
        }
    }
    #[test]
    fn permutes_scalar_words_without_arithmetic() {
        let words = [
            0x80000000u32,
            0x7fc00001,
            0x00000001,
            0x3f800000,
            0xff800000,
            0x7f800000,
        ];
        let input: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mut out = vec![0; 64];
        pack(
            &input,
            &["nCells".into(), "nVertLevels".into()],
            &[3, 2],
            &field(),
            Some(1),
            &mut out,
        )
        .unwrap();
        for c in 0..3 {
            for z in 0..2 {
                let dst = 1 + 2 * z + 4 * c;
                assert_eq!(&out[dst * 4..dst * 4 + 4], &words[2 * c + z].to_le_bytes());
            }
        }
        assert_eq!(&out[48..], &[0; 16]);
    }
    #[test]
    fn rejects_extent_and_repeated_axis() {
        let mut out = vec![0; 64];
        assert!(
            pack(
                &[0; 16],
                &["nCells".into(), "nVertLevels".into()],
                &[2, 2],
                &field(),
                Some(0),
                &mut out
            )
            .is_err()
        );
        assert!(
            pack(
                &[0; 16],
                &["nVertLevels".into(), "nVertLevels".into()],
                &[2, 2],
                &field(),
                Some(0),
                &mut out
            )
            .is_err()
        );
    }

    #[test]
    fn ordered_large_payload_keeps_ieee_words_and_trailing_dummy() {
        let mut f = field();
        f.dimensions = vec!["nVertLevels".into(), "nCells".into()];
        f.shape = vec![55, 5001];
        f.components.clear();
        let words = [0x80000000u32, 0x7fc12345, 0x7f812345, 1, 0xff800000, 0x3f800000];
        let input: Vec<_> = (0..55 * 5000).flat_map(|i| words[i % words.len()].to_le_bytes()).collect();
        let mut out = vec![0xa5; 55 * 5001 * 4];
        pack(&input, &["nCells".into(), "nVertLevels".into()], &[5000, 55], &f, None, &mut out).unwrap();
        assert_eq!(&out[..input.len()], &input);
        assert!(out[input.len()..].iter().all(|b| *b == 0xa5));
    }

    #[test]
    fn interior_dummy_and_reordered_axes_still_permute() {
        let mut f = field();
        f.dimensions = vec!["nCells".into(), "nVertLevels".into()];
        f.shape = vec![4, 2];
        f.components.clear();
        let words = [0x80000000u32, 0x7f812345, 1, 0xff800000, 0x3f800000, 0x7fc12345];
        for reversed in [false, true] {
            let (dims, counts) = if reversed {
                (vec!["nVertLevels".into(), "nCells".into()], vec![2, 3])
            } else {
                (vec!["nCells".into(), "nVertLevels".into()], vec![3, 2])
            };
            let input: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let mut out = vec![0xa5; 32];
            pack(&input, &dims, &counts, &f, None, &mut out).unwrap();
            for z in 0..2 {
                for c in 0..3 {
                    let src = if reversed { z * 3 + c } else { c * 2 + z };
                    assert_eq!(&out[(c + 4 * z) * 4..(c + 4 * z + 1) * 4], &words[src].to_le_bytes());
                }
                assert_eq!(&out[(3 + 4 * z) * 4..(4 + 4 * z) * 4], &[0xa5; 4]);
            }
        }
    }
}

#[cfg(test)]
mod rrfs_empty_pool_tests {
    use super::*;
    #[test]
    fn registry_zero_dimension_produces_associated_empty_payload() {
        let root = std::env::var("RRFS_POOL_TEST_ROOT").unwrap_or_else(|_| format!("{}/target",env!("CARGO_MANIFEST_DIR")));
        let root = Path::new(&root).join(format!("rrfs-empty-transport-{}",std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let request=root.join("request.json");
        let r=serde_json::json!({"schema":"mpas-native-pool-request-v1","sources":[],
            "fields":[{"pool":"diag_physics","name":"ddvel","kind":"real",
            "dimensions":["nChem","nCells"],"shape":[0,4],"initial":"0"},
            {"pool":"diag_physics","name":"empty_int","kind":"integer",
            "dimensions":["nCells","nChem"],"shape":[4,0],"initial":"-1"}]});
        fs::write(&request,serde_json::to_vec(&r).unwrap()).unwrap();
        run(&request,&root.join("output")).unwrap();
        let result:serde_json::Value=serde_json::from_slice(
            &fs::read(root.join("output/manifest.json")).unwrap()).unwrap();
        for record in result["fields"].as_array().unwrap() {
            let bytes=fs::read(root.join("output").join(record["payload"].as_str().unwrap())).unwrap();
            assert!(bytes.is_empty());
            assert_eq!(record["sha256"],format!("{:x}",Sha256::digest(&bytes)));
        }
        fs::remove_dir_all(root).unwrap();
    }
}
