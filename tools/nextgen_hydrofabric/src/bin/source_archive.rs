//! Source acquisition tool. libarchive is linked only by this optional binary,
//! never the production raw-buffer bridge or routing process.
use nextgen_hydrofabric::Result;
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

#[path = "source_archive/plan.rs"]
mod source_plan;

#[cfg_attr(target_os = "linux", link(name = ":libarchive.so.13"))]
#[cfg_attr(not(target_os = "linux"), link(name = "archive"))]
extern "C" {
    fn archive_read_new() -> *mut c_void;
    fn archive_read_support_filter_all(a: *mut c_void) -> c_int;
    fn archive_read_support_format_all(a: *mut c_void) -> c_int;
    fn archive_read_open_filename(a: *mut c_void, path: *const c_char, block: usize) -> c_int;
    fn archive_read_next_header(a: *mut c_void, entry: *mut *mut c_void) -> c_int;
    fn archive_entry_pathname(entry: *mut c_void) -> *const c_char;
    fn archive_entry_size(entry: *mut c_void) -> i64;
    fn archive_entry_filetype(entry: *mut c_void) -> u32;
    fn archive_entry_hardlink(entry: *mut c_void) -> *const c_char;
    fn archive_read_data(a: *mut c_void, data: *mut u8, length: usize) -> isize;
    fn archive_read_data_skip(a: *mut c_void) -> c_int;
    fn archive_error_string(a: *mut c_void) -> *const c_char;
    fn archive_read_free(a: *mut c_void) -> c_int;
    fn archive_version_string() -> *const c_char;
}
struct Archive(*mut c_void);
impl Archive {
    fn check(&self, rc: c_int) -> Result<()> {
        if rc < 0 {
            let p = unsafe { archive_error_string(self.0) };
            return Err(if p.is_null() {
                format!("libarchive status {rc}")
            } else {
                unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
            }
            .into());
        }
        Ok(())
    }
}
impl Drop for Archive {
    fn drop(&mut self) {
        unsafe {
            archive_read_free(self.0);
        }
    }
}
#[derive(Deserialize)]
struct Job {
    archive: PathBuf,
    sha256: String,
    /// Exact member paths; an empty list inventories without extracting.
    members: Vec<String>,
    output: PathBuf,
    max_extracted_bytes: u64,
    /// For bounded HTTP prefix audits only. The receipt explicitly distinguishes
    /// an early stop from a complete inventory; never infer absent members.
    #[serde(default)]
    inventory_entry_limit: Option<usize>,
    #[serde(default)]
    inventory_skip_budget: Option<u64>,
}
fn file_sha(path: &Path) -> Result<String> {
    let mut input = std::fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = input.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
fn run(job: &Job) -> Result<serde_json::Value> {
    let start = Instant::now();
    let sha = file_sha(&job.archive)?;
    if sha != job.sha256 {
        return Err("source archive SHA256 mismatch".into());
    }
    let selected: BTreeSet<_> = job.members.iter().cloned().collect();
    if selected.len() != job.members.len() {
        return Err("duplicate requested member".into());
    }
    let mut found = BTreeSet::new();
    let mut entries = Vec::new();
    let mut extracted_bytes = 0u64;
    let mut complete_scan = false;
    let mut skipped_bytes = 0u64;
    let mut stopped_before_member_data = false;
    if job.inventory_entry_limit == Some(0)
        || (job.inventory_entry_limit.is_some() && !selected.is_empty())
    {
        return Err("bounded inventory requires a positive limit and no extraction".into());
    }
    let a = Archive(unsafe { archive_read_new() });
    if a.0.is_null() {
        return Err("archive allocation failed".into());
    }
    a.check(unsafe { archive_read_support_filter_all(a.0) })?;
    a.check(unsafe { archive_read_support_format_all(a.0) })?;
    let filename = CString::new(job.archive.to_str().ok_or("non-UTF8 archive path")?)?;
    a.check(unsafe { archive_read_open_filename(a.0, filename.as_ptr(), 1024 * 1024) })?;
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let mut entry = std::ptr::null_mut();
        let code = unsafe { archive_read_next_header(a.0, &mut entry) };
        if code == 1 {
            complete_scan = true;
            break;
        }
        a.check(code)?;
        if entry.is_null() {
            return Err("missing archive entry".into());
        }
        let p = unsafe { archive_entry_pathname(entry) };
        if p.is_null() {
            return Err("missing member pathname".into());
        }
        let name = unsafe { CStr::from_ptr(p) }.to_str()?.to_owned();
        let size = unsafe { archive_entry_size(entry) };
        let mut record = json!({"path":name,"bytes":size,"extracted":false});
        if selected.is_empty()
            && job.inventory_skip_budget.is_some_and(|budget| {
                size < 0 || skipped_bytes.saturating_add(size as u64) > budget
            })
        {
            entries.push(record);
            stopped_before_member_data = true;
            break;
        }
        if selected.contains(&name) {
            // Official archives need only regular files. Do not materialize links
            // or paths outside a newly created acquisition directory.
            let path = Path::new(&name);
            if path
                .components()
                .any(|c| !matches!(c, Component::Normal(_)))
                || name.contains('\\')
                || name.contains(':')
                || unsafe { archive_entry_filetype(entry) } != 0o100000
                || !unsafe { archive_entry_hardlink(entry) }.is_null()
                || size < 0
            {
                return Err(format!("unsupported selected member {name}").into());
            }
            if !found.insert(name.clone()) {
                return Err("duplicate archive member".into());
            }
            let size = size as u64;
            extracted_bytes = extracted_bytes
                .checked_add(size)
                .ok_or("member size overflow")?;
            if extracted_bytes > job.max_extracted_bytes {
                return Err("selected members exceed configured extraction budget".into());
            }
            let dest = job.output.join(path);
            std::fs::create_dir_all(dest.parent().ok_or("missing member parent")?)?;
            let mut out = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&dest)?;
            let mut hash = Sha256::new();
            let mut written = 0u64;
            loop {
                let n = unsafe { archive_read_data(a.0, buffer.as_mut_ptr(), buffer.len()) };
                if n < 0 {
                    a.check(n as c_int)?;
                }
                if n == 0 {
                    break;
                }
                written += n as u64;
                if written > size {
                    return Err("member exceeds declared size".into());
                }
                out.write_all(&buffer[..n as usize])?;
                hash.update(&buffer[..n as usize]);
            }
            if written != size {
                return Err("incomplete archive member".into());
            }
            record["extracted"] = json!(true);
            record["sha256"] = json!(format!("{:x}", hash.finalize()));
        } else if job.inventory_entry_limit != Some(entries.len() + 1) {
            a.check(unsafe { archive_read_data_skip(a.0) })?;
            skipped_bytes = skipped_bytes.saturating_add(size.max(0) as u64);
        }
        entries.push(record);
        if job.inventory_entry_limit == Some(entries.len()) {
            break;
        }
    }
    if found != selected {
        return Err(format!(
            "requested members absent: {:?}",
            selected.difference(&found)
        )
        .into());
    }
    Ok(json!({"archive":job.archive,"sha256":sha,"entries":entries,
        "output":job.output,"extracted_bytes":extracted_bytes,
        "complete_archive_inventory":complete_scan,"inventory_entry_limit":job.inventory_entry_limit,
        "stopped_before_member_data":stopped_before_member_data,"inventory_skip_budget":job.inventory_skip_budget,
        "wall_ms":start.elapsed().as_secs_f64()*1000.0}))
}
fn main() {
    if let Err(e) = main_result() {
        eprintln!("rw_hydrofabric_source_archive: {e}");
        std::process::exit(1);
    }
}
fn main_result() -> Result<()> {
    if matches!(
        std::env::args().nth(1).as_deref(),
        Some("--plan" | "--check" | "--fetch")
    ) {
        return source_plan::main_result();
    }
    let config = std::env::args()
        .nth(1)
        .ok_or("usage: rw_hydrofabric_source_archive JOBS.json")?;
    let jobs: Vec<Job> = serde_json::from_slice(&std::fs::read(config)?)?;
    let start = Instant::now();
    let results = jobs.par_iter().map(run).collect::<Result<Vec<_>>>()?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({"source":nextgen_hydrofabric::SOURCE_REVISION,
        "library":unsafe { CStr::from_ptr(archive_version_string()) }.to_str()?,
        "rayon_workers":rayon::current_num_threads(),"parallelism":"one independent archive per Rayon job; each compressed stream is sequential",
        "wall_ms":start.elapsed().as_secs_f64()*1000.0,"results":results}))?
    );
    Ok(())
}
