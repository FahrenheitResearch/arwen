//! File-based visual-workspace handoff; the TUI remains the CLI job owner.
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    io::{self, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
#[path="local_progress.rs"]
pub(crate) mod local_progress;

pub fn now_ms() -> u128 { SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() }
pub fn digest(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }
pub fn read_json(path: &Path, limit: u64) -> Result<Value, String> {
    let mut bytes = Vec::new();
    fs::File::open(path).map_err(|e| format!("Cannot read {}: {e}", path.display()))?
        .take(limit + 1).read_to_end(&mut bytes).map_err(|e| e.to_string())?;
    if bytes.len() as u64 > limit { return Err("JSON document is too large.".into()); }
    serde_json::from_slice(&bytes).map_err(|_| "Invalid JSON document.".into())
}
fn atomic_json(path: &Path, value: &Value) -> io::Result<()> {
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
    let temporary = path.with_file_name(format!(".{}-{}-{stamp}.tmp", path.file_name().unwrap().to_string_lossy(), std::process::id()));
    let result = (|| {
        let mut file = fs::OpenOptions::new().write(true).create_new(true).open(&temporary)?;
        serde_json::to_writer_pretty(&mut file, value)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temporary, path)
    })();
    if result.is_err() { let _ = fs::remove_file(temporary); }
    result
}

#[derive(Clone, Debug)]
pub enum Action {
    BrowseRuns,
    OpenRun(String),
    CloseRun(String),
    ReviewPlan(PathBuf),
    LaunchPlan(PathBuf),
    StopJob(String),
    SyncArtifacts { job: String, domain: u32, sequence: Option<u64>, reader_leases: bool },
    SyncProcessedFrame { job: String, domain: u32, sequence: Option<u64> },
    SyncProcessedFrameV2 { job: String, domain: u32, sequence: Option<u64>, options: crate::remote::ViewerOptions, reader_leases: bool, cache_bytes: Option<u64> },
    SyncNativePlots { job: String, domain: u32, sequence: u64 },
    ArtifactIndex { job: String, domain: u32, after_sequence: u64 },
    /// Offline downscaling of a finished local run: the parent's history
    /// and its restart evidence become a standalone child forecast. Local
    /// only, because both inputs are read from this computer's disk.
    LaunchDownscale(Box<DownscaleRequest>),
    OpenConfig(PathBuf),
    ResetSetup,
    FocusLogs,
    FocusJobLogs(String),
    FocusNodes,
    FocusSetup,
    SelectTarget,
    /// A second desktop launch over the same output root asks the live
    /// controller to show its workspace instead of starting a duplicate.
    OpenWorkspace,
}
/// One `launch_downscale` request, already checked field by field. Every
/// value here becomes a Downscale guide answer, never a command token of
/// its own: the guide is the single place that turns settings into
/// `gpuwm downscale` arguments, so the controller cannot drift from what
/// the terminal builds for the same answers.
#[derive(Clone, Debug, Default)]
pub struct DownscaleRequest {
    pub parent_run_dir: String,
    pub parent_domain: Option<u32>,
    /// `None` asks the engine for the parent's newest complete checkpoint
    /// set (`--parent-restart=latest`), which is what a caller holding
    /// only the parent's run directory can name.
    pub parent_restart: Option<String>,
    pub point: Option<(f64, f64)>,
    pub child_config: Option<String>,
    pub ratio: u32,
    pub child_size: Option<(u32, u32)>,
    pub vram_gib: Option<f64>,
    pub auto_vram: bool,
    pub hours: Option<f64>,
    pub output_interval_seconds: Option<f64>,
    pub tiles: Option<String>,
    pub accept_parent_cadence: bool,
    pub max_boundary_interval_seconds: Option<f64>,
    pub out_dir: String,
    /// `true` is `mode: "plan"`: validate, derive and price the child,
    /// write its TOML and plan document, run no forecast.
    pub plan: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Target { Local, Ssh { node_id: String, connection_sha256: String } }
impl Target {
    pub fn value(&self) -> Value { match self {
        Self::Local => json!({"kind":"local"}),
        Self::Ssh { node_id, connection_sha256 } => json!({"kind":"ssh","node_id":node_id,"connection_sha256":connection_sha256}),
    }}
}
#[derive(Clone, Debug)]
pub struct Request {
    pub id: String, pub name: String, pub action: Action, pub target: Option<Target>,
    pub plan_sha256: Option<String>, pub config_sha256: Option<String>,
    pub review_id: Option<String>, pub review_sha256: Option<String>,
}
/// Every key `launch_downscale` accepts. The whitelist stays explicit:
/// an unknown key is refused rather than ignored, so a caller never
/// believes it set something the controller dropped.
const DOWNSCALE_KEYS: &[&str] = &["parent_run_dir", "parent_domain", "parent_restart", "point",
    "child_config", "ratio", "child_size", "vram_gib", "auto_vram", "hours",
    "output_interval_seconds", "tiles", "accept_parent_cadence",
    "max_boundary_interval_seconds", "out_dir", "mode"];

fn absolute_field(value: &Value, key: &str, label: &str) -> Result<String, String> {
    let text = value[key].as_str().filter(|text| !text.is_empty() && text.len() <= 8192
        && !text.chars().any(char::is_control)).ok_or_else(|| format!("{label} is required."))?;
    if !Path::new(text).is_absolute() { return Err(format!("{label} must be an absolute path.")); }
    Ok(text.to_owned())
}

fn finite(value: &Value, key: &str, label: &str) -> Result<Option<f64>, String> {
    match value.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(number) => Ok(Some(number.as_f64().filter(|value| value.is_finite())
            .ok_or_else(|| format!("{label} must be a finite number."))?)),
    }
}

fn count(value: &Value, key: &str, range: std::ops::RangeInclusive<u64>, label: &str) -> Result<Option<u32>, String> {
    match value.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(number) => Ok(Some(number.as_u64().filter(|value| range.contains(value))
            .ok_or_else(|| format!("{label} must be an integer between {} and {}.", range.start(), range.end()))? as u32)),
    }
}

fn parse_downscale(value: &Value) -> Result<DownscaleRequest, String> {
    let mut request = DownscaleRequest {
        parent_run_dir: absolute_field(value, "parent_run_dir", "The parent run directory")?,
        out_dir: absolute_field(value, "out_dir", "The downscaled output directory")?,
        ratio: count(value, "ratio", 2..=99, "Refinement ratio")?.unwrap_or(3),
        parent_domain: count(value, "parent_domain", 1..=99, "Parent domain")?,
        hours: finite(value, "hours", "Child duration in hours")?,
        output_interval_seconds: finite(value, "output_interval_seconds", "Child output interval")?,
        max_boundary_interval_seconds: finite(value, "max_boundary_interval_seconds", "Maximum boundary interval")?,
        vram_gib: finite(value, "vram_gib", "GPU memory capacity")?,
        ..DownscaleRequest::default()
    };
    if !Path::new(&request.parent_run_dir).is_dir() {
        return Err("The parent run directory does not exist on this computer. Choose a finished local forecast in My forecasts.".into());
    }
    request.parent_restart = match value.get("parent_restart") {
        None | Some(Value::Null) => None,
        Some(_) => Some(absolute_field(value, "parent_restart", "The parent restart checkpoint")?),
    };
    request.child_config = match value.get("child_config") {
        None | Some(Value::Null) => None,
        Some(_) => Some(absolute_field(value, "child_config", "The child configuration")?),
    };
    request.point = match value.get("point") {
        None | Some(Value::Null) => None,
        Some(Value::Object(point)) if point.len() == 2 => {
            let coordinate = |key: &str, limit: f64| point.get(key).and_then(Value::as_f64)
                .filter(|value| value.is_finite() && value.abs() <= limit)
                .ok_or_else(|| "The child centre needs a finite latitude and longitude inside geographic bounds.".to_owned());
            Some((coordinate("lat", 90.0)?, coordinate("lon", 360.0)?))
        }
        Some(_) => return Err("The child centre must be an object with lat and lon.".into()),
    };
    if request.point.is_some() == request.child_config.is_some() {
        return Err("Choose either a child centre point or an existing child configuration.".into());
    }
    request.child_size = match value.get("child_size") {
        None | Some(Value::Null) => None,
        Some(Value::Object(size)) if size.len() == 2 => {
            let extent = |key: &str| size.get(key).and_then(Value::as_u64).filter(|value| (2..=8192).contains(value))
                .map(|value| value as u32).ok_or_else(|| "Explicit child size needs nx and ny cell counts.".to_owned());
            Some((extent("nx")?, extent("ny")?))
        }
        Some(_) => return Err("Explicit child size must be an object with nx and ny.".into()),
    };
    request.auto_vram = match value.get("auto_vram") {
        None | Some(Value::Null) => request.child_size.is_none() && request.vram_gib.is_none() && request.child_config.is_none(),
        Some(flag) => flag.as_bool().ok_or("Fit to this GPU must be true or false.")?,
    };
    if request.auto_vram && (request.child_size.is_some() || request.vram_gib.is_some() || request.child_config.is_some()) {
        return Err("Fitting the child to this GPU measures it: leave explicit size, capacity and a supplied child configuration unset, or turn fitting off.".into());
    }
    if request.child_size.is_some() && request.vram_gib.is_some() {
        return Err("Choose an explicit child size or a GPU capacity to size against, not both.".into());
    }
    request.accept_parent_cadence = match value.get("accept_parent_cadence") {
        None | Some(Value::Null) => request.max_boundary_interval_seconds.is_none(),
        Some(flag) => flag.as_bool().ok_or("Accepting the parent cadence must be true or false.")?,
    };
    if request.accept_parent_cadence && request.max_boundary_interval_seconds.is_some() {
        return Err("Choose a maximum boundary interval or accept the parent's own cadence, not both.".into());
    }
    request.tiles = match value.get("tiles") {
        None | Some(Value::Null) => None,
        Some(Value::String(mode)) if matches!(mode.as_str(), "on" | "auto") => Some(mode.clone()),
        Some(_) => return Err("Streaming must be on or auto.".into()),
    };
    request.plan = match value["mode"].as_str() {
        Some("plan") => true,
        Some("run") => false,
        _ => return Err("Downscale mode must be plan or run.".into()),
    };
    Ok(request)
}

fn valid_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 80 && id.bytes().all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-'))
}
fn parse_request(value: &Value, id: &str, session: &str) -> Result<Request, String> {
    if value["schema"] != "arwen.companion-request.v1" || value["session_id"] != session || value["id"] != id {
        return Err("Request schema, session or ID does not match this control queue.".into());
    }
    let name = value["action"].as_str().ok_or("Request action is missing.")?;
    let field = match name { "review_plan" | "launch_plan" => Some("plan_path"), "stop_job" | "sync_artifacts" | "sync_processed_frame" | "sync_processed_frame_v2" | "sync_native_plots" | "artifact_index" | "open_run" | "close_run" => Some("job_id"), "open_config" => Some("config_path"), "reset_setup" | "focus_logs" | "focus_nodes" | "focus_setup" | "select_target" | "browse_runs" | "open_workspace" | "launch_downscale" => None,
        _ => return Err("Unsupported companion action.".into()) };
    let object = value.as_object().ok_or("Request must be a JSON object.")?;
    let plan_action = matches!(name, "review_plan" | "launch_plan");
    if object.keys().any(|key| !["schema", "session_id", "id", "action"].contains(&key.as_str())
        && !(key == "target" && !matches!(name,"open_config"|"reset_setup"|"focus_nodes"|"focus_setup"|"open_workspace"))
        && !(matches!(name,"sync_artifacts"|"sync_processed_frame"|"artifact_index") && key=="domain")
        && !(name=="sync_artifacts" && matches!(key.as_str(),"sequence"|"reader_leases"))
        && !(name=="sync_processed_frame" && key=="sequence")
        && !(name=="sync_native_plots" && matches!(key.as_str(),"domain"|"sequence"))
        && !(name=="focus_logs" && key=="job_id")
        && !(name=="sync_processed_frame_v2" && matches!(key.as_str(),"domain"|"sequence"|"profile"|"products"|"expected_run_id"|"prefetch_sequences"|"reader_leases"|"cache_bytes"))
        && !(name=="artifact_index" && key=="after_sequence")
        && !(plan_action && ["plan_sha256","config_sha256"].contains(&key.as_str()))
        && !(name=="launch_plan" && ["review_id","review_sha256"].contains(&key.as_str()))
        && !(name=="launch_downscale" && DOWNSCALE_KEYS.contains(&key.as_str()))
        && Some(key.as_str()) != field) {
        return Err("Unknown field in companion request.".into());
    }
    let hash = |key:&str| -> Result<Option<String>,String> {match value.get(key) {
        None => Ok(None), Some(Value::String(text)) if text.len()==64 && text.bytes().all(|c|c.is_ascii_hexdigit()) => Ok(Some(text.to_ascii_lowercase())),
        _ => Err(format!("{key} must be a SHA-256 digest.")),
    }};
    let target = match value.get("target") {
        None => None,
        Some(Value::Object(target)) if target.get("kind")==Some(&json!("local")) && target.len()==1 => Some(Target::Local),
        Some(Value::Object(target)) if target.get("kind")==Some(&json!("ssh")) && target.len()==3 => {
            let node_id=target.get("node_id").and_then(Value::as_str).filter(|id|valid_id(id)).ok_or("SSH target needs a valid node ID.")?;
            let connection=target.get("connection_sha256").and_then(Value::as_str).filter(|h|h.len()==64&&h.bytes().all(|c|c.is_ascii_hexdigit())).ok_or("SSH target needs its connection SHA-256.")?;
            Some(Target::Ssh{node_id:node_id.into(),connection_sha256:connection.to_ascii_lowercase()})
        }
        _ => return Err("Unsupported or incomplete companion target.".into()),
    };
    let ssh=matches!(target,Some(Target::Ssh{..}));
    if matches!(name,"browse_runs"|"open_run"|"close_run")&&target.is_none(){return Err("Runs requests require an explicit local or saved SSH target.".into());}
    if name=="select_target"&&target.is_none(){return Err("Select target requires its explicit local or SSH identity.".into());}
    // Named here, at the whitelist, so the reason travels with the refusal:
    // downscaling reads the parent's history and its restart from THIS
    // computer's disk, and a node has no staged-inputs review for them.
    if name=="launch_downscale"&&!matches!(target,Some(Target::Local)){
        return Err("Downscaling reads the parent's history and restart from this computer's disk; a node has no staged-inputs review for them. Run `gpuwm downscale` in the node's terminal.".into());
    }
    let plan_sha256=hash("plan_sha256")?;
    let config_sha256=hash("config_sha256")?;
    let review_sha256=hash("review_sha256")?;
    let review_id=match value.get("review_id") {None=>None,Some(Value::String(id)) if valid_id(id)=>Some(id.clone()),_=>return Err("Invalid review ID.".into())};
    if (name=="review_plan"||ssh&&name=="launch_plan") && (plan_sha256.is_none()||config_sha256.is_none()) {
        return Err("A remote plan needs the saved plan and configuration SHA-256 bindings.".into());
    }
    if name=="review_plan"&&!ssh {return Err("Remote review requires an explicit SSH target.".into());}
    if matches!(name,"sync_artifacts"|"sync_processed_frame"|"sync_processed_frame_v2"|"sync_native_plots"|"artifact_index")&&!ssh {return Err("Remote artifacts require an explicit SSH target.".into());}
    if ssh&&name=="launch_plan"&&(review_id.is_none()||review_sha256.is_none()) {return Err("A remote launch needs its completed node review.".into());}
    let argument = field.map(|key| value[key].as_str().filter(|s| !s.is_empty() && s.len() <= 8192 && !s.chars().any(char::is_control))
        .ok_or_else(|| "Request needs a valid absolute path.".to_owned())).transpose()?;
    if argument.is_some_and(|value| !(matches!(name,"stop_job"|"sync_artifacts"|"sync_processed_frame"|"sync_processed_frame_v2"|"sync_native_plots"|"artifact_index"|"open_run"|"close_run")&&ssh) && !Path::new(value).is_absolute()) {
        return Err("Companion paths must be absolute.".into());
    }
    if matches!(name,"stop_job"|"sync_artifacts"|"sync_processed_frame"|"sync_processed_frame_v2"|"sync_native_plots"|"artifact_index"|"open_run"|"close_run")&&ssh {crate::remote::valid_job(argument.unwrap())?;}
    let domain=||match value.get("domain"){
        None=>Ok(1),Some(value)=>value.as_u64().filter(|n|(1..=999).contains(n)).map(|n|n as u32).ok_or("Artifact domain must be an integer between 1 and 999."),
    };
    let action = match name {
        "focus_logs" if value.get("job_id").is_some()=>{
            if !ssh{return Err("A saved remote job log requires an explicit SSH target.".into());}
            let job=value["job_id"].as_str().ok_or("A saved job log requires its job ID.")?;
            crate::remote::valid_job(job)?;Action::FocusJobLogs(job.into())
        },
        "browse_runs" => Action::BrowseRuns,
        "open_run" => Action::OpenRun(argument.unwrap().to_owned()),
        "close_run" => Action::CloseRun(argument.unwrap().to_owned()),
        "review_plan" => Action::ReviewPlan(PathBuf::from(argument.unwrap())),
        "launch_plan" => Action::LaunchPlan(PathBuf::from(argument.unwrap())),
        "stop_job" => Action::StopJob(argument.unwrap().to_owned()),
        "sync_artifacts" => Action::SyncArtifacts { job: argument.unwrap().to_owned(), domain:domain()?,
            sequence:value.get("sequence").map(|v|v.as_u64().filter(|n|*n>0&&*n<=i64::MAX as u64).ok_or("Frame sequence must be a positive integer.")).transpose()?,
            reader_leases:match value.get("reader_leases"){None=>false,Some(value)=>value.as_bool().ok_or("Reader leases must be a boolean.")?}},
        "sync_processed_frame" => Action::SyncProcessedFrame { job:argument.unwrap().to_owned(),domain:domain()?,
            sequence:value.get("sequence").map(|v|v.as_u64().filter(|n|*n>0&&*n<=i64::MAX as u64).ok_or("Frame sequence must be a positive integer.")).transpose()? },
        "sync_processed_frame_v2" => Action::SyncProcessedFrameV2 {job:argument.unwrap().to_owned(),domain:domain()?,
            sequence:value.get("sequence").map(|v|v.as_u64().filter(|n|*n>0&&*n<=i64::MAX as u64).ok_or("Frame sequence must be a positive integer.")).transpose()?,
            options:crate::remote::ViewerOptions::from_value(value)?,
            reader_leases:value.get("reader_leases").map(|v|v.as_bool().ok_or("Reader leases must be a boolean.")).transpose()?.unwrap_or(false),
            cache_bytes:value.get("cache_bytes").map(|v|v.as_u64().filter(|n|(64*1024*1024..=1024_u64.pow(4)).contains(n)).ok_or("Viewer cache must be 64 MiB to 1 TiB.")).transpose()?},
        "sync_native_plots"=>Action::SyncNativePlots{job:argument.unwrap().to_owned(),domain:domain()?,sequence:value["sequence"].as_u64().filter(|n|*n>0&&*n<=i64::MAX as u64).ok_or("Native plots require an exact positive frame sequence.")?},
        "artifact_index"=>Action::ArtifactIndex{job:argument.unwrap().to_owned(),domain:domain()?,
            after_sequence:match value.get("after_sequence"){None=>0,Some(value)=>value.as_u64().filter(|n|*n<=i64::MAX as u64).ok_or("Artifact cursor must be a nonnegative integer.")?}},
        "launch_downscale" => Action::LaunchDownscale(Box::new(parse_downscale(value)?)),
        "open_config" => Action::OpenConfig(PathBuf::from(argument.unwrap())),
        "reset_setup" => Action::ResetSetup,
        "focus_nodes" => Action::FocusNodes,
        "focus_setup" => Action::FocusSetup,
        "select_target" => Action::SelectTarget,
        "open_workspace" => Action::OpenWorkspace,
        _ => Action::FocusLogs,
    };
    Ok(Request { id: id.into(), name: name.into(), action, target, plan_sha256, config_sha256, review_id, review_sha256 })
}

/// A claim older than this in a session that is closed or has stopped
/// heartbeating is abandoned. Matches COMPANION_QUEUE_TIMEOUT (main.rs) and the
/// worker readiness limit (job.rs); the visual workspace waits 120-240 s.
const ABANDONED_CLAIM: Duration = Duration::from_secs(30);

fn claimed_action(value: Option<&Value>) -> String {
    value.and_then(|value| value["action"].as_str()).filter(|name| valid_id(name)).unwrap_or("unknown").to_owned()
}

/// The one response writer. Every claimed request id ends in exactly this
/// document, whichever session (current or a previous, dead one) answers it.
fn write_response(directory:&Path,session_id:&str,id:&str,action:&str,result:Result<String,String>,details:Value)->Result<(),String>{
    let ok = result.is_ok();
    let message = match result { Ok(message) | Err(message) => message };
    let mut response=json!({
        "schema":"arwen.companion-response.v1", "session_id":session_id, "id":id, "action":action,
        "ok":ok, "message":message, "job_id":null, "job_dir":null, "tui_version":TUI_VERSION,
    });
    if let Some(fields)=details.as_object(){for(key,value)in fields{
        if !["target","job_id","job_dir","remote_output_root","review_id","review_sha256","review_path",
            "artifact_manifest_path","artifact_manifest_sha256","artifact_index_path","artifact_index_sha256",
            "transferred_bytes","waiting","cache_recovery","jobs","handoff","processed_frame","native_plots",
            "downscale_plan_path","child_config_path","mode"].contains(&key.as_str()) {return Err("Unsupported companion response detail.".into());}
        response[key]=value.clone();
    }}
    atomic_json(&directory.join("responses").join(format!("{id}.json")), &response).map_err(|e|e.to_string())
}

fn unix_ms(time: SystemTime) -> u128 { time.duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() }

/// This terminal's own release; travels with every status, response and
/// handoff so a GUI can tell a 2.6.x terminal (no sync_processed_frame_v2)
/// from the one it was packaged with.
pub const TUI_VERSION: &str = env!("CARGO_PKG_VERSION");

/// The installed engine's version as the configured interpreter reports it
/// (`gpuwm.__version__`, from distribution metadata; "0+unknown" for an
/// uninstalled source tree). The probe is bounded even when startup/import
/// hangs; -P keeps the launch folder off its path.
pub fn engine_version(python: &Path) -> Result<String, String> {
    let mut command = Command::new(python);
    command.args(["-P", "-B", "-c", "import gpuwm,sys;sys.stdout.write(gpuwm.__version__)"])
        .stdin(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(windows_sys::Win32::System::Threading::CREATE_NO_WINDOW);
    }
    probe_version(&mut command, Duration::from_secs(20))
}

struct ProbeLog { path: PathBuf, file: fs::File }
impl ProbeLog {
    fn create(label: &str) -> io::Result<Self> {
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
        let path = env::temp_dir().join(format!("arwen-runtime-{}-{stamp}-{label}.log", std::process::id()));
        let file = fs::OpenOptions::new().read(true).write(true).create_new(true).open(&path)?;
        Ok(Self { path, file })
    }
    fn text(&mut self, limit: u64) -> io::Result<String> {
        self.file.seek(SeekFrom::Start(0))?;
        let mut bytes = Vec::new();
        Read::by_ref(&mut self.file).take(limit).read_to_end(&mut bytes)?;
        Ok(String::from_utf8_lossy(&bytes).trim().to_owned())
    }
}
impl Drop for ProbeLog {
    fn drop(&mut self) { let _ = fs::remove_file(&self.path); }
}

fn probe_version(command: &mut Command, timeout: Duration) -> Result<String, String> {
    // Regular files avoid pipe-reader threads and inherited-pipe EOF waits.
    // A noisy startup is stopped before its diagnostic files can grow freely.
    let mut stdout = ProbeLog::create("stdout").map_err(|e| format!("Cannot capture the ArWen runtime version: {e}"))?;
    let mut stderr = ProbeLog::create("stderr").map_err(|e| format!("Cannot capture ArWen runtime diagnostics: {e}"))?;
    command.stdout(Stdio::from(stdout.file.try_clone().map_err(|e| e.to_string())?))
        .stderr(Stdio::from(stderr.file.try_clone().map_err(|e| e.to_string())?));
    let mut child = command.spawn().map_err(|e| format!("Cannot start the selected ArWen runtime: {e}"))?;
    let deadline = Instant::now() + timeout;
    let outcome = loop {
        let error = match child.try_wait() {
            Ok(Some(outcome)) => break outcome,
            Ok(None) if Instant::now() >= deadline => Some(format!("ArWen runtime version check timed out after {} seconds.", timeout.as_secs_f64())),
            Ok(None) if stdout.file.metadata().map(|v| v.len() > 4096).unwrap_or(true)
                || stderr.file.metadata().map(|v| v.len() > 65536).unwrap_or(true) => Some("ArWen runtime version check produced excessive output.".into()),
            Ok(None) => None,
            Err(error) => Some(format!("Cannot check the selected ArWen runtime: {error}")),
        };
        if let Some(error) = error {
            let _ = child.kill();
            let _ = child.wait();
            return Err(error);
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    if !outcome.success() {
        let detail = stderr.text(4096).unwrap_or_default();
        return Err(format!("The selected runtime could not load ArWen ({outcome}). {detail}"));
    }
    if stdout.file.metadata().map(|v| v.len() > 4096).unwrap_or(true)
        || stderr.file.metadata().map(|v| v.len() > 65536).unwrap_or(true) {
        return Err("ArWen runtime version check produced excessive output.".into());
    }
    let version = stdout.text(4097).map_err(|e| format!("Cannot read the ArWen runtime version: {e}"))?;
    if version.is_empty() || version.len() > 64 || !version.bytes().all(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'+' | b'-' | b'_')) {
        return Err("The selected runtime returned an invalid ArWen version.".into());
    }
    Ok(version)
}

/// Bound eligible work, not the history of completed or already claimed work.
/// create_new at dispatch remains the authoritative claim against races.
fn pending_request_paths(directory: &Path, paths: impl Iterator<Item = PathBuf>) -> Vec<PathBuf> {
    let mut pending: Vec<_> = paths.filter(|path| {
        if !path.extension().is_some_and(|extension| extension == "json") { return false; }
        let Some(id) = path.file_stem().and_then(|name| name.to_str()).filter(|id| valid_id(id)) else { return false; };
        !directory.join("responses").join(format!("{id}.json")).exists()
            && !directory.join("claimed").join(format!("{id}.json")).exists()
    }).take(4096).collect();
    pending.sort();
    pending
}

/// A previous control session is dead once it published `closed`, or once its
/// status heartbeat (500 ms cadence) is older than ABANDONED_CLAIM, or when it
/// never published a status and is older than that limit.
fn session_dead(directory: &Path, now: u128) -> bool {
    let limit = ABANDONED_CLAIM.as_millis();
    match read_json(&directory.join("status.json"), 1024 * 1024) {
        Ok(status) => status["state"] == "closed"
            || status["heartbeat_unix_ms"].as_u64().is_none_or(|beat| now.saturating_sub(u128::from(beat)) >= limit),
        Err(_) => fs::metadata(directory).and_then(|meta| meta.modified()).map(unix_ms)
            .is_ok_and(|created| now.saturating_sub(created) >= limit),
    }
}

/// Answer every claim a dead previous session left without a response. Only
/// sibling `companion-*` sessions under the same `.arwen-tui` are visited; the
/// current session is skipped. Best effort: an unwritable response changes
/// nothing, the next start tries again.
fn reap_abandoned_claims(parent: &Path, current_id: &str) {
    let Ok(sessions) = fs::read_dir(parent) else { return; };
    let now = now_ms();
    for session in sessions.filter_map(Result::ok).take(4096) {
        let directory = session.path();
        let Some(name) = directory.file_name().and_then(|name| name.to_str()) else { continue; };
        let Some(session_id) = name.strip_prefix("companion-") else { continue; };
        if session_id == current_id || !directory.is_dir() || !session_dead(&directory, now) { continue; }
        let Ok(claims) = fs::read_dir(directory.join("claimed")) else { continue; };
        for claim in claims.filter_map(Result::ok).take(4096) {
            let path = claim.path();
            let Some(id) = path.file_stem().and_then(|name| name.to_str()).filter(|id| valid_id(id)) else { continue; };
            if !path.extension().is_some_and(|extension| extension == "json") { continue; }
            if directory.join("responses").join(format!("{id}.json")).exists() { continue; }
            let evidence = read_json(&path, 64 * 1024).ok();
            let claimed = evidence.as_ref().and_then(|value| value["claimed_unix_ms"].as_u64()).map(u128::from)
                .or_else(|| fs::metadata(&path).and_then(|meta| meta.modified()).ok().map(unix_ms));
            if claimed.is_some_and(|at| now.saturating_sub(at) < ABANDONED_CLAIM.as_millis()) { continue; }
            let action = claimed_action(evidence.as_ref().map(|value| &value["request"]));
            let _ = write_response(&directory, session_id, id, &action,
                Err("This request was claimed by a previous control center session that did not respond. Retry it in the current session.".into()),
                Value::Null);
        }
    }
}

/// A session still publishing inside this window may still be running and is
/// never removed, whoever started it. It matches the abandoned-claim limit and
/// the 30 s the single-controller rule already uses.
const VIEWER_SESSION_LIVE: Duration = ABANDONED_CLAIM;

/// Whether `pid` still names a running process. Anything that cannot be
/// established answers `true`: a sweep must never remove the directory a
/// running viewer is still publishing into, while keeping a dead session's
/// directory until one more start costs nothing.
fn process_is_running(pid: u32) -> bool {
    if pid == 0 { return true; }
    #[cfg(unix)]
    {
        if unsafe { libc::kill(pid as libc::pid_t, 0) } == 0 { return true; }
        // EPERM: the process exists and belongs to another user.
        io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, ERROR_INVALID_PARAMETER};
        use windows_sys::Win32::System::Threading::{GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};
        unsafe {
            let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
            // A pid with no process answers ERROR_INVALID_PARAMETER; a refusal
            // for any other reason means the process is there.
            if handle.is_null() { return GetLastError() != ERROR_INVALID_PARAMETER; }
            let mut code = 0u32;
            let read = GetExitCodeProcess(handle, &mut code);
            CloseHandle(handle);
            // 0x103 is STILL_ACTIVE. A process that genuinely exited with that
            // code reads as running, which only keeps its directory.
            read == 0 || code == 0x103
        }
    }
    #[cfg(not(any(unix, windows)))]
    { true }
}

/// The published status of a read-only run viewer session under `directory`
/// that nothing can still be using, if that is what `directory` is. Three
/// shapes qualify. The viewer published `closed`, which only a session's own
/// end writes, so it is finished whoever owns it -- that is the shape a removal
/// a held-open file blocked leaves behind, and the process that wrote it is
/// usually gone by the time anything sweeps. Or it stopped publishing longer
/// ago than VIEWER_SESSION_LIVE and its process is no longer running, which is
/// what a killed TUI leaves. Or the directory is empty, no longer young, and
/// named for this process or for a pid that has ended, which is what a removal
/// blocked on the status file itself leaves once its reader lets go.
/// Everything else answers `None`: a viewer that still heartbeats, an
/// unfinished session whose own process is still running -- this process
/// included, because a run viewer runs inside the controller's process and a
/// live one whose heartbeat has paused must never be swept -- a session that
/// is not read-only (a controller owns forecasts and answers requests), a
/// directory whose status is missing or names another session, and every
/// `job-*` receipt directory, which never carries a `companion-` name in the
/// first place.
fn finished_viewer_session(directory: &Path, now: u128) -> Option<Value> {
    let id = directory.file_name().and_then(|name| name.to_str())?.strip_prefix("companion-")?;
    if !directory.is_dir() { return None; }
    let Ok(status) = read_json(&directory.join("status.json"), 1024 * 1024) else {
        // An empty session directory carrying no status at all is what a
        // removal a reader blocked on the status file itself leaves behind once
        // that reader lets go. Nothing inside it can say whether it was
        // read-only, so its owner comes from its name, which every session
        // builds as `<pid>-<nanos>`: this process, whose own viewers it may
        // finish while it runs, or a pid that has ended. Another live TUI's
        // debris is left to that TUI. A session that is starting always holds
        // its request queues, and a young directory is waited for, so no live
        // session of any kind is ever this shape.
        let owner = id.split_once('-').and_then(|(pid, _)| pid.parse::<u32>().ok())?;
        if owner != std::process::id() && process_is_running(owner) { return None; }
        let empty = fs::read_dir(directory).ok()?.next().is_none();
        let old = fs::metadata(directory).and_then(|meta| meta.modified()).map(unix_ms)
            .is_ok_and(|at| now.saturating_sub(at) >= VIEWER_SESSION_LIVE.as_millis());
        return (empty && old).then_some(Value::Null);
    };
    if status["schema"] != "arwen.companion-status.v1" || status["session_id"] != id || status["read_only"] != true { return None; }
    if status["state"] == "closed" { return Some(status); }
    let quiet = status["heartbeat_unix_ms"].as_u64().map(u128::from)
        .is_none_or(|beat| now.saturating_sub(beat) >= VIEWER_SESSION_LIVE.as_millis());
    let pid = status["tui_pid"].as_u64().and_then(|pid| u32::try_from(pid).ok())?;
    // Not `pid == std::process::id()`: a run viewer lives in the controller's
    // own process, so this process's pid on an unfinished session is a viewer
    // that is very likely still open. A viewer of this process that has ended
    // published `closed` above, and one that never published a status has no
    // status to reach here, so nothing needs the shortcut and it was the only
    // thing that made a live viewer of this process removable.
    let ended = !process_is_running(pid);
    (quiet && ended).then_some(status)
}

/// Removes one finished viewer session directory, with the raw and processed
/// frame caches that viewer downloaded into it. Best effort: a file a reader
/// still holds open leaves the rest of the directory in place, and the closed
/// status stays or is written again so the leftovers remain a recognisable
/// finished session for the next sweep rather than a statusless directory that
/// a starting controller would wait VIEWER_SESSION_LIVE for. A directory that
/// is already gone is the wanted state, not a blocked removal.
fn discard_viewer_session(directory: &Path, closed: &Value) -> bool {
    let status = directory.join("status.json");
    // Everything but the status first. A file another process holds open
    // cannot be replaced on Windows once it is marked for deletion, so the
    // status a blocked removal has to leave behind is only removed when
    // nothing else is left to block it.
    let mut blocked = false;
    if let Ok(entries) = fs::read_dir(directory) {
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            if path == status { continue; }
            let removed = if entry.file_type().is_ok_and(|kind| kind.is_dir()) { fs::remove_dir_all(&path) } else { fs::remove_file(&path) };
            blocked |= removed.is_err() && path.exists();
        }
    }
    if !blocked {
        let _ = fs::remove_file(&status);
        match fs::remove_dir(directory) {
            Ok(()) => return true,
            // Already gone is the wanted state, not a blocked removal:
            // recreating the directory here would leave the leftover this
            // exists to take away.
            Err(error) if error.kind() == io::ErrorKind::NotFound => return true,
            Err(_) => {}
        }
    }
    if closed.is_object() && fs::create_dir_all(directory).is_ok() && !status.exists() {
        let _ = atomic_json(&status, closed);
    }
    false
}

/// The controller's startup and exit sweep: every finished read-only viewer
/// session directory under `parent`, except `keep`, is removed. A live session,
/// a controller session and a job directory are never touched. Leftovers exist
/// only when a viewer's own removal could not finish, so this normally removes
/// nothing. Returns how many directories went.
fn remove_finished_viewer_sessions(parent: &Path, keep: &str) -> usize {
    let Ok(entries) = fs::read_dir(parent) else { return 0; };
    let keep = format!("companion-{keep}");
    let now = now_ms();
    let mut removed = 0;
    for entry in entries.filter_map(Result::ok).take(4096) {
        let directory = entry.path();
        if directory.file_name().and_then(|name| name.to_str()) == Some(keep.as_str()) { continue; }
        let Some(status) = finished_viewer_session(&directory, now) else { continue; };
        if discard_viewer_session(&directory, &status) { removed += 1; }
    }
    removed
}

/// A controller whose status heartbeat (500 ms cadence) is older than this is
/// not asked to reopen its workspace. It matches the abandoned-claim limit
/// rather than the desktop's 10 s rule because a controller's heartbeat pauses
/// for the engine-version probe each time a run viewer opens (up to 20 s).
const LIVE_CONTROLLER_HEARTBEAT: Duration = ABANDONED_CLAIM;

/// A controller publishes its first status only once its workspace has started,
/// seconds after its session directory appears. Two launches inside that window
/// would both become controllers, so a young session directory without a status
/// is waited for until it publishes, closes or the abandonment limit passes.
fn wait_for_starting_controllers(parent: &Path) {
    let deadline = Instant::now() + ABANDONED_CLAIM;
    loop {
        let now = now_ms();
        let starting = fs::read_dir(parent).ok().into_iter().flatten().filter_map(Result::ok).take(4096).any(|entry| {
            let directory = entry.path();
            directory.file_name().and_then(|name| name.to_str()).is_some_and(|name| name.starts_with("companion-"))
                && directory.is_dir() && !directory.join("status.json").exists()
                && fs::metadata(&directory).and_then(|meta| meta.modified()).map(unix_ms)
                    .is_ok_and(|created| now.saturating_sub(created) < ABANDONED_CLAIM.as_millis())
        });
        if !starting || Instant::now() >= deadline { return; }
        std::thread::sleep(Duration::from_millis(200));
    }
}

/// The live controller session under `parent`, if any: a heartbeating,
/// unclosed `arwen.companion-status.v1` publisher whose directory name matches
/// its session id. Read-only run viewers publish the same schema with
/// `read_only: true` and own no forecast, so they are never the answer. With
/// several candidates the freshest heartbeat wins. Returns the session id, its
/// directory and its published status.
pub(crate) fn live_controller(parent: &Path, now: u128) -> Option<(String, PathBuf, Value)> {
    let sessions = fs::read_dir(parent).ok()?;
    let mut best: Option<(u128, String, PathBuf, Value)> = None;
    for session in sessions.filter_map(Result::ok).take(4096) {
        let directory = session.path();
        let Some(id) = directory.file_name().and_then(|name| name.to_str()).and_then(|name| name.strip_prefix("companion-")) else { continue; };
        let Ok(status) = read_json(&directory.join("status.json"), 1024 * 1024) else { continue; };
        if status["schema"] != "arwen.companion-status.v1" || status["session_id"] != id
            || status["read_only"] == true || status["state"] == "closed" { continue; }
        let Some(beat) = status["heartbeat_unix_ms"].as_u64().map(u128::from) else { continue; };
        if now.saturating_sub(beat) >= LIVE_CONTROLLER_HEARTBEAT.as_millis() { continue; }
        if best.as_ref().is_none_or(|(freshest, ..)| beat > *freshest) { best = Some((beat, id.to_owned(), directory, status)); }
    }
    best.map(|(_, id, directory, status)| (id, directory, status))
}

/// One controller per output root. A second `--headless-companion` start over
/// a root whose controller still heartbeats would own nothing: the first
/// controller's forecast is only a read-only saved run to it and cannot be
/// stopped from the second workspace. So the newcomer asks the live controller
/// to reopen its workspace and, on success, exits. `None` means no live
/// controller and the newcomer becomes one. `Some(Err(_))` means a live
/// controller refused or did not answer and still lives: the newcomer must
/// exit with that message, never become a second owner. An earlier terminal
/// that does not know the request refuses it the same way.
pub fn reopen_live_controller(output: &Path) -> Option<Result<String, String>> {
    let parent = output.join(".arwen-tui");
    wait_for_starting_controllers(&parent);
    let (session_id, directory, status) = live_controller(&parent, now_ms())?;
    if let Some(pid) = status["tui_pid"].as_u64().and_then(|pid| u32::try_from(pid).ok()).filter(|pid| *pid > 0) { allow_foreground(pid); }
    let outcome = request_open_workspace(&directory, &session_id, LIVE_CONTROLLER_HEARTBEAT);
    let reason = match outcome { Ok(message) => return Some(Ok(message)), Err(reason) => reason };
    // Only a controller that has since gone quiet leaves the root to the newcomer.
    match live_controller(&parent, now_ms()) {
        Some((live, ..)) if live == session_id => {
            let version = status["tui_version"].as_str().unwrap_or("an earlier version");
            Some(Err(format!("ArWen (terminal {version}) is already running for this workspace and could not reopen it: {reason} Let its forecast finish or stop it there, or close that ArWen, before opening ArWen again.")))
        }
        _ => None,
    }
}

/// Grants the live controller the right to bring its workspace window to the
/// foreground: the newcomer was just started by the user, the controller was
/// not. Best effort; Windows only.
fn allow_foreground(pid: u32) {
    #[cfg(windows)]
    unsafe {
        #[link(name = "user32")]
        unsafe extern "system" { fn AllowSetForegroundWindow(process_id: u32) -> i32; }
        AllowSetForegroundWindow(pid);
    }
    #[cfg(not(windows))]
    let _ = pid;
}

/// Writes an `open_workspace` request the way the visual workspace writes its
/// own (a temporary file renamed into `requests/`), then waits for the answer.
fn request_open_workspace(directory: &Path, session_id: &str, wait: Duration) -> Result<String, String> {
    let id = format!("open-workspace-{}-{}", std::process::id(), now_ms());
    let request = json!({"schema": "arwen.companion-request.v1", "session_id": session_id, "id": id, "action": "open_workspace"});
    atomic_json(&directory.join("requests").join(format!("{id}.json")), &request)
        .map_err(|error| format!("Cannot ask the running ArWen controller to reopen its workspace: {error}"))?;
    let response = directory.join("responses").join(format!("{id}.json"));
    let deadline = Instant::now() + wait;
    loop {
        if let Ok(value) = read_json(&response, 64 * 1024) {
            let message = value["message"].as_str().unwrap_or("no message").to_owned();
            return if value["ok"] == true { Ok(message) } else { Err(format!("The running ArWen controller could not reopen its workspace: {message}")) };
        }
        if Instant::now() >= deadline {
            return Err(format!("The running ArWen controller did not answer within {} seconds.", wait.as_secs()));
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// `<output>/.arwen-tui/controller.log`: the desktop launcher's controller has
/// a hidden console or no console at all, so its fatal errors and startup notes
/// are appended here (UTC stamp, one line) as well as written to stderr. The
/// launcher reads the tail when the controller exits nonzero.
pub fn controller_log(output: &Path, message: &str) {
    let parent = output.join(".arwen-tui");
    if fs::create_dir_all(&parent).is_err() { return; }
    let Ok(mut log) = fs::OpenOptions::new().create(true).append(true).open(parent.join("controller.log")) else { return; };
    let stamp = i64::try_from(now_ms()).ok().and_then(|now| local_progress::utc_text(now).ok()).unwrap_or_default();
    let _ = writeln!(log, "{stamp} {}", message.trim_end());
}

pub struct Session {
    pub id: String,
    pub directory: PathBuf,
    pub handoff: PathBuf,
    /// A read-only run viewer's session: it owns no forecast, answers only
    /// read requests, and its directory lives exactly as long as the viewer.
    read_only: bool,
    /// When this controller last swept finished viewer sessions. A removal a
    /// held-open file blocked has to be retried while the workspace is open:
    /// the reader releases the file seconds later, and waiting for the next
    /// start would leave every remote viewer of an evening on disk.
    swept: Instant,
    last_status: Value,
    published: Instant,
}
impl Session {
    #[cfg(test)]
    pub(crate) fn test_session(output:&Path)->Result<Self,String>{Self::create(output)}
    pub(crate) fn create(output: &Path) -> Result<Self, String> { Self::open(output, false) }
    /// A read-only run viewer's session. Its directory is scratch: `Drop`
    /// publishes `closed` and then removes it, so opening saved runs several
    /// times in an evening leaves nothing behind under `.arwen-tui`.
    pub(crate) fn create_read_only(output: &Path) -> Result<Self, String> { Self::open(output, true) }
    fn open(output: &Path, read_only: bool) -> Result<Self, String> {
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
        let id = format!("{}-{stamp}", std::process::id());
        let parent = output.join(".arwen-tui");
        fs::create_dir_all(&parent).map_err(|e| e.to_string())?;
        let directory = parent.join(format!("companion-{id}"));
        fs::create_dir(&directory).map_err(|e| e.to_string())?;
        let directory = directory.canonicalize().map_err(|e| e.to_string())?;
        for child in ["requests", "responses", "claimed"] { fs::create_dir(directory.join(child)).map_err(|e| e.to_string())?; }
        reap_abandoned_claims(&parent, &id);
        // The controller sweeps at startup; a viewer removes only its own
        // directory, so one viewer never decides another viewer is finished.
        if !read_only { remove_finished_viewer_sessions(&parent, &id); }
        Ok(Self { id, handoff: directory.join("handoff.json"), directory, read_only, swept: Instant::now(),
            last_status: Value::Null, published: Instant::now() - Duration::from_secs(1) })
    }
    pub fn publish(&mut self, mut status: Value, force: bool) -> Result<(), String> {
        if !force && self.published.elapsed() < Duration::from_millis(500) { return Ok(()); }
        status["schema"] = json!("arwen.companion-status.v1");
        status["session_id"] = json!(self.id);
        status["tui_pid"] = json!(std::process::id());
        status["tui_version"] = json!(TUI_VERSION);
        status["heartbeat_unix_ms"] = json!(now_ms());
        if self.read_only { status["read_only"] = json!(true); }
        atomic_json(&self.directory.join("status.json"), &status).map_err(|e| e.to_string())?;
        self.last_status = status;
        self.published = Instant::now();
        // The controller retries the blocked removals on its own heartbeat, so
        // a viewer whose frames a reader still held stops surviving the
        // session that opened it. One scan per VIEWER_SESSION_LIVE reads only
        // the `companion-*` statuses beside this one.
        if !self.read_only && self.swept.elapsed() >= VIEWER_SESSION_LIVE {
            self.swept = Instant::now();
            if let Some(parent) = self.directory.parent() { remove_finished_viewer_sessions(parent, &self.id); }
        }
        Ok(())
    }
    pub fn due(&self) -> bool { self.published.elapsed() >= Duration::from_millis(500) }
    pub(crate) fn publish_handoff(&self, mut context:Value)->Result<Value,String>{
        context["schema"]=json!("arwen.companion-handoff.v1");
        context["session_id"]=json!(self.id);context["tui_pid"]=json!(std::process::id());
        context["control_dir"]=json!(self.directory);context["status_path"]=json!(self.directory.join("status.json"));
        // Additive: the schema string is unchanged, so a current GUI still
        // parses the handoff. The GUI side is expected to read and compare
        // these before it assumes which actions the terminal understands.
        context["tui_version"]=json!(TUI_VERSION);
        let python = context["python"].as_str().ok_or("The visual workspace requires a selected ArWen runtime.")?;
        context["engine_version"] = Value::String(engine_version(Path::new(python))?);
        atomic_json(&self.handoff,&context).map_err(|error|error.to_string())?;
        Ok(context)
    }
    pub fn requests(&mut self) -> Vec<Request> {
        let Ok(entries) = fs::read_dir(self.directory.join("requests")) else { return Vec::new(); };
        let paths = pending_request_paths(&self.directory, entries.filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_file()))
            .map(|entry| entry.path()));
        let mut requests = Vec::new();
        for path in paths {
            if requests.len() >= 8 { break; }
            let Some(id) = path.file_stem().and_then(|name| name.to_str()).filter(|id| valid_id(id)) else { continue; };
            if self.directory.join("responses").join(format!("{id}.json")).exists() { continue; }
            let claim = self.directory.join("claimed").join(format!("{id}.json"));
            let Ok(mut claimed) = fs::OpenOptions::new().write(true).create_new(true).open(claim) else { continue; };
            // From here the id is claimed: every exit below writes a response,
            // because a claim without a response is skipped forever by this
            // loop and the visual workspace can only wait out its timeout.
            let value = read_json(&path, 64 * 1024);
            let evidence = json!({"id": id, "claimed_unix_ms": now_ms(), "request": value.as_ref().ok()});
            let recorded = serde_json::to_writer(&mut claimed, &evidence).and_then(|_| claimed.sync_all().map_err(serde_json::Error::io))
                .map_err(|error| format!("The control center could not record its claim on this request: {error}"));
            drop(claimed);
            let action = claimed_action(value.as_ref().ok());
            let parsed = recorded.and_then(|_| value).and_then(|value| parse_request(&value, id, &self.id));
            match parsed {
                Ok(request) => requests.push(request),
                Err(error) => { let _ = self.respond(id, &action, Err(error), None); }
            }
        }
        requests
    }
    pub fn respond(&self, id: &str, action: &str, result: Result<String, String>, job: Option<&Path>) -> Result<(), String> {
        self.respond_with(id,action,result,json!({"job_id":job,"job_dir":job}))
    }
    pub fn respond_with(&self,id:&str,action:&str,result:Result<String,String>,details:Value)->Result<(),String>{
        write_response(&self.directory,&self.id,id,action,result,details)
    }
    pub fn save_review(&self,id:&str,review:&Value)->Result<(PathBuf,String),String>{
        let path=self.review_path(id)?;
        fs::create_dir_all(path.parent().unwrap()).map_err(|e|e.to_string())?;
        if path.exists(){return Err("This control request already has a saved review.".into());}
        atomic_json(&path,review).map_err(|e|e.to_string())?;
        let bytes=fs::read(&path).map_err(|e|e.to_string())?;
        Ok((path,digest(&bytes)))
    }
    pub fn review_path(&self,id:&str)->Result<PathBuf,String>{
        if !valid_id(id){return Err("Invalid companion review ID.".into());}
        Ok(self.directory.join("reviews").join(format!("{id}.json")))
    }
    pub fn save_artifacts(&self,id:&str,value:&Value)->Result<(PathBuf,String),String>{
        if !valid_id(id){return Err("Invalid artifact request ID.".into());}
        let directory=self.directory.join("artifacts");
        fs::create_dir_all(&directory).map_err(|e|e.to_string())?;
        let path=directory.join(format!("{id}.json"));
        if path.exists(){return Err("This control request already has an artifact receipt.".into());}
        atomic_json(&path,value).map_err(|e|e.to_string())?;
        Ok((path.clone(),digest(&fs::read(path).map_err(|e|e.to_string())?)))
    }
}
impl Drop for Session {
    fn drop(&mut self) {
        if self.last_status.is_object() {
            let mut status = self.last_status.clone();
            status["state"] = json!("closed");
            let _ = self.publish(status, true);
        }
        if self.read_only {
            // The viewer published `closed` above and every transfer it
            // started has ended. Its directory holds that session's handoff,
            // queues and receipts, and the raw and processed frames it
            // downloaded into `remote-artifacts` and `processed-store`. The
            // desktop confines a frame lease to the control directory of the
            // handoff it is holding, and every viewer publishes a new session,
            // so those two caches are reachable only through this viewer and
            // go with it. The caches keyed by node and job under the output
            // root, `.arwen-viewer-cache` and `.arwen-native-plots-cache`,
            // outlive every viewer and are never touched here.
            discard_viewer_session(&self.directory, &self.last_status);
        } else if let Some(parent) = self.directory.parent() {
            // The controller's exit sweep: anything a viewer could not remove.
            remove_finished_viewer_sessions(parent, &self.id);
        }
    }
}

#[derive(Default)]
pub struct Controller {
    pub explicit_path: Option<PathBuf>,
    pub session: Option<Session>,
    child: Option<Child>,
}
impl Controller {
    pub fn ensure_session(&mut self,output:&Path)->Result<(),String>{
        if self.session.is_none(){self.session=Some(Session::create(output)?);}
        Ok(())
    }
    pub fn open(&mut self, cwd: &Path, output: &Path, context: Value) -> Result<String, String> {
        if self.child.as_mut().is_some_and(|child| child.try_wait().ok() == Some(None)) {
            return Ok("Visual workspace is already open.".into());
        }
        let executable = self.explicit_path.clone().or_else(|| env::var_os("ARWEN_COMPANION").map(PathBuf::from))
            .unwrap_or_else(|| env::current_exe().unwrap_or_default().with_file_name(if cfg!(windows) { "arwen-companion.exe" } else { "arwen-companion" }));
        let executable = if executable.is_absolute() { executable } else { cwd.join(executable) };
        if !executable.is_file() {
            return Err("Visual workspace is not installed. Set --companion PATH or ARWEN_COMPANION to its application.".into());
        }
        if self.session.is_none() { self.session = Some(Session::create(output)?); }
        let session = self.session.as_mut().unwrap();
        session.publish_handoff(context)?;
        let log_path = session.directory.join("companion.log");
        let mut log = fs::OpenOptions::new().create(true).append(true).open(&log_path)
            .map_err(|e| format!("Cannot open the visual workspace diagnostic log {}: {e}", log_path.display()))?;
        writeln!(log, "\n[ArWen TUI] Visual workspace launch at {} ms", now_ms())
            .map_err(|e| format!("Cannot write the visual workspace diagnostic log: {e}"))?;
        let stdout = log.try_clone().map_err(|e| format!("Cannot attach the visual workspace diagnostic log: {e}"))?;
        self.child = Some(Command::new(executable).arg("--handoff").arg(&session.handoff)
            .current_dir(cwd).stdin(Stdio::null()).stdout(Stdio::from(stdout)).stderr(Stdio::from(log))
            .spawn().map_err(|e| format!("Could not open the visual workspace: {e}. Diagnostic log: {}", log_path.display()))?);
        Ok(format!("Visual workspace process started. Diagnostic log: {}. Forecast jobs stay in this control center.", log_path.display()))
    }
    pub fn requests(&mut self) -> Vec<Request> { self.session.as_mut().map(Session::requests).unwrap_or_default() }
    /// The visual workspace process this controller spawned, while it is alive.
    pub fn child_pid(&mut self) -> Option<u32> {
        let child = self.child.as_mut()?;
        (child.try_wait().ok() == Some(None)).then(|| child.id())
    }
    pub fn child_status(&mut self) -> io::Result<Option<std::process::ExitStatus>> {
        self.child.as_mut().ok_or_else(|| io::Error::new(io::ErrorKind::NotConnected,
            "The visual workspace has not started."))?.try_wait()
    }
}

pub fn plan_run_dir(path: &Path) -> Option<PathBuf> {
    let plan = read_json(path, 4 * 1024 * 1024).ok()?;
    let output = plan["output_root"].as_str().unwrap_or("out/run");
    let output = PathBuf::from(output);
    Some(if output.is_absolute() { output } else { path.parent()?.join(output) })
}

pub fn job_status(job: &crate::job::Job, output: &Path) -> Value {
    let plan = (job.action == "run-plan").then(|| job.command.get(4).map(PathBuf::from)).flatten();
    let run = plan.as_deref().and_then(plan_run_dir);
    let manifest_path = run.as_ref().map(|root| root.join("run-manifest.json"));
    let process_path = job.dir.join("process.json");
    let process = read_json(&process_path, 64 * 1024).ok();
    let fresh_manifest = manifest_path.as_ref().and_then(|path| fs::metadata(path).ok()?.modified().ok())
        .zip(fs::metadata(job.dir.join("job.json")).ok().and_then(|value| value.modified().ok()))
        .is_some_and(|(manifest, launched)| manifest >= launched);
    let manifest = manifest_path.as_deref().and_then(|path| read_json(path, 256 * 1024).ok())
        .filter(|manifest| fresh_manifest && manifest["schema"] == "gpuwm.run-manifest.v1"
            && manifest["pid"].as_u64().is_some()
            && manifest["pid"].as_u64() == process.as_ref().and_then(|value| value["pid"].as_u64()));
    let named = |key: &str| manifest.as_ref().and_then(|value| value[key].as_str()).map(PathBuf::from);
    let roots: Vec<_> = run.iter().flat_map(|root| [root.clone(), root.join("run"), root.join("evidence"), root.join("chain/run"), root.join("chain/run/evidence")]).collect();
    let existing = |name: &str| if manifest.is_some() { roots.iter().map(|root| root.join(name)).find(|path| path.exists()) } else { None };
    let state = if job.is_stopping() { "stopping" } else { match job.outcome {
        Some(_) if job.interrupted() => "stopped", Some(0) => "completed", Some(_) => "failed",
        None if !job.dir.join("start").is_file() => "starting", None => "running",
    }};
    let mut status=json!({"job_id":job.dir,"job_dir":job.dir,"action":job.action,"state":state,"exit_code":job.outcome,
        "log_path":job.dir.join("job.log"),"process_path":process_path,"result_path":job.dir.join("result.json"),
        "plan_path":plan,"run_dir":run,"run_manifest_path":manifest_path,"manifest_ready":manifest.is_some(),
        "events_path":named("events_path"),"progress_path":named("progress_path"),
        "step_progress_path":existing("progress.jsonl"),"ready_dir":existing("ready"),
        "outputs_dir":named("outputs_dir").or_else(|| run.clone()).unwrap_or_else(|| output.to_owned()),
        "latest_run_pointer":output.join("latest-run.txt")});
    // A finished failure carries the worker's recorded reason -- an uncaught
    // exception or the refusal sentence the CLI printed -- the way a saved
    // row does, so a reader of the live status shows the sentence, not
    // "failed".
    if state=="failed"{
        if let Some(message)=read_json(&job.dir.join("result.json"),128*1024).ok()
            .and_then(|result|result["error"]["message"].as_str().map(str::trim).filter(|text|!text.is_empty()).map(str::to_owned)){
            status["error"]=json!(message);
        }
    }
    // The worker writes process.json before its ready handshake. Until that
    // handshake is released, an absent process receipt is ordinary startup.
    // Running and terminal jobs still require the complete receipt binding.
    if state!="starting"&&matches!(job.action.as_str(),"run-plan"|"go"|"sim"|"run"|"resume"|"downscale")&&!job.command.iter().any(|arg|matches!(arg.as_str(),"--dry-run"|"--estimate"|"--resolve"|"--physics-profiles")){
        match local_progress::cached(&job.dir,&job.command){
            Ok(native)=>{if let Some(fields)=native.as_object(){for(key,value)in fields{status[key]=value.clone();}}}
            Err(error)=>{status["progress_error"]=json!(error);status["manifest_ready"]=json!(false);for key in ["events_path","progress_path","ready_dir"]{status[key]=Value::Null;}}
        }
    }
    status
}

/// The launcher records its canonical working directory (`\\?\C:\...` on
/// Windows) while the worker reports `Path.cwd()` (`C:\...`). One directory,
/// two spellings: compare the directory, not the string. A string comparison
/// here refused every Windows job and emptied the desktop's run list.
pub(crate) fn same_directory(recorded:&str,reported:&str)->bool{
    if recorded==reported{return true;}
    let plain=|value:&str|value.strip_prefix(r"\\?\").unwrap_or(value).trim_end_matches(['\\','/']).to_owned();
    let (a,b)=(plain(recorded),plain(reported));
    if if cfg!(windows){a.eq_ignore_ascii_case(&b)}else{a==b}{return true;}
    matches!((Path::new(recorded).canonicalize(),Path::new(reported).canonicalize()),(Ok(x),Ok(y)) if x==y)
}

/// Historical local runs use their saved launcher/process/native receipts;
/// attaching a reader does not recreate a Child handle or claim its lifetime.
pub(crate) fn saved_job_status(directory:&Path)->Result<Value,String>{
    let directory=directory.canonicalize().map_err(|error|error.to_string())?;
    let launcher=read_json(&directory.join("job.json"),128*1024)?;
    let command=launcher["command"].as_array().filter(|values|values.len()>=5)
        .ok_or("Saved local job has no launch command.")?.iter().map(|value|value.as_str().map(str::to_owned)
            .ok_or_else(||"Saved local command contains a non-string argument.".to_owned())).collect::<Result<Vec<_>,_>>()?;
    if launcher["schema"]!="gpuwm-tui-job-v1"||launcher["action"]!=command[3]
        ||!matches!(command[3].as_str(),"run-plan"|"go"|"sim"|"run"|"resume"|"downscale")
        ||command.iter().any(|arg|matches!(arg.as_str(),"--dry-run"|"--estimate"|"--resolve"|"--physics-profiles"|"--help")){
        return Err("This saved command is not a forecast run.".into());
    }
    let process=read_json(&directory.join("process.json"),64*1024)?;
    let same_cwd=launcher["cwd"].as_str().zip(process["cwd"].as_str()).is_some_and(|(recorded,reported)|same_directory(recorded,reported));
    if process["schema"]!="gpuwm-tui-process-v1"||process["cli_args"]!=json!(&command[3..])
        ||!same_cwd||process["pid"].as_u64().is_none_or(|pid|pid==0){
        return Err("Saved local process does not match its original launch command.".into());
    }
    let result_path=directory.join("result.json");
    let result=if result_path.is_file(){Some(read_json(&result_path,128*1024)?)}else{None};
    // Two receipts end a job: the worker's own, bound to process.json by pid and
    // arguments, or the launcher's, written when the worker was terminated before
    // it could write (a Windows stop ends the job object at once) or when its
    // receipt failed verification. The launcher receipt is read only beside the
    // validated process receipt and never records a success.
    let launcher_receipt=result.as_ref().is_some_and(|value|value["schema"]=="gpuwm-tui-launcher-result-v1");
    if result.as_ref().is_some_and(|result|if launcher_receipt{
            !matches!(result["status"].as_str(),Some("stopped"|"failed"))||result["exit_code"].as_i64().is_none_or(|code|code==0)
        }else{result["schema"]!="gpuwm-tui-result-v1"
            ||result["pid"]!=process["pid"]||result["cli_args"]!=process["cli_args"]||result["exit_code"].as_i64().is_none()}){
        return Err("Saved local completion does not match its original process.".into());
    }
    let state=match result.as_ref(){
        Some(value) if launcher_receipt=>if value["status"]=="stopped"{"stopped"}else{"failed"},
        Some(value)=>match value["exit_code"].as_i64(){Some(0)=>"completed",Some(130)=>"stopped",_=>"failed"},
        None=>"running",
    };
    let created=fs::metadata(directory.join("job.json")).and_then(|metadata|metadata.modified()).ok()
        .and_then(|time|time.duration_since(UNIX_EPOCH).ok()).and_then(|elapsed|i64::try_from(elapsed.as_millis()).ok())
        .and_then(|milliseconds|local_progress::utc_text(milliseconds).ok());
    let mut status=json!({"id":directory,"job_id":directory,"job_dir":directory,"target":{"kind":"local"},
        "action":command[3],"state":state,"exit_code":result.as_ref().map(|value|value["exit_code"].clone()),
        "created_at":created,"started_at":process["started_at"],"ended_at":result.as_ref().map(|value|value["ended_at"].clone()),
        "pid":process["pid"],"process_path":directory.join("process.json"),"result_path":result_path,
        "log_path":directory.join("job.log"),"command":command,"cwd":launcher["cwd"]});
    // Native receipts exist once the run's own manifest does. A job that ended
    // before that point (failed while acquiring inputs, stopped while preparing)
    // is still a finished job with a reason worth showing; a running one without
    // them is the owning controller's to describe.
    match local_progress::cached(&directory,&command){
        Ok(native)=>{if let Some(fields)=native.as_object(){for(key,value)in fields{status[key]=value.clone();}}}
        Err(error) if state!="running"=>{
            status["progress_error"]=json!(error);status["manifest_ready"]=json!(false);
            if let Some(message)=result.as_ref().and_then(|value|value["message"].as_str().or_else(||value["error"]["message"].as_str())){status["error"]=json!(message);}
        }
        Err(error)=>return Err(error),
    }
    Ok(status)
}

#[cfg(test)]
mod downscale_requests {
    use super::*;
    fn scratch(label: &str) -> PathBuf {
        let path = env::temp_dir().join(format!("arwen-downscale-{label}-{}-{}", std::process::id(), now_ms()));
        fs::create_dir_all(&path).unwrap();
        path.canonicalize().unwrap()
    }
    fn payload(parent: &Path, out: &Path) -> Value {
        json!({"schema":"arwen.companion-request.v1","session_id":"s","id":"r","action":"launch_downscale",
            "target":{"kind":"local"},"parent_run_dir":parent,"point":{"lat":39.5,"lon":-84.0},
            "out_dir":out,"mode":"plan"})
    }
    fn parse(value: &Value) -> Result<DownscaleRequest, String> {
        match parse_request(value, "r", "s")?.action {
            Action::LaunchDownscale(body) => Ok(*body),
            _ => Err("not a downscale action".into()),
        }
    }
    #[test]
    fn the_defaults_are_the_narrow_door_and_every_exclusion_is_refused_by_name() {
        let root = scratch("defaults");
        let parent = root.join("parent-run");
        fs::create_dir(&parent).unwrap();
        let out = root.join("child-run");
        let body = parse(&payload(&parent, &out)).unwrap();
        assert_eq!(body.ratio, 3);
        assert!(body.auto_vram && body.accept_parent_cadence && body.plan);
        assert_eq!(body.parent_restart, None);
        assert_eq!(body.child_config, None);
        assert_eq!(body.point.map(|(lat, _)| lat), Some(39.5));
        assert_eq!(body.out_dir, out.to_string_lossy());

        // A node has neither input on its disk, and the refusal says which.
        let mut remote = payload(&parent, &out);
        remote["target"] = json!({"kind":"ssh","node_id":"node-1","connection_sha256":"a".repeat(64)});
        let refusal = parse(&remote).unwrap_err();
        assert!(refusal.contains("this computer's disk") && refusal.contains("gpuwm downscale"), "{refusal}");

        let mut unknown = payload(&parent, &out);
        unknown["child_levels"] = json!("40,2.5");
        assert!(parse(&unknown).unwrap_err().contains("Unknown field"));

        let mut both = payload(&parent, &out);
        both["child_config"] = json!(root.join("child.toml"));
        assert!(parse(&both).unwrap_err().contains("either a child centre point"));

        let mut neither = payload(&parent, &out);
        neither["point"] = Value::Null;
        assert!(parse(&neither).unwrap_err().contains("either a child centre point"));

        let mut sized = payload(&parent, &out);
        sized["child_size"] = json!({"nx":300,"ny":300});
        sized["auto_vram"] = json!(true);
        assert!(parse(&sized).unwrap_err().contains("measures it"));
        // Absent, fitting turns itself off for a caller that named an
        // extent: the default is the narrow door, not a contradiction.
        sized["auto_vram"] = Value::Null;
        let body = parse(&sized).unwrap();
        assert_eq!(body.child_size, Some((300, 300)));
        assert!(!body.auto_vram);

        let mut cadence = payload(&parent, &out);
        cadence["max_boundary_interval_seconds"] = json!(600);
        cadence["accept_parent_cadence"] = json!(true);
        assert!(parse(&cadence).unwrap_err().contains("not both"));
        // Absent, an explicit ceiling turns acceptance off by itself, the
        // way the guide's own two settings resolve each other.
        cadence["accept_parent_cadence"] = Value::Null;
        let body = parse(&cadence).unwrap();
        assert_eq!(body.max_boundary_interval_seconds, Some(600.0));
        assert!(!body.accept_parent_cadence);

        let mut missing = payload(&parent, &out);
        missing["mode"] = json!("go");
        assert!(parse(&missing).unwrap_err().contains("plan or run"));

        let mut relative = payload(&parent, &out);
        relative["out_dir"] = json!("child-run");
        assert!(parse(&relative).unwrap_err().contains("absolute"));

        let mut absent = payload(&parent, &out);
        absent["parent_run_dir"] = json!(root.join("never-ran"));
        assert!(parse(&absent).unwrap_err().contains("does not exist on this computer"));
        fs::remove_dir_all(root).ok();
    }
}

#[cfg(test)]
mod request_queue_regressions {
    use super::*;
    #[test]
    fn starting_job_defers_native_progress_but_released_jobs_keep_strict_receipts() {
        let python=PathBuf::from(env::var_os("GPUWM_TUI_TEST_PYTHON").expect("set test Python path"));
        let root=env::temp_dir().join(format!("arwen-companion-startup-{}-{}",std::process::id(),now_ms()));
        fs::create_dir(&root).unwrap();let root=root.canonicalize().unwrap();
        let package=root.join("gpuwm");fs::create_dir(&package).unwrap();
        // Gate package import so the real worker has not written process.json
        // when Job::start returns. No model is run and no Job handle is forged.
        fs::write(package.join("__init__.py"),"from pathlib import Path\nimport time\ngate=Path(__file__).parent.parent/'allow-worker'\ndeadline=time.monotonic()+15\nwhile not gate.exists():\n if time.monotonic()>deadline: raise RuntimeError('fixture import gate timed out')\n time.sleep(0.01)\n").unwrap();
        fs::copy(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../gpuwm/tui_worker.py"),package.join("tui_worker.py")).unwrap();
        fs::write(package.join("cli.py"),"from pathlib import Path\nimport time\ndef main(argv):\n gate=Path(__file__).parent.parent/'finish'\n deadline=time.monotonic()+15\n while not gate.exists():\n  if time.monotonic()>deadline: return 2\n  time.sleep(0.01)\n return 0\n").unwrap();
        struct OwnedTestJob(crate::job::Job);
        impl Drop for OwnedTestJob {fn drop(&mut self){let _=self.0.stop();let _=self.0.stop();let _=self.0.poll();}}
        let mut owned=OwnedTestJob(crate::job::Job::start_with_module_path(&python,"run-plan",&[root.join("plan.json").display().to_string(),"--execute".into()],&root.join("job"),&root,Some(&root)).unwrap());
        let job=&mut owned.0;
        assert!(!job.dir.join("process.json").exists());
        let starting=job_status(job,&root);assert_eq!(starting["state"],"starting");assert!(starting.get("progress_error").is_none());
        fs::write(root.join("allow-worker"),b"").unwrap();
        let deadline=Instant::now()+Duration::from_secs(10);
        while !job.dir.join("start").is_file(){assert!(job.poll().unwrap().is_none());assert!(Instant::now()<deadline,"{}",job.log_tail(20));std::thread::sleep(Duration::from_millis(10));}
        let process_path=job.dir.join("process.json");let process=fs::read(&process_path).unwrap();
        fs::remove_file(&process_path).unwrap();
        let missing=job_status(job,&root);assert_eq!(missing["state"],"running");assert!(missing["progress_error"].as_str().unwrap().contains("process.json"));
        let mut changed:Value=serde_json::from_slice(&process).unwrap();changed["cli_args"]=json!(["another-command"]);atomic_json(&process_path,&changed).unwrap();
        std::thread::sleep(Duration::from_millis(510));
        let mismatched=job_status(job,&root);assert_eq!(mismatched["state"],"running");assert!(mismatched["progress_error"].as_str().unwrap().contains("does not match its launch command"));
        fs::write(&process_path,&process).unwrap();fs::write(root.join("finish"),b"").unwrap();
        while job.poll().unwrap().is_none(){assert!(Instant::now()<deadline,"{}",job.log_tail(20));std::thread::sleep(Duration::from_millis(10));}
        assert_eq!(job.outcome,Some(0));fs::remove_file(&process_path).unwrap();
        std::thread::sleep(Duration::from_millis(510));
        let completed=job_status(job,&root);assert_eq!(completed["state"],"completed");assert!(completed["progress_error"].as_str().unwrap().contains("process.json"));
        drop(owned);assert!(root.starts_with(env::temp_dir().canonicalize().unwrap()));fs::remove_dir_all(root).unwrap();
    }

    fn directory(label: &str) -> PathBuf {
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root = env::temp_dir().join(format!("arwen-request-queue-{label}-{}-{stamp}", std::process::id()));
        fs::create_dir_all(root.join("responses")).unwrap();
        fs::create_dir_all(root.join("claimed")).unwrap();
        root
    }

    #[test]
    fn completed_history_does_not_consume_the_pending_limit() {
        let root = directory("responses");
        let mut paths = Vec::new();
        for index in 0..4096 {
            let name = format!("done-{index:05}.json");
            fs::write(root.join("responses").join(&name), b"{}").unwrap();
            paths.push(root.join("requests").join(name));
        }
        let live = root.join("requests/live.json");
        paths.push(live.clone());
        assert_eq!(pending_request_paths(&root, paths.into_iter()), vec![live]);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn claimed_history_does_not_consume_the_pending_limit() {
        let root = directory("claims");
        let mut paths = Vec::new();
        for index in 0..4096 {
            let name = format!("claimed-{index:05}.json");
            fs::write(root.join("claimed").join(&name), b"{}").unwrap();
            paths.push(root.join("requests").join(name));
        }
        let live = root.join("requests/live.json");
        paths.push(live.clone());
        assert_eq!(pending_request_paths(&root, paths.into_iter()), vec![live]);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn temporary_files_do_not_consume_the_pending_limit() {
        let root = directory("temporary");
        let mut paths: Vec<_> = (0..4096).map(|i| root.join(format!("requests/{i}.tmp"))).collect();
        let live = root.join("requests/live.json");
        paths.push(live.clone());
        assert_eq!(pending_request_paths(&root, paths.into_iter()), vec![live]);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn pending_work_is_still_bounded_and_sorted() {
        let root = directory("bounded");
        let paths = (0..4100).rev().map(|i| root.join(format!("requests/{i:05}.json")));
        let selected = pending_request_paths(&root, paths);
        assert_eq!(selected.len(), 4096);
        assert!(selected.windows(2).all(|pair| pair[0] <= pair[1]));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn already_open_workspace_keeps_its_runtime_binding_without_another_probe() {
        let root = directory("already-open");
        let python = PathBuf::from(env::var_os("GPUWM_TUI_TEST_PYTHON").expect("set test Python path"));
        let mut command = Command::new(&python);
        command.args(["-I", "-B", "-c", "import time;time.sleep(60)"])
            .stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
        #[cfg(windows)] {
            use std::os::windows::process::CommandExt;
            command.creation_flags(windows_sys::Win32::System::Threading::CREATE_NO_WINDOW);
        }
        struct OwnedController(Controller);
        impl Drop for OwnedController {
            fn drop(&mut self) {
                if let Some(child) = self.0.child.as_mut() { let _ = child.kill(); let _ = child.wait(); }
            }
        }
        let session = Session::create(&root).unwrap();
        let original = json!({"schema":"arwen.companion-handoff.v1", "python":python, "engine_version":"2.7.0"});
        atomic_json(&session.handoff, &original).unwrap();
        let handoff_path = session.handoff.clone();
        let mut controller = OwnedController(Controller { explicit_path:Some(python), session:Some(session), child:Some(command.spawn().unwrap()) });
        let began = Instant::now();
        let result = controller.0.open(&root, &root, json!({"python":root.join("unavailable-interpreter"), "cwd":root})).unwrap();
        assert_eq!(result, "Visual workspace is already open.");
        assert!(began.elapsed() < Duration::from_secs(1));
        assert_eq!(read_json(&handoff_path, 65536).unwrap(), original);
        drop(controller);
        fs::remove_dir_all(root).unwrap();
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn compact_viewer_request_keeps_explicit_target_exact_frame_and_prefetch(){
        let mut value=serde_json::json!({"schema":"arwen.companion-request.v1","session_id":"s","id":"compact-1","action":"sync_processed_frame_v2",
            "target":{"kind":"ssh","node_id":"node-2","connection_sha256":"a".repeat(64)},"job_id":"job-1","domain":2,"sequence":42,
            "profile":"viewer-2d-v1","products":["mslp_10m_winds"],"expected_run_id":"run-1","reader_leases":true,"prefetch_sequences":[43,44]});
        assert!(matches!(super::parse_request(&value,"compact-1","s").unwrap().action,super::Action::SyncProcessedFrameV2{domain:2,sequence:Some(42),reader_leases:true,..}));
        value["prefetch_sequences"]=serde_json::json!([1,2,3,4,5,6,7,8,9]);assert!(super::parse_request(&value,"compact-1","s").is_err());
        value["prefetch_sequences"]=serde_json::json!([]);value["target"]=serde_json::json!({"kind":"local"});assert!(super::parse_request(&value,"compact-1","s").is_err());
    }
    use super::*;
    #[test]
    fn runs_and_converted_field_requests_keep_explicit_targets_and_bounded_selectors(){
        let target=json!({"kind":"ssh","node_id":"saved-node","connection_sha256":"a".repeat(64)});
        let mut value=json!({"schema":"arwen.companion-request.v1","session_id":"s","id":"r","action":"browse_runs","target":target});
        assert!(matches!(parse_request(&value,"r","s").unwrap().action,Action::BrowseRuns));
        value.as_object_mut().unwrap().remove("target");assert!(parse_request(&value,"r","s").is_err());value["target"]=target.clone();
        value["action"]=json!("open_run");value["job_id"]=json!("saved-job");assert!(matches!(parse_request(&value,"r","s").unwrap().action,Action::OpenRun(_)));
        value["action"]=json!("close_run");assert!(matches!(parse_request(&value,"r","s").unwrap().action,Action::CloseRun(_)));
        value["action"]=json!("sync_processed_frame");value["domain"]=json!(3);value["sequence"]=json!(42);
        assert!(matches!(parse_request(&value,"r","s").unwrap().action,Action::SyncProcessedFrame{domain:3,sequence:Some(42),..}));
        value["sequence"]=json!(0);assert!(parse_request(&value,"r","s").is_err());value["sequence"]=json!(42);
        value["reader_leases"]=json!(true);assert!(parse_request(&value,"r","s").is_err());value.as_object_mut().unwrap().remove("reader_leases");
        value["target"]=json!({"kind":"local"});assert!(parse_request(&value,"r","s").is_err());
    }
    #[test]
    fn companion_protocol_rejects_commands_foreign_sessions_and_relative_paths() {
        let mut value = json!({"schema":"arwen.companion-request.v1","session_id":"s","id":"a","action":"focus_logs"});
        assert!(parse_request(&value,"a","s").is_ok());
        value["command"] = json!("go");
        assert!(parse_request(&value,"a","s").is_err());
        value.as_object_mut().unwrap().remove("command");
        assert!(parse_request(&value,"a","other").is_err());
        value["action"] = json!("launch_plan"); value["plan_path"] = json!("relative.json");
        assert!(parse_request(&value,"a","s").is_err());
    }
    #[test]
    fn ssh_review_launch_and_stop_require_explicit_complete_bindings(){
        let path=env::temp_dir().join("saved-map-plan.json");
        let target=json!({"kind":"ssh","node_id":"node-1","connection_sha256":"a".repeat(64)});
        let mut value=json!({"schema":"arwen.companion-request.v1","session_id":"s","id":"request-1",
            "action":"review_plan","plan_path":path,"plan_sha256":"b".repeat(64),"config_sha256":"c".repeat(64),"target":target});
        let request=parse_request(&value,"request-1","s").unwrap();
        assert!(matches!(request.action,Action::ReviewPlan(_)));
        value["plan_sha256"]=json!("changed");assert!(parse_request(&value,"request-1","s").is_err());
        value["plan_sha256"]=json!("b".repeat(64));value["action"]=json!("launch_plan");
        assert!(parse_request(&value,"request-1","s").is_err());
        value["review_id"]=json!("review-1");value["review_sha256"]=json!("d".repeat(64));
        assert!(parse_request(&value,"request-1","s").is_ok());
        let mut stop=json!({"schema":"arwen.companion-request.v1","session_id":"s","id":"stop-1","action":"stop_job","job_id":"remote-job-1","target":target});
        assert!(parse_request(&stop,"stop-1","s").is_ok());
        stop.as_object_mut().unwrap().remove("target");assert!(parse_request(&stop,"stop-1","s").is_err());
        let mut focus=json!({"schema":"arwen.companion-request.v1","session_id":"s","id":"focus-1","action":"focus_nodes"});
        assert!(matches!(parse_request(&focus,"focus-1","s").unwrap().action,Action::FocusNodes));
        focus["host"]=json!("other-node");assert!(parse_request(&focus,"focus-1","s").is_err());
    }
    #[test]
    fn setup_focus_and_artifact_requests_allow_only_the_typed_payload(){
        let mut focus=json!({"schema":"arwen.companion-request.v1","session_id":"s","id":"focus-1","action":"focus_setup"});
        assert!(matches!(parse_request(&focus,"focus-1","s").unwrap().action,Action::FocusSetup));
        focus["target"]=json!({"kind":"local"});assert!(parse_request(&focus,"focus-1","s").is_err());
        let mut request=json!({"schema":"arwen.companion-request.v1","session_id":"s","id":"frame-1","action":"sync_artifacts",
            "job_id":"job-1","target":{"kind":"ssh","node_id":"node-1","connection_sha256":"a".repeat(64)}});
        assert!(matches!(parse_request(&request,"frame-1","s").unwrap().action,Action::SyncArtifacts{domain:1,..}));
        request["domain"]=json!(2);assert!(matches!(parse_request(&request,"frame-1","s").unwrap().action,Action::SyncArtifacts{domain:2,..}));
        request["domain"]=json!(true);assert!(parse_request(&request,"frame-1","s").is_err());
        request["domain"]=json!(1);request["path"]=json!("/other-file");assert!(parse_request(&request,"frame-1","s").is_err());
    }
    #[test]
    fn artifact_timeline_and_exact_frame_requests_keep_bounded_selectors_and_lease_opt_in(){
        let mut value=json!({"schema":"arwen.companion-request.v1","session_id":"s","id":"index-1","action":"artifact_index",
            "job_id":"job-1","domain":3,"after_sequence":256,"target":{"kind":"ssh","node_id":"node-1","connection_sha256":"a".repeat(64)}});
        assert!(matches!(parse_request(&value,"index-1","s").unwrap().action,Action::ArtifactIndex{domain:3,after_sequence:256,..}));
        for bad in [json!(-1),json!(true),json!(u64::MAX)]{
            value["after_sequence"]=bad;assert!(parse_request(&value,"index-1","s").is_err());
        }
        value.as_object_mut().unwrap().remove("after_sequence");value["action"]=json!("sync_artifacts");
        value["sequence"]=json!(42);value["reader_leases"]=json!(true);
        assert!(matches!(parse_request(&value,"index-1","s").unwrap().action,Action::SyncArtifacts{sequence:Some(42),reader_leases:true,..}));
        value["sequence"]=json!(0);assert!(parse_request(&value,"index-1","s").is_err());
        value.as_object_mut().unwrap().remove("sequence");value.as_object_mut().unwrap().remove("reader_leases");
        assert!(matches!(parse_request(&value,"index-1","s").unwrap().action,Action::SyncArtifacts{sequence:None,reader_leases:false,..}));
        value["reader_leases"]=json!("true");assert!(parse_request(&value,"index-1","s").is_err());
    }
    #[test]
    fn selecting_a_target_requires_only_its_exact_typed_identity(){
        let mut value=json!({"schema":"arwen.companion-request.v1","session_id":"s","id":"target-1","action":"select_target"});
        assert!(parse_request(&value,"target-1","s").is_err());
        value["target"]=json!({"kind":"local"});
        assert!(matches!(parse_request(&value,"target-1","s").unwrap().action,Action::SelectTarget));
        value["target"]=json!({"kind":"ssh","node_id":"node-1","connection_sha256":"b".repeat(64)});
        assert!(parse_request(&value,"target-1","s").is_ok());
        value["config_path"]=json!("/unexpected.toml");assert!(parse_request(&value,"target-1","s").is_err());
    }
    #[test]
    fn open_workspace_accepts_only_an_empty_typed_payload(){
        let value=json!({"schema":"arwen.companion-request.v1","session_id":"s","id":"open-1","action":"open_workspace"});
        assert!(matches!(parse_request(&value,"open-1","s").unwrap().action,Action::OpenWorkspace));
        for (key,extra) in [("target",json!({"kind":"local"})),("target",json!({"kind":"ssh","node_id":"node-1","connection_sha256":"a".repeat(64)})),
                            ("config_path",json!("/saved.toml")),("job_id",json!("job-1")),("plan_sha256",json!("a".repeat(64)))] {
            let mut other=value.clone();other[key]=extra;
            assert!(parse_request(&other,"open-1","s").is_err(),"{key}");
        }
    }
    #[test]
    fn a_read_only_viewer_session_removes_its_directory_when_it_ends(){
        let root=env::temp_dir().join(format!("arwen-companion-viewer-end-{}-{}",std::process::id(),now_ms()));
        fs::create_dir_all(&root).unwrap();
        let mut viewer=Session::create_read_only(&root).unwrap();
        let session=viewer.directory.clone();
        viewer.publish(json!({"state":"ready"}),true).unwrap();
        let published=read_json(&session.join("status.json"),128*1024).unwrap();
        assert_eq!(published["read_only"],true,"a viewer session declares itself read-only: {published}");
        drop(viewer);
        assert!(!session.exists(),"the run viewer's session directory outlived the viewer");
        let mut controller=Session::create(&root).unwrap();
        let owned=controller.directory.clone();
        controller.publish(json!({"state":"ready"}),true).unwrap();
        assert!(read_json(&owned.join("status.json"),128*1024).unwrap()["read_only"].is_null());
        drop(controller);
        assert_eq!(read_json(&owned.join("status.json"),128*1024).unwrap()["state"],"closed",
            "the controller's own session stays readable after it closes");
        let _=fs::remove_dir_all(root);
    }
    /// A process to attribute a session to. `running` stays alive until it is
    /// killed; the other has ended before its pid is used, which is what a
    /// killed or crashed TUI leaves behind in a session status.
    fn other_process(running:bool)->Child{
        #[cfg(windows)]
        let mut command={let mut command=Command::new("cmd");command.args(["/c",if running{"ping -n 60 127.0.0.1"}else{"exit"}]);command};
        #[cfg(unix)]
        let mut command={let mut command=Command::new("sh");command.args(["-c",if running{"sleep 60"}else{"exit 0"}]);command};
        command.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).spawn().unwrap()
    }
    /// The pid of a process that has already ended.
    fn ended_process()->u32{
        let mut child=other_process(false);
        child.wait().unwrap();
        child.id()
    }
    #[test]
    fn a_running_process_is_told_from_one_that_ended(){
        assert!(process_is_running(std::process::id()),"this process is running");
        let mut alive=other_process(true);
        assert!(process_is_running(alive.id()),"a spawned child that has not exited is running");
        assert!(!process_is_running(ended_process()),"a child that already exited is not running");
        let _=alive.kill();let _=alive.wait();
    }
    #[test]
    fn finished_viewer_sessions_are_swept_and_live_controller_and_job_directories_are_kept(){
        let root=env::temp_dir().join(format!("arwen-companion-viewer-sweep-{}-{}",std::process::id(),now_ms()));
        let parent=root.join(".arwen-tui");
        fs::create_dir_all(&parent).unwrap();
        let now=now_ms();
        let publish=|id:&str,state:&str,read_only:bool,pid:u32,beat:u128|{
            let directory=parent.join(format!("companion-{id}"));
            fs::create_dir_all(directory.join("responses")).unwrap();
            atomic_json(&directory.join("status.json"),&json!({"schema":"arwen.companion-status.v1","session_id":id,"state":state,
                "read_only":read_only,"tui_pid":pid,"heartbeat_unix_ms":beat as u64})).unwrap();
            directory
        };
        let own=std::process::id();
        let mut running=other_process(true);let foreign=running.id();
        let ended=ended_process();
        // `closed` is written only by a session's own end, so it is finished
        // whoever owns it, including a viewer of a TUI that is still running.
        let closed=publish("closed-viewer","closed",true,ended,now);
        let closed_foreign=publish("closed-foreign-viewer","closed",true,foreign,now);
        // A run viewer lives inside the controller's own process, so an
        // unfinished session carrying this pid is a viewer that is still open,
        // whatever its heartbeat says. It is never swept.
        let quiet_own=publish("quiet-own-viewer","ready",true,own,now-60_000);
        let quiet_ended=publish("quiet-ended-viewer","ready",true,ended,now-60_000);
        let live=publish("live-viewer","ready",true,foreign,now);
        let quiet_foreign=publish("quiet-foreign-viewer","ready",true,foreign,now-60_000);
        let controller=publish("dead-controller","closed",false,ended,now-60_000);
        let current=publish("current-viewer","closed",true,own,now);
        let starting=parent.join("companion-starting");fs::create_dir_all(starting.join("requests")).unwrap();
        let job=parent.join("job-1");fs::create_dir_all(&job).unwrap();fs::write(job.join("job.json"),"{}").unwrap();
        assert_eq!(remove_finished_viewer_sessions(&parent,"current-viewer"),3);
        assert!(!closed.exists(),"a closed run viewer's directory is swept");
        assert!(!closed_foreign.exists(),"a closed run viewer is finished whoever owns it");
        assert!(quiet_own.is_dir(),"an open viewer of this process whose heartbeat paused was swept");
        assert!(!quiet_ended.exists(),"a viewer of a TUI that was killed is swept");
        assert!(live.is_dir(),"a heartbeating viewer is never touched");
        assert!(quiet_foreign.is_dir(),"a viewer of a TUI that is still running is left to it");
        assert!(controller.is_dir(),"a controller session is never swept");
        assert!(current.is_dir(),"the caller's own session is never swept");
        assert!(starting.is_dir(),"a session that has not published a status yet is waited for, not removed");
        assert!(job.join("job.json").is_file(),"a job directory is never a sweep target");
        let _=running.kill();let _=running.wait();
        let _=fs::remove_dir_all(root);
    }
    /// A run viewer runs inside the controller's process, and a viewer that is
    /// busy downloading frames can go longer than VIEWER_SESSION_LIVE without
    /// republishing its heartbeat. Nothing about the order in which the poll
    /// republishes heartbeats and the controller sweeps may decide whether that
    /// viewer's directory survives: the predicate itself has to keep it, or the
    /// user watching that run loses its frames mid-read.
    #[test]
    fn an_open_viewer_of_this_process_survives_a_paused_heartbeat(){
        let root=env::temp_dir().join(format!("arwen-companion-viewer-paused-{}-{}",std::process::id(),now_ms()));
        fs::create_dir_all(&root).unwrap();
        let mut controller=Session::create(&root).unwrap();
        controller.publish(json!({"state":"ready"}),true).unwrap();
        let mut viewer=Session::create_read_only(&root).unwrap();
        let session=viewer.directory.clone();
        // The heartbeat this viewer published a minute ago, before it started a
        // transfer that has held its poll ever since.
        viewer.publish(json!({"state":"ready"}),true).unwrap();
        let path=session.join("status.json");
        let mut published=read_json(&path,128*1024).unwrap();
        published["heartbeat_unix_ms"]=json!((now_ms()-60_000) as u64);
        atomic_json(&path,&published).unwrap();
        assert!(finished_viewer_session(&session,now_ms()).is_none(),"a live viewer of this process was called finished: {published}");
        let parent=root.join(".arwen-tui");
        assert_eq!(remove_finished_viewer_sessions(&parent,&controller.id),0,"the sweep removed a live viewer of this process");
        assert!(session.join("status.json").is_file(),"the live viewer's session was swept");
        // And at the controller's exit, whichever of the two is dropped first.
        drop(controller);
        assert!(session.join("status.json").is_file(),"the controller's exit sweep took a live viewer with it");
        drop(viewer);
        assert!(!session.exists(),"the viewer's own end left its directory");
        let _=fs::remove_dir_all(root);
    }
    #[test]
    fn a_viewer_session_directory_that_is_already_gone_is_never_put_back(){
        let root=env::temp_dir().join(format!("arwen-companion-viewer-gone-{}-{}",std::process::id(),now_ms()));
        fs::create_dir_all(&root).unwrap();
        let mut viewer=Session::create_read_only(&root).unwrap();
        let session=viewer.directory.clone();
        viewer.publish(json!({"state":"ready"}),true).unwrap();
        // Anything may have taken the directory first: the controller's own
        // exit sweep, a cleaning tool, the user. Its end must not rebuild it.
        fs::remove_dir_all(&session).unwrap();
        drop(viewer);
        assert!(!session.exists(),"a viewer that ended after its directory was already gone recreated it");
        let _=fs::remove_dir_all(root);
    }
    /// A reader that still holds a downloaded frame blocks the removal. What is
    /// left has to stay an identifiable finished session, and the controller
    /// has to finish the job once the reader lets go -- otherwise every remote
    /// viewer of an evening survives on disk, which is the defect.
    #[test]
    fn a_removal_a_reader_blocks_stays_identifiable_and_the_controller_finishes_it(){
        let root=env::temp_dir().join(format!("arwen-companion-viewer-blocked-{}-{}",std::process::id(),now_ms()));
        fs::create_dir_all(&root).unwrap();
        let mut viewer=Session::create_read_only(&root).unwrap();
        let session=viewer.directory.clone();
        viewer.publish(json!({"state":"ready"}),true).unwrap();
        // The state a viewer's own end publishes before it removes itself.
        viewer.publish(json!({"state":"closed"}),true).unwrap();
        let frames=session.join("remote-artifacts").join("node-2").join("job-1").join("objects");
        fs::create_dir_all(&frames).unwrap();
        let frame=frames.join("frame.wrf");fs::write(&frame,b"raw").unwrap();
        // A native reader that does not share delete access holds the file on
        // Windows; on Unix, where an open file never blocks a removal, a
        // session directory that cannot be written to does.
        #[cfg(windows)]
        let (held,blocking)={
            use std::os::windows::fs::OpenOptionsExt;
            // FILE_SHARE_READ only: no delete while this handle is open.
            (fs::OpenOptions::new().read(true).share_mode(1).open(&frame).unwrap(),true)
        };
        #[cfg(unix)]
        let (held,blocking)={
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&session,fs::Permissions::from_mode(0o555)).unwrap();
            (fs::File::open(&frame).unwrap(),unsafe{libc::geteuid()}!=0)
        };
        let closed=viewer.last_status.clone();
        let blocked=!discard_viewer_session(&session,&closed);
        if blocking{
            assert!(blocked,"the removal finished while a reader still held a downloaded frame");
            let status=read_json(&session.join("status.json"),128*1024).unwrap();
            assert_eq!(status["read_only"],true);assert_eq!(status["session_id"],viewer.id);
            assert!(finished_viewer_session(&session,now_ms()).is_some(),"a blocked removal stopped being a recognisable finished session: {status}");
        }
        drop(held);
        #[cfg(unix)]
        {use std::os::unix::fs::PermissionsExt;let _=fs::set_permissions(&session,fs::Permissions::from_mode(0o755));}
        // The next controller's startup sweep, once the reader has let go.
        let controller=Session::create(&root).unwrap();
        assert!(!session.exists(),"a viewer whose frames a reader held survived the session that opened it");
        assert_eq!(remove_finished_viewer_sessions(&root.join(".arwen-tui"),&controller.id),0,"the sweep found more than the one leftover");
        drop(viewer);
        assert!(!session.exists(),"the viewer's end put back a directory the sweep had taken");
        drop(controller);let _=fs::remove_dir_all(root);
    }
    #[test]
    fn an_emptied_session_directory_that_is_no_longer_young_is_swept(){
        let root=env::temp_dir().join(format!("arwen-companion-viewer-empty-{}-{}",std::process::id(),now_ms()));
        let parent=root.join(".arwen-tui");
        let empty=|id:String|{let directory=parent.join(format!("companion-{id}"));fs::create_dir_all(&directory).unwrap();directory};
        let young=empty(format!("{}-1",std::process::id()));
        let later=now_ms()+VIEWER_SESSION_LIVE.as_millis()+1_000;
        // A starting controller always holds its queues; only a removal that a
        // reader blocked on the status file itself leaves an empty directory.
        let starting=parent.join("companion-starting");fs::create_dir_all(starting.join("requests")).unwrap();
        // An empty directory says nothing about what it was, so its name has to
        // name an owner this process may finish. Another TUI that is still
        // running keeps its own debris, and a name that is not a session's is
        // not this sweep's to take.
        let mut running=other_process(true);
        let foreign=empty(format!("{}-2",running.id()));
        let ended=empty(format!("{}-3",ended_process()));
        let unnamed=empty("young".into());
        assert!(finished_viewer_session(&young,now_ms()).is_none(),"a young directory is waited for, not swept");
        assert!(finished_viewer_session(&starting,later).is_none(),"a session that holds its queues is not empty debris");
        assert!(finished_viewer_session(&foreign,later).is_none(),"a live TUI's empty session directory is not this process's to remove");
        assert!(finished_viewer_session(&unnamed,later).is_none(),"a directory whose name names no session was treated as debris");
        assert!(finished_viewer_session(&ended,later).is_some(),"an emptied session of a TUI that ended is never removed");
        assert!(finished_viewer_session(&young,later).is_some(),"an emptied session directory is never removed");
        let _=running.kill();let _=running.wait();
        let _=fs::remove_dir_all(root);
    }
    #[test]
    fn the_controller_retries_a_blocked_removal_on_its_own_heartbeat(){
        let root=env::temp_dir().join(format!("arwen-companion-viewer-heartbeat-{}-{}",std::process::id(),now_ms()));
        fs::create_dir_all(&root).unwrap();
        let mut controller=Session::create(&root).unwrap();
        controller.publish(json!({"state":"ready"}),true).unwrap();
        let parent=root.join(".arwen-tui");
        let leftover=parent.join("companion-blocked-leftover");
        fs::create_dir_all(&leftover).unwrap();
        atomic_json(&leftover.join("status.json"),&json!({"schema":"arwen.companion-status.v1","session_id":"blocked-leftover","state":"closed",
            "read_only":true,"tui_pid":4242,"heartbeat_unix_ms":now_ms() as u64})).unwrap();
        controller.publish(json!({"state":"ready"}),true).unwrap();
        assert!(leftover.is_dir(),"the heartbeat sweep must not read every status twice a second");
        controller.swept=Instant::now()-VIEWER_SESSION_LIVE;
        controller.publish(json!({"state":"ready"}),true).unwrap();
        assert!(!leftover.exists(),"a blocked removal waited for the next TUI start instead of the next heartbeat");
        drop(controller);let _=fs::remove_dir_all(root);
    }
    #[test]
    fn the_controller_sweeps_leftover_viewer_sessions_at_startup_and_at_exit(){
        let root=env::temp_dir().join(format!("arwen-companion-viewer-controller-{}-{}",std::process::id(),now_ms()));
        let parent=root.join(".arwen-tui");
        let leftover=|name:&str|{
            let directory=parent.join(format!("companion-{name}"));
            fs::create_dir_all(&directory).unwrap();
            atomic_json(&directory.join("status.json"),&json!({"schema":"arwen.companion-status.v1","session_id":name,"state":"closed",
                "read_only":true,"tui_pid":4242,"heartbeat_unix_ms":now_ms() as u64})).unwrap();
            directory
        };
        let at_startup=leftover("startup-leftover");
        let mut controller=Session::create(&root).unwrap();
        assert!(!at_startup.exists(),"the controller sweeps leftover viewer sessions when it starts");
        controller.publish(json!({"state":"ready"}),true).unwrap();
        let at_exit=leftover("exit-leftover");
        drop(controller);
        assert!(!at_exit.exists(),"the controller sweeps leftover viewer sessions when it exits");
        let _=fs::remove_dir_all(root);
    }
    #[test]
    fn live_controller_scan_ignores_closed_stale_and_read_only_sessions(){
        let root=env::temp_dir().join(format!("arwen-live-controller-{}-{}",std::process::id(),now_ms()));
        let parent=root.join(".arwen-tui");
        let now=now_ms();
        let publish=|id:&str,status:Value|{let directory=parent.join(format!("companion-{id}"));fs::create_dir_all(&directory).unwrap();atomic_json(&directory.join("status.json"),&status).unwrap();};
        let status=|id:&str,state:&str,beat:u128|json!({"schema":"arwen.companion-status.v1","session_id":id,"state":state,"heartbeat_unix_ms":beat as u64,"tui_pid":4242});
        publish("closed",status("closed","closed",now));
        publish("stale",status("stale","ready",now-LIVE_CONTROLLER_HEARTBEAT.as_millis()));
        let mut viewer=status("viewer","ready",now);viewer["read_only"]=json!(true);publish("viewer",viewer);
        publish("foreign",json!({"schema":"arwen.other.v1","session_id":"foreign","state":"ready","heartbeat_unix_ms":now as u64}));
        publish("renamed",status("elsewhere","ready",now));
        fs::create_dir_all(parent.join("companion-empty")).unwrap();
        assert!(live_controller(&parent,now).is_none());
        publish("older",status("older","running",now-2_000));
        publish("live",status("live","ready",now-500));
        let (id,directory,found)=live_controller(&parent,now).unwrap();
        assert_eq!(id,"live");assert_eq!(directory,parent.join("companion-live"));assert_eq!(found["tui_pid"],4242);
        assert!(live_controller(&root.join("no-such-root"),now).is_none());
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn a_newcomer_reopens_the_live_controller_workspace_through_its_request_queue(){
        let root=env::temp_dir().join(format!("arwen-reopen-live-{}-{}",std::process::id(),now_ms()));
        let mut session=Session::create(&root).unwrap();
        session.publish(json!({"state":"ready"}),true).unwrap();
        let (id,directory,_)=live_controller(&root.join(".arwen-tui"),now_ms()).unwrap();
        assert_eq!(id,session.id);
        let requests=directory.join("requests");
        let asked=std::thread::spawn(move||request_open_workspace(&directory,&id,Duration::from_secs(10)));
        let deadline=Instant::now()+Duration::from_secs(10);
        let request=loop{
            if let Some(request)=session.requests().pop(){break request;}
            assert!(Instant::now()<deadline,"the reopen request never reached the controller queue");
            std::thread::sleep(Duration::from_millis(20));
        };
        assert!(matches!(request.action,Action::OpenWorkspace),"{:?}",request.action);
        assert_eq!(request.name,"open_workspace");
        session.respond(&request.id,&request.name,Ok("Visual workspace is already open.".into()),None).unwrap();
        assert_eq!(asked.join().unwrap().unwrap(),"Visual workspace is already open.");
        assert!(!fs::read_dir(requests).unwrap().filter_map(Result::ok).any(|entry|entry.path().extension().is_some_and(|extension|extension=="tmp")));
        // A refused reopen sends the newcomer on as a controller, as does a session it cannot reach.
        let refused=std::thread::spawn({let directory=session.directory.clone();let id=session.id.clone();move||request_open_workspace(&directory,&id,Duration::from_secs(10))});
        let request=loop{if let Some(request)=session.requests().pop(){break request;}assert!(Instant::now()<deadline);std::thread::sleep(Duration::from_millis(20));};
        session.respond(&request.id,&request.name,Err("Save or Save As before opening the visual workspace.".into()),None).unwrap();
        assert!(refused.join().unwrap().unwrap_err().contains("Save or Save As"));
        drop(session);
        assert!(request_open_workspace(&root.join(".arwen-tui").join("no-such-session"),"none",Duration::from_millis(200)).is_err());
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn reset_setup_accepts_only_an_empty_typed_payload(){
        let value=json!({"schema":"arwen.companion-request.v1","session_id":"s","id":"reset-1","action":"reset_setup"});
        assert!(matches!(parse_request(&value,"reset-1","s").unwrap().action,Action::ResetSetup));
        for (key,extra) in [("target",json!({"kind":"local"})),("config_path",json!("/saved.toml")),
                            ("job_id",json!("job-1")),("plan_sha256",json!("a".repeat(64)))] {
            let mut other=value.clone();other[key]=extra;
            assert!(parse_request(&other,"reset-1","s").is_err(),"{key}");
        }
    }
    #[test]
    fn companion_protocol_claims_once_and_durably_replays_response() {
        let root = env::temp_dir().join(format!("arwen-companion-check-{}-{}", std::process::id(), now_ms()));
        assert!(!root.exists());
        let mut session = Session::create(&root).unwrap();
        let value = json!({"schema":"arwen.companion-request.v1","session_id":session.id,"id":"one","action":"focus_logs"});
        atomic_json(&session.directory.join("requests/one.json"), &value).unwrap();
        assert_eq!(session.requests().len(), 1);
        assert!(session.requests().is_empty());
        session.respond("one", "focus_logs", Ok("Shown".into()), None).unwrap();
        assert!(session.requests().is_empty());
        let response = read_json(&session.directory.join("responses/one.json"), 4096).unwrap();
        assert_eq!(response["ok"], true);
        assert_eq!(response["id"], "one");
        drop(session);
        let verified = root.canonicalize().unwrap();
        assert!(verified.starts_with(env::temp_dir().canonicalize().unwrap()));
        assert!(verified.file_name().unwrap().to_string_lossy().starts_with("arwen-companion-check-"));
        fs::remove_dir_all(verified).unwrap();
    }
    #[test]
    fn handoff_status_and_responses_carry_the_terminal_and_engine_versions() {
        let root = env::temp_dir().join(format!("arwen-companion-versions-{}-{}", std::process::id(), now_ms()));
        let mut session = Session::create(&root).unwrap();
        let missing = root.join("no-such-interpreter");
        let error = session.publish_handoff(json!({"python":missing,"cwd":root})).unwrap_err();
        assert!(error.contains("Cannot start the selected ArWen runtime"), "{error}");
        assert!(!session.handoff.is_file());
        session.publish(json!({"state":"idle"}), true).unwrap();
        assert_eq!(read_json(&session.directory.join("status.json"), 64 * 1024).unwrap()["tui_version"], env!("CARGO_PKG_VERSION"));
        session.respond("v", "focus_logs", Ok("Shown".into()), None).unwrap();
        assert_eq!(read_json(&session.directory.join("responses/v.json"), 64 * 1024).unwrap()["tui_version"], env!("CARGO_PKG_VERSION"));
        assert!(engine_version(&missing).is_err());
        if let Some(python) = env::var_os("GPUWM_TUI_TEST_PYTHON").map(PathBuf::from) {
            let version = engine_version(&python).expect("the test interpreter reports gpuwm.__version__");
            assert!(version.chars().next().is_some_and(|c| c.is_ascii_digit()), "{version}");
            let handoff = session.publish_handoff(json!({"python":python,"cwd":root})).unwrap();
            assert_eq!(handoff["engine_version"], version);
            assert_eq!(handoff["schema"], "arwen.companion-handoff.v1");
            assert_eq!(handoff["tui_version"], env!("CARGO_PKG_VERSION"));
            assert_eq!(read_json(&session.handoff, 64 * 1024).unwrap()["tui_version"], env!("CARGO_PKG_VERSION"));
        }
        drop(session);
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn a_claimed_request_that_cannot_be_parsed_still_gets_an_error_response() {
        let root = env::temp_dir().join(format!("arwen-companion-malformed-{}-{}", std::process::id(), now_ms()));
        let mut session = Session::create(&root).unwrap();
        fs::write(session.directory.join("requests/broken.json"), br#"{"schema":"arwen.companion-request.v1","action":"focus_logs","#).unwrap();
        fs::write(session.directory.join("requests/foreign.json"), serde_json::to_vec(&json!({"schema":"arwen.companion-request.v1","session_id":"someone-else","id":"foreign","action":"focus_logs"})).unwrap()).unwrap();
        assert!(session.requests().is_empty());
        for id in ["broken", "foreign"] {
            assert!(session.directory.join("claimed").join(format!("{id}.json")).is_file());
            let response = read_json(&session.directory.join("responses").join(format!("{id}.json")), 64 * 1024).unwrap();
            assert_eq!(response["schema"], "arwen.companion-response.v1");
            assert_eq!(response["session_id"], session.id);
            assert_eq!(response["id"], id);
            assert_eq!(response["ok"], false);
            assert!(response["message"].as_str().is_some_and(|text| !text.is_empty()), "{response}");
        }
        assert_eq!(read_json(&session.directory.join("responses/broken.json"), 64 * 1024).unwrap()["action"], "unknown");
        assert_eq!(read_json(&session.directory.join("responses/foreign.json"), 64 * 1024).unwrap()["action"], "focus_logs");
        // Already answered: the next poll neither re-claims nor re-answers.
        assert!(session.requests().is_empty());
        drop(session);
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn a_new_session_answers_claims_abandoned_by_dead_sessions_only() {
        let root = env::temp_dir().join(format!("arwen-companion-reaper-{}-{}", std::process::id(), now_ms()));
        let stale = now_ms() - 2 * ABANDONED_CLAIM.as_millis();
        let mut dead = Session::create(&root).unwrap();
        for id in ["lost", "answered"] {
            fs::write(dead.directory.join("claimed").join(format!("{id}.json")),
                serde_json::to_vec(&json!({"id":id,"claimed_unix_ms":stale,"request":{"action":"launch_plan"}})).unwrap()).unwrap();
        }
        dead.respond("answered", "launch_plan", Ok("Run plan accepted.".into()), None).unwrap();
        dead.publish(json!({"state":"idle"}), true).unwrap();
        let dead_directory = dead.directory.clone();
        let dead_id = dead.id.clone();
        drop(dead); // publishes state "closed"
        let mut live = Session::create(&root).unwrap();
        fs::write(live.directory.join("claimed/inflight.json"),
            serde_json::to_vec(&json!({"id":"inflight","claimed_unix_ms":stale,"request":{"action":"review_plan"}})).unwrap()).unwrap();
        live.publish(json!({"state":"busy"}), true).unwrap();
        let next = Session::create(&root).unwrap();
        let reaped = read_json(&dead_directory.join("responses/lost.json"), 64 * 1024).unwrap();
        assert_eq!(reaped["schema"], "arwen.companion-response.v1");
        assert_eq!(reaped["session_id"], dead_id);
        assert_eq!(reaped["id"], "lost");
        assert_eq!(reaped["action"], "launch_plan");
        assert_eq!(reaped["ok"], false);
        assert!(reaped["message"].as_str().unwrap().contains("previous control center session"));
        assert_eq!(read_json(&dead_directory.join("responses/answered.json"), 64 * 1024).unwrap()["ok"], true);
        assert!(!live.directory.join("responses/inflight.json").exists(), "a heartbeating session keeps its in-flight claim");
        assert!(next.directory.join("responses").read_dir().unwrap().next().is_none());
        drop(next);
        drop(live);
        fs::remove_dir_all(root).unwrap();
    }
}

#[cfg(test)]
mod saved_job_identity {
    use super::*;
    #[test]
    fn worker_cwd_spelling_differs_from_the_launcher_canonical_path_without_breaking_identity(){
        let root=env::temp_dir().join(format!("arwen-saved-job-cwd-{}-{}",std::process::id(),now_ms()));
        fs::create_dir_all(&root).unwrap();
        let canonical=root.canonicalize().unwrap();
        let recorded=canonical.to_string_lossy().into_owned();
        let reported=recorded.strip_prefix(r"\\?\").unwrap_or(&recorded).to_owned();
        #[cfg(windows)]
        assert_ne!(reported,recorded,"a canonical Windows path carries the verbatim prefix the worker never writes");
        assert!(same_directory(&recorded,&reported));
        assert!(same_directory(&reported,&recorded));
        assert!(same_directory(&recorded,&recorded));
        let other=root.join("elsewhere");fs::create_dir_all(&other).unwrap();
        assert!(!same_directory(&recorded,&other.to_string_lossy()));
        assert!(!same_directory(&recorded,&format!("{reported}-missing")));
        // The worker's unprefixed spelling passes the process identity check;
        // validation proceeds to the native receipts instead of refusing here.
        let job=root.join("job");fs::create_dir(&job).unwrap();
        let command=vec!["python".to_owned(),"-m".into(),"gpuwm.cli".into(),"run-plan".into(),root.join("plan.json").to_string_lossy().into_owned()];
        atomic_json(&job.join("job.json"),&json!({"schema":"gpuwm-tui-job-v1","command":command,"cwd":recorded,"action":"run-plan"})).unwrap();
        atomic_json(&job.join("process.json"),&json!({"schema":"gpuwm-tui-process-v1","pid":4242,"started_at":"2026-09-10T04:45:21.086137+00:00","cwd":reported,"cli_args":&command[3..]})).unwrap();
        let error=saved_job_status(&job).unwrap_err();
        assert!(!error.contains("does not match its original launch command"),"{error}");
        let mut changed:Value=read_json(&job.join("process.json"),65536).unwrap();changed["cwd"]=json!(other);atomic_json(&job.join("process.json"),&changed).unwrap();
        assert!(saved_job_status(&job).unwrap_err().contains("does not match its original launch command"));
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn jobs_that_ended_before_their_native_manifest_keep_their_terminal_state(){
        let root=env::temp_dir().join(format!("arwen-saved-job-terminal-{}-{}",std::process::id(),now_ms()));
        fs::create_dir_all(&root).unwrap();let root=root.canonicalize().unwrap();
        let command=vec!["python".to_owned(),"-m".into(),"gpuwm.cli".into(),"run-plan".into(),root.join("plan.json").to_string_lossy().into_owned()];
        let job=|name:&str|{let dir=root.join(name);fs::create_dir(&dir).unwrap();
            atomic_json(&dir.join("job.json"),&json!({"schema":"gpuwm-tui-job-v1","command":command,"cwd":root,"action":"run-plan"})).unwrap();
            atomic_json(&dir.join("process.json"),&json!({"schema":"gpuwm-tui-process-v1","pid":4242,"started_at":"2026-09-10T04:45:21.086137+00:00","cwd":root,"cli_args":&command[3..]})).unwrap();dir};
        // A Windows stop terminates the worker before it writes its receipt; the launcher's receipt ends the job.
        let stopped=job("stopped");
        atomic_json(&stopped.join("result.json"),&json!({"schema":"gpuwm-tui-launcher-result-v1","exit_code":130,"os_exit_code":130,"status":"stopped","worker_receipt_valid":false,"message":"Stopped owned process tree (OS exit 130)."})).unwrap();
        let status=saved_job_status(&stopped).unwrap();
        assert_eq!(status["state"],"stopped");assert_eq!(status["manifest_ready"],false);assert!(status["progress_error"].as_str().is_some());
        assert_eq!(status["error"],"Stopped owned process tree (OS exit 130).");assert!(status["created_at"].as_str().unwrap().ends_with('Z'));
        // A worker that failed while acquiring inputs has its own receipt but no run manifest yet.
        let failed=job("failed");
        atomic_json(&failed.join("result.json"),&json!({"schema":"gpuwm-tui-result-v1","pid":4242,"cli_args":&command[3..],"started_at":"2026-09-10T04:45:21.086137+00:00","ended_at":"2026-09-10T04:45:40+00:00","exit_code":2,"status":"failed","error":{"type":"RuntimeError","message":"source window wider than the global grid"}})).unwrap();
        let status=saved_job_status(&failed).unwrap();
        assert_eq!(status["state"],"failed");assert_eq!(status["error"],"source window wider than the global grid");assert_eq!(status["manifest_ready"],false);
        // A launcher receipt never records a success, and a running job without native receipts stays the controller's to describe.
        let forged=job("forged");
        atomic_json(&forged.join("result.json"),&json!({"schema":"gpuwm-tui-launcher-result-v1","exit_code":0,"status":"stopped"})).unwrap();
        assert!(saved_job_status(&forged).unwrap_err().contains("does not match its original process"));
        let running=job("running");
        assert!(saved_job_status(&running).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
