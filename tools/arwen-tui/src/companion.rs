//! File-based visual-workspace handoff; the TUI remains the CLI job owner.
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    io::{self, Read, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
#[path="local_progress.rs"]
mod local_progress;

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
    ArtifactIndex { job: String, domain: u32, after_sequence: u64 },
    OpenConfig(PathBuf),
    ResetSetup,
    FocusLogs,
    FocusNodes,
    FocusSetup,
    SelectTarget,
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
fn valid_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 80 && id.bytes().all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-'))
}
fn parse_request(value: &Value, id: &str, session: &str) -> Result<Request, String> {
    if value["schema"] != "arwen.companion-request.v1" || value["session_id"] != session || value["id"] != id {
        return Err("Request schema, session or ID does not match this control queue.".into());
    }
    let name = value["action"].as_str().ok_or("Request action is missing.")?;
    let field = match name { "review_plan" | "launch_plan" => Some("plan_path"), "stop_job" | "sync_artifacts" | "sync_processed_frame" | "artifact_index" | "open_run" | "close_run" => Some("job_id"), "open_config" => Some("config_path"), "reset_setup" | "focus_logs" | "focus_nodes" | "focus_setup" | "select_target" | "browse_runs" => None,
        _ => return Err("Unsupported companion action.".into()) };
    let object = value.as_object().ok_or("Request must be a JSON object.")?;
    let plan_action = matches!(name, "review_plan" | "launch_plan");
    if object.keys().any(|key| !["schema", "session_id", "id", "action"].contains(&key.as_str())
        && !(key == "target" && !matches!(name,"open_config"|"reset_setup"|"focus_nodes"|"focus_setup"))
        && !(matches!(name,"sync_artifacts"|"sync_processed_frame"|"artifact_index") && key=="domain")
        && !(name=="sync_artifacts" && matches!(key.as_str(),"sequence"|"reader_leases"))
        && !(name=="sync_processed_frame" && key=="sequence")
        && !(name=="artifact_index" && key=="after_sequence")
        && !(plan_action && ["plan_sha256","config_sha256"].contains(&key.as_str()))
        && !(name=="launch_plan" && ["review_id","review_sha256"].contains(&key.as_str()))
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
    let plan_sha256=hash("plan_sha256")?;
    let config_sha256=hash("config_sha256")?;
    let review_sha256=hash("review_sha256")?;
    let review_id=match value.get("review_id") {None=>None,Some(Value::String(id)) if valid_id(id)=>Some(id.clone()),_=>return Err("Invalid review ID.".into())};
    if (name=="review_plan"||ssh&&name=="launch_plan") && (plan_sha256.is_none()||config_sha256.is_none()) {
        return Err("A remote plan needs the saved plan and configuration SHA-256 bindings.".into());
    }
    if name=="review_plan"&&!ssh {return Err("Remote review requires an explicit SSH target.".into());}
    if matches!(name,"sync_artifacts"|"sync_processed_frame"|"artifact_index")&&!ssh {return Err("Remote artifacts require an explicit SSH target.".into());}
    if ssh&&name=="launch_plan"&&(review_id.is_none()||review_sha256.is_none()) {return Err("A remote launch needs its completed node review.".into());}
    let argument = field.map(|key| value[key].as_str().filter(|s| !s.is_empty() && s.len() <= 8192 && !s.chars().any(char::is_control))
        .ok_or_else(|| "Request needs a valid absolute path.".to_owned())).transpose()?;
    if argument.is_some_and(|value| !(matches!(name,"stop_job"|"sync_artifacts"|"sync_processed_frame"|"artifact_index"|"open_run"|"close_run")&&ssh) && !Path::new(value).is_absolute()) {
        return Err("Companion paths must be absolute.".into());
    }
    if matches!(name,"stop_job"|"sync_artifacts"|"sync_processed_frame"|"artifact_index"|"open_run"|"close_run")&&ssh {crate::remote::valid_job(argument.unwrap())?;}
    let domain=||match value.get("domain"){
        None=>Ok(1),Some(value)=>value.as_u64().filter(|n|(1..=999).contains(n)).map(|n|n as u32).ok_or("Artifact domain must be an integer between 1 and 999."),
    };
    let action = match name {
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
        "artifact_index"=>Action::ArtifactIndex{job:argument.unwrap().to_owned(),domain:domain()?,
            after_sequence:match value.get("after_sequence"){None=>0,Some(value)=>value.as_u64().filter(|n|*n<=i64::MAX as u64).ok_or("Artifact cursor must be a nonnegative integer.")?}},
        "open_config" => Action::OpenConfig(PathBuf::from(argument.unwrap())),
        "reset_setup" => Action::ResetSetup,
        "focus_nodes" => Action::FocusNodes,
        "focus_setup" => Action::FocusSetup,
        "select_target" => Action::SelectTarget,
        _ => Action::FocusLogs,
    };
    Ok(Request { id: id.into(), name: name.into(), action, target, plan_sha256, config_sha256, review_id, review_sha256 })
}

pub struct Session {
    pub id: String,
    pub directory: PathBuf,
    pub handoff: PathBuf,
    last_status: Value,
    published: Instant,
}
impl Session {
    #[cfg(test)]
    pub(crate) fn test_session(output:&Path)->Result<Self,String>{Self::create(output)}
    pub(crate) fn create(output: &Path) -> Result<Self, String> {
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
        let id = format!("{}-{stamp}", std::process::id());
        let parent = output.join(".arwen-tui");
        fs::create_dir_all(&parent).map_err(|e| e.to_string())?;
        let directory = parent.join(format!("companion-{id}"));
        fs::create_dir(&directory).map_err(|e| e.to_string())?;
        let directory = directory.canonicalize().map_err(|e| e.to_string())?;
        for child in ["requests", "responses", "claimed"] { fs::create_dir(directory.join(child)).map_err(|e| e.to_string())?; }
        Ok(Self { id, handoff: directory.join("handoff.json"), directory,
            last_status: Value::Null, published: Instant::now() - Duration::from_secs(1) })
    }
    pub fn publish(&mut self, mut status: Value, force: bool) -> Result<(), String> {
        if !force && self.published.elapsed() < Duration::from_millis(500) { return Ok(()); }
        status["schema"] = json!("arwen.companion-status.v1");
        status["session_id"] = json!(self.id);
        status["tui_pid"] = json!(std::process::id());
        status["heartbeat_unix_ms"] = json!(now_ms());
        atomic_json(&self.directory.join("status.json"), &status).map_err(|e| e.to_string())?;
        self.last_status = status;
        self.published = Instant::now();
        Ok(())
    }
    pub fn due(&self) -> bool { self.published.elapsed() >= Duration::from_millis(500) }
    pub(crate) fn publish_handoff(&self, mut context:Value)->Result<Value,String>{
        context["schema"]=json!("arwen.companion-handoff.v1");
        context["session_id"]=json!(self.id);context["tui_pid"]=json!(std::process::id());
        context["control_dir"]=json!(self.directory);context["status_path"]=json!(self.directory.join("status.json"));
        atomic_json(&self.handoff,&context).map_err(|error|error.to_string())?;
        Ok(context)
    }
    pub fn requests(&mut self) -> Vec<Request> {
        let Ok(entries) = fs::read_dir(self.directory.join("requests")) else { return Vec::new(); };
        let mut paths: Vec<_> = entries.filter_map(Result::ok).take(4096)
            .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_file()))
            .map(|entry| entry.path()).filter(|path| path.extension().is_some_and(|extension| extension == "json")).collect();
        paths.sort();
        let mut requests = Vec::new();
        for path in paths {
            if requests.len() >= 8 { break; }
            let Some(id) = path.file_stem().and_then(|name| name.to_str()).filter(|id| valid_id(id)) else { continue; };
            if self.directory.join("responses").join(format!("{id}.json")).exists() { continue; }
            let claim = self.directory.join("claimed").join(format!("{id}.json"));
            let Ok(mut claimed) = fs::OpenOptions::new().write(true).create_new(true).open(claim) else { continue; };
            let value = read_json(&path, 64 * 1024);
            let evidence = json!({"id": id, "claimed_unix_ms": now_ms(), "request": value.as_ref().ok()});
            if serde_json::to_writer(&mut claimed, &evidence).and_then(|_| claimed.sync_all().map_err(serde_json::Error::io)).is_err() { continue; }
            let parsed = value.and_then(|value| parse_request(&value, id, &self.id));
            match parsed {
                Ok(request) => requests.push(request),
                Err(error) => { let _ = self.respond(id, "unknown", Err(error), None); }
            }
        }
        requests
    }
    pub fn respond(&self, id: &str, action: &str, result: Result<String, String>, job: Option<&Path>) -> Result<(), String> {
        self.respond_with(id,action,result,json!({"job_id":job,"job_dir":job}))
    }
    pub fn respond_with(&self,id:&str,action:&str,result:Result<String,String>,details:Value)->Result<(),String>{
        let ok = result.is_ok();
        let message = match result { Ok(message) | Err(message) => message };
        let mut response=json!({
            "schema":"arwen.companion-response.v1", "session_id":self.id, "id":id, "action":action,
            "ok":ok, "message":message, "job_id":null, "job_dir":null,
        });
        if let Some(fields)=details.as_object(){for(key,value)in fields{
            if !["target","job_id","job_dir","remote_output_root","review_id","review_sha256","review_path",
                "artifact_manifest_path","artifact_manifest_sha256","artifact_index_path","artifact_index_sha256",
                "transferred_bytes","waiting","cache_recovery","jobs","handoff","processed_frame"].contains(&key.as_str()) {return Err("Unsupported companion response detail.".into());}
            response[key]=value.clone();
        }}
        atomic_json(&self.directory.join("responses").join(format!("{id}.json")), &response).map_err(|e|e.to_string())
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
        let executable = self.explicit_path.clone().or_else(|| env::var_os("ARWEN_COMPANION").map(PathBuf::from))
            .unwrap_or_else(|| env::current_exe().unwrap_or_default().with_file_name(if cfg!(windows) { "arwen-companion.exe" } else { "arwen-companion" }));
        let executable = if executable.is_absolute() { executable } else { cwd.join(executable) };
        if !executable.is_file() {
            return Err("Visual workspace is not installed. Set --companion PATH or ARWEN_COMPANION to its application.".into());
        }
        if self.session.is_none() { self.session = Some(Session::create(output)?); }
        let session = self.session.as_mut().unwrap();
        session.publish_handoff(context)?;
        if self.child.as_mut().is_some_and(|child| child.try_wait().ok() == Some(None)) {
            return Ok("Visual workspace is already open.".into());
        }
        self.child = Some(Command::new(executable).arg("--handoff").arg(&session.handoff)
            .current_dir(cwd).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null())
            .spawn().map_err(|e| format!("Could not open the visual workspace: {e}"))?);
        Ok("Visual workspace opened. Forecast jobs stay in this control center.".into())
    }
    pub fn requests(&mut self) -> Vec<Request> { self.session.as_mut().map(Session::requests).unwrap_or_default() }
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
    if matches!(job.action.as_str(),"run-plan"|"go"|"sim"|"run"|"resume")&&!job.command.iter().any(|arg|matches!(arg.as_str(),"--dry-run"|"--estimate"|"--resolve"|"--physics-profiles")){
        match local_progress::cached(&job.dir,&job.command){
            Ok(native)=>{if let Some(fields)=native.as_object(){for(key,value)in fields{status[key]=value.clone();}}}
            Err(error)=>{status["progress_error"]=json!(error);status["manifest_ready"]=json!(false);for key in ["events_path","progress_path","ready_dir"]{status[key]=Value::Null;}}
        }
    }
    status
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
        ||!matches!(command[3].as_str(),"run-plan"|"go"|"sim"|"run"|"resume")
        ||command.iter().any(|arg|matches!(arg.as_str(),"--dry-run"|"--estimate"|"--resolve"|"--physics-profiles"|"--help")){
        return Err("This saved command is not a forecast run.".into());
    }
    let process=read_json(&directory.join("process.json"),64*1024)?;
    if process["schema"]!="gpuwm-tui-process-v1"||process["cli_args"]!=json!(&command[3..])
        ||process["cwd"]!=launcher["cwd"]||process["pid"].as_u64().is_none_or(|pid|pid==0){
        return Err("Saved local process does not match its original launch command.".into());
    }
    let result_path=directory.join("result.json");
    let result=if result_path.is_file(){Some(read_json(&result_path,128*1024)?)}else{None};
    if result.as_ref().is_some_and(|result|result["schema"]!="gpuwm-tui-result-v1"
        ||result["pid"]!=process["pid"]||result["cli_args"]!=process["cli_args"]||result["exit_code"].as_i64().is_none()){
        return Err("Saved local completion does not match its original process.".into());
    }
    let state=match result.as_ref().and_then(|value|value["exit_code"].as_i64()){
        Some(0)=>"completed",Some(130)=>"stopped",Some(_)=>"failed",None=>"running",
    };
    let native=local_progress::cached(&directory,&command)?;
    let mut status=json!({"id":directory,"job_id":directory,"job_dir":directory,"target":{"kind":"local"},
        "action":command[3],"state":state,"exit_code":result.as_ref().map(|value|value["exit_code"].clone()),
        "started_at":process["started_at"],"ended_at":result.as_ref().map(|value|value["ended_at"].clone()),
        "pid":process["pid"],"process_path":directory.join("process.json"),"result_path":result_path,
        "log_path":directory.join("job.log"),"command":command,"cwd":launcher["cwd"]});
    if let Some(fields)=native.as_object(){for(key,value)in fields{status[key]=value.clone();}}
    Ok(status)
}

#[cfg(test)]
mod tests {
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
}
