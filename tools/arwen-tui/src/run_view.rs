//! Read-only run browsers and viewers, independent of the forecast draft and
//! selected node. Every viewer owns its session, frozen connection and queues.
use crate::{companion::{self, Action, Session, Target}, remote};
use serde_json::{json, Value};
use std::{collections::BTreeSet, fs, path::{Path, PathBuf}, time::{Duration, Instant}};

const MAX_VIEWERS: usize = 8;
const STATUS_INTERVAL: Duration = Duration::from_secs(2);

#[derive(Clone)]
struct Client { python: PathBuf, output: PathBuf, cwd: PathBuf }

pub struct Completion {
    pub session_id: String,
    pub request: companion::Request,
    pub result: Result<String, String>,
    pub details: Value,
}

enum Opening { Browse, Status, Index { status: Value, binding: Binding } }
struct Pending {
    parent: String,
    request: companion::Request,
    target: Target,
    node: remote::Node,
    client: Client,
    transport: remote::Request,
    stage: Opening,
}

#[derive(Default)]
pub struct Manager {
    pending: Vec<Pending>,
    viewers: Vec<Viewer>,
    local_roots: BTreeSet<PathBuf>,
}

impl Manager {
    pub fn handles(action: &Action) -> bool { matches!(action, Action::BrowseRuns | Action::OpenRun(_)) }

    /// `live` is the controller's own status for the job it currently owns
    /// (`companion::job_status`). It lists and opens that job before its saved
    /// receipts exist; every other directory still goes through the saved
    /// launcher/process/native receipt validation.
    pub fn begin(&mut self, request: companion::Request, parent: &Session, nodes: &remote::Store,
        python: &Path, output: &Path, cwd: &Path, live: Option<&Value>) -> Result<Option<Completion>, String> {
        self.local_roots.insert(output.to_owned());
        self.local_roots.insert(cwd.join("arwen-runs"));
        if self.pending.len() >= MAX_VIEWERS { return Err("Run browsing requests are already in progress. Retry when they finish.".into()); }
        if matches!(request.action, Action::OpenRun(_)) && self.viewers.len() + self.pending.len() >= MAX_VIEWERS {
            return Err("Close an existing Runs viewer before opening another.".into());
        }
        let target = request.target.clone().ok_or("Run browsing requires an explicit target.")?;
        let client = Client { python: python.into(), output: output.into(), cwd: cwd.into() };
        let node = resolve_node(nodes, &target)?;
        let Some(node) = node else {
            let (message, details) = match &request.action {
                Action::BrowseRuns => {
                    let jobs = local_jobs(&self.local_roots, live);
                    ("Saved local runs loaded.".to_owned(), json!({"target":target.value(),"jobs":jobs}))
                }
                Action::OpenRun(id) => {
                    let directory = owned_local_job(id, &self.local_roots)?;
                    // The owned job is readable from the controller before its
                    // receipts settle; a bound identity still comes only from them.
                    let viewer = match (companion::saved_job_status(&directory), live_status_for(live, &directory)) {
                        (Ok(status), _) => { let identity = local_identity(&status); Viewer::local(client, directory, Some(identity), status)? }
                        (Err(_), Some(status)) => Viewer::local(client, directory, None, status)?,
                        (Err(error), None) => return Err(error),
                    };
                    let handoff = viewer.handoff.clone();
                    self.viewers.push(viewer);
                    ("Saved run opened in a read-only viewer.".to_owned(), json!({"target":target.value(),"job_id":id,"handoff":handoff}))
                }
                _ => return Err("Unsupported run browser action.".into()),
            };
            return Ok(Some(Completion { session_id: parent.id.clone(), request, result: Ok(message), details }));
        };
        let (operation, stage) = match &request.action {
            Action::BrowseRuns => (remote::Operation::List, Opening::Browse),
            Action::OpenRun(id) => (remote::Operation::Status { job: id.clone() }, Opening::Status),
            _ => return Err("Unsupported run browser action.".into()),
        };
        let transport = remote::Request::start(&node, operation, python, output, cwd)?;
        self.pending.push(Pending { parent: parent.id.clone(), request, target, node, client, transport, stage });
        Ok(None)
    }

    pub fn active(&self) -> bool { !self.viewers.is_empty() || !self.pending.is_empty() }

    pub fn poll(&mut self, live: Option<&Value>) -> Vec<Completion> {
        let mut replies = Vec::new();
        for mut pending in std::mem::take(&mut self.pending) {
            let reply = match pending.transport.poll() {
                Ok(None) => { self.pending.push(pending); continue; }
                Ok(Some(reply)) => remote::validate_readonly_reply(&pending.node, &pending.transport.operation, &reply).map(|_| reply),
                Err(error) => Err(error),
            };
            let mut details = json!({"target":pending.target.value()});
            let result = match reply {
                Err(error) => Err(error),
                Ok(reply) => match std::mem::replace(&mut pending.stage, Opening::Browse) {
                    Opening::Browse => {
                        details["jobs"] = json!(reply["jobs"].as_array().unwrap().iter().map(run_summary).collect::<Vec<_>>());
                        Ok("Saved node runs loaded.".into())
                    }
                    Opening::Status => {
                        let status = reply["job"].clone();
                        let started = (|| {
                            let binding = Binding::new(&status)?;
                            let operation = remote::Operation::ArtifactIndex { job: binding.job.clone(), domain: 1, after_sequence: 0 };
                            let transport = remote::Request::start(&pending.node, operation, &pending.client.python, &pending.client.output, &pending.client.cwd)?;
                            Ok::<_, String>((binding, transport))
                        })();
                        match started {
                            Ok((binding, transport)) => {
                                pending.stage = Opening::Index { status, binding };
                                pending.transport = transport;
                                self.pending.push(pending);
                                continue;
                            }
                            Err(error) => Err(error),
                        }
                    }
                    Opening::Index { status, mut binding } => {
                        (|| {
                            binding.authority(&reply["artifact_index"])?;
                            let viewer = Viewer::remote(pending.client.clone(), pending.node.clone(), pending.target.clone(), status, binding, &reply["artifact_index"])?;
                            details["job_id"] = json!(viewer.job_id());
                            details["handoff"] = viewer.handoff.clone();
                            self.viewers.push(viewer);
                            Ok("Saved run opened in a read-only viewer.".into())
                        })()
                    }
                },
            };
            replies.push(Completion { session_id: pending.parent, request: pending.request, result, details });
        }
        for viewer in &mut self.viewers { viewer.poll(live); }
        // No cache/receipt deletion: readers may still hold their native file
        // leases after the UI closes. Pending transfers finish before detach.
        self.viewers.retain(|viewer| !viewer.finished());
        replies
    }
}

fn resolve_node(nodes: &remote::Store, target: &Target) -> Result<Option<remote::Node>, String> {
    match target {
        Target::Local => Ok(None),
        Target::Ssh { node_id, connection_sha256 } => {
            let node = nodes.nodes.iter().find(|node| node.id == *node_id).ok_or("That saved node no longer exists. Refresh Runs targets.")?;
            if companion::digest(node.connection_key().as_bytes()) != *connection_sha256 {
                return Err("That saved node connection changed. Refresh Runs targets.".into());
            }
            node.validate(false)?;
            Ok(Some(node.clone()))
        }
    }
}

fn owned_local_job(id: &str, roots: &BTreeSet<PathBuf>) -> Result<PathBuf, String> {
    let path = Path::new(id);
    if !path.is_absolute() || path.is_symlink() || path.components().any(|part| matches!(part, std::path::Component::ParentDir)) {
        return Err("Choose a saved local run from Runs.".into());
    }
    let path = path.canonicalize().map_err(|error| error.to_string())?;
    if !roots.iter().filter_map(|root| root.join(".arwen-tui").canonicalize().ok()).any(|root| path.parent() == Some(root.as_path())) {
        return Err("That local job is outside the known run stores.".into());
    }
    Ok(path)
}

/// The controller's own job status when `directory` is the job it owns,
/// normalized to the saved-run row shape. Any other directory is `None`.
fn live_status_for(live: Option<&Value>, directory: &Path) -> Option<Value> {
    let live = live?;
    let job_dir = live["job_dir"].as_str().filter(|value| !value.is_empty())?;
    if !companion::same_directory(job_dir, &directory.to_string_lossy()) { return None; }
    let mut status = live.clone();
    if status["id"].is_null() { status["id"] = status["job_id"].clone(); }
    if status["target"].is_null() { status["target"] = json!({"kind":"local"}); }
    Some(status)
}

/// The run list is a forecast list. Sources probes, physics-profile listings,
/// dry runs and version checks share the same job root and are skipped by
/// the same predicate saved_job_status applies; they are not failures to show.
fn forecast_job_directory(directory: &Path) -> bool {
    let Ok(launcher) = companion::read_json(&directory.join("job.json"), 128 * 1024) else { return true; };
    let command: Vec<&str> = launcher["command"].as_array().map(|values| values.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
    command.get(3).is_some_and(|action| matches!(*action, "run-plan" | "go" | "sim" | "run" | "resume"))
        && !command.iter().any(|argument| matches!(*argument, "--dry-run" | "--estimate" | "--resolve" | "--physics-profiles" | "--help"))
}

/// A job directory whose receipts do not validate is listed with the exact
/// reason instead of vanishing; opening it repeats that reason. The id is
/// the canonical directory, the same spelling verified rows and the owning
/// controller use, so a row keeps its identity when it becomes verified.
fn unverified_summary(directory: &Path, error: String) -> Value {
    let directory = directory.canonicalize().unwrap_or_else(|_| directory.to_owned());
    let launcher = companion::read_json(&directory.join("job.json"), 128 * 1024).ok();
    let created = fs::metadata(directory.join("job.json")).and_then(|metadata| metadata.modified()).ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .and_then(|elapsed| i64::try_from(elapsed.as_millis()).ok())
        .and_then(|milliseconds| companion::local_progress::utc_text(milliseconds).ok());
    json!({"id":directory,"job_id":directory,"job_dir":directory,"target":{"kind":"local"},"state":"unverified","error":error,
        "action":launcher.as_ref().and_then(|value| value["action"].as_str()),"created_at":created,
        "name":Value::Null,"forecast_start_time":Value::Null,"run_seconds":Value::Null})
}

fn local_jobs(roots: &BTreeSet<PathBuf>, live: Option<&Value>) -> Vec<Value> {
    let mut directories = BTreeSet::new();
    for root in roots {
        if let Ok(entries) = fs::read_dir(root.join(".arwen-tui")) {
            for entry in entries.filter_map(Result::ok).take(4096) {
                if entry.file_type().is_ok_and(|kind| kind.is_dir()) && entry.path().join("job.json").is_file() {
                    directories.insert(entry.path());
                }
            }
        }
    }
    directories.into_iter().rev().filter(|path| forecast_job_directory(path)).take(50).map(|path| {
        // Saved receipts are the authority once they validate; the owning
        // controller's status stands in only until then.
        match companion::saved_job_status(&path) {
            Ok(status) => run_summary(&status),
            Err(error) => match live_status_for(live, &path) {
                Some(status) => run_summary(&status),
                None => unverified_summary(&path, error),
            },
        }
    }).collect()
}

fn snapshot_summary(status: &Value) -> Value {
    let mut result = json!({"name":null,"forecast_start_time":null,"run_seconds":status["progress"]["run_seconds"]});
    let Some(path) = status["source_config_path"].as_str().map(Path::new).filter(|path| path.is_absolute()) else { return result; };
    let Some(expected) = status["source_config_sha256"].as_str() else { return result; };
    let Ok(metadata) = fs::metadata(path) else { return result; };
    if !metadata.is_file() || metadata.len() > 128 * 1024 { return result; }
    let Ok(bytes) = fs::read(path) else { return result; };
    if companion::digest(&bytes) != expected { return result; }
    let Some(document) = std::str::from_utf8(&bytes).ok().and_then(|text| text.parse::<toml_edit::DocumentMut>().ok()) else { return result; };
    let Some(experiment) = document.get("experiment").and_then(toml_edit::Item::as_table_like) else { return result; };
    result["name"] = json!(experiment.get("name").and_then(toml_edit::Item::as_str));
    result["forecast_start_time"] = json!(experiment.get("start_time").and_then(toml_edit::Item::as_value).map(|value| value.to_string().trim_matches('"').to_owned()));
    result["run_seconds"] = json!(experiment.get("run_seconds").and_then(|value| value.as_float().or_else(|| value.as_integer().map(|value| value as f64))));
    result
}

fn run_summary(status: &Value) -> Value {
    let mut value = snapshot_summary(status);
    for key in ["id", "job_id", "job_dir", "state", "action", "created_at", "started_at", "ended_at", "exit_code", "outdir", "run_dir", "source_config_path", "source_config_sha256", "model_elapsed_seconds", "valid_time", "phase", "stage", "error"] {
        value[key] = status[key].clone();
    }
    if value["job_id"].is_null() { value["job_id"] = status["id"].clone(); }
    value["domain_count"] = json!(status["progress"]["domains"].as_array().map(Vec::len));
    value["rendered_png_count"] = status["render_summary"]["rendered_png_count"].clone();
    value
}

fn sha(value: &Value) -> bool { value.as_str().is_some_and(|value| value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())) }
fn immutable_job(status: &Value) -> Value {
    json!({"id":status["id"],"config":status["config"],"outdir":status["outdir"],"action":status["action"],"created_at":status["created_at"],
        "source_config_path":status["source_config_path"],"source_config_sha256":status["source_config_sha256"]})
}

struct Binding {
    job: String,
    fixed: Value,
    producer: Option<(String, String)>,
    snapshot_sha256: Option<String>,
    pid: Option<u64>,
}
impl Binding {
    fn new(status: &Value) -> Result<Self, String> {
        let job = status["id"].as_str().ok_or("Saved node run has no job ID.")?.to_owned();
        remote::valid_job(&job)?;
        if status["outdir"].as_str().is_none_or(|path| !path.starts_with('/') || path.chars().any(char::is_control)) {
            return Err("Saved node run has no original output directory.".into());
        }
        if !status["source_config_sha256"].is_null() && !sha(&status["source_config_sha256"]) {
            return Err("Saved node source configuration hash is invalid.".into());
        }
        let mut binding = Self { job, fixed: immutable_job(status), producer: None, snapshot_sha256: None, pid: None };
        binding.status(status)?;
        Ok(binding)
    }
    fn producer(&mut self, run: &str, digest: &str) -> Result<(), String> {
        let identity = (run.to_owned(), digest.to_owned());
        if self.producer.as_ref().is_some_and(|old| *old != identity) { return Err("The saved run's native producer or manifest changed. This viewer retained its original run.".into()); }
        self.producer = Some(identity);
        Ok(())
    }
    fn status(&mut self, status: &Value) -> Result<(), String> {
        if immutable_job(status) != self.fixed { return Err("The node status no longer matches this viewer's original job and saved configuration.".into()); }
        let progress = &status["progress"];
        if !progress.is_null() {
            let source = &progress["source"];
            let run = source["run_id"].as_str().filter(|value| !value.is_empty()).ok_or("Native status has no producer run ID.")?;
            if progress["schema"] != "arwen.forecast-progress.v1" || !sha(&source["manifest_sha256"]) || !sha(&source["snapshot_config_sha256"]) {
                return Err("Native status has incomplete manifest/configuration authority.".into());
            }
            self.producer(run, source["manifest_sha256"].as_str().unwrap())?;
            let snapshot = source["snapshot_config_sha256"].as_str().unwrap();
            if self.snapshot_sha256.as_deref().is_some_and(|old| old != snapshot) { return Err("The native saved configuration changed during viewing.".into()); }
            self.snapshot_sha256 = Some(snapshot.to_owned());
        }
        Ok(())
    }
    fn authority(&mut self, artifact: &Value) -> Result<(), String> {
        if artifact["job_id"] != self.job || artifact["remote_output_root"] != self.fixed["outdir"] {
            return Err("The artifact belongs to another job or original output directory.".into());
        }
        if artifact["waiting"] == true && artifact["run_manifest"].is_null() { return Ok(()); }
        let raw = artifact["run_manifest"]["utf8"].as_str().ok_or("The artifact has no exact native manifest bytes.")?;
        let digest = companion::digest(raw.as_bytes());
        if artifact["run_manifest"]["sha256"] != digest { return Err("Native manifest bytes disagree with their receipt.".into()); }
        let manifest: Value = serde_json::from_str(raw).map_err(|_| "Native manifest is not valid JSON.")?;
        let run = manifest["run_id"].as_str().filter(|value| !value.is_empty()).ok_or("Native manifest has no run ID.")?;
        let pid = manifest["pid"].as_u64().filter(|pid| *pid > 0).ok_or("Native manifest has no producer PID.")?;
        if artifact["run_id"] != run || artifact["remote_pid"].as_u64() != Some(pid) || self.pid.is_some_and(|old| old != pid) {
            return Err("The native artifact producer PID or run ID changed.".into());
        }
        self.producer(run, &digest)?;
        if let Some(snapshot) = artifact["producer_binding"]["producer_resolved"]["utf8"].as_str() {
            let resolved: Value = serde_json::from_str(snapshot).map_err(|_| "Native source receipt is not JSON.")?;
            if self.snapshot_sha256.as_deref().is_some_and(|expected| resolved["config_sha256"] != expected) {
                return Err("The native artifact source configuration differs from this saved job.".into());
            }
        }
        self.pid = Some(pid);
        Ok(())
    }
}

struct ArtifactPending { request: companion::Request, transport: remote::Request }
enum Source {
    Remote { node: remote::Node, binding: Binding },
    /// `identity` binds once the saved receipts validate; until then the
    /// viewer may only mirror the controller's own status for its owned job.
    Local { directory: PathBuf, identity: Option<Value> },
}
struct Viewer {
    session: Session,
    handoff: Value,
    target: Target,
    client: Client,
    source: Source,
    status: Value,
    status_pending: Option<remote::Request>,
    artifact_pending: Option<ArtifactPending>,
    last_status_request: Instant,
    artifacts: Option<Value>,
    error: Option<String>,
    closing: bool,
}

fn local_identity(status: &Value) -> Value {
    json!({"job_id":status["job_id"],"command":status["command"],"pid":status["pid"],"cwd":status["cwd"],"source_config_path":status["source_config_path"],
        "source_config_sha256":status["source_config_sha256"],"run_id":status["progress"]["source"]["run_id"],"manifest_sha256":status["progress"]["source"]["manifest_sha256"]})
}
fn remote_status(status: &Value, target: &Target) -> Value {
    let mut value = status.clone();
    value["job_id"] = status["id"].clone(); value["job_dir"] = Value::Null;
    value["target"] = target.value(); value["remote_output_root"] = status["outdir"].clone();
    value["action"] = if status["action"] == "start-plan" { json!("run-plan") } else { status["action"].clone() };
    for key in ["log_path", "progress_path", "events_path", "ready_dir"] { value[key] = Value::Null; }
    value["manifest_ready"] = json!(false);
    value
}

impl Viewer {
    fn remote(client: Client, node: remote::Node, target: Target, status: Value, binding: Binding, initial_index: &Value) -> Result<Self, String> {
        let mut viewer = Self::create(client, target, Source::Remote { node, binding }, status)?;
        let mut index = initial_index.clone(); index["schema"] = json!("arwen.remote-artifact-index.v1"); index["target"] = viewer.target.value();
        let _ = viewer.session.save_artifacts("initial-index", &index)?;
        viewer.publish(true)?;
        Ok(viewer)
    }
    fn local(client: Client, directory: PathBuf, identity: Option<Value>, status: Value) -> Result<Self, String> {
        let mut viewer = Self::create(client, Target::Local, Source::Local { directory, identity }, status)?;
        viewer.publish(true)?;
        Ok(viewer)
    }
    fn create(client: Client, target: Target, source: Source, status: Value) -> Result<Self, String> {
        let session = Session::create(&client.output)?;
        let mut endpoint = target.value();
        if let Source::Remote { node, .. } = &source {
            endpoint["name"] = json!(node.name); endpoint["workspace"] = json!(node.workspace);
            // This broker implements only the committed read APIs. It never
            // advertises or probes forecast/review/GPU capabilities.
            endpoint["capabilities"] = json!({"artifact_index_v1":true,"artifact_sync_v1":true,"artifact_sequence_v1":true,"sync_processed_frame_v1":true,
                "processed_frame_v2":status["viewer_capabilities"]["processed_frame_v2"]==true});
        }
        let context = json!({"read_only":true,"config_path":status["source_config_path"],"config_sha256":status["source_config_sha256"],
            "python":client.python,"cwd":client.cwd,"output_root":client.output,"geog_root":null,"prepared_root":null,
            "render_products":status["render_summary"]["requested_specs"][0].as_str().or_else(||status["products"].as_str()).unwrap_or(""),
            "current_job_dir":status["job_dir"],"job_id":status["id"].as_str().or_else(||status["job_id"].as_str()),
            "target":endpoint,"available_targets":[target.value()],"run_summary":run_summary(&status)});
        let handoff = session.publish_handoff(context)?;
        Ok(Self { session, handoff, target, client, source, status, status_pending: None, artifact_pending: None,
            last_status_request: Instant::now() - STATUS_INTERVAL, artifacts: None, error: None, closing: false })
    }
    fn job_id(&self) -> &str { self.status["id"].as_str().or_else(|| self.status["job_id"].as_str()).unwrap_or("") }
    fn publish(&mut self, force: bool) -> Result<(), String> {
        let mut value = self.handoff.clone();
        value["state"] = json!(if self.closing { "closing" } else { "ready" });
        value["draft_dirty"] = json!(false);
        value["job"] = match &self.source { Source::Remote { .. } => remote_status(&self.status, &self.target), Source::Local { .. } => self.status.clone() };
        value["target_connection_error"] = json!(self.error);
        if let Some(artifacts) = &self.artifacts { for key in ["artifact_manifest_path", "artifact_manifest_sha256"] { value["job"][key] = artifacts[key].clone(); } }
        self.session.publish(value, force)
    }
    fn begin(&mut self, request: &companion::Request) -> Result<String, String> {
        if request.target.as_ref() != Some(&self.target) { return Err("This viewer is bound to a different original run target.".into()); }
        let requested_job = match &request.action {
            Action::ArtifactIndex { job, .. } | Action::SyncArtifacts { job, .. } | Action::SyncProcessedFrame { job, .. } | Action::SyncProcessedFrameV2 { job, .. } | Action::SyncNativePlots { job, .. } | Action::CloseRun(job) => job,
            _ => return Err("Runs viewers are read-only. Draft edits, target selection and job control stay in the control center.".into()),
        };
        if requested_job != self.job_id() { return Err("This request names a different saved job.".into()); }
        if matches!(request.action, Action::CloseRun(_)) { self.closing = true; return Ok("Viewer closing. The forecast job continues independently.".into()); }
        if self.closing { return Err("This run viewer is closing.".into()); }
        if self.artifact_pending.is_some() { return Err("A frame request is already in progress for this viewer.".into()); }
        let Source::Remote { node, .. } = &self.source else { return Err("Local run frames are read from their native saved manifest.".into()); };
        let operation = match &request.action {
            Action::ArtifactIndex { job, domain, after_sequence } => remote::Operation::ArtifactIndex { job: job.clone(), domain: *domain, after_sequence: *after_sequence },
            Action::SyncArtifacts { job, domain, sequence, reader_leases } => remote::Operation::SyncArtifacts {
                job: job.clone(), domain: *domain, sequence: *sequence, reader_leases: *reader_leases,
                cache: self.session.directory.join("remote-artifacts").join(&node.id).join(job),
            },
            Action::SyncProcessedFrame { job, domain, sequence } => remote::Operation::SyncProcessedFrame {
                job:job.clone(),domain:*domain,sequence:*sequence,cache:self.session.directory.join("processed-store").join(&node.id).join(job),
            },
            Action::SyncProcessedFrameV2 { job, domain, sequence, options, reader_leases, cache_bytes } => remote::Operation::SyncProcessedFrameV2 {
                job:job.clone(),domain:*domain,sequence:*sequence,options:options.clone(),reader_leases:*reader_leases,cache_bytes:*cache_bytes,
                cache:self.client.output.join(".arwen-viewer-cache").join(&node.id).join(job),
            },
            Action::SyncNativePlots { job, domain, sequence } => remote::Operation::SyncNativePlots {
                job:job.clone(),domain:*domain,sequence:*sequence,cache:self.client.output.join(".arwen-native-plots-cache").join(&node.id).join(job),
            },
            _ => unreachable!(),
        };
        let transport = remote::Request::start(node, operation, &self.client.python, &self.client.output, &self.client.cwd)?;
        self.artifact_pending = Some(ArtifactPending { request: request.clone(), transport });
        Ok(String::new())
    }
    fn poll(&mut self, live: Option<&Value>) {
        for request in self.session.requests() {
            let result = self.begin(&request);
            if !matches!(&result, Ok(message) if message.is_empty()) {
                let _ = self.session.respond_with(&request.id, &request.name, result, json!({"target":self.target.value(),"job_id":self.job_id()}));
            }
        }
        if let Some(mut pending) = self.artifact_pending.take() {
            match pending.transport.poll() {
                Ok(None) => self.artifact_pending = Some(pending),
                result => {
                    let mut details = json!({"target":self.target.value(),"job_id":self.job_id()});
                    let result = result.and_then(|reply| {
                        let reply = reply.ok_or("Frame request ended without a reply.")?;
                        let Source::Remote { node, binding } = &mut self.source else { unreachable!() };
                        remote::validate_readonly_reply(node, &pending.transport.operation, &reply)?;
                        if matches!(pending.request.action,Action::SyncNativePlots{..}) {
                            binding.authority(&reply["native_plots"])?;
                            details["native_plots"]=reply["native_plots"].clone();
                            return Ok(if reply["native_plots"]["waiting"]==true {"Native plots are still being prepared."}else{"Native plot gallery is ready."}.into());
                        }
                        if matches!(pending.request.action,Action::SyncProcessedFrame{..}|Action::SyncProcessedFrameV2{..}){
                            binding.authority(&reply["processed_frame"])?;
                            details["processed_frame"]=reply["processed_frame"].clone();
                            let result=remote::processed_message(&details["processed_frame"]);
                            if let Ok(message)=&result{if details["processed_frame"]["processing"].is_object(){details["processed_frame"]["processing"]["message"]=json!(message);}}
                            return result;
                        }
                        let (key, schema) = if matches!(pending.request.action, Action::ArtifactIndex { .. }) {
                            ("artifact_index", "arwen.remote-artifact-index.v1")
                        } else { ("artifacts", "arwen.remote-artifacts.v1") };
                        binding.authority(&reply[key])?;
                        details["waiting"] = reply[key]["waiting"].clone();
                        if key == "artifacts" {
                            details["transferred_bytes"] = reply["transferred_bytes"].clone();
                            if let Some(recovery) = reply.get("cache_recovery") {
                                self.artifacts = None; details["cache_recovery"] = recovery.clone();
                                return Ok("Refreshing a retained frame whose cached bytes changed.".into());
                            }
                            if reply[key]["waiting"] == true { return Ok("Waiting for the selected domain's committed frame.".into()); }
                        }
                        let mut record = reply[key].clone(); record["schema"] = json!(schema); record["target"] = self.target.value();
                        let (path, digest) = self.session.save_artifacts(&pending.request.id, &record)?;
                        if key == "artifact_index" {
                            details["artifact_index_path"] = json!(path); details["artifact_index_sha256"] = json!(digest);
                        } else {
                            details["artifact_manifest_path"] = json!(path); details["artifact_manifest_sha256"] = json!(digest); self.artifacts = Some(details.clone());
                        }
                        Ok("Saved run metadata and frame identity verified.".into())
                    });
                    if let Err(error) = &result { self.error = Some(error.clone()); }
                    let _ = self.session.respond_with(&pending.request.id, &pending.request.name, result, details);
                }
            }
        }
        // Artifact polling never updates this clock or owns this request slot.
        if let Some(mut pending) = self.status_pending.take() {
            match pending.poll() {
                Ok(None) => self.status_pending = Some(pending),
                result => {
                    let result = result.and_then(|reply| {
                        let reply = reply.ok_or("Run status ended without a reply.")?;
                        let Source::Remote { node, binding } = &mut self.source else { unreachable!() };
                        remote::validate_readonly_reply(node, &pending.operation, &reply)?;
                        binding.status(&reply["job"])?;
                        self.status = reply["job"].clone(); Ok(())
                    });
                    self.error = result.err();
                }
            }
        }
        if !self.closing && self.status_pending.is_none() && self.last_status_request.elapsed() >= STATUS_INTERVAL {
            self.last_status_request = Instant::now();
            match &self.source {
                Source::Remote { node, binding } => {
                    match remote::Request::start(node, remote::Operation::Status { job: binding.job.clone() }, &self.client.python, &self.client.output, &self.client.cwd) {
                        Ok(request) => self.status_pending = Some(request), Err(error) => self.error = Some(error),
                    }
                }
                Source::Local { directory, identity } => {
                    let (directory, bound) = (directory.clone(), identity.clone());
                    let outcome = match companion::saved_job_status(&directory) {
                        Ok(status) => match &bound {
                            Some(bound) if local_identity(&status) != *bound => Err("The saved local run's original process/source/manifest identity changed.".to_owned()),
                            Some(_) => Ok((status, None)),
                            None => { let identity = local_identity(&status); Ok((status, Some(identity))) }
                        },
                        // Receipts still settling: mirror the owning controller
                        // until they validate, never once an identity is bound.
                        Err(error) => match live_status_for(live, &directory) {
                            Some(status) if bound.is_none() => Ok((status, None)),
                            _ => Err(error),
                        },
                    };
                    match outcome {
                        Ok((status, newly_bound)) => {
                            self.status = status; self.error = None;
                            if let (Some(identity), Source::Local { identity: slot, .. }) = (newly_bound, &mut self.source) { *slot = Some(identity); }
                        }
                        Err(error) => self.error = Some(error),
                    }
                }
            }
        }
        if let Err(error) = self.publish(false) { self.error = Some(error); }
    }
    fn finished(&self) -> bool { self.closing && self.artifact_pending.is_none() && self.status_pending.is_none() }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn directory(label:&str)->PathBuf{
        let path=std::env::temp_dir().join(format!("arwen-run-view-{label}-{}",remote::stamp()));
        fs::create_dir_all(&path).unwrap();path.canonicalize().unwrap()
    }
    fn write(path:&Path,value:&Value){fs::write(path,serde_json::to_vec_pretty(value).unwrap()).unwrap();}
    fn request(action:Action,target:Target)->companion::Request{
        companion::Request{id:"request-1".into(),name:"fixture".into(),action,target:Some(target),plan_sha256:None,config_sha256:None,review_id:None,review_sha256:None}
    }
    fn node(id:&str)->remote::Node{
        let mut node=remote::Node::blank();node.id=id.into();node.host=format!("{id}.invalid");node.workspace="/node/runs".into();node
    }
    fn target(node:&remote::Node)->Target{Target::Ssh{node_id:node.id.clone(),connection_sha256:companion::digest(node.connection_key().as_bytes())}}
    /// Opening a viewer publishes a handoff, which probes the selected runtime's
    /// engine version; the local-run tests therefore need a real interpreter.
    fn test_python()->PathBuf{PathBuf::from(std::env::var_os("GPUWM_TUI_TEST_PYTHON").expect("set test Python path"))}
    #[test]
    fn runs_resolve_the_requested_saved_node_without_changing_the_active_one(){
        let mut store=remote::Store::default();let first=node("node-1");let second=node("node-2");
        store.active=Some(first.id.clone());store.nodes=vec![first.clone(),second.clone()];
        let before=store.nodes.clone();
        assert_eq!(resolve_node(&store,&target(&second)).unwrap(),Some(second.clone()));
        assert_eq!(store.active.as_deref(),Some(first.id.as_str()));assert_eq!(store.nodes,before);
        let mut stale=target(&second);if let Target::Ssh{connection_sha256,..}=&mut stale{*connection_sha256="0".repeat(64);}
        assert!(resolve_node(&store,&stale).is_err());
        assert!(resolve_node(&store,&Target::Local).unwrap().is_none());
    }
    fn bound_status()->(Value,Value){
        let manifest=json!({"schema":"gpuwm.run-manifest.v1","run_id":"native-original","pid":42,"run_dir":"/run","outputs_dir":"/run"});
        let raw=serde_json::to_string(&manifest).unwrap();let digest=companion::digest(raw.as_bytes());
        let status=json!({"id":"job-original","outdir":"/run","config":"/run/case.toml","action":"start-plan","created_at":"2026-09-08T00:00:00Z","state":"completed",
            "source_config_path":"original.toml","source_config_sha256":"a".repeat(64),"progress":{"schema":"arwen.forecast-progress.v1",
            "source":{"run_id":"native-original","manifest_sha256":digest,"snapshot_config_sha256":"b".repeat(64)}}});
        let artifact=json!({"job_id":"job-original","remote_output_root":"/run","run_id":"native-original","remote_pid":42,"waiting":false,
            "run_manifest":{"utf8":raw,"sha256":digest}});
        (status,artifact)
    }
    #[test]
    fn viewer_rejects_foreign_job_config_producer_pid_and_manifest(){
        let(status,artifact)=bound_status();
        let mut binding=Binding::new(&status).unwrap();binding.authority(&artifact).unwrap();
        for key in ["id","outdir","config","source_config_path","source_config_sha256"]{
            let mut wrong=status.clone();wrong[key]=json!("different");assert!(binding.status(&wrong).is_err(),"{key}");
        }
        let mut wrong=status.clone();wrong["progress"]["source"]["snapshot_config_sha256"]=json!("c".repeat(64));assert!(binding.status(&wrong).is_err());
        let mut wrong=artifact.clone();wrong["remote_pid"]=json!(43);assert!(binding.authority(&wrong).is_err());
        let mut wrong=artifact.clone();wrong["run_manifest"]["sha256"]=json!("d".repeat(64));assert!(binding.authority(&wrong).is_err());
        let mut wrong=artifact.clone();wrong["run_id"]=json!("other-native");assert!(binding.authority(&wrong).is_err());
        binding.status(&status).unwrap();binding.authority(&artifact).unwrap();
    }
    #[test]
    fn readonly_native_validator_refuses_stop_and_launch_before_any_profile_write(){
        let node=node("node-2");
        for operation in [remote::Operation::Stop{job:"job-1".into()},remote::Operation::Probe,
            remote::Operation::Start{products:"all".into(),preview:false,binding:None}]{
            assert!(remote::validate_readonly_reply(&node,&operation,&json!({"ok":true})).unwrap_err().contains("read-only"));
        }
    }
    fn local_fixture(root:&Path)->(PathBuf,PathBuf){
        let output=root.join("saved-output");let job=output.join(".arwen-tui/job-1");let run=root.join("native-run");
        fs::create_dir_all(&job).unwrap();fs::create_dir(&run).unwrap();
        let config=root.join("original.toml");fs::write(&config,"[experiment]\nname='Saved fixture'\nstart_time=2013-05-20T00:00:00\nrun_seconds=3600\nrestart_interval_s=0\n[[domain]]\ngrid_id=1\nhistory_interval_s=3600\n").unwrap();
        let config_sha=companion::digest(&fs::read(&config).unwrap());
        let plan=root.join("plan.json");write(&plan,&json!({"schema":"gpuwm.run-plan.v1","config":{"path":config},"output_root":run}));
        let command=vec!["fixture-python".to_owned(),"-m".into(),"gpuwm.cli".into(),"run-plan".into(),plan.to_string_lossy().into_owned(),"--execute".into()];
        write(&job.join("job.json"),&json!({"schema":"gpuwm-tui-job-v1","command":command,"cwd":root,"action":"run-plan"}));
        write(&job.join("process.json"),&json!({"schema":"gpuwm-tui-process-v1","pid":42,"started_at":"2026-09-08T00:00:00Z","cwd":root,"cli_args":&command[3..]}));
        write(&job.join("result.json"),&json!({"schema":"gpuwm-tui-result-v1","pid":42,"ended_at":"2026-09-08T00:00:05Z","exit_code":0,"cli_args":&command[3..]}));
        write(&run.join("run-manifest.json"),&json!({"schema":"gpuwm.run-manifest.v1","run_id":"local-original","pid":42,"started_at_utc":"2026-09-08T00:00:01Z",
            "run_dir":run,"outputs_dir":run,"events_path":run.join("events.jsonl"),"plan_source":plan,"plan_sha256":companion::digest(&fs::read(&plan).unwrap()),"route":"experiment"}));
        let events=[json!({"schema_version":"gpuwm.run-plan.event.v1","sequence":1,"event":"resolved_plan","emitted_unix_ms":1788825601100i64,"config_source":config,"config_sha256":config_sha}),
            json!({"schema_version":"gpuwm.run-plan.event.v1","sequence":2,"event":"model_progress","emitted_unix_ms":1788825602000i64,"domain":1,"model_seconds":3600,"outer_step":60})];
        fs::write(run.join("events.jsonl"),events.iter().map(|event|serde_json::to_string(event).unwrap()+"\n").collect::<String>()).unwrap();
        (output,job)
    }
    #[test]
    fn local_runs_browse_and_open_from_receipts_while_an_ssh_node_is_selected(){
        let root=directory("local");let(output,job)=local_fixture(&root);
        companion::saved_job_status(&job).expect("the local native receipt fixture must validate before browsing");
        let parent=Session::create(&output).unwrap();let mut store=remote::Store::default();let selected=node("selected-ssh");
        store.active=Some(selected.id.clone());store.nodes.push(selected.clone());
        let mut manager=Manager::default();
        let browse=manager.begin(request(Action::BrowseRuns,Target::Local),&parent,&store,&test_python(),&output,&root,None).unwrap().unwrap();
        assert_eq!(browse.details["jobs"].as_array().unwrap().len(),1,"{}",browse.details);
        assert_eq!(browse.details["jobs"][0]["name"],"Saved fixture");assert_eq!(browse.details["jobs"][0]["run_seconds"],3600.0);
        let opened=manager.begin(request(Action::OpenRun(job.to_string_lossy().into_owned()),Target::Local),&parent,&store,&test_python(),&output,&root,None).unwrap().unwrap();
        assert_eq!(opened.details["handoff"]["read_only"],true);assert_ne!(opened.details["handoff"]["session_id"],parent.id);
        assert_eq!(store.active.as_deref(),Some(selected.id.as_str()));
        let viewer=&mut manager.viewers[0];
        for action in [Action::SelectTarget,Action::ResetSetup,Action::OpenConfig(root.join("another.toml")),Action::LaunchPlan(root.join("plan.json")),Action::StopJob(viewer.job_id().into())]{
            assert!(viewer.begin(&request(action,Target::Local)).unwrap_err().contains("read-only"));
        }
        let session=viewer.session.directory.clone();let held=fs::File::open(session.join("status.json")).unwrap();
        viewer.begin(&request(Action::CloseRun(viewer.job_id().into()),Target::Local)).unwrap();
        manager.poll(None);assert!(manager.viewers.is_empty());assert!(session.join("status.json").is_file());drop(held);
        assert_eq!(companion::read_json(&session.join("status.json"),128*1024).unwrap()["state"],"closed");
        drop(parent);let _=fs::remove_dir_all(root);
    }
    #[test]
    fn unverifiable_saved_jobs_are_listed_with_their_reason_instead_of_vanishing(){
        let root=directory("unverified");let(output,job)=local_fixture(&root);
        let mut process=companion::read_json(&job.join("process.json"),65536).unwrap();process["cwd"]=json!(root.join("elsewhere"));write(&job.join("process.json"),&process);
        // A sources probe shares the job root and is not a forecast: never listed, never "unverified".
        let probe=output.join(".arwen-tui/job-0-sources");fs::create_dir_all(&probe).unwrap();
        write(&probe.join("job.json"),&json!({"schema":"gpuwm-tui-job-v1","command":["fixture-python","-m","gpuwm.cli","sources"],"cwd":root,"action":"sources"}));
        let rows=local_jobs(&BTreeSet::from([output.clone()]),None);
        assert_eq!(rows.len(),1,"{rows:?}");
        assert_eq!(rows[0]["state"],"unverified");assert_eq!(rows[0]["job_id"],json!(job.canonicalize().unwrap()));assert_eq!(rows[0]["action"],"run-plan");
        assert!(rows[0]["error"].as_str().unwrap().contains("does not match its original launch command"),"{}",rows[0]);
        assert!(rows[0]["created_at"].as_str().unwrap().ends_with('Z'));
        let parent=Session::create(&output).unwrap();let store=remote::Store::default();let mut manager=Manager::default();
        let opened=manager.begin(request(Action::OpenRun(job.to_string_lossy().into_owned()),Target::Local),&parent,&store,&test_python(),&output,&root,None);
        assert!(opened.err().expect("an unverifiable saved job is refused with its reason").contains("does not match its original launch command"));
        // The worker's unprefixed cwd spelling is the same directory, not a mismatch.
        process["cwd"]=json!(root.to_string_lossy().strip_prefix(r"\\?\").unwrap_or(&root.to_string_lossy()));write(&job.join("process.json"),&process);
        let rows=local_jobs(&BTreeSet::from([output]),None);assert_eq!(rows[0]["state"],"completed","{}",rows[0]);
        drop(parent);let _=fs::remove_dir_all(root);
    }
    #[test]
    fn the_owned_job_is_listed_and_opened_from_the_controller_before_its_receipts_settle(){
        let root=directory("live");let(output,job)=local_fixture(&root);
        // Before the worker reports: no process receipt, no native manifest.
        let manifest=fs::read(root.join("native-run/run-manifest.json")).unwrap();
        fs::remove_file(job.join("process.json")).unwrap();fs::remove_file(job.join("result.json")).unwrap();fs::remove_file(root.join("native-run/run-manifest.json")).unwrap();
        let live=json!({"job_id":job,"job_dir":job,"action":"run-plan","state":"starting","exit_code":null,"log_path":job.join("job.log"),"manifest_ready":false});
        let rows=local_jobs(&BTreeSet::from([output.clone()]),Some(&live));
        assert_eq!(rows.len(),1);assert_eq!(rows[0]["state"],"starting");assert_eq!(rows[0]["job_id"],json!(job));assert!(rows[0]["error"].is_null());
        let rows=local_jobs(&BTreeSet::from([output.clone()]),None);assert_eq!(rows[0]["state"],"unverified","another controller's or no live status never vouches for the directory");
        let parent=Session::create(&output).unwrap();let store=remote::Store::default();let mut manager=Manager::default();
        let opened=manager.begin(request(Action::OpenRun(job.to_string_lossy().into_owned()),Target::Local),&parent,&store,&test_python(),&output,&root,Some(&live)).unwrap().unwrap();
        assert_eq!(opened.details["handoff"]["read_only"],true);
        let viewer=&mut manager.viewers[0];assert!(matches!(&viewer.source,Source::Local{identity:None,..}));
        assert_eq!(companion::read_json(&viewer.session.directory.join("status.json"),128*1024).unwrap()["job"]["state"],"starting");
        let mut running=live.clone();running["state"]=json!("running");running["progress"]=json!({"schema":"arwen.forecast-progress.v1","model_seconds":600.,"run_seconds":3600.});
        viewer.last_status_request=Instant::now()-STATUS_INTERVAL;viewer.poll(Some(&running));
        assert!(viewer.error.is_none(),"{:?}",viewer.error);assert_eq!(viewer.status["state"],"running");assert_eq!(viewer.status["progress"]["model_seconds"],600.);
        // Receipts settle: the identity binds from them and the controller mirror stops being consulted.
        let cli_args=json!(["run-plan",root.join("plan.json").to_string_lossy(),"--execute"]);
        write(&job.join("process.json"),&json!({"schema":"gpuwm-tui-process-v1","pid":42,"started_at":"2026-09-08T00:00:00Z","cwd":root,"cli_args":cli_args}));
        write(&job.join("result.json"),&json!({"schema":"gpuwm-tui-result-v1","pid":42,"ended_at":"2026-09-08T00:00:05Z","exit_code":0,"cli_args":cli_args}));
        fs::write(root.join("native-run/run-manifest.json"),manifest).unwrap();
        viewer.last_status_request=Instant::now()-STATUS_INTERVAL;viewer.poll(None);
        assert!(viewer.error.is_none(),"{:?}",viewer.error);assert_eq!(viewer.status["state"],"completed");
        assert!(matches!(&viewer.source,Source::Local{identity:Some(_),..}));
        let mut foreign=running.clone();foreign["state"]=json!("stopped");
        viewer.last_status_request=Instant::now()-STATUS_INTERVAL;viewer.poll(Some(&foreign));
        assert_eq!(viewer.status["state"],"completed","a bound viewer reads receipts, not the controller");
        drop(parent);let _=fs::remove_dir_all(root);
    }
}
