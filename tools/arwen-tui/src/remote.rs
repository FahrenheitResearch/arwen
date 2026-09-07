//! Node preferences and short controller requests. Simulations remain on the node.
use crate::job::Job;
use serde_json::{json, Value};
use std::fs::{self, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const STORE_SCHEMA: &str = "gpuwm.tui.nodes.v1";
const REPLY_SCHEMA: &str = "gpuwm.remote.result.v1";
const MAX_STORE_BYTES: u64 = 1024 * 1024;
const MAX_REPLY_BYTES: u64 = 2 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Node {
    pub id: String,
    pub name: String,
    pub host: String,
    pub python: String,
    pub workspace: String,
    pub config: String,
    pub output: String,
    pub geography: String,
    pub port: String,
    pub identity: String,
    pub ssh_config: String,
    pub prepared: String,
    pub wps_namelist: String,
    pub last_job: Option<String>,
    pub plot_label: Option<String>,
    pub plot_products: Option<String>,
}

impl Node {
    pub fn blank() -> Self {
        Self {
            id: stamp(),
            name: "Linux node".into(),
            host: String::new(),
            python: "/usr/bin/python3".into(),
            workspace: String::new(),
            config: String::new(),
            output: String::new(),
            geography: String::new(),
            port: String::new(),
            identity: String::new(),
            ssh_config: String::new(),
            prepared: String::new(),
            wps_namelist: String::new(),
            last_job: None,
            plot_label: None,
            plot_products: None,
        }
    }

    pub fn connection_key(&self) -> String {
        json!([
            self.host,
            self.python,
            self.workspace,
            self.port,
            self.identity,
            self.ssh_config
        ])
        .to_string()
    }

    pub fn validate(&self, launch: bool) -> Result<(), String> {
        if self.name.trim().is_empty() {
            return Err("Give this node a name.".into());
        }
        if self.host.is_empty()
            || self.host.starts_with('-')
            || self.host.chars().any(char::is_whitespace)
        {
            return Err(
                "SSH host must be an SSH alias or user@host, without spaces or a leading dash."
                    .into(),
            );
        }
        for (name, value) in self.fields() {
            if value.len() > 4096 || value.chars().any(char::is_control) {
                return Err(format!(
                    "{name} is too long or contains a control character."
                ));
            }
        }
        if let Some(products) = &self.plot_products {
            if products.is_empty()
                || products.len() > 32 * 1024
                || products.chars().any(char::is_control)
            {
                return Err(
                    "Saved node plot selection is invalid. Review Plots before launching.".into(),
                );
            }
        }
        posix_absolute(&self.python, "Python on node")?;
        posix_absolute(&self.workspace, "Node workspace")?;
        if !self.port.is_empty() && self.port.parse::<u16>().ok().filter(|p| *p > 0).is_none() {
            return Err("SSH port must be between 1 and 65535.".into());
        }
        if launch {
            posix_absolute(&self.config, "Remote configuration")?;
            if !self.output.is_empty() {
                posix_absolute(&self.output, "Remote output folder")?;
            }
            if !self.geography.is_empty() {
                posix_absolute(&self.geography, "Remote geography folder")?;
            }
            if !self.prepared.is_empty() {
                posix_absolute(&self.prepared, "Prepared folder on node")?;
            }
            if !self.wps_namelist.is_empty() {
                posix_absolute(&self.wps_namelist, "WPS namelist on node")?;
            }
        }
        Ok(())
    }

    pub fn output_directory(&self) -> String {
        if self.output.is_empty() {
            self.workspace.clone()
        } else {
            self.output.clone()
        }
    }

    pub fn fields(&self) -> [(&'static str, &str); 12] {
        [
            ("Name", &self.name),
            ("SSH host", &self.host),
            ("Python on node", &self.python),
            ("Node workspace", &self.workspace),
            ("Remote configuration", &self.config),
            ("Remote output folder", &self.output),
            ("Remote geography folder", &self.geography),
            ("SSH port (optional)", &self.port),
            ("Identity file on this computer (optional)", &self.identity),
            ("SSH config on this computer (optional)", &self.ssh_config),
            ("Prepared folder on node (optional)", &self.prepared),
            ("WPS namelist on node (optional)", &self.wps_namelist),
        ]
    }

    pub fn set_field(&mut self, index: usize, value: String) {
        let before = self.connection_key();
        match index {
            0 => self.name = value,
            1 => self.host = value,
            2 => self.python = value,
            3 => self.workspace = value,
            4 => self.config = value,
            5 => self.output = value,
            6 => self.geography = value,
            7 => self.port = value,
            8 => self.identity = value,
            9 => self.ssh_config = value,
            10 => self.prepared = value,
            11 => self.wps_namelist = value,
            _ => return,
        }
        if self.connection_key() != before {
            self.last_job = None;
        }
    }

    pub fn args(&self, operation: &Operation) -> Result<Vec<String>, String> {
        self.validate(matches!(operation, Operation::Start { .. }))?;
        let mut args = vec![
            operation.action().into(),
            "--host".into(),
            self.host.clone(),
            "--python".into(),
            self.python.clone(),
            "--workspace".into(),
            self.workspace.clone(),
            "--json".into(),
        ];
        for (flag, value) in [
            ("--port", &self.port),
            ("--identity", &self.identity),
            ("--ssh-config", &self.ssh_config),
        ] {
            if !value.is_empty() {
                args.extend([flag.into(), value.clone()]);
            }
        }
        match operation {
            Operation::Probe => {}
            Operation::List => args.extend(["--limit".into(), "50".into()]),
            Operation::Start {
                products,
                preview,
                binding,
            } => {
                args.extend([
                    "--config".into(),
                    self.config.clone(),
                    "--outdir".into(),
                    binding
                        .as_ref()
                        .map(|v| v.output.clone())
                        .unwrap_or_else(|| {
                            format!(
                                "{}/arwen-{}",
                                self.output_directory().trim_end_matches('/'),
                                stamp()
                            )
                        }),
                    "--products".into(),
                    products.clone(),
                ]);
                if !self.geography.is_empty() {
                    args.extend(["--geog-root".into(), self.geography.clone()]);
                }
                if !self.prepared.is_empty() {
                    args.extend(["--prepared-root".into(), self.prepared.clone()]);
                }
                if !self.wps_namelist.is_empty() {
                    args.extend(["--wps-namelist".into(), self.wps_namelist.clone()]);
                }
                if *preview {
                    args.push("--dry-run".into());
                } else {
                    binding
                        .as_ref()
                        .ok_or("Review the remote launch before starting it.")?
                        .append(&mut args);
                }
            }
            Operation::Status { job } | Operation::Stop { job } => {
                valid_job(job)?;
                args.extend(["--job".into(), job.clone()]);
            }
            Operation::Logs { job, cursor } => {
                valid_job(job)?;
                args.extend([
                    "--job".into(),
                    job.clone(),
                    "--cursor".into(),
                    cursor.to_string(),
                    "--limit".into(),
                    "65536".into(),
                ]);
            }
            Operation::Resume {
                job,
                checkpoint,
                output,
                preview,
                binding,
            } => {
                valid_job(job)?;
                if checkpoint != "latest" {
                    posix_absolute(checkpoint, "Remote checkpoint")?;
                }
                if !output.is_empty() {
                    posix_absolute(output, "New remote output folder")?;
                }
                args.extend([
                    "--job".into(),
                    job.clone(),
                    "--from".into(),
                    binding
                        .as_ref()
                        .and_then(|v| v.checkpoint.clone())
                        .unwrap_or_else(|| checkpoint.clone()),
                ]);
                let output = binding
                    .as_ref()
                    .map(|v| v.output.clone())
                    .unwrap_or_else(|| {
                        if output.is_empty() {
                            format!(
                                "{}/arwen-resume-{}",
                                self.output_directory().trim_end_matches('/'),
                                stamp()
                            )
                        } else {
                            output.clone()
                        }
                    });
                args.extend(["--outdir".into(), output]);
                if *preview {
                    args.push("--dry-run".into());
                } else {
                    binding
                        .as_ref()
                        .ok_or("Review the remote resume before starting it.")?
                        .append(&mut args);
                }
            }
        }
        Ok(args)
    }

    fn value(&self) -> Value {
        json!({"id":self.id,"name":self.name,"host":self.host,"python":self.python,
            "workspace":self.workspace,"config":self.config,"output":self.output,
            "geography":self.geography,"port":self.port,"identity":self.identity,
            "ssh_config":self.ssh_config,"last_job":self.last_job,
            "prepared":self.prepared,"wps_namelist":self.wps_namelist,
            "plot_label":self.plot_label,"plot_products":self.plot_products})
    }

    fn from_value(value: &Value) -> Result<Self, String> {
        let text = |key: &str| {
            value
                .get(key)
                .and_then(Value::as_str)
                .map(str::to_owned)
                .ok_or_else(|| format!("Saved node has no valid {key}."))
        };
        let optional = |key: &str| match value.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(text)) => Ok(Some(text.clone())),
            _ => Err(format!(
                "Saved node has no valid {key}; its preferences were preserved."
            )),
        };
        let node = Self {
            id: text("id")?,
            name: text("name")?,
            host: text("host")?,
            python: text("python")?,
            workspace: text("workspace")?,
            config: text("config")?,
            output: text("output")?,
            geography: text("geography")?,
            port: text("port")?,
            identity: text("identity")?,
            ssh_config: text("ssh_config")?,
            prepared: optional("prepared")?.unwrap_or_default(),
            wps_namelist: optional("wps_namelist")?.unwrap_or_default(),
            plot_label: optional("plot_label")?,
            plot_products: optional("plot_products")?,
            last_job: match value.get("last_job") {
                None | Some(Value::Null) => None,
                Some(Value::String(id)) => {
                    valid_job(id)?;
                    Some(id.clone())
                }
                _ => return Err("Saved node job ID is invalid.".into()),
            },
        };
        valid_job(&node.id)?;
        node.validate(false)?;
        Ok(node)
    }
}

fn posix_absolute(value: &str, label: &str) -> Result<(), String> {
    if !value.starts_with('/') || value.contains('\\') || value.split('/').any(|part| part == "..")
    {
        Err(format!(
            "{label} must be an absolute Linux path, such as /srv/weather."
        ))
    } else {
        Ok(())
    }
}

pub fn valid_job(value: &str) -> Result<(), String> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"-_".contains(&c))
    {
        Err("The job ID is not valid. Refresh the node's job list.".into())
    } else {
        Ok(())
    }
}

pub fn stamp() -> String {
    format!(
        "{}-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
        std::process::id()
    )
}

#[derive(Clone, Debug, Default)]
pub struct Store {
    pub nodes: Vec<Node>,
    pub active: Option<String>,
    loaded_bytes: Option<Vec<u8>>,
    legacy_imports: Vec<String>,
}

impl Store {
    pub fn load(path: &Path) -> Result<Self, String> {
        let file = match fs::File::open(path) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(error) => return Err(format!("Cannot read node preferences: {error}")),
        };
        let mut bytes = Vec::new();
        file.take(MAX_STORE_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        if bytes.len() as u64 > MAX_STORE_BYTES {
            return Err("Node preferences exceed 1 MiB.".into());
        }
        let data: Value =
            serde_json::from_slice(&bytes).map_err(|e| format!("Invalid node preferences: {e}"))?;
        if data["schema"] != STORE_SCHEMA {
            return Err("Unknown node preferences format; the file was preserved.".into());
        }
        let rows = data["nodes"]
            .as_array()
            .ok_or("Node preferences have no node list.")?;
        let nodes = rows
            .iter()
            .map(Node::from_value)
            .collect::<Result<Vec<_>, _>>()?;
        let mut ids = std::collections::HashSet::new();
        if nodes.iter().any(|node| !ids.insert(node.id.clone())) {
            return Err("Saved node IDs are duplicated.".into());
        }
        let active = match &data["active"] {
            Value::Null => None,
            Value::String(id) if ids.contains(id) => Some(id.clone()),
            _ => return Err("The selected saved node does not exist.".into()),
        };
        let legacy_imports = match data.get("legacy_imports") {
            None => Vec::new(),
            Some(Value::Array(values)) => values.iter().map(|value|
                value.as_str().map(str::to_owned).ok_or("Invalid legacy profile import record."))
                .collect::<Result<Vec<_>, _>>()?,
            _ => return Err("Invalid legacy profile import record.".into()),
        };
        Ok(Self {
            nodes,
            active,
            loaded_bytes: Some(bytes),
            legacy_imports,
        })
    }

    pub fn selected(&self) -> Option<&Node> {
        self.active
            .as_ref()
            .and_then(|id| self.nodes.iter().find(|node| &node.id == id))
    }

    pub fn save(&mut self, path: &Path) -> Result<(), String> {
        for node in &self.nodes {
            node.validate(false)?;
        }
        if self.active.is_some() && self.selected().is_none() {
            return Err("The selected node does not exist.".into());
        }
        let previous = match fs::read(path) {
            Ok(bytes) => Some(bytes),
            Err(error) if error.kind() == io::ErrorKind::NotFound => None,
            Err(error) => {
                return Err(format!(
                    "Cannot verify node preferences before saving: {error}"
                ))
            }
        };
        if previous != self.loaded_bytes {
            return Err(
                "Node preferences changed outside this window. Reload Nodes before saving.".into(),
            );
        }
        let parent = path
            .parent()
            .ok_or("Node preferences need a parent folder.")?;
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        let temporary = parent.join(format!(".arwen-nodes-{}.tmp", stamp()));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let data = json!({"schema":STORE_SCHEMA,"active":self.active,"nodes":self.nodes.iter().map(Node::value).collect::<Vec<_>>(), "legacy_imports":self.legacy_imports});
        let mut bytes = serde_json::to_vec_pretty(&data).map_err(|e| e.to_string())?;
        bytes.push(b'\n');
        if bytes.len() as u64 > MAX_STORE_BYTES {
            return Err("Node preferences exceed 1 MiB.".into());
        }
        let result = (|| {
            let mut file = options.open(&temporary)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
            drop(file);
            fs::rename(&temporary, path)
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        result.map_err(|error: io::Error| format!("Could not save node preferences: {error}"))?;
        self.loaded_bytes = Some(bytes);
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Operation {
    Probe,
    List,
    Start {
        products: String,
        preview: bool,
        binding: Option<Binding>,
    },
    Status {
        job: String,
    },
    Logs {
        job: String,
        cursor: u64,
    },
    Stop {
        job: String,
    },
    Resume {
        job: String,
        checkpoint: String,
        output: String,
        preview: bool,
        binding: Option<Binding>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Binding {
    pub output: String,
    pub checkpoint: Option<String>,
    pub hashes: Vec<(String, String)>,
}

impl Binding {
    fn from_review(review: &Value, resume: bool) -> Result<Self, String> {
        let output = review["outdir"]
            .as_str()
            .ok_or("Launch review has no output directory.")?
            .to_string();
        posix_absolute(&output, "Reviewed output directory")?;
        let mut hashes = Vec::new();
        for name in [
            "config",
            "wps",
            "input",
            "prepared",
            "checkpoint",
            "checkpoint_set",
        ] {
            let key = format!("{name}_sha256");
            if matches!(name, "wps" | "prepared" | "checkpoint" | "checkpoint_set")
                && review[&key].is_null()
            {
                if name == "prepared" && review["prepared_root"].is_string() {
                    return Err("Launch review has no prepared-input hash.".into());
                }
                if name.starts_with("checkpoint") && resume {
                    return Err("Resume review has no checkpoint hash.".into());
                }
                continue;
            }
            let hash = review[&key]
                .as_str()
                .filter(|v| v.len() == 64 && v.bytes().all(|b| b.is_ascii_hexdigit()))
                .ok_or_else(|| format!("Launch review has no valid {name} input hash."))?;
            hashes.push((
                format!("--expected-{}-sha256", name.replace('_', "-")),
                hash.into(),
            ));
        }
        let checkpoint = if resume {
            let path = review["checkpoint"]
                .as_str()
                .ok_or("Resume review has no resolved checkpoint.")?;
            posix_absolute(path, "Reviewed checkpoint")?;
            Some(path.into())
        } else {
            None
        };
        Ok(Self {
            output,
            checkpoint,
            hashes,
        })
    }
    fn append(&self, args: &mut Vec<String>) {
        for (flag, hash) in &self.hashes {
            args.extend([flag.clone(), hash.clone()]);
        }
    }
}

impl Operation {
    pub fn confirmed(&self, review: &Value) -> Result<Self, String> {
        let mut operation = self.clone();
        let resume = matches!(operation, Self::Resume { .. });
        match &mut operation {
            Self::Start {
                preview, binding, ..
            }
            | Self::Resume {
                preview, binding, ..
            } => {
                *binding = Some(Binding::from_review(review, resume)?);
                *preview = false;
                Ok(operation)
            }
            _ => Err("This request has no launch review.".into()),
        }
    }
    pub fn action(&self) -> &'static str {
        match self {
            Self::Probe => "probe",
            Self::List => "list",
            Self::Start { .. } => "start",
            Self::Status { .. } => "status",
            Self::Logs { .. } => "logs",
            Self::Stop { .. } => "stop",
            Self::Resume { .. } => "resume",
        }
    }
    pub fn mutates(&self) -> bool {
        matches!(
            self,
            Self::Start { preview: false, .. }
                | Self::Stop { .. }
                | Self::Resume { preview: false, .. }
        )
    }
}

pub struct Request {
    pub node: Node,
    pub operation: Operation,
    pub job: Job,
}

impl Request {
    pub fn start(
        node: &Node,
        operation: Operation,
        python: &Path,
        logs: &Path,
        cwd: &Path,
    ) -> Result<Self, String> {
        let args = node.args(&operation)?;
        let directory = logs.join(".arwen-tui").join(format!("remote-{}", stamp()));
        let job = Job::start(python, "remote", &args, &directory, cwd)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            node: node.clone(),
            operation,
            job,
        })
    }

    pub fn poll(&mut self) -> Result<Option<Value>, String> {
        let Some(code) = self.job.poll().map_err(|e| e.to_string())? else {
            return Ok(None);
        };
        let file = fs::File::open(self.job.dir.join("job.log")).map_err(|e| e.to_string())?;
        let mut bytes = Vec::new();
        file.take(MAX_REPLY_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        if bytes.len() as u64 > MAX_REPLY_BYTES {
            return Err(
                "Node reply was too large. The remote simulation may still be running.".into(),
            );
        }
        let text = String::from_utf8(bytes).map_err(|_| "Node reply was not UTF-8.".to_string())?;
        let reply = parse_reply(&text, self.operation.action())?;
        if code != 0 || reply["ok"] != true {
            let error = reply["error"]["message"]
                .as_str()
                .or_else(|| reply["error"].as_str())
                .map(str::to_owned)
                .unwrap_or_else(|| format!("Node request failed (exit {code})."));
            return Err(error);
        }
        Ok(Some(reply))
    }
}

pub fn parse_reply(text: &str, action: &str) -> Result<Value, String> {
    let mut replies = text
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|value| value["schema"] == REPLY_SCHEMA);
    let Some(reply) = replies.next() else {
        return Err("No complete node response was received. The simulation may still be running; reconnect and refresh its jobs.".into());
    };
    if replies.next().is_some() || reply["action"] != action || !reply["ok"].is_boolean() {
        return Err("The node response did not match this request. Refresh the node's jobs before retrying a start.".into());
    }
    Ok(reply)
}

/// Bounded viewer state. A transport failure leaves the last known run state visible.
pub struct View {
    pub runtime: Option<Value>,
    pub status: Option<Value>,
    pub jobs: Vec<Value>,
    pub log: String,
    pub cursor: u64,
    pub eof: Option<bool>,
    terminal_eof_confirmed: bool,
    pub connection_error: Option<String>,
    pub last_refresh: Option<Instant>,
}

impl Default for View {
    fn default() -> Self {
        Self {
            runtime: None,
            status: None,
            jobs: Vec::new(),
            log: String::new(),
            cursor: 0,
            eof: None,
            terminal_eof_confirmed: false,
            connection_error: None,
            last_refresh: None,
        }
    }
}

pub struct Controller {
    pub store: Store,
    pub path: PathBuf,
    pub view: View,
    pub pending: Option<Request>,
    pub load_error: Option<String>,
}

pub enum Update {
    Connected,
    Jobs,
    Status,
    Logs,
    Preview { operation: Operation, review: Value },
    Started(String),
    Stopped(String),
    Failed(String),
}

impl Controller {
    /// Shared user preferences, with one-time import of older workspace files.
    pub fn load_user(cwd: &Path) -> Self {
        let base = if cfg!(windows) {
            std::env::var_os("APPDATA").map(PathBuf::from)
                .or_else(|| std::env::var_os("USERPROFILE").map(|home| PathBuf::from(home).join("AppData/Roaming")))
                .map(|base| base.join("ArWen"))
        } else {
            std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from).filter(|path| path.is_absolute())
                .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
                .map(|base| base.join("arwen"))
        };
        match base.filter(|path| path.is_absolute()) {
            Some(base) => Self::load_shared(cwd, base.join("nodes.json")),
            None => {
                let mut controller = Self::load(cwd);
                controller.load_error = Some("ArWen cannot locate your user configuration folder. Set HOME (Linux) or APPDATA (Windows); existing node files are preserved.".into());
                controller
            }
        }
    }

    fn load_shared(cwd: &Path, path: PathBuf) -> Self {
        let mut controller = Self::load_path(path);
        if controller.load_error.is_some() { return controller; }
        let legacy_path = cwd.join(".arwen-nodes.json");
        if !legacy_path.is_file() { return controller; }
        let key = legacy_path.canonicalize().unwrap_or(legacy_path.clone()).to_string_lossy().into_owned();
        let key = if cfg!(windows) { key.to_lowercase() } else { key };
        if controller.store.legacy_imports.contains(&key) { return controller; }
        let legacy = match Store::load(&legacy_path) {
            Ok(legacy) => legacy,
            Err(error) => {
                controller.load_error = Some(format!("Could not import older node profiles from {}: {error}. The original file is preserved.", legacy_path.display()));
                return controller;
            }
        };
        let previous = controller.store.clone();
        let new_store = controller.store.loaded_bytes.is_none();
        for node in legacy.nodes {
            if !controller.store.nodes.iter().any(|known| known.id == node.id) {
                controller.store.nodes.push(node);
            }
        }
        if new_store { controller.store.active = legacy.active; }
        controller.store.legacy_imports.push(key);
        if let Err(error) = controller.store.save(&controller.path) {
            controller.store = previous;
            controller.load_error = Some(format!("Could not retain imported node profiles: {error}. The older workspace file is preserved."));
        }
        controller
    }

    pub fn load(cwd: &Path) -> Self {
        Self::load_path(cwd.join(".arwen-nodes.json"))
    }

    fn load_path(path: PathBuf) -> Self {
        let (store, load_error) = match Store::load(&path) {
            Ok(store) => (store, None),
            Err(error) => (Store::default(), Some(error)),
        };
        Self {
            store,
            path,
            view: View::default(),
            pending: None,
            load_error,
        }
    }

    pub fn reload(&mut self) -> Result<(), String> {
        if self.pending.is_some() {
            return Err("Wait for the current node request before reloading profiles.".into());
        }
        let store = Store::load(&self.path)?;
        self.store = store;
        self.load_error = None;
        self.view = View::default();
        Ok(())
    }

    pub fn select(&mut self, id: Option<String>) -> Result<(), String> {
        if self.pending.is_some() {
            return Err("Wait for the current node request before changing target.".into());
        }
        if let Some(error) = &self.load_error {
            return Err(error.clone());
        }
        let old = self.store.active.clone();
        self.store.active = id;
        if let Err(error) = self.store.save(&self.path) {
            self.store.active = old;
            return Err(error);
        }
        self.view = View::default();
        Ok(())
    }

    pub fn save_node(&mut self, node: Node) -> Result<(), String> {
        if self.pending.is_some() {
            return Err("Wait for the current node request before editing its target.".into());
        }
        if let Some(error) = &self.load_error {
            return Err(error.clone());
        }
        node.validate(false)?;
        let previous = self.store.clone();
        if let Some(existing) = self.store.nodes.iter_mut().find(|row| row.id == node.id) {
            *existing = node;
        } else {
            self.store.nodes.push(node);
        }
        if let Err(error) = self.store.save(&self.path) {
            self.store = previous;
            return Err(error);
        }
        let identity = |store: &Store| {
            store
                .selected()
                .map(|node| (node.connection_key(), node.last_job.clone()))
        };
        if identity(&previous) != identity(&self.store) {
            self.view = View::default();
        }
        Ok(())
    }

    pub fn remove_node(&mut self, id: &str) -> Result<(), String> {
        if self.pending.is_some() {
            return Err("Wait for the current node request before removing its profile.".into());
        }
        if let Some(error) = &self.load_error { return Err(error.clone()); }
        if !self.store.nodes.iter().any(|node| node.id == id) {
            return Err("That node profile no longer exists. Reload Nodes.".into());
        }
        let previous = self.store.clone();
        self.store.nodes.retain(|node| node.id != id);
        if self.store.active.as_deref() == Some(id) { self.store.active = None; }
        if let Err(error) = self.store.save(&self.path) {
            self.store = previous;
            return Err(error);
        }
        self.view = View::default();
        Ok(())
    }

    pub fn remember_job(&mut self, job: &str) -> Result<(), String> {
        if self.pending.is_some() {
            return Err("Wait for the current node request before choosing another job.".into());
        }
        valid_job(job)?;
        let active = self.store.active.as_ref().ok_or("Choose a node first.")?;
        let node = self
            .store
            .nodes
            .iter_mut()
            .find(|row| &row.id == active)
            .ok_or("The selected node no longer exists.")?;
        if node.last_job.as_deref() != Some(job) {
            self.view.eof = None;
            self.view.terminal_eof_confirmed = false;
        }
        node.last_job = Some(job.into());
        if let Err(error) = self.store.save(&self.path) {
            // Preserve the observed ID in this window even if persistence fails.
            // Losing a local preference must not hide an already-created remote run.
            return Err(error);
        }
        Ok(())
    }

    pub fn begin(
        &mut self,
        operation: Operation,
        python: &Path,
        logs: &Path,
        cwd: &Path,
    ) -> Result<(), String> {
        if self.pending.is_some() {
            return Err("A node request is already in progress.".into());
        }
        let node = self.store.selected().ok_or("Choose a Linux node first.")?;
        self.pending = Some(Request::start(node, operation, python, logs, cwd)?);
        Ok(())
    }

    pub fn poll(&mut self) -> Option<Update> {
        let request = self.pending.as_mut()?;
        let result = match request.poll() {
            Ok(None) => return None,
            result => result,
        };
        let request = self.pending.take().expect("the completed request");
        self.view.last_refresh = Some(Instant::now());
        if self.store.selected().map(Node::connection_key) != Some(request.node.connection_key()) {
            return Some(Update::Failed(
                "A reply arrived for a different saved node. Refresh the selected node.".into(),
            ));
        }
        let result = result
            .and_then(|value| self.accept(&request.operation, value.expect("completed response")));
        match result {
            Ok(update) => {
                self.view.connection_error = None;
                Some(update)
            }
            Err(error) => {
                self.view.connection_error = Some(error.clone());
                Some(Update::Failed(error))
            }
        }
    }

    fn accept(&mut self, operation: &Operation, reply: Value) -> Result<Update, String> {
        if reply.get("ok") == Some(&Value::Bool(false)) {
            return Err(
                "The node refused this request; its last observed state is preserved.".into(),
            );
        }
        match operation {
            Operation::Probe => {
                if !reply["runtime"].is_object() || !reply["capabilities"].is_object() {
                    return Err("Node probe reply is incomplete.".into());
                }
                self.view.runtime = Some(reply);
                Ok(Update::Connected)
            }
            Operation::List => {
                let jobs = reply["jobs"]
                    .as_array()
                    .ok_or("Node job list reply is incomplete.")?;
                if jobs.len() > 50 {
                    return Err("Node returned more jobs than requested.".into());
                }
                for job in jobs {
                    job_identity(job)?;
                }
                if let Some(job) = self
                    .store
                    .selected()
                    .and_then(|node| node.last_job.as_ref())
                {
                    if let Some(status) = jobs.iter().find(|status| status["id"] == *job) {
                        self.view.observe_status(status.clone());
                    }
                }
                self.view.jobs = jobs.clone();
                Ok(Update::Jobs)
            }
            Operation::Start { preview: true, .. } | Operation::Resume { preview: true, .. } => {
                if reply["dry_run"] != true || !reply["review"].is_object() {
                    return Err("Node did not return a launch review. No start is assumed.".into());
                }
                Ok(Update::Preview {
                    operation: operation.clone(),
                    review: reply["review"].clone(),
                })
            }
            Operation::Start { preview: false, .. } | Operation::Resume { preview: false, .. } => {
                let id = job_identity(&reply["job"])?;
                self.view.status = Some(reply["job"].clone());
                self.view.log.clear();
                self.view.cursor = 0;
                self.view.eof = None;
                self.view.terminal_eof_confirmed = false;
                if let Err(error) = self.remember_job(&id) {
                    return Err(format!("Remote job {id} was created. Could not save its reconnect ID: {error}. Keep this ID; do not start a duplicate."));
                }
                Ok(Update::Started(id))
            }
            Operation::Status { job } | Operation::Stop { job } => {
                if job_identity(&reply["job"])? != *job {
                    return Err("Node returned a different job ID. No change is assumed.".into());
                }
                self.view.observe_status(reply["job"].clone());
                if matches!(operation, Operation::Stop { .. }) {
                    let state = reply["job"]["state"].as_str().unwrap_or("");
                    if !matches!(
                        state,
                        "stopped" | "interrupted" | "completed" | "failed" | "cancelled"
                    ) {
                        return Err(format!(
                            "Node has not confirmed termination; current state is {state}."
                        ));
                    }
                    Ok(Update::Stopped(job.clone()))
                } else {
                    Ok(Update::Status)
                }
            }
            Operation::Logs { job, cursor } => {
                if job_identity(&reply["job"])? != *job {
                    return Err("Node logs belong to a different job.".into());
                }
                if *cursor != self.view.cursor {
                    return Err(
                        "Node log reply used a stale cursor. Refresh the selected job.".into(),
                    );
                }
                let text = reply["text"]
                    .as_str()
                    .ok_or("Node log reply has no text.")?;
                let next = reply["cursor"]
                    .as_u64()
                    .filter(|next| next >= cursor)
                    .ok_or("Node log cursor moved backwards.")?;
                let eof = reply["eof"]
                    .as_bool()
                    .ok_or("Node log reply has no valid EOF flag.")?;
                if text.len() > 128 * 1024 {
                    return Err("Node log chunk exceeds the allowed size.".into());
                }
                // Cursors count raw bytes; invalid UTF-8 is decoded with replacement.
                // A replacement can consume one, two, or three original bytes.
                let advanced = next - cursor;
                let replaced = text.chars().filter(|c| *c == '\u{fffd}').count();
                let minimum = text.len().saturating_sub(replaced.saturating_mul(2)) as u64;
                if advanced > 65_536
                    || advanced < minimum
                    || advanced > text.len() as u64
                    || (!eof && advanced == 0)
                {
                    return Err("Node log cursor does not match the returned byte chunk.".into());
                }
                // The worker samples log bytes before job status. Once terminal is
                // first observed, collect one more reply so its final line is covered.
                let terminal_was_known = self.view.terminal_for(job);
                self.view.status = Some(reply["job"].clone());
                self.view.append_log(text);
                self.view.cursor = next;
                self.view.eof = Some(eof);
                self.view.terminal_eof_confirmed =
                    eof && terminal_was_known && self.view.terminal_for(job);
                Ok(Update::Logs)
            }
        }
    }
}

fn job_identity(job: &Value) -> Result<String, String> {
    let id = job["id"].as_str().ok_or("Node response has no job ID.")?;
    valid_job(id)?;
    if job["state"]
        .as_str()
        .filter(|state| !state.is_empty())
        .is_none()
    {
        return Err("Node response has no job state.".into());
    }
    Ok(id.into())
}

impl View {
    fn terminal_for(&self, job: &str) -> bool {
        self.status.as_ref().is_some_and(|status| {
            status["id"] == job
                && matches!(
                    status["state"].as_str(),
                    Some("stopped" | "interrupted" | "completed" | "failed" | "cancelled")
                )
        })
    }
    pub fn logs_complete_for(&self, job: &str) -> bool {
        self.eof == Some(true) && self.terminal_eof_confirmed && self.terminal_for(job)
    }
    pub fn needs_log_drain(&self, job: &str) -> bool {
        self.eof == Some(false) || (self.terminal_for(job) && !self.logs_complete_for(job))
    }
    fn observe_status(&mut self, status: Value) {
        if self.status.as_ref().is_none_or(|previous| {
            previous["id"] != status["id"] || previous["state"] != status["state"]
        }) {
            self.eof = None;
            self.terminal_eof_confirmed = false;
        }
        self.status = Some(status);
    }
    pub fn append_log(&mut self, text: &str) {
        self.log.push_str(text);
        if self.log.len() > 128 * 1024 {
            let mut start = self.log.len() - 128 * 1024;
            while !self.log.is_char_boundary(start) {
                start += 1;
            }
            self.log.drain(..start);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn node() -> Node {
        let mut node = Node::blank();
        node.host = "research-node".into();
        node.workspace = "/srv/weather/Weather runs".into();
        node.config = "/srv/weather/Weather runs/storm.toml".into();
        node
    }
    fn review(resume: bool) -> Value {
        let hash = "a".repeat(64);
        json!({"outdir":"/new output/unique", "config_sha256":hash, "wps_sha256":null,
            "input_sha256":hash,"checkpoint":if resume { Some("/old/run/checkpoint.npz") } else { None },
            "checkpoint_sha256":if resume { Some(hash.clone()) } else { None },
            "checkpoint_set_sha256":if resume { Some(hash) } else { None }})
    }
    #[test]
    fn node_paths_are_arguments_and_explicit_products_survive() {
        let mut node = node();
        node.python = "/opt/ArWen env/bin/python".into();
        node.identity = "C:\\ArWen\\Operator's Keys\\weather key".into();
        for products in ["none", "all", "t2m,total_qpf"] {
            let operation = Operation::Start {
                products: products.into(),
                preview: true,
                binding: None,
            }
            .confirmed(&review(false))
            .unwrap();
            let args = node.args(&operation).unwrap();
            assert_eq!(args[0], "start");
            for (flag, expected) in [
                ("--python", node.python.as_str()),
                ("--config", node.config.as_str()),
                ("--identity", node.identity.as_str()),
                ("--products", products),
            ] {
                assert_eq!(
                    args[args.iter().position(|v| v == flag).unwrap() + 1],
                    expected
                );
            }
            assert!(!args.iter().any(|v| v == "--dry-run"));
        }
    }
    #[test]
    fn node_validation_refuses_ambiguous_target_and_local_paths() {
        for host in ["", "-oProxyCommand=evil", "user host", "host\nnext"] {
            let mut n = node();
            n.host = host.into();
            assert!(n.validate(false).is_err());
        }
        for path in ["C:\\weather", "relative", "/srv/../other"] {
            let mut n = node();
            n.workspace = path.into();
            assert!(n.validate(false).is_err());
        }
        for port in ["0", "65536", "22; command"] {
            let mut n = node();
            n.port = port.into();
            assert!(n.validate(false).is_err());
        }
    }
    #[test]
    fn invalid_optional_preferences_never_silently_change_a_saved_request() {
        for field in ["prepared", "wps_namelist", "plot_products", "plot_label"] {
            let mut value = node().value();
            value[field] = json!(["invalid type"]);
            assert!(Node::from_value(&value).unwrap_err().contains(field));
        }
        let mut value = node().value();
        value.as_object_mut().unwrap().remove("prepared");
        assert_eq!(Node::from_value(&value).unwrap().prepared, "");
    }
    #[test]
    fn profile_target_edit_drops_previous_node_job_but_label_edit_does_not() {
        let mut n = node();
        n.last_job = Some("job-123".into());
        n.set_field(0, "My weather node".into());
        assert_eq!(n.last_job.as_deref(), Some("job-123"));
        n.set_field(1, "different-node".into());
        assert!(n.last_job.is_none());
    }
    #[test]
    fn reply_requires_unique_bound_complete_envelope() {
        let good = json!({"schema":REPLY_SCHEMA,"ok":true,"action":"status","state":"running"})
            .to_string();
        assert_eq!(
            parse_reply(
                &format!("gpuwm remote: installed wheel\n{good}\n"),
                "status"
            )
            .unwrap()["state"],
            "running"
        );
        for bad in [
            "partial {\"schema\":",
            "{\"ok\":true}",
            &format!("{good}\n{good}"),
            &good.replace("status", "start"),
        ] {
            assert!(parse_reply(bad, "status").is_err());
        }
    }
    #[test]
    fn preference_round_trip_preserves_targets_and_reconnect_job() {
        let path = std::env::temp_dir()
            .join(format!("arwen-nodes-test-{}", stamp()))
            .join("nodes.json");
        let mut n = node();
        n.last_job = Some("owned-job-123".into());
        let mut store = Store {
            nodes: vec![n.clone()],
            active: Some(n.id.clone()),
            loaded_bytes: None,
            legacy_imports: Vec::new(),
        };
        store.save(&path).unwrap();
        let restored = Store::load(&path).unwrap();
        assert_eq!(restored.selected(), Some(&n));
        let mut updated = restored;
        updated.nodes[0].name = "Updated".into();
        updated.save(&path).unwrap();
        assert_eq!(Store::load(&path).unwrap().nodes[0].name, "Updated");
        fs::write(&path, b"{\"schema\":\"future\"}").unwrap();
        assert!(Store::load(&path).is_err());
        assert!(updated.save(&path).unwrap_err().contains("changed outside"));
        assert_eq!(fs::read(&path).unwrap(), b"{\"schema\":\"future\"}");
        fs::remove_file(&path).unwrap();
        fs::remove_dir(path.parent().unwrap()).unwrap();
    }
    #[test]
    fn resume_keeps_original_job_and_explicit_checkpoint_destination_separate() {
        let args = node()
            .args(&Operation::Resume {
                job: "existing-job".into(),
                checkpoint: "latest".into(),
                output: "/new output".into(),
                preview: true,
                binding: None,
            })
            .unwrap();
        assert!(args.contains(&"--dry-run".into()));
        assert!(args.windows(2).any(|v| v == ["--job", "existing-job"]));
        assert!(args.windows(2).any(|v| v == ["--from", "latest"]));
        assert!(args.windows(2).any(|v| v == ["--outdir", "/new output"]));
        assert!(!Operation::Resume {
            job: "x".into(),
            checkpoint: "latest".into(),
            output: String::new(),
            preview: true,
            binding: None
        }
        .mutates());
        assert!(Operation::Stop { job: "x".into() }.mutates());
    }
    #[test]
    fn shared_profiles_follow_the_user_and_import_legacy_jobs_once() {
        let root = std::env::temp_dir().join(format!("arwen-shared-nodes-{}", stamp()));
        let first = root.join("storms");
        let second = root.join("another-folder");
        let shared = root.join("preferences/nodes.json");
        let mut old = Controller::load(&first);
        let mut n = node();
        n.last_job = Some("ongoing-forecast".into());
        old.save_node(n.clone()).unwrap();
        old.select(Some(n.id.clone())).unwrap();
        let original = fs::read(&old.path).unwrap();
        let mut imported = Controller::load_shared(&first, shared.clone());
        assert!(imported.load_error.is_none());
        assert_eq!(imported.store.selected(), Some(&n));
        let reopened = Controller::load_shared(&second, shared.clone());
        assert_eq!(reopened.store.selected(), Some(&n));
        assert_eq!(fs::read(&old.path).unwrap(), original);
        imported.remove_node(&n.id).unwrap();
        assert!(imported.store.selected().is_none());
        // The preserved old file cannot resurrect a deliberately removed profile.
        let reopened = Controller::load_shared(&first, shared);
        assert!(reopened.store.nodes.is_empty());
        assert_eq!(fs::read(&old.path).unwrap(), original);
    }
    #[test]
    fn importing_another_workspace_preserves_the_newer_shared_profile() {
        let root = std::env::temp_dir().join(format!("arwen-merge-nodes-{}", stamp()));
        let shared = root.join("preferences/nodes.json");
        let mut live = Controller::load_shared(&root.join("empty"), shared.clone());
        let mut n = node();
        n.last_job = Some("newer-job".into());
        live.save_node(n.clone()).unwrap();
        let legacy_dir = root.join("old");
        let mut old = Controller::load(&legacy_dir);
        let mut stale = n.clone();
        stale.last_job = Some("older-job".into());
        old.save_node(stale).unwrap();
        let mut other = node();
        other.id = stamp();
        old.save_node(other.clone()).unwrap();
        let imported = Controller::load_shared(&legacy_dir, shared);
        assert!(imported.load_error.is_none());
        assert_eq!(imported.store.nodes.len(), 2);
        assert_eq!(imported.store.nodes.iter().find(|row| row.id == n.id), Some(&n));
        assert_eq!(imported.store.nodes.iter().find(|row| row.id == other.id), Some(&other));
    }
    #[test]
    fn removing_a_profile_preserves_a_concurrent_external_change() {
        let root = std::env::temp_dir().join(format!("arwen-remove-node-{}", stamp()));
        let mut c = Controller::load(&root);
        let n = node();
        c.save_node(n.clone()).unwrap();
        c.select(Some(n.id.clone())).unwrap();
        let mut external = Store::load(&c.path).unwrap();
        external.nodes[0].name = "Changed elsewhere".into();
        external.save(&c.path).unwrap();
        let external_bytes = fs::read(&c.path).unwrap();
        assert!(c.remove_node(&n.id).unwrap_err().contains("changed outside"));
        assert_eq!(c.store.selected(), Some(&n));
        assert_eq!(fs::read(&c.path).unwrap(), external_bytes);
    }
    #[test]
    fn log_view_is_bounded_without_breaking_utf8() {
        let mut view = View::default();
        view.append_log(&"☁".repeat(100_000));
        assert!(view.log.len() <= 128 * 1024);
        assert!(view.log.chars().all(|c| c == '☁'));
        view.connection_error = Some("SSH connection lost".into());
        view.status = Some(json!({"state":"running"}));
        assert_eq!(view.status.as_ref().unwrap()["state"], "running");
    }
    fn controller() -> Controller {
        let directory = std::env::temp_dir().join(format!("arwen-node-controller-{}", stamp()));
        let mut controller = Controller::load(&directory);
        let n = node();
        controller.store.active = Some(n.id.clone());
        controller.store.nodes.push(n);
        controller
    }
    fn clean(controller: &Controller) {
        if controller.path.is_file() {
            fs::remove_file(&controller.path).unwrap();
        }
        if controller.path.parent().unwrap().is_dir() {
            fs::remove_dir(controller.path.parent().unwrap()).unwrap();
        }
    }
    #[test]
    fn stop_acknowledgement_cannot_turn_running_into_stopped() {
        let mut c = controller();
        let error = c
            .accept(
                &Operation::Stop {
                    job: "job-123".into(),
                },
                json!({"job":{"id":"job-123","state":"running"}}),
            )
            .err()
            .unwrap();
        assert!(error.contains("not confirmed termination"));
        assert_eq!(c.view.status.as_ref().unwrap()["state"], "running");
        assert!(matches!(
            c.accept(
                &Operation::Stop {
                    job: "job-123".into()
                },
                json!({"job":{"id":"job-123","state":"completed"}})
            ),
            Ok(Update::Stopped(_))
        ));
    }
    #[test]
    fn lost_local_preferences_cannot_hide_created_remote_job() {
        let mut c = controller();
        c.store.save(&c.path).unwrap();
        fs::write(&c.path, b"external change").unwrap();
        let error = c
            .accept(
                &Operation::Start {
                    products: "none".into(),
                    preview: false,
                    binding: None,
                },
                json!({"job":{"id":"created-123","state":"running"}}),
            )
            .err()
            .unwrap();
        assert!(error.contains("created-123 was created"));
        assert_eq!(c.view.status.as_ref().unwrap()["id"], "created-123");
        assert_eq!(
            c.store.selected().unwrap().last_job.as_deref(),
            Some("created-123")
        );
        assert_eq!(fs::read(&c.path).unwrap(), b"external change");
        clean(&c);
    }
    #[test]
    fn stale_or_mismatched_log_reply_never_changes_current_log() {
        let mut c = controller();
        c.view.log = "kept".into();
        c.view.cursor = 5;
        let request = Operation::Logs {
            job: "current".into(),
            cursor: 5,
        };
        for reply in [
            json!({"job":{"id":"other","state":"running"},"text":"wrong","cursor":20}),
            json!({"job":{"id":"current","state":"running"},"text":"old","cursor":3}),
        ] {
            assert!(c.accept(&request, reply).is_err());
            assert_eq!(c.view.log, "kept");
            assert_eq!(c.view.cursor, 5);
        }
    }
    fn observed(view: &View) -> Value {
        json!({"runtime":view.runtime,"status":view.status,"jobs":view.jobs,
            "log":view.log,"cursor":view.cursor,"eof":view.eof,
            "terminal_eof_confirmed":view.terminal_eof_confirmed,
            "connection_error":view.connection_error})
    }
    #[test]
    fn malformed_log_chunks_preserve_every_observed_field() {
        let mut c = controller();
        c.view.status = Some(json!({"id":"current","state":"running"}));
        c.view.log = "kept".into();
        c.view.cursor = 5;
        c.view.eof = Some(false);
        let original = observed(&c.view);
        let request = Operation::Logs {
            job: "current".into(),
            cursor: 5,
        };
        let good = json!({"job":{"id":"current","state":"completed"},
            "text":"next","cursor":9,"eof":true});
        let mut bad = Vec::new();
        for (key, values) in [
            ("eof", vec![Value::Null, json!("true"), json!(1)]),
            (
                "cursor",
                vec![
                    Value::Null,
                    json!(-1),
                    json!(5.5),
                    json!("9"),
                    json!(4),
                    json!(10),
                ],
            ),
            ("text", vec![Value::Null, json!(1), json!("")]),
            (
                "job",
                vec![
                    json!({"id":"other","state":"completed"}),
                    json!({"id":"current"}),
                ],
            ),
            ("ok", vec![json!(false)]),
        ] {
            for value in values {
                let mut reply = good.clone();
                reply[key] = value;
                bad.push(reply);
            }
        }
        bad.push(json!({"job":{"id":"current","state":"running"},"text":"",
            "cursor":5,"eof":false}));
        bad.push(
            json!({"job":{"id":"current","state":"running"},"text":"x".repeat(65_537),
            "cursor":65_542,"eof":false}),
        );
        for reply in bad {
            assert!(c.accept(&request, reply).is_err());
            assert_eq!(observed(&c.view), original);
        }
        assert!(c
            .accept(
                &Operation::Logs {
                    job: "current".into(),
                    cursor: 4
                },
                good
            )
            .is_err());
        assert_eq!(observed(&c.view), original);
    }
    #[test]
    fn byte_cursor_handles_split_unicode_replacements_and_final_terminal_tail() {
        let mut c = controller();
        c.view.status = Some(json!({"id":"current","state":"running"}));
        for (text, advanced, state, eof) in [
            ("a".repeat(16_383), 16_383, "running", false),
            ("🌦".into(), 4, "running", false),
            ("\u{fffd}".into(), 1, "running", false),
            (String::new(), 0, "running", true),
            (String::new(), 0, "completed", true),
            ("\nFINAL line".into(), 11, "completed", true),
        ] {
            let cursor = c.view.cursor;
            c.accept(
                &Operation::Logs {
                    job: "current".into(),
                    cursor,
                },
                json!({"job":{"id":"current","state":state},"text":text,
                    "cursor":cursor+advanced,"eof":eof}),
            )
            .unwrap();
            assert_eq!(c.view.eof, Some(eof));
            assert_eq!(
                c.view.logs_complete_for("current"),
                c.view.log.ends_with("FINAL line")
            );
        }
        assert!(c.view.logs_complete_for("current"));
        assert!(!c.view.logs_complete_for("different"));
    }
    #[test]
    fn harmless_profile_edits_preserve_the_connected_view_and_target_edits_reset_it() {
        let mut c = controller();
        c.store.nodes[0].last_job = Some("current".into());
        c.store.save(&c.path).unwrap();
        c.view.runtime = Some(json!({"version":"2.7.0"}));
        c.view.status = Some(json!({"id":"current","state":"running"}));
        c.view.log = "observed output".into();
        c.view.cursor = 15;
        c.view.eof = Some(false);
        let original = observed(&c.view);
        let mut n = c.store.selected().unwrap().clone();
        n.name = "Weather node".into();
        n.plot_label = Some("Snow".into());
        n.plot_products = Some("var:SNOWH,total_qpf".into());
        c.save_node(n.clone()).unwrap();
        assert_eq!(observed(&c.view), original);
        c.save_node(node()).unwrap(); // Adding an inactive profile keeps this view too.
        assert_eq!(observed(&c.view), original);
        fs::write(&c.path, "external change").unwrap();
        n.plot_products = Some("none".into());
        assert!(c.save_node(n.clone()).is_err());
        assert_eq!(observed(&c.view), original);
        fs::write(&c.path, c.store.loaded_bytes.as_ref().unwrap()).unwrap();
        n.set_field(1, "different-node".into());
        c.save_node(n).unwrap();
        assert_eq!(observed(&c.view), observed(&View::default()));
        clean(&c);
    }
    #[test]
    fn list_updates_only_selected_job_and_forces_a_new_final_log_read() {
        let mut c = controller();
        c.store.nodes[0].last_job = Some("current".into());
        c.view.status = Some(json!({"id":"current","state":"running"}));
        c.view.eof = Some(true);
        c.accept(
            &Operation::List,
            json!({"jobs":[
            {"id":"other","state":"failed"},{"id":"current","state":"completed"}]}),
        )
        .unwrap();
        assert_eq!(c.view.status.as_ref().unwrap()["state"], "completed");
        assert_eq!(c.view.eof, None);
        assert!(c.view.needs_log_drain("current"));
        c.accept(
            &Operation::List,
            json!({"jobs":[{"id":"other","state":"running"}]}),
        )
        .unwrap();
        assert_eq!(c.view.status.as_ref().unwrap()["id"], "current");
        assert_eq!(c.view.status.as_ref().unwrap()["state"], "completed");
        let original = observed(&c.view);
        assert!(c
            .accept(
                &Operation::List,
                json!({"jobs":[
            {"id":"current","state":"failed"},{"id":"malformed"}]})
            )
            .is_err());
        assert_eq!(observed(&c.view), original);
    }
    #[test]
    fn polling_drains_bounded_chunks_then_stops_after_confirmed_terminal_eof() {
        use crate::node_ui::{Panel, Screen};
        use std::time::Duration;
        let mut c = controller();
        c.store.nodes[0].last_job = Some("current".into());
        let mut panel = Panel::default();
        panel.screen = Screen::Job;
        let accept = |c: &mut Controller, state: &str, eof: bool| {
            let cursor = c.view.cursor;
            c.accept(
                &Operation::Logs {
                    job: "current".into(),
                    cursor,
                },
                json!({"job":{"id":"current","state":state},"text":"x",
                    "cursor":cursor+1,"eof":eof}),
            )
            .unwrap();
        };
        accept(&mut c, "running", false);
        c.view.last_refresh = Some(Instant::now() - Duration::from_millis(50));
        assert!(!panel.should_refresh(&c));
        c.view.last_refresh = Some(Instant::now() - Duration::from_millis(250));
        assert!(panel.should_refresh(&c));
        accept(&mut c, "running", true);
        c.view.last_refresh = Some(Instant::now() - Duration::from_secs(4));
        assert!(!panel.should_refresh(&c));
        c.view.last_refresh = Some(Instant::now() - Duration::from_secs(6));
        assert!(panel.should_refresh(&c));
        accept(&mut c, "completed", true);
        c.view.last_refresh = Some(Instant::now() - Duration::from_millis(250));
        assert!(panel.should_refresh(&c)); // First terminal sample still needs final tail.
        accept(&mut c, "completed", true);
        c.view.last_refresh = Some(Instant::now() - Duration::from_secs(60));
        assert!(!panel.should_refresh(&c));
        c.view.connection_error = Some("transport lost".into());
        c.view.last_refresh = Some(Instant::now() - Duration::from_secs(14));
        assert!(!panel.should_refresh(&c));
        c.view.last_refresh = Some(Instant::now() - Duration::from_secs(16));
        assert!(panel.should_refresh(&c));
        panel.screen = Screen::Jobs;
        assert!(!panel.should_refresh(&c));
    }
    #[test]
    fn remote_preview_has_no_saved_job_or_local_output_side_effect() {
        let mut c = controller();
        let update = c
            .accept(
                &Operation::Start {
                    products: "none".into(),
                    preview: true,
                    binding: None,
                },
                json!({"dry_run":true,"review":{"config":"/remote/storm.toml"}}),
            )
            .unwrap();
        assert!(matches!(update, Update::Preview { .. }));
        assert!(c.store.selected().unwrap().last_job.is_none());
        assert!(!c.path.exists());
        assert!(c
            .accept(
                &Operation::Start {
                    products: "none".into(),
                    preview: true,
                    binding: None
                },
                json!({"job":{"id":"unexpected","state":"running"}})
            )
            .is_err());
    }
    #[test]
    fn confirmed_launch_and_resume_bind_exact_output_inputs_and_complete_checkpoint_set() {
        let n = node();
        let preview = Operation::Start {
            products: "none".into(),
            preview: true,
            binding: None,
        };
        let confirmed = preview.confirmed(&review(false)).unwrap();
        let args = n.args(&confirmed).unwrap();
        assert!(args
            .windows(2)
            .any(|v| v == ["--outdir", "/new output/unique"]));
        assert!(args.contains(&"--expected-config-sha256".into()));
        assert!(args.contains(&"--expected-input-sha256".into()));
        assert!(n
            .args(&Operation::Start {
                products: "none".into(),
                preview: false,
                binding: None
            })
            .is_err());
        let resume = Operation::Resume {
            job: "old".into(),
            checkpoint: "latest".into(),
            output: String::new(),
            preview: true,
            binding: None,
        };
        let args = n.args(&resume.confirmed(&review(true)).unwrap()).unwrap();
        assert!(args
            .windows(2)
            .any(|v| v == ["--from", "/old/run/checkpoint.npz"]));
        assert!(args.contains(&"--expected-checkpoint-set-sha256".into()));
        let mut incomplete = review(true);
        incomplete["checkpoint_set_sha256"] = Value::Null;
        assert!(resume.confirmed(&incomplete).is_err());
        let mut malformed = review(false);
        malformed["input_sha256"] = json!("bad hash");
        assert!(preview.confirmed(&malformed).is_err());
    }
}
