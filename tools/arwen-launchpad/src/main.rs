//! Loopback-only visual client for the existing ArWen command engine.
//! An actor owns every process handle; no OS handle crosses server threads.
#[path = "../../arwen-tui/src/job.rs"]
mod job;
use axum::{
    extract::{DefaultBodyLimit, State},
    http::{HeaderMap, StatusCode},
    response::{Html, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::mpsc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

type Answer = Result<Value, String>;
struct Request {
    action: String,
    body: Value,
    reply: mpsc::Sender<Answer>,
}
#[derive(Clone)]
struct Web {
    sender: mpsc::Sender<Request>,
    token: String,
    host: String,
}
struct Backend {
    python: PathBuf,
    source: PathBuf,
    workspace: PathBuf,
    geography: String,
    config: Option<PathBuf>,
    original: String,
    job: Option<job::Job>,
    catalog: Option<Value>,
}
fn poll_keeps_ownership(result: io::Result<Option<i32>>) -> bool {
    !matches!(result, Ok(Some(_)))
}
fn prepared_args(
    prepared: &str,
    config: &Path,
    output: &Path,
    wps: Option<&str>,
    print_only: bool,
) -> Vec<String> {
    let mut args = vec![
        prepared.into(),
        "--experiment-config".into(),
        display(config),
        "--outdir".into(),
        display(output),
    ];
    let companion = config.with_file_name(format!(
        "{}.namelist.wps",
        config.file_stem().unwrap_or_default().to_string_lossy()
    ));
    let selected = wps
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .or_else(|| companion.is_file().then_some(companion));
    if let Some(path) = selected {
        args.extend(["--wps-namelist".into(), display(&path)]);
    }
    if print_only {
        args.push("--print-command".into());
    }
    args
}
fn result_directory(command: &[String], log: &str) -> Option<PathBuf> {
    let explicit = log
        .lines()
        .filter_map(|line| line.strip_prefix("Output: "))
        .last()
        .map(PathBuf::from);
    explicit
        .or_else(|| {
            command
                .windows(2)
                .find(|a| a[0] == "--outdir")
                .map(|a| PathBuf::from(&a[1]))
        })
        .filter(|p| p.is_dir())
        .and_then(|p| p.canonicalize().ok())
}
fn open_directory(path: &Path) -> Result<(), String> {
    #[cfg(windows)]
    let mut command = Command::new("explorer.exe");
    #[cfg(not(windows))]
    let mut command = Command::new("xdg-open");
    command
        .arg(path)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    command
        .spawn()
        .map_err(|e| format!("Cannot open results folder: {e}"))?;
    Ok(())
}
fn stamp() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}
fn display(path: &Path) -> String {
    path.to_string_lossy()
        .strip_prefix(r"\\?\")
        .unwrap_or(&path.to_string_lossy())
        .to_string()
}
fn python_call(python: &Path, source: &Path, payload: Value) -> Answer {
    let mut command = Command::new(python);
    command
        .args(["-m", "gpuwm.launchpad_api"])
        .current_dir(source)
        .env("PYTHONPATH", source)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    let mut child = command
        .spawn()
        .map_err(|e| format!("Cannot start the configured Python: {e}"))?;
    child
        .stdin
        .take()
        .unwrap()
        .write_all(payload.to_string().as_bytes())
        .map_err(|e| e.to_string())?;
    let output = child.wait_with_output().map_err(|e| e.to_string())?;
    let response: Value = serde_json::from_slice(&output.stdout).map_err(|_| {
        format!(
            "ArWen could not read this request. {} {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    })?;
    if response["ok"] == true {
        Ok(response["value"].clone())
    } else {
        Err(response["error"]
            .as_str()
            .unwrap_or("ArWen refused this operation")
            .to_owned())
    }
}
fn save_existing(path: &Path, original: &str, text: &str) -> Result<PathBuf, String> {
    if fs::read_to_string(path).map_err(|e| e.to_string())? != original {
        return Err("This file changed outside the launchpad. Your draft is retained; reload or save a new copy.".into());
    }
    if path.extension().and_then(|v| v.to_str()) == Some("toml") {
        text.parse::<toml_edit::DocumentMut>()
            .map_err(|e| e.to_string())?;
    }
    let parent = path.parent().ok_or("No parent folder")?;
    let id = stamp();
    let backup = parent.join(format!(".arwen-backup-{id}"));
    let temporary = parent.join(format!(".arwen-saving-{id}"));
    let write = |p: &Path, value: &str| -> Result<(), String> {
        let mut f = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(p)
            .map_err(|e| e.to_string())?;
        f.write_all(value.as_bytes())
            .and_then(|_| f.sync_all())
            .map_err(|e| e.to_string())
    };
    write(&backup, original)?;
    write(&temporary, text)?;
    if fs::read_to_string(path).map_err(|e| e.to_string())? != original {
        return Err(format!(
            "External edit detected. Draft retained at {}",
            display(&temporary)
        ));
    }
    fs::rename(&temporary, path).map_err(|e| e.to_string())?;
    #[cfg(unix)]
    fs::File::open(parent)
        .and_then(|f| f.sync_all())
        .map_err(|e| e.to_string())?;
    Ok(backup)
}
fn toml_value(value: &Value) -> Result<toml_edit::Value, String> {
    Ok(match value {
        Value::String(s) => s.clone().into(),
        Value::Bool(b) => (*b).into(),
        Value::Number(n) if n.is_i64() => n.as_i64().unwrap().into(),
        Value::Number(n) => n.as_f64().ok_or("Not a finite number")?.into(),
        Value::Array(items) => {
            let mut a = toml_edit::Array::new();
            for i in items {
                a.push(toml_value(i)?);
            }
            a.into()
        }
        _ => return Err("Use the complete editor for table changes".into()),
    })
}
fn patch_document(text: &str, changes: &Value) -> Result<String, String> {
    let mut doc = text
        .parse::<toml_edit::DocumentMut>()
        .map_err(|e| e.to_string())?;
    for change in changes.as_array().ok_or("changes must be an array")? {
        let path = change["path"].as_str().ok_or("Missing field path")?;
        let keys: Vec<_> = path.split('/').filter(|s| !s.is_empty()).collect();
        let new = toml_value(&change["value"])?;
        let item = match keys.as_slice() {
            [table, key] => &mut doc[*table][*key],
            ["domain", index, key] => {
                let domains = doc["domain"]
                    .as_array_of_tables_mut()
                    .ok_or("No domain tables")?;
                let i: usize = index.parse().map_err(|_| "Invalid domain index")?;
                &mut domains.get_mut(i).ok_or("Domain is absent")?[*key]
            }
            _ => return Err("Unsupported field path; use the complete config editor".into()),
        };
        // Keep comments and spacing attached to the value being edited.
        let mut new = new;
        if let Some(old) = item.as_value() {
            *new.decor_mut() = old.decor().clone();
        }
        *item = toml_edit::Item::Value(new);
    }
    Ok(doc.to_string())
}
impl Backend {
    fn busy(&mut self) -> bool {
        self.job
            .as_mut()
            .map(|j| poll_keeps_ownership(j.poll()))
            .unwrap_or(false)
    }
    fn loaded(&mut self, path: PathBuf) -> Answer {
        if self.busy() {
            return Err(
                "Finish or stop the active command before changing its configuration.".into(),
            );
        }
        let path = path
            .canonicalize()
            .map_err(|e| format!("Cannot open that configuration: {e}"))?;
        let text = fs::read_to_string(&path).map_err(|e| e.to_string())?;
        let inspection = python_call(
            &self.python,
            &self.source,
            json!({"action":"inspect","text":text,"path":display(&path)}),
        )?;
        self.config = Some(path.clone());
        self.original = text.clone();
        Ok(json!({"path":display(&path),"text":text,"inspection":inspection}))
    }
    fn call(&mut self, action: &str, mut body: Value) -> Answer {
        match action {
            "catalog" => {
                if self.catalog.is_none() {
                    self.catalog = Some(python_call(
                        &self.python,
                        &self.source,
                        json!({"action":"catalog"}),
                    )?)
                }
                Ok(
                    json!({"catalog":self.catalog,"workspace":display(&self.workspace),"geography":self.geography,"python":display(&self.python)}),
                )
            }
            "open" => self.loaded(PathBuf::from(
                body["path"].as_str().ok_or("Enter a config path")?,
            )),
            "create" => {
                if self.busy() {
                    return Err(
                        "A command is running; wait before creating another forecast.".into(),
                    );
                }
                let folder = self
                    .workspace
                    .join("forecasts")
                    .join(format!("forecast-{}", stamp()));
                fs::create_dir_all(&folder).map_err(|e| e.to_string())?;
                body["directory"] = json!(display(&folder));
                body["action"] = json!("create");
                let made = python_call(&self.python, &self.source, body)?;
                let mut opened = self.loaded(PathBuf::from(
                    made["path"].as_str().ok_or("Wizard returned no config")?,
                ))?;
                opened["creation_log"] = made["creation_log"].clone();
                opened["requested_region"] = made["requested_region"].clone();
                Ok(opened)
            }
            "inspect" => {
                body["path"] = json!(self.config.as_ref().map(|p| display(p)));
                body["action"] = json!("inspect");
                python_call(&self.python, &self.source, body)
            }
            "patch" => {
                let text = patch_document(
                    body["text"].as_str().ok_or("No draft text")?,
                    &body["changes"],
                )?;
                let inspection = python_call(
                    &self.python,
                    &self.source,
                    json!({"action":"inspect","text":text,"path":self.config.as_ref().map(|p|display(p))}),
                )?;
                Ok(json!({"text":text,"inspection":inspection}))
            }
            "save" => {
                if self.busy() {
                    return Err("The active run owns its saved configuration. Your edits are still here; save after it finishes.".into());
                }
                let path = self
                    .config
                    .as_ref()
                    .ok_or("Open or create a forecast first")?;
                let text = body["text"].as_str().ok_or("No draft text")?;
                let backup = save_existing(path, &self.original, text)?;
                self.original = text.into();
                Ok(json!({"saved":display(path),"backup":display(&backup)}))
            }
            "save-as" => {
                let text = body["text"].as_str().ok_or("No draft text")?;
                let path = PathBuf::from(body["path"].as_str().ok_or("Name the new file")?);
                if !path.is_absolute() {
                    return Err("Use the full path for the new file so relative inputs have an explicit home.".into());
                }
                let mut f = fs::OpenOptions::new()
                    .create_new(true)
                    .write(true)
                    .open(&path)
                    .map_err(|e| e.to_string())?;
                f.write_all(text.as_bytes())
                    .and_then(|_| f.sync_all())
                    .map_err(|e| e.to_string())?;
                Ok(
                    json!({"saved":display(&path),"message":"New copy written. Relative input paths resolve from its folder; the original active config is unchanged."}),
                )
            }
            "external-plan" => {
                body["action"] = json!("external-plan");
                python_call(&self.python, &self.source, body)
            }
            "start" => {
                if self.busy() {
                    return Err(
                        "A command is already running. Follow it below, or stop that run first."
                            .into(),
                    );
                }
                let mode = body["mode"].as_str().ok_or("Choose an action")?;
                let output = body["output"]
                    .as_str()
                    .filter(|s| !s.is_empty())
                    .map(PathBuf::from)
                    .unwrap_or_else(|| self.workspace.join("runs"));
                let geog = body["geography"]
                    .as_str()
                    .unwrap_or(&self.geography)
                    .to_string();
                let (command, mut args) = match mode {
                    "doctor" => ("doctor", vec![]),
                    "external-run" => {
                        let kind = body["kind"].as_str().ok_or("Choose WRF input type")?;
                        if !["met-em", "wrfinput"].contains(&kind) {
                            return Err("Unknown external input type".into());
                        }
                        let directory = body["directory"]
                            .as_str()
                            .ok_or("Choose the input folder")?;
                        let mut a = vec![
                            format!("--{kind}"),
                            directory.into(),
                            "--outdir".into(),
                            display(&output),
                        ];
                        if let Some(seconds) = body["run_seconds"].as_f64() {
                            a.extend(["--run-seconds".into(), seconds.to_string()]);
                        }
                        ("run", a)
                    }
                    "advanced" => {
                        let values = body["argv"]
                            .as_array()
                            .ok_or("Enter a JSON list of ArWen arguments")?;
                        let a: Vec<String> = values
                            .iter()
                            .map(|v| {
                                v.as_str()
                                    .map(str::to_owned)
                                    .ok_or("Every command argument must be text")
                            })
                            .collect::<Result<_, _>>()?;
                        let first = a.first().ok_or("Enter an ArWen command")?;
                        // Existing engine verbs only, never shell execution.
                        let c = match first.as_str() {
                            "prep" => "prep",
                            "sim" => "sim",
                            "run" => "run",
                            "go" => "go",
                            "check" => "check",
                            "fetch" => "fetch",
                            "run-plan" => "run-plan",
                            _ => {
                                return Err(
                                    "Choose prep, sim, run, go, check, fetch or run-plan.".into()
                                )
                            }
                        };
                        (c, a[1..].to_vec())
                    }
                    _ => {
                        let config = self
                            .config
                            .as_ref()
                            .ok_or("Create or import a forecast first")?;
                        if body["text"].as_str() != Some(self.original.as_str()) {
                            return Err("Save your draft before planning or launching it.".into());
                        }
                        if fs::read_to_string(config).map_err(|e| e.to_string())? != self.original {
                            return Err(
                                "The config changed outside this window. Reload it before running."
                                    .into(),
                            );
                        }
                        let c = display(config);
                        match mode {
                            "check" => ("check", vec![c]),
                            "plan" => (
                                "go",
                                vec![c, "--outdir".into(), display(&output), "--dry-run".into()],
                            ),
                            "run" => ("go", vec![c, "--outdir".into(), display(&output)]),
                            "prepared" | "prepared-plan" => (
                                "sim",
                                prepared_args(
                                    body["prepared"]
                                        .as_str()
                                        .ok_or("Choose a prepared folder")?,
                                    config,
                                    &output,
                                    body["wps"].as_str(),
                                    mode == "prepared-plan",
                                ),
                            ),
                            _ => return Err("Unknown launch action".into()),
                        }
                    }
                };
                if command == "go" && !geog.is_empty() {
                    args.extend(["--geog-root".into(), geog]);
                }
                if command == "go" {
                    if let Some(products) = body["products"].as_str().filter(|s| !s.is_empty()) {
                        args.extend(["--products".into(), products.into()]);
                    }
                }
                let folder = self
                    .workspace
                    .join("jobs")
                    .join(format!("{}-{mode}", stamp()));
                self.job = Some(
                    job::Job::start(&self.python, command, &args, &folder, &self.source)
                        .map_err(|e| e.to_string())?,
                );
                self.call("status", json!({}))
            }
            "status" => {
                if let Some(j) = &mut self.job {
                    let log = j.log_tail(1000);
                    let output = result_directory(&j.command, &log);
                    let poll_error=j.poll().err().map(|e|format!("Process status is unknown: {e}. This job retains ownership until its status is resolved."));
                    Ok(
                        json!({"active":j.outcome.is_none() || poll_error.is_some(),"poll_error":poll_error,"exit_code":j.outcome,"command":j.command,"action":j.action,"seconds":j.started.elapsed().as_secs_f64(),"log":log,"output":output.map(|p|display(&p)),"folder":display(&j.dir)}),
                    )
                } else {
                    Ok(json!({"active":false,"log":"","command":[]}))
                }
            }
            "open-results" => {
                let j = self
                    .job
                    .as_ref()
                    .ok_or("No command has produced a results folder yet")?;
                let path = result_directory(&j.command, &j.log_tail(1000))
                    .ok_or("Results folder does not exist yet")?;
                open_directory(&path)?;
                Ok(json!({"opened":display(&path)}))
            }
            "stop" => {
                if let Some(j) = &mut self.job {
                    j.stop().map_err(|e| e.to_string())?;
                }
                self.call("status", json!({}))
            }
            _ => Err("Unknown launchpad endpoint".into()),
        }
    }
}
fn authorized(headers: &HeaderMap, web: &Web) -> bool {
    headers.get("host").and_then(|v| v.to_str().ok()) == Some(web.host.as_str())
        && headers.get("x-arwen-session").and_then(|v| v.to_str().ok()) == Some(web.token.as_str())
        && headers
            .get("origin")
            .map(|v| v.to_str().ok() == Some(format!("http://{}", web.host).as_str()))
            .unwrap_or(true)
}
async fn api(
    State(web): State<Web>,
    headers: HeaderMap,
    Json(mut payload): Json<Value>,
) -> impl IntoResponse {
    if !authorized(&headers, &web) {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"ok":false,"error":"Open this launchpad's local window to use it."})),
        );
    }
    let action = payload["action"].take().as_str().unwrap_or("").to_owned();
    let (send, recv) = mpsc::channel();
    if web
        .sender
        .send(Request {
            action,
            body: payload,
            reply: send,
        })
        .is_err()
    {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"ok":false,"error":"The local engine has closed"})),
        );
    }
    let result = tokio::task::spawn_blocking(move || recv.recv()).await;
    match result {
        Ok(Ok(Ok(value))) => (StatusCode::OK, Json(json!({"ok":true,"value":value}))),
        Ok(Ok(Err(error))) => (
            StatusCode::BAD_REQUEST,
            Json(json!({"ok":false,"error":error})),
        ),
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"ok":false,"error":"The local engine could not complete this request"})),
        ),
    }
}
async fn index(State(web): State<Web>, headers: HeaderMap) -> impl IntoResponse {
    if headers.get("host").and_then(|v| v.to_str().ok()) != Some(web.host.as_str()) {
        return (
            StatusCode::FORBIDDEN,
            Html("Local host required".to_owned()),
        );
    }
    (
        StatusCode::OK,
        Html(include_str!("../web/index.html").replace("__SESSION__", &web.token)),
    )
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut python = PathBuf::from("python");
    let mut source = std::env::current_dir()?;
    let mut workspace = source.join("launchpad-work");
    let mut geography = String::new();
    let mut port = 0u16;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--python" => python = args.next().ok_or("Missing Python")?.into(),
            "--source" => source = args.next().ok_or("Missing source")?.into(),
            "--workspace" => workspace = args.next().ok_or("Missing workspace")?.into(),
            "--geog-root" => geography = args.next().ok_or("Missing geography")?,
            "--port" => port = args.next().ok_or("Missing port")?.parse()?,
            "--help" => {
                println!("arwen-launchpad --python PYTHON --source ARWEN --workspace FORECASTS [--geog-root DIR] [--port PORT]\nOpens no forecast automatically. Visit the printed loopback URL.");
                return Ok(());
            }
            _ => return Err(format!("Unknown option {a}").into()),
        }
    }
    fs::create_dir_all(&workspace)?;
    workspace = workspace.canonicalize()?;
    source = source.canonicalize()?;
    // Exact lane imports also apply to descendants of the durable CLI worker.
    std::env::set_var("PYTHONPATH", &source);
    if let Some(parent) = python.parent().filter(|p| !p.as_os_str().is_empty()) {
        let mut paths = vec![parent.to_path_buf()];
        if let Some(existing) = std::env::var_os("PATH") {
            paths.extend(std::env::split_paths(&existing));
        }
        std::env::set_var("PATH", std::env::join_paths(paths)?);
    }
    let (sender, receiver) = mpsc::channel::<Request>();
    std::thread::spawn(move || {
        let mut b = Backend {
            python,
            source,
            workspace,
            geography,
            config: None,
            original: String::new(),
            job: None,
            catalog: None,
        };
        loop {
            match receiver.recv_timeout(Duration::from_millis(200)) {
                Ok(r) => {
                    let value = b.call(&r.action, r.body);
                    let _ = r.reply.send(value);
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    b.busy();
                }
                Err(_) => break,
            }
        }
    });
    let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, port)).await?;
    let host = listener.local_addr()?.to_string();
    let mut random = [0u8; 32];
    getrandom::fill(&mut random).map_err(|e| io::Error::other(e.to_string()))?;
    let web = Web {
        sender,
        token: random.iter().map(|b| format!("{b:02x}")).collect(),
        host: host.clone(),
    };
    let app = Router::new()
        .route("/", get(index))
        .route(
            "/app.js",
            get(|| async {
                (
                    [(axum::http::header::CONTENT_TYPE, "text/javascript")],
                    include_str!("../web/app.js"),
                )
            }),
        )
        .route(
            "/style.css",
            get(|| async {
                (
                    [(axum::http::header::CONTENT_TYPE, "text/css")],
                    include_str!("../web/style.css"),
                )
            }),
        )
        .route(
            "/world.svg",
            get(|| async {
                (
                    [(axum::http::header::CONTENT_TYPE, "image/svg+xml")],
                    include_str!("../web/world.svg"),
                )
            }),
        )
        .route(
            "/geography.json",
            get(|| async {
                (
                    [
                        (axum::http::header::CONTENT_TYPE, "application/json"),
                        (axum::http::header::CONTENT_ENCODING, "gzip"),
                    ],
                    include_bytes!("../web/geography.json.gz").as_slice(),
                )
            }),
        )
        .route(
            "/geography-sources",
            get(|| async {
                (
                    [(
                        axum::http::header::CONTENT_TYPE,
                        "text/plain; charset=utf-8",
                    )],
                    include_str!("../web/GEOGRAPHY-SOURCES.txt"),
                )
            }),
        )
        .route("/api", post(api))
        .layer(DefaultBodyLimit::max(2 * 1024 * 1024))
        .with_state(web);
    println!("ArWen launchpad: http://{host}\nLocal only. Opening the window starts no download or forecast.");
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn patch_keeps_comments_unknown_choices_and_other_domains() {
        let original="# owner comment\n[shared]\nmp_physics = 8 # chosen\nfuture_option = 'keep'\n[[domain]]\nnx = 64\n[[domain]]\nnx = 32\n";
        let got = patch_document(
            original,
            &json!([{"path":"shared/mp_physics","value":50},{"path":"domain/0/nx","value":96}]),
        )
        .unwrap();
        assert!(got.contains("# owner comment"));
        assert!(got.contains("mp_physics = 50 # chosen"));
        assert!(got.contains("future_option = 'keep'"));
        assert!(got.contains("nx = 32"));
    }
    #[test]
    fn inactive_and_arbitrary_toml_values_survive_patch() {
        let got = patch_document(
            "[shared]\no3input=0\nprivate_choice=901\n",
            &json!([{"path":"shared/mp_physics","value":8}]),
        )
        .unwrap();
        assert!(got.contains("private_choice=901"));
        assert!(got.contains("o3input=0"));
    }
    #[test]
    fn save_refuses_external_change_and_keeps_original() {
        let folder = std::env::temp_dir().join(format!("arwen-save-{}", stamp()));
        fs::create_dir(&folder).unwrap();
        let path = folder.join("case.toml");
        fs::write(&path, "# external\n[a]\nx=2\n").unwrap();
        assert!(save_existing(&path, "[a]\nx=1\n", "[a]\nx=3\n").is_err());
        assert!(fs::read_to_string(path).unwrap().contains("# external"));
    }
    #[test]
    fn save_roundtrips_unicode_with_backup() {
        let folder = std::env::temp_dir().join(format!("arwen-unicode-{}", stamp()));
        fs::create_dir(&folder).unwrap();
        let path = folder.join("case.toml");
        let old = "# 冬 — source\n[a]\nx=1\n";
        let new = "# 冬 — source\n[a]\nx=2\n";
        fs::write(&path, old).unwrap();
        let backup = save_existing(&path, old, new).unwrap();
        assert_eq!(fs::read_to_string(backup).unwrap(), old);
        assert_eq!(fs::read_to_string(path).unwrap(), new);
    }
    #[test]
    fn poll_error_never_releases_job_ownership() {
        assert!(poll_keeps_ownership(Err(io::Error::other("unknown"))));
        assert!(poll_keeps_ownership(Ok(None)));
        assert!(!poll_keeps_ownership(Ok(Some(0))));
    }
    #[test]
    fn prepared_plan_passes_explicit_wps_and_print_only() {
        let args = prepared_args(
            "prepared",
            Path::new("case.toml"),
            Path::new("out"),
            Some("original.wps"),
            true,
        );
        assert!(args
            .windows(2)
            .any(|v| v == ["--wps-namelist", "original.wps"]));
        assert!(args.contains(&"--print-command".into()));
    }
    #[test]
    fn unrelated_origins_cannot_start_jobs() {
        let (sender, _) = mpsc::channel();
        let web = Web {
            sender,
            token: "secret".into(),
            host: "127.0.0.1:32100".into(),
        };
        let mut h = HeaderMap::new();
        h.insert("host", "127.0.0.1:32100".parse().unwrap());
        h.insert("x-arwen-session", "secret".parse().unwrap());
        h.insert("origin", "https://unrelated.example".parse().unwrap());
        assert!(!authorized(&h, &web));
        h.remove("origin");
        assert!(authorized(&h, &web));
        h.insert("host", "other.example:32100".parse().unwrap());
        assert!(!authorized(&h, &web));
    }
}
