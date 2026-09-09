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
        for (label, value) in [
            ("Identity file on this computer", &self.identity),
            ("SSH config on this computer", &self.ssh_config),
        ] {
            if !value.is_empty() && !Path::new(value).is_absolute() {
                return Err(format!(
                    "{label} must be an absolute path. A relative path resolves against the folder the terminal was opened in, so that folder could supply an SSH configuration whose ProxyCommand runs a local program when you connect."
                ));
            }
        }
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
            Operation::ReviewPlan { plan, plan_sha256, config_sha256, output } => {
                if !plan.is_absolute() { return Err("Saved companion plan must use an absolute local path.".into()); }
                posix_absolute(output, "Remote output directory")?;
                valid_sha(plan_sha256, "plan")?;
                valid_sha(config_sha256, "configuration")?;
                args.extend(["--plan".into(), plan.to_string_lossy().into_owned(),
                    "--expected-plan-sha256".into(), plan_sha256.clone(),
                    "--expected-config-sha256".into(), config_sha256.clone(),
                    "--outdir".into(), output.clone()]);
                if !self.geography.is_empty() {
                    posix_absolute(&self.geography, "Remote geography directory")?;
                    args.extend(["--geog-root".into(), self.geography.clone()]);
                }
            }
            Operation::StartPlan { review } => {
                let bundle = review["bundle_id"].as_str().ok_or("Remote review has no bundle ID.")?;
                valid_job(bundle)?;
                args.extend(["--bundle-id".into(), bundle.into()]);
                for name in ["bundle", "plan", "config", "input"] {
                    let hash = review[format!("{name}_sha256")].as_str().ok_or("Remote review is missing an input hash.")?;
                    valid_sha(hash, name)?;
                    args.extend([format!("--expected-{name}-sha256"), hash.into()]);
                }
                if review["memory"]["measured"] != true || review["memory"]["refuse"] != false {
                    return Err("The remote memory review is not launch-ready. Review the selected node again.".into());
                }
                if review["source_blobs"].as_array().is_some_and(|blobs|!blobs.is_empty()){
                    let path=PathBuf::from(review["local_source_manifest"].as_str().ok_or("Large inputs need the completed local review for background verification.")?);
                    if !path.is_absolute(){return Err("Completed source-input review path must be absolute.".into());}
                    args.extend(["--source-inputs-file".into(),path.to_string_lossy().into_owned()]);
                }
            }
            Operation::SyncArtifacts { job, domain, cache, sequence, reader_leases } => {
                valid_job(job)?;
                if !(1..=999).contains(domain)||!cache.is_absolute(){return Err("Remote artifacts need a valid domain and absolute owned cache.".into());}
                args.extend(["--job".into(),job.clone(),"--domain".into(),domain.to_string(),
                    "--cache-root".into(),cache.to_string_lossy().into_owned()]);
                if let Some(sequence)=sequence{
                    if *sequence==0||*sequence>i64::MAX as u64{return Err("Artifact sequence must be a positive integer.".into());}
                    args.extend(["--sequence".into(),sequence.to_string()]);
                }
                if *reader_leases{args.push("--reader-leases".into());}
            }
            Operation::SyncNativePlots{job,domain,cache,sequence}=>{
                valid_job(job)?;
                if !(1..=999).contains(domain)||!cache.is_absolute()||*sequence==0||*sequence>i64::MAX as u64{return Err("Native plots need a valid domain, exact frame sequence and absolute cache.".into());}
                args.extend(["--job".into(),job.clone(),"--domain".into(),domain.to_string(),"--sequence".into(),sequence.to_string(),"--cache-root".into(),cache.to_string_lossy().into_owned()]);
            }
            Operation::ArtifactIndex{job,domain,after_sequence}=>{
                valid_job(job)?;
                if !(1..=999).contains(domain)||*after_sequence>i64::MAX as u64{return Err("Invalid artifact timeline selector.".into());}
                args.extend(["--job".into(),job.clone(),"--domain".into(),domain.to_string(),"--after-sequence".into(),after_sequence.to_string()]);
            }
            Operation::SyncProcessedFrame{job,domain,cache,sequence}=>{
                valid_job(job)?;
                if !(1..=999).contains(domain)||!cache.is_absolute(){return Err("Converted fields need a valid domain and absolute owned cache.".into());}
                args.extend(["--job".into(),job.clone(),"--domain".into(),domain.to_string(),"--cache-root".into(),cache.to_string_lossy().into_owned()]);
                if let Some(sequence)=sequence{
                    if *sequence==0||*sequence>i64::MAX as u64{return Err("Converted frame sequence must be a positive integer.".into());}
                    args.extend(["--sequence".into(),sequence.to_string()]);
                }
            }
            Operation::SyncProcessedFrameV2{job,domain,cache,sequence,options,reader_leases,cache_bytes}=>{
                valid_job(job)?;
                if !(1..=999).contains(domain)||!cache.is_absolute(){return Err("Viewer fields need a valid domain and absolute owned cache.".into());}
                args.extend(["--job".into(),job.clone(),"--domain".into(),domain.to_string(),"--cache-root".into(),cache.to_string_lossy().into_owned()]);
                if let Some(sequence)=sequence{if *sequence==0||*sequence>i64::MAX as u64{return Err("Viewer frame sequence must be positive.".into());}args.extend(["--sequence".into(),sequence.to_string()]);}
                options.args(&mut args);
                if *reader_leases{args.push("--reader-leases".into());}
                if let Some(bytes)=cache_bytes{if !(64*1024*1024..=1024_u64.pow(4)).contains(bytes){return Err("Viewer cache must be 64 MiB to 1 TiB.".into());}args.extend(["--cache-bytes".into(),bytes.to_string()]);}
            }
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

fn valid_sha(value:&str,label:&str)->Result<(),String>{
    if value.len()==64 && value.bytes().all(|c|c.is_ascii_hexdigit()){Ok(())}
    else{Err(format!("Invalid {label} SHA-256 binding."))}
}

fn artifact_producer_root(artifacts:&Value,manifest:&Value)->Result<String,String>{
    let outer=artifacts["remote_output_root"].as_str().ok_or("Remote artifact has no job output root.")?;
    let Some(binding)=artifacts.get("producer_binding") else{return Ok(outer.to_owned());};
    if binding["schema"]!="gpuwm.remote-producer-binding.v1"{return Err("Unknown native producer binding.".into());}
    let authority=|value:&Value|->Result<Value,String>{
        let raw=value["utf8"].as_str().ok_or("Native producer authority has no exact bytes.")?;
        if raw.len()>128*1024||value["sha256"]!=crate::companion::digest(raw.as_bytes()){
            return Err("Native producer authority hash changed.".into());
        }
        serde_json::from_str(raw).map_err(|_|"Native producer authority is not valid JSON.".into())
    };
    let parent=authority(&binding["parent_manifest"])?;
    let resolved=authority(&binding["parent_resolved"])?;
    let produced=authority(&binding["producer_resolved"])?;
    let pointer=binding["chain_pointer"]["utf8"].as_str().ok_or("Native chain pointer has no exact bytes.")?;
    let name=pointer.trim();
    if pointer.len()>256||!name.starts_with("run-")||name.contains('/')||name.contains('\\')
        ||name.contains("..")||name.chars().any(char::is_control)
        ||binding["chain_pointer"]["sha256"]!=crate::companion::digest(pointer.as_bytes()){
        return Err("Native chain pointer does not name one owned stamped run.".into());
    }
    let root=format!("{outer}/chain/{name}");
    let config=resolved["config_source"].as_str().ok_or("Native resolved receipt has no config source.")?;
    valid_sha(resolved["config_sha256"].as_str().unwrap_or(""),"native producer configuration")?;
    if parent["schema"]!="gpuwm.run-manifest.v1"||parent["route"]!="prepared"
        ||parent["run_dir"]!=outer||parent["outputs_dir"]!=outer||parent["pid"]!=manifest["pid"]
        ||parent["run_id"]==manifest["run_id"]
        ||binding["parent_manifest"]["remote_path"]!=format!("{outer}/run-manifest.json")
        ||binding["chain_pointer"]["remote_path"]!=format!("{outer}/chain/latest-run.txt")
        ||artifacts["run_manifest"]["remote_path"]!=format!("{root}/run-manifest.json")
        ||manifest["outputs_dir"]!=root||manifest["plan_source"]!=format!("gpuwm go {config}")
        ||resolved["schema_version"]!="gpuwm.run-plan.event.v1"||resolved["event"]!="resolved_plan"
        ||produced["schema_version"]!="gpuwm.run-plan.event.v1"||produced["event"]!="resolved_plan"
        ||resolved["config_source"]!=produced["config_source"]||resolved["config_sha256"]!=produced["config_sha256"]
        ||resolved["sequence"]!=binding["parent_resolved"]["sequence"]
        ||produced["sequence"]!=binding["producer_resolved"]["sequence"]
        ||parent["events_path"]!=binding["parent_resolved"]["remote_path"]
        ||manifest["events_path"]!=binding["producer_resolved"]["remote_path"]{
        return Err("Native producer does not match this job's parent manifest, pointer and configuration receipts.".into());
    }
    Ok(root)
}

fn validate_processed_frame(job:&str,domain:u32,cache:&Path,sequence:Option<u64>,value:&Value)->Result<(),String>{
    if value["schema"]!="arwen.remote-processed-frame.v1"||value["job_id"]!=job||value["domain"]!=domain||!value["waiting"].is_boolean(){
        return Err("Converted fields belong to a different job or domain.".into());
    }
    let returned_sequence=value["sequence"].as_u64().filter(|n|*n>0&&*n<=i64::MAX as u64);
    if sequence.is_some_and(|wanted|returned_sequence!=Some(wanted))
        &&!(value["waiting"]==true&&value["sequence"].is_null()&&value["commit"].is_null()){
        return Err("Converted fields belong to a different requested forecast time.".into());
    }
    let processing=&value["processing"];
    if !processing.is_null()&&(processing["schema"]!="arwen.native-store-queue.v1"||processing["job_id"]!=job
        ||!matches!(processing["state"].as_str(),Some("queued"|"processing"|"waiting_for_output"|"complete"|"failed"))){
        return Err("Native conversion progress belongs to another job or schema.".into());
    }
    let authority=|record:&Value|->Result<Value,String>{
        let raw=record["utf8"].as_str().ok_or("Converted fields have no exact native authority bytes.")?;
        if raw.len()>128*1024||record["sha256"]!=crate::companion::digest(raw.as_bytes()){
            return Err("Converted-field native authority checksum changed.".into());
        }
        serde_json::from_str(raw).map_err(|_|"Converted-field authority is not JSON.".into())
    };
    if !value["run_manifest"].is_null(){
        let manifest=authority(&value["run_manifest"])?;let root=artifact_producer_root(value,&manifest)?;
        if manifest["schema"]!="gpuwm.run-manifest.v1"||manifest["run_id"]!=value["run_id"]||manifest["pid"]!=value["remote_pid"]
            ||manifest["pid"].as_u64().is_none_or(|pid|pid==0)||manifest["run_dir"]!=root||manifest["outputs_dir"]!=root{
            return Err("Converted fields disagree with their original native producer.".into());
        }
        if !value["commit"].is_null(){
            let commit=authority(&value["commit"])?;
            if commit["schema_version"]!="gpuwm.run-plan.event.v1"||commit["event"]!="output_committed"||commit["domain"]!=domain
                ||commit["sequence"].as_u64()!=returned_sequence||value["commit"]["sequence"].as_u64()!=returned_sequence
                ||commit["valid_time"]!=value["valid_time"]||manifest["events_path"]!=value["commit"]["remote_path"]
                ||commit.get("run_id").is_some_and(|id|id!=&manifest["run_id"]){
                return Err("Converted fields disagree with the selected native frame commit.".into());
            }
        }else if value["waiting"]!=true{return Err("Ready converted fields have no committed-frame authority.".into());}
    }else if value["waiting"]!=true{return Err("Ready converted fields have no native producer manifest.".into());}
    if value["waiting"]==true{
        if !value["local_result_path"].is_null(){return Err("A pending conversion cannot supply a ready local result.".into());}
        return Ok(());
    }
    if returned_sequence.is_none()||!value["transferred_bytes"].is_u64(){return Err("Ready converted fields have no exact sequence or transfer count.".into());}
    valid_sha(value["source_sha256"].as_str().unwrap_or(""),"converted source")?;
    let archive=value["archive"]["sha256"].as_str().ok_or("Converted store has no archive identity.")?;
    valid_sha(archive,"converted archive")?;
    if value["archive"]["size_bytes"].as_u64().is_none_or(|n|n==0){return Err("Converted store archive has no byte length.".into());}
    let cache=cache.canonicalize().map_err(|error|error.to_string())?;
    let root=cache.join(archive);
    if root.is_symlink()||root.canonicalize().map_err(|error|error.to_string())?!=root{return Err("Converted store escaped its immutable cache directory.".into());}
    let path=PathBuf::from(value["local_result_path"].as_str().ok_or("Converted fields have no local result receipt.")?);
    if !path.is_absolute()||path.is_symlink()||path.canonicalize().map_err(|error|error.to_string())?!=root.join("native-result.json"){
        return Err("Converted result receipt is outside its owned archive cache.".into());
    }
    let result=crate::companion::read_json(&path,4*1024*1024)?;
    let frame=&result["frame"];let source=&result["remote_source"];
    if result["schema"]!="arwen.wrf-process-result.v1"||result["domain"]!=format!("d{domain:02}")||frame!=&value["frame"]
        ||frame["schema"]!="arwen.companion-store-frame.v1"||frame["identity"]["source_sha256"]!=value["source_sha256"]
        ||frame["identity"]["case_id"]!=value["run_id"]||source["job_id"]!=job||source["domain"]!=domain
        ||source["sequence"].as_u64()!=returned_sequence||source["run_id"]!=value["run_id"]||source["source_sha256"]!=value["source_sha256"]
        ||source["run_manifest"]!=value["run_manifest"]||source["commit"]!=value["commit"]{
        return Err("The local converted receipt lost its source, run, domain or time authority.".into());
    }
    let store=PathBuf::from(frame["store_root"].as_str().ok_or("Converted frame has no local store root.")?);
    let hour=PathBuf::from(frame["hour_path"].as_str().ok_or("Converted frame has no local hour file.")?);
    let store_canonical=store.canonicalize().map_err(|error|error.to_string())?;
    if !store.is_absolute()||store.is_symlink()||store_canonical==root.join("store")||!store_canonical.starts_with(root.join("store"))
        ||!hour.is_absolute()||hour.is_symlink()||!hour.canonicalize().map_err(|error|error.to_string())?.starts_with(&store_canonical)
        ||hour.extension().is_none_or(|extension|extension!="rws")
        ||frame["rws_bytes"].as_u64().filter(|bytes|*bytes>0)!=Some(fs::metadata(&hour).map_err(|error|error.to_string())?.len()){
        return Err("Converted native files are outside their verified local store or have changed length.".into());
    }
    valid_sha(frame["rws_sha256"].as_str().unwrap_or(""),"converted hour")?;
    valid_sha(frame["grid_sha256"].as_str().unwrap_or(""),"converted grid")?;
    Ok(())
}

fn validate_processed_frame_v2(job:&str,domain:u32,cache:&Path,sequence:Option<u64>,options:&ViewerOptions,value:&Value)->Result<(),String>{
    if value["schema"]!="arwen.remote-processed-frame.v2"||value["job_id"]!=job||value["domain"]!=domain
        ||!value["waiting"].is_boolean()||value["profile"]!=options.profile
        ||options.expected_run_id.as_ref().is_some_and(|id|!value["run_id"].is_null()&&value["run_id"]!=*id){
        return Err("Native viewer reply belongs to another job, run, domain or profile.".into());
    }
    let returned_sequence=value["sequence"].as_u64().filter(|n|*n>0&&*n<=i64::MAX as u64);
    if sequence.is_some_and(|wanted|returned_sequence!=Some(wanted)){return Err("Native viewer selected another committed time.".into());}
    if value["processing"]["schema"]!="arwen.native-store-queue.v2"||value["processing"]["job_id"]!=job{
        return Err("Native viewer progress belongs to another job or schema.".into());
    }
    if !options.products.is_empty(){
        let mut expected=options.products.clone();expected.sort();expected.dedup();
        if options.profile=="viewer-2d-v1"&&value["selection_products"]!=json!(expected){return Err("Native viewer products changed during the request.".into());}
    }
    let authority=|record:&Value|->Result<Value,String>{
        let raw=record["utf8"].as_str().filter(|text|text.len()<=128*1024).ok_or("Viewer authority is missing or excessive.")?;
        if record["sha256"]!=crate::companion::digest(raw.as_bytes()){return Err("Viewer source authority checksum changed.".into());}
        serde_json::from_str(raw).map_err(|_|"Viewer source authority is not JSON.".into())
    };
    if !value["run_manifest"].is_null(){
        let manifest=authority(&value["run_manifest"])?;let root=artifact_producer_root(value,&manifest)?;
        if manifest["schema"]!="gpuwm.run-manifest.v1"||manifest["run_id"]!=value["run_id"]||manifest["pid"]!=value["remote_pid"]
            ||manifest["pid"].as_u64().is_none_or(|pid|pid==0)||manifest["run_dir"]!=root||manifest["outputs_dir"]!=root{
            return Err("Viewer reply lost its original native producer.".into());
        }
        if !value["commit"].is_null(){
            let commit=authority(&value["commit"])?;
            if commit["schema_version"]!="gpuwm.run-plan.event.v1"||commit["event"]!="output_committed"||commit["domain"]!=domain
                ||commit["sequence"].as_u64()!=returned_sequence||value["commit"]["sequence"].as_u64()!=returned_sequence
                ||commit["valid_time"]!=value["valid_time"]||manifest["events_path"]!=value["commit"]["remote_path"]{
                return Err("Viewer reply lost its selected native frame commit.".into());
            }
        }else if value["waiting"]!=true{return Err("Ready viewer frame has no source commit.".into());}
    }else if value["waiting"]!=true{return Err("Ready viewer frame has no run manifest.".into());}
    if value["waiting"]==true{
        if !value["local_result_path"].is_null(){return Err("Pending viewer frame supplied a ready local receipt.".into());}
        return Ok(());
    }
    let publication=value["publication_sha256"].as_str().ok_or("Viewer publication has no immutable identity.")?;
    valid_sha(publication,"viewer publication")?;valid_sha(value["source_sha256"].as_str().unwrap_or(""),"viewer source")?;
    let cache=cache.canonicalize().map_err(|e|e.to_string())?;let root=cache.join("objects").join(publication);
    let canonical=|value:&Value|->Result<PathBuf,String>{
        let path=PathBuf::from(value.as_str().ok_or("Viewer local file has no path.")?);
        if !path.is_absolute()||path.is_symlink(){return Err("Viewer local path is not an owned absolute file.".into());}
        path.canonicalize().map_err(|e|e.to_string())
    };
    if root.is_symlink()||root.canonicalize().map_err(|e|e.to_string())?!=root||canonical(&value["cache_root"])?!=cache||canonical(&value["object_root"])?!=root{
        return Err("Viewer publication escaped its bounded local cache.".into());
    }
    let lease_path=cache.join("leases").join(format!("{publication}.lock"));
    if canonical(&value["cache_lease_path"])?!=lease_path{return Err("Viewer reader lease belongs to another cache object.".into());}
    let lease=OpenOptions::new().read(true).write(true).open(&lease_path).map_err(|e|e.to_string())?;
    lease.lock_shared().map_err(|e|e.to_string())?;
    let path=canonical(&value["local_result_path"])?;
    if path!=root.join("native-result.json"){return Err("Viewer result receipt escaped its object directory.".into());}
    let result=crate::companion::read_json(&path,512*1024)?;let frame=&result["frame"];let source=&result["remote_source"];
    if result["schema"]!=if options.profile=="viewer-2d-v1"{"arwen.wrf-process-result.v2"}else{"arwen.wrf-process-result.v1"}
        ||result["domain"]!=format!("d{domain:02}")||frame!=&value["frame"]||frame["schema"]!="arwen.companion-store-frame.v1"
        ||frame["identity"]["source"]!="arwen"||frame["identity"]["model"]!=format!("wrf-d{domain:02}")||!frame["identity"]["member"].is_null()
        ||frame["identity"]["case_id"]!=value["run_id"]||frame["identity"]["source_sha256"]!=value["source_sha256"]
        ||canonical(&frame["cache_lease_path"])?!=lease_path{
        return Err("Local viewer result changed its source, native frame or reader lease.".into());
    }
    for key in ["job_id","run_id","domain","sequence","source_sha256","run_manifest","commit","publication_sha256"]{
        if source[key]!=value[key]{return Err(format!("Local viewer result lost its {key} authority."));}
    }
    let valid=frame["identity"]["valid_unix"].as_i64().ok_or("Viewer frame has no exact valid UTC.")?;
    let lead=frame["identity"]["lead_seconds"].as_u64().ok_or("Viewer frame has no exact lead seconds.")?;
    let utc=crate::companion::local_progress::utc_ms(value["valid_time"].as_str().ok_or("Viewer frame has no committed valid time.")?)?;
    if i128::from(valid)*1000!=i128::from(utc)||value["lead_seconds"].as_u64()!=Some(lead)
        ||value["initialization_unix"].as_i64().map(i128::from)!=Some(i128::from(valid)-i128::from(lead)){
        return Err("Viewer frame changed its exact UTC initialization, lead or valid time.".into());
    }
    let members=value["members"].as_array().filter(|rows|!rows.is_empty()&&rows.len()<=16).ok_or("Viewer member catalog is missing or excessive.")?;
    let local=result["files"].as_array().filter(|rows|rows.len()==members.len()).ok_or("Local viewer member catalog changed.")?;
    let mut keys=std::collections::BTreeSet::new();
    for member in members{
        let key=member["key"].as_str().ok_or("Viewer member key is missing.")?;
        if !keys.insert(key){return Err("Viewer member key is duplicated.".into());}
        valid_sha(member["sha256"].as_str().unwrap_or(""),"viewer member")?;
        if member["grid_sha256"]!=frame["grid_sha256"]||member["bytes"].as_u64().is_none_or(|n|n==0||n>4*1024_u64.pow(3)){return Err("Viewer member has invalid size or moving-grid identity.".into());}
        let item=local.iter().find(|item|item["key"]==key).ok_or("Local viewer member is missing.")?;
        let path=canonical(&item["path"])?;
        if !path.starts_with(root.join("store"))||item["relative_path"]!=member["relative_path"]||fs::metadata(&path).map_err(|e|e.to_string())?.len()!=item["bytes"].as_u64().unwrap_or(0){return Err("Local viewer member escaped its store or changed length.".into());}
        if member["kind"]!="metadata"&&(item["sha256"]!=member["sha256"]||item["bytes"]!=member["bytes"]){return Err("Native weather member changed during local transfer.".into());}
        use sha2::{Digest,Sha256};let mut file=fs::File::open(path).map_err(|e|e.to_string())?;let mut hash=Sha256::new();let mut bytes=[0_u8;65536];
        loop{let count=file.read(&mut bytes).map_err(|e|e.to_string())?;if count==0{break;}hash.update(&bytes[..count]);}
        if item["sha256"]!=format!("{:x}",hash.finalize()){return Err("Local viewer member checksum changed.".into());}
    }
    valid_sha(frame["grid_sha256"].as_str().unwrap_or(""),"viewer grid")?;
    Ok(())
}

pub(crate) fn processed_message(value:&Value)->Result<String,String>{
    if value["schema"]=="arwen.remote-processed-frame.v2"{
        if value["state"]=="failed"{return Err(format!("Viewer derivation failed: {}",value["error"].as_str().unwrap_or("the native processor reported a failure")));}
        if value["state"]=="backpressure"{return Ok(value["error"].as_str().unwrap_or("Viewer derivation is waiting for cache space.").to_owned());}
        if value["waiting"]==true{return Ok(match value["state"].as_str(){Some("deriving")=>"Deriving the selected viewer fields on the node CPU.",Some("evicted")=>"Rebuilding the selected fields from retained WRF history.",Some("waiting_for_output")=>"Waiting for a committed forecast frame.",_=>"The selected viewer fields are queued on the node CPU."}.into());}
        return Ok("Selected native viewer fields are ready.".into());
    }
    if value["processing"]["state"]=="failed"{
        return Err(format!("Native field conversion could not finish: {}",value["processing"]["error"].as_str().unwrap_or("the node reported a conversion failure")));
    }
    if value["waiting"]==true{
        let processing=&value["processing"];
        return Ok(match (processing["ready"].as_u64(),processing["committed"].as_u64()){
            (Some(ready),Some(total))=>format!("Converting saved weather fields on the node: {ready} of {total} frames ready."),
            _=>"Waiting for the selected native weather fields to finish converting on the node.".into(),
        });
    }
    Ok("Converted native weather fields are ready.".into())
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
pub struct ViewerOptions {
    pub profile: String,
    pub products: Vec<String>,
    pub expected_run_id: Option<String>,
    pub prefetch_sequences: Vec<u64>,
}
impl ViewerOptions {
    pub(crate) fn from_value(value:&Value)->Result<Self,String>{
        let profile=value.get("profile").map(|v|v.as_str().ok_or("Viewer profile must be a string.")).transpose()?.unwrap_or("viewer-2d-v1");
        if !matches!(profile,"viewer-2d-v1"|"full-science-v1"){return Err("Unsupported native viewer profile.".into());}
        let products=match value.get("products"){
            None=>vec![],Some(value)=>{
                let rows=value.as_array().filter(|rows|!rows.is_empty()&&rows.len()<=96).ok_or("Choose 1..96 canonical viewer products.")?;
                rows.iter().map(|v|v.as_str().filter(|s|!s.is_empty()&&s.len()<=96&&s.bytes().all(|c|c.is_ascii_lowercase()||c.is_ascii_digit()||matches!(c,b'_'|b'-')))
                    .map(str::to_owned).ok_or_else(||"Invalid native viewer product slug.".to_owned())).collect::<Result<Vec<_>,_>>()?
            }};
        let expected_run_id=value.get("expected_run_id").map(|v|v.as_str().filter(|s|!s.is_empty()&&s.len()<=256&&!s.chars().any(char::is_control)).map(str::to_owned).ok_or("Invalid native run identity.")).transpose()?;
        let prefetch_sequences=match value.get("prefetch_sequences"){
            None=>vec![],Some(value)=>value.as_array().filter(|rows|rows.len()<=8).ok_or("Loop prefetch supports at most eight frames.")?.iter()
                .map(|v|v.as_u64().filter(|n|*n>0&&*n<=i64::MAX as u64).ok_or_else(||"Loop prefetch needs positive committed sequences.".to_owned())).collect::<Result<Vec<_>,_>>()?};
        Ok(Self{profile:profile.into(),products,expected_run_id,prefetch_sequences})
    }
    fn args(&self,args:&mut Vec<String>){
        args.extend(["--profile".into(),self.profile.clone()]);
        if !self.products.is_empty(){args.extend(["--products".into(),self.products.join(",")]);}
        if let Some(run)=&self.expected_run_id{args.extend(["--expected-run-id".into(),run.clone()]);}
        if !self.prefetch_sequences.is_empty(){args.extend(["--prefetch-sequences".into(),self.prefetch_sequences.iter().map(u64::to_string).collect::<Vec<_>>().join(",")]);}
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Operation {
    Probe,
    List,
    ReviewPlan { plan: PathBuf, plan_sha256: String, config_sha256: String, output: String },
    StartPlan { review: Value },
    SyncArtifacts { job: String, domain: u32, cache: PathBuf, sequence: Option<u64>, reader_leases: bool },
    SyncProcessedFrame { job: String, domain: u32, cache: PathBuf, sequence: Option<u64> },
    SyncProcessedFrameV2 { job: String, domain: u32, cache: PathBuf, sequence: Option<u64>, options: ViewerOptions, reader_leases: bool, cache_bytes: Option<u64> },
    SyncNativePlots { job: String, domain: u32, cache: PathBuf, sequence: u64 },
    ArtifactIndex { job: String, domain: u32, after_sequence: u64 },
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
        if review["memory"].is_object() && (review["memory"]["measured"] != true || review["memory"]["refuse"] == true) {
            return Err(format!("Remote memory review is not launch-ready: {}", review["memory"]["verdict"].as_str().unwrap_or("actual node capacity is unknown")));
        }
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
            Self::ReviewPlan { .. } => "review-plan",
            Self::StartPlan { .. } => "start-plan",
            Self::SyncArtifacts { .. } => "sync-artifacts",
            Self::SyncProcessedFrame { .. } => "sync-processed-frame",
            Self::SyncProcessedFrameV2 { .. } => "sync-processed-frame-v2",
            Self::SyncNativePlots { .. } => "sync-native-plots",
            Self::ArtifactIndex { .. } => "artifact-index",
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
                | Self::StartPlan { .. }
                | Self::Stop { .. }
                | Self::Resume { preview: false, .. }
        )
    }
    /// Polling reads whose launch record is not evidence of anything the
    /// user decided: connect, job list, status, log chunk, timeline page.
    /// Their request directories are rotated, not accumulated.
    pub fn transient(&self) -> bool {
        matches!(
            self,
            Self::Probe | Self::List | Self::Status { .. } | Self::Logs { .. } | Self::ArtifactIndex { .. }
        )
    }
}

/// One retained record per node for transient polls: the completed request
/// directory replaces `remote-last-<node>`; the previous record is removed.
/// In-flight requests keep unique `remote-<stamp>` directories, so concurrent
/// pollers on one node (the Nodes panel, each Runs viewer) never collide; a
/// lost rename race removes the completed directory instead.
fn retain_last_poll(directory: &Path, node_id: &str) {
    let Some(parent) = directory.parent() else { return; };
    let last = parent.join(format!("remote-last-{node_id}"));
    let _ = fs::remove_dir_all(&last);
    if fs::rename(directory, &last).is_err() {
        let _ = fs::remove_dir_all(directory);
    }
}

/// Remove every retained poll record under `<logs>/.arwen-tui` when the
/// terminal closes. Reviewed and mutating request records are kept.
pub fn remove_poll_records(logs: &Path) {
    let Ok(entries) = fs::read_dir(logs.join(".arwen-tui")) else { return; };
    for entry in entries.filter_map(Result::ok) {
        if entry.file_name().to_string_lossy().starts_with("remote-last-") {
            let _ = fs::remove_dir_all(entry.path());
        }
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
        let outcome = Self::completed(&self.job.dir, self.operation.action(), code);
        if self.operation.transient() {
            retain_last_poll(&self.job.dir, &self.node.id);
        }
        outcome
    }

    fn completed(directory: &Path, action: &str, code: i32) -> Result<Option<Value>, String> {
        let file = fs::File::open(directory.join("job.log")).map_err(|e| e.to_string())?;
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
        let reply = parse_reply(&text, action)?;
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
    /// Status/log polling has its own clock; timeline traffic cannot postpone it.
    pub last_job_refresh: Option<Instant>,
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
            last_job_refresh: None,
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
    PlanReviewed(Value),
    ArtifactsSynced(Value),
    ProcessedFrameSynced(Value),
    ProcessedFrameSyncedV2(Value),
    NativePlotsSynced(Value),
    ArtifactIndexed(Value),
    Started(String),
    Stopped(String),
    Failed(String),
}

impl Controller {
    /// Shared user preferences. Nothing is read from the launch folder: a
    /// `.arwen-nodes.json` there is untrusted input and is only opened through
    /// an explicit `--nodes-file ABSOLUTE_PATH`.
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
            Some(base) => Self::load_shared(base.join("nodes.json")),
            None => Self {
                store: Store::default(),
                path: cwd.join(".arwen-nodes.json"),
                view: View::default(),
                pending: None,
                load_error: Some("ArWen cannot locate your user configuration folder. Set HOME (Linux) or APPDATA (Windows); no node profiles were loaded from this folder.".into()),
            },
        }
    }

    fn load_shared(path: PathBuf) -> Self {
        Self::load_path(path)
    }

    #[cfg(test)]
    pub fn load(cwd: &Path) -> Self {
        Self::load_path(cwd.join(".arwen-nodes.json"))
    }

    pub(crate) fn load_path(path: PathBuf) -> Self {
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
        if matches!(request.operation,Operation::Status{..}|Operation::Logs{..}|Operation::List|Operation::Stop{..}|Operation::StartPlan{..}|Operation::Start{preview:false,..}|Operation::Resume{preview:false,..}){
            self.view.last_job_refresh=Some(Instant::now());
        }
        if self.store.selected().map(|node| (node.id.clone(), node.connection_key()))
            != Some((request.node.id.clone(), request.node.connection_key())) {
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
            Operation::ReviewPlan { plan_sha256, config_sha256, .. } => {
                let review = &reply["review"];
                if reply["dry_run"] != true || !review.is_object()
                    || review["source"]["plan_sha256"] != *plan_sha256
                    || review["source"]["config_sha256"] != *config_sha256 {
                    return Err("The node review does not match the saved companion plan and configuration.".into());
                }
                for name in ["bundle", "plan", "config", "input"] {
                    valid_sha(review[format!("{name}_sha256")].as_str().unwrap_or(""), name)?;
                }
                if !review["memory"].is_object() {
                    return Err("The node returned no actual memory review.".into());
                }
                let runtime = self.view.runtime.get_or_insert_with(|| json!({}));
                runtime["runtime"] = review["runtime"].clone();
                runtime["probe"] = review["probe"].clone();
                if !runtime["capabilities"].is_object(){runtime["capabilities"]=json!({});}
                for key in ["stage_plan_v1","review_plan_v1","start_plan_v1"]{runtime["capabilities"][key]=json!(true);}
                Ok(Update::PlanReviewed(review.clone()))
            }
            Operation::SyncArtifacts { job, domain, cache, sequence, reader_leases } => {
                let artifacts=&reply["artifacts"];
                if artifacts["schema"]!="gpuwm.remote-artifacts.v1"||artifacts["job_id"]!=*job
                    ||!artifacts["waiting"].is_boolean()||!reply["transferred_bytes"].is_u64(){
                    return Err("Node artifact reply is incomplete or belongs to a different job.".into());
                }
                if let Some(recovery)=reply.get("cache_recovery"){
                    if !reader_leases||artifacts["waiting"]!=true||recovery["schema"]!="arwen.artifact-cache-recovery.v1"
                        ||recovery["reason"]!="corrupt_retained_object"{return Err("Invalid retained-frame cache recovery response.".into());}
                    valid_sha(recovery["sha256"].as_str().unwrap_or(""),"retained frame")?;
                }
                if artifacts["waiting"]!=true {
                    let frames=artifacts["frames"].as_array().filter(|f|f.len()==1).ok_or("Node must return one selected-domain frame.")?;
                    let frame=&frames[0];
                    if frame["domain"]!=*domain{return Err("Node artifact belongs to a different domain.".into());}
                    if sequence.is_some_and(|value|frame["commit"]["sequence"].as_u64()!=Some(value)){return Err("Node artifact belongs to a different requested time.".into());}
                    for key in ["sha256","id"]{valid_sha(frame[key].as_str().unwrap_or(""),"artifact")?;}
                    let path=PathBuf::from(frame["path"].as_str().ok_or("Downloaded frame has no local path.")?);
                    let objects=cache.join("objects").canonicalize().map_err(|e|e.to_string())?;
                    if !path.is_absolute()||path.is_symlink()||path.canonicalize().map_err(|e|e.to_string())?!=objects.join(format!("{}.wrf",frame["sha256"].as_str().unwrap()))
                        ||frame["size_bytes"].as_u64().filter(|n|*n>0&&*n<=512*1024*1024)!=Some(fs::metadata(&path).map_err(|e|e.to_string())?.len()){
                        return Err("Downloaded artifact path or byte length disagrees with its owned cache.".into());
                    }
                    for authority in [&artifacts["run_manifest"],&frame["commit"]]{
                        let raw=authority["utf8"].as_str().ok_or("Artifact authority has no exact UTF-8 bytes.")?;
                        if authority["sha256"]!=crate::companion::digest(raw.as_bytes())||serde_json::from_str::<Value>(raw).is_err(){
                            return Err("Artifact source authority changed during transfer.".into());
                        }
                    }
                    let manifest:Value=serde_json::from_str(artifacts["run_manifest"]["utf8"].as_str().unwrap()).unwrap();
                    let commit:Value=serde_json::from_str(frame["commit"]["utf8"].as_str().unwrap()).unwrap();
                    let producer_root=artifact_producer_root(artifacts,&manifest)?;
                    if manifest["schema"]!="gpuwm.run-manifest.v1"||manifest["run_id"]!=artifacts["run_id"]
                        ||manifest["pid"]!=artifacts["remote_pid"]||manifest["run_dir"]!=producer_root
                        ||manifest["events_path"]!=frame["commit"]["remote_path"]||commit["schema_version"]!="gpuwm.run-plan.event.v1"
                        ||commit["event"]!="output_committed"||commit["domain"]!=frame["domain"]||commit["path"]!=frame["remote_path"]
                        ||commit["valid_time"]!=frame["valid_time"]||commit["sequence"]!=frame["commit"]["sequence"]{
                        return Err("Downloaded frame disagrees with its native run and commit identity.".into());
                    }
                }
                Ok(Update::ArtifactsSynced(reply))
            }
            Operation::SyncProcessedFrame{job,domain,cache,sequence}=>{
                validate_processed_frame(job,*domain,cache,*sequence,&reply["processed_frame"])?;
                Ok(Update::ProcessedFrameSynced(reply))
            }
            Operation::SyncProcessedFrameV2{job,domain,cache,sequence,options,..}=>{
                validate_processed_frame_v2(job,*domain,cache,*sequence,options,&reply["processed_frame"])?;
                Ok(Update::ProcessedFrameSyncedV2(reply))
            }
            Operation::SyncNativePlots{job,domain,cache,sequence}=>{
                validate_native_plots(job,*domain,cache,*sequence,&reply["native_plots"])?;
                Ok(Update::NativePlotsSynced(reply))
            }
            Operation::ArtifactIndex{job,domain,after_sequence}=>{
                let index=&reply["artifact_index"];
                if index["schema"]!="gpuwm.remote-artifact-index.v1"||index["job_id"]!=*job||index["domain"]!=*domain||!index["waiting"].is_boolean(){
                    return Err("Node timeline reply belongs to a different job or domain.".into());
                }
                let entries=index["entries"].as_array().filter(|rows|rows.len()<=256).ok_or("Node timeline page is excessive or missing.")?;
                let mut last=*after_sequence;
                for entry in entries{
                    let sequence=entry["sequence"].as_u64().filter(|n|*n>last&&*n<=i64::MAX as u64).ok_or("Node timeline sequences must increase.")?;
                    if entry["domain"]!=*domain||entry["valid_time"].as_str().is_none_or(|v|v.is_empty()||v.len()>64){return Err("Node timeline has an invalid forecast time.".into());}
                    last=sequence;
                }
                if !index["next_after_sequence"].is_null()&&(entries.is_empty()||index["next_after_sequence"].as_u64()!=Some(last)){
                    return Err("Node timeline cursor disagrees with its last entry.".into());
                }
                if let Some(raw)=index["run_manifest"]["utf8"].as_str(){
                    if index["run_manifest"]["sha256"]!=crate::companion::digest(raw.as_bytes()){return Err("Native timeline manifest bytes changed.".into());}
                    let manifest:Value=serde_json::from_str(raw).map_err(|_|"Native timeline manifest is not JSON.")?;
                    let root=artifact_producer_root(index,&manifest)?;
                    if manifest["schema"]!="gpuwm.run-manifest.v1"||manifest["run_id"]!=index["run_id"]||manifest["pid"]!=index["remote_pid"]||manifest["run_dir"]!=root{
                        return Err("Node timeline disagrees with its native producer.".into());
                    }
                }else if index["waiting"]!=true||!entries.is_empty(){return Err("Node timeline has no native run authority.".into());}
                Ok(Update::ArtifactIndexed(reply))
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
            Operation::Start { preview: false, .. } | Operation::Resume { preview: false, .. } | Operation::StartPlan { .. } => {
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

/// Reuse the native receipt validator for an independent read-only session.
/// The ephemeral controller never loads, selects or persists a user profile.
pub(crate) fn validate_readonly_reply(node:&Node,operation:&Operation,reply:&Value)->Result<(),String>{
    if !matches!(operation,Operation::List|Operation::Status{..}|Operation::ArtifactIndex{..}|Operation::SyncArtifacts{..}|Operation::SyncProcessedFrame{..}|Operation::SyncProcessedFrameV2{..}|Operation::SyncNativePlots{..}){
        return Err("This run viewer only accepts read-only job and artifact requests.".into());
    }
    let mut store=Store::default();store.nodes.push(node.clone());store.active=Some(node.id.clone());
    let mut controller=Controller{store,path:PathBuf::new(),view:View::default(),pending:None,load_error:None};
    controller.accept(operation,reply.clone()).map(|_|())
}

fn validate_native_plots(job:&str,domain:u32,cache:&Path,sequence:u64,value:&Value)->Result<(),String>{
    if value["schema"]!="arwen.native-plots.v1"||value["job_id"]!=job||value["domain"]!=domain
        ||value["sequence"]!=sequence||!value["waiting"].is_boolean(){return Err("Native plots belong to a different job, domain or forecast time.".into());}
    if value["waiting"]==true{return Ok(());}
    for authority in [&value["run_manifest"],&value["commit"]]{
        let raw=authority["utf8"].as_str().ok_or("Native plots have no exact source authority.")?;
        if authority["sha256"]!=crate::companion::digest(raw.as_bytes()){return Err("Native plot authority changed.".into());}
    }
    let manifest:Value=serde_json::from_str(value["run_manifest"]["utf8"].as_str().unwrap()).map_err(|e|e.to_string())?;
    let commit:Value=serde_json::from_str(value["commit"]["utf8"].as_str().unwrap()).map_err(|e|e.to_string())?;
    let root=artifact_producer_root(value,&manifest)?;
    if manifest["schema"]!="gpuwm.run-manifest.v1"||manifest["run_id"]!=value["run_id"]||manifest["pid"]!=value["remote_pid"]||manifest["run_dir"]!=root
        ||manifest["events_path"]!=value["commit"]["remote_path"]||commit["event"]!="output_committed"||commit["sequence"]!=sequence
        ||commit["domain"]!=domain||commit["valid_time"]!=value["valid_time"]{return Err("Native plots disagree with the committed forecast frame.".into());}
    let path=PathBuf::from(value["gallery_path"].as_str().ok_or("Native gallery has no local path.")?);
    let owned=cache.canonicalize().map_err(|e|e.to_string())?;
    let resolved=path.canonicalize().map_err(|e|e.to_string())?;
    if !path.is_absolute()||path.is_symlink()||!resolved.starts_with(&owned)||path.file_name().and_then(|p|p.to_str())!=Some("index.html")
        ||fs::metadata(&path).map_err(|e|e.to_string())?.len()>512*1024{return Err("Native gallery is outside its owned cache.".into());}
    let bytes=fs::read(&path).map_err(|e|e.to_string())?;
    if value["gallery_sha256"]!=crate::companion::digest(&bytes){return Err("Native gallery checksum changed.".into());}
    Ok(())
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
    #[test]
    fn compact_viewer_options_pin_run_products_and_eight_frame_prefetch(){
        let options=ViewerOptions::from_value(&json!({"profile":"viewer-2d-v1","products":["mslp_10m_winds"],"expected_run_id":"run-1","prefetch_sequences":[43,44]})).unwrap();
        let operation=Operation::SyncProcessedFrameV2{job:"job-1".into(),domain:2,sequence:Some(42),cache:std::env::temp_dir().join("viewer-cache"),options,reader_leases:true,cache_bytes:None};
        let args=node().args(&operation).unwrap();
        assert_eq!(args[0],"sync-processed-frame-v2");assert!(!operation.mutates());
        for (flag,wanted) in [("--sequence","42"),("--expected-run-id","run-1"),("--prefetch-sequences","43,44"),("--products","mslp_10m_winds")]{
            assert_eq!(args[args.iter().position(|arg|arg==flag).unwrap()+1],wanted);
        }
        assert!(args.iter().any(|arg|arg=="--reader-leases"));
        assert!(ViewerOptions::from_value(&json!({"prefetch_sequences":[0]})).is_err());
        assert!(ViewerOptions::from_value(&json!({"prefetch_sequences":[1,2,3,4,5,6,7,8,9]})).is_err());
        assert!(ViewerOptions::from_value(&json!({"profile":"unknown"})).is_err());
        assert_eq!(ViewerOptions::from_value(&json!({"profile":"full-science-v1"})).unwrap().profile,"full-science-v1");
    }
    #[test]
    fn compact_viewer_pending_reply_cannot_change_selected_run_time_or_profile(){
        let options=ViewerOptions::from_value(&json!({"expected_run_id":"run-1"})).unwrap();
        let value=json!({"schema":"arwen.remote-processed-frame.v2","job_id":"job-1","domain":2,"sequence":42,
            "waiting":true,"state":"queued","profile":"viewer-2d-v1","run_id":"run-1",
            "processing":{"schema":"arwen.native-store-queue.v2","job_id":"job-1","state":"idle"}});
        assert!(validate_processed_frame_v2("job-1",2,Path::new("unused"),Some(42),&options,&value).is_ok());
        for (key,wrong) in [("run_id",json!("other-run")),("domain",json!(1)),("sequence",json!(43)),("profile",json!("full-science-v1"))]{
            let mut changed=value.clone();changed[key]=wrong;
            assert!(validate_processed_frame_v2("job-1",2,Path::new("unused"),Some(42),&options,&changed).is_err(),"{key}");
        }
    }
    #[test]
    #[ignore="Requires ARWEN_COMPACT_READY pointing to the Python member-transport protocol fixture"]
    fn compact_viewer_python_transport_fixture_passes_tui_validation(){
        let path=PathBuf::from(std::env::var_os("ARWEN_COMPACT_READY").expect("explicit Python protocol fixture"));
        let value:Value=serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        let options=ViewerOptions::from_value(&json!({"profile":value["profile"],"expected_run_id":value["run_id"],"products":value["selection_products"]})).unwrap();
        let cache=PathBuf::from(value["cache_root"].as_str().unwrap());
        validate_processed_frame_v2(value["job_id"].as_str().unwrap(),value["domain"].as_u64().unwrap() as u32,&cache,value["sequence"].as_u64(),&options,&value).unwrap();
        let mut changed=value.clone();changed["members"][0]["grid_sha256"]=json!("f".repeat(64));
        assert!(validate_processed_frame_v2(value["job_id"].as_str().unwrap(),value["domain"].as_u64().unwrap() as u32,&cache,value["sequence"].as_u64(),&options,&changed).is_err());
    }
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
    fn map_review_uses_local_saved_plan_and_never_profile_configuration(){
        let mut node=node();node.config="/unrelated/profile-case.toml".into();
        let plan=std::env::temp_dir().join("saved map plan.json");
        let operation=Operation::ReviewPlan{plan:plan.clone(),plan_sha256:"a".repeat(64),config_sha256:"b".repeat(64),output:"/new/output".into()};
        let args=node.args(&operation).unwrap();
        assert_eq!(args[0],"review-plan");
        assert_eq!(args[args.iter().position(|v|v=="--plan").unwrap()+1],plan.to_string_lossy());
        assert!(!args.contains(&node.config));assert!(!operation.mutates());
        let review=json!({"bundle_id":"1234567890abcdef1234567890abcdef","bundle_sha256":"a".repeat(64),
            "plan_sha256":"b".repeat(64),"config_sha256":"c".repeat(64),"input_sha256":"d".repeat(64),
            "memory":{"measured":true,"refuse":false}});
        let launch=Operation::StartPlan{review:review.clone()};
        assert!(launch.mutates());assert!(node.args(&launch).is_ok());
        let mut with_blobs=review.clone();with_blobs["source_blobs"]=json!([{"source_path":"selected.grib"}]);
        assert!(node.args(&Operation::StartPlan{review:with_blobs.clone()}).is_err());
        let local_review=std::env::temp_dir().join("completed-node-review.json");
        with_blobs["local_source_manifest"]=json!(local_review);
        let args=node.args(&Operation::StartPlan{review:with_blobs}).unwrap();
        assert_eq!(args[args.iter().position(|v|v=="--source-inputs-file").unwrap()+1],local_review.to_string_lossy());
        let mut unknown=review;unknown["memory"]["measured"]=json!(false);
        assert!(node.args(&Operation::StartPlan{review:unknown}).is_err());
    }
    #[test]
    fn hosted_artifact_keeps_exact_parent_pointer_and_both_config_receipts(){
        let outer="/owned/run";let name="run-20260907-180001Z_i201305311200Z";
        let root=format!("{outer}/chain/{name}");
        let authority=|path:&str,value:Value|{let utf8=format!("{}\n",value);json!({"remote_path":path,"sha256":crate::companion::digest(utf8.as_bytes()),"utf8":utf8,"sequence":1})};
        let parent=json!({"schema":"gpuwm.run-manifest.v1","route":"prepared","pid":123,"run_id":"parent",
            "run_dir":outer,"outputs_dir":outer,"events_path":format!("{outer}/events.jsonl")});
        let manifest=json!({"pid":123,"run_id":"producer","outputs_dir":root,"events_path":format!("{root}/events.jsonl"),"plan_source":"gpuwm go /saved/case.toml"});
        let resolved=json!({"schema_version":"gpuwm.run-plan.event.v1","event":"resolved_plan","sequence":1,
            "config_source":"/saved/case.toml","config_sha256":"b".repeat(64)});
        let pointer=format!("{name}\n");
        let artifacts=json!({"remote_output_root":outer,"run_manifest":{"remote_path":format!("{root}/run-manifest.json")},
            "producer_binding":{"schema":"gpuwm.remote-producer-binding.v1",
                "parent_manifest":authority(&format!("{outer}/run-manifest.json"),parent),
                "parent_resolved":authority(&format!("{outer}/events.jsonl"),resolved.clone()),
                "producer_resolved":authority(&format!("{root}/events.jsonl"),resolved),
                "chain_pointer":{"remote_path":format!("{outer}/chain/latest-run.txt"),"utf8":pointer,"sha256":crate::companion::digest(pointer.as_bytes())}}});
        assert_eq!(artifact_producer_root(&artifacts,&manifest).unwrap(),root);
        let mut wrong=artifacts.clone();wrong["producer_binding"]["chain_pointer"]["utf8"]=json!("../escape");
        assert!(artifact_producer_root(&wrong,&manifest).is_err());
        let mut wrong=artifacts.clone();wrong["producer_binding"]["producer_resolved"]["sha256"]=json!("c".repeat(64));
        assert!(artifact_producer_root(&wrong,&manifest).is_err());
        let mut foreign=manifest.clone();foreign["pid"]=json!(999);
        assert!(artifact_producer_root(&artifacts,&foreign).is_err());
        let mut foreign=manifest;foreign["plan_source"]=json!("gpuwm go /other/case.toml");
        assert!(artifact_producer_root(&artifacts,&foreign).is_err());
    }
    #[test]
    fn timeline_page_and_explicit_time_args_are_bound_without_opening_a_raw_frame(){
        let root=std::env::temp_dir().join(format!("arwen-index-control-{}",stamp()));
        let manifest=json!({"schema":"gpuwm.run-manifest.v1","run_id":"run-1","pid":123,"run_dir":"/owned/run"}).to_string();
        let reply=json!({"schema":REPLY_SCHEMA,"ok":true,"action":"artifact-index","artifact_index":{
            "schema":"gpuwm.remote-artifact-index.v1","job_id":"job-1","domain":3,"waiting":false,
            "run_id":"run-1","remote_pid":123,"remote_output_root":"/owned/run",
            "run_manifest":{"utf8":manifest,"sha256":crate::companion::digest(manifest.as_bytes())},
            "entries":[{"sequence":42,"domain":3,"valid_time":"2013-05-31T18:00:00Z"}],"next_after_sequence":null,"latest_sequence":42}});
        let operation=Operation::ArtifactIndex{job:"job-1".into(),domain:3,after_sequence:40};
        let args=node().args(&operation).unwrap();assert_eq!(args[0],"artifact-index");
        assert!(args.contains(&"--after-sequence".into())&&!operation.mutates());
        let mut controller=Controller::load(&root);
        assert!(matches!(controller.accept(&operation,reply.clone()),Ok(Update::ArtifactIndexed(_))));
        let mut wrong=reply.clone();wrong["artifact_index"]["entries"][0]["sequence"]=json!(40);assert!(controller.accept(&operation,wrong).is_err());
        let mut wrong=reply;wrong["artifact_index"]["next_after_sequence"]=json!(43);assert!(controller.accept(&operation,wrong).is_err());
        let operation=Operation::SyncArtifacts{job:"job-1".into(),domain:3,cache:root.join("cache"),sequence:Some(42),reader_leases:true};
        let args=node().args(&operation).unwrap();
        assert_eq!(args[args.iter().position(|a|a=="--sequence").unwrap()+1],"42");
        assert!(args.contains(&"--reader-leases".into()));
    }
    #[test]
    fn converted_store_transfer_is_read_only_and_ready_receipts_keep_owned_paths_and_source(){
        let root=std::env::temp_dir().join(format!("arwen-converted-control-{}",stamp()));fs::create_dir_all(&root).unwrap();let root=root.canonicalize().unwrap();
        let archive="c".repeat(64);let object=root.join(&archive);let store=object.join("store/identity/rw-store");let hour=store.join("wrf/native/f006.rws");
        fs::create_dir_all(hour.parent().unwrap()).unwrap();let bytes=b"metadata-validator fixture, not a scientific array";fs::write(&hour,bytes).unwrap();
        let authority=|path:&str,value:Value|{let utf8=serde_json::to_string(&value).unwrap();json!({"remote_path":path,"sha256":crate::companion::digest(utf8.as_bytes()),"utf8":utf8})};
        let manifest=authority("/original/run-manifest.json",json!({"schema":"gpuwm.run-manifest.v1","run_id":"native-original","pid":42,"run_dir":"/original","outputs_dir":"/original","events_path":"/original/events.jsonl"}));
        let mut commit=authority("/original/events.jsonl",json!({"schema_version":"gpuwm.run-plan.event.v1","event":"output_committed","domain":3,"sequence":42,"valid_time":"2013-05-20T06:00:00Z","path":"/original/frame"}));commit["sequence"]=json!(42);
        let source_sha="b".repeat(64);let frame=json!({"schema":"arwen.companion-store-frame.v1","identity":{"source_sha256":source_sha,"case_id":"native-original"},
            "store_root":store,"hour_path":hour,"rws_bytes":bytes.len(),"rws_sha256":crate::companion::digest(bytes),"grid_sha256":"d".repeat(64)});
        let source=json!({"job_id":"saved-job","domain":3,"sequence":42,"run_id":"native-original","source_sha256":source_sha,"run_manifest":manifest,"commit":commit});
        let result_path=object.join("native-result.json");let result=json!({"schema":"arwen.wrf-process-result.v1","domain":"d03","frame":frame,"remote_source":source});
        fs::write(&result_path,serde_json::to_vec(&result).unwrap()).unwrap();
        let value=json!({"schema":"arwen.remote-processed-frame.v1","job_id":"saved-job","domain":3,"sequence":42,"waiting":false,
            "run_id":"native-original","remote_pid":42,"remote_output_root":"/original","run_manifest":manifest,"commit":commit,"valid_time":"2013-05-20T06:00:00Z",
            "source_sha256":source_sha,"archive":{"sha256":archive,"size_bytes":123},"frame":frame,"local_result_path":result_path,"transferred_bytes":123,
            "processing":{"schema":"arwen.native-store-queue.v1","job_id":"saved-job","state":"complete"}});
        let operation=Operation::SyncProcessedFrame{job:"saved-job".into(),domain:3,sequence:Some(42),cache:root.clone()};
        let args=node().args(&operation).unwrap();assert_eq!(args[0],"sync-processed-frame");assert!(!operation.mutates());
        assert_eq!(args[args.iter().position(|arg|arg=="--sequence").unwrap()+1],"42");
        validate_processed_frame("saved-job",3,&root,Some(42),&value).unwrap();
        for(key,replacement)in[("job_id",json!("other")),("domain",json!(2)),("sequence",json!(43)),("source_sha256",json!("a".repeat(64))),
            ("local_result_path",json!(root.join("outside.json")))]{let mut wrong=value.clone();wrong[key]=replacement;assert!(validate_processed_frame("saved-job",3,&root,Some(42),&wrong).is_err(),"{key}");}
        let mut wrong=value.clone();wrong["commit"]["sha256"]=json!("a".repeat(64));assert!(validate_processed_frame("saved-job",3,&root,Some(42),&wrong).is_err());
        let mut wrong_result=result;wrong_result["remote_source"]["job_id"]=json!("other");fs::write(&result_path,serde_json::to_vec(&wrong_result).unwrap()).unwrap();
        assert!(validate_processed_frame("saved-job",3,&root,Some(42),&value).unwrap_err().contains("authority"));
        let _=fs::remove_dir_all(root);
    }
    #[test]
    fn converted_field_waiting_and_failure_are_explicit_without_a_raw_frame(){
        let value=json!({"schema":"arwen.remote-processed-frame.v1","job_id":"saved-job","domain":1,"sequence":null,"waiting":true,
            "processing":{"schema":"arwen.native-store-queue.v1","job_id":"saved-job","state":"processing","committed":57,"ready":12}});
        validate_processed_frame("saved-job",1,Path::new("uncreated-cache"),None,&value).unwrap();
        assert!(processed_message(&value).unwrap().contains("12 of 57"));
        let mut failed=value;failed["processing"]["state"]=json!("failed");failed["processing"]["error"]=json!("source checksum changed");
        assert!(processed_message(&failed).unwrap_err().contains("source checksum changed"));
    }
    #[test]
    #[ignore="explicit real converted reply/cache; validates existing metadata and lengths without decoding fields or contacting nodes"]
    fn actual_converted_reply_keeps_its_native_authorities_and_local_store(){
        let reply_path=PathBuf::from(std::env::var_os("ARWEN_PROCESSED_REPLY").expect("ARWEN_PROCESSED_REPLY"));
        let cache=PathBuf::from(std::env::var_os("ARWEN_PROCESSED_CACHE").expect("ARWEN_PROCESSED_CACHE"));
        let job=std::env::var("ARWEN_PROCESSED_JOB").expect("ARWEN_PROCESSED_JOB");
        let domain=std::env::var("ARWEN_PROCESSED_DOMAIN").unwrap().parse::<u32>().unwrap();
        let sequence=std::env::var("ARWEN_PROCESSED_SEQUENCE").unwrap().parse::<u64>().unwrap();
        let reply=crate::companion::read_json(&reply_path,2*1024*1024).unwrap();
        validate_processed_frame(&job,domain,&cache,Some(sequence),&reply["processed_frame"]).unwrap();
        assert_eq!(reply["processed_frame"]["waiting"],false);
        println!("Verified real converted job={job}, domain={domain}, sequence={sequence}, fields={}, local_result={}",
            reply["processed_frame"]["frame"]["variables"].as_array().unwrap().len(),reply["processed_frame"]["local_result_path"]);
    }
    #[test]
    fn artifact_reply_is_bound_to_job_domain_cache_and_exact_native_commit(){
        let root=std::env::temp_dir().join(format!("arwen-artifact-control-{}",stamp()));
        let cache=root.join("cache");fs::create_dir_all(cache.join("objects")).unwrap();
        let bytes=b"raw protocol fixture; not weather";let hash=crate::companion::digest(bytes);
        let path=cache.join("objects").join(format!("{hash}.wrf"));fs::write(&path,bytes).unwrap();
        let manifest=json!({"schema":"gpuwm.run-manifest.v1","run_id":"run-1","pid":123,"run_dir":"/owned/run","events_path":"/owned/run/events.jsonl"}).to_string();
        let commit=json!({"schema_version":"gpuwm.run-plan.event.v1","event":"output_committed","sequence":4,"domain":2,
            "path":"/owned/run/wrfout_d02","valid_time":"2026-09-07T18:00:00Z"}).to_string();
        let reply=json!({"schema":REPLY_SCHEMA,"ok":true,"action":"sync-artifacts","transferred_bytes":bytes.len(),
            "artifacts":{"schema":"gpuwm.remote-artifacts.v1","job_id":"job-1","waiting":false,"run_id":"run-1","remote_pid":123,"remote_output_root":"/owned/run",
            "run_manifest":{"utf8":manifest,"sha256":crate::companion::digest(manifest.as_bytes())},
            "frames":[{"id":"a".repeat(64),"domain":2,"path":path,"remote_path":"/owned/run/wrfout_d02","sha256":hash,"size_bytes":bytes.len(),"valid_time":"2026-09-07T18:00:00Z",
                "commit":{"utf8":commit,"sha256":crate::companion::digest(commit.as_bytes()),"remote_path":"/owned/run/events.jsonl","sequence":4}}]}});
        let op=Operation::SyncArtifacts{job:"job-1".into(),domain:2,cache,sequence:None,reader_leases:false};let args=node().args(&op).unwrap();
        assert_eq!(args[0],"sync-artifacts");assert!(!op.mutates());assert!(!args.contains(&node().config));
        let mut controller=Controller::load(&root);
        assert!(matches!(controller.accept(&op,reply.clone()),Ok(Update::ArtifactsSynced(_))));
        let mut wrong=reply.clone();wrong["artifacts"]["job_id"]=json!("job-2");assert!(controller.accept(&op,wrong).is_err());
        let mut wrong=reply.clone();wrong["artifacts"]["frames"][0]["domain"]=json!(1);assert!(controller.accept(&op,wrong).is_err());
        let mut wrong=reply.clone();wrong["artifacts"]["frames"][0]["valid_time"]=json!("2026-09-07T19:00:00Z");assert!(controller.accept(&op,wrong).is_err());
        let mut wrong=reply;wrong["artifacts"]["frames"][0]["commit"]["utf8"]=json!("{}");assert!(controller.accept(&op,wrong).is_err());
    }
    #[test]
    fn node_paths_are_arguments_and_explicit_products_survive() {
        let mut node = node();
        node.python = "/opt/ArWen env/bin/python".into();
        node.identity = if cfg!(windows) { "C:\\ArWen\\Operator's Keys\\weather key".into() } else { "/opt/ArWen/Operator's Keys/weather key".into() };
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
    fn shared_profiles_ignore_node_files_in_the_launch_folder() {
        let root = std::env::temp_dir().join(format!("arwen-shared-nodes-{}", stamp()));
        let planted_folder = root.join("storms");
        let shared = root.join("preferences/nodes.json");
        // A folder the user merely opened the terminal in carries a profile
        // whose active node points at an SSH configuration of its own.
        let mut planted = Controller::load(&planted_folder);
        let mut n = node();
        n.name = "Linux node".into();
        n.ssh_config = if cfg!(windows) { "C:\\planted\\sshconf".into() } else { "/planted/sshconf".into() };
        planted.save_node(n.clone()).unwrap();
        planted.select(Some(n.id.clone())).unwrap();
        let original = fs::read(&planted.path).unwrap();
        let shared_store = Controller::load_shared(shared.clone());
        assert!(shared_store.load_error.is_none());
        assert!(shared_store.store.nodes.is_empty());
        assert!(shared_store.store.selected().is_none());
        assert!(!shared.exists(), "nothing was merged or written");
        assert_eq!(fs::read(&planted.path).unwrap(), original);
        // The explicit --nodes-file route still opens exactly that file.
        let explicit = Controller::load_path(planted.path.clone());
        assert!(explicit.load_error.is_none());
        assert_eq!(explicit.store.selected(), Some(&n));
    }
    #[test]
    fn identity_and_ssh_config_paths_must_be_absolute() {
        let mut n = node();
        n.identity = "keys/weather".into();
        let error = n.validate(false).unwrap_err();
        assert!(error.contains("Identity file on this computer must be an absolute path"), "{error}");
        assert!(error.contains("ProxyCommand"), "{error}");
        n.identity.clear();
        n.ssh_config = "sshconf".into();
        let error = n.validate(false).unwrap_err();
        assert!(error.contains("SSH config on this computer must be an absolute path"), "{error}");
        n.ssh_config = if cfg!(windows) { "C:\\ArWen\\ssh\\config".into() } else { "/opt/arwen/ssh/config".into() };
        n.identity = if cfg!(windows) { "C:\\ArWen\\ssh\\weather".into() } else { "/opt/arwen/ssh/weather".into() };
        n.validate(false).unwrap();
        // A saved store carrying a relative path is refused with the same
        // message and the file is preserved for the user to correct.
        let path = std::env::temp_dir().join(format!("arwen-relative-identity-{}", stamp())).join("nodes.json");
        let mut store = Store { nodes: vec![n.clone()], active: None, loaded_bytes: None, legacy_imports: Vec::new() };
        store.save(&path).unwrap();
        let mut planted: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        planted["nodes"][0]["identity"] = json!("weather");
        fs::write(&path, serde_json::to_vec(&planted).unwrap()).unwrap();
        let loaded = Controller::load_path(path.clone());
        assert!(loaded.load_error.as_deref().is_some_and(|error| error.contains("absolute path")), "{:?}", loaded.load_error);
        assert!(loaded.store.nodes.is_empty());
        assert_eq!(serde_json::from_slice::<Value>(&fs::read(&path).unwrap()).unwrap(), planted);
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
        c.view.last_refresh = Some(Instant::now() - Duration::from_millis(500));
        assert!(!panel.should_refresh(&c));
        c.view.last_refresh = Some(Instant::now() - Duration::from_secs(6));
        assert!(panel.should_refresh(&c));
        accept(&mut c, "completed", true);
        c.view.last_refresh = Some(Instant::now() - Duration::from_millis(250));
        assert!(panel.should_refresh(&c)); // First terminal sample still needs final tail.
        accept(&mut c, "completed", true);
        c.view.last_refresh = Some(Instant::now() - Duration::from_secs(60));
        assert!(!panel.should_refresh(&c));
        c.view.status.as_mut().unwrap()["native_plots"]=json!({"done":false});
        assert!(panel.should_refresh(&c));
        c.view.last_refresh=Some(Instant::now()-Duration::from_secs(1));
        assert!(!panel.should_refresh(&c));
        c.view.status.as_mut().unwrap()["native_plots"]=json!({"done":true});
        c.view.last_refresh=Some(Instant::now()-Duration::from_secs(60));
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
