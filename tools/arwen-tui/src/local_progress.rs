//! Read-only local counterpart of the node's native progress summary.
//! Bind the launcher, process, saved configuration and native run before
//! reading bounded JSON metadata. No model or weather array is opened.
use super::{digest,read_json};
use serde_json::{json,Value};
use std::{collections::BTreeMap,fs,io::{Read,Seek,SeekFrom},path::{Path,PathBuf},sync::{Mutex,OnceLock},time::{Duration,Instant}};

struct Cached {directory:PathBuf,command:Vec<String>,at:Instant,value:Result<Value,String>}
static CACHE:OnceLock<Mutex<Option<Cached>>>=OnceLock::new();
pub fn cached(directory:&Path,command:&[String])->Result<Value,String>{
    let mut cache=CACHE.get_or_init(||Mutex::new(None)).lock().map_err(|_|"Local progress cache is unavailable")?;
    if let Some(old)=cache.as_ref(){if old.directory==directory&&old.command==command&&old.at.elapsed()<Duration::from_millis(500){return old.value.clone();}}
    let value=read(directory,command);*cache=Some(Cached{directory:directory.into(),command:command.to_vec(),at:Instant::now(),value:value.clone()});value
}
fn raw(path:&Path,limit:u64)->Result<Vec<u8>,String>{
    let mut bytes=Vec::new();fs::File::open(path).map_err(|e|format!("{}: {e}",path.display()))?.take(limit+1).read_to_end(&mut bytes).map_err(|e|e.to_string())?;
    if bytes.len() as u64>limit{return Err(format!("{} exceeds the progress metadata limit",path.display()));}Ok(bytes)
}
fn text<'a>(v:&'a Value,key:&str)->Result<&'a str,String>{v[key].as_str().filter(|s|!s.is_empty()&&!s.chars().any(char::is_control)).ok_or_else(||format!("Missing native {key}"))}
fn canonical(path:&Path)->Result<PathBuf,String>{path.canonicalize().map_err(|e|format!("{}: {e}",path.display()))}
fn resolve(path:&str,base:&Path)->Result<PathBuf,String>{let p=Path::new(path);canonical(&if p.is_absolute(){p.into()}else{base.join(p)})}
fn same(path:&str,expected:&Path)->bool{Path::new(path).is_absolute()&&canonical(Path::new(path)).as_deref()==Ok(expected)}
fn owned(path:&str,root:&Path)->Result<PathBuf,String>{
    let path=Path::new(path);if !path.is_absolute()||path.components().any(|part|matches!(part,std::path::Component::ParentDir)){return Err("Native metadata path must be absolute without traversal".into());}
    let result=canonical(path)?;if !result.starts_with(root){return Err("Native metadata is outside this run".into());}Ok(result)
}
fn hash(value:&Value)->bool{value.as_str().is_some_and(|s|s.len()==64&&s.bytes().all(|b|b.is_ascii_digit()||(b'a'..=b'f').contains(&b)))}
fn stamp_name(name:&str)->bool{
    let b=name.as_bytes();if !name.is_ascii()||b.len()<20||!b.starts_with(b"run-")||b[12]!=b'-'||b[19]!=b'Z'||!b[4..12].iter().chain(&b[13..19]).all(u8::is_ascii_digit){return false;}
    let mut rest=&b[20..];if rest.starts_with(b"_i"){if rest.len()<15||rest[14]!=b'Z'||!rest[2..14].iter().all(u8::is_ascii_digit){return false;}rest=&rest[15..];}
    rest.is_empty()||rest.len()>1&&rest[0]==b'-'&&rest[1..].iter().all(u8::is_ascii_digit)
}
fn pointed(parent:&Path)->Result<Option<PathBuf>,String>{
    let pointer=parent.join("latest-run.txt");if !pointer.exists(){return Ok(None);}
    let bytes=raw(&pointer,256)?;let name=std::str::from_utf8(&bytes).map_err(|_|"Native run pointer is not UTF-8")?.trim();
    if !stamp_name(name){return Err("Native pointer does not name one stamped run".into());}
    owned(&parent.join(name).to_string_lossy(),parent).map(Some)
}
fn events(path:&Path,tail:bool)->Result<Vec<Value>,String>{
    let mut file=fs::File::open(path).map_err(|e|e.to_string())?;let size=file.metadata().map_err(|e|e.to_string())?.len();let cap=if tail{512*1024}else{1024*1024};let offset=if tail{size.saturating_sub(cap)}else{0};
    file.seek(SeekFrom::Start(offset)).map_err(|e|e.to_string())?;let mut bytes=Vec::new();file.take(cap).read_to_end(&mut bytes).map_err(|e|e.to_string())?;
    let begin=if offset>0{bytes.iter().position(|c|*c==b'\n').map(|n|n+1).unwrap_or(bytes.len())}else{0};
    let mut result=Vec::new();let mut sequence=0;
    for line in bytes[begin..].split_inclusive(|b|*b==b'\n'){
        if !line.ends_with(b"\n"){break;}if line.len()>128*1024{return Err("Native progress event exceeds its byte limit".into());}
        let value:Value=serde_json::from_slice(line).map_err(|_|"Native progress event is not JSON")?;
        let next=value["sequence"].as_u64().filter(|n|*n>sequence).ok_or("Native event sequence moved backward")?;
        if value["schema_version"]!="gpuwm.run-plan.event.v1"{return Err("Unknown native progress event schema".into());}sequence=next;result.push(value);
        if !tail&&result.len()>=64{break;}
    }Ok(result)
}
fn resolved(manifest:&Value,root:&Path,start:i64,config:&Path,sha:&str)->Result<(),String>{
    let path=owned(text(manifest,"events_path")?,root)?;
    let values=events(&path,false)?;let value=values.iter().find(|v|v["event"]=="resolved_plan").ok_or("Waiting for the native configuration receipt")?;
    if value["emitted_unix_ms"].as_i64().is_none_or(|ms|ms<start)||value["config_sha256"]!=sha||!same(text(value,"config_source")?,config){return Err("Native configuration receipt does not match this job's saved setup".into());}Ok(())
}
fn manifest(root:&Path,pid:u64,earliest:i64,latest:Option<i64>)->Result<(Value,Vec<u8>,i64),String>{
    let bytes=raw(&root.join("run-manifest.json"),256*1024)?;let value:Value=serde_json::from_slice(&bytes).map_err(|_|"Native manifest is not JSON")?;
    let start=utc_ms(text(&value,"started_at_utc")?)?;
    if value["schema"]!="gpuwm.run-manifest.v1"||value["pid"].as_u64()!=Some(pid)||start<earliest||latest.is_some_and(|last|start>last)
        ||!same(text(&value,"run_dir")?,root)||!same(text(&value,"outputs_dir")?,root)||!hash(&value["plan_sha256"]){return Err("Native manifest does not belong to this launcher's process and output".into());}
    text(&value,"run_id")?;Ok((value,bytes,start))
}
fn read(directory:&Path,command:&[String])->Result<Value,String>{
    let launcher=read_json(&directory.join("job.json"),128*1024)?;let process=read_json(&directory.join("process.json"),64*1024)?;
    if command.len()<5||launcher["schema"]!="gpuwm-tui-job-v1"||launcher["command"]!=json!(command)||process["schema"]!="gpuwm-tui-process-v1"||process["cli_args"]!=json!(&command[3..]){return Err("The local process receipt does not match its launch command".into());}
    let cwd=canonical(Path::new(text(&launcher,"cwd")?))?;if !same(text(&process,"cwd")?,&cwd){return Err("The local process working directory changed".into());}
    let pid=process["pid"].as_u64().filter(|n|*n>0).ok_or("Local process has no PID")?;let started=utc_ms(text(&process,"started_at")?)?;
    let ended=read_json(&directory.join("result.json"),128*1024).ok().filter(|r|r["schema"]=="gpuwm-tui-result-v1"&&r["pid"].as_u64()==Some(pid)&&r["cli_args"]==process["cli_args"])
        .and_then(|r|r["ended_at"].as_str().and_then(|s|utc_ms(s).ok()));
    let flag=|name:&str|command.iter().position(|v|v==name).and_then(|i|command.get(i+1)).map(String::as_str);
    let (config,base,plan)=if command[3]=="run-plan"{
        let path=resolve(&command[4],&cwd)?;let bytes=raw(&path,4*1024*1024)?;let value:Value=serde_json::from_slice(&bytes).map_err(|_|"The saved plan is not JSON")?;
        if value["schema"]!="gpuwm.run-plan.v1"{return Err("Unknown local run plan schema".into());}
        let parent=path.parent().ok_or("Plan has no parent")?;
        (resolve(text(&value["config"],"path")?,parent)?,resolve(text(&value,"output_root")?,parent)?,Some((path,digest(&bytes))))
    }else if matches!(command[3].as_str(),"go"|"sim"|"run"|"resume"){
        let name=flag("--experiment-config").or_else(||flag("--config")).or_else(||(command[3]=="go").then_some(command[4].as_str())).ok_or("This local route has not named its saved configuration")?;
        let config=resolve(name,&cwd)?;let base=flag("--outdir").map(|p|resolve(p,&cwd)).transpose()?.unwrap_or_else(||config.with_file_name(format!("{}-go",config.file_stem().unwrap_or_default().to_string_lossy())));
        (config,canonical(&base)?,None)
    }else{return Err("This command is not a forecast run".into());};
    let config_bytes=raw(&config,128*1024)?;let config_sha=digest(&config_bytes);
    let root=if plan.is_some(){base.clone()}else{pointed(&base)?.unwrap_or(base.clone())};
    let (parent,parent_bytes,parent_started)=manifest(&root,pid,started,ended)?;
    if let Some((path,sha))=&plan{
        if parent["plan_sha256"]!=*sha||!same(text(&parent,"plan_source")?,path){return Err("Native manifest does not match the exact reviewed local plan".into());}
    }else if !parent["plan_source"].as_str().and_then(|s|s.strip_prefix("gpuwm go ")).is_some_and(|p|same(p,&config)){
        return Err("Native manifest does not identify this saved local configuration".into());
    }
    resolved(&parent,&root,parent_started,&config,&config_sha)?;
    let native=&parent["native_run"];
    let next=if native.is_object(){
        if native["schema"]!="gpuwm.native-run-binding.v1"||native["pid"].as_u64()!=Some(pid)||native["config_sha256"]!=config_sha||!same(text(native,"config_source")?,&config){return Err("Native child binding does not match this local job".into());}
        Some(owned(text(native,"run_dir")?,&root)?)
    }else if parent["route"]=="prepared"&&root.join("chain").is_dir(){pointed(&canonical(&root.join("chain"))?)?}else{None};
    let (root,manifest,manifest_bytes,start)=if let Some(next)=next{
        if next.parent()!=Some(canonical(&root.join("chain"))?.as_path()){return Err("Native child is outside this job's owned chain".into());}
        let (value,bytes,start)=manifest(&next,pid,parent_started,ended)?;
        if !value["plan_source"].as_str().and_then(|s|s.strip_prefix("gpuwm go ")).is_some_and(|p|same(p,&config))||value["run_id"]==parent["run_id"]{return Err("Native child does not identify the saved configuration".into());}
        if native.is_object()&&(native["manifest_sha256"]!=digest(&bytes)||native["run_id"]!=value["run_id"]||!same(text(native,"manifest_path")?,&next.join("run-manifest.json"))){return Err("Native child manifest changed after its parent bound it".into());}
        resolved(&value,&next,start,&config,&config_sha)?;(next,value,bytes,start)
    }else{(root,parent,parent_bytes,parent_started)};
    let event_path=owned(text(&manifest,"events_path")?,&root)?;let mut stream=events(&event_path,true)?;
    stream.retain(|e|e["emitted_unix_ms"].as_i64().is_some_and(|ms|ms>=start));
    let heartbeat_path=manifest["progress_path"].as_str().map(|p|owned(p,&root)).transpose()?;
    let heartbeat=heartbeat_path.as_ref().and_then(|p|read_json(p,64*1024).ok()).filter(|h|h["schema"]=="gpuwm.run-progress/v1"&&h["run_id"]==manifest["run_id"]&&h["pid"].as_u64()==Some(pid)&&h["config_digest"]==config_sha
        &&h["started_at_utc"].as_str().and_then(|s|utc_ms(s).ok())==Some(start)&&h["updated_at_utc"].as_str().and_then(|s|utc_ms(s).ok()).is_some_and(|ms|ms>=start));
    let mut result=summarize(&config_bytes,&config_sha,&manifest,&manifest_bytes,&stream,heartbeat.as_ref(),&root)?;
    result["pipeline_progress"]=pipeline_progress(&stream,&result,start,ended);
    for (key,value) in [("run_dir",json!(root)),("outputs_dir",json!(root)),("run_manifest_path",json!(root.join("run-manifest.json"))),("events_path",json!(event_path)),("progress_path",json!(heartbeat_path)),("manifest_ready",json!(true)),("source_config_path",json!(config)),("source_config_sha256",json!(config_sha))]{result[key]=value;}
    result["ready_dir"]=if root.join("ready").is_dir(){json!(root.join("ready"))}else{Value::Null};Ok(result)
}

fn leap(year:i64)->bool{year%4==0&&(year%100!=0||year%400==0)}
fn month_days(year:i64,month:u8)->i64{match month{2=>if leap(year){29}else{28},4|6|9|11=>30,_=>31}}
fn before_year(year:i64)->i64{let previous=year-1;365*previous+previous/4-previous/100+previous/400}
fn utc_ms(value:&str)->Result<i64,String>{
    let parsed=value.parse::<toml_edit::Datetime>().map_err(|_|"Invalid UTC timestamp in native metadata")?;
    let date=parsed.date.ok_or("Native timestamp has no date")?;let time=parsed.time.ok_or("Native timestamp has no time")?;
    let year=i64::from(date.year);
    if year<1||year>9999||!(1..=12).contains(&date.month)||i64::from(date.day)>month_days(year,date.month)||date.day==0||time.second>59{return Err("Invalid calendar timestamp in native metadata".into());}
    let days=before_year(year)-before_year(1970)+(1..date.month).map(|m|month_days(year,m)).sum::<i64>()+i64::from(date.day)-1;
    let offset=match parsed.offset{Some(toml_edit::Offset::Custom{minutes})=>i64::from(minutes)*60_000,_=>0};
    Ok((days*86400+i64::from(time.hour)*3600+i64::from(time.minute)*60+i64::from(time.second))*1000+i64::from(time.nanosecond/1_000_000)-offset)
}
fn utc_text(milliseconds:i64)->Result<String,String>{
    let days=milliseconds.div_euclid(86_400_000)+before_year(1970);let within=milliseconds.rem_euclid(86_400_000);
    if days<0||days>=before_year(10000){return Err("Forecast time is outside the supported calendar".into());}
    let(mut low,mut high)=(1,10000);while low+1<high{let middle=(low+high)/2;if before_year(middle)<=days{low=middle}else{high=middle}}
    let mut day=days-before_year(low);let mut month=1;while day>=month_days(low,month){day-=month_days(low,month);month+=1;}
    let seconds=within/1000;let base=format!("{low:04}-{month:02}-{:02}T{:02}:{:02}:{:02}",day+1,seconds/3600,seconds/60%60,seconds%60);
    Ok(if within%1000==0{format!("{base}Z")}else{format!("{base}.{:03}Z",within%1000)})
}
fn seconds(value:&Value)->Option<f64>{value.as_f64().filter(|v|v.is_finite()&&*v>=0.)}
fn pipeline_progress(events:&[Value],result:&Value,started:i64,ended:Option<i64>)->Value{
    let stage=result["stage"].as_str().unwrap_or("starting");let stage=if stage.starts_with("preparing:"){"prepare"}else{stage};
    let mut phase=result["phase"].as_str().unwrap_or(stage).to_owned();let(mut began,mut updated,mut finished)=(started,started,None);
    let mut acquisition=json!({});let mut preparation=Value::Null;let mut files:BTreeMap<String,Value>=BTreeMap::new();
    for event in events{
        let Some(emitted)=event["emitted_unix_ms"].as_i64()else{continue};updated=updated.max(emitted);let tag=event["event"].as_str().unwrap_or("");
        if tag=="stage_started"{began=emitted;finished=None;}else if tag=="stage_finished"{finished=seconds(&event["wall_seconds"]);}
        let supplied=&event["acquisition"];if supplied["schema"]=="arwen.acquisition-progress.v1"{
            for key in ["source","provider","phase","dataset"]{if supplied[key].as_str().is_some_and(|v|v.len()<=160){acquisition[key]=supplied[key].clone();}}
            for key in ["requests_completed","requests_total","request_index","files_completed","files_total","forcing_hours","forcing_times_total","forcing_times_completed","bytes_available","transferred_bytes"]{if supplied[key].as_u64().is_some_and(|v|v<i64::MAX as u64){acquisition[key]=supplied[key].clone();}}
            if supplied["reused"].is_boolean(){acquisition["reused"]=supplied["reused"].clone();}
        }
        if matches!(tag,"fetch_started"|"fetch_progress"|"fetch_completed"){if let Some(name)=event["file"].as_str().filter(|s|!s.is_empty()&&s.len()<=512){
            let item=files.entry(name.to_owned()).or_insert_with(||json!({}));if tag=="fetch_started"{*item=json!({});}
            for key in ["bytes","expected_bytes"]{if event[key].as_u64().is_some_and(|v|v<i64::MAX as u64){item[key]=event[key].clone();}}
            if tag=="fetch_completed"{item["completed"]=json!(event["failed"]==false);}
        }}
        let supplied=&event["preparation"];if event["code"]=="preparation_progress"&&matches!(supplied["schema"].as_str(),Some("gpuwm.prep-stage.v1"|"gpuwm.prepare-progress/v1")){
            preparation=json!({});for key in ["schema","label","stage","event","backend","index","count","phase","phase_index","phases_total","elapsed_seconds","outcome"]{let value=&supplied[key];if value.is_number()||value.as_str().is_some_and(|v|v.len()<=512){preparation[key]=value.clone();}}
        }
    }
    if !files.is_empty(){
        if acquisition["transferred_bytes"].is_null(){acquisition["transferred_bytes"]=json!(files.values().fold(0_u64,|sum,v|sum.saturating_add(v["bytes"].as_u64().unwrap_or(0))));}
        if acquisition["files_completed"].is_null(){acquisition["files_completed"]=json!(files.values().filter(|v|v["completed"]==true).count());}
        let expected=files.values().try_fold(0_u64,|sum,v|sum.checked_add(v["expected_bytes"].as_u64()?));
        acquisition["expected_bytes"]=json!(expected.filter(|_|acquisition["files_total"].as_u64()==Some(files.len() as u64)));
    }
    if stage=="fetch"{if let Some(value)=acquisition["phase"].as_str(){phase=value.to_owned();}}else if matches!(stage,"prepare"|"initialize"){
        if let Some(value)=preparation["label"].as_str().or_else(||preparation["phase"].as_str()){phase=value.to_owned();}
    }
    let now=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis().min(i64::MAX as u128) as i64;let now=ended.map_or(now,|end|end.min(now));
    let mut value=json!({"schema":"arwen.pipeline-progress.v1","stage":stage,"phase":phase,"state":Value::Null,"started_unix_ms":began,"updated_unix_ms":updated,"wall_seconds":finished.unwrap_or_else(||((now-began) as f64/1000.).max(0.))});
    if acquisition.as_object().is_some_and(|v|!v.is_empty()){value["acquisition"]=acquisition;}if !preparation.is_null(){value["preparation"]=preparation;}value
}
fn table_number(table:&toml_edit::Table,key:&str)->Option<f64>{table.get(key).and_then(|v|v.as_float().or_else(||v.as_integer().map(|n|n as f64))).filter(|v|v.is_finite()&&*v>=0.)}
fn planned(current:Option<f64>,interval:f64,total:f64)->Option<f64>{let now=current?;if interval<=0.||now>=total{return None;}let next=((now/interval+1e-9).floor()+1.)*interval;(next<=total+1e-7).then_some(next)}
fn summarize(config:&[u8],config_sha:&str,manifest:&Value,manifest_bytes:&[u8],events:&[Value],heartbeat:Option<&Value>,root:&Path)->Result<Value,String>{
    let doc=std::str::from_utf8(config).map_err(|_|"Saved configuration is not UTF-8")?.parse::<toml_edit::DocumentMut>().map_err(|_|"Saved configuration is not valid TOML")?;
    let experiment=doc.get("experiment").and_then(|v|v.as_table()).ok_or("Saved configuration has no experiment")?;
    let start=experiment.get("start_time").ok_or("Saved configuration has no start time")?;
    let start=utc_ms(&start.as_str().map(str::to_owned).or_else(||start.as_datetime().map(ToString::to_string)).ok_or("Invalid saved start time")?)?;
    let total=table_number(experiment,"run_seconds").filter(|v|*v>0.).ok_or("Saved configuration has no positive forecast duration")?;
    let restart=table_number(experiment,"restart_interval_s").ok_or("Saved configuration has no checkpoint policy")?;
    let tables=doc.get("domain").and_then(|v|v.as_array_of_tables()).filter(|v|!v.is_empty()&&v.len()<=999).ok_or("Saved configuration has no domain timing")?;
    let mut schedule=BTreeMap::new();for table in tables{
        let id=table.get("grid_id").and_then(|v|v.as_integer()).filter(|n|(1..=999).contains(n)).ok_or("Invalid saved domain ID")? as u64;
        let interval=table_number(table,"history_interval_s").filter(|v|*v>0.).ok_or("Invalid saved output interval")?;
        if schedule.insert(id,interval).is_some(){return Err("Duplicate saved domain ID".into());}
    }
    let mut result=json!({});let mut model=None;let mut outputs=BTreeMap::new();
    if let Some(h)=heartbeat{
        if let Some(elapsed)=seconds(&h["model_elapsed_seconds"]){result["model_elapsed_seconds"]=json!(elapsed);}
        if let Some(phase)=h["status"].as_str().filter(|p|p.len()<=160){result["stage"]=json!(phase);result["phase"]=json!(phase);result["phase_updated_unix_ms"]=json!(utc_ms(text(h,"updated_at_utc")?)?);}
    }
    for event in events{
        if let Some(phase)=event["phase"].as_str().filter(|p|!p.is_empty()&&p.len()<=160){if event["emitted_unix_ms"].as_i64()>=result["phase_updated_unix_ms"].as_i64(){result["phase"]=json!(phase);result["phase_updated_unix_ms"]=event["emitted_unix_ms"].clone();}}
        let tag=event["event"].as_str().unwrap_or("");
        if matches!(tag,"stage_started"|"stage_finished"|"failed"){if let Some(stage)=event["stage"].as_str().filter(|p|p.len()<=160){result["stage"]=json!(stage);}}
        if tag=="output_committed"{if let (Some(id),Some(valid))=(event["domain"].as_u64(),event["valid_time"].as_str()){
            let saved=(utc_ms(valid)?-start) as f64/1000.;if saved>=0.&&saved<=total{outputs.insert(id,saved);}result["valid_time"]=json!(valid);
        }}
        if tag=="model_progress"{if let Some(elapsed)=seconds(&event["model_seconds"]){
            if model.is_some_and(|old:&Value|seconds(&old["model_seconds"]).is_some_and(|old|old>elapsed)){return Err("Native model clock moved backward".into());}
            model=Some(event);result["model_elapsed_seconds"]=json!(elapsed.max(seconds(&result["model_elapsed_seconds"]).unwrap_or(0.)));
        }}
        if tag=="failed"{if let Some(message)=event["message"].as_str(){result["error"]=json!(message.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(1600).collect::<String>());}}
        let summary=event.get("render_summary").or_else(||event["summary"].get("render_summary"));
        if let Some(summary)=summary{if summary["schema"]=="gpuwm.render-summary.v1"&&serde_json::to_vec(summary).is_ok_and(|v|v.len()<=64*1024){result["render_summary"]=summary.clone();}}
        if tag=="completed"{result["stage"]=json!("completed");}
    }
    let empty=Value::Null;let model=model.unwrap_or(&empty);let hb=heartbeat.unwrap_or(&empty);
    let elapsed=seconds(&model["model_seconds"]).or_else(||seconds(&hb["model_elapsed_seconds"]));
    if elapsed.is_some_and(|v|v>total+1e-6){return Err("Native model clock exceeds this saved forecast duration".into());}
    let mut clocks=BTreeMap::new();if let Some(domains)=model["domains"].as_array(){for d in domains{if let(Some(id),Some(value))=(d["domain"].as_u64(),seconds(&d["model_seconds"]).filter(|v|*v<=total+1e-6)){clocks.insert(id,value);}}}
    if let(Some(id),Some(value))=(model["domain"].as_u64(),elapsed){clocks.insert(id,value);}
    let domains=schedule.iter().map(|(id,interval)|{
        let current=clocks.get(id).copied().or_else(||(schedule.len()==1).then_some(elapsed).flatten());let next=planned(current,*interval,total);
        json!({"grid_id":id,"model_seconds":current,"history_interval_s":interval,"last_save_model_seconds":outputs.get(id),"next_save_model_seconds":next,"next_save_in_seconds":next.zip(current).map(|(next,now)|(next-now).max(0.))})
    }).collect::<Vec<_>>();
    let mut checkpoint_saved=None;
    if let Some(path)=model["last_checkpoint"].as_str().or_else(||hb["last_checkpoint"].as_str()){
        if let Ok(path)=owned(path,root){if path.is_file(){if let Some(name)=path.file_name().and_then(|p|p.to_str()).filter(|p|p.is_ascii()){
            if let Some((id,instant))=name.strip_prefix("gpuwmrst_d").and_then(|s|s.split_once('_')){
                if id.parse::<u64>().ok()==schedule.keys().next().copied()&&instant.len()>=23&&instant.ends_with(".npz"){
                    let date=&instant[..19];let suffix=&instant[19..];if suffix==".npz"||suffix.starts_with("__"){
                        let stamp=date.replacen('_',"T",1).replace('_',":");if let Ok(ms)=utc_ms(&stamp){let saved=(ms-start) as f64/1000.;if saved>=0.&&elapsed.is_some_and(|now|saved<=now){checkpoint_saved=Some(saved);}}
                    }
                }
            }
        }}}
    }
    let next=planned(elapsed,restart,total);let valid=elapsed.map(|value|start.checked_add((value*1000.).round() as i64).ok_or_else(||"Forecast UTC overflow".to_string()).and_then(utc_text)).transpose()?;
    let updated=model["emitted_unix_ms"].as_i64().or_else(||hb["updated_at_utc"].as_str().and_then(|s|utc_ms(s).ok()));
    let positive=|key:&str|seconds(&model[key]).filter(|v|*v>0.);
    result["progress"]=json!({"schema":"arwen.forecast-progress.v1","outer_step":model["outer_step"].as_u64().or_else(||hb["outer_step"].as_u64()),"model_seconds":elapsed,"run_seconds":total,
        "wall_seconds":seconds(&model["wall_seconds"]),"speed_x":positive("speed_x"),"step_ms":positive("step_ms"),"valid_time":valid,"updated_unix_ms":updated,"phase":result["phase"].as_str().or_else(||result["stage"].as_str()),
        "domains":domains,"checkpoint":{"interval_seconds":restart,"last_saved_model_seconds":checkpoint_saved,"next_model_seconds":next,"in_seconds":next.zip(elapsed).map(|(next,now)|(next-now).max(0.))},
        "source":{"run_id":manifest["run_id"],"manifest_sha256":digest(manifest_bytes),"snapshot_config_sha256":config_sha,"event_sequence":model["sequence"]}});
    Ok(result)
}

#[cfg(test)]
mod tests{
    use super::*;
    fn write(path:&Path,value:&Value){fs::write(path,serde_json::to_vec_pretty(value).unwrap()).unwrap();}
    struct Fixture{root:PathBuf,job:PathBuf,config:PathBuf,plan:PathBuf,run:PathBuf,command:Vec<String>,expected:Value}
    fn fixture(hosted:bool)->Fixture{
        let path=std::env::temp_dir().join(format!("local-progress-{}",crate::remote::stamp()));fs::create_dir_all(&path).unwrap();let root=path.canonicalize().unwrap();
        let job=root.join("job");let run=root.join("run");fs::create_dir(&job).unwrap();fs::create_dir(&run).unwrap();
        let config=root.join("case.toml");fs::write(&config,"[experiment]\nstart_time=2013-05-20T00:00:00\nrun_seconds=21600\nrestart_interval_s=3600\n[[domain]]\ngrid_id=1\nhistory_interval_s=3600\n[[domain]]\ngrid_id=2\nhistory_interval_s=900\n[[domain]]\ngrid_id=3\nhistory_interval_s=900\n").unwrap();
        let sha=digest(&fs::read(&config).unwrap());let plan=root.join("plan.json");write(&plan,&json!({"schema":"gpuwm.run-plan.v1","config":{"path":config},"output_root":run}));
        let command=vec!["unused-fixture-python".into(),"-m".into(),"gpuwm.cli".into(),"run-plan".into(),plan.display().to_string(),"--execute".into()];
        write(&job.join("job.json"),&json!({"schema":"gpuwm-tui-job-v1","command":command,"cwd":root,"action":"run-plan"}));
        write(&job.join("process.json"),&json!({"schema":"gpuwm-tui-process-v1","pid":4242,"started_at":"2026-09-07T00:00:00Z","cwd":root,"cli_args":&command[3..]}));
        let receipt:Value=serde_json::from_str(include_str!("../tests/fixtures/native-progress-newcastle-2013.json")).unwrap();let expected=receipt["native_result"]["progress"].clone();
        let mut parent=json!({"schema":"gpuwm.run-manifest.v1","run_id":"local-parent","pid":4242,"started_at_utc":"2026-09-07T00:00:01Z","run_dir":run,"outputs_dir":run,"events_path":run.join("events.jsonl"),"plan_source":plan,"plan_sha256":digest(&fs::read(&plan).unwrap()),"route":if hosted{"prepared"}else{"experiment"}});
        let native=if hosted{let path=run.join("chain/run-20260907-000002Z_i201305200000Z");fs::create_dir_all(&path).unwrap();path}else{run.clone()};
        let mut manifest=parent.clone();if hosted{
            manifest["run_id"]=json!("local-native");manifest["started_at_utc"]=json!("2026-09-07T00:00:02Z");manifest["run_dir"]=json!(native);manifest["outputs_dir"]=json!(native);manifest["events_path"]=json!(native.join("events.jsonl"));manifest["plan_source"]=json!(format!("gpuwm go {}",config.display()));manifest["plan_sha256"]=json!("d".repeat(64));manifest["route"]=json!("experiment");
        }
        let started=utc_ms(manifest["started_at_utc"].as_str().unwrap()).unwrap();
        let checkpoint=native.join("gpuwmrst_d01_2013-05-20_06_00_00__fixture-set.npz");fs::write(&checkpoint,b"metadata fixture only; never read as a checkpoint").unwrap();
        let resolved=json!({"schema_version":"gpuwm.run-plan.event.v1","sequence":1,"event":"resolved_plan","emitted_unix_ms":started+1,"config_source":config,"config_sha256":sha});
        let mut stream=vec![resolved.clone()];
        for id in 1..=3{stream.push(json!({"schema_version":"gpuwm.run-plan.event.v1","sequence":id+1,"event":"output_committed","emitted_unix_ms":started+5000,"domain":id,"valid_time":"2013-05-20T06:00:00","path":native.join(format!("wrfout_d{id:02}_not_opened"))}));}
        let mut model=json!({"schema_version":"gpuwm.run-plan.event.v1","sequence":5,"event":"model_progress","emitted_unix_ms":started+6000,"domain":1,"phase":"post-d01-sync","last_checkpoint":checkpoint});
        for key in ["outer_step","model_seconds","wall_seconds","speed_x","step_ms"]{model[key]=expected[key].clone();}
        model["domains"]=json!(expected["domains"].as_array().unwrap().iter().map(|d|json!({"domain":d["grid_id"],"model_seconds":d["model_seconds"]})).collect::<Vec<_>>());stream.push(model);
        stream.push(json!({"schema_version":"gpuwm.run-plan.event.v1","sequence":6,"event":"stage_started","emitted_unix_ms":started+7000,"stage":"finalize","phase":"render"}));
        fs::write(native.join("events.jsonl"),stream.iter().map(|e|serde_json::to_string(e).unwrap()+"\n").collect::<String>()).unwrap();
        write(&native.join("run-manifest.json"),&manifest);
        if hosted{
            parent["native_run"]=json!({"schema":"gpuwm.native-run-binding.v1","pid":4242,"config_source":config,"config_sha256":sha,"run_id":manifest["run_id"],"run_dir":native,"manifest_path":native.join("run-manifest.json"),"manifest_sha256":digest(&fs::read(native.join("run-manifest.json")).unwrap())});
            let mut first=resolved;first["emitted_unix_ms"]=json!(utc_ms("2026-09-07T00:00:01Z").unwrap()+1);fs::write(run.join("events.jsonl"),serde_json::to_string(&first).unwrap()+"\n").unwrap();write(&run.join("run-manifest.json"),&parent);
        }
        Fixture{root,job,config,plan,run,command,expected}
    }
    #[test]
    fn utc_dates_match_native_timestamps_with_offsets_and_historical_days(){
        assert_eq!(utc_ms("1970-01-01T00:00:00Z").unwrap(),0);assert_eq!(utc_ms("1969-12-31 23:59:59").unwrap(),-1000);
        assert_eq!(utc_ms("2026-09-08T03:20:51.630828Z").unwrap(),1788837651630);
        assert_eq!(utc_ms("2000-02-29T01:02:03+02:00").unwrap(),utc_ms("2000-02-28T23:02:03Z").unwrap());
        assert!(utc_ms("1900-02-29T00:00:00Z").is_err());
        for time in ["1950-01-01T12:00:00Z","2000-02-29T23:59:59Z","2026-09-08T03:20:51.630Z"]{assert_eq!(utc_text(utc_ms(time).unwrap()).unwrap(),time);}
    }
    #[test]
    fn local_plan_and_native_child_reuse_actual_native_timing_without_reading_weather(){
        for hosted in [false,true]{let f=fixture(hosted);let value=read(&f.job,&f.command).unwrap();let p=&value["progress"];
            for key in ["outer_step","model_seconds","run_seconds","wall_seconds","speed_x","step_ms","valid_time","domains","checkpoint"]{assert_eq!(p[key],f.expected[key],"{key}");}
            assert_eq!(value["source_config_sha256"],digest(&fs::read(&f.config).unwrap()));assert_eq!(value["source_config_path"],json!(f.config));assert_eq!(value["phase"],"render");assert_eq!(value["manifest_ready"],true);
            assert_eq!(Path::new(value["run_dir"].as_str().unwrap()).starts_with(&f.run),true);
            assert!(f.root.is_dir());
        }
    }
    #[test]
    fn changed_plan_config_process_and_child_manifest_are_refused(){
        let f=fixture(false);let original=fs::read(&f.plan).unwrap();let mut changed=original.clone();changed.extend_from_slice(b"\n");fs::write(&f.plan,changed).unwrap();assert!(read(&f.job,&f.command).unwrap_err().contains("reviewed local plan"));fs::write(&f.plan,original).unwrap();
        fs::write(&f.config,b"# externally changed saved configuration").unwrap();assert!(read(&f.job,&f.command).unwrap_err().contains("configuration receipt"));
        let f=fixture(false);let mut p=read_json(&f.job.join("process.json"),65536).unwrap();p["pid"]=json!(9999);write(&f.job.join("process.json"),&p);assert!(read(&f.job,&f.command).unwrap_err().contains("process and output"));
        let f=fixture(true);let path=f.run.join("chain/run-20260907-000002Z_i201305200000Z/run-manifest.json");let mut p=read_json(&path,65536).unwrap();p["name"]=json!("changed");write(&path,&p);assert!(read(&f.job,&f.command).unwrap_err().contains("changed after its parent"));
    }
    #[test]
    fn ordinary_go_accepts_canonical_path_spelling_and_publishes_the_same_job_summary(){
        let mut f=fixture(false);f.command=vec!["unused-fixture-python".into(),"-m".into(),"gpuwm.cli".into(),"go".into(),f.config.display().to_string(),"--outdir".into(),f.run.display().to_string()];
        let mut launcher=read_json(&f.job.join("job.json"),65536).unwrap();launcher["command"]=json!(f.command);write(&f.job.join("job.json"),&launcher);
        let mut process=read_json(&f.job.join("process.json"),65536).unwrap();process["cli_args"]=json!(&f.command[3..]);write(&f.job.join("process.json"),&process);
        let path=f.run.join("run-manifest.json");let mut m=read_json(&path,65536).unwrap();m["plan_source"]=json!(format!("gpuwm go {}",f.config.display()));write(&path,&m);
        let value=read(&f.job,&f.command).unwrap();assert_eq!(value["progress"]["outer_step"],360);assert_eq!(value["progress"]["speed_x"],45.5852);
    }
    #[test]
    fn pipeline_shows_real_acquisition_and_preparation_counts_before_integration(){
        let events=vec![json!({"event":"stage_started","stage":"fetch","emitted_unix_ms":1000}),json!({"event":"fetch_progress","emitted_unix_ms":2000,"acquisition":{"schema":"arwen.acquisition-progress.v1","phase":"cds_running","requests_completed":1,"requests_total":2,"files_total":2,"forcing_hours":3,"forcing_times_total":4}}),json!({"event":"fetch_completed","emitted_unix_ms":3000,"file":"part-0.grib","bytes":4194304,"failed":false}),json!({"event":"fetch_progress","emitted_unix_ms":4000,"file":"part-1.grib","bytes":2097152})];
        let p=pipeline_progress(&events,&json!({"stage":"fetch"}),0,Some(60000));assert_eq!(p["phase"],"cds_running");assert_eq!(p["wall_seconds"],59.);assert_eq!(p["acquisition"]["transferred_bytes"],6291456);assert_eq!(p["acquisition"]["files_completed"],1);assert!(p["acquisition"]["expected_bytes"].is_null());
        let mut events=events;events.push(json!({"event":"warning","emitted_unix_ms":5000,"code":"preparation_progress","preparation":{"schema":"gpuwm.prep-stage.v1","label":"Preparing geography","index":2,"count":3,"backend":"native"}}));
        let p=pipeline_progress(&events,&json!({"stage":"prepare"}),0,Some(60000));assert_eq!(p["phase"],"Preparing geography");assert_eq!(p["preparation"]["index"],2);assert_eq!(p["preparation"]["count"],3);
    }
}
