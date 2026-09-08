//! Opt-in process proof: real file IPC handlers, qualified Python/SSH RPCs,
//! native metadata and one bounded raw transfer. Never starts a forecast.
use super::*;
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{io::Read,path::Path,time::{Duration,Instant}};

fn submit(directory:&Path,session:&str,id:&str,action:&str,mut fields:Value){
    fields["schema"]=json!("arwen.companion-request.v1");fields["session_id"]=json!(session);fields["id"]=json!(id);fields["action"]=json!(action);
    fs::write(directory.join("requests").join(format!("{id}.json")),serde_json::to_vec(&fields).unwrap()).unwrap();
}
fn wait_response(app:&mut App,directory:&Path,id:&str)->Value{
    let path=directory.join("responses").join(format!("{id}.json"));let deadline=Instant::now()+Duration::from_secs(180);
    while Instant::now()<deadline{
        app.poll_companion_requests();app.poll_run_views();
        if path.is_file(){return companion::read_json(&path,2*1024*1024).unwrap();}
        std::thread::sleep(Duration::from_millis(20));
    }
    panic!("Timed out waiting for real IPC response {}",path.display());
}
fn hash_file(path:&Path)->String{
    let mut input=fs::File::open(path).unwrap();let mut hash=Sha256::new();let mut buffer=vec![0;1024*1024];
    loop{let count=input.read(&mut buffer).unwrap();if count==0{break;}hash.update(&buffer[..count]);}format!("{:x}",hash.finalize())
}

#[test]
#[ignore="explicit qualified Python, saved Node 1/2 profiles and completed Node 2 job; writes only a new local QA session/cache"]
fn real_node2_runs_ipc_preserves_node1_draft_and_target(){
    let profiles=PathBuf::from(env::var_os("ARWEN_RUN_VIEW_PROFILES").expect("ARWEN_RUN_VIEW_PROFILES"));
    let python=PathBuf::from(env::var_os("ARWEN_RUN_VIEW_PYTHON").expect("ARWEN_RUN_VIEW_PYTHON"));
    let node_id=env::var("ARWEN_RUN_VIEW_NODE_ID").expect("ARWEN_RUN_VIEW_NODE_ID");
    let expected_job=env::var("ARWEN_RUN_VIEW_JOB_ID").expect("ARWEN_RUN_VIEW_JOB_ID");
    let qa=PathBuf::from(env::var_os("ARWEN_RUN_VIEW_QA_DIR").expect("ARWEN_RUN_VIEW_QA_DIR"));
    assert!(qa.is_absolute()&&!qa.exists());fs::create_dir_all(&qa).unwrap();let qa=qa.canonicalize().unwrap();
    let original_profiles=fs::read(&profiles).unwrap();let copied_profiles=qa.join("nodes.json");fs::write(&copied_profiles,&original_profiles).unwrap();
    let draft=qa.join("main-node1-draft.toml");fs::write(&draft,"[experiment]\nname='Runs IPC draft isolation proof'\nstart_time=2000-01-01T00:00:00\nrun_seconds=3600\n").unwrap();
    let saved_draft=fs::read(&draft).unwrap();
    let mut app=App::new().unwrap();app.nodes=remote::Controller::load_path(copied_profiles.clone());
    assert!(app.nodes.load_error.is_none());
    let node=app.nodes.store.nodes.iter().find(|node|node.id==node_id).expect("saved viewer node").clone();
    let main_node=app.nodes.store.selected().expect("saved main target").clone();assert_ne!(main_node.id,node.id);
    let target=companion::Target::Ssh{node_id:node.id.clone(),connection_sha256:companion::digest(node.connection_key().as_bytes())};
    app.python=python;app.output=qa.join("local-runs");app.cwd=qa.clone();app.editor=Some(Editor::load(draft.clone()).unwrap());
    app.editor.as_mut().unwrap().insert("# Unsaved main draft remains on the other node.\n");assert!(app.dirty());
    app.companion.ensure_session(&app.output).unwrap();app.publish_companion_status(true);
    let parent=app.companion.session.as_ref().unwrap().directory.clone();let parent_id=app.companion.session.as_ref().unwrap().id.clone();
    let before=app.companion_context();let draft_text=app.editor.as_ref().unwrap().text();let node_state=app.nodes.store.nodes.clone();
    submit(&parent,&parent_id,"browse-real","browse_runs",json!({"target":target.value()}));
    let browse=wait_response(&mut app,&parent,"browse-real");assert_eq!(browse["ok"],true,"{browse}");
    assert!(browse["jobs"].as_array().unwrap().iter().any(|job|job["id"]==expected_job),"{browse}");
    submit(&parent,&parent_id,"open-real","open_run",json!({"target":target.value(),"job_id":expected_job}));
    let opened=wait_response(&mut app,&parent,"open-real");assert_eq!(opened["ok"],true,"{opened}");
    let handoff=&opened["handoff"];assert_eq!(handoff["schema"],"arwen.companion-handoff.v1");assert_eq!(handoff["read_only"],true);
    let viewer=PathBuf::from(handoff["control_dir"].as_str().unwrap());let viewer_id=handoff["session_id"].as_str().unwrap();assert_ne!(viewer_id,parent_id);
    assert_eq!(handoff["job_id"],expected_job);assert_ne!(handoff["config_path"],before["config_path"]);
    let first_status=companion::read_json(&viewer.join("status.json"),256*1024).unwrap();
    assert_eq!(first_status["job"]["job_id"],expected_job);assert_eq!(first_status["job"]["target"],target.value());

    submit(&viewer,viewer_id,"index-real","artifact_index",json!({"target":target.value(),"job_id":expected_job,"domain":1,"after_sequence":0}));
    let indexed=wait_response(&mut app,&viewer,"index-real");assert_eq!(indexed["ok"],true,"{indexed}");
    let index_path=PathBuf::from(indexed["artifact_index_path"].as_str().unwrap());assert_eq!(indexed["artifact_index_sha256"],hash_file(&index_path));
    let index=companion::read_json(&index_path,256*1024).unwrap();assert_eq!(index["job_id"],expected_job);assert_eq!(index["target"],target.value());
    let sequence=index["entries"].as_array().unwrap().last().unwrap()["sequence"].as_u64().unwrap();
    assert_eq!(index["run_id"],first_status["job"]["progress"]["source"]["run_id"]);
    assert_eq!(index["run_manifest"]["sha256"],first_status["job"]["progress"]["source"]["manifest_sha256"]);
    let native_manifest:Value=serde_json::from_str(index["run_manifest"]["utf8"].as_str().unwrap()).unwrap();assert_eq!(native_manifest["pid"],index["remote_pid"]);

    let raw=if env::var("ARWEN_RUN_VIEW_SYNC_RAW").as_deref()==Ok("1"){
        submit(&viewer,viewer_id,"frame-real","sync_artifacts",json!({"target":target.value(),"job_id":expected_job,"domain":1,"sequence":sequence,"reader_leases":true}));
        let synced=wait_response(&mut app,&viewer,"frame-real");assert_eq!(synced["ok"],true,"{synced}");
        let artifacts_path=PathBuf::from(synced["artifact_manifest_path"].as_str().unwrap());assert_eq!(synced["artifact_manifest_sha256"],hash_file(&artifacts_path));
        let artifacts=companion::read_json(&artifacts_path,256*1024).unwrap();let frame=&artifacts["frames"][0];
        assert_eq!(artifacts["job_id"],expected_job);assert_eq!(artifacts["remote_pid"],index["remote_pid"]);assert_eq!(artifacts["run_id"],index["run_id"]);
        assert_eq!(frame["domain"],1);assert_eq!(frame["commit"]["sequence"],sequence);
        let frame_path=PathBuf::from(frame["path"].as_str().unwrap());assert_eq!(frame["sha256"],hash_file(&frame_path));
        Some((artifacts_path,frame_path,frame.clone()))
    }else{None};

    // The normal one-second timeline cadence keeps running while the
    // independent two-second status request clock continues to advance.
    for poll in 0..5{
        let id=format!("fair-index-{poll}");submit(&viewer,viewer_id,&id,"artifact_index",json!({"target":target.value(),"job_id":expected_job,"domain":1,"after_sequence":sequence}));
        let answer=wait_response(&mut app,&viewer,&id);assert_eq!(answer["ok"],true,"{answer}");
        let until=Instant::now()+Duration::from_secs(1);while Instant::now()<until{app.poll_run_views();std::thread::sleep(Duration::from_millis(20));}
    }
    for (id,action,payload) in [("reject-stop","stop_job",json!({"target":target.value(),"job_id":expected_job})),("reject-select","select_target",json!({"target":target.value()})),
        ("reject-reset","reset_setup",json!({})),("reject-open","open_config",json!({"config_path":draft}))]{
        submit(&viewer,viewer_id,id,action,payload);let answer=wait_response(&mut app,&viewer,id);assert_eq!(answer["ok"],false,"{answer}");
    }
    let held_path=raw.as_ref().map(|(_,path,_)|path).unwrap_or(&index_path);
    let held_reader=fs::File::open(held_path).unwrap();
    submit(&viewer,viewer_id,"close-real","close_run",json!({"target":target.value(),"job_id":expected_job}));
    let closed=wait_response(&mut app,&viewer,"close-real");assert_eq!(closed["ok"],true,"{closed}");
    let until=Instant::now()+Duration::from_secs(10);
    while Instant::now()<until{app.poll_run_views();if companion::read_json(&viewer.join("status.json"),256*1024).unwrap()["state"]=="closed"{break;}std::thread::sleep(Duration::from_millis(20));}
    assert!(held_path.is_file()&&index_path.is_file());drop(held_reader);
    app.publish_companion_status(true);let after=app.companion_context();
    assert_eq!(before,after);assert_eq!(app.editor.as_ref().unwrap().text(),draft_text);assert!(app.dirty());assert!(app.job.is_none()&&app.nodes.pending.is_none());
    assert_eq!(app.nodes.store.nodes,node_state);assert_eq!(app.nodes.store.active.as_deref(),Some(main_node.id.as_str()));
    assert_eq!(fs::read(&profiles).unwrap(),original_profiles);assert_eq!(fs::read(&copied_profiles).unwrap(),original_profiles);assert_eq!(fs::read(&draft).unwrap(),saved_draft);
    let mut transports=Vec::new();
    for entry in fs::read_dir(app.output.join(".arwen-tui")).unwrap().filter_map(Result::ok){
        if entry.file_name().to_string_lossy().starts_with("remote-"){
            let launch=companion::read_json(&entry.path().join("job.json"),128*1024).unwrap();let command=launch["command"].as_array().unwrap();
            let action=command[4].as_str().unwrap();assert!(matches!(action,"list"|"status"|"artifact-index"|"sync-artifacts"),"{launch}");
            let host=command.iter().position(|value|value=="--host").unwrap()+1;assert_eq!(command[host],node.host);
            transports.push(json!({"directory":entry.path(),"action":action,"command":command}));
        }
    }
    let status_polls=transports.iter().filter(|row|row["action"]=="status").count();assert!(status_polls>=3,"status polls starved: {transports:?}");
    let report=json!({"schema":"arwen.runs-viewer-real-ipc-proof.v1","status":"PASS","process_id":std::process::id(),"qa_dir":qa,
        "main_target_before":before["target"],"main_target_after":after["target"],"main_config_path":before["config_path"],"main_unsaved_draft_sha256":companion::digest(draft_text.as_bytes()),
        "main_context_unchanged":before==after,"profile_bytes_unchanged":true,"saved_draft_unchanged":true,"viewer_target":target.value(),"viewer_job_id":expected_job,
        "viewer_handoff":handoff,"viewer_session":viewer,"browse_response":parent.join("responses/browse-real.json"),"open_response":parent.join("responses/open-real.json"),
        "artifact_index_path":index_path,"artifact_manifest_path":raw.as_ref().map(|(path,_,_)|path),"native_run_id":index["run_id"],"native_pid":index["remote_pid"],"native_manifest_sha256":index["run_manifest"]["sha256"],
        "raw_frame_path":raw.as_ref().map(|(_,path,_)|path),"raw_frame_sha256":raw.as_ref().map(|(_,_,frame)|&frame["sha256"]),"raw_frame_bytes":raw.as_ref().map(|(_,_,frame)|&frame["size_bytes"]),
        "valid_time":index["entries"].as_array().unwrap().last().unwrap()["valid_time"],"sequence":sequence,"metadata_only":raw.is_none(),"raw_transfer_requested":raw.is_some(),
        "status_requests_during_viewing":status_polls,"transports":transports,"viewer_close_retained_reader_files":true,
        "forecast_started":false,"gpu_probe":false,"node1_contacted":false,"original_run_write_actions_requested":false});
    fs::write(qa.join("PROOF.json"),serde_json::to_vec_pretty(&report).unwrap()).unwrap();println!("Real Runs viewer IPC proof: {}",qa.join("PROOF.json").display());
}
