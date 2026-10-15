//! Real Rust comparator and gpuwm-hex CLI on synthetic stored-word tapes.
//! These prove the instrument and refusals, never a forecast identity claim.
//!
//! They drive the gpuwm-hex CLI (`python3 -m hexcore.cli`, or MPAS_VERIFY_CLI) and write
//! scratch under MPAS_VERIFY_TEST_ROOT, neither of which an engine checkout has. So they are
//! ignored by name in a plain `cargo test` (the engine release's cargo tier, which panicked
//! on the missing variable before this) and run with hex installed:
//!   MPAS_VERIFY_TEST_ROOT=DIR cargo test -p rw-mpas --test verify_mpas_cli -- --ignored
//! The fixtures shard of tools/battery/cargo_gates.txt lists them.
//!
//! Unix only: they stage the native run's input links with std::os::unix::fs::symlink
//! and drive `python3`, as the native MPAS reference does. Breakage: 2.9.0's public CI
//! (run 38718383587, native windows-2025) could not compile this file, so the whole
//! rw-mpas cargo gate failed on Windows before any test ran.
#![cfg(unix)]
use rw_store::netcdf_classic::{NcClassicWriter,NcFormat,NcDim,NcVarDef,NcType,NcData};
use serde_json::{json,Value};
use std::path::{Path,PathBuf};
use std::process::Command;

fn reference(path: &Path) -> Value {
    json!({"path":path,"sha256":rw_mpas::sha256_file(path).unwrap()})
}
fn write_json(path: &Path,value: &Value) {
    std::fs::write(path,serde_json::to_vec_pretty(value).unwrap()).unwrap();
}
fn tape(path: &Path,delta: u32,omit: bool,water_leak: bool,records: u64) {
    let dims=vec![NcDim::record("Time"),NcDim::fixed("nCells",2),NcDim::fixed("nVertLevels",3),NcDim::fixed("StrLen",19)];
    let mut vars=vec![NcVarDef::new("xtime",NcType::Char,vec![0,3]),NcVarDef::new("areaCell",NcType::Double,vec![1]),
        NcVarDef::new("mask",NcType::Int,vec![0,1]),NcVarDef::new("double",NcType::Double,vec![0,1]),
        NcVarDef::new("padding",NcType::Float,vec![0,1])];
    for name in ["pressure","height","east","north","temperature"] {
        if !omit || name!="temperature" { vars.push(NcVarDef::new(name,NcType::Float,vec![0,1,2])); }
    }
    for name in ["t2m","q2","precip","waterStorage","waterSource","waterSink"] {
        vars.push(NcVarDef::new(name,NcType::Float,vec![0,1]));
    }
    let mut f=NcClassicWriter::create(path,NcFormat::Data64,dims,vec![],vars,records).unwrap();
    f.put("areaCell",NcData::Doubles(&[1.,2.])).unwrap();
    for r in 0..records {
        f.put_record_text("xtime",r,&format!("2000-01-01_{r:02}:00:00")).unwrap();
        f.put_record("mask",r,NcData::Ints(&[-1,16777217])).unwrap();
        f.put_record("double",r,NcData::Doubles(&[1.,-0.])).unwrap();
        f.put_record("padding",r,NcData::Floats(&[f32::from_bits(0x7fc00001),-0.])).unwrap();
        for (name,values) in [("pressure",vec![100000.,50000.,10000.,100000.,50000.,10000.]),
            ("height",vec![0.,5000.,15000.,0.,5000.,15000.]),("east",vec![5.,10.,20.,5.,10.,20.]),
            ("north",vec![1.,2.,3.,1.,2.,3.]),("temperature",vec![280.,270.,240.,280.,270.,240.])] {
            if omit && name=="temperature" { continue; }
            let mut values: Vec<f32>=values;
            if name=="temperature" && r==2 { values[0]=f32::from_bits(values[0].to_bits()+delta); }
            f.put_record(name,r,NcData::Floats(&values)).unwrap();
        }
        for (name,v) in [("t2m",280f32),("q2",0.001),("precip",0.),("waterStorage",10.),("waterSource",0.),("waterSink",0.)] {
            let mut v=v;
            if name=="waterStorage" && water_leak && r==2 { v+=1.; }
            f.put_record(name,r,NcData::Floats(&[v,v])).unwrap();
        }
    }
    f.finish().unwrap();
}
/// The comparator is the `verify` tool of rw_mpas_static. A copy under its
/// former name selects it, which is how a deck or a hex install that names
/// `rw_mpas_verify` keeps reaching it. A copy, not a link: the hex CLI
/// resolves the comparator path, and a resolved link names rw_mpas_static.
fn comparator(root: &Path) -> PathBuf {
    let bin=root.join("bin");
    let alias=bin.join(if cfg!(windows) {"rw_mpas_verify.exe"} else {"rw_mpas_verify"});
    if !alias.exists() {
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::copy(env!("CARGO_BIN_EXE_rw_mpas_static"),&alias).unwrap();
    }
    alias
}
fn invoke(deck: &Path,root: &Path,name: &str,baseline: Option<&Path>) -> (bool,Value) {
    invoke_mode(deck,root,name,baseline,true)
}
fn invoke_mode(deck: &Path,root: &Path,name: &str,baseline: Option<&Path>,comparison_only: bool) -> (bool,Value) {
    invoke_job_exit(deck,root,name,baseline,comparison_only,None)
}
fn invoke_job_exit(deck: &Path,root: &Path,name: &str,baseline: Option<&Path>,comparison_only: bool,failed_exit: Option<i32>) -> (bool,Value) {
    let work=root.join(format!("scratch-{name}"));
    let receipt=root.join(format!("{name}.json"));
    let cli=std::env::var("MPAS_VERIFY_CLI").ok();
    let mut command=Command::new(cli.clone().unwrap_or_else(||std::env::var("MPAS_VERIFY_PYTHON").unwrap_or_else(|_|"python3".into())));
    if cli.is_none() { command.args(["-m","hexcore.cli"]); }
    command.args(["verify-mpas","--deck"]).arg(deck)
        .arg("--receipt").arg(&receipt).arg("--work-root").arg(&work)
        .arg("--comparator").arg(comparator(root));
    if comparison_only { command.arg("--hex-output").arg(root.join("hex")); }
    if let Some(code)=failed_exit { command.arg("--failed-model-exit").arg(code.to_string()); }
    if let Some(path)=baseline { command.arg("--ratchet").arg(path); }
    let output=command.output().unwrap();
    assert!(receipt.is_file(),"CLI produced no receipt: {}",String::from_utf8_lossy(&output.stderr));
    let value: Value=serde_json::from_slice(&std::fs::read(receipt).unwrap()).unwrap();
    let label=if value["passed"]==true {value["acceptance_tier"].as_str().unwrap()} else {"REFUSED"};
    assert!(String::from_utf8_lossy(&output.stdout).contains(&format!("verify-mpas: {label};")),
        "console must report actual acceptance: {}",String::from_utf8_lossy(&output.stdout));
    (output.status.success(),value)
}

#[test]
#[ignore = "needs gpuwm-hex (hexcore.cli) and MPAS_VERIFY_TEST_ROOT; run with --ignored where hex is installed"]
fn real_door_identity_perturbation_missing_and_spread() {
    let parent=PathBuf::from(std::env::var("MPAS_VERIFY_TEST_ROOT").expect("keep scratch in lane: set MPAS_VERIFY_TEST_ROOT"));
    let root=parent.join(format!("verify-instrument-{}",std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    for dir in ["strict","hex","optimized"] { std::fs::create_dir(root.join(dir)).unwrap(); }
    for name in ["namelist","streams","init"] { std::fs::write(root.join(name),format!("synthetic {name}")).unwrap(); }
    let inputs=json!({"namelist":reference(&root.join("namelist")),"streams":reference(&root.join("streams")),"init":reference(&root.join("init"))});
    let hashes=json!({"namelist":inputs["namelist"]["sha256"],"streams":inputs["streams"]["sha256"],"init":inputs["init"]["sha256"]});
    for dir in ["strict","hex","optimized"] { tape(&root.join(dir).join("history.nc"),if dir=="optimized" {4} else {0},false,false,3); }
    let times=json!(["2000-01-01_00:00:00","2000-01-01_01:00:00","2000-01-01_02:00:00"]);
    for dir in ["strict","optimized"] {
        let manifest=json!({"schema":"mpas-native-run-v1","process_count":1,"pin_set":"synthetic-pins","inputs":hashes,
            "complete":true,"prognostic_fields":{"history":["temperature"]},"output_root":root.join(dir),"outputs":[{"stream":"history","path":"history.nc",
            "sha256":rw_mpas::sha256_file(&root.join(dir).join("history.nc")).unwrap(),"steps":[0,1,2],"times":times}]});
        write_json(&root.join(format!("{dir}.json")),&manifest);
    }
    let mut deck=json!({"schema":"mpas-verification-deck-v1","process_count":1,"pin_set":"synthetic-pins","mesh":"synthetic-mesh",
        "date":"synthetic-date","id":"instrument","inputs":inputs,"native":reference(&root.join("strict.json")),
        "optimized":reference(&root.join("optimized.json")),"declared_libm_residue":["synthetic deliberate temperature perturbation; NOT MODEL PROOF"],
        "duration_steps":2,"expected_samples":{"history":[0,1,2]},
        "diagnostics":{"stream":"history","area":reference(&root.join("strict/history.nc")),"area_field":"areaCell",
            "pressure":"pressure","height":"height","eastward_wind":"east","northward_wind":"north",
            "t2m":"t2m","q2":"q2","precipitation":["precip"],"prognostics":["temperature"],
            "water":{"storage":["waterStorage"],"cumulative_sources":["waterSource"],"cumulative_sinks":["waterSink"]}}});
    let path=root.join("deck.json");write_json(&path,&deck);
    let (ok,receipt)=invoke(&path,&root,"identical",None);
    assert!(ok,"{receipt}"); assert_eq!(receipt["tier"],"BITWISE");
    assert_eq!(receipt["evidence_kind"],"COMPARISON-ONLY");
    assert_eq!(receipt["process_count"],1);
    assert!(receipt["comparison"]["fields"].as_array().unwrap().iter().all(|r|r["differing_words"]==0));
    let baseline=root.join("baseline.json");write_json(&baseline,&receipt["candidate_ratchet"]);
    tape(&root.join("hex/history.nc"),1,false,false,3);
    let (ok,perturb)=invoke(&path,&root,"perturbed",Some(&baseline));
    assert!(!ok); assert_eq!(perturb["tier"],"ONE-STEP-IDENTICAL");
    assert!(perturb["acceptance_tier"].is_null());
    let (ok,diagnostic)=invoke(&path,&root,"diagnostic-only-spread",None);
    assert!(!ok);assert_eq!(diagnostic["tier"],"ONE-STEP-IDENTICAL");
    assert!(diagnostic["acceptance_tier"].is_null());
    assert!(!perturb["ratchet_regressions"].as_array().unwrap().is_empty());
    let row=perturb["candidate_ratchet"]["entries"].as_array().unwrap().iter().find(|r|r["field"]=="temperature").unwrap();
    assert_eq!(row["differing_words"],1);assert_eq!(row["max_ulp"],1);assert_eq!(row["first_differing_step"],2);
    tape(&root.join("hex/history.nc"),8,false,false,3);
    let (ok,outside)=invoke(&path,&root,"outside-spread",Some(&baseline));
    assert!(!ok);assert!(outside["tier"].is_null());
    tape(&root.join("hex/history.nc"),0,true,false,3);
    let (ok,missing)=invoke(&path,&root,"missing-field",Some(&baseline));
    assert!(!ok);assert!(missing["candidate_ratchet"]["entries"].as_array().unwrap().iter().any(|r|r["field"]=="temperature" && r["missing_samples"]==3));
    // The 07:45Z foundation policy admits this exact recorded missing field,
    // retains its words, and cannot enroll another gap or any computed error.
    let deferred=root.join("deferred.json");write_json(&deferred,&missing["candidate_ratchet"]);
    let (ok,recorded)=invoke(&path,&root,"recorded-missing",Some(&deferred));
    assert!(ok,"{recorded}");
    assert_eq!(recorded["acceptance_tier"],"BITWISE-WRITTEN-FIELDS");
    assert_eq!(recorded["all_native_fields_bitwise"],false);
    assert_eq!(recorded["evidence_kind"],"COMPARISON-ONLY"); // cannot pass the forecast gate
    assert_eq!(recorded["candidate_ratchet"],missing["candidate_ratchet"]);
    assert!(recorded["failures"].as_array().unwrap().is_empty());
    assert!(!recorded["diagnostic_failures"].as_array().unwrap().is_empty());
    tape(&root.join("hex/history.nc"),1,false,false,3);
    let (ok,computed)=invoke(&path,&root,"recorded-computed-error",Some(&deferred));
    assert!(!ok);assert!(computed["acceptance_tier"].is_null());
    tape(&root.join("hex/history.nc"),0,false,false,3);
    let (ok,fixed)=invoke(&path,&root,"recorded-fixed",Some(&deferred));
    assert!(ok);assert_eq!(fixed["acceptance_tier"],"BITWISE");
    let retired=root.join("retired.json");write_json(&retired,&fixed["candidate_ratchet"]);
    tape(&root.join("hex/history.nc"),0,true,false,3);
    let (ok,again)=invoke(&path,&root,"recorded-cannot-disappear",Some(&retired));
    assert!(!ok);assert!(!again["ratchet_regressions"].as_array().unwrap().is_empty());
    tape(&root.join("hex/history.nc"),0,false,false,2);
    let (ok,short)=invoke(&path,&root,"missing-time",Some(&baseline));
    assert!(!ok);assert!(short["comparison"]["fields"].as_array().unwrap().iter().any(|r|r["status"]=="output time missing in hex"));
    tape(&root.join("hex/history.nc"),0,false,true,3);
    let (ok,water)=invoke(&path,&root,"water-leak",Some(&baseline));
    assert!(!ok);assert!(!water["comparison"]["spread_samples"][2]["water_budget"]["passed"].as_bool().unwrap());
    std::fs::remove_file(root.join("hex/history.nc")).unwrap();
    let (ok,stream)=invoke(&path,&root,"missing-stream",Some(&baseline));
    assert!(!ok);assert!(stream["comparison"]["fields"].as_array().unwrap().iter().all(|r|r["status"]=="field missing in hex"));
    tape(&root.join("hex/history.nc"),0,false,false,3);
    // A field stream with a non-.nc suffix must still enter the ratchet.
    tape(&root.join("hex/unlisted.stream"),0,false,false,3);
    let (ok,extra)=invoke(&path,&root,"extra-stream",Some(&baseline));
    assert!(!ok);assert!(extra["comparison"]["fields"].as_array().unwrap().iter().any(|r|r["status"]=="extra output stream in hex"));
    std::fs::remove_file(root.join("hex/unlisted.stream")).unwrap();
    tape(&root.join("strict/unlisted.stream"),0,false,false,3);
    let (ok,inventory)=invoke(&path,&root,"incomplete-native-inventory",Some(&baseline));
    assert!(!ok);assert!(inventory["failures"][0].as_str().unwrap().contains("omits or invents"));
    std::fs::remove_file(root.join("strict/unlisted.stream")).unwrap();
    deck["diagnostics"]["prognostics"]=json!([]);write_json(&path,&deck);
    let (ok,prognostics)=invoke(&path,&root,"incomplete-prognostics",Some(&baseline));
    assert!(!ok);assert!(prognostics["failures"][0].as_str().unwrap().contains("entire prognostic inventory"));
    deck["diagnostics"]["prognostics"]=json!(["temperature"]);write_json(&path,&deck);
    // Input provenance and native inventory cannot be made to pass by a
    // manifest which simply forgets the file being measured.
    std::fs::write(root.join("streams"),"changed stream field list").unwrap();
    let (ok,changed)=invoke(&path,&root,"changed-input",Some(&baseline));
    assert!(!ok);assert_eq!(changed["evidence_kind"],"REFUSAL");
    deck["inputs"]["streams"]=reference(&root.join("streams"));write_json(&path,&deck);
    let (ok,mismatch)=invoke(&path,&root,"native-input-mismatch",Some(&baseline));
    assert!(!ok);assert!(mismatch["failures"][0].as_str().unwrap().contains("bytes differ"));
    let evidence=parent.join("receipts");std::fs::create_dir_all(&evidence).unwrap();
    for name in ["identical","perturbed","diagnostic-only-spread","outside-spread","missing-field","recorded-missing","recorded-computed-error","recorded-fixed","recorded-cannot-disappear","missing-time","water-leak","missing-stream","changed-input","native-input-mismatch","extra-stream","incomplete-native-inventory","incomplete-prognostics"] {
        std::fs::copy(root.join(format!("{name}.json")),evidence.join(format!("instrument-{name}.json"))).unwrap();
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
#[ignore = "needs gpuwm-hex (hexcore.cli) and MPAS_VERIFY_TEST_ROOT; run with --ignored where hex is installed"]
fn published_f0_cache_uses_sealed_inventory_and_actual_timestamps() {
    let parent=PathBuf::from(std::env::var("MPAS_VERIFY_TEST_ROOT").unwrap());
    let root=parent.join(format!("verify-publication-{}",std::process::id()));
    std::fs::create_dir_all(root.join("strict")).unwrap();
    std::fs::create_dir(root.join("hex")).unwrap();
    std::fs::write(root.join("namelist"),"synthetic publication namelist").unwrap();
    std::fs::write(root.join("init"),"synthetic publication init").unwrap();
    std::fs::write(root.join("streams"),r#"<streams><stream name="history" type="output" filename_template="history.nc"><file name="stream_list.history"/></stream></streams>"#).unwrap();
    std::fs::write(root.join("stream_list.history"),"temperature\n").unwrap();
    for name in ["strict","hex"] { tape(&root.join(name).join("history.nc"),0,false,false,3); }
    let inputs=json!({"namelist":reference(&root.join("namelist")),"streams":reference(&root.join("streams")),"init":reference(&root.join("init"))});
    let mut cache=json!({"schema":"mpas.native-forecast-cache/v1","ranks":1,"pin_set_hash":"synthetic-publication-pins",
        "namelist_sha256":inputs["namelist"]["sha256"],"streams_sha256":inputs["streams"]["sha256"],
        "inputs":{"init":inputs["init"]["sha256"],"stream_list.history":reference(&root.join("stream_list.history"))["sha256"]},
        "runtime":{"start":"2000-01-01_00:00:00","dt":3600},
        "outputs":{"history.nc":reference(&root.join("strict/history.nc"))["sha256"]}});
    write_json(&root.join("strict/MANIFEST.json"),&cache);
    std::fs::write(root.join("strict/log.atmosphere.0000.out"),"Finished running the atmosphere core").unwrap();
    let mut publication=json!({"schema":"mpas.native-cache-publication/v1","files":{
        "MANIFEST.json":reference(&root.join("strict/MANIFEST.json"))["sha256"],
        "history.nc":reference(&root.join("strict/history.nc"))["sha256"],
        "log.atmosphere.0000.out":reference(&root.join("strict/log.atmosphere.0000.out"))["sha256"]}});
    let publication_path=root.join("strict/PUBLICATION.json");
    write_json(&publication_path,&publication);
    let mut deck=json!({"schema":"mpas-verification-deck-v1","process_count":1,"pin_set":"synthetic-publication-pins",
        "mesh":"instrument-mesh","date":"instrument-date","id":"published-instrument", "inputs":inputs,
        "native":reference(&publication_path),"auxiliary_inputs":{"stream_list.history":reference(&root.join("stream_list.history"))}});
    let path=root.join("deck.json");write_json(&path,&deck);
    let (ok,receipt)=invoke(&path,&root,"published",None);
    assert!(ok,"{receipt}"); assert_eq!(receipt["tier"],"BITWISE");
    assert_eq!(receipt["evidence_kind"],"COMPARISON-ONLY");
    assert_eq!(receipt["process_count"],1);
    assert_eq!(receipt["comparison"]["fields"].as_array().unwrap().iter().map(|r|r["step"].as_u64().unwrap()).max(),Some(2));
    assert!(receipt["provenance"]["native_cache_manifest_sha256"].is_string());
    // Native stages actual netCDF input symlinks beside its outputs. Only
    // pinned input bytes may be excluded; no reference output disappears.
    tape(&root.join("grid.nc"),0,false,false,1);
    std::os::unix::fs::symlink(root.join("grid.nc"),root.join("strict/grid.nc")).unwrap();
    deck["auxiliary_inputs"]["grid.nc"]=reference(&root.join("grid.nc"));
    cache["inputs"]["grid.nc"]=reference(&root.join("grid.nc"))["sha256"].clone();
    write_json(&root.join("strict/MANIFEST.json"),&cache);
    publication["files"]["MANIFEST.json"]=reference(&root.join("strict/MANIFEST.json"))["sha256"].clone();
    write_json(&publication_path,&publication);
    deck["native"]=reference(&publication_path);write_json(&path,&deck);
    let (ok,staged_input)=invoke(&path,&root,"published-auxiliary-input-inventory",None);
    assert!(ok,"{staged_input}");assert_eq!(staged_input["tier"],"BITWISE");
    // f0 publishes source-bound field inventories separately from immutable
    // forecast caches. Bind their native bytes, rank and Registry before use.
    std::fs::create_dir(root.join("field-inventory")).unwrap();
    std::fs::write(root.join("Registry.xml"),"instrument Registry source").unwrap();
    let mut names: Vec<_> = receipt["comparison"]["fields"].as_array().unwrap().iter()
        .map(|r|r["field"].as_str().unwrap().to_owned()).collect();
    names.sort();names.dedup();
    let mut field_inventory=json!({"schema":"mpas.native-source-field-inventory/v1","ranks":1,
        "native_publication_sha256":reference(&publication_path)["sha256"],
        "native_manifest_sha256":reference(&root.join("strict/MANIFEST.json"))["sha256"],
        "source_sha256":{"src/core_atmosphere/Registry.xml":reference(&root.join("Registry.xml"))["sha256"]},
        "examples":{"history":{"file":"history.nc","sha256":reference(&root.join("strict/history.nc"))["sha256"]}},
        "streams":{"history":names.iter().map(|n|json!({"field":n})).collect::<Vec<_>>()},
        "prognostic_fields":{"history":["temperature"]},"water_integral_complete":false,
        "water_integral_limitation":"instrument streams have no complete native water integral"});
    let inventory_path=root.join("field-inventory/MANIFEST.json");
    let inventory_publication=root.join("field-inventory/PUBLICATION.json");
    let seal_inventory=|data: &Value, deck: &mut Value| {
        write_json(&inventory_path,data);
        write_json(&inventory_publication,&json!({"schema":"mpas.native-cache-publication/v1",
            "files":{"MANIFEST.json":reference(&inventory_path)["sha256"]}}));
        deck["native_field_inventory"]=reference(&inventory_publication);
        deck["native_field_inventory"]["registry"]=reference(&root.join("Registry.xml"));
        write_json(&path,deck);
    };
    seal_inventory(&field_inventory,&mut deck);
    let (ok,bound)=invoke(&path,&root,"published-source-inventory",None);
    assert!(ok,"{bound}");
    assert_eq!(bound["provenance"]["native_field_inventory"]["prognostic_fields"]["history"],json!(["temperature"]));
    assert_eq!(bound["provenance"]["native_field_inventory"]["water_integral_complete"],false);
    field_inventory["native_publication_sha256"]=json!("wrong-native-publication");
    seal_inventory(&field_inventory,&mut deck);
    let (ok,wrong_native)=invoke(&path,&root,"published-source-wrong-native",None);
    assert!(!ok);assert!(wrong_native["failures"][0].as_str().unwrap().contains("another native forecast"));
    field_inventory["native_publication_sha256"]=reference(&publication_path)["sha256"].clone();
    field_inventory["streams"]["history"].as_array_mut().unwrap().push(json!({"field":"invented-native-field"}));
    seal_inventory(&field_inventory,&mut deck);
    let (ok,wrong_fields)=invoke(&path,&root,"published-source-wrong-fields",None);
    assert!(!ok);assert!(wrong_fields["failures"][0].as_str().unwrap().contains("stored native fields"));
    deck.as_object_mut().unwrap().remove("native_field_inventory");
    write_json(&path,&deck);
    let (ok,timeout_full)=invoke_job_exit(&path,&root,"published-outer-timeout-full",None,true,Some(124));
    assert!(!ok);assert_eq!(timeout_full["evidence_kind"],"FORECAST-FAILED");
    assert_eq!(timeout_full["failed_model_exit"],124);
    assert_eq!(timeout_full["tier"],"BITWISE");
    assert!(timeout_full["comparison"]["fields"].as_array().unwrap().iter().all(|r|r["differing_words"]==0));
    std::fs::remove_file(root.join("hex/history.nc")).unwrap();
    let (ok,timeout_empty)=invoke_job_exit(&path,&root,"published-outer-timeout-empty",None,true,Some(124));
    assert!(!ok);assert_eq!(timeout_empty["evidence_kind"],"FORECAST-FAILED");
    assert!(timeout_empty["candidate_ratchet"]["entries"].as_array().unwrap().iter().all(|r|r["missing_samples"].as_u64().unwrap()>0));
    // A killed writer can leave a recognized but truncated netCDF header.
    // Its native words still count, including in the directory inventory.
    std::fs::write(root.join("hex/history.nc"),b"CDF\x01\x00\x00\x00\x00").unwrap();
    let (ok,corrupt)=invoke(&path,&root,"published-corrupt-output",None);
    assert!(!ok);assert!(corrupt["comparison"]["fields"].as_array().unwrap().iter().all(|r|r["status"].as_str().unwrap().starts_with("stream unreadable in hex")));
    assert_eq!(corrupt["comparison"]["fields"].as_array().unwrap().iter().map(|r|r["compared_words"].as_u64().unwrap()).sum::<u64>(),
        timeout_empty["comparison"]["fields"].as_array().unwrap().iter().map(|r|r["differing_words"].as_u64().unwrap()).sum::<u64>());
    tape(&root.join("hex/history.nc"),0,false,false,3);
    let auxiliary=deck.as_object_mut().unwrap().remove("auxiliary_inputs").unwrap();
    write_json(&path,&deck);
    let (ok,unbound)=invoke(&path,&root,"published-unbound-stream-list",None);
    assert!(!ok);assert!(unbound["failures"][0].as_str().unwrap().contains("field list is not pinned"));
    deck["auxiliary_inputs"]=auxiliary;
    // Deliberate unknown argparse flag guarantees this real subprocess stays
    // CPU-only. It witnesses receipt/log preservation and fresh input cwd.
    deck["hex_command"]=json!(["gpuwm-hex","forecast","--verification-instrument-parser-refusal",
        "--physics-namelist","{namelist}","--streams","{streams}","--init","{init}","--out","{out}"]);
    write_json(&path,&deck);
    let (ok,door)=invoke_mode(&path,&root,"published-door-refusal",None,false);
    assert!(!ok);assert!(door["failures"][0].as_str().unwrap().contains("hex model door failed"));
    assert_eq!(door["evidence_kind"],"FORECAST-FAILED");
    assert_eq!(door["process_count"],1);
    assert!(door["candidate_ratchet"]["entries"].as_array().unwrap().iter().all(|r|r["missing_samples"].as_u64().unwrap()>0));
    assert!(root.join("scratch-published-door-refusal/hex.log").is_file());
    assert!(root.join("scratch-published-door-refusal/run-inputs/stream_list.history").is_symlink());
    assert!(!root.join("scratch-published-door-refusal/hex-output").exists());
    // Re-hash the source manifest and publication deliberately, so a wrong
    // clock is refused based on the Rust-read tape rather than a stale hash.
    let mut offclock=cache.clone();offclock["runtime"]["dt"]=json!(7200);
    write_json(&root.join("strict/MANIFEST.json"),&offclock);
    publication["files"]["MANIFEST.json"]=reference(&root.join("strict/MANIFEST.json"))["sha256"].clone();
    write_json(&publication_path,&publication);
    deck["native"]=reference(&publication_path);write_json(&path,&deck);
    let (ok,clock)=invoke(&path,&root,"published-offclock",None);
    assert!(!ok);assert!(clock["failures"][0].as_str().unwrap().contains("off its timestep clock"));
    let mut bad_dt=cache.clone();bad_dt["runtime"]["dt"]=json!("invalid");
    write_json(&root.join("strict/MANIFEST.json"),&bad_dt);
    publication["files"]["MANIFEST.json"]=reference(&root.join("strict/MANIFEST.json"))["sha256"].clone();
    write_json(&publication_path,&publication);
    deck["native"]=reference(&publication_path);write_json(&path,&deck);
    let (ok,invalid_clock)=invoke(&path,&root,"published-invalid-clock",None);
    assert!(!ok);assert_eq!(invalid_clock["evidence_kind"],"REFUSAL");
    std::fs::write(root.join("streams"),"<streams>").unwrap();
    let mut bad_xml=cache.clone();bad_xml["streams_sha256"]=reference(&root.join("streams"))["sha256"].clone();
    write_json(&root.join("strict/MANIFEST.json"),&bad_xml);
    publication["files"]["MANIFEST.json"]=reference(&root.join("strict/MANIFEST.json"))["sha256"].clone();
    write_json(&publication_path,&publication);
    deck["native"]=reference(&publication_path);deck["inputs"]["streams"]=reference(&root.join("streams"));write_json(&path,&deck);
    let (ok,invalid_stream)=invoke(&path,&root,"published-invalid-stream-xml",None);
    assert!(!ok);assert_eq!(invalid_stream["evidence_kind"],"REFUSAL");
    std::fs::write(root.join("strict/log.atmosphere.0000.out"),"native aborted").unwrap();
    publication["files"]["log.atmosphere.0000.out"]=reference(&root.join("strict/log.atmosphere.0000.out"))["sha256"].clone();
    write_json(&publication_path,&publication);
    deck["native"]=reference(&publication_path);write_json(&path,&deck);
    let (ok,unfinished)=invoke(&path,&root,"published-unfinished",None);
    assert!(!ok);assert!(unfinished["failures"][0].as_str().unwrap().contains("completion witness"));
    std::fs::create_dir_all(parent.join("receipts")).unwrap();
    for name in ["published","published-offclock","published-unfinished","published-unbound-stream-list",
                 "published-door-refusal","published-invalid-clock","published-invalid-stream-xml","published-auxiliary-input-inventory",
                 "published-source-inventory","published-source-wrong-native","published-source-wrong-fields",
                 "published-outer-timeout-full","published-outer-timeout-empty","published-corrupt-output"] {
        std::fs::copy(root.join(format!("{name}.json")),parent.join("receipts").join(format!("instrument-{name}.json"))).unwrap();
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
#[ignore = "needs gpuwm-hex (hexcore.cli) and MPAS_VERIFY_TEST_ROOT; run with --ignored where hex is installed"]
fn surface_without_xtime_uses_sealed_filename_and_record_clock() {
    let parent=PathBuf::from(std::env::var("MPAS_VERIFY_TEST_ROOT").unwrap());
    let root=parent.join(format!("verify-surface-{}",std::process::id()));
    for dir in ["strict","hex"] { std::fs::create_dir_all(root.join(dir)).unwrap(); }
    for name in ["namelist","init"] { std::fs::write(root.join(name),"instrument input").unwrap(); }
    let template="surface.$Y-$M-$D_$h.$m.$s.nc";
    let filename="surface.2000-01-01_23.00.00.nc";
    std::fs::write(root.join("streams"),format!(r#"<streams><stream name="surface" type="input;output" filename_template="{template}" output_interval="0_01:00:00" filename_interval="none"/></streams>"#)).unwrap();
    let surface=|path: &Path, delta: bool, records: u64| {
        let mut f=NcClassicWriter::create(path,NcFormat::Data64,
            vec![NcDim::record("Time"),NcDim::fixed("nCells",2)],vec![],
            vec![NcVarDef::new("sst",NcType::Float,vec![0,1]),NcVarDef::new("landmask",NcType::Int,vec![1])],records).unwrap();
        f.put("landmask",NcData::Ints(&[1,0])).unwrap();
        for r in 0..records { let v=if delta && r==2 {f32::from_bits(280f32.to_bits()+1)} else {280.};
            f.put_record("sst",r,NcData::Floats(&[v,281.])).unwrap(); }
        f.finish().unwrap();
    };
    surface(&root.join("strict").join(filename),false,3);
    surface(&root.join("hex").join(filename),false,3);
    let inputs=json!({"namelist":reference(&root.join("namelist")),"streams":reference(&root.join("streams")),"init":reference(&root.join("init"))});
    let cache=json!({"schema":"mpas.native-forecast-cache/v1","ranks":1,"pin_set_hash":"instrument-surface-pins",
        "namelist_sha256":inputs["namelist"]["sha256"],"streams_sha256":inputs["streams"]["sha256"],
        "inputs":{"init":inputs["init"]["sha256"]},"runtime":{"start":"2000-01-01_23:00:00","dt":3600},
        "outputs":{filename:reference(&root.join("strict").join(filename))["sha256"]}});
    write_json(&root.join("strict/MANIFEST.json"),&cache);
    std::fs::write(root.join("strict/log.atmosphere.0000.out"),"Finished running the atmosphere core").unwrap();
    let publication=json!({"schema":"mpas.native-cache-publication/v1","files":{
        "MANIFEST.json":reference(&root.join("strict/MANIFEST.json"))["sha256"],
        filename:reference(&root.join("strict").join(filename))["sha256"],
        "log.atmosphere.0000.out":reference(&root.join("strict/log.atmosphere.0000.out"))["sha256"]}});
    write_json(&root.join("strict/PUBLICATION.json"),&publication);
    let deck=json!({"schema":"mpas-verification-deck-v1","process_count":1,"pin_set":"instrument-surface-pins","mesh":"instrument-mesh",
        "date":"instrument-date","id":"surface-instrument","inputs":inputs,"native":reference(&root.join("strict/PUBLICATION.json"))});
    let path=root.join("deck.json");write_json(&path,&deck);
    let (ok,receipt)=invoke(&path,&root,"surface-clock",None);
    assert!(ok,"{receipt}");assert_eq!(receipt["tier"],"BITWISE");
    assert_eq!(receipt["comparison"]["fields"].as_array().unwrap().len(),4);
    assert_eq!(receipt["provenance"]["native_stream_clocks"][0]["times"][2],"2000-01-02_01:00:00");
    surface(&root.join("hex").join(filename),true,3);
    let (ok,changed)=invoke(&path,&root,"surface-perturbed",None);
    assert!(!ok);assert_eq!(changed["candidate_ratchet"]["entries"].as_array().unwrap().iter().find(|e|e["field"]=="sst").unwrap()["first_differing_step"],2);
    surface(&root.join("hex").join(filename),false,2);
    let (ok,missing)=invoke(&path,&root,"surface-missing-time",None);
    assert!(!ok);assert!(missing["comparison"]["fields"].as_array().unwrap().iter().any(|r|r["status"]=="output time missing in hex"));
    // A request cannot relabel a no-xtime native file arbitrarily.
    let request=root.join("scratch-surface-clock/compare-request.json");
    let mut data: Value=serde_json::from_slice(&std::fs::read(&request).unwrap()).unwrap();
    data["samples"][0]["times"][0]=json!("2000-01-02_23:00:00");write_json(&request,&data);
    let result=Command::new(env!("CARGO_BIN_EXE_rw_mpas_static")).args(["verify","--request",request.to_str().unwrap(),"--out",root.join("wrong-clock.json").to_str().unwrap()]).output().unwrap();
    assert!(!result.status.success());assert!(String::from_utf8_lossy(&result.stderr).contains("filename differs"));
    std::fs::create_dir_all(parent.join("receipts")).unwrap();
    for name in ["surface-clock","surface-perturbed","surface-missing-time"] {
        std::fs::copy(root.join(format!("{name}.json")),parent.join("receipts").join(format!("instrument-{name}.json"))).unwrap();
    }
    std::fs::remove_dir_all(root).unwrap();
}
