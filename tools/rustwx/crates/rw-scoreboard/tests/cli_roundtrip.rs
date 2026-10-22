use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};
// Parallel tests on Windows can read the same SystemTime nanoseconds; the counter keeps their scratch folders apart.
static NEXT_SCRATCH: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
fn scratch() -> PathBuf {
    let base = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    fs::create_dir_all(&base).unwrap();
    let p = base.join(format!(
        "rw-scoreboard-test-{}-{}-{}",
        std::process::id(),
        NEXT_SCRATCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&p).unwrap();
    p
}
fn frame(
    path: &Path,
    time: &str,
    temperature: f32,
    rain: f32,
    start_metadata: bool,
    elapsed_minutes: f32,
) {
    let mut schema = Schema::new(NcFormat::Offset64);
    let t = schema.def_dim("Time", 0, true).unwrap();
    let strlen = schema.def_dim("DateStrLen", 19, false).unwrap();
    let y = schema.def_dim("south_north", 2, false).unwrap();
    let x = schema.def_dim("west_east", 2, false).unwrap();
    let z = schema.def_dim("bottom_top", 2, false).unwrap();
    schema
        .put_global_attr("MAP_PROJ", AttrValue::Ints(vec![1]))
        .unwrap();
    if start_metadata {
        schema
            .put_global_attr(
                "SIMULATION_START_DATE",
                AttrValue::Text("2026-10-01_00:00:00".into()),
            )
            .unwrap();
    }
    let times = schema.def_var("Times", NcType::Char, &[t, strlen]).unwrap();
    let xtime = schema.def_var("XTIME", NcType::Float, &[t]).unwrap();
    schema
        .put_var_attr(
            xtime,
            "units",
            AttrValue::Text("minutes since 2026-10-01 00:00:00".into()),
        )
        .unwrap();
    let fields = [
        ("XLAT", vec![0., 0., 1., 1.]),
        ("XLONG", vec![0., 1., 0., 1.]),
        ("T2", vec![temperature; 4]),
        ("Q2", vec![0.01; 4]),
        ("PSFC", vec![100000.; 4]),
        ("HGT", vec![100.; 4]),
        ("U10", vec![10.; 4]),
        ("V10", vec![0.; 4]),
        ("COSALPHA", vec![0.; 4]),
        ("SINALPHA", vec![1.; 4]),
        ("RAINNC", vec![rain; 4]),
        ("RAINC", vec![1.; 4]),
    ];
    let vars: Vec<_> = fields
        .iter()
        .map(|(name, _)| schema.def_var(name, NcType::Float, &[t, y, x]).unwrap())
        .collect();
    let refl = schema
        .def_var("REFL_10CM", NcType::Float, &[t, z, y, x])
        .unwrap();
    let mut writer = NcWriter::create(path, schema).unwrap();
    writer
        .write_record(0, times, VarData::Char(time.as_bytes()))
        .unwrap();
    writer
        .write_record(0, xtime, VarData::F32(&[elapsed_minutes]))
        .unwrap();
    for ((_, values), var) in fields.iter().zip(vars) {
        writer.write_record(0, var, VarData::F32(values)).unwrap();
    }
    writer
        .write_record(
            0,
            refl,
            VarData::F32(&[10., 10., 10., 10., 20., 20., 20., 20.]),
        )
        .unwrap();
    writer.finish().unwrap();
}
fn call(root: &Path, name: &str, request: Value) -> Value {
    let path = root.join(format!("{name}.request.json"));
    fs::write(&path, serde_json::to_vec(&request).unwrap()).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_rw_scoreboard"))
        .args(["--request", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).unwrap()
}
fn clean(root: &Path) {
    for entry in fs::read_dir(root).unwrap() {
        let entry = entry.unwrap();
        eprintln!(
            "deleted_test_fixture {} {} bytes",
            entry.file_name().to_string_lossy(),
            entry.metadata().unwrap().len()
        );
    }
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn executable_withdraws_corrected_fields_and_preserves_absent_reports() {
    let root = scratch();
    let stations = json!([{"station_id":"A","lat":0.5,"lon":0.5,"elevation_m":0.0}]);
    let observe = |name: &str, row: &str| {
        let csv = root.join(format!("{name}.csv"));
        let artifact = root.join(format!("{name}.json"));
        fs::write(
            &csv,
            format!("station,valid,lon,lat,elevation,tmpf,dwpf,sknt,drct,alti\n{row}"),
        )
        .unwrap();
        let receipt = call(
            &root,
            name,
            json!({"action":"observe","format":"iem_csv","input":csv,"valid_time":"2026-10-01T01:00:00Z","stations":stations,"output":artifact}),
        );
        (artifact, receipt)
    };
    let (good, good_receipt) = observe("accepted", "A,2026-10-01 01:00,0.5,0.5,0,60,50,10,90,30\n");
    let archive = root.join("accepted.obs.zst");
    call(
        &root,
        "archive-once",
        json!({"action":"observe","format":"iem_csv","input":root.join("accepted.csv"),"valid_time":"2026-10-01T01:00:00Z","stations":[],"archive_output":archive,"output":root.join("all-source.json")}),
    );
    let projection = call(
        &root,
        "archive-project",
        json!({"action":"observe","format":"archive","input":archive,"valid_time":"2026-10-01T01:00:00Z","stations":stations,"output":root.join("projected.json")}),
    );
    assert_eq!(
        projection["semantic_sha256"],
        good_receipt["semantic_sha256"]
    );
    let (bad, _) = observe("rejected", "A,2026-10-01 01:00,0.5,0.5,0,999,50,10,90,30\n");
    let (missing, _) = observe("missing", "A,2026-10-01 01:00,0.5,0.5,0,,50,10,90,30\n");
    let (absent, _) = observe("partial", "B,2026-10-01 01:00,0.5,0.5,0,60,50,10,90,30\n");
    let merge = |name: &str, inputs: Vec<PathBuf>| {
        let output = root.join(format!("{name}.json"));
        let receipt = call(
            &root,
            name,
            json!({"action":"merge","inputs":inputs,"valid_time":"2026-10-01T01:00:00Z","stations":stations,"output":output}),
        );
        let artifact: Value = serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
        (output, receipt, artifact)
    };
    for (name, path) in [("qc-withdraw", bad.clone()), ("missing-withdraw", missing)] {
        let (_, receipt, a) = merge(name, vec![path, good.clone()]);
        assert!(a["points"][0]["values"]["temperature_2m"].is_null());
        assert!(a["points"][0]["values"]["temperature_2m_raw"].is_null());
        assert!(a["points"][0]["values"]["dewpoint_2m"].is_number());
        assert_ne!(receipt["semantic_sha256"], good_receipt["semantic_sha256"]);
        assert!(
            a["tombstones"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v["variable"] == "temperature_2m")
        );
    }
    let (_, _, a) = merge("preserve-absent", vec![absent.clone(), good.clone()]);
    assert!(a["points"][0]["values"]["temperature_2m"].is_number());
    let (withdrawn, _, _) = merge("withdrawn", vec![bad, good.clone()]);
    let (_, _, a) = merge("absent-after-withdrawal", vec![absent, withdrawn.clone()]);
    assert!(a["points"][0]["values"]["temperature_2m"].is_null());
    let (_, _, a) = merge("restored", vec![good.clone(), withdrawn]);
    assert!(a["points"][0]["values"]["temperature_2m"].is_number());
    let awc_csv = root.join("awc.csv.gz");
    use std::io::Write;
    let mut gzip = flate2::write::GzEncoder::new(
        fs::File::create(&awc_csv).unwrap(),
        flate2::Compression::default(),
    );
    gzip.write_all(b"data source=metars\nstation_id,observation_time,latitude,longitude,elevation_m,temp_c,dewpoint_c,wind_speed_kt,wind_dir_degrees,altim_in_hg\nA,2026-10-01T01:00:00.000Z,0.5,0.5,0,20,10,10,90,30\n").unwrap();
    gzip.finish().unwrap();
    let awc = root.join("awc.json");
    call(
        &root,
        "awc-cache",
        json!({"action":"observe","format":"awc_csv","input":awc_csv,"valid_time":"2026-10-01T01:00:00Z","stations":stations,"output":awc}),
    );
    let (_, _, a) = merge(
        "other-feed-retained",
        vec![root.join("rejected.json"), awc, good],
    );
    assert_eq!(a["points"][0]["values"]["temperature_2m"], 293.15);
    assert_eq!(
        a["points"][0]["observation_sources"]["temperature_2m"],
        "awc-metar"
    );
    clean(&root);
}
#[test]
fn executable_reports_zero_pair_coverage_without_scoring_unpaired_grids() {
    let root = scratch();
    let mut a = json!({"schema":"woof.scoreboard.artifact.v1","method_id":"scoreboard-v1","valid_time":"2026-10-01T01:00:00Z","init_time":"2026-10-01T00:00:00Z","domain":"d01","lead_minutes":60,"source_sha256":"same","support_hash":"same","stations":[],"points":[],"grids":[{"spec":{"south":0.0,"west":0.0,"dlat":0.01,"dlon":0.01,"nx":2,"ny":2,"dx_m":1000.0},"variable":"reflectivity","window_minutes":0,"values":[0.0,1.0,null,5.0]}]});
    let forecast = root.join("forecast.json");
    let obs = root.join("obs.json");
    let baseline = root.join("baseline.json");
    fs::write(&forecast, serde_json::to_vec(&a).unwrap()).unwrap();
    fs::write(&obs, serde_json::to_vec(&a).unwrap()).unwrap();
    a["grids"] = json!([]);
    fs::write(&baseline, serde_json::to_vec(&a).unwrap()).unwrap();
    let result = call(
        &root,
        "missing-baseline-grid",
        json!({"action":"score","forecast":forecast,"observations":obs,"baseline":baseline,"variable":"reflectivity"}),
    );
    assert_eq!(result["pairs"], json!([]));
    assert_eq!(
        result["coverage"],
        json!({"forecastSamples":3,"baselineSamples":0,"observationSamples":3})
    );
    clean(&root);
}
#[test]
fn executable_reads_wrf_and_scores_large_finite_errors() {
    let root = scratch();
    let input = root.join("wrfout");
    let previous = root.join("previous");
    frame(&input, "2026-10-01_01:00:00", 500., 4., true, 60.);
    frame(&previous, "2026-10-01_00:00:00", 300., 1., true, 0.);
    let stations = json!([{"station_id":"A","lat":0.5,"lon":0.5,"elevation_m":200.0},{"station_id":"OUT","lat":3.0,"lon":3.0,"elevation_m":0.0}]);
    let grid = json!({"south":0.0,"west":0.0,"dlat":1.0,"dlon":1.0,"nx":2,"ny":2,"dx_m":100000.0});
    let forecast = root.join("forecast.json");
    call(
        &root,
        "extract",
        json!({"action":"extract","format":"wrf","input":input,"previous_input":previous,"window_minutes":60,"valid_time":"2026-10-01T01:00:00Z","init_time":"2026-10-01T00:00:00Z","lead_minutes":60,"stations":stations,"grid":grid,"output":forecast}),
    );
    let a: Value = serde_json::from_slice(&fs::read(&forecast).unwrap()).unwrap();
    assert!((a["points"][0]["values"]["temperature_2m"].as_f64().unwrap() - 499.35).abs() < 1e-8);
    assert!(
        a["points"][0]["values"]["u_wind_10m"]
            .as_f64()
            .unwrap()
            .abs()
            < 1e-10
    );
    assert_eq!(a["points"][0]["values"]["v_wind_10m"], 10.);
    assert_eq!(a["grids"][0]["values"][0], 3.);
    assert_eq!(
        a["diagnostics"]["accumulation_start"],
        "2026-10-01T00:00:00Z"
    );
    let csv = root.join("obs.csv");
    fs::write(&csv,"station,valid,lat,lon,elevation,tmpf,dwpf,sknt,drct,alti\nA,2026-10-01 01:00,0.5,0.5,200,80.33,65,10,180,29.92\nOUT,2026-10-01 01:00,3,3,0,80.33,65,10,180,29.92\n").unwrap();
    let obs = root.join("obs.json");
    call(
        &root,
        "observe",
        json!({"action":"observe","format":"iem_csv","input":csv,"valid_time":"2026-10-01T01:00:00Z","stations":stations,"grid":grid,"output":obs}),
    );
    let result = call(
        &root,
        "score",
        json!({"action":"score","forecast":forecast,"baseline":forecast,"observations":obs,"variable":"t2"}),
    );
    assert_eq!(result["pairs"][0]["woof"]["count"], 1);
    assert!(result["summary"][0]["woof"]["mae"].as_f64().unwrap() > 190.);
    let inferred_path = root.join("inferred.json");
    let inferred = call(
        &root,
        "infer-init",
        json!({"action":"extract","format":"wrf","input":input,"valid_time":"2026-10-01T01:00:00Z","init_time":null,"lead_minutes":0,"stations":stations,"grid":grid,"output":inferred_path}),
    );
    assert_eq!(inferred["init_time"], "2026-10-01T00:00:00Z");
    assert_eq!(inferred["lead_minutes"], 60);
    let xtime_frame = root.join("xtime-frame");
    frame(&xtime_frame, "2026-10-01_01:00:00", 300., 4., false, 60.);
    let fallback = call(
        &root,
        "infer-xtime",
        json!({"action":"extract","format":"wrf","input":xtime_frame,"valid_time":"2026-10-01T01:00:00Z","stations":stations,"output":root.join("xtime.json")}),
    );
    assert_eq!(fallback["init_time"], "2026-10-01T00:00:00Z");
    assert_eq!(fallback["lead_minutes"], 60);
    assert_eq!(
        fallback["diagnostics"]["init_time_source"],
        "Times minus XTIME"
    );
    let seconds_frame = root.join("seconds-frame");
    frame(&seconds_frame, "2026-10-01_00:00:30", 300., 0., true, 0.5);
    let seconds_path = root.join("seconds.json");
    let seconds = call(
        &root,
        "infer-seconds",
        json!({"action":"extract","format":"wrf","input":seconds_frame,"valid_time":"2026-10-01T00:00:30Z","stations":stations,"output":seconds_path}),
    );
    assert_eq!(seconds["lead_seconds"], 30);
    assert_eq!(seconds["lead_minutes"], 0);
    let seconds_csv = root.join("seconds.csv");
    fs::write(&seconds_csv,"station,valid,lat,lon,elevation,tmpf,dwpf,sknt,drct,alti\nA,2026-10-01 00:00,0.5,0.5,200,80.33,65,10,180,29.92\n").unwrap();
    let seconds_obs = root.join("seconds-obs.json");
    call(
        &root,
        "observe-seconds",
        json!({"action":"observe","format":"iem_csv","input":seconds_csv,"valid_time":"2026-10-01T00:00:30Z","stations":stations,"output":seconds_obs}),
    );
    let seconds_score = call(
        &root,
        "score-seconds",
        json!({"action":"score","forecast":seconds_path,"baseline":seconds_path,"observations":seconds_obs,"variable":"t2"}),
    );
    assert_eq!(seconds_score["leadSeconds"], 30);
    assert_eq!(seconds_score["pairs"][0]["woof"]["count"], 1);
    let wrong = root.join("wrong-init.request.json");
    fs::write(&wrong,serde_json::to_vec(&json!({"action":"extract","format":"wrf","input":input,"valid_time":"2026-10-01T01:00:00Z","init_time":"2026-10-01T01:00:00Z","stations":stations,"output":root.join("wrong.json")})).unwrap()).unwrap();
    let rejected = Command::new(env!("CARGO_BIN_EXE_rw_scoreboard"))
        .args(["--request", wrong.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    assert!(
        String::from_utf8_lossy(&rejected.stderr)
            .contains("differs from the native WRF simulation start")
    );
    let columns = call(
        &root,
        "score-columns",
        json!({"action":"score","forecast":format!("{}.columns.zst",forecast.display()),"baseline":format!("{}.columns.zst",forecast.display()),"observations":format!("{}.columns.zst",obs.display()),"variable":"t2"}),
    );
    assert_eq!(
        result, columns,
        "columnar read must preserve metadata, per-variable observation times and exact scores"
    );
    let mut equivalent = a.clone();
    equivalent["init_time"] = json!("2026-10-01T01:00:00.000+01:00");
    let equivalent_path = root.join("equivalent-cycle.json");
    fs::write(&equivalent_path, serde_json::to_vec(&equivalent).unwrap()).unwrap();
    let same = call(
        &root,
        "same-cycle",
        json!({"action":"score","forecast":forecast,"baseline":equivalent_path,"observations":obs,"variable":"t2"}),
    );
    assert_eq!(
        same["pairs"], result["pairs"],
        "equivalent ISO cycle strings must compare as timestamps"
    );
    let mut earlier = a.clone();
    earlier["init_time"] = json!("2026-09-30T23:00:00Z");
    let earlier_path = root.join("earlier-cycle.json");
    fs::write(&earlier_path, serde_json::to_vec(&earlier).unwrap()).unwrap();
    let older = call(
        &root,
        "older-cycle",
        json!({"action":"score","forecast":forecast,"baseline":earlier_path,"observations":obs,"variable":"t2","baseline_policy":"latest_at_or_before_init"}),
    );
    assert_eq!(older["baseline_cycle_gap_seconds"], 3600);
    assert_eq!(older["baseline_lead_seconds"], 7200);
    let mut later = a.clone();
    later["init_time"] = json!("2026-10-01T00:30:00Z");
    let later_path = root.join("later-cycle.json");
    fs::write(&later_path, serde_json::to_vec(&later).unwrap()).unwrap();
    let later_request = root.join("later-cycle.request.json");
    fs::write(&later_request,serde_json::to_vec(&json!({"action":"score","forecast":forecast,"baseline":later_path,"observations":obs,"variable":"t2","baseline_policy":"latest_at_or_before_init"})).unwrap()).unwrap();
    let rejected = Command::new(env!("CARGO_BIN_EXE_rw_scoreboard"))
        .args(["--request", later_request.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    assert!(
        String::from_utf8_lossy(&rejected.stderr).contains("newer than the WOOF initialization")
    );
    clean(&root);
}
