// SPDX-License-Identifier: Apache-2.0
use super::*;
use std::process::Command;

fn base() -> State {
    State {
        f: BTreeMap::from([
            ("soil_reservoir.storage_max_m".into(), 0.878),
            ("gw_reservoir.storage_max_m".into(), 16.0),
        ]),
        i: BTreeMap::from([
            ("surface_runoff_scheme".into(), 1),
            ("soil_reservoir.is_sft_coupled".into(), 0),
            ("soil_reservoir.is_aet_rootzone".into(), 0),
            ("N_nash_subsurface".into(), 2),
            ("num_giuh_ordinates".into(), 5),
        ]),
        vectors: BTreeMap::new(),
        raw_initial: false,
    }
}
fn raw() -> Config {
    Config(
        RAW_INITIAL_KEYS
            .into_iter()
            .zip([
                "0.499238123374",
                "8",
                "0.001,0.002",
                "0.003,0.004,0.005,0.006,0.007",
                "0.0001",
            ])
            .map(|(k, v)| (k.to_owned(), v.to_owned()))
            .collect(),
    )
}
#[test]
fn reject_incomplete_and_unrepresentable_state() {
    for key in RAW_INITIAL_KEYS {
        let mut c = raw();
        c.0.remove(key);
        assert!(apply_raw_initial(&c, &mut base())
            .unwrap_err()
            .contains("all five"));
    }
    for (key, value) in [
        (RAW_INITIAL_KEYS[0], "1"),
        (RAW_INITIAL_KEYS[1], "17"),
        (RAW_INITIAL_KEYS[4], "-0.1"),
        (RAW_INITIAL_KEYS[4], "NaN"),
        (RAW_INITIAL_KEYS[2], "0"),
        (RAW_INITIAL_KEYS[3], "0,0,0,0,0,0"),
        (RAW_INITIAL_KEYS[2], "-0.1,0"),
        (RAW_INITIAL_KEYS[2], "1e308,1e308"),
    ] {
        let mut c = raw();
        c.0.insert(key.into(), value.into());
        assert!(apply_raw_initial(&c, &mut base()).is_err(), "{key}={value}");
    }
    for (key, value) in [
        ("surface_runoff_scheme", 2),
        ("soil_reservoir.is_sft_coupled", 1),
        ("soil_reservoir.is_aet_rootzone", 1),
    ] {
        let mut s = base();
        s.i(key, value);
        assert!(apply_raw_initial(&raw(), &mut s)
            .unwrap_err()
            .contains("lacks"));
    }
}

fn read_json(p: &Path) -> Value {
    serde_json::from_slice(&fs::read(p).unwrap()).unwrap()
}
fn f64s(p: &Path) -> Vec<f64> {
    let bytes = fs::read(p).unwrap();
    assert_eq!(bytes.len() % 8, 0);
    bytes
        .chunks_exact(8)
        .map(|x| f64::from_le_bytes(x.try_into().unwrap()))
        .collect()
}
fn i32s(p: &Path) -> Vec<i32> {
    fs::read(p)
        .unwrap()
        .chunks_exact(4)
        .map(|x| i32::from_le_bytes(x.try_into().unwrap()))
        .collect()
}
fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}
fn csv(v: &[f64]) -> String {
    v.iter()
        .map(|x| format!("{x:.17e}"))
        .collect::<Vec<_>>()
        .join(",")
}
fn invoke(exe: &Path, req: &Value, path: &Path, good: bool) -> Value {
    fs::write(path, serde_json::to_vec_pretty(req).unwrap()).unwrap();
    let t = Instant::now();
    let p = Command::new(exe).arg(path).output().unwrap();
    let result = json!({"command":[exe,path],"exit":p.status.code(),"wall_ms":t.elapsed().as_secs_f64()*1000.0,
        "stdout":String::from_utf8_lossy(&p.stdout),"stderr":String::from_utf8_lossy(&p.stderr)});
    assert_eq!(p.status.success(), good, "{result}");
    result
}

// This receipt test exercises the actual CLI and packed output, using only
// retained, sealed Apache CFE native boundary packets. It writes in TMPDIR.
#[test]
#[ignore = "requires retained sealed native packets and before/after CLI executables"]
fn native_raw_words_cli_and_fraction_regression() {
    let root = std::env::var_os("CFE_RAW_FIXTURES").expect("CFE_RAW_FIXTURES");
    let root = PathBuf::from(root);
    let exe = PathBuf::from(std::env::var_os("CFE_RAW_EXE").expect("CFE_RAW_EXE"));
    let old = PathBuf::from(std::env::var_os("CFE_RAW_BEFORE_EXE").expect("CFE_RAW_BEFORE_EXE"));
    let work = PathBuf::from(std::env::var_os("TMPDIR").expect("TMPDIR")).join("raw-cli");
    fs::create_dir(&work).unwrap();
    let mut records = vec![];
    let mut count = 0;
    for packet in ["bmi", "next"] {
        let dir = root.join(packet);
        let m = read_json(&dir.join("manifest.json"));
        let comp = read_json(&dir.join("compare.json"));
        let n = m["catchments"].as_array().unwrap().len();
        let names = m["field_names"].as_array().unwrap();
        let opts = m["option_names"].as_array().unwrap();
        let option = i32s(&dir.join("input/options.bin"));
        let offsets = i32s(&dir.join("input/offsets.bin"));
        let oi = |k: &str| opts.iter().position(|x| x == k).unwrap();
        for side in ["input", "expected"] {
            for key in ["fields", "runoff_queue", "nash_subsurface"] {
                let sha = if side == "input" {
                    m["buffers"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|x| x["name"] == key)
                        .unwrap()["sha256"]
                        .as_str()
                        .unwrap()
                } else {
                    comp["fields"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|x| x["name"] == key)
                        .unwrap()["expected"]["sha256"]
                        .as_str()
                        .unwrap()
                };
                assert_eq!(
                    file_sha256(&dir.join(side).join(format!("{key}.bin"))).unwrap(),
                    sha
                );
            }
            // Options and offsets control raw slice extents and need seals too.
            for key in ["options", "offsets"] {
                let seal = m["buffers"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|x| x["name"] == key)
                    .unwrap();
                assert_eq!(
                    file_sha256(&dir.join("input").join(format!("{key}.bin"))).unwrap(),
                    seal["sha256"].as_str().unwrap()
                );
            }
            let fields = f64s(&dir.join(side).join("fields.bin"));
            let nq = f64s(&dir.join(side).join("runoff_queue.bin"));
            let ns = f64s(&dir.join(side).join("nash_subsurface.bin"));
            let mut cols = vec![];
            let mut expected = vec![];
            let case = work.join(format!("{packet}-{side}"));
            fs::create_dir(&case).unwrap();
            for col in 0..n {
                let f = |k: &str| fields[names.iter().position(|x| x == k).unwrap() * n + col];
                let id = m["bmi"]["framework_witness"]["catchment_ids"][col]
                    .as_str()
                    .unwrap();
                let config = m["native_recipe"]["adaptations"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter_map(|v| v["adapted"].as_str())
                    .find(|p| p.ends_with(&format!("/cfe/{id}_bmi_config.ini")))
                    .unwrap();
                // Source artifact seals live in the manifest's provenance list.
                fn sealed_in(v: &Value, path: &str) -> Option<String> {
                    match v {
                        Value::Object(o) => {
                            if o.get("path").and_then(Value::as_str) == Some(path) {
                                o.get("sha256").and_then(Value::as_str).map(str::to_owned)
                            } else {
                                o.values().find_map(|v| sealed_in(v, path))
                            }
                        }
                        Value::Array(a) => a.iter().find_map(|v| sealed_in(v, path)),
                        _ => None,
                    }
                }
                let canonical = Path::new(config).canonicalize().unwrap();
                let configsha =
                    sealed_in(&m, canonical.to_str().unwrap()).expect("native config seal");
                assert_eq!(file_sha256(&canonical).unwrap(), configsha);
                let mut text = fs::read_to_string(config).unwrap();
                let nn = option[oi("N_nash_subsurface") * n + col] as usize;
                let ng = option[oi("num_giuh_ordinates") * n + col] as usize;
                // layout offsets are fixed in the typed ABI: GIUH, queue, Nash.
                let q = offsets[n + col] as usize;
                let a = offsets[2 * n + col] as usize;
                assert_eq!(nq[q + ng].to_bits(), 0);
                let soil = f("soil_reservoir.storage_m");
                let gw = f("gw_reservoir.storage_m");
                let perc = f("flux_perc_m");
                let nash = &ns[a..a + nn];
                let queue = &nq[q..q + ng];
                text+=&format!("\nN_nash_subsurface={nn}\ninitial_soil_storage_m={soil:.17e}\ninitial_gw_storage_m={gw:.17e}\ninitial_nash_subsurface_m={}\ninitial_giuh_queue_m={}\ninitial_flux_perc_m={perc:.17e}\n",csv(nash),csv(queue));
                let path = case.join(format!("{id}.ini"));
                fs::write(&path, text).unwrap();
                cols.push(
                    json!({"id":id,"config":{"path":path,"sha256":file_sha256(&path).unwrap()}}),
                );
                expected.push((soil, gw, perc, nash.to_vec(), queue.to_vec()));
            }
            let req = json!({"schema":"arwen.cfe.prepare/v1","time_origin":"native-boundary-witness-fresh-clock","columns":cols,"output":case.join("out")});
            let run = invoke(&exe, &req, &case.join("request.json"), true);
            let out = case.join("out");
            let outm = read_json(&out.join("manifest.json"));
            let actual = f64s(&out.join("fields.bin"));
            let queue = f64s(&out.join("runoff_queue.bin"));
            let nash = f64s(&out.join("nash_subsurface.bin"));
            for key in ["runoff_queue", "nash_subsurface"] {
                assert_eq!(
                    fs::read(out.join(format!("{key}.bin"))).unwrap(),
                    fs::read(dir.join(side).join(format!("{key}.bin"))).unwrap()
                );
            }
            let mut a = 0;
            let mut q = 0;
            for (col, (soil, gw, perc, ns, nq)) in expected.iter().enumerate() {
                for (k, v) in [
                    ("soil_reservoir.storage_m", soil),
                    ("gw_reservoir.storage_m", gw),
                    ("flux_perc_m", perc),
                ] {
                    assert_eq!(
                        actual[names.iter().position(|x| x == k).unwrap() * n + col].to_bits(),
                        v.to_bits()
                    );
                    count += 1;
                }
                assert_eq!(bits(&nash[a..a + ns.len()]), bits(ns));
                a += ns.len();
                count += ns.len();
                assert_eq!(bits(&queue[q..q + nq.len()]), bits(nq));
                q += nq.len() + 1;
                count += nq.len() + 1;
                let inv = &outm["initial_water_inventory"][col]["water"];
                assert_eq!(
                    inv["total_held_m"].as_f64().unwrap().to_bits(),
                    (((gw + soil) + ns.iter().sum::<f64>()) + nq.iter().sum::<f64>()).to_bits()
                );
                assert_eq!(inv["storage_source"], "raw initial_* keys");
            }
            records.push(json!({"packet":packet,"side":side,"cli":run,"stages_ms":outm["stages_ms"],"water":outm["initial_water_inventory"]}));
            // Sealed configuration tampering must fail before an output exists.
            if packet == "bmi" && side == "input" {
                let p = PathBuf::from(req["columns"][0]["config"]["path"].as_str().unwrap());
                fs::OpenOptions::new()
                    .append(true)
                    .open(p)
                    .unwrap()
                    .write_all(b"\n#tampered\n")
                    .unwrap();
                let mut bad = req.clone();
                bad["output"] = json!(case.join("bad"));
                records.push(invoke(&exe, &bad, &case.join("bad-request.json"), false));
                assert!(!case.join("bad").exists());
            }
        }
    }
    // Legacy fraction initialization must preserve every typed buffer byte.
    let config = root.join("bmi/manifest.json");
    let m = read_json(&config);
    let path = m["native_recipe"]["adaptations"][0]["adapted"]
        .as_str()
        .unwrap();
    let column = json!({"id":"legacy","config":{"path":path,"sha256":file_sha256(Path::new(path)).unwrap()}});
    let mut req = json!({"schema":"arwen.cfe.prepare/v1","time_origin":"fraction-regression","columns":[column],"output":work.join("before")});
    records.push(invoke(&old, &req, &work.join("before.json"), true));
    req["output"] = json!(work.join("after"));
    records.push(invoke(&exe, &req, &work.join("after.json"), true));
    let before = read_json(&work.join("before/manifest.json"));
    let after = read_json(&work.join("after/manifest.json"));
    for b in before["buffers"].as_array().unwrap() {
        let key = b["name"].as_str().unwrap();
        let a = after["buffers"]
            .as_array()
            .unwrap()
            .iter()
            .find(|x| x["name"] == key)
            .unwrap();
        assert_eq!(b["sha256"], a["sha256"]);
        assert_eq!(b["shape"], a["shape"]);
    }
    fs::write(work.join("verdict.json"),serde_json::to_vec_pretty(&json!({"raw_exact_words":count,"boundaries":4,"legacy_buffers_exact":before["buffers"].as_array().unwrap().len(),"runs":records,"scope":"native words loaded into fresh initialization; not a native restart, world-init execution or physical admission"})).unwrap()).unwrap();
    println!(
        "exact raw native words {count}; all legacy typed buffers exact; sealed tampering refused"
    );
}
