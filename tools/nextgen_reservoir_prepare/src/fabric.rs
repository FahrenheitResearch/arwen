//! Original Apache-2.0 startup binding of existing lake attributes to the CUDA ABI.
//! No native capture, reservoir time step, or implicit physical initializer.
use super::{doubles, floats, hash, CONTROLS, PARAMETERS, PIN};
use rayon::prelude::*;
use serde_json::{json, Value};
use serde::Deserialize;
use std::{collections::{BTreeMap, BTreeSet}, fs, path::{Component, Path}, time::Instant};

type Result<T> = std::result::Result<T, String>;
type Payload = (String, &'static str, Vec<usize>, Vec<u8>);

#[derive(Deserialize)]
struct LakeAttributes { lakes: Vec<Value> }

fn read(path: &Path) -> Result<Vec<u8>> {
    fs::read(path).map_err(|e| format!("{}: {e}", path.display()))
}
fn text<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v[key].as_str().filter(|s| !s.trim().is_empty()).ok_or_else(|| format!("missing {key}"))
}
fn number(v: &Value, key: &str) -> Result<f32> {
    let x = v[key].as_f64().ok_or_else(|| format!("missing numeric {key}"))? as f32;
    if !x.is_finite() { return Err(format!("nonfinite/out-of-f32-range {key}")); }
    Ok(x)
}
fn double(v: &Value, key: &str) -> Result<f64> {
    let x = v[key].as_f64().ok_or_else(|| format!("missing numeric {key}"))?;
    if !x.is_finite() { return Err(format!("nonfinite {key}")); }
    Ok(x)
}
fn id(v: &Value) -> Result<i64> {
    if let Some(x) = v.as_i64().filter(|x| *x > 0) { return Ok(x); }
    if let Some(x) = v.as_f64().filter(|x| x.is_finite() && *x > 0. && *x <= 9007199254740991. && x.fract() == 0.) { return Ok(x as i64); }
    Err("lake/state ID must be a positive exact integer".into())
}
fn sha(v: &str) -> bool { v.len() == 64 && v.bytes().all(|b| b.is_ascii_hexdigit()) }

pub fn run(manifest_path: &Path, initial_path: &Path, output: &Path) -> Result<()> {
    let started = Instant::now();
    let bytes = read(manifest_path)?;
    let manifest: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    if manifest["schema"] != "arwen.nextgen.hydrofabric.v1" { return Err("expected prepared hydrofabric manifest".into()); }
    let spec = &manifest["auxiliary"]["attributes.json"];
    // The existing reader keys this auxiliary by its fixed filename; older
    // prepared manifests deliberately contain only its hash, not a path.
    let relative = Path::new(spec.get("path").and_then(Value::as_str).unwrap_or("attributes.json"));
    if relative.is_absolute() || !relative.components().all(|c| matches!(c, Component::Normal(_))) { return Err("attributes path escapes prepared fabric".into()); }
    let attrs_bytes = read(&manifest_path.parent().unwrap_or(Path::new(".")).join(relative))?;
    let attrs_sha = hash(&attrs_bytes);
    if text(spec, "sha256")? != attrs_sha { return Err("attributes SHA256 mismatch".into()); }
    let attrs: LakeAttributes = serde_json::from_slice(&attrs_bytes).map_err(|e| e.to_string())?;
    drop(attrs_bytes);
    let lakes = &attrs.lakes;
    let initial_bytes = read(initial_path)?;
    let initial: Value = serde_json::from_slice(&initial_bytes).map_err(|e| e.to_string())?;
    if initial["schema"] != "arwen.nextgen.reservoir-initial.v1" { return Err("expected reservoir-initial.v1 packet".into()); }
    let origin = text(&initial, "time_origin")?;
    let datum = text(&initial, "vertical_datum")?;
    let source = &initial["source"];
    if !matches!(text(source, "method")?, "analysis" | "retrospective" | "replay-spinup") || !sha(text(source, "sha256")?) { return Err("initial state requires analysis/retrospective/replay-spinup source SHA256".into()); }
    // Datum/time/identity are producer attestations, not guessed from a filename.
    if text(source, "vertical_datum")? != datum || text(source, "time_origin")? != origin { return Err("initial source epoch/datum differs from packet".into()); }
    if text(source, "fabric_manifest_sha256")? != hash(&bytes) { return Err("initial state identifies another fabric".into()); }
    let mut states = BTreeMap::new();
    for row in initial["rows"].as_array().ok_or("missing initial rows")? {
        let key = id(&row["id"])?;
        let h = number(row, "height_m")?;
        let q = double(row, "release_m3_s")?;
        if q < 0. { return Err("initial release must be nonnegative".into()); }
        if states.insert(key, (h,q)).is_some() { return Err("duplicate initial reservoir ID".into()); }
    }
    let mut ordered = BTreeMap::new();
    for row in lakes {
        let key = id(&row["hl_link"])?;
        if row.get("reservoir_type").is_some_and(|v| v.as_i64() != Some(1)) {
            return Err(format!("reservoir {key} declares DA/type controls; this level-pool packer cannot replace them"));
        }
        if ordered.insert(key, row).is_some() { return Err("duplicate lake ID".into()); }
    }
    if states.keys().collect::<BTreeSet<_>>() != ordered.keys().collect::<BTreeSet<_>>() { return Err("initial and lake ID axes differ; no dropped/extra state rows allowed".into()); }
    // Decode/validate independent lake columns on all preparation workers.
    let rows = ordered.iter().collect::<Vec<_>>().par_iter().map(|(key,row)| -> Result<_> {
        let mut p = [0f32;9];
        for (i,name) in ["LkArea","WeirE","LkMxE","WeirC","WeirL","Dam_Length","OrificeE","OrificeC","OrificeA"].iter().enumerate() { p[i] = number(row,name)?; }
        let fraction = number(row,"ifd")?;
        if p[0] <= 0. || p[5] <= 0. || p[2] < p[1] || p[1] < p[6] || [3,4,7,8].iter().any(|i| p[*i] < 0.) || !(0. ..=1.).contains(&fraction) { return Err(format!("invalid physical lake parameters for {key}")); }
        let (h,q) = states[key];
        if h < p[6] { return Err(format!("initial height below orifice datum for {key}")); }
        // No cold sentinel/ifd fallback: initialization belongs to its producer.
        Ok((p,h,q,json!({"id":key,"nexus_id":text(row,"id")?,"outlet_flowpath":text(row,"toid")?})))
    }).collect::<Result<Vec<_>>>()?;
    let n = rows.len();
    let ids: Vec<u8> = ordered.keys().flat_map(|x|x.to_le_bytes()).collect();
    let mut parameters = Vec::with_capacity(9*n);
    for i in 0..9 { parameters.extend(rows.iter().map(|r| r.0[i])); }
    let heights = rows.iter().map(|r|r.1).collect::<Vec<_>>();
    let releases = rows.iter().map(|r|r.2).collect::<Vec<_>>();
    let mut payloads: Vec<Payload> = vec![];
    let mut push = |name: &str,dtype,shape,data| payloads.push((name.into(),dtype,shape,data));
    push("parameters","<f4",vec![9,n],floats(&parameters));
    for name in ["height","initial_height","levelpool_height"] { push(name,"<f4",vec![n],floats(&heights)); }
    for name in ["previous_inflow","inflow","lateral","levelpool_release"] { push(name,"<f4",vec![n],vec![0;n*4]); }
    push("types","<i4",vec![n],(0..n).flat_map(|_|1i32.to_le_bytes()).collect());
    for name in ["indices","dynamic_type","active_source","fault"] { push(name,"<i4",vec![n],vec![0;n*4]); }
    push("offsets","<i8",vec![n+1],vec![0;(n+1)*8]);
    push("controls","<f8",vec![8,n],vec![0;8*n*8]);
    push("carriers","<f8",vec![3,n],vec![0;3*n*8]);
    push("clock","<f8",vec![n],vec![0;n*8]);
    push("release","<f8",vec![n],doubles(&releases));
    push("assimilated","<f8",vec![n],doubles(&vec![-9999.;n]));
    for name in ["series","times"] { push(name,"<f8",vec![0],vec![]); }
    // Validate everything before creating an output. Existing output is never overwritten.
    fs::create_dir(output).map_err(|e|e.to_string())?;
    let files = payloads.par_iter().map(|(name,dtype,shape,data)| -> Result<_> {
        let key = format!("reservoir_{name}"); let path = format!("{key}.bin");
        fs::write(output.join(&path),data).map_err(|e|e.to_string())?;
        Ok((key,json!({"path":path,"dtype":dtype,"shape":shape,"bytes":data.len(),"sha256":hash(data)})))
    }).collect::<Result<BTreeMap<_,_>>>()?;
    fs::write(output.join("reservoir_ids.i64"),&ids).map_err(|e|e.to_string())?;
    let bindings = payloads.iter().map(|(name,_,_,_)|(name.clone(),format!("reservoir_{name}"))).collect::<BTreeMap<_,_>>();
    let result = json!({"schema":"arwen.nextgen.reservoir-payloads.v1","files":files,
        "reservoirs":{"arrays":bindings,"source_pin":PIN,"parameter_columns":PARAMETERS,"control_columns":CONTROLS,"validated_csr":true,"time_origin":origin,"reservoir_order_sha256":hash(&ids),"initialization":{"method":source["method"],"source_sha256":source["sha256"],"packet_sha256":hash(&initial_bytes),"vertical_datum":datum}},
        "reservoir_ids":{"path":"reservoir_ids.i64","dtype":"<i8","shape":[n],"sha256":hash(&ids)},
        "lake_outlets":rows.iter().map(|r|&r.3).collect::<Vec<_>>(),
        "source_manifest":{"path":manifest_path,"sha256":hash(&bytes)},"attributes_sha256":attrs_sha,
        "data_license":manifest["input"]["data_license"],"data_scope":"inherits producer data terms; no licence clearance inferred by this packer",
        "scope":"native-free type1 level-pool startup fragment; no topology collapse, DA acquisition, physical parameter derivation or native trajectory proof",
        "rows":n,"parallelism":{"rayon_threads":rayon::current_num_threads(),"work":"independent lake validation and payload writes; no CPU reservoir step"},"wall_ms":started.elapsed().as_secs_f64()*1000.});
    fs::write(output.join("manifest.json"),serde_json::to_vec_pretty(&result).unwrap()).map_err(|e|e.to_string())?;
    println!("{}",serde_json::to_string(&result).unwrap());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    struct Case { root: std::path::PathBuf, manifest: Value, attrs: Value, initial: Value }
    impl Case {
        fn new() -> Self {
            let base = std::env::var("HYDRO_TEST_ROOT").expect("HYDRO_TEST_ROOT must name owned data-disk scratch");
            let root = Path::new(&base).join(format!("case-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));
            fs::create_dir_all(&root).unwrap();
            let mut c = Self {root,manifest:json!({"schema":"arwen.nextgen.hydrofabric.v1","input":{"data_license":"synthetic ABI fixture; no dataset"},"auxiliary":{}}),
                attrs:json!({"lakes":[
                    {"hl_link":27,"id":"nex-27","toid":"wb-29","LkArea":2.,"WeirE":110.,"LkMxE":115.,"WeirC":1.5,"WeirL":12.,"Dam_Length":27.,"OrificeE":100.,"OrificeC":0.6,"OrificeA":2.,"ifd":0.4},
                    {"hl_link":3,"id":"nex-3","toid":"wb-5","LkArea":1.,"WeirE":12.,"LkMxE":15.,"WeirC":1.5,"WeirL":8.,"Dam_Length":5.,"OrificeE":10.,"OrificeC":0.6,"OrificeA":1.,"ifd":0.4}]}),
                initial:json!({"schema":"arwen.nextgen.reservoir-initial.v1","time_origin":"2023-01-05T00:00:00Z","vertical_datum":"synthetic-local-metres","source":{"method":"replay-spinup","sha256":hash(b"synthetic ABI fixture only"),"time_origin":"2023-01-05T00:00:00Z","vertical_datum":"synthetic-local-metres"},"rows":[{"id":27,"height_m":111.,"release_m3_s":2.5},{"id":3,"height_m":13.,"release_m3_s":1.25}]})};
            c.seal(); c
        }
        fn seal(&mut self) {
            let attrs = serde_json::to_vec(&self.attrs).unwrap();
            fs::write(self.root.join("attributes.json"),&attrs).unwrap();
            self.manifest["auxiliary"]["attributes.json"] = json!({"sha256":hash(&attrs)});
            let manifest = serde_json::to_vec(&self.manifest).unwrap();
            fs::write(self.root.join("fabric.json"),&manifest).unwrap();
            self.initial["source"]["fabric_manifest_sha256"] = json!(hash(&manifest));
            fs::write(self.root.join("initial.json"),serde_json::to_vec(&self.initial).unwrap()).unwrap();
        }
        fn invoke(&self, name: &str) -> Result<()> { run(&self.root.join("fabric.json"),&self.root.join("initial.json"),&self.root.join(name)) }
        fn refused(&self, message: &str) {
            let error = self.invoke("refused").unwrap_err();
            assert!(error.contains(message),"{error}");
            assert!(!self.root.join("refused").exists(),"invalid input must not leave a partial fragment");
        }
    }
    #[test]
    fn packed_axes_values_and_hashes() {
        let c = Case::new(); c.invoke("packed").unwrap();
        let root = c.root.join("packed");
        let m: Value = serde_json::from_slice(&read(&root.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(read(&root.join("reservoir_ids.i64")).unwrap(),[3i64.to_le_bytes(),27i64.to_le_bytes()].concat());
        assert_eq!(read(&root.join("reservoir_height.bin")).unwrap(),floats(&[13.,111.]));
        assert_eq!(read(&root.join("reservoir_release.bin")).unwrap(),doubles(&[1.25,2.5]));
        let p=read(&root.join("reservoir_parameters.bin")).unwrap();
        assert_eq!(&p[5*2*4..6*2*4],floats(&[5.,27.]));
        assert_eq!(m["lake_outlets"][0]["outlet_flowpath"],"wb-5");
        assert_eq!(m["files"].as_object().unwrap().len(),21);
        for spec in m["files"].as_object().unwrap().values() {
            let bytes=read(&root.join(text(spec,"path").unwrap())).unwrap();
            assert_eq!(spec["sha256"],hash(&bytes));
            assert_eq!(spec["bytes"].as_u64().unwrap(),bytes.len() as u64);
        }
        // Retain this tiny actual Rust output for the CUDA factory consumer.
        fs::write(Path::new(&std::env::var("HYDRO_TEST_ROOT").unwrap()).join("cuda-input-path.txt"),root.to_string_lossy().as_bytes()).unwrap();
    }
    #[test]
    fn unrelated_continental_attributes_leave_reservoir_payloads_unchanged() {
        let mut c=Case::new(); c.invoke("before").unwrap();
        c.attrs["divides"] = json!([{"divide_id":"cat-1","model_parameters":vec![1.2345678901234567;8193]}]);
        c.attrs["flowpath_attributes"] = json!([{"id":"wb-1","unused_nested":{"rows":[1,2,3]}}]);
        c.seal(); c.invoke("after").unwrap();
        let m: Value=serde_json::from_slice(&read(&c.root.join("before/manifest.json")).unwrap()).unwrap();
        for spec in m["files"].as_object().unwrap().values() {
            let path=text(spec,"path").unwrap();
            assert_eq!(read(&c.root.join("before").join(path)).unwrap(),read(&c.root.join("after").join(path)).unwrap());
        }
        let after: Value=serde_json::from_slice(&read(&c.root.join("after/manifest.json")).unwrap()).unwrap();
        assert_eq!(after["attributes_sha256"],c.manifest["auxiliary"]["attributes.json"]["sha256"]);
        fs::write(c.root.join("attributes.json"),b"{}").unwrap();c.refused("SHA256");
    }
    #[test]
    fn initial_release_preserves_float64_words() {
        let mut c = Case::new();
        let release = 1.2345678901234567_f64;
        c.initial["rows"][1]["release_m3_s"] = json!(release);
        c.seal();
        c.invoke("packed").unwrap();
        assert_eq!(read(&c.root.join("packed/reservoir_release.bin")).unwrap(),
                   doubles(&[release, 2.5]));
        fs::write(Path::new(&std::env::var("HYDRO_TEST_ROOT").unwrap()).join("cuda-precise-input-path.txt"),
                  c.root.join("packed").to_string_lossy().as_bytes()).unwrap();
    }
    #[test]
    fn source_row_permutation_preserves_every_payload() {
        let mut c=Case::new(); c.invoke("before").unwrap();
        c.attrs["lakes"].as_array_mut().unwrap().reverse();
        c.initial["rows"].as_array_mut().unwrap().reverse(); c.seal(); c.invoke("after").unwrap();
        let m: Value=serde_json::from_slice(&read(&c.root.join("before/manifest.json")).unwrap()).unwrap();
        for spec in m["files"].as_object().unwrap().values() {
            let path=text(spec,"path").unwrap();
            assert_eq!(read(&c.root.join("before").join(path)).unwrap(),read(&c.root.join("after").join(path)).unwrap());
        }
        assert_eq!(read(&c.root.join("before/reservoir_ids.i64")).unwrap(),read(&c.root.join("after/reservoir_ids.i64")).unwrap());
    }
    #[test]
    fn missing_lake_parameter_and_invalid_geometry_refuse() {
        let mut c=Case::new(); c.attrs["lakes"][0]["Dam_Length"]=Value::Null;c.seal();c.refused("Dam_Length");
        let mut c=Case::new();c.attrs["lakes"][0]["WeirE"]=json!(90.);c.seal();c.refused("invalid physical");
    }
    #[test]
    fn incomplete_extra_and_duplicate_ids_refuse() {
        let mut c=Case::new();c.initial["rows"].as_array_mut().unwrap().pop();c.seal();c.refused("ID axes");
        let mut c=Case::new();c.initial["rows"][0]["id"]=json!(28);c.seal();c.refused("ID axes");
        let mut c=Case::new();c.initial["rows"][0]["id"]=json!(3);c.seal();c.refused("duplicate initial");
        let mut c=Case::new();c.attrs["lakes"][0]["hl_link"]=json!(3);c.seal();c.refused("duplicate lake");
    }
    #[test]
    fn initial_datum_and_release_validation() {
        let mut c=Case::new();c.initial["rows"][0]["height_m"]=json!(99.);c.seal();c.refused("below orifice");
        let mut c=Case::new();c.initial["rows"][0]["release_m3_s"]=json!(-1.);c.seal();c.refused("nonnegative");
        let mut c=Case::new();c.initial["source"]["vertical_datum"]=json!("wrong");c.seal();c.refused("epoch/datum");
        let mut c=Case::new();c.initial["source"]["time_origin"]=json!("wrong");c.seal();c.refused("epoch/datum");
    }
    #[test]
    fn hashes_and_declared_da_controls_cannot_be_silently_ignored() {
        let c=Case::new();fs::write(c.root.join("attributes.json"),b"{}").unwrap();c.refused("SHA256");
        let mut c=Case::new();c.initial["source"]["fabric_manifest_sha256"]=json!("0".repeat(64));
        fs::write(c.root.join("initial.json"),serde_json::to_vec(&c.initial).unwrap()).unwrap();c.refused("another fabric");
        let mut c=Case::new();c.attrs["lakes"][0]["reservoir_type"]=json!(3);c.seal();c.refused("DA/type");
    }
    #[test]
    fn output_is_never_overwritten() { let c=Case::new();c.invoke("packed").unwrap();assert!(c.invoke("packed").is_err()); }
}
