// SPDX-License-Identifier: Apache-2.0
// Synthetic exact diagnostic fixture, never a native/observed routing reference.
use nextgen_replay::file_sha256;
use serde_json::{json, Map, Value};
use std::{fs, path::PathBuf};
fn main() {
    let root = PathBuf::from(std::env::args().nth(1).expect("new fixture directory under lane scratch"));
    fs::create_dir(&root).unwrap();
    let mut arrays = Map::new();
    let missing = f32::from_bits(0x7fc00000);
    for (name, values) in [
        ("initial_depth", vec![0., 0.125, 0.5, 1., 2., 0.25]),
        ("next_depth", vec![0.25, 0.0625, 8., 0.75, 3., 9.]),
        ("expected_depth", vec![0.25, 0.0625, missing, 0.75, 3., missing]),
        ("expected_maximum", vec![0.25, 0.125, missing, 1., 3., missing]),
        ("expected_failed_depth", vec![missing; 6]),
    ] {
        let path = root.join(format!("{name}.raw"));
        fs::write(&path, values.into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>()).unwrap();
        arrays.insert(name.into(), json!({"path":path,"sha256":file_sha256(&path).unwrap(),"shape":[2,3],"dtype":"f32","order":"c"}));
    }
    let path = root.join("active.raw");
    fs::write(&path, [1i32,1,0,1,1,0].into_iter().flat_map(i32::to_le_bytes).collect::<Vec<_>>()).unwrap();
    arrays.insert("active".into(),json!({"path":path,"sha256":file_sha256(&path).unwrap(),"shape":[2,3],"dtype":"i32","order":"c"}));
    for (name,bits) in [("healthy",0u32),("source_failed",1),("bad_depth",2),("bad_mask",4)] {
        let path = root.join(format!("{name}.raw")); fs::write(&path,bits.to_le_bytes()).unwrap();
        arrays.insert(name.into(),json!({"path":path,"sha256":file_sha256(&path).unwrap(),"shape":[4],"dtype":"u8","order":"c"}));
    }
    let manifest = json!({"schema":"nextgen.inundation.fixture/v1","origin":"tool-self-test","description":"Original exact copy/max/mask/health diagnostic fixture; no native or observed physics", "arrays":Value::Object(arrays)});
    fs::write(root.join("fixture.json"),serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
}
