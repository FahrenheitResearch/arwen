// SPDX-License-Identifier: Apache-2.0
//! Actual Rust preparer/serialized-word check; synthetic profiles, no physics claim.
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, time::Instant};

fn sha(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 4, "NAMELIST PARAMETERS NEW_OUTPUT");
    let out = Path::new(&args[3]);
    fs::create_dir(out)?;
    let started = Instant::now();
    let schema: Value =
        serde_json::from_slice(include_bytes!("../../../gpuwm/hydro/noahowp/schema.json"))?;
    let profiles = [[260.123456789, 270., 280., 290.], [301., 302., 303., 304.]];
    let rows: Vec<_> = ["profile-a", "profile-b"]
        .iter()
        .map(|id| json!({"id":id,"namelist":args[1],"parameter_dir":args[2]}))
        .collect();
    let cold = json!({"catchments":rows});
    let mut warm = cold.clone();
    for (row, profile) in warm["catchments"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .zip(profiles)
    {
        row["overrides"] = json!({"soil_temperature_k": profile});
    }
    let run = |name: &str, request: &Value| {
        let p = out.join(format!("{name}.request.json"));
        fs::write(&p, serde_json::to_vec_pretty(request).unwrap()).unwrap();
        arwen_noahowp_init::prepare(&p, &out.join(name))
    };
    let baseline = run("cold", &cold)?;
    let prepared = run("profile", &warm)?;
    assert_ne!(baseline["request_sha256"], prepared["request_sha256"]);
    assert_ne!(baseline["parameter_sha256"], prepared["parameter_sha256"]);
    let stc = schema["fields"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["name"] == "energy.stc")
        .unwrap();
    let offset = stc["offset"].as_u64().unwrap() as usize;
    assert_eq!(stc["bounds"], json!([[-2, 4]]));
    let mut compared = 0;
    let mut changed = 0;
    for (dtype, width) in [("f32", 4), ("i32", 4), ("f64", 8)] {
        let mut expected = fs::read(out.join(format!("cold/{dtype}.slab")))?;
        let actual = fs::read(out.join(format!("profile/{dtype}.slab")))?;
        assert_eq!(expected.len(), actual.len());
        if dtype == "f32" {
            for (c, profile) in profiles.iter().enumerate() {
                for (k, value) in profile.iter().enumerate() {
                    let p = ((offset + 3 + k) * 2 + c) * 4;
                    assert_eq!(&expected[p..p + 4], &298f32.to_le_bytes());
                    expected[p..p + 4].copy_from_slice(&(*value as f32).to_le_bytes());
                    changed += 1;
                }
            }
        }
        assert_eq!(expected, actual, "serialized {dtype} words");
        assert_eq!(prepared["files"][dtype]["sha256"], sha(&actual));
        compared += expected.len() / width;
        fs::write(out.join(format!("expected-{dtype}.slab")), expected)?;
    }
    let bad = [
        json!([270.]),
        json!([270., 271., 272.]),
        json!([270., 270., 270., 270., 270.]),
        json!([270., 0., 272., 273.]),
        json!([270., -1., 272., 273.]),
        json!([270., "NaN", 272., 273.]),
        json!([270., "inf", 272., 273.]),
        json!([270., 1e99, 272., 273.]),
    ];
    let mut controls = vec![];
    for (i, profile) in bad.into_iter().enumerate() {
        let name = format!("invalid-{i}");
        let mut request = warm.clone();
        request["catchments"][0]["overrides"]["soil_temperature_k"] = profile;
        let error = run(&name, &request).unwrap_err();
        assert!(!out.join(&name).exists());
        controls.push(json!({"name":name,"refusal":error,"destination_absent":true}));
    }
    let receipt = json!({"status":"PASS","scope":"synthetic profiles through actual Rust initializer; no native warm-start or snow mapping claim",
        "compared_words":compared,"differing_words":0,"changed_soil_words":changed,
        "controls":controls,"wall_ms":started.elapsed().as_secs_f64()*1000.,
        "threads":rayon::current_num_threads(),"source_revision":prepared["source_revision"]});
    fs::write(out.join("check.json"), serde_json::to_vec_pretty(&receipt)?)?;
    println!("{receipt}");
    Ok(())
}
