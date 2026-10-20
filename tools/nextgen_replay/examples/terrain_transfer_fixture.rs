// SPDX-License-Identifier: Apache-2.0
// Closed-form carrier expectations, not a native or physical-event referee.
use nextgen_replay::file_sha256;
use serde_json::{json, Map, Value};
use std::{fs, path::Path, time::Instant};

fn raw(root: &Path, name: &str, dtype: &str, shape: Value, bytes: Vec<u8>) -> Value {
    let path = root.join(format!("{name}.raw"));
    fs::write(&path, bytes).unwrap();
    json!({"path":path,"sha256":file_sha256(&path).unwrap(),
           "dtype":dtype,"shape":shape,"order":"c"})
}

fn main() {
    let root = std::path::PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("new scratch fixture directory"),
    );
    fs::create_dir(&root).unwrap();
    let started = Instant::now();
    if std::env::args().nth(2).as_deref() == Some("intervals") {
        interval_fixture(&root);
        println!(
            "{}",
            json!({"stage":"interval-fixture-generation",
            "wall_ms":started.elapsed().as_secs_f64()*1000., "parallelism":"one Rust process"})
        );
        return;
    }
    let mut cases = Vec::new();
    // dx=2, dt=1, area=4. No neighbours or external source rates.
    // Surface: depth=head drop=1 => Manning drain rate=1/4 m/s.
    // Ground: thickness=head drop=1/2 => Darcy drain rate=1/16 m/s.
    // Exfiltration: initial storage3/4 minus capacity1/2 => spill1/4 m.
    for (name, z, h, w, sy, k, width, next_h, next_w, ds, dg, ex) in [
        (
            "surface",
            0.,
            1.,
            0.,
            1.,
            0.,
            1.,
            3. / 4.,
            0.,
            1. / 4.,
            0.,
            0.,
        ),
        (
            "ground",
            1.,
            0.,
            1. / 2.,
            1.,
            1.,
            1.,
            0.,
            7. / 16.,
            0.,
            1. / 16.,
            0.,
        ),
        (
            "exfil",
            0.,
            0.,
            3. / 4.,
            1. / 2.,
            0.,
            0.,
            1. / 4.,
            1. / 2.,
            0.,
            0.,
            1. / 4.,
        ),
    ] {
        let mut inputs = Map::new();
        for (field, value) in [
            ("elevation", z),
            ("surface_depth", h),
            ("groundwater_storage", w),
            ("specific_yield", sy),
            ("conductivity", k),
            ("outlet_width", width),
            ("soil_depth", 1.),
            ("manning", 1.),
            ("outlet_stage", 0.),
            ("surface_source", 0.),
            ("recharge", 0.),
        ] {
            inputs.insert(
                field.into(),
                raw(
                    &root,
                    &format!("{name}-initial-{field}"),
                    "f32",
                    json!([1, 1]),
                    (value as f32).to_le_bytes().to_vec(),
                ),
            );
        }
        inputs.insert(
            "active".into(),
            raw(
                &root,
                &format!("{name}-active"),
                "i32",
                json!([1, 1]),
                1i32.to_le_bytes().to_vec(),
            ),
        );
        let mut expected = Map::new();
        for (field, value) in [("surface_depth", next_h), ("groundwater_storage", next_w)] {
            expected.insert(
                field.into(),
                raw(
                    &root,
                    &format!("{name}-expected-{field}"),
                    "f32",
                    json!([1, 1]),
                    (value as f32).to_le_bytes().to_vec(),
                ),
            );
        }
        for (field, rate) in [
            ("surface_outlet_volume", ds),
            ("groundwater_outlet_volume", dg),
            ("exfiltration_volume", ex),
        ] {
            inputs.insert(
                field.into(),
                raw(
                    &root,
                    &format!("{name}-initial-{field}"),
                    "f64",
                    json!([1, 1]),
                    0.125f64.to_le_bytes().to_vec(),
                ),
            );
            expected.insert(
                field.into(),
                raw(
                    &root,
                    &format!("{name}-expected-{field}"),
                    "f64",
                    json!([1, 1]),
                    (0.125f64 + 4. * rate).to_le_bytes().to_vec(),
                ),
            );
        }
        expected.insert(
            "health".into(),
            raw(
                &root,
                &format!("{name}-healthy"),
                "u8",
                json!([4]),
                0u32.to_le_bytes().to_vec(),
            ),
        );
        cases.push(json!({"name":name,"inputs":inputs,"expected":expected}));
    }
    let fixture = json!({"schema":"nextgen.terrain-transfer.fixture/v1",
        "origin":"analytical-carrier-check","dx":2.,"dt":1.,"period_seconds":1.,
        "scope":"Three independent one-cell transfer checks; nonzero seeded cumulative volumes; no native replay, flood event or observation skill",
        "cases":cases});
    fs::write(
        root.join("fixture.json"),
        serde_json::to_vec_pretty(&fixture).unwrap(),
    )
    .unwrap();
    println!(
        "{}",
        json!({"stage":"fixture-generation","wall_ms":started.elapsed().as_secs_f64()*1000.,
        "parallelism":"one Rust process; three closed-form one-cell cases"})
    );
}

fn interval_fixture(root: &Path) {
    let scalar = |name: &str, value: f64, double: bool| {
        if double {
            raw(
                root,
                name,
                "f64",
                json!([1, 1]),
                value.to_le_bytes().to_vec(),
            )
        } else {
            raw(
                root,
                name,
                "f32",
                json!([1, 1]),
                (value as f32).to_le_bytes().to_vec(),
            )
        }
    };
    let map_offsets = raw(
        root,
        "offsets",
        "i64",
        json!([4]),
        [0i64, 0, 1, 1]
            .into_iter()
            .flat_map(i64::to_le_bytes)
            .collect(),
    );
    let map_indices = raw(
        root,
        "indices",
        "i32",
        json!([1]),
        0i32.to_le_bytes().to_vec(),
    );
    let mut cases = Vec::new();
    for (name, z, h, w, sy, k, width, source) in [
        ("surface", 0., 0.75, 0., 1., 0., 1., 0.25),
        ("ground", 1., 0., 0.5, 1., 1., 1., 0.),
        ("exfil", 0., 0., 0.75, 0.5, 0., 0., 0.),
    ] {
        let mut inputs = Map::new();
        for (field, value) in [
            ("elevation", z),
            ("surface_depth", h),
            ("groundwater_storage", w),
            ("specific_yield", sy),
            ("conductivity", k),
            ("outlet_width", width),
            ("soil_depth", 1.),
            ("manning", 1.),
            ("outlet_stage", 0.),
            ("surface_source", source),
            ("recharge", 0.),
        ] {
            inputs.insert(
                field.into(),
                scalar(&format!("{name}-initial-{field}"), value, false),
            );
        }
        inputs.insert(
            "active".into(),
            raw(
                root,
                &format!("{name}-active"),
                "i32",
                json!([1, 1]),
                1i32.to_le_bytes().to_vec(),
            ),
        );
        let seed = if name == "surface" {
            2f64.powi(60)
        } else {
            0.125
        };
        for field in [
            "surface_outlet_volume",
            "groundwater_outlet_volume",
            "exfiltration_volume",
        ] {
            inputs.insert(
                field.into(),
                scalar(&format!("{name}-initial-{field}"), seed, true),
            );
        }
        for field in ["surface_interval_volume", "groundwater_interval_volume"] {
            inputs.insert(
                field.into(),
                scalar(&format!("{name}-initial-{field}"), 32., true),
            );
        }
        let mut observations = Vec::new();
        for step in 0..2 {
            // All arithmetic here is binary exact. The huge surface total stays
            // unchanged when1m3 is added, while its direct interval must be1m3.
            let (next_h, next_w, sv, gv, ev, si, gi) = match name {
                "surface" => (0.75, 0., seed, seed, seed, 1., 0.),
                "ground" if step == 0 => (0., 7. / 16., seed, seed + 1. / 4., seed, 0., 1. / 4.),
                "ground" => (
                    0.,
                    399. / 1024.,
                    seed,
                    seed + 1. / 4. + 49. / 256.,
                    seed,
                    0.,
                    49. / 256.,
                ),
                _ => (0.25, 0.5, seed, seed, seed + 1., 0., 0.),
            };
            let mut expected = Map::new();
            for (field, value, double) in [
                ("surface_depth", next_h, false),
                ("groundwater_storage", next_w, false),
                ("surface_outlet_volume", sv, true),
                ("groundwater_outlet_volume", gv, true),
                ("exfiltration_volume", ev, true),
                ("surface_interval_volume", si, true),
                ("groundwater_interval_volume", gi, true),
            ] {
                expected.insert(
                    field.into(),
                    scalar(&format!("{name}-step{step}-{field}"), value, double),
                );
            }
            for field in ["health", "lateral_health"] {
                expected.insert(
                    field.into(),
                    raw(
                        root,
                        &format!("{name}-step{step}-{field}"),
                        "u8",
                        json!([4]),
                        0u32.to_le_bytes().to_vec(),
                    ),
                );
            }
            expected.insert(
                "terrain_lateral".into(),
                raw(
                    root,
                    &format!("{name}-step{step}-lateral"),
                    "f32",
                    json!([3]),
                    [0f32, (si + gi) as f32, 0.]
                        .into_iter()
                        .flat_map(f32::to_le_bytes)
                        .collect(),
                ),
            );
            observations.push(Value::Object(expected));
        }
        cases.push(json!({"name":name,"inputs":inputs,"observations":observations}));
    }
    // Ordered surface-then-ground additions at ascending cells must round to1.
    // Grouping the three tiny surface terms before the midpoint ground term
    // would round to the next f32. Positive volumes only, no cancellation.
    let epsilon = 2f64.powi(-53);
    let midpoint = 1. + 2f64.powi(-24);
    let mut witness = Map::new();
    for (field, values) in [
        ("surface_interval_volume", [epsilon; 3]),
        ("groundwater_interval_volume", [midpoint, 0., 0.]),
    ] {
        witness.insert(
            field.into(),
            raw(
                root,
                &format!("witness-{field}"),
                "f64",
                json!([1, 3]),
                values.into_iter().flat_map(f64::to_le_bytes).collect(),
            ),
        );
    }
    witness.insert(
        "offsets".into(),
        raw(
            root,
            "witness-offsets",
            "i64",
            json!([4]),
            [0i64, 0, 3, 3]
                .into_iter()
                .flat_map(i64::to_le_bytes)
                .collect(),
        ),
    );
    witness.insert(
        "indices".into(),
        raw(
            root,
            "witness-indices",
            "i32",
            json!([3]),
            [0i32, 1, 2]
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect(),
        ),
    );
    let witness_expected = raw(
        root,
        "witness-expected",
        "f32",
        json!([3]),
        [0f32, 1., 0.]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect(),
    );
    let healthy = raw(
        root,
        "witness-healthy",
        "u8",
        json!([4]),
        0u32.to_le_bytes().to_vec(),
    );
    fs::write(root.join("fixture.json"),serde_json::to_vec_pretty(&json!({
        "schema":"nextgen.terrain-transfer.fixture/v2","origin":"analytical-carrier-check",
        "dx":2.,"dt":1.,"period_seconds":1.,"cases":cases,
        "mapping":{"offsets":map_offsets,"indices":map_indices},
        "ordered_witness":{"inputs":witness,"expected":witness_expected,"healthy":healthy},
        "scope":"Two advances, direct interval reset, large cumulative rounding and ordered CSR producer; no MC/reservoir binding or physical-event proof"
    })).unwrap()).unwrap();
}
