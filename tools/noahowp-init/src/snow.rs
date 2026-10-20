// SPDX-License-Identifier: Apache-2.0
// Aggregate cold-pack recipe. DIVIDE/COMBO follow the pinned Apache Noah-OWP
// SnowLayerChange.f90, revision 0242a96c. This is initialization, not a claim
// to recover an unobserved liquid fraction, snow age or temperature profile.
use super::*;

#[derive(Clone, Copy, Default)]
struct Layer {
    dz: f32,
    ice: f32,
    t: f32,
}

fn combo(a: &mut Layer, b: Layer, cice: f32, tfrz: f32) {
    let ice = a.ice + b.ice;
    let h = (cice * a.ice) * (a.t - tfrz);
    let h2 = (cice * b.ice) * (b.t - tfrz);
    let hc = h + h2;
    // All-ice specialization of native COMBO, including its rounding order.
    a.t = if hc < 0. {
        tfrz + hc / (cice * ice)
    } else if hc <= 0. {
        tfrz
    } else {
        tfrz + hc / (cice * ice)
    };
    a.dz += b.dz;
    a.ice = ice;
}

pub(super) fn initialize(state: &mut State, values: &Values) -> Result<&'static str> {
    let keys = [
        "snow_water_equivalent_mm",
        "snow_depth_m",
        "snow_temperature_k",
    ];
    if keys.iter().all(|k| !values.contains_key(*k)) {
        return Ok("native-cold-start");
    }
    let [swe, depth, temperature] = [
        number(values, keys[0])?,
        number(values, keys[1])?,
        number(values, keys[2])?,
    ];
    if !swe.is_finite()
        || !depth.is_finite()
        || !temperature.is_finite()
        || swe < 0.
        || depth < 0.
        || temperature <= 0.
    {
        return Err(
            "snow state requires finite nonnegative SWE/depth and positive Kelvin temperature"
                .into(),
        );
    }
    if (swe == 0.) != (depth == 0.) {
        return Err(
            "snow SWE/depth must both be zero or both positive; otherwise snow mass has no volume"
                .into(),
        );
    }
    if swe == 0. {
        return Ok("aggregate-all-ice-divide-v1");
    }
    let tfrz = take(state, "parameters.tfrz");
    if temperature > tfrz || swe / depth > take(state, "parameters.denice") {
        return Err("all-ice snow recipe requires temperature <= TFRZ and bulk density <= DENICE; cannot infer a liquid pack".into());
    }
    let mut layers = [Layer::default(); 3];
    let mut count = 0;
    // Pinned SnowFall starts a layer at 0.025 m. Shallower snow uses native
    // aggregate storage with ISNOW=0 and no snow thermal layers.
    if depth >= 0.025 {
        count = 1;
        layers[0] = Layer {
            dz: depth,
            ice: swe,
            t: temperature,
        };
        if layers[0].dz > 0.05 {
            count = 2;
            layers[0].dz /= 2.;
            layers[0].ice /= 2.;
            layers[1] = layers[0];
        }
        if count > 1 && layers[0].dz > 0.05 {
            let drr = layers[0].dz - 0.05;
            let removed = (drr / layers[0].dz) * layers[0].ice;
            layers[0].ice *= 0.05 / layers[0].dz;
            layers[0].dz = 0.05;
            let moved = Layer {
                dz: drr,
                ice: removed,
                t: layers[0].t,
            };
            combo(&mut layers[1], moved, take(state, "parameters.cice"), tfrz);
            if layers[1].dz > 0.20 {
                count = 3;
                let dtdz = (layers[0].t - layers[1].t) / ((layers[0].dz + layers[1].dz) / 2.);
                layers[1].dz /= 2.;
                layers[1].ice /= 2.;
                layers[2] = layers[1];
                layers[2].t = layers[1].t - dtdz * layers[1].dz / 2.;
                if layers[2].t >= tfrz {
                    layers[2].t = layers[1].t;
                } else {
                    layers[1].t += dtdz * layers[1].dz / 2.;
                }
            }
        }
        if count > 2 && layers[1].dz > 0.2 {
            let drr = layers[1].dz - 0.2;
            let removed = (drr / layers[1].dz) * layers[1].ice;
            layers[1].ice *= 0.2 / layers[1].dz;
            layers[1].dz = 0.2;
            let moved = Layer {
                dz: drr,
                ice: removed,
                t: layers[1].t,
            };
            combo(&mut layers[2], moved, take(state, "parameters.cice"), tfrz);
        }
    }
    put(state, "water.isnow", -(count as f32))?;
    put(state, "water.sneqv", swe)?;
    put(state, "water.sneqvo", swe)?;
    put(state, "water.snowh", depth)?;
    for k in ["water.snice", "water.snliq", "water.ficeold"] {
        put(state, k, 0.)?;
    }
    let mut z = 0f32;
    for j in 0..3 {
        let active = j >= 3 - count;
        let layer = if active {
            layers[j - (3 - count)]
        } else {
            Layer::default()
        };
        state.get_mut("domain.dzsnso").unwrap()[j] = layer.dz as f64;
        state.get_mut("water.snice").unwrap()[j] = layer.ice as f64;
        state.get_mut("water.ficeold").unwrap()[j] = if active { 1. } else { 0. };
        state.get_mut("energy.stc").unwrap()[j] = layer.t as f64;
        z -= layer.dz;
        state.get_mut("domain.zsnso").unwrap()[j] = z as f64;
    }
    for j in 3..7 {
        z -= state["domain.dzsnso"][j] as f32;
        state.get_mut("domain.zsnso").unwrap()[j] = z as f64;
    }
    // Native SnowWater's mass-weighted diagnostic; shallow snow has no layers.
    let mut heat = 0f32;
    let mut mass = 0f32;
    for layer in &layers[..count] {
        heat += layer.t * layer.ice;
        mass += layer.ice;
    }
    put(
        state,
        "energy.snowt_avg",
        if count > 0 { heat / mass } else { -999999. },
    )?;
    for name in [
        "domain.dzsnso",
        "domain.zsnso",
        "water.snice",
        "energy.stc",
        "energy.snowt_avg",
    ] {
        if state[name].iter().any(|v| !v.is_finite()) {
            return Err("snow layer recipe exceeds finite native REAL representation".into());
        }
    }
    Ok("aggregate-all-ice-divide-v1")
}
