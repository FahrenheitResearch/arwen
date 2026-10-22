//! Native visibility at the same station support as other ASOS quantities.
use super::*;
pub fn field(file: &WrfFile) -> Result<Vec<f64>> {
    let read = |name: &str| file.read_var(name, 0).map(|v| v.to_vec());
    let optional = |name: &str| -> Result<Option<Vec<f64>>> {
        Ok(if file.has_var(name) {
            Some(read(name)?)
        } else {
            None
        })
    };
    let p = file.full_pressure(0)?;
    let t = file.temperature(0)?;
    let q = read("QVAPOR")?;
    let c = read("QCLOUD")?;
    let r = optional("QRAIN")?;
    let i = optional("QICE")?;
    let s = optional("QSNOW")?;
    let g = optional("QGRAUP")?;
    let out =
        rustwx_calc::visibility::compute_visibility(rustwx_calc::visibility::VisibilityInputs {
            grid: rustwx_calc::GridShape::new(file.nx, file.ny)?,
            nz: file.nz,
            pressure_pa: &p,
            temperature_k: &t,
            vapor_kgkg: &q,
            cloud_kgkg: &c,
            rain_kgkg: r.as_deref(),
            ice_kgkg: i.as_deref(),
            snow_kgkg: s.as_deref(),
            graupel_kgkg: g.as_deref(),
        })?;
    Ok(out.visibility_m)
}

/// Station scores use a common, complete roster and strict low-visibility events.
pub fn scores(observed: &[f64], covered: &[bool], members: &[Vec<f64>]) -> Value {
    let valid: Vec<usize> = (0..observed.len())
        .filter(|&i| {
            covered[i] && observed[i].is_finite() && members.iter().all(|m| m[i].is_finite())
        })
        .collect();
    let thresholds: Vec<Value> = [1000.0, 400.0]
        .into_iter()
        .map(|threshold| {
            let (mut hits, mut misses, mut false_alarms, mut correct_negatives) =
                (0u64, 0u64, 0u64, 0u64);
            let mut brier = 0.0;
            for &i in &valid {
                let event = observed[i] < threshold;
                let probability = members.iter().filter(|m| m[i] < threshold).count() as f64
                    / members.len() as f64;
                brier += (probability - u8::from(event) as f64).powi(2);
                match (event, probability >= 0.5) {
                    (true, true) => hits += 1,
                    (true, false) => misses += 1,
                    (false, true) => false_alarms += 1,
                    (false, false) => correct_negatives += 1,
                }
            }
            let ratio = |a: u64, b: u64| (b > 0).then_some(a as f64 / b as f64);
            json!({"threshold_m":threshold,"comparison":"lt","samples":valid.len(),
            "observed_events":hits+misses,"hits":hits,"misses":misses,
            "false_alarms":false_alarms,"correct_negatives":correct_negatives,
            "contingency_probability_cutoff":0.5,
            "brier_score":(!valid.is_empty()).then_some(brier / valid.len() as f64),
            "pod":ratio(hits,hits+misses),"far":ratio(false_alarms,hits+false_alarms),
            "csi":ratio(hits,hits+misses+false_alarms)})
        })
        .collect();
    json!({"thresholds":thresholds,"roster":"all original members finite at common observed stations",
        "metar_cap_m":16093.44})
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn strict_thresholds_and_complete_roster_are_scored() {
        let obs = [399.0, 400.0, 999.0, 1000.0, 0.0];
        let members = vec![
            vec![399.0, 400.0, 999.0, 1000.0, f64::NAN],
            vec![399.0, 400.0, 999.0, 1000.0, 0.0],
        ];
        let score = scores(&obs, &[true; 5], &members);
        let limits = score["thresholds"].as_array().unwrap();
        assert_eq!(limits[0]["samples"], 4);
        assert_eq!(limits[0]["observed_events"], 3);
        assert_eq!(limits[1]["observed_events"], 1);
        assert_eq!(limits[0]["brier_score"], 0.0);
        assert_eq!(limits[1]["brier_score"], 0.0);
    }
}
