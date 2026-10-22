use crate::{finite, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Continuous,
    Integer,
    Categorical,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Parameter {
    pub name: String,
    pub kind: Kind,
    #[serde(default)]
    pub min: Option<f64>,
    #[serde(default)]
    pub max: Option<f64>,
    #[serde(default)]
    pub values: Option<Vec<Value>>,
    pub default: Value,
    #[serde(default)]
    pub schemes: Vec<Value>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampleRequest {
    pub parameters: Vec<Parameter>,
    pub count: usize,
    pub seed: u64,
    #[serde(default = "default_prefix")]
    pub id_prefix: String,
}

fn default_prefix() -> String {
    "member".into()
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Member {
    pub id: String,
    pub parameters: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SampleResult {
    pub members: Vec<Member>,
    pub seed: u64,
    pub method: &'static str,
    pub applicability: BTreeMap<String, Vec<Value>>,
}

// SplitMix64 makes the seeded design reproducible across platforms and builds.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut x = self.0;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / 9_007_199_254_740_992.0
    }

    fn index(&mut self, upper: usize) -> usize {
        let upper = upper as u64;
        let limit = u64::MAX - u64::MAX % upper;
        loop {
            let value = self.next();
            if value < limit {
                return (value % upper) as usize;
            }
        }
    }

    fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.index(i + 1);
            items.swap(i, j);
        }
    }
}

fn numeric_bounds(parameter: &Parameter) -> Result<(f64, f64)> {
    let lo = finite(
        parameter
            .min
            .ok_or_else(|| format!("{} needs min", parameter.name))?,
        "min",
    )?;
    let hi = finite(
        parameter
            .max
            .ok_or_else(|| format!("{} needs max", parameter.name))?,
        "max",
    )?;
    if lo >= hi || !(hi - lo).is_finite() {
        return Err(format!(
            "{} needs a finite nonzero increasing range",
            parameter.name
        ));
    }
    let default = parameter
        .default
        .as_f64()
        .ok_or_else(|| format!("{} needs a numeric default", parameter.name))?;
    if !default.is_finite() || default < lo || default > hi {
        return Err(format!("{} default is outside its range", parameter.name));
    }
    if parameter.values.is_some() {
        return Err(format!(
            "{} numeric range cannot also specify values",
            parameter.name
        ));
    }
    if matches!(parameter.kind, Kind::Integer)
        && (lo.fract() != 0.0
            || hi.fract() != 0.0
            || default.fract() != 0.0
            || lo.abs() > 9_007_199_254_740_991.0
            || hi.abs() > 9_007_199_254_740_991.0)
    {
        return Err(format!(
            "{} integer bounds and default must be exact integers",
            parameter.name
        ));
    }
    Ok((lo, hi))
}

fn category_values(parameter: &Parameter) -> Result<&[Value]> {
    if parameter.min.is_some() || parameter.max.is_some() {
        return Err(format!(
            "{} categorical values cannot also specify a range",
            parameter.name
        ));
    }
    let values = parameter
        .values
        .as_deref()
        .ok_or_else(|| format!("{} needs categorical values", parameter.name))?;
    if values.is_empty() {
        return Err(format!("{} has no categorical values", parameter.name));
    }
    let mut distinct = BTreeSet::new();
    for value in values {
        if value.is_null()
            || value.is_array()
            || value.is_object()
            || !distinct.insert(serde_json::to_string(value).map_err(|e| e.to_string())?)
        {
            return Err(format!(
                "{} needs unique scalar categorical values",
                parameter.name
            ));
        }
    }
    if !values.contains(&parameter.default) {
        return Err(format!("{} default is not a category", parameter.name));
    }
    Ok(values)
}

pub fn sample(request: SampleRequest) -> Result<SampleResult> {
    if request.count == 0 || request.parameters.is_empty() || request.id_prefix.trim().is_empty() {
        return Err(
            "sample needs a positive count, a nonempty parameter table and an ID prefix".into(),
        );
    }
    let mut names = BTreeSet::new();
    for parameter in &request.parameters {
        if parameter.name.trim().is_empty() || !names.insert(parameter.name.clone()) {
            return Err("parameter names must be nonempty and unique".into());
        }
        let mut schemes = BTreeSet::new();
        if parameter.schemes.iter().any(|s| {
            s.is_null()
                || s.is_array()
                || s.is_object()
                || s.as_str().is_some_and(|label| label.trim().is_empty())
                || !schemes.insert(s.to_string())
        }) {
            return Err(format!(
                "{} schemes must be nonempty and unique",
                parameter.name
            ));
        }
        match parameter.kind {
            Kind::Categorical => {
                category_values(parameter)?;
            }
            _ => {
                numeric_bounds(parameter)?;
            }
        }
    }
    let mut random = Random(request.seed);
    let mut members: Vec<Member> = (0..request.count)
        .map(|i| Member {
            id: format!("{}-{:03}", request.id_prefix, i + 1),
            parameters: BTreeMap::new(),
        })
        .collect();
    let mut applicability = BTreeMap::new();
    for parameter in &request.parameters {
        applicability.insert(parameter.name.clone(), parameter.schemes.clone());
        let mut values = match parameter.kind {
            Kind::Categorical => {
                let categories = category_values(parameter)?;
                // Rotate the remainder before shuffling so repeated small designs
                // do not always favor the first category in the table.
                let offset = random.index(categories.len());
                (0..request.count)
                    .map(|i| categories[(i + offset) % categories.len()].clone())
                    .collect()
            }
            Kind::Continuous | Kind::Integer => {
                let (lo, hi) = numeric_bounds(parameter)?;
                (0..request.count)
                    .map(|i| {
                        let unit = (i as f64 + random.unit()) / request.count as f64;
                        let value = if matches!(parameter.kind, Kind::Integer) {
                            (lo + unit * (hi - lo + 1.0)).floor().min(hi)
                        } else {
                            lo + unit * (hi - lo)
                        };
                        if matches!(parameter.kind, Kind::Integer) {
                            Value::from(value as i64)
                        } else {
                            Value::from(value)
                        }
                    })
                    .collect::<Vec<_>>()
            }
        };
        random.shuffle(&mut values);
        for (member, value) in members.iter_mut().zip(values) {
            member.parameters.insert(parameter.name.clone(), value);
        }
    }
    Ok(SampleResult {
        members,
        seed: request.seed,
        method: "seeded_latin_hypercube_balanced_categories",
        applicability,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request(seed: u64) -> SampleRequest {
        serde_json::from_value(json!({"count": 24, "seed": seed, "parameters": [
            {"name":"coefficient","kind":"continuous","min":0.2,"max":1.8,"default":1.0},
            {"name":"mixing","kind":"integer","min":0,"max":2,"default":1},
            {"name":"land_model","kind":"categorical","values":["a","b","c"],"default":"a"}
        ]}))
        .unwrap()
    }

    #[test]
    fn reproducible_latin_hypercube_and_balanced_categories() {
        let a = sample(request(17)).unwrap();
        let b = sample(request(17)).unwrap();
        assert_eq!(a.members, b.members);
        assert_ne!(a.members, sample(request(18)).unwrap().members);
        let mut bins = BTreeSet::new();
        let mut categories = BTreeMap::new();
        for member in &a.members {
            let x = member.parameters["coefficient"].as_f64().unwrap();
            bins.insert(((x - 0.2) / 1.6 * 24.0).floor() as usize);
            let integer = member.parameters["mixing"].as_f64().unwrap();
            assert_eq!(integer.fract(), 0.0);
            assert!((0.0..=2.0).contains(&integer));
            *categories
                .entry(member.parameters["land_model"].as_str().unwrap())
                .or_insert(0) += 1;
        }
        assert_eq!(bins, (0..24).collect());
        assert!(categories.values().all(|&count| count == 8));
    }

    #[test]
    fn invalid_defaults_and_duplicate_parameters_are_rejected() {
        let mut r = request(2);
        r.parameters[0].default = json!(3);
        assert!(sample(r).unwrap_err().contains("outside"));
        let mut r = request(2);
        r.parameters.push(r.parameters[0].clone());
        assert!(sample(r).unwrap_err().contains("unique"));
    }

    #[test]
    fn numeric_scheme_selectors_and_descriptive_table_metadata_are_accepted() {
        let request = serde_json::from_value(json!({"count": 4, "seed": 8, "parameters": [
            {"name":"surface_scale","kind":"continuous","min":0.5,"max":1.5,"default":1,
             "schemes":[2,3,4],"products":["city","research"],"units":"fraction"}
        ]}))
        .unwrap();
        let result = sample(request).unwrap();
        assert_eq!(
            result.applicability["surface_scale"],
            vec![json!(2), json!(3), json!(4)]
        );
        assert_eq!(result.members.len(), 4);
    }
}
