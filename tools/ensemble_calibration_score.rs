//! Native scores for collocated ensemble forecasts and observations.
//!
//! Compile with rustc --edition 2021 -O tools/ensemble_calibration_score.rs.
//! Input TSV header: sample_id, weight, observed, then distinct member labels.
//! Columns use tabs. Missing observations or incomplete member rows are counted
//! and omitted, never filled with zeros. Geometry and observation QC belong to
//! the native observation readers and remapper before this scoring seam.
//!
//! CRPS is the empirical-distribution score, with pair denominator N*N.
//! Fair CRPS uses N*(N-1), and is reported separately for exchangeable members.
//! Spread is RMS sample standard deviation (ddof=1); the corrected ratio
//! multiplies spread/RMSE by sqrt((N+1)/N). Neither correction removes biased
//! sampling, unequal quality, or dependence between lagged/model members.
use std::collections::HashSet;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Debug)]
struct Scores {
    members: usize,
    thresholds: Vec<f64>,
    samples: usize,
    missing_observations: usize,
    missing_members: usize,
    weight: f64,
    squared_error: f64,
    bias: f64,
    variance: f64,
    crps: f64,
    fair_crps: f64,
    ranks: Vec<f64>,
    brier: Vec<f64>,
    events: Vec<f64>,
    reliability_weights: Vec<Vec<f64>>,
    reliability_events: Vec<Vec<f64>>,
}

impl Scores {
    fn new(members: usize, thresholds: Vec<f64>) -> Result<Self, String> {
        if members < 1 {
            return Err("at least one forecast member is required".into());
        }
        if thresholds.iter().any(|v| !v.is_finite()) {
            return Err("thresholds must be finite".into());
        }
        let k = thresholds.len();
        Ok(Self { members, thresholds, samples: 0, missing_observations: 0,
            missing_members: 0, weight: 0.0, squared_error: 0.0, bias: 0.0,
            variance: 0.0, crps: 0.0, fair_crps: 0.0,
            ranks: vec![0.0; members + 1], brier: vec![0.0; k],
            events: vec![0.0; k],
            reliability_weights: vec![vec![0.0; members + 1]; k],
            reliability_events: vec![vec![0.0; members + 1]; k] })
    }

    fn add(&mut self, weight: f64, observed: f64, members: &[f64]) -> Result<(), String> {
        if !weight.is_finite() || weight <= 0.0 {
            return Err("sample weight must be finite and positive".into());
        }
        if members.len() != self.members {
            return Err("every row must have the declared member count".into());
        }
        if !observed.is_finite() {
            self.missing_observations += 1;
            return Ok(());
        }
        if members.iter().any(|x| !x.is_finite()) {
            self.missing_members += 1;
            return Ok(());
        }
        let n = self.members as f64;
        let mean = members.iter().sum::<f64>() / n;
        let error = mean - observed;
        let absolute_error = members.iter().map(|x| (x - observed).abs()).sum::<f64>() / n;
        let mut sorted = members.to_vec();
        sorted.sort_by(|a,b| a.partial_cmp(b).unwrap());
        // sum_{i<j}|x_i-x_j| from the sorted order, with no N*N allocation.
        let pair_sum = sorted.windows(2).enumerate().map(|(i,x)| {
            (i + 1) as f64 * (self.members - i - 1) as f64 * (x[1] - x[0])
        }).sum::<f64>();
        self.samples += 1;
        self.weight += weight;
        self.squared_error += weight * error * error;
        self.bias += weight * error;
        self.crps += weight * (absolute_error - pair_sum / (n * n));
        if self.members > 1 {
            self.variance += weight * members.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
            self.fair_crps += weight * (absolute_error - pair_sum / (n * (n - 1.0)));
        }
        let below = members.iter().filter(|&&x| x < observed).count();
        let tied = members.iter().filter(|&&x| x == observed).count();
        for rank in below..=below + tied {
            self.ranks[rank] += weight / (tied + 1) as f64;
        }
        for (i,threshold) in self.thresholds.iter().enumerate() {
            let count = members.iter().filter(|&&x| x >= *threshold).count();
            let p = count as f64 / n;
            let event = if observed >= *threshold { 1.0 } else { 0.0 };
            self.brier[i] += weight * (p - event).powi(2);
            self.events[i] += weight * event;
            self.reliability_weights[i][count] += weight;
            self.reliability_events[i][count] += weight * event;
        }
        Ok(())
    }

    fn json(&self) -> Result<String, String> {
        if self.samples == 0 {
            return Err("no complete forecast-observation pairs; no scores were measured".into());
        }
        if [self.weight, self.squared_error, self.bias, self.variance,
            self.crps, self.fair_crps].iter().chain(self.ranks.iter())
            .chain(self.brier.iter()).any(|v| !v.is_finite()) {
            return Err("score arithmetic overflowed; input values or weights are out of range".into());
        }
        let rmse = (self.squared_error / self.weight).sqrt();
        let spread = if self.members > 1 { Some((self.variance / self.weight).sqrt()) } else { None };
        let ratio = spread.and_then(|x| if rmse > 0.0 { Some(x / rmse) } else { None });
        let corrected = ratio.map(|x| x * ((self.members + 1) as f64 / self.members as f64).sqrt());
        let threshold_scores: Vec<String> = self.thresholds.iter().enumerate().map(|(i,t)| {
            let bins: Vec<String> = (0..=self.members).map(|k| {
                let w = self.reliability_weights[i][k];
                let observed_frequency = if w > 0.0 { Some(self.reliability_events[i][k] / w) } else { None };
                format!("{{\"probability\":{},\"weight\":{},\"observed_frequency\":{}}}", k as f64 / self.members as f64, w, number(observed_frequency))
            }).collect();
            format!("{{\"threshold\":{},\"event\":\"value >= threshold\",\"brier_score\":{},\"observed_frequency\":{},\"reliability\":[{}]}}",t,self.brier[i] / self.weight,self.events[i] / self.weight,bins.join(","))
        }).collect();
        Ok(format!(concat!("{{\"schema\":\"gpuwm-ensemble-calibration.scores.v1\",",
            "\"members\":{},\"samples\":{},\"missing_observations\":{},\"missing_members\":{},\"weight\":{},",
            "\"ensemble_mean_rmse\":{},\"ensemble_mean_bias\":{},\"rms_member_sample_spread\":{},",
            "\"spread_skill_ratio\":{},\"finite_ensemble_corrected_spread_skill_ratio\":{},",
            "\"crps\":{},\"fair_crps\":{},\"rank_ties\":\"fractional uniform among admissible ranks\",",
            "\"rank_weight\":{:?},\"threshold_scores\":[{}]}}"),
            self.members,self.samples,self.missing_observations,self.missing_members,self.weight,
            rmse,self.bias / self.weight,number(spread),number(ratio),number(corrected),
            self.crps / self.weight,number(if self.members > 1 {Some(self.fair_crps / self.weight)} else {None}),
            self.ranks,threshold_scores.join(",")))
    }
}

fn number(value: Option<f64>) -> String {
    value.filter(|x| x.is_finite()).map_or_else(|| "null".into(), |x| x.to_string())
}

fn score(path: &str, thresholds: Vec<f64>) -> Result<String,String> {
    let input = File::open(path).map_err(|e| e.to_string())?;
    let mut lines = BufReader::new(input).lines();
    let header = lines.next().ok_or("input is empty")?.map_err(|e|e.to_string())?;
    let columns: Vec<&str> = header.split('\t').collect();
    if columns.len() < 4 || columns[..3] != ["sample_id", "weight", "observed"] {
        return Err("TSV header must be sample_id, weight, observed, then member labels".into());
    }
    let mut labels = HashSet::new();
    for label in &columns[3..] {
        if label.is_empty() || !labels.insert(*label) {
            return Err("member labels must be nonempty and unique".into());
        }
    }
    let mut scores = Scores::new(columns.len()-3, thresholds)?;
    let mut ids = HashSet::new();
    for (line_no, raw) in lines.enumerate() {
        let raw = raw.map_err(|e| e.to_string())?;
        let cells: Vec<&str> = raw.split('\t').collect();
        if cells.len() != columns.len() {
            return Err(format!("line {} has {} columns, expected {}",line_no+2,cells.len(),columns.len()));
        }
        if cells[0].is_empty() || !ids.insert(cells[0].to_string()) {
            return Err(format!("line {} has an empty or repeated sample id",line_no+2));
        }
        let values: Result<Vec<f64>,String> = cells[1..].iter().map(|s| s.parse::<f64>().map_err(|_|format!("line {} has a nonnumeric value",line_no+2))).collect();
        let values = values?;
        scores.add(values[0], values[1], &values[2..])?;
    }
    scores.json()
}

fn run() -> Result<(),String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        return Err("usage: ensemble_calibration_score MATCHED.tsv THRESHOLDS_COMMA_SEPARATED (or - for none)".into());
    }
    let thresholds = if args[2] == "-" { Vec::new() } else {
        args[2].split(',').map(|s|s.parse::<f64>().map_err(|_|"threshold is not numeric".into())).collect::<Result<Vec<f64>,String>>()?
    };
    println!("{}",score(&args[1],thresholds)?);
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}
