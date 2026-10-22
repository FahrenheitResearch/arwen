use crate::{finite, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Row {
    pub member: String,
    pub day: String,
    pub station: String,
    pub valid: String,
    pub period: Period,
    pub forecast: f64,
    pub observation: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Period {
    Day,
    Night,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub train_days: Vec<String>,
    pub heldout_days: Vec<String>,
    pub baseline_member: String,
    pub expected_members: Vec<String>,
    pub minimum_pairs: usize,
    /// Controls can be omitted from the spread without changing the score mask.
    #[serde(default)]
    pub ensemble_members: Option<Vec<String>>,
    #[serde(default)]
    pub frozen_winner: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScoreRequest {
    pub rows: Vec<Row>,
    pub config: Config,
}

#[derive(Clone, Debug, Serialize)]
pub struct Metrics {
    pub rmse: f64,
    pub bias: f64,
    pub count: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct MemberDayPeriod {
    pub member: String,
    pub day: String,
    pub period: &'static str,
    #[serde(flatten)]
    pub metrics: Metrics,
}

#[derive(Clone, Debug, Serialize)]
pub struct Aggregate {
    pub mean_daily_rmse: f64,
    pub mean_daily_bias: f64,
    pub days: usize,
    pub count: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct Rank {
    pub member: String,
    pub training_score: f64,
    pub heldout_score: Option<f64>,
    pub training: Aggregate,
    pub heldout: Option<Aggregate>,
}

#[derive(Clone, Debug, Serialize)]
pub struct MaskAudit {
    pub day: String,
    pub common_pairs: usize,
    pub common_day_pairs: usize,
    pub common_night_pairs: usize,
    pub input_pairs: BTreeMap<String, usize>,
    pub dropped_pairs: BTreeMap<String, usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Spread {
    pub count: usize,
    pub members: usize,
    pub rms_sample_spread: f64,
    pub ensemble_mean_rmse: f64,
    pub ensemble_mean_bias: f64,
    /// Undefined when the ensemble mean is exactly correct.
    pub spread_to_skill: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct DailySpread {
    pub day: String,
    #[serde(flatten)]
    pub spread: Spread,
}

#[derive(Clone, Debug, Serialize)]
pub struct Comparison {
    pub baseline_rmse: f64,
    pub winner_rmse: f64,
    pub rmse_gain: f64,
    pub relative_gain: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct DailyComparison {
    pub day: String,
    pub baseline_rmse: f64,
    pub winner_rmse: f64,
    pub wins: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct PeriodComparison {
    pub day: String,
    pub period: &'static str,
    pub baseline_rmse: f64,
    pub winner_rmse: f64,
    pub rmse_gain: f64,
    pub not_worse: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct Decision {
    pub winner: String,
    pub baseline_member: String,
    pub selection_metric: &'static str,
    pub training: Comparison,
    pub heldout: Option<Comparison>,
    pub heldout_days: Vec<DailyComparison>,
    pub heldout_periods: Vec<PeriodComparison>,
    pub heldout_passed: bool,
    pub recommendation: &'static str,
    pub rule: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct ScoreResult {
    pub ranking: Vec<Rank>,
    pub metrics: Vec<MemberDayPeriod>,
    pub masks: Vec<MaskAudit>,
    pub decision: Decision,
    pub train_spread: Spread,
    pub heldout_spread: Option<Spread>,
    pub daily_spread: Vec<DailySpread>,
    pub ensemble_members: Vec<String>,
    pub train_features: FeatureOutput,
}

#[derive(Clone, Debug, Serialize)]
pub struct FeatureLabel {
    pub day: String,
    pub station: String,
    pub period: &'static str,
    pub count: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct FeatureOutput {
    pub members: Vec<String>,
    pub labels: Vec<FeatureLabel>,
    pub predictions: Vec<Vec<f64>>,
    pub observations: Vec<f64>,
    pub units: &'static str,
    pub method: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    day: String,
    station: String,
    valid: String,
}

#[derive(Default)]
struct Sum {
    value: f64,
    correction: f64,
}

impl Sum {
    fn add(&mut self, x: f64) -> Result<()> {
        let y = x - self.correction;
        let next = self.value + y;
        self.correction = (next - self.value) - y;
        self.value = finite(next, "metric sum")?;
        Ok(())
    }
}

#[derive(Default)]
struct Accumulator {
    count: usize,
    error: Sum,
    squared_error: Sum,
}

impl Accumulator {
    fn add(&mut self, forecast: f64, observation: f64) -> Result<()> {
        let error = finite(forecast - observation, "forecast error")?;
        self.error.add(error)?;
        self.squared_error
            .add(finite(error * error, "squared forecast error")?)?;
        self.count += 1;
        Ok(())
    }

    fn finish(self) -> Result<Metrics> {
        if self.count == 0 {
            return Err("cannot score an empty mask".into());
        }
        Ok(Metrics {
            rmse: finite(
                (self.squared_error.value / self.count as f64).sqrt(),
                "RMSE",
            )?,
            bias: finite(self.error.value / self.count as f64, "bias")?,
            count: self.count,
        })
    }
}

fn unique_labels(labels: &[String], name: &str) -> Result<BTreeSet<String>> {
    let mut set = BTreeSet::new();
    if labels.is_empty() {
        return Err(format!("{name} cannot be empty"));
    }
    for label in labels {
        if label.trim().is_empty() || !set.insert(label.clone()) {
            return Err(format!("{name} must be nonempty unique labels"));
        }
    }
    Ok(set)
}

fn aggregate(
    days: &[String],
    member: &str,
    daily: &BTreeMap<(String, String), Metrics>,
) -> Result<Aggregate> {
    let mut rmse = Sum::default();
    let mut bias = Sum::default();
    let mut count = 0;
    for day in days {
        let metrics = daily
            .get(&(member.to_owned(), day.clone()))
            .ok_or_else(|| format!("missing scored member {member} on day {day}"))?;
        rmse.add(metrics.rmse)?;
        bias.add(metrics.bias)?;
        count += metrics.count;
    }
    Ok(Aggregate {
        mean_daily_rmse: rmse.value / days.len() as f64,
        mean_daily_bias: bias.value / days.len() as f64,
        days: days.len(),
        count,
    })
}

fn comparison(winner: &Aggregate, baseline: &Aggregate) -> Comparison {
    let rmse_gain = baseline.mean_daily_rmse - winner.mean_daily_rmse;
    Comparison {
        baseline_rmse: baseline.mean_daily_rmse,
        winner_rmse: winner.mean_daily_rmse,
        rmse_gain,
        relative_gain: (baseline.mean_daily_rmse > 0.0)
            .then(|| rmse_gain / baseline.mean_daily_rmse),
    }
}

fn spread<'a>(
    keys: impl Iterator<Item = &'a Key>,
    rows: &BTreeMap<(String, Key), Row>,
    members: &[String],
) -> Result<Spread> {
    let mut variance_sum = Sum::default();
    let mut mean_score = Accumulator::default();
    for key in keys {
        let mut mean = 0.0;
        let mut m2 = 0.0;
        let mut observation = None;
        // Welford's centered calculation avoids cancellation for large offsets.
        for (i, member) in members.iter().enumerate() {
            let row = &rows[&(member.clone(), key.clone())];
            let delta = finite(row.forecast - mean, "spread difference")?;
            mean = finite(mean + delta / (i + 1) as f64, "ensemble mean")?;
            m2 = finite(m2 + delta * (row.forecast - mean), "ensemble variance")?;
            observation = Some(row.observation);
        }
        variance_sum.add((m2 / (members.len() - 1) as f64).max(0.0))?;
        mean_score.add(mean, observation.ok_or("empty spread ensemble")?)?;
    }
    let metrics = mean_score.finish()?;
    let rms_sample_spread = finite(
        (variance_sum.value / metrics.count as f64).sqrt(),
        "RMS spread",
    )?;
    let spread_to_skill = if metrics.rmse > 0.0 {
        Some(finite(
            rms_sample_spread / metrics.rmse,
            "spread-to-skill ratio",
        )?)
    } else {
        None
    };
    Ok(Spread {
        count: metrics.count,
        members: members.len(),
        rms_sample_spread,
        ensemble_mean_rmse: metrics.rmse,
        ensemble_mean_bias: metrics.bias,
        spread_to_skill,
    })
}

fn features(
    train: &BTreeSet<String>,
    common: &BTreeMap<String, BTreeSet<Key>>,
    rows: &BTreeMap<(String, Key), Row>,
    members: &[String],
) -> Result<FeatureOutput> {
    let mut groups: BTreeMap<(String, String, &'static str), Vec<&Key>> = BTreeMap::new();
    for (day, keys) in common.iter().filter(|(day, _)| train.contains(*day)) {
        for key in keys {
            let row = &rows[&(members[0].clone(), key.clone())];
            let period = match row.period {
                Period::Day => "day",
                Period::Night => "night",
            };
            groups
                .entry((day.clone(), key.station.clone(), period))
                .or_default()
                .push(key);
        }
    }
    let mut labels = Vec::new();
    let mut predictions = vec![Vec::new(); members.len()];
    let mut observations = Vec::new();
    for ((day, station, period), keys) in groups {
        let mut observation = Sum::default();
        for key in &keys {
            observation
                .add(rows[&(members[0].clone(), (*key).clone())].observation / keys.len() as f64)?;
        }
        observations.push(observation.value);
        for (i, member) in members.iter().enumerate() {
            let mut prediction = Sum::default();
            for key in &keys {
                prediction
                    .add(rows[&(member.clone(), (*key).clone())].forecast / keys.len() as f64)?;
            }
            predictions[i].push(prediction.value);
        }
        labels.push(FeatureLabel {
            day,
            station,
            period,
            count: keys.len(),
        });
    }
    Ok(FeatureOutput {
        members: members.to_vec(),
        labels,
        predictions,
        observations,
        units: "same_as_input",
        method: "training_only_station_day_period_mean",
    })
}

pub fn score(request: ScoreRequest) -> Result<ScoreResult> {
    let config = request.config;
    let train = unique_labels(&config.train_days, "train_days")?;
    let heldout = if config.heldout_days.is_empty() {
        BTreeSet::new()
    } else {
        unique_labels(&config.heldout_days, "heldout_days")?
    };
    if !train.is_disjoint(&heldout) {
        return Err("training and held-out days must be disjoint".into());
    }
    let expected = unique_labels(&config.expected_members, "expected_members")?;
    if !expected.contains(&config.baseline_member) {
        return Err("baseline member is not expected".into());
    }
    if config.minimum_pairs == 0 {
        return Err("minimum_pairs must be positive".into());
    }
    let ensemble_members = config
        .ensemble_members
        .unwrap_or_else(|| config.expected_members.clone());
    let ensemble = unique_labels(&ensemble_members, "ensemble_members")?;
    if ensemble.len() < 2 || !ensemble.is_subset(&expected) {
        return Err("spread needs at least two expected ensemble members".into());
    }
    let days: BTreeSet<_> = train.union(&heldout).cloned().collect();
    let mut indexed = BTreeMap::new();
    let mut truth: BTreeMap<Key, (f64, Period)> = BTreeMap::new();
    let mut member_day_keys: BTreeMap<(String, String), BTreeSet<Key>> = BTreeMap::new();
    for row in request.rows {
        if !expected.contains(&row.member) || !days.contains(&row.day) {
            return Err(format!(
                "unexpected member or day: {} {}",
                row.member, row.day
            ));
        }
        if row.station.trim().is_empty() || row.valid.trim().is_empty() {
            return Err("station and valid labels must be nonempty".into());
        }
        finite(row.forecast, "forecast")?;
        finite(row.observation, "observation")?;
        let key = Key {
            day: row.day.clone(),
            station: row.station.clone(),
            valid: row.valid.clone(),
        };
        if let Some((observation, period)) = truth.get(&key) {
            if *observation != row.observation || *period != row.period {
                return Err(format!(
                    "observation or period mismatch at {} {} {}",
                    key.day, key.station, key.valid
                ));
            }
        } else {
            truth.insert(key.clone(), (row.observation, row.period));
        }
        member_day_keys
            .entry((row.member.clone(), row.day.clone()))
            .or_default()
            .insert(key.clone());
        let member = row.member.clone();
        if indexed.insert((member.clone(), key.clone()), row).is_some() {
            return Err(format!(
                "duplicate row at {member} {} {} {}",
                key.day, key.station, key.valid
            ));
        }
    }
    let mut masks = Vec::new();
    let mut common: BTreeMap<String, BTreeSet<Key>> = BTreeMap::new();
    for day in &days {
        let mut mask: Option<BTreeSet<Key>> = None;
        let mut input_pairs = BTreeMap::new();
        for member in &expected {
            let keys = member_day_keys
                .get(&(member.clone(), day.clone()))
                .ok_or_else(|| format!("missing member {member} on required day {day}"))?;
            input_pairs.insert(member.clone(), keys.len());
            mask = Some(match mask {
                None => keys.clone(),
                Some(previous) => previous.intersection(keys).cloned().collect(),
            });
        }
        let mask = mask.ok_or("empty expected-member mask")?;
        let day_count = mask
            .iter()
            .filter(|key| truth[*key].1 == Period::Day)
            .count();
        let night_count = mask.len() - day_count;
        if mask.len() < config.minimum_pairs
            || day_count < config.minimum_pairs
            || night_count < config.minimum_pairs
        {
            return Err(format!("day {day} common mask has {} total, {day_count} day and {night_count} night pairs; each period needs {}", mask.len(), config.minimum_pairs));
        }
        let dropped_pairs = input_pairs
            .iter()
            .map(|(member, count)| (member.clone(), count - mask.len()))
            .collect();
        masks.push(MaskAudit {
            day: day.clone(),
            common_pairs: mask.len(),
            common_day_pairs: day_count,
            common_night_pairs: night_count,
            input_pairs,
            dropped_pairs,
        });
        common.insert(day.clone(), mask);
    }
    let mut metrics = Vec::new();
    let mut daily = BTreeMap::new();
    for member in &expected {
        for (day, keys) in &common {
            let mut all = Accumulator::default();
            let mut daytime = Accumulator::default();
            let mut nighttime = Accumulator::default();
            for key in keys {
                let row = &indexed[&(member.clone(), key.clone())];
                all.add(row.forecast, row.observation)?;
                match row.period {
                    Period::Day => daytime.add(row.forecast, row.observation)?,
                    Period::Night => nighttime.add(row.forecast, row.observation)?,
                }
            }
            let all_metrics = all.finish()?;
            daily.insert((member.clone(), day.clone()), all_metrics.clone());
            for (period, result) in [
                ("all", all_metrics),
                ("day", daytime.finish()?),
                ("night", nighttime.finish()?),
            ] {
                metrics.push(MemberDayPeriod {
                    member: member.clone(),
                    day: day.clone(),
                    period,
                    metrics: result,
                });
            }
        }
    }
    let mut ranking: Vec<Rank> = expected
        .iter()
        .map(|member| {
            let training = aggregate(&config.train_days, member, &daily)?;
            let heldout = if config.heldout_days.is_empty() {
                None
            } else {
                Some(aggregate(&config.heldout_days, member, &daily)?)
            };
            Ok(Rank {
                member: member.clone(),
                training_score: training.mean_daily_rmse,
                heldout_score: heldout.as_ref().map(|a| a.mean_daily_rmse),
                training,
                heldout,
            })
        })
        .collect::<Result<_>>()?;
    ranking.sort_by(|a, b| {
        a.training
            .mean_daily_rmse
            .total_cmp(&b.training.mean_daily_rmse)
            .then_with(|| {
                (b.member == config.baseline_member).cmp(&(a.member == config.baseline_member))
            })
            .then_with(|| a.member.cmp(&b.member))
    });
    let winner = &ranking[0];
    if config
        .frozen_winner
        .as_ref()
        .is_some_and(|frozen| *frozen != winner.member)
    {
        return Err(format!(
            "frozen winner does not match training-only rank: {}",
            winner.member
        ));
    }
    let baseline = ranking
        .iter()
        .find(|rank| rank.member == config.baseline_member)
        .ok_or("missing baseline")?;
    let heldout_days: Vec<DailyComparison> = config
        .heldout_days
        .iter()
        .map(|day| {
            let winner_rmse = daily[&(winner.member.clone(), day.clone())].rmse;
            let baseline_rmse = daily[&(config.baseline_member.clone(), day.clone())].rmse;
            DailyComparison {
                day: day.clone(),
                baseline_rmse,
                winner_rmse,
                wins: winner_rmse < baseline_rmse,
            }
        })
        .collect();
    let mut heldout_periods = Vec::new();
    for day in &config.heldout_days {
        for period in ["day", "night"] {
            let winner_rmse = metrics
                .iter()
                .find(|row| row.member == winner.member && row.day == *day && row.period == period)
                .ok_or("missing winner period score")?
                .metrics
                .rmse;
            let baseline_rmse = metrics
                .iter()
                .find(|row| {
                    row.member == config.baseline_member && row.day == *day && row.period == period
                })
                .ok_or("missing baseline period score")?
                .metrics
                .rmse;
            heldout_periods.push(PeriodComparison {
                day: day.clone(),
                period,
                baseline_rmse,
                winner_rmse,
                rmse_gain: baseline_rmse - winner_rmse,
                not_worse: winner_rmse <= baseline_rmse,
            });
        }
    }
    let training_comparison = comparison(&winner.training, &baseline.training);
    let heldout_comparison = winner
        .heldout
        .as_ref()
        .zip(baseline.heldout.as_ref())
        .map(|(w, b)| comparison(w, b));
    let heldout_passed = winner.member != config.baseline_member
        && training_comparison.rmse_gain > 0.0
        && heldout_comparison
            .as_ref()
            .is_some_and(|c| c.rmse_gain > 0.0)
        && !heldout_days.is_empty()
        && heldout_days.iter().all(|day| day.wins)
        && heldout_periods.iter().all(|period| period.not_worse);
    let decision = Decision { winner: winner.member.clone(), baseline_member: config.baseline_member.clone(),
        selection_metric: "mean_daily_rmse_training_only", training: training_comparison,
        heldout: heldout_comparison, heldout_days, heldout_periods, heldout_passed,
        recommendation: if heldout_passed { "eligible_for_review" } else { "retain_current_setting" },
        rule: "Freeze the training winner. Require strict RMSE improvement on every held-out day and no day/night period regression. No held-out reselection." };
    let train_spread = spread(
        common
            .iter()
            .filter(|(day, _)| train.contains(*day))
            .flat_map(|(_, keys)| keys.iter()),
        &indexed,
        &ensemble_members,
    )?;
    let heldout_spread = if heldout.is_empty() {
        None
    } else {
        Some(spread(
            common
                .iter()
                .filter(|(day, _)| heldout.contains(*day))
                .flat_map(|(_, keys)| keys.iter()),
            &indexed,
            &ensemble_members,
        )?)
    };
    let daily_spread = common
        .iter()
        .map(|(day, keys)| {
            Ok(DailySpread {
                day: day.clone(),
                spread: spread(keys.iter(), &indexed, &ensemble_members)?,
            })
        })
        .collect::<Result<_>>()?;
    let train_features = features(&train, &common, &indexed, &ensemble_members)?;
    Ok(ScoreResult {
        ranking,
        metrics,
        masks,
        decision,
        train_spread,
        heldout_spread,
        daily_spread,
        ensemble_members,
        train_features,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> ScoreRequest {
        let config = Config {
            train_days: vec!["t1".into(), "t2".into()],
            heldout_days: vec!["h1".into(), "h2".into()],
            baseline_member: "control".into(),
            expected_members: vec!["control".into(), "fit".into(), "other".into()],
            minimum_pairs: 1,
            ensemble_members: Some(vec!["fit".into(), "other".into()]),
            frozen_winner: None,
        };
        let mut rows = Vec::new();
        for day in ["t1", "t2", "h1", "h2"] {
            for member in ["control", "fit", "other"] {
                let error = match (member, day.starts_with('t')) {
                    ("control", _) => 2.0,
                    ("fit", true) => 1.0,
                    ("fit", false) => 3.0,
                    ("other", true) => 1.5,
                    _ => 0.0,
                };
                for (valid, period) in [("12", Period::Day), ("00", Period::Night)] {
                    rows.push(Row {
                        member: member.into(),
                        day: day.into(),
                        station: "station".into(),
                        valid: valid.into(),
                        period,
                        forecast: 10.0 + error,
                        observation: 10.0,
                    });
                }
            }
        }
        ScoreRequest { rows, config }
    }

    #[test]
    fn heldout_cannot_reselect_training_winner() {
        let result = score(fixture()).unwrap();
        assert_eq!(result.decision.winner, "fit");
        assert!(!result.decision.heldout_passed);
        assert_eq!(result.decision.heldout.unwrap().rmse_gain, -1.0);
        let mut altered = fixture();
        for row in &mut altered.rows {
            if row.day.starts_with('h') && row.member == "other" {
                row.forecast = -100.0;
            }
        }
        assert_eq!(score(altered).unwrap().decision.winner, "fit");
    }

    #[test]
    fn common_mask_discards_unpaired_extreme_and_audits_counts() {
        let mut input = fixture();
        let mut unpaired = input.rows[0].clone();
        unpaired.station = "extra".into();
        unpaired.forecast = 1000.0;
        input.rows.push(unpaired);
        let result = score(input).unwrap();
        let mask = result.masks.iter().find(|mask| mask.day == "t1").unwrap();
        assert_eq!(mask.common_pairs, 2);
        assert_eq!(mask.dropped_pairs["control"], 1);
        let control = result
            .ranking
            .iter()
            .find(|rank| rank.member == "control")
            .unwrap();
        assert_eq!(control.training.mean_daily_rmse, 2.0);
    }

    #[test]
    fn missing_day_duplicate_truth_and_nonfinite_are_rejected() {
        let mut input = fixture();
        input
            .rows
            .retain(|row| !(row.member == "fit" && row.day == "h2"));
        assert!(score(input).unwrap_err().contains("missing member fit"));
        let mut input = fixture();
        input.rows.push(input.rows[0].clone());
        assert!(score(input).unwrap_err().contains("duplicate"));
        let mut input = fixture();
        input.rows[2].observation = 12.0;
        assert!(score(input).unwrap_err().contains("mismatch"));
        let mut input = fixture();
        input.rows[0].forecast = f64::NAN;
        assert!(score(input).unwrap_err().contains("finite"));
    }

    #[test]
    fn training_and_heldout_must_be_disjoint_and_periods_covered() {
        let mut input = fixture();
        input.config.heldout_days.push("t1".into());
        assert!(score(input).unwrap_err().contains("disjoint"));
        let mut input = fixture();
        input.rows.retain(|row| row.period == Period::Day);
        assert!(score(input).unwrap_err().contains("night pairs"));
    }

    #[test]
    fn exact_tie_prefers_baseline_and_daily_holdout_regression_blocks_acceptance() {
        let mut input = fixture();
        for row in &mut input.rows {
            if row.day.starts_with('t') {
                row.forecast = 12.0;
            }
        }
        assert_eq!(score(input).unwrap().decision.winner, "control");
        let mut input = fixture();
        for row in &mut input.rows {
            if row.member == "fit" && row.day == "h1" {
                row.forecast = 10.0;
            }
            if row.member == "fit" && row.day == "h2" {
                row.forecast = 12.5;
            }
        }
        let result = score(input).unwrap();
        assert!(result.decision.heldout.unwrap().rmse_gain > 0.0);
        assert!(!result.decision.heldout_passed);
    }

    #[test]
    fn spread_uses_sample_standard_deviation_and_requested_members() {
        let result = score(fixture()).unwrap();
        let spread = result.train_spread;
        assert_eq!(spread.members, 2);
        assert!((spread.rms_sample_spread - (0.125_f64).sqrt()).abs() < 1e-12);
        assert!((spread.ensemble_mean_rmse - 1.25).abs() < 1e-12);
        assert!((spread.spread_to_skill.unwrap() - (0.125_f64).sqrt() / 1.25).abs() < 1e-12);
    }

    #[test]
    fn days_have_equal_weight_even_when_pair_counts_differ() {
        let mut input = fixture();
        for row in &mut input.rows {
            if row.member == "fit" && row.day == "t1" {
                row.forecast = 10.0;
            }
            if row.member == "fit" && row.day == "t2" {
                row.forecast = 13.0;
            }
        }
        let repeated: Vec<_> = input
            .rows
            .iter()
            .filter(|row| row.day == "t1")
            .cloned()
            .collect();
        for i in 0..8 {
            for mut row in repeated.clone() {
                row.station = format!("station-{i}");
                input.rows.push(row);
            }
        }
        let result = score(input).unwrap();
        let rank = result
            .ranking
            .iter()
            .find(|rank| rank.member == "fit")
            .unwrap();
        assert_eq!(rank.training.mean_daily_rmse, 1.5);
    }

    #[test]
    fn training_only_freezes_winner_and_features_exclude_heldout() {
        let full = score(fixture()).unwrap();
        assert!(full
            .train_features
            .labels
            .iter()
            .all(|label| label.day.starts_with('t')));
        let mut input = fixture();
        input.rows.retain(|row| row.day.starts_with('t'));
        input.config.heldout_days.clear();
        let selected = score(input).unwrap();
        assert_eq!(selected.decision.winner, full.decision.winner);
        assert!(selected.decision.heldout.is_none());
        assert!(!selected.decision.heldout_passed);
        let mut input = fixture();
        input.config.frozen_winner = Some(selected.decision.winner);
        assert_eq!(score(input).unwrap().decision.winner, "fit");
        let mut input = fixture();
        input.config.frozen_winner = Some("other".into());
        assert!(score(input).unwrap_err().contains("frozen winner"));
    }

    #[test]
    fn daytime_gain_cannot_hide_heldout_nighttime_regression() {
        let mut input = fixture();
        for row in &mut input.rows {
            if row.member == "fit" && row.day.starts_with('h') {
                row.forecast = if row.period == Period::Day {
                    10.0
                } else {
                    12.5
                };
            }
        }
        let result = score(input).unwrap();
        assert!(result.decision.heldout.as_ref().unwrap().rmse_gain > 0.0);
        assert!(result.decision.heldout_days.iter().all(|day| day.wins));
        assert!(!result.decision.heldout_passed);
        assert!(result
            .decision
            .heldout_periods
            .iter()
            .filter(|period| period.period == "night")
            .all(|period| !period.not_worse));
    }
}
