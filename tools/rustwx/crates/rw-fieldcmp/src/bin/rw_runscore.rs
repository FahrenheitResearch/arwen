//! `rw_runscore` -- distance between two runs of the same case.
//!
//! Two run directories in, one distance per registered metric out.  The
//! domain ladder, the grid spacings, the field list, every threshold and
//! every metric-key spelling arrive on the command line, so the binary
//! carries no campaign, case, or physics-suite knowledge of its own: a caller
//! that has pinned a metric set passes its pins, and a caller that has not
//! gets WRF-convention defaults and generic key spellings.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use rw_fieldcmp::runscore::{DomainSpec, MetricKeys, RunScoreRequest};

const USAGE: &str = "\
usage: rw_runscore <left-dir> <right-dir> --start WHEN --run-seconds N \\
                   --cadence-seconds N --domain LABEL=DX [options]

  --start WHEN            first valid time, YYYY-MM-DD[T ]HH:MM:SS (required)
  --run-seconds N         scored duration (required)
  --cadence-seconds N     history interval (required)
  --domain LABEL=DX_M     a domain and its grid spacing; repeatable, required
  --neighborhood-domain LABEL
                          also score the neighbourhood row on LABEL;
                          repeatable, and no domain carries it by default
  --field VAR             score VAR; repeatable, and the first use replaces
                          the default field set
  --frame-prefix TEXT     history frames start with TEXT (default: wrfout)
  --low-pass-width-m V    physical width of the low-pass filter (default: 6000)
  --interior-cells N      cells excluded from each interior edge (default: 5)
  --boundary-cells N      width of the scored boundary frame (default: 5)
  --composite VAR         volume field reduced per column (default: REFL_10CM)
  --threshold V           event threshold on the composite (default: 40)
  --neighborhood-radius-m V
                          neighbourhood half-width in metres (default: 5000)
  --object-connectivity N 4 or 8 (default: 8)
  --object-min-area-km2 V smallest qualifying object (default: 25)
  --key-state TEXT        metric-key category for the state rows
  --key-boundary TEXT     metric-key category for the boundary rows
  --key-object TEXT       metric-key category for the object-timing rows
  --key-object-subject TEXT       third field of the object-timing keys
  --key-neighborhood TEXT         metric-key category for the neighbourhood row
  --key-neighborhood-subject TEXT third field of the neighbourhood key
  --json PATH             write the distances and the read counts to PATH
  --quiet                 do not write the table to stdout
  --threads N             worker threads (default: one per core)

defaults: fields U V W T PH MU QVAPOR; key categories low_pass_state_rmse,
applied_boundary_increment_error, storm_object_timing_difference and
neighborhood_fss_distance
";

struct Options {
    request: RunScoreRequest,
    json_path: Option<PathBuf>,
    quiet: bool,
    threads: Option<usize>,
}

fn parse(mut args: impl Iterator<Item = String>) -> Result<Options, String> {
    let mut positional: Vec<String> = Vec::new();
    let mut start = None;
    let mut run_seconds = None;
    let mut cadence_seconds = None;
    let mut domains: Vec<(String, f64)> = Vec::new();
    let mut neighborhood_domains: Vec<String> = Vec::new();
    let mut fields: Option<Vec<String>> = None;
    let mut frame_prefix = "wrfout".to_string();
    let mut low_pass_width_m = 6000.0f64;
    let mut interior_cells = 5usize;
    let mut boundary_cells = 5usize;
    let mut composite = "REFL_10CM".to_string();
    let mut threshold = 40.0f64;
    let mut neighborhood_radius_m = 5000.0f64;
    let mut object_connectivity = 8u8;
    let mut object_min_area_km2 = 25.0f64;
    let mut keys = MetricKeys::default();
    let mut json_path = None;
    let mut quiet = false;
    let mut threads = None;

    while let Some(arg) = args.next() {
        let mut value = |name: &str| -> Result<String, String> {
            args.next().ok_or_else(|| format!("{name} needs a value"))
        };
        let number = |raw: String, name: &str| -> Result<f64, String> {
            raw.parse::<f64>()
                .map_err(|_| format!("{name} {raw} is not a number"))
        };
        let count = |raw: String, name: &str| -> Result<usize, String> {
            raw.parse::<usize>()
                .map_err(|_| format!("{name} {raw} is not a count"))
        };
        match arg.as_str() {
            "-h" | "--help" => return Err(USAGE.to_string()),
            "--start" => start = Some(value("--start")?),
            "--run-seconds" => {
                let raw = value("--run-seconds")?;
                run_seconds = Some(
                    raw.parse::<i64>()
                        .map_err(|_| format!("--run-seconds {raw} is not a duration"))?,
                );
            }
            "--cadence-seconds" => {
                let raw = value("--cadence-seconds")?;
                cadence_seconds = Some(
                    raw.parse::<i64>()
                        .map_err(|_| format!("--cadence-seconds {raw} is not a duration"))?,
                );
            }
            "--domain" => {
                let raw = value("--domain")?;
                let (label, spacing) = raw
                    .split_once('=')
                    .ok_or_else(|| format!("--domain {raw} is not LABEL=DX_M"))?;
                let dx = spacing
                    .parse::<f64>()
                    .map_err(|_| format!("--domain {raw} has no numeric spacing"))?;
                if !(dx.is_finite() && dx > 0.0) {
                    return Err(format!("--domain {raw} has a non-physical spacing"));
                }
                domains.push((label.to_string(), dx));
            }
            "--neighborhood-domain" => {
                neighborhood_domains.push(value("--neighborhood-domain")?);
            }
            "--field" => {
                let name = value("--field")?;
                fields.get_or_insert_with(Vec::new).push(name);
            }
            "--frame-prefix" => frame_prefix = value("--frame-prefix")?,
            "--low-pass-width-m" => {
                low_pass_width_m = number(value("--low-pass-width-m")?, "--low-pass-width-m")?;
            }
            "--interior-cells" => {
                interior_cells = count(value("--interior-cells")?, "--interior-cells")?;
            }
            "--boundary-cells" => {
                boundary_cells = count(value("--boundary-cells")?, "--boundary-cells")?;
            }
            "--composite" => composite = value("--composite")?,
            "--threshold" => threshold = number(value("--threshold")?, "--threshold")?,
            "--neighborhood-radius-m" => {
                neighborhood_radius_m =
                    number(value("--neighborhood-radius-m")?, "--neighborhood-radius-m")?;
            }
            "--object-connectivity" => {
                let raw = value("--object-connectivity")?;
                object_connectivity = match raw.as_str() {
                    "4" => 4,
                    "8" => 8,
                    _ => return Err(format!("--object-connectivity {raw} is not 4 or 8")),
                };
            }
            "--object-min-area-km2" => {
                object_min_area_km2 =
                    number(value("--object-min-area-km2")?, "--object-min-area-km2")?;
            }
            "--key-state" => keys.state = value("--key-state")?,
            "--key-boundary" => keys.boundary = value("--key-boundary")?,
            "--key-object" => keys.object = value("--key-object")?,
            "--key-object-subject" => keys.object_subject = value("--key-object-subject")?,
            "--key-neighborhood" => keys.neighborhood = value("--key-neighborhood")?,
            "--key-neighborhood-subject" => {
                keys.neighborhood_subject = value("--key-neighborhood-subject")?;
            }
            "--json" => json_path = Some(PathBuf::from(value("--json")?)),
            "--quiet" => quiet = true,
            "--threads" => threads = Some(count(value("--threads")?, "--threads")?),
            other if other.starts_with('-') => return Err(format!("unknown option {other}")),
            other => positional.push(other.to_string()),
        }
    }

    if positional.len() != 2 {
        return Err(format!(
            "expected two run directories, got {}\n\n{USAGE}",
            positional.len()
        ));
    }
    let start = start.ok_or_else(|| format!("--start is required\n\n{USAGE}"))?;
    let run_seconds = run_seconds.ok_or("--run-seconds is required")?;
    let cadence_seconds = cadence_seconds.ok_or("--cadence-seconds is required")?;
    if domains.is_empty() {
        return Err(format!("at least one --domain is required\n\n{USAGE}"));
    }
    for label in &neighborhood_domains {
        if !domains.iter().any(|(name, _)| name == label) {
            return Err(format!("--neighborhood-domain {label} is not a --domain"));
        }
    }

    let request = RunScoreRequest {
        left: PathBuf::from(&positional[0]),
        right: PathBuf::from(&positional[1]),
        frame_prefix,
        start,
        run_seconds,
        cadence_seconds,
        domains: domains
            .into_iter()
            .map(|(label, dx_m)| DomainSpec {
                score_neighborhood: neighborhood_domains.contains(&label),
                label,
                dx_m,
            })
            .collect(),
        fields: fields.unwrap_or_else(|| {
            ["U", "V", "W", "T", "PH", "MU", "QVAPOR"]
                .into_iter()
                .map(str::to_string)
                .collect()
        }),
        low_pass_width_m,
        interior_exclusion_cells: interior_cells,
        boundary_width_cells: boundary_cells,
        composite_variable: composite,
        threshold,
        neighborhood_radius_m,
        object_connectivity,
        object_min_area_km2,
        keys,
    };

    Ok(Options {
        request,
        json_path,
        quiet,
        threads,
    })
}

fn run() -> Result<(), String> {
    let options = parse(std::env::args().skip(1))?;
    if let Some(threads) = options.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .map_err(|error| format!("cannot size the worker pool: {error}"))?;
    }

    let began = Instant::now();
    let score = rw_fieldcmp::runscore::score(&options.request).map_err(|error| error.to_string())?;
    let elapsed = began.elapsed().as_secs_f64();

    if !options.quiet {
        let width = score
            .distances
            .keys()
            .map(String::len)
            .max()
            .unwrap_or(0)
            .max(6);
        println!("{:<width$}  distance", "metric", width = width);
        for (metric, value) in &score.distances {
            println!(
                "{:<width$}  {}",
                metric,
                rw_fieldcmp::numfmt::general(*value, 12, false),
                width = width
            );
        }
        println!();
        println!(
            "{} distances over {} domains: {} variable decodes, {} frame opens, {elapsed:.2} s",
            score.distances.len(),
            score.domains.len(),
            score.variable_decodes,
            score.frame_opens
        );
    }
    if let Some(path) = &options.json_path {
        let encoded = serde_json::to_string_pretty(&serde_json::json!({
            "score": &score,
            "elapsed_seconds": elapsed,
        }))
        .map_err(|error| format!("cannot encode the score: {error}"))?;
        std::fs::write(path, encoded)
            .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}
