//! `rw_fieldcmp` -- compare two directories of history frames.
//!
//! Two directories in, one metric table out, optionally the same numbers as
//! JSON for a driver to consume.  Everything the table names -- the arm
//! labels, the field labels, the accumulation fields, the composite variable
//! and its thresholds -- is supplied on the command line, so the binary
//! carries no campaign, case, or physics-suite knowledge of its own.

use std::path::PathBuf;
use std::process::ExitCode;

use rw_fieldcmp::{CompareRequest, CompositeSpec, FieldSpec};

const USAGE: &str = "\
usage: rw_fieldcmp <left-dir> <right-dir> [options]

  --left-label NAME       arm name printed for the left directory (default: left)
  --right-label NAME      arm name printed for the right directory (default: right)
  --frame-prefix TEXT     compare files whose names start with TEXT (default: wrfout)
  --field VAR[=LABEL]     score VAR, printed as LABEL; repeatable, and the
                          first use replaces the default field set
  --accumulation VAR      report VAR's domain sum and ratio; repeatable, and
                          the first use replaces the default set
  --composite VAR         reduce VAR over its level axis and report coverage
  --composite-label TEXT  name printed for the composite rows (default: comp-dBZ)
  --threshold VALUE       composite coverage threshold; repeatable, and the
                          first use replaces the default set
  --no-composite          skip the composite entirely
  --table PATH            write the table to PATH as well as to stdout
  --json PATH             write the full numeric record to PATH
  --quiet                 do not write the table to stdout
  --threads N             worker threads (default: one per core)

defaults: fields RAINNC RAINC PBLH T2 Q2 GLW OLR SWDOWN, accumulations
RAINNC RAINC, composite REFL_10CM at 20 and 40
";

struct Options {
    request: CompareRequest,
    table_path: Option<PathBuf>,
    json_path: Option<PathBuf>,
    quiet: bool,
    threads: Option<usize>,
}

fn parse(mut args: impl Iterator<Item = String>) -> Result<Options, String> {
    let mut positional: Vec<String> = Vec::new();
    let mut left_label = None;
    let mut right_label = None;
    let mut frame_prefix = None;
    let mut fields: Option<Vec<FieldSpec>> = None;
    let mut accumulations: Option<Vec<String>> = None;
    let mut composite_variable = None;
    let mut composite_label = None;
    let mut thresholds: Option<Vec<f64>> = None;
    let mut no_composite = false;
    let mut table_path = None;
    let mut json_path = None;
    let mut quiet = false;
    let mut threads = None;

    while let Some(arg) = args.next() {
        let mut value = |name: &str| -> Result<String, String> {
            args.next()
                .ok_or_else(|| format!("{name} needs a value"))
        };
        match arg.as_str() {
            "-h" | "--help" => return Err(USAGE.to_string()),
            "--left-label" => left_label = Some(value("--left-label")?),
            "--right-label" => right_label = Some(value("--right-label")?),
            "--frame-prefix" => frame_prefix = Some(value("--frame-prefix")?),
            "--field" => {
                let spec = value("--field")?;
                let parsed = match spec.split_once('=') {
                    Some((variable, label)) => FieldSpec::labelled(variable, label),
                    None => FieldSpec::new(spec),
                };
                fields.get_or_insert_with(Vec::new).push(parsed);
            }
            "--accumulation" => {
                let variable = value("--accumulation")?;
                accumulations.get_or_insert_with(Vec::new).push(variable);
            }
            "--composite" => composite_variable = Some(value("--composite")?),
            "--composite-label" => composite_label = Some(value("--composite-label")?),
            "--threshold" => {
                let raw = value("--threshold")?;
                let parsed = raw
                    .parse::<f64>()
                    .map_err(|_| format!("--threshold {raw} is not a number"))?;
                thresholds.get_or_insert_with(Vec::new).push(parsed);
            }
            "--no-composite" => no_composite = true,
            "--table" => table_path = Some(PathBuf::from(value("--table")?)),
            "--json" => json_path = Some(PathBuf::from(value("--json")?)),
            "--quiet" => quiet = true,
            "--threads" => {
                let raw = value("--threads")?;
                threads = Some(
                    raw.parse::<usize>()
                        .map_err(|_| format!("--threads {raw} is not a count"))?,
                );
            }
            other if other.starts_with('-') => return Err(format!("unknown option {other}")),
            other => positional.push(other.to_string()),
        }
    }

    if positional.len() != 2 {
        return Err(format!(
            "expected two directories, got {}\n\n{USAGE}",
            positional.len()
        ));
    }

    let mut request = CompareRequest::new(&positional[0], &positional[1]);
    if let Some(label) = left_label {
        request.left_label = label;
    }
    if let Some(label) = right_label {
        request.right_label = label;
    }
    if let Some(prefix) = frame_prefix {
        request.frame_prefix = prefix;
    }
    if let Some(specs) = fields {
        request.fields = specs;
    }
    if let Some(variables) = accumulations {
        request.accumulations = variables;
    }
    request.composite = if no_composite {
        None
    } else {
        let default = CompositeSpec {
            variable: "REFL_10CM".to_string(),
            label: "comp-dBZ".to_string(),
            thresholds: vec![20.0, 40.0],
        };
        let mut spec = request.composite.unwrap_or(default);
        if let Some(variable) = composite_variable {
            spec.variable = variable;
        }
        if let Some(label) = composite_label {
            spec.label = label;
        }
        if let Some(values) = thresholds {
            spec.thresholds = values;
        }
        Some(spec)
    };

    Ok(Options {
        request,
        table_path,
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

    let comparison = rw_fieldcmp::compare(&options.request).map_err(|error| error.to_string())?;
    let table = rw_fieldcmp::report::table(&comparison);

    if !options.quiet {
        print!("{table}");
    }
    if let Some(path) = &options.table_path {
        std::fs::write(path, &table)
            .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    }
    if let Some(path) = &options.json_path {
        let encoded = serde_json::to_string_pretty(&comparison)
            .map_err(|error| format!("cannot encode the comparison: {error}"))?;
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
