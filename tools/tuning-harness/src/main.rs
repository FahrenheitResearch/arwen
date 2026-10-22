use serde::Serialize;
use serde_json::Value;
use std::io::{self, Read, Write};
use tuning_harness_core::{eki, import, sample, score, Result};

fn execute(command: &str, input: &str) -> Result<Value> {
    match command {
        "sample" => {
            let request =
                serde_json::from_str(input).map_err(|e| format!("invalid sample JSON: {e}"))?;
            serde_json::to_value(sample::sample(request)?).map_err(|e| e.to_string())
        }
        "score" => {
            let request =
                serde_json::from_str(input).map_err(|e| format!("invalid score JSON: {e}"))?;
            serde_json::to_value(score::score(request)?).map_err(|e| e.to_string())
        }
        "eki" => {
            let request =
                serde_json::from_str(input).map_err(|e| format!("invalid EKI JSON: {e}"))?;
            serde_json::to_value(eki::update(request)?).map_err(|e| e.to_string())
        }
        "import" => {
            let request =
                serde_json::from_str(input).map_err(|e| format!("invalid import JSON: {e}"))?;
            serde_json::to_value(import::import(request)?).map_err(|e| e.to_string())
        }
        _ => Err("usage: tuning-harness-core sample|score|eki|import < request.json".into()),
    }
}

fn write_json<T: Serialize>(mut writer: impl Write, value: &T) -> io::Result<()> {
    serde_json::to_writer(&mut writer, value).map_err(io::Error::other)?;
    writeln!(writer)
}

fn main() {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let result = if arguments.len() != 1 {
        Err("usage: tuning-harness-core sample|score|eki|import < request.json".into())
    } else {
        let mut input = String::new();
        io::stdin()
            .read_to_string(&mut input)
            .map_err(|e| format!("cannot read request: {e}"))
            .and_then(|_| execute(&arguments[0], &input))
    };
    match result {
        Ok(value) => {
            if write_json(io::stdout().lock(), &value).is_err() {
                std::process::exit(2);
            }
        }
        Err(error) => {
            let _ = write_json(io::stderr().lock(), &serde_json::json!({"error": error}));
            std::process::exit(2);
        }
    }
}
