//! rw_radar_feeds: national radar-volume feeds as JSON rows.
//!
//! ```text
//! rw_radar_feeds providers
//! rw_radar_feeds sites   [--provider ID]                 (embedded tables, no network)
//! rw_radar_feeds latest  --provider ID --site SITE
//! rw_radar_feeds recent  --provider ID --site SITE [--count N]
//! rw_radar_feeds window  --provider ID --site SITE --start ISO --end ISO [--max N]
//! rw_radar_feeds --abi | --version
//! ```
//!
//! Every command prints one JSON document on stdout. A frame row is
//! `{provider, site, format, identity, stamp, merge, parts: [{url, filename}]}`;
//! `window` adds `route` (`archive` or `recent`). Errors go to stderr with a
//! non-zero exit. Hand-parsed for the same reason `rw_odim` is: this
//! workspace builds offline and clap is not in its vendored closure.

use std::process::ExitCode;

use chrono::{DateTime, NaiveDateTime, TimeZone, Utc};
use rw_radar_feeds::{FrameRow, feed, feeds, frame_format, window};
use serde_json::json;

const SCHEMA: &str = "gpuwm-obs.radar-feeds.v1";
const SOURCE_REV: &str = env!("GPUWM_BRIDGE_SOURCE_REV");

/// The record contract this build writes (`gpuwm.obs.frontdoor.RADAR_FEEDS`
/// compares it exactly): not a version, the fields a consumer reads.
const ABI_MARKER: &str = "gpuwm-obs.radar-feeds.v1\tproviders\tsites\tlatest\trecent\twindow\tprovider\tsite\tformat\tidentity\tstamp\tmerge\tparts\turl\tfilename\troute";

/// `GPUWM_BRIDGE_SOURCE_REV=<40-hex commit>`, embedded so the release cut can
/// prove a staged binary matches the commit being released by reading bytes
/// alone; `main` references it so the linker keeps it.
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

fn usage() -> String {
    "usage: rw_radar_feeds providers | sites [--provider ID] | latest --provider ID --site SITE \
     | recent --provider ID --site SITE [--count N] | window --provider ID --site SITE --start ISO \
     --end ISO [--max N]"
        .to_string()
}

struct Args {
    command: String,
    provider: Option<String>,
    site: Option<String>,
    count: usize,
    start: Option<DateTime<Utc>>,
    end: Option<DateTime<Utc>>,
    max: usize,
}

fn parse_time(text: &str) -> Result<DateTime<Utc>, String> {
    if let Ok(t) = DateTime::parse_from_rfc3339(text) {
        return Ok(t.with_timezone(&Utc));
    }
    let trimmed = text.trim_end_matches('Z');
    for format in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y%m%d%H%M"] {
        if let Ok(naive) = NaiveDateTime::parse_from_str(trimmed, format) {
            return Ok(Utc.from_utc_datetime(&naive));
        }
    }
    Err(format!(
        "{text:?} is not a UTC time (2026-10-16T08:00Z or 202610160800)"
    ))
}

fn parse(argv: &[String]) -> Result<Args, String> {
    let command = argv.first().cloned().ok_or_else(usage)?;
    let mut args = Args {
        command,
        provider: None,
        site: None,
        count: 12,
        start: None,
        end: None,
        max: 10_000,
    };
    let mut i = 1;
    while i < argv.len() {
        let flag = argv[i].as_str();
        let value = argv
            .get(i + 1)
            .ok_or_else(|| format!("{flag} needs a value; {}", usage()))?;
        match flag {
            "--provider" => args.provider = Some(value.clone()),
            "--site" => args.site = Some(value.clone()),
            "--count" => args.count = value.parse().map_err(|_| format!("--count {value:?}"))?,
            "--max" => args.max = value.parse().map_err(|_| format!("--max {value:?}"))?,
            "--start" => args.start = Some(parse_time(value)?),
            "--end" => args.end = Some(parse_time(value)?),
            _ => return Err(format!("unknown option {flag}; {}", usage())),
        }
        i += 2;
    }
    Ok(args)
}

fn need<'a>(value: &'a Option<String>, flag: &str) -> Result<&'a str, String> {
    value
        .as_deref()
        .ok_or_else(|| format!("{flag} is required; {}", usage()))
}

fn run(args: Args) -> Result<serde_json::Value, String> {
    let header = json!({"schema": SCHEMA, "command": args.command, "source_rev": SOURCE_REV});
    let mut out = header;
    match args.command.as_str() {
        "providers" => {
            out["providers"] = feeds()
                .iter()
                .map(|p| {
                    json!({
                        "id": p.id(), "label": p.label(), "country": p.country(),
                        "format": frame_format(p.id()), "recent": p.supports_recent(),
                        "archive": p.supports_archive(), "static_sites": p.static_sites().len(),
                    })
                })
                .collect();
        }
        "sites" => {
            let mut rows = Vec::new();
            for p in feeds() {
                if args.provider.as_deref().is_some_and(|want| want != p.id()) {
                    continue;
                }
                for s in p.static_sites() {
                    rows.push(json!({
                        "provider": p.id(), "site": s.site_id, "label": s.label,
                        "country": s.country, "latitude": s.latitude_deg,
                        "longitude": s.longitude_deg, "format": frame_format(p.id()),
                    }));
                }
            }
            if rows.is_empty() {
                return Err(format!("no sites for provider {:?}", args.provider));
            }
            out["sites"] = rows.into();
        }
        "latest" => {
            let provider = feed(need(&args.provider, "--provider")?)?;
            let site = need(&args.site, "--site")?;
            let plan = provider.latest(site)?;
            out["frames"] = json!([FrameRow::new(provider.id(), site, plan).to_json()]);
        }
        "recent" => {
            let provider = feed(need(&args.provider, "--provider")?)?;
            let site = need(&args.site, "--site")?;
            let plans = provider.recent(site, args.count.max(1))?;
            out["frames"] = plans
                .into_iter()
                .map(|plan| FrameRow::new(provider.id(), site, plan).to_json())
                .collect();
        }
        "window" => {
            let provider = feed(need(&args.provider, "--provider")?)?;
            let site = need(&args.site, "--site")?;
            let start = args.start.ok_or("--start is required")?;
            let end = args.end.ok_or("--end is required")?;
            let (route, rows) = window(provider.as_ref(), site, start, end, args.max)?;
            out["route"] = route.name().into();
            out["start"] = start.format("%Y-%m-%dT%H:%M:%SZ").to_string().into();
            out["end"] = end.format("%Y-%m-%dT%H:%M:%SZ").to_string().into();
            out["frames"] = rows.iter().map(FrameRow::to_json).collect();
        }
        other => return Err(format!("unknown subcommand {other:?}; {}", usage())),
    }
    Ok(out)
}

fn main() -> ExitCode {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    let argv: Vec<String> = std::env::args().skip(1).collect();
    match argv.first().map(String::as_str) {
        Some("--abi") => {
            println!("{ABI_MARKER}");
            return ExitCode::SUCCESS;
        }
        Some("--version" | "-V") => {
            println!("rw_radar_feeds {}", env!("CARGO_PKG_VERSION"));
            return ExitCode::SUCCESS;
        }
        Some("--help" | "-h") | None => {
            println!("{}", usage());
            return ExitCode::SUCCESS;
        }
        _ => {}
    }
    match parse(&argv).and_then(run) {
        Ok(doc) => {
            println!("{}", serde_json::to_string_pretty(&doc).unwrap_or_default());
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("rw_radar_feeds: {message}");
            ExitCode::from(2)
        }
    }
}
