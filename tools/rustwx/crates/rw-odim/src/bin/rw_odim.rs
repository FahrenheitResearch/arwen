//! `rw_odim` — the gpuwm front door over the ODIM_H5 polar-volume decoder.
//!
//! ```text
//! rw_odim pack (--file F [--file F2 ...] [--site S] | --dir D [--stamp S])
//!                               --out P [--quantities Q,Q]
//!                               [--max-elevation-deg D] [--max-range-km R]
//!                               [--dual-prf on|off]
//!                                   decode one volume and write a
//!                                   gpuwm-obs.radar-sweeps.v3 pack
//! rw_odim volumes --dir D           which volumes a directory of scan files
//!                                   holds, grouped by nominal time
//! rw_odim nyquist --file F [--site S]
//!                                   the dealias handoff: per-sweep interval
//!                                   and its provenance
//! rw_odim stations --file F         the stations a multi-station file
//!                                   (a JMA tar) holds
//! ```
//!
//! `--file` reads ODIM_H5 and, chosen by the file's own magic bytes, the
//! polar formats the rest of the open world publishes: JMA GRIB2 tars,
//! IRIS/Sigmet RAW, Universal Format, CfRadial 1 and DORADE, gzipped or not
//! (`rw_odim::world`). Repeat `--file` for one station that arrives in
//! several files (JMA's reflectivity and velocity tars, an IRIS site's
//! separate tasks); `--site` picks a station out of a multi-station file.
//!
//! ```text
//! rw_odim --abi                     the record contract this build writes
//! rw_odim --version                 the build's version
//! ```
//!
//! `--file` and `--dir` are the two shapes national feeds come in, not two
//! ways of saying one thing. The Netherlands publishes one `PVOL` per volume;
//! Germany publishes one `SCAN` per (elevation, quantity), so a ten-elevation
//! volume carrying `DBZH` and `VRADH` is twenty files that mean nothing apart.
//! `--dir` assembles those and refuses to mix two nominal times. Run
//! `volumes` first to see what a directory holds; `--stamp` is required only
//! when it holds more than one, so that packing a directory can never
//! silently pick one of several.
//!
//! Arguments are hand-parsed rather than clap-driven: this workspace builds
//! `--locked --offline` from its checked-in `vendor/crates-io` closure and
//! clap is not in it. The clap front end with `inspect`/`decode`/`export`
//! lives upstream on the Studio repo's `lane/odim-polar-decoder`, where the
//! workspace carries clap; nothing here needs it.
//!
//! Every subcommand writes one JSON record to stdout carrying a `schema`
//! string, so a consumer pins the contract rather than the binary's version.

use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use rw_nexrad::pack::write_pack;
use rw_odim::pack::{PackOptions, build_pack_reported};
use rw_odim::{
    DecodeOptions, VolumeGroup, assemble, read_volume_with, survey_dir, volume::NyquistSource,
};

/// The record contract this build writes.
///
/// Not a version number: it names the fields a consumer was written against,
/// so a rebuilt-but-unchanged binary still matches and one whose records
/// changed shape does not. Same idea as `rw_opera --abi` on the gpuwm side.
const ABI_MARKER: &str = "gpuwm-obs.odim-pack.v1\tgpuwm-obs.odim-nyquist.v1\t\
gpuwm-obs.odim-volumes.v1\tgpuwm-obs.radar-sweeps.v3\t\
site\telangle\tnbins\tnrays\trscale\trstart\tazimuth\tnyquist_ms\tnyquist_source\t\
nyquist_granularity\tcensor\tmeasured\tundetect\tnodata\tsentinel_ambiguous\t\
assembled\tmanifest_sha256\tstamp\tdual_prf\t\
gpuwm-obs.odim-stations.v1\tformat\todim-h5\tjma-grib2-tar\tiris-raw\t\
universal-format\tcfradial1\tdorade\tDerivedDualPrf\tjma-no-echo-undetect";

/// `GPUWM_BRIDGE_SOURCE_REV=<40-hex commit>`: the source revision this
/// binary was built from, embedded so the gpuwm release cut can prove a
/// staged bridge matches the commit being released by reading bytes alone
/// (`tools/build_bridge_bundle.py pin --source-rev`). `build.rs` injects the
/// value; `main` references the constant so the linker cannot discard it.
///
/// Required because `rw_odim` is a bundled artifact: the cut refuses to pin
/// a bundle whose binaries do not carry it, and a bridge the cut cannot
/// prove must not ship.
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

const PACK_SCHEMA: &str = "gpuwm-obs.odim-pack.v1";
const NYQUIST_SCHEMA: &str = "gpuwm-obs.odim-nyquist.v1";
const VOLUMES_SCHEMA: &str = "gpuwm-obs.odim-volumes.v1";
const STATIONS_SCHEMA: &str = "gpuwm-obs.odim-stations.v1";

const USAGE: &str = "\
rw_odim pack (--file F [--file F2 ...] [--site S] | --dir D [--stamp S]) --out P \
[--quantities Q,Q] [--max-elevation-deg D] [--max-range-km R] [--dual-prf on|off]
rw_odim volumes --dir D
rw_odim nyquist --file F [--site S]
rw_odim stations --file F
rw_odim --abi
rw_odim --version";

fn main() -> ExitCode {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("rw_odim: {error}");
            ExitCode::FAILURE
        }
    }
}

/// One `--flag value` pair off the command line.
struct Args {
    values: Vec<(String, String)>,
}

impl Args {
    fn parse(raw: &[String]) -> Result<Self, Box<dyn Error>> {
        let mut values = Vec::new();
        let mut index = 0;
        while index < raw.len() {
            let flag = &raw[index];
            if !flag.starts_with("--") {
                return Err(format!("unexpected argument {flag:?}\n\n{USAGE}").into());
            }
            let value = raw.get(index + 1).ok_or_else(|| {
                Box::<dyn Error>::from(format!("{flag} needs a value\n\n{USAGE}"))
            })?;
            values.push((flag.trim_start_matches("--").to_string(), value.clone()));
            index += 2;
        }
        Ok(Args { values })
    }

    fn get(&self, name: &str) -> Option<&str> {
        self.values
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }

    fn all(&self, name: &str) -> Vec<&str> {
        self.values
            .iter()
            .filter(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
            .collect()
    }

    fn require(&self, name: &str) -> Result<&str, Box<dyn Error>> {
        self.get(name)
            .ok_or_else(|| format!("--{name} is required\n\n{USAGE}").into())
    }

    fn number(&self, name: &str, fallback: f64) -> Result<f64, Box<dyn Error>> {
        match self.get(name) {
            None => Ok(fallback),
            Some(text) => text
                .parse::<f64>()
                .map_err(|error| format!("--{name} {text:?} is not a number: {error}").into()),
        }
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    match argv.first().map(String::as_str) {
        None => {
            println!("{USAGE}");
            Ok(())
        }
        Some("--abi") => {
            println!("{ABI_MARKER}");
            Ok(())
        }
        Some("--version" | "-V") => {
            println!("rw_odim {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Some("--help" | "-h") => {
            println!("{USAGE}");
            Ok(())
        }
        Some("pack") => pack(&Args::parse(&argv[1..])?),
        Some("volumes") => volumes(&Args::parse(&argv[1..])?),
        Some("nyquist") => nyquist(&Args::parse(&argv[1..])?),
        Some("stations") => stations(&Args::parse(&argv[1..])?),
        Some(other) => Err(format!("unknown subcommand {other:?}\n\n{USAGE}").into()),
    }
}

fn read_file_identity(path: &Path) -> Result<(String, usize, String), Box<dyn Error>> {
    use sha2::{Digest, Sha256};
    let bytes =
        std::fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let digest: String = hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| path.display().to_string());
    Ok((name, bytes.len(), digest))
}

/// Resolve `--dir` (and optional `--stamp`) to the member files of one volume.
///
/// Refuses a directory holding more than one nominal time unless the caller
/// named which one. Picking the newest would be the plausible wrong answer:
/// the record would look ordinary and half a user's cases would silently be
/// the volume they did not ask for.
fn group_for(dir: &Path, stamp: Option<&str>) -> Result<VolumeGroup, Box<dyn Error>> {
    let groups = survey_dir(dir)?;
    if groups.is_empty() {
        return Err(format!(
            "{} holds no ODIM files (*.h5, *.hdf, *.hdf5)",
            dir.display()
        )
        .into());
    }
    let chosen = match stamp {
        Some(wanted) => groups
            .into_iter()
            .find(|group| group.stamp == wanted)
            .ok_or_else(|| {
                Box::<dyn Error>::from(format!(
                    "{} holds no volume stamped {wanted:?}; run `rw_odim \
                     volumes --dir {}` for the stamps it does hold",
                    dir.display(),
                    dir.display()
                ))
            })?,
        None if groups.len() == 1 => groups.into_iter().next().expect("length checked"),
        None => {
            let stamps: Vec<&str> = groups.iter().map(|g| g.stamp.as_str()).collect();
            return Err(format!(
                "{} holds {} volumes ({}); pass --stamp to say which one. \
                 Taking the newest would put a volume the caller did not ask \
                 for behind a record that looks ordinary",
                dir.display(),
                stamps.len(),
                stamps.join(", ")
            )
            .into());
        }
    };
    if let Some(detail) = &chosen.site_disagreement {
        return Err(format!(
            "the files stamped {} are not all one radar: {detail}",
            chosen.stamp
        )
        .into());
    }
    Ok(chosen)
}

fn volumes(args: &Args) -> Result<(), Box<dyn Error>> {
    let dir = PathBuf::from(args.require("dir")?);
    let groups = survey_dir(&dir)?;
    let record = serde_json::json!({
        "schema": VOLUMES_SCHEMA,
        "dir": dir.display().to_string(),
        "volumes": groups.iter().map(|group| serde_json::json!({
            "stamp": group.stamp,
            "site": group.site,
            "files": group.files.len(),
            "bytes": group.bytes,
            "objects": group.objects,
            "quantities": group.quantities,
            "elevations_deg": group.elevations_deg,
            "split_files": group.files.len() > 1,
            "site_disagreement": group.site_disagreement,
        })).collect::<Vec<_>>(),
    });
    println!("{}", serde_json::to_string_pretty(&record)?);
    Ok(())
}

fn pack(args: &Args) -> Result<(), Box<dyn Error>> {
    let out = PathBuf::from(args.require("out")?);
    let quantities: Vec<String> = match args.get("quantities") {
        None => Vec::new(),
        Some(list) => list
            .split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(str::to_string)
            .collect(),
    };
    // Dual-PRF outlier correction (rw_odim::dualprf): on unless `--dual-prf off`.
    let dual_prf = match args.get("dual-prf") {
        None | Some("on") => true,
        Some("off") => false,
        Some(other) => {
            return Err(format!("--dual-prf takes on or off, not {other:?}\n\n{USAGE}").into());
        }
    };
    let options = PackOptions {
        quantities: quantities.clone(),
        max_elevation_deg: args.number("max-elevation-deg", 90.0)?,
        max_range_km: args.number("max-range-km", f64::INFINITY)?,
    };

    let decode = DecodeOptions {
        quantities: if quantities.is_empty() {
            None
        } else {
            Some(quantities)
        },
        sweep_indices: None,
        geometry_only: false,
    };
    // Exactly one of --file and --dir. Accepting both and preferring one
    // would let a caller believe they packed the directory when they packed
    // the file.
    let files: Vec<PathBuf> = args.all("file").into_iter().map(PathBuf::from).collect();
    let mut formats: Vec<&'static str> = Vec::new();
    let (volume, name, bytes, digest, report) = match (files.is_empty(), args.get("dir")) {
        (false, Some(_)) => {
            return Err(
                format!("--file and --dir are two different volumes; pass one\n\n{USAGE}").into(),
            );
        }
        (true, None) => {
            return Err(format!("pack needs --file or --dir\n\n{USAGE}").into());
        }
        (false, None) => {
            let (volume, used) = read_world(&files, args.get("site"), &decode)?;
            formats = used;
            if files.len() == 1 {
                let (name, bytes, digest) = read_file_identity(&files[0])?;
                (volume, name, bytes, digest, None)
            } else {
                let report = world_report(&files, &volume)?;
                let name = format!(
                    "{} @ {} ({} files)",
                    volume.source.label(),
                    report.stamp,
                    report.files
                );
                let bytes = report.total_bytes;
                let digest = report.manifest_sha256.clone();
                (volume, name, bytes, digest, Some(report))
            }
        }
        (true, Some(dir)) => {
            let dir = PathBuf::from(dir);
            let group = group_for(&dir, args.get("stamp"))?;
            let (volume, report) = assemble(&group.files, &decode)?;
            let name = format!("{} @ {} ({} files)", group.site, report.stamp, report.files);
            let bytes = report.total_bytes;
            let digest = report.manifest_sha256.clone();
            (volume, name, bytes, digest, Some(report))
        }
    };
    let (meta, payload, census) = build_pack_reported(
        &volume,
        &options,
        dual_prf,
        &name,
        bytes,
        &digest,
        report.as_ref(),
    )?;

    // The writer's own reader, on the writer's own bytes, before anything
    // downstream is asked to trust the file.
    let written = write_pack(&out, &meta, &payload)?;
    let round_trip = std::fs::read(&out)?;
    let (checked, _) = rw_nexrad::pack::decode_pack(&round_trip).map_err(|error| {
        format!(
            "rw_odim wrote {} and then refused to read it back: {error}",
            out.display()
        )
    })?;

    let record = serde_json::json!({
        "schema": PACK_SCHEMA,
        "pack": out.display().to_string(),
        "pack_bytes": written,
        "pack_schema": checked.schema,
        "source_file": name,
        "source_bytes": bytes,
        "source_sha256": digest,
        "format": if formats.is_empty() { vec!["odim-h5"] } else { formats },
        "site": checked.site.id,
        "site_alt_m": checked.site.alt_m,
        "valid_time": checked.volume.valid_time,
        "sweeps": checked.sweeps.len(),
        "dropped_sweeps": checked.dropped_sweeps,
        "dropped_moments": checked.dropped_moments,
        "trimmed_gates": checked.trimmed_gates,
        "sweeps_with_nyquist": checked
            .sweeps
            .iter()
            .filter(|sweep| sweep.nyquist_velocity_ms.is_some())
            .count(),
        "nyquist_granularity": rw_odim::pack::NYQUIST_GRANULARITY,
        // Dual-PRF outliers removed before the pack was written (on by
        // default; rw_odim::dualprf). A velocity on a dual-PRF sweep with no
        // declared NI or PRF pair is counted as skipped, never guessed.
        "dual_prf": {
            "correction": if dual_prf { "on" } else { "off" },
            "moments_examined": census.dual_prf.moments_examined,
            "moments_with_ray_prf": census.dual_prf.moments_with_ray_prf,
            "moments_skipped": census.dual_prf.moments_skipped,
            "gates_measured": census.dual_prf.gates_measured,
            "gates_corrected": census.dual_prf.gates_corrected,
        },
        "gates": checked
            .sweeps
            .iter()
            .map(|sweep| {
                sweep
                    .moments
                    .iter()
                    .map(|moment| sweep.radial_count * moment.gate_count)
                    .sum::<usize>()
            })
            .sum::<usize>(),
        // Present only for an assembled volume, so a reader can tell "one
        // file" from "twenty files that agreed" without inspecting the pack.
        // `source_sha256` above is a manifest digest when this is present.
        "assembled": report.as_ref().map(|report| serde_json::json!({
            "files": report.files,
            "sweeps": report.sweeps,
            "sweeps_merged": report.merged,
            "stamp": report.stamp,
            "manifest_sha256": report.manifest_sha256,
            "members": report.members.iter().map(|member| serde_json::json!({
                "name": member.name,
                "bytes": member.bytes,
                "sha256": member.sha256,
            })).collect::<Vec<_>>(),
        })),
    });
    println!("{}", serde_json::to_string_pretty(&record)?);
    Ok(())
}

fn nyquist(args: &Args) -> Result<(), Box<dyn Error>> {
    let file = PathBuf::from(args.require("file")?);
    let volume = match rw_odim::world::sniff_path(&file)? {
        Some(rw_odim::world::Format::OdimH5) | None => {
            read_volume_with(&file, &DecodeOptions::geometry_only())?
        }
        Some(_) => {
            read_world(
                std::slice::from_ref(&file),
                args.get("site"),
                &DecodeOptions::all(),
            )?
            .0
        }
    };
    let sweeps: Vec<_> = volume
        .sweeps
        .iter()
        .map(|sweep| {
            serde_json::json!({
                "sweep_index": sweep.index,
                "elevation_deg": sweep.elevation_deg,
                "nyquist_ms": sweep.nyquist.interval_ms,
                "source": format!("{:?}", sweep.nyquist.source),
                "dual_prf": sweep.nyquist.dual_prf,
                "high_prf_hz": sweep.nyquist.high_prf_hz,
                "low_prf_hz": sweep.nyquist.low_prf_hz,
                "per_ray_prf": sweep.ray_prf_hz.is_some(),
                "dealiasable": !matches!(sweep.nyquist.source, NyquistSource::Unavailable),
            })
        })
        .collect();
    let record = serde_json::json!({
        "schema": NYQUIST_SCHEMA,
        "file": file.display().to_string(),
        "granularity": rw_odim::pack::NYQUIST_GRANULARITY,
        "sweeps": sweeps,
    });
    println!("{}", serde_json::to_string_pretty(&record)?);
    Ok(())
}

/// Read one station from one or more files of any format `rw_odim::world`
/// recognises, and fold them into one volume.
fn read_world(
    files: &[PathBuf],
    site: Option<&str>,
    decode: &DecodeOptions,
) -> Result<(rw_odim::PolarVolume, Vec<&'static str>), Box<dyn Error>> {
    let mut volumes = Vec::new();
    let mut formats = Vec::new();
    for file in files {
        let read = rw_odim::world::read_path(file, site, decode)?;
        if !formats.contains(&read.format.token()) {
            formats.push(read.format.token());
        }
        volumes.extend(read.volumes);
    }
    let what = files
        .iter()
        .map(|file| file.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let chosen = rw_odim::world::select_site(volumes, site, &what)?;
    Ok((rw_odim::world::merge_station(chosen, &what)?, formats))
}

/// The member manifest for a volume folded from several files, in the shape
/// the pack's `assembled` block already carries for split ODIM scans.
fn world_report(
    files: &[PathBuf],
    volume: &rw_odim::PolarVolume,
) -> Result<rw_odim::AssembleReport, Box<dyn Error>> {
    let mut members = Vec::new();
    let mut total = 0usize;
    for file in files {
        let (name, bytes, sha256) = read_file_identity(file)?;
        total += bytes;
        members.push(rw_odim::MemberFile {
            name,
            bytes,
            sha256,
        });
    }
    let stamp = volume
        .nominal_time
        .map(|time| time.format("%Y%m%dT%H%M%SZ").to_string())
        .unwrap_or_else(|| "unstamped".to_string());
    Ok(rw_odim::AssembleReport {
        files: files.len(),
        sweeps: volume.sweeps.len(),
        merged: 0,
        stamp,
        manifest_sha256: rw_odim::assemble::manifest_sha256(&members),
        members,
        total_bytes: total,
    })
}

fn stations(args: &Args) -> Result<(), Box<dyn Error>> {
    let file = PathBuf::from(args.require("file")?);
    let bytes =
        std::fs::read(&file).map_err(|error| format!("cannot read {}: {error}", file.display()))?;
    let name = file.display().to_string();
    let list = match rw_odim::world::sniff(&bytes) {
        Some(rw_odim::world::Format::JmaGrib2Tar) => rw_odim::world::jma::stations(&bytes, &name)?,
        Some(format) => {
            return Err(format!(
                "{name} is {}, which holds one station; `stations` lists multi-station files",
                format.token()
            )
            .into());
        }
        None => return Err(format!("{name} is not a format rw_odim reads").into()),
    };
    let record = serde_json::json!({
        "schema": STATIONS_SCHEMA,
        "file": name,
        "stations": list,
    });
    println!("{}", serde_json::to_string_pretty(&record)?);
    Ok(())
}
