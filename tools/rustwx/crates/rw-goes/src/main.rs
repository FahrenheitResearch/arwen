//! `rw_goes` — gpuwm's GOES ABI L2 cloud-product front door.
//!
//! The satellite twin of `rw_nexrad`, built to
//! `docs/obs-goes-cwp-bridge-design.md`.  A fail-closed CLI with three
//! jobs and no opinions beyond them:
//!
//! * **acquire** — list and download ABI L2 cloud granules for a
//!   `(satellite, sector, product set, time window)` from the anonymous
//!   `noaa-goes{NN}` buckets, through the rw-sat content-addressed cache;
//! * **derive** — decode one scan's COD + CPS + ACTP trio through the
//!   patched `rw-sat` (DQF-gated, counts kept), refuse unless their fixed
//!   grids are bit-identical, and derive WoFS-style cloud water path;
//! * **pack** — write flat `GPWMGOES` packs that the Python obs layer
//!   reads with `json` + `numpy.frombuffer` and nothing else.
//!
//! It moves bytes, gates them by the numbers the algorithm published, and
//! reports facts.  It does **not** superob, regrid, or assign obs error:
//! those are the Python stage's.
//!
//! Two rules run through the whole thing, and both are refusals rather
//! than repairs.  A granule set that cannot be completed is a hard error
//! naming what is missing, never a partial pack.  Planes are only ever
//! combined across granules whose navigation is identical — combining
//! planes across two fixed grids would be fabrication, so it is refused
//! with the grids' own numbers in the message.
//!
//! That second rule has a measured consequence: at CONUS the CWP trio is
//! the 2 km grid (2500 x 1500) and ACHA/CTP are the 10 km grid (500 x 300),
//! so the vertical-placement fields get their OWN pack — `cloud-top`,
//! schema `gpuwm-obs.goes-cloudtop.v1` — paired to the CWP pack by
//! `(satellite, sector, scan_start)`.  Two packs per scan, deliberately;
//! see `cloudtop` for the ruling and the reason.
//!
//! In the payload, `NaN` means *no observation* and `0.0` means
//! *clear-sky zero*; nothing in this crate ever writes one for the other.
//!
//! ```text
//! rw_goes list      --satellite G19 --sector C --products COD,CPS,ACTP \
//!                   --start 2026-08-04T18:00:00Z --end 2026-08-04T18:30:00Z
//! rw_goes fetch     --satellite G19 --sector C --products COD,CPS,ACTP \
//!                   --start ... --end ... --cache DIR [--out DIR]
//! rw_goes cwp       --cod FILE --cps FILE --actp FILE \
//!                   --out FILE.goespack [--window XS,XC,YS,YC]
//! rw_goes cloud-top --acha FILE [--ctp FILE] --out FILE.goespack \
//!                   [--pairs-with FILE.goespack] [--window XS,XC,YS,YC]
//! rw_goes verify    --pack FILE.goespack
//! ```

mod cloudtop;
mod pack;

use std::collections::BTreeMap;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use chrono::{DateTime, NaiveDateTime, TimeZone, Timelike, Utc};
use serde::Serialize;

use rw_sat::abi::{AbiFixedGrid, AbiSector, read_goes_abi_field, read_goes_abi_field_window};
use rw_sat::cloud::{
    CloudProduct, CloudProductField, DqfReport, DqfRule, read_cloud_product_field,
    read_cloud_product_field_window,
};
use rw_sat::cwp::{CwpCounts, cloud_water_path_plane};
use rw_sat::goes::{GoesSatellite, parse_goes_abi_filename};
use rw_sat::s3::{
    S3Object, Sector, bucket_for_satellite, build_agent, download_object, list_s3_objects,
    object_filename, object_url, product_hour_prefix,
};
use rw_store::atomic::atomic_write_bytes;

use cloudtop::{
    CLOUDTOP_READABLE_SCHEMAS, CLOUDTOP_SCHEMA, CloudTopMeta, NO_REGRID, SiblingEntry,
    decode_cloudtop_pack, pairs_with_schema, write_cloudtop_pack,
};
use pack::{
    ArrayEntry, CWP_READABLE_SCHEMAS, CWP_SCHEMA, CoefficientTable, CwpRow, DqfRow, PackMeta, PayloadBuilder,
    ProjectionEntry, SourceEntry, boxed_error, decode_pack, hex_sha256, pack_schema, write_pack,
};

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// `GPUWM_BRIDGE_SOURCE_REV=<40-hex commit>`: the source revision this
/// binary was built from, embedded so the gpuwm release cut can prove a
/// staged bridge matches the commit being released by reading bytes
/// alone (`tools/build_bridge_bundle.py pin --source-rev`).  `build.rs`
/// injects the value; `main` references the constant so the linker
/// cannot discard it.
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

pub const LIST_SCHEMA: &str = "gpuwm-obs.goes-list.v1";
pub const FETCH_SCHEMA: &str = "gpuwm-obs.goes-fetch.v1";
pub const BUILD_SCHEMA: &str = "gpuwm-obs.goes-cwp-build.v1";
pub const CLOUDTOP_BUILD_SCHEMA: &str = "gpuwm-obs.goes-cloudtop-build.v1";
pub const VERIFY_SCHEMA: &str = "gpuwm-obs.goes-cwp-verify.v1";
pub const CLOUDTOP_VERIFY_SCHEMA: &str = "gpuwm-obs.goes-cloudtop-verify.v1";

/// The exact `--abi` line the Python bridge pins, in the shape
/// `rw_nexrad` and `rw-obs`'s three bins established: the record schema
/// the acquisition subcommand prints, then the schemas of the two packs
/// this bin writes.  It is not a version number -- a rebuilt-but-
/// unchanged binary still matches, and a binary whose record or pack
/// shapes moved does not, which is the only question the Python
/// wrapper's probe and the release cut are asking.
///
/// It is a literal rather than a `concat!` of the three constants
/// because `concat!` takes literals only and this workspace builds
/// `--locked --offline` from a fixed vendor closure, so a const-format
/// crate is not available to spell it. `abi_marker_names_every_contract
/// _it_pins` below binds the literal to the three constants instead, so
/// a schema bump that forgets this line fails the crate's own tests.
const ABI_MARKER: &str = "gpuwm-obs.goes-fetch.v1\tgpuwm-obs.goes-cwp.v2\t\
gpuwm-obs.goes-cloudtop.v2";

/// The ABI scan mode the operational feed has run since 2019.  A flip to
/// the contingency mode is a flag, not a rebuild.
const DEFAULT_MODE: u8 = 6;

/// The CWP relation the pack declares, verbatim from `rw_sat::cwp`.
const CWP_FORMULA: &str = "CWP[g m^-2] = (2/3) * tau * r_e[um] * rho[g cm^-3]";

/// Where the cache lands when neither `--cache` nor `--out` says.
const DEFAULT_CACHE_DIR: &str = ".rw-goes-cache";

const USAGE: &str = "\
usage: rw_goes <list|fetch|cwp|cloud-top|verify> [OPTIONS]
       rw_goes --version | --help | --abi

  list       report the ABI L2 cloud granules a (satellite, sector, product
             set, window) resolves to, grouped into scans, moving no payload
  fetch      download the granules of the complete scans in that window into
             the content-addressed cache and print a fetch record
  cwp        decode one scan's COD + CPS + ACTP trio, derive cloud water path,
             and write a `gpuwm-obs.goes-cwp.v1` pack (the 2 km pack)
  cloud-top  decode the same scan's ACHA / CTP and write a
             `gpuwm-obs.goes-cloudtop.v1` pack (the 10 km pack)
  verify     re-read either pack and re-prove its header, schema and payload
             digest

two packs per scan, deliberately
  Measured on real GOES-19 CONUS granules: COD/CPS/ACTP are the 2 km fixed
  grid (2500 x 1500) and ACHA/CTP are the 10 km fixed grid (500 x 300) --
  same satellite, same sector, same scan start, same projection, different
  grid.  A pack's whole promise is that every plane in it is bit-identically
  on the grid it states, so `cwp` REFUSES a mixed-grid pack (naming both
  shapes) and the vertical-placement fields get their own pack instead.
  Upsampling cloud-top height to 2 km here would break the guarantee the
  format exists to make and would bury an interpolation choice in an ingest
  tool where no science reviewed it.  The join happens at the consumer,
  where the interpolation is explicit and recorded in that stage's receipt.
  Both packs carry the same (satellite, sector, scan_start) triple, which is
  the pairing key; `cloud-top --pairs-with` additionally pins the CWP pack's
  payload digest so the pairing can be proved rather than assumed.

acquisition options (list, fetch)
  --satellite ID        G16 / G17 / G18 / G19 (data, not a gate).  Selects the
                        anonymous open-data bucket noaa-goes{NN} unless
                        --bucket overrides it
  --sector TOKEN        C (CONUS), F (full disk), M1, M2.  NOAA publishes no
                        mesoscale COD or CTP, so a meso request naming either
                        is refused rather than silently listing nothing
  --products LIST       comma-separated family tokens: COD,CPS,ACTP and
                        optionally ACHA,CTP (also ACM).  Required: the product
                        set defines what makes a scan complete, so it is never
                        guessed
  --start TIME          window start, 2026-08-04T18:00:00Z or 20260804T180000
  --end TIME            window end (inclusive)
  --bucket NAME         override the satellite's open-data bucket
  --mode N              ABI scan mode token in the key prefix (default 6; 3 is
                        the contingency mode)

fetch options
  --cache DIR           object cache root (default: <out>/.cache, else
                        ./.rw-goes-cache).  Layout is rw-sat's:
                        <cache>/satellite/<bucket>/<key>
  --no-cache            always re-download, never read the cache
  --out DIR             also publish each granule here under its plain object
                        file name
  --limit-scans N       keep at most the first N complete scans in the window
  --complete-only       refuse the whole run if ANY scan in the window is
                        missing a requested product, naming the scan times and
                        the missing products.  Without it an incomplete scan is
                        reported and skipped, and only complete scans are ever
                        downloaded -- a half-fetched scan can never become a
                        pack, so it is not fetched

reading the DQF plane: index it over the WHOLE pack
  The `_dqf` planes are full-grid planes and must be indexed over every
  pixel of the pack, NEVER over the surviving-CWP mask.  The obs-error
  inflation factor is needed per pixel BEFORE superobbing decides which
  cells survive, so the two populations are different and are not
  interchangeable: on the measured CONUS scan 47,162 pixels have a NaN
  DQF (the flag itself was fill), and reading those as a bitfield gives
  garbage, silently -- `astype(np.uint16)` on a NaN does not raise.  Mask
  on `np.isfinite(dqf)` first, then read bits from the finite pixels.
  `dqf_missing` in the metadata is the count of exactly those pixels, so
  it is the number to check your mask against.

pairing a WINDOWED pack (nested campaigns)
  ORDER MATTERS, and it is the one thing worth not rediscovering:
    1.  rw_goes cwp       ... --window XS,XC,YS,YC --out nest.goespack
    2.  rw_goes cloud-top ... --pairs-with nest.goespack --out nest.cloudtop.goespack
  The sibling block can only pin a pack that already exists, so the
  windowed CWP pack is built first and the cloud-top points at it.  The
  cloud-top's sibling block then pins that window's digest and records
  its 2 km window, so the pairing is provable rather than asserted: a
  consumer hashes the CWP pack it holds and compares.  Pass the window
  you actually built -- a cloud-top whose sibling pins the full-sector
  pack will (correctly) not match a windowed one, because they are
  different packs with different digests.  A worked example with real
  digests is in docs/obs-goes-cwp-bridge-design.md.
  The cloud-top stays full-sector on purpose.  A 2 km window has no exact
  10 km counterpart in general -- 512 two-km columns is 102.4 ten-km
  columns -- so this tool will not round one into existence; that is the
  same silent resampling the two-pack split exists to prevent.  Window
  the cloud-top yourself with --window if you want a smaller one, and
  choose a box that covers your 2 km window; the pack records whichever
  window it was given, on its own grid.

per-pixel DQF planes (schema v2, 2026-08-06)
  Both packs carry one `<product>_dqf` plane per source granule -- cod_dqf,
  cps_dqf, actp_dqf (and acha_dqf / ctp_dqf in the cloud-top pack) -- holding
  the DQF exactly as published and UNGATED, appended after every v1 plane so
  no existing plane index moves.  `<f4` like every other plane, which is
  lossless for a u16 flag word (every integer below 2^24 is exact in f32);
  NaN means that DQF pixel was itself fill or out of range, never a stand-in
  for the real value 0.  Each `sources` row names its plane in `dqf_plane`,
  so a consumer looks the plane up rather than guessing the name.
  v1 carried only the DQF counts and the condemn mask.  Those are a summary:
  they say how many pixels each cause condemned, not which -- so the thin
  (256) and thick (512) DCOMP bits, which the condemn mask deliberately does
  NOT gate and which the CWP observation operator inflates obs error on,
  could not be recovered from a v1 pack at all.  The counts and the mask are
  unchanged and still there; the plane is what they could not answer.

cwp options (planes: cwp, phase, cod, cps, lat, lon, then the _dqf planes)
  --cod FILE            ABI-L2-COD granule (required)
  --cps FILE            ABI-L2-CPS granule (required)
  --actp FILE           ABI-L2-ACTP granule (required)
  --acha FILE           ABI-L2-ACHA granule.  Accepted, and refused with both
                        grid shapes wherever ACHA is not on the trio's grid --
                        which at CONUS it never is.  Use `cloud-top` instead
  --ctp FILE            ABI-L2-CTP granule; same rule as --acha
  --out FILE            pack destination (required)
  --window XS,XC,YS,YC  decode only this fixed-grid window (x start, x count,
                        y start, y count), the same window for every product.
                        Recorded in the pack so a consumer can place it back
                        on the full sector

cloud-top options (planes: cloud_top_height_m, cloud_top_pressure_hpa, lat,
                   lon, then the _dqf planes)
  --acha FILE           ABI-L2-ACHA granule; adds the cloud_top_height_m plane
  --ctp FILE            ABI-L2-CTP granule; adds the cloud_top_pressure_hpa
                        plane.  At least one of --acha / --ctp is required,
                        and given both they must share one scan and one grid
  --out FILE            pack destination (required)
  --pairs-with FILE     the `gpuwm-obs.goes-cwp.v1` pack of the same scan.
                        Its scan identity is checked against this one and
                        refused on any difference, and its payload digest is
                        recorded, so a consumer can prove the pairing
  --window XS,XC,YS,YC  window into THIS pack's grid.  Not comparable to a
                        --window index in the CWP sibling: the grids differ

DQF gating (both pack subcommands)
  Every granule is gated by its product's default DQF rule: enumerated
  (good is exactly DQF == 0) for ACHA/ACM/ACTP/CTP, and the measured DCOMP
  bitfield mask 88 (snow/sea-ice 8, twilight 16, sun glint 64, plus any
  fill DQF) for COD/CPS.  The rule name and the condemn mask are recorded
  per product in the pack, so how a pixel was gated is a value and not an
  inference.  The ice and mixed-phase CWP coefficients are PROVISIONAL and
  the pack says so.  Neither subcommand resamples anything.

verify options
  --pack FILE           the pack to re-prove (--out is accepted too).  The
                        family is read from the pack's own schema, not from
                        its file name.  Reads every version this tool has
                        ever written -- v1 as well as v2 -- because a
                        receipt that names a pack's digest is worth nothing
                        once that pack can no longer be re-verified.  The
                        receipt reports `per_pixel_dqf`, false for a v1
                        pack: a fact about the version, not a fault
";

fn main() -> ExitCode {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(output) => {
            print!("{output}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("rw_goes: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[String]) -> Result<String, Box<dyn Error>> {
    let Some(first) = args.first() else {
        return Ok(USAGE.to_string());
    };
    match first.as_str() {
        "--help" | "-h" | "help" => return Ok(USAGE.to_string()),
        "--version" | "-V" => return Ok(format!("rw_goes {VERSION}\n")),
        "--abi" => return Ok(format!("{ABI_MARKER}\n")),
        _ => {}
    }
    let options = Options::parse(&args[1..])?;
    match first.as_str() {
        "list" => cmd_list(&options),
        "fetch" => cmd_fetch(&options),
        "cwp" => cmd_cwp(&options),
        "cloud-top" => cmd_cloud_top(&options),
        "verify" => cmd_verify(&options),
        other => Err(boxed_error(format!(
            "unknown subcommand {other:?}\n\n{USAGE}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// options
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct Options {
    satellite: Option<String>,
    sector: Option<String>,
    products: Option<String>,
    start: Option<String>,
    end: Option<String>,
    bucket: Option<String>,
    mode: Option<u8>,
    cache: Option<PathBuf>,
    no_cache: bool,
    out: Option<PathBuf>,
    limit_scans: Option<usize>,
    complete_only: bool,
    cod: Option<PathBuf>,
    cps: Option<PathBuf>,
    actp: Option<PathBuf>,
    acha: Option<PathBuf>,
    ctp: Option<PathBuf>,
    window: Option<[usize; 4]>,
    pack: Option<PathBuf>,
    pairs_with: Option<PathBuf>,
}

impl Options {
    fn parse(args: &[String]) -> Result<Self, Box<dyn Error>> {
        let mut options = Options::default();
        let mut index = 0;
        while index < args.len() {
            let flag = args[index].as_str();
            let mut value = || -> Result<String, Box<dyn Error>> {
                index += 1;
                args.get(index)
                    .cloned()
                    .ok_or_else(|| boxed_error(format!("{flag} needs a value")))
            };
            match flag {
                "--satellite" => options.satellite = Some(value()?),
                "--sector" => options.sector = Some(value()?),
                "--products" => options.products = Some(value()?),
                "--start" => options.start = Some(value()?),
                "--end" => options.end = Some(value()?),
                "--bucket" => options.bucket = Some(value()?),
                "--mode" => {
                    let raw = value()?;
                    options.mode = Some(raw.parse::<u8>().map_err(|_| {
                        boxed_error(format!("--mode expects an ABI scan mode token, got {raw:?}"))
                    })?)
                }
                "--cache" => options.cache = Some(PathBuf::from(value()?)),
                "--no-cache" => options.no_cache = true,
                "--out" => options.out = Some(PathBuf::from(value()?)),
                "--limit-scans" => {
                    let raw = value()?;
                    let count = raw.parse::<usize>().map_err(|_| {
                        boxed_error(format!("--limit-scans expects a count, got {raw:?}"))
                    })?;
                    if count == 0 {
                        return Err(boxed_error(
                            "--limit-scans must be at least 1; zero scans is not a fetch",
                        ));
                    }
                    options.limit_scans = Some(count)
                }
                "--complete-only" => options.complete_only = true,
                "--cod" => options.cod = Some(PathBuf::from(value()?)),
                "--cps" => options.cps = Some(PathBuf::from(value()?)),
                "--actp" => options.actp = Some(PathBuf::from(value()?)),
                "--acha" => options.acha = Some(PathBuf::from(value()?)),
                "--ctp" => options.ctp = Some(PathBuf::from(value()?)),
                "--window" => {
                    let raw = value()?;
                    options.window = Some(parse_window(&raw)?)
                }
                "--pack" => options.pack = Some(PathBuf::from(value()?)),
                "--pairs-with" => options.pairs_with = Some(PathBuf::from(value()?)),
                other => {
                    return Err(boxed_error(format!(
                        "unknown option {other:?}\n\n{USAGE}"
                    )));
                }
            }
            index += 1;
        }
        Ok(options)
    }

    fn satellite(&self) -> Result<GoesSatellite, Box<dyn Error>> {
        let raw = self
            .satellite
            .as_deref()
            .ok_or_else(|| boxed_error("--satellite is required"))?;
        Ok(GoesSatellite::parse(raw))
    }

    /// The bucket to read.  `--bucket` wins; otherwise the satellite's own
    /// open-data bucket, which is where the refusal for an unknown
    /// satellite comes from.
    fn bucket(&self) -> Result<String, Box<dyn Error>> {
        if let Some(bucket) = &self.bucket {
            let trimmed = bucket.trim();
            if trimmed.is_empty() {
                return Err(boxed_error("--bucket is empty"));
            }
            return Ok(trimmed.to_string());
        }
        let satellite = self
            .satellite
            .as_deref()
            .ok_or_else(|| boxed_error("--satellite is required (or name a --bucket)"))?;
        bucket_for_satellite(satellite)
    }

    fn sector(&self) -> Result<Sector, Box<dyn Error>> {
        let raw = self
            .sector
            .as_deref()
            .ok_or_else(|| boxed_error("--sector is required (C, F, M1, M2)"))?;
        Sector::parse(raw)
            .ok_or_else(|| boxed_error(format!("unknown ABI sector {raw:?}; expected C, F, M1, M2")))
    }

    /// The requested product set, in the order the caller named it and
    /// with duplicates collapsed.  Order is kept because it is the order
    /// every scan's granule slots, and every receipt row, are reported in.
    fn products(&self) -> Result<Vec<CloudProduct>, Box<dyn Error>> {
        let raw = self.products.as_deref().ok_or_else(|| {
            boxed_error(
                "--products is required (e.g. COD,CPS,ACTP); the product set is what makes \
                 a scan complete, so it is never guessed",
            )
        })?;
        let mut products: Vec<CloudProduct> = Vec::new();
        for token in raw.split(',') {
            let token = token.trim();
            if token.is_empty() {
                continue;
            }
            let product = CloudProduct::parse(token).ok_or_else(|| {
                boxed_error(format!(
                    "unknown cloud product {token:?}; expected ACHA, ACM, ACTP, COD, CPS or CTP"
                ))
            })?;
            if !products.contains(&product) {
                products.push(product);
            }
        }
        if products.is_empty() {
            return Err(boxed_error(format!(
                "--products {raw:?} names no cloud product"
            )));
        }
        Ok(products)
    }

    fn window(&self) -> Result<(DateTime<Utc>, DateTime<Utc>), Box<dyn Error>> {
        let start = parse_time(
            self.start
                .as_deref()
                .ok_or_else(|| boxed_error("--start is required"))?,
        )?;
        let end = parse_time(
            self.end
                .as_deref()
                .ok_or_else(|| boxed_error("--end is required"))?,
        )?;
        if end < start {
            return Err(boxed_error(format!(
                "--end {} precedes --start {}",
                iso8601(end),
                iso8601(start)
            )));
        }
        Ok((start, end))
    }

    fn mode(&self) -> u8 {
        self.mode.unwrap_or(DEFAULT_MODE)
    }

    fn cache_dir(&self) -> PathBuf {
        if let Some(cache) = &self.cache {
            return cache.clone();
        }
        match &self.out {
            Some(out) => out.join(".cache"),
            None => PathBuf::from(DEFAULT_CACHE_DIR),
        }
    }
}

fn parse_window(raw: &str) -> Result<[usize; 4], Box<dyn Error>> {
    let parts: Vec<&str> = raw.split(',').map(str::trim).collect();
    if parts.len() != 4 {
        return Err(boxed_error(format!(
            "--window expects X_START,X_COUNT,Y_START,Y_COUNT, got {raw:?} ({} field(s), not 4)",
            parts.len()
        )));
    }
    let mut values = [0usize; 4];
    for (slot, text) in values.iter_mut().zip(&parts) {
        *slot = text.parse::<usize>().map_err(|_| {
            boxed_error(format!(
                "--window component {text:?} is not a non-negative whole number"
            ))
        })?;
    }
    if values[1] == 0 || values[3] == 0 {
        return Err(boxed_error(format!(
            "--window {raw:?} selects no pixels: X_COUNT={} Y_COUNT={}",
            values[1], values[3]
        )));
    }
    Ok(values)
}

/// The time formats the acquisition flags accept, mirroring `rw_nexrad`.
fn parse_time(raw: &str) -> Result<DateTime<Utc>, Box<dyn Error>> {
    let trimmed = raw.trim();
    if let Ok(parsed) = DateTime::parse_from_rfc3339(trimmed) {
        return Ok(parsed.with_timezone(&Utc));
    }
    for format in [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%MZ",
        "%Y-%m-%dT%H:%M",
        "%Y%m%dT%H%M%SZ",
        "%Y%m%dT%H%M%S",
        "%Y%m%dT%H%MZ",
        "%Y%m%dT%H%M",
        "%Y%m%d%H%M",
    ] {
        if let Ok(naive) = NaiveDateTime::parse_from_str(trimmed, format) {
            return Ok(Utc.from_utc_datetime(&naive));
        }
    }
    Err(boxed_error(format!(
        "cannot read {raw:?} as a UTC time; try 2026-08-04T18:00:00Z or 20260804T180000"
    )))
}

/// ISO8601, carrying the tenth-of-a-second the GOES filename actually
/// states when it is not zero.  Scan start IS the scan's identity here, so
/// it is never rounded away.
fn iso8601(when: DateTime<Utc>) -> String {
    if when.timestamp_subsec_millis() == 0 {
        when.format("%Y-%m-%dT%H:%M:%SZ").to_string()
    } else {
        when.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string()
    }
}

/// Every hour directory the window touches, inclusive of both ends.
fn window_hours(start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<DateTime<Utc>> {
    let mut hours = Vec::new();
    let Some(mut cursor) = start
        .with_minute(0)
        .and_then(|value| value.with_second(0))
        .and_then(|value| value.with_nanosecond(0))
    else {
        return hours;
    };
    while cursor <= end {
        hours.push(cursor);
        cursor += chrono::Duration::hours(1);
    }
    hours
}

// ---------------------------------------------------------------------------
// listing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Granule {
    product: CloudProduct,
    object: S3Object,
    scan_start: DateTime<Utc>,
    scan_end: DateTime<Utc>,
}

/// One scan time and the requested products it published, in the order the
/// caller named them.
#[derive(Debug, Clone)]
struct Scan {
    start: DateTime<Utc>,
    slots: Vec<Option<Granule>>,
}

impl Scan {
    fn complete(&self) -> bool {
        self.slots.iter().all(Option::is_some)
    }

    fn granules(&self) -> impl Iterator<Item = &Granule> {
        self.slots.iter().flatten()
    }

    fn missing(&self, products: &[CloudProduct]) -> Vec<String> {
        products
            .iter()
            .zip(&self.slots)
            .filter(|(_, slot)| slot.is_none())
            .map(|(product, _)| product.family().to_string())
            .collect()
    }
}

#[derive(Debug)]
struct Listing {
    scans: Vec<Scan>,
    prefixes: Vec<String>,
    listed_objects: usize,
    matched_granules: usize,
}

/// List every requested product over every hour the window touches and
/// group what came back into scans keyed by the granule's own scan start.
fn list_scans(
    satellite: &GoesSatellite,
    sector: Sector,
    products: &[CloudProduct],
    bucket: &str,
    mode: u8,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
) -> Result<Listing, Box<dyn Error>> {
    // Refuse a sector a product is not published in before a single
    // request goes out: listing an absent prefix reports an empty sky,
    // which is a different fact from "NOAA does not publish this".
    let mut abi_products: Vec<String> = Vec::with_capacity(products.len());
    for product in products {
        let abi_product = product.abi_product(sector).ok_or_else(|| {
            boxed_error(format!(
                "NOAA publishes no {} product for sector {}; the ABI L2 cloud suite has no \
                 mesoscale COD or CTP, so a mesoscale CWP scan cannot be completed",
                product.family(),
                sector.slug()
            ))
        })?;
        abi_products.push(abi_product);
    }

    let agent = build_agent();
    let mut prefixes = Vec::new();
    let mut listed_objects = 0usize;
    let mut matched_granules = 0usize;
    let mut scans: BTreeMap<DateTime<Utc>, Scan> = BTreeMap::new();

    for hour in window_hours(start, end) {
        for (slot, (product, abi_product)) in products.iter().zip(&abi_products).enumerate() {
            let prefix = product_hour_prefix(abi_product, satellite, mode, hour);
            let objects = list_s3_objects(&agent, bucket, &prefix, None)?;
            listed_objects += objects.len();
            prefixes.push(prefix);
            for object in objects {
                let Ok(parsed) = parse_goes_abi_filename(object_filename(&object.key)) else {
                    // A key under the prefix whose name this build cannot
                    // read is not silently treated as a granule.
                    continue;
                };
                if parsed.start_time_utc < start || parsed.start_time_utc > end {
                    continue;
                }
                matched_granules += 1;
                let entry = scans.entry(parsed.start_time_utc).or_insert_with(|| Scan {
                    start: parsed.start_time_utc,
                    slots: vec![None; products.len()],
                });
                let granule = Granule {
                    product: *product,
                    object,
                    scan_start: parsed.start_time_utc,
                    scan_end: parsed.end_time_utc,
                };
                // A reprocessed granule shares its scan start with the
                // original and differs only in the `c` creation token, so
                // the lexicographically greater key is the newer file.
                match &entry.slots[slot] {
                    Some(existing) if existing.object.key >= granule.object.key => {}
                    _ => entry.slots[slot] = Some(granule),
                }
            }
        }
    }

    Ok(Listing {
        scans: scans.into_values().collect(),
        prefixes,
        listed_objects,
        matched_granules,
    })
}

#[derive(Serialize)]
struct WindowRecord {
    start: String,
    end: String,
    hours: usize,
    prefixes: Vec<String>,
}

#[derive(Serialize)]
struct ListCounts {
    listed_objects: usize,
    matched_granules: usize,
    scans: usize,
    complete_scans: usize,
    incomplete_scans: usize,
}

#[derive(Serialize)]
struct GranuleRecord {
    product: String,
    key: String,
    filename: String,
    url: String,
    scan_start: String,
    scan_end: String,
    size_bytes: u64,
    last_modified: String,
}

fn granule_record(bucket: &str, granule: &Granule) -> GranuleRecord {
    GranuleRecord {
        product: granule.product.family().to_string(),
        key: granule.object.key.clone(),
        filename: object_filename(&granule.object.key).to_string(),
        url: object_url(bucket, &granule.object.key),
        scan_start: iso8601(granule.scan_start),
        scan_end: iso8601(granule.scan_end),
        size_bytes: granule.object.size_bytes,
        last_modified: granule.object.last_modified.clone(),
    }
}

#[derive(Serialize)]
struct ScanRecord {
    scan_start: String,
    complete: bool,
    missing_products: Vec<String>,
    granules: Vec<GranuleRecord>,
}

fn cmd_list(options: &Options) -> Result<String, Box<dyn Error>> {
    let satellite = options.satellite()?;
    let sector = options.sector()?;
    let products = options.products()?;
    let bucket = options.bucket()?;
    let mode = options.mode();
    let (start, end) = options.window()?;

    let listing = list_scans(&satellite, sector, &products, &bucket, mode, start, end)?;
    let complete = listing.scans.iter().filter(|scan| scan.complete()).count();

    #[derive(Serialize)]
    struct ListRecord {
        schema: &'static str,
        status: &'static str,
        satellite: String,
        sector: String,
        bucket: String,
        mode: u8,
        products: Vec<String>,
        window: WindowRecord,
        scans: Vec<ScanRecord>,
        counts: ListCounts,
        total_bytes: u64,
    }

    let record = ListRecord {
        schema: LIST_SCHEMA,
        status: if complete == 0 { "EMPTY" } else { "READY" },
        satellite: satellite.as_str().to_string(),
        sector: sector_token_from_sector(sector).to_string(),
        bucket: bucket.clone(),
        mode,
        products: products.iter().map(|p| p.family().to_string()).collect(),
        window: WindowRecord {
            start: iso8601(start),
            end: iso8601(end),
            hours: window_hours(start, end).len(),
            prefixes: listing.prefixes.clone(),
        },
        counts: ListCounts {
            listed_objects: listing.listed_objects,
            matched_granules: listing.matched_granules,
            scans: listing.scans.len(),
            complete_scans: complete,
            incomplete_scans: listing.scans.len() - complete,
        },
        total_bytes: listing
            .scans
            .iter()
            .flat_map(Scan::granules)
            .map(|granule| granule.object.size_bytes)
            .sum(),
        scans: listing
            .scans
            .iter()
            .map(|scan| ScanRecord {
                scan_start: iso8601(scan.start),
                complete: scan.complete(),
                missing_products: scan.missing(&products),
                granules: scan
                    .granules()
                    .map(|granule| granule_record(&bucket, granule))
                    .collect(),
            })
            .collect(),
    };
    Ok(format!("{}\n", serde_json::to_string_pretty(&record)?))
}

// ---------------------------------------------------------------------------
// fetch
// ---------------------------------------------------------------------------

fn cmd_fetch(options: &Options) -> Result<String, Box<dyn Error>> {
    let satellite = options.satellite()?;
    let sector = options.sector()?;
    let products = options.products()?;
    let bucket = options.bucket()?;
    let mode = options.mode();
    let (start, end) = options.window()?;
    let cache_dir = options.cache_dir();

    let listing = list_scans(&satellite, sector, &products, &bucket, mode, start, end)?;
    let incomplete: Vec<&Scan> = listing.scans.iter().filter(|s| !s.complete()).collect();
    if options.complete_only && !incomplete.is_empty() {
        let detail = incomplete
            .iter()
            .map(|scan| format!("{} missing {}", iso8601(scan.start), scan.missing(&products).join("+")))
            .collect::<Vec<_>>()
            .join("; ");
        return Err(boxed_error(format!(
            "--complete-only: {} of {} scan(s) between {} and {} cannot be completed from \
             s3://{bucket}/ -- {detail}",
            incomplete.len(),
            listing.scans.len(),
            iso8601(start),
            iso8601(end)
        )));
    }

    let mut selected: Vec<&Scan> = listing.scans.iter().filter(|s| s.complete()).collect();
    let complete_scans = selected.len();
    if let Some(limit) = options.limit_scans {
        selected.truncate(limit);
    }
    if selected.is_empty() {
        let detail = if listing.scans.is_empty() {
            format!(
                "no {} granule at all was published under s3://{bucket}/ in that window",
                products
                    .iter()
                    .map(|p| p.family())
                    .collect::<Vec<_>>()
                    .join("/")
            )
        } else {
            format!(
                "{} scan(s) matched, none complete: {}",
                listing.scans.len(),
                listing
                    .scans
                    .iter()
                    .map(|scan| format!(
                        "{} missing {}",
                        iso8601(scan.start),
                        scan.missing(&products).join("+")
                    ))
                    .collect::<Vec<_>>()
                    .join("; ")
            )
        };
        return Err(boxed_error(format!(
            "no complete {} scan for {} sector {} between {} and {}: {detail}",
            products
                .iter()
                .map(|p| p.family())
                .collect::<Vec<_>>()
                .join("+"),
            satellite.as_str(),
            sector_token_from_sector(sector),
            iso8601(start),
            iso8601(end)
        )));
    }

    #[derive(Serialize)]
    struct FetchFileRecord {
        product: String,
        key: String,
        filename: String,
        path: String,
        cache_path: String,
        bytes: usize,
        sha256: String,
        cache_hit: bool,
    }

    #[derive(Serialize)]
    struct FetchScanRecord {
        scan_start: String,
        complete: bool,
        files: Vec<FetchFileRecord>,
    }

    #[derive(Serialize)]
    struct FetchCounts {
        matched_scans: usize,
        complete_scans: usize,
        incomplete_scans: usize,
        fetched_scans: usize,
        files: usize,
        cache_hits: usize,
    }

    #[derive(Serialize)]
    struct FetchRecord {
        schema: &'static str,
        status: &'static str,
        satellite: String,
        sector: String,
        bucket: String,
        mode: u8,
        products: Vec<String>,
        window: WindowRecord,
        cache_dir: String,
        out_dir: Option<String>,
        skipped_scans: Vec<ScanRecord>,
        scans: Vec<FetchScanRecord>,
        counts: FetchCounts,
        total_bytes: usize,
    }

    let agent = build_agent();
    let mut scan_records: Vec<FetchScanRecord> = Vec::new();
    let mut cache_hits = 0usize;
    let mut files = 0usize;
    let mut total_bytes = 0usize;
    for scan in &selected {
        let mut file_records = Vec::new();
        for granule in scan.granules() {
            let downloaded = download_object(
                &agent,
                &bucket,
                &cache_dir,
                &granule.object,
                !options.no_cache,
            )?;
            if downloaded.cache_hit {
                cache_hits += 1;
            }
            // The sha256 of record is taken over the bytes that are on
            // disk now, not over what the listing said they would be.
            let bytes = std::fs::read(&downloaded.path).map_err(|err| {
                boxed_error(format!(
                    "cannot re-read the fetched granule {}: {err}",
                    downloaded.path.display()
                ))
            })?;
            let sha256 = hex_sha256(&bytes);
            let name = object_filename(&granule.object.key).to_string();
            let published = match &options.out {
                Some(out) => {
                    let target = out.join(&name);
                    atomic_write_bytes(&target, &bytes)?;
                    target
                }
                None => downloaded.path.clone(),
            };
            files += 1;
            total_bytes += bytes.len();
            file_records.push(FetchFileRecord {
                product: granule.product.family().to_string(),
                key: granule.object.key.clone(),
                filename: name,
                path: published.to_string_lossy().to_string(),
                cache_path: downloaded.path.to_string_lossy().to_string(),
                bytes: bytes.len(),
                sha256,
                cache_hit: downloaded.cache_hit,
            });
        }
        scan_records.push(FetchScanRecord {
            scan_start: iso8601(scan.start),
            complete: true,
            files: file_records,
        });
    }

    let record = FetchRecord {
        schema: FETCH_SCHEMA,
        status: "READY",
        satellite: satellite.as_str().to_string(),
        sector: sector_token_from_sector(sector).to_string(),
        bucket: bucket.clone(),
        mode,
        products: products.iter().map(|p| p.family().to_string()).collect(),
        window: WindowRecord {
            start: iso8601(start),
            end: iso8601(end),
            hours: window_hours(start, end).len(),
            prefixes: listing.prefixes.clone(),
        },
        cache_dir: cache_dir.to_string_lossy().to_string(),
        out_dir: options.out.as_ref().map(|p| p.to_string_lossy().to_string()),
        skipped_scans: incomplete
            .iter()
            .map(|scan| ScanRecord {
                scan_start: iso8601(scan.start),
                complete: false,
                missing_products: scan.missing(&products),
                granules: scan
                    .granules()
                    .map(|granule| granule_record(&bucket, granule))
                    .collect(),
            })
            .collect(),
        counts: FetchCounts {
            matched_scans: listing.scans.len(),
            complete_scans,
            incomplete_scans: incomplete.len(),
            fetched_scans: scan_records.len(),
            files,
            cache_hits,
        },
        total_bytes,
        scans: scan_records,
    };
    Ok(format!("{}\n", serde_json::to_string_pretty(&record)?))
}

// ---------------------------------------------------------------------------
// cwp
// ---------------------------------------------------------------------------

/// One granule, decoded and DQF-gated, with the identity of the bytes it
/// came from.
#[derive(Debug)]
struct DecodedSource {
    product: CloudProduct,
    path: PathBuf,
    filename: String,
    bytes: usize,
    sha256: String,
    decoded: CloudProductField,
    /// The product's DQF plane exactly as published: decoded, and
    /// **ungated** — the flags are the evidence, so they are never
    /// masked by the rule that reads them.  NaN is a DQF pixel that was
    /// itself fill or out of range (what `DqfReport::dqf_missing`
    /// counts), never a substitute for the real value 0.
    dqf_values: Vec<f32>,
}

impl DecodedSource {
    fn scene(&self) -> &rw_sat::abi::GoesAbiScene {
        &self.decoded.field.scene
    }

    fn values(&self) -> &[f32] {
        &self.decoded.field.values
    }

    fn label(&self) -> String {
        format!("{} {}", self.product.family(), self.filename)
    }
}

fn decode_source(
    path: &Path,
    product: CloudProduct,
    window: Option<[usize; 4]>,
) -> Result<DecodedSource, Box<dyn Error>> {
    let filename = path
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string_lossy().to_string());
    // A file whose own name says it is a different product is refused
    // before a byte is read, not three variable lookups later.
    let parsed = parse_goes_abi_filename(&filename)?;
    let token = parsed
        .product
        .to_ascii_uppercase()
        .strip_prefix("ABI-L2-")
        .map(str::to_string)
        .unwrap_or_else(|| parsed.product.to_ascii_uppercase());
    if !token.starts_with(product.family()) {
        return Err(boxed_error(format!(
            "{filename} is a {} granule, but it was passed as --{}; the pack's product rows \
             are taken from the flag, so a swapped file would mislabel a plane",
            parsed.product,
            product.slug()
        )));
    }

    let raw = std::fs::read(path).map_err(|err| {
        boxed_error(format!(
            "cannot read the {} granule {}: {err}",
            product.family(),
            path.display()
        ))
    })?;
    let bytes = raw.len();
    let sha256 = hex_sha256(&raw);
    drop(raw);

    let decoded = match window {
        Some([x_start, x_count, y_start, y_count]) => {
            read_cloud_product_field_window(path, product, x_start, x_count, y_start, y_count)?
        }
        None => read_cloud_product_field(path, product)?,
    };
    // The gate consumes the DQF plane and keeps only its counts, so read
    // it again for the payload.  A second read of the same variable, not
    // a second interpretation of it: this is the plane `gate_by_dqf`
    // just saw, and the counts recorded beside it must add up against it.
    let dqf_field = match window {
        Some([x_start, x_count, y_start, y_count]) => read_goes_abi_field_window(
            path,
            product.dqf_variable(),
            x_start,
            x_count,
            y_start,
            y_count,
        )?,
        None => read_goes_abi_field(path, product.dqf_variable())?,
    };
    if dqf_field.values.len() != decoded.field.values.len() {
        return Err(boxed_error(format!(
            "{filename}: the {} DQF plane holds {} values but its {} plane holds {}; the pack \
             will not carry a quality flag that does not line up pixel for pixel with the \
             value it judges",
            product.dqf_variable(),
            dqf_field.values.len(),
            product.primary_variable(),
            decoded.field.values.len()
        )));
    }
    Ok(DecodedSource {
        product,
        path: path.to_path_buf(),
        filename,
        bytes,
        sha256,
        decoded,
        dqf_values: dqf_field.values,
    })
}

/// The payload plane carrying one product's per-pixel DQF.
fn dqf_plane_name(product: CloudProduct) -> String {
    format!("{}_dqf", product.slug())
}

fn sector_token_from_sector(sector: Sector) -> &'static str {
    match sector {
        Sector::Conus => "C",
        Sector::FullDisk => "F",
        Sector::Meso1 => "M1",
        Sector::Meso2 => "M2",
    }
}

/// The sector token a decoded scene reports, as a string the pack can
/// carry.  `Unknown` keeps the product token verbatim rather than being
/// mapped to a sector that was not observed.
fn sector_token_from_scene(sector: &AbiSector) -> String {
    match sector {
        AbiSector::Conus => "C".to_string(),
        AbiSector::FullDisk => "F".to_string(),
        AbiSector::Mesoscale1 => "M1".to_string(),
        AbiSector::Mesoscale2 => "M2".to_string(),
        AbiSector::Mesoscale => "M".to_string(),
        AbiSector::Unknown(value) => value.clone(),
    }
}

/// Where two fixed grids first stop being identical, for a refusal that
/// carries the numbers.
fn grid_difference(left: &AbiFixedGrid, right: &AbiFixedGrid) -> String {
    if left.nx != right.nx || left.ny != right.ny {
        return format!(
            "{}x{} against {}x{} (nx x ny)",
            left.nx, left.ny, right.nx, right.ny
        );
    }
    for (index, (a, b)) in left.x_scan_rad.iter().zip(&right.x_scan_rad).enumerate() {
        if a != b {
            return format!("x_scan_rad[{index}] {a:.12e} against {b:.12e} rad");
        }
    }
    for (index, (a, b)) in left.y_scan_rad.iter().zip(&right.y_scan_rad).enumerate() {
        if a != b {
            return format!("y_scan_rad[{index}] {a:.12e} against {b:.12e} rad");
        }
    }
    format!(
        "axis lengths {}/{} against {}/{}",
        left.x_scan_rad.len(),
        left.y_scan_rad.len(),
        right.x_scan_rad.len(),
        right.y_scan_rad.len()
    )
}

/// The pointer `cwp` appends to a grid refusal: at CONUS the trio is 2 km
/// and ACHA/CTP are 10 km, so the honest answer is a second pack, not a
/// resample.
const CLOUD_TOP_HINT: &str =
    " The ABI cloud suite publishes cloud-top height and pressure on a coarser fixed grid \
      than the CWP trio, so they get their own pack: run `rw_goes cloud-top` for the same \
      scan and pair the two by (satellite, sector, scan_start).";

/// Every reason two granules may not share one pack.  The reference is the
/// first decoded source; everything else is measured against it.
/// `grid_hint` is appended to a fixed-grid refusal and nothing else.
fn assert_one_scan(sources: &[&DecodedSource], grid_hint: &str) -> Result<(), Box<dyn Error>> {
    let Some((reference, rest)) = sources.split_first() else {
        return Err(boxed_error("no granules were decoded"));
    };
    for source in rest {
        if source.scene().start_time_utc != reference.scene().start_time_utc {
            return Err(boxed_error(format!(
                "these granules are not one scan: {} starts {}, {} starts {}; a pack carries \
                 exactly one scan time",
                reference.label(),
                iso8601(reference.scene().start_time_utc),
                source.label(),
                iso8601(source.scene().start_time_utc)
            )));
        }
        if source.scene().satellite != reference.scene().satellite {
            return Err(boxed_error(format!(
                "these granules are not one satellite: {} is {}, {} is {}",
                reference.label(),
                reference.scene().satellite.as_str(),
                source.label(),
                source.scene().satellite.as_str()
            )));
        }
        if source.scene().sector != reference.scene().sector {
            return Err(boxed_error(format!(
                "these granules are not one sector: {} is sector {}, {} is sector {}",
                reference.label(),
                sector_token_from_scene(&reference.scene().sector),
                source.label(),
                sector_token_from_scene(&source.scene().sector)
            )));
        }
        if source.scene().fixed_grid != reference.scene().fixed_grid {
            return Err(boxed_error(format!(
                "the fixed grids are not bit-identical: {} against {} differ at {}; combining \
                 planes across two grids would be fabrication, so this pack is refused.{}",
                reference.label(),
                source.label(),
                grid_difference(&reference.scene().fixed_grid, &source.scene().fixed_grid),
                grid_hint
            )));
        }
        if source.scene().projection != reference.scene().projection {
            return Err(boxed_error(format!(
                "the geostationary navigation differs: {} declares {:?}, {} declares {:?}; the \
                 projection IS the pack's navigation of record, so this pack is refused",
                reference.label(),
                reference.scene().projection,
                source.label(),
                source.scene().projection
            )));
        }
    }
    Ok(())
}

fn dqf_row(report: &DqfReport) -> DqfRow {
    DqfRow {
        total: report.total,
        primary_missing: report.primary_missing,
        dqf_missing: report.dqf_missing,
        dqf_bad: report.dqf_bad,
        masked: report.masked,
        finite: report.finite,
    }
}

fn cwp_row(counts: &CwpCounts) -> CwpRow {
    CwpRow {
        clear_zero: counts.clear_zero,
        liquid: counts.liquid,
        supercooled: counts.supercooled,
        mixed: counts.mixed,
        ice: counts.ice,
        unknown: counts.unknown,
        phase_missing: counts.phase_missing,
        input_missing: counts.input_missing,
        finite: counts.finite(),
    }
}

/// The DQF gate a product's default rule actually applies, as the pack
/// records it: the rule's name and, for the bitfield rule, the condemn
/// mask that was really in force.
fn dqf_rule_row(product: CloudProduct) -> (String, Option<u16>) {
    match product.dqf_rule() {
        DqfRule::Enumerated => ("enumerated".to_string(), None),
        DqfRule::Bitfield { condemn } => ("bitfield".to_string(), Some(condemn)),
    }
}

fn source_entry(source: &DecodedSource) -> SourceEntry {
    let (dqf_rule, condemn_mask) = dqf_rule_row(source.product);
    SourceEntry {
        product: source.product.family().to_string(),
        filename: source.filename.clone(),
        bytes: source.bytes,
        sha256: source.sha256.clone(),
        dqf_rule,
        condemn_mask,
        dqf: dqf_row(&source.decoded.dqf),
        dqf_plane: dqf_plane_name(source.product),
    }
}

fn push_plane(
    builder: &mut PayloadBuilder,
    planes: &mut BTreeMap<String, String>,
    order: &mut Vec<String>,
    name: &str,
    values: &[f32],
    shape: &[usize],
) -> Result<(), Box<dyn Error>> {
    let expected: usize = shape.iter().product();
    if values.len() != expected {
        return Err(boxed_error(format!(
            "plane {name} holds {} values but the declared shape {shape:?} needs {expected}",
            values.len()
        )));
    }
    let key = builder.push_f32(values, shape.to_vec());
    planes.insert(name.to_string(), key);
    order.push(name.to_string());
    Ok(())
}

fn cmd_cwp(options: &Options) -> Result<String, Box<dyn Error>> {
    let cod_path = options
        .cod
        .as_deref()
        .ok_or_else(|| boxed_error("--cod is required (the cloud optical depth granule)"))?;
    let cps_path = options
        .cps
        .as_deref()
        .ok_or_else(|| boxed_error("--cps is required (the cloud particle size granule)"))?;
    let actp_path = options
        .actp
        .as_deref()
        .ok_or_else(|| boxed_error("--actp is required (the cloud-top phase granule)"))?;
    let out = options
        .out
        .as_deref()
        .ok_or_else(|| boxed_error("--out is required (the pack destination)"))?;
    if out.is_dir() {
        return Err(boxed_error(format!(
            "--out {} is a directory; give the pack file path",
            out.display()
        )));
    }
    let window = options.window;

    let cod = decode_source(cod_path, CloudProduct::OpticalDepth, window)?;
    let cps = decode_source(cps_path, CloudProduct::ParticleSize, window)?;
    let actp = decode_source(actp_path, CloudProduct::CloudTopPhase, window)?;
    let acha = match options.acha.as_deref() {
        Some(path) => Some(decode_source(path, CloudProduct::CloudTopHeight, window)?),
        None => None,
    };
    let ctp = match options.ctp.as_deref() {
        Some(path) => Some(decode_source(path, CloudProduct::CloudTopPressure, window)?),
        None => None,
    };

    let mut sources: Vec<&DecodedSource> = vec![&cod, &cps, &actp];
    sources.extend(acha.as_ref());
    sources.extend(ctp.as_ref());
    assert_one_scan(&sources, CLOUD_TOP_HINT)?;

    let scene = cod.scene();
    let nx = scene.fixed_grid.nx;
    let ny = scene.fixed_grid.ny;
    let shape = vec![ny, nx];

    let (cwp_plane, counts) =
        cloud_water_path_plane(cod.values(), cps.values(), actp.values())?;

    let (lat, lon) = scene.lat_lon_mesh();

    let mut builder = PayloadBuilder::new();
    let mut planes: BTreeMap<String, String> = BTreeMap::new();
    let mut plane_order: Vec<String> = Vec::new();
    push_plane(
        &mut builder,
        &mut planes,
        &mut plane_order,
        "cwp",
        &cwp_plane,
        &shape,
    )?;
    push_plane(
        &mut builder,
        &mut planes,
        &mut plane_order,
        "phase",
        actp.values(),
        &shape,
    )?;
    push_plane(
        &mut builder,
        &mut planes,
        &mut plane_order,
        "cod",
        cod.values(),
        &shape,
    )?;
    push_plane(
        &mut builder,
        &mut planes,
        &mut plane_order,
        "cps",
        cps.values(),
        &shape,
    )?;
    if let Some(acha) = &acha {
        push_plane(
            &mut builder,
            &mut planes,
            &mut plane_order,
            "cloud_top_height_m",
            acha.values(),
            &shape,
        )?;
    }
    if let Some(ctp) = &ctp {
        push_plane(
            &mut builder,
            &mut planes,
            &mut plane_order,
            "cloud_top_pressure_hpa",
            ctp.values(),
            &shape,
        )?;
    }
    push_plane(
        &mut builder,
        &mut planes,
        &mut plane_order,
        "lat",
        &lat,
        &shape,
    )?;
    push_plane(
        &mut builder,
        &mut planes,
        &mut plane_order,
        "lon",
        &lon,
        &shape,
    )?;
    // The per-pixel DQF planes go LAST, after every v1 plane, so the
    // indices a positional reader already relies on do not move.
    for source in &sources {
        push_plane(
            &mut builder,
            &mut planes,
            &mut plane_order,
            &dqf_plane_name(source.product),
            &source.dqf_values,
            &shape,
        )?;
    }
    let (payload, arrays) = builder.finish();

    let meta = PackMeta {
        schema: CWP_SCHEMA.to_string(),
        status: "READY".to_string(),
        satellite: scene.satellite.as_str().to_string(),
        sector: sector_token_from_scene(&scene.sector),
        scan_start: iso8601(scene.start_time_utc),
        scan_end: iso8601(scene.end_time_utc),
        sources: sources.iter().map(|source| source_entry(source)).collect(),
        window,
        projection: ProjectionEntry {
            perspective_point_height_m: scene.projection.perspective_point_height_m,
            semi_major_axis_m: scene.projection.semi_major_axis_m,
            semi_minor_axis_m: scene.projection.semi_minor_axis_m,
            longitude_of_projection_origin_deg: scene
                .projection
                .longitude_of_projection_origin_deg,
            sweep_angle_axis: scene.projection.sweep_angle_axis.as_str().to_string(),
        },
        nx,
        ny,
        x_scan_rad: scene.fixed_grid.x_scan_rad.clone(),
        y_scan_rad: scene.fixed_grid.y_scan_rad.clone(),
        planes,
        plane_order,
        arrays,
        payload_bytes: payload.len(),
        content_sha256: hex_sha256(&payload),
        cwp_counts: cwp_row(&counts),
        coefficients: CoefficientTable {
            formula: CWP_FORMULA.to_string(),
            liquid_density_g_cm3: rw_sat::cwp::LIQUID_WATER_DENSITY_G_CM3,
            ice_density_g_cm3: rw_sat::cwp::BULK_ICE_DENSITY_G_CM3,
            ice_coefficient_provisional: true,
            mixed_phase_takes_ice_branch_provisional: true,
            clear_sky_emits_zero: true,
        },
    };
    let pack_bytes = write_pack(out, &meta, &payload)?;

    #[derive(Serialize)]
    struct PackRecord<'a> {
        path: String,
        schema: &'a str,
        bytes: usize,
        payload_bytes: usize,
        content_sha256: &'a str,
    }

    #[derive(Serialize)]
    struct ScanIdentity<'a> {
        satellite: &'a str,
        sector: &'a str,
        scan_start: &'a str,
        scan_end: &'a str,
        granules: Vec<String>,
    }

    #[derive(Serialize)]
    struct BuildRecord<'a> {
        schema: &'static str,
        status: &'static str,
        pack: PackRecord<'a>,
        scan: ScanIdentity<'a>,
        nx: usize,
        ny: usize,
        window: Option<[usize; 4]>,
        planes: &'a [String],
        sources: &'a [SourceEntry],
        cwp_counts: &'a CwpRow,
        coefficients: &'a CoefficientTable,
    }

    let record = BuildRecord {
        schema: BUILD_SCHEMA,
        status: "READY",
        pack: PackRecord {
            path: out.to_string_lossy().to_string(),
            schema: &meta.schema,
            bytes: pack_bytes,
            payload_bytes: meta.payload_bytes,
            content_sha256: &meta.content_sha256,
        },
        scan: ScanIdentity {
            satellite: &meta.satellite,
            sector: &meta.sector,
            scan_start: &meta.scan_start,
            scan_end: &meta.scan_end,
            granules: sources
                .iter()
                .map(|source| source.path.to_string_lossy().to_string())
                .collect(),
        },
        nx: meta.nx,
        ny: meta.ny,
        window: meta.window,
        planes: &meta.plane_order,
        sources: &meta.sources,
        cwp_counts: &meta.cwp_counts,
        coefficients: &meta.coefficients,
    };
    Ok(format!("{}\n", serde_json::to_string_pretty(&record)?))
}

// ---------------------------------------------------------------------------
// cloud-top: the 10 km sibling
// ---------------------------------------------------------------------------

/// The CWP pack a cloud-top pack is being paired with: read, re-proved,
/// and refused on any scan-identity difference, so `--pairs-with` records
/// a pairing rather than asserting one.
fn read_sibling(
    path: &Path,
    satellite: &str,
    sector: &str,
    scan_start: &str,
) -> Result<SiblingEntry, Box<dyn Error>> {
    let bytes = std::fs::read(path).map_err(|err| {
        boxed_error(format!(
            "cannot read the --pairs-with pack {}: {err}",
            path.display()
        ))
    })?;
    let (meta, _) = decode_pack(&bytes)?;
    for (field, mine, theirs) in [
        ("satellite", satellite, meta.satellite.as_str()),
        ("sector", sector, meta.sector.as_str()),
        ("scan start", scan_start, meta.scan_start.as_str()),
    ] {
        if mine != theirs {
            return Err(boxed_error(format!(
                "--pairs-with {} is a different scan: its {field} is {theirs}, these granules' \
                 {field} is {mine}. The two packs of one scan pair on \
                 (satellite, sector, scan_start); pairing across scans would place cloud tops \
                 on the wrong cloud",
                path.display()
            )));
        }
    }
    Ok(SiblingEntry {
        schema: meta.schema.clone(),
        filename: path
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_else(|| path.to_string_lossy().to_string()),
        content_sha256: meta.content_sha256.clone(),
        nx: meta.nx,
        ny: meta.ny,
        window: meta.window,
    })
}

fn cmd_cloud_top(options: &Options) -> Result<String, Box<dyn Error>> {
    if options.acha.is_none() && options.ctp.is_none() {
        return Err(boxed_error(
            "cloud-top needs at least one of --acha (cloud-top height) or --ctp (cloud-top \
             pressure); an empty vertical-placement pack is not a product",
        ));
    }
    for (flag, present) in [
        ("--cod", options.cod.is_some()),
        ("--cps", options.cps.is_some()),
        ("--actp", options.actp.is_some()),
    ] {
        if present {
            return Err(boxed_error(format!(
                "cloud-top does not take {flag}: the CWP trio is on a different fixed grid and \
                 belongs in the `rw_goes cwp` pack. Two packs per scan, paired by \
                 (satellite, sector, scan_start)"
            )));
        }
    }
    let out = options
        .out
        .as_deref()
        .ok_or_else(|| boxed_error("--out is required (the pack destination)"))?;
    if out.is_dir() {
        return Err(boxed_error(format!(
            "--out {} is a directory; give the pack file path",
            out.display()
        )));
    }
    let window = options.window;

    let acha = match options.acha.as_deref() {
        Some(path) => Some(decode_source(path, CloudProduct::CloudTopHeight, window)?),
        None => None,
    };
    let ctp = match options.ctp.as_deref() {
        Some(path) => Some(decode_source(path, CloudProduct::CloudTopPressure, window)?),
        None => None,
    };

    let mut sources: Vec<&DecodedSource> = Vec::new();
    sources.extend(acha.as_ref());
    sources.extend(ctp.as_ref());
    // ACHA and CTP share the 10 km grid, but that is measured here rather
    // than assumed: same refusal, no hint, because there is no third pack
    // to send the caller to.
    assert_one_scan(&sources, "")?;

    let reference = sources[0];
    let scene = reference.scene();
    let nx = scene.fixed_grid.nx;
    let ny = scene.fixed_grid.ny;
    let shape = vec![ny, nx];
    let satellite = scene.satellite.as_str().to_string();
    let sector = sector_token_from_scene(&scene.sector);
    let scan_start = iso8601(scene.start_time_utc);

    let sibling = match options.pairs_with.as_deref() {
        Some(path) => Some(read_sibling(path, &satellite, &sector, &scan_start)?),
        None => None,
    };

    let (lat, lon) = scene.lat_lon_mesh();

    let mut builder = PayloadBuilder::new();
    let mut planes: BTreeMap<String, String> = BTreeMap::new();
    let mut plane_order: Vec<String> = Vec::new();
    if let Some(acha) = &acha {
        push_plane(
            &mut builder,
            &mut planes,
            &mut plane_order,
            "cloud_top_height_m",
            acha.values(),
            &shape,
        )?;
    }
    if let Some(ctp) = &ctp {
        push_plane(
            &mut builder,
            &mut planes,
            &mut plane_order,
            "cloud_top_pressure_hpa",
            ctp.values(),
            &shape,
        )?;
    }
    push_plane(
        &mut builder,
        &mut planes,
        &mut plane_order,
        "lat",
        &lat,
        &shape,
    )?;
    push_plane(
        &mut builder,
        &mut planes,
        &mut plane_order,
        "lon",
        &lon,
        &shape,
    )?;
    for source in &sources {
        push_plane(
            &mut builder,
            &mut planes,
            &mut plane_order,
            &dqf_plane_name(source.product),
            &source.dqf_values,
            &shape,
        )?;
    }
    let (payload, arrays) = builder.finish();

    let meta = CloudTopMeta {
        schema: CLOUDTOP_SCHEMA.to_string(),
        status: "READY".to_string(),
        satellite,
        sector,
        scan_start,
        scan_end: iso8601(scene.end_time_utc),
        sources: sources.iter().map(|source| source_entry(source)).collect(),
        window,
        projection: ProjectionEntry {
            perspective_point_height_m: scene.projection.perspective_point_height_m,
            semi_major_axis_m: scene.projection.semi_major_axis_m,
            semi_minor_axis_m: scene.projection.semi_minor_axis_m,
            longitude_of_projection_origin_deg: scene
                .projection
                .longitude_of_projection_origin_deg,
            sweep_angle_axis: scene.projection.sweep_angle_axis.as_str().to_string(),
        },
        nx,
        ny,
        x_scan_rad: scene.fixed_grid.x_scan_rad.clone(),
        y_scan_rad: scene.fixed_grid.y_scan_rad.clone(),
        planes,
        plane_order,
        arrays,
        payload_bytes: payload.len(),
        content_sha256: hex_sha256(&payload),
        pairs_with_schema: pairs_with_schema(),
        regrid: NO_REGRID.to_string(),
        sibling,
    };
    let pack_bytes = write_cloudtop_pack(out, &meta, &payload)?;

    #[derive(Serialize)]
    struct PackRecord<'a> {
        path: String,
        schema: &'a str,
        bytes: usize,
        payload_bytes: usize,
        content_sha256: &'a str,
    }

    #[derive(Serialize)]
    struct ScanIdentity<'a> {
        satellite: &'a str,
        sector: &'a str,
        scan_start: &'a str,
        scan_end: &'a str,
        granules: Vec<String>,
    }

    #[derive(Serialize)]
    struct CloudTopBuildRecord<'a> {
        schema: &'static str,
        status: &'static str,
        pack: PackRecord<'a>,
        scan: ScanIdentity<'a>,
        nx: usize,
        ny: usize,
        window: Option<[usize; 4]>,
        planes: &'a [String],
        sources: &'a [SourceEntry],
        pairs_with_schema: &'a str,
        regrid: &'a str,
        sibling: &'a Option<SiblingEntry>,
    }

    let record = CloudTopBuildRecord {
        schema: CLOUDTOP_BUILD_SCHEMA,
        status: "READY",
        pack: PackRecord {
            path: out.to_string_lossy().to_string(),
            schema: &meta.schema,
            bytes: pack_bytes,
            payload_bytes: meta.payload_bytes,
            content_sha256: &meta.content_sha256,
        },
        scan: ScanIdentity {
            satellite: &meta.satellite,
            sector: &meta.sector,
            scan_start: &meta.scan_start,
            scan_end: &meta.scan_end,
            granules: sources
                .iter()
                .map(|source| source.path.to_string_lossy().to_string())
                .collect(),
        },
        nx: meta.nx,
        ny: meta.ny,
        window: meta.window,
        planes: &meta.plane_order,
        sources: &meta.sources,
        pairs_with_schema: &meta.pairs_with_schema,
        regrid: &meta.regrid,
        sibling: &meta.sibling,
    };
    Ok(format!("{}\n", serde_json::to_string_pretty(&record)?))
}

// ---------------------------------------------------------------------------
// verify
// ---------------------------------------------------------------------------

/// Prove every declared plane names an array that lies inside the payload
/// it indexes, and that the grid the metadata states is the grid the
/// arrays and the scan-angle axes hold.  Shared by both pack families:
/// the container is one container.
fn check_grid_and_planes(
    nx: usize,
    ny: usize,
    x_scan_rad: usize,
    y_scan_rad: usize,
    plane_order: &[String],
    planes: &BTreeMap<String, String>,
    arrays: &BTreeMap<String, ArrayEntry>,
) -> Result<(), Box<dyn Error>> {
    let expected: usize = nx.saturating_mul(ny);
    for name in plane_order {
        let key = planes.get(name).ok_or_else(|| {
            boxed_error(format!(
                "pack declares plane {name:?} in plane_order but no array key for it"
            ))
        })?;
        let entry: &ArrayEntry = arrays.get(key).ok_or_else(|| {
            boxed_error(format!(
                "pack plane {name:?} names array {key:?}, which the pack does not hold"
            ))
        })?;
        let elements: usize = entry.shape.iter().product();
        if elements != expected {
            return Err(boxed_error(format!(
                "pack plane {name:?} holds {elements} values but the declared grid is \
                 {nx}x{ny} ({expected})"
            )));
        }
    }
    if x_scan_rad != nx || y_scan_rad != ny {
        return Err(boxed_error(format!(
            "pack states a {nx}x{ny} grid but carries {x_scan_rad}/{y_scan_rad} scan-angle \
             values"
        )));
    }
    Ok(())
}

/// Every source must name a DQF plane the pack actually holds.  A source
/// row promising per-pixel flags that are not there is worse than a pack
/// without them: a consumer would build an error model on a plane it
/// never read.
fn check_dqf_planes(
    sources: &[SourceEntry],
    planes: &BTreeMap<String, String>,
    require: bool,
) -> Result<(), Box<dyn Error>> {
    for source in sources {
        if source.dqf_plane.is_empty() {
            if !require {
                // A v1 pack never carried one.  That is the version's
                // known limit, not a corrupt pack, and it is reported in
                // the receipt rather than raised as a fault.
                continue;
            }
            return Err(boxed_error(format!(
                "pack source {} names no DQF plane; this schema requires per-pixel flags for \
                 every source",
                source.product
            )));
        }
        if !planes.contains_key(&source.dqf_plane) {
            return Err(boxed_error(format!(
                "pack source {} names DQF plane {:?}, which the pack does not hold (it has: \
                 {})",
                source.product,
                source.dqf_plane,
                planes.keys().cloned().collect::<Vec<_>>().join(", ")
            )));
        }
    }
    Ok(())
}

fn cmd_verify(options: &Options) -> Result<String, Box<dyn Error>> {
    let path = options
        .pack
        .as_deref()
        .or(options.out.as_deref())
        .ok_or_else(|| boxed_error("verify needs the pack path in --pack (or --out)"))?;
    let bytes = std::fs::read(path)
        .map_err(|err| boxed_error(format!("cannot read {}: {err}", path.display())))?;
    // Which family this is comes from the pack's own schema, never from
    // its file name.
    let declared = pack_schema(&bytes)?;
    // Both families read every version they have ever written, not just
    // the one they write now: a receipt naming a v1 pack's digest is only
    // worth something while that pack can still be re-verified.
    if CWP_READABLE_SCHEMAS.contains(&declared.as_str()) {
        return verify_cwp_pack(path, &bytes, &declared);
    }
    if CLOUDTOP_READABLE_SCHEMAS.contains(&declared.as_str()) {
        return verify_cloudtop_pack(path, &bytes, &declared);
    }
    Err(boxed_error(format!(
        "{} declares schema {declared:?}; this build reads {} and {}",
        path.display(),
        CWP_READABLE_SCHEMAS.join(", "),
        CLOUDTOP_READABLE_SCHEMAS.join(", ")
    )))
}

/// Whether a declared schema is one that promises a per-pixel DQF plane
/// for every source.  v1 packs do not, and are not faulted for it.
fn carries_per_pixel_dqf(declared: &str) -> bool {
    declared == CWP_SCHEMA || declared == CLOUDTOP_SCHEMA
}

fn verify_cwp_pack(path: &Path, bytes: &[u8], declared: &str) -> Result<String, Box<dyn Error>> {
    let (meta, payload) = decode_pack(bytes)?;
    check_grid_and_planes(
        meta.nx,
        meta.ny,
        meta.x_scan_rad.len(),
        meta.y_scan_rad.len(),
        &meta.plane_order,
        &meta.planes,
        &meta.arrays,
    )?;
    let per_pixel_dqf = carries_per_pixel_dqf(declared);
    check_dqf_planes(&meta.sources, &meta.planes, per_pixel_dqf)?;

    #[derive(Serialize)]
    struct VerifyRecord<'a> {
        schema: &'static str,
        status: &'static str,
        path: String,
        pack_schema: &'a str,
        pack_status: &'a str,
        bytes: usize,
        payload_bytes: usize,
        content_sha256: &'a str,
        /// Whether this pack's schema carries a per-pixel DQF plane per
        /// source.  False for v1, which is a fact about the version and
        /// not a fault in the pack.
        per_pixel_dqf: bool,
        satellite: &'a str,
        sector: &'a str,
        scan_start: &'a str,
        scan_end: &'a str,
        window: Option<[usize; 4]>,
        nx: usize,
        ny: usize,
        planes: &'a [String],
        arrays: usize,
        sources: &'a [SourceEntry],
        cwp_counts: &'a CwpRow,
        coefficients: &'a CoefficientTable,
    }

    let record = VerifyRecord {
        schema: VERIFY_SCHEMA,
        status: "PASS",
        path: path.to_string_lossy().to_string(),
        pack_schema: &meta.schema,
        pack_status: &meta.status,
        bytes: bytes.len(),
        payload_bytes: payload.len(),
        content_sha256: &meta.content_sha256,
        per_pixel_dqf,
        satellite: &meta.satellite,
        sector: &meta.sector,
        scan_start: &meta.scan_start,
        scan_end: &meta.scan_end,
        window: meta.window,
        nx: meta.nx,
        ny: meta.ny,
        planes: &meta.plane_order,
        arrays: meta.arrays.len(),
        sources: &meta.sources,
        cwp_counts: &meta.cwp_counts,
        coefficients: &meta.coefficients,
    };
    Ok(format!("{}\n", serde_json::to_string_pretty(&record)?))
}

fn verify_cloudtop_pack(path: &Path, bytes: &[u8], declared: &str) -> Result<String, Box<dyn Error>> {
    let (meta, payload) = decode_cloudtop_pack(bytes)?;
    check_grid_and_planes(
        meta.nx,
        meta.ny,
        meta.x_scan_rad.len(),
        meta.y_scan_rad.len(),
        &meta.plane_order,
        &meta.planes,
        &meta.arrays,
    )?;
    let per_pixel_dqf = carries_per_pixel_dqf(declared);
    check_dqf_planes(&meta.sources, &meta.planes, per_pixel_dqf)?;

    #[derive(Serialize)]
    struct VerifyRecord<'a> {
        schema: &'static str,
        status: &'static str,
        path: String,
        pack_schema: &'a str,
        pack_status: &'a str,
        bytes: usize,
        payload_bytes: usize,
        content_sha256: &'a str,
        /// Whether this pack's schema carries a per-pixel DQF plane per
        /// source.  False for v1, which is a fact about the version and
        /// not a fault in the pack.
        per_pixel_dqf: bool,
        satellite: &'a str,
        sector: &'a str,
        scan_start: &'a str,
        scan_end: &'a str,
        window: Option<[usize; 4]>,
        nx: usize,
        ny: usize,
        planes: &'a [String],
        arrays: usize,
        sources: &'a [SourceEntry],
        pairs_with_schema: &'a str,
        regrid: &'a str,
        sibling: &'a Option<SiblingEntry>,
    }

    let record = VerifyRecord {
        schema: CLOUDTOP_VERIFY_SCHEMA,
        status: "PASS",
        path: path.to_string_lossy().to_string(),
        pack_schema: &meta.schema,
        pack_status: &meta.status,
        bytes: bytes.len(),
        payload_bytes: payload.len(),
        content_sha256: &meta.content_sha256,
        per_pixel_dqf,
        satellite: &meta.satellite,
        sector: &meta.sector,
        scan_start: &meta.scan_start,
        scan_end: &meta.scan_end,
        window: meta.window,
        nx: meta.nx,
        ny: meta.ny,
        planes: &meta.plane_order,
        arrays: meta.arrays.len(),
        sources: &meta.sources,
        pairs_with_schema: &meta.pairs_with_schema,
        regrid: &meta.regrid,
        sibling: &meta.sibling,
    };
    Ok(format!("{}\n", serde_json::to_string_pretty(&record)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|value| value.to_string()).collect()
    }

    fn flags(list: &[&str]) -> Options {
        Options::parse(&args(list)).expect("parses")
    }

    #[test]
    fn help_and_version_are_stable_surfaces() {
        assert!(run(&[]).unwrap().contains("usage: rw_goes"));
        let help = run(&["--help".to_string()]).unwrap();
        for subcommand in ["list", "fetch", "cwp", "cloud-top", "verify"] {
            assert!(help.contains(subcommand), "usage must document {subcommand}");
        }
        for flag in [
            "--satellite",
            "--sector",
            "--products",
            "--start",
            "--end",
            "--bucket",
            "--mode",
            "--cache",
            "--no-cache",
            "--out",
            "--limit-scans",
            "--complete-only",
            "--cod",
            "--cps",
            "--actp",
            "--acha",
            "--ctp",
            "--window",
            "--pack",
            "--pairs-with",
        ] {
            assert!(help.contains(flag), "usage must document {flag}");
        }
        assert!(
            run(&["--version".to_string()])
                .unwrap()
                .starts_with("rw_goes ")
        );
        assert_eq!(run(&["-h".to_string()]).unwrap(), USAGE);
        assert_eq!(run(&["--abi".to_string()]).unwrap(), format!("{ABI_MARKER}\n"));
        assert!(USAGE.contains("--abi"), "usage must document --abi");
    }

    #[test]
    fn abi_marker_names_every_contract_it_pins() {
        // The marker is a literal (no const-format crate in the offline
        // vendor closure), so this is what keeps it honest: bump a schema
        // without touching the marker and this test is the refusal.
        for needle in [FETCH_SCHEMA, pack::CWP_SCHEMA, cloudtop::CLOUDTOP_SCHEMA] {
            assert!(
                ABI_MARKER.contains(needle),
                "--abi does not pin {needle}, so a wrapper written against \
                 it could not tell a drifted binary from a current one"
            );
        }
        assert_eq!(ABI_MARKER.split('\t').count(), 3);
    }

    #[test]
    fn the_help_states_the_gate_that_was_actually_applied() {
        // How a pixel was gated is a value in the pack; --help is where the
        // operator reads what the default is before they get a pack.
        for needle in ["enumerated", "bitfield", "88", "PROVISIONAL", "DQF == 0"] {
            assert!(USAGE.contains(needle), "--help does not mention {needle:?}");
        }
    }

    #[test]
    fn the_help_states_why_there_are_two_packs_per_scan_with_both_grids() {
        // A reader who finds two packs for one scan must be able to read
        // WHY here, with the measured shapes, not infer it from a refusal.
        for needle in [
            "two packs per scan, deliberately",
            "2500 x 1500",
            "500 x 300",
            "REFUSES",
            "pairing key",
            "gpuwm-obs.goes-cloudtop.v1",
        ] {
            assert!(USAGE.contains(needle), "--help does not mention {needle:?}");
        }
    }

    #[test]
    fn every_source_names_a_dqf_plane_and_a_pack_that_lacks_it_is_refused() {
        let mut planes = BTreeMap::new();
        planes.insert("cod_dqf".to_string(), "a00006".to_string());
        let row = |product: CloudProduct, plane: &str| {
            let (dqf_rule, condemn_mask) = dqf_rule_row(product);
            SourceEntry {
                product: product.family().to_string(),
                filename: "x.nc".to_string(),
                bytes: 1,
                sha256: String::new(),
                dqf_rule,
                condemn_mask,
                dqf: DqfRow::default(),
                dqf_plane: plane.to_string(),
            }
        };

        // The name every source's plane is derived from, per product.
        assert_eq!(dqf_plane_name(CloudProduct::OpticalDepth), "cod_dqf");
        assert_eq!(dqf_plane_name(CloudProduct::ParticleSize), "cps_dqf");
        assert_eq!(dqf_plane_name(CloudProduct::CloudTopPhase), "actp_dqf");
        assert_eq!(dqf_plane_name(CloudProduct::CloudTopHeight), "acha_dqf");
        assert_eq!(dqf_plane_name(CloudProduct::CloudTopPressure), "ctp_dqf");

        let good = vec![row(CloudProduct::OpticalDepth, "cod_dqf")];
        assert!(check_dqf_planes(&good, &planes, true).is_ok());

        // A row promising a plane the pack does not hold: refused, with
        // what the pack does hold named.
        let missing = vec![row(CloudProduct::ParticleSize, "cps_dqf")];
        let err = check_dqf_planes(&missing, &planes, true).unwrap_err().to_string();
        assert!(err.contains("cps_dqf"), "{err}");
        assert!(err.contains("cod_dqf"), "{err}");

        // A row naming no plane at all is refused by this schema.
        let empty = vec![row(CloudProduct::OpticalDepth, "")];
        let err = check_dqf_planes(&empty, &planes, true).unwrap_err().to_string();
        assert!(err.contains("names no DQF plane"), "{err}");
    }

    #[test]
    fn f32_carries_every_dqf_bit_pattern_the_dcomp_word_can_hold() {
        // The plane is `<f4` and the flag word is u16, so the claim that
        // it is lossless has to hold for EVERY value, not the ones that
        // happen to appear in one granule -- the thin (256) and thick
        // (512) bits the operator inflates on among them.
        for raw in 0u16..=u16::MAX {
            let round_tripped = f32::from_le_bytes((raw as f32).to_le_bytes()) as u16;
            assert_eq!(round_tripped, raw, "f32 lost DQF value {raw}");
        }
        // And the bits the condemn mask deliberately leaves alone survive
        // the gate, which is the whole reason the plane is recoverable.
        let rule = DqfRule::Bitfield {
            condemn: rw_sat::cloud::DCOMP_CONDEMN_DEFAULT,
        };
        for thin_or_thick in [256.0f32, 512.0, 256.0 + 512.0, 2.0 + 256.0] {
            assert!(
                rule.is_good(thin_or_thick),
                "{thin_or_thick} must survive the gate so its bits reach the plane"
            );
        }
    }

    #[test]
    fn the_help_states_what_the_counts_could_not_answer() {
        for needle in ["cod_dqf", "actp_dqf", "dqf_plane", "256", "512", "UNGATED"] {
            assert!(USAGE.contains(needle), "--help does not mention {needle:?}");
        }
        // The bump is stated where a reader of the format will find it.
        assert!(CWP_SCHEMA.ends_with(".v2"), "{CWP_SCHEMA}");
        assert!(CLOUDTOP_SCHEMA.ends_with(".v2"), "{CLOUDTOP_SCHEMA}");
    }

    #[test]
    fn the_help_says_to_index_the_dqf_plane_over_the_whole_pack() {
        // The first version of this note said NaN pixels were "gated out
        // of cwp anyway", which is true and useless: the inflation factor
        // is needed per pixel BEFORE superobbing picks cells, so indexing
        // over the surviving-CWP mask would read NaN as a bitfield for
        // 47,162 pixels on the measured scan.
        for needle in [
            "WHOLE pack",
            "NEVER over the surviving-CWP mask",
            "BEFORE superobbing",
            "47,162",
            "np.isfinite",
            "does not raise",
        ] {
            assert!(USAGE.contains(needle), "--help does not mention {needle:?}");
        }
    }

    #[test]
    fn the_help_answers_how_a_windowed_pack_pairs() {
        for needle in [
            "pairing a WINDOWED pack",
            "ORDER MATTERS",
            "--pairs-with nest.goespack",
            "102.4 ten-km",
            "full-sector on purpose",
            "docs/obs-goes-cwp-bridge-design.md",
        ] {
            assert!(USAGE.contains(needle), "--help does not mention {needle:?}");
        }
        // The steps must be stated in the order they have to be run: the
        // sibling block can only pin a pack that already exists.
        let cwp_step = USAGE.find("1.  rw_goes cwp").expect("step 1 stated");
        let ctop_step = USAGE.find("2.  rw_goes cloud-top").expect("step 2 stated");
        assert!(cwp_step < ctop_step, "the windowed pack is built first");
    }

    #[test]
    fn v1_packs_stay_readable_and_report_the_plane_they_never_had() {
        assert!(carries_per_pixel_dqf(CWP_SCHEMA));
        assert!(carries_per_pixel_dqf(CLOUDTOP_SCHEMA));
        assert!(!carries_per_pixel_dqf(pack::CWP_SCHEMA_V1));
        assert!(!carries_per_pixel_dqf(cloudtop::CLOUDTOP_SCHEMA_V1));

        // A v1 source row names no plane. Required (v2) that is a fault;
        // not required (v1) it is the version's known limit and passes.
        let planes = BTreeMap::new();
        let v1_row = vec![SourceEntry {
            product: "COD".to_string(),
            filename: "x.nc".to_string(),
            bytes: 1,
            sha256: String::new(),
            dqf_rule: "bitfield".to_string(),
            condemn_mask: Some(88),
            dqf: DqfRow::default(),
            dqf_plane: String::new(),
        }];
        assert!(check_dqf_planes(&v1_row, &planes, false).is_ok());
        assert!(check_dqf_planes(&v1_row, &planes, true).is_err());
    }

    #[test]
    fn verify_reads_a_real_v1_pack_and_says_it_has_no_per_pixel_dqf() {
        use pack::{CWP_SCHEMA_V1, encode_pack};
        let dir = std::env::temp_dir().join(format!("rw-goes-v1-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let mut builder = PayloadBuilder::new();
        let key = builder.push_f32(&[1.0, 2.0], vec![1, 2]);
        let (payload, arrays) = builder.finish();
        let mut planes = BTreeMap::new();
        planes.insert("cwp".to_string(), key);
        let meta = PackMeta {
            schema: CWP_SCHEMA_V1.to_string(),
            status: "READY".to_string(),
            satellite: "G19".to_string(),
            sector: "C".to_string(),
            scan_start: "2026-08-04T18:01:17Z".to_string(),
            scan_end: "2026-08-04T18:03:54.500Z".to_string(),
            sources: vec![SourceEntry {
                product: "COD".to_string(),
                filename: "OR_ABI-L2-CODC-M6_G19_s2026216.nc".to_string(),
                bytes: 3666798,
                sha256: "0".repeat(64),
                dqf_rule: "bitfield".to_string(),
                condemn_mask: Some(88),
                dqf: DqfRow::default(),
                // v1 named no plane; serde(default) is what makes the
                // real historical JSON deserialize here.
                dqf_plane: String::new(),
            }],
            window: None,
            projection: ProjectionEntry {
                perspective_point_height_m: 35786023.0,
                semi_major_axis_m: 6378137.0,
                semi_minor_axis_m: 6356752.31414,
                longitude_of_projection_origin_deg: -75.0,
                sweep_angle_axis: "x".to_string(),
            },
            nx: 2,
            ny: 1,
            x_scan_rad: vec![0.0, 1.0e-4],
            y_scan_rad: vec![0.0],
            planes,
            plane_order: vec!["cwp".to_string()],
            arrays,
            payload_bytes: payload.len(),
            content_sha256: hex_sha256(&payload),
            cwp_counts: CwpRow::default(),
            coefficients: CoefficientTable {
                formula: CWP_FORMULA.to_string(),
                liquid_density_g_cm3: 1.0,
                ice_density_g_cm3: 0.917,
                ice_coefficient_provisional: true,
                mixed_phase_takes_ice_branch_provisional: true,
                clear_sky_emits_zero: true,
            },
        };
        let path = dir.join("historical.goespack");
        std::fs::write(&path, encode_pack(&meta, &payload).unwrap()).unwrap();

        let json = cmd_verify(&Options {
            pack: Some(path),
            ..Options::default()
        })
        .unwrap();
        assert!(json.contains("\"status\": \"PASS\""), "{json}");
        assert!(json.contains(CWP_SCHEMA_V1), "{json}");
        assert!(json.contains("\"per_pixel_dqf\": false"), "{json}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_grid_refusal_points_at_the_sibling_pack_and_never_at_a_resample() {
        assert!(CLOUD_TOP_HINT.contains("rw_goes cloud-top"));
        assert!(CLOUD_TOP_HINT.contains("own pack"));
        for forbidden in ["resample", "regrid", "interpolat"] {
            assert!(
                !CLOUD_TOP_HINT.contains(forbidden),
                "the hint must never offer to {forbidden}"
            );
        }
        // The pack itself states the resampling it did not do.
        assert!(NO_REGRID.starts_with("none:"), "{NO_REGRID}");
        assert_eq!(pairs_with_schema(), CWP_SCHEMA);
        // And it names no schema version of its own: a literal version in
        // that sentence went stale the moment CWP bumped to v2 and
        // shipped a wrong name in real metadata. One place for the string.
        assert!(
            !NO_REGRID.contains(".v1") && !NO_REGRID.contains(".v2"),
            "the regrid note must not spell a version: {NO_REGRID}"
        );
    }

    #[test]
    fn unknown_subcommands_and_options_fail_closed() {
        assert!(run(&["superob".to_string()]).is_err());
        assert!(Options::parse(&args(&["--nope"])).is_err());
        assert!(Options::parse(&args(&["--satellite"])).is_err());
        assert!(Options::parse(&args(&["--cod"])).is_err());
        assert!(Options::parse(&args(&["--window"])).is_err());
        // A missing value must not swallow the next flag silently: it is
        // taken as the value, which is wrong but visible, so the flags that
        // parse a number must reject it.
        assert!(Options::parse(&args(&["--limit-scans", "--out"])).is_err());
        assert!(Options::parse(&args(&["--mode", "six"])).is_err());
        assert!(Options::parse(&args(&["--limit-scans", "0"])).is_err());
    }

    #[test]
    fn window_parses_four_counts_and_refuses_anything_else() {
        assert_eq!(parse_window("100,256,200,128").unwrap(), [100, 256, 200, 128]);
        assert_eq!(parse_window(" 0, 4 ,0, 4 ").unwrap(), [0, 4, 0, 4]);
        for bad in [
            "100,256,200",
            "100,256,200,128,7",
            "100,256,200,x",
            "-1,256,200,128",
            "100,0,200,128",
            "100,256,200,0",
            "",
        ] {
            assert!(parse_window(bad).is_err(), "--window {bad:?} must fail closed");
        }
        assert_eq!(flags(&["--window", "1,2,3,4"]).window, Some([1, 2, 3, 4]));
        assert!(Options::parse(&args(&["--window", "1,2,3"])).is_err());
    }

    #[test]
    fn the_acquisition_options_are_required_rather_than_guessed() {
        let bare = Options::default();
        assert!(bare.satellite().is_err());
        assert!(bare.sector().is_err());
        assert!(bare.products().is_err());
        assert!(bare.window().is_err());
        assert!(bare.bucket().is_err());
        assert_eq!(bare.mode(), DEFAULT_MODE);
    }

    #[test]
    fn products_keep_the_requested_order_and_refuse_junk() {
        let options = flags(&["--products", "COD, cps ,cloud-top-phase,COD"]);
        assert_eq!(
            options.products().unwrap(),
            vec![
                CloudProduct::OpticalDepth,
                CloudProduct::ParticleSize,
                CloudProduct::CloudTopPhase,
            ],
            "duplicates collapse, order is the caller's"
        );
        assert!(flags(&["--products", "COD,ORANGES"]).products().is_err());
        assert!(flags(&["--products", " , "]).products().is_err());
    }

    #[test]
    fn the_bucket_follows_the_satellite_unless_it_is_named() {
        assert_eq!(
            flags(&["--satellite", "G19"]).bucket().unwrap(),
            "noaa-goes19"
        );
        assert_eq!(
            flags(&["--satellite", "goes-16"]).bucket().unwrap(),
            "noaa-goes16"
        );
        assert!(flags(&["--satellite", "himawari"]).bucket().is_err());
        assert_eq!(
            flags(&["--satellite", "himawari", "--bucket", "some-mirror"])
                .bucket()
                .unwrap(),
            "some-mirror"
        );
    }

    #[test]
    fn sector_tokens_parse_and_unknown_ones_are_refused() {
        assert_eq!(flags(&["--sector", "C"]).sector().unwrap(), Sector::Conus);
        assert_eq!(
            flags(&["--sector", "full-disk"]).sector().unwrap(),
            Sector::FullDisk
        );
        assert_eq!(flags(&["--sector", "m2"]).sector().unwrap(), Sector::Meso2);
        assert!(flags(&["--sector", "quadrant"]).sector().is_err());
        assert_eq!(sector_token_from_sector(Sector::Conus), "C");
        assert_eq!(sector_token_from_sector(Sector::Meso1), "M1");
    }

    #[test]
    fn a_backwards_window_is_refused_with_both_ends_in_the_message() {
        let options = flags(&[
            "--start",
            "2026-08-04T18:30:00Z",
            "--end",
            "2026-08-04T18:00:00Z",
        ]);
        let err = options.window().unwrap_err().to_string();
        assert!(err.contains("precedes"), "{err}");
        assert!(err.contains("2026-08-04T18:00:00Z"), "{err}");

        let good = flags(&[
            "--start",
            "20260804T180000",
            "--end",
            "2026-08-04T18:30:00Z",
        ]);
        let (start, end) = good.window().unwrap();
        assert_eq!(iso8601(start), "2026-08-04T18:00:00Z");
        assert_eq!(iso8601(end), "2026-08-04T18:30:00Z");
        assert!(flags(&["--start", "yesterday", "--end", "now"]).window().is_err());
    }

    #[test]
    fn scan_identity_keeps_the_tenth_of_a_second_the_filename_states() {
        // s20262161801170 is 18:01:17.0; a scan start that rounded would
        // merge two scans that the bucket keeps apart.
        let parsed = parse_goes_abi_filename(
            "OR_ABI-L2-CODC-M6_G19_s20262161801170_e20262161803543_c20262161805066.nc",
        )
        .unwrap();
        assert_eq!(iso8601(parsed.start_time_utc), "2026-08-04T18:01:17Z");
        let tenth = parse_time("2026-08-04T18:01:17.100Z").unwrap();
        assert_eq!(iso8601(tenth), "2026-08-04T18:01:17.100Z");
    }

    #[test]
    fn the_hour_walk_covers_both_ends_of_the_window() {
        let start = parse_time("2026-08-04T18:41:00Z").unwrap();
        let end = parse_time("2026-08-04T20:03:00Z").unwrap();
        let hours = window_hours(start, end);
        assert_eq!(hours.len(), 3);
        assert_eq!(iso8601(hours[0]), "2026-08-04T18:00:00Z");
        assert_eq!(iso8601(hours[2]), "2026-08-04T20:00:00Z");
        let single = window_hours(start, start);
        assert_eq!(single.len(), 1);
    }

    #[test]
    fn a_scan_is_complete_only_when_every_requested_product_is_present() {
        let products = vec![
            CloudProduct::OpticalDepth,
            CloudProduct::ParticleSize,
            CloudProduct::CloudTopPhase,
        ];
        let start = parse_time("2026-08-04T18:01:17Z").unwrap();
        let granule = |product: CloudProduct| Granule {
            product,
            object: S3Object {
                key: format!("ABI-L2-{}C/2026/216/18/OR.nc", product.family()),
                size_bytes: 10,
                last_modified: String::new(),
            },
            scan_start: start,
            scan_end: start,
        };
        let mut scan = Scan {
            start,
            slots: vec![
                Some(granule(CloudProduct::OpticalDepth)),
                None,
                Some(granule(CloudProduct::CloudTopPhase)),
            ],
        };
        assert!(!scan.complete());
        assert_eq!(scan.missing(&products), vec!["CPS".to_string()]);
        assert_eq!(scan.granules().count(), 2);
        scan.slots[1] = Some(granule(CloudProduct::ParticleSize));
        assert!(scan.complete());
        assert!(scan.missing(&products).is_empty());
    }

    #[test]
    fn the_pack_records_the_rule_and_the_mask_that_were_really_applied() {
        assert_eq!(
            dqf_rule_row(CloudProduct::OpticalDepth),
            ("bitfield".to_string(), Some(rw_sat::cloud::DCOMP_CONDEMN_DEFAULT))
        );
        assert_eq!(
            dqf_rule_row(CloudProduct::ParticleSize),
            ("bitfield".to_string(), Some(rw_sat::cloud::DCOMP_CONDEMN_DEFAULT))
        );
        assert_eq!(
            dqf_rule_row(CloudProduct::CloudTopPhase),
            ("enumerated".to_string(), None)
        );
        assert_eq!(
            dqf_rule_row(CloudProduct::CloudTopHeight),
            ("enumerated".to_string(), None)
        );
        assert_eq!(rw_sat::cloud::DCOMP_CONDEMN_DEFAULT, 8 | 16 | 64);
    }

    #[test]
    fn cwp_counts_carry_the_derived_finite_total() {
        let counts = CwpCounts {
            clear_zero: 3,
            liquid: 5,
            supercooled: 1,
            mixed: 2,
            ice: 4,
            unknown: 6,
            phase_missing: 7,
            input_missing: 8,
        };
        let row = cwp_row(&counts);
        assert_eq!(row.finite, 15);
        assert_eq!(row.finite, counts.finite());
        assert_eq!(row.phase_missing, 7);
        assert_eq!(row.input_missing, 8);
    }

    #[test]
    fn grid_differences_are_reported_with_the_numbers() {
        let base = AbiFixedGrid {
            nx: 2,
            ny: 1,
            x_scan_rad: vec![0.0, 1.0e-4],
            y_scan_rad: vec![0.0],
        };
        let bigger = AbiFixedGrid {
            nx: 3,
            ny: 1,
            x_scan_rad: vec![0.0, 1.0e-4, 2.0e-4],
            y_scan_rad: vec![0.0],
        };
        assert!(grid_difference(&base, &bigger).contains("2x1 against 3x1"));
        let shifted = AbiFixedGrid {
            x_scan_rad: vec![0.0, 1.000_000_1e-4],
            ..base.clone()
        };
        assert!(grid_difference(&base, &shifted).contains("x_scan_rad[1]"));
        let nudged_y = AbiFixedGrid {
            y_scan_rad: vec![1.0e-9],
            ..base.clone()
        };
        assert!(grid_difference(&base, &nudged_y).contains("y_scan_rad[0]"));
    }

    #[test]
    fn the_scene_sector_token_never_invents_a_sector() {
        assert_eq!(sector_token_from_scene(&AbiSector::Conus), "C");
        assert_eq!(sector_token_from_scene(&AbiSector::FullDisk), "F");
        assert_eq!(sector_token_from_scene(&AbiSector::Mesoscale1), "M1");
        assert_eq!(sector_token_from_scene(&AbiSector::Mesoscale2), "M2");
        assert_eq!(
            sector_token_from_scene(&AbiSector::Unknown("ABI-L2-XX".to_string())),
            "ABI-L2-XX"
        );
    }

    #[test]
    fn cwp_and_verify_require_their_inputs_before_touching_a_file() {
        assert!(cmd_cwp(&Options::default()).is_err());
        assert!(
            cmd_cwp(&Options {
                cod: Some(PathBuf::from("cod.nc")),
                ..Options::default()
            })
            .is_err()
        );
        assert!(
            cmd_cwp(&Options {
                cod: Some(PathBuf::from("cod.nc")),
                cps: Some(PathBuf::from("cps.nc")),
                actp: Some(PathBuf::from("actp.nc")),
                ..Options::default()
            })
            .unwrap_err()
            .to_string()
            .contains("--out"),
            "the destination is required before any granule is read"
        );
        assert!(cmd_verify(&Options::default()).is_err());
        assert!(
            cmd_verify(&Options {
                pack: Some(PathBuf::from("absent.goespack")),
                ..Options::default()
            })
            .is_err()
        );
    }

    #[test]
    fn cloud_top_refuses_an_empty_pack_and_refuses_the_trio_outright() {
        let err = cmd_cloud_top(&Options::default()).unwrap_err().to_string();
        assert!(err.contains("--acha"), "{err}");
        assert!(err.contains("--ctp"), "{err}");

        // The trio belongs in the other pack; taking it here would be the
        // one-pack-by-resampling outcome the format exists to prevent.
        for flag in ["cod", "cps", "actp"] {
            let options = Options {
                acha: Some(PathBuf::from("acha.nc")),
                out: Some(PathBuf::from("out.goespack")),
                cod: (flag == "cod").then(|| PathBuf::from("cod.nc")),
                cps: (flag == "cps").then(|| PathBuf::from("cps.nc")),
                actp: (flag == "actp").then(|| PathBuf::from("actp.nc")),
                ..Options::default()
            };
            let err = cmd_cloud_top(&options).unwrap_err().to_string();
            assert!(err.contains(&format!("--{flag}")), "{err}");
            assert!(err.contains("different fixed grid"), "{err}");
        }

        // And the destination is required before any granule is read.
        let err = cmd_cloud_top(&Options {
            acha: Some(PathBuf::from("acha.nc")),
            ..Options::default()
        })
        .unwrap_err()
        .to_string();
        assert!(err.contains("--out"), "{err}");
    }

    #[test]
    fn a_pairing_across_scans_is_refused_by_the_identity_it_would_have_broken() {
        // Build a real CWP pack in memory, then try to pair a cloud-top
        // pack of a different scan with it.
        use pack::encode_pack;
        let dir = std::env::temp_dir().join(format!("rw-goes-pairs-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sibling.goespack");

        let mut builder = PayloadBuilder::new();
        let key = builder.push_f32(&[1.0, 2.0], vec![1, 2]);
        let (payload, arrays) = builder.finish();
        let mut planes = BTreeMap::new();
        planes.insert("cwp".to_string(), key);
        let meta = PackMeta {
            schema: CWP_SCHEMA.to_string(),
            status: "READY".to_string(),
            satellite: "G19".to_string(),
            sector: "C".to_string(),
            scan_start: "2026-08-04T18:01:17Z".to_string(),
            scan_end: "2026-08-04T18:03:54.500Z".to_string(),
            sources: Vec::new(),
            window: None,
            projection: ProjectionEntry {
                perspective_point_height_m: 35786023.0,
                semi_major_axis_m: 6378137.0,
                semi_minor_axis_m: 6356752.31414,
                longitude_of_projection_origin_deg: -75.0,
                sweep_angle_axis: "x".to_string(),
            },
            nx: 2,
            ny: 1,
            x_scan_rad: vec![0.0, 1.0e-4],
            y_scan_rad: vec![0.0],
            planes,
            plane_order: vec!["cwp".to_string()],
            arrays,
            payload_bytes: payload.len(),
            content_sha256: hex_sha256(&payload),
            cwp_counts: CwpRow::default(),
            coefficients: CoefficientTable {
                formula: CWP_FORMULA.to_string(),
                liquid_density_g_cm3: 1.0,
                ice_density_g_cm3: 0.917,
                ice_coefficient_provisional: true,
                mixed_phase_takes_ice_branch_provisional: true,
                clear_sky_emits_zero: true,
            },
        };
        std::fs::write(&path, encode_pack(&meta, &payload).unwrap()).unwrap();

        // Matching identity pairs, and records the OTHER grid.
        let paired = read_sibling(&path, "G19", "C", "2026-08-04T18:01:17Z").unwrap();
        assert_eq!(paired.content_sha256, meta.content_sha256);
        assert_eq!((paired.nx, paired.ny), (2, 1));
        assert_eq!(paired.schema, CWP_SCHEMA);

        // Every axis of the identity refuses on its own, by name.
        for (satellite, sector, scan_start, needle) in [
            ("G18", "C", "2026-08-04T18:01:17Z", "satellite"),
            ("G19", "F", "2026-08-04T18:01:17Z", "sector"),
            ("G19", "C", "2026-08-04T18:06:17Z", "scan start"),
        ] {
            let err = read_sibling(&path, satellite, sector, scan_start)
                .unwrap_err()
                .to_string();
            assert!(err.contains(needle), "{err}");
            assert!(err.contains("wrong cloud"), "{err}");
        }

        // A cloud-top pack is not a CWP pack, so it cannot be its own
        // sibling.
        assert!(read_sibling(&dir.join("absent"), "G19", "C", "x").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn verify_picks_the_family_from_the_schema_not_the_file_name() {
        use cloudtop::CloudTopMeta;
        use pack::encode_container;
        let dir = std::env::temp_dir().join(format!("rw-goes-verify-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let mut builder = PayloadBuilder::new();
        let key = builder.push_f32(&[9500.0, f32::NAN], vec![1, 2]);
        let (payload, arrays) = builder.finish();
        let mut planes = BTreeMap::new();
        planes.insert("cloud_top_height_m".to_string(), key);
        let meta = CloudTopMeta {
            schema: CLOUDTOP_SCHEMA.to_string(),
            status: "READY".to_string(),
            satellite: "G19".to_string(),
            sector: "C".to_string(),
            scan_start: "2026-08-04T18:01:17Z".to_string(),
            scan_end: "2026-08-04T18:03:54.500Z".to_string(),
            sources: Vec::new(),
            window: None,
            projection: ProjectionEntry {
                perspective_point_height_m: 35786023.0,
                semi_major_axis_m: 6378137.0,
                semi_minor_axis_m: 6356752.31414,
                longitude_of_projection_origin_deg: -75.0,
                sweep_angle_axis: "x".to_string(),
            },
            nx: 2,
            ny: 1,
            x_scan_rad: vec![0.0, 5.0e-4],
            y_scan_rad: vec![0.0],
            planes,
            plane_order: vec!["cloud_top_height_m".to_string()],
            arrays,
            payload_bytes: payload.len(),
            content_sha256: hex_sha256(&payload),
            pairs_with_schema: pairs_with_schema(),
            regrid: NO_REGRID.to_string(),
            sibling: None,
        };
        // Deliberately a name that says nothing, and a name that lies.
        for name in ["anonymous.bin", "definitely_a_cwp_pack.goespack"] {
            let path = dir.join(name);
            std::fs::write(&path, encode_container(&meta, &payload).unwrap()).unwrap();
            let json = cmd_verify(&Options {
                pack: Some(path),
                ..Options::default()
            })
            .unwrap();
            assert!(json.contains(CLOUDTOP_VERIFY_SCHEMA), "{json}");
            assert!(json.contains(CLOUDTOP_SCHEMA), "{json}");
            assert!(json.contains("cloud_top_height_m"), "{json}");
            assert!(!json.contains("cwp_counts"), "{json}");
        }

        // A foreign schema in a well-formed container is named, not guessed.
        let path = dir.join("foreign.goespack");
        #[derive(Serialize)]
        struct Foreign {
            schema: &'static str,
            content_sha256: String,
        }
        std::fs::write(
            &path,
            encode_container(
                &Foreign {
                    schema: "gpuwm-obs.something-else.v9",
                    content_sha256: hex_sha256(&payload),
                },
                &payload,
            )
            .unwrap(),
        )
        .unwrap();
        let err = cmd_verify(&Options {
            pack: Some(path),
            ..Options::default()
        })
        .unwrap_err()
        .to_string();
        assert!(err.contains("gpuwm-obs.something-else.v9"), "{err}");
        assert!(err.contains(CLOUDTOP_SCHEMA), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_granule_passed_under_the_wrong_flag_is_refused_by_name() {
        // No file is read past its name: the refusal beats the IO error.
        let err = decode_source(
            Path::new("OR_ABI-L2-ACTPC-M6_G19_s20262161801170_e20262161803543_c20262161805066.nc"),
            CloudProduct::CloudTopPressure,
            None,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("--ctp"), "{err}");
        assert!(err.contains("ABI-L2-ACTPC"), "{err}");
        // The matching flag gets past the name gate and fails on the
        // absent file instead, which is the next honest error.
        let err = decode_source(
            Path::new("OR_ABI-L2-CTPC-M6_G19_s20262161801170_e20262161803543_c20262161805066.nc"),
            CloudProduct::CloudTopPressure,
            None,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("cannot read the CTP granule"), "{err}");
    }

    #[test]
    fn cache_dir_defaults_next_to_the_output_then_to_the_cwd() {
        assert_eq!(
            Options {
                out: Some(PathBuf::from("granules")),
                ..Options::default()
            }
            .cache_dir(),
            PathBuf::from("granules").join(".cache")
        );
        assert_eq!(
            Options::default().cache_dir(),
            PathBuf::from(DEFAULT_CACHE_DIR)
        );
        assert_eq!(
            flags(&["--cache", "explicit", "--out", "granules"]).cache_dir(),
            PathBuf::from("explicit")
        );
    }

    #[test]
    fn a_mesoscale_cwp_request_is_refused_before_any_request_goes_out() {
        // COD and CTP publish no mesoscale sector, so a meso CWP scan can
        // never be completed. Listing an absent prefix would report an
        // empty sky, which is a different fact.
        let err = list_scans(
            &GoesSatellite::G19,
            Sector::Meso1,
            &[
                CloudProduct::OpticalDepth,
                CloudProduct::ParticleSize,
                CloudProduct::CloudTopPhase,
            ],
            "noaa-goes19",
            DEFAULT_MODE,
            parse_time("2026-08-04T18:00:00Z").unwrap(),
            parse_time("2026-08-04T18:05:00Z").unwrap(),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("COD"), "{err}");
        assert!(err.contains("meso"), "{err}");
    }
}
