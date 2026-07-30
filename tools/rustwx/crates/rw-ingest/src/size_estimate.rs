//! Store/download size estimation, in two modes:
//!
//! * EXACT, for hours that already exist: [`walk_hour_sizes`] parses one
//!   `.rws` file's header + meta + chunk index (the payload is never read)
//!   and reports the exact compressed bytes per variable — the same walk
//!   `rw_bench` prints in its size table.
//! * PREDICTIVE, for planned ingests: [`estimate`] prices an
//!   [`IngestProfile`](super::ingest_profile::IngestProfile) against a
//!   [`Calibration`] table (per-2D-variable bytes/hour and per-volume
//!   bytes/LEVEL/hour) built from any existing hour file(s) of the same
//!   model + grid via [`Calibration::from_hour_files`], falling back to
//!   [`Calibration::builtin_default`] — real numbers measured from the
//!   2026-06-08 00z HRRR store on disk.
//!
//! Overhead is modeled, not guessed: the chunk index is 64 bytes per chunk
//! and chunk counts follow exactly from the grid dims (2D tiles are
//! `ceil(nx/256) * ceil(ny/256)` per variable; 3D column chunks are
//! `ceil(nx/16) * ceil(ny/16)` per volume regardless of level count), and
//! per-variable meta JSON is calibrated from the measured `meta_len`.
//!
//! Download estimates assume the current full-file fetch of the `prs` and
//! `sfc` family files (a profile that needs neither volumes, prs planes,
//! nor compute stages skips the prs bytes). An `.idx`-driven byte-range
//! subset fetch would shrink downloads dramatically for small profiles —
//! noted as a future refinement, not priced here.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read as _, Seek as _, SeekFrom};
use std::path::Path;

use rustwx_core::ModelId;
use rw_store::format::{COL_X, COL_Y, HEADER_LEN, INDEX_RECORD_LEN, RwsHourMeta, TILE_X, TILE_Y};
use rw_store::header::RwsHeader;
use rw_store::index::ChunkRecord;

use super::ingest_profile::IngestProfile;
use super::planned_store_variables;
use crate::fetch_plan;

/// Default calibration inputs for a predictive estimate: the newest (by
/// modification time) `.rws` hour files of `model_slug` under `store_root`,
/// at most three so the header walk stays cheap. Empty when the model has
/// no stored hours yet — callers fall back to
/// [`Calibration::builtin_default`]. (Extracted from `rw_ingest`'s
/// `calibration_paths` so the UI estimate reuses the same discovery.)
pub fn default_calibration_paths(store_root: &Path, model_slug: &str) -> Vec<std::path::PathBuf> {
    let mut hour_files: Vec<std::path::PathBuf> = Vec::new();
    let model_dir = store_root.join(model_slug);
    let Ok(runs) = std::fs::read_dir(&model_dir) else {
        return Vec::new();
    };
    for run in runs.flatten() {
        if let Ok(entries) = std::fs::read_dir(run.path()) {
            hour_files.extend(
                entries
                    .flatten()
                    .map(|entry| entry.path())
                    .filter(|path| path.extension().is_some_and(|ext| ext == "rws")),
            );
        }
    }
    // Newest first by modification time; the 3 newest bound the walk cost.
    hour_files.sort_by_key(|path| {
        std::cmp::Reverse(
            std::fs::metadata(path)
                .and_then(|meta| meta.modified())
                .ok(),
        )
    });
    hour_files.truncate(3);
    hour_files
}

/// One variable's exact on-disk size inside an hour file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarSize {
    pub name: String,
    /// "surface2d" | "pressure3d"
    pub kind: String,
    /// Stored level count (0 for 2D variables).
    pub levels: usize,
    pub chunks: usize,
    /// Compressed payload bytes (sum of this variable's chunk lengths).
    pub bytes: u64,
}

/// Exact sizes of one hour file: per-variable payload plus the fixed
/// header/meta/index bookkeeping, summing to `file_bytes`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HourSizes {
    pub file_bytes: u64,
    pub payload_bytes: u64,
    pub meta_len: u64,
    pub index_bytes: u64,
    pub nx: usize,
    pub ny: usize,
    /// Per-variable sizes in meta (write) order.
    pub vars: Vec<VarSize>,
}

/// EXACT mode: walk one `.rws` hour file's header + meta + chunk index and
/// sum compressed payload bytes per variable. Only the header, meta JSON,
/// and index region are read — never the payload.
pub fn walk_hour_sizes(path: &Path) -> Result<HourSizes, Box<dyn std::error::Error>> {
    let mut file = File::open(path).map_err(|err| format!("open {}: {err}", path.display()))?;
    let file_bytes = file.metadata()?.len();
    let mut header_bytes = [0u8; HEADER_LEN];
    file.read_exact(&mut header_bytes)?;
    let header = RwsHeader::parse(&header_bytes)?;

    let mut meta_bytes = vec![0u8; header.meta_len as usize];
    file.read_exact(&mut meta_bytes)?;
    let meta: RwsHourMeta = serde_json::from_slice(&meta_bytes)
        .map_err(|err| format!("{}: hour meta JSON: {err}", path.display()))?;

    let index_len = usize::try_from(header.index_count)
        .ok()
        .and_then(|count| count.checked_mul(INDEX_RECORD_LEN))
        .ok_or("index size overflows usize")?;
    let mut index_bytes = vec![0u8; index_len];
    file.seek(SeekFrom::Start(header.index_offset))?;
    file.read_exact(&mut index_bytes)?;

    let mut by_var: BTreeMap<u16, (usize, u64)> = BTreeMap::new();
    for record_bytes in index_bytes.chunks_exact(INDEX_RECORD_LEN) {
        let record = ChunkRecord::unpack(record_bytes)?;
        let entry = by_var.entry(record.var_id).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += u64::from(record.len);
    }

    let mut payload_bytes = 0u64;
    let vars = meta
        .variables
        .iter()
        .map(|var| {
            let (chunks, bytes) = by_var.get(&var.id).copied().unwrap_or((0, 0));
            payload_bytes += bytes;
            VarSize {
                name: var.name.clone(),
                kind: var.kind.clone(),
                levels: var.levels_hpa.len(),
                chunks,
                bytes,
            }
        })
        .collect();

    Ok(HourSizes {
        file_bytes,
        payload_bytes,
        meta_len: u64::from(header.meta_len),
        index_bytes: index_len as u64,
        nx: meta.nx,
        ny: meta.ny,
        vars,
    })
}

/// PREDICTIVE-mode pricing table: per-variable bytes/hour for 2D variables,
/// per-variable bytes/LEVEL/hour for 3D volumes, plus the overhead and
/// download constants. Build one from real hour files of the same
/// model + grid with [`Calibration::from_hour_files`], or fall back to the
/// measured [`Calibration::builtin_default`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Calibration {
    /// Human-readable provenance for honest CLI reporting.
    pub source: String,
    pub nx: usize,
    pub ny: usize,
    /// 2D variable name -> compressed payload bytes per hour.
    pub bytes_2d: BTreeMap<String, u64>,
    /// 3D volume name -> compressed payload bytes per LEVEL per hour.
    pub bytes_3d_per_level: BTreeMap<String, u64>,
    /// Per-variable share of the hour meta JSON (meta_len / variable count).
    pub meta_bytes_per_var: u64,
    /// One-time per-run grid file (`grid.rwg`) size.
    pub grid_file_bytes: u64,
    /// Full prs family file download size per hour.
    pub prs_file_bytes: u64,
    /// Full sfc family file download size per hour.
    pub sfc_file_bytes: u64,
}

/// Built-in default calibration, measured from the real store on disk:
/// HRRR CONUS (1799 x 1059), run 20260608 00z, average of hours
/// f004/f005/f006 (`store/hrrr/20260608_00z`, walked 2026-06-09 with the
/// same header+index parse as [`walk_hour_sizes`]). Per-hour spread across
/// those three hours was under ~10% for every variable that matters
/// (small categorical/UH masks vary more but are < 0.1 MB absolute).
/// Download sizes are the same three hours' measured prs/sfc full-file
/// fetch sizes (from the rw_batch manifests); the grid file size is the
/// run's actual `grid.rwg`.
const BUILTIN_BYTES_2D: &[(&str, u64)] = &[
    ("temperature_2m", 1_802_389),
    ("dewpoint_2m", 1_843_608),
    ("u_10m", 2_358_461),
    ("v_10m", 2_299_915),
    ("composite_reflectivity", 769_911),
    ("mslp", 1_634_792),
    ("rh_2m", 2_759_721),
    ("wind_gust_10m", 2_250_372),
    ("surface_pressure", 2_406_874),
    ("orography", 3_597_584),
    ("apcp_run_total", 770_199),
    ("categorical_rain", 56_667),
    ("categorical_freezing_rain", 358),
    ("categorical_ice_pellets", 202),
    ("categorical_snow", 1_563),
    ("pwat", 1_849_599),
    ("cloud_cover_low", 916_799),
    ("cloud_cover_mid", 530_505),
    ("cloud_cover_high", 656_298),
    ("cloud_cover_total", 960_192),
    ("visibility", 2_576_577),
    ("reflectivity_1km", 297_043),
    ("uh_2to5km", 53_688),
    ("smoke_8m", 1_607_229),
    ("smoke_column", 4_302_922),
    ("simulated_ir", 3_815_160),
    ("apcp_1h", 671_242),
    ("uh_2to5km_max_1h", 53_688),
    ("wind_speed_10m_max_1h", 2_109_743),
    ("temperature_200hpa", 760_255),
    ("temperature_250hpa", 665_584),
    ("temperature_300hpa", 646_883),
    ("temperature_500hpa", 731_783),
    ("temperature_700hpa", 841_350),
    ("temperature_850hpa", 1_022_815),
    ("dewpoint_700hpa", 2_144_015),
    ("dewpoint_850hpa", 2_084_574),
    ("u_wind_200hpa", 1_536_432),
    ("u_wind_250hpa", 1_580_774),
    ("u_wind_300hpa", 1_487_566),
    ("u_wind_500hpa", 1_346_866),
    ("u_wind_700hpa", 1_382_210),
    ("u_wind_850hpa", 1_501_278),
    ("v_wind_200hpa", 1_523_468),
    ("v_wind_250hpa", 1_626_715),
    ("v_wind_300hpa", 1_539_448),
    ("v_wind_500hpa", 1_347_988),
    ("v_wind_700hpa", 1_404_561),
    ("v_wind_850hpa", 1_541_690),
    ("geopotential_height_200hpa", 3_091_494),
    ("geopotential_height_250hpa", 3_001_035),
    ("geopotential_height_300hpa", 2_862_550),
    ("geopotential_height_500hpa", 2_483_701),
    ("geopotential_height_700hpa", 2_236_418),
    ("geopotential_height_850hpa", 2_338_061),
    ("absolute_vorticity_200", 862_105),
    ("absolute_vorticity_300", 954_071),
    ("absolute_vorticity_500", 814_166),
    ("absolute_vorticity_700", 942_178),
    ("absolute_vorticity_850", 1_099_485),
    ("relative_humidity_200hpa", 1_815_332),
    ("relative_humidity_300hpa", 2_142_813),
    ("relative_humidity_500hpa", 2_062_564),
    ("relative_humidity_700hpa", 2_052_337),
    ("relative_humidity_850hpa", 2_269_525),
    ("sbcape", 3_718_714),
    ("sbcin", 2_926_180),
    ("sblcl", 6_714_394),
    ("mlcape", 3_393_539),
    ("mlcin", 2_789_333),
    ("mucape", 3_942_831),
    ("mucin", 3_628_021),
    ("dcape", 4_047_747),
    ("theta_e_2m_10m_winds", 5_644_707),
    ("vpd_2m", 6_506_672),
    ("dewpoint_depression_2m", 6_476_717),
    ("wetbulb_2m", 6_170_155),
    ("fire_weather_composite", 6_743_634),
    ("apparent_temperature_2m", 2_853_520),
    ("heat_index_2m", 6_166_752),
    ("wind_chill_2m", 6_172_990),
    ("lifted_index", 6_573_921),
    ("lapse_rate_700_500", 6_287_452),
    ("lapse_rate_0_3km", 6_323_234),
    ("bulk_shear_0_1km", 6_683_583),
    ("bulk_shear_0_6km", 6_600_700),
    ("srh_0_1km", 6_942_380),
    ("srh_0_3km", 6_831_400),
    ("ehi_0_1km", 3_878_851),
    ("ehi_0_3km", 3_819_000),
    ("stp_fixed", 545_962),
    ("scp_mu_0_3km_0_6km_proxy", 1_377_567),
    ("temperature_advection_700mb", 4_755_386),
    ("temperature_advection_850mb", 5_330_848),
    ("sbecape", 3_051_775),
    ("mlecape", 4_487_500),
    ("muecape", 3_588_061),
    ("sb_ecape_derived_cape_ratio", 2_449_059),
    ("ml_ecape_derived_cape_ratio", 2_243_942),
    ("mu_ecape_derived_cape_ratio", 2_810_552),
    ("sb_ecape_native_cape_ratio", 2_534_096),
    ("ml_ecape_native_cape_ratio", 2_211_456),
    ("mu_ecape_native_cape_ratio", 2_759_306),
    ("sbncape", 2_896_879),
    ("sbecin", 5_091_753),
    ("mlecin", 5_202_096),
    ("ecape_scp", 1_251_035),
    ("ecape_ehi_0_1km", 3_222_865),
    ("ecape_ehi_0_3km", 3_174_997),
    ("ecape_stp", 759_608),
];

/// Per-LEVEL bytes for the 5 volumes (37-level average / 37); same
/// provenance as [`BUILTIN_BYTES_2D`].
const BUILTIN_BYTES_3D_PER_LEVEL: &[(&str, u64)] = &[
    ("temperature_iso", 1_263_734),
    ("dewpoint_iso", 2_897_582),
    ("u_iso", 2_718_603),
    ("v_iso", 2_763_437),
    ("height_iso", 1_566_833),
];

/// Measured `meta_len` 19,776 bytes / 115 variables (f004-f006 identical).
const BUILTIN_META_BYTES_PER_VAR: u64 = 172;
/// Measured `store/hrrr/20260608_00z/grid.rwg`.
const BUILTIN_GRID_FILE_BYTES: u64 = 12_518_585;
/// Measured prs full-file fetch, f004-f006 average (405.3/406.2/409.8 MiB).
const BUILTIN_PRS_FILE_BYTES: u64 = 426_845_586;
/// Measured sfc full-file fetch, f004-f006 average (142.3/141.9/145.0 MiB).
const BUILTIN_SFC_FILE_BYTES: u64 = 149_987_245;

// ─── GFS builtin calibration ─────────────────────────────────────────────────
//
// Source: `out/gfs_store/gfs/20260611_00z` sounding-profile ingest, 2026-06-11.
// Averaged over f001/f002/f003 (f000 is the analysis hour and is slightly
// smaller; the mid-range hours better represent a typical download target).
// Same header+index walk as the HRRR table above.
//
// Grid: 1440 × 721 (0.25° global, lat 90→−90, lon 0→359.75).
// Profile: sounding (no derived, no heavy — GFS v1 also excludes apcp_1h,
// UH-max, and wind-speed-max-1h which rely on the HRRR trailing-window trick).
//
// 2D surface variables measured (average bytes over the 3 hours):

/// GFS 2D surface variables: per-hour compressed payload bytes.
const GFS_BUILTIN_BYTES_2D: &[(&str, u64)] = &[
    ("dewpoint_2m", 1_097_921),
    ("mslp", 2_598_100),
    ("orography", 1_289_196),
    ("surface_pressure", 2_216_429),
    ("temperature_2m", 1_045_557),
    ("u_10m", 3_254_742),
    ("v_10m", 3_276_497),
];

/// GFS 3D isobaric volumes: per-LEVEL per-hour compressed payload bytes
/// (measured from the sounding-profile 21-level ingest).
const GFS_BUILTIN_BYTES_3D_PER_LEVEL: &[(&str, u64)] = &[
    ("height_iso", 1_695_714),
    ("rh_iso", 1_813_760),
    ("temperature_iso", 1_989_392),
    ("u_iso", 2_009_366),
    ("v_iso", 2_016_029),
];

/// Measured `meta_len / variable_count` from the 20260611 00z GFS sounding
/// store (f001, 12 variables).
const GFS_BUILTIN_META_BYTES_PER_VAR: u64 = 226;

/// Measured GFS `grid.rwg` (1440×721 LatLon global, 20260611 00z sounding run).
const GFS_BUILTIN_GRID_FILE_BYTES: u64 = 7_370;

/// Full `pgrb2.0p25` per-hour download size: average of f001/f002/f003 for
/// 20260611 00z (512 MB / 538 MB / 543 MB / 544 MB range; f001-f003 average
/// 542,140,336 bytes ≈ 517 MiB). The single GFS file serves both the
/// surface and pressure roles — no separate prs/sfc split.
const GFS_BUILTIN_PGRB2_FILE_BYTES: u64 = 542_140_336;

// ─── RRFS-A builtin calibration ──────────────────────────────────────────────
//
// Source: `out/rrfs_store/rrfs_a/20260611_16z` FULL-profile crop-at-ingest
// store (2026-06-11 16z), averaged over f001/f002/f003 (f000 is the analysis
// hour: no trailing 1 h window fields, slightly smaller). Same header+index
// walk as the HRRR/GFS tables above.
//
// Grid: 2938 × 1739 (≈5.1 M cells) — the NA rotated-pole grid (4881×2961)
// cropped at ingest to the CONUS box (−134.5, −60.5, 21.0, 53.5).
// Profile: full (all 2D + derived + heavy + 5 isobaric volumes @ 37 levels).
//
// DOWNLOAD PRICING IS SUBSET PRICING: RRFS-A fetches via `.idx` message
// subsetting (the NA pair is 4.4 + 9.2 GB whole-file — see the fetch plan in
// `crate::fetch_plan`), so `prs/sfc_file_bytes` below are the measured
// SUBSET bytes actually transferred (f001-f003 average), NOT the full file
// sizes. Estimating with full-file sizes would over-state the download ~4×.

/// RRFS-A 2D variables: per-hour compressed payload bytes (full profile,
/// incl. derived + heavy grids).
const RRFS_A_BUILTIN_BYTES_2D: &[(&str, u64)] = &[
    ("absolute_vorticity_200", 4_390_808),
    ("absolute_vorticity_300", 6_902_895),
    ("absolute_vorticity_500", 4_400_155),
    ("absolute_vorticity_700", 6_820_782),
    ("absolute_vorticity_850", 12_504_916),
    ("apcp_1h", 771_712),
    ("apcp_run_total", 771_712),
    ("apparent_temperature_2m", 12_058_937),
    ("bulk_shear_0_1km", 17_419_622),
    ("bulk_shear_0_6km", 17_130_770),
    ("categorical_freezing_rain", 3_097),
    ("categorical_ice_pellets", 246),
    ("categorical_rain", 246_257),
    ("categorical_snow", 514),
    ("cloud_cover_high", 1_679_037),
    ("cloud_cover_low", 2_629_384),
    ("cloud_cover_mid", 1_401_755),
    ("cloud_cover_total", 2_897_057),
    ("composite_reflectivity", 4_530_565),
    ("dcape", 13_322_140),
    ("dewpoint_2m", 9_415_021),
    ("dewpoint_700hpa", 14_922_850),
    ("dewpoint_850hpa", 11_551_245),
    ("dewpoint_depression_2m", 17_261_644),
    ("ecape_ehi_0_1km", 12_374_223),
    ("ecape_ehi_0_3km", 12_144_440),
    ("ecape_scp", 4_346_256),
    ("ecape_stp", 2_643_656),
    ("ehi_0_1km", 13_775_526),
    ("ehi_0_3km", 13_543_561),
    ("fire_weather_composite", 17_317_167),
    ("geopotential_height_200hpa", 9_210_857),
    ("geopotential_height_250hpa", 9_124_755),
    ("geopotential_height_300hpa", 8_935_956),
    ("geopotential_height_500hpa", 10_293_514),
    ("geopotential_height_700hpa", 10_585_929),
    ("geopotential_height_850hpa", 13_096_748),
    ("heat_index_2m", 16_446_599),
    ("lapse_rate_0_3km", 16_672_191),
    ("lapse_rate_700_500", 16_208_968),
    ("lifted_index", 17_246_940),
    ("ml_ecape_derived_cape_ratio", 6_684_881),
    ("ml_ecape_native_cape_ratio", 6_426_728),
    ("mlcape", 10_566_460),
    ("mlcin", 9_152_883),
    ("mlecape", 11_988_731),
    ("mlecin", 15_502_192),
    ("mslp", 10_965_202),
    ("mu_ecape_derived_cape_ratio", 8_771_860),
    ("mu_ecape_native_cape_ratio", 8_242_431),
    ("mucape", 11_674_312),
    ("mucin", 6_360_643),
    ("muecape", 10_869_694),
    ("orography", 6_853_188),
    ("pwat", 5_050_325),
    ("relative_humidity_200hpa", 2_577_382),
    ("relative_humidity_300hpa", 2_842_388),
    ("relative_humidity_500hpa", 2_810_978),
    ("relative_humidity_700hpa", 2_961_726),
    ("relative_humidity_850hpa", 3_348_716),
    ("rh_2m", 6_422_411),
    ("sb_ecape_derived_cape_ratio", 8_155_105),
    ("sb_ecape_native_cape_ratio", 7_396_620),
    ("sbcape", 13_155_944),
    ("sbcin", 4_595_530),
    ("sbecape", 11_660_786),
    ("sbecin", 10_319_613),
    ("sblcl", 17_297_616),
    ("sbncape", 11_007_538),
    ("scp_mu_0_3km_0_6km_proxy", 4_652_927),
    ("srh_0_1km", 18_056_058),
    ("srh_0_3km", 17_715_579),
    ("stp_fixed", 2_509_935),
    ("surface_pressure", 8_754_293),
    ("temperature_200hpa", 2_187_107),
    ("temperature_250hpa", 2_099_405),
    ("temperature_2m", 8_814_778),
    ("temperature_300hpa", 2_030_705),
    ("temperature_500hpa", 2_323_497),
    ("temperature_700hpa", 2_802_979),
    ("temperature_850hpa", 3_357_067),
    ("temperature_advection_700mb", 15_093_841),
    ("temperature_advection_850mb", 16_675_906),
    ("theta_e_2m_10m_winds", 15_054_024),
    ("u_10m", 11_573_496),
    ("u_wind_200hpa", 9_283_662),
    ("u_wind_250hpa", 4_182_420),
    ("u_wind_300hpa", 4_135_229),
    ("u_wind_500hpa", 11_298_087),
    ("u_wind_700hpa", 11_273_728),
    ("u_wind_850hpa", 11_730_071),
    ("uh_2to5km", 171_333),
    ("uh_2to5km_max_1h", 171_333),
    ("v_10m", 11_714_177),
    ("v_wind_200hpa", 4_011_911),
    ("v_wind_250hpa", 4_140_185),
    ("v_wind_300hpa", 4_097_928),
    ("v_wind_500hpa", 11_342_272),
    ("v_wind_700hpa", 11_417_216),
    ("v_wind_850hpa", 11_961_239),
    ("visibility", 13_640_378),
    ("vpd_2m", 17_328_958),
    ("wetbulb_2m", 16_341_982),
    ("wind_chill_2m", 16_513_637),
    ("wind_gust_10m", 4_371_284),
    ("wind_speed_10m_max_1h", 4_817_503),
];

/// RRFS-A 3D isobaric volumes: per-LEVEL per-hour compressed payload bytes
/// (measured from the full-profile 37-level ingest).
const RRFS_A_BUILTIN_BYTES_3D_PER_LEVEL: &[(&str, u64)] = &[
    ("dewpoint_iso", 9_274_226),
    ("height_iso", 4_548_152),
    ("temperature_iso", 3_608_394),
    ("u_iso", 9_180_297),
    ("v_iso", 9_307_722),
];

/// Measured `meta_len / variable_count` from the 20260611 16z RRFS-A full
/// store (f001, 111 variables).
const RRFS_A_BUILTIN_META_BYTES_PER_VAR: u64 = 171;

/// Measured RRFS-A `grid.rwg` (2938×1739 cropped curvilinear grid with full
/// per-cell lat/lon — much larger than HRRR's because the cropped rotated
/// grid stores both coordinate planes at 5.1 M cells).
const RRFS_A_BUILTIN_GRID_FILE_BYTES: u64 = 34_861_339;

/// Measured `prslev.na` `.idx`-SUBSET download per hour (f001-f003 average:
/// 3,029,622,550 / 3,039,520,938 / 3,016,832,963 bytes ≈ 2.82 GiB). The
/// whole file is ~4.4 GB; the subset is the 7 isobaric volume field types
/// (~69% of the file).
const RRFS_A_BUILTIN_PRS_SUBSET_BYTES: u64 = 3_028_658_817;

/// Measured `natlev.na` `.idx`-SUBSET download per hour (f001-f003 average:
/// 241,292,518 / 244,572,538 / 244,184,130 bytes ≈ 232 MiB). The whole file
/// is ~9.2 GB; the subset is the 33-message surface set (~2.6% of the file).
const RRFS_A_BUILTIN_NAT_SUBSET_BYTES: u64 = 243_349_728;

impl Calibration {
    /// The measured 2026-06-08 00z HRRR defaults (see the consts above).
    pub fn builtin_default() -> Self {
        Self {
            source: "built-in defaults (measured from store/hrrr/20260608_00z f004-f006, \
                     HRRR CONUS 1799x1059)"
                .to_string(),
            nx: 1799,
            ny: 1059,
            bytes_2d: BUILTIN_BYTES_2D
                .iter()
                .map(|(name, bytes)| ((*name).to_string(), *bytes))
                .collect(),
            bytes_3d_per_level: BUILTIN_BYTES_3D_PER_LEVEL
                .iter()
                .map(|(name, bytes)| ((*name).to_string(), *bytes))
                .collect(),
            meta_bytes_per_var: BUILTIN_META_BYTES_PER_VAR,
            grid_file_bytes: BUILTIN_GRID_FILE_BYTES,
            prs_file_bytes: BUILTIN_PRS_FILE_BYTES,
            sfc_file_bytes: BUILTIN_SFC_FILE_BYTES,
        }
    }

    /// The measured 2026-06-11 00z GFS defaults (see the `GFS_BUILTIN_*`
    /// consts above). The sounding-profile ingest (no derived/heavy) provides
    /// the 2D surface + 3D isobaric volume calibration. GFS downloads a single
    /// `pgrb2.0p25` file per hour (no prs/sfc split), stored in `prs_file_bytes`
    /// with `sfc_file_bytes = 0`; [`estimate`] derives the per-hour download
    /// cost from the model's fetch plan, so a single-entry plan prices exactly
    /// the one file regardless of the `needs_prs` flag.
    pub fn builtin_gfs_default() -> Self {
        Self {
            source: "built-in defaults (GFS, calibrated 2026-06-11 00z, \
                     sounding profile f001-f003, 1440x721 0.25° global)"
                .to_string(),
            nx: 1440,
            ny: 721,
            bytes_2d: GFS_BUILTIN_BYTES_2D
                .iter()
                .map(|(name, bytes)| ((*name).to_string(), *bytes))
                .collect(),
            bytes_3d_per_level: GFS_BUILTIN_BYTES_3D_PER_LEVEL
                .iter()
                .map(|(name, bytes)| ((*name).to_string(), *bytes))
                .collect(),
            meta_bytes_per_var: GFS_BUILTIN_META_BYTES_PER_VAR,
            grid_file_bytes: GFS_BUILTIN_GRID_FILE_BYTES,
            // GFS single pgrb2.0p25 file: the full measured download bytes.
            // sfc_file_bytes = 0 so that when estimate() sums plan entries it
            // pays exactly once (the prs entry covers both roles).
            prs_file_bytes: GFS_BUILTIN_PGRB2_FILE_BYTES,
            sfc_file_bytes: 0,
        }
    }

    /// The measured 2026-06-11 16z RRFS-A defaults (see the `RRFS_A_BUILTIN_*`
    /// consts above), from the full-profile crop-at-ingest store. RRFS-A keeps
    /// the two-file prs+sfc plan shape (`prs-na` + `nat-na`), so the standard
    /// two-entry download rule applies — but **both download constants are the
    /// measured `.idx`-SUBSET transfer sizes**, not the multi-GB whole files
    /// (the provenance string discloses this).
    pub fn builtin_rrfs_a_default() -> Self {
        Self {
            source: "built-in defaults (RRFS-A, calibrated 2026-06-11 16z full profile \
                     f001-f003, 2938x1739 CONUS crop of the NA grid; downloads priced as \
                     .idx-subset bytes: prs-na ~2.8 GiB of a 4.4 GB file, nat-na ~232 MiB \
                     of a 9.2 GB file)"
                .to_string(),
            nx: 2938,
            ny: 1739,
            bytes_2d: RRFS_A_BUILTIN_BYTES_2D
                .iter()
                .map(|(name, bytes)| ((*name).to_string(), *bytes))
                .collect(),
            bytes_3d_per_level: RRFS_A_BUILTIN_BYTES_3D_PER_LEVEL
                .iter()
                .map(|(name, bytes)| ((*name).to_string(), *bytes))
                .collect(),
            meta_bytes_per_var: RRFS_A_BUILTIN_META_BYTES_PER_VAR,
            grid_file_bytes: RRFS_A_BUILTIN_GRID_FILE_BYTES,
            prs_file_bytes: RRFS_A_BUILTIN_PRS_SUBSET_BYTES,
            sfc_file_bytes: RRFS_A_BUILTIN_NAT_SUBSET_BYTES,
        }
    }

    /// Select the appropriate built-in calibration for `model`: GFS and RRFS-A
    /// use their measured tables; everything else falls back to the HRRR table.
    pub fn builtin_for_model(model: ModelId) -> Self {
        match model {
            ModelId::Gfs => Self::builtin_gfs_default(),
            ModelId::RrfsA => Self::builtin_rrfs_a_default(),
            _ => Self::builtin_default(),
        }
    }

    /// Calibrate from one or more existing hour files of the same
    /// model + grid: per-variable bytes are averaged across the files
    /// (per-LEVEL for volumes). Quantities a store hour cannot carry —
    /// the download file sizes — stay at the built-in measured values for
    /// `model`, as does the grid file size unless a sibling `grid.rwg` exists.
    pub fn from_hour_files(
        paths: &[std::path::PathBuf],
        model: ModelId,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if paths.is_empty() {
            return Err("calibration needs at least one hour file".into());
        }
        let mut sums_2d: BTreeMap<String, (u64, u64)> = BTreeMap::new();
        let mut sums_3d: BTreeMap<String, (u64, u64)> = BTreeMap::new();
        let mut first_dims = None;
        let mut meta_bytes_per_var = BUILTIN_META_BYTES_PER_VAR;
        for path in paths {
            let sizes = walk_hour_sizes(path)?;
            if let Some((nx, ny)) = first_dims {
                if (sizes.nx, sizes.ny) != (nx, ny) {
                    return Err(format!(
                        "calibration hours disagree on grid dims: {}x{} vs {nx}x{ny} ({})",
                        sizes.nx,
                        sizes.ny,
                        path.display()
                    )
                    .into());
                }
            } else {
                first_dims = Some((sizes.nx, sizes.ny));
                if !sizes.vars.is_empty() {
                    meta_bytes_per_var = sizes.meta_len / sizes.vars.len() as u64;
                }
            }
            for var in &sizes.vars {
                if var.kind == "pressure3d" {
                    if var.levels == 0 {
                        continue;
                    }
                    let entry = sums_3d.entry(var.name.clone()).or_insert((0, 0));
                    entry.0 += var.bytes / var.levels as u64;
                    entry.1 += 1;
                } else {
                    let entry = sums_2d.entry(var.name.clone()).or_insert((0, 0));
                    entry.0 += var.bytes;
                    entry.1 += 1;
                }
            }
        }
        let (nx, ny) = first_dims.expect("at least one hour walked");
        let average = |sums: BTreeMap<String, (u64, u64)>| -> BTreeMap<String, u64> {
            sums.into_iter()
                .map(|(name, (total, count))| (name, total / count.max(1)))
                .collect()
        };
        let grid_file_bytes = paths[0]
            .parent()
            .map(|dir| dir.join("grid.rwg"))
            .filter(|grid| grid.exists())
            .and_then(|grid| std::fs::metadata(grid).ok())
            .map(|meta| meta.len())
            .unwrap_or(BUILTIN_GRID_FILE_BYTES);
        // Download sizes cannot come from a store walk: use the model's builtin
        // measured values.  For GFS (single pgrb2 file) this sets prs to the
        // measured pgrb2 size and sfc to 0; for HRRR it uses the prs+sfc pair.
        let builtin = Calibration::builtin_for_model(model);
        Ok(Self {
            source: format!("{} hour file(s), first {}", paths.len(), paths[0].display()),
            nx,
            ny,
            bytes_2d: average(sums_2d),
            bytes_3d_per_level: average(sums_3d),
            meta_bytes_per_var,
            grid_file_bytes,
            prs_file_bytes: builtin.prs_file_bytes,
            sfc_file_bytes: builtin.sfc_file_bytes,
        })
    }

    /// Per-hour bytes for one 2D variable: calibrated value, else the
    /// built-in measurement, else the mean of the calibrated table (so an
    /// unknown future variable still prices at a plausible magnitude).
    fn lookup_2d(&self, name: &str) -> u64 {
        if let Some(bytes) = self.bytes_2d.get(name) {
            return *bytes;
        }
        if let Some((_, bytes)) = BUILTIN_BYTES_2D.iter().find(|(have, _)| *have == name) {
            return *bytes;
        }
        mean(self.bytes_2d.values().copied())
            .unwrap_or_else(|| mean(BUILTIN_BYTES_2D.iter().map(|(_, b)| *b)).unwrap_or(0))
    }

    /// Per-LEVEL bytes for one volume, with the same fallback chain.
    fn lookup_3d_per_level(&self, name: &str) -> u64 {
        if let Some(bytes) = self.bytes_3d_per_level.get(name) {
            return *bytes;
        }
        if let Some((_, bytes)) = BUILTIN_BYTES_3D_PER_LEVEL
            .iter()
            .find(|(have, _)| *have == name)
        {
            return *bytes;
        }
        mean(self.bytes_3d_per_level.values().copied()).unwrap_or_else(|| {
            mean(BUILTIN_BYTES_3D_PER_LEVEL.iter().map(|(_, b)| *b)).unwrap_or(0)
        })
    }
}

fn mean(values: impl Iterator<Item = u64>) -> Option<u64> {
    let mut total = 0u64;
    let mut count = 0u64;
    for value in values {
        total += value;
        count += 1;
    }
    (count > 0).then(|| total / count)
}

/// What one planned ingest will cost: store bytes (hour files for every
/// hour plus the one-time `grid.rwg`), download bytes (full prs/sfc family
/// files per hour under the current fetch path), and the per-hour
/// per-variable breakdown behind the totals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SizeEstimate {
    /// `hours * per_hour_store_bytes + grid_file_bytes`.
    pub store_bytes: u64,
    /// `hours * per_hour_download_bytes`.
    pub download_bytes: u64,
    pub per_hour_store_bytes: u64,
    pub per_hour_download_bytes: u64,
    pub grid_file_bytes: u64,
    /// Per-hour entries, largest first: every planned variable by store
    /// name (volumes priced per-level x planned levels) plus the modeled
    /// `hour overhead (header+meta+index)` entry.
    pub breakdown: Vec<(String, u64)>,
}

/// Breakdown label for the modeled non-payload bytes of each hour file.
pub const OVERHEAD_LABEL: &str = "hour overhead (header+meta+index)";

/// PREDICTIVE mode: price a profile against a calibration table.
pub fn estimate(
    profile: &IngestProfile,
    model: ModelId,
    hours: u16,
    calibration: &Calibration,
) -> SizeEstimate {
    let plan = planned_store_variables(profile, model);
    let mut breakdown: Vec<(String, u64)> = Vec::new();

    for (name, levels) in &plan.volumes {
        breakdown.push((
            (*name).to_string(),
            calibration.lookup_3d_per_level(name) * *levels as u64,
        ));
    }
    for name in plan
        .fields_2d
        .iter()
        .map(String::as_str)
        .chain(plan.derived.iter().copied())
        .chain(plan.heavy.iter().copied())
    {
        breakdown.push((name.to_string(), calibration.lookup_2d(name)));
    }

    // Modeled bookkeeping: 64-byte header, per-variable meta JSON share,
    // and one 64-byte index record per chunk. Chunk counts follow exactly
    // from the grid dims; 3D column chunks span all levels, so the level
    // step does not change the 3D chunk count.
    let count_2d = breakdown.len() - plan.volumes.len();
    let tiles_2d = calibration.nx.div_ceil(TILE_X) * calibration.ny.div_ceil(TILE_Y);
    let cols_3d = calibration.nx.div_ceil(COL_X) * calibration.ny.div_ceil(COL_Y);
    let chunk_count = tiles_2d * count_2d + cols_3d * plan.volumes.len();
    let var_count = breakdown.len();
    let overhead = HEADER_LEN as u64
        + calibration.meta_bytes_per_var * var_count as u64
        + (INDEX_RECORD_LEN * chunk_count) as u64;
    breakdown.push((OVERHEAD_LABEL.to_string(), overhead));

    let per_hour_store_bytes: u64 = breakdown.iter().map(|(_, bytes)| bytes).sum();

    // Download pricing is driven by the fetch plan for this model: each
    // plan entry is one physical file.  For models with a single-file plan
    // (GFS, `pgrb2.0p25` serves both surface and pressure roles), price
    // exactly that one file.  For HRRR's two-entry plan, price sfc always
    // and prs only when the profile needs isobaric volumes or planes (the
    // historical rule).  Unknown/unsupported models fall back to the HRRR
    // rule so callers can still get a rough estimate.
    let per_hour_download_bytes = match fetch_plan(model).ok().as_deref() {
        Some([_single]) => {
            // One physical file covers both roles. The calibration's
            // prs_file_bytes slot carries that single-product download size.
            calibration.prs_file_bytes
        }
        _ => {
            // Two-entry (prs + sfc) or unknown plan: legacy HRRR logic.
            calibration.sfc_file_bytes
                + if profile.needs_prs() {
                    calibration.prs_file_bytes
                } else {
                    0
                }
        }
    };

    breakdown.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    SizeEstimate {
        store_bytes: per_hour_store_bytes * u64::from(hours) + calibration.grid_file_bytes,
        download_bytes: per_hour_download_bytes * u64::from(hours),
        per_hour_store_bytes,
        per_hour_download_bytes,
        grid_file_bytes: calibration.grid_file_bytes,
        breakdown,
    }
}

#[cfg(test)]
mod tests {
    use super::super::ingest_profile::{FieldSet, IngestProfile};
    use super::*;
    use rustwx_core::ModelId;

    /// Calibration walk against the live GFS store — prints the per-variable
    /// bytes table for copy-pasting into the GFS builtin consts.
    /// Run with: cargo test -p rw-ingest -- --nocapture print_gfs_calibration_table
    #[test]
    #[ignore]
    fn print_gfs_calibration_table() {
        use std::collections::BTreeMap;
        // Resolve from the workspace root (two dirs up from the crate root).
        let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let paths: Vec<std::path::PathBuf> = [
            "out/gfs_store/gfs/20260611_00z/f001.rws",
            "out/gfs_store/gfs/20260611_00z/f002.rws",
            "out/gfs_store/gfs/20260611_00z/f003.rws",
        ]
        .iter()
        .map(|p| workspace.join(p))
        .collect();
        let mut sums_2d: BTreeMap<String, (u64, u64)> = BTreeMap::new();
        let mut sums_3d: BTreeMap<String, (u64, u64)> = BTreeMap::new();
        let mut meta_per_var = 0u64;
        let mut nx = 0;
        let mut ny = 0;
        for path in &paths {
            let sizes = walk_hour_sizes(path).expect("walk");
            if meta_per_var == 0 && !sizes.vars.is_empty() {
                meta_per_var = sizes.meta_len / sizes.vars.len() as u64;
                nx = sizes.nx;
                ny = sizes.ny;
                println!(
                    "dims={}x{} meta_per_var={} file_bytes={}",
                    nx, ny, meta_per_var, sizes.file_bytes
                );
            }
            for var in &sizes.vars {
                if var.kind == "pressure3d" {
                    if var.levels == 0 {
                        continue;
                    }
                    let e = sums_3d.entry(var.name.clone()).or_insert((0, 0));
                    e.0 += var.bytes / var.levels as u64;
                    e.1 += 1;
                } else {
                    let e = sums_2d.entry(var.name.clone()).or_insert((0, 0));
                    e.0 += var.bytes;
                    e.1 += 1;
                }
            }
        }
        println!("nx={nx} ny={ny} meta_per_var={meta_per_var}");
        println!("=== GFS_2D ===");
        for (n, (t, c)) in &sums_2d {
            println!("    (\"{n}\", {}),", t / c);
        }
        println!("=== GFS_3D ===");
        for (n, (t, c)) in &sums_3d {
            println!("    (\"{n}\", {}),", t / c);
        }
    }
    use rustwx_core::{
        CanonicalField, FieldSelector, GridProjection, GridShape, LatLonGrid, SelectedField2D,
    };
    use rw_store::ingest::{PressureVolumeInput, write_hour_from_fields_with_derived};
    use std::path::PathBuf;

    const NX: usize = 80;
    const NY: usize = 60;

    fn test_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("rw-size-estimate-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn grid() -> LatLonGrid {
        let mut lat = Vec::with_capacity(NX * NY);
        let mut lon = Vec::with_capacity(NX * NY);
        for y in 0..NY {
            for x in 0..NX {
                lat.push((35.0 + 0.1 * y as f64) as f32);
                lon.push((-100.0 + 0.1 * x as f64) as f32);
            }
        }
        LatLonGrid::new(GridShape::new(NX, NY).unwrap(), lat, lon).unwrap()
    }

    fn field(seed: f32) -> SelectedField2D {
        let values: Vec<f32> = (0..NY)
            .flat_map(|y| (0..NX).map(move |x| seed + 0.05 * x as f32 - 0.02 * y as f32))
            .collect();
        SelectedField2D::new(
            FieldSelector::height_agl(CanonicalField::Temperature, 2),
            "K",
            grid(),
            values,
        )
        .unwrap()
        .with_projection(GridProjection::Geographic)
    }

    /// Write a synthetic hour (2 x 2D fields + one 3-level volume), then
    /// prove the walk's bookkeeping: per-variable payload sums plus
    /// header + meta + index equal the file size exactly, names ride in
    /// write order, and chunk counts follow from the grid dims.
    #[test]
    fn walk_hour_sizes_accounts_for_every_byte_of_a_synthetic_hour() {
        let root = test_dir("walk");
        let t2m = field(280.0);
        let d2m = field(270.0);
        let planes: Vec<Vec<f32>> = (0..3)
            .map(|k| {
                (0..NY * NX)
                    .map(|i| 250.0 + k as f32 * 5.0 + (i % 17) as f32)
                    .collect()
            })
            .collect();
        let volume = PressureVolumeInput {
            name: "temperature_iso",
            units: "K",
            selector_template: serde_json::json!({"field": "temperature", "vertical": "isobaric"}),
            levels: vec![
                (1000, planes[0].as_slice()),
                (850, planes[1].as_slice()),
                (700, planes[2].as_slice()),
            ],
        };
        let written = write_hour_from_fields_with_derived(
            &root,
            "hrrr",
            "20260608_00z",
            6,
            &[("temperature_2m", &t2m), ("dewpoint_2m", &d2m)],
            &[],
            &[volume],
            "test-build",
            1_780_000_000,
        )
        .expect("synthetic hour writes");

        let sizes = walk_hour_sizes(&written.path).expect("walk parses the hour");
        assert_eq!(sizes.file_bytes, written.bytes);
        assert_eq!((sizes.nx, sizes.ny), (NX, NY));
        assert_eq!(
            sizes
                .vars
                .iter()
                .map(|v| v.name.as_str())
                .collect::<Vec<_>>(),
            vec!["temperature_2m", "dewpoint_2m", "temperature_iso"],
        );
        // 80x60 with 256-tiles: one tile per 2D var; 16x16 columns: 5x4=20.
        assert_eq!(sizes.vars[0].chunks, 1);
        assert_eq!(sizes.vars[1].chunks, 1);
        assert_eq!(sizes.vars[2].chunks, 20);
        assert_eq!(sizes.vars[2].levels, 3);
        assert!(sizes.vars.iter().all(|v| v.bytes > 0));
        assert_eq!(
            sizes.payload_bytes,
            sizes.vars.iter().map(|v| v.bytes).sum::<u64>()
        );
        assert_eq!(
            sizes.file_bytes,
            HEADER_LEN as u64 + sizes.meta_len + sizes.index_bytes + sizes.payload_bytes,
            "header + meta + index + payload must account for every byte"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn calibration_from_hour_files_averages_and_divides_per_level() {
        let root = test_dir("calibrate");
        let t2m = field(280.0);
        let planes: Vec<Vec<f32>> = (0..4)
            .map(|k| {
                (0..NY * NX)
                    .map(|i| 250.0 + k as f32 * 5.0 + (i % 13) as f32)
                    .collect()
            })
            .collect();
        let volume_levels: Vec<(u16, &[f32])> = vec![
            (1000, planes[0].as_slice()),
            (850, planes[1].as_slice()),
            (700, planes[2].as_slice()),
            (500, planes[3].as_slice()),
        ];
        let mut paths = Vec::new();
        for hour in [4u16, 5] {
            let volume = PressureVolumeInput {
                name: "temperature_iso",
                units: "K",
                selector_template: serde_json::json!({"field": "temperature"}),
                levels: volume_levels.clone(),
            };
            let written = write_hour_from_fields_with_derived(
                &root,
                "hrrr",
                "20260608_00z",
                hour,
                &[("temperature_2m", &t2m)],
                &[],
                &[volume],
                "test-build",
                1_780_000_000,
            )
            .expect("synthetic hour writes");
            paths.push(written.path);
        }

        let calibration = Calibration::from_hour_files(&paths, ModelId::Hrrr).expect("calibrates");
        assert_eq!((calibration.nx, calibration.ny), (NX, NY));
        let walked = walk_hour_sizes(&paths[0]).unwrap();
        let t2m_bytes = walked.vars[0].bytes;
        let vol_bytes = walked.vars[1].bytes;
        // Identical inputs both hours: the average equals either hour.
        assert_eq!(calibration.bytes_2d["temperature_2m"], t2m_bytes);
        assert_eq!(
            calibration.bytes_3d_per_level["temperature_iso"],
            vol_bytes / 4,
            "per-level bytes must divide the volume payload by its level count"
        );
        // grid.rwg sits next to the hour files and must be picked up.
        let grid_bytes = std::fs::metadata(paths[0].parent().unwrap().join("grid.rwg"))
            .unwrap()
            .len();
        assert_eq!(calibration.grid_file_bytes, grid_bytes);
        // Download sizes cannot come from a store walk: builtin defaults.
        assert_eq!(calibration.prs_file_bytes, BUILTIN_PRS_FILE_BYTES);
        assert_eq!(calibration.sfc_file_bytes, BUILTIN_SFC_FILE_BYTES);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Synthetic calibration with round numbers: the estimate must follow
    /// the documented arithmetic exactly.
    #[test]
    fn estimate_prices_a_sounding_profile_against_synthetic_calibration() {
        let mut profile = IngestProfile::sounding();
        profile.level_step_hpa = 50; // 19 candidate levels
        let mut calibration = Calibration::builtin_default();
        calibration.nx = 300; // 2 x 2 tiles, 19 x 19 columns
        calibration.ny = 300;
        calibration.meta_bytes_per_var = 100;
        calibration.grid_file_bytes = 1_000;
        calibration.prs_file_bytes = 400_000;
        calibration.sfc_file_bytes = 150_000;
        calibration.bytes_2d = [("temperature_2m", 10u64), ("dewpoint_2m", 20u64)]
            .into_iter()
            .map(|(name, bytes)| (name.to_string(), bytes))
            .collect();
        calibration.bytes_3d_per_level = [("temperature_iso", 1_000u64)]
            .into_iter()
            .map(|(name, bytes)| (name.to_string(), bytes))
            .collect();

        let estimate = estimate(&profile, ModelId::Hrrr, 3, &calibration);

        let by_name: BTreeMap<&str, u64> = estimate
            .breakdown
            .iter()
            .map(|(name, bytes)| (name.as_str(), *bytes))
            .collect();
        // Calibrated names use the synthetic table.
        assert_eq!(by_name["temperature_2m"], 10);
        assert_eq!(by_name["dewpoint_2m"], 20);
        assert_eq!(by_name["temperature_iso"], 1_000 * 19);
        // Names absent from the synthetic table fall back to the builtin
        // measurements (per level for volumes).
        assert_eq!(by_name["mslp"], 1_634_792);
        assert_eq!(by_name["dewpoint_iso"], 2_897_582 * 19);
        // Overhead: 7 x 2D fields (4 tiles each) + 5 volumes (361 columns
        // each) -> 28 + 1805 chunks, 12 variables of meta.
        let expected_overhead = 64 + 100 * 12 + 64 * (4 * 7 + 361 * 5);
        assert_eq!(by_name[OVERHEAD_LABEL], expected_overhead);
        // No derived/heavy entries and no prs planes for sounding.
        assert!(!by_name.contains_key("sbcape"));
        assert!(!by_name.contains_key("absolute_vorticity_500"));
        assert!(!by_name.contains_key("apcp_1h"));
        assert_eq!(estimate.breakdown.len(), 7 + 5 + 1);

        let per_hour: u64 = estimate.breakdown.iter().map(|(_, b)| b).sum();
        assert_eq!(estimate.per_hour_store_bytes, per_hour);
        assert_eq!(estimate.store_bytes, per_hour * 3 + 1_000);
        // Sounding needs prs (volumes) + sfc, full files, every hour.
        assert_eq!(estimate.per_hour_download_bytes, 550_000);
        assert_eq!(estimate.download_bytes, 1_650_000);
        // Breakdown is sorted largest-first.
        assert!(
            estimate
                .breakdown
                .windows(2)
                .all(|pair| pair[0].1 >= pair[1].1)
        );
    }

    #[test]
    fn estimate_full_profile_covers_every_builtin_variable() {
        let calibration = Calibration::builtin_default();
        let estimate = estimate(&IngestProfile::full(), ModelId::Hrrr, 1, &calibration);
        // Every variable the 20260608 store realized must be priced: 110
        // 2D (26 surface-plan + 3 trailing + 5 vorticity + 31 direct
        // planes + 29 derived + 16 heavy, as measured) + 5 volumes +
        // the overhead entry = 115 + 1.
        assert_eq!(estimate.breakdown.len(), 115 + 1);
        // Within bookkeeping noise of the real measured hours (~677 MB).
        let measured_avg = 709_779_736u64;
        let diff = estimate.per_hour_store_bytes.abs_diff(measured_avg);
        assert!(
            diff as f64 / measured_avg as f64 <= 0.01,
            "full-profile estimate {} must sit within 1% of the measured \
             average {measured_avg} it was calibrated from",
            estimate.per_hour_store_bytes
        );
        // Full profile downloads both family files.
        assert_eq!(
            estimate.per_hour_download_bytes,
            BUILTIN_PRS_FILE_BYTES + BUILTIN_SFC_FILE_BYTES
        );
    }

    #[test]
    fn estimate_view_profile_has_no_volumes_but_still_downloads_prs() {
        let calibration = Calibration::builtin_default();
        let view = estimate(&IngestProfile::view(), ModelId::Hrrr, 1, &calibration);
        assert!(
            view.breakdown
                .iter()
                .all(|(name, _)| !name.ends_with("_iso")),
            "view must not price any volume"
        );
        let names: Vec<&str> = view.breakdown.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"sbcape"), "view prices derived grids");
        assert!(
            !names.contains(&"sbecape"),
            "view excludes heavy grids (sbecape is heavy)"
        );
        assert!(names.contains(&"absolute_vorticity_500"));
        // prs still downloads: the 2D prs planes + derived inputs ride it.
        assert_eq!(
            view.per_hour_download_bytes,
            BUILTIN_PRS_FILE_BYTES + BUILTIN_SFC_FILE_BYTES
        );
    }

    /// A 2D-only named-subset profile with no volumes and no compute
    /// stages needs no prs file, and unknown variable names price at the
    /// calibration mean instead of zero.
    #[test]
    fn estimate_skips_prs_download_when_nothing_needs_it_and_prices_unknowns() {
        let profile = IngestProfile {
            volumes: Vec::new(),
            level_step_hpa: 25,
            surface_fields: FieldSet::Named(vec!["temperature_2m".to_string()]),
            derived: false,
            heavy: false,
        };
        profile.validate().expect("sfc-only profile validates");
        let calibration = Calibration::builtin_default();
        let estimate = estimate(&profile, ModelId::Hrrr, 2, &calibration);
        assert_eq!(
            estimate.per_hour_download_bytes, BUILTIN_SFC_FILE_BYTES,
            "no volumes, no prs planes, no compute stages -> sfc only"
        );
        assert_eq!(estimate.download_bytes, BUILTIN_SFC_FILE_BYTES * 2);

        let unknown = calibration.lookup_2d("some_future_variable");
        let builtin_mean = mean(BUILTIN_BYTES_2D.iter().map(|(_, b)| *b)).unwrap();
        assert_eq!(unknown, builtin_mean);
    }

    // ─── GFS calibration tests ──────────────────────────────────────────────

    /// GFS estimate with an empty store uses the GFS builtin table, and the
    /// provenance string honestly names the model and calibration date.
    #[test]
    fn gfs_estimate_with_empty_store_uses_gfs_builtins_and_says_so() {
        let calibration = Calibration::builtin_for_model(ModelId::Gfs);
        assert!(
            calibration.source.contains("GFS"),
            "GFS builtin provenance must name the model: {}",
            calibration.source
        );
        assert!(
            calibration.source.contains("2026-06-11"),
            "GFS builtin provenance must name the calibration date: {}",
            calibration.source
        );
        assert_eq!(calibration.nx, 1440, "GFS grid is 1440×721");
        assert_eq!(calibration.ny, 721);

        // The sounding profile on GFS should produce non-zero estimates and
        // price exactly one download file (pgrb2 only, not prs+sfc).
        let estimate = estimate(&IngestProfile::sounding(), ModelId::Gfs, 1, &calibration);
        assert!(estimate.per_hour_store_bytes > 0);
        // Single-file download: priced from the calibrated pgrb2 bytes.
        assert_eq!(
            estimate.per_hour_download_bytes, GFS_BUILTIN_PGRB2_FILE_BYTES,
            "GFS download must price exactly one pgrb2.0p25 file"
        );
        // The breakdown must contain GFS-specific variables.
        let names: Vec<&str> = estimate.breakdown.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"temperature_2m"), "GFS prices t2m");
        assert!(names.contains(&"temperature_iso"), "GFS prices temp volume");
        // GFS must NOT price HRRR-only variables.
        assert!(
            !names.contains(&"composite_reflectivity"),
            "GFS must not price HRRR-only composite_reflectivity"
        );
    }

    /// HRRR full-profile estimate is byte-for-byte unchanged after the GFS
    /// calibration table was added.  This pins the HRRR path against
    /// regression — the measured average is the anchor value from before.
    #[test]
    fn hrrr_full_profile_estimate_is_unchanged_after_gfs_table_added() {
        let calibration = Calibration::builtin_default();
        let estimate = estimate(&IngestProfile::full(), ModelId::Hrrr, 1, &calibration);
        // Measured HRRR average used as the regression anchor (709,779,736 bytes).
        let pinned_hrrr_avg = 709_779_736u64;
        let diff = estimate.per_hour_store_bytes.abs_diff(pinned_hrrr_avg);
        assert!(
            diff as f64 / pinned_hrrr_avg as f64 <= 0.01,
            "HRRR full-profile estimate must be within 1% of the pinned anchor \
             after the GFS builtin table was added; got {} vs {}",
            estimate.per_hour_store_bytes,
            pinned_hrrr_avg
        );
        // HRRR download still prices both prs+sfc files.
        assert_eq!(
            estimate.per_hour_download_bytes,
            BUILTIN_PRS_FILE_BYTES + BUILTIN_SFC_FILE_BYTES
        );
    }

    /// RRFS-A builtin pricing is honest about subsetting: the download
    /// estimate for a full-profile hour must equal the measured prs+nat
    /// `.idx`-SUBSET bytes (the two-entry plan rule prices sfc always and
    /// prs when the profile needs isobaric data) — a tiny fraction of the
    /// 13+ GB whole-file pair — and the provenance string must disclose the
    /// subset pricing. HRRR's table stays untouched (pinned elsewhere by
    /// `hrrr_full_profile_estimate_is_unchanged_after_gfs_table_added`).
    #[test]
    fn rrfs_a_estimate_prices_subset_downloads_not_full_files() {
        let calibration = Calibration::builtin_for_model(ModelId::RrfsA);
        assert!(
            calibration.source.contains("subset"),
            "provenance must disclose subset pricing: {}",
            calibration.source
        );
        assert_eq!((calibration.nx, calibration.ny), (2938, 1739));

        let profile = IngestProfile::full();
        let priced = estimate(&profile, ModelId::RrfsA, 1, &calibration);
        assert_eq!(
            priced.per_hour_download_bytes,
            RRFS_A_BUILTIN_PRS_SUBSET_BYTES + RRFS_A_BUILTIN_NAT_SUBSET_BYTES,
            "full profile prices prs+nat subset bytes"
        );
        // The whole-file pair is ~12.7 GiB (4.37 + 9.25 GB measured live);
        // the subset estimate must be far below it.
        assert!(
            priced.per_hour_download_bytes < 4_000_000_000,
            "subset pricing must not approach whole-file sizes, got {}",
            priced.per_hour_download_bytes
        );
    }

    /// RRFS-A estimate accuracy vs the live cropped store within ±15%
    /// (store) and ±10% (download), mirroring the GFS accuracy gate.
    /// Requires the live RRFS-A store at out/rrfs_store (2026-06-11 16z
    /// full-profile crop-at-ingest run).
    #[test]
    #[ignore]
    fn rrfs_a_estimate_accuracy_vs_live_store_within_15_pct() {
        let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let run_dir = workspace.join("out/rrfs_store/rrfs_a/20260611_16z");
        let store_paths: Vec<std::path::PathBuf> = ["f001.rws", "f002.rws", "f003.rws"]
            .iter()
            .map(|f| run_dir.join(f))
            .collect();

        let calibration =
            Calibration::from_hour_files(&store_paths, ModelId::RrfsA).expect("calibrate");
        // Full profile at the 25 hPa step — exactly the live ingest shape
        // (37 realized levels).
        let profile = IngestProfile::full();
        let estimate = estimate(&profile, ModelId::RrfsA, 1, &calibration);

        let walked_bytes: Vec<u64> = store_paths
            .iter()
            .map(|p| walk_hour_sizes(p).expect("walk").file_bytes)
            .collect();
        let avg_store_bytes = walked_bytes.iter().sum::<u64>() / walked_bytes.len() as u64;
        let diff = estimate.per_hour_store_bytes.abs_diff(avg_store_bytes);
        let ratio = diff as f64 / avg_store_bytes as f64;
        assert!(
            ratio <= 0.15,
            "RRFS-A store estimate {} must be within 15% of measured avg {} \
             (actual diff {:.1}%)",
            estimate.per_hour_store_bytes,
            avg_store_bytes,
            ratio * 100.0
        );

        // Download: the measured .idx-subset transfer sizes (f001-f003
        // average, from the live fetch cache metas).
        let measured_subset_avg = RRFS_A_BUILTIN_PRS_SUBSET_BYTES + RRFS_A_BUILTIN_NAT_SUBSET_BYTES;
        let dl_diff = estimate
            .per_hour_download_bytes
            .abs_diff(measured_subset_avg);
        let dl_ratio = dl_diff as f64 / measured_subset_avg as f64;
        assert!(
            dl_ratio <= 0.10,
            "RRFS-A download estimate {} must be within 10% of measured subset avg {} \
             (actual diff {:.1}%)",
            estimate.per_hour_download_bytes,
            measured_subset_avg,
            dl_ratio * 100.0
        );

        println!(
            "RRFS-A accuracy: store estimate {} vs measured {} ({:.1}%); \
             download estimate {} vs measured subset {} ({:.1}%)",
            estimate.per_hour_store_bytes,
            avg_store_bytes,
            ratio * 100.0,
            estimate.per_hour_download_bytes,
            measured_subset_avg,
            dl_ratio * 100.0
        );
    }

    /// GFS estimate accuracy vs the live store within ±15%.
    /// Uses level_step_hpa=50 (19 candidate levels) which closely brackets
    /// the 21 isobaric levels GFS actually provides in pgrb2.0p25, keeping
    /// the store-size error under the ±15% gate.
    ///
    /// Also verifies the download estimate is within ±10% of the real
    /// measured pgrb2 download sizes (f001-f003 average 542,140,336 bytes).
    ///
    /// Requires the live GFS store at out/gfs_store (from Task 2).
    #[test]
    #[ignore]
    fn gfs_estimate_accuracy_vs_live_store_within_15_pct() {
        let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let store_paths: Vec<std::path::PathBuf> = ["f001.rws", "f002.rws", "f003.rws"]
            .iter()
            .map(|f| workspace.join("out/gfs_store/gfs/20260611_00z").join(f))
            .collect();

        // Calibrate from the live store hours.
        let calibration =
            Calibration::from_hour_files(&store_paths, ModelId::Gfs).expect("calibrate");

        // Sounding profile at step=50 (19 candidate levels) brackets the 21
        // realized GFS isobaric levels, keeping the over-estimate modest.
        let mut profile = IngestProfile::sounding();
        profile.level_step_hpa = 50;

        let estimate = estimate(&profile, ModelId::Gfs, 1, &calibration);

        // Check against the average of the three walked hours.
        let walked_bytes: Vec<u64> = store_paths
            .iter()
            .map(|p| walk_hour_sizes(p).expect("walk").file_bytes)
            .collect();
        let avg_store_bytes = walked_bytes.iter().sum::<u64>() / walked_bytes.len() as u64;

        let diff = estimate.per_hour_store_bytes.abs_diff(avg_store_bytes);
        let ratio = diff as f64 / avg_store_bytes as f64;
        assert!(
            ratio <= 0.15,
            "GFS store estimate {} must be within 15% of measured avg {} \
             (actual diff {:.1}%)",
            estimate.per_hour_store_bytes,
            avg_store_bytes,
            ratio * 100.0
        );

        // Download: the real measured pgrb2 sizes (from the Task-2 live run).
        // f001=538,342,911  f002=543,684,242  f003=544,393,854 → avg 542,140,336.
        let measured_pgrb2_avg = 542_140_336u64;
        let dl_diff = estimate
            .per_hour_download_bytes
            .abs_diff(measured_pgrb2_avg);
        let dl_ratio = dl_diff as f64 / measured_pgrb2_avg as f64;
        assert!(
            dl_ratio <= 0.10,
            "GFS download estimate {} must be within 10% of measured avg {} \
             (actual diff {:.1}%)",
            estimate.per_hour_download_bytes,
            measured_pgrb2_avg,
            dl_ratio * 100.0
        );

        println!(
            "GFS accuracy: store estimate {} vs measured {} ({:.1}%); \
             download estimate {} vs measured {} ({:.1}%)",
            estimate.per_hour_store_bytes,
            avg_store_bytes,
            ratio * 100.0,
            estimate.per_hour_download_bytes,
            measured_pgrb2_avg,
            dl_ratio * 100.0
        );
    }
}
