//! `rw_goes forward`: the ABI clear-sky infrared forward operator on model
//! columns, the Rust data path the ensemble filter runs.
//!
//! Input: a `gpuwm-da.abi-columns.v2` stream (the analysed or forecast
//! columns of the global model at the observation points, top of
//! atmosphere first, with the model's own vertical coordinate riding
//! along) and a `gpuwm-da.abi-fast-model.v1` coefficient table trained on
//! CRTM's layer optical depths (`gpuwm.arwen_global.abi_fast_model`).
//! Output: the `gpuwm-da.abi-crtm.v1` stream the reference driver writes,
//! so the scorer reads the operator and the reference alike: brightness
//! temperature, the emissivity used, the radiance, the skin Jacobian, the
//! layer temperature and vapor Jacobians (central finite differences) and
//! the nadir layer optical depths.
//!
//! The arithmetic is the emission march CRTM's clear-sky solver takes
//! (upwelling layer emission, surface emission, specular reflection of the
//! downwelling along the same slant path) with the instrument's own
//! band-corrected Planck relation.  Every layer optical depth is a product
//! sum over named per-layer features (`features`), each feature held
//! inside the range the table saw in training.  A column set on another
//! vertical coordinate than the table's is refused with both identities.

use std::collections::BTreeMap;
use std::error::Error;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::pack::{boxed_error, hex_sha256};

pub const COLUMNS_MAGIC: i32 = 1_128_874_561; // "ABIC"
pub const OUTPUT_MAGIC: i32 = 1_380_532_801; // "ABIR"
pub const FORWARD_SCHEMA: &str = "gpuwm-da.abi-forward.v1";
pub const FAST_MODEL_SCHEMA: &str = "gpuwm-da.abi-fast-model.v1";

const T_REF_K: f64 = 250.0;
const T_SCALE_K: f64 = 50.0;
const U_FLOOR: f64 = 1.0e-8;
const E_FACTOR: f64 = 1.608e-3;
const PW_FLOOR_HPA: f64 = 1.0e-3;
const STEP_T_K: f64 = 0.1;
const STEP_Q_RELATIVE: f64 = 0.01;
const STEP_SKIN_K: f64 = 0.1;
const COORDINATE_TOLERANCE: f64 = 1.0e-9;

/// The feature vocabulary, in the table's order.
pub const FEATURE_NAMES: [&str; 15] = [
    "one", "dp", "u", "e", "lnp", "Tn", "lu", "lUs", "Twn", "lP", "lsec", "lu_up", "lu_dn", "Tn_up", "Tn_dn",
];

#[derive(Debug, Clone, Default)]
pub struct ForwardOptions {
    pub columns: Option<PathBuf>,
    pub table: Option<PathBuf>,
    pub out: Option<PathBuf>,
    pub emis_mode: Option<u8>,
    pub bands: Vec<u8>,
    pub threads: Option<usize>,
}

// ---------------------------------------------------------------------------
// the table
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct Table {
    pub schema: String,
    pub vertical: Vertical,
    pub bands: BTreeMap<String, BandTable>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Vertical {
    pub nlev: usize,
    pub a_half_pa: Vec<f64>,
    pub b_half: Vec<f64>,
    pub sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Planck {
    pub fk1: f64,
    pub fk2: f64,
    pub bc1: f64,
    pub bc2: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BandTable {
    pub band: u8,
    pub form: String,
    pub planck: Planck,
    #[serde(default)]
    pub terms_linear: Vec<Vec<String>>,
    #[serde(default)]
    pub terms_wet: Vec<Vec<String>>,
    #[serde(default)]
    pub terms_dry: Vec<Vec<String>>,
    pub layers: Vec<Layer>,
    #[serde(default)]
    pub clip: BTreeMap<String, ClipRow>,
    pub emissivity: Emissivity,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Layer {
    #[serde(default)]
    pub linear: Vec<f64>,
    #[serde(default)]
    pub wet: Vec<f64>,
    #[serde(default)]
    pub dry: Vec<f64>,
    #[serde(default)]
    pub ln_od_max: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ClipRow {
    pub min: Vec<f64>,
    pub max: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Emissivity {
    pub water: f64,
    pub land_by_igbp_class: Vec<f64>,
    pub snow: f64,
    pub ice: f64,
    #[serde(default = "default_snow_threshold")]
    pub snow_depth_threshold_m: f64,
}

fn default_snow_threshold() -> f64 {
    0.01
}

/// A band's table with its terms resolved to feature indices.
struct Resolved<'a> {
    band: &'a BandTable,
    linear: Vec<Vec<usize>>,
    wet: Vec<Vec<usize>>,
    dry: Vec<Vec<usize>>,
    clip: Vec<(usize, &'a ClipRow)>,
}

fn feature_index(name: &str) -> Result<usize, Box<dyn Error>> {
    FEATURE_NAMES
        .iter()
        .position(|n| *n == name)
        .ok_or_else(|| boxed_error(format!("the table names an unknown feature {name:?}; the vocabulary is {FEATURE_NAMES:?}")))
}

fn resolve<'a>(band: &'a BandTable, nlay: usize) -> Result<Resolved<'a>, Box<dyn Error>> {
    let terms = |list: &Vec<Vec<String>>| -> Result<Vec<Vec<usize>>, Box<dyn Error>> {
        list.iter().map(|t| t.iter().map(|n| feature_index(n)).collect()).collect()
    };
    if band.layers.len() != nlay {
        return Err(boxed_error(format!(
            "band {} table carries {} layers, the columns {nlay}",
            band.band,
            band.layers.len()
        )));
    }
    let resolved = Resolved {
        band,
        linear: terms(&band.terms_linear)?,
        wet: terms(&band.terms_wet)?,
        dry: terms(&band.terms_dry)?,
        clip: band
            .clip
            .iter()
            .map(|(name, row)| feature_index(name).map(|i| (i, row)))
            .collect::<Result<Vec<_>, _>>()?,
    };
    for (i, row) in &resolved.clip {
        if row.min.len() != nlay || row.max.len() != nlay {
            return Err(boxed_error(format!(
                "clip range of {} has {} / {} layers, the columns {nlay}",
                FEATURE_NAMES[*i],
                row.min.len(),
                row.max.len()
            )));
        }
    }
    match band.form.as_str() {
        "linear" => {
            for (k, layer) in band.layers.iter().enumerate() {
                if layer.linear.len() != resolved.linear.len() {
                    return Err(boxed_error(format!(
                        "band {} layer {k} has {} linear coefficients for {} terms",
                        band.band,
                        layer.linear.len(),
                        resolved.linear.len()
                    )));
                }
            }
        }
        "two_term" => {
            for (k, layer) in band.layers.iter().enumerate() {
                if layer.wet.len() != resolved.wet.len() || layer.dry.len() != resolved.dry.len() {
                    return Err(boxed_error(format!(
                        "band {} layer {k} has {} wet / {} dry coefficients for {} / {} terms",
                        band.band,
                        layer.wet.len(),
                        layer.dry.len(),
                        resolved.wet.len(),
                        resolved.dry.len()
                    )));
                }
            }
        }
        other => return Err(boxed_error(format!("band {} table form {other:?} is not linear or two_term", band.band))),
    }
    if band.emissivity.land_by_igbp_class.len() != 20 {
        return Err(boxed_error(format!(
            "band {} emissivity table has {} land classes, the IGBP table has 20",
            band.band,
            band.emissivity.land_by_igbp_class.len()
        )));
    }
    Ok(resolved)
}

// ---------------------------------------------------------------------------
// the columns
// ---------------------------------------------------------------------------

/// Every field of the stream is decoded (the length proof needs them all);
/// the operator reads the atmosphere, the surface and the angle.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Columns {
    pub n: usize,
    pub nlay: usize,
    pub a_half_pa: Vec<f64>,
    pub b_half: Vec<f64>,
    pub user_channels: Vec<i32>,
    pub lat: Vec<f64>,
    pub lon: Vec<f64>,
    pub zenith: Vec<f64>,
    pub land_fraction: Vec<f64>,
    pub skin_k: Vec<f64>,
    pub psfc_hpa: Vec<f64>,
    pub wind10: Vec<f64>,
    pub snowh_m: Vec<f64>,
    pub seaice: Vec<f64>,
    pub climatology: Vec<i32>,
    pub land_type: Vec<i32>,
    /// `(n, nlay + 1)` row-major, top first.
    pub p_half_hpa: Vec<f64>,
    /// `(n, nlay)` row-major, top first.
    pub p_full_hpa: Vec<f64>,
    pub temperature_k: Vec<f64>,
    pub q_gkg: Vec<f64>,
    pub o3_ppmv: Vec<f64>,
    /// `(n, n_user_chan)` row-major.
    pub emissivity_user: Vec<f64>,
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
    path: &'a Path,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, count: usize, what: &str) -> Result<&'a [u8], Box<dyn Error>> {
        let end = self.offset.checked_add(count).ok_or_else(|| boxed_error("stream offset overflow"))?;
        if end > self.bytes.len() {
            return Err(boxed_error(format!(
                "{} ends after {} bytes while reading {what} at offset {} ({count} bytes wanted)",
                self.path.display(),
                self.bytes.len(),
                self.offset
            )));
        }
        let slice = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(slice)
    }
    fn i32s(&mut self, count: usize, what: &str) -> Result<Vec<i32>, Box<dyn Error>> {
        Ok(self.take(count * 4, what)?.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
    fn f64s(&mut self, count: usize, what: &str) -> Result<Vec<f64>, Box<dyn Error>> {
        Ok(self
            .take(count * 8, what)?
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect())
    }
}

/// Decode a `gpuwm-da.abi-columns.v2` stream (v1, without the coordinate, is
/// refused: the operator cannot prove the table applies).
pub fn read_columns(path: &Path) -> Result<Columns, Box<dyn Error>> {
    let bytes = std::fs::read(path).map_err(|err| boxed_error(format!("cannot read {}: {err}", path.display())))?;
    let mut cur = Cursor { bytes: &bytes, offset: 0, path };
    let header = cur.i32s(5, "the header")?;
    if header[0] != COLUMNS_MAGIC {
        return Err(boxed_error(format!("{} is not a gpuwm-da.abi-columns stream (magic {})", path.display(), header[0])));
    }
    if header[1] != 2 {
        return Err(boxed_error(format!(
            "{} is a gpuwm-da.abi-columns.v{} stream; the operator needs v2 (the vertical coordinate rides along so the \
             table can be proven to apply)",
            path.display(),
            header[1]
        )));
    }
    let n = usize::try_from(header[2]).map_err(|_| boxed_error("negative column count"))?;
    let nlay = usize::try_from(header[3]).map_err(|_| boxed_error("negative layer count"))?;
    let n_user = usize::try_from(header[4]).map_err(|_| boxed_error("negative user channel count"))?;
    if n == 0 || nlay == 0 {
        return Err(boxed_error(format!("{} carries {n} columns of {nlay} layers; nothing to evaluate", path.display())));
    }
    let a_half_pa = cur.f64s(nlay + 1, "a_half")?;
    let b_half = cur.f64s(nlay + 1, "b_half")?;
    let user_channels = cur.i32s(n_user, "user channels")?;
    let lat = cur.f64s(n, "lat")?;
    let lon = cur.f64s(n, "lon")?;
    let zenith = cur.f64s(n, "zenith")?;
    let land_fraction = cur.f64s(n, "land_fraction")?;
    let skin_k = cur.f64s(n, "skin")?;
    let psfc_hpa = cur.f64s(n, "psfc")?;
    let wind10 = cur.f64s(n, "wind10")?;
    let snowh_m = cur.f64s(n, "snowh")?;
    let seaice = cur.f64s(n, "seaice")?;
    let climatology = cur.i32s(n, "climatology")?;
    let land_type = cur.i32s(n, "land_type")?;
    let p_half_hpa = cur.f64s(n * (nlay + 1), "p_half")?;
    let p_full_hpa = cur.f64s(n * nlay, "p_full")?;
    let temperature_k = cur.f64s(n * nlay, "temperature")?;
    let q_gkg = cur.f64s(n * nlay, "q")?;
    let o3_ppmv = cur.f64s(n * nlay, "o3")?;
    let emissivity_user = cur.f64s(n * n_user, "user emissivity")?;
    if cur.offset != bytes.len() {
        return Err(boxed_error(format!(
            "{} holds {} bytes beyond the {} its header describes",
            path.display(),
            bytes.len() - cur.offset,
            cur.offset
        )));
    }
    Ok(Columns {
        n,
        nlay,
        a_half_pa,
        b_half,
        user_channels,
        lat,
        lon,
        zenith,
        land_fraction,
        skin_k,
        psfc_hpa,
        wind10,
        snowh_m,
        seaice,
        climatology,
        land_type,
        p_half_hpa,
        p_full_hpa,
        temperature_k,
        q_gkg,
        o3_ppmv,
        emissivity_user,
    })
}

fn same_coordinate(table: &Vertical, columns: &Columns) -> Result<(), Box<dyn Error>> {
    let mismatch = |name: &str, ours: &[f64], theirs: &[f64]| -> Result<(), Box<dyn Error>> {
        if ours.len() != theirs.len() {
            return Err(boxed_error(format!(
                "vertical coordinate mismatch: the table's {name} has {} half levels, the columns {}",
                ours.len(),
                theirs.len()
            )));
        }
        for (k, (a, b)) in ours.iter().zip(theirs).enumerate() {
            let scale = a.abs().max(b.abs()).max(1.0);
            if (a - b).abs() > COORDINATE_TOLERANCE * scale {
                return Err(boxed_error(format!(
                    "vertical coordinate mismatch at half level {k}: the table's {name} is {a}, the columns' {b}; the \
                     per-layer coefficients (table coordinate sha256 {}) do not apply to these columns",
                    table.sha256
                )));
            }
        }
        Ok(())
    };
    if table.nlev != columns.nlay {
        return Err(boxed_error(format!(
            "the table was trained on {} layers, the columns carry {}",
            table.nlev, columns.nlay
        )));
    }
    mismatch("a_half_pa", &table.a_half_pa, &columns.a_half_pa)?;
    mismatch("b_half", &table.b_half, &columns.b_half)
}

// ---------------------------------------------------------------------------
// one column
// ---------------------------------------------------------------------------

/// The per-layer features of one column, `features[f][k]`.
pub fn features(t: &[f64], q: &[f64], p_half: &[f64], p_full: &[f64], zenith_deg: f64) -> Vec<Vec<f64>> {
    let nlay = t.len();
    let sec = 1.0 / zenith_deg.to_radians().cos();
    let lsec_v = sec.ln();
    let mut f: Vec<Vec<f64>> = (0..FEATURE_NAMES.len()).map(|_| vec![0.0; nlay]).collect();
    let mut cum_u = 0.0;
    let mut cum_ut = 0.0;
    let mut cum_up = 0.0;
    for k in 0..nlay {
        let dp = p_half[k + 1] - p_half[k];
        let u = q[k] * dp;
        let tn = (t[k] - T_REF_K) / T_SCALE_K;
        cum_u += u;
        cum_ut += u * t[k];
        cum_up += u * p_full[k];
        let u_mid = (cum_u - 0.5 * u).max(U_FLOOR);
        let tw = (cum_ut - 0.5 * u * t[k]) / u_mid;
        let pw = (cum_up - 0.5 * u * p_full[k]) / u_mid;
        f[0][k] = 1.0;
        f[1][k] = dp;
        f[2][k] = u;
        f[3][k] = q[k] * p_full[k] * E_FACTOR;
        f[4][k] = p_full[k].ln();
        f[5][k] = tn;
        f[6][k] = u.max(U_FLOOR).ln();
        f[7][k] = (u_mid * sec).ln();
        f[8][k] = (tw - T_REF_K) / T_SCALE_K;
        f[9][k] = pw.max(PW_FLOOR_HPA).ln();
        f[10][k] = lsec_v;
    }
    for k in 0..nlay {
        let up = if k == 0 { k } else { k - 1 };
        let dn = if k + 1 == nlay { k } else { k + 1 };
        f[11][k] = f[6][up];
        f[12][k] = f[6][dn];
        f[13][k] = f[5][up];
        f[14][k] = f[5][dn];
    }
    f
}

fn clip_in_place(f: &mut [Vec<f64>], clip: &[(usize, &ClipRow)]) {
    for (i, row) in clip {
        for (k, v) in f[*i].iter_mut().enumerate() {
            *v = v.max(row.min[k]).min(row.max[k]);
        }
    }
}

fn term_value(f: &[Vec<f64>], term: &[usize], k: usize) -> f64 {
    term.iter().fold(1.0, |acc, &i| acc * f[i][k])
}

/// Nadir layer optical depths of one column.
fn layer_od(r: &Resolved, f: &[Vec<f64>]) -> Vec<f64> {
    let nlay = f[0].len();
    let mut od = vec![0.0; nlay];
    for k in 0..nlay {
        let layer = &r.band.layers[k];
        od[k] = match r.band.form.as_str() {
            "linear" => {
                let s: f64 = r.linear.iter().zip(&layer.linear).map(|(t, c)| c * term_value(f, t, k)).sum();
                s.max(0.0)
            }
            _ => {
                let w: f64 = r.wet.iter().zip(&layer.wet).map(|(t, c)| c * term_value(f, t, k)).sum();
                let d: f64 = r.dry.iter().zip(&layer.dry).map(|(t, c)| c * term_value(f, t, k)).sum();
                w.min(layer.ln_od_max).exp() + d.max(0.0)
            }
        };
    }
    od
}

pub fn planck_radiance(p: &Planck, t: f64) -> f64 {
    p.fk1 / ((p.fk2 / (p.bc1 + p.bc2 * t)).exp() - 1.0)
}

pub fn planck_temperature(p: &Planck, radiance: f64) -> f64 {
    (p.fk2 / (p.fk1 / radiance + 1.0).ln() - p.bc1) / p.bc2
}

/// The clear-sky emission march on slant optical depths: TOA radiance.
pub fn emission_radiance(p: &Planck, od_slant: &[f64], t: &[f64], skin_k: f64, emissivity: f64) -> f64 {
    let nlay = od_slant.len();
    let mut up = 0.0;
    let mut trans = 1.0;
    let b: Vec<f64> = t.iter().map(|&tk| planck_radiance(p, tk)).collect();
    for k in 0..nlay {
        let next = trans * (-od_slant[k]).exp();
        up += b[k] * (trans - next);
        trans = next;
    }
    let t_sfc = trans;
    // downwelling at the surface along the same path: from the surface upward
    let mut down = 0.0;
    let mut trans_to_sfc = 1.0;
    for k in (0..nlay).rev() {
        let next = trans_to_sfc * (-od_slant[k]).exp();
        down += b[k] * (trans_to_sfc - next);
        trans_to_sfc = next;
    }
    up + emissivity * planck_radiance(p, skin_k) * t_sfc + (1.0 - emissivity) * down * t_sfc
}

fn table_emissivity(e: &Emissivity, land_fraction: f64, land_type: i32, seaice: f64, snowh_m: f64) -> f64 {
    let land = land_fraction.clamp(0.0, 1.0);
    let mut water = 1.0 - land;
    let ice = seaice.clamp(0.0, 1.0) * water;
    water -= ice;
    let class = land_type.clamp(1, 20) as usize;
    let snow = snowh_m > e.snow_depth_threshold_m || land_type == 15;
    let land_e = if snow { e.snow } else { e.land_by_igbp_class[class - 1] };
    water * e.water + ice * e.ice + land * land_e
}

struct ColumnResult {
    bt: f64,
    emissivity: f64,
    radiance: f64,
    jac_tskin: f64,
    sfc_planck: f64,
    jac_t: Vec<f64>,
    jac_q: Vec<f64>,
    od: Vec<f64>,
}

fn evaluate_column(r: &Resolved, t: &[f64], q: &[f64], p_half: &[f64], p_full: &[f64], zenith: f64, skin: f64,
                   emissivity: f64) -> (f64, Vec<f64>, f64) {
    let mut f = features(t, q, p_half, p_full, zenith);
    clip_in_place(&mut f, &r.clip);
    let od = layer_od(r, &f);
    let sec = 1.0 / zenith.to_radians().cos();
    let slant: Vec<f64> = od.iter().map(|v| v * sec).collect();
    let rad = emission_radiance(&r.band.planck, &slant, t, skin, emissivity);
    (planck_temperature(&r.band.planck, rad), od, rad)
}

fn column(r: &Resolved, c: &Columns, m: usize, emissivity: f64) -> ColumnResult {
    let nlay = c.nlay;
    let t = &c.temperature_k[m * nlay..(m + 1) * nlay];
    let q = &c.q_gkg[m * nlay..(m + 1) * nlay];
    let p_half = &c.p_half_hpa[m * (nlay + 1)..(m + 1) * (nlay + 1)];
    let p_full = &c.p_full_hpa[m * nlay..(m + 1) * nlay];
    let zen = c.zenith[m];
    let skin = c.skin_k[m];
    let (bt, od, rad) = evaluate_column(r, t, q, p_half, p_full, zen, skin, emissivity);
    let mut jac_t = vec![0.0; nlay];
    let mut jac_q = vec![0.0; nlay];
    let mut tp = t.to_vec();
    let mut qp = q.to_vec();
    for k in 0..nlay {
        tp[k] = t[k] + STEP_T_K;
        let plus = evaluate_column(r, &tp, q, p_half, p_full, zen, skin, emissivity).0;
        tp[k] = t[k] - STEP_T_K;
        let minus = evaluate_column(r, &tp, q, p_half, p_full, zen, skin, emissivity).0;
        tp[k] = t[k];
        jac_t[k] = (plus - minus) / (2.0 * STEP_T_K);
        let dq = (q[k] * STEP_Q_RELATIVE).max(1.0e-9);
        qp[k] = q[k] + dq;
        let plus = evaluate_column(r, t, &qp, p_half, p_full, zen, skin, emissivity).0;
        qp[k] = q[k] - dq;
        let minus = evaluate_column(r, t, &qp, p_half, p_full, zen, skin, emissivity).0;
        qp[k] = q[k];
        jac_q[k] = (plus - minus) / (2.0 * dq);
    }
    let plus = evaluate_column(r, t, q, p_half, p_full, zen, skin + STEP_SKIN_K, emissivity).0;
    let minus = evaluate_column(r, t, q, p_half, p_full, zen, skin - STEP_SKIN_K, emissivity).0;
    ColumnResult {
        bt,
        emissivity,
        radiance: rad,
        jac_tskin: (plus - minus) / (2.0 * STEP_SKIN_K),
        sfc_planck: planck_radiance(&r.band.planck, skin),
        jac_t,
        jac_q,
        od,
    }
}

// ---------------------------------------------------------------------------
// the subcommand
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct ForwardRecord<'a> {
    schema: &'static str,
    status: &'static str,
    columns: String,
    columns_sha256: String,
    table: String,
    table_sha256: String,
    coordinate_sha256: &'a str,
    n: usize,
    layers: usize,
    bands: Vec<u8>,
    emissivity_mode: u8,
    forms: BTreeMap<String, String>,
    out: String,
    threads: usize,
    wall_s: f64,
    jacobian_steps: BTreeMap<&'static str, f64>,
}

pub fn cmd_forward(options: &ForwardOptions) -> Result<String, Box<dyn Error>> {
    let started = std::time::Instant::now();
    let columns_path = options.columns.as_deref().ok_or_else(|| boxed_error("forward needs --columns FILE"))?;
    let table_path = options.table.as_deref().ok_or_else(|| boxed_error("forward needs --table FILE"))?;
    let out = options.out.as_deref().ok_or_else(|| boxed_error("forward needs --out FILE"))?;
    let emis_mode = options.emis_mode.unwrap_or(0);
    if emis_mode > 1 {
        return Err(boxed_error("--emis-mode is 0 (the table's surface emissivity) or 1 (the columns' own)"));
    }
    let table_bytes = std::fs::read(table_path).map_err(|err| boxed_error(format!("cannot read {}: {err}", table_path.display())))?;
    let table: Table = serde_json::from_slice(&table_bytes)
        .map_err(|err| boxed_error(format!("{} is not a fast-model table: {err}", table_path.display())))?;
    if table.schema != FAST_MODEL_SCHEMA {
        return Err(boxed_error(format!("{} declares schema {:?}, not {FAST_MODEL_SCHEMA}", table_path.display(), table.schema)));
    }
    let columns = read_columns(columns_path)?;
    same_coordinate(&table.vertical, &columns)?;
    let bands: Vec<u8> = if options.bands.is_empty() {
        table.bands.values().map(|b| b.band).collect()
    } else {
        options.bands.clone()
    };
    if bands.is_empty() {
        return Err(boxed_error("the table carries no band"));
    }
    let mut resolved = Vec::new();
    for band in &bands {
        let entry = table
            .bands
            .get(&band.to_string())
            .ok_or_else(|| boxed_error(format!("band {band} is not in the table (bands {:?})", table.bands.keys().collect::<Vec<_>>())))?;
        resolved.push(resolve(entry, columns.nlay)?);
    }
    if emis_mode == 1 {
        for band in &bands {
            if !columns.user_channels.contains(&i32::from(*band)) {
                return Err(boxed_error(format!(
                    "--emis-mode 1 asks for the columns' own emissivity but the columns carry none for band {band} \
                     (channels {:?})",
                    columns.user_channels
                )));
            }
        }
    }
    let threads = options
        .threads
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))
        .clamp(1, 64);
    let n = columns.n;
    let nlay = columns.nlay;
    let nb = bands.len();
    let mut results: Vec<Option<ColumnResult>> = Vec::with_capacity(n * nb);
    results.resize_with(n * nb, || None);
    {
        let chunk = n.div_ceil(threads).max(1);
        let slots: Vec<&mut [Option<ColumnResult>]> = results.chunks_mut(chunk * nb).collect();
        std::thread::scope(|scope| {
            for (i, slot) in slots.into_iter().enumerate() {
                let start = i * chunk;
                let resolved = &resolved;
                let columns = &columns;
                let bands = &bands;
                scope.spawn(move || {
                    for (local, m) in (start..(start + slot.len() / nb)).enumerate() {
                        for (bi, r) in resolved.iter().enumerate() {
                            let emissivity = if emis_mode == 1 {
                                let idx = columns.user_channels.iter().position(|c| *c == i32::from(bands[bi])).unwrap_or(0);
                                columns.emissivity_user[m * columns.user_channels.len() + idx]
                            } else {
                                table_emissivity(
                                    &r.band.emissivity,
                                    columns.land_fraction[m],
                                    columns.land_type[m],
                                    columns.seaice[m],
                                    columns.snowh_m[m],
                                )
                            };
                            slot[local * nb + bi] = Some(column(r, columns, m, emissivity));
                        }
                    }
                });
            }
        });
    }
    // the output stream, channel-major like the reference driver's
    let mut buf: Vec<u8> = Vec::with_capacity(28 + 4 * nb + 8 * nb * n * (5 + 3 * nlay));
    for v in [OUTPUT_MAGIC, 1, n as i32, nlay as i32, nb as i32, i32::from(emis_mode), 0] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    for band in &bands {
        buf.extend_from_slice(&i32::from(*band).to_le_bytes());
    }
    let get = |m: usize, bi: usize| results[m * nb + bi].as_ref().expect("every column evaluated");
    for bi in 0..nb {
        for m in 0..n {
            buf.extend_from_slice(&get(m, bi).bt.to_le_bytes());
        }
    }
    for bi in 0..nb {
        for m in 0..n {
            buf.extend_from_slice(&get(m, bi).emissivity.to_le_bytes());
        }
    }
    for bi in 0..nb {
        for m in 0..n {
            buf.extend_from_slice(&get(m, bi).radiance.to_le_bytes());
        }
    }
    for bi in 0..nb {
        for m in 0..n {
            buf.extend_from_slice(&get(m, bi).jac_tskin.to_le_bytes());
        }
    }
    for bi in 0..nb {
        for m in 0..n {
            buf.extend_from_slice(&get(m, bi).sfc_planck.to_le_bytes());
        }
    }
    for field in 0..3 {
        for bi in 0..nb {
            for m in 0..n {
                let r = get(m, bi);
                let arr = match field {
                    0 => &r.jac_t,
                    1 => &r.jac_q,
                    _ => &r.od,
                };
                for v in arr {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
    }
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut file = std::fs::File::create(out).map_err(|err| boxed_error(format!("cannot write {}: {err}", out.display())))?;
    file.write_all(&buf)?;
    let columns_bytes = {
        let mut f = std::fs::File::open(columns_path)?;
        let mut v = Vec::new();
        f.read_to_end(&mut v)?;
        v
    };
    let record = ForwardRecord {
        schema: FORWARD_SCHEMA,
        status: "READY",
        columns: columns_path.to_string_lossy().to_string(),
        columns_sha256: hex_sha256(&columns_bytes),
        table: table_path.to_string_lossy().to_string(),
        table_sha256: hex_sha256(&table_bytes),
        coordinate_sha256: &table.vertical.sha256,
        n,
        layers: nlay,
        bands: bands.clone(),
        emissivity_mode: emis_mode,
        forms: resolved.iter().map(|r| (r.band.band.to_string(), r.band.form.clone())).collect(),
        out: out.to_string_lossy().to_string(),
        threads,
        wall_s: started.elapsed().as_secs_f64(),
        jacobian_steps: BTreeMap::from([
            ("temperature_k", STEP_T_K),
            ("vapor_relative", STEP_Q_RELATIVE),
            ("skin_k", STEP_SKIN_K),
        ]),
    };
    Ok(format!("{}\n", serde_json::to_string_pretty(&record)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn planck() -> Planck {
        Planck { fk1: 10860.400390625, fk2: 1395.18994140625, bc1: 0.07480999827384949, bc2: 0.999750018119812 }
    }

    #[test]
    fn planck_round_trips() {
        let p = planck();
        for t in [200.0, 250.0, 300.0, 330.0] {
            let back = planck_temperature(&p, planck_radiance(&p, t));
            assert!((back - t).abs() < 1e-9, "{t} -> {back}");
        }
    }

    #[test]
    fn a_transparent_atmosphere_reads_the_skin_and_an_opaque_isothermal_one_its_temperature() {
        let p = planck();
        let t = vec![250.0; 10];
        let od = vec![0.0; 10];
        let bt = planck_temperature(&p, emission_radiance(&p, &od, &t, 301.5, 1.0));
        assert!((bt - 301.5).abs() < 1e-9);
        let od = vec![5.0; 10];
        let bt = planck_temperature(&p, emission_radiance(&p, &od, &t, 301.5, 1.0));
        assert!((bt - 250.0).abs() < 1e-6, "{bt}");
        // a gray surface under a transparent sky reflects nothing when nothing comes down
        let od = vec![0.0; 10];
        let bt = planck_temperature(&p, emission_radiance(&p, &od, &t, 300.0, 0.9));
        let expected = planck_temperature(&p, 0.9 * planck_radiance(&p, 300.0));
        assert!((bt - expected).abs() < 1e-9);
    }

    #[test]
    fn features_follow_their_definitions() {
        let t = vec![220.0, 250.0, 280.0];
        let q = vec![0.01, 1.0, 10.0];
        let p_half = vec![100.0, 300.0, 600.0, 1000.0];
        let p_full = vec![173.2, 424.3, 774.6];
        let f = features(&t, &q, &p_half, &p_full, 60.0);
        assert_eq!(f.len(), FEATURE_NAMES.len());
        assert!((f[1][0] - 200.0).abs() < 1e-12);
        assert!((f[2][1] - 300.0).abs() < 1e-12);
        assert!((f[5][1]).abs() < 1e-12);
        assert!((f[10][0] - 2.0f64.ln()).abs() < 1e-9);
        // the slant path above the middle of layer 1: (2 + 150) * sec 60 = 304
        assert!((f[7][1] - (304.0f64).ln()).abs() < 1e-9, "{}", f[7][1]);
        // neighbours: the top layer repeats its own lu above, the bottom its own below
        assert_eq!(f[11][0], f[6][0]);
        assert_eq!(f[12][2], f[6][2]);
        assert_eq!(f[11][2], f[6][1]);
    }

    #[test]
    fn emissivity_composes_the_fractions() {
        let e = Emissivity { water: 0.98, land_by_igbp_class: vec![0.96; 20], snow: 0.99, ice: 0.97, snow_depth_threshold_m: 0.01 };
        assert!((table_emissivity(&e, 0.0, 17, 0.0, 0.0) - 0.98).abs() < 1e-12);
        assert!((table_emissivity(&e, 1.0, 10, 0.0, 0.0) - 0.96).abs() < 1e-12);
        assert!((table_emissivity(&e, 1.0, 10, 0.0, 0.5) - 0.99).abs() < 1e-12);
        assert!((table_emissivity(&e, 0.0, 17, 1.0, 0.0) - 0.97).abs() < 1e-12);
        assert!((table_emissivity(&e, 0.5, 10, 0.0, 0.0) - 0.97).abs() < 1e-12);
    }

    #[test]
    fn a_short_stream_and_a_wrong_version_are_refused_by_name() {
        let dir = std::env::temp_dir().join(format!("rw-goes-forward-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("short.bin");
        let mut bytes = Vec::new();
        for v in [COLUMNS_MAGIC, 2, 3, 4, 0] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&path, &bytes).unwrap();
        let err = read_columns(&path).unwrap_err().to_string();
        assert!(err.contains("ends after"), "{err}");
        let mut bytes = Vec::new();
        for v in [COLUMNS_MAGIC, 1, 3, 4, 0] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&path, &bytes).unwrap();
        let err = read_columns(&path).unwrap_err().to_string();
        assert!(err.contains("needs v2"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
