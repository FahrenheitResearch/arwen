//! Group D (specification section 7): reflectivity, cloud cover, cloud
//! base, top and ceiling, visibility, and the simulated infrared brightness
//! temperature (`simulated_ir`, 2.8.5's product, restored by the b2-ir lane).
//!
//! Ported from the group D Python + CuPy lane into this crate (audit
//! blocker 4).  The data path is Rust and prebuilt CUDA only:
//! * CPU: [`column::column`], binary64 from the binary32 carriers, rayon
//!   over columns;
//! * GPU: `woof_post_d_fields_v1` in `kernels/group_d.cu`, the same
//!   statements, compiled with `--fmad=false` and exact division, with
//!   transcendentals from the shared maths library (D30), so both devices
//!   return the same bits.
//!
//! Inputs are Group A's shared column state (p_full, tk, r, z_mass) and the
//! packed hydrometeor carriers that the state build already uploaded, plus
//! the group D carriers read here: `REFL_10CM`, `CLDFRA`, the 550 nm aerosol
//! extinction, `HGT` and `OLR` ([`history`]).  `simulated_ir` has its own
//! kernel, `woof_post_d_ir_v1`, twin of [`column::ir_column`]; it also reads
//! Group A's interface pressure.
//!
//! Public sources: see `kernels/group_d.cu`.

pub mod column;
pub mod history;

#[cfg(any(windows, target_os = "linux"))]
pub mod cuda;

use rayon::prelude::*;

use crate::state::{ColumnState, MassInput, MassState, StateInputs};
use column::*;

/// Vertical interpolation of `reflectivity_1km` (D17, RULINGS 5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReflInterp {
    /// Linear in the reflectivity factor Z (mm6 m-3), then dBZ.
    LinearZ,
    /// Linear in dBZ.
    LinearDbz,
}

/// RULINGS 5 (lead, 2026-10-05): interpolate linear Z.  The MRMS check on
/// the graded frames may switch it to `LinearDbz`; this line is the switch.
pub const REFL_INTERP: ReflInterp = ReflInterp::LinearZ;

/// Where the cloud fraction of the six cloud fields comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CloudFractionSource {
    /// The model's own CLDFRA when the history carries it, else the
    /// diagnosed fraction (RULINGS 1).
    ModelThenDiagnosed,
    /// The model's CLDFRA only; without it the six fields are omitted (the
    /// policy before RULINGS 1).
    ModelOnly,
    /// Always the diagnosed fraction, even when CLDFRA is present.
    Diagnosed,
}

/// RULINGS 1 (lead, 2026-10-05): the model's radiation cloud fraction by
/// default, the diagnosed fraction for histories without it.  One-line
/// switch: `ModelOnly` drops the six cloud fields when CLDFRA is absent.
pub const CLOUD_FRACTION_SOURCE: CloudFractionSource = CloudFractionSource::ModelThenDiagnosed;

/// Diagnosed cloud-fraction method (used only without CLDFRA).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiagnosedCloud {
    /// Xu and Randall, J. Atmos. Sci. 53 (1996) 3084-3102: RH^0.25 (1 -
    /// exp(-100 l / ((1 - RH) q*)^0.49)), l = QCLOUD + QICE, q* over water
    /// at or above 0 C and over ice below (AERK / AERKi [R6]).
    XuRandall1996,
    /// 1 where QCLOUD + QICE (+ QSNOW) exceeds 1e-6 kg/kg, else 0 (the
    /// group D lane's measurement mode; kept for comparisons).
    CondensatePresence,
}

/// PROVISIONAL (not named by RULINGS 1, which says only "the spec's
/// published method"; SPEC 7.2 cites none): Xu and Randall (1996).  One-line
/// switch to `CondensatePresence`.
pub const DIAGNOSED_CLOUD: DiagnosedCloud = DiagnosedCloud::XuRandall1996;

/// Koschmieder constant for the WMO meteorological optical range, 5 %
/// contrast, -ln(0.05) (D21, RULINGS 6).
pub const CONTRAST_MOR_5PCT: f64 = 2.995732273553991;
/// The 2 % contrast alternative, -ln(0.02).
pub const CONTRAST_2PCT: f64 = 3.912023005428146;
/// RULINGS 6: 5 % MOR unless ASOS clearly prefers 2 %.  This is the switch.
pub const VIS_CONTRAST: f64 = CONTRAST_MOR_5PCT;

/// RULINGS 10 (lead, 2026-10-05): visibility is ON by default when the
/// history has no aerosol extinction carrier.  It is then the hydrometeor
/// extinction plus the clear-air baseline [`CLEAR_AIR_BASELINE`], with no
/// aerosol term, and the manifest says so ([`aerosol_note`]).  Aerosol
/// carriers are added wherever the history has them.  `false` is the
/// one-line switch back to omitting visibility without a carrier.
pub const VIS_WITHOUT_AEROSOL: bool = true;

/// The clear-air baseline of the visibility without an aerosol carrier, as
/// the manifest records it.  It is the RH haze term the column already
/// applies in every case (spec 7.4, D21): the FRAM-L relation of Gultepe
/// et al., BAMS 90 (2009) 341 [R40], Vis_km = 40.10 - 5.19e-10 RH^5.44 with
/// RH held at 30 % or more, turned into extinction with the same contrast
/// as the rest of the column.  Its dry-air value (40.1 km) lies beyond the
/// 20 km cap, so dry clear air reports the cap.  The ASOS visibility score
/// (RULINGS 6 and 10) picks the constant; changing it changes this text and
/// the `HAZE_*` constants of `column.rs` and `kernels/group_d.cu` together.
pub const CLEAR_AIR_BASELINE: &str = "clear-air baseline: FRAM-L RH haze relation (Gultepe et al., BAMS 90 (2009) 341, Vis_km = 40.10 - 5.19e-10 RH^5.44, RH >= 30 %), no aerosol extinction term";

/// Manifest text for the aerosol term of one frame's visibility: the
/// carriers used, or the clear-air baseline when there are none.
#[must_use]
pub fn aerosol_note(aerosol_source: &str, opt: &Options) -> String {
    if aerosol_source == "absent" || aerosol_source.is_empty() {
        if opt.allow_missing_aerosol {
            format!("absent (no AEXTC55, EXTCOF55 or AOD3D_SMOKE in the history); {CLEAR_AIR_BASELINE} (RULINGS 10)")
        } else {
            "absent; visibility omitted".to_owned()
        }
    } else {
        format!("{aerosol_source} (550 nm extinction added to the hydrometeor and haze terms)")
    }
}

/// Options of one group D run.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub refl_interp: ReflInterp,
    pub cloud_fraction: CloudFractionSource,
    pub diagnosed: DiagnosedCloud,
    pub contrast: f64,
    /// Produce visibility without an aerosol carrier: hydrometeor
    /// extinction plus the clear-air baseline.  On by default
    /// ([`VIS_WITHOUT_AEROSOL`], RULINGS 10); `false` omits visibility when
    /// no aerosol carrier exists.
    pub allow_missing_aerosol: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            refl_interp: REFL_INTERP,
            cloud_fraction: CLOUD_FRACTION_SOURCE,
            diagnosed: DIAGNOSED_CLOUD,
            contrast: VIS_CONTRAST,
            allow_missing_aerosol: VIS_WITHOUT_AEROSOL,
        }
    }
}

/// One group D field: identity and method string (manifest text).
#[derive(Clone, Copy, Debug)]
pub struct FieldSpec {
    pub id: &'static str,
    pub discipline: u8,
    pub category: u8,
    pub number: u8,
    pub level_type: u8,
    pub level_value: f64,
    pub units: &'static str,
    pub method: &'static str,
}

/// The ten planes, in output order (`O_*`): the nine of the column kernel,
/// then `simulated_ir` (`O_IR`).
pub const FIELDS: [FieldSpec; N_OUT] = [
    FieldSpec { id: "composite_reflectivity", discipline: 0, category: 16, number: 196, level_type: 200, level_value: 0.0, units: "dBZ",
        method: "column maximum of REFL_10CM, or of the RIP-note equivalent reflectivity (exponential distributions, fixed intercepts, Smith 1984 ice factor 0.224) [R29][R30]; floor -20 dBZ (D17)" },
    FieldSpec { id: "reflectivity_1km", discipline: 0, category: 16, number: 195, level_type: 103, level_value: 1000.0, units: "dBZ",
        method: "1000 m above ground between bracketing mass levels, linear in Z (D17, RULINGS 5); same volume and floor as composite_reflectivity" },
    FieldSpec { id: "low_cloud", discipline: 0, category: 6, number: 3, level_type: 214, level_value: 0.0, units: "%",
        method: "maximum cloud fraction (maximum overlap [R33]) for p > 680 hPa (ISCCP [R34])" },
    FieldSpec { id: "mid_cloud", discipline: 0, category: 6, number: 4, level_type: 224, level_value: 0.0, units: "%",
        method: "maximum cloud fraction for 440 < p <= 680 hPa (ISCCP [R34])" },
    FieldSpec { id: "high_cloud", discipline: 0, category: 6, number: 5, level_type: 234, level_value: 0.0, units: "%",
        method: "maximum cloud fraction for p <= 440 hPa (ISCCP [R34])" },
    FieldSpec { id: "cloud_base", discipline: 0, category: 3, number: 5, level_type: 2, level_value: 0.0, units: "gpm",
        method: "lowest crossing of cloud fraction > 0.01 and QCLOUD+QICE+QSNOW > 1e-6 kg/kg (ECMWF 228023 [R35]), height MSL" },
    FieldSpec { id: "cloud_top", discipline: 0, category: 3, number: 5, level_type: 3, level_value: 0.0, units: "gpm",
        method: "highest crossing of the cloud_base criterion, height MSL" },
    FieldSpec { id: "cloud_ceiling", discipline: 0, category: 3, number: 5, level_type: 215, level_value: 0.0, units: "gpm",
        method: "lowest crossing of cloud fraction >= 0.5 with condensate > 1e-6 kg/kg (FMH-1 broken or overcast [R36], ECMWF ceiling [R35]), height MSL; no ceiling is missing (D19)" },
    FieldSpec { id: "visibility", discipline: 0, category: 19, number: 0, level_type: 1, level_value: 0.0, units: "m",
        method: "lowest mass level: Stoelinga-Warner 1999 / Kunkel 1984 hydrometeor extinction [R37][R38], FRAM-L RH haze relation [R40] as the clear-air baseline, 550 nm aerosol extinction where the history carries it (RULINGS 10); MOR 5 % contrast (WMO-No. 8, RULINGS 6), cap 20 km" },
    FieldSpec { id: "simulated_ir", discipline: 0, category: 192, number: 0, level_type: 8, level_value: 0.0, units: "K",
        method: SIMULATED_IR_METHOD },
];

/// Manifest method of `simulated_ir` (no instrument band, not a radiative
/// transfer operator such as CRTM).
pub const SIMULATED_IR_METHOD: &str = "window brightness temperature: where the column is opaque, the air temperature at unit infrared absorption optical depth below the model top (Eddington-Barbier emission level; 0.145 m2 g-1 cloud water, 0.272 m2 g-1 cloud ice plus snow, PROVISIONAL, kept by a GOES-16/18 band 13 score); elsewhere the model's own OLR through T_f = (OLR/sigma)^(1/4) = T_b (1.228 - 1.106e-3 T_b) (Ohring, Gruber and Ellingson, J. Climate Appl. Meteor. 23 (1984) 416; Yang and Slingo, Mon. Wea. Rev. 129 (2001) 784); no instrument band, not CRTM";

/// Plane index of a field id.
#[must_use]
pub fn plane_index(id: &str) -> Option<usize> {
    FIELDS.iter().position(|f| f.id == id)
}

/// The group D carriers beyond Group A's state, one frame.
#[derive(Clone, Copy, Debug)]
pub struct Carriers<'a> {
    /// `REFL_10CM`, `[nz][ncell]` dBZ.
    pub refl: Option<&'a [f32]>,
    /// `CLDFRA`, `[nz][ncell]` as a fraction 0..1.
    pub cldfra: Option<&'a [f32]>,
    /// Lowest-level 550 nm aerosol extinction, km-1, `[ncell]`.
    pub aerosol_km: Option<&'a [f32]>,
    /// `HGT`, `[ncell]`.
    pub hgt: Option<&'a [f32]>,
    /// `OLR`, top-of-atmosphere outgoing longwave flux, W m-2, `[ncell]`.
    pub olr: Option<&'a [f32]>,
}

/// Group D output of one frame.
#[derive(Clone, Debug)]
pub struct Output {
    pub nx: usize,
    pub ny: usize,
    /// `[N_OUT][ncell]`, f32, NaN = missing.  Omitted fields are all NaN.
    pub planes: Vec<f32>,
    pub omitted: Vec<(&'static str, String)>,
    /// Where the cloud fraction came from (manifest text), when used.
    pub cloud_fraction_source: Option<&'static str>,
    /// Which branches `simulated_ir` used (manifest text), when written.
    pub ir_source: Option<&'static str>,
    pub device: String,
}

impl Output {
    #[must_use]
    pub fn plane(&self, id: &str) -> Option<&[f32]> {
        let i = plane_index(id)?;
        let n = self.nx * self.ny;
        Some(&self.planes[i * n..(i + 1) * n])
    }
}

/// Flags of one run, and the manifest text for the cloud-fraction source.
#[must_use]
pub fn flags(inp: &StateInputs, c: &Carriers<'_>, opt: &Options) -> (i32, Option<&'static str>) {
    let mut f = 0;
    let mut set = |on: bool, bit: i32| {
        if on {
            f |= bit;
        }
    };
    set(c.refl.is_some(), DF_REFL);
    set(inp.has(MassInput::QRain), DF_QR);
    set(inp.has(MassInput::QSnow), DF_QS);
    set(inp.has(MassInput::QGraupel), DF_QG);
    set(inp.has(MassInput::QCloud), DF_QC);
    set(inp.has(MassInput::QIce), DF_QI);
    set(c.aerosol_km.is_some(), DF_AER);
    set(c.hgt.is_some(), DF_HGT);
    set(opt.refl_interp == ReflInterp::LinearDbz, DF_INTERP_DBZ);
    set(c.olr.is_some(), DF_OLR);
    let water = inp.has(MassInput::QCloud) && inp.has(MassInput::QIce);
    let use_model = c.cldfra.is_some() && opt.cloud_fraction != CloudFractionSource::Diagnosed;
    let use_diag = !use_model && water && opt.cloud_fraction != CloudFractionSource::ModelOnly;
    let mut source = None;
    if use_model && water {
        f |= DF_CF_MODEL;
        source = Some("CLDFRA (the model's radiation cloud fraction)");
    } else if use_diag {
        f |= DF_CF_DIAG;
        source = Some(match opt.diagnosed {
            DiagnosedCloud::XuRandall1996 => "diagnosed: Xu and Randall (1996), J. Atmos. Sci. 53, 3084 (history has no CLDFRA)",
            DiagnosedCloud::CondensatePresence => {
                f |= DF_CF_BINARY;
                "diagnosed: condensate presence QCLOUD+QICE+QSNOW > 1e-6 kg/kg (history has no CLDFRA)"
            }
        });
    }
    (f, source)
}

/// Fields the carriers cannot support, with the reason.  Absence is never
/// read as zero (spec 3.1).
#[must_use]
pub fn omissions(inp: &StateInputs, c: &Carriers<'_>, opt: &Options) -> Vec<(&'static str, String)> {
    let (f, _) = flags(inp, c, opt);
    let mut v = Vec::new();
    if f & (DF_REFL | DF_QR) == 0 {
        for id in ["composite_reflectivity", "reflectivity_1km"] {
            v.push((id, "carrier absent: REFL_10CM (or QRAIN for the RIP-note volume)".to_owned()));
        }
    } else if f & DF_HGT == 0 {
        v.push(("reflectivity_1km", "carrier absent: HGT".to_owned()));
    }
    if f & (DF_CF_MODEL | DF_CF_DIAG) == 0 {
        let why = if f & (DF_QC | DF_QI) != (DF_QC | DF_QI) {
            "carrier absent: QCLOUD and QICE (condensate)".to_owned()
        } else {
            "carrier absent: CLDFRA (cloud-fraction source ModelOnly)".to_owned()
        };
        for id in ["low_cloud", "mid_cloud", "high_cloud", "cloud_base", "cloud_top", "cloud_ceiling"] {
            v.push((id, why.clone()));
        }
    }
    let mut vis_missing = Vec::new();
    for (bit, name) in [(DF_QC, "QCLOUD"), (DF_QR, "QRAIN"), (DF_QI, "QICE")] {
        if f & bit == 0 {
            vis_missing.push(name);
        }
    }
    if !vis_missing.is_empty() {
        v.push(("visibility", format!("carrier absent: {}", vis_missing.join(", "))));
    } else if c.aerosol_km.is_none() && !opt.allow_missing_aerosol {
        v.push((
            "visibility",
            "aerosol extinction absent (AEXTC55, AOD3D_SMOKE or EXTCOF55) and the clear-air baseline is switched off (VIS_WITHOUT_AEROSOL); absent smoke is not zero smoke".to_owned(),
        ));
    }
    if f & DF_OLR == 0 {
        v.push(("simulated_ir", "carrier absent: OLR (top-of-atmosphere outgoing longwave); absence is not a flux".to_owned()));
    }
    v
}

/// Manifest text for the branches `simulated_ir` uses, from the run flags.
#[must_use]
pub fn ir_source(flags: i32) -> Option<&'static str> {
    if flags & DF_OLR == 0 {
        None
    } else if flags & DF_QC != 0 && flags & DF_QI != 0 {
        Some("emission level at unit optical depth where opaque (QCLOUD, QICE, QSNOW), OLR relation elsewhere")
    } else {
        Some("OLR relation only (QCLOUD or QICE absent, so no opaque-column test)")
    }
}

/// Check carrier sizes against the state.
pub fn validate(state: &ColumnState, inp: &StateInputs, c: &Carriers<'_>) -> Result<(), String> {
    if state.shape != inp.shape {
        return Err("state and carriers have different shapes".into());
    }
    let n = state.shape.ncell();
    let v = state.shape.mass_volume();
    for (name, got, want) in [
        ("REFL_10CM", c.refl.map(<[f32]>::len), v),
        ("CLDFRA", c.cldfra.map(<[f32]>::len), v),
        ("aerosol extinction", c.aerosol_km.map(<[f32]>::len), n),
        ("HGT", c.hgt.map(<[f32]>::len), n),
        ("OLR", c.olr.map(<[f32]>::len), n),
    ] {
        if let Some(got) = got {
            if got != want {
                return Err(format!("{name} has {got} words, expected {want}"));
            }
        }
    }
    Ok(())
}

pub(crate) fn blank_omitted(planes: &mut [f32], n: usize, omitted: &[(&'static str, String)]) {
    for (id, _) in omitted {
        if let Some(i) = plane_index(id) {
            planes[i * n..(i + 1) * n].fill(crate::math::NAN32);
        }
    }
}

/// Group D on the CPU.
pub fn compute_cpu(state: &ColumnState, inp: &StateInputs, c: &Carriers<'_>, opt: &Options) -> Result<Output, String> {
    validate(state, inp, c)?;
    let shape = state.shape;
    let n = shape.ncell();
    let (flags, source) = flags(inp, c, opt);
    let mut planes = vec![0.0f32; N_OUT * n];
    let cols: Vec<([f32; N_COLUMN], f32)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let col = Col {
                st: &state.mass,
                in3: &inp.mass,
                refl: c.refl,
                cf: c.cldfra,
                vol: shape.mass_volume(),
                n,
                i,
                flags,
            };
            let o = column::column(&col, shape.nz, c.aerosol_km, c.hgt, opt.contrast);
            let ir = column::ir_column(&col, &state.interface, c.olr, shape.nz);
            (o.map(crate::math::to_f32), crate::math::to_f32(ir))
        })
        .collect();
    for (i, (o, ir)) in cols.iter().enumerate() {
        for f in 0..N_COLUMN {
            planes[f * n + i] = o[f];
        }
        planes[O_IR * n + i] = *ir;
    }
    let omitted = omissions(inp, c, opt);
    blank_omitted(&mut planes, n, &omitted);
    let cf_used = FIELDS[O_LOW..=O_CEIL].iter().any(|f| !omitted.iter().any(|(id, _)| *id == f.id));
    Ok(Output {
        nx: shape.nx,
        ny: shape.ny,
        planes,
        omitted,
        cloud_fraction_source: if cf_used { source } else { None },
        ir_source: ir_source(flags),
        device: "cpu".into(),
    })
}

/// Group D on the requested device.  `Auto` falls back to the CPU when no
/// GPU is given or the device path fails; `Gpu` fails instead.
#[cfg(any(windows, target_os = "linux"))]
pub fn compute(
    state: &ColumnState,
    inp: &StateInputs,
    c: &Carriers<'_>,
    opt: &Options,
    gpu: Option<&crate::gpu::GpuPost>,
    device: crate::PostDevice,
) -> Result<Output, String> {
    use crate::PostDevice;
    match (device, gpu) {
        (PostDevice::Cpu, _) => compute_cpu(state, inp, c, opt),
        (PostDevice::Gpu, Some(g)) => cuda::compute_host(g, state, inp, c, opt),
        (PostDevice::Gpu, None) => Err("PostDevice::Gpu needs an opened GPU".to_owned()),
        (PostDevice::Auto, Some(g)) => cuda::compute_host(g, state, inp, c, opt).or_else(|_| compute_cpu(state, inp, c, opt)),
        (PostDevice::Auto, None) => compute_cpu(state, inp, c, opt),
    }
}

/// The state slots group D reads (documentation and tests).
pub const STATE_SLOTS: [MassState; 4] = [MassState::PFull, MassState::Tk, MassState::R, MassState::ZMass];
