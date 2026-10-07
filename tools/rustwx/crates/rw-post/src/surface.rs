//! Group A shelter and surface fields (specification section 4.2).
//!
//! [`surface_cell`] is the CPU twin of `woof_post_surface_v1`.  Inputs are
//! packed 2D planes ([`SurfaceInput`]), outputs are planes in the order of
//! [`SurfaceField`].  A carrier that is absent leaves its products missing
//! (NaN) and the field catalog says why; absence is never zero.

use rayon::prelude::*;

use crate::consts::*;
use crate::solar::{SolarFrame, cosz};
use crate::thermo::{dewpoint, expf, floor0, powpos, rh, vapour_pressure};

/// Slots of the packed 2D input (`IN2_*` in the kernel).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum SurfaceInput {
    Hgt = 0,
    Psfc = 1,
    T2 = 2,
    Th2 = 3,
    Q2 = 4,
    Tsk = 5,
    Snow = 6,
    SnowH = 7,
    SnowC = 8,
    Hfx = 9,
    Lh = 10,
    Qfx = 11,
    GrdFlx = 12,
    Ust = 13,
    Glw = 14,
    SwDown = 15,
    VegFra = 16,
    IvgTyp = 17,
    SeaIce = 18,
    Znt = 19,
    ShdMin = 20,
    ShdMax = 21,
    Lai = 22,
    XLat = 23,
    XLong = 24,
    /// Snow-depletion threshold of the cell's vegetation class, m of water
    /// (0: water or ice class; NaN: class without a threshold).
    Snup = 25,
    /// Virtual temperature of the lowest mass level, from the state.
    TvLowest = 26,
    /// Solar zenith cosine supplied with the inputs (used as given instead
    /// of the computed one).
    Cosz = 27,
}

impl SurfaceInput {
    pub const COUNT: usize = 28;
    /// History carriers read straight into a slot.
    pub const CARRIERS: [(SurfaceInput, &'static str); 25] = [
        (Self::Hgt, "HGT"),
        (Self::Psfc, "PSFC"),
        (Self::T2, "T2"),
        (Self::Th2, "TH2"),
        (Self::Q2, "Q2"),
        (Self::Tsk, "TSK"),
        (Self::Snow, "SNOW"),
        (Self::SnowH, "SNOWH"),
        (Self::SnowC, "SNOWC"),
        (Self::Hfx, "HFX"),
        (Self::Lh, "LH"),
        (Self::Qfx, "QFX"),
        (Self::GrdFlx, "GRDFLX"),
        (Self::Ust, "UST"),
        (Self::Glw, "GLW"),
        (Self::SwDown, "SWDOWN"),
        (Self::VegFra, "VEGFRA"),
        (Self::IvgTyp, "IVGTYP"),
        (Self::SeaIce, "SEAICE"),
        (Self::Znt, "ZNT"),
        (Self::ShdMin, "SHDMIN"),
        (Self::ShdMax, "SHDMAX"),
        (Self::Lai, "LAI"),
        (Self::XLat, "XLAT"),
        (Self::XLong, "XLONG"),
    ];
}

/// Output planes (`OUT_*` in the kernel).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum SurfaceField {
    Terrain = 0,
    SurfacePressure = 1,
    T2 = 2,
    Td2 = 3,
    Rh2 = 4,
    PotentialTemperature2 = 5,
    SpecificHumidity2 = 6,
    SkinTemperature = 7,
    SnowWater = 8,
    SnowDepth = 9,
    SnowCover = 10,
    SensibleHeatFlux = 11,
    LatentHeatFlux = 12,
    GroundHeatFlux = 13,
    FrictionVelocity = 14,
    DownwardLongwave = 15,
    DownwardShortwave = 16,
    VegetationFraction = 17,
    VegetationType = 18,
    SeaIceFraction = 19,
    RoughnessLength = 20,
    VegetationMin = 21,
    VegetationMax = 22,
    LeafAreaIndex = 23,
    /// Solar zenith cosine at valid time (shared helper, not a GRIB row).
    Cosz = 24,
    /// Shelter pressure (shared helper, not a GRIB row).
    ShelterPressure = 25,
}

/// GRIB2 identity and method of one Group A product.
#[derive(Clone, Copy, Debug)]
pub struct FieldSpec {
    pub field: SurfaceField,
    pub id: &'static str,
    pub discipline: u8,
    pub category: u8,
    pub number: u8,
    pub level_type: u8,
    pub level_value: f64,
    pub units: &'static str,
    /// Carriers that must all be present.
    pub needs: &'static [&'static str],
    /// Manifest method string with its public source.
    pub method: &'static str,
}

impl SurfaceField {
    pub const COUNT: usize = 26;

    /// The 24 GRIB products of specification section 4.2, in its order.
    pub const PRODUCTS: [FieldSpec; 24] = [
        FieldSpec { field: Self::Terrain, id: "terrain", discipline: 0, category: 3, number: 5, level_type: 1, level_value: 0.0, units: "gpm", needs: &["HGT"], method: "HGT, WRF geometric metres reported as gpm (D2)" },
        FieldSpec { field: Self::SurfacePressure, id: "surface_pressure", discipline: 0, category: 3, number: 0, level_type: 1, level_value: 0.0, units: "Pa", needs: &["PSFC"], method: "PSFC (D3)" },
        FieldSpec { field: Self::T2, id: "t2", discipline: 0, category: 0, number: 0, level_type: 103, level_value: 2.0, units: "K", needs: &["T2|TH2"], method: "surface scheme T2; else TH2 (p2/p0)^(Rd/cp), p2 hypsometric from PSFC (D4, AMS Glossary)" },
        FieldSpec { field: Self::Td2, id: "td2", discipline: 0, category: 0, number: 6, level_type: 103, level_value: 2.0, units: "K", needs: &["Q2", "PSFC"], method: "inverse AERK Magnus dewpoint of Q2 at shelter pressure, capped at T2 (JAM 35 (1996) 601, D4, D5)" },
        FieldSpec { field: Self::Rh2, id: "rh2", discipline: 0, category: 1, number: 1, level_type: 103, level_value: 2.0, units: "%", needs: &["Q2", "PSFC", "T2|TH2"], method: "RH over water, AERK Magnus, clipped [0,100] (JAM 35 (1996) 601, WMO-No. 8, D5)" },
        FieldSpec { field: Self::PotentialTemperature2, id: "potential_temperature2", discipline: 0, category: 0, number: 2, level_type: 103, level_value: 2.0, units: "K", needs: &["TH2"], method: "TH2" },
        FieldSpec { field: Self::SpecificHumidity2, id: "specific_humidity2", discipline: 0, category: 1, number: 0, level_type: 103, level_value: 2.0, units: "kg kg-1", needs: &["Q2"], method: "Q2/(1+Q2), Q2 floored at 0" },
        FieldSpec { field: Self::SkinTemperature, id: "skin_temperature", discipline: 0, category: 0, number: 0, level_type: 1, level_value: 0.0, units: "K", needs: &["TSK"], method: "TSK" },
        FieldSpec { field: Self::SnowWater, id: "snow_water", discipline: 0, category: 1, number: 13, level_type: 1, level_value: 0.0, units: "kg m-2", needs: &["SNOW"], method: "SNOW, floored at 0" },
        FieldSpec { field: Self::SnowDepth, id: "snow_depth", discipline: 0, category: 1, number: 11, level_type: 1, level_value: 0.0, units: "m", needs: &["SNOWH"], method: "SNOWH, floored at 0" },
        FieldSpec { field: Self::SnowCover, id: "snow_cover", discipline: 0, category: 1, number: 42, level_type: 1, level_value: 0.0, units: "%", needs: &["SNOWC|SNOW+IVGTYP"], method: "SNOWC x 100; else Noah depletion curve of SNOW and the VEGPARM.TBL SNUP threshold (JGR 108 (2003) 8851, D26)" },
        FieldSpec { field: Self::SensibleHeatFlux, id: "sensible_heat_flux", discipline: 0, category: 0, number: 11, level_type: 1, level_value: 0.0, units: "W m-2", needs: &["HFX"], method: "HFX, upward positive" },
        FieldSpec { field: Self::LatentHeatFlux, id: "latent_heat_flux", discipline: 0, category: 0, number: 10, level_type: 1, level_value: 0.0, units: "W m-2", needs: &["LH|QFX"], method: "LH; RUC land model: QFX x 2.501e6 (D28), upward positive" },
        FieldSpec { field: Self::GroundHeatFlux, id: "ground_heat_flux", discipline: 2, category: 0, number: 10, level_type: 1, level_value: 0.0, units: "W m-2", needs: &["GRDFLX"], method: "GRDFLX" },
        FieldSpec { field: Self::FrictionVelocity, id: "friction_velocity", discipline: 0, category: 2, number: 30, level_type: 1, level_value: 0.0, units: "m s-1", needs: &["UST"], method: "UST" },
        FieldSpec { field: Self::DownwardLongwave, id: "downward_longwave", discipline: 0, category: 5, number: 3, level_type: 1, level_value: 0.0, units: "W m-2", needs: &["GLW"], method: "GLW" },
        FieldSpec { field: Self::DownwardShortwave, id: "downward_shortwave", discipline: 0, category: 4, number: 7, level_type: 1, level_value: 0.0, units: "W m-2", needs: &["SWDOWN"], method: "SWDOWN, 0 where the Solar Energy 40 (1988) 227 solar zenith cosine <= 0 (D27)" },
        FieldSpec { field: Self::VegetationFraction, id: "vegetation_fraction", discipline: 2, category: 0, number: 4, level_type: 1, level_value: 0.0, units: "%", needs: &["VEGFRA"], method: "VEGFRA in percent" },
        FieldSpec { field: Self::VegetationType, id: "vegetation_type", discipline: 2, category: 0, number: 198, level_type: 1, level_value: 0.0, units: "index", needs: &["IVGTYP"], method: "IVGTYP" },
        FieldSpec { field: Self::SeaIceFraction, id: "sea_ice_fraction", discipline: 10, category: 2, number: 0, level_type: 1, level_value: 0.0, units: "proportion", needs: &["SEAICE"], method: "SEAICE, not thresholded" },
        FieldSpec { field: Self::RoughnessLength, id: "roughness_length", discipline: 2, category: 0, number: 1, level_type: 1, level_value: 0.0, units: "m", needs: &["ZNT"], method: "ZNT" },
        FieldSpec { field: Self::VegetationMin, id: "vegetation_min", discipline: 2, category: 0, number: 231, level_type: 1, level_value: 0.0, units: "%", needs: &["SHDMIN"], method: "SHDMIN in percent" },
        FieldSpec { field: Self::VegetationMax, id: "vegetation_max", discipline: 2, category: 0, number: 232, level_type: 1, level_value: 0.0, units: "%", needs: &["SHDMAX"], method: "SHDMAX in percent" },
        FieldSpec { field: Self::LeafAreaIndex, id: "leaf_area_index", discipline: 0, category: 7, number: 198, level_type: 1, level_value: 0.0, units: "1", needs: &["LAI"], method: "LAI" },
    ];
}

/// Scalar options of the surface computation (`WpSurfaceOptions`).
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct SurfaceOptions {
    pub sin_dec: f32,
    pub cos_dec: f32,
    pub ha0_deg: f32,
    /// 1 when VEGFRA is already percent, 100 when it is a fraction.
    pub vegfra_scale: f32,
    /// The same for SHDMIN/SHDMAX.
    pub shd_scale: f32,
    /// 1: RUC land model, latent heat = QFX x Lv (D28).
    pub latent_from_qfx: u32,
    /// 1: no SNOWC carrier, derive the cover from SNOW (D26).
    pub snow_cover_derived: u32,
    /// Noah snow-depletion shape parameter.
    pub snow_salp: f32,
}

#[cfg(any(windows, target_os = "linux"))]
unsafe impl cudarc::driver::DeviceRepr for SurfaceOptions {}

impl SurfaceOptions {
    pub fn new(solar: &SolarFrame) -> Self {
        Self {
            sin_dec: solar.sin_dec,
            cos_dec: solar.cos_dec,
            ha0_deg: solar.ha0_deg,
            vegfra_scale: 1.0,
            shd_scale: 1.0,
            latent_from_qfx: 0,
            snow_cover_derived: 0,
            snow_salp: crate::snow::NOAH_SALP,
        }
    }
}

/// Packed 2D inputs.
#[derive(Clone, Debug)]
pub struct SurfaceInputs {
    pub ncell: usize,
    /// `SurfaceInput::COUNT` planes of `ncell`; absent planes hold zeros.
    pub planes: Vec<f32>,
    pub present: u32,
}

impl SurfaceInputs {
    pub fn new(ncell: usize) -> Self {
        Self { ncell, planes: vec![0.0; SurfaceInput::COUNT * ncell], present: 0 }
    }
    pub fn has(&self, s: SurfaceInput) -> bool {
        (self.present >> s as u32) & 1 == 1
    }
    pub fn set(&mut self, s: SurfaceInput, values: &[f32]) {
        assert_eq!(values.len(), self.ncell, "plane size");
        let n = self.ncell;
        self.planes[s as usize * n..(s as usize + 1) * n].copy_from_slice(values);
        self.present |= 1 << s as u32;
    }
    pub fn plane(&self, s: SurfaceInput) -> &[f32] {
        let n = self.ncell;
        &self.planes[s as usize * n..(s as usize + 1) * n]
    }
}

/// One cell of every surface output: the CPU twin of `woof_post_surface_v1`.
pub fn surface_cell(inp: &SurfaceInputs, opt: &SurfaceOptions, c: usize) -> [f32; SurfaceField::COUNT] {
    use SurfaceField as F;
    use SurfaceInput as I;
    let n = inp.ncell;
    let get = |s: I| inp.planes[s as usize * n + c];
    let has = |s: I| inp.has(s);
    let nan = f32::NAN;
    let mut o = [nan; SurfaceField::COUNT];

    o[F::Terrain as usize] = if has(I::Hgt) { get(I::Hgt) } else { nan };
    let psfc = if has(I::Psfc) { get(I::Psfc) } else { nan };
    o[F::SurfacePressure as usize] = psfc;

    let mut p2 = nan;
    if has(I::TvLowest) {
        p2 = psfc * expf(-(G * 2.0) / (RD * get(I::TvLowest)));
    }
    o[F::ShelterPressure as usize] = p2;

    let mut t2 = nan;
    if has(I::T2) {
        t2 = get(I::T2);
    } else if has(I::Th2) {
        t2 = get(I::Th2) * powpos(p2 / P0, KAPPA);
    }
    o[F::T2 as usize] = t2;

    let mut q2 = nan;
    if has(I::Q2) {
        let r2 = floor0(get(I::Q2));
        q2 = r2 / (1.0 + r2);
    }
    o[F::SpecificHumidity2 as usize] = q2;
    let e2 = vapour_pressure(q2, p2);
    let mut td2 = dewpoint(e2);
    if td2 > t2 {
        td2 = t2;
    }
    o[F::Td2 as usize] = td2;
    o[F::Rh2 as usize] = rh(e2, t2);
    o[F::PotentialTemperature2 as usize] = if has(I::Th2) { get(I::Th2) } else { nan };
    o[F::SkinTemperature as usize] = if has(I::Tsk) { get(I::Tsk) } else { nan };
    o[F::SnowWater as usize] = if has(I::Snow) { floor0(get(I::Snow)) } else { nan };
    o[F::SnowDepth as usize] = if has(I::SnowH) { floor0(get(I::SnowH)) } else { nan };

    let mut snowc = nan;
    if opt.snow_cover_derived == 0 {
        if has(I::SnowC) {
            snowc = get(I::SnowC) * 100.0;
        }
    } else if has(I::Snow) && has(I::Snup) {
        let snup = get(I::Snup);
        let swe_m = floor0(get(I::Snow)) / 1000.0;
        if snup.is_nan() {
            snowc = nan;
        } else if snup == 0.0 {
            snowc = 0.0;
        } else if swe_m >= snup {
            snowc = 100.0;
        } else {
            let rs = swe_m / snup;
            let frac = 1.0 - (expf(-opt.snow_salp * rs) - rs * expf(-opt.snow_salp));
            snowc = 100.0 * frac;
        }
    }
    o[F::SnowCover as usize] = snowc;

    o[F::SensibleHeatFlux as usize] = if has(I::Hfx) { get(I::Hfx) } else { nan };
    let mut lh = nan;
    if opt.latent_from_qfx != 0 {
        if has(I::Qfx) {
            lh = get(I::Qfx) * LV;
        }
    } else if has(I::Lh) {
        lh = get(I::Lh);
    }
    o[F::LatentHeatFlux as usize] = lh;
    o[F::GroundHeatFlux as usize] = if has(I::GrdFlx) { get(I::GrdFlx) } else { nan };
    o[F::FrictionVelocity as usize] = if has(I::Ust) { get(I::Ust) } else { nan };
    o[F::DownwardLongwave as usize] = if has(I::Glw) { get(I::Glw) } else { nan };

    let mut cz = nan;
    if has(I::Cosz) {
        cz = get(I::Cosz);
    } else if has(I::XLat) && has(I::XLong) {
        cz = cosz(get(I::XLat), get(I::XLong), opt.sin_dec, opt.cos_dec, opt.ha0_deg);
    }
    o[F::Cosz as usize] = cz;
    let mut sw = nan;
    if has(I::SwDown) {
        sw = get(I::SwDown);
        if cz <= 0.0 {
            sw = 0.0;
        }
    }
    o[F::DownwardShortwave as usize] = sw;
    o[F::VegetationFraction as usize] =
        if has(I::VegFra) { get(I::VegFra) * opt.vegfra_scale } else { nan };
    o[F::VegetationType as usize] = if has(I::IvgTyp) { get(I::IvgTyp) } else { nan };
    o[F::SeaIceFraction as usize] = if has(I::SeaIce) { get(I::SeaIce) } else { nan };
    o[F::RoughnessLength as usize] = if has(I::Znt) { get(I::Znt) } else { nan };
    o[F::VegetationMin as usize] = if has(I::ShdMin) { get(I::ShdMin) * opt.shd_scale } else { nan };
    o[F::VegetationMax as usize] = if has(I::ShdMax) { get(I::ShdMax) * opt.shd_scale } else { nan };
    o[F::LeafAreaIndex as usize] = if has(I::Lai) { get(I::Lai) } else { nan };
    o
}

/// All surface planes on the CPU, `SurfaceField::COUNT * ncell`.
pub fn surface_cpu(inp: &SurfaceInputs, opt: &SurfaceOptions) -> Vec<f32> {
    let n = inp.ncell;
    let cells: Vec<[f32; SurfaceField::COUNT]> =
        (0..n).into_par_iter().map(|c| surface_cell(inp, opt, c)).collect();
    let mut out = vec![0.0f32; SurfaceField::COUNT * n];
    for (c, v) in cells.iter().enumerate() {
        for (f, x) in v.iter().enumerate() {
            out[f * n + c] = *x;
        }
    }
    out
}

/// Whether a product can be computed from the carriers present, and if not
/// the reason recorded in the manifest.
pub fn availability(spec: &FieldSpec, inp: &SurfaceInputs, opt: &SurfaceOptions) -> Result<(), String> {
    use SurfaceInput as I;
    let missing = |what: &str| Err(format!("carrier {what} absent"));
    match spec.field {
        SurfaceField::T2 => {
            if inp.has(I::T2) {
                return Ok(());
            }
            if !inp.has(I::Th2) {
                return missing("T2 and TH2");
            }
            if !inp.has(I::TvLowest) || !inp.has(I::Psfc) {
                return missing("PSFC or the 3D state for shelter pressure");
            }
            Ok(())
        }
        SurfaceField::Td2 | SurfaceField::Rh2 => {
            if !inp.has(I::Q2) {
                return missing("Q2");
            }
            if !inp.has(I::Psfc) || !inp.has(I::TvLowest) {
                return missing("PSFC or the 3D state for shelter pressure");
            }
            if !inp.has(I::T2) && !inp.has(I::Th2) {
                return missing("T2 and TH2");
            }
            Ok(())
        }
        SurfaceField::SnowCover => {
            if opt.snow_cover_derived == 0 {
                if inp.has(I::SnowC) { Ok(()) } else { missing("SNOWC") }
            } else if inp.has(I::Snow) && inp.has(I::Snup) {
                Ok(())
            } else {
                missing("SNOWC, and SNOW or a known land-use table for the derived cover")
            }
        }
        SurfaceField::LatentHeatFlux => {
            if opt.latent_from_qfx != 0 {
                if inp.has(I::Qfx) { Ok(()) } else { missing("QFX (RUC land model)") }
            } else if inp.has(I::Lh) {
                Ok(())
            } else {
                missing("LH")
            }
        }
        SurfaceField::DownwardShortwave => {
            if !inp.has(I::SwDown) {
                return missing("SWDOWN");
            }
            if !inp.has(I::Cosz) && (!inp.has(I::XLat) || !inp.has(I::XLong)) {
                return missing("XLAT/XLONG for the night guard");
            }
            Ok(())
        }
        _ => {
            for need in spec.needs {
                let slot = SurfaceInput::CARRIERS.iter().find(|(_, n)| n == need).map(|(s, _)| *s);
                match slot {
                    Some(s) if inp.has(s) => {}
                    _ => return missing(need),
                }
            }
            Ok(())
        }
    }
}
