//! The decoded polar volume: sites, sweeps, moments and their geometry.
//!
//! The shape mirrors ODIM's own nesting -- a volume holds sweeps
//! (`/datasetN`), a sweep holds moments (`/datasetN/dataM`) -- which is also
//! the shape `wx_radar::level2::Level2File` uses for NEXRAD, and the opposite
//! of `wx_field::RadialField`, which is one product across all sweeps. Keeping
//! the file's own nesting means the decoder never has to transpose, and a
//! sweep's geometry is stated once beside the moments it applies to.
//!
//! Units are named in the field, per the newer workspace convention
//! (`_deg`, `_m`, `_ms`): the vendor radar crates predate it and pay for it.
//!
//! The bulk arrays (`values`, `censor`) are `#[serde(skip)]`, so serialising a
//! volume yields its metadata and its census -- the provenance record -- and
//! not two hundred megabytes of gates.

use std::fmt;

use chrono::{DateTime, NaiveDate, NaiveDateTime, NaiveTime, Utc};
use serde::{Deserialize, Serialize};

use crate::censor::{self, Census};
use crate::quantity::QuantityKind;

/// Where the antenna is. Every field is required by ODIM `/where`.
///
/// `height_m` is the one the frozen MeteoGate site table cannot supply -- the
/// locations feed publishes `null` for all 136 radars -- and it is the reason
/// a polar volume has to be read before a European site is assimilable at all.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Site {
    pub latitude_deg: f64,
    pub longitude_deg: f64,
    /// Antenna height above mean sea level, metres.
    pub height_m: f64,
}

/// `/what/source`, split on its `KEY:value` pairs.
///
/// ODIM writes this as a comma-separated list whose membership varies by
/// country: Finland ships `WIGOS`, `WMO`, `RAD`, `PLC` and `NOD`; Romania
/// ships `NOD` alone. The raw string is kept so nothing is lost to a parser
/// that did not expect a key.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Source {
    pub raw: String,
    /// `NOD:` -- the OPERA node identifier, e.g. `deboo`. The join key to the
    /// MeteoGate site table.
    pub nod: Option<String>,
    /// `WMO:` -- the five-digit WMO station number.
    pub wmo: Option<String>,
    /// `WIGOS:` -- the WIGOS station identifier.
    pub wigos: Option<String>,
    /// `PLC:` -- the place name.
    pub place: Option<String>,
    /// `RAD:` -- the national radar identifier.
    pub rad: Option<String>,
}

impl Source {
    /// Split `WMO:10132,NOD:deboo` into its parts.
    pub fn parse(raw: &str) -> Self {
        let mut source = Source {
            raw: raw.to_string(),
            ..Source::default()
        };
        for field in raw.split(',') {
            let Some((key, value)) = field.split_once(':') else {
                continue;
            };
            let value = value.trim().to_string();
            if value.is_empty() {
                continue;
            }
            match key.trim().to_ascii_uppercase().as_str() {
                "NOD" => source.nod = Some(value),
                "WMO" => source.wmo = Some(value),
                "WIGOS" => source.wigos = Some(value),
                "PLC" => source.place = Some(value),
                "RAD" => source.rad = Some(value),
                _ => {}
            }
        }
        source
    }

    /// The most specific identifier available, for labelling output.
    pub fn label(&self) -> String {
        self.nod
            .clone()
            .or_else(|| self.wigos.clone())
            .or_else(|| self.wmo.clone())
            .unwrap_or_else(|| self.raw.clone())
    }
}

/// Radar hardware and processing notes from `/how`, all optional in ODIM.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SystemNotes {
    /// `/how/wavelength`, centimetres. Needed to derive a Nyquist interval
    /// when one is not declared, and the thing that reveals that three
    /// European radars are S-band rather than C-band.
    pub wavelength_cm: Option<f64>,
    pub beamwidth_deg: Option<f64>,
    pub system: Option<String>,
    pub software: Option<String>,
    pub sw_version: Option<String>,
}

/// How a sweep's per-ray azimuths were arrived at.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AzimuthSource {
    /// Circular mean of `/datasetN/how/startazA` and `stopazA`: the azimuth
    /// the antenna actually swept through, per ray.
    MeasuredStartStopMean,
    /// ODIM section 5.1's nominal geometry: the first ray points north and
    /// the sweep proceeds clockwise, so ray `i` is centred on
    /// `astart + (i + 1/2) * 360/nrays`.
    NominalFromRayCount { astart_deg: f64 },
}

impl fmt::Display for AzimuthSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AzimuthSource::MeasuredStartStopMean => write!(f, "measured (startazA/stopazA mean)"),
            AzimuthSource::NominalFromRayCount { astart_deg } => {
                write!(f, "nominal (astart={astart_deg}, clockwise from north)")
            }
        }
    }
}

/// Where a sweep's Nyquist interval came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NyquistSource {
    /// `/datasetN/how/NI`, stated by the writer. Always preferred: for a
    /// dual-PRF scheme it is the *extended* interval, which cannot be
    /// recovered from one PRF.
    Declared,
    /// Derived as `wavelength * PRF / 4` from a single-PRF sweep.
    DerivedSinglePrf,
    /// Not available. The velocity moments of this sweep cannot be unfolded.
    Unavailable,
}

/// The unambiguous velocity interval of a sweep, and its provenance.
///
/// This is the handoff to the dealiaser, so it is a struct rather than an
/// `Option<f64>`: a region-global unfolding engine needs to know not only the
/// number but whether it was declared or inferred, and a dual-PRF sweep whose
/// `NI` is missing must be refused rather than given a single-PRF estimate
/// that is a factor of three too small.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Nyquist {
    /// Unambiguous velocity, m s-1. `None` when it could not be established.
    pub interval_ms: Option<f64>,
    pub source: NyquistSource,
    pub high_prf_hz: Option<f64>,
    pub low_prf_hz: Option<f64>,
    /// True when the sweep declares two different PRFs, i.e. the interval is
    /// extended by a dual-PRF scheme and is not `wavelength * PRF / 4`.
    pub dual_prf: bool,
}

impl Nyquist {
    /// Establish the interval from what the sweep declares.
    ///
    /// Order matters. A declared `NI` wins outright. Only a sweep that is
    /// unambiguously single-PRF gets a derived value, because the dual-PRF
    /// extension factor depends on the PRF ratio and the vendor's unfolding
    /// scheme, and guessing it would hand the dealiaser a wrong ceiling --
    /// which does not fail loudly, it just produces wind.
    pub fn establish(
        declared_ni: Option<f64>,
        high_prf_hz: Option<f64>,
        low_prf_hz: Option<f64>,
        wavelength_cm: Option<f64>,
    ) -> Self {
        let dual_prf = match (high_prf_hz, low_prf_hz) {
            (Some(high), Some(low)) => high > 0.0 && low > 0.0 && (high - low).abs() > 1e-6,
            _ => false,
        };
        if let Some(ni) = declared_ni
            && ni.is_finite()
            && ni > 0.0
        {
            return Nyquist {
                interval_ms: Some(ni),
                source: NyquistSource::Declared,
                high_prf_hz,
                low_prf_hz,
                dual_prf,
            };
        }
        if !dual_prf
            && let Some(prf) = high_prf_hz.or(low_prf_hz)
            && let Some(wavelength_cm) = wavelength_cm
            && prf > 0.0
            && wavelength_cm > 0.0
        {
            // v_nyq = lambda * PRF / 4, lambda in metres.
            let interval = (wavelength_cm / 100.0) * prf / 4.0;
            return Nyquist {
                interval_ms: Some(interval),
                source: NyquistSource::DerivedSinglePrf,
                high_prf_hz,
                low_prf_hz,
                dual_prf,
            };
        }
        Nyquist {
            interval_ms: None,
            source: NyquistSource::Unavailable,
            high_prf_hz,
            low_prf_hz,
            dual_prf,
        }
    }

    /// Whether a velocity moment on this sweep can be handed to a dealiaser.
    pub fn is_usable(&self) -> bool {
        self.interval_ms.is_some_and(|v| v.is_finite() && v > 0.0)
    }
}

/// The `what` calibration of one moment, verbatim from the file.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Calibration {
    pub gain: f64,
    pub offset: f64,
    /// Raw storage value meaning "not measured".
    pub nodata: Option<f64>,
    /// Raw storage value meaning "looked, found nothing".
    pub undetect: Option<f64>,
    /// True when the file declared both sentinels as the same raw value, so
    /// the two states cannot be told apart. Every such gate is classified
    /// [`censor::SENTINEL_AMBIGUOUS`] rather than being assigned to either.
    pub sentinels_collide: bool,
}

impl Calibration {
    /// Apply `raw * gain + offset`.
    ///
    /// Computed and kept in `f64`. ODIM gain and offset are `f64` attributes
    /// and a 16-bit payload spans 65 536 steps, so narrowing to `f32` here
    /// would round the physical value before anyone had a chance to use it --
    /// the mistake `rw-glm`'s `granule.rs` documents against `rw-sat`.
    pub fn apply(&self, raw: f64) -> f64 {
        raw * self.gain + self.offset
    }
}

/// One decoded moment of one sweep.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Moment {
    /// ODIM quantity name, e.g. `DBZH`, `VRADH`.
    pub quantity: String,
    /// The ODIM group this came from, e.g. `/dataset3/data2`.
    pub path: String,
    pub unit: String,
    pub kind: QuantityKind,
    pub calibration: Calibration,
    pub nrays: usize,
    pub nbins: usize,
    /// Physical values, row-major `[ray][bin]`, `NaN` wherever the gate is
    /// not a measurement. Every `NaN` has a reason in [`Moment::censor`].
    #[serde(skip)]
    pub values: Vec<f64>,
    /// One [`censor`] code per gate, parallel to `values` and always the same
    /// length.
    #[serde(skip)]
    pub censor: Vec<u8>,
    pub census: Census,
}

impl Moment {
    /// The physical value at `[ray][bin]`, or `NaN`.
    pub fn value(&self, ray: usize, bin: usize) -> f64 {
        self.values[ray * self.nbins + bin]
    }

    /// The censor code at `[ray][bin]`.
    pub fn code(&self, ray: usize, bin: usize) -> u8 {
        self.censor[ray * self.nbins + bin]
    }

    /// Smallest and largest measured value, or `None` if nothing was measured.
    pub fn measured_range(&self) -> Option<(f64, f64)> {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        let mut any = false;
        for (value, code) in self.values.iter().zip(&self.censor) {
            if *code == censor::MEASURED {
                lo = lo.min(*value);
                hi = hi.max(*value);
                any = true;
            }
        }
        any.then_some((lo, hi))
    }
}

/// One sweep: a single elevation cut and every moment recorded on it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sweep {
    /// The `N` of `/datasetN`. Sweeps are ordered by this, not by elevation:
    /// a Dutch volume opens with its 90-degree birdbath cut and a reorder
    /// would silently renumber every sweep index a consumer quoted.
    pub index: usize,
    pub path: String,
    /// `/datasetN/where/elangle`, degrees above the horizon.
    pub elevation_deg: f64,
    pub nrays: usize,
    pub nbins: usize,
    /// `/datasetN/where/rscale`, metres between successive bins.
    pub range_scale_m: f64,
    /// Start of the first range bin, metres.
    ///
    /// ODIM states `rstart` in **kilometres**; it is converted here so that
    /// every length on this struct is metres.
    pub range_start_m: f64,
    /// `/datasetN/where/a1gate`: index of the ray radiated first. The rays are
    /// stored starting at north regardless, so this is the acquisition-order
    /// key, not a permutation that has been applied.
    pub a1gate: Option<usize>,
    pub start_time: Option<DateTime<Utc>>,
    pub end_time: Option<DateTime<Utc>>,
    pub nyquist: Nyquist,
    /// Per-ray azimuth of the ray centre, degrees clockwise from north, in
    /// `[0, 360)`.
    #[serde(skip)]
    pub azimuth_deg: Vec<f64>,
    pub azimuth_source: AzimuthSource,
    /// Per-ray elevation, when the file records the antenna's actual
    /// elevation per ray rather than only the nominal cut.
    #[serde(skip)]
    pub ray_elevation_deg: Option<Vec<f64>>,
    pub moments: Vec<Moment>,
}

impl Sweep {
    /// Range to the **centre** of bin `bin`, metres.
    ///
    /// ODIM defines `rstart` as the range of the *start* of the first bin, so
    /// the centre of bin `i` is `rstart + (i + 1/2) * rscale`. Py-ART reports
    /// `rstart * 1000` as `meters_to_center_of_first_gate`, which places every
    /// gate half a bin too close; at a 250 m bin that is 125 m, and it biases
    /// the beam-height of every assimilated gate in the same direction. The
    /// ODIM-defined geometry is implemented here and the divergence is
    /// deliberate.
    pub fn gate_centre_range_m(&self, bin: usize) -> f64 {
        self.range_start_m + (bin as f64 + 0.5) * self.range_scale_m
    }

    /// Range to the far edge of the last bin, metres.
    pub fn max_range_m(&self) -> f64 {
        self.range_start_m + self.nbins as f64 * self.range_scale_m
    }

    /// The moment with this ODIM quantity name, if the sweep carries it.
    pub fn moment(&self, quantity: &str) -> Option<&Moment> {
        self.moments.iter().find(|m| m.quantity == quantity)
    }

    /// The quantities this sweep carries, in file order.
    pub fn quantities(&self) -> Vec<&str> {
        self.moments.iter().map(|m| m.quantity.as_str()).collect()
    }
}

/// A decoded ODIM_H5 polar volume.
///
/// One file. A `PVOL` carries many sweeps; a `SCAN` carries one. Both are
/// read the same way, which is what lets a German single-sweep single-moment
/// file and a Dutch sixteen-sweep nine-moment file come out of the same call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolarVolume {
    /// `/what/object`: `PVOL` or `SCAN`.
    pub object: String,
    /// The root `Conventions` attribute, e.g. `ODIM_H5/V2_2`.
    pub conventions: Option<String>,
    /// `/what/version`, e.g. `H5rad 2.2`.
    pub version: Option<String>,
    pub source: Source,
    /// `/what/date` + `/what/time`, the volume's nominal time.
    pub nominal_time: Option<DateTime<Utc>>,
    pub site: Site,
    pub system: SystemNotes,
    pub sweeps: Vec<Sweep>,
}

impl PolarVolume {
    /// Every distinct quantity anywhere in the volume, sorted.
    pub fn quantities(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .sweeps
            .iter()
            .flat_map(|s| s.moments.iter().map(|m| m.quantity.clone()))
            .collect();
        names.sort();
        names.dedup();
        names
    }

    /// Sweeps carrying at least one radial-velocity moment.
    pub fn velocity_sweeps(&self) -> Vec<&Sweep> {
        self.sweeps
            .iter()
            .filter(|s| {
                s.moments
                    .iter()
                    .any(|m| m.kind == QuantityKind::RadialVelocity)
            })
            .collect()
    }

    /// Whether every sweep that carries a velocity moment also has a usable
    /// Nyquist interval.
    ///
    /// This is the one-line answer to "can this volume be dealiased", and it
    /// is false when even one velocity sweep is missing its ceiling.
    pub fn velocity_is_dealiasable(&self) -> bool {
        let sweeps = self.velocity_sweeps();
        !sweeps.is_empty() && sweeps.iter().all(|s| s.nyquist.is_usable())
    }
}

/// Parse ODIM's `YYYYMMDD` + `HHMMSS` pair into an instant.
///
/// ODIM times are UTC by definition, so no zone is inferred.
pub(crate) fn parse_odim_datetime(date: &str, time: &str) -> Option<DateTime<Utc>> {
    let date = NaiveDate::parse_from_str(date.trim(), "%Y%m%d").ok()?;
    let time = time.trim();
    // Some writers ship HHMM rather than HHMMSS.
    let parsed = NaiveTime::parse_from_str(time, "%H%M%S")
        .or_else(|_| NaiveTime::parse_from_str(time, "%H%M"))
        .ok()?;
    Some(NaiveDateTime::new(date, parsed).and_utc())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_source_string_splits_into_its_keys() {
        let source = Source::parse("WIGOS:0-20010-0-02933,WMO:02933,RAD:FI46,PLC:Korpo,NOD:fikor");
        assert_eq!(source.nod.as_deref(), Some("fikor"));
        assert_eq!(source.wmo.as_deref(), Some("02933"));
        assert_eq!(source.wigos.as_deref(), Some("0-20010-0-02933"));
        assert_eq!(source.place.as_deref(), Some("Korpo"));
        assert_eq!(source.rad.as_deref(), Some("FI46"));
        assert_eq!(source.label(), "fikor");
    }

    #[test]
    fn a_sparse_source_string_keeps_what_it_has() {
        // Romania ships NOD alone; the rest must stay None rather than empty.
        let source = Source::parse("NOD:robuc");
        assert_eq!(source.nod.as_deref(), Some("robuc"));
        assert_eq!(source.wmo, None);
        assert_eq!(source.label(), "robuc");
        assert_eq!(source.raw, "NOD:robuc");
    }

    #[test]
    fn a_declared_nyquist_always_wins_over_a_derivable_one() {
        // Bucharest: dual PRF 652/434.67 with a declared extended NI. The
        // single-PRF formula on the high PRF would give
        // 0.10221 * 652 / 4 = 16.66 m/s, half the truth.
        let nyquist = Nyquist::establish(Some(33.3205), Some(652.0), Some(434.67), Some(10.221));
        assert_eq!(nyquist.source, NyquistSource::Declared);
        assert_eq!(nyquist.interval_ms, Some(33.3205));
        assert!(nyquist.dual_prf);
        assert!(nyquist.is_usable());
    }

    #[test]
    fn a_dual_prf_sweep_without_a_declared_interval_is_refused_not_guessed() {
        let nyquist = Nyquist::establish(None, Some(652.0), Some(434.67), Some(10.221));
        assert_eq!(nyquist.source, NyquistSource::Unavailable);
        assert_eq!(nyquist.interval_ms, None);
        assert!(nyquist.dual_prf);
        assert!(!nyquist.is_usable());
    }

    #[test]
    fn a_single_prf_sweep_may_be_derived_from_wavelength_and_prf() {
        // Korpo: 570 Hz single PRF, C band. lambda*PRF/4 with lambda = 5.33 cm
        // gives 7.60 m/s, which is what the file's own NI says (7.595).
        let nyquist = Nyquist::establish(None, Some(570.0), Some(570.0), Some(5.33));
        assert_eq!(nyquist.source, NyquistSource::DerivedSinglePrf);
        assert!(!nyquist.dual_prf);
        let interval = nyquist.interval_ms.expect("derived");
        assert!((interval - 7.595).abs() < 0.02, "{interval}");
    }

    #[test]
    fn a_sweep_with_neither_interval_nor_prf_says_so() {
        let nyquist = Nyquist::establish(None, None, None, Some(5.33));
        assert_eq!(nyquist.source, NyquistSource::Unavailable);
        assert!(!nyquist.is_usable());
    }

    #[test]
    fn a_nonpositive_declared_interval_is_not_taken_at_face_value() {
        let nyquist = Nyquist::establish(Some(0.0), None, None, None);
        assert_eq!(nyquist.source, NyquistSource::Unavailable);
    }

    #[test]
    fn gate_ranges_are_bin_centres_measured_from_the_start_of_the_first_bin() {
        let sweep = test_sweep(0.0, 250.0, 720);
        // Centre of bin 0 is half a bin out, not zero.
        assert_eq!(sweep.gate_centre_range_m(0), 125.0);
        assert_eq!(sweep.gate_centre_range_m(1), 375.0);
        assert_eq!(sweep.max_range_m(), 180_000.0);
    }

    #[test]
    fn a_nonzero_rstart_shifts_every_gate() {
        // rstart is in km in the file; by the time it reaches here it is
        // metres, so a 2 km start is 2000.
        let sweep = test_sweep(2000.0, 500.0, 100);
        assert_eq!(sweep.gate_centre_range_m(0), 2250.0);
        assert_eq!(sweep.max_range_m(), 52_000.0);
    }

    #[test]
    fn odim_date_and_time_parse_as_utc() {
        let when = parse_odim_datetime("20260812", "233520").expect("parses");
        assert_eq!(when.to_rfc3339(), "2026-08-12T23:35:20+00:00");
        // The shorter spelling some writers use.
        let short = parse_odim_datetime("20260812", "2335").expect("parses");
        assert_eq!(short.to_rfc3339(), "2026-08-12T23:35:00+00:00");
        assert!(parse_odim_datetime("not-a-date", "233520").is_none());
    }

    #[test]
    fn calibration_stays_in_f64_through_the_affine_step() {
        // A 16-bit DWD reflectivity: gain is ~2.9e-3, so a raw step is far
        // below f32's resolution at the top of the range.
        let cal = Calibration {
            gain: 0.002_929_821_616_590_115,
            offset: -64.002_929_821_616_59,
            nodata: Some(65535.0),
            undetect: Some(0.0),
            sentinels_collide: false,
        };
        let a = cal.apply(40000.0);
        let b = cal.apply(40001.0);
        assert!(a != b, "adjacent raw steps collapsed");
        assert!((b - a - cal.gain).abs() < 1e-12);
    }

    fn test_sweep(range_start_m: f64, range_scale_m: f64, nbins: usize) -> Sweep {
        Sweep {
            index: 1,
            path: "/dataset1".to_string(),
            elevation_deg: 0.5,
            nrays: 360,
            nbins,
            range_scale_m,
            range_start_m,
            a1gate: Some(101),
            start_time: None,
            end_time: None,
            nyquist: Nyquist::establish(Some(31.9), Some(800.0), Some(600.0), Some(5.33)),
            azimuth_deg: Vec::new(),
            azimuth_source: AzimuthSource::MeasuredStartStopMean,
            ray_elevation_deg: None,
            moments: Vec::new(),
        }
    }
}
