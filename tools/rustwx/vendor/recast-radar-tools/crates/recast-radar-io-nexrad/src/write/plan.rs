//! Everything the writer decides before encoding: which sweeps and fields
//! are written, their Message 31 moments, gate geometry and codings, the
//! elevation numbers, radial times, site identifier, VCP and the constant
//! blocks. Every refusal happens here, so encoding never fails half way.

use recast_radar_core::bounded_read::MAX_GATES_PER_RADIAL;
use recast_radar_core::model::{
    ArrayBuf, AttrValue, CycleTracker, Field, FieldName, Polarization, Quantity, RangeCoord,
    Scalar, SourceFormat, Sweep, SweepMode, Volume, collection_order,
};

use super::quantize::{self, FieldEncoding};
use super::{
    DataMessage, MomentReport, RecordLayout, SkippedField, SourceMetadata, WriteError,
    WriteOptions, WriteSummary, WrittenRays,
};
use crate::messages::RawMessages;
use crate::messages::msg31_blocks::{
    ElevationDataBlock, ProcessingStatus, RadialDataBlock, VolumeDataBlock,
};
use crate::messages::vcp::VolumeCoveragePattern;

/// Most elevation cuts a Level II volume numbers (Table XVII-A byte 22:
/// elevation number 1 to 32).
pub(crate) const MAX_ELEVATION_CUTS: usize = 32;
/// Fixed frame of the metadata record.
const FRAME_BYTES: usize = 2432;
/// Milliseconds per day.
const DAY_MS: i64 = 86_400_000;
/// Largest first-gate range and gate spacing in metres: the fast decoder
/// reads both as signed 16-bit numbers.
const MAX_RANGE_FIELD_M: f64 = 32_767.0;

/// A Message 31 data moment (Table XVII-I).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum Moment {
    /// `REF`: reflectivity, dBZ.
    Ref,
    /// `VEL`: radial velocity, m/s.
    Vel,
    /// `SW`: spectrum width, m/s.
    Sw,
    /// `ZDR`: differential reflectivity, dB.
    Zdr,
    /// `PHI`: differential phase, degrees.
    Phi,
    /// `RHO`: correlation coefficient.
    Rho,
    /// `CFP`: clutter filter power removed, dB.
    Cfp,
}

impl Moment {
    /// Every moment, in the order radials carry them.
    pub const ALL: [Moment; 7] = [
        Moment::Ref,
        Moment::Vel,
        Moment::Sw,
        Moment::Zdr,
        Moment::Phi,
        Moment::Rho,
        Moment::Cfp,
    ];

    /// The three name bytes of the data moment block (`"SW "` padded).
    pub fn block_name(self) -> [u8; 3] {
        match self {
            Moment::Ref => *b"REF",
            Moment::Vel => *b"VEL",
            Moment::Sw => *b"SW ",
            Moment::Zdr => *b"ZDR",
            Moment::Phi => *b"PHI",
            Moment::Rho => *b"RHO",
            Moment::Cfp => *b"CFP",
        }
    }

    /// The moment named `name` ([`Moment::name`], ASCII case-insensitive,
    /// surrounding spaces ignored), or `None`.
    pub fn parse(name: &str) -> Option<Moment> {
        let name = name.trim();
        Moment::ALL
            .into_iter()
            .find(|moment| moment.name().eq_ignore_ascii_case(name))
    }

    /// The ICD's typical coding of the moment, as NOAA's current files carry
    /// it and [`super::Quantization::Standard`] writes it: word size in bits,
    /// scale and offset (`value = (code - offset) / scale`).
    pub fn standard_coding(self) -> (u8, f32, f32) {
        let coding = quantize::standard_codings(self)[0];
        (coding.word_size, coding.scale, coding.offset)
    }

    /// The name without padding.
    pub fn name(self) -> &'static str {
        match self {
            Moment::Ref => "REF",
            Moment::Vel => "VEL",
            Moment::Sw => "SW",
            Moment::Zdr => "ZDR",
            Moment::Phi => "PHI",
            Moment::Rho => "RHO",
            Moment::Cfp => "CFP",
        }
    }
}

impl std::fmt::Display for Moment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// A Level II date (day 1 is 1970-01-01) and milliseconds past midnight.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct NexradTime {
    pub date: u16,
    pub ms: u32,
}

impl NexradTime {
    fn from_epoch_ms(ms: i64) -> Option<Self> {
        let date = u16::try_from(ms.checked_div_euclid(DAY_MS)? + 1).ok()?;
        let ms = u32::try_from(ms.rem_euclid(DAY_MS)).ok()?;
        (date >= 1).then_some(Self { date, ms })
    }

    fn epoch_ms(self) -> i64 {
        (i64::from(self.date) - 1) * DAY_MS + i64::from(self.ms)
    }
}

/// One written moment of a sweep.
pub(crate) struct MomentPlan<'a> {
    pub moment: Moment,
    pub field: &'a Field,
    /// Leading native gates left out (negative range).
    pub skip_gates: usize,
    pub first_gate_m: u16,
    pub gate_spacing_m: u16,
    pub gate_count: u16,
    pub tover_raw: u16,
    pub snr_threshold_raw: i16,
    pub control_flags: u8,
    pub encoding: FieldEncoding,
}

impl SweepPlan<'_> {
    /// The VOL, ELV and RAD blocks of ray `ray` (the RAD block's Nyquist
    /// velocity and unambiguous range are set from the volume on writing).
    pub fn blocks(&self, ray: usize) -> (&VolumeDataBlock, &ElevationDataBlock, &RadialDataBlock) {
        match self.rays.get(ray) {
            Some(own) => (&own.volume, &own.elevation, &own.radial),
            None => (
                &self.constants.volume,
                &self.constants.elevation,
                &self.constants.radial,
            ),
        }
    }
}

impl Plan<'_> {
    /// The coding of each moment the plan writes (its first field's).
    pub fn codings(&self) -> Vec<PinnedCoding> {
        let mut codings: Vec<PinnedCoding> = Vec::new();
        for moment in self.sweeps.iter().flat_map(|sweep| &sweep.moments) {
            if !codings.iter().any(|(known, _)| *known == moment.moment) {
                codings.push((moment.moment, moment.encoding.coding));
            }
        }
        codings
    }
}

impl MomentPlan<'_> {
    /// Bytes of the moment block in a radial: the 28-byte header and the
    /// gates.
    pub fn block_len(&self) -> usize {
        28 + usize::from(self.gate_count) * usize::from(self.encoding.coding.word_size / 8)
    }
}

/// The constant blocks every radial of a sweep carries (the RAD block's
/// Nyquist velocity and unambiguous range are set per radial).
pub(crate) struct SweepConstants {
    pub volume: VolumeDataBlock,
    pub elevation: ElevationDataBlock,
    pub radial: RadialDataBlock,
}

/// One radial's own constant blocks and Data Header Block items, from the
/// source metadata ([`crate::RadialConstants`]); the RAD block's Nyquist
/// velocity and unambiguous range still come from the volume.
pub(crate) struct RayConstants {
    pub volume: VolumeDataBlock,
    pub elevation: ElevationDataBlock,
    pub radial: RadialDataBlock,
    /// The radial's own radar identifier, written back unless
    /// [`WriteOptions::icao`] names a site ([`Plan::keep_radial_sites`]).
    pub radar_identifier: [u8; 4],
    pub azimuth_number: u16,
    pub spare: u8,
    pub azimuth_resolution_code: u8,
    pub radial_status_code: u8,
    pub cut_sector_number: u8,
    pub spot_blanking: u8,
    pub azimuth_indexing_raw: u8,
}

/// One written elevation cut.
pub(crate) struct SweepPlan<'a> {
    /// Index of the sweep in the source volume.
    pub index: usize,
    pub sweep: &'a Sweep,
    pub elevation_number: u8,
    /// The source rays written, in the order written ([`ray_order`]).
    pub order: Vec<usize>,
    /// Table XVII-A byte 20: 1 for 0.5 degree, 2 for 1 degree.
    pub azimuth_resolution: u8,
    /// Time of every source ray.
    pub times: Vec<NexradTime>,
    pub nyquist_raw: Vec<u16>,
    pub unambiguous_raw: Vec<u16>,
    pub moments: Vec<MomentPlan<'a>>,
    pub constants: SweepConstants,
    /// Table XVII-A byte 29 of radials without their own source constants:
    /// the azimuth indexing angle in 0.01 degree steps, 0 for none
    /// ([`azimuth_indexing_raw`]).
    pub azimuth_indexing_raw: u8,
    /// Each source ray's message generation date and time as a Level II
    /// source recorded them ([`source_message_times`]); empty otherwise
    /// (every radial's message time is its collection time).
    pub message_times: Vec<NexradTime>,
    /// Each source ray's message header channel byte as a Level II source
    /// recorded it ([`source_channels`]); empty otherwise (every radial on
    /// the Open RDA channel).
    pub channels: Vec<u8>,
    /// Each ray's own blocks when the source metadata has every radial of
    /// the sweep; empty otherwise (every radial carries `constants`, radial
    /// number = ray + 1, cut sector 1, no spot blanking or indexing).
    pub rays: Vec<RayConstants>,
    /// Message 5 values of the cut.
    pub fixed_angle_deg: f32,
    pub azimuth_rate_deg_per_s: f32,
    pub waveform: u8,
    pub super_resolution: u8,
}

/// Values of the synthesised Message 18.
pub(crate) struct Adaptation {
    pub latitude_deg: f64,
    pub longitude_deg: f64,
    pub frequency_mhz: Option<i32>,
    pub antenna_gain_db: Option<f32>,
    pub beam_width_deg: Option<f32>,
}

/// A source's metadata record written again, and what is changed in it so
/// that it agrees with the written volume.
pub(crate) struct CarriedRecord<'a> {
    /// The record as the source has it (whole 2432-byte frames).
    pub bytes: &'a [u8],
    /// What becomes of its Message 5.
    pub vcp: VcpEdit,
    /// First of the four frames whose Message 18 is replaced by the
    /// synthesised one: its body is shorter than Table XV's 9468 bytes,
    /// which MetPy and Py-ART cannot unpack.
    pub adaptation_frame: Option<usize>,
    /// Bytes replaced in place (offset in the record, new bytes): the VCP of
    /// Messages 2 and 5 and the Message 18 site name, when the written VCP
    /// or site differs from the record's.
    pub patches: Vec<(usize, Vec<u8>)>,
}

/// What becomes of a carried-over record's Message 5.
pub(crate) enum VcpEdit {
    /// Written as it is (with the pattern number patched when it differs),
    /// or there is none.
    Keep,
    /// Replaced by the synthesised Message 5 (it does not decode, or does
    /// not list a cut for every written sweep).
    Synthesise {
        /// Zero-based frame of the record.
        frame: usize,
    },
    /// Its cuts listed again in the order of the written sweeps, which
    /// readers number 1 to n: a volume that leaves out or reorders the
    /// source's cuts.
    Reindex {
        /// Zero-based frame of the record.
        frame: usize,
        /// Zero-based source cut of each written sweep.
        cuts: Vec<usize>,
        /// Bytes per cut in the source message.
        cut_len: usize,
    },
}

/// The complete write decision.
pub(crate) struct Plan<'a> {
    pub site: [u8; 4],
    /// Radials with their own source constants keep their own radar
    /// identifier (no [`WriteOptions::icao`] given); the others carry `site`.
    pub keep_radial_sites: bool,
    /// The header's tape name, `AR2V00nn`.
    pub tape: [u8; 8],
    pub volume_number: u16,
    pub vcp: u16,
    pub header_time: NexradTime,
    pub sweeps: Vec<SweepPlan<'a>>,
    /// A source metadata record to write instead of the synthesised one.
    pub metadata_record: Option<CarriedRecord<'a>>,
    /// The source's data messages, in the order of their places among the
    /// radials.
    pub data_messages: Vec<&'a DataMessage>,
    pub adaptation: Adaptation,
    /// Message 5 halfword 6 upper byte: 2 for 0.5 m/s, 4 for 1 m/s.
    pub velocity_resolution: u8,
    /// Message 5 halfword 6 lower byte, the pulse width: 2 short, 4 long
    /// ([`pulse_width_code`]).
    pub pulse_width: u8,
    pub radials_per_record: usize,
    pub record_layout: RecordLayout,
    pub summary: WriteSummary,
}

/// Decide how `volume` is written.
pub(crate) fn plan<'a>(
    volume: &'a Volume,
    source: SourceMetadata<'a>,
    options: &WriteOptions,
) -> Result<Plan<'a>, WriteError> {
    plan_with_codings(volume, source, options, None)
}

/// A moment's coding fixed before planning.
pub(crate) type PinnedCoding = (Moment, quantize::Coding);

/// [`plan`], with every moment's coding fixed beforehand when `pinned` is
/// given (the real-time writer codes every sweep of a moment alike, the
/// sweeps arriving one by one): a moment it does not list is refused
/// ([`WriteError::UnplannedMoment`]).
pub(crate) fn plan_with_codings<'a>(
    volume: &'a Volume,
    source: SourceMetadata<'a>,
    options: &WriteOptions,
    pinned: Option<&[PinnedCoding]>,
) -> Result<Plan<'a>, WriteError> {
    if options.radials_per_record == 0 || options.radials_per_record > usize::from(u16::MAX) {
        return Err(WriteError::InvalidOption(format!(
            "radials_per_record {} is outside 1 to 65535",
            options.radials_per_record
        )));
    }
    if let Some(limit) = options.max_range_error_m
        && (limit.is_nan() || limit < 0.0)
    {
        return Err(WriteError::InvalidOption(format!(
            "max_range_error_m {limit} must be zero or more"
        )));
    }
    check_location(volume)?;
    let fallbacks = [
        (
            "nyquist_velocity_mps",
            options.nyquist_velocity_mps,
            NYQUIST_SCALE,
        ),
        (
            "unambiguous_range_m",
            options.unambiguous_range_m,
            UNAMBIGUOUS_RANGE_SCALE,
        ),
    ];
    for (name, value, scale) in fallbacks {
        if let Some(value) = value
            && !(value.is_finite()
                && value > 0.0
                && (1.0..=f64::from(i16::MAX)).contains(&(f64::from(value) * scale).round()))
        {
            return Err(WriteError::InvalidOption(format!(
                "{name} {value} is outside what the RAD block holds (1 to {} steps of {})",
                i16::MAX,
                1.0 / scale
            )));
        }
    }
    let data_messages = checked_data_messages(source.data_messages)?;
    let mut summary = WriteSummary::default();
    let site = site_id(options.icao.as_deref(), volume)?;
    summary.icao = String::from_utf8_lossy(&site).into_owned();
    let volume_number = volume_number(options.volume_number, volume)?;
    let tape = tape_name(volume, &mut summary);
    let vcp = options.vcp.or(volume.scan.vcp_pattern).unwrap_or(0);

    let mut written: Vec<(usize, &Sweep)> = volume
        .sweeps
        .iter()
        .enumerate()
        .filter(|(index, sweep)| {
            let keep = sweep.nrays() > 0;
            if !keep {
                summary.skipped_sweeps.push(*index);
            }
            keep
        })
        .collect();
    if written.is_empty() {
        return Err(WriteError::EmptyVolume);
    }

    let reference_ms = volume.time_reference.timestamp_millis();
    let sweep_blocks = source
        .metadata
        .and_then(|metadata| metadata.per_sweep_elevation_data.as_deref())
        .unwrap_or_default();
    // Level II stores radials in the order they were collected: a foreign
    // volume's cuts in the order their sweeps were collected, each from its
    // earliest ray. A Level II source keeps the order it is given (its own,
    // or the one a caller chose for its cuts), and so do the cuts of any
    // volume under `WriteOptions::keep_sweep_order` (ECCC scans from the top
    // down; readers such as GR2Analyst expect the lowest cut first).
    let keep_order = volume.provenance.source_format == SourceFormat::NexradLevel2;
    if !keep_order && !options.keep_sweep_order {
        let mut rank = vec![0; volume.sweeps.len()];
        for (position, index) in collection_order(volume).into_iter().enumerate() {
            if let Some(slot) = rank.get_mut(index) {
                *slot = position;
            }
        }
        written.sort_by_key(|(index, _)| rank.get(*index).copied().unwrap_or(usize::MAX));
    }
    let mut sweeps = Vec::with_capacity(written.len());
    for (index, sweep) in written {
        check_sweep_mode(index, sweep)?;
        check_ray_arrays(index, sweep)?;
        let moments = plan_moments(index, sweep, options, &mut summary)?;
        let times = ray_times(index, sweep, reference_ms)?;
        check_angles(index, sweep)?;
        let order = ray_order(sweep, &moments, &times, keep_order);
        if order.is_empty() {
            summary.skipped_sweeps.push(index);
            summary.notes.push(format!(
                "sweep {index}: no ray has data of a Message 31 moment{}; the sweep is left out",
                if moments.is_empty() {
                    " (no field was written)"
                } else {
                    " (every row of the written fields is absent)"
                }
            ));
            continue;
        }
        note_ray_order(index, sweep, &order, &mut summary);
        let template = source_blocks(sweep_blocks, index, sweep);
        let constants = sweep_constants(
            volume,
            vcp,
            template.map(|t| (&t.volume, &t.elevation, &t.radial)),
        );
        let rays = template
            .filter(|t| t.radials.len() == sweep.nrays())
            .map(|t| ray_constants(volume, vcp, &constants, &t.radials))
            .unwrap_or_default();
        let waveform = waveform(&moments);
        // A Level II source's own code (its first written radial's), else
        // from the azimuth steps.
        let azimuth_resolution = order
            .first()
            .and_then(|ray| rays.get(*ray))
            .map(|own| own.azimuth_resolution_code)
            .filter(|code| matches!(code, 1 | 2))
            .unwrap_or_else(|| azimuth_resolution(sweep));
        sweeps.push(SweepPlan {
            index,
            sweep,
            // Numbered once the written sweeps are known.
            elevation_number: 0,
            azimuth_resolution,
            nyquist_raw: per_ray_raw(
                sweep.ray_vars.nyquist_velocity_mps.as_deref(),
                sweep.nrays(),
                NYQUIST_SCALE,
                options.nyquist_velocity_mps,
            ),
            unambiguous_raw: per_ray_raw(
                sweep.ray_vars.unambiguous_range_m.as_deref(),
                sweep.nrays(),
                UNAMBIGUOUS_RANGE_SCALE,
                options.unambiguous_range_m,
            ),
            azimuth_rate_deg_per_s: azimuth_rate(sweep, &times, &order),
            order,
            times,
            fixed_angle_deg: fixed_angle(index, sweep, &mut summary),
            waveform,
            super_resolution: u8::from(azimuth_resolution == 1),
            azimuth_indexing_raw: azimuth_indexing_raw(sweep),
            channels: if volume.provenance.source_format == SourceFormat::NexradLevel2 {
                source_channels(sweep)
            } else {
                Vec::new()
            },
            message_times: if volume.provenance.source_format == SourceFormat::NexradLevel2 {
                source_message_times(sweep)
            } else {
                Vec::new()
            },
            constants,
            rays,
            moments,
        });
    }
    summary.skipped_sweeps.sort_unstable();
    ray_order_notes(volume, &mut summary);
    if sweeps.is_empty() {
        return Err(WriteError::NoMoments);
    }
    summary.written_sweeps = sweeps.iter().map(|sweep| sweep.index).collect();
    if summary
        .written_sweeps
        .windows(2)
        .any(|pair| pair[0] > pair[1])
        && !keep_order
    {
        summary.notes.push(format!(
            "cuts written in the order their sweeps were collected, as Level II holds radials: \
             sweeps {}",
            summary
                .written_sweeps
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    check_one_scan_cycle(volume, &sweeps)?;
    if sweeps.len() > MAX_ELEVATION_CUTS {
        return Err(WriteError::TooManySweeps {
            count: sweeps.len(),
            max: MAX_ELEVATION_CUTS,
        });
    }
    for (number, sweep) in sweeps.iter_mut().enumerate() {
        sweep.elevation_number = u8::try_from(number + 1).unwrap_or(u8::MAX);
    }
    if let Some(pinned) = pinned {
        check_pinned(&sweeps, pinned)?;
    }
    assign_codings(
        &mut sweeps,
        options.quantization,
        pinned.unwrap_or_default(),
    );
    check_clipping(&sweeps, pinned.is_some())?;
    check_radial_lengths(&sweeps)?;

    let header_time = header_time(source, &sweeps)?;
    summary.volume_time = chrono::DateTime::from_timestamp_millis(header_time.epoch_ms());
    summary.sweeps = sweeps.len();
    summary.radials = sweeps.iter().map(|sweep| sweep.order.len()).sum();
    for sweep in &sweeps {
        for moment in &sweep.moments {
            summary.moments.push(MomentReport {
                sweep: sweep.index,
                moment: moment.moment,
                field: moment.field.name.clone(),
                word_size: moment.encoding.coding.word_size,
                scale: moment.encoding.coding.scale,
                offset: moment.encoding.coding.offset,
                exact: moment.encoding.exact,
                max_abs_error: moment.encoding.max_abs_error,
                absent_rays: moment.field.absent_rows.len(),
                dropped_gates: moment.skip_gates,
            });
        }
    }

    if volume.provenance.source_format == SourceFormat::NexradLevel2 {
        renumbering_note(&sweeps, &mut summary);
    } else {
        nyquist_note(&sweeps, &mut summary);
    }
    pyart_spacing_note(&sweeps, &mut summary);
    let metadata_record = match source.metadata_record {
        Some(record) => checked_metadata_record(record, &mut summary)?,
        None => None,
    }
    .and_then(|record| carried_record(record, &sweeps, vcp, &site, &mut summary));
    let velocity_resolution = sweeps
        .iter()
        .flat_map(|sweep| &sweep.moments)
        .find(|moment| moment.moment == Moment::Vel)
        .map_or(2, |moment| {
            if moment.encoding.coding.scale == 1.0 && moment.encoding.coding.word_size == 8 {
                4
            } else {
                2
            }
        });
    let pulse_width = pulse_width_code(volume, &sweeps);
    Ok(Plan {
        site,
        keep_radial_sites: options.icao.is_none(),
        tape,
        volume_number,
        vcp,
        header_time,
        adaptation: adaptation(volume),
        sweeps,
        metadata_record,
        data_messages,
        velocity_resolution,
        pulse_width,
        radials_per_record: options.radials_per_record,
        record_layout: options.record_layout,
        summary,
    })
}

/// The source rays of a sweep in the order they are written: the rays on
/// which at least one written moment has data (Py-ART and xradar take a
/// cut's moments from its first radial, and a radial without moment blocks
/// carries nothing), from the earliest one when the storage order is the
/// order they were collected in turned round (the times run forward but
/// for one step back, and the last ray is no later than the first), unless
/// `keep_order`. ODIM stores a sweep's rays from north while the antenna
/// starts anywhere (`where/a1gate`); Level II has radials in the order
/// they were collected, the first one
/// opening the cut. Any other order of times is kept as stored: NOXP's
/// DORADE clock steps back a second every 29 rays or so while the antenna
/// runs on, so its storage order is the order of collection.
fn ray_order(
    sweep: &Sweep,
    moments: &[MomentPlan<'_>],
    times: &[NexradTime],
    keep_order: bool,
) -> Vec<usize> {
    let mut order: Vec<usize> = (0..sweep.nrays())
        .filter(|&ray| moments.iter().any(|moment| !moment.field.is_absent(ray)))
        .collect();
    if keep_order {
        return order;
    }
    let time = |position: usize| {
        order
            .get(position)
            .and_then(|ray| times.get(*ray))
            .map_or(0, |time| time.epoch_ms())
    };
    let mut steps_back = (1..order.len()).filter(|&position| time(position) < time(position - 1));
    if let (Some(start), None) = (steps_back.next(), steps_back.next())
        && time(order.len() - 1) <= time(0)
    {
        order.rotate_left(start);
    }
    order
}

/// Record a sweep whose radials are not its rays in storage order in
/// [`WriteSummary::written_rays`].
fn note_ray_order(index: usize, sweep: &Sweep, order: &[usize], summary: &mut WriteSummary) {
    let reordered = order.windows(2).any(|pair| pair[0] > pair[1]);
    if order.len() != sweep.nrays() || reordered {
        summary.written_rays.push(WrittenRays {
            sweep: index,
            rays: order.to_vec(),
        });
    }
}

/// One note for the sweeps written from another ray than their first, one
/// for the sweeps with rays left out.
fn ray_order_notes(volume: &Volume, summary: &mut WriteSummary) {
    let mut turned = Vec::new();
    let mut thinned = Vec::new();
    for written in &summary.written_rays {
        let rays = volume.sweeps.get(written.sweep).map_or(0, Sweep::nrays);
        if written.rays.windows(2).any(|pair| pair[0] > pair[1]) {
            turned.push(format!(
                "{} (from ray {})",
                written.sweep,
                written.rays.first().copied().unwrap_or_default()
            ));
        }
        if written.rays.len() != rays {
            thinned.push(format!(
                "{} ({} of {rays})",
                written.sweep,
                rays - written.rays.len()
            ));
        }
    }
    if !turned.is_empty() {
        summary.notes.push(format!(
            "rays written in the order they were collected, from the earliest, in sweeps stored \
             from another azimuth: {}",
            turned.join(", ")
        ));
    }
    if !thinned.is_empty() {
        summary.notes.push(format!(
            "rays without data of a written moment left out of sweeps {}",
            thinned.join(", ")
        ));
    }
}

/// Under the real-time writer, every written moment must have a coding
/// fixed by the planned volume.
fn check_pinned(sweeps: &[SweepPlan<'_>], pinned: &[PinnedCoding]) -> Result<(), WriteError> {
    for sweep in sweeps {
        for moment in &sweep.moments {
            if !pinned.iter().any(|(known, _)| *known == moment.moment) {
                return Err(WriteError::UnplannedMoment {
                    sweep: sweep.index,
                    moment: moment.moment,
                    field: moment.field.name.to_string(),
                });
            }
        }
    }
    Ok(())
}

/// Refuse a volume with a value its moment's coding cannot hold: the
/// writer never clips or drops a source value. Only a coding fixed
/// beforehand (`pinned`, the real-time writer's) leaves values outside it;
/// a coding chosen from the values holds every one of them.
fn check_clipping(sweeps: &[SweepPlan<'_>], pinned: bool) -> Result<(), WriteError> {
    for sweep in sweeps {
        for moment in &sweep.moments {
            let gates = moment.encoding.clipped_gates;
            if gates > 0 {
                let (low, high) = moment.encoding.coding.value_range();
                return Err(WriteError::ValueOutsideCoding {
                    sweep: sweep.index,
                    moment: moment.moment,
                    field: moment.field.name.to_string(),
                    gates,
                    low,
                    high,
                    planned: pinned,
                });
            }
        }
    }
    Ok(())
}

/// The 4-character site identifier: the option, else derived from the
/// instrument name (see `docs/level2/writer.md`).
fn site_id(option: Option<&str>, volume: &Volume) -> Result<[u8; 4], WriteError> {
    let valid = |c: char| c.is_ascii_alphanumeric() || c == '_';
    let chosen: String = match option {
        Some(text) => {
            if text.is_empty() || text.len() > 4 || !text.chars().all(valid) {
                return Err(WriteError::InvalidSiteId(format!(
                    "{:?} must be 1 to 4 characters from [A-Za-z0-9_]",
                    text
                )));
            }
            text.to_owned()
        }
        None => {
            let name = volume.attrs.instrument_name.trim();
            let alnum: String = name.chars().filter(char::is_ascii_alphanumeric).collect();
            let derived = if name.len() == 4 && name.chars().all(valid) {
                name.to_owned()
            } else if alnum.len() == 5
                && name.len() == 5
                && alnum.chars().take(2).all(|c| c.is_ascii_alphabetic())
            {
                // ODIM NOD: two-letter country, three-letter radar
                // ("DKROM" -> "DROM").
                let mut id = String::new();
                id.push(alnum.chars().next().unwrap_or('_'));
                id.extend(alnum.chars().skip(2));
                id.to_ascii_uppercase()
            } else {
                alnum
                    .chars()
                    .take(4)
                    .collect::<String>()
                    .to_ascii_uppercase()
            };
            if derived.is_empty() {
                return Err(WriteError::InvalidSiteId(format!(
                    "instrument name {:?} has no letters or digits; set WriteOptions::icao",
                    volume.attrs.instrument_name
                )));
            }
            derived
        }
    };
    let mut site = [b'_'; 4];
    for (slot, byte) in site.iter_mut().zip(chosen.bytes()) {
        *slot = byte;
    }
    Ok(site)
}

/// The header extension's volume number.
fn volume_number(option: Option<u16>, volume: &Volume) -> Result<u16, WriteError> {
    if let Some(number) = option {
        if !(1..=999).contains(&number) {
            return Err(WriteError::InvalidOption(format!(
                "volume_number {number} is outside 1 to 999"
            )));
        }
        return Ok(number);
    }
    let from_source = (volume.provenance.source_format == SourceFormat::NexradLevel2)
        .then_some(volume.provenance.source_version.as_deref())
        .flatten()
        .filter(|version| version.starts_with("AR2V"))
        .and_then(|version| version.rsplit_once('.'))
        .and_then(|(_, number)| number.trim().parse::<u16>().ok())
        .filter(|number| (1..=999).contains(number));
    Ok(from_source.unwrap_or(1))
}

/// The header's tape name: a Level II source's own `AR2V00nn` when it names
/// a Message 31 archive (version 2 or later), so a re-encoded file keeps its
/// version; else `AR2V0006`, the version of current Message 31 files.
/// `ARCHIVE2` and `AR2V0001` name Message 1 archives, and readers that
/// trust the name (RSL) parse a file so labelled as Message 1 radials; a
/// source so labelled (Message 1 files, and early Build 10 files such as
/// KVWX 2008 whose Message 31 radials carry `AR2V0001`) is relabelled, with
/// a note in `summary`.
fn tape_name(volume: &Volume, summary: &mut WriteSummary) -> [u8; 8] {
    let mut tape = *b"AR2V0006";
    if volume.provenance.source_format != SourceFormat::NexradLevel2 {
        return tape;
    }
    let Some(version) = volume.provenance.source_version.as_deref() else {
        return tape;
    };
    let digits = version
        .strip_prefix("AR2V")
        .filter(|digits| digits.len() >= 4 && digits.as_bytes()[..4].iter().all(u8::is_ascii_digit))
        .map(|digits| &digits.as_bytes()[..4]);
    match digits {
        Some(digits) if digits > b"0001".as_slice() => tape[4..].copy_from_slice(digits),
        _ => summary.notes.push(format!(
            "the source's header version {} names a Message 1 archive; written as AR2V0006 \
             for the Message 31 radials",
            version.split('.').next().unwrap_or(version).trim()
        )),
    }
    tape
}

/// PPI-type sweeps only: Message 31 radials sweep in azimuth at one
/// elevation.
fn check_sweep_mode(index: usize, sweep: &Sweep) -> Result<(), WriteError> {
    match sweep.sweep_mode {
        SweepMode::AzimuthSurveillance
        | SweepMode::Sector
        | SweepMode::ManualPpi
        | SweepMode::VerticalPointing => Ok(()),
        ref other => Err(WriteError::UnsupportedSweepMode {
            sweep: index,
            mode: other.as_str().to_owned(),
        }),
    }
}

/// Every per-ray array the writer reads has one entry per ray (the azimuths
/// set the ray count, as [`Sweep::nrays`] counts them). Decoders seal their
/// sweeps, which checks this; a volume built or changed by hand may not be.
fn check_ray_arrays(index: usize, sweep: &Sweep) -> Result<(), WriteError> {
    let nrays = sweep.nrays();
    let vars = &sweep.ray_vars;
    let arrays = [
        ("rays.time_s", Some(sweep.rays.time_s.len())),
        ("rays.elevation_deg", Some(sweep.rays.elevation_deg.len())),
        (
            "ray_vars.nyquist_velocity_mps",
            vars.nyquist_velocity_mps.as_ref().map(Vec::len),
        ),
        (
            "ray_vars.unambiguous_range_m",
            vars.unambiguous_range_m.as_ref().map(Vec::len),
        ),
    ];
    for (what, len) in arrays {
        if let Some(len) = len
            && len != nrays
        {
            return Err(WriteError::Inconsistent {
                sweep: index,
                reason: format!("{what} has {len} entries for {nrays} rays"),
            });
        }
    }
    Ok(())
}

/// A written field has one row of `ngates` values per ray, absent rows
/// that are ascending ray indices, and a gate mapping its range allows.
fn check_field_shape(index: usize, sweep: &Sweep, field: &Field) -> Result<(), WriteError> {
    let nrays = sweep.nrays();
    let inconsistent = |reason: String| WriteError::Inconsistent {
        sweep: index,
        reason: format!("field {}: {reason}", field.name),
    };
    if field.nrays as usize != nrays {
        return Err(inconsistent(format!(
            "{} rows for {nrays} rays",
            field.nrays
        )));
    }
    if nrays.checked_mul(field.ngates as usize) != Some(field.data.len()) {
        return Err(inconsistent(format!(
            "{} values for {nrays} rows of {} gates",
            field.data.len(),
            field.ngates
        )));
    }
    let ascending = field.absent_rows.windows(2).all(|pair| pair[0] < pair[1]);
    if !ascending
        || field
            .absent_rows
            .last()
            .is_some_and(|last| *last as usize >= nrays)
    {
        return Err(inconsistent(
            "absent rows are not ascending indices of the sweep's rays".to_owned(),
        ));
    }
    let explicit = matches!(sweep.range, RangeCoord::Explicit { .. });
    if field.gates.stride == 0 || (explicit && field.gates.stride != 1) {
        return Err(inconsistent(format!(
            "gate stride {} on a{} range",
            field.gates.stride,
            if explicit { "n explicit" } else { " uniform" }
        )));
    }
    Ok(())
}

/// The cut's Message 5 angle: the sweep's fixed angle, else (when it is not
/// finite) the median elevation of its rays, with a note.
fn fixed_angle(index: usize, sweep: &Sweep, summary: &mut WriteSummary) -> f32 {
    if sweep.fixed_angle_deg.is_finite() {
        return sweep.fixed_angle_deg;
    }
    let mut elevations: Vec<f32> = sweep
        .rays
        .elevation_deg
        .iter()
        .copied()
        .filter(|elevation| elevation.is_finite())
        .collect();
    elevations.sort_by(f32::total_cmp);
    let median = elevations
        .get(elevations.len() / 2)
        .copied()
        .unwrap_or_default();
    summary.notes.push(format!(
        "sweep {index}: fixed angle {} is not finite; Message 5 has the median ray elevation, \
         {median} degrees",
        sweep.fixed_angle_deg
    ));
    median
}

/// Azimuth and elevation of every ray must be finite.
fn check_angles(index: usize, sweep: &Sweep) -> Result<(), WriteError> {
    for (ray, (azimuth, elevation)) in sweep
        .rays
        .azimuth_deg
        .iter()
        .zip(&sweep.rays.elevation_deg)
        .enumerate()
    {
        if !azimuth.is_finite() || !elevation.is_finite() {
            return Err(WriteError::Ray {
                sweep: index,
                ray,
                reason: format!("azimuth {azimuth} or elevation {elevation} is not finite"),
            });
        }
    }
    Ok(())
}

/// Level II time of every ray (millisecond precision).
fn ray_times(
    index: usize,
    sweep: &Sweep,
    reference_ms: i64,
) -> Result<Vec<NexradTime>, WriteError> {
    sweep
        .rays
        .time_s
        .iter()
        .enumerate()
        .map(|(ray, time_s)| {
            let error = |reason: String| WriteError::Ray {
                sweep: index,
                ray,
                reason,
            };
            if !time_s.is_finite() || time_s.abs() > 1e11 {
                return Err(error(format!("time {time_s} s is not a usable offset")));
            }
            let ms = reference_ms + (time_s * 1000.0).round() as i64;
            NexradTime::from_epoch_ms(ms).ok_or_else(|| {
                error(format!(
                    "time {ms} ms since 1970 is outside the Level II dates (1970 to 2149)"
                ))
            })
        })
        .collect()
}

/// RAD block steps per unit: Nyquist velocity in 0.01 m/s, unambiguous range
/// in 0.1 km (Table XVII-H).
const NYQUIST_SCALE: f64 = 100.0;
const UNAMBIGUOUS_RANGE_SCALE: f64 = 0.01;

/// Per-ray raw values of the RAD block (`value * scale`, rounded), one per
/// ray ([`check_ray_arrays`] has checked the length). A ray whose value is
/// missing, not finite or not positive takes `fallback`
/// ([`WriteOptions::nyquist_velocity_mps`] and
/// [`WriteOptions::unambiguous_range_m`], checked by the planner), else 0.
fn per_ray_raw(values: Option<&[f32]>, rays: usize, scale: f64, fallback: Option<f32>) -> Vec<u16> {
    let raw = |value: f32| {
        let raw = (f64::from(value) * scale).round();
        (raw.is_finite() && raw > 0.0).then(|| raw.min(f64::from(i16::MAX as u16)) as u16)
    };
    let fallback = fallback.and_then(raw).unwrap_or(0);
    (0..rays)
        .map(|ray| {
            values
                .and_then(|values| values.get(ray))
                .and_then(|value| raw(*value))
                .unwrap_or(fallback)
        })
        .collect()
}

/// A note naming the sweeps of a foreign volume whose VEL radials carry no
/// Nyquist velocity (the RAD block holds 0, which readers take as unknown),
/// so the caller can supply one ([`WriteOptions::nyquist_velocity_mps`]).
/// A Level II source's RAD blocks are written as recorded, without a note.
fn nyquist_note(sweeps: &[SweepPlan<'_>], summary: &mut WriteSummary) {
    let without: Vec<String> = sweeps
        .iter()
        .filter(|sweep| {
            sweep.moments.iter().any(|moment| {
                moment.moment == Moment::Vel
                    && sweep.order.iter().any(|&ray| {
                        !moment.field.is_absent(ray)
                            && sweep.nyquist_raw.get(ray).copied().unwrap_or(0) == 0
                    })
            })
        })
        .map(|sweep| sweep.index.to_string())
        .collect();
    if !without.is_empty() {
        summary.notes.push(format!(
            "VEL written without a Nyquist velocity (0 in the RAD block) on radials of sweeps {}; \
             the source has none (WriteOptions::nyquist_velocity_mps supplies one)",
            without.join(", ")
        ));
    }
}

/// Table XVII-A byte 29 for a sweep: the indexing angle a Level II source
/// recorded (the sweep attribute `nexrad_azimuth_indexing_angle_deg`), else,
/// when the sweep's rays are indexed (`Sweep::rays_are_indexed`), its ray
/// angle resolution (`rays_angle_resolution_deg`, 0.01 to 2.55 degrees),
/// else 0 (no indexing).
fn azimuth_indexing_raw(sweep: &Sweep) -> u8 {
    let to_raw = |degrees: f64| {
        let raw = (degrees * 100.0).round();
        (raw >= 1.0 && raw <= f64::from(u8::MAX)).then_some(raw as u8)
    };
    let recorded = sweep
        .other
        .iter()
        .find(|(name, _)| &**name == "nexrad_azimuth_indexing_angle_deg")
        .and_then(|(_, value)| match value {
            AttrValue::Scalar(Scalar::F32(degrees)) => Some(f64::from(*degrees)),
            _ => None,
        });
    if let Some(degrees) = recorded {
        return to_raw(degrees).unwrap_or(0);
    }
    if sweep.rays_are_indexed == Some(true) {
        return sweep
            .rays_angle_resolution_deg
            .and_then(|degrees| to_raw(f64::from(degrees)))
            .unwrap_or(0);
    }
    0
}

/// The message header generation date and time (Table II halfwords 4 to 6)
/// of every ray of a Level II sweep, as the decoder carries them
/// (`nexrad_message_date`, `nexrad_message_milliseconds`). Empty when the
/// sweep has neither, or they do not have one entry per ray.
fn source_message_times(sweep: &Sweep) -> Vec<NexradTime> {
    let find = |name: &str| {
        sweep
            .extra_vars
            .iter()
            .find(|variable| &*variable.name == name)
            .map(|variable| &variable.values)
    };
    match (
        find("nexrad_message_date"),
        find("nexrad_message_milliseconds"),
    ) {
        (Some(ArrayBuf::U16(dates)), Some(ArrayBuf::U32(milliseconds)))
            if dates.len() == sweep.nrays() && milliseconds.len() == sweep.nrays() =>
        {
            dates
                .iter()
                .zip(milliseconds)
                .map(|(&date, &ms)| NexradTime { date, ms })
                .collect()
        }
        _ => Vec::new(),
    }
}

/// The message header channel byte (Table II halfword 2, high byte) of every
/// ray of a Level II sweep, as the decoder carries it: the per-ray
/// `nexrad_message_channels` variable when the sweep's radials differ, else
/// the sweep attribute of that name. Empty when the sweep has neither.
fn source_channels(sweep: &Sweep) -> Vec<u8> {
    const NAME: &str = "nexrad_message_channels";
    if let Some(variable) = sweep
        .extra_vars
        .iter()
        .find(|variable| &*variable.name == NAME)
        && let ArrayBuf::U8(values) = &variable.values
        && values.len() == sweep.nrays()
    {
        return values.clone();
    }
    match sweep.other.iter().find(|(name, _)| &**name == NAME) {
        Some((_, AttrValue::Scalar(Scalar::U8(channels)))) => vec![*channels; sweep.nrays()],
        _ => Vec::new(),
    }
}

/// The radar's position must be known: Message 31's VOL block and the RDA
/// adaptation data hold it, and a made-up one (such as 0, 0) would place
/// the radar wrongly for every reader.
fn check_location(volume: &Volume) -> Result<(), WriteError> {
    let location = volume.location;
    let within =
        |value: Option<f64>, limit: f64| value.is_some_and(|v| v.is_finite() && v.abs() <= limit);
    if !within(location.latitude_deg, 90.0) {
        return Err(WriteError::MissingLocation("latitude"));
    }
    if !within(location.longitude_deg, 360.0) {
        return Err(WriteError::MissingLocation("longitude"));
    }
    if !location.altitude_m.is_some_and(f64::is_finite) {
        return Err(WriteError::MissingLocation("height"));
    }
    Ok(())
}

/// Longest pulse, in seconds, written as a short pulse: between NEXRAD's
/// short (1.57 us) and long (4.57 us) pulses.
const SHORT_PULSE_MAX_S: f32 = 3.0e-6;

/// Message 5 pulse width code (Table XI halfword 6): 4 (long) when the
/// radar's pulse is longer than [`SHORT_PULSE_MAX_S`], else 2 (short). The
/// pulse is the median of the written sweeps' per-ray pulse widths, else
/// the volume's `radar_parameters` or first calibration's; a volume that
/// names none gets 2, the pulse of NEXRAD's precipitation patterns.
fn pulse_width_code(volume: &Volume, sweeps: &[SweepPlan<'_>]) -> u8 {
    let mut widths: Vec<f32> = sweeps
        .iter()
        .filter_map(|plan| plan.sweep.ray_vars.pulse_width_s.as_deref())
        .flatten()
        .copied()
        .filter(|width| width.is_finite() && *width > 0.0)
        .collect();
    let width = if widths.is_empty() {
        volume
            .radar_parameters
            .pulse_width_s
            .or_else(|| {
                volume
                    .radar_calibration
                    .first()
                    .and_then(|calibration| calibration.pulse_width_s)
            })
            .filter(|width| width.is_finite() && *width > 0.0)
    } else {
        widths.sort_by(f32::total_cmp);
        widths.get(widths.len() / 2).copied()
    };
    match width {
        Some(width) if width > SHORT_PULSE_MAX_S => 4,
        _ => 2,
    }
}

/// Largest azimuth spacing, in degrees, flagged as 0.5 degree resolution:
/// the midpoint of 0.5 and 1 degree less a margin, so that NEXRAD's 720-
/// and 360-radial cuts fall either side, and so do foreign cuts of 600 or
/// more radials (0.60 degree steps and finer, nearer 0.5 degree) and of
/// 512 or fewer (0.70 degree and coarser).
const HALF_DEGREE_MAX_SPACING_DEG: f32 = 0.65;

/// Table XVII-A byte 20: 1 (0.5 degree) when the sweep's ray angle
/// resolution, else its median azimuth step, is at most
/// [`HALF_DEGREE_MAX_SPACING_DEG`]; else 2 (1 degree).
fn azimuth_resolution(sweep: &Sweep) -> u8 {
    let spacing = sweep
        .rays_angle_resolution_deg
        .filter(|resolution| resolution.is_finite() && *resolution > 0.0)
        .or_else(|| median_azimuth_step(sweep));
    match spacing {
        Some(spacing) if spacing <= HALF_DEGREE_MAX_SPACING_DEG => 1,
        _ => 2,
    }
}

/// The median azimuth step between consecutive rays (steps of 0 left out).
fn median_azimuth_step(sweep: &Sweep) -> Option<f32> {
    let mut steps: Vec<f32> = sweep
        .rays
        .azimuth_deg
        .windows(2)
        .map(|pair| {
            let step = (pair[1] - pair[0]).rem_euclid(360.0);
            step.min(360.0 - step)
        })
        .filter(|step| *step > 0.0)
        .collect();
    if steps.is_empty() {
        return None;
    }
    steps.sort_by(f32::total_cmp);
    Some(steps[steps.len() / 2])
}

/// Azimuth rate for Message 5: the sweep's target rate, else azimuth
/// covered over the time taken by the written rays (`order`), else 0.
fn azimuth_rate(sweep: &Sweep, times: &[NexradTime], order: &[usize]) -> f32 {
    if let Some(rate) = sweep
        .target_scan_rate_deg_per_s
        .filter(|rate| rate.is_finite())
    {
        return rate;
    }
    let time = |ray: &usize| times.get(*ray).map(|time| time.epoch_ms());
    let (Some(first), Some(last)) = (order.first().and_then(time), order.last().and_then(time))
    else {
        return 0.0;
    };
    let seconds = (last - first) as f64 / 1000.0;
    let azimuth = |ray: usize| sweep.rays.azimuth_deg.get(ray).copied().unwrap_or_default();
    let covered: f64 = order
        .windows(2)
        .map(|pair| {
            let step = f64::from((azimuth(pair[1]) - azimuth(pair[0])).rem_euclid(360.0));
            step.min(360.0 - step)
        })
        .sum();
    if seconds > 0.0 && covered > 0.0 {
        (covered / seconds) as f32
    } else {
        0.0
    }
}

/// Message 5 waveform (E2 lower byte): 1 contiguous surveillance for a
/// reflectivity-only cut, 2 contiguous Doppler with ambiguity resolution
/// for a Doppler-only cut, 3 contiguous Doppler without ambiguity
/// resolution for both (the waveform of NEXRAD's upper cuts, which carry
/// every moment in one scan).
fn waveform(moments: &[MomentPlan<'_>]) -> u8 {
    let has = |moment: Moment| moments.iter().any(|plan| plan.moment == moment);
    let reflectivity = has(Moment::Ref);
    let doppler = has(Moment::Vel) || has(Moment::Sw);
    match (reflectivity, doppler) {
        (true, false) => 1,
        (false, true) => 2,
        _ => 3,
    }
}

/// Rank of `field` for `moment` (lower is preferred), or `None` when the
/// field does not map to it automatically.
fn auto_rank(field: &Field, moment: Moment) -> Option<u8> {
    use FieldName as N;
    // Single-polarization moments come from the horizontal channel; the
    // dual-polarization ones relate both channels.
    let polarization_fits = match moment {
        Moment::Zdr | Moment::Phi | Moment::Rho => matches!(
            field.polarization,
            Polarization::Hv | Polarization::Unspecified | Polarization::H
        ),
        _ => matches!(
            field.polarization,
            Polarization::H | Polarization::CopolarH | Polarization::Unspecified
        ),
    };
    let horizontal = matches!(
        field.polarization,
        Polarization::H | Polarization::CopolarH | Polarization::Unspecified
    );
    let pair = |exact: &[FieldName], quantity: Quantity, fallback: &[FieldName]| {
        if let Some(rank) = exact.iter().position(|name| *name == field.name) {
            Some(rank as u8)
        } else if fallback.contains(&field.name) {
            Some(20)
        } else if field.quantity == quantity && polarization_fits {
            Some(10)
        } else {
            None
        }
    };
    match moment {
        Moment::Ref => pair(
            &[N::Dbzh, N::Dbz],
            Quantity::Reflectivity,
            &[N::Dbth, N::Th],
        )
        .or_else(|| (field.quantity == Quantity::TotalPower && horizontal).then_some(21)),
        Moment::Vel => pair(&[N::Vradh, N::Vrad], Quantity::RadialVelocity, &[N::Vraddh]),
        Moment::Sw => pair(&[N::Wradh, N::Wrad], Quantity::SpectrumWidth, &[]),
        Moment::Zdr => pair(&[N::Zdr], Quantity::DifferentialReflectivity, &[N::Uzdr]),
        Moment::Phi => pair(&[N::Phidp], Quantity::DifferentialPhase, &[N::Uphidp]),
        Moment::Rho => pair(&[N::Rhohv], Quantity::CorrelationCoefficient, &[N::Urhohv]),
        Moment::Cfp => pair(&[N::Ccorh], Quantity::ClutterCorrection, &[]),
    }
}

/// Choose, place and code the moments of one sweep.
fn plan_moments<'a>(
    index: usize,
    sweep: &'a Sweep,
    options: &WriteOptions,
    summary: &mut WriteSummary,
) -> Result<Vec<MomentPlan<'a>>, WriteError> {
    // (moment, rank, field index)
    let mut chosen: Vec<(Moment, u8, usize)> = Vec::new();
    let mut unmapped = Vec::new();
    for (field_index, field) in sweep.fields.iter().enumerate() {
        let explicit = options
            .field_map
            .iter()
            .find(|(name, _)| *name == field.name)
            .map(|(_, moment)| (*moment, 0u8));
        let candidate = explicit.or_else(|| {
            Moment::ALL
                .iter()
                .filter_map(|moment| auto_rank(field, *moment).map(|rank| (*moment, rank + 1)))
                .min_by_key(|(_, rank)| *rank)
        });
        match candidate {
            Some((moment, rank)) => match chosen.iter_mut().find(|(m, _, _)| *m == moment) {
                Some(entry) if rank < entry.1 => {
                    unmapped.push((entry.2, format!("{moment} carries {} instead", field.name)));
                    *entry = (moment, rank, field_index);
                }
                Some(entry) => {
                    let winner = &sweep.fields[entry.2].name;
                    unmapped.push((field_index, format!("{moment} carries {winner} instead")));
                }
                None => chosen.push((moment, rank, field_index)),
            },
            None => unmapped.push((
                field_index,
                "no Message 31 moment for this quantity".to_owned(),
            )),
        }
    }
    chosen.sort_by_key(|(moment, _, _)| *moment);
    let mut moments = Vec::with_capacity(chosen.len());
    for (moment, _, field_index) in chosen {
        let field = &sweep.fields[field_index];
        check_field_shape(index, sweep, field)?;
        if field.ngates == 0 {
            unmapped.push((field_index, "the field has no gates".to_owned()));
            continue;
        }
        let geometry = gate_geometry(
            index,
            sweep,
            field,
            options.max_range_error_m,
            options.drop_negative_range_gates,
        )?;
        let Some((first_gate_m, gate_spacing_m, error_m, skip_gates)) = geometry else {
            unmapped.push((field_index, "every gate lies before the radar".to_owned()));
            continue;
        };
        summary.max_range_error_m = summary.max_range_error_m.max(error_m);
        let gate_count = u16::try_from(field.ngates as usize - skip_gates)
            .ok()
            .filter(|count| usize::from(*count) <= MAX_GATES_PER_RADIAL)
            .ok_or_else(|| WriteError::Geometry {
                sweep: index,
                field: field.name.to_string(),
                reason: format!(
                    "{} gates exceed the {MAX_GATES_PER_RADIAL}-gate Message 31 limit",
                    field.ngates
                ),
            })?;
        // Placeholder until `assign_codings` sees every sweep's fields.
        let encoding = quantize::FieldEncoding {
            coding: quantize::standard_codings(moment)[0],
            encoder: quantize::GateEncoder::Raw,
            exact: true,
            max_abs_error: 0.0,
            clipped_gates: 0,
        };
        moments.push(MomentPlan {
            moment,
            field,
            skip_gates,
            first_gate_m,
            gate_spacing_m,
            gate_count,
            tover_raw: attr_f64(field, "nexrad_tover_db")
                .map_or(0, |db| (db * 10.0).round().clamp(0.0, 65_535.0) as u16),
            snr_threshold_raw: attr_f64(field, "nexrad_snr_threshold_db").map_or(0, |db| {
                (db * 8.0)
                    .round()
                    .clamp(f64::from(i16::MIN), f64::from(i16::MAX)) as i16
            }),
            control_flags: attr_f64(field, "nexrad_recombination")
                .map_or(0, |code| code.clamp(0.0, 255.0) as u8),
            encoding,
        });
    }
    unmapped.sort_by_key(|(field_index, _)| *field_index);
    for (field_index, reason) in unmapped {
        summary.skipped_fields.push(SkippedField {
            sweep: index,
            field: sweep.fields[field_index].name.clone(),
            reason,
        });
    }
    Ok(moments)
}

/// Choose each moment's coding from its fields in every sweep.
fn assign_codings(
    sweeps: &mut [SweepPlan<'_>],
    policy: super::Quantization,
    pinned: &[PinnedCoding],
) {
    for moment in Moment::ALL {
        let fields: Vec<&Field> = sweeps
            .iter()
            .flat_map(|sweep| &sweep.moments)
            .filter(|plan| plan.moment == moment)
            .map(|plan| plan.field)
            .collect();
        if fields.is_empty() {
            continue;
        }
        let encodings = match pinned.iter().find(|(pinned, _)| *pinned == moment) {
            Some((_, coding)) => quantize::with_coding(&fields, *coding),
            None => quantize::choose(&fields, moment, policy),
        };
        let mut encodings = encodings.into_iter();
        for plan in sweeps
            .iter_mut()
            .flat_map(|sweep| sweep.moments.iter_mut())
            .filter(|plan| plan.moment == moment)
        {
            if let Some(encoding) = encodings.next() {
                plan.encoding = encoding;
            }
        }
    }
}

/// A numeric source attribute of the field (the Message 31 moment header
/// values the NEXRAD decoder keeps).
fn attr_f64(field: &Field, name: &str) -> Option<f64> {
    field
        .attrs
        .other
        .iter()
        .find(|(key, _)| key.as_ref() == name)
        .and_then(|(_, value)| match value {
            AttrValue::Scalar(scalar) => Some(scalar.as_f64()),
            _ => None,
        })
        .filter(|value: &f64| value.is_finite())
}

/// First gate and spacing in whole metres, the range error at the last
/// gate, and the leading gates dropped for lying before the radar (`None`
/// when every gate does).
fn gate_geometry(
    index: usize,
    sweep: &Sweep,
    field: &Field,
    max_error: Option<f64>,
    drop_negative: bool,
) -> Result<Option<(u16, u16, f64, usize)>, WriteError> {
    let error = |reason: String| WriteError::Geometry {
        sweep: index,
        field: field.name.to_string(),
        reason,
    };
    let (first, spacing) = field
        .native_geometry(&sweep.range)
        .ok_or_else(|| error("the field's gates are outside the sweep range".to_owned()))?;
    if let RangeCoord::Explicit { centers_m } = &sweep.range {
        let start = field.gates.start as usize;
        for gate in 0..field.ngates as usize {
            let Some(center) = centers_m.get(start + gate) else {
                return Err(error(
                    "the field's gates are outside the sweep range".to_owned(),
                ));
            };
            let expected = first + gate as f64 * spacing;
            if (f64::from(*center) - expected).abs() > 0.5_f64.max(1e-3 * spacing.abs()) {
                return Err(error(
                    "gate spacing is not constant; Message 31 gates are evenly spaced".to_owned(),
                ));
            }
        }
    }
    if !(first.is_finite() && spacing.is_finite()) || (spacing <= 0.0 && field.ngates > 1) {
        return Err(error(format!("first gate {first} m, spacing {spacing} m")));
    }
    let mut first = first;
    let mut skip = 0usize;
    if drop_negative && first.round() < 0.0 {
        if spacing <= 0.0 {
            return Ok(None);
        }
        // First gate whose centre rounds to 0 m or more.
        skip = ((-first - 0.5) / spacing).ceil().max(0.0) as usize;
        while first + skip as f64 * spacing < -0.5 {
            skip += 1;
        }
        if skip >= field.ngates as usize {
            return Ok(None);
        }
        first += skip as f64 * spacing;
    }
    let gates = f64::from(field.ngates.max(1)) - skip as f64;
    let first_m = first.round();
    let spacing_m = if field.ngates > 1 {
        spacing.round()
    } else {
        spacing.round().max(1.0)
    };
    if first_m < 0.0 {
        return Err(error(format!(
            "first gate at {first} m, before the radar; Message 31 holds 0 to \
             {MAX_RANGE_FIELD_M} m (WriteOptions::drop_negative_range_gates leaves out the gates \
             before the radar)"
        )));
    }
    if first_m > MAX_RANGE_FIELD_M {
        return Err(error(format!(
            "first gate at {first} m; Message 31 holds 0 to {MAX_RANGE_FIELD_M} m"
        )));
    }
    if !(1.0..=MAX_RANGE_FIELD_M).contains(&spacing_m) {
        return Err(error(format!(
            "gate spacing {spacing} m; Message 31 holds 1 to {MAX_RANGE_FIELD_M} m"
        )));
    }
    let range_error = (first_m - first).abs() + (gates - 1.0) * (spacing_m - spacing).abs();
    let limit = max_error.unwrap_or(spacing_m / 2.0);
    if range_error > limit + 1e-9 {
        return Err(error(format!(
            "first gate {first} m and spacing {spacing} m are not whole metres: \
             {range_error:.3} m off at the last gate, above the {limit} m allowed \
             (WriteOptions::max_range_error_m)"
        )));
    }
    Ok(Some((first_m as u16, spacing_m as u16, range_error, skip)))
}

/// Radials must fit the 16-bit radial length of the Data Header Block.
fn check_radial_lengths(sweeps: &[SweepPlan<'_>]) -> Result<(), WriteError> {
    for sweep in sweeps {
        for &ray in &sweep.order {
            let bytes = super::encode::radial_len(sweep, ray);
            if bytes > usize::from(u16::MAX) {
                return Err(WriteError::RadialTooLarge {
                    sweep: sweep.index,
                    ray,
                    bytes,
                });
            }
        }
    }
    Ok(())
}

/// The volume header time: the source's when it is Level II, else the
/// first written radial's, the one that opens the volume (the earliest in a
/// volume whose sweeps run in collection order, as NEXRAD's do). The
/// real-time start chunk, which goes out with the first sweep, has the same.
fn header_time(
    source: SourceMetadata<'_>,
    sweeps: &[SweepPlan<'_>],
) -> Result<NexradTime, WriteError> {
    if let Some(time) = source
        .metadata
        .and_then(|metadata| metadata.volume_header_time)
        .and_then(|time| NexradTime::from_epoch_ms(time.timestamp_millis()))
    {
        return Ok(time);
    }
    // The volume scan starts with its earliest radial.
    sweeps
        .iter()
        .flat_map(|sweep| sweep.order.iter().filter_map(|ray| sweep.times.get(*ray)))
        .copied()
        .min_by_key(|time| time.epoch_ms())
        .ok_or(WriteError::EmptyVolume)
}

/// A Level II file holds one volume scan: the written sweeps, taken in the
/// order they were collected, must be one scan cycle
/// ([`recast_radar_core::model::scan_cycles`]); sweeps of another cycle are
/// refused, never written beside this one's.
fn check_one_scan_cycle(volume: &Volume, sweeps: &[SweepPlan<'_>]) -> Result<(), WriteError> {
    let mut tracker = CycleTracker::new();
    for index in collection_order(volume)
        .into_iter()
        .filter(|index| sweeps.iter().any(|sweep| sweep.index == *index))
    {
        if let Some(begins) = tracker.check(volume, index, index) {
            return Err(WriteError::MixedScanCycles { begins });
        }
        tracker.add(volume, index, index);
    }
    Ok(())
}

/// A note when Py-ART cannot open the file: it takes the smallest first gate
/// and gate spacing of all moments as the volume's range and reads a moment
/// on another first gate or spacing only when its spacing is 2 or 4 times
/// the smallest (`pyart.io.nexrad_archive._find_scans_to_interp`, Py-ART
/// 2.3: "Gate spacing is neither 1/4 or 1/2"), as NEXRAD's 1 km and 250 m
/// gates are. A 30 m vertically pointing sweep beside 250 m sweeps is not.
fn pyart_spacing_note(sweeps: &[SweepPlan<'_>], summary: &mut WriteSummary) {
    let moments = || {
        sweeps
            .iter()
            .flat_map(|sweep| sweep.moments.iter().map(move |moment| (sweep, moment)))
    };
    let (Some(first), Some(spacing)) = (
        moments().map(|(_, moment)| moment.first_gate_m).min(),
        moments().map(|(_, moment)| moment.gate_spacing_m).min(),
    ) else {
        return;
    };
    let mut finest = Vec::new();
    let mut other = Vec::new();
    for (sweep, moment) in moments() {
        let own = u32::from(moment.gate_spacing_m);
        let smallest = u32::from(spacing);
        if moment.gate_spacing_m == spacing && !finest.contains(&sweep.index) {
            finest.push(sweep.index);
        }
        let differs = moment.first_gate_m != first || moment.gate_spacing_m != spacing;
        if differs && own != 2 * smallest && own != 4 * smallest && !other.contains(&sweep.index) {
            other.push(sweep.index);
        }
    }
    if other.is_empty() {
        return;
    }
    let list = |indices: &[usize]| {
        indices
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    };
    summary.notes.push(format!(
        "Py-ART 2.3 cannot open this file: it reads every moment on the smallest gate spacing \
         ({spacing} m, sweeps {}) from the smallest first gate ({first} m), and a moment on \
         another only when its gates are 2 or 4 times as long (\"Gate spacing is neither 1/4 or \
         1/2\"), which those of sweeps {} are not; leave out one kind for Py-ART",
        list(&finest),
        list(&other)
    ));
}

/// VOL, ELV and RAD blocks of a sweep: the source's own blocks when given,
/// with the location and VCP of the volume, else built from the volume.
fn sweep_constants(
    volume: &Volume,
    vcp: u16,
    template: Option<(
        &Option<VolumeDataBlock>,
        &Option<ElevationDataBlock>,
        &Option<RadialDataBlock>,
    )>,
) -> SweepConstants {
    let calibration = volume.radar_calibration.first();
    let location = volume.location;
    let latitude = location.latitude_deg.unwrap_or(0.0) as f32;
    let longitude = location.longitude_deg.unwrap_or(0.0) as f32;
    let dbz0 = calibration
        .and_then(|calibration| calibration.base_1km_hc_dbz)
        .unwrap_or(0.0);
    let mut vol = VolumeDataBlock {
        block_size: 52,
        version_major: 3,
        version_minor: 0,
        latitude_deg: latitude,
        longitude_deg: longitude,
        site_height_m: 0,
        feedhorn_height_m: 0,
        calibration_constant_db: dbz0,
        horizontal_shv_tx_power_kw: 0.0,
        vertical_shv_tx_power_kw: 0.0,
        system_differential_reflectivity_db: calibration
            .and_then(|calibration| calibration.zdr_correction_db)
            .unwrap_or(0.0),
        initial_system_differential_phase_deg: calibration
            .and_then(|calibration| calibration.system_phidp_deg)
            .unwrap_or(0.0),
        vcp_number: vcp,
        processing_status: ProcessingStatus(0),
        zdr_bias_estimate_raw: Some(0),
    };
    let mut elevation = ElevationDataBlock {
        block_size: 12,
        atmospheric_attenuation_raw: 0,
        calibration_constant_db: dbz0,
    };
    let mut radial = RadialDataBlock {
        block_size: 28,
        unambiguous_range_raw: 0,
        horizontal_noise_level_dbm: calibration
            .and_then(|calibration| calibration.noise_hc_dbm)
            .unwrap_or(0.0),
        vertical_noise_level_dbm: calibration
            .and_then(|calibration| calibration.noise_vc_dbm)
            .unwrap_or(0.0),
        nyquist_velocity_raw: 0,
        radial_flags: 0,
        horizontal_calibration_constant_dbz: Some(dbz0),
        vertical_calibration_constant_dbz: Some(
            calibration
                .and_then(|calibration| calibration.base_1km_vc_dbz)
                .unwrap_or(dbz0),
        ),
    };
    if let Some((source_vol, source_elv, source_rad)) = template {
        if let Some(source) = source_vol {
            vol = *source;
        }
        if let Some(source) = source_elv {
            elevation = *source;
        }
        if let Some(source) = source_rad {
            radial = *source;
        }
    }
    fit_volume_block(&mut vol, volume, vcp);
    SweepConstants {
        volume: vol,
        elevation,
        radial,
    }
}

/// Location and VCP always follow the volume being written; the source
/// blocks already agree for an unmodified Level II volume.
fn fit_volume_block(vol: &mut VolumeDataBlock, volume: &Volume, vcp: u16) {
    let location = volume.location;
    let latitude = location.latitude_deg.unwrap_or(0.0) as f32;
    let longitude = location.longitude_deg.unwrap_or(0.0) as f32;
    if f64::from(vol.latitude_deg) != f64::from(latitude)
        || f64::from(vol.longitude_deg) != f64::from(longitude)
    {
        vol.latitude_deg = latitude;
        vol.longitude_deg = longitude;
    }
    let block_altitude = f64::from(f32::from(vol.site_height_m) + f32::from(vol.feedhorn_height_m));
    if let Some(altitude) = location.altitude_m
        && block_altitude != altitude
    {
        vol.feedhorn_height_m = 0;
        vol.site_height_m = altitude
            .round()
            .clamp(f64::from(i16::MIN), f64::from(i16::MAX)) as i16;
    }
    vol.vcp_number = vcp;
}

/// Each radial's own blocks and header items from the source metadata, with
/// the location and VCP of the volume; a block the radial lacked is the
/// sweep's.
fn ray_constants(
    volume: &Volume,
    vcp: u16,
    constants: &SweepConstants,
    radials: &[crate::RadialConstants],
) -> Vec<RayConstants> {
    radials
        .iter()
        .map(|source| {
            let mut vol = source.volume.unwrap_or(constants.volume);
            fit_volume_block(&mut vol, volume, vcp);
            RayConstants {
                volume: vol,
                elevation: source.elevation.unwrap_or(constants.elevation),
                radial: source.radial.unwrap_or(constants.radial),
                radar_identifier: source.radar_identifier,
                azimuth_number: source.azimuth_number,
                spare: source.spare,
                azimuth_resolution_code: source.azimuth_resolution_code,
                radial_status_code: source.radial_status_code,
                cut_sector_number: source.cut_sector_number,
                spot_blanking: source.spot_blanking.0,
                azimuth_indexing_raw: source.azimuth_indexing_raw,
            }
        })
        .collect()
}

/// Values of the synthesised adaptation data.
fn adaptation(volume: &Volume) -> Adaptation {
    let parameters = &volume.radar_parameters;
    Adaptation {
        latitude_deg: volume.location.latitude_deg.unwrap_or(0.0),
        longitude_deg: volume.location.longitude_deg.unwrap_or(0.0),
        frequency_mhz: parameters
            .frequency_hz
            .first()
            .map(|hz| (hz / 1e6).round())
            .filter(|mhz| mhz.is_finite() && *mhz > 0.0 && *mhz <= f64::from(i32::MAX))
            .map(|mhz| mhz as i32),
        antenna_gain_db: parameters.antenna_gain_h_db.filter(|gain| gain.is_finite()),
        beam_width_deg: parameters
            .beam_width_h_deg
            .filter(|width| width.is_finite()),
    }
}

/// The source metadata's constant blocks of `sweep` (index `index` in the
/// volume being written): the entry of the same sweep index when it has the
/// sweep's elevation number, else the one entry with that elevation number
/// (the volume leaves out or reorders the source's sweeps), else none.
fn source_blocks<'m>(
    blocks: &'m [crate::SweepElevationData],
    index: usize,
    sweep: &Sweep,
) -> Option<&'m crate::SweepElevationData> {
    let by_index = blocks.iter().find(|blocks| blocks.sweep_index == index);
    let Some(number) = sweep.elevation_number else {
        return by_index;
    };
    let same_number =
        |blocks: &&crate::SweepElevationData| u16::from(blocks.elevation_number) == number;
    by_index.filter(same_number).or_else(|| {
        let mut matching = blocks.iter().filter(same_number);
        let first = matching.next();
        if matching.next().is_some() {
            None
        } else {
            first
        }
    })
}

/// Level II readers number cuts 1 to n (Py-ART groups radials by elevation
/// numbers 1 to the largest, MetPy adds an empty sweep for a missing
/// number), so a volume that leaves out or reorders its source's cuts is
/// numbered again; the note lists the sweeps whose number changed. Only
/// Level II sources carry ICD elevation numbers (CfRadial's decoder keeps
/// its 0-based sweep index there).
fn renumbering_note(sweeps: &[SweepPlan<'_>], summary: &mut WriteSummary) {
    let changed: Vec<String> = sweeps
        .iter()
        .filter_map(|plan| {
            let source = plan.sweep.elevation_number?;
            (source != u16::from(plan.elevation_number)).then(|| {
                format!(
                    "sweep {} ({source} to {})",
                    plan.index, plan.elevation_number
                )
            })
        })
        .collect();
    if !changed.is_empty() {
        summary.notes.push(format!(
            "elevation numbers written 1 to {} in sweep order, as readers number cuts: {}",
            sweeps.len(),
            changed.join(", ")
        ));
    }
}

/// Offset of a Message 5 body's pattern number, Message 2's VCP halfword and
/// Message 18's site name within their bodies (Tables XI, IV and XV).
const VCP_PATTERN_OFFSET: usize = 4;
const RDA_STATUS_VCP_OFFSET: usize = 14;
const ADAPTATION_SITE_OFFSET: usize = 8368;
/// CTM and message header before a message body in a frame.
const FRAME_HEADER_BYTES: usize = 28;
/// Table XI header halfwords before the cuts.
const VCP_HEADER_HALFWORDS: usize = 11;

/// The body bytes of the message in a fixed frame (by its size halfword,
/// within the frame).
fn frame_body(frame: &[u8]) -> &[u8] {
    let size = usize::from(u16::from_be_bytes([frame[12], frame[13]])) * 2;
    &frame[FRAME_HEADER_BYTES..(12 + size).clamp(FRAME_HEADER_BYTES, frame.len())]
}

/// Zero-based frames of `record` holding a message of `message_type`.
fn frames_of(record: &[u8], message_type: u8) -> impl Iterator<Item = (usize, &[u8])> {
    record
        .chunks_exact(FRAME_BYTES)
        .enumerate()
        .filter(move |(_, frame)| {
            frame[15] == message_type && u16::from_be_bytes([frame[12], frame[13]]) != 0
        })
}

/// A checked source metadata record, with what must change in it to agree
/// with the written volume: its Message 5 cuts, the VCP of Messages 2 and 5
/// and the Message 18 site name. `None` when the record cannot be used.
fn carried_record<'a>(
    record: &'a [u8],
    sweeps: &[SweepPlan<'_>],
    vcp: u16,
    site: &[u8; 4],
    summary: &mut WriteSummary,
) -> Option<CarriedRecord<'a>> {
    let adaptation_frame = match short_adaptation_frames(record) {
        ShortAdaptation::None => None,
        ShortAdaptation::FourFrames(first) => {
            summary.notes.push(format!(
                "the source metadata record's Message 18 (frames {} to {}) is shorter than \
                 Table XV; the synthesised Message 18 replaces it",
                first + 1,
                first + 4
            ));
            Some(first)
        }
        ShortAdaptation::Other => {
            summary.notes.push(
                "the source metadata record's Message 18 is shorter than Table XV and not in \
                 four frames; synthesised messages 2, 5 and 18 written instead"
                    .to_owned(),
            );
            return None;
        }
    };
    let mut patches = Vec::new();
    let vcp_edit = vcp_edit(record, sweeps, vcp, summary, &mut patches);

    // Message 2: the VCP halfword (negative for a locally selected pattern).
    let mut status_patched = false;
    for (index, frame) in frames_of(record, 2) {
        let body = frame_body(frame);
        let Some(bytes) = body.get(RDA_STATUS_VCP_OFFSET..RDA_STATUS_VCP_OFFSET + 2) else {
            continue;
        };
        let selection = crate::messages::rda_status::VcpSelection::from_code(u16::from_be_bytes([
            bytes[0], bytes[1],
        ]));
        if selection.pattern().unwrap_or(0) == vcp {
            continue;
        }
        let magnitude = i16::try_from(vcp).unwrap_or(i16::MAX);
        let code = match selection {
            crate::messages::rda_status::VcpSelection::Local(_) => -magnitude,
            _ => magnitude,
        };
        let at = index * FRAME_BYTES + FRAME_HEADER_BYTES + RDA_STATUS_VCP_OFFSET;
        patches.push((at, code.to_be_bytes().to_vec()));
        status_patched = true;
    }
    if status_patched {
        summary.notes.push(format!(
            "the source metadata record's Message 2 names another VCP; written with VCP {vcp}"
        ));
    }

    // Message 18: the site name, unless the synthesised one replaces it.
    if adaptation_frame.is_none() {
        let mut body_offset = 0;
        let mut positions = Vec::new();
        for (index, frame) in frames_of(record, 18) {
            let body = frame_body(frame);
            for at in ADAPTATION_SITE_OFFSET..ADAPTATION_SITE_OFFSET + 4 {
                if let Some(within) = at.checked_sub(body_offset).filter(|w| *w < body.len()) {
                    positions.push((
                        index * FRAME_BYTES + FRAME_HEADER_BYTES + within,
                        body[within],
                    ));
                }
            }
            body_offset += body.len();
        }
        let current: Vec<u8> = positions.iter().map(|(_, byte)| *byte).collect();
        if positions.len() == 4 && current != site {
            for ((at, _), byte) in positions.iter().zip(site) {
                patches.push((*at, vec![*byte]));
            }
            summary.notes.push(format!(
                "the source metadata record's Message 18 names site {:?}; written with {:?}",
                String::from_utf8_lossy(&current),
                String::from_utf8_lossy(site)
            ));
        }
    }
    Some(CarriedRecord {
        bytes: record,
        vcp: vcp_edit,
        adaptation_frame,
        patches,
    })
}

/// What becomes of the record's Message 5: readers index its cuts by the
/// written elevation numbers (1 to n in sweep order), so it must list the
/// written sweeps' cuts in that order and name the written VCP.
fn vcp_edit(
    record: &[u8],
    sweeps: &[SweepPlan<'_>],
    vcp: u16,
    summary: &mut WriteSummary,
    patches: &mut Vec<(usize, Vec<u8>)>,
) -> VcpEdit {
    let Some((frame, body)) = frames_of(record, 5)
        .next()
        .map(|(index, frame)| (index, frame_body(frame)))
    else {
        return VcpEdit::Keep;
    };
    let synthesise = |summary: &mut WriteSummary, why: &str| {
        summary.notes.push(format!(
            "the source metadata record's Message 5 (frame {}) {why}; the synthesised Message 5 \
             replaces it",
            frame + 1
        ));
        VcpEdit::Synthesise { frame }
    };
    let pattern = match VolumeCoveragePattern::decode(body) {
        Ok(pattern) => pattern,
        Err(_) => return synthesise(summary, "does not decode"),
    };
    // The source cut of each written sweep: its own elevation number, else
    // its position.
    let wanted: Vec<usize> = sweeps
        .iter()
        .enumerate()
        .map(|(position, plan)| {
            plan.sweep
                .elevation_number
                .map_or(position + 1, usize::from)
        })
        .collect();
    let cuts = pattern.cuts.len();
    if wanted.iter().any(|number| *number == 0 || *number > cuts) {
        return synthesise(
            summary,
            &format!("has {cuts} cuts, not one for every written sweep"),
        );
    }
    if wanted
        .iter()
        .enumerate()
        .all(|(position, number)| *number == position + 1)
    {
        if pattern.pattern_number != vcp {
            let at = frame * FRAME_BYTES + FRAME_HEADER_BYTES + VCP_PATTERN_OFFSET;
            patches.push((at, vcp.to_be_bytes().to_vec()));
            summary.notes.push(format!(
                "the source metadata record's Message 5 names VCP {}; written with VCP {vcp}",
                pattern.pattern_number
            ));
        }
        return VcpEdit::Keep;
    }
    // The decoder accepted the sizes: the cuts fill the message after its
    // header, at least 23 halfwords each.
    let cut_halfwords = usize::from(pattern.message_size).saturating_sub(VCP_HEADER_HALFWORDS)
        / usize::from(pattern.number_of_cuts).max(1);
    let cut_len = cut_halfwords * 2;
    if (VCP_HEADER_HALFWORDS * 2 + cut_len * wanted.len()) > FRAME_BYTES - FRAME_HEADER_BYTES {
        return synthesise(summary, "cannot list the written cuts in one frame");
    }
    summary.notes.push(format!(
        "the source metadata record's Message 5 lists the cuts of elevation numbers {}; they \
         are listed again in the order of the written sweeps",
        wanted
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    ));
    VcpEdit::Reindex {
        frame,
        cuts: wanted.iter().map(|number| number - 1).collect(),
        cut_len,
    }
}

/// A carried-over record's Message 18 that no reader can unpack.
enum ShortAdaptation {
    /// None, or one of Table XV's full length.
    None,
    /// A short one in four frames from this zero-based frame.
    FourFrames(usize),
    /// A short one in some other number of frames.
    Other,
}

/// Find a Message 18 whose joined body is shorter than Table XV's 9468
/// bytes in `record` (fixed frames).
fn short_adaptation_frames(record: &[u8]) -> ShortAdaptation {
    let frames: Vec<(usize, usize)> = record
        .chunks_exact(FRAME_BYTES)
        .enumerate()
        .filter(|(_, frame)| frame[15] == 18)
        .map(|(index, frame)| {
            let size = usize::from(u16::from_be_bytes([frame[12], frame[13]])) * 2;
            (index, size.saturating_sub(16))
        })
        .filter(|(_, body)| *body > 0)
        .collect();
    let body: usize = frames.iter().map(|(_, body)| body).sum();
    if frames.is_empty() || body >= crate::messages::adaptation::RDA_ADAPTATION_DATA_LEN {
        return ShortAdaptation::None;
    }
    let consecutive = frames.windows(2).all(|pair| pair[1].0 == pair[0].0 + 1);
    match frames.first() {
        Some((first, _)) if frames.len() == 4 && consecutive => ShortAdaptation::FourFrames(*first),
        _ => ShortAdaptation::Other,
    }
}

/// Most bytes of data messages carried over (a volume's RDA status updates
/// take a few frames).
const MAX_DATA_MESSAGE_BYTES: usize = 16 << 20;

/// Data messages passed through from a Level II source: whole fixed frames
/// of non-radial messages that fit their frames, ordered by their places
/// among the radials.
fn checked_data_messages(messages: &[DataMessage]) -> Result<Vec<&DataMessage>, WriteError> {
    let total = messages.iter().fold(0usize, |sum, message| {
        sum.saturating_add(message.frames.len())
    });
    if total > MAX_DATA_MESSAGE_BYTES {
        return Err(WriteError::LimitExceeded(format!(
            "{total} bytes of data messages, more than {MAX_DATA_MESSAGE_BYTES}"
        )));
    }
    for (index, message) in messages.iter().enumerate() {
        let error = |reason: String| WriteError::DataMessage { index, reason };
        if message.frames.is_empty() || !message.frames.len().is_multiple_of(FRAME_BYTES) {
            return Err(error(format!(
                "{} bytes is not a whole number of {FRAME_BYTES}-byte frames",
                message.frames.len()
            )));
        }
        for frame in message.frames.chunks_exact(FRAME_BYTES) {
            let message_type = frame[15];
            let size = usize::from(u16::from_be_bytes([frame[12], frame[13]])) * 2;
            if matches!(message_type, 0 | 1 | 31) {
                return Err(error(format!(
                    "message type {message_type} is not a non-radial message"
                )));
            }
            if !(16..=FRAME_BYTES - 12).contains(&size) {
                return Err(error(format!(
                    "a frame declares {size} bytes; a message fills 16 to {} bytes of its frame",
                    FRAME_BYTES - 12
                )));
            }
        }
    }
    let mut ordered: Vec<&DataMessage> = messages.iter().collect();
    ordered.sort_by_key(|message| message.after_radials);
    Ok(ordered)
}

/// Use a metadata record passed through from a Level II source: it must
/// frame, and it must not hold radials (files from before 2005 have radials
/// in those frames; they are replaced by synthesised messages).
fn checked_metadata_record<'a>(
    record: &'a [u8],
    summary: &mut WriteSummary,
) -> Result<Option<&'a [u8]>, WriteError> {
    if record.is_empty() {
        summary.notes.push(
            "the source metadata record is empty; synthesised messages 2, 5 and 18 written"
                .to_owned(),
        );
        return Ok(None);
    }
    let mut walker = RawMessages::new(record);
    let mut radials = false;
    let mut unframed = None;
    for item in walker.by_ref() {
        match item {
            Ok(message) => {
                if matches!(message.header.message_type, 1 | 31) {
                    radials = true;
                }
            }
            // Stale segments and zero-filled messages of real records are
            // reported by the walker but framed; only an unframed record is
            // refused.
            Err(error @ crate::NexradError::Truncated { .. }) => {
                unframed.get_or_insert(error.to_string());
            }
            Err(_) => {}
        }
    }
    if radials {
        summary.notes.push(
            "the source metadata record holds radials (files from before 2005, some converted feeds); synthesised messages 2, 5 and 18 written instead"
                .to_owned(),
        );
        return Ok(None);
    }
    if let Some(error) = unframed {
        return Err(WriteError::MetadataRecord(error));
    }
    // Without radials the record is fixed 2432-byte frames, each message
    // fitting its frame.
    if !record.len().is_multiple_of(FRAME_BYTES) {
        return Err(WriteError::MetadataRecord(format!(
            "{} bytes is not a whole number of {FRAME_BYTES}-byte frames",
            record.len()
        )));
    }
    for (index, frame) in record.chunks_exact(FRAME_BYTES).enumerate() {
        let size = u16::from_be_bytes([frame[12], frame[13]]);
        if usize::from(size) * 2 > FRAME_BYTES - 12 {
            return Err(WriteError::MetadataRecord(format!(
                "frame {} declares {size} halfwords, more than a frame holds",
                index + 1
            )));
        }
    }
    if walker.position() != record.len() {
        return Err(WriteError::MetadataRecord(format!(
            "frames end at byte {} of {}",
            walker.position(),
            record.len()
        )));
    }
    Ok(Some(record))
}
