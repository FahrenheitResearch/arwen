//! Byte layout of the written archive: the volume header, fixed 2432-byte
//! metadata frames, Message 31 radials and the synthesised Messages 2, 5 and
//! 18 (ICD 2620002AA Tables II, IV, XI, XV and XVII; ICD 2620010 Table I).

use super::plan::{Adaptation, CarriedRecord, NexradTime, Plan, SweepPlan, VcpEdit};
use super::quantize;
use super::{RecordLayout, WriteError};
use crate::messages::msg31_blocks::{
    ELEVATION_BLOCK_LEN, ElevationDataBlock, RADIAL_BLOCK_CALIBRATED_LEN, RADIAL_BLOCK_LEN,
    RadialDataBlock, VOLUME_BLOCK_LEN, VOLUME_BLOCK_ZDR_BIAS_LEN, VolumeDataBlock,
};

/// Archive II volume header length.
pub(crate) const VOLUME_HEADER_LEN: usize = 24;
/// Communications-manager header before every message.
pub(crate) const CTM_LEN: usize = 12;
/// Message header (Table II).
const MESSAGE_HEADER_LEN: usize = 16;
/// Fixed frame of a non-radial message.
pub(crate) const FRAME_LEN: usize = 2432;
/// Offset of the radial status byte in a Message 31 frame: the CTM and
/// message headers, then Table XVII-A byte 21.
pub(crate) const RADIAL_STATUS_OFFSET: usize = CTM_LEN + MESSAGE_HEADER_LEN + 21;
/// Largest message body in one fixed frame.
const FRAME_BODY_LEN: usize = FRAME_LEN - CTM_LEN - MESSAGE_HEADER_LEN;
/// Frames of the Archive II metadata record.
const METADATA_FRAMES: usize = 134;
/// Data Header Block with ten block pointers (Build 19 on).
const DATA_HEADER_LEN: usize = 72;
/// Block pointer slots of the Data Header Block.
const POINTER_SLOTS: usize = 10;
/// Data moment block header.
const MOMENT_HEADER_LEN: usize = 28;
/// RDA redundant channel byte: Open RDA, single channel (bit 3 set, as
/// every current WSR-88D records; readers pick the ORDA layouts of
/// Messages 2 and 18 from it).
const CHANNELS_ORDA: u8 = 8;
/// Message 2 body: 60 halfwords (Build 18 on).
const RDA_STATUS_HALFWORDS: usize = 60;
/// Message 18 body (Table XV).
const ADAPTATION_LEN: usize = crate::messages::adaptation::RDA_ADAPTATION_DATA_LEN;
/// Bytes of each Message 18 segment body as real files split it.
const ADAPTATION_SEGMENT_LEN: usize = 2400;
/// Zero-based metadata-record frames of the messages, as real files place
/// them: Message 18 in frames 127 to 130, Message 5 in 133, Message 2 in 134.
const ADAPTATION_FRAME: usize = 126;
const VCP_FRAME: usize = 132;
const RDA_STATUS_FRAME: usize = 133;
/// RDA build number in Message 2 (20.00): the first build whose layouts
/// match what the writer produces (72-byte Data Header Block, 52-byte VOL,
/// 28-byte RAD blocks).
const RDA_BUILD_RAW: u16 = 2000;
/// Message sequence numbers wrap at 0x7FFF (Table II).
const SEQUENCE_MODULUS: u32 = 0x8000;

/// The encoded archive before compression.
pub(crate) struct Encoded {
    pub header: [u8; VOLUME_HEADER_LEN],
    /// The metadata record, then the radial records.
    pub records: Vec<Vec<u8>>,
}

/// Sequence number of the first radial: the metadata messages take 1 to 3.
pub(crate) const FIRST_RADIAL_SEQUENCE: u32 = 4;

/// Encode the planned archive.
pub(crate) fn encode(plan: &Plan<'_>) -> Result<Encoded, WriteError> {
    let header = volume_header(plan);
    let mut records = vec![metadata_record_bytes(plan)];
    let context = RadialContext {
        opens_volume: true,
        last_cut: plan.sweeps.last().map_or(0, |sweep| sweep.elevation_number),
        ends_volume: true,
    };
    let mut sequence = FIRST_RADIAL_SEQUENCE;
    records.extend(
        radial_records(plan, &context, &mut sequence, plan.radials_per_record)?
            .into_iter()
            .map(|record| record.bytes),
    );
    Ok(Encoded { header, records })
}

/// The metadata record: the source's with the plan's edits, else the
/// synthesised one.
pub(crate) fn metadata_record_bytes(plan: &Plan<'_>) -> Vec<u8> {
    match &plan.metadata_record {
        Some(carried) => carried_record(plan, carried),
        None => metadata_record(plan),
    }
}

/// Where a plan's radials sit in their volume, which sets their statuses
/// (Table III-C).
pub(crate) struct RadialContext {
    /// The plan's first radial opens the volume.
    pub opens_volume: bool,
    /// Elevation number of the volume's last cut: its first radial has
    /// status 5.
    pub last_cut: u8,
    /// The plan's last radial ends the volume.
    pub ends_volume: bool,
}

/// One record of radials, uncompressed.
#[derive(Clone, Default)]
pub(crate) struct RadialRecord {
    pub bytes: Vec<u8>,
    /// Radials in it.
    pub radials: usize,
    /// Offset of its last radial's status byte.
    pub last_status: usize,
}

/// Append `bytes` to `out`, a failed allocation being a typed error.
fn try_extend(out: &mut Vec<u8>, bytes: &[u8]) -> Result<(), WriteError> {
    out.try_reserve(bytes.len())
        .map_err(|err| WriteError::LimitExceeded(format!("record buffer: {err}")))?;
    out.extend_from_slice(bytes);
    Ok(())
}

/// The plan's radials in records of `radials_per_record` radials, the first
/// one holding at most `first_room` (the rest of a record the real-time
/// writer continues), with the source's data messages among them where the
/// source had them. Under [`RecordLayout::Continuous`] records run on across
/// cuts; under [`RecordLayout::WithinCuts`] each cut starts a record, as
/// NOAA's LDM files and real-time chunks have it (their cuts are 360 or 720
/// radials, so the two agree there). `sequence` is the next message sequence
/// number, and is advanced.
pub(crate) fn radial_records(
    plan: &Plan<'_>,
    context: &RadialContext,
    sequence: &mut u32,
    first_room: usize,
) -> Result<Vec<RadialRecord>, WriteError> {
    let mut records = Vec::new();
    for_each_radial_record(plan, context, sequence, first_room, |record| {
        records.push(record);
        Ok(())
    })?;
    Ok(records)
}

/// [`radial_records`], handing each record to `emit` as soon as it is
/// complete instead of collecting them, so that a streaming writer holds one
/// record at a time.
pub(crate) fn for_each_radial_record(
    plan: &Plan<'_>,
    context: &RadialContext,
    sequence: &mut u32,
    first_room: usize,
    mut emit: impl FnMut(RadialRecord) -> Result<(), WriteError>,
) -> Result<(), WriteError> {
    let per_record = plan.radials_per_record.max(1);
    let mut room = first_room.clamp(1, per_record);
    let mut current = RadialRecord::default();
    let mut messages = plan.data_messages.iter().peekable();
    let mut radial = 0usize;
    for (index, sweep) in plan.sweeps.iter().enumerate() {
        if plan.record_layout == RecordLayout::WithinCuts && current.radials > 0 {
            emit(std::mem::take(&mut current))?;
            room = per_record;
        }
        for (position, &ray) in sweep.order.iter().enumerate() {
            if current.radials == room {
                emit(std::mem::take(&mut current))?;
                room = per_record;
            }
            while let Some(message) = messages.next_if(|m| m.after_radials <= radial) {
                try_extend(&mut current.bytes, &message.frames)?;
            }
            let status = radial_status(plan, context, index, position, ray);
            current
                .bytes
                .try_reserve(CTM_LEN + MESSAGE_HEADER_LEN + radial_len(sweep, ray))
                .map_err(|err| WriteError::LimitExceeded(format!("record buffer: {err}")))?;
            current.last_status = current.bytes.len() + RADIAL_STATUS_OFFSET;
            encode_radial(
                plan,
                sweep,
                position,
                ray,
                status,
                *sequence as u16,
                &mut current.bytes,
            );
            current.radials += 1;
            *sequence = (*sequence + 1) % SEQUENCE_MODULUS;
            radial += 1;
        }
    }
    // Messages after the last radial end the last record.
    for message in messages {
        try_extend(&mut current.bytes, &message.frames)?;
    }
    if !current.bytes.is_empty() {
        emit(current)?;
    }
    Ok(())
}

/// A carried-over metadata record with the plan's edits: its Message 5
/// replaced or its cuts listed again, a short Message 18 replaced, and the
/// VCP and site patches.
fn carried_record(plan: &Plan<'_>, carried: &CarriedRecord<'_>) -> Vec<u8> {
    let mut record = padded_record(carried.bytes);
    let replace = |record: &mut Vec<u8>, first: usize, frames: &[u8]| {
        if let Some(target) = record.get_mut(first * FRAME_LEN..first * FRAME_LEN + frames.len()) {
            target.copy_from_slice(frames);
        }
    };
    let synthesised_vcp = |out: &mut Vec<u8>| {
        fixed_frame(out, 5, 3, plan.header_time, 1, 1, &vcp_body(plan));
    };
    match &carried.vcp {
        VcpEdit::Keep => {}
        VcpEdit::Synthesise { frame } => {
            let mut replacement = Vec::with_capacity(FRAME_LEN);
            synthesised_vcp(&mut replacement);
            replace(&mut record, *frame, &replacement);
        }
        VcpEdit::Reindex {
            frame,
            cuts,
            cut_len,
        } => {
            let replacement = reindexed_vcp_frame(plan, carried.bytes, *frame, cuts, *cut_len)
                .unwrap_or_else(|| {
                    let mut synthesised = Vec::with_capacity(FRAME_LEN);
                    synthesised_vcp(&mut synthesised);
                    synthesised
                });
            replace(&mut record, *frame, &replacement);
        }
    }
    if let Some(first) = carried.adaptation_frame {
        let mut replacement = Vec::with_capacity(4 * FRAME_LEN);
        adaptation_frames(plan, &mut replacement);
        replace(&mut record, first, &replacement);
    }
    for (at, bytes) in &carried.patches {
        if let Some(target) = record.get_mut(*at..*at + bytes.len()) {
            target.copy_from_slice(bytes);
        }
    }
    record
}

/// The source's Message 5 frame with its cuts listed again: the header as
/// the source has it with the written VCP, cut count and sizes, then the
/// source cut of each written sweep in order. `None` when the frame is not
/// where the plan found it.
fn reindexed_vcp_frame(
    plan: &Plan<'_>,
    record: &[u8],
    frame: usize,
    cuts: &[usize],
    cut_len: usize,
) -> Option<Vec<u8>> {
    let source = record.get(frame * FRAME_LEN..(frame + 1) * FRAME_LEN)?;
    let body = source.get(CTM_LEN + MESSAGE_HEADER_LEN..)?;
    let body_halfwords = 11 + cut_len / 2 * cuts.len();
    let mut out = Vec::with_capacity(FRAME_LEN);
    out.extend_from_slice(source.get(..CTM_LEN + MESSAGE_HEADER_LEN)?);
    out[CTM_LEN..CTM_LEN + 2]
        .copy_from_slice(&u16::try_from(8 + body_halfwords).ok()?.to_be_bytes());
    let header_start = out.len();
    out.extend_from_slice(body.get(..22)?);
    out[header_start..header_start + 2]
        .copy_from_slice(&u16::try_from(body_halfwords).ok()?.to_be_bytes());
    out[header_start + 4..header_start + 6].copy_from_slice(&plan.vcp.to_be_bytes());
    out[header_start + 6..header_start + 8]
        .copy_from_slice(&u16::try_from(cuts.len()).ok()?.to_be_bytes());
    for cut in cuts {
        let at = 22 + cut * cut_len;
        out.extend_from_slice(body.get(at..at + cut_len)?);
    }
    (out.len() <= FRAME_LEN).then(|| {
        out.resize(FRAME_LEN, 0);
        out
    })
}

/// A carried-over metadata record, padded with empty frames to the 134
/// frames of Archive II when it is shorter (trimmed fixtures and some
/// converted files carry only a few), so every reader frames the radials
/// after it.
fn padded_record(record: &[u8]) -> Vec<u8> {
    let mut padded = record.to_vec();
    if padded.len() < METADATA_FRAMES * FRAME_LEN && padded.len().is_multiple_of(FRAME_LEN) {
        padded.resize(METADATA_FRAMES * FRAME_LEN, 0);
    }
    padded
}

/// Table I: `AR2V00nn.NNN`, the date (day 1 is 1970-01-01), milliseconds
/// past midnight and the ICAO.
pub(crate) fn volume_header(plan: &Plan<'_>) -> [u8; VOLUME_HEADER_LEN] {
    let mut header = [0u8; VOLUME_HEADER_LEN];
    let extension = format!(".{:03}", plan.volume_number % 1000);
    header[..8].copy_from_slice(&plan.tape);
    header[8..12].copy_from_slice(&extension.as_bytes()[..4]);
    header[12..16].copy_from_slice(&u32::from(plan.header_time.date).to_be_bytes());
    header[16..20].copy_from_slice(&plan.header_time.ms.to_be_bytes());
    header[20..24].copy_from_slice(&plan.site);
    header
}

/// Radial status (Table III-C) of the radial at `position` in its cut,
/// source ray `ray`: 3 opens the volume, 0 an elevation, 5 the last
/// elevation; 2 ends an elevation, 4 the volume; 1 in between. A one-radial
/// cut keeps its opening status.
fn radial_status(
    plan: &Plan<'_>,
    context: &RadialContext,
    sweep_index: usize,
    position: usize,
    ray: usize,
) -> u8 {
    let sweep = &plan.sweeps[sweep_index];
    let opens_volume = context.opens_volume && sweep_index == 0;
    let last_cut = sweep.elevation_number == context.last_cut;
    let ends_volume = context.ends_volume && sweep_index + 1 == plan.sweeps.len();
    let last_ray = position + 1 == sweep.order.len();
    let status = match (position == 0, opens_volume, last_cut, last_ray, ends_volume) {
        (true, true, _, _, _) => 3,
        (true, false, true, _, _) => 5,
        (true, false, false, _, _) => 0,
        (false, _, _, true, true) => 4,
        (false, _, _, true, false) => 2,
        _ => 1,
    };
    // The source radial's own code where it plays the same part: a cut
    // start as 0 or 5 (the RDA writes 0 for a last cut it did not plan as
    // last, as AVSET does), and codes outside Table III-C mid-cut. Volume
    // start and cut and volume ends follow the written sweeps.
    let Some(source) = sweep.rays.get(ray).map(|own| own.radial_status_code) else {
        return status;
    };
    let same_part = match status {
        0 | 5 => matches!(source, 0 | 5),
        1 => !matches!(source, 0 | 2 | 3 | 4 | 5),
        _ => false,
    };
    if same_part { source } else { status }
}

/// Block offsets of the Message 31 body of ray `ray` and its length.
///
/// Blocks follow the 72-byte Data Header Block in pointer order: VOL, ELV,
/// RAD, then the moments the ray carries. A block of odd length (8-bit gates,
/// odd gate count) is followed by a pad byte, so blocks start on halfwords,
/// except before the last block: an odd last block starts one byte late
/// instead, so it ends the body. The message size counts halfwords, and the
/// `nexrad` crate reads the next message where the last block ends.
struct RadialLayout {
    pointers: Vec<u32>,
    len: usize,
}

fn radial_layout(sweep: &SweepPlan<'_>, ray: usize) -> RadialLayout {
    let (volume, _, radial) = sweep.blocks(ray);
    let mut lens = vec![
        volume_block_len(volume),
        ELEVATION_BLOCK_LEN,
        radial_block_len(radial),
    ];
    lens.extend(
        sweep
            .moments
            .iter()
            .filter(|moment| !moment.field.is_absent(ray))
            .map(|moment| moment.block_len()),
    );
    let mut pointers = Vec::with_capacity(lens.len());
    let mut offset = DATA_HEADER_LEN;
    let last = lens.len() - 1;
    for (index, len) in lens.iter().enumerate() {
        if index == last {
            offset += len % 2;
        }
        pointers.push(offset as u32);
        offset += len;
        if index != last {
            offset += len % 2;
        }
    }
    RadialLayout {
        pointers,
        len: offset,
    }
}

/// Bytes of the Message 31 body of ray `ray`: the Data Header Block, the
/// constant blocks and the moments the ray carries.
pub(crate) fn radial_len(sweep: &SweepPlan<'_>, ray: usize) -> usize {
    radial_layout(sweep, ray).len
}

fn volume_block_len(block: &VolumeDataBlock) -> usize {
    if block.zdr_bias_estimate_raw.is_some() {
        VOLUME_BLOCK_ZDR_BIAS_LEN
    } else {
        VOLUME_BLOCK_LEN
    }
}

fn radial_block_len(block: &RadialDataBlock) -> usize {
    if block.horizontal_calibration_constant_dbz.is_some() {
        RADIAL_BLOCK_CALIBRATED_LEN
    } else {
        RADIAL_BLOCK_LEN
    }
}

#[allow(clippy::too_many_arguments)]
fn message_header(
    out: &mut Vec<u8>,
    size_halfwords: u16,
    channels: u8,
    message_type: u8,
    sequence: u16,
    time: NexradTime,
    segments: u16,
    segment: u16,
) {
    out.extend_from_slice(&size_halfwords.to_be_bytes());
    out.push(channels);
    out.push(message_type);
    out.extend_from_slice(&sequence.to_be_bytes());
    out.extend_from_slice(&time.date.to_be_bytes());
    out.extend_from_slice(&time.ms.to_be_bytes());
    out.extend_from_slice(&segments.to_be_bytes());
    out.extend_from_slice(&segment.to_be_bytes());
}

/// One Message 31 frame: CTM header, message header, radial. `ray` is the
/// source ray, `position` its place in the written cut.
#[allow(clippy::too_many_arguments)]
fn encode_radial(
    plan: &Plan<'_>,
    sweep: &SweepPlan<'_>,
    position: usize,
    ray: usize,
    status: u8,
    sequence: u16,
    out: &mut Vec<u8>,
) {
    let RadialLayout {
        pointers,
        len: body_len,
    } = radial_layout(sweep, ray);
    // The planner checked that every per-ray array has one entry per ray;
    // the fallbacks below are never taken.
    let time = sweep.times.get(ray).copied().unwrap_or(plan.header_time);
    out.extend_from_slice(&[0u8; CTM_LEN]);
    let size_halfwords = ((MESSAGE_HEADER_LEN + body_len) / 2) as u16;
    // A Level II source radial's own channel byte and message generation
    // time, else the Open RDA's channel and the radial's collection time.
    let channels = sweep.channels.get(ray).copied().unwrap_or(CHANNELS_ORDA);
    let generated = sweep.message_times.get(ray).copied().unwrap_or(time);
    message_header(out, size_halfwords, channels, 31, sequence, generated, 1, 1);
    let body_start = out.len();
    let (volume, elevation_constants, radial_constants) = sweep.blocks(ray);
    // Radial number, spare, azimuth resolution, cut sector, spot blanking
    // and azimuth indexing: the source radial's, else the radial's place in
    // the cut from 1, the sweep's resolution, in sector 1. The site: the
    // source radial's own identifier unless the options name a site.
    let own = sweep.rays.get(ray);
    let site = match own {
        Some(own) if plan.keep_radial_sites => own.radar_identifier,
        _ => plan.site,
    };
    let (azimuth_number, spare, resolution, cut_sector, spot_blanking, indexing) = match own {
        Some(own) => (
            own.azimuth_number,
            own.spare,
            own.azimuth_resolution_code,
            own.cut_sector_number,
            own.spot_blanking,
            own.azimuth_indexing_raw,
        ),
        None => (
            ((position % usize::from(u16::MAX)) + 1) as u16,
            0,
            sweep.azimuth_resolution,
            1,
            0,
            sweep.azimuth_indexing_raw,
        ),
    };

    // Data Header Block (Table XVII-A).
    let rays = &sweep.sweep.rays;
    let azimuth = rays.azimuth_deg.get(ray).copied().unwrap_or_default();
    let elevation = rays.elevation_deg.get(ray).copied().unwrap_or_default();
    out.extend_from_slice(&site);
    out.extend_from_slice(&time.ms.to_be_bytes());
    out.extend_from_slice(&time.date.to_be_bytes());
    out.extend_from_slice(&azimuth_number.to_be_bytes());
    out.extend_from_slice(&azimuth.to_be_bytes());
    out.push(0); // compression indicator: uncompressed
    out.push(spare);
    out.extend_from_slice(&(body_len as u16).to_be_bytes());
    out.push(resolution);
    out.push(status);
    out.push(sweep.elevation_number);
    out.push(cut_sector);
    out.extend_from_slice(&elevation.to_be_bytes());
    out.push(spot_blanking);
    out.push(indexing);
    out.extend_from_slice(&(pointers.len() as u16).to_be_bytes());
    for slot in 0..POINTER_SLOTS {
        out.extend_from_slice(&pointers.get(slot).copied().unwrap_or(0).to_be_bytes());
    }

    let mut blocks = pointers.iter();
    let mut seek = |out: &mut Vec<u8>| {
        if let Some(pointer) = blocks.next() {
            out.resize(body_start + *pointer as usize, 0);
        }
    };
    seek(out);
    volume_block(volume, out);
    seek(out);
    elevation_block(elevation_constants, out);
    let mut radial = *radial_constants;
    radial.nyquist_velocity_raw = sweep.nyquist_raw.get(ray).copied().unwrap_or_default();
    radial.unambiguous_range_raw = sweep.unambiguous_raw.get(ray).copied().unwrap_or_default();
    seek(out);
    radial_block(&radial, out);

    for moment in &sweep.moments {
        if moment.field.is_absent(ray) {
            continue;
        }
        seek(out);
        let coding = moment.encoding.coding;
        out.push(b'D');
        out.extend_from_slice(&moment.moment.block_name());
        out.extend_from_slice(&0u32.to_be_bytes());
        out.extend_from_slice(&moment.gate_count.to_be_bytes());
        out.extend_from_slice(&moment.first_gate_m.to_be_bytes());
        out.extend_from_slice(&moment.gate_spacing_m.to_be_bytes());
        out.extend_from_slice(&moment.tover_raw.to_be_bytes());
        out.extend_from_slice(&moment.snr_threshold_raw.to_be_bytes());
        out.push(moment.control_flags);
        out.push(coding.word_size);
        out.extend_from_slice(&coding.scale.to_be_bytes());
        out.extend_from_slice(&coding.offset.to_be_bytes());
        let data_start = out.len();
        quantize::encode_row(moment.field, ray, moment.skip_gates, &moment.encoding, out);
        // Rows are the field's width; fill a short row.
        out.resize(data_start + moment.block_len() - MOMENT_HEADER_LEN, 0);
    }
    out.resize(body_start + body_len, 0);
    debug_assert_eq!(out.len() - body_start, body_len);
}

fn volume_block(block: &VolumeDataBlock, out: &mut Vec<u8>) {
    let len = volume_block_len(block);
    out.extend_from_slice(b"RVOL");
    out.extend_from_slice(&(len as u16).to_be_bytes());
    out.push(block.version_major);
    out.push(block.version_minor);
    out.extend_from_slice(&block.latitude_deg.to_be_bytes());
    out.extend_from_slice(&block.longitude_deg.to_be_bytes());
    out.extend_from_slice(&block.site_height_m.to_be_bytes());
    out.extend_from_slice(&block.feedhorn_height_m.to_be_bytes());
    out.extend_from_slice(&block.calibration_constant_db.to_be_bytes());
    out.extend_from_slice(&block.horizontal_shv_tx_power_kw.to_be_bytes());
    out.extend_from_slice(&block.vertical_shv_tx_power_kw.to_be_bytes());
    out.extend_from_slice(&block.system_differential_reflectivity_db.to_be_bytes());
    out.extend_from_slice(&block.initial_system_differential_phase_deg.to_be_bytes());
    out.extend_from_slice(&block.vcp_number.to_be_bytes());
    out.extend_from_slice(&block.processing_status.0.to_be_bytes());
    if let Some(bias) = block.zdr_bias_estimate_raw {
        out.extend_from_slice(&bias.to_be_bytes());
        out.extend_from_slice(&[0u8; 6]);
    }
}

fn elevation_block(block: &ElevationDataBlock, out: &mut Vec<u8>) {
    out.extend_from_slice(b"RELV");
    out.extend_from_slice(&(ELEVATION_BLOCK_LEN as u16).to_be_bytes());
    out.extend_from_slice(&block.atmospheric_attenuation_raw.to_be_bytes());
    out.extend_from_slice(&block.calibration_constant_db.to_be_bytes());
}

fn radial_block(block: &RadialDataBlock, out: &mut Vec<u8>) {
    let len = radial_block_len(block);
    out.extend_from_slice(b"RRAD");
    out.extend_from_slice(&(len as u16).to_be_bytes());
    out.extend_from_slice(&block.unambiguous_range_raw.to_be_bytes());
    out.extend_from_slice(&block.horizontal_noise_level_dbm.to_be_bytes());
    out.extend_from_slice(&block.vertical_noise_level_dbm.to_be_bytes());
    out.extend_from_slice(&block.nyquist_velocity_raw.to_be_bytes());
    out.extend_from_slice(&block.radial_flags.to_be_bytes());
    if let Some(horizontal) = block.horizontal_calibration_constant_dbz {
        out.extend_from_slice(&horizontal.to_be_bytes());
        out.extend_from_slice(
            &block
                .vertical_calibration_constant_dbz
                .unwrap_or(horizontal)
                .to_be_bytes(),
        );
    }
}

/// One fixed 2432-byte frame holding `body` (at most 2400 bytes).
#[allow(clippy::too_many_arguments)]
fn fixed_frame(
    out: &mut Vec<u8>,
    message_type: u8,
    sequence: u16,
    time: NexradTime,
    segments: u16,
    segment: u16,
    body: &[u8],
) {
    debug_assert!(body.len() <= FRAME_BODY_LEN && body.len().is_multiple_of(2));
    let start = out.len();
    out.extend_from_slice(&[0u8; CTM_LEN]);
    let size_halfwords = ((MESSAGE_HEADER_LEN + body.len()) / 2) as u16;
    message_header(
        out,
        size_halfwords,
        CHANNELS_ORDA,
        message_type,
        sequence,
        time,
        segments,
        segment,
    );
    out.extend_from_slice(body);
    out.resize(start + FRAME_LEN, 0);
}

/// The synthesised 134-frame metadata record.
fn metadata_record(plan: &Plan<'_>) -> Vec<u8> {
    let mut record = Vec::with_capacity(METADATA_FRAMES * FRAME_LEN);
    let time = plan.header_time;
    let vcp = vcp_body(plan);
    let status = rda_status_body(plan);
    let mut frame = 0;
    while frame < METADATA_FRAMES {
        if frame == ADAPTATION_FRAME {
            adaptation_frames(plan, &mut record);
            frame += ADAPTATION_SEGMENTS;
            continue;
        }
        match frame {
            VCP_FRAME => fixed_frame(&mut record, 5, 3, time, 1, 1, &vcp),
            RDA_STATUS_FRAME => fixed_frame(&mut record, 2, 2, time, 1, 1, &status),
            _ => record.resize(record.len() + FRAME_LEN, 0),
        }
        frame += 1;
    }
    record
}

/// Segments of the synthesised Message 18.
const ADAPTATION_SEGMENTS: usize = ADAPTATION_LEN.div_ceil(ADAPTATION_SEGMENT_LEN);

/// The four fixed frames of the synthesised Message 18.
fn adaptation_frames(plan: &Plan<'_>, out: &mut Vec<u8>) {
    let adaptation = adaptation_body(plan);
    for (index, segment) in adaptation.chunks(ADAPTATION_SEGMENT_LEN).enumerate() {
        fixed_frame(
            out,
            18,
            1,
            plan.header_time,
            ADAPTATION_SEGMENTS as u16,
            index as u16 + 1,
            segment,
        );
    }
}

/// Table XV body: zero except the site name and position, the transmitter
/// frequency, the antenna gain and the beam width.
fn adaptation_body(plan: &Plan<'_>) -> Vec<u8> {
    let Adaptation {
        latitude_deg,
        longitude_deg,
        frequency_mhz,
        antenna_gain_db,
        beam_width_deg,
    } = plan.adaptation;
    let mut body = vec![0u8; ADAPTATION_LEN];
    let mut put = |offset: usize, bytes: &[u8]| {
        body[offset..offset + bytes.len()].copy_from_slice(bytes);
    };
    if let Some(mhz) = frequency_mhz {
        put(1092, &mhz.to_be_bytes());
    }
    if let Some(width) = beam_width_deg {
        put(1132, &width.to_be_bytes());
    }
    if let Some(gain) = antenna_gain_db {
        put(1136, &gain.to_be_bytes());
    }
    let (lat_deg, lat_min, lat_sec) = degrees_minutes_seconds(latitude_deg);
    let (lon_deg, lon_min, lon_sec) = degrees_minutes_seconds(longitude_deg);
    put(1288, &lat_sec.to_be_bytes());
    put(1292, &lon_sec.to_be_bytes());
    put(1300, &lat_deg.to_be_bytes());
    put(1304, &lat_min.to_be_bytes());
    put(1308, &lon_deg.to_be_bytes());
    put(1312, &lon_min.to_be_bytes());
    put(1316, if latitude_deg < 0.0 { b"S   " } else { b"N   " });
    put(
        1320,
        if longitude_deg < 0.0 {
            b"W   "
        } else {
            b"E   "
        },
    );
    put(8368, &plan.site);
    body
}

/// Magnitude of `angle` as whole degrees, whole minutes and seconds.
fn degrees_minutes_seconds(angle: f64) -> (i32, i32, f32) {
    let magnitude = angle.abs();
    let degrees = magnitude.floor();
    let minutes_total = (magnitude - degrees) * 60.0;
    let minutes = minutes_total.floor();
    let seconds = (minutes_total - minutes) * 60.0;
    (degrees as i32, minutes as i32, seconds as f32)
}

/// A Table III-A angle code: 360 / 65536 degrees per unit; negative angles
/// as the angle plus 360.
fn angle_code(degrees: f32) -> u16 {
    let wrapped = f64::from(degrees).rem_euclid(360.0);
    ((wrapped * 65_536.0 / 360.0).round() as u32 % 65_536) as u16
}

/// Table XI body: one cut per written sweep.
fn vcp_body(plan: &Plan<'_>) -> Vec<u8> {
    let cuts = plan.sweeps.len();
    let mut body = Vec::with_capacity((11 + 23 * cuts) * 2);
    let mut hw = |value: u16| body.extend_from_slice(&value.to_be_bytes());
    hw((11 + 23 * cuts) as u16); // message size, halfwords
    hw(2); // pattern type: constant elevation cut
    hw(plan.vcp);
    hw(cuts as u16);
    hw(0x0101); // VCP version 1, clutter map group 1
    hw(u16::from(plan.velocity_resolution) << 8 | u16::from(plan.pulse_width)); // resolution, pulse width
    hw(0);
    hw(0);
    hw(0); // sequencing
    hw(0); // supplemental data
    hw(0);
    for sweep in &plan.sweeps {
        hw(angle_code(sweep.fixed_angle_deg)); // E1
        hw(u16::from(sweep.waveform)); // E2: channel configuration 0, waveform
        hw(u16::from(sweep.super_resolution) << 8); // E3
        hw(0); // E4 surveillance pulses
        let rate = (f64::from(sweep.azimuth_rate_deg_per_s) * 65_536.0 / 90.0)
            .round()
            .clamp(f64::from(i16::MIN), f64::from(i16::MAX)) as i16;
        hw(rate as u16); // E5
        for moment in super::plan::Moment::ALL.iter().take(6) {
            let snr = sweep
                .moments
                .iter()
                .find(|plan| plan.moment == *moment)
                .map_or(0, |plan| plan.snr_threshold_raw);
            hw(snr as u16); // E6 to E11
        }
        for _ in 12..=22 {
            hw(0); // Doppler sectors, supplemental data, EBC
        }
        hw(0); // E23
    }
    body
}

/// Table IV body (Open RDA, 60 halfwords): operating, on line, remote
/// control, the VCP, build 20.00, operational mode.
fn rda_status_body(plan: &Plan<'_>) -> Vec<u8> {
    let mut halfwords = [0u16; RDA_STATUS_HALFWORDS];
    let mut set = |number: usize, value: u16| halfwords[number - 1] = value;
    set(1, 16); // RDA state: operate
    set(2, 2); // operability: on line
    set(3, 4); // control status: remote only
    set(7, 28); // data transmission enabled: reflectivity, velocity, width
    set(8, (plan.vcp as i16).max(0) as u16); // VCP, selected remotely
    set(10, RDA_BUILD_RAW);
    set(11, 4); // operational mode: operational
    let super_resolution = plan
        .sweeps
        .iter()
        .any(|sweep| sweep.azimuth_resolution == 1);
    set(12, if super_resolution { 2 } else { 4 });
    halfwords
        .iter()
        .flat_map(|value| value.to_be_bytes())
        .collect()
}
