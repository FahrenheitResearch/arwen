//! The NEXRAD metadata messages in the FM301 model.
//!
//! The volume decoder reads the non-radial messages of an Archive II file
//! and writes each decoded value into [`Volume::extra_vars`] as a root
//! variable named after its ICD field, with its ICD units
//! (`docs/design/fm301-model.md` section 2; FM301 has no slot for any of
//! them). The FM301 view writes them with
//! [`Passthrough::All`](recast_radar_core::fm301::Passthrough::All).
//!
//! | Message | Variables | Dimensions |
//! |---|---|---|
//! | 2, RDA Status Data (Table IV), every one in the file | `nexrad_rda_status_<field>`, `nexrad_rda_status_time`, `nexrad_rda_status_layout` | `nexrad_rda_status` (one entry per message); alarm codes also `nexrad_rda_status_alarm_slot` |
//! | 3, Performance/Maintenance Data (Table V) | `nexrad_performance_<field>` ([`performance_fields`]) | scalars; test bits `nexrad_performance_<field>_index` |
//! | 5 or 7, Volume Coverage Pattern (Table XI) | `nexrad_vcp_<field>`, `nexrad_vcp_message_type` | scalars; cut values `nexrad_vcp_cut`, Doppler sectors also `nexrad_vcp_sector` |
//! | 8, Clutter Censor Zones (Table XII) | `nexrad_clutter_censor_<field>` | `nexrad_clutter_censor_zone` |
//! | 13, Clutter Filter Bypass Map (Table IX) | `nexrad_bypass_map_<field>` | `nexrad_bypass_map_segment`, `nexrad_bypass_map_radial`, `nexrad_bypass_map_halfword` |
//! | 15, Clutter Filter Map (Table XIV) | `nexrad_clutter_filter_map_<field>` | `nexrad_clutter_filter_map_segment`, `nexrad_clutter_filter_map_azimuth` and the ragged `nexrad_clutter_filter_map_zone` |
//! | 18, RDA Adaptation Data (Table XV) | `nexrad_adaptation_<field>` ([`adaptation_fields`]) | scalars; tables `nexrad_adaptation_<field>_index` |
//! | 32, RDA PRF Data (Table XVIII) | `nexrad_prf_<field>` | `nexrad_prf_waveform`, `nexrad_prf_number` |
//! | 4 and 10, Console Message (Table VI), every one | `nexrad_console_<field>`, `nexrad_console_message_time` | `nexrad_console_message`, ragged `nexrad_console_byte` |
//! | 6, RDA Control Commands (Table X), every one | `nexrad_control_commands_<field>`, `nexrad_control_commands_time` | `nexrad_control_commands` |
//! | 9, Request for Data (Table XIII), every one | `nexrad_request_for_data_type`, `nexrad_request_for_data_time` | `nexrad_request_for_data` |
//! | 11 and 12, Loop Back Test (Table VIII), every one | `nexrad_loopback_<field>`, `nexrad_loopback_time` | `nexrad_loopback`, ragged `nexrad_loopback_byte` |
//! | 33, RDA Log Data (Table XVIV), every one | `nexrad_rda_log_<field>`, `nexrad_rda_log_time` | `nexrad_rda_log`, ragged `nexrad_rda_log_byte` |
//! | 3 and 18 of a legacy RDA (layouts not decoded), 29, and types Table I does not define, every one | `nexrad_unparsed_message_<field>`, the frames verbatim (a variable-length message: its header and body) | `nexrad_unparsed_message`, ragged `nexrad_unparsed_message_byte` |
//! | Every non-radial frame's message header (Table II), as stored | `nexrad_metadata_message_<field>` (`message_header_table`) | `nexrad_metadata_message` |
//!
//! Messages 3, 5 or 7, 8, 13, 15, 18 and 32 also get `<prefix>message_time`,
//! the generation time in their message header, in seconds since the
//! volume's time reference (NaN when the header date is 0), and
//! `<prefix>message_channels`; the tables have a time and a channels column. A later message of one of these types whose content
//! differs from every earlier one is carried too, up to
//! [`MAX_COPIES_PER_MESSAGE_TYPE`] copies: copy `n` (from 1) has
//! `_message<n>` after every variable and dimension name. Messages 13 and
//! 15 with body bytes after their map carry them as
//! `<prefix>trailing_bytes`. Counted in root variables, when not zero:
//! messages that do not decode (`nexrad_metadata_messages_not_decoded`),
//! later messages identical to a carried one apart from the message header
//! (`nexrad_metadata_messages_repeated`), copies past the cap
//! (`nexrad_metadata_messages_not_carried`) and frames past the builder's
//! cap (`nexrad_metadata_frames_not_kept`).
//!
//! Every variable has a `comment` naming its ICD table and location, a
//! `long_name`, and `units` where the ICD gives a physical unit; code fields
//! keep the ICD code. Values are those of the typed decoders in
//! [`crate::messages`]. Each message's header channel byte (the RDA
//! redundant channel) is `<prefix>message_channels`, or the `<table>_channels`
//! column of a table, beside its generation time. The whole header of every
//! frame, as stored, is also in the `nexrad_metadata_message_*` table.
//!
//! Non-radial messages in variable framing (a size of 65535, or message 29)
//! are kept in their own framing, up to 16 MiB per volume in all; an
//! extended-size message 33 is read like a fixed one.
//!
//! Where FM301 has a slot the value goes there instead: the VCP cut azimuth
//! rate of each sweep's elevation number is `Sweep::target_scan_rate_deg_per_s`,
//! and message 18 already gives `Volume::radar_parameters`.

use std::collections::BTreeMap;

use recast_radar_core::bounded_read::DecodeBudget;
use recast_radar_core::model::{
    ArrayBuf, AttrValue, ExtraVariable, PrtMode, Scalar, Sweep, Volume,
};

use crate::MessageHeader;
use crate::messages::adaptation::RdaAdaptationData;
use crate::messages::bypass_map::{BypassMapLayout, ClutterFilterBypassMap};
use crate::messages::clutter_censor::ClutterCensorZones;
use crate::messages::clutter_filter_map::ClutterFilterMap;
use crate::messages::console::ConsoleMessage;
use crate::messages::control::RdaControlCommands;
use crate::messages::loopback::LoopbackTest;
use crate::messages::performance::PerformanceMaintenance;
use crate::messages::prf::RdaPrfData;
use crate::messages::rda_log::{MAX_RDA_LOG_BYTES, RdaLogCompression, RdaLogData};
use crate::messages::rda_status::{LegacyRdaStatus, OrdaRdaStatus, RdaStatus};
use crate::messages::request::RequestForData;
use crate::messages::vcp::{PulseWidth, VcpCut, VolumeCoveragePattern, WaveformType};
use crate::messages::{MessageBody, RawMessages};

/// Where a value sits in its message body.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Location {
    /// A 1-based halfword number (Tables IV and V); a value of several
    /// halfwords starts here.
    Halfword(u16),
    /// Byte 0 (most significant) or 1 of a 1-based halfword.
    HalfwordByte(u16, u8),
    /// A 0-based byte offset (Table XV).
    Byte(u16),
}

/// A decoded metadata value.
#[derive(Clone, Debug, PartialEq)]
pub enum FieldValue<'a> {
    /// A number in the decoder's type.
    Scalar(Scalar),
    /// Text with NUL padding removed.
    Text(&'a str),
    /// A Table XV "T"/"F" flag; `None` when the bytes are neither.
    Flag(Option<bool>),
    /// A fixed-length table.
    Array(ArrayBuf),
}

/// One decoded field of a metadata message.
#[derive(Clone, Debug, PartialEq)]
pub struct MessageField<'a> {
    /// Field name: the ICD mnemonic (message 18) or the ICD field name in
    /// lower case. The model variable is the message prefix plus this name.
    pub name: &'static str,
    /// Location in the message body.
    pub location: Location,
    /// UDUNITS units, or `""` for codes, counts of things and text.
    pub units: &'static str,
    /// The ICD description.
    pub long_name: &'static str,
    /// The decoded value.
    pub value: FieldValue<'a>,
}

/// Collects [`MessageField`]s.
struct Fields<'a>(Vec<MessageField<'a>>);

impl<'a> Fields<'a> {
    fn push(
        &mut self,
        name: &'static str,
        location: Location,
        units: &'static str,
        long_name: &'static str,
        value: FieldValue<'a>,
    ) {
        self.0.push(MessageField {
            name,
            location,
            units,
            long_name,
            value,
        });
    }
}

type Value<'a> = FieldValue<'a>;

/// Every field of a message 3 (Table V, Build 24.0 names), in halfword order.
/// The model variables are `nexrad_performance_<name>`.
pub fn performance_fields(performance: &PerformanceMaintenance) -> Vec<MessageField<'_>> {
    let mut fields = Fields(Vec::with_capacity(270));
    performance_table(performance, &mut fields);
    fields.0
}

/// Every field of a message 18 (Table XV, Build 24.0 names), in byte order.
/// The model variables are `nexrad_adaptation_<name>`.
pub fn adaptation_fields(adaptation: &RdaAdaptationData) -> Vec<MessageField<'_>> {
    let mut fields = Fields(Vec::with_capacity(230));
    adaptation_table(adaptation, &mut fields);
    fields.0
}

/// Every field of a message 2 (Table IV of ICD 2620002AA for an Open RDA,
/// of ICD 2620002B for a legacy RDA), in halfword order. Code fields are the
/// halfwords as sent. The model variables are `nexrad_rda_status_<name>`.
pub fn rda_status_fields(status: &RdaStatus) -> Vec<MessageField<'static>> {
    let mut fields = Fields(Vec::with_capacity(40));
    let u16v = |value: u16| Value::Scalar(Scalar::U16(value));
    match status {
        RdaStatus::Orda(orda) => orda_fields(orda, &mut fields, u16v),
        RdaStatus::Legacy(legacy) => legacy_fields(legacy, &mut fields, u16v),
    }
    fields.0
}

fn shared_status_head(
    f: &mut Fields<'static>,
    u16v: impl Fn(u16) -> Value<'static>,
    rda_state: u16,
    operability: u16,
    control_status: u16,
    auxiliary_power: u16,
    average_transmitter_power: u16,
) {
    use Location::Halfword;
    f.push("rda_status", Halfword(1), "", "RDA status", u16v(rda_state));
    f.push(
        "operability_status",
        Halfword(2),
        "",
        "Operability status",
        u16v(operability),
    );
    f.push(
        "control_status",
        Halfword(3),
        "",
        "Control status",
        u16v(control_status),
    );
    f.push(
        "auxiliary_power_generator_state",
        Halfword(4),
        "",
        "Auxiliary power generator state",
        u16v(auxiliary_power),
    );
    f.push(
        "average_transmitter_power",
        Halfword(5),
        "W",
        "Average transmitter power",
        u16v(average_transmitter_power),
    );
}

fn orda_fields(
    s: &OrdaRdaStatus,
    f: &mut Fields<'static>,
    u16v: impl Fn(u16) -> Value<'static> + Copy,
) {
    use Location::Halfword;
    shared_status_head(
        f,
        u16v,
        s.rda_state.code(),
        s.operability.code(),
        s.control_status.code(),
        s.auxiliary_power.0,
        s.average_transmitter_power,
    );
    f.push(
        "horizontal_reflectivity_calibration_correction",
        Halfword(6),
        "dB",
        "Horizontal reflectivity calibration correction (delta dBZ0)",
        Value::Scalar(Scalar::F32(
            s.horizontal_reflectivity_calibration_correction,
        )),
    );
    f.push(
        "data_transmission_enabled",
        Halfword(7),
        "",
        "Data transmission enabled",
        u16v(s.data_transmission.0),
    );
    f.push(
        "volume_coverage_pattern",
        Halfword(8),
        "",
        "Volume coverage pattern number (negative when selected locally)",
        vcp_value(s.volume_coverage_pattern.signed()),
    );
    f.push(
        "rda_control_authorization",
        Halfword(9),
        "",
        "RDA control authorization",
        u16v(s.control_authorization.code()),
    );
    f.push(
        "rda_build_number",
        Halfword(10),
        "",
        "RDA build number",
        u16v(s.rda_build.raw()),
    );
    f.push(
        "operational_mode",
        Halfword(11),
        "",
        "Operational mode",
        u16v(s.operational_mode.code()),
    );
    f.push(
        "super_resolution_status",
        Halfword(12),
        "",
        "Super resolution status",
        u16v(s.super_resolution.code()),
    );
    f.push(
        "clutter_mitigation_decision_status",
        Halfword(13),
        "",
        "Clutter mitigation decision status",
        u16v(s.clutter_mitigation_decision.0),
    );
    f.push(
        "rda_scan_and_data_flags",
        Halfword(14),
        "",
        "RDA scan and data flags (AVSET, EBC, RDA log data, time series recording)",
        u16v(s.scan_data_flags.0),
    );
    f.push(
        "rda_alarm_summary",
        Halfword(15),
        "",
        "RDA alarm summary",
        u16v(s.alarm_summary.0),
    );
    status_tail(
        f,
        u16v,
        s.command_acknowledgment.code(),
        s.channel_control.code(),
        s.spot_blanking.code(),
        (
            s.bypass_map_generation.date,
            s.bypass_map_generation.minutes,
        ),
    );
    f.push(
        "clutter_filter_map_generation_date",
        Halfword(21),
        "",
        "Clutter filter map generation date (days, 1 January 1970 = 1)",
        u16v(s.clutter_filter_map_generation.date),
    );
    f.push(
        "clutter_filter_map_generation_time",
        Halfword(22),
        "min",
        "Clutter filter map generation time (minutes after midnight UTC)",
        u16v(s.clutter_filter_map_generation.minutes),
    );
    f.push(
        "vertical_reflectivity_calibration_correction",
        Halfword(23),
        "dB",
        "Vertical reflectivity calibration correction (delta dBZ0)",
        Value::Scalar(Scalar::F32(s.vertical_reflectivity_calibration_correction)),
    );
    f.push(
        "transition_power_source_status",
        Halfword(24),
        "",
        "Transition power source status",
        u16v(s.transition_power_source.code()),
    );
    f.push(
        "rms_control_status",
        Halfword(25),
        "",
        "RMS control status",
        u16v(s.rms_control.code()),
    );
    f.push(
        "performance_check_status",
        Halfword(26),
        "",
        "Performance check status",
        u16v(s.performance_check.code()),
    );
    alarm_codes(f, &s.alarm_codes);
    if let Some(options) = s.signal_processing_options {
        f.push(
            "signal_processor_options",
            Halfword(41),
            "",
            "Signal processor options",
            u16v(options.0),
        );
    }
    if let Some(pattern) = s.downloaded_pattern_number {
        f.push(
            "downloaded_pattern_number",
            Halfword(59),
            "",
            "Remote VCP number the RDA acknowledges (0 for none)",
            u16v(pattern),
        );
    }
    if let Some(version) = s.status_version {
        f.push(
            "status_version",
            Halfword(60),
            "",
            "RDA status message version",
            u16v(version),
        );
    }
}

fn legacy_fields(
    s: &LegacyRdaStatus,
    f: &mut Fields<'static>,
    u16v: impl Fn(u16) -> Value<'static> + Copy,
) {
    use Location::Halfword;
    shared_status_head(
        f,
        u16v,
        s.rda_state.code(),
        s.operability.code(),
        s.control_status.code(),
        s.auxiliary_power.0,
        s.average_transmitter_power,
    );
    f.push(
        "reflectivity_calibration_correction",
        Halfword(6),
        "",
        "Reflectivity calibration correction, as sent (ICD 2620002B gives no scale)",
        Value::Scalar(Scalar::I16(s.reflectivity_calibration_correction_raw)),
    );
    f.push(
        "data_transmission_enabled",
        Halfword(7),
        "",
        "Data transmission enabled",
        u16v(s.data_transmission.0),
    );
    f.push(
        "volume_coverage_pattern",
        Halfword(8),
        "",
        "Volume coverage pattern number (negative when selected locally)",
        vcp_value(s.volume_coverage_pattern.signed()),
    );
    f.push(
        "rda_control_authorization",
        Halfword(9),
        "",
        "RDA control authorization",
        u16v(s.control_authorization.code()),
    );
    f.push(
        "interference_detection_rate",
        Halfword(10),
        "s-1",
        "Interference detection rate (pulses per second)",
        u16v(s.interference_detection_rate),
    );
    f.push(
        "operational_mode",
        Halfword(11),
        "",
        "Operational mode",
        u16v(s.operational_mode.code()),
    );
    f.push(
        "interference_suppression_unit",
        Halfword(12),
        "",
        "Interference suppression unit",
        u16v(s.interference_suppression_unit.code()),
    );
    f.push(
        "archive_ii_status",
        Halfword(13),
        "",
        "Archive II status",
        u16v(s.archive_ii_status),
    );
    f.push(
        "archive_ii_remaining_capacity",
        Halfword(14),
        "",
        "Archive II estimated remaining capacity (volume scans)",
        u16v(s.archive_ii_remaining_capacity),
    );
    f.push(
        "rda_alarm_summary",
        Halfword(15),
        "",
        "RDA alarm summary",
        u16v(s.alarm_summary.0),
    );
    status_tail(
        f,
        u16v,
        s.command_acknowledgment.code(),
        s.channel_control.code(),
        s.spot_blanking.code(),
        (
            s.bypass_map_generation.date,
            s.bypass_map_generation.minutes,
        ),
    );
    f.push(
        "notch_width_map_generation_date",
        Halfword(21),
        "",
        "Notch width map generation date (days, 1 January 1970 = 1)",
        u16v(s.notch_width_map_generation.date),
    );
    f.push(
        "notch_width_map_generation_time",
        Halfword(22),
        "min",
        "Notch width map generation time (minutes after midnight UTC)",
        u16v(s.notch_width_map_generation.minutes),
    );
    f.push(
        "transition_power_source_status",
        Halfword(24),
        "",
        "Transition power source status",
        u16v(s.transition_power_source.code()),
    );
    f.push(
        "rms_control_status",
        Halfword(25),
        "",
        "RMS control status",
        u16v(s.rms_control.code()),
    );
    alarm_codes(f, &s.alarm_codes);
}

fn vcp_value(signed: i32) -> Value<'static> {
    Value::Scalar(Scalar::I16(i16::try_from(signed).unwrap_or(i16::MIN)))
}

fn status_tail(
    f: &mut Fields<'static>,
    u16v: impl Fn(u16) -> Value<'static>,
    command_acknowledgment: u16,
    channel_control: u16,
    spot_blanking: u16,
    (bypass_date, bypass_minutes): (u16, u16),
) {
    use Location::Halfword;
    f.push(
        "command_acknowledgment",
        Halfword(16),
        "",
        "Command acknowledgment",
        u16v(command_acknowledgment),
    );
    f.push(
        "channel_control_status",
        Halfword(17),
        "",
        "Channel control status",
        u16v(channel_control),
    );
    f.push(
        "spot_blanking_status",
        Halfword(18),
        "",
        "Spot blanking status",
        u16v(spot_blanking),
    );
    f.push(
        "bypass_map_generation_date",
        Halfword(19),
        "",
        "Bypass map generation date (days, 1 January 1970 = 1)",
        u16v(bypass_date),
    );
    f.push(
        "bypass_map_generation_time",
        Halfword(20),
        "min",
        "Bypass map generation time (minutes after midnight UTC)",
        u16v(bypass_minutes),
    );
}

fn alarm_codes(f: &mut Fields<'static>, codes: &[u16]) {
    f.push(
        "alarm_codes",
        Location::Halfword(27),
        "",
        "RDA alarm codes (bit 15 set: alarm cleared; 0: slot unused)",
        Value::Array(ArrayBuf::U16(codes.to_vec())),
    );
}

/// Ceiling on the inflated RDA log data (message 33) one volume keeps, all
/// logs together. Each log is also limited to [`MAX_RDA_LOG_BYTES`] and
/// charged to the volume's [`DecodeBudget`]; a log that does not fit is not
/// decoded and counts in `nexrad_metadata_messages_not_decoded`.
pub(crate) const MAX_RDA_LOG_VOLUME_BYTES: usize = MAX_RDA_LOG_BYTES;

/// Copies of one message 3, 5 or 7, 8, 13, 15, 18 or 32 a volume keeps: the
/// first, and each later one whose content differs from every kept copy.
/// A real file has one of each; the cap bounds the variables a hostile file
/// can make from the 1024 metadata frames the builder keeps.
pub const MAX_COPIES_PER_MESSAGE_TYPE: usize = 16;

/// A message and the header it came with (its generation time).
type Timed<T> = (MessageHeader, T);

/// The metadata messages of one volume, as the volume decoder reads them.
#[derive(Clone, Debug, Default)]
pub(crate) struct MetadataMessages {
    rda_status: Vec<Timed<RdaStatus>>,
    performance: Vec<Timed<Box<PerformanceMaintenance>>>,
    vcp: Vec<Timed<VolumeCoveragePattern>>,
    /// Each message 18 with its body, for the bytes of flag fields that are
    /// neither "T" nor "F".
    adaptation: Vec<Timed<(Box<RdaAdaptationData>, Vec<u8>)>>,
    /// Each message 15 with the body bytes after its map.
    clutter_filter_map: Vec<Timed<(ClutterFilterMap, Vec<u8>)>>,
    /// Each message 13 with the body bytes after its map.
    bypass_map: Vec<Timed<(ClutterFilterBypassMap, Vec<u8>)>>,
    clutter_censor_zones: Vec<Timed<ClutterCensorZones>>,
    prf: Vec<Timed<RdaPrfData>>,
    console: Vec<Timed<ConsoleMessage>>,
    rda_logs: Vec<Timed<RdaLogData>>,
    control_commands: Vec<Timed<RdaControlCommands>>,
    requests: Vec<Timed<RequestForData>>,
    loopback: Vec<Timed<LoopbackTest>>,
    /// Messages whose layout the decoders do not read (the legacy RDA
    /// messages 3 and 18): their frames from the message header on,
    /// verbatim ([`unparsed_frames`]).
    unparsed: Vec<Timed<Vec<u8>>>,
    /// Body bytes of the kept copies of each message 3, 5 or 7, 8, 13, 15,
    /// 18 and 32, by message type (7 under 5), to tell a repeat.
    bodies: BTreeMap<u8, Vec<Vec<u8>>>,
    /// Messages that did not decode: framing errors, table errors, and RDA
    /// logs beyond the limits.
    not_decoded: usize,
    /// Later messages 3, 5 or 7, 8, 13, 15, 18 or 32 identical to a kept
    /// copy (their content is carried; their header is not).
    repeated: usize,
    /// Later messages past [`MAX_COPIES_PER_MESSAGE_TYPE`] distinct copies.
    not_carried: usize,
    /// Fixed frames past the volume builder's frame cap, never read.
    frames_not_kept: usize,
}

/// What [`keep`] did with a message.
enum Outcome {
    Kept,
    Repeated,
    OverCap,
}

/// Keep `value` unless a kept message of its type had the same body bytes
/// (`bodies`, one entry per kept copy) or the type has
/// [`MAX_COPIES_PER_MESSAGE_TYPE`] copies. Bodies are compared rather than
/// decoded values, whose floats may be NaN.
fn keep<T>(
    copies: &mut Vec<Timed<T>>,
    bodies: &mut Vec<Vec<u8>>,
    header: MessageHeader,
    value: T,
    body: Vec<u8>,
) -> Outcome {
    if bodies.contains(&body) {
        Outcome::Repeated
    } else if copies.len() >= MAX_COPIES_PER_MESSAGE_TYPE {
        Outcome::OverCap
    } else {
        copies.push((header, value));
        bodies.push(body);
        Outcome::Kept
    }
}

/// Archive II fixed frame (Table II): a 12-byte CTM header, then the
/// message header and body, 2432 bytes in all.
const FRAME_BYTES: usize = 2432;
const CTM_BYTES: usize = 12;

/// The frames of the message whose (first) header is at `offset` in
/// `frames` and which spans `count` frames: each frame from the message
/// header to the frame end ([`FRAME_BYTES`] - [`CTM_BYTES`] bytes), one after
/// another. This keeps the whole of each frame the message occupies, whatever
/// its header sizes say: a converted file's legacy-channel message 18 may declare sizes that
/// leave out the message header, so its last 16 body bytes in each frame lie
/// past the declared size.
fn unparsed_frames(frames: &[u8], offset: usize, count: usize) -> Option<Vec<u8>> {
    let start = offset.checked_sub(CTM_BYTES)?;
    let end = start.checked_add(count.checked_mul(FRAME_BYTES)?)?;
    let span = frames.get(start..end)?;
    let mut out = Vec::with_capacity(count * (FRAME_BYTES - CTM_BYTES));
    for frame in span.chunks_exact(FRAME_BYTES) {
        out.extend_from_slice(&frame[CTM_BYTES..]);
    }
    Some(out)
}

/// The bytes of an unparsed message whose header is at `offset` in
/// `frames`: its [`unparsed_frames`], or the message header and body of a
/// variable-length message, which is kept in its own framing.
fn unparsed_bytes(
    frames: &[u8],
    header: &MessageHeader,
    offset: usize,
    count: usize,
) -> Option<Vec<u8>> {
    if header.is_variable_length() {
        frames
            .get(offset..offset.checked_add(header.message_len())?)
            .map(<[u8]>::to_vec)
    } else {
        unparsed_frames(frames, offset, count)
    }
}

/// The last `trailing` bytes of `body`.
fn tail(body: &[u8], trailing: usize) -> Vec<u8> {
    body[body.len().saturating_sub(trailing)..].to_vec()
}

impl MetadataMessages {
    /// Decode the messages of `frames` (record bytes: 2432-byte frames, each
    /// a 12-byte CTM header, the message header and the body). Messages that
    /// do not decode are counted and skipped; `NexradMetadata` reports them.
    /// RDA log data is limited to [`MAX_RDA_LOG_VOLUME_BYTES`] and charged to
    /// `budget`. `frames_not_kept` is the number of frames the builder did
    /// not keep.
    pub(crate) fn from_frames(
        frames: &[u8],
        frames_not_kept: usize,
        budget: &mut DecodeBudget,
    ) -> Self {
        let mut messages = Self {
            frames_not_kept,
            ..Self::default()
        };
        let mut log_allowance = MAX_RDA_LOG_VOLUME_BYTES;
        for item in RawMessages::new(frames) {
            let Ok(raw) = item else {
                messages.not_decoded += 1;
                continue;
            };
            if raw.header.message_type == 33 {
                let limit = log_allowance.min(budget.remaining());
                match RdaLogData::decode_limited(&raw.body, limit) {
                    Ok(log) if budget.charge(1, log.data.len(), "RDA log data").is_ok() => {
                        log_allowance = log_allowance.saturating_sub(log.data.len());
                        messages.rda_logs.push((raw.header, log));
                    }
                    _ => messages.not_decoded += 1,
                }
                continue;
            }
            // The body of a message 3, 5 or 7, 8, 13, 15, 18 or 32 tells a
            // repeat; messages 13, 15 and 18 also keep bytes the typed
            // decoders do not hold.
            let kind = raw.header.message_type;
            let body = matches!(kind, 3 | 5 | 7 | 8 | 13 | 15 | 18 | 32).then(|| raw.body.to_vec());
            let (offset, frame_count) = (raw.offset, raw.frames);
            let Ok((header, decoded)) = raw.decode() else {
                messages.not_decoded += 1;
                continue;
            };
            let body = body.unwrap_or_default();
            let bodies = messages
                .bodies
                .entry(if kind == 7 { 5 } else { kind })
                .or_default();
            let kept = match decoded {
                MessageBody::RdaStatus(status) => {
                    messages.rda_status.push((header, status));
                    Outcome::Kept
                }
                MessageBody::Performance(data) => {
                    keep(&mut messages.performance, bodies, header, data, body)
                }
                MessageBody::Vcp(vcp) => keep(&mut messages.vcp, bodies, header, vcp, body),
                MessageBody::Adaptation(data) => keep(
                    &mut messages.adaptation,
                    bodies,
                    header,
                    (data, body.clone()),
                    body,
                ),
                MessageBody::ClutterFilterMap(map) => {
                    let trailing = tail(&body, map.trailing_bytes);
                    keep(
                        &mut messages.clutter_filter_map,
                        bodies,
                        header,
                        (map, trailing),
                        body,
                    )
                }
                MessageBody::BypassMap(map) => {
                    let trailing = tail(&body, map.trailing_bytes);
                    keep(
                        &mut messages.bypass_map,
                        bodies,
                        header,
                        (map, trailing),
                        body,
                    )
                }
                MessageBody::ClutterCensorZones(zones) => keep(
                    &mut messages.clutter_censor_zones,
                    bodies,
                    header,
                    zones,
                    body,
                ),
                MessageBody::Prf(prf) => keep(&mut messages.prf, bodies, header, prf, body),
                MessageBody::Console(console) => {
                    messages.console.push((header, console));
                    Outcome::Kept
                }
                MessageBody::ControlCommands(commands) => {
                    messages.control_commands.push((header, commands));
                    Outcome::Kept
                }
                MessageBody::RequestForData(request) => {
                    messages.requests.push((header, request));
                    Outcome::Kept
                }
                MessageBody::Loopback(test) => {
                    messages.loopback.push((header, test));
                    Outcome::Kept
                }
                MessageBody::Unparsed(_) => {
                    match unparsed_bytes(frames, &header, offset, frame_count) {
                        Some(bytes) => {
                            messages.unparsed.push((header, bytes));
                            Outcome::Kept
                        }
                        None => {
                            messages.not_decoded += 1;
                            Outcome::Kept
                        }
                    }
                }
                // The builder keeps no message 31 frame here, and message 33
                // is read above.
                MessageBody::DigitalRadarDataGeneric(_) | MessageBody::RdaLog(_) => Outcome::Kept,
            };
            match kept {
                Outcome::Kept => {}
                Outcome::Repeated => messages.repeated += 1,
                Outcome::OverCap => messages.not_carried += 1,
            }
        }
        messages
    }

    /// The VCP that sets the sweeps' scan rates and pulse values: the first
    /// message 5, or else the first message 7.
    fn scan_vcp(&self) -> Option<&VolumeCoveragePattern> {
        self.vcp
            .iter()
            .find(|(header, _)| header.message_type == 5)
            .or_else(|| self.vcp.first())
            .map(|(_, vcp)| vcp)
    }

    /// Write every message into `volume` (see the module documentation).
    pub(crate) fn attach(self, volume: &mut Volume) {
        let reference = volume.time_reference;
        let out = &mut volume.extra_vars;
        for (name, count, long_name) in [
            (
                "nexrad_metadata_messages_not_decoded",
                self.not_decoded,
                "Metadata messages that did not decode (framing or table errors, RDA log data beyond the limits)",
            ),
            (
                "nexrad_metadata_messages_repeated",
                self.repeated,
                "Later messages 3, 5 or 7, 8, 13, 15, 18 or 32 identical to a carried one apart from the message header",
            ),
            (
                "nexrad_metadata_messages_not_carried",
                self.not_carried,
                "Later messages 3, 5 or 7, 8, 13, 15, 18 or 32 past the decoder's cap of distinct copies of a type",
            ),
            (
                "nexrad_metadata_frames_not_kept",
                self.frames_not_kept,
                "Metadata frames past the decoder's caps (1024 fixed frames; 1024 variable-length messages or 16 MiB of them), not kept",
            ),
        ] {
            if count > 0 {
                out.push(variable(
                    name.to_owned(),
                    Vec::new(),
                    ArrayBuf::U32(vec![len_u32(count)]),
                    vec![("long_name".into(), AttrValue::text(long_name))],
                ));
            }
        }
        if !self.rda_status.is_empty() {
            rda_status_table(&self.rda_status, &reference, out);
        }
        for (copy, (header, performance)) in self.performance.iter().enumerate() {
            let mut vars = Vec::new();
            single_message(
                "nexrad_performance_",
                "ICD 2620002AA Table V",
                &performance_fields(performance),
                &[],
                &mut vars,
            );
            push_copy(out, vars, "nexrad_performance_", header, &reference, copy);
        }
        for (copy, (header, vcp)) in self.vcp.iter().enumerate() {
            let mut vars = vec![scalar_variable(
                "nexrad_vcp_message_type",
                Scalar::U8(header.message_type),
                table_attrs(
                    "",
                    "Message type (5 RDA to RPG, 7 RPG to RDA)",
                    "Table II",
                    "message header",
                ),
            )];
            vcp_variables(vcp, &mut vars);
            push_copy(out, vars, "nexrad_vcp_", header, &reference, copy);
        }
        for (copy, (header, (adaptation, body))) in self.adaptation.iter().enumerate() {
            let mut vars = Vec::new();
            single_message(
                "nexrad_adaptation_",
                "ICD 2620002AA Table XV",
                &adaptation_fields(adaptation),
                body,
                &mut vars,
            );
            push_copy(out, vars, "nexrad_adaptation_", header, &reference, copy);
        }
        for (copy, (header, zones)) in self.clutter_censor_zones.iter().enumerate() {
            let mut vars = Vec::new();
            censor_variables(zones, &mut vars);
            push_copy(
                out,
                vars,
                "nexrad_clutter_censor_",
                header,
                &reference,
                copy,
            );
        }
        for (copy, (header, (map, trailing))) in self.bypass_map.iter().enumerate() {
            let mut vars = Vec::new();
            bypass_variables(map, &mut vars);
            trailing_variable("nexrad_bypass_map_", "Table IX", trailing, &mut vars);
            push_copy(out, vars, "nexrad_bypass_map_", header, &reference, copy);
        }
        for (copy, (header, (map, trailing))) in self.clutter_filter_map.iter().enumerate() {
            let mut vars = Vec::new();
            clutter_filter_map_variables(map, &mut vars);
            trailing_variable(
                "nexrad_clutter_filter_map_",
                "Table XIV",
                trailing,
                &mut vars,
            );
            push_copy(
                out,
                vars,
                "nexrad_clutter_filter_map_",
                header,
                &reference,
                copy,
            );
        }
        for (copy, (header, prf)) in self.prf.iter().enumerate() {
            let mut vars = Vec::new();
            prf_variables(prf, &mut vars);
            push_copy(out, vars, "nexrad_prf_", header, &reference, copy);
        }
        if !self.console.is_empty() {
            console_variables(&self.console, &reference, out);
        }
        if !self.control_commands.is_empty() {
            control_variables(&self.control_commands, &reference, out);
        }
        if !self.requests.is_empty() {
            request_variables(&self.requests, &reference, out);
        }
        if !self.loopback.is_empty() {
            loopback_variables(&self.loopback, &reference, out);
        }
        if !self.unparsed.is_empty() {
            unparsed_variables(&self.unparsed, &reference, out);
        }
        if let Some(vcp) = self.scan_vcp() {
            let prf = self.prf.first().map(|(_, prf)| prf);
            let adaptation = self
                .adaptation
                .first()
                .map(|(_, (adaptation, _))| &**adaptation);
            for sweep in &mut volume.sweeps {
                let cut = sweep
                    .elevation_number
                    .and_then(|number| usize::from(number).checked_sub(1))
                    .and_then(|index| vcp.cuts.get(index));
                if let Some(cut) = cut {
                    sweep.target_scan_rate_deg_per_s = Some(cut.azimuth_rate_deg_per_s);
                    attach_cut_pulses(sweep, vcp, cut, prf, adaptation);
                }
            }
        }
        if !self.rda_logs.is_empty() {
            rda_log_variables(self.rda_logs, &reference, &mut volume.extra_vars);
        }
    }
}

/// Seconds from the volume's time reference to the generation time in a
/// message header (Table II halfwords 5 to 8); NaN when the date is 0.
fn header_seconds(header: &MessageHeader, reference: &chrono::DateTime<chrono::Utc>) -> f64 {
    if header.date == 0 {
        return f64::NAN;
    }
    let ms = (i64::from(header.date) - 1) * 86_400_000 + i64::from(header.milliseconds);
    (ms - reference.timestamp_millis()) as f64 / 1000.0
}

/// Attributes of a message generation time variable.
fn time_attrs(reference: &chrono::DateTime<chrono::Utc>, what: &str) -> Vec<(Box<str>, AttrValue)> {
    vec![
        (
            "long_name".into(),
            AttrValue::Text(format!("generation time of the {what}").into_boxed_str()),
        ),
        (
            "units".into(),
            AttrValue::Text(
                format!("seconds since {}", reference.format("%Y-%m-%dT%H:%M:%SZ"))
                    .into_boxed_str(),
            ),
        ),
        (
            "comment".into(),
            AttrValue::text("ICD 2620002AA Table II message header date and time"),
        ),
    ]
}

/// The generation time of each message of a table, over `dim`.
fn time_column<T>(
    messages: &[Timed<T>],
    reference: &chrono::DateTime<chrono::Utc>,
    name: &str,
    dim: &str,
    what: &str,
) -> ExtraVariable {
    vector(
        name,
        dim,
        ArrayBuf::F64(
            messages
                .iter()
                .map(|(header, _)| header_seconds(header, reference))
                .collect(),
        ),
        time_attrs(reference, what),
    )
}

/// The channel byte of each message of a table, over `dim`.
fn channels_column<T>(messages: &[Timed<T>], name: &str, dim: &str) -> ExtraVariable {
    vector(
        name,
        dim,
        ArrayBuf::U8(messages.iter().map(|(header, _)| header.channels).collect()),
        channels_attrs(),
    )
}

/// The generation time and the channel byte of each message of a table,
/// over `dim`: `<table>_time` and `<table>_channels`.
fn header_columns<T>(
    messages: &[Timed<T>],
    reference: &chrono::DateTime<chrono::Utc>,
    table: &str,
    dim: &str,
    what: &str,
) -> [ExtraVariable; 2] {
    [
        time_column(messages, reference, &format!("{table}_time"), dim, what),
        channels_column(messages, &format!("{table}_channels"), dim),
    ]
}

/// Attributes of a message header channel byte.
fn channels_attrs() -> Vec<(Box<str>, AttrValue)> {
    vec![
        (
            "long_name".into(),
            AttrValue::text("RDA redundant channel of the message"),
        ),
        (
            "comment".into(),
            AttrValue::text(
                "ICD 2620002AA Table II halfword 2, high byte: bits 0-1 the redundant channel (0 single channel, 1 or 2), bit 3 set for an Open RDA",
            ),
        ),
    ]
}

/// Add the variables of one copy of a message: `<prefix>message_time`, the
/// generation time in its header, and `<prefix>message_channels`, its
/// channel byte, then `vars`. Copy 0 (the first message
/// of its type) keeps the names; a later copy `n` gets `_message<n>` after
/// every variable name, every `nexrad_*` dimension name and every
/// `sample_dimension` attribute.
fn push_copy(
    out: &mut Vec<ExtraVariable>,
    vars: Vec<ExtraVariable>,
    prefix: &str,
    header: &MessageHeader,
    reference: &chrono::DateTime<chrono::Utc>,
    copy: usize,
) {
    let time = scalar_variable(
        &format!("{prefix}message_time"),
        Scalar::F64(header_seconds(header, reference)),
        time_attrs(reference, "message"),
    );
    let channels = scalar_variable(
        &format!("{prefix}message_channels"),
        Scalar::U8(header.channels),
        channels_attrs(),
    );
    out.reserve(vars.len() + 2);
    let all = [time, channels].into_iter().chain(vars);
    if copy == 0 {
        out.extend(all);
        return;
    }
    let suffix = format!("_message{copy}");
    let renamed = |name: &str| -> Box<str> {
        if name.starts_with("nexrad_") {
            format!("{name}{suffix}").into_boxed_str()
        } else {
            name.into()
        }
    };
    out.extend(all.map(|mut variable| {
        variable.name = renamed(&variable.name);
        for dim in &mut variable.dims {
            *dim = renamed(dim);
        }
        for (key, value) in &mut variable.attrs {
            if &**key == "sample_dimension"
                && let AttrValue::Text(dim) = value
            {
                *dim = renamed(dim);
            }
        }
        variable
    }));
}

/// The body bytes after a clutter map (`<prefix>trailing_bytes`), when there
/// are any.
fn trailing_variable(prefix: &str, table: &str, trailing: &[u8], out: &mut Vec<ExtraVariable>) {
    if trailing.is_empty() {
        return;
    }
    out.push(vector(
        &format!("{prefix}trailing_bytes"),
        &format!("{prefix}trailing_byte"),
        ArrayBuf::U8(trailing.to_vec()),
        table_attrs(
            "",
            "Body bytes after the map, verbatim (not part of the table: stale segment content)",
            table,
            "after the last elevation segment",
        ),
    ));
}

/// The per-ray FM301 `n_samples`, `prt` and `pulse_width` of a sweep, and
/// its `prt_mode`, from the file's own values:
///
/// - `n_samples`: the VCP cut's surveillance pulse count (contiguous
///   surveillance cuts) or the pulse count of the Doppler sector that holds
///   the ray's azimuth (every other waveform; batch cuts report the Doppler
///   part, as the radial block's unambiguous range does).
/// - `prt`: 1 / PRF, the PRF being the message 32 value for the cut's
///   surveillance PRF number or the sector's Doppler PRF number (Table XVIII
///   note 1). Without a message 32 there is no `prt`: PRF numbers are codes
///   whose values depend on the site's PRF set. Staggered pulse pair cuts
///   (two PRTs) get no `prt`.
/// - `pulse_width`: the message 18 transmitter pulse width of the VCP's
///   pulse (TAU_SP or TAU_LP, ns), when the file has an Open RDA message 18.
/// - `prt_mode`: `fixed` for contiguous cuts, `staggered` for staggered
///   pulse pair; batch cuts, which interleave two PRFs, leave it unset.
fn attach_cut_pulses(
    sweep: &mut Sweep,
    vcp: &VolumeCoveragePattern,
    cut: &VcpCut,
    prf: Option<&RdaPrfData>,
    adaptation: Option<&RdaAdaptationData>,
) {
    let nrays = sweep.nrays();
    if nrays == 0 {
        return;
    }
    let tau_ns = match (vcp.pulse_width, adaptation) {
        (PulseWidth::Short, Some(adaptation)) => Some(adaptation.tau_sp),
        (PulseWidth::Long, Some(adaptation)) => Some(adaptation.tau_lp),
        _ => None,
    };
    if let Some(ns) = tau_ns.filter(|ns| *ns > 0) {
        sweep.ray_vars.pulse_width_s = Some(vec![(f64::from(ns) * 1e-9) as f32; nrays]);
    }
    let prt_s = |hz: Option<f64>| hz.filter(|hz| *hz > 0.0).map(|hz| (1.0 / hz) as f32);
    match cut.waveform {
        WaveformType::Unknown(_) => {}
        WaveformType::ContiguousSurveillance => {
            if cut.surveillance_pulse_count > 0 {
                sweep.ray_vars.n_samples =
                    Some(vec![i32::from(cut.surveillance_pulse_count); nrays]);
            }
            if let Some(prt) = prt_s(prf.and_then(|prf| prf.surveillance_prf_hz(cut))) {
                sweep.ray_vars.prt_s = Some(vec![prt; nrays]);
                sweep.prt_mode = Some(PrtMode::Fixed);
            }
        }
        waveform => {
            let sectors: Option<Vec<usize>> = sweep
                .rays
                .azimuth_deg
                .iter()
                .map(|azimuth| doppler_sector(cut, *azimuth))
                .collect();
            let Some(sectors) = sectors else {
                return;
            };
            sweep.ray_vars.n_samples = Some(
                sectors
                    .iter()
                    .map(|sector| i32::from(cut.doppler_sectors[*sector].pulse_count))
                    .collect(),
            );
            if waveform == WaveformType::StaggeredPulsePair {
                sweep.prt_mode = Some(PrtMode::Staggered);
                return;
            }
            if let Some(prf) = prf {
                let prts: Option<Vec<f32>> = sectors
                    .iter()
                    .map(|sector| prt_s(prf.doppler_prf_hz(cut, *sector)))
                    .collect();
                if let Some(prts) = prts {
                    sweep.ray_vars.prt_s = Some(prts);
                    if waveform != WaveformType::Batch {
                        sweep.prt_mode = Some(PrtMode::Fixed);
                    }
                }
            }
        }
    }
}

/// The Doppler sector (0 to 2) of a cut that holds `azimuth_deg`: among the
/// sectors with a PRF number, the one whose clockwise start edge is the
/// last at or before the azimuth, wrapping to the last edge below the first.
/// `None` when no sector has a PRF number.
fn doppler_sector(cut: &VcpCut, azimuth_deg: f32) -> Option<usize> {
    let azimuth = azimuth_deg.rem_euclid(360.0);
    let used = || {
        cut.doppler_sectors
            .iter()
            .enumerate()
            .filter(|(_, sector)| sector.prf_number != 0)
    };
    let edge = |index: &usize| cut.doppler_sectors[*index].edge_angle_deg;
    used()
        .filter(|(_, sector)| sector.edge_angle_deg <= azimuth)
        .map(|(index, _)| index)
        .max_by(|a, b| edge(a).total_cmp(&edge(b)))
        .or_else(|| {
            used()
                .map(|(index, _)| index)
                .max_by(|a, b| edge(a).total_cmp(&edge(b)))
        })
}

fn location_text(location: Location) -> String {
    match location {
        Location::Halfword(halfword) => format!("halfword {halfword}"),
        Location::HalfwordByte(halfword, byte) => format!("halfword {halfword} byte {byte}"),
        Location::Byte(byte) => format!("byte {byte}"),
    }
}

fn attrs(units: &str, long_name: &str, comment: String) -> Vec<(Box<str>, AttrValue)> {
    let mut attrs = Vec::with_capacity(3);
    attrs.push(("long_name".into(), AttrValue::text(long_name)));
    if !units.is_empty() {
        attrs.push(("units".into(), AttrValue::text(units)));
    }
    attrs.push(("comment".into(), AttrValue::Text(comment.into_boxed_str())));
    attrs
}

fn variable(
    name: String,
    dims: Vec<Box<str>>,
    values: ArrayBuf,
    attrs: Vec<(Box<str>, AttrValue)>,
) -> ExtraVariable {
    let shape = match dims.len() {
        0 => Vec::new(),
        1 => vec![len_u32(values.len())],
        _ => Vec::new(),
    };
    ExtraVariable {
        name: name.into_boxed_str(),
        dims,
        shape,
        values,
        attrs,
    }
}

fn len_u32(len: usize) -> u32 {
    u32::try_from(len).unwrap_or(u32::MAX)
}

/// One element of a scalar as a buffer.
fn scalar_buf(value: Scalar) -> ArrayBuf {
    match value {
        Scalar::I8(v) => ArrayBuf::I8(vec![v]),
        Scalar::U8(v) => ArrayBuf::U8(vec![v]),
        Scalar::I16(v) => ArrayBuf::I16(vec![v]),
        Scalar::U16(v) => ArrayBuf::U16(vec![v]),
        Scalar::I32(v) => ArrayBuf::I32(vec![v]),
        Scalar::U32(v) => ArrayBuf::U32(vec![v]),
        Scalar::I64(v) => ArrayBuf::I64(vec![v]),
        Scalar::U64(v) => ArrayBuf::I64(vec![i64::try_from(v).unwrap_or(i64::MAX)]),
        Scalar::F32(v) => ArrayBuf::F32(vec![v]),
        Scalar::F64(v) => ArrayBuf::F64(vec![v]),
    }
}

fn flag_text(flag: bool) -> &'static str {
    if flag { "T" } else { "F" }
}

/// Variables for the fields of a message that occurs once.
/// `body` is the message body, for flag fields that hold neither "T" nor
/// "F": their stored byte is carried instead (empty when not available).
fn single_message(
    prefix: &str,
    table: &str,
    fields: &[MessageField<'_>],
    body: &[u8],
    out: &mut Vec<ExtraVariable>,
) {
    out.reserve(fields.len());
    for field in fields {
        let name = format!("{prefix}{}", field.name);
        let comment = format!("{table} {}", location_text(field.location));
        let attrs = attrs(field.units, field.long_name, comment);
        let (dims, values) = match &field.value {
            FieldValue::Scalar(value) => (Vec::new(), scalar_buf(*value)),
            FieldValue::Text(text) => (Vec::new(), ArrayBuf::Text(vec![(*text).into()])),
            FieldValue::Flag(Some(flag)) => {
                (Vec::new(), ArrayBuf::Text(vec![flag_text(*flag).into()]))
            }
            // Neither "T" nor "F": the stored byte, verbatim.
            FieldValue::Flag(None) => {
                let byte = match field.location {
                    Location::Byte(offset) => body.get(usize::from(offset)).copied(),
                    _ => None,
                };
                let Some(byte) = byte else { continue };
                let mut attrs = attrs;
                attrs.push((
                    "flag_comment".into(),
                    AttrValue::text("the stored byte, which is neither \"T\" nor \"F\""),
                ));
                out.push(variable(name, Vec::new(), ArrayBuf::U8(vec![byte]), attrs));
                continue;
            }
            FieldValue::Array(values) => (
                vec![format!("{name}_index").into_boxed_str()],
                values.clone(),
            ),
        };
        out.push(variable(name, dims, values, attrs));
    }
}

/// One column of a table of messages.
enum Column {
    U16(Vec<u16>),
    I16(Vec<i16>),
    F32(Vec<f32>),
    /// Rows of a fixed length.
    U16Rows(usize, Vec<u16>),
}

const U16_FILL: u16 = u16::MAX;
const I16_FILL: i16 = i16::MIN;

impl Column {
    fn new(value: &FieldValue<'_>) -> Option<Self> {
        Some(match value {
            FieldValue::Scalar(Scalar::U16(_)) => Self::U16(Vec::new()),
            FieldValue::Scalar(Scalar::I16(_)) => Self::I16(Vec::new()),
            FieldValue::Scalar(Scalar::F32(_)) => Self::F32(Vec::new()),
            FieldValue::Array(ArrayBuf::U16(values)) => Self::U16Rows(values.len(), Vec::new()),
            _ => return None,
        })
    }

    fn push(&mut self, value: Option<&FieldValue<'_>>) {
        match (self, value) {
            (Self::U16(column), Some(FieldValue::Scalar(Scalar::U16(v)))) => column.push(*v),
            (Self::U16(column), _) => column.push(U16_FILL),
            (Self::I16(column), Some(FieldValue::Scalar(Scalar::I16(v)))) => column.push(*v),
            (Self::I16(column), _) => column.push(I16_FILL),
            (Self::F32(column), Some(FieldValue::Scalar(Scalar::F32(v)))) => column.push(*v),
            (Self::F32(column), _) => column.push(f32::NAN),
            (Self::U16Rows(len, column), Some(FieldValue::Array(ArrayBuf::U16(values))))
                if values.len() == *len =>
            {
                column.extend_from_slice(values);
            }
            (Self::U16Rows(len, column), _) => column.extend(std::iter::repeat_n(0, *len)),
        }
    }
}

/// Every message 2 as a table: one entry per message, in file order.
fn rda_status_table(
    statuses: &[Timed<RdaStatus>],
    reference: &chrono::DateTime<chrono::Utc>,
    out: &mut Vec<ExtraVariable>,
) {
    const DIM: &str = "nexrad_rda_status";
    let rows: Vec<Vec<MessageField<'static>>> = statuses
        .iter()
        .map(|(_, status)| rda_status_fields(status))
        .collect();
    let mut columns: Vec<(&MessageField<'static>, Column)> = Vec::new();
    for field in rows.iter().flatten() {
        if columns.iter().any(|(known, _)| known.name == field.name) {
            continue;
        }
        if let Some(column) = Column::new(&field.value) {
            columns.push((field, column));
        }
    }
    for row in &rows {
        for (known, column) in &mut columns {
            column.push(
                row.iter()
                    .find(|field| field.name == known.name)
                    .map(|field| &field.value),
            );
        }
    }
    let count = len_u32(statuses.len());
    out.extend(header_columns(
        statuses,
        reference,
        "nexrad_rda_status",
        DIM,
        "RDA status message",
    ));
    out.push(variable(
        "nexrad_rda_status_layout".to_owned(),
        vec![DIM.into()],
        ArrayBuf::Text(
            statuses
                .iter()
                .map(|(_, status)| match status {
                    RdaStatus::Orda(_) => "orda".into(),
                    RdaStatus::Legacy(_) => "legacy".into(),
                })
                .collect(),
        ),
        vec![(
            "long_name".into(),
            AttrValue::text(
                "Table IV layout: orda (ICD 2620002AA) or legacy (ICD 2620002B), from the message header channel byte",
            ),
        )],
    ));
    for (field, column) in columns {
        let name = format!("{DIM}_{}", field.name);
        let comment = format!("ICD 2620002 Table IV {}", location_text(field.location));
        let mut attrs = attrs(field.units, field.long_name, comment);
        let (dims, shape, values) = match column {
            Column::U16(values) => {
                if values.contains(&U16_FILL) {
                    attrs.push((
                        "_FillValue".into(),
                        AttrValue::Scalar(Scalar::U16(U16_FILL)),
                    ));
                }
                (vec![DIM.into()], vec![count], ArrayBuf::U16(values))
            }
            Column::I16(values) => {
                if values.contains(&I16_FILL) {
                    attrs.push((
                        "_FillValue".into(),
                        AttrValue::Scalar(Scalar::I16(I16_FILL)),
                    ));
                }
                (vec![DIM.into()], vec![count], ArrayBuf::I16(values))
            }
            Column::F32(values) => (vec![DIM.into()], vec![count], ArrayBuf::F32(values)),
            Column::U16Rows(len, values) => (
                vec![DIM.into(), format!("{DIM}_alarm_slot").into_boxed_str()],
                vec![count, len_u32(len)],
                ArrayBuf::U16(values),
            ),
        };
        out.push(ExtraVariable {
            name: name.into_boxed_str(),
            dims,
            shape,
            values,
            attrs,
        });
    }
}

fn table_attrs(
    units: &str,
    long_name: &str,
    table: &str,
    location: &str,
) -> Vec<(Box<str>, AttrValue)> {
    attrs(
        units,
        long_name,
        format!("ICD 2620002AA {table} {location}"),
    )
}

fn scalar_variable(name: &str, value: Scalar, attrs: Vec<(Box<str>, AttrValue)>) -> ExtraVariable {
    variable(name.to_owned(), Vec::new(), scalar_buf(value), attrs)
}

fn vector(
    name: &str,
    dim: &str,
    values: ArrayBuf,
    attrs: Vec<(Box<str>, AttrValue)>,
) -> ExtraVariable {
    variable(name.to_owned(), vec![dim.into()], values, attrs)
}

fn matrix(
    name: &str,
    dims: [&str; 2],
    shape: [usize; 2],
    values: ArrayBuf,
    attrs: Vec<(Box<str>, AttrValue)>,
) -> ExtraVariable {
    ExtraVariable {
        name: name.into(),
        dims: dims.iter().map(|dim| (*dim).into()).collect(),
        shape: shape.iter().map(|len| len_u32(*len)).collect(),
        values,
        attrs,
    }
}

fn vcp_variables(vcp: &VolumeCoveragePattern, out: &mut Vec<ExtraVariable>) {
    const TABLE: &str = "Table XI";
    let hw = |n: u16| format!("halfword {n}");
    let scalars: [(&str, Scalar, &str, String); 10] = [
        (
            "message_size",
            Scalar::U16(vcp.message_size),
            "Message size in halfwords",
            hw(1),
        ),
        (
            "pattern_type",
            Scalar::U16(vcp.pattern_type.code()),
            "Pattern type",
            hw(2),
        ),
        (
            "pattern_number",
            Scalar::U16(vcp.pattern_number),
            "Pattern number",
            hw(3),
        ),
        (
            "number_of_cuts",
            Scalar::U16(vcp.number_of_cuts),
            "Number of elevation cuts",
            hw(4),
        ),
        (
            "version",
            Scalar::U8(vcp.version),
            "VCP version",
            "halfword 5 byte 0".to_owned(),
        ),
        (
            "clutter_map_group",
            Scalar::U8(vcp.clutter_map_group),
            "Clutter map group number",
            "halfword 5 byte 1".to_owned(),
        ),
        (
            "doppler_velocity_resolution",
            Scalar::U8(vcp.doppler_velocity_resolution.code()),
            "Doppler velocity resolution code (2 = 0.5 m/s, 4 = 1.0 m/s)",
            "halfword 6 byte 0".to_owned(),
        ),
        (
            "pulse_width",
            Scalar::U8(vcp.pulse_width.code()),
            "Pulse width code (2 = short, 4 = long)",
            "halfword 6 byte 1".to_owned(),
        ),
        (
            "sequencing",
            Scalar::U16(vcp.sequencing.code),
            "VCP sequencing",
            hw(9),
        ),
        (
            "supplemental_data",
            Scalar::U16(vcp.supplemental.code),
            "VCP supplemental data",
            hw(10),
        ),
    ];
    for (name, value, long_name, location) in scalars {
        out.push(scalar_variable(
            &format!("nexrad_vcp_{name}"),
            value,
            table_attrs("", long_name, TABLE, &location),
        ));
    }
    const CUT: &str = "nexrad_vcp_cut";
    let cuts = &vcp.cuts;
    let f32s = |get: fn(&crate::messages::vcp::VcpCut) -> f32| {
        ArrayBuf::F32(cuts.iter().map(get).collect())
    };
    let mut cut = |name: &str, values: ArrayBuf, units: &str, long_name: &str, element: &str| {
        out.push(vector(
            &format!("nexrad_vcp_{name}"),
            CUT,
            values,
            table_attrs(units, long_name, TABLE, element),
        ));
    };
    cut(
        "elevation_angle",
        f32s(|c| c.elevation_angle_deg),
        "degree",
        "Elevation angle",
        "E1",
    );
    cut(
        "channel_configuration",
        ArrayBuf::U8(
            cuts.iter()
                .map(|c| c.channel_configuration.code())
                .collect(),
        ),
        "",
        "Channel configuration",
        "E2 byte 0",
    );
    cut(
        "waveform_type",
        ArrayBuf::U16(cuts.iter().map(|c| c.waveform.code()).collect()),
        "",
        "Waveform type",
        "E2 byte 1",
    );
    cut(
        "super_resolution_control",
        ArrayBuf::U8(cuts.iter().map(|c| c.super_resolution.code).collect()),
        "",
        "Super resolution control bits",
        "E3 byte 0",
    );
    cut(
        "surveillance_prf_number",
        ArrayBuf::U8(cuts.iter().map(|c| c.surveillance_prf_number).collect()),
        "",
        "Surveillance PRF number",
        "E3 byte 1",
    );
    cut(
        "surveillance_pulse_count",
        ArrayBuf::U16(cuts.iter().map(|c| c.surveillance_pulse_count).collect()),
        "",
        "Surveillance pulse count per radial",
        "E4",
    );
    cut(
        "azimuth_rate",
        f32s(|c| c.azimuth_rate_deg_per_s),
        "degree s-1",
        "Azimuth rate",
        "E5",
    );
    cut(
        "snr_threshold_reflectivity",
        f32s(|c| c.snr_threshold_db.reflectivity),
        "dB",
        "Reflectivity SNR threshold",
        "E6",
    );
    cut(
        "snr_threshold_velocity",
        f32s(|c| c.snr_threshold_db.velocity),
        "dB",
        "Velocity SNR threshold",
        "E7",
    );
    cut(
        "snr_threshold_spectrum_width",
        f32s(|c| c.snr_threshold_db.spectrum_width),
        "dB",
        "Spectrum width SNR threshold",
        "E8",
    );
    cut(
        "snr_threshold_differential_reflectivity",
        f32s(|c| c.snr_threshold_db.differential_reflectivity),
        "dB",
        "Differential reflectivity SNR threshold",
        "E9",
    );
    cut(
        "snr_threshold_differential_phase",
        f32s(|c| c.snr_threshold_db.differential_phase),
        "dB",
        "Differential phase SNR threshold",
        "E10",
    );
    cut(
        "snr_threshold_correlation_coefficient",
        f32s(|c| c.snr_threshold_db.correlation_coefficient),
        "dB",
        "Correlation coefficient SNR threshold",
        "E11",
    );
    cut(
        "cut_supplemental_data",
        ArrayBuf::U16(cuts.iter().map(|c| c.supplemental.code).collect()),
        "",
        "Cut supplemental data",
        "E15",
    );
    cut(
        "ebc_angle",
        f32s(|c| c.ebc_angle_deg),
        "degree",
        "Elevation-based clutter (EBC) angle correction",
        "E19",
    );
    let sectors = [CUT, "nexrad_vcp_sector"];
    let shape = [cuts.len(), 3];
    let per_sector = |get: fn(&crate::messages::vcp::DopplerSector) -> u16| {
        ArrayBuf::U16(
            cuts.iter()
                .flat_map(|c| c.doppler_sectors.iter().map(get))
                .collect(),
        )
    };
    out.push(matrix(
        "nexrad_vcp_doppler_edge_angle",
        sectors,
        shape,
        ArrayBuf::F32(
            cuts.iter()
                .flat_map(|c| c.doppler_sectors.iter().map(|s| s.edge_angle_deg))
                .collect(),
        ),
        table_attrs(
            "degree",
            "Doppler sector clockwise edge angle",
            TABLE,
            "E12, E16, E20",
        ),
    ));
    out.push(matrix(
        "nexrad_vcp_doppler_prf_number",
        sectors,
        shape,
        per_sector(|s| s.prf_number),
        table_attrs("", "Doppler PRF number", TABLE, "E13, E17, E21"),
    ));
    out.push(matrix(
        "nexrad_vcp_doppler_pulse_count",
        sectors,
        shape,
        per_sector(|s| s.pulse_count),
        table_attrs("", "Doppler pulse count per radial", TABLE, "E14, E18, E22"),
    ));
}

fn censor_variables(zones: &ClutterCensorZones, out: &mut Vec<ExtraVariable>) {
    const TABLE: &str = "Table XII";
    const DIM: &str = "nexrad_clutter_censor_zone";
    out.push(scalar_variable(
        "nexrad_clutter_censor_number_of_regions",
        Scalar::U16(u16::try_from(zones.regions.len()).unwrap_or(u16::MAX)),
        table_attrs("", "Number of override regions", TABLE, "halfword 1"),
    ));
    let regions = &zones.regions;
    let column = |get: fn(&crate::messages::clutter_censor::CensorZone) -> u16| {
        ArrayBuf::U16(regions.iter().map(get).collect())
    };
    let items: [(&str, ArrayBuf, &str, &str, &str); 6] = [
        (
            "start_range",
            column(|z| z.start_range_km),
            "km",
            "Start range",
            "R1",
        ),
        (
            "stop_range",
            column(|z| z.stop_range_km),
            "km",
            "Stop range",
            "R2",
        ),
        (
            "start_azimuth",
            column(|z| z.start_azimuth_deg),
            "degree",
            "Start azimuth",
            "R3",
        ),
        (
            "stop_azimuth",
            column(|z| z.stop_azimuth_deg),
            "degree",
            "Stop azimuth",
            "R4",
        ),
        (
            "elevation_segment",
            column(|z| z.elevation_segment),
            "",
            "Elevation segment number",
            "R5",
        ),
        (
            "operator_select_code",
            column(|z| z.operator_select.code()),
            "",
            "Operator select code (0 bypass filter, 1 bypass map in control, 2 force filter)",
            "R6",
        ),
    ];
    for (name, values, units, long_name, element) in items {
        out.push(vector(
            &format!("nexrad_clutter_censor_{name}"),
            DIM,
            values,
            table_attrs(units, long_name, TABLE, element),
        ));
    }
}

fn bypass_variables(map: &ClutterFilterBypassMap, out: &mut Vec<ExtraVariable>) {
    let (table, layout) = match map.layout {
        BypassMapLayout::Current => ("ICD 2620002AA Table IX", "current"),
        BypassMapLayout::Legacy => ("ICD 2620002B Table IX", "legacy"),
    };
    let comment = |location: &str| format!("{table} {location}");
    out.push(variable(
        "nexrad_bypass_map_layout".to_owned(),
        Vec::new(),
        ArrayBuf::Text(vec![layout.into()]),
        attrs(
            "",
            "Table IX revision: current (360 radials of 1 degree) or legacy (256 radials of 1.40625 degrees)",
            comment("layout"),
        ),
    ));
    if let Some(date) = map.generation_date {
        out.push(scalar_variable(
            "nexrad_bypass_map_generation_date",
            Scalar::U16(date),
            attrs(
                "",
                "Generation date (days, 1 January 1970 = 1)",
                comment("halfword 1"),
            ),
        ));
    }
    if let Some(minutes) = map.generation_minutes {
        out.push(scalar_variable(
            "nexrad_bypass_map_generation_time",
            Scalar::U16(minutes),
            attrs(
                "min",
                "Generation time (minutes after midnight UTC)",
                comment("halfword 2"),
            ),
        ));
    }
    const SEGMENT: &str = "nexrad_bypass_map_segment";
    out.push(vector(
        "nexrad_bypass_map_segment_number",
        SEGMENT,
        ArrayBuf::U16(map.segments.iter().map(|s| s.segment_number).collect()),
        attrs("", "Elevation segment number", comment("segment number")),
    ));
    let radials = map.layout.radial_count();
    let halfwords = crate::messages::bypass_map::HALFWORDS_PER_RADIAL;
    let mut bits = Vec::with_capacity(map.segments.len() * radials * halfwords);
    for segment in &map.segments {
        for radial in 0..radials {
            match segment.radials.get(radial) {
                Some(words) => bits.extend_from_slice(words),
                None => bits.extend(std::iter::repeat_n(0, halfwords)),
            }
        }
    }
    out.push(ExtraVariable {
        name: "nexrad_bypass_map_bins".into(),
        dims: vec![
            SEGMENT.into(),
            "nexrad_bypass_map_radial".into(),
            "nexrad_bypass_map_halfword".into(),
        ],
        shape: vec![
            len_u32(map.segments.len()),
            len_u32(radials),
            len_u32(halfwords),
        ],
        values: ArrayBuf::U16(bits),
        attrs: attrs(
            "",
            "Range bin bits of each radial: bin 0 (1 km) is the most significant bit of halfword 0; 1 = bypass the clutter filters",
            comment("radial data"),
        ),
    });
}

fn clutter_filter_map_variables(map: &ClutterFilterMap, out: &mut Vec<ExtraVariable>) {
    const TABLE: &str = "Table XIV";
    out.push(scalar_variable(
        "nexrad_clutter_filter_map_generation_date",
        Scalar::U16(map.generation_date),
        table_attrs(
            "",
            "Generation date (days, 1 January 1970 = 1)",
            TABLE,
            "halfword 1",
        ),
    ));
    out.push(scalar_variable(
        "nexrad_clutter_filter_map_generation_time",
        Scalar::U16(map.generation_minutes),
        table_attrs(
            "min",
            "Generation time (minutes after midnight UTC)",
            TABLE,
            "halfword 2",
        ),
    ));
    const ZONE: &str = "nexrad_clutter_filter_map_zone";
    let azimuths = map
        .segments
        .iter()
        .map(|segment| segment.azimuth_count())
        .max()
        .unwrap_or(0);
    let mut counts = Vec::with_capacity(map.segments.len() * azimuths);
    let mut op_codes = Vec::new();
    let mut end_ranges = Vec::new();
    for segment in &map.segments {
        for azimuth in 0..azimuths {
            let zones = segment.azimuth(azimuth).unwrap_or(&[]);
            counts.push(u16::try_from(zones.len()).unwrap_or(u16::MAX));
            op_codes.extend(zones.iter().map(|zone| zone.op_code.code()));
            end_ranges.extend(zones.iter().map(|zone| zone.end_range_km));
        }
    }
    let mut count_attrs = table_attrs(
        "",
        "Number of range zones of each azimuth segment (1 degree) of each elevation segment",
        TABLE,
        "range zone count",
    );
    count_attrs.push(("sample_dimension".into(), AttrValue::text(ZONE)));
    out.push(matrix(
        "nexrad_clutter_filter_map_zone_count",
        [
            "nexrad_clutter_filter_map_segment",
            "nexrad_clutter_filter_map_azimuth",
        ],
        [map.segments.len(), azimuths],
        ArrayBuf::U16(counts),
        count_attrs,
    ));
    out.push(vector(
        "nexrad_clutter_filter_map_op_code",
        ZONE,
        ArrayBuf::U16(op_codes),
        table_attrs(
            "",
            "Range zone op code (0 bypass filter, 1 bypass map in control, 2 force filter), zones of every azimuth segment in order",
            TABLE,
            "R1",
        ),
    ));
    out.push(vector(
        "nexrad_clutter_filter_map_end_range",
        ZONE,
        ArrayBuf::U16(end_ranges),
        table_attrs("km", "Range zone stop range", TABLE, "R2"),
    ));
}

fn prf_variables(prf: &RdaPrfData, out: &mut Vec<ExtraVariable>) {
    const TABLE: &str = "Table XVIII";
    const WAVEFORM: &str = "nexrad_prf_waveform";
    out.push(scalar_variable(
        "nexrad_prf_number_of_waveforms",
        Scalar::U16(prf.number_of_waveforms),
        table_attrs("", "Number of waveforms", TABLE, "halfword 1"),
    ));
    out.push(vector(
        "nexrad_prf_waveform_type",
        WAVEFORM,
        ArrayBuf::U16(prf.waveforms.iter().map(|w| w.waveform.code()).collect()),
        table_attrs("", "Waveform type", TABLE, "P1"),
    ));
    out.push(vector(
        "nexrad_prf_count",
        WAVEFORM,
        ArrayBuf::U16(
            prf.waveforms
                .iter()
                .map(|w| u16::try_from(w.prfs_mhz.len()).unwrap_or(u16::MAX))
                .collect(),
        ),
        table_attrs("", "Number of PRFs", TABLE, "P2"),
    ));
    let widest = prf
        .waveforms
        .iter()
        .map(|w| w.prfs_mhz.len())
        .max()
        .unwrap_or(0);
    let mut values = Vec::with_capacity(prf.waveforms.len() * widest);
    for waveform in &prf.waveforms {
        values.extend_from_slice(&waveform.prfs_mhz);
        values.extend(std::iter::repeat_n(
            u32::MAX,
            widest - waveform.prfs_mhz.len(),
        ));
    }
    let mut attrs = table_attrs(
        "mHz",
        "PRF of each PRF number (index 0 is PRF number 1)",
        TABLE,
        "P3 onward",
    );
    if prf.waveforms.iter().any(|w| w.prfs_mhz.len() < widest) {
        attrs.push((
            "_FillValue".into(),
            AttrValue::Scalar(Scalar::U32(u32::MAX)),
        ));
    }
    out.push(matrix(
        "nexrad_prf_value",
        [WAVEFORM, "nexrad_prf_number"],
        [prf.waveforms.len(), widest],
        ArrayBuf::U32(values),
        attrs,
    ));
}

fn console_variables(
    messages: &[Timed<ConsoleMessage>],
    reference: &chrono::DateTime<chrono::Utc>,
    out: &mut Vec<ExtraVariable>,
) {
    const TABLE: &str = "Table VI";
    const DIM: &str = "nexrad_console_message";
    out.push(vector(
        "nexrad_console_message_type",
        DIM,
        ArrayBuf::U8(
            messages
                .iter()
                .map(|(header, _)| header.message_type)
                .collect(),
        ),
        table_attrs(
            "",
            "Message type (4 RDA to RPG, 10 RPG to RDA)",
            TABLE,
            "message header",
        ),
    ));
    out.extend(header_columns(
        messages,
        reference,
        "nexrad_console_message",
        DIM,
        "console message",
    ));
    let mut size_attrs = table_attrs("byte", "Number of bytes of text", TABLE, "halfword 1");
    size_attrs.push((
        "sample_dimension".into(),
        AttrValue::text("nexrad_console_byte"),
    ));
    out.push(vector(
        "nexrad_console_message_size",
        DIM,
        ArrayBuf::U16(messages.iter().map(|(_, m)| m.message_size).collect()),
        size_attrs,
    ));
    out.push(vector(
        "nexrad_console_text",
        "nexrad_console_byte",
        ArrayBuf::U8(
            messages
                .iter()
                .flat_map(|(_, m)| m.bytes.iter().copied())
                .collect(),
        ),
        table_attrs(
            "",
            "Message text bytes of every message in order",
            TABLE,
            "halfwords 2 to 203",
        ),
    ));
}

/// Every message 6 (RDA Control Commands, Table X) as a table over
/// `nexrad_control_commands`: each command halfword's code as sent.
fn control_variables(
    messages: &[Timed<RdaControlCommands>],
    reference: &chrono::DateTime<chrono::Utc>,
    out: &mut Vec<ExtraVariable>,
) {
    const TABLE: &str = "Table X";
    const DIM: &str = "nexrad_control_commands";
    out.extend(header_columns(
        messages,
        reference,
        "nexrad_control_commands",
        DIM,
        "RDA control commands message",
    ));
    let column = |get: fn(&RdaControlCommands) -> u16| {
        ArrayBuf::U16(messages.iter().map(|(_, commands)| get(commands)).collect())
    };
    let items: [(&str, ArrayBuf, &str, &str); 13] = [
        (
            "rda_state",
            column(|c| c.rda_state.code()),
            "RDA state command (0 no change, 32769 standby, 32772 operate, 32776 restart)",
            "halfword 1",
        ),
        (
            "rda_log",
            column(|c| c.rda_log.log_code()),
            "RDA log command (0 no change, 1 enable, 2 disable)",
            "halfword 2",
        ),
        (
            "auxiliary_power",
            column(|c| c.auxiliary_power.code()),
            "Auxiliary power generator control (0 no change, 32770 switch to utility, 32772 switch to auxiliary)",
            "halfword 3",
        ),
        (
            "control_authorization",
            column(|c| c.control_authorization.code()),
            "RDA control commands and authorization",
            "halfword 4",
        ),
        (
            "restart",
            column(|c| c.restart.code()),
            "Restart VCP or elevation cut (32768 restart VCP; plus the cut number to restart a cut)",
            "halfword 5",
        ),
        (
            "select_local_vcp",
            column(|c| c.select_local_vcp.code()),
            "Select local VCP number for the next volume scan (0 use remote pattern, 32767 no change)",
            "halfword 6",
        ),
        (
            "super_resolution",
            column(|c| c.super_resolution.code()),
            "Super resolution control (0 no change, 2 enable, 4 disable)",
            "halfword 8",
        ),
        (
            "clutter_mitigation_decision",
            column(|c| c.clutter_mitigation_decision.code()),
            "Clutter mitigation decision (CMD) control (0 no change, 2 enable, 4 disable)",
            "halfword 9",
        ),
        (
            "avset",
            column(|c| c.avset.code()),
            "AVSET control (0 no change, 2 enable, 4 disable)",
            "halfword 10",
        ),
        (
            "channel_control",
            column(|c| c.channel_control.code()),
            "Channel control command (0 no change, 1 set controlling, 2 set non-controlling)",
            "halfword 12",
        ),
        (
            "performance_check",
            column(|c| c.performance_check.code()),
            "Performance check control (0 no change, 1 force a performance check)",
            "halfword 13",
        ),
        (
            "zdr_bias_estimate",
            column(|c| c.zdr_bias_estimate.code()),
            "ZDR bias estimate weighted mean (0 not available, 1 no change, 2 to 1058 coded as (code - 418) / 32 dB)",
            "halfword 14",
        ),
        (
            "spot_blanking",
            column(|c| c.spot_blanking.code()),
            "Spot blanking control (0 no change, 2 enable, 4 disable)",
            "halfword 21",
        ),
    ];
    for (name, values, long_name, location) in items {
        out.push(vector(
            &format!("{DIM}_{name}"),
            DIM,
            values,
            table_attrs("", long_name, TABLE, location),
        ));
    }
}

/// Every message 9 (Request for Data, Table XIII) as a table over
/// `nexrad_request_for_data`.
fn request_variables(
    messages: &[Timed<RequestForData>],
    reference: &chrono::DateTime<chrono::Utc>,
    out: &mut Vec<ExtraVariable>,
) {
    const DIM: &str = "nexrad_request_for_data";
    out.extend(header_columns(
        messages,
        reference,
        "nexrad_request_for_data",
        DIM,
        "request for data message",
    ));
    out.push(vector(
        "nexrad_request_for_data_type",
        DIM,
        ArrayBuf::U16(
            messages
                .iter()
                .map(|(_, request)| request.request.code())
                .collect(),
        ),
        table_attrs(
            "",
            "Data request type (129 RDA status, 130 performance/maintenance data, 132 clutter filter bypass map, 136 clutter filter map, 144 RDA adaptation data, 160 volume coverage pattern)",
            "Table XIII",
            "halfword 1",
        ),
    ));
}

/// Every message 11 and 12 (Loop Back Test, Table VIII) as a table over
/// `nexrad_loopback`, with the bit patterns one after another over
/// `nexrad_loopback_byte`.
fn loopback_variables(
    messages: &[Timed<LoopbackTest>],
    reference: &chrono::DateTime<chrono::Utc>,
    out: &mut Vec<ExtraVariable>,
) {
    const TABLE: &str = "Table VIII";
    const DIM: &str = "nexrad_loopback";
    out.push(vector(
        "nexrad_loopback_message_type",
        DIM,
        ArrayBuf::U8(
            messages
                .iter()
                .map(|(header, _)| header.message_type)
                .collect(),
        ),
        table_attrs(
            "",
            "Message type (11 RDA to RPG, 12 RPG to RDA)",
            TABLE,
            "message header",
        ),
    ));
    out.extend(header_columns(
        messages,
        reference,
        "nexrad_loopback",
        DIM,
        "loop back test message",
    ));
    out.push(vector(
        "nexrad_loopback_message_size",
        DIM,
        ArrayBuf::U16(messages.iter().map(|(_, test)| test.message_size).collect()),
        table_attrs(
            "",
            "Message size in halfwords, including this one and not the message header",
            TABLE,
            "halfword 1",
        ),
    ));
    let mut length_attrs = table_attrs("byte", "Length of the bit pattern", TABLE, "halfword 1");
    length_attrs.push((
        "sample_dimension".into(),
        AttrValue::text("nexrad_loopback_byte"),
    ));
    out.push(vector(
        "nexrad_loopback_pattern_length",
        DIM,
        ArrayBuf::U32(
            messages
                .iter()
                .map(|(_, test)| len_u32(test.bit_pattern.len()))
                .collect(),
        ),
        length_attrs,
    ));
    out.push(vector(
        "nexrad_loopback_bit_pattern",
        "nexrad_loopback_byte",
        ArrayBuf::U8(
            messages
                .iter()
                .flat_map(|(_, test)| test.bit_pattern.iter().copied())
                .collect(),
        ),
        table_attrs(
            "",
            "Test bit pattern of every message in order",
            TABLE,
            "halfwords 2 onward",
        ),
    ));
}

/// Every message whose layout the decoders do not read (the legacy RDA
/// messages 3 and 18, message 29 and the types Table I does not define) as a
/// table over `nexrad_unparsed_message`, with the frames of each
/// ([`unparsed_frames`]; the message header and body of a variable-length
/// message) one after another over `nexrad_unparsed_message_byte`.
fn unparsed_variables(
    messages: &[Timed<Vec<u8>>],
    reference: &chrono::DateTime<chrono::Utc>,
    out: &mut Vec<ExtraVariable>,
) {
    const DIM: &str = "nexrad_unparsed_message";
    out.push(vector(
        "nexrad_unparsed_message_type",
        DIM,
        ArrayBuf::U8(
            messages
                .iter()
                .map(|(header, _)| header.message_type)
                .collect(),
        ),
        table_attrs("", "Message type", "Table II", "message header"),
    ));
    out.push(vector(
        "nexrad_unparsed_message_channels",
        DIM,
        ArrayBuf::U8(messages.iter().map(|(header, _)| header.channels).collect()),
        table_attrs(
            "",
            "RDA redundant channel byte (for messages 3 and 18, bit 3 clear: a legacy RDA, whose layout of these messages is not decoded)",
            "Table II",
            "message header",
        ),
    ));
    out.push(time_column(
        messages,
        reference,
        "nexrad_unparsed_message_time",
        DIM,
        "message",
    ));
    let mut length_attrs = table_attrs(
        "byte",
        "Length of the message's bytes: 2420 per fixed frame, the message size of a variable-length message",
        "Table II",
        "frames",
    );
    length_attrs.push((
        "sample_dimension".into(),
        AttrValue::text("nexrad_unparsed_message_byte"),
    ));
    out.push(vector(
        "nexrad_unparsed_message_length",
        DIM,
        ArrayBuf::U32(
            messages
                .iter()
                .map(|(_, frames)| len_u32(frames.len()))
                .collect(),
        ),
        length_attrs,
    ));
    out.push(vector(
        "nexrad_unparsed_message_frames",
        "nexrad_unparsed_message_byte",
        ArrayBuf::U8(
            messages
                .iter()
                .flat_map(|(_, frames)| frames.iter().copied())
                .collect(),
        ),
        table_attrs(
            "",
            "Every frame of every message in order, verbatim from the 16-byte message header to the end of the 2432-byte frame (each segment's header gives its declared size and segment number); a variable-length message (a size of 65535, or message 29) is its message header and body",
            "Table II",
            "frames",
        ),
    ));
}

/// The message header (Table II) of every non-radial message frame the
/// volume decoder kept, in file order, one entry per frame (each segment
/// of a segmented message, each variable-length message), as stored, over
/// `nexrad_metadata_message`: `nexrad_metadata_message_type`, `_channels`,
/// `_size`, `_sequence_number`, `_date`, `_milliseconds`, `_segments` and
/// `_segment_number`. Nothing when there is none.
pub(crate) fn message_header_table(headers: &[MessageHeader], out: &mut Vec<ExtraVariable>) {
    if headers.is_empty() {
        return;
    }
    const DIM: &str = "nexrad_metadata_message";
    let column = |pick: fn(&MessageHeader) -> u16| -> ArrayBuf {
        ArrayBuf::U16(headers.iter().map(pick).collect())
    };
    let table = "Table II";
    let where_ = |halfwords: &str| {
        format!("message header {halfwords}, each non-radial message frame in file order")
    };
    out.push(vector(
        "nexrad_metadata_message_type",
        DIM,
        ArrayBuf::U8(headers.iter().map(|header| header.message_type).collect()),
        table_attrs("", "Message type", table, &where_("halfword 2, low byte")),
    ));
    out.push(vector(
        "nexrad_metadata_message_channels",
        DIM,
        ArrayBuf::U8(headers.iter().map(|header| header.channels).collect()),
        table_attrs(
            "",
            "RDA redundant channel byte",
            table,
            &where_("halfword 2, high byte"),
        ),
    ));
    out.push(vector(
        "nexrad_metadata_message_size",
        DIM,
        column(|header| header.size_halfwords),
        table_attrs(
            "",
            "Message size in halfwords (65535: halfwords 7-8 hold the size in bytes)",
            table,
            &where_("halfword 1"),
        ),
    ));
    out.push(vector(
        "nexrad_metadata_message_sequence_number",
        DIM,
        column(|header| header.sequence_id),
        table_attrs("", "Message sequence number", table, &where_("halfword 3")),
    ));
    out.push(vector(
        "nexrad_metadata_message_date",
        DIM,
        column(|header| header.date),
        table_attrs(
            "days since 1969-12-31T00:00:00Z",
            "Message generation date (1 January 1970 is day 1)",
            table,
            &where_("halfword 4"),
        ),
    ));
    out.push(vector(
        "nexrad_metadata_message_milliseconds",
        DIM,
        ArrayBuf::U32(headers.iter().map(|header| header.milliseconds).collect()),
        table_attrs(
            "ms",
            "Message generation time, milliseconds past midnight UTC of nexrad_metadata_message_date",
            table,
            &where_("halfwords 5-6"),
        ),
    ));
    out.push(vector(
        "nexrad_metadata_message_segments",
        DIM,
        column(|header| header.segments),
        table_attrs(
            "",
            "Number of message segments (with a size of 65535: the high halfword of the size in bytes)",
            table,
            &where_("halfword 7"),
        ),
    ));
    out.push(vector(
        "nexrad_metadata_message_segment_number",
        DIM,
        column(|header| header.segment_number),
        table_attrs(
            "",
            "Message segment number (with a size of 65535: the low halfword of the size in bytes)",
            table,
            &where_("halfword 8"),
        ),
    ));
}

fn rda_log_variables(
    logs: Vec<Timed<RdaLogData>>,
    reference: &chrono::DateTime<chrono::Utc>,
    out: &mut Vec<ExtraVariable>,
) {
    const TABLE: &str = "Table XVIV";
    const DIM: &str = "nexrad_rda_log";
    out.extend(header_columns(
        &logs,
        reference,
        "nexrad_rda_log",
        DIM,
        "RDA log data message",
    ));
    let u32s =
        |get: fn(&RdaLogData) -> u32| ArrayBuf::U32(logs.iter().map(|(_, log)| get(log)).collect());
    out.push(vector(
        "nexrad_rda_log_identifier",
        DIM,
        ArrayBuf::Text(
            logs.iter()
                .map(|(_, l)| l.identifier.as_str().into())
                .collect(),
        ),
        table_attrs("", "Log file name", TABLE, "halfwords 2 to 14"),
    ));
    let items: [(&str, ArrayBuf, &str, &str, &str); 5] = [
        (
            "version",
            u32s(|l| l.version),
            "",
            "Message format version",
            "halfwords 0 to 1",
        ),
        (
            "data_version",
            u32s(|l| l.data_version),
            "",
            "Log version",
            "halfwords 15 to 16",
        ),
        (
            "compression",
            u32s(|l| match l.compression {
                RdaLogCompression::Uncompressed => 0,
                RdaLogCompression::Gzip => 1,
                RdaLogCompression::Bzip2 => 2,
                RdaLogCompression::Zip => 3,
                RdaLogCompression::Unknown(code) => code,
            }),
            "",
            "Compression type (0 none, 1 gzip, 2 bzip2, 3 zip)",
            "halfwords 17 to 18",
        ),
        (
            "compressed_size",
            u32s(|l| l.compressed_size),
            "byte",
            "Compressed size",
            "halfwords 19 to 20",
        ),
        (
            "decompressed_size",
            u32s(|l| l.decompressed_size),
            "byte",
            "Decompressed size",
            "halfwords 21 to 22",
        ),
    ];
    for (name, values, units, long_name, location) in items {
        out.push(vector(
            &format!("nexrad_rda_log_{name}"),
            DIM,
            values,
            table_attrs(units, long_name, TABLE, location),
        ));
    }
    let mut count_attrs = table_attrs("byte", "Length of the decompressed log data", TABLE, "data");
    count_attrs.push((
        "sample_dimension".into(),
        AttrValue::text("nexrad_rda_log_byte"),
    ));
    out.push(vector(
        "nexrad_rda_log_data_length",
        DIM,
        ArrayBuf::U32(logs.iter().map(|(_, l)| len_u32(l.data.len())).collect()),
        count_attrs,
    ));
    // The log data moves into one buffer: one log's is taken as it is, and
    // several are appended one by one, each freed once copied.
    let mut data = Vec::new();
    for (_, log) in logs {
        if data.is_empty() {
            data = log.data;
        } else {
            data.extend_from_slice(&log.data);
        }
    }
    out.push(vector(
        "nexrad_rda_log_data",
        "nexrad_rda_log_byte",
        ArrayBuf::U8(data),
        table_attrs(
            "",
            "Decompressed log data of every message in order",
            TABLE,
            "halfword 34 onward",
        ),
    ));
}

// Tables generated from the field documentation of `messages::performance`
// and `messages::adaptation`.

#[rustfmt::skip]
fn performance_table<'a>(p: &'a PerformanceMaintenance, f: &mut Fields<'a>) {
    use Location::{Halfword, HalfwordByte};
    // Halfwords 1 to 57.
    f.push("loop_back_test_status", Halfword(2), "", "Loop back test status", Value::Scalar(Scalar::U16(p.communications.loop_back_test_status)));
    f.push("t1_output_frames", Halfword(3), "", "T1 output frames", Value::Scalar(Scalar::U32(p.communications.t1_output_frames)));
    f.push("t1_input_frames", Halfword(5), "", "T1 input frames", Value::Scalar(Scalar::U32(p.communications.t1_input_frames)));
    f.push("router_memory_used", Halfword(7), "byte", "Router memory used by applications", Value::Scalar(Scalar::U32(p.communications.router_memory_used)));
    f.push("router_memory_free", Halfword(9), "byte", "Router memory free", Value::Scalar(Scalar::U32(p.communications.router_memory_free)));
    f.push("router_memory_utilization", Halfword(11), "percent", "Router memory utilization", Value::Scalar(Scalar::U16(p.communications.router_memory_utilization)));
    f.push("route_to_rpg", Halfword(12), "", "Route to RPG", Value::Scalar(Scalar::U16(p.communications.route_to_rpg)));
    f.push("t1_port_status", Halfword(13), "", "T1 port status", Value::Scalar(Scalar::U16(p.communications.t1_port_status)));
    f.push("router_dedicated_ethernet_port_status", Halfword(14), "", "Router dedicated (local) Ethernet port to the RPG", Value::Scalar(Scalar::U16(p.communications.router_dedicated_ethernet_port_status)));
    f.push("router_commercial_ethernet_port_status", Halfword(15), "", "Router commercial Ethernet port to the RPG", Value::Scalar(Scalar::U16(p.communications.router_commercial_ethernet_port_status)));
    f.push("csu_24hr_errored_seconds", Halfword(21), "s", "CSU errored seconds in the previous 24 hours", Value::Scalar(Scalar::U32(p.communications.csu_24hr_errored_seconds)));
    f.push("csu_24hr_severely_errored_seconds", Halfword(23), "s", "CSU severely errored seconds in the previous 24 hours", Value::Scalar(Scalar::U32(p.communications.csu_24hr_severely_errored_seconds)));
    f.push("csu_24hr_severely_errored_framing_seconds", Halfword(25), "s", "CSU severely errored framing seconds in the previous 24 hours", Value::Scalar(Scalar::U32(p.communications.csu_24hr_severely_errored_framing_seconds)));
    f.push("csu_24hr_unavailable_seconds", Halfword(27), "s", "CSU unavailable seconds in the previous 24 hours", Value::Scalar(Scalar::U32(p.communications.csu_24hr_unavailable_seconds)));
    f.push("csu_24hr_controlled_slip_seconds", Halfword(29), "s", "CSU controlled slip seconds in the previous 24 hours", Value::Scalar(Scalar::U32(p.communications.csu_24hr_controlled_slip_seconds)));
    f.push("csu_24hr_path_coding_violations", Halfword(31), "1", "CSU path coding violations in the previous 24 hours", Value::Scalar(Scalar::U32(p.communications.csu_24hr_path_coding_violations)));
    f.push("csu_24hr_line_errored_seconds", Halfword(33), "s", "CSU line errored seconds in the previous 24 hours", Value::Scalar(Scalar::U32(p.communications.csu_24hr_line_errored_seconds)));
    f.push("csu_24hr_bursty_errored_seconds", Halfword(35), "s", "CSU bursty errored seconds in the previous 24 hours", Value::Scalar(Scalar::U32(p.communications.csu_24hr_bursty_errored_seconds)));
    f.push("csu_24hr_degraded_minutes", Halfword(37), "min", "CSU degraded minutes in the previous 24 hours", Value::Scalar(Scalar::U32(p.communications.csu_24hr_degraded_minutes)));
    f.push("lan_switch_cpu_utilization", Halfword(41), "percent", "LAN switch CPU utilization", Value::Scalar(Scalar::U32(p.communications.lan_switch_cpu_utilization)));
    f.push("lan_switch_memory_utilization", Halfword(43), "percent", "LAN switch memory utilization", Value::Scalar(Scalar::U16(p.communications.lan_switch_memory_utilization)));
    f.push("ifdr_chassis_temperature", Halfword(45), "degC", "IFDR chassis (case) temperature", Value::Scalar(Scalar::I16(p.communications.ifdr_chassis_temperature)));
    f.push("ifdr_fpga_temperature", Halfword(46), "degC", "IFDR FPGA temperature", Value::Scalar(Scalar::I16(p.communications.ifdr_fpga_temperature)));
    f.push("ntp_status", Halfword(47), "", "NTP synchronization status", Value::Scalar(Scalar::U16(p.communications.ntp_status)));
    f.push("ipc_status", Halfword(53), "", "Status of the communications between the channels of a redundant system", Value::Scalar(Scalar::U16(p.communications.ipc_status)));
    f.push("commanded_channel_control", Halfword(54), "", "Channel the RDA has commanded to be the controlling channel (not necessarily the one in control)", Value::Scalar(Scalar::U16(p.communications.commanded_channel_control)));
    // Halfwords 58 to 98.
    f.push("polarization", Halfword(58), "", "AME polarization", Value::Scalar(Scalar::U16(p.ame.polarization)));
    f.push("internal_temperature", Halfword(59), "degC", "AME internal temperature", Value::Scalar(Scalar::F32(p.ame.internal_temperature)));
    f.push("receiver_module_temperature", Halfword(61), "degC", "AME receiver module temperature", Value::Scalar(Scalar::F32(p.ame.receiver_module_temperature)));
    f.push("bite_cal_module_temperature", Halfword(63), "degC", "AME BITE/CAL module temperature", Value::Scalar(Scalar::F32(p.ame.bite_cal_module_temperature)));
    f.push("peltier_pulse_width_modulation", Halfword(65), "percent", "AME Peltier pulse width modulation", Value::Scalar(Scalar::U16(p.ame.peltier_pulse_width_modulation)));
    f.push("peltier_status", Halfword(66), "", "AME Peltier status", Value::Scalar(Scalar::U16(p.ame.peltier_status)));
    f.push("ad_converter_status", Halfword(67), "", "AME A/D converter status", Value::Scalar(Scalar::U16(p.ame.ad_converter_status)));
    f.push("state", Halfword(68), "", "AME state", Value::Scalar(Scalar::U16(p.ame.state)));
    f.push("ps_3_3v_voltage", Halfword(69), "V", "AME +3.3 V power supply voltage", Value::Scalar(Scalar::F32(p.ame.ps_3_3v_voltage)));
    f.push("ps_5v_voltage", Halfword(71), "V", "AME +5 V power supply voltage", Value::Scalar(Scalar::F32(p.ame.ps_5v_voltage)));
    f.push("ps_6_5v_voltage", Halfword(73), "V", "AME +6.5 V power supply voltage", Value::Scalar(Scalar::F32(p.ame.ps_6_5v_voltage)));
    f.push("ps_15v_voltage", Halfword(75), "V", "AME +15 V power supply voltage", Value::Scalar(Scalar::F32(p.ame.ps_15v_voltage)));
    f.push("ps_48v_voltage", Halfword(77), "V", "AME +48 V power supply voltage", Value::Scalar(Scalar::F32(p.ame.ps_48v_voltage)));
    f.push("stalo_power", Halfword(79), "V", "AME STALO power", Value::Scalar(Scalar::F32(p.ame.stalo_power)));
    f.push("peltier_current", Halfword(81), "A", "Peltier current", Value::Scalar(Scalar::F32(p.ame.peltier_current)));
    f.push("adc_calibration_reference_voltage", Halfword(83), "V", "ADC calibration reference voltage", Value::Scalar(Scalar::F32(p.ame.adc_calibration_reference_voltage)));
    f.push("mode", Halfword(85), "", "AME mode", Value::Scalar(Scalar::U16(p.ame.mode)));
    f.push("peltier_mode", Halfword(86), "", "AME Peltier mode", Value::Scalar(Scalar::U16(p.ame.peltier_mode)));
    f.push("peltier_inside_fan_current", Halfword(87), "A", "AME Peltier inside fan current", Value::Scalar(Scalar::F32(p.ame.peltier_inside_fan_current)));
    f.push("peltier_outside_fan_current", Halfword(89), "A", "AME Peltier outside fan current", Value::Scalar(Scalar::F32(p.ame.peltier_outside_fan_current)));
    f.push("horizontal_tr_limiter_voltage", Halfword(91), "V", "Horizontal TR limiter voltage", Value::Scalar(Scalar::F32(p.ame.horizontal_tr_limiter_voltage)));
    f.push("vertical_tr_limiter_voltage", Halfword(93), "V", "Vertical TR limiter voltage", Value::Scalar(Scalar::F32(p.ame.vertical_tr_limiter_voltage)));
    f.push("adc_calibration_offset_voltage", Halfword(95), "mV", "ADC calibration offset voltage", Value::Scalar(Scalar::F32(p.ame.adc_calibration_offset_voltage)));
    f.push("adc_calibration_gain_correction", Halfword(97), "1", "ADC calibration gain correction", Value::Scalar(Scalar::F32(p.ame.adc_calibration_gain_correction)));
    // Halfwords 99 to 110.
    f.push("rcp_status", Halfword(99), "", "Status of the third-party radar control program (RCP)", Value::Scalar(Scalar::U16(p.rcp_spip.rcp_status)));
    f.push("rcp_string", Halfword(100), "", "Descriptive string for the radar control program state", Value::Text(p.rcp_spip.rcp_string.as_str()));
    f.push("spip_power_buttons", Halfword(108), "", "State of the SPIP power buttons (bit field)", Value::Scalar(Scalar::U16(p.rcp_spip.spip_power_buttons)));
    // Halfwords 111 to 136.
    f.push("master_power_administrator_load", Halfword(111), "A", "Master power administrator load", Value::Scalar(Scalar::F32(p.power.master_power_administrator_load)));
    f.push("expansion_power_administrator_load", Halfword(113), "A", "Expansion power administrator load", Value::Scalar(Scalar::F32(p.power.expansion_power_administrator_load)));
    // Halfwords 137 to 228.
    f.push("ps_5vdc", Halfword(137), "", "+5 VDC power supply", Value::Scalar(Scalar::U16(p.transmitter.ps_5vdc)));
    f.push("ps_15vdc", Halfword(138), "", "+15 VDC power supply", Value::Scalar(Scalar::U16(p.transmitter.ps_15vdc)));
    f.push("ps_28vdc", Halfword(139), "", "+28 VDC power supply", Value::Scalar(Scalar::U16(p.transmitter.ps_28vdc)));
    f.push("ps_neg_15vdc", Halfword(140), "", "-15 VDC power supply", Value::Scalar(Scalar::U16(p.transmitter.ps_neg_15vdc)));
    f.push("ps_45vdc", Halfword(141), "", "+45 VDC power supply", Value::Scalar(Scalar::U16(p.transmitter.ps_45vdc)));
    f.push("filament_ps_voltage", Halfword(142), "", "Filament power supply voltage", Value::Scalar(Scalar::U16(p.transmitter.filament_ps_voltage)));
    f.push("vacuum_pump_ps_voltage", Halfword(143), "", "Vacuum pump power supply voltage", Value::Scalar(Scalar::U16(p.transmitter.vacuum_pump_ps_voltage)));
    f.push("focus_coil_ps_voltage", Halfword(144), "", "Focus coil power supply voltage", Value::Scalar(Scalar::U16(p.transmitter.focus_coil_ps_voltage)));
    f.push("filament_ps", Halfword(145), "", "Filament power supply", Value::Scalar(Scalar::U16(p.transmitter.filament_ps)));
    f.push("klystron_warmup", Halfword(146), "", "Klystron warmup", Value::Scalar(Scalar::U16(p.transmitter.klystron_warmup)));
    f.push("transmitter_available", Halfword(147), "", "Transmitter available", Value::Scalar(Scalar::U16(p.transmitter.transmitter_available)));
    f.push("wg_switch_position", Halfword(148), "", "Waveguide switch position", Value::Scalar(Scalar::U16(p.transmitter.wg_switch_position)));
    f.push("wg_pfn_transfer_interlock", Halfword(149), "", "Waveguide/PFN transfer interlock", Value::Scalar(Scalar::U16(p.transmitter.wg_pfn_transfer_interlock)));
    f.push("maintenance_mode", Halfword(150), "", "Maintenance mode", Value::Scalar(Scalar::U16(p.transmitter.maintenance_mode)));
    f.push("maintenance_required", Halfword(151), "", "Maintenance required", Value::Scalar(Scalar::U16(p.transmitter.maintenance_required)));
    f.push("pfn_switch_position", Halfword(152), "", "PFN switch position", Value::Scalar(Scalar::U16(p.transmitter.pfn_switch_position)));
    f.push("modulator_overload", Halfword(153), "", "Modulator overload", Value::Scalar(Scalar::U16(p.transmitter.modulator_overload)));
    f.push("modulator_inv_current", Halfword(154), "", "Modulator inverse current", Value::Scalar(Scalar::U16(p.transmitter.modulator_inv_current)));
    f.push("modulator_switch_fail", Halfword(155), "", "Modulator switch fail", Value::Scalar(Scalar::U16(p.transmitter.modulator_switch_fail)));
    f.push("main_power_voltage", Halfword(156), "", "Main power voltage", Value::Scalar(Scalar::U16(p.transmitter.main_power_voltage)));
    f.push("charging_system_fail", Halfword(157), "", "Charging system fail", Value::Scalar(Scalar::U16(p.transmitter.charging_system_fail)));
    f.push("inverse_diode_current", Halfword(158), "", "Inverse diode current", Value::Scalar(Scalar::U16(p.transmitter.inverse_diode_current)));
    f.push("trigger_amplifier", Halfword(159), "", "Trigger amplifier", Value::Scalar(Scalar::U16(p.transmitter.trigger_amplifier)));
    f.push("circulator_temperature", Halfword(160), "", "Circulator temperature", Value::Scalar(Scalar::U16(p.transmitter.circulator_temperature)));
    f.push("spectrum_filter_pressure", Halfword(161), "", "Spectrum filter pressure", Value::Scalar(Scalar::U16(p.transmitter.spectrum_filter_pressure)));
    f.push("wg_arc_vswr", Halfword(162), "", "Waveguide arc/VSWR", Value::Scalar(Scalar::U16(p.transmitter.wg_arc_vswr)));
    f.push("cabinet_interlock", Halfword(163), "", "Cabinet interlock", Value::Scalar(Scalar::U16(p.transmitter.cabinet_interlock)));
    f.push("cabinet_air_temperature", Halfword(164), "", "Cabinet air temperature", Value::Scalar(Scalar::U16(p.transmitter.cabinet_air_temperature)));
    f.push("cabinet_airflow", Halfword(165), "", "Cabinet airflow", Value::Scalar(Scalar::U16(p.transmitter.cabinet_airflow)));
    f.push("klystron_current", Halfword(166), "", "Klystron current", Value::Scalar(Scalar::U16(p.transmitter.klystron_current)));
    f.push("klystron_filament_current", Halfword(167), "", "Klystron filament current", Value::Scalar(Scalar::U16(p.transmitter.klystron_filament_current)));
    f.push("klystron_vacion_current", Halfword(168), "", "Klystron VacIon current", Value::Scalar(Scalar::U16(p.transmitter.klystron_vacion_current)));
    f.push("klystron_air_temperature", Halfword(169), "", "Klystron air temperature", Value::Scalar(Scalar::U16(p.transmitter.klystron_air_temperature)));
    f.push("klystron_airflow", Halfword(170), "", "Klystron airflow", Value::Scalar(Scalar::U16(p.transmitter.klystron_airflow)));
    f.push("modulator_switch_maintenance", Halfword(171), "", "Modulator switch maintenance", Value::Scalar(Scalar::U16(p.transmitter.modulator_switch_maintenance)));
    f.push("post_charge_regulator_maintenance", Halfword(172), "", "Post charge regulator maintenance", Value::Scalar(Scalar::U16(p.transmitter.post_charge_regulator_maintenance)));
    f.push("wg_pressure_humidity", Halfword(173), "", "Waveguide pressure/humidity", Value::Scalar(Scalar::U16(p.transmitter.wg_pressure_humidity)));
    f.push("transmitter_overvoltage", Halfword(174), "", "Transmitter overvoltage", Value::Scalar(Scalar::U16(p.transmitter.transmitter_overvoltage)));
    f.push("transmitter_overcurrent", Halfword(175), "", "Transmitter overcurrent", Value::Scalar(Scalar::U16(p.transmitter.transmitter_overcurrent)));
    f.push("focus_coil_current", Halfword(176), "", "Focus coil current", Value::Scalar(Scalar::U16(p.transmitter.focus_coil_current)));
    f.push("focus_coil_airflow", Halfword(177), "", "Focus coil airflow", Value::Scalar(Scalar::U16(p.transmitter.focus_coil_airflow)));
    f.push("oil_temperature", Halfword(178), "", "Oil temperature", Value::Scalar(Scalar::U16(p.transmitter.oil_temperature)));
    f.push("prf_limit", Halfword(179), "", "PRF limit", Value::Scalar(Scalar::U16(p.transmitter.prf_limit)));
    f.push("transmitter_oil_level", Halfword(180), "", "Transmitter oil level", Value::Scalar(Scalar::U16(p.transmitter.transmitter_oil_level)));
    f.push("transmitter_battery_charging", Halfword(181), "", "Transmitter battery charging", Value::Scalar(Scalar::U16(p.transmitter.transmitter_battery_charging)));
    f.push("high_voltage_status", Halfword(182), "", "High voltage (HV) status", Value::Scalar(Scalar::U16(p.transmitter.high_voltage_status)));
    f.push("transmitter_recycling_summary", Halfword(183), "", "Transmitter recycling summary", Value::Scalar(Scalar::U16(p.transmitter.transmitter_recycling_summary)));
    f.push("transmitter_inoperable", Halfword(184), "", "Transmitter inoperable", Value::Scalar(Scalar::U16(p.transmitter.transmitter_inoperable)));
    f.push("transmitter_air_filter", Halfword(185), "", "Transmitter air filter", Value::Scalar(Scalar::U16(p.transmitter.transmitter_air_filter)));
    f.push("zero_test_bits", Halfword(186), "", "Zero test bits 0 to 7", Value::Array(ArrayBuf::U16(p.transmitter.zero_test_bits.to_vec())));
    f.push("one_test_bits", Halfword(194), "", "One test bits 0 to 7", Value::Array(ArrayBuf::U16(p.transmitter.one_test_bits.to_vec())));
    f.push("xmtr_spip_interface", Halfword(202), "", "Transmitter/SPIP interface", Value::Scalar(Scalar::U16(p.transmitter.xmtr_spip_interface)));
    f.push("transmitter_summary_status", Halfword(203), "", "Transmitter summary status", Value::Scalar(Scalar::U16(p.transmitter.transmitter_summary_status)));
    f.push("transmitter_rf_power", Halfword(205), "mW", "Transmitter RF power (sensor)", Value::Scalar(Scalar::F32(p.transmitter.transmitter_rf_power)));
    f.push("horizontal_xmtr_peak_power", Halfword(207), "kW", "Horizontal transmitter peak power", Value::Scalar(Scalar::F32(p.transmitter.horizontal_xmtr_peak_power)));
    f.push("xmtr_peak_power", Halfword(209), "kW", "Transmitter peak power", Value::Scalar(Scalar::F32(p.transmitter.xmtr_peak_power)));
    f.push("vertical_xmtr_peak_power", Halfword(211), "kW", "Vertical transmitter peak power", Value::Scalar(Scalar::F32(p.transmitter.vertical_xmtr_peak_power)));
    f.push("xmtr_rf_avg_power", Halfword(213), "W", "Transmitter RF average power", Value::Scalar(Scalar::F32(p.transmitter.xmtr_rf_avg_power)));
    f.push("xmtr_recycle_count", Halfword(217), "", "Transmitter recycle count (0 to 999,999)", Value::Scalar(Scalar::U32(p.transmitter.xmtr_recycle_count)));
    f.push("receiver_bias", Halfword(219), "dB", "Receiver bias (measurement)", Value::Scalar(Scalar::F32(p.transmitter.receiver_bias)));
    f.push("transmit_imbalance", Halfword(221), "dB", "Transmit imbalance", Value::Scalar(Scalar::F32(p.transmitter.transmit_imbalance)));
    f.push("xmtr_power_meter_zero", Halfword(223), "V", "Transmitter power meter zero", Value::Scalar(Scalar::F32(p.transmitter.xmtr_power_meter_zero)));
    // Halfwords 229 to 249.
    f.push("ac_unit_1_compressor_shut_off", Halfword(229), "", "AC unit 1 compressor shut off", Value::Scalar(Scalar::U16(p.tower_utilities.ac_unit_1_compressor_shut_off)));
    f.push("ac_unit_2_compressor_shut_off", Halfword(230), "", "AC unit 2 compressor shut off", Value::Scalar(Scalar::U16(p.tower_utilities.ac_unit_2_compressor_shut_off)));
    f.push("generator_maintenance_required", Halfword(231), "", "Generator maintenance required", Value::Scalar(Scalar::U16(p.tower_utilities.generator_maintenance_required)));
    f.push("generator_battery_voltage", Halfword(232), "", "Generator battery voltage", Value::Scalar(Scalar::U16(p.tower_utilities.generator_battery_voltage)));
    f.push("generator_engine", Halfword(233), "", "Generator engine", Value::Scalar(Scalar::U16(p.tower_utilities.generator_engine)));
    f.push("generator_volt_frequency", Halfword(234), "", "Generator volt/frequency", Value::Scalar(Scalar::U16(p.tower_utilities.generator_volt_frequency)));
    f.push("power_source", Halfword(235), "", "Power source", Value::Scalar(Scalar::U16(p.tower_utilities.power_source)));
    f.push("transitional_power_source", Halfword(236), "", "Transitional power source (TPS)", Value::Scalar(Scalar::U16(p.tower_utilities.transitional_power_source)));
    f.push("generator_auto_run_off_switch", Halfword(237), "", "Generator auto/run/off switch", Value::Scalar(Scalar::U16(p.tower_utilities.generator_auto_run_off_switch)));
    f.push("aircraft_hazard_lighting", Halfword(238), "", "Aircraft hazard lighting", Value::Scalar(Scalar::U16(p.tower_utilities.aircraft_hazard_lighting)));
    // Halfwords 250 to 299.
    f.push("fire_detection_system", Halfword(250), "", "Equipment shelter fire detection system", Value::Scalar(Scalar::U16(p.equipment_shelter.fire_detection_system)));
    f.push("equipment_shelter_fire_smoke", Halfword(251), "", "Equipment shelter fire/smoke", Value::Scalar(Scalar::U16(p.equipment_shelter.equipment_shelter_fire_smoke)));
    f.push("generator_shelter_fire_smoke", Halfword(252), "", "Generator shelter fire/smoke", Value::Scalar(Scalar::U16(p.equipment_shelter.generator_shelter_fire_smoke)));
    f.push("utility_voltage_frequency", Halfword(253), "", "Utility voltage/frequency", Value::Scalar(Scalar::U16(p.equipment_shelter.utility_voltage_frequency)));
    f.push("site_security_alarm", Halfword(254), "", "Site security alarm", Value::Scalar(Scalar::U16(p.equipment_shelter.site_security_alarm)));
    f.push("security_equipment", Halfword(255), "", "Security equipment", Value::Scalar(Scalar::U16(p.equipment_shelter.security_equipment)));
    f.push("security_system", Halfword(256), "", "Security system", Value::Scalar(Scalar::U16(p.equipment_shelter.security_system)));
    f.push("receiver_connected_to_antenna", Halfword(257), "", "Receiver connected to antenna", Value::Scalar(Scalar::U16(p.equipment_shelter.receiver_connected_to_antenna)));
    f.push("radome_hatch", Halfword(258), "", "Radome hatch", Value::Scalar(Scalar::U16(p.equipment_shelter.radome_hatch)));
    f.push("ac_unit_1_filter_dirty", Halfword(259), "", "AC unit 1 filter", Value::Scalar(Scalar::U16(p.equipment_shelter.ac_unit_1_filter_dirty)));
    f.push("ac_unit_2_filter_dirty", Halfword(260), "", "AC unit 2 filter", Value::Scalar(Scalar::U16(p.equipment_shelter.ac_unit_2_filter_dirty)));
    f.push("equipment_shelter_temperature", Halfword(261), "degC", "Equipment shelter temperature", Value::Scalar(Scalar::F32(p.equipment_shelter.equipment_shelter_temperature)));
    f.push("outside_ambient_temperature", Halfword(263), "degC", "Outside ambient temperature", Value::Scalar(Scalar::F32(p.equipment_shelter.outside_ambient_temperature)));
    f.push("transmitter_leaving_air_temperature", Halfword(265), "degC", "Transmitter leaving air temperature", Value::Scalar(Scalar::F32(p.equipment_shelter.transmitter_leaving_air_temperature)));
    f.push("ac_unit_1_discharge_air_temperature", Halfword(267), "degC", "AC unit 1 discharge air temperature", Value::Scalar(Scalar::F32(p.equipment_shelter.ac_unit_1_discharge_air_temperature)));
    f.push("generator_shelter_temperature", Halfword(269), "degC", "Generator shelter temperature", Value::Scalar(Scalar::F32(p.equipment_shelter.generator_shelter_temperature)));
    f.push("radome_air_temperature", Halfword(271), "degC", "Radome air temperature", Value::Scalar(Scalar::F32(p.equipment_shelter.radome_air_temperature)));
    f.push("ac_unit_2_discharge_air_temperature", Halfword(273), "degC", "AC unit 2 discharge air temperature", Value::Scalar(Scalar::F32(p.equipment_shelter.ac_unit_2_discharge_air_temperature)));
    f.push("spip_15v_ps", Halfword(275), "V", "SPIP +15 V power supply", Value::Scalar(Scalar::F32(p.equipment_shelter.spip_15v_ps)));
    f.push("spip_neg_15v_ps", Halfword(277), "V", "SPIP -15 V power supply", Value::Scalar(Scalar::F32(p.equipment_shelter.spip_neg_15v_ps)));
    f.push("spip_28v_ps_status", Halfword(279), "", "SPIP +28 V power supply status", Value::Scalar(Scalar::U16(p.equipment_shelter.spip_28v_ps_status)));
    f.push("spip_5v_ps", Halfword(281), "V", "SPIP +5 V power supply", Value::Scalar(Scalar::F32(p.equipment_shelter.spip_5v_ps)));
    f.push("converted_generator_fuel_level", Halfword(283), "percent", "Converted generator fuel level", Value::Scalar(Scalar::U16(p.equipment_shelter.converted_generator_fuel_level)));
    // Halfwords 300 to 340.
    f.push("elevation_pos_dead_limit", Halfword(300), "", "Elevation + dead limit (antenna in the upper dead limit)", Value::Scalar(Scalar::U16(p.antenna_pedestal.elevation_pos_dead_limit)));
    f.push("pos_150v_overvoltage", Halfword(301), "", "+150 V overvoltage", Value::Scalar(Scalar::U16(p.antenna_pedestal.pos_150v_overvoltage)));
    f.push("pos_150v_undervoltage", Halfword(302), "", "+150 V undervoltage", Value::Scalar(Scalar::U16(p.antenna_pedestal.pos_150v_undervoltage)));
    f.push("elevation_servo_amp_inhibit", Halfword(303), "", "Elevation servo amplifier inhibit", Value::Scalar(Scalar::U16(p.antenna_pedestal.elevation_servo_amp_inhibit)));
    f.push("elevation_servo_amp_short_circuit", Halfword(304), "", "Elevation servo amplifier short circuit", Value::Scalar(Scalar::U16(p.antenna_pedestal.elevation_servo_amp_short_circuit)));
    f.push("elevation_servo_amp_overtemp", Halfword(305), "", "Elevation servo amplifier overtemperature", Value::Scalar(Scalar::U16(p.antenna_pedestal.elevation_servo_amp_overtemp)));
    f.push("elevation_motor_overtemp", Halfword(306), "", "Elevation motor overtemperature", Value::Scalar(Scalar::U16(p.antenna_pedestal.elevation_motor_overtemp)));
    f.push("elevation_stow_pin", Halfword(307), "", "Elevation stow pin", Value::Scalar(Scalar::U16(p.antenna_pedestal.elevation_stow_pin)));
    f.push("elevation_housing_5v_ps", Halfword(308), "", "Elevation housing DC-to-DC converter (+5 V) power supply", Value::Scalar(Scalar::U16(p.antenna_pedestal.elevation_housing_5v_ps)));
    f.push("elevation_neg_dead_limit", Halfword(309), "", "Elevation - dead limit (antenna in the lower dead limit)", Value::Scalar(Scalar::U16(p.antenna_pedestal.elevation_neg_dead_limit)));
    f.push("elevation_pos_normal_limit", Halfword(310), "", "Elevation + normal limit (antenna in the upper normal limit)", Value::Scalar(Scalar::U16(p.antenna_pedestal.elevation_pos_normal_limit)));
    f.push("elevation_neg_normal_limit", Halfword(311), "", "Elevation - normal limit", Value::Scalar(Scalar::U16(p.antenna_pedestal.elevation_neg_normal_limit)));
    f.push("elevation_encoder_light", Halfword(312), "", "Elevation encoder light", Value::Scalar(Scalar::U16(p.antenna_pedestal.elevation_encoder_light)));
    f.push("elevation_gearbox_oil", Halfword(313), "", "Elevation gearbox oil", Value::Scalar(Scalar::U16(p.antenna_pedestal.elevation_gearbox_oil)));
    f.push("elevation_handwheel", Halfword(314), "", "Elevation handwheel", Value::Scalar(Scalar::U16(p.antenna_pedestal.elevation_handwheel)));
    f.push("elevation_amp_ps", Halfword(315), "", "Elevation amplifier power supply", Value::Scalar(Scalar::U16(p.antenna_pedestal.elevation_amp_ps)));
    f.push("azimuth_servo_amp_inhibit", Halfword(316), "", "Azimuth servo amplifier inhibit", Value::Scalar(Scalar::U16(p.antenna_pedestal.azimuth_servo_amp_inhibit)));
    f.push("azimuth_servo_amp_short_circuit", Halfword(317), "", "Azimuth servo amplifier short circuit", Value::Scalar(Scalar::U16(p.antenna_pedestal.azimuth_servo_amp_short_circuit)));
    f.push("azimuth_servo_amp_overtemp", Halfword(318), "", "Azimuth servo amplifier overtemperature", Value::Scalar(Scalar::U16(p.antenna_pedestal.azimuth_servo_amp_overtemp)));
    f.push("azimuth_motor_overtemp", Halfword(319), "", "Azimuth motor overtemperature", Value::Scalar(Scalar::U16(p.antenna_pedestal.azimuth_motor_overtemp)));
    f.push("azimuth_stow_pin", Halfword(320), "", "Azimuth stow pin", Value::Scalar(Scalar::U16(p.antenna_pedestal.azimuth_stow_pin)));
    f.push("azimuth_housing_5v_ps", Halfword(321), "", "Azimuth housing DC-to-DC converter (+5 V) power supply", Value::Scalar(Scalar::U16(p.antenna_pedestal.azimuth_housing_5v_ps)));
    f.push("azimuth_encoder_light", Halfword(322), "", "Azimuth encoder light", Value::Scalar(Scalar::U16(p.antenna_pedestal.azimuth_encoder_light)));
    f.push("azimuth_gearbox_oil", Halfword(323), "", "Azimuth gearbox oil", Value::Scalar(Scalar::U16(p.antenna_pedestal.azimuth_gearbox_oil)));
    f.push("azimuth_bull_gear_oil", Halfword(324), "", "Azimuth bull gear oil", Value::Scalar(Scalar::U16(p.antenna_pedestal.azimuth_bull_gear_oil)));
    f.push("azimuth_handwheel", Halfword(325), "", "Azimuth handwheel", Value::Scalar(Scalar::U16(p.antenna_pedestal.azimuth_handwheel)));
    f.push("azimuth_servo_amp_ps", Halfword(326), "", "Azimuth servo amplifier power supply", Value::Scalar(Scalar::U16(p.antenna_pedestal.azimuth_servo_amp_ps)));
    f.push("servo", Halfword(327), "", "Servo", Value::Scalar(Scalar::U16(p.antenna_pedestal.servo)));
    f.push("pedestal_interlock_switch", Halfword(328), "", "Pedestal interlock switch", Value::Scalar(Scalar::U16(p.antenna_pedestal.pedestal_interlock_switch)));
    // Halfwords 341 to 362.
    f.push("coho_clock", Halfword(341), "", "COHO/clock", Value::Scalar(Scalar::U16(p.rf_generator_receiver.coho_clock)));
    f.push("frequency_select_oscillator", Halfword(342), "", "RF generator frequency select oscillator", Value::Scalar(Scalar::U16(p.rf_generator_receiver.frequency_select_oscillator)));
    f.push("rf_stalo", Halfword(343), "", "RF generator RF/STALO", Value::Scalar(Scalar::U16(p.rf_generator_receiver.rf_stalo)));
    f.push("phase_shifted_coho", Halfword(344), "", "RF generator phase shifted COHO", Value::Scalar(Scalar::U16(p.rf_generator_receiver.phase_shifted_coho)));
    f.push("receiver_ps_9v", Halfword(345), "", "+9 V receiver power supply", Value::Scalar(Scalar::U16(p.rf_generator_receiver.receiver_ps_9v)));
    f.push("receiver_ps_5v", Halfword(346), "", "+5 V receiver power supply", Value::Scalar(Scalar::U16(p.rf_generator_receiver.receiver_ps_5v)));
    f.push("receiver_ps_18v", Halfword(347), "", "+/-18 V receiver power supply", Value::Scalar(Scalar::U16(p.rf_generator_receiver.receiver_ps_18v)));
    f.push("receiver_ps_neg_9v", Halfword(348), "", "-9 V receiver power supply", Value::Scalar(Scalar::U16(p.rf_generator_receiver.receiver_ps_neg_9v)));
    f.push("rdaiu_ps_5v", Halfword(349), "", "+5 V single channel RDAIU power supply", Value::Scalar(Scalar::U16(p.rf_generator_receiver.rdaiu_ps_5v)));
    f.push("horizontal_short_pulse_noise", Halfword(351), "dBm", "Horizontal short pulse noise", Value::Scalar(Scalar::F32(p.rf_generator_receiver.horizontal_short_pulse_noise)));
    f.push("horizontal_long_pulse_noise", Halfword(353), "dBm", "Horizontal long pulse noise", Value::Scalar(Scalar::F32(p.rf_generator_receiver.horizontal_long_pulse_noise)));
    f.push("horizontal_noise_temperature", Halfword(355), "K", "Horizontal noise temperature", Value::Scalar(Scalar::F32(p.rf_generator_receiver.horizontal_noise_temperature)));
    f.push("vertical_short_pulse_noise", Halfword(357), "dBm", "Vertical short pulse noise", Value::Scalar(Scalar::F32(p.rf_generator_receiver.vertical_short_pulse_noise)));
    f.push("vertical_long_pulse_noise", Halfword(359), "dBm", "Vertical long pulse noise", Value::Scalar(Scalar::F32(p.rf_generator_receiver.vertical_long_pulse_noise)));
    f.push("vertical_noise_temperature", Halfword(361), "K", "Vertical noise temperature", Value::Scalar(Scalar::F32(p.rf_generator_receiver.vertical_noise_temperature)));
    // Halfwords 363 to 430.
    f.push("horizontal_linearity", Halfword(363), "1", "Horizontal linearity", Value::Scalar(Scalar::F32(p.calibration.horizontal_linearity)));
    f.push("horizontal_dynamic_range", Halfword(365), "dB", "Horizontal dynamic range", Value::Scalar(Scalar::F32(p.calibration.horizontal_dynamic_range)));
    f.push("horizontal_delta_dbz0", Halfword(367), "dB", "Horizontal delta dBZ0", Value::Scalar(Scalar::F32(p.calibration.horizontal_delta_dbz0)));
    f.push("vertical_delta_dbz0", Halfword(369), "dB", "Vertical delta dBZ0", Value::Scalar(Scalar::F32(p.calibration.vertical_delta_dbz0)));
    f.push("kd_peak_measured", Halfword(371), "dBm", "KD peak measured", Value::Scalar(Scalar::F32(p.calibration.kd_peak_measured)));
    f.push("short_pulse_horizontal_dbz0", Halfword(375), "dBZ", "Short pulse horizontal dBZ0", Value::Scalar(Scalar::F32(p.calibration.short_pulse_horizontal_dbz0)));
    f.push("long_pulse_horizontal_dbz0", Halfword(377), "dBZ", "Long pulse horizontal dBZ0", Value::Scalar(Scalar::F32(p.calibration.long_pulse_horizontal_dbz0)));
    f.push("velocity_processed", Halfword(379), "", "Velocity check (processed)", Value::Scalar(Scalar::U16(p.calibration.velocity_processed)));
    f.push("width_processed", Halfword(380), "", "Spectrum width check (processed)", Value::Scalar(Scalar::U16(p.calibration.width_processed)));
    f.push("velocity_rf_gen", Halfword(381), "", "Velocity check (RF generator)", Value::Scalar(Scalar::U16(p.calibration.velocity_rf_gen)));
    f.push("width_rf_gen", Halfword(382), "", "Spectrum width check (RF generator)", Value::Scalar(Scalar::U16(p.calibration.width_rf_gen)));
    f.push("horizontal_i0", Halfword(383), "dBm", "Horizontal I0", Value::Scalar(Scalar::F32(p.calibration.horizontal_i0)));
    f.push("vertical_i0", Halfword(385), "dBm", "Vertical I0", Value::Scalar(Scalar::F32(p.calibration.vertical_i0)));
    f.push("vertical_dynamic_range", Halfword(387), "dB", "Vertical dynamic range", Value::Scalar(Scalar::F32(p.calibration.vertical_dynamic_range)));
    f.push("short_pulse_vertical_dbz0", Halfword(389), "dBZ", "Short pulse vertical dBZ0", Value::Scalar(Scalar::F32(p.calibration.short_pulse_vertical_dbz0)));
    f.push("long_pulse_vertical_dbz0", Halfword(391), "dBZ", "Long pulse vertical dBZ0", Value::Scalar(Scalar::F32(p.calibration.long_pulse_vertical_dbz0)));
    f.push("horizontal_power_sense", Halfword(397), "dBm", "Horizontal power sense", Value::Scalar(Scalar::F32(p.calibration.horizontal_power_sense)));
    f.push("vertical_power_sense", Halfword(399), "dBm", "Vertical power sense", Value::Scalar(Scalar::F32(p.calibration.vertical_power_sense)));
    f.push("zdr_offset", Halfword(401), "dB", "ZDR offset (called ZDR bias before Build 22.0)", Value::Scalar(Scalar::F32(p.calibration.zdr_offset)));
    f.push("clutter_suppression_delta", Halfword(409), "dB", "Clutter suppression delta", Value::Scalar(Scalar::F32(p.calibration.clutter_suppression_delta)));
    f.push("clutter_suppression_unfiltered_power", Halfword(411), "dBZ", "Clutter suppression unfiltered power", Value::Scalar(Scalar::F32(p.calibration.clutter_suppression_unfiltered_power)));
    f.push("clutter_suppression_filtered_power", Halfword(413), "dBZ", "Clutter suppression filtered power", Value::Scalar(Scalar::F32(p.calibration.clutter_suppression_filtered_power)));
    f.push("vertical_linearity", Halfword(425), "1", "Vertical linearity", Value::Scalar(Scalar::F32(p.calibration.vertical_linearity)));
    // Halfwords 431 to 460.
    f.push("state_file_read", Halfword(431), "", "State file read status", Value::Scalar(Scalar::U16(p.file_status.state_file_read)));
    f.push("state_file_write", Halfword(432), "", "State file write status", Value::Scalar(Scalar::U16(p.file_status.state_file_write)));
    f.push("bypass_map_file_read", Halfword(433), "", "Bypass map file read status", Value::Scalar(Scalar::U16(p.file_status.bypass_map_file_read)));
    f.push("bypass_map_file_write", Halfword(434), "", "Bypass map file write status", Value::Scalar(Scalar::U16(p.file_status.bypass_map_file_write)));
    f.push("current_adaptation_file_read", Halfword(437), "", "Current adaptation file read status", Value::Scalar(Scalar::U16(p.file_status.current_adaptation_file_read)));
    f.push("current_adaptation_file_write", Halfword(438), "", "Current adaptation file write status", Value::Scalar(Scalar::U16(p.file_status.current_adaptation_file_write)));
    f.push("censor_zone_file_read", Halfword(439), "", "Censor zone file read status", Value::Scalar(Scalar::U16(p.file_status.censor_zone_file_read)));
    f.push("censor_zone_file_write", Halfword(440), "", "Censor zone file write status", Value::Scalar(Scalar::U16(p.file_status.censor_zone_file_write)));
    f.push("remote_vcp_file_read", Halfword(441), "", "Remote VCP file read status", Value::Scalar(Scalar::U16(p.file_status.remote_vcp_file_read)));
    f.push("remote_vcp_file_write", Halfword(442), "", "Remote VCP file write status", Value::Scalar(Scalar::U16(p.file_status.remote_vcp_file_write)));
    f.push("baseline_adaptation_file_read", Halfword(443), "", "Baseline adaptation file read status", Value::Scalar(Scalar::U16(p.file_status.baseline_adaptation_file_read)));
    f.push("prf_sets_read", Halfword(444), "", "Read status of the PRF sets (bit field; per bit 0 = fail, 1 = OK)", Value::Scalar(Scalar::U16(p.file_status.prf_sets_read)));
    f.push("clutter_filter_map_file_read", Halfword(445), "", "Clutter filter map file read status", Value::Scalar(Scalar::U16(p.file_status.clutter_filter_map_file_read)));
    f.push("clutter_filter_map_file_write", Halfword(446), "", "Clutter filter map file write status", Value::Scalar(Scalar::U16(p.file_status.clutter_filter_map_file_write)));
    f.push("general_disk_io_error", Halfword(447), "", "General disk I/O error", Value::Scalar(Scalar::U16(p.file_status.general_disk_io_error)));
    f.push("rsp_status", HalfwordByte(448, 0), "", "RSP health status (bit field; per bit 1 = fail, 0 = OK)", Value::Scalar(Scalar::U8(p.file_status.rsp_status)));
    f.push("rsp_cpu1_temperature", HalfwordByte(449, 0), "degC", "RSP CPU 1 temperature", Value::Scalar(Scalar::U8(p.file_status.rsp_cpu1_temperature)));
    f.push("rsp_cpu2_temperature", HalfwordByte(449, 1), "degC", "RSP CPU 2 temperature", Value::Scalar(Scalar::U8(p.file_status.rsp_cpu2_temperature)));
    f.push("rsp_motherboard_power", Halfword(450), "W", "RSP power used, as measured by the motherboard sensor", Value::Scalar(Scalar::U16(p.file_status.rsp_motherboard_power)));
    // Halfwords 461 to 479.
    f.push("spip_comm_status", Halfword(461), "", "SPIP communication status", Value::Scalar(Scalar::U16(p.device_status.spip_comm_status)));
    f.push("hci_comm_status", Halfword(462), "", "HCI communication status", Value::Scalar(Scalar::U16(p.device_status.hci_comm_status)));
    f.push("signal_processor_command_status", Halfword(464), "", "Signal processor command status", Value::Scalar(Scalar::U16(p.device_status.signal_processor_command_status)));
    f.push("ame_communication_status", Halfword(465), "", "AME communication status", Value::Scalar(Scalar::U16(p.device_status.ame_communication_status)));
    f.push("rms_link_status", Halfword(466), "", "RMS link status", Value::Scalar(Scalar::U16(p.device_status.rms_link_status)));
    f.push("rpg_link_status", Halfword(467), "", "RPG link status", Value::Scalar(Scalar::U16(p.device_status.rpg_link_status)));
    f.push("interpanel_link_status", Halfword(468), "", "Interpanel link (channel 1 SPIP to channel 2 SPIP power and communications)", Value::Scalar(Scalar::U16(p.device_status.interpanel_link_status)));
    f.push("performance_check_time", Halfword(469), "seconds since 1970-01-01T00:00:00Z", "Time the next performance check is due", Value::Scalar(Scalar::U32(p.device_status.performance_check_time)));
    f.push("version", Halfword(480), "", "Version number of the performance data message", Value::Scalar(Scalar::U16(p.version)));
}
#[rustfmt::skip]
fn adaptation_table<'a>(a: &'a RdaAdaptationData, f: &mut Fields<'a>) {
    use Location::Byte;
    f.push("adap_file_name", Byte(0), "", "Name of the adaptation data file (\"baseline\" or \"current\")", Value::Text(a.adap_file_name.as_str()));
    f.push("adap_format", Byte(12), "", "Format of the adaptation data file (for example \"14\")", Value::Text(a.adap_format.as_str()));
    f.push("adap_revision", Byte(16), "", "Revision number of the adaptation data file", Value::Text(a.adap_revision.as_str()));
    f.push("adap_date", Byte(20), "", "Last modified date of the adaptation data file", Value::Text(a.adap_date.as_str()));
    f.push("adap_time", Byte(32), "", "Last modified time of the adaptation data file", Value::Text(a.adap_time.as_str()));
    f.push("lower_pre_limit", Byte(44), "degree", "Angle of the lower pre-limit switch", Value::Scalar(Scalar::F32(a.lower_pre_limit)));
    f.push("az_lat", Byte(48), "s", "Latency of the azimuth encoder measurement", Value::Scalar(Scalar::F32(a.az_lat)));
    f.push("upper_pre_limit", Byte(52), "degree", "Angle of the upper pre-limit switch", Value::Scalar(Scalar::F32(a.upper_pre_limit)));
    f.push("el_lat", Byte(56), "s", "Latency of the elevation encoder measurement", Value::Scalar(Scalar::F32(a.el_lat)));
    f.push("parkaz", Byte(60), "degree", "Pedestal park position in azimuth", Value::Scalar(Scalar::F32(a.parkaz)));
    f.push("parkel", Byte(64), "degree", "Pedestal park position in elevation", Value::Scalar(Scalar::F32(a.parkel)));
    f.push("a_fuel_conv", Byte(68), "", "Generator fuel level height to capacity conversion table", Value::Array(ArrayBuf::F32(a.a_fuel_conv.to_vec())));
    f.push("a_min_shelter_temp", Byte(112), "degC", "Minimum equipment shelter alarm temperature", Value::Scalar(Scalar::F32(a.a_min_shelter_temp)));
    f.push("a_max_shelter_temp", Byte(116), "degC", "Maximum equipment shelter alarm temperature", Value::Scalar(Scalar::F32(a.a_max_shelter_temp)));
    f.push("a_min_shelter_ac_temp_diff", Byte(120), "degC", "Minimum A/C discharge air temperature differential", Value::Scalar(Scalar::F32(a.a_min_shelter_ac_temp_diff)));
    f.push("a_max_xmtr_air_temp", Byte(124), "degC", "Maximum transmitter leaving air alarm temperature", Value::Scalar(Scalar::F32(a.a_max_xmtr_air_temp)));
    f.push("a_max_rad_temp", Byte(128), "degC", "Maximum radome alarm temperature", Value::Scalar(Scalar::F32(a.a_max_rad_temp)));
    f.push("a_max_rad_temp_rise", Byte(132), "degC", "Maximum radome minus ambient temperature difference", Value::Scalar(Scalar::F32(a.a_max_rad_temp_rise)));
    f.push("lower_dead_limit", Byte(136), "degree", "Angle of the lower dead limit switch", Value::Scalar(Scalar::F32(a.lower_dead_limit)));
    f.push("upper_dead_limit", Byte(140), "degree", "Angle of the upper dead limit switch", Value::Scalar(Scalar::F32(a.upper_dead_limit)));
    f.push("a_min_gen_room_temp", Byte(148), "degC", "Minimum generator shelter alarm temperature", Value::Scalar(Scalar::F32(a.a_min_gen_room_temp)));
    f.push("a_max_gen_room_temp", Byte(152), "degC", "Maximum generator shelter alarm temperature", Value::Scalar(Scalar::F32(a.a_max_gen_room_temp)));
    f.push("spip_5v_reg_lim", Byte(156), "percent", "SPIP +5 V power supply tolerance", Value::Scalar(Scalar::F32(a.spip_5v_reg_lim)));
    f.push("spip_15v_reg_lim", Byte(160), "percent", "SPIP +/-15 V power supply tolerance", Value::Scalar(Scalar::F32(a.spip_15v_reg_lim)));
    f.push("rpg_co_located", Byte(176), "", "RPG co-located", Value::Flag(a.rpg_co_located));
    f.push("spec_filter_installed", Byte(180), "", "Transmitter spectrum filter installed", Value::Flag(a.spec_filter_installed));
    f.push("tps_installed", Byte(184), "", "Transition power source installed", Value::Flag(a.tps_installed));
    f.push("rms_installed", Byte(188), "", "FAA RMS installed", Value::Flag(a.rms_installed));
    f.push("a_hvdl_tst_int", Byte(192), "h", "Performance test interval", Value::Scalar(Scalar::I32(a.a_hvdl_tst_int)));
    f.push("a_rpg_lt_int", Byte(196), "min", "RPG loop test interval", Value::Scalar(Scalar::I32(a.a_rpg_lt_int)));
    f.push("a_min_stab_util_pwr_time", Byte(200), "min", "Required interval time for stable utility power", Value::Scalar(Scalar::I32(a.a_min_stab_util_pwr_time)));
    f.push("a_gen_auto_exer_interval", Byte(204), "h", "Maximum generator automatic exercise interval", Value::Scalar(Scalar::I32(a.a_gen_auto_exer_interval)));
    f.push("a_util_pwr_sw_req_interval", Byte(208), "min", "Recommended switch to utility power time interval", Value::Scalar(Scalar::I32(a.a_util_pwr_sw_req_interval)));
    f.push("a_low_fuel_level", Byte(212), "percent", "Low fuel tank warning level", Value::Scalar(Scalar::F32(a.a_low_fuel_level)));
    f.push("config_chan_number", Byte(216), "", "Configuration channel number (1 or 2)", Value::Scalar(Scalar::I32(a.config_chan_number)));
    f.push("redundant_chan_config", Byte(224), "", "Redundant channel configuration", Value::Scalar(Scalar::I32(a.redundant_chan_config)));
    f.push("atten_table", Byte(228), "dB", "Test signal attenuator insertion losses for 0 dB to 103 dB of attenuation", Value::Array(ArrayBuf::F32(a.atten_table.to_vec())));
    f.push("path_losses_7", Byte(668), "dB", "Path loss, vertical IF heliax to 4AT16", Value::Scalar(Scalar::F32(a.path_losses_7)));
    f.push("path_losses_13", Byte(692), "dB", "Path loss, 2A9A9 RF delay line", Value::Scalar(Scalar::F32(a.path_losses_13)));
    f.push("path_losses_28", Byte(752), "dB", "Path loss, horizontal IF heliax to 4AT17", Value::Scalar(Scalar::F32(a.path_losses_28)));
    f.push("h_coupler_xmt_loss", Byte(756), "dB", "RF pallet horizontal coupler transmitter loss", Value::Scalar(Scalar::F32(a.h_coupler_xmt_loss)));
    f.push("path_losses_32", Byte(768), "dB", "Path loss, WG02 harmonic filter", Value::Scalar(Scalar::F32(a.path_losses_32)));
    f.push("path_losses_33", Byte(772), "dB", "Path loss, waveguide klystron to switch", Value::Scalar(Scalar::F32(a.path_losses_33)));
    f.push("path_losses_35", Byte(780), "dB", "Path loss, WG06 spectrum filter", Value::Scalar(Scalar::F32(a.path_losses_35)));
    f.push("path_losses_39", Byte(796), "dB", "Path loss, WG04 circulator", Value::Scalar(Scalar::F32(a.path_losses_39)));
    f.push("path_losses_40", Byte(800), "dB", "Path loss, A6 arc detector", Value::Scalar(Scalar::F32(a.path_losses_40)));
    f.push("path_losses_42", Byte(808), "dB", "Path loss, 1DC1 transmitter coupler coupling", Value::Scalar(Scalar::F32(a.path_losses_42)));
    f.push("path_losses_43", Byte(812), "dB", "Path loss, A33 pad", Value::Scalar(Scalar::F32(a.path_losses_43)));
    f.push("path_losses_44", Byte(816), "dB", "Path loss, coax transmitter RF sample to A33 pad", Value::Scalar(Scalar::F32(a.path_losses_44)));
    f.push("path_losses_45", Byte(820), "dB", "Path loss, A20J1_4 power splitter", Value::Scalar(Scalar::F32(a.path_losses_45)));
    f.push("path_losses_46", Byte(824), "dB", "Path loss, A20J1_3 power splitter", Value::Scalar(Scalar::F32(a.path_losses_46)));
    f.push("path_losses_47", Byte(828), "dB", "Path loss, A20J1_2 power splitter", Value::Scalar(Scalar::F32(a.path_losses_47)));
    f.push("h_coupler_cw_loss", Byte(832), "dB", "RF pallet horizontal coupler test signal loss", Value::Scalar(Scalar::F32(a.h_coupler_cw_loss)));
    f.push("v_coupler_xmt_loss", Byte(836), "dB", "RF pallet vertical coupler transmitter loss", Value::Scalar(Scalar::F32(a.v_coupler_xmt_loss)));
    f.push("ame_ts_bias", Byte(844), "dB", "AME test signal bias", Value::Scalar(Scalar::F32(a.ame_ts_bias)));
    f.push("path_losses_52", Byte(848), "dB", "Path loss, 1AT4 transmitter coupler pad", Value::Scalar(Scalar::F32(a.path_losses_52)));
    f.push("v_coupler_cw_loss", Byte(852), "dB", "RF pallet vertical coupler test signal loss", Value::Scalar(Scalar::F32(a.v_coupler_cw_loss)));
    f.push("pwr_sense_bias", Byte(864), "dB", "Power sense calibration offset bias", Value::Scalar(Scalar::F32(a.pwr_sense_bias)));
    f.push("ame_v_noise_enr", Byte(868), "dB", "AME noise source vertical excess noise ratio", Value::Scalar(Scalar::F32(a.ame_v_noise_enr)));
    f.push("path_losses_58", Byte(872), "dB", "Path loss, 4AT17 attenuator", Value::Scalar(Scalar::F32(a.path_losses_58)));
    f.push("path_losses_59", Byte(876), "dB", "Path loss, IFDR IF anti-alias filter", Value::Scalar(Scalar::F32(a.path_losses_59)));
    f.push("path_losses_60", Byte(880), "dB", "Path loss, A20J1_5 power splitter", Value::Scalar(Scalar::F32(a.path_losses_60)));
    f.push("path_losses_61", Byte(884), "dB", "Path loss, AT5 50 dB attenuator", Value::Scalar(Scalar::F32(a.path_losses_61)));
    f.push("path_losses_63", Byte(892), "dB", "Path loss, A39 RF/IF burst mixer", Value::Scalar(Scalar::F32(a.path_losses_63)));
    f.push("path_losses_64", Byte(896), "dB", "Path loss (gain), AR1 burst IF amplifier", Value::Scalar(Scalar::F32(a.path_losses_64)));
    f.push("path_losses_65", Byte(900), "dB", "Path loss, IFDR burst anti-alias filter", Value::Scalar(Scalar::F32(a.path_losses_65)));
    f.push("path_losses_66", Byte(904), "", "Path loss", Value::Scalar(Scalar::F32(a.path_losses_66)));
    f.push("path_losses_67", Byte(908), "dB", "Path loss, 4DC3J1 to 4A39 L", Value::Scalar(Scalar::F32(a.path_losses_67)));
    f.push("path_losses_68", Byte(912), "dB", "Path loss, AT2+AT3 26 dB COHO attenuator", Value::Scalar(Scalar::F32(a.path_losses_68)));
    f.push("chan_cal_diff", Byte(920), "dB", "Non-controlling channel calibration difference", Value::Scalar(Scalar::F32(a.chan_cal_diff)));
    f.push("v_ts_cw", Byte(936), "dBm", "AME vertical test signal power", Value::Scalar(Scalar::F32(a.v_ts_cw)));
    f.push("h_rnscale", Byte(940), "1", "Horizontal receiver noise normalization per elevation sector (-1.0 to -0.5 deg, -0.5 to 0.0 deg, ... 4.5 to 5.0 deg, above 5.0 deg)", Value::Array(ArrayBuf::F32(a.h_rnscale.to_vec())));
    f.push("atmos", Byte(992), "dB km-1", "Two-way atmospheric loss per km for the elevation sectors of Table XVI (-1.0 to -0.5 deg, ... above 5.0 deg)", Value::Array(ArrayBuf::F32(a.atmos.to_vec())));
    f.push("el_index", Byte(1044), "degree", "Bypass map generation elevation angles", Value::Array(ArrayBuf::F32(a.el_index.to_vec())));
    f.push("tfreq_mhz", Byte(1092), "MHz", "Transmitter frequency", Value::Scalar(Scalar::I32(a.tfreq_mhz)));
    f.push("base_data_tcn", Byte(1096), "dB", "Point clutter suppression threshold (TCN)", Value::Scalar(Scalar::F32(a.base_data_tcn)));
    f.push("refl_data_tover", Byte(1100), "dB", "Range unfolding overlay threshold (TOVER)", Value::Scalar(Scalar::F32(a.refl_data_tover)));
    f.push("tar_h_dbz0_lp", Byte(1104), "dBZ", "Horizontal target system calibration (dBZ0) for long pulse", Value::Scalar(Scalar::F32(a.tar_h_dbz0_lp)));
    f.push("tar_v_dbz0_lp", Byte(1108), "dBZ", "Vertical target system calibration (dBZ0) for long pulse", Value::Scalar(Scalar::F32(a.tar_v_dbz0_lp)));
    f.push("init_phi_dp", Byte(1112), "degree", "Initial system differential phase", Value::Scalar(Scalar::I32(a.init_phi_dp)));
    f.push("norm_init_phi_dp", Byte(1116), "degree", "Normalized initial system differential phase", Value::Scalar(Scalar::I32(a.norm_init_phi_dp)));
    f.push("lx_lp", Byte(1120), "dB", "Matched filter loss for long pulse", Value::Scalar(Scalar::F32(a.lx_lp)));
    f.push("lx_sp", Byte(1124), "dB", "Matched filter loss for short pulse", Value::Scalar(Scalar::F32(a.lx_sp)));
    f.push("meteor_param", Byte(1128), "1", "Hydrometeor refractivity factor |K|^2", Value::Scalar(Scalar::F32(a.meteor_param)));
    f.push("beamwidth", Byte(1132), "degree", "Antenna beamwidth", Value::Scalar(Scalar::F32(a.beamwidth)));
    f.push("antenna_gain", Byte(1136), "dB", "Antenna gain including radome", Value::Scalar(Scalar::F32(a.antenna_gain)));
    f.push("vel_degrad_limit", Byte(1152), "m s-1", "Velocity check delta degrade limit", Value::Scalar(Scalar::F32(a.vel_degrad_limit)));
    f.push("wth_degrad_limit", Byte(1156), "m s-1", "Spectrum width check delta degrade limit", Value::Scalar(Scalar::F32(a.wth_degrad_limit)));
    f.push("h_noisetemp_dgrad_limit", Byte(1160), "K", "Horizontal system noise temperature degrade limit", Value::Scalar(Scalar::F32(a.h_noisetemp_dgrad_limit)));
    f.push("h_min_noisetemp", Byte(1164), "K", "Horizontal system noise temperature too-low limit", Value::Scalar(Scalar::I32(a.h_min_noisetemp)));
    f.push("v_noisetemp_dgrad_limit", Byte(1168), "K", "Vertical system noise temperature degrade limit", Value::Scalar(Scalar::F32(a.v_noisetemp_dgrad_limit)));
    f.push("v_min_noisetemp", Byte(1172), "K", "Vertical system noise temperature too-low limit", Value::Scalar(Scalar::I32(a.v_min_noisetemp)));
    f.push("kly_degrade_limit", Byte(1176), "dB", "Klystron output target consistency degrade limit", Value::Scalar(Scalar::F32(a.kly_degrade_limit)));
    f.push("ts_coho", Byte(1180), "dBm", "COHO power at A1J4", Value::Scalar(Scalar::F32(a.ts_coho)));
    f.push("h_ts_cw", Byte(1184), "dBm", "AME horizontal test signal power", Value::Scalar(Scalar::F32(a.h_ts_cw)));
    f.push("ts_stalo", Byte(1196), "dBm", "STALO power at A1J2", Value::Scalar(Scalar::F32(a.ts_stalo)));
    f.push("ame_h_noise_enr", Byte(1200), "dB", "AME noise source horizontal excess noise ratio", Value::Scalar(Scalar::F32(a.ame_h_noise_enr)));
    f.push("xmtr_peak_pwr_high_limit", Byte(1204), "kW", "Maximum transmitter peak power alarm level", Value::Scalar(Scalar::F32(a.xmtr_peak_pwr_high_limit)));
    f.push("xmtr_peak_pwr_low_limit", Byte(1208), "kW", "Minimum transmitter peak power alarm level", Value::Scalar(Scalar::F32(a.xmtr_peak_pwr_low_limit)));
    f.push("h_dbz0_delta_limit", Byte(1212), "dB", "Difference between computed and target horizontal dBZ0 limit", Value::Scalar(Scalar::F32(a.h_dbz0_delta_limit)));
    f.push("threshold1", Byte(1216), "dB", "Bypass map generator noise threshold", Value::Scalar(Scalar::F32(a.threshold1)));
    f.push("threshold2", Byte(1220), "dB", "Bypass map generator rejection ratio threshold", Value::Scalar(Scalar::F32(a.threshold2)));
    f.push("clut_supp_dgrad_lim", Byte(1224), "dB", "Clutter suppression degrade limit", Value::Scalar(Scalar::F32(a.clut_supp_dgrad_lim)));
    f.push("range0_value", Byte(1232), "km", "True range at the start of the first range bin", Value::Scalar(Scalar::F32(a.range0_value)));
    f.push("xmtr_pwr_mtr_scale", Byte(1236), "W", "Scale factor converting transmitter power byte data to watts, the value of the LSB of the power measurement", Value::Scalar(Scalar::F32(a.xmtr_pwr_mtr_scale)));
    f.push("v_dbz0_delta_limit", Byte(1240), "dB", "Difference between computed and target vertical dBZ0 limit", Value::Scalar(Scalar::F32(a.v_dbz0_delta_limit)));
    f.push("tar_h_dbz0_sp", Byte(1244), "dBZ", "Horizontal target system calibration (dBZ0) for short pulse", Value::Scalar(Scalar::F32(a.tar_h_dbz0_sp)));
    f.push("tar_v_dbz0_sp", Byte(1248), "dBZ", "Vertical target system calibration (dBZ0) for short pulse", Value::Scalar(Scalar::F32(a.tar_v_dbz0_sp)));
    f.push("deltaprf", Byte(1252), "", "Site PRF set", Value::Scalar(Scalar::I32(a.deltaprf)));
    f.push("tau_sp", Byte(1264), "ns", "Pulse width of the transmitter output in short pulse", Value::Scalar(Scalar::I32(a.tau_sp)));
    f.push("tau_lp", Byte(1268), "ns", "Pulse width of the transmitter output in long pulse", Value::Scalar(Scalar::I32(a.tau_lp)));
    f.push("nc_dead_value", Byte(1272), "", "Number of 1/4 km bins of corrupted data at the end of a sweep (1 to 10)", Value::Scalar(Scalar::I32(a.nc_dead_value)));
    f.push("tau_rf_sp", Byte(1276), "ns", "RF drive pulse width in short pulse", Value::Scalar(Scalar::I32(a.tau_rf_sp)));
    f.push("tau_rf_lp", Byte(1280), "ns", "RF drive pulse width in long pulse", Value::Scalar(Scalar::I32(a.tau_rf_lp)));
    f.push("seg1lim", Byte(1284), "degree", "Clutter map boundary elevation between segments 1 and 2", Value::Scalar(Scalar::F32(a.seg1lim)));
    f.push("slatsec", Byte(1288), "arc_second", "Site latitude", Value::Scalar(Scalar::F32(a.slatsec)));
    f.push("slonsec", Byte(1292), "arc_second", "Site longitude", Value::Scalar(Scalar::F32(a.slonsec)));
    f.push("slatdeg", Byte(1300), "degree", "Site latitude", Value::Scalar(Scalar::I32(a.slatdeg)));
    f.push("slatmin", Byte(1304), "arc_minute", "Site latitude", Value::Scalar(Scalar::I32(a.slatmin)));
    f.push("slondeg", Byte(1308), "degree", "Site longitude", Value::Scalar(Scalar::I32(a.slondeg)));
    f.push("slonmin", Byte(1312), "arc_minute", "Site longitude", Value::Scalar(Scalar::I32(a.slonmin)));
    f.push("slatdir", Byte(1316), "", "Site latitude direction", Value::Text(a.slatdir.as_str()));
    f.push("slondir", Byte(1320), "", "Site longitude direction", Value::Text(a.slondir.as_str()));
    f.push("dig_rcvr_clock_freq", Byte(2500), "MHz", "Digital receiver clock frequency", Value::Scalar(Scalar::F64(a.dig_rcvr_clock_freq)));
    f.push("coho_freq", Byte(2508), "MHz", "COHO frequency", Value::Scalar(Scalar::F64(a.coho_freq)));
    f.push("az_correction_factor", Byte(8360), "degree", "Azimuth boresight correction factor", Value::Scalar(Scalar::F32(a.az_correction_factor)));
    f.push("el_correction_factor", Byte(8364), "degree", "Elevation boresight correction factor", Value::Scalar(Scalar::F32(a.el_correction_factor)));
    f.push("site_name", Byte(8368), "", "Site name designation (ICAO identifier)", Value::Text(a.site_name.as_str()));
    f.push("ant_manual_setup_ielmin", Byte(8372), "", "Minimum elevation angle as a two's complement binary angle (-7281 to 7281); multiply by 360/65536 for degrees (-39.99573 to 39.99573)", Value::Scalar(Scalar::I32(a.ant_manual_setup_ielmin)));
    f.push("ant_manual_setup_ielmax", Byte(8376), "", "Maximum elevation angle as a binary angle (0 to 40049); multiply by 360/65536 for degrees (0.00000 to 219.99573)", Value::Scalar(Scalar::I32(a.ant_manual_setup_ielmax)));
    f.push("ant_manual_setup_fazvelmax", Byte(8380), "degree s-1", "Maximum azimuth velocity", Value::Scalar(Scalar::I32(a.ant_manual_setup_fazvelmax)));
    f.push("ant_manual_setup_felvelmax", Byte(8384), "degree s-1", "Maximum elevation velocity", Value::Scalar(Scalar::I32(a.ant_manual_setup_felvelmax)));
    f.push("ant_manual_setup_ignd_hgt", Byte(8388), "m", "Site ground height above sea level", Value::Scalar(Scalar::I32(a.ant_manual_setup_ignd_hgt)));
    f.push("ant_manual_setup_irad_hgt", Byte(8392), "m", "Site radar height above ground", Value::Scalar(Scalar::I32(a.ant_manual_setup_irad_hgt)));
    f.push("az_pos_sustain_drive", Byte(8396), "1", "Azimuth motor positive sustaining drive", Value::Scalar(Scalar::F32(a.az_pos_sustain_drive)));
    f.push("az_neg_sustain_drive", Byte(8400), "1", "Azimuth motor negative sustaining drive", Value::Scalar(Scalar::F32(a.az_neg_sustain_drive)));
    f.push("az_nom_pos_drive_slope", Byte(8404), "1", "Initial estimate for the azimuth positive drive slope", Value::Scalar(Scalar::F32(a.az_nom_pos_drive_slope)));
    f.push("az_nom_neg_drive_slope", Byte(8408), "1", "Initial estimate for the azimuth negative drive slope", Value::Scalar(Scalar::F32(a.az_nom_neg_drive_slope)));
    f.push("az_feedback_slope", Byte(8412), "1", "Azimuth velocity feedback slope", Value::Scalar(Scalar::F32(a.az_feedback_slope)));
    f.push("el_pos_sustain_drive", Byte(8416), "1", "Elevation motor positive sustaining drive", Value::Scalar(Scalar::F32(a.el_pos_sustain_drive)));
    f.push("el_neg_sustain_drive", Byte(8420), "1", "Elevation motor negative sustaining drive", Value::Scalar(Scalar::F32(a.el_neg_sustain_drive)));
    f.push("el_nom_pos_drive_slope", Byte(8424), "1", "Initial estimate for the elevation positive drive slope", Value::Scalar(Scalar::F32(a.el_nom_pos_drive_slope)));
    f.push("el_nom_neg_drive_slope", Byte(8428), "1", "Initial estimate for the elevation negative drive slope", Value::Scalar(Scalar::F32(a.el_nom_neg_drive_slope)));
    f.push("el_feedback_slope", Byte(8432), "1", "Elevation velocity feedback slope", Value::Scalar(Scalar::F32(a.el_feedback_slope)));
    f.push("el_first_slope", Byte(8436), "1", "Slope for the first interval of the elevation position feedback curve", Value::Scalar(Scalar::F32(a.el_first_slope)));
    f.push("el_second_slope", Byte(8440), "1", "Slope for the second interval of the elevation position feedback curve", Value::Scalar(Scalar::F32(a.el_second_slope)));
    f.push("el_third_slope", Byte(8444), "1", "Slope for the third interval of the elevation position feedback curve", Value::Scalar(Scalar::F32(a.el_third_slope)));
    f.push("el_droop_pos", Byte(8448), "degree", "Neutral droop angle", Value::Scalar(Scalar::F32(a.el_droop_pos)));
    f.push("el_off_neutral_drive", Byte(8452), "1", "90 degree off-neutral drive", Value::Scalar(Scalar::F32(a.el_off_neutral_drive)));
    f.push("az_inertia", Byte(8456), "1", "Azimuth moment of inertia", Value::Scalar(Scalar::F32(a.az_inertia)));
    f.push("el_inertia", Byte(8460), "1", "Elevation moment of inertia", Value::Scalar(Scalar::F32(a.el_inertia)));
    f.push("az_stow_angle", Byte(8496), "degree", "Azimuth stow angle for encoder alignment", Value::Scalar(Scalar::F32(a.az_stow_angle)));
    f.push("el_stow_angle", Byte(8500), "degree", "Elevation stow angle for encoder alignment", Value::Scalar(Scalar::F32(a.el_stow_angle)));
    f.push("az_encoder_alignment", Byte(8504), "degree", "Azimuth encoder alignment ETU angle", Value::Scalar(Scalar::F32(a.az_encoder_alignment)));
    f.push("el_encoder_alignment", Byte(8508), "degree", "Elevation encoder alignment ETU angle", Value::Scalar(Scalar::F32(a.el_encoder_alignment)));
    f.push("refined_park", Byte(8688), "", "Refined park in use", Value::Flag(a.refined_park));
    f.push("rvp8nv_iwaveguide_length", Byte(8696), "m", "Waveguide length", Value::Scalar(Scalar::I32(a.rvp8nv_iwaveguide_length)));
    f.push("v_rnscale", Byte(8700), "1", "Vertical receiver noise normalization for the elevation sectors of H_RNSCALE", Value::Array(ArrayBuf::F32(a.v_rnscale.to_vec())));
    f.push("vel_data_tover", Byte(8744), "dB", "Velocity unfolding overlay threshold", Value::Scalar(Scalar::F32(a.vel_data_tover)));
    f.push("width_data_tover", Byte(8748), "dB", "Spectrum width unfolding overlay threshold", Value::Scalar(Scalar::F32(a.width_data_tover)));
    f.push("doppler_range_start", Byte(8764), "km", "Start range for the first Doppler radial", Value::Scalar(Scalar::F32(a.doppler_range_start)));
    f.push("max_el_index", Byte(8768), "", "Maximum index for the EL_INDEX parameters (0 to 11)", Value::Scalar(Scalar::I32(a.max_el_index)));
    f.push("seg2lim", Byte(8772), "degree", "Clutter map boundary elevation between segments 2 and 3", Value::Scalar(Scalar::F32(a.seg2lim)));
    f.push("seg3lim", Byte(8776), "degree", "Clutter map boundary elevation between segments 3 and 4", Value::Scalar(Scalar::F32(a.seg3lim)));
    f.push("seg4lim", Byte(8780), "degree", "Clutter map boundary elevation between segments 4 and 5", Value::Scalar(Scalar::F32(a.seg4lim)));
    f.push("nbr_el_segments", Byte(8784), "", "Number of elevation segments in the ORDA clutter map (1 to 5)", Value::Scalar(Scalar::I32(a.nbr_el_segments)));
    f.push("h_noise_long", Byte(8788), "dBm", "Horizontal receiver noise for long pulse", Value::Scalar(Scalar::F32(a.h_noise_long)));
    f.push("ant_noise_temp", Byte(8792), "K", "Antenna noise temperature", Value::Scalar(Scalar::F32(a.ant_noise_temp)));
    f.push("h_noise_short", Byte(8796), "dBm", "Horizontal receiver noise for short pulse", Value::Scalar(Scalar::F32(a.h_noise_short)));
    f.push("h_noise_tolerance", Byte(8800), "dB", "Horizontal receiver noise tolerance", Value::Scalar(Scalar::F32(a.h_noise_tolerance)));
    f.push("min_h_dyn_range", Byte(8804), "dB", "Minimum horizontal dynamic range", Value::Scalar(Scalar::F32(a.min_h_dyn_range)));
    f.push("gen_installed", Byte(8808), "", "Auxiliary generator installed (FAA only)", Value::Flag(a.gen_installed));
    f.push("gen_exercise", Byte(8812), "", "Auxiliary generator automatic exercise enabled (FAA only)", Value::Flag(a.gen_exercise));
    f.push("v_noise_tolerance", Byte(8816), "dB", "Vertical receiver noise tolerance", Value::Scalar(Scalar::F32(a.v_noise_tolerance)));
    f.push("min_v_dyn_range", Byte(8820), "dB", "Minimum vertical dynamic range", Value::Scalar(Scalar::F32(a.min_v_dyn_range)));
    f.push("zdr_offset_dgrad_lim", Byte(8824), "dB", "System differential reflectivity offset degrade limit", Value::Scalar(Scalar::F32(a.zdr_offset_dgrad_lim)));
    f.push("baseline_zdr_offset", Byte(8828), "dB", "Baseline system differential reflectivity offset", Value::Scalar(Scalar::F32(a.baseline_zdr_offset)));
    f.push("v_noise_long", Byte(8844), "dBm", "Vertical receiver noise for long pulse", Value::Scalar(Scalar::F32(a.v_noise_long)));
    f.push("v_noise_short", Byte(8848), "dBm", "Vertical receiver noise for short pulse", Value::Scalar(Scalar::F32(a.v_noise_short)));
    f.push("zdr_data_tover", Byte(8852), "dB", "ZDR unfolding overlay threshold", Value::Scalar(Scalar::F32(a.zdr_data_tover)));
    f.push("phi_data_tover", Byte(8856), "dB", "PHI unfolding overlay threshold", Value::Scalar(Scalar::F32(a.phi_data_tover)));
    f.push("rho_data_tover", Byte(8860), "dB", "RHO unfolding overlay threshold", Value::Scalar(Scalar::F32(a.rho_data_tover)));
    f.push("stalo_power_dgrad_limit", Byte(8864), "V", "STALO power degrade limit", Value::Scalar(Scalar::F32(a.stalo_power_dgrad_limit)));
    f.push("stalo_power_maint_limit", Byte(8868), "V", "STALO power maintenance limit", Value::Scalar(Scalar::F32(a.stalo_power_maint_limit)));
    f.push("min_h_pwr_sense", Byte(8872), "dBm", "Minimum horizontal power sense", Value::Scalar(Scalar::F32(a.min_h_pwr_sense)));
    f.push("min_v_pwr_sense", Byte(8876), "dBm", "Minimum vertical power sense", Value::Scalar(Scalar::F32(a.min_v_pwr_sense)));
    f.push("h_pwr_sense_offset", Byte(8880), "dB", "Horizontal power sense calibration offset", Value::Scalar(Scalar::F32(a.h_pwr_sense_offset)));
    f.push("v_pwr_sense_offset", Byte(8884), "dB", "Vertical power sense calibration offset", Value::Scalar(Scalar::F32(a.v_pwr_sense_offset)));
    f.push("ps_gain_ref", Byte(8888), "dB", "Power sense gain reference value", Value::Scalar(Scalar::F32(a.ps_gain_ref)));
    f.push("rf_pallet_broad_loss", Byte(8892), "dB", "RF pallet broadband loss", Value::Scalar(Scalar::F32(a.rf_pallet_broad_loss)));
    f.push("ame_ps_tolerance", Byte(8960), "percent", "AME power supply tolerance", Value::Scalar(Scalar::F32(a.ame_ps_tolerance)));
    f.push("ame_max_temp", Byte(8964), "degC", "Maximum AME internal alarm temperature", Value::Scalar(Scalar::F32(a.ame_max_temp)));
    f.push("ame_min_temp", Byte(8968), "degC", "Minimum AME internal alarm temperature", Value::Scalar(Scalar::F32(a.ame_min_temp)));
    f.push("rcvr_mod_max_temp", Byte(8972), "degC", "Maximum AME receiver module alarm temperature", Value::Scalar(Scalar::F32(a.rcvr_mod_max_temp)));
    f.push("rcvr_mod_min_temp", Byte(8976), "degC", "Minimum AME receiver module alarm temperature", Value::Scalar(Scalar::F32(a.rcvr_mod_min_temp)));
    f.push("bite_mod_max_temp", Byte(8980), "degC", "Maximum AME BITE module alarm temperature", Value::Scalar(Scalar::F32(a.bite_mod_max_temp)));
    f.push("bite_mod_min_temp", Byte(8984), "degC", "Minimum AME BITE module alarm temperature", Value::Scalar(Scalar::F32(a.bite_mod_min_temp)));
    f.push("default_polarization", Byte(8988), "", "Default (H+V) microwave assembly phase shifter position (0 to 60000)", Value::Scalar(Scalar::I32(a.default_polarization)));
    f.push("tr_limit_dgrad_limit", Byte(8992), "V", "TR limiter degrade limit", Value::Scalar(Scalar::F32(a.tr_limit_dgrad_limit)));
    f.push("tr_limit_fail_limit", Byte(8996), "V", "TR limiter failure limit", Value::Scalar(Scalar::F32(a.tr_limit_fail_limit)));
    f.push("rfp_stepper_enabled", Byte(9000), "", "Whether the RF pallet stepper motor is enabled", Value::Flag(a.rfp_stepper_enabled));
    f.push("ame_current_tolerance", Byte(9008), "percent", "AME Peltier current tolerance", Value::Scalar(Scalar::F32(a.ame_current_tolerance)));
    f.push("h_only_polarization", Byte(9012), "", "Horizontal-only microwave assembly phase shifter position (0 to 60000)", Value::Scalar(Scalar::I32(a.h_only_polarization)));
    f.push("v_only_polarization", Byte(9016), "", "Vertical-only microwave assembly phase shifter position (0 to 60000)", Value::Scalar(Scalar::I32(a.v_only_polarization)));
    f.push("sun_bias", Byte(9028), "dB", "Sun measurement bias", Value::Scalar(Scalar::F32(a.sun_bias)));
    f.push("a_min_shelter_temp_warn", Byte(9032), "degC", "Low equipment shelter temperature warning limit", Value::Scalar(Scalar::F32(a.a_min_shelter_temp_warn)));
    f.push("power_meter_zero", Byte(9036), "V", "Power meter zero bias voltage", Value::Scalar(Scalar::F32(a.power_meter_zero)));
    f.push("txb_baseline", Byte(9040), "dB", "Expected value of the RDA transmit bias (TXB)", Value::Scalar(Scalar::F32(a.txb_baseline)));
    f.push("txb_alarm_thresh", Byte(9044), "dB", "Threshold on the difference between a measured transmit bias and TXB_BASELINE above which the RDA sets an alarm", Value::Scalar(Scalar::F32(a.txb_alarm_thresh)));
    f.push("normal_tps_power_time", Byte(9048), "s", "Normal TPS power time", Value::Scalar(Scalar::I32(a.normal_tps_power_time)));
}
