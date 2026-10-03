//! RDA Status Data (message 2).
//!
//! Two layouts exist, told apart by the RDA redundant channel byte of the
//! message header (Table II): bit 3 set means an Open RDA (ORDA), clear means
//! the legacy RDA that ORDA replaced in 2005-2008.
//!
//! - [`OrdaRdaStatus`] follows ICD 2620002AA (Build 24.0) Table IV. The body
//!   is 40 halfwords up to Build 17 and 60 halfwords from Build 18.0; the
//!   extra halfwords (41 to 60) decode to `Some` only when present. Halfwords
//!   that were spare in the build that wrote a file hold zero, so fields added
//!   by later builds read as their zero code. TDWR supplemental product
//!   generator files also set bit 3 and use this layout.
//! - [`LegacyRdaStatus`] follows ICD 2620002B (Open Build 1.0, 2001)
//!   Table IV, the last revision that documents the legacy RDA. It is 40
//!   halfwords; halfwords 6, 10 to 15, 21 to 23 and 26 differ from ORDA.
//!
//! Units follow the ICD: W for transmitter power, dB for calibration
//! corrections, days and minutes for map generation times.
//!
//! The alarm codes of halfwords 27 to 40 are kept as numbers; the alarm text
//! of Table IV-A is not included.

use std::borrow::Cow;

use chrono::{DateTime, TimeDelta, Utc};

use super::MessageBody;
use crate::{MessageHeader, Result};

/// Body length of the legacy RDA layout and of ORDA builds before 18.0:
/// 40 halfwords.
pub const RDA_STATUS_SHORT_LEN: usize = 80;

/// Body length of the ORDA layout from Build 18.0 on: 60 halfwords.
pub const RDA_STATUS_LONG_LEN: usize = 120;

/// Number of alarm code halfwords (27 to 40).
pub const RDA_STATUS_ALARM_SLOTS: usize = 14;

/// Which RDA produced a message, from the RDA redundant channel byte of the
/// message header (Table II).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RdaSystem {
    /// Bit 3 clear: legacy RDA (channel codes 0, 1, 2).
    Legacy,
    /// Bit 3 set: Open RDA (channel codes 8, 9, 10).
    Orda,
}

impl RdaSystem {
    /// Classify a header's `channels` byte.
    pub fn from_channels(channels: u8) -> Self {
        if channels & 0x08 != 0 {
            Self::Orda
        } else {
            Self::Legacy
        }
    }
}

/// Decoded RDA Status Data in the layout of the RDA that sent it.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum RdaStatus {
    /// Open RDA layout (ICD 2620002AA Table IV).
    Orda(OrdaRdaStatus),
    /// Legacy RDA layout (ICD 2620002B Table IV).
    Legacy(LegacyRdaStatus),
}

impl RdaStatus {
    /// Decode a message body (the bytes after the 16-byte message header),
    /// choosing the layout from the header's `channels` byte.
    pub fn decode(channels: u8, body: &[u8]) -> Result<Self> {
        match RdaSystem::from_channels(channels) {
            RdaSystem::Orda => OrdaRdaStatus::decode(body).map(Self::Orda),
            RdaSystem::Legacy => LegacyRdaStatus::decode(body).map(Self::Legacy),
        }
    }

    /// Halfword 1: RDA state.
    pub fn rda_state(&self) -> RdaState {
        match self {
            Self::Orda(status) => status.rda_state,
            Self::Legacy(status) => status.rda_state,
        }
    }

    /// Halfword 8: volume coverage pattern and how it was selected.
    pub fn volume_coverage_pattern(&self) -> VcpSelection {
        match self {
            Self::Orda(status) => status.volume_coverage_pattern,
            Self::Legacy(status) => status.volume_coverage_pattern,
        }
    }

    /// Halfword 10 of the ORDA layout; `None` for the legacy RDA, which sends
    /// an interference detection rate there.
    pub fn rda_build(&self) -> Option<RdaBuild> {
        match self {
            Self::Orda(status) => Some(status.rda_build),
            Self::Legacy(_) => None,
        }
    }

    /// Halfwords 27 to 40: the alarms reported, skipping empty slots.
    pub fn alarms(&self) -> impl Iterator<Item = RdaAlarm> + '_ {
        let codes = match self {
            Self::Orda(status) => &status.alarm_codes,
            Self::Legacy(status) => &status.alarm_codes,
        };
        codes.iter().copied().filter_map(RdaAlarm::from_halfword)
    }
}

/// RDA Status Data from an Open RDA (ICD 2620002AA Table IV).
#[derive(Clone, Debug, PartialEq)]
pub struct OrdaRdaStatus {
    /// Halfword 1.
    pub rda_state: RdaState,
    /// Halfword 2.
    pub operability: OperabilityStatus,
    /// Halfword 3.
    pub control_status: ControlStatus,
    /// Halfword 4.
    pub auxiliary_power: AuxiliaryPower,
    /// Halfword 5: average transmitter power, W (0 to 9999).
    pub average_transmitter_power: u16,
    /// Halfword 6: horizontal reflectivity calibration correction (delta
    /// dBZ0), the difference from adaptation data, dB (-198.00 to +198.00;
    /// sent as hundredths).
    pub horizontal_reflectivity_calibration_correction: f32,
    /// Halfword 7.
    pub data_transmission: DataTransmission,
    /// Halfword 8.
    pub volume_coverage_pattern: VcpSelection,
    /// Halfword 9.
    pub control_authorization: ControlAuthorization,
    /// Halfword 10.
    pub rda_build: RdaBuild,
    /// Halfword 11.
    pub operational_mode: OperationalMode,
    /// Halfword 12.
    pub super_resolution: EnableStatus,
    /// Halfword 13.
    pub clutter_mitigation_decision: ClutterMitigationDecision,
    /// Halfword 14: AVSET, EBC, RDA log data and time series recording.
    pub scan_data_flags: ScanDataFlags,
    /// Halfword 15.
    pub alarm_summary: AlarmSummary,
    /// Halfword 16.
    pub command_acknowledgment: CommandAcknowledgment,
    /// Halfword 17.
    pub channel_control: ChannelControlStatus,
    /// Halfword 18.
    pub spot_blanking: SpotBlanking,
    /// Halfwords 19 and 20: bypass map generation date and time.
    pub bypass_map_generation: MapGenerationTime,
    /// Halfwords 21 and 22: clutter filter map generation date and time.
    pub clutter_filter_map_generation: MapGenerationTime,
    /// Halfword 23: vertical reflectivity calibration correction, the
    /// difference from adaptation data, dB (-198.00 to +198.00; sent as
    /// hundredths).
    pub vertical_reflectivity_calibration_correction: f32,
    /// Halfword 24.
    pub transition_power_source: TransitionPowerSource,
    /// Halfword 25.
    pub rms_control: RmsControl,
    /// Halfword 26.
    pub performance_check: PerformanceCheckStatus,
    /// Halfwords 27 to 40: alarm codes, one per halfword, zero when unused.
    /// See [`RdaStatus::alarms`] / [`RdaAlarm`].
    pub alarm_codes: [u16; RDA_STATUS_ALARM_SLOTS],
    /// Halfword 41; `None` for 40-halfword bodies.
    pub signal_processing_options: Option<SignalProcessingOptions>,
    /// Halfword 59: the remote VCP number (1 to 767) the RDA acknowledges,
    /// 0 for none; `None` for 40-halfword bodies.
    pub downloaded_pattern_number: Option<u16>,
    /// Halfword 60: version of the status message; `None` for 40-halfword
    /// bodies.
    pub status_version: Option<u16>,
}

impl OrdaRdaStatus {
    /// Decode a message body (the bytes after the 16-byte message header).
    /// Needs at least 40 halfwords; halfwords 41 to 60 are read when the
    /// body holds 60.
    pub fn decode(body: &[u8]) -> Result<Self> {
        crate::require_len(body, 0, RDA_STATUS_SHORT_LEN, "RDA status data")?;
        let halfword = |number: usize| crate::be_u16(body, (number - 1) * 2);
        let long = body.len() >= RDA_STATUS_LONG_LEN;
        Ok(Self {
            rda_state: RdaState::from_code(halfword(1)),
            operability: OperabilityStatus::from_code(halfword(2)),
            control_status: ControlStatus::from_code(halfword(3)),
            auxiliary_power: AuxiliaryPower(halfword(4)),
            average_transmitter_power: halfword(5),
            horizontal_reflectivity_calibration_correction: hundredths(halfword(6)),
            data_transmission: DataTransmission(halfword(7)),
            volume_coverage_pattern: VcpSelection::from_code(halfword(8)),
            control_authorization: ControlAuthorization::from_code(halfword(9)),
            rda_build: RdaBuild(halfword(10)),
            operational_mode: OperationalMode::from_code(halfword(11)),
            super_resolution: EnableStatus::from_code(halfword(12)),
            clutter_mitigation_decision: ClutterMitigationDecision(halfword(13)),
            scan_data_flags: ScanDataFlags(halfword(14)),
            alarm_summary: AlarmSummary(halfword(15)),
            command_acknowledgment: CommandAcknowledgment::from_code(halfword(16)),
            channel_control: ChannelControlStatus::from_code(halfword(17)),
            spot_blanking: SpotBlanking::from_code(halfword(18)),
            bypass_map_generation: MapGenerationTime {
                date: halfword(19),
                minutes: halfword(20),
            },
            clutter_filter_map_generation: MapGenerationTime {
                date: halfword(21),
                minutes: halfword(22),
            },
            vertical_reflectivity_calibration_correction: hundredths(halfword(23)),
            transition_power_source: TransitionPowerSource::from_code(halfword(24)),
            rms_control: RmsControl::from_code(halfword(25)),
            performance_check: PerformanceCheckStatus::from_code(halfword(26)),
            alarm_codes: std::array::from_fn(|slot| halfword(27 + slot)),
            signal_processing_options: long.then(|| SignalProcessingOptions(halfword(41))),
            downloaded_pattern_number: long.then(|| halfword(59)),
            status_version: long.then(|| halfword(60)),
        })
    }
}

/// RDA Status Data from a legacy RDA (ICD 2620002B Table IV).
#[derive(Clone, Debug, PartialEq)]
pub struct LegacyRdaStatus {
    /// Halfword 1.
    pub rda_state: RdaState,
    /// Halfword 2.
    pub operability: OperabilityStatus,
    /// Halfword 3.
    pub control_status: ControlStatus,
    /// Halfword 4.
    pub auxiliary_power: AuxiliaryPower,
    /// Halfword 5: average transmitter power, W (0 to 9999).
    pub average_transmitter_power: u16,
    /// Halfword 6: reflectivity calibration correction, the difference from
    /// adaptation data, as sent. ICD 2620002B gives it as a fixed-point
    /// scaled integer in dB (-10 to +10, precision 0.25 dB) without stating
    /// the scale, so the raw value is kept.
    pub reflectivity_calibration_correction_raw: i16,
    /// Halfword 7.
    pub data_transmission: DataTransmission,
    /// Halfword 8.
    pub volume_coverage_pattern: VcpSelection,
    /// Halfword 9.
    pub control_authorization: ControlAuthorization,
    /// Halfword 10: interference detection rate, pulses per second
    /// (0 to 32767).
    pub interference_detection_rate: u16,
    /// Halfword 11.
    pub operational_mode: OperationalMode,
    /// Halfword 12: interference suppression unit.
    pub interference_suppression_unit: EnableStatus,
    /// Halfword 13: Archive II status code (tape drive state bits plus a tape
    /// number field), as sent.
    pub archive_ii_status: u16,
    /// Halfword 14: Archive II estimated remaining capacity, volume scans
    /// (worst case, 1 to 900).
    pub archive_ii_remaining_capacity: u16,
    /// Halfword 15.
    pub alarm_summary: LegacyAlarmSummary,
    /// Halfword 16.
    pub command_acknowledgment: CommandAcknowledgment,
    /// Halfword 17.
    pub channel_control: ChannelControlStatus,
    /// Halfword 18.
    pub spot_blanking: SpotBlanking,
    /// Halfwords 19 and 20: bypass map generation date and time.
    pub bypass_map_generation: MapGenerationTime,
    /// Halfwords 21 and 22: notch width map generation date and time.
    pub notch_width_map_generation: MapGenerationTime,
    /// Halfword 24.
    pub transition_power_source: TransitionPowerSource,
    /// Halfword 25 (code 4 means the maintenance console, MMI, is in
    /// control).
    pub rms_control: RmsControl,
    /// Halfwords 27 to 40: alarm codes, one per halfword, zero when unused.
    pub alarm_codes: [u16; RDA_STATUS_ALARM_SLOTS],
}

impl LegacyRdaStatus {
    /// Decode a message body (the bytes after the 16-byte message header).
    pub fn decode(body: &[u8]) -> Result<Self> {
        crate::require_len(body, 0, RDA_STATUS_SHORT_LEN, "legacy RDA status data")?;
        let halfword = |number: usize| crate::be_u16(body, (number - 1) * 2);
        Ok(Self {
            rda_state: RdaState::from_code(halfword(1)),
            operability: OperabilityStatus::from_code(halfword(2)),
            control_status: ControlStatus::from_code(halfword(3)),
            auxiliary_power: AuxiliaryPower(halfword(4)),
            average_transmitter_power: halfword(5),
            reflectivity_calibration_correction_raw: halfword(6) as i16,
            data_transmission: DataTransmission(halfword(7)),
            volume_coverage_pattern: VcpSelection::from_code(halfword(8)),
            control_authorization: ControlAuthorization::from_code(halfword(9)),
            interference_detection_rate: halfword(10),
            operational_mode: OperationalMode::from_code(halfword(11)),
            interference_suppression_unit: EnableStatus::from_code(halfword(12)),
            archive_ii_status: halfword(13),
            archive_ii_remaining_capacity: halfword(14),
            alarm_summary: LegacyAlarmSummary(halfword(15)),
            command_acknowledgment: CommandAcknowledgment::from_code(halfword(16)),
            channel_control: ChannelControlStatus::from_code(halfword(17)),
            spot_blanking: SpotBlanking::from_code(halfword(18)),
            bypass_map_generation: MapGenerationTime {
                date: halfword(19),
                minutes: halfword(20),
            },
            notch_width_map_generation: MapGenerationTime {
                date: halfword(21),
                minutes: halfword(22),
            },
            transition_power_source: TransitionPowerSource::from_code(halfword(24)),
            rms_control: RmsControl::from_code(halfword(25)),
            alarm_codes: std::array::from_fn(|slot| halfword(27 + slot)),
        })
    }
}

/// A scaled Integer*2 in hundredths (Table IV note 5).
fn hundredths(raw: u16) -> f32 {
    f32::from(raw as i16) / 100.0
}

/// RDA state (halfword 1; mutually exclusive codes).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RdaState {
    /// 2 (bit 1).
    StartUp,
    /// 4 (bit 2).
    Standby,
    /// 8 (bit 3).
    Restart,
    /// 16 (bit 4).
    Operate,
    /// 32 (bit 5): playback, legacy RDA only (spare for ORDA).
    Playback,
    /// 64 (bit 6): off-line operate, listed through Build 18.0 (spare in
    /// 2620002AA).
    OfflineOperate,
    /// Any other code.
    Unknown(u16),
}

impl RdaState {
    /// Map a Table IV code.
    pub fn from_code(code: u16) -> Self {
        match code {
            2 => Self::StartUp,
            4 => Self::Standby,
            8 => Self::Restart,
            16 => Self::Operate,
            32 => Self::Playback,
            64 => Self::OfflineOperate,
            other => Self::Unknown(other),
        }
    }

    /// The Table IV code, the inverse of [`Self::from_code`].
    pub fn code(self) -> u16 {
        match self {
            Self::StartUp => 2,
            Self::Standby => 4,
            Self::Restart => 8,
            Self::Operate => 16,
            Self::Playback => 32,
            Self::OfflineOperate => 64,
            Self::Unknown(code) => code,
        }
    }
}

/// Operability status (halfword 2).
///
/// Legacy RDAs and some ORDA builds (for example 2620002F, Build 10.0) add 1
/// (bit 0) when automatic calibration is disabled; [`OperabilityStatus::from_code`]
/// separates that bit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OperabilityStatus {
    /// The status with bit 0 removed.
    pub state: OperabilityState,
    /// Bit 0: automatic calibration disabled.
    pub automatic_calibration_disabled: bool,
}

impl OperabilityStatus {
    /// Map a Table IV code.
    pub fn from_code(code: u16) -> Self {
        Self {
            state: OperabilityState::from_code(code & !1),
            automatic_calibration_disabled: code & 1 != 0,
        }
    }

    /// The halfword 2 code, the inverse of [`Self::from_code`].
    pub fn code(self) -> u16 {
        self.state.code() | u16::from(self.automatic_calibration_disabled)
    }
}

/// Operability state (halfword 2 without bit 0).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum OperabilityState {
    /// 2 (bit 1): on-line.
    OnLine,
    /// 4 (bit 2): maintenance action required.
    MaintenanceActionRequired,
    /// 8 (bit 3): maintenance action mandatory.
    MaintenanceActionMandatory,
    /// 16 (bit 4): commanded shut down.
    CommandedShutDown,
    /// 32 (bit 5): inoperable.
    Inoperable,
    /// Any other code.
    Unknown(u16),
}

impl OperabilityState {
    /// Map a Table IV code (bit 0 already removed).
    pub fn from_code(code: u16) -> Self {
        match code {
            2 => Self::OnLine,
            4 => Self::MaintenanceActionRequired,
            8 => Self::MaintenanceActionMandatory,
            16 => Self::CommandedShutDown,
            32 => Self::Inoperable,
            other => Self::Unknown(other),
        }
    }

    /// The Table IV code, the inverse of [`Self::from_code`].
    pub fn code(self) -> u16 {
        match self {
            Self::OnLine => 2,
            Self::MaintenanceActionRequired => 4,
            Self::MaintenanceActionMandatory => 8,
            Self::CommandedShutDown => 16,
            Self::Inoperable => 32,
            Self::Unknown(code) => code,
        }
    }
}

/// Control status (halfword 3; mutually exclusive codes).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ControlStatus {
    /// 2 (bit 1): local only.
    LocalOnly,
    /// 4 (bit 2): RPG (remote) only.
    RemoteOnly,
    /// 8 (bit 3): either.
    Either,
    /// Any other code.
    Unknown(u16),
}

impl ControlStatus {
    /// Map a Table IV code.
    pub fn from_code(code: u16) -> Self {
        match code {
            2 => Self::LocalOnly,
            4 => Self::RemoteOnly,
            8 => Self::Either,
            other => Self::Unknown(other),
        }
    }

    /// The Table IV code, the inverse of [`Self::from_code`].
    pub fn code(self) -> u16 {
        match self {
            Self::LocalOnly => 2,
            Self::RemoteOnly => 4,
            Self::Either => 8,
            Self::Unknown(code) => code,
        }
    }
}

/// Auxiliary power generator state (halfword 4; any combination of bits).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AuxiliaryPower(pub u16);

impl AuxiliaryPower {
    /// Bit 0: switched to auxiliary power.
    pub fn switched_to_auxiliary_power(self) -> bool {
        self.0 & 1 != 0
    }
    /// Bit 1: utility power available.
    pub fn utility_power_available(self) -> bool {
        self.0 & 2 != 0
    }
    /// Bit 2: generator on.
    pub fn generator_on(self) -> bool {
        self.0 & 4 != 0
    }
    /// Bit 3: transfer switch in manual.
    pub fn transfer_switch_manual(self) -> bool {
        self.0 & 8 != 0
    }
    /// Bit 4: commanded switchover.
    pub fn commanded_switchover(self) -> bool {
        self.0 & 16 != 0
    }
}

/// Data transmission enabled (halfword 7; any combination of bits).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DataTransmission(pub u16);

impl DataTransmission {
    /// Bit 1: none.
    pub fn none(self) -> bool {
        self.0 & 2 != 0
    }
    /// Bit 2: reflectivity.
    pub fn reflectivity(self) -> bool {
        self.0 & 4 != 0
    }
    /// Bit 3: velocity.
    pub fn velocity(self) -> bool {
        self.0 & 8 != 0
    }
    /// Bit 4: spectrum width.
    pub fn width(self) -> bool {
        self.0 & 16 != 0
    }
}

/// Volume coverage pattern number (halfword 8, a signed Integer*2): the
/// magnitude is the pattern, the sign how it was selected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum VcpSelection {
    /// 0: no pattern.
    NoPattern,
    /// Negative: pattern selected locally at the RDA.
    Local(u16),
    /// Positive: pattern selected remotely (by the RPG).
    Remote(u16),
}

impl VcpSelection {
    /// Map the halfword.
    pub fn from_code(code: u16) -> Self {
        let signed = code as i16;
        match signed {
            0 => Self::NoPattern,
            s if s < 0 => Self::Local(s.unsigned_abs()),
            s => Self::Remote(s.unsigned_abs()),
        }
    }

    /// The pattern number, or `None` for no pattern.
    pub fn pattern(self) -> Option<u16> {
        match self {
            Self::NoPattern => None,
            Self::Local(pattern) | Self::Remote(pattern) => Some(pattern),
        }
    }

    /// The halfword as a signed number (negative for local selection).
    pub fn signed(self) -> i32 {
        match self {
            Self::NoPattern => 0,
            Self::Local(pattern) => -i32::from(pattern),
            Self::Remote(pattern) => i32::from(pattern),
        }
    }
}

/// RDA control authorization (halfword 9; mutually exclusive codes).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ControlAuthorization {
    /// 0: no action.
    NoAction,
    /// 2 (bit 1): local control requested.
    LocalControlRequested,
    /// 4 (bit 2): remote control requested, a.k.a. local control released
    /// ("remote control enabled" in the legacy table).
    RemoteControlRequested,
    /// Any other code.
    Unknown(u16),
}

impl ControlAuthorization {
    /// Map a Table IV code.
    pub fn from_code(code: u16) -> Self {
        match code {
            0 => Self::NoAction,
            2 => Self::LocalControlRequested,
            4 => Self::RemoteControlRequested,
            other => Self::Unknown(other),
        }
    }

    /// The Table IV code, the inverse of [`Self::from_code`].
    pub fn code(self) -> u16 {
        match self {
            Self::NoAction => 0,
            Self::LocalControlRequested => 2,
            Self::RemoteControlRequested => 4,
            Self::Unknown(code) => code,
        }
    }
}

/// RDA build number (halfword 10, scaled Integer*2, note 6). Message 31
/// layouts are selected from block sizes, but the build explains them; see
/// [`super::msg31_blocks`].
///
/// Encoding: when the raw value divided by 100 is greater than 2, the build
/// is the value divided by 100 (1320 is 13.2, 2410 is 24.1); otherwise it is
/// the value divided by 10 (100 is Build 10.0; TDWR records 20, 2.0). The
/// scale changed to 100 with ICD revision H (Build 11.2).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct RdaBuild(pub u16);

impl RdaBuild {
    /// Wrap a raw halfword 10 value.
    pub const fn from_raw(raw: u16) -> Self {
        Self(raw)
    }

    /// Read halfword 10 from a message 2 body (bytes 18-19), or `None` when
    /// the body is shorter. The halfword is read whatever the layout: in a
    /// legacy RDA body (see [`RdaSystem`]) it is the interference detection
    /// rate, not a build, and [`RdaStatus::rda_build`] returns `None`.
    pub fn from_rda_status_body(body: &[u8]) -> Option<Self> {
        (body.len() >= 20).then(|| Self::from_raw(crate::be_u16(body, 18)))
    }

    /// Halfword 10 of the first message 2 in record bytes that start at a
    /// frame boundary (for example [`super::metadata_record`]), or `None`
    /// when there is none. Like [`Self::from_rda_status_body`], this does not
    /// check the layout.
    pub fn from_records(records: &[u8]) -> Option<Self> {
        super::RawMessages::new(records)
            .flatten()
            .find(|message| message.header.message_type == 2)
            .and_then(|message| Self::from_rda_status_body(&message.body))
    }

    /// The raw halfword.
    pub const fn raw(self) -> u16 {
        self.0
    }

    /// Build number in hundredths (1320 for 13.2, 100 for 10.0).
    pub fn hundredths(self) -> u32 {
        if self.0 > 200 {
            u32::from(self.0)
        } else {
            u32::from(self.0) * 10
        }
    }

    /// Major build number (13 for 13.2).
    pub fn major(self) -> u32 {
        self.hundredths() / 100
    }

    /// Hundredths after the major number (20 for 13.2, 1 for 13.01).
    pub fn minor_hundredths(self) -> u32 {
        self.hundredths() % 100
    }

    /// Build number as a real value (13.2).
    pub fn version(self) -> f32 {
        self.hundredths() as f32 / 100.0
    }
}

impl std::fmt::Display for RdaBuild {
    /// "13.2", "10.0", "19.96".
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let minor = self.minor_hundredths();
        if minor.is_multiple_of(10) {
            write!(f, "{}.{}", self.major(), minor / 10)
        } else {
            write!(f, "{}.{minor:02}", self.major())
        }
    }
}

/// Operational mode (halfword 11; mutually exclusive codes).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum OperationalMode {
    /// 2 (bit 1): test (ORDA tables through Build 13.0). The legacy RDA table
    /// uses 2 for maintenance.
    Test,
    /// 4 (bit 2): operational.
    Operational,
    /// 8 (bit 3): maintenance.
    Maintenance,
    /// Any other code.
    Unknown(u16),
}

impl OperationalMode {
    /// Map a Table IV code.
    pub fn from_code(code: u16) -> Self {
        match code {
            2 => Self::Test,
            4 => Self::Operational,
            8 => Self::Maintenance,
            other => Self::Unknown(other),
        }
    }

    /// The Table IV code, the inverse of [`Self::from_code`].
    pub fn code(self) -> u16 {
        match self {
            Self::Test => 2,
            Self::Operational => 4,
            Self::Maintenance => 8,
            Self::Unknown(code) => code,
        }
    }
}

/// Enabled/disabled status coded 2/4 (super resolution in halfword 12 of the
/// ORDA layout, the interference suppression unit in the legacy layout).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum EnableStatus {
    /// 2 (bit 1).
    Enabled,
    /// 4 (bit 2).
    Disabled,
    /// Any other code, including 0 from builds where the halfword was spare.
    Unknown(u16),
}

impl EnableStatus {
    /// Map a Table IV code.
    pub fn from_code(code: u16) -> Self {
        match code {
            2 => Self::Enabled,
            4 => Self::Disabled,
            other => Self::Unknown(other),
        }
    }

    /// The Table IV code, the inverse of [`Self::from_code`].
    pub fn code(self) -> u16 {
        match self {
            Self::Enabled => 2,
            Self::Disabled => 4,
            Self::Unknown(code) => code,
        }
    }
}

/// Clutter mitigation decision status (halfword 13).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ClutterMitigationDecision(pub u16);

impl ClutterMitigationDecision {
    /// Bit 0: CMD enabled (0 means disabled).
    pub fn enabled(self) -> bool {
        self.0 & 1 != 0
    }

    /// Bits 1 to 5: whether CMD is applied in bypass map elevation segment
    /// `segment` (1 to 5); `false` for other segment numbers.
    pub fn applied_in_segment(self, segment: u8) -> bool {
        (1..=5).contains(&segment) && self.0 & (1 << segment) != 0
    }
}

/// RDA scan and data flags (halfword 14, Table IV note 10).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ScanDataFlags(pub u16);

impl ScanDataFlags {
    /// Bit 1: AVSET enabled.
    pub fn avset_enabled(self) -> bool {
        self.0 & 2 != 0
    }
    /// Bit 2: AVSET disabled.
    pub fn avset_disabled(self) -> bool {
        self.0 & 4 != 0
    }
    /// Bit 3: EBC (elevation-based clutter) enabled.
    pub fn ebc_enabled(self) -> bool {
        self.0 & 8 != 0
    }
    /// Bit 4: RDA log data (message 33) enabled.
    pub fn rda_log_data_enabled(self) -> bool {
        self.0 & 16 != 0
    }
    /// Bit 5: local time series data recording at the RDA.
    pub fn time_series_recording(self) -> bool {
        self.0 & 32 != 0
    }
}

/// RDA alarm summary (halfword 15, ORDA layout; any combination of bits, 0
/// for no alarms).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AlarmSummary(pub u16);

impl AlarmSummary {
    /// No bits set.
    pub fn no_alarms(self) -> bool {
        self.0 == 0
    }
    /// Bit 1: tower/utilities.
    pub fn tower_utilities(self) -> bool {
        self.0 & 2 != 0
    }
    /// Bit 2: pedestal.
    pub fn pedestal(self) -> bool {
        self.0 & 4 != 0
    }
    /// Bit 3: transmitter.
    pub fn transmitter(self) -> bool {
        self.0 & 8 != 0
    }
    /// Bit 4: receiver.
    pub fn receiver(self) -> bool {
        self.0 & 16 != 0
    }
    /// Bit 5: RDA control.
    pub fn rda_control(self) -> bool {
        self.0 & 32 != 0
    }
    /// Bit 6: communication.
    pub fn communication(self) -> bool {
        self.0 & 64 != 0
    }
    /// Bit 7: signal processor.
    pub fn signal_processor(self) -> bool {
        self.0 & 128 != 0
    }
}

/// RDA alarm summary (halfword 15, legacy layout; any combination of bits, 0
/// for no alarms).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LegacyAlarmSummary(pub u16);

impl LegacyAlarmSummary {
    /// No bits set.
    pub fn no_alarms(self) -> bool {
        self.0 == 0
    }
    /// Bit 1: tower/utilities.
    pub fn tower_utilities(self) -> bool {
        self.0 & 2 != 0
    }
    /// Bit 2: pedestal.
    pub fn pedestal(self) -> bool {
        self.0 & 4 != 0
    }
    /// Bit 3: transmitter.
    pub fn transmitter(self) -> bool {
        self.0 & 8 != 0
    }
    /// Bit 4: receiver/signal processor.
    pub fn receiver_signal_processor(self) -> bool {
        self.0 & 16 != 0
    }
    /// Bit 5: RDA control.
    pub fn rda_control(self) -> bool {
        self.0 & 32 != 0
    }
    /// Bit 6: RPG communication.
    pub fn rpg_communication(self) -> bool {
        self.0 & 64 != 0
    }
    /// Bit 7: user communication.
    pub fn user_communication(self) -> bool {
        self.0 & 128 != 0
    }
    /// Bit 8: Archive II.
    pub fn archive_ii(self) -> bool {
        self.0 & 256 != 0
    }
}

/// Command acknowledgment (halfword 16).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CommandAcknowledgment {
    /// 0: no acknowledgment.
    None,
    /// 1: remote VCP received.
    RemoteVcpReceived,
    /// 2: clutter bypass map received.
    BypassMapReceived,
    /// 3: clutter censor zones received.
    CensorZonesReceived,
    /// 4: redundant channel control command accepted (legacy: standby
    /// command accepted).
    RedundantChannelControlAccepted,
    /// Any other code.
    Unknown(u16),
}

impl CommandAcknowledgment {
    /// Map a Table IV code.
    pub fn from_code(code: u16) -> Self {
        match code {
            0 => Self::None,
            1 => Self::RemoteVcpReceived,
            2 => Self::BypassMapReceived,
            3 => Self::CensorZonesReceived,
            4 => Self::RedundantChannelControlAccepted,
            other => Self::Unknown(other),
        }
    }

    /// The Table IV code, the inverse of [`Self::from_code`].
    pub fn code(self) -> u16 {
        match self {
            Self::None => 0,
            Self::RemoteVcpReceived => 1,
            Self::BypassMapReceived => 2,
            Self::CensorZonesReceived => 3,
            Self::RedundantChannelControlAccepted => 4,
            Self::Unknown(code) => code,
        }
    }
}

/// Channel control status (halfword 17).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ChannelControlStatus {
    /// 0: this channel is the controlling channel.
    Controlling,
    /// 1 (bit 0): non-controlling.
    NonControlling,
    /// Any other code.
    Unknown(u16),
}

impl ChannelControlStatus {
    /// Map a Table IV code.
    pub fn from_code(code: u16) -> Self {
        match code {
            0 => Self::Controlling,
            1 => Self::NonControlling,
            other => Self::Unknown(other),
        }
    }

    /// The Table IV code, the inverse of [`Self::from_code`].
    pub fn code(self) -> u16 {
        match self {
            Self::Controlling => 0,
            Self::NonControlling => 1,
            Self::Unknown(code) => code,
        }
    }
}

/// Spot blanking status (halfword 18; mutually exclusive codes).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SpotBlanking {
    /// 0: not installed.
    NotInstalled,
    /// 2 (bit 1): enabled.
    Enabled,
    /// 4 (bit 2): disabled.
    Disabled,
    /// Any other code.
    Unknown(u16),
}

impl SpotBlanking {
    /// Map a Table IV code.
    pub fn from_code(code: u16) -> Self {
        match code {
            0 => Self::NotInstalled,
            2 => Self::Enabled,
            4 => Self::Disabled,
            other => Self::Unknown(other),
        }
    }

    /// The Table IV code, the inverse of [`Self::from_code`].
    pub fn code(self) -> u16 {
        match self {
            Self::NotInstalled => 0,
            Self::Enabled => 2,
            Self::Disabled => 4,
            Self::Unknown(code) => code,
        }
    }
}

/// Generation date and time of a clutter map (halfwords 19-20 and 21-22).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MapGenerationTime {
    /// Modified Julian date, days (Table IV note 1: 1 January 1970 is day 1);
    /// 0 when no map has been generated.
    pub date: u16,
    /// Minutes since midnight GMT (0 to 1440).
    pub minutes: u16,
}

impl MapGenerationTime {
    /// The generation time, or `None` when the date is 0.
    pub fn datetime(self) -> Option<DateTime<Utc>> {
        if self.date == 0 {
            return None;
        }
        let days = TimeDelta::days(i64::from(self.date) - 1);
        let minutes = TimeDelta::minutes(i64::from(self.minutes));
        DateTime::<Utc>::UNIX_EPOCH
            .checked_add_signed(days)?
            .checked_add_signed(minutes)
    }
}

/// Transition power source status (halfword 24).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TransitionPowerSource {
    /// 0: not installed.
    NotInstalled,
    /// 1 (bit 0): off.
    Off,
    /// 3 (bits 0 and 1): OK.
    Ok,
    /// 4 (bit 2): the RDA reports the state as unknown.
    StateUnknown,
    /// Any other code.
    Unknown(u16),
}

impl TransitionPowerSource {
    /// Map a Table IV code.
    pub fn from_code(code: u16) -> Self {
        match code {
            0 => Self::NotInstalled,
            1 => Self::Off,
            3 => Self::Ok,
            4 => Self::StateUnknown,
            other => Self::Unknown(other),
        }
    }

    /// The Table IV code, the inverse of [`Self::from_code`].
    pub fn code(self) -> u16 {
        match self {
            Self::NotInstalled => 0,
            Self::Off => 1,
            Self::Ok => 3,
            Self::StateUnknown => 4,
            Self::Unknown(code) => code,
        }
    }
}

/// RMS control status (halfword 25; mutually exclusive codes).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RmsControl {
    /// 0: non-RMS system.
    NonRms,
    /// 2 (bit 1): RMS in control.
    RmsInControl,
    /// 4 (bit 2): RDA in control (legacy: MMI in control).
    RdaInControl,
    /// Any other code.
    Unknown(u16),
}

impl RmsControl {
    /// Map a Table IV code.
    pub fn from_code(code: u16) -> Self {
        match code {
            0 => Self::NonRms,
            2 => Self::RmsInControl,
            4 => Self::RdaInControl,
            other => Self::Unknown(other),
        }
    }

    /// The Table IV code, the inverse of [`Self::from_code`].
    pub fn code(self) -> u16 {
        match self {
            Self::NonRms => 0,
            Self::RmsInControl => 2,
            Self::RdaInControl => 4,
            Self::Unknown(code) => code,
        }
    }
}

/// Performance check status (halfword 26; mutually exclusive codes).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum PerformanceCheckStatus {
    /// 0: no command pending.
    NoCommandPending,
    /// 1 (bit 0): force performance check pending.
    ForcePerformanceCheckPending,
    /// 2 (bit 1): in progress.
    InProgress,
    /// Any other code.
    Unknown(u16),
}

impl PerformanceCheckStatus {
    /// Map a Table IV code.
    pub fn from_code(code: u16) -> Self {
        match code {
            0 => Self::NoCommandPending,
            1 => Self::ForcePerformanceCheckPending,
            2 => Self::InProgress,
            other => Self::Unknown(other),
        }
    }

    /// The Table IV code, the inverse of [`Self::from_code`].
    pub fn code(self) -> u16 {
        match self {
            Self::NoCommandPending => 0,
            Self::ForcePerformanceCheckPending => 1,
            Self::InProgress => 2,
            Self::Unknown(code) => code,
        }
    }
}

/// Signal processing options (halfword 41).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SignalProcessingOptions(pub u16);

impl SignalProcessingOptions {
    /// Bit 0: the CMD rho-hv test is enabled.
    pub fn cmd_rho_hv_test_enabled(self) -> bool {
        self.0 & 1 != 0
    }
}

/// One alarm from halfwords 27 to 40.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RdaAlarm {
    /// Alarm code (Table IV-A; 1 to 800).
    pub code: u16,
    /// The most significant bit was set: the alarm has been cleared.
    pub cleared: bool,
}

impl RdaAlarm {
    /// Split an alarm halfword; `None` for an empty (zero) slot.
    pub fn from_halfword(halfword: u16) -> Option<Self> {
        let code = halfword & 0x7fff;
        (code != 0).then_some(Self {
            code,
            cleared: halfword & 0x8000 != 0,
        })
    }
}

/// Walker hook: the typed body for message 2.
pub(crate) fn message_body<'a>(
    header: &MessageHeader,
    body: Cow<'a, [u8]>,
) -> Result<MessageBody<'a>> {
    RdaStatus::decode(header.channels, &body).map(MessageBody::RdaStatus)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_number_follows_note_6() {
        assert_eq!(RdaBuild(2410).version(), 24.1);
        assert_eq!(RdaBuild(1820).version(), 18.2);
        assert_eq!(RdaBuild(100).version(), 10.0);
        assert_eq!(RdaBuild(20).version(), 2.0);
    }

    #[test]
    fn rda_build_scales_follow_table_iv_note_6() {
        assert_eq!(RdaBuild::from_raw(100).to_string(), "10.0");
        assert_eq!(RdaBuild::from_raw(1320).to_string(), "13.2");
        assert_eq!(RdaBuild::from_raw(1301).to_string(), "13.01");
        assert_eq!(RdaBuild::from_raw(2410).major(), 24);
        assert_eq!(RdaBuild::from_raw(2410).minor_hundredths(), 10);
        assert_eq!(RdaBuild::from_raw(20).to_string(), "2.0");
        assert_eq!(RdaBuild::from_raw(200).to_string(), "20.0");
        assert_eq!(RdaBuild::from_raw(201).to_string(), "2.01");
        assert_eq!(RdaBuild::from_raw(0).to_string(), "0.0");
    }

    #[test]
    fn vcp_sign_selects_local_or_remote() {
        assert_eq!(VcpSelection::from_code(212), VcpSelection::Remote(212));
        assert_eq!(VcpSelection::from_code(65456), VcpSelection::Local(80));
        assert_eq!(VcpSelection::from_code(0).pattern(), None);
        assert_eq!(VcpSelection::Local(80).signed(), -80);
    }

    #[test]
    fn alarm_halfword_splits_cleared_bit() {
        assert_eq!(RdaAlarm::from_halfword(0), None);
        assert_eq!(
            RdaAlarm::from_halfword(0x8000 | 205),
            Some(RdaAlarm {
                code: 205,
                cleared: true
            })
        );
    }

    #[test]
    fn map_generation_time_counts_from_day_one() {
        let time = MapGenerationTime {
            date: 1,
            minutes: 90,
        };
        assert_eq!(
            time.datetime().map(|t| t.to_rfc3339()),
            Some("1970-01-01T01:30:00+00:00".to_owned())
        );
        assert_eq!(MapGenerationTime::default().datetime(), None);
    }
}
