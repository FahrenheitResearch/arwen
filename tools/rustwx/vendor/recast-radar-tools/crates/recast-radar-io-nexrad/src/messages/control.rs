//! RDA Control Commands (message 6, ICD 2620002AA Table X).
//!
//! Sent from the RPG to the RDA, so Archive II files do not record it; no file
//! in the test corpus holds one and this decoder follows the ICD without a
//! verified real sample. Every field keeps codes the ICD does not list as
//! `Unknown(raw)`.

use std::borrow::Cow;

use super::MessageBody;
use crate::Result;

/// Body length of Table X: 26 halfwords.
pub const CONTROL_COMMANDS_LEN: usize = 52;

/// Decoded RDA control commands (Table X). Halfwords 7, 11, 15 to 20 and 22
/// to 26 are spare.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RdaControlCommands {
    /// Halfword 1.
    pub rda_state: RdaStateCommand,
    /// Halfword 2.
    pub rda_log: EnableDisable,
    /// Halfword 3.
    pub auxiliary_power: AuxiliaryPowerCommand,
    /// Halfword 4.
    pub control_authorization: ControlAuthorization,
    /// Halfword 5.
    pub restart: RestartCommand,
    /// Halfword 6.
    pub select_local_vcp: LocalVcpSelection,
    /// Halfword 8.
    pub super_resolution: EnableDisable,
    /// Halfword 9: clutter mitigation decision (CMD).
    pub clutter_mitigation_decision: EnableDisable,
    /// Halfword 10.
    pub avset: EnableDisable,
    /// Halfword 12.
    pub channel_control: ChannelControlCommand,
    /// Halfword 13.
    pub performance_check: PerformanceCheckCommand,
    /// Halfword 14.
    pub zdr_bias_estimate: ZdrBiasEstimate,
    /// Halfword 21.
    pub spot_blanking: EnableDisable,
}

impl RdaControlCommands {
    /// Decode a message body (the bytes after the 16-byte message header).
    pub fn decode(body: &[u8]) -> Result<Self> {
        crate::require_len(body, 0, CONTROL_COMMANDS_LEN, "RDA control commands")?;
        let halfword = |number: usize| crate::be_u16(body, (number - 1) * 2);
        Ok(Self {
            rda_state: RdaStateCommand::from_code(halfword(1)),
            rda_log: EnableDisable::from_log_code(halfword(2)),
            auxiliary_power: AuxiliaryPowerCommand::from_code(halfword(3)),
            control_authorization: ControlAuthorization::from_code(halfword(4)),
            restart: RestartCommand::from_code(halfword(5)),
            select_local_vcp: LocalVcpSelection::from_code(halfword(6)),
            super_resolution: EnableDisable::from_code(halfword(8)),
            clutter_mitigation_decision: EnableDisable::from_code(halfword(9)),
            avset: EnableDisable::from_code(halfword(10)),
            channel_control: ChannelControlCommand::from_code(halfword(12)),
            performance_check: PerformanceCheckCommand::from_code(halfword(13)),
            zdr_bias_estimate: ZdrBiasEstimate::from_code(halfword(14)),
            spot_blanking: EnableDisable::from_code(halfword(21)),
        })
    }
}

/// RDA state command (halfword 1).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RdaStateCommand {
    /// 0.
    NoChange,
    /// 32769 (bits 0 and 15).
    StandBy,
    /// 32772 (bits 2 and 15).
    Operate,
    /// 32776 (bits 3 and 15).
    Restart,
    /// Any other code.
    Unknown(u16),
}

impl RdaStateCommand {
    /// Map a Table X code.
    pub fn from_code(code: u16) -> Self {
        match code {
            0 => Self::NoChange,
            32769 => Self::StandBy,
            32772 => Self::Operate,
            32776 => Self::Restart,
            other => Self::Unknown(other),
        }
    }

    /// The Table X code.
    pub fn code(self) -> u16 {
        match self {
            Self::NoChange => 0,
            Self::StandBy => 32769,
            Self::Operate => 32772,
            Self::Restart => 32776,
            Self::Unknown(code) => code,
        }
    }
}

/// No change / enable / disable command. Super resolution, CMD, AVSET and
/// spot blanking use codes 0, 2, 4; the RDA log command uses 0, 1, 2.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum EnableDisable {
    /// 0.
    NoChange,
    /// 2 (bit 1), or 1 (bit 0) for the RDA log command.
    Enable,
    /// 4 (bit 2), or 2 (bit 1) for the RDA log command.
    Disable,
    /// Any other code.
    Unknown(u16),
}

impl EnableDisable {
    /// Map the 0/2/4 codes of halfwords 8, 9, 10 and 21.
    pub fn from_code(code: u16) -> Self {
        match code {
            0 => Self::NoChange,
            2 => Self::Enable,
            4 => Self::Disable,
            other => Self::Unknown(other),
        }
    }

    /// Map the 0/1/2 codes of the RDA log command (halfword 2).
    pub fn from_log_code(code: u16) -> Self {
        match code {
            0 => Self::NoChange,
            1 => Self::Enable,
            2 => Self::Disable,
            other => Self::Unknown(other),
        }
    }

    /// The 0/2/4 code of halfwords 8, 9, 10 and 21 ([`Self::from_code`]).
    pub fn code(self) -> u16 {
        match self {
            Self::NoChange => 0,
            Self::Enable => 2,
            Self::Disable => 4,
            Self::Unknown(code) => code,
        }
    }

    /// The 0/1/2 code of the RDA log command ([`Self::from_log_code`]).
    pub fn log_code(self) -> u16 {
        match self {
            Self::NoChange => 0,
            Self::Enable => 1,
            Self::Disable => 2,
            Self::Unknown(code) => code,
        }
    }
}

/// Auxiliary power generator control (halfword 3).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum AuxiliaryPowerCommand {
    /// 0.
    NoChange,
    /// 32772 (bits 2 and 15).
    SwitchToAuxiliary,
    /// 32770 (bits 1 and 15).
    SwitchToUtility,
    /// Any other code.
    Unknown(u16),
}

impl AuxiliaryPowerCommand {
    /// Map a Table X code.
    pub fn from_code(code: u16) -> Self {
        match code {
            0 => Self::NoChange,
            32772 => Self::SwitchToAuxiliary,
            32770 => Self::SwitchToUtility,
            other => Self::Unknown(other),
        }
    }

    /// The Table X code.
    pub fn code(self) -> u16 {
        match self {
            Self::NoChange => 0,
            Self::SwitchToAuxiliary => 32772,
            Self::SwitchToUtility => 32770,
            Self::Unknown(code) => code,
        }
    }
}

/// RDA control commands and authorization (halfword 4).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ControlAuthorization {
    /// 0.
    NoChange,
    /// 2 (bit 1).
    ControlCommandClear,
    /// 4 (bit 2).
    LocalControlEnabled,
    /// 8 (bit 3).
    RemoteControlAccepted,
    /// 16 (bit 4).
    RemoteControlRequested,
    /// Any other code.
    Unknown(u16),
}

impl ControlAuthorization {
    /// Map a Table X code.
    pub fn from_code(code: u16) -> Self {
        match code {
            0 => Self::NoChange,
            2 => Self::ControlCommandClear,
            4 => Self::LocalControlEnabled,
            8 => Self::RemoteControlAccepted,
            16 => Self::RemoteControlRequested,
            other => Self::Unknown(other),
        }
    }

    /// The Table X code.
    pub fn code(self) -> u16 {
        match self {
            Self::NoChange => 0,
            Self::ControlCommandClear => 2,
            Self::LocalControlEnabled => 4,
            Self::RemoteControlAccepted => 8,
            Self::RemoteControlRequested => 16,
            Self::Unknown(code) => code,
        }
    }
}

/// Restart VCP or elevation cut (halfword 5).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RestartCommand {
    /// 0.
    None,
    /// 32768 (bit 15).
    RestartVcp,
    /// 32768 plus the cut number in bits 0 to 7.
    RestartElevationCut(u8),
    /// Any other code.
    Unknown(u16),
}

impl RestartCommand {
    /// Map a Table X code.
    pub fn from_code(code: u16) -> Self {
        match code {
            0 => Self::None,
            32768 => Self::RestartVcp,
            32769..=33023 => Self::RestartElevationCut((code & 0x00ff) as u8),
            other => Self::Unknown(other),
        }
    }

    /// The Table X code.
    pub fn code(self) -> u16 {
        match self {
            Self::None => 0,
            Self::RestartVcp => 32768,
            Self::RestartElevationCut(cut) => 32768 | u16::from(cut),
            Self::Unknown(code) => code,
        }
    }
}

/// Select local VCP number for the next volume scan (halfword 6).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum LocalVcpSelection {
    /// 0: use the remote pattern.
    UseRemotePattern,
    /// 1 to 767.
    Pattern(u16),
    /// 32767.
    NoChange,
    /// Any other code.
    Unknown(u16),
}

impl LocalVcpSelection {
    /// Map a Table X code.
    pub fn from_code(code: u16) -> Self {
        match code {
            0 => Self::UseRemotePattern,
            1..=767 => Self::Pattern(code),
            32767 => Self::NoChange,
            other => Self::Unknown(other),
        }
    }

    /// The Table X code.
    pub fn code(self) -> u16 {
        match self {
            Self::UseRemotePattern => 0,
            Self::Pattern(code) | Self::Unknown(code) => code,
            Self::NoChange => 32767,
        }
    }
}

/// Channel control command (halfword 12).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ChannelControlCommand {
    /// 0.
    NoChange,
    /// 1 (bit 0).
    SetControlling,
    /// 2 (bit 1).
    SetNonControlling,
    /// Any other code.
    Unknown(u16),
}

impl ChannelControlCommand {
    /// Map a Table X code.
    pub fn from_code(code: u16) -> Self {
        match code {
            0 => Self::NoChange,
            1 => Self::SetControlling,
            2 => Self::SetNonControlling,
            other => Self::Unknown(other),
        }
    }

    /// The Table X code.
    pub fn code(self) -> u16 {
        match self {
            Self::NoChange => 0,
            Self::SetControlling => 1,
            Self::SetNonControlling => 2,
            Self::Unknown(code) => code,
        }
    }
}

/// Performance check control (halfword 13).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum PerformanceCheckCommand {
    /// 0.
    NoChange,
    /// 1 (bit 0): force a performance check at the end of the current VCP.
    ForcePerformanceCheck,
    /// Any other code.
    Unknown(u16),
}

impl PerformanceCheckCommand {
    /// Map a Table X code.
    pub fn from_code(code: u16) -> Self {
        match code {
            0 => Self::NoChange,
            1 => Self::ForcePerformanceCheck,
            other => Self::Unknown(other),
        }
    }

    /// The Table X code.
    pub fn code(self) -> u16 {
        match self {
            Self::NoChange => 0,
            Self::ForcePerformanceCheck => 1,
            Self::Unknown(code) => code,
        }
    }
}

/// ZDR bias estimate weighted mean (halfword 14, Table X note 8).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ZdrBiasEstimate {
    /// 0.
    NotAvailable,
    /// 1.
    NoChange,
    /// 2 to 1058, encoded like the 16-bit "ZDR" moment of Table XVII-I.
    Coded(u16),
    /// Any other code.
    Unknown(u16),
}

impl ZdrBiasEstimate {
    /// Scale of the 16-bit "ZDR" moment encoding in Table XVII-I.
    pub const SCALE: f32 = 32.0;
    /// Offset of the 16-bit "ZDR" moment encoding in Table XVII-I.
    pub const OFFSET: f32 = 418.0;

    /// Map a Table X code.
    pub fn from_code(code: u16) -> Self {
        match code {
            0 => Self::NotAvailable,
            1 => Self::NoChange,
            2..=1058 => Self::Coded(code),
            other => Self::Unknown(other),
        }
    }

    /// The Table X code.
    pub fn code(self) -> u16 {
        match self {
            Self::NotAvailable => 0,
            Self::NoChange => 1,
            Self::Coded(code) | Self::Unknown(code) => code,
        }
    }

    /// Bias in dB for a coded value: `(code - 418) / 32`, spanning -13 to
    /// +20 dB.
    pub fn db(self) -> Option<f32> {
        match self {
            Self::Coded(code) => Some((f32::from(code) - Self::OFFSET) / Self::SCALE),
            _ => None,
        }
    }
}

/// Walker hook: the typed body for message 6.
pub(crate) fn message_body(body: Cow<'_, [u8]>) -> Result<MessageBody<'_>> {
    RdaControlCommands::decode(&body).map(MessageBody::ControlCommands)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zdr_bias_code_range_matches_table_xvii_i_span() {
        assert_eq!(ZdrBiasEstimate::from_code(2).db(), Some(-13.0));
        assert_eq!(ZdrBiasEstimate::from_code(1058).db(), Some(20.0));
        assert_eq!(ZdrBiasEstimate::from_code(418).db(), Some(0.0));
        assert_eq!(ZdrBiasEstimate::from_code(1).db(), None);
    }

    /// Every halfword value maps back to itself through the typed command.
    #[test]
    fn command_codes_round_trip() {
        for code in 0..=u16::MAX {
            assert_eq!(RdaStateCommand::from_code(code).code(), code);
            assert_eq!(EnableDisable::from_code(code).code(), code);
            assert_eq!(EnableDisable::from_log_code(code).log_code(), code);
            assert_eq!(AuxiliaryPowerCommand::from_code(code).code(), code);
            assert_eq!(ControlAuthorization::from_code(code).code(), code);
            assert_eq!(RestartCommand::from_code(code).code(), code);
            assert_eq!(LocalVcpSelection::from_code(code).code(), code);
            assert_eq!(ChannelControlCommand::from_code(code).code(), code);
            assert_eq!(PerformanceCheckCommand::from_code(code).code(), code);
            assert_eq!(ZdrBiasEstimate::from_code(code).code(), code);
        }
    }

    #[test]
    fn restart_code_carries_cut_number_in_low_byte() {
        assert_eq!(RestartCommand::from_code(32768), RestartCommand::RestartVcp);
        assert_eq!(
            RestartCommand::from_code(32768 + 5),
            RestartCommand::RestartElevationCut(5)
        );
        assert_eq!(RestartCommand::from_code(1), RestartCommand::Unknown(1));
    }
}
