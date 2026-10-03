//! xradar's NEXRAD-specific group attributes for the FM301 view
//! (`docs/design/fm301-model.md` section 12.1, `ExtraAttrs`).
//!
//! xradar 0.12 (`nexrad_level2.py`, `get_attrs`, `_assign_sweep_attrs`)
//! writes root attributes from the Message 5 VCP definition and the Message 2
//! RDA status, and per-sweep attributes from the VCP's elevation cut of the
//! same index (cut `i` for `sweep_i`, whatever the radial's elevation number).
//! [`NexradVolume`] reproduces them in the [`Flavor::Xradar012`] flavor, with
//! xradar's names, types and spellings, so a DataTree built from the view
//! carries what `open_nexradlevel2_datatree` carries. The [`Flavor::Wmo2022`]
//! flavor has no slot for them and gets none.
//!
//! Where xradar reads a halfword by position, the same halfword is used
//! whatever the message layout: on a legacy RDA status (Message 2 of the
//! ICD 2620002B layout) halfword 10 is the interference detection rate,
//! halfword 12 the interference suppression unit and halfword 14 the Archive
//! II remaining capacity, and xradar still reports them as
//! `rda_build_number`, `super_res_status` and the AVSET/EBC flags.
//!
//! xradar's waveform table names ICD code 3 `batch` and 4
//! `staggered_pulse_pair` (Table XI E2 says 3 is contiguous Doppler without
//! ambiguity resolution, 4 batch, 5 staggered pulse pair); the flavor copies
//! xradar's table, since its purpose is to match xradar's output. The ICD
//! reading is [`crate::messages::vcp::WaveformType`].

use std::borrow::Cow;

use recast_radar_core::fm301::{ExtraAttrs, Flavor};
use recast_radar_core::model::{AttrValue, Scalar};

use crate::messages::rda_status::{EnableStatus, OperationalMode, RdaStatus};
use crate::messages::vcp::{
    ChannelConfiguration, DopplerVelocityResolution, PulseWidth, VcpCut, VolumeCoveragePattern,
};
use crate::metadata::NexradVolume;

type Attrs = Vec<(Cow<'static, str>, AttrValue)>;

fn text(value: impl Into<String>) -> AttrValue {
    AttrValue::Text(value.into().into())
}

fn boolean(value: bool) -> AttrValue {
    AttrValue::Bool(value)
}

fn int(value: i64) -> AttrValue {
    AttrValue::Scalar(Scalar::I64(value))
}

/// xradar's `_WAVEFORM_TYPES` (keyed by the Table XI E2 lower byte).
fn xradar_waveform_type(code: u16) -> String {
    match code {
        0 => "not_applicable".to_owned(),
        1 => "contiguous_surveillance".to_owned(),
        2 => "contiguous_doppler".to_owned(),
        3 => "batch".to_owned(),
        4 => "staggered_pulse_pair".to_owned(),
        other => other.to_string(),
    }
}

/// xradar's `_CHANNEL_CONFIGS`.
fn xradar_channel_config(config: ChannelConfiguration) -> String {
    match config {
        ChannelConfiguration::ConstantPhase => "constant_phase".to_owned(),
        ChannelConfiguration::RandomPhase => "random_phase".to_owned(),
        ChannelConfiguration::Sz2Phase => "sz2_phase_coding".to_owned(),
        ChannelConfiguration::Unknown(code) => code.to_string(),
    }
}

/// xradar's `_get_dynamic_scan_type`.
fn dynamic_scan_type(vcp: &VolumeCoveragePattern) -> String {
    let supplemental = vcp.supplemental;
    if supplemental.sails() {
        match supplemental.sails_cuts() {
            0 => "SAILS".to_owned(),
            n => format!("SAILS x {n}"),
        }
    } else if supplemental.mrle() {
        match supplemental.mrle_cuts() {
            0 => "MRLE".to_owned(),
            n => format!("MRLE x {n}"),
        }
    } else {
        "standard".to_owned()
    }
}

/// xradar's `_attrs_from_msg_5`, apart from `scan_name`, which the view
/// writes from `Volume::scan.name`.
fn vcp_root_attrs(vcp: &VolumeCoveragePattern) -> Attrs {
    let velocity_resolution = match vcp.doppler_velocity_resolution {
        DopplerVelocityResolution::HalfMetrePerSecond => 0.5,
        _ => 1.0,
    };
    let pulse_width = match vcp.pulse_width {
        PulseWidth::Short => "short".to_owned(),
        PulseWidth::Long => "long".to_owned(),
        PulseWidth::Unknown(code) => code.to_string(),
    };
    vec![
        ("dynamic_scan_type".into(), text(dynamic_scan_type(vcp))),
        ("mpda_vcp".into(), boolean(vcp.supplemental.mpda())),
        (
            "base_tilt_vcp".into(),
            boolean(vcp.supplemental.base_tilt()),
        ),
        (
            "num_base_tilts".into(),
            int(i64::from(vcp.supplemental.base_tilt_cuts())),
        ),
        ("vcp_truncated".into(), boolean(vcp.sequencing.truncated())),
        (
            "vcp_sequence_active".into(),
            boolean(vcp.sequencing.sequence_active()),
        ),
        (
            "number_elevation_cuts".into(),
            int(i64::from(vcp.number_of_cuts)),
        ),
        (
            "doppler_velocity_resolution".into(),
            AttrValue::Scalar(Scalar::F64(velocity_resolution)),
        ),
        ("vcp_pulse_width".into(), text(pulse_width)),
    ]
}

/// xradar's `_attrs_from_msg_2`: halfwords 10, 11, 12 and 14 of Message 2.
fn rda_status_root_attrs(status: &RdaStatus) -> Attrs {
    let (build, mode, super_res, flags) = match status {
        RdaStatus::Orda(orda) => (
            orda.rda_build.0,
            orda.operational_mode,
            orda.super_resolution,
            orda.scan_data_flags.0,
        ),
        RdaStatus::Legacy(legacy) => (
            legacy.interference_detection_rate,
            legacy.operational_mode,
            legacy.interference_suppression_unit,
            legacy.archive_ii_remaining_capacity,
        ),
    };
    let mode = match mode {
        OperationalMode::Test => 2,
        OperationalMode::Operational => 4,
        OperationalMode::Maintenance => 8,
        OperationalMode::Unknown(code) => code,
    };
    let super_res = match super_res {
        EnableStatus::Enabled => 2,
        EnableStatus::Disabled => 4,
        EnableStatus::Unknown(code) => code,
    };
    vec![
        ("avset_enabled".into(), boolean(flags & 0x0002 != 0)),
        ("ebc_enabled".into(), boolean(flags & 0x0008 != 0)),
        ("super_res_status".into(), int(i64::from(super_res))),
        ("rda_build_number".into(), int(i64::from(build))),
        ("operational_mode".into(), int(i64::from(mode))),
    ]
}

/// xradar's `_assign_sweep_attrs` for one VCP cut.
fn cut_sweep_attrs(cut: &VcpCut) -> Attrs {
    let supplemental = cut.supplemental;
    vec![
        (
            "waveform_type".into(),
            text(xradar_waveform_type(cut.waveform.code())),
        ),
        (
            "channel_config".into(),
            text(xradar_channel_config(cut.channel_configuration)),
        ),
        (
            "super_resolution".into(),
            int(i64::from(cut.super_resolution.code)),
        ),
        ("sails_cut".into(), boolean(supplemental.sails_cut())),
        (
            "sails_sequence_number".into(),
            int(i64::from(supplemental.sails_sequence_number())),
        ),
        ("mrle_cut".into(), boolean(supplemental.mrle_cut())),
        (
            "mrle_sequence_number".into(),
            int(i64::from(supplemental.mrle_sequence_number())),
        ),
        ("mpda_cut".into(), boolean(supplemental.mpda_cut())),
        (
            "base_tilt_cut".into(),
            boolean(supplemental.base_tilt_cut()),
        ),
    ]
}

impl ExtraAttrs for NexradVolume {
    fn root_attrs(&self, flavor: Flavor) -> Attrs {
        if flavor != Flavor::Xradar012 {
            return Vec::new();
        }
        let mut attrs = Vec::new();
        if let Some(vcp) = &self.metadata.vcp {
            attrs.extend(vcp_root_attrs(vcp));
        }
        if let Some(status) = &self.metadata.rda_status {
            attrs.extend(rda_status_root_attrs(status));
        }
        attrs.push((
            "actual_elevation_cuts".into(),
            int(self.volume.sweeps.len() as i64),
        ));
        attrs
    }

    fn sweep_attrs(&self, sweep: usize, flavor: Flavor) -> Attrs {
        if flavor != Flavor::Xradar012 {
            return Vec::new();
        }
        self.metadata
            .vcp
            .as_ref()
            .and_then(|vcp| vcp.cuts.get(sweep))
            .map(cut_sweep_attrs)
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn waveform_and_channel_spellings_follow_xradar() {
        assert_eq!(xradar_waveform_type(1), "contiguous_surveillance");
        assert_eq!(xradar_waveform_type(2), "contiguous_doppler");
        assert_eq!(xradar_waveform_type(3), "batch");
        assert_eq!(xradar_waveform_type(4), "staggered_pulse_pair");
        assert_eq!(xradar_waveform_type(9), "9");
        assert_eq!(
            xradar_channel_config(ChannelConfiguration::Sz2Phase),
            "sz2_phase_coding"
        );
    }
}
