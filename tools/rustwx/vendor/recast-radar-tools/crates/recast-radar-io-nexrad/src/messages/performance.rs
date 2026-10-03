//! Performance/Maintenance Data (message 3, ICD 2620002AA Table V).
//!
//! The RDA sends this 480-halfword message at wideband connection, at the
//! start of every volume scan and on request. [`PerformanceMaintenance`]
//! follows the Build 24.0 table: one struct per section of Table V, each
//! field documented with its halfword location, units and ICD range, and
//! status codes kept as the numbers the ICD lists. Spare halfwords are not
//! kept.
//!
//! # Earlier builds
//!
//! Every Open RDA build since 10.0 sends 480 halfwords, but Table V has
//! reassigned locations over time; a file written by an older build carries
//! the older content in these fields (checked against ICD revisions F, J, M,
//! N, P, R, T, U, V, W and Y):
//!
//! - Build 17.0 (2620002P) replaced the DAU with the SPIP. Before it,
//!   halfwords 55-57 held the DAU A/D tests, 99-110 the UPS status and
//!   readings, 202 the transmitter/DAU interface, 215 the transmitter power
//!   meter zero (an Integer*2), 239 the DAU UART status, 275-282 the DAU
//!   +15 V, -15 V, +28 V and +5 V supplies (all Real*4), 291-300 the pedestal
//!   supply voltages (Real*4), 308 and 321 the elevation and azimuth PCU
//!   parity, 331-333 the self tests, 415-418 the transmit burst power and
//!   phase, and 461 and 463 the DAU and pedestal communication status.
//!   Halfwords 223-224 (power meter zero, Real*4) and 468 (interpanel link)
//!   were spare.
//! - Build 18.0 (2620002R) added halfword 12 (route to RPG) and 444 (PRF set
//!   read status); through Build 17.0 halfwords 45-52 held NTP rejected
//!   packets, NTP estimated time error, GPS satellites and GPS maximum signal
//!   strength (Integer*4 each).
//! - Build 19.0 (2620002T) added halfwords 45-46 (IFDR temperatures), 448-450
//!   (RSP status) and 480 (version). Builds 19 and 20 used byte 1 of
//!   halfword 448 for the RSP motherboard temperature and halfwords 450-454
//!   for fan speeds; Build 22.0 (2620002W) put the motherboard power in 450.
//!   Through Build 19.0, halfwords 13-20 held the CSU loss of signal, loss of
//!   frames, yellow alarm and blue alarm counts (Integer*4 each).
//! - Build 20.0 (2620002U) added halfword 47 (NTP status); Build 23.0
//!   (2620002Y) added 13-15 (T1 and router Ethernet port status).
//! - Build 13.0 (2620002M) added the AME section (58-98) and the vertical
//!   channel readings (211-212, 357-362, 387-392, 397-402, 425-426; 357-358,
//!   the vertical short pulse noise, is marked spare from Build 14.0 to 18.0
//!   and defined again from 19.0); Build 14.0
//!   (2620002N) added 113-114 (expansion power administrator load) and
//!   219-222 (receiver bias, transmit imbalance) and replaced the LAN switch
//!   memory used and free counts (39-42) with CPU utilization (41-42).
//!
//! Messages from a legacy (pre-ORDA) RDA use the unrelated 520-halfword
//! layout of ICD 2620002B Table V, with non-IEEE floating point; the walker
//! yields those bodies unparsed.

use std::borrow::Cow;

use super::MessageBody;
use super::rda_status::RdaSystem;
use crate::{MessageHeader, Result};

/// Body length of Table V: 480 halfwords.
pub const PERFORMANCE_MAINTENANCE_LEN: usize = 960;

/// Decoded Performance/Maintenance Data (Table V, Build 24.0).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PerformanceMaintenance {
    /// Halfwords 1 to 57.
    pub communications: Communications,
    /// Halfwords 58 to 98.
    pub ame: Ame,
    /// Halfwords 99 to 110.
    pub rcp_spip: RcpSpip,
    /// Halfwords 111 to 136.
    pub power: Power,
    /// Halfwords 137 to 228.
    pub transmitter: Transmitter,
    /// Halfwords 229 to 249.
    pub tower_utilities: TowerUtilities,
    /// Halfwords 250 to 299.
    pub equipment_shelter: EquipmentShelter,
    /// Halfwords 300 to 340.
    pub antenna_pedestal: AntennaPedestal,
    /// Halfwords 341 to 362.
    pub rf_generator_receiver: RfGeneratorReceiver,
    /// Halfwords 363 to 430.
    pub calibration: Calibration,
    /// Halfwords 431 to 460.
    pub file_status: FileStatus,
    /// Halfwords 461 to 479.
    pub device_status: DeviceStatus,
    /// Version number of the performance data message (halfword 480), expected to change with any
    /// other change to the message. Zero or unrelated data before Build 19.0.
    pub version: u16,
}

impl PerformanceMaintenance {
    /// Decode a message body (the bytes after the 16-byte message header) in the Build 24.0 layout.
    /// Needs at least 480 halfwords.
    pub fn decode(body: &[u8]) -> Result<Self> {
        crate::require_len(
            body,
            0,
            PERFORMANCE_MAINTENANCE_LEN,
            "performance/maintenance data",
        )?;
        Ok(Self {
            communications: Communications::decode(body),
            ame: Ame::decode(body),
            rcp_spip: RcpSpip::decode(body),
            power: Power::decode(body),
            transmitter: Transmitter::decode(body),
            tower_utilities: TowerUtilities::decode(body),
            equipment_shelter: EquipmentShelter::decode(body),
            antenna_pedestal: AntennaPedestal::decode(body),
            rf_generator_receiver: RfGeneratorReceiver::decode(body),
            calibration: Calibration::decode(body),
            file_status: FileStatus::decode(body),
            device_status: DeviceStatus::decode(body),
            version: u16_at(body, 480),
        })
    }
}

/// Byte offset of a 1-based halfword.
fn offset(halfword: usize) -> usize {
    (halfword - 1) * 2
}

fn u16_at(body: &[u8], halfword: usize) -> u16 {
    crate::be_u16(body, offset(halfword))
}

fn u32_at(body: &[u8], halfword: usize) -> u32 {
    crate::be_u32(body, offset(halfword))
}

fn f32_at(body: &[u8], halfword: usize) -> f32 {
    crate::be_f32(body, offset(halfword))
}

fn halfwords(body: &[u8], first: usize, count: usize) -> &[u8] {
    &body[offset(first)..offset(first + count)]
}

/// Communications status (halfwords 1 to 57).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Communications {
    /// Loop back test status (halfword 2): 0 = pass, 1 = fail, 2 = timeout, 3 = not tested (not
    /// connected or not configured).
    pub loop_back_test_status: u16,
    /// T1 output frames (halfwords 3-4): octets received on the interface, including frame octets.
    pub t1_output_frames: u32,
    /// T1 input frames (halfwords 5-6): octets sent on the interface, including frame octets.
    pub t1_input_frames: u32,
    /// Router memory used by applications (halfwords 7-8), bytes.
    pub router_memory_used: u32,
    /// Router memory free (halfwords 9-10), bytes.
    pub router_memory_free: u32,
    /// Router memory utilization (halfword 11), % (0 to 100).
    pub router_memory_utilization: u16,
    /// Route to RPG (halfword 12), the status of the backup communications route: 0 = normal,
    /// 1 = backup in use, 2 = backup down failure, 3 = backup commanded down, 4 = backup not
    /// installed.
    pub route_to_rpg: u16,
    /// T1 port status (halfword 13): 1 = up, 2 = down, 3 = test; other values unknown.
    pub t1_port_status: u16,
    /// Router dedicated (local) Ethernet port to the RPG (halfword 14): 1 = up, 2 = down, 3 = test.
    pub router_dedicated_ethernet_port_status: u16,
    /// Router commercial Ethernet port to the RPG (halfword 15): 1 = up, 2 = down, 3 = test.
    pub router_commercial_ethernet_port_status: u16,
    /// CSU errored seconds in the previous 24 hours (halfwords 21-22), s (updated every 15
    /// minutes).
    pub csu_24hr_errored_seconds: u32,
    /// CSU severely errored seconds in the previous 24 hours (halfwords 23-24), s.
    pub csu_24hr_severely_errored_seconds: u32,
    /// CSU severely errored framing seconds in the previous 24 hours (halfwords 25-26), s.
    pub csu_24hr_severely_errored_framing_seconds: u32,
    /// CSU unavailable seconds in the previous 24 hours (halfwords 27-28), s.
    pub csu_24hr_unavailable_seconds: u32,
    /// CSU controlled slip seconds in the previous 24 hours (halfwords 29-30), s.
    pub csu_24hr_controlled_slip_seconds: u32,
    /// CSU path coding violations in the previous 24 hours (halfwords 31-32), count.
    pub csu_24hr_path_coding_violations: u32,
    /// CSU line errored seconds in the previous 24 hours (halfwords 33-34), s.
    pub csu_24hr_line_errored_seconds: u32,
    /// CSU bursty errored seconds in the previous 24 hours (halfwords 35-36), s.
    pub csu_24hr_bursty_errored_seconds: u32,
    /// CSU degraded minutes in the previous 24 hours (halfwords 37-38), min.
    pub csu_24hr_degraded_minutes: u32,
    /// LAN switch CPU utilization (halfwords 41-42), % (0 to 100).
    pub lan_switch_cpu_utilization: u32,
    /// LAN switch memory utilization (halfword 43), % (0 to 100).
    pub lan_switch_memory_utilization: u16,
    /// IFDR chassis (case) temperature (halfword 45), deg C (-30 to 150).
    pub ifdr_chassis_temperature: i16,
    /// IFDR FPGA temperature (halfword 46), deg C (-30 to 150).
    pub ifdr_fpga_temperature: i16,
    /// NTP synchronization status (halfword 47): 0 = OK, 1 = fail.
    pub ntp_status: u16,
    /// Status of the communications between the channels of a redundant system (halfword 53):
    /// 0 = OK, 1 = fail, 2 = N/A (single channel).
    pub ipc_status: u16,
    /// Channel the RDA has commanded to be the controlling channel (not necessarily the one in
    /// control) (halfword 54): 0 = N/A, 1 = channel 1, 2 = channel 2.
    pub commanded_channel_control: u16,
}

impl Communications {
    fn decode(body: &[u8]) -> Self {
        Self {
            loop_back_test_status: u16_at(body, 2),
            t1_output_frames: u32_at(body, 3),
            t1_input_frames: u32_at(body, 5),
            router_memory_used: u32_at(body, 7),
            router_memory_free: u32_at(body, 9),
            router_memory_utilization: u16_at(body, 11),
            route_to_rpg: u16_at(body, 12),
            t1_port_status: u16_at(body, 13),
            router_dedicated_ethernet_port_status: u16_at(body, 14),
            router_commercial_ethernet_port_status: u16_at(body, 15),
            csu_24hr_errored_seconds: u32_at(body, 21),
            csu_24hr_severely_errored_seconds: u32_at(body, 23),
            csu_24hr_severely_errored_framing_seconds: u32_at(body, 25),
            csu_24hr_unavailable_seconds: u32_at(body, 27),
            csu_24hr_controlled_slip_seconds: u32_at(body, 29),
            csu_24hr_path_coding_violations: u32_at(body, 31),
            csu_24hr_line_errored_seconds: u32_at(body, 33),
            csu_24hr_bursty_errored_seconds: u32_at(body, 35),
            csu_24hr_degraded_minutes: u32_at(body, 37),
            lan_switch_cpu_utilization: u32_at(body, 41),
            lan_switch_memory_utilization: u16_at(body, 43),
            ifdr_chassis_temperature: u16_at(body, 45) as i16,
            ifdr_fpga_temperature: u16_at(body, 46) as i16,
            ntp_status: u16_at(body, 47),
            ipc_status: u16_at(body, 53),
            commanded_channel_control: u16_at(body, 54),
        }
    }
}

/// Antenna mounted electronics (AME) status (halfwords 58 to 98).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Ame {
    /// AME polarization (halfword 58): 0 = H only, 1 = H + V, 2 = V only.
    pub polarization: u16,
    /// AME internal temperature (halfwords 59-60), deg C (-40.0 to +125.0).
    pub internal_temperature: f32,
    /// AME receiver module temperature (halfwords 61-62), deg C (-40.0 to +125.0).
    pub receiver_module_temperature: f32,
    /// AME BITE/CAL module temperature (halfwords 63-64), deg C (-40.0 to +125.0).
    pub bite_cal_module_temperature: f32,
    /// AME Peltier pulse width modulation (halfword 65), % (0 to 100).
    pub peltier_pulse_width_modulation: u16,
    /// AME Peltier status (halfword 66): 0 = off, 1 = on.
    pub peltier_status: u16,
    /// AME A/D converter status (halfword 67): 0 = OK, 1 = fail.
    pub ad_converter_status: u16,
    /// AME state (halfword 68): 0 = start, 1 = running, 2 = flash, 3 = error.
    pub state: u16,
    /// AME +3.3 V power supply voltage (halfwords 69-70), V (0.00 to 4.09).
    pub ps_3_3v_voltage: f32,
    /// AME +5 V power supply voltage (halfwords 71-72), V (0.00 to 6.10).
    pub ps_5v_voltage: f32,
    /// AME +6.5 V power supply voltage (halfwords 73-74), V (0.00 to 7.50).
    pub ps_6_5v_voltage: f32,
    /// AME +15 V power supply voltage (halfwords 75-76), V (0.00 to 19.00).
    pub ps_15v_voltage: f32,
    /// AME +48 V power supply voltage (halfwords 77-78), V (0.00 to 60.00).
    pub ps_48v_voltage: f32,
    /// AME STALO power (halfwords 79-80), V (0.00 to 4.09).
    pub stalo_power: f32,
    /// Peltier current (halfwords 81-82), A (0.00 to 16.00).
    pub peltier_current: f32,
    /// ADC calibration reference voltage (halfwords 83-84), V (0.000 to 2.048).
    pub adc_calibration_reference_voltage: f32,
    /// AME mode (halfword 85): 0 = ready, 1 = maintenance.
    pub mode: u16,
    /// AME Peltier mode (halfword 86): 0 = cool, 1 = heat.
    pub peltier_mode: u16,
    /// AME Peltier inside fan current (halfwords 87-88), A (0.00 to 4.00).
    pub peltier_inside_fan_current: f32,
    /// AME Peltier outside fan current (halfwords 89-90), A (0.00 to 4.00).
    pub peltier_outside_fan_current: f32,
    /// Horizontal TR limiter voltage (halfwords 91-92), V (0.00 to 5.00).
    pub horizontal_tr_limiter_voltage: f32,
    /// Vertical TR limiter voltage (halfwords 93-94), V (0.00 to 5.00).
    pub vertical_tr_limiter_voltage: f32,
    /// ADC calibration offset voltage (halfwords 95-96), mV (-50.000 to +50.000).
    pub adc_calibration_offset_voltage: f32,
    /// ADC calibration gain correction (halfwords 97-98), unitless (0.990 to 1.010).
    pub adc_calibration_gain_correction: f32,
}

impl Ame {
    fn decode(body: &[u8]) -> Self {
        Self {
            polarization: u16_at(body, 58),
            internal_temperature: f32_at(body, 59),
            receiver_module_temperature: f32_at(body, 61),
            bite_cal_module_temperature: f32_at(body, 63),
            peltier_pulse_width_modulation: u16_at(body, 65),
            peltier_status: u16_at(body, 66),
            ad_converter_status: u16_at(body, 67),
            state: u16_at(body, 68),
            ps_3_3v_voltage: f32_at(body, 69),
            ps_5v_voltage: f32_at(body, 71),
            ps_6_5v_voltage: f32_at(body, 73),
            ps_15v_voltage: f32_at(body, 75),
            ps_48v_voltage: f32_at(body, 77),
            stalo_power: f32_at(body, 79),
            peltier_current: f32_at(body, 81),
            adc_calibration_reference_voltage: f32_at(body, 83),
            mode: u16_at(body, 85),
            peltier_mode: u16_at(body, 86),
            peltier_inside_fan_current: f32_at(body, 87),
            peltier_outside_fan_current: f32_at(body, 89),
            horizontal_tr_limiter_voltage: f32_at(body, 91),
            vertical_tr_limiter_voltage: f32_at(body, 93),
            adc_calibration_offset_voltage: f32_at(body, 95),
            adc_calibration_gain_correction: f32_at(body, 97),
        }
    }
}

/// Radar control program and SPIP power button status (halfwords 99 to 110).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RcpSpip {
    /// Status of the third-party radar control program (RCP) (halfword 99): 0 = OK, 1 = not OK.
    pub rcp_status: u16,
    /// Descriptive string for the radar control program state (halfwords 100-107), NUL padding
    /// removed.
    pub rcp_string: String,
    /// State of the SPIP power buttons (bit field) (halfword 108): bit 0 = this channel's DAQ power
    /// button is off, bit 1 = this channel's DAQ PED power button is off, bit 2 = channel 2 DAQ
    /// power button is off (channel 1 only), bit 3 = channel 2 DAQ PED power button is off (channel
    /// 1 only), bit 4 = this is channel 1 of a redundant configuration.
    pub spip_power_buttons: u16,
}

impl RcpSpip {
    fn decode(body: &[u8]) -> Self {
        Self {
            rcp_status: u16_at(body, 99),
            rcp_string: crate::ascii_trim(halfwords(body, 100, 8)),
            spip_power_buttons: u16_at(body, 108),
        }
    }
}

/// Power administrator loads (halfwords 111 to 136).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Power {
    /// Master power administrator load (halfwords 111-112), A (0.00 to 12.00).
    pub master_power_administrator_load: f32,
    /// Expansion power administrator load (halfwords 113-114), A (0.00 to 12.00).
    pub expansion_power_administrator_load: f32,
}

impl Power {
    fn decode(body: &[u8]) -> Self {
        Self {
            master_power_administrator_load: f32_at(body, 111),
            expansion_power_administrator_load: f32_at(body, 113),
        }
    }
}

/// Transmitter status (halfwords 137 to 228).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Transmitter {
    /// +5 VDC power supply (halfword 137): 0 = OK, 1 = fail.
    pub ps_5vdc: u16,
    /// +15 VDC power supply (halfword 138): 0 = OK, 1 = fail.
    pub ps_15vdc: u16,
    /// +28 VDC power supply (halfword 139): 0 = OK, 1 = fail.
    pub ps_28vdc: u16,
    /// -15 VDC power supply (halfword 140): 0 = OK, 1 = fail.
    pub ps_neg_15vdc: u16,
    /// +45 VDC power supply (halfword 141): 0 = OK, 1 = fail.
    pub ps_45vdc: u16,
    /// Filament power supply voltage (halfword 142): 0 = OK, 1 = fail.
    pub filament_ps_voltage: u16,
    /// Vacuum pump power supply voltage (halfword 143): 0 = OK, 1 = fail.
    pub vacuum_pump_ps_voltage: u16,
    /// Focus coil power supply voltage (halfword 144): 0 = OK, 1 = fail.
    pub focus_coil_ps_voltage: u16,
    /// Filament power supply (halfword 145): 0 = on, 1 = off.
    pub filament_ps: u16,
    /// Klystron warmup (halfword 146): 0 = normal, 1 = preheat.
    pub klystron_warmup: u16,
    /// Transmitter available (halfword 147): 0 = yes, 1 = no.
    pub transmitter_available: u16,
    /// Waveguide switch position (halfword 148): 0 = antenna, 1 = dummy load.
    pub wg_switch_position: u16,
    /// Waveguide/PFN transfer interlock (halfword 149): 0 = OK, 1 = open.
    pub wg_pfn_transfer_interlock: u16,
    /// Maintenance mode (halfword 150): 0 = no, 1 = yes.
    pub maintenance_mode: u16,
    /// Maintenance required (halfword 151): 0 = no, 1 = required.
    pub maintenance_required: u16,
    /// PFN switch position (halfword 152): 0 = short pulse, 1 = long pulse.
    pub pfn_switch_position: u16,
    /// Modulator overload (halfword 153): 0 = OK, 1 = fail.
    pub modulator_overload: u16,
    /// Modulator inverse current (halfword 154): 0 = OK, 1 = fail.
    pub modulator_inv_current: u16,
    /// Modulator switch fail (halfword 155): 0 = OK, 1 = fail.
    pub modulator_switch_fail: u16,
    /// Main power voltage (halfword 156): 0 = OK, 1 = over.
    pub main_power_voltage: u16,
    /// Charging system fail (halfword 157): 0 = OK, 1 = fail.
    pub charging_system_fail: u16,
    /// Inverse diode current (halfword 158): 0 = OK, 1 = fail.
    pub inverse_diode_current: u16,
    /// Trigger amplifier (halfword 159): 0 = OK, 1 = fail.
    pub trigger_amplifier: u16,
    /// Circulator temperature (halfword 160): 0 = OK, 1 = fail.
    pub circulator_temperature: u16,
    /// Spectrum filter pressure (halfword 161): 0 = OK, 1 = fail.
    pub spectrum_filter_pressure: u16,
    /// Waveguide arc/VSWR (halfword 162): 0 = OK, 1 = fail.
    pub wg_arc_vswr: u16,
    /// Cabinet interlock (halfword 163): 0 = OK, 1 = open.
    pub cabinet_interlock: u16,
    /// Cabinet air temperature (halfword 164): 0 = OK, 1 = fail.
    pub cabinet_air_temperature: u16,
    /// Cabinet airflow (halfword 165): 0 = OK, 1 = fail.
    pub cabinet_airflow: u16,
    /// Klystron current (halfword 166): 0 = OK, 1 = fail.
    pub klystron_current: u16,
    /// Klystron filament current (halfword 167): 0 = OK, 1 = fail.
    pub klystron_filament_current: u16,
    /// Klystron VacIon current (halfword 168): 0 = OK, 1 = fail.
    pub klystron_vacion_current: u16,
    /// Klystron air temperature (halfword 169): 0 = OK, 1 = fail.
    pub klystron_air_temperature: u16,
    /// Klystron airflow (halfword 170): 0 = OK, 1 = fail.
    pub klystron_airflow: u16,
    /// Modulator switch maintenance (halfword 171): 0 = OK, 1 = required.
    pub modulator_switch_maintenance: u16,
    /// Post charge regulator maintenance (halfword 172): 0 = OK, 1 = maintenance.
    pub post_charge_regulator_maintenance: u16,
    /// Waveguide pressure/humidity (halfword 173): 0 = OK, 1 = fail.
    pub wg_pressure_humidity: u16,
    /// Transmitter overvoltage (halfword 174): 0 = OK, 1 = over.
    pub transmitter_overvoltage: u16,
    /// Transmitter overcurrent (halfword 175): 0 = OK, 1 = over.
    pub transmitter_overcurrent: u16,
    /// Focus coil current (halfword 176): 0 = OK, 1 = fail.
    pub focus_coil_current: u16,
    /// Focus coil airflow (halfword 177): 0 = OK, 1 = fail.
    pub focus_coil_airflow: u16,
    /// Oil temperature (halfword 178): 0 = OK, 1 = fail.
    pub oil_temperature: u16,
    /// PRF limit (halfword 179): 0 = OK, 1 = fail.
    pub prf_limit: u16,
    /// Transmitter oil level (halfword 180): 0 = OK, 1 = fail.
    pub transmitter_oil_level: u16,
    /// Transmitter battery charging (halfword 181): 0 = yes, 1 = no.
    pub transmitter_battery_charging: u16,
    /// High voltage (HV) status (halfword 182): 0 = on, 1 = off.
    pub high_voltage_status: u16,
    /// Transmitter recycling summary (halfword 183): 0 = normal, 1 = recycling.
    pub transmitter_recycling_summary: u16,
    /// Transmitter inoperable (halfword 184): 0 = OK, 1 = inoperable.
    pub transmitter_inoperable: u16,
    /// Transmitter air filter (halfword 185): 0 = dirty, 1 = OK.
    pub transmitter_air_filter: u16,
    /// Zero test bits 0 to 7 (halfwords 186-193): 0 = OK, 1 = fail.
    pub zero_test_bits: [u16; 8],
    /// One test bits 0 to 7 (halfwords 194-201): 0 = fail, 1 = OK.
    pub one_test_bits: [u16; 8],
    /// Transmitter/SPIP interface (halfword 202): 0 = fail, 1 = OK.
    pub xmtr_spip_interface: u16,
    /// Transmitter summary status (halfword 203): 0 = ready, 1 = alarm, 2 = maintenance,
    /// 3 = recycle, 4 = preheat.
    pub transmitter_summary_status: u16,
    /// Transmitter RF power (sensor) (halfwords 205-206), mW (0.0000 to 10.0000).
    pub transmitter_rf_power: f32,
    /// Horizontal transmitter peak power (halfwords 207-208), kW (0 to 999.9).
    pub horizontal_xmtr_peak_power: f32,
    /// Transmitter peak power (halfwords 209-210), kW (0 to 999.9).
    pub xmtr_peak_power: f32,
    /// Vertical transmitter peak power (halfwords 211-212), kW (0 to 999.9).
    pub vertical_xmtr_peak_power: f32,
    /// Transmitter RF average power (halfwords 213-214), W (0 to 9999.9).
    pub xmtr_rf_avg_power: f32,
    /// Transmitter recycle count (0 to 999,999) (halfwords 217-218).
    pub xmtr_recycle_count: u32,
    /// Receiver bias (measurement) (halfwords 219-220), dB (-999.9999 to 999.9999).
    pub receiver_bias: f32,
    /// Transmit imbalance (halfwords 221-222), dB (-999.9999 to 999.99).
    pub transmit_imbalance: f32,
    /// Transmitter power meter zero (halfwords 223-224), V (0.01 to 8.00).
    pub xmtr_power_meter_zero: f32,
}

impl Transmitter {
    fn decode(body: &[u8]) -> Self {
        Self {
            ps_5vdc: u16_at(body, 137),
            ps_15vdc: u16_at(body, 138),
            ps_28vdc: u16_at(body, 139),
            ps_neg_15vdc: u16_at(body, 140),
            ps_45vdc: u16_at(body, 141),
            filament_ps_voltage: u16_at(body, 142),
            vacuum_pump_ps_voltage: u16_at(body, 143),
            focus_coil_ps_voltage: u16_at(body, 144),
            filament_ps: u16_at(body, 145),
            klystron_warmup: u16_at(body, 146),
            transmitter_available: u16_at(body, 147),
            wg_switch_position: u16_at(body, 148),
            wg_pfn_transfer_interlock: u16_at(body, 149),
            maintenance_mode: u16_at(body, 150),
            maintenance_required: u16_at(body, 151),
            pfn_switch_position: u16_at(body, 152),
            modulator_overload: u16_at(body, 153),
            modulator_inv_current: u16_at(body, 154),
            modulator_switch_fail: u16_at(body, 155),
            main_power_voltage: u16_at(body, 156),
            charging_system_fail: u16_at(body, 157),
            inverse_diode_current: u16_at(body, 158),
            trigger_amplifier: u16_at(body, 159),
            circulator_temperature: u16_at(body, 160),
            spectrum_filter_pressure: u16_at(body, 161),
            wg_arc_vswr: u16_at(body, 162),
            cabinet_interlock: u16_at(body, 163),
            cabinet_air_temperature: u16_at(body, 164),
            cabinet_airflow: u16_at(body, 165),
            klystron_current: u16_at(body, 166),
            klystron_filament_current: u16_at(body, 167),
            klystron_vacion_current: u16_at(body, 168),
            klystron_air_temperature: u16_at(body, 169),
            klystron_airflow: u16_at(body, 170),
            modulator_switch_maintenance: u16_at(body, 171),
            post_charge_regulator_maintenance: u16_at(body, 172),
            wg_pressure_humidity: u16_at(body, 173),
            transmitter_overvoltage: u16_at(body, 174),
            transmitter_overcurrent: u16_at(body, 175),
            focus_coil_current: u16_at(body, 176),
            focus_coil_airflow: u16_at(body, 177),
            oil_temperature: u16_at(body, 178),
            prf_limit: u16_at(body, 179),
            transmitter_oil_level: u16_at(body, 180),
            transmitter_battery_charging: u16_at(body, 181),
            high_voltage_status: u16_at(body, 182),
            transmitter_recycling_summary: u16_at(body, 183),
            transmitter_inoperable: u16_at(body, 184),
            transmitter_air_filter: u16_at(body, 185),
            zero_test_bits: std::array::from_fn(|bit| u16_at(body, 186 + bit)),
            one_test_bits: std::array::from_fn(|bit| u16_at(body, 194 + bit)),
            xmtr_spip_interface: u16_at(body, 202),
            transmitter_summary_status: u16_at(body, 203),
            transmitter_rf_power: f32_at(body, 205),
            horizontal_xmtr_peak_power: f32_at(body, 207),
            xmtr_peak_power: f32_at(body, 209),
            vertical_xmtr_peak_power: f32_at(body, 211),
            xmtr_rf_avg_power: f32_at(body, 213),
            xmtr_recycle_count: u32_at(body, 217),
            receiver_bias: f32_at(body, 219),
            transmit_imbalance: f32_at(body, 221),
            xmtr_power_meter_zero: f32_at(body, 223),
        }
    }
}

/// Tower and utilities status (halfwords 229 to 249).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TowerUtilities {
    /// AC unit 1 compressor shut off (halfword 229): 0 = OK, 1 = shut off.
    pub ac_unit_1_compressor_shut_off: u16,
    /// AC unit 2 compressor shut off (halfword 230): 0 = OK, 1 = shut off.
    pub ac_unit_2_compressor_shut_off: u16,
    /// Generator maintenance required (halfword 231): 0 = yes, 1 = no.
    pub generator_maintenance_required: u16,
    /// Generator battery voltage (halfword 232): 0 = low, 1 = OK.
    pub generator_battery_voltage: u16,
    /// Generator engine (halfword 233): 0 = fail, 1 = OK.
    pub generator_engine: u16,
    /// Generator volt/frequency (halfword 234): 0 = not available, 1 = available.
    pub generator_volt_frequency: u16,
    /// Power source (halfword 235): 0 = utility power, 1 = generator power.
    pub power_source: u16,
    /// Transitional power source (TPS) (halfword 236): 0 = OK, 1 = off.
    pub transitional_power_source: u16,
    /// Generator auto/run/off switch (halfword 237): 0 = manual, 1 = auto.
    pub generator_auto_run_off_switch: u16,
    /// Aircraft hazard lighting (halfword 238): 0 = fail, 1 = OK.
    pub aircraft_hazard_lighting: u16,
}

impl TowerUtilities {
    fn decode(body: &[u8]) -> Self {
        Self {
            ac_unit_1_compressor_shut_off: u16_at(body, 229),
            ac_unit_2_compressor_shut_off: u16_at(body, 230),
            generator_maintenance_required: u16_at(body, 231),
            generator_battery_voltage: u16_at(body, 232),
            generator_engine: u16_at(body, 233),
            generator_volt_frequency: u16_at(body, 234),
            power_source: u16_at(body, 235),
            transitional_power_source: u16_at(body, 236),
            generator_auto_run_off_switch: u16_at(body, 237),
            aircraft_hazard_lighting: u16_at(body, 238),
        }
    }
}

/// Equipment shelter status (halfwords 250 to 299).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EquipmentShelter {
    /// Equipment shelter fire detection system (halfword 250): 0 = OK, 1 = fail.
    pub fire_detection_system: u16,
    /// Equipment shelter fire/smoke (halfword 251): 0 = OK, 1 = fire.
    pub equipment_shelter_fire_smoke: u16,
    /// Generator shelter fire/smoke (halfword 252): 0 = fire, 1 = OK.
    pub generator_shelter_fire_smoke: u16,
    /// Utility voltage/frequency (halfword 253): 0 = not available, 1 = available.
    pub utility_voltage_frequency: u16,
    /// Site security alarm (halfword 254): 0 = alarm, 1 = OK.
    pub site_security_alarm: u16,
    /// Security equipment (halfword 255): 0 = fail, 1 = OK.
    pub security_equipment: u16,
    /// Security system (halfword 256): 0 = disabled, 1 = OK.
    pub security_system: u16,
    /// Receiver connected to antenna (halfword 257): 0 = connected, 1 = not connected, 2 = N/A
    /// (single channel system).
    pub receiver_connected_to_antenna: u16,
    /// Radome hatch (halfword 258): 0 = open, 1 = closed.
    pub radome_hatch: u16,
    /// AC unit 1 filter (halfword 259): 0 = dirty, 1 = OK.
    pub ac_unit_1_filter_dirty: u16,
    /// AC unit 2 filter (halfword 260): 0 = dirty, 1 = OK.
    pub ac_unit_2_filter_dirty: u16,
    /// Equipment shelter temperature (halfwords 261-262), deg C (0.00 to +50.00).
    pub equipment_shelter_temperature: f32,
    /// Outside ambient temperature (halfwords 263-264), deg C (-50.00 to +50.00).
    pub outside_ambient_temperature: f32,
    /// Transmitter leaving air temperature (halfwords 265-266), deg C (-10.00 to +60.00).
    pub transmitter_leaving_air_temperature: f32,
    /// AC unit 1 discharge air temperature (halfwords 267-268), deg C (0.00 to +50.00).
    pub ac_unit_1_discharge_air_temperature: f32,
    /// Generator shelter temperature (halfwords 269-270), deg C (0.00 to +50.00).
    pub generator_shelter_temperature: f32,
    /// Radome air temperature (halfwords 271-272), deg C (-50.00 to +50.00).
    pub radome_air_temperature: f32,
    /// AC unit 2 discharge air temperature (halfwords 273-274), deg C (0.00 to +50.00).
    pub ac_unit_2_discharge_air_temperature: f32,
    /// SPIP +15 V power supply (halfwords 275-276), V.
    pub spip_15v_ps: f32,
    /// SPIP -15 V power supply (halfwords 277-278), V.
    pub spip_neg_15v_ps: f32,
    /// SPIP +28 V power supply status (halfword 279): 0 = fail, 1 = OK.
    pub spip_28v_ps_status: u16,
    /// SPIP +5 V power supply (halfwords 281-282), V (0.00 to 6.64).
    pub spip_5v_ps: f32,
    /// Converted generator fuel level (halfword 283), % (0 to 100).
    pub converted_generator_fuel_level: u16,
}

impl EquipmentShelter {
    fn decode(body: &[u8]) -> Self {
        Self {
            fire_detection_system: u16_at(body, 250),
            equipment_shelter_fire_smoke: u16_at(body, 251),
            generator_shelter_fire_smoke: u16_at(body, 252),
            utility_voltage_frequency: u16_at(body, 253),
            site_security_alarm: u16_at(body, 254),
            security_equipment: u16_at(body, 255),
            security_system: u16_at(body, 256),
            receiver_connected_to_antenna: u16_at(body, 257),
            radome_hatch: u16_at(body, 258),
            ac_unit_1_filter_dirty: u16_at(body, 259),
            ac_unit_2_filter_dirty: u16_at(body, 260),
            equipment_shelter_temperature: f32_at(body, 261),
            outside_ambient_temperature: f32_at(body, 263),
            transmitter_leaving_air_temperature: f32_at(body, 265),
            ac_unit_1_discharge_air_temperature: f32_at(body, 267),
            generator_shelter_temperature: f32_at(body, 269),
            radome_air_temperature: f32_at(body, 271),
            ac_unit_2_discharge_air_temperature: f32_at(body, 273),
            spip_15v_ps: f32_at(body, 275),
            spip_neg_15v_ps: f32_at(body, 277),
            spip_28v_ps_status: u16_at(body, 279),
            spip_5v_ps: f32_at(body, 281),
            converted_generator_fuel_level: u16_at(body, 283),
        }
    }
}

/// Antenna and pedestal status (halfwords 300 to 340).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AntennaPedestal {
    /// Elevation + dead limit (antenna in the upper dead limit) (halfword 300): 0 = OK, 1 = in
    /// limit.
    pub elevation_pos_dead_limit: u16,
    /// +150 V overvoltage (halfword 301): 0 = OK, 1 = overvoltage.
    pub pos_150v_overvoltage: u16,
    /// +150 V undervoltage (halfword 302): 0 = OK, 1 = undervoltage (the table says "overvoltage").
    pub pos_150v_undervoltage: u16,
    /// Elevation servo amplifier inhibit (halfword 303): 0 = normal, 1 = inhibit.
    pub elevation_servo_amp_inhibit: u16,
    /// Elevation servo amplifier short circuit (halfword 304): 0 = normal, 1 = short circuit.
    pub elevation_servo_amp_short_circuit: u16,
    /// Elevation servo amplifier overtemperature (halfword 305): 0 = normal, 1 = overtemp.
    pub elevation_servo_amp_overtemp: u16,
    /// Elevation motor overtemperature (halfword 306): 0 = OK, 1 = overtemp.
    pub elevation_motor_overtemp: u16,
    /// Elevation stow pin (halfword 307): 0 = operational, 1 = engaged.
    pub elevation_stow_pin: u16,
    /// Elevation housing DC-to-DC converter (+5 V) power supply (halfword 308): 0 = OK, 1 = fail.
    pub elevation_housing_5v_ps: u16,
    /// Elevation - dead limit (antenna in the lower dead limit) (halfword 309): 0 = OK, 1 = in
    /// limit.
    pub elevation_neg_dead_limit: u16,
    /// Elevation + normal limit (antenna in the upper normal limit) (halfword 310): 0 = OK, 1 = in
    /// limit.
    pub elevation_pos_normal_limit: u16,
    /// Elevation - normal limit (halfword 311): 0 = OK, 1 = in limit.
    pub elevation_neg_normal_limit: u16,
    /// Elevation encoder light (halfword 312): 0 = OK, 1 = fail.
    pub elevation_encoder_light: u16,
    /// Elevation gearbox oil (halfword 313): 0 = OK, 1 = oil level low.
    pub elevation_gearbox_oil: u16,
    /// Elevation handwheel (halfword 314): 0 = operational, 1 = engaged.
    pub elevation_handwheel: u16,
    /// Elevation amplifier power supply (halfword 315): 0 = OK, 1 = fail.
    pub elevation_amp_ps: u16,
    /// Azimuth servo amplifier inhibit (halfword 316): 0 = OK, 1 = inhibit.
    pub azimuth_servo_amp_inhibit: u16,
    /// Azimuth servo amplifier short circuit (halfword 317): 0 = OK, 1 = short circuit.
    pub azimuth_servo_amp_short_circuit: u16,
    /// Azimuth servo amplifier overtemperature (halfword 318): 0 = OK, 1 = overtemp.
    pub azimuth_servo_amp_overtemp: u16,
    /// Azimuth motor overtemperature (halfword 319): 0 = OK, 1 = overtemp.
    pub azimuth_motor_overtemp: u16,
    /// Azimuth stow pin (halfword 320): 0 = operational, 1 = engaged.
    pub azimuth_stow_pin: u16,
    /// Azimuth housing DC-to-DC converter (+5 V) power supply (halfword 321): 0 = OK, 1 = fail.
    pub azimuth_housing_5v_ps: u16,
    /// Azimuth encoder light (halfword 322): 0 = OK, 1 = fail.
    pub azimuth_encoder_light: u16,
    /// Azimuth gearbox oil (halfword 323): 0 = OK, 1 = oil level low.
    pub azimuth_gearbox_oil: u16,
    /// Azimuth bull gear oil (halfword 324): 0 = OK, 1 = oil level low.
    pub azimuth_bull_gear_oil: u16,
    /// Azimuth handwheel (halfword 325): 0 = operational, 1 = engaged.
    pub azimuth_handwheel: u16,
    /// Azimuth servo amplifier power supply (halfword 326): 0 = OK, 1 = fail.
    pub azimuth_servo_amp_ps: u16,
    /// Servo (halfword 327): 0 = on, 1 = off.
    pub servo: u16,
    /// Pedestal interlock switch (halfword 328): 0 = operational, 1 = safe.
    pub pedestal_interlock_switch: u16,
}

impl AntennaPedestal {
    fn decode(body: &[u8]) -> Self {
        Self {
            elevation_pos_dead_limit: u16_at(body, 300),
            pos_150v_overvoltage: u16_at(body, 301),
            pos_150v_undervoltage: u16_at(body, 302),
            elevation_servo_amp_inhibit: u16_at(body, 303),
            elevation_servo_amp_short_circuit: u16_at(body, 304),
            elevation_servo_amp_overtemp: u16_at(body, 305),
            elevation_motor_overtemp: u16_at(body, 306),
            elevation_stow_pin: u16_at(body, 307),
            elevation_housing_5v_ps: u16_at(body, 308),
            elevation_neg_dead_limit: u16_at(body, 309),
            elevation_pos_normal_limit: u16_at(body, 310),
            elevation_neg_normal_limit: u16_at(body, 311),
            elevation_encoder_light: u16_at(body, 312),
            elevation_gearbox_oil: u16_at(body, 313),
            elevation_handwheel: u16_at(body, 314),
            elevation_amp_ps: u16_at(body, 315),
            azimuth_servo_amp_inhibit: u16_at(body, 316),
            azimuth_servo_amp_short_circuit: u16_at(body, 317),
            azimuth_servo_amp_overtemp: u16_at(body, 318),
            azimuth_motor_overtemp: u16_at(body, 319),
            azimuth_stow_pin: u16_at(body, 320),
            azimuth_housing_5v_ps: u16_at(body, 321),
            azimuth_encoder_light: u16_at(body, 322),
            azimuth_gearbox_oil: u16_at(body, 323),
            azimuth_bull_gear_oil: u16_at(body, 324),
            azimuth_handwheel: u16_at(body, 325),
            azimuth_servo_amp_ps: u16_at(body, 326),
            servo: u16_at(body, 327),
            pedestal_interlock_switch: u16_at(body, 328),
        }
    }
}

/// RF generator and receiver status (halfwords 341 to 362).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RfGeneratorReceiver {
    /// COHO/clock (halfword 341): 0 = OK, 1 = fail.
    pub coho_clock: u16,
    /// RF generator frequency select oscillator (halfword 342): 0 = OK, 1 = fail.
    pub frequency_select_oscillator: u16,
    /// RF generator RF/STALO (halfword 343): 0 = OK, 1 = fail.
    pub rf_stalo: u16,
    /// RF generator phase shifted COHO (halfword 344): 0 = OK, 1 = fail.
    pub phase_shifted_coho: u16,
    /// +9 V receiver power supply (halfword 345): 0 = OK, 1 = fail.
    pub receiver_ps_9v: u16,
    /// +5 V receiver power supply (halfword 346): 0 = OK, 1 = fail.
    pub receiver_ps_5v: u16,
    /// +/-18 V receiver power supply (halfword 347): 0 = OK, 1 = fail.
    pub receiver_ps_18v: u16,
    /// -9 V receiver power supply (halfword 348): 0 = OK, 1 = fail.
    pub receiver_ps_neg_9v: u16,
    /// +5 V single channel RDAIU power supply (halfword 349): 0 = OK, 1 = fail.
    pub rdaiu_ps_5v: u16,
    /// Horizontal short pulse noise (halfwords 351-352), dBm (-100.00 to -50.00).
    pub horizontal_short_pulse_noise: f32,
    /// Horizontal long pulse noise (halfwords 353-354), dBm (-100.00 to -50.00).
    pub horizontal_long_pulse_noise: f32,
    /// Horizontal noise temperature (halfwords 355-356), K (0 to 9999.99).
    pub horizontal_noise_temperature: f32,
    /// Vertical short pulse noise (halfwords 357-358), dBm (-100.00 to -50.00).
    pub vertical_short_pulse_noise: f32,
    /// Vertical long pulse noise (halfwords 359-360), dBm (-100.00 to -50.00).
    pub vertical_long_pulse_noise: f32,
    /// Vertical noise temperature (halfwords 361-362), K (0 to 9999.99).
    pub vertical_noise_temperature: f32,
}

impl RfGeneratorReceiver {
    fn decode(body: &[u8]) -> Self {
        Self {
            coho_clock: u16_at(body, 341),
            frequency_select_oscillator: u16_at(body, 342),
            rf_stalo: u16_at(body, 343),
            phase_shifted_coho: u16_at(body, 344),
            receiver_ps_9v: u16_at(body, 345),
            receiver_ps_5v: u16_at(body, 346),
            receiver_ps_18v: u16_at(body, 347),
            receiver_ps_neg_9v: u16_at(body, 348),
            rdaiu_ps_5v: u16_at(body, 349),
            horizontal_short_pulse_noise: f32_at(body, 351),
            horizontal_long_pulse_noise: f32_at(body, 353),
            horizontal_noise_temperature: f32_at(body, 355),
            vertical_short_pulse_noise: f32_at(body, 357),
            vertical_long_pulse_noise: f32_at(body, 359),
            vertical_noise_temperature: f32_at(body, 361),
        }
    }
}

/// Calibration results (halfwords 363 to 430).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Calibration {
    /// Horizontal linearity (halfwords 363-364), unitless (0.5000 to 1.5000).
    pub horizontal_linearity: f32,
    /// Horizontal dynamic range (halfwords 365-366), dB (0.000 to 120.000).
    pub horizontal_dynamic_range: f32,
    /// Horizontal delta dBZ0 (halfwords 367-368), dB (-198.00 to +198.00).
    pub horizontal_delta_dbz0: f32,
    /// Vertical delta dBZ0 (halfwords 369-370), dB (-198.00 to +198.00).
    pub vertical_delta_dbz0: f32,
    /// KD peak measured (halfwords 371-372), dBm (-99.90 to +99.90).
    pub kd_peak_measured: f32,
    /// Short pulse horizontal dBZ0 (halfwords 375-376), dBZ (-99.9000 to +99.9000).
    pub short_pulse_horizontal_dbz0: f32,
    /// Long pulse horizontal dBZ0 (halfwords 377-378), dBZ (-99.9000 to +99.9000).
    pub long_pulse_horizontal_dbz0: f32,
    /// Velocity check (processed) (halfword 379): 0 = good, 1 = fail.
    pub velocity_processed: u16,
    /// Spectrum width check (processed) (halfword 380): 0 = good, 1 = fail.
    pub width_processed: u16,
    /// Velocity check (RF generator) (halfword 381): 0 = good, 1 = fail.
    pub velocity_rf_gen: u16,
    /// Spectrum width check (RF generator) (halfword 382): 0 = good, 1 = fail.
    pub width_rf_gen: u16,
    /// Horizontal I0 (halfwords 383-384), dBm (-999.9000 to +999.9000).
    pub horizontal_i0: f32,
    /// Vertical I0 (halfwords 385-386), dBm (-999.9000 to +999.9000).
    pub vertical_i0: f32,
    /// Vertical dynamic range (halfwords 387-388), dB (0.000 to 120.000).
    pub vertical_dynamic_range: f32,
    /// Short pulse vertical dBZ0 (halfwords 389-390), dBZ (-99.9000 to +99.9000).
    pub short_pulse_vertical_dbz0: f32,
    /// Long pulse vertical dBZ0 (halfwords 391-392), dBZ (-99.9000 to +99.9000).
    pub long_pulse_vertical_dbz0: f32,
    /// Horizontal power sense (halfwords 397-398), dBm (-999.9000 to +999.9000).
    pub horizontal_power_sense: f32,
    /// Vertical power sense (halfwords 399-400), dBm (-999.9000 to +999.9000).
    pub vertical_power_sense: f32,
    /// ZDR offset (called ZDR bias before Build 22.0) (halfwords 401-402), dB (-999.9000 to
    /// +999.9000).
    pub zdr_offset: f32,
    /// Clutter suppression delta (halfwords 409-410), dB (-99.90 to +99.90).
    pub clutter_suppression_delta: f32,
    /// Clutter suppression unfiltered power (halfwords 411-412), dBZ (-99.90 to +99.90).
    pub clutter_suppression_unfiltered_power: f32,
    /// Clutter suppression filtered power (halfwords 413-414), dBZ (-99.90 to +99.90).
    pub clutter_suppression_filtered_power: f32,
    /// Vertical linearity (halfwords 425-426), unitless (0.5000 to 1.5000).
    pub vertical_linearity: f32,
}

impl Calibration {
    fn decode(body: &[u8]) -> Self {
        Self {
            horizontal_linearity: f32_at(body, 363),
            horizontal_dynamic_range: f32_at(body, 365),
            horizontal_delta_dbz0: f32_at(body, 367),
            vertical_delta_dbz0: f32_at(body, 369),
            kd_peak_measured: f32_at(body, 371),
            short_pulse_horizontal_dbz0: f32_at(body, 375),
            long_pulse_horizontal_dbz0: f32_at(body, 377),
            velocity_processed: u16_at(body, 379),
            width_processed: u16_at(body, 380),
            velocity_rf_gen: u16_at(body, 381),
            width_rf_gen: u16_at(body, 382),
            horizontal_i0: f32_at(body, 383),
            vertical_i0: f32_at(body, 385),
            vertical_dynamic_range: f32_at(body, 387),
            short_pulse_vertical_dbz0: f32_at(body, 389),
            long_pulse_vertical_dbz0: f32_at(body, 391),
            horizontal_power_sense: f32_at(body, 397),
            vertical_power_sense: f32_at(body, 399),
            zdr_offset: f32_at(body, 401),
            clutter_suppression_delta: f32_at(body, 409),
            clutter_suppression_unfiltered_power: f32_at(body, 411),
            clutter_suppression_filtered_power: f32_at(body, 413),
            vertical_linearity: f32_at(body, 425),
        }
    }
}

/// File and RSP status (halfwords 431 to 460).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FileStatus {
    /// State file read status (halfword 431): 0 = OK, 1 = fail.
    pub state_file_read: u16,
    /// State file write status (halfword 432): 0 = OK, 1 = fail.
    pub state_file_write: u16,
    /// Bypass map file read status (halfword 433): 0 = OK, 1 = fail.
    pub bypass_map_file_read: u16,
    /// Bypass map file write status (halfword 434): 0 = OK, 1 = fail.
    pub bypass_map_file_write: u16,
    /// Current adaptation file read status (halfword 437): 0 = OK, 1 = fail.
    pub current_adaptation_file_read: u16,
    /// Current adaptation file write status (halfword 438): 0 = OK, 1 = fail.
    pub current_adaptation_file_write: u16,
    /// Censor zone file read status (halfword 439): 0 = OK, 1 = fail.
    pub censor_zone_file_read: u16,
    /// Censor zone file write status (halfword 440): 0 = OK, 1 = fail.
    pub censor_zone_file_write: u16,
    /// Remote VCP file read status (halfword 441): 0 = OK, 1 = fail.
    pub remote_vcp_file_read: u16,
    /// Remote VCP file write status (halfword 442): 0 = OK, 1 = fail.
    pub remote_vcp_file_write: u16,
    /// Baseline adaptation file read status (halfword 443): 0 = OK, 1 = fail.
    pub baseline_adaptation_file_read: u16,
    /// Read status of the PRF sets (bit field; per bit 0 = fail, 1 = OK) (halfword 444): bit
    /// 0 = surveillance, bit 1 = Doppler, bit 2 = staggered PRT.
    pub prf_sets_read: u16,
    /// Clutter filter map file read status (halfword 445): 0 = OK, 1 = fail.
    pub clutter_filter_map_file_read: u16,
    /// Clutter filter map file write status (halfword 446): 0 = OK, 1 = fail.
    pub clutter_filter_map_file_write: u16,
    /// General disk I/O error (halfword 447): 0 = OK, 1 = fail.
    pub general_disk_io_error: u16,
    /// RSP health status (bit field; per bit 1 = fail, 0 = OK) (halfword 448 byte 0): bit
    /// 0 = system drive SMART status, bit 1 = data drive SMART status, bit 2 = CPU 1
    /// overtemperature, bit 3 = CPU 2 overtemperature. Byte 1 of the halfword is spare.
    pub rsp_status: u8,
    /// RSP CPU 1 temperature (halfword 449 byte 0), deg C (0 to 255; 255 denotes a suspected sensor
    /// failure).
    pub rsp_cpu1_temperature: u8,
    /// RSP CPU 2 temperature (halfword 449 byte 1), deg C (0 to 255; 255 denotes a suspected sensor
    /// failure).
    pub rsp_cpu2_temperature: u8,
    /// RSP power used (halfword 450), as measured by the motherboard sensor, W (0 to 2000; 65535
    /// denotes a suspected sensor failure).
    pub rsp_motherboard_power: u16,
}

impl FileStatus {
    fn decode(body: &[u8]) -> Self {
        Self {
            state_file_read: u16_at(body, 431),
            state_file_write: u16_at(body, 432),
            bypass_map_file_read: u16_at(body, 433),
            bypass_map_file_write: u16_at(body, 434),
            current_adaptation_file_read: u16_at(body, 437),
            current_adaptation_file_write: u16_at(body, 438),
            censor_zone_file_read: u16_at(body, 439),
            censor_zone_file_write: u16_at(body, 440),
            remote_vcp_file_read: u16_at(body, 441),
            remote_vcp_file_write: u16_at(body, 442),
            baseline_adaptation_file_read: u16_at(body, 443),
            prf_sets_read: u16_at(body, 444),
            clutter_filter_map_file_read: u16_at(body, 445),
            clutter_filter_map_file_write: u16_at(body, 446),
            general_disk_io_error: u16_at(body, 447),
            rsp_status: body[offset(448)],
            rsp_cpu1_temperature: body[offset(449)],
            rsp_cpu2_temperature: body[offset(449) + 1],
            rsp_motherboard_power: u16_at(body, 450),
        }
    }
}

/// Device communication status (halfwords 461 to 479).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DeviceStatus {
    /// SPIP communication status (halfword 461): 0 = OK, 1 = fail.
    pub spip_comm_status: u16,
    /// HCI communication status (halfword 462): 0 = OK, 1 = fail.
    pub hci_comm_status: u16,
    /// Signal processor command status (halfword 464): 0 = OK, 1 = fail.
    pub signal_processor_command_status: u16,
    /// AME communication status (halfword 465): 0 = OK, 1 = fail.
    pub ame_communication_status: u16,
    /// RMS link status (halfword 466): 0 = connected, 1 = not connected.
    pub rms_link_status: u16,
    /// RPG link status (halfword 467): 0 = connected, 1 = not connected.
    pub rpg_link_status: u16,
    /// Interpanel link (channel 1 SPIP to channel 2 SPIP power and communications) (halfword 468):
    /// 0 = OK, 1 = fail, 2 = N/A (single channel system).
    pub interpanel_link_status: u16,
    /// Time the next performance check is due (halfwords 469-470), Unix epoch seconds (32-bit
    /// time_t).
    pub performance_check_time: u32,
}

impl DeviceStatus {
    fn decode(body: &[u8]) -> Self {
        Self {
            spip_comm_status: u16_at(body, 461),
            hci_comm_status: u16_at(body, 462),
            signal_processor_command_status: u16_at(body, 464),
            ame_communication_status: u16_at(body, 465),
            rms_link_status: u16_at(body, 466),
            rpg_link_status: u16_at(body, 467),
            interpanel_link_status: u16_at(body, 468),
            performance_check_time: u32_at(body, 469),
        }
    }
}
/// Walker hook: the typed body for message 3. Legacy RDA bodies are yielded
/// unparsed.
pub(crate) fn message_body<'a>(
    header: &MessageHeader,
    body: Cow<'a, [u8]>,
) -> Result<MessageBody<'a>> {
    match RdaSystem::from_channels(header.channels) {
        RdaSystem::Orda => PerformanceMaintenance::decode(&body)
            .map(|decoded| MessageBody::Performance(Box::new(decoded))),
        RdaSystem::Legacy => Ok(MessageBody::Unparsed(body)),
    }
}
