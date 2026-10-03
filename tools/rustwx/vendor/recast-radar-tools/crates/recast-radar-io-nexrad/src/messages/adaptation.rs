//! RDA Adaptation Data (message 18, ICD 2620002AA Table XV).
//!
//! The RDA sends its adaptation data (site parameters, alarm thresholds,
//! signal processing and calibration constants) at wideband connection and
//! whenever it changes. The 9468-byte body is split over four segments that
//! the walker joins. [`RdaAdaptationData`] follows the Build 24.0 table:
//! every field is named after its ICD mnemonic (lower case) and documented
//! with its byte location, units and ICD range. Spare bytes are not kept;
//! strings drop NUL padding and "T"/"F" flags decode to `Some(true)` /
//! `Some(false)` (`None` for anything else).
//!
//! # Earlier builds
//!
//! The body has been 9468 bytes since Build 10.0, but some locations were
//! reassigned; a file from an older build carries the older content in these
//! fields (checked against ICD revisions F, J, M, N, P, R, U and W, and
//! against the corpus):
//!
//! - Through Build 13 (2620002M), bytes 1328-8359 held six default VCP
//!   definitions in Message 5 format (VCPAT11, VCPAT21, VCPAT31, VCPAT32,
//!   VCPAT300 and VCPAT301, 1172 bytes each); Build 17.0 (2620002P) kept the
//!   first four and Build 18.0 (2620002R) made the whole range spare. The
//!   Build 24.0 table puts DIG_RCVR_CLOCK_FREQ and COHO_FREQ at bytes
//!   2500-2515, inside the former VCPAT21; corpus files carry them from
//!   Build 23.1. These tables are not decoded here.
//! - Before Build 17.0, bytes 44 and 52 held the position gain factors K1 and
//!   K3 (now LOWER_PRE_LIMIT and UPPER_PRE_LIMIT), bytes 136-147 the pedestal
//!   +28 V, +5 V and +/-15 V regulation limits (now LOWER_DEAD_LIMIT,
//!   UPPER_DEAD_LIMIT and a spare), and bytes 156-167 the DAU +5 V, +/-15 V
//!   and +28 V limits (now the SPIP limits and a spare).
//! - Bytes 1132-1135 held BEAMWIDTH (MetPy's layout). The Build 24.0 table
//!   makes them spare; corpus files from 2008 to 2016 carry 0.89 to 0.94
//!   degrees there, and files from 2020 on hold zero.
//!   [`RdaAdaptationData::beamwidth`] keeps the value.
//! - Bytes 1164 and 1172 held the horizontal and vertical noise temperature
//!   maintenance limits (Real*4) through Build 13, were spare in Builds 17
//!   and 18, and hold the Integer*4 H_MIN_NOISETEMP and V_MIN_NOISETEMP in
//!   corpus files from Build 19.1 on.
//! - Build 18.0 renamed RNSCALE to H_RNSCALE, listed V_RNSCALE (corpus files
//!   from Build 12.0 on already carry values there; Build 10.0 files hold
//!   zero), and replaced REFLECTOR_BIAS (byte 9028) with SUN_BIAS. REFINED_PARK
//!   (byte 8688) is zero in corpus files before Build 21.0. Build 22.0
//!   (2620002W) renamed ZDR_BIAS_DGRAD_LIM and the baseline ZDR bias to
//!   their "offset" names.
//!
//! Messages from a legacy (pre-ORDA) RDA carry a 9600-byte message 18 that
//! no available ICD revision documents; the walker yields those bodies
//! unparsed.

use std::borrow::Cow;

use super::MessageBody;
use super::rda_status::RdaSystem;
use crate::{MessageHeader, Result};

/// Body length of Table XV: 9468 bytes.
pub const RDA_ADAPTATION_DATA_LEN: usize = 9468;

/// TFREQ_MHZ location (Table XV bytes 1092-1095).
const TFREQ_MHZ_OFFSET: usize = 1092;
/// BEAMWIDTH location (bytes 1132-1135, before Build 18).
const BEAMWIDTH_OFFSET: usize = 1132;
/// ANTENNA_GAIN location (Table XV bytes 1136-1139).
const ANTENNA_GAIN_OFFSET: usize = 1136;

/// Plausible transmitter frequencies, MHz: L band to Ka band. The ICD range
/// of the WSR-88D is 2700 to 3000 MHz; Level II files the writer makes from
/// C- and X-band radars carry theirs.
pub(crate) const PLAUSIBLE_FREQUENCY_MHZ: std::ops::RangeInclusive<i32> = 1000..=40_000;
/// Plausible antenna gains, dB (ICD range of the WSR-88D: 43 to 47 dB).
pub(crate) const PLAUSIBLE_ANTENNA_GAIN_DB: std::ops::RangeInclusive<f32> = 20.0..=60.0;
/// Plausible beam widths, degrees; leaves out the zero of files from Build
/// 18 on.
pub(crate) const PLAUSIBLE_BEAM_WIDTH_DEG: std::ops::RangeInclusive<f32> = 0.1..=10.0;

/// Site constants of the FM301 model (`RadarParameters`) at the start of an
/// Open RDA message 18 body.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SiteConstants {
    /// TFREQ_MHZ in Hz; `None` outside [`PLAUSIBLE_FREQUENCY_MHZ`].
    pub frequency_hz: Option<f64>,
    /// ANTENNA_GAIN in dB; `None` outside [`PLAUSIBLE_ANTENNA_GAIN_DB`].
    pub antenna_gain_db: Option<f32>,
    /// BEAMWIDTH in degrees; `None` outside [`PLAUSIBLE_BEAM_WIDTH_DEG`].
    pub beam_width_deg: Option<f32>,
}

/// The [`SiteConstants`] of a message 18 body. They lie in the first segment
/// (the first 1204 body bytes), so the volume decoder reads them without
/// reassembling the message. `None` when `body` is too short to hold them.
pub(crate) fn site_constants(body: &[u8]) -> Option<SiteConstants> {
    if body.len() < ANTENNA_GAIN_OFFSET + 4 {
        return None;
    }
    let frequency_mhz = i32_at(body, TFREQ_MHZ_OFFSET);
    let antenna_gain_db = f32_at(body, ANTENNA_GAIN_OFFSET);
    let beam_width_deg = f32_at(body, BEAMWIDTH_OFFSET);
    Some(SiteConstants {
        frequency_hz: PLAUSIBLE_FREQUENCY_MHZ
            .contains(&frequency_mhz)
            .then(|| f64::from(frequency_mhz) * 1e6),
        antenna_gain_db: PLAUSIBLE_ANTENNA_GAIN_DB
            .contains(&antenna_gain_db)
            .then_some(antenna_gain_db),
        beam_width_deg: PLAUSIBLE_BEAM_WIDTH_DEG
            .contains(&beam_width_deg)
            .then_some(beam_width_deg),
    })
}

/// Decoded RDA Adaptation Data (Table XV, Build 24.0).
#[derive(Clone, Debug, PartialEq)]
pub struct RdaAdaptationData {
    /// Name of the adaptation data file ("baseline" or "current") (ADAP_FILE_NAME, bytes 0-11).
    pub adap_file_name: String,
    /// Format of the adaptation data file (for example "14") (ADAP_FORMAT, bytes 12-15).
    pub adap_format: String,
    /// Revision number of the adaptation data file (ADAP_REVISION, bytes 16-19), the message
    /// revision number (for example "20"; it increments when the message format changes).
    pub adap_revision: String,
    /// Last modified date of the adaptation data file (ADAP_DATE, bytes 20-31), "mm/dd/yy".
    pub adap_date: String,
    /// Last modified time of the adaptation data file (ADAP_TIME, bytes 32-43), "hh-mm-ss".
    pub adap_time: String,
    /// Angle of the lower pre-limit switch (LOWER_PRE_LIMIT, bytes 44-47), deg (-3.000 to 0.000;
    /// the table prints 3.000).
    pub lower_pre_limit: f32,
    /// Latency of the azimuth encoder measurement (AZ_LAT, bytes 48-51), s (0.0000 to 2.0000).
    pub az_lat: f32,
    /// Angle of the upper pre-limit switch (UPPER_PRE_LIMIT, bytes 52-55), deg (55.000 to 66.000).
    pub upper_pre_limit: f32,
    /// Latency of the elevation encoder measurement (EL_LAT, bytes 56-59), s (0.000 to 2.000).
    pub el_lat: f32,
    /// Pedestal park position in azimuth (PARKAZ, bytes 60-63), deg (0.00 to 359.99).
    pub parkaz: f32,
    /// Pedestal park position in elevation (PARKEL, bytes 64-67), deg (-1.00 to 55.00).
    pub parkel: f32,
    /// Generator fuel level height to capacity conversion table (A_FUEL_CONV(0..10), bytes 68-111),
    /// % of capacity at 0%, 10%, ... 100% of tank height (0.0 to 100.0).
    pub a_fuel_conv: [f32; 11],
    /// Minimum equipment shelter alarm temperature (A_MIN_SHELTER_TEMP, bytes 112-115), deg C
    /// (0.0 to 50.0).
    pub a_min_shelter_temp: f32,
    /// Maximum equipment shelter alarm temperature (A_MAX_SHELTER_TEMP, bytes 116-119), deg C
    /// (0.0 to 50.0).
    pub a_max_shelter_temp: f32,
    /// Minimum A/C discharge air temperature differential (A_MIN_SHELTER_AC_TEMP_DIFF, bytes
    /// 120-123), deg C (0.0 to 10.0).
    pub a_min_shelter_ac_temp_diff: f32,
    /// Maximum transmitter leaving air alarm temperature (A_MAX_XMTR_AIR_TEMP, bytes 124-127), deg
    /// C (0.0 to 55.0).
    pub a_max_xmtr_air_temp: f32,
    /// Maximum radome alarm temperature (A_MAX_RAD_TEMP, bytes 128-131), deg C (0.0 to 50.0).
    pub a_max_rad_temp: f32,
    /// Maximum radome minus ambient temperature difference (A_MAX_RAD_TEMP_RISE, bytes 132-135),
    /// deg C (0.0 to 10.0).
    pub a_max_rad_temp_rise: f32,
    /// Angle of the lower dead limit switch (LOWER_DEAD_LIMIT, bytes 136-139), deg (-4.000 to
    /// 0.000).
    pub lower_dead_limit: f32,
    /// Angle of the upper dead limit switch (UPPER_DEAD_LIMIT, bytes 140-143), deg (60.000 to
    /// 66.000).
    pub upper_dead_limit: f32,
    /// Minimum generator shelter alarm temperature (A_MIN_GEN_ROOM_TEMP, bytes 148-151), deg C
    /// (0.0 to 50.0).
    pub a_min_gen_room_temp: f32,
    /// Maximum generator shelter alarm temperature (A_MAX_GEN_ROOM_TEMP, bytes 152-155), deg C
    /// (0.0 to 50.0).
    pub a_max_gen_room_temp: f32,
    /// SPIP +5 V power supply tolerance (SPIP_5V_REG_LIM, bytes 156-159), % (0.0 to 20.0).
    pub spip_5v_reg_lim: f32,
    /// SPIP +/-15 V power supply tolerance (SPIP_15V_REG_LIM, bytes 160-163), % (0.0 to 20.0).
    pub spip_15v_reg_lim: f32,
    /// RPG co-located (RPG_CO_LOCATED, bytes 176-179).
    pub rpg_co_located: Option<bool>,
    /// Transmitter spectrum filter installed (SPEC_FILTER_INSTALLED, bytes 180-183).
    pub spec_filter_installed: Option<bool>,
    /// Transition power source installed (TPS_INSTALLED, bytes 184-187).
    pub tps_installed: Option<bool>,
    /// FAA RMS installed (RMS_INSTALLED, bytes 188-191).
    pub rms_installed: Option<bool>,
    /// Performance test interval (A_HVDL_TST_INT, bytes 192-195), h (2 to 72).
    pub a_hvdl_tst_int: i32,
    /// RPG loop test interval (A_RPG_LT_INT, bytes 196-199), min (1 to 20).
    pub a_rpg_lt_int: i32,
    /// Required interval time for stable utility power (A_MIN_STAB_UTIL_PWR_TIME, bytes 200-203),
    /// min (1 to 20).
    pub a_min_stab_util_pwr_time: i32,
    /// Maximum generator automatic exercise interval (A_GEN_AUTO_EXER_INTERVAL, bytes 204-207), h
    /// (5 to 1000).
    pub a_gen_auto_exer_interval: i32,
    /// Recommended switch to utility power time interval (A_UTIL_PWR_SW_REQ_INTERVAL, bytes
    /// 208-211), min (5 to 30).
    pub a_util_pwr_sw_req_interval: i32,
    /// Low fuel tank warning level (A_LOW_FUEL_LEVEL, bytes 212-215), % (0.0 to 100.0).
    pub a_low_fuel_level: f32,
    /// Configuration channel number (1 or 2) (CONFIG_CHAN_NUMBER, bytes 216-219).
    pub config_chan_number: i32,
    /// Redundant channel configuration (REDUNDANT_CHAN_CONFIG, bytes 224-227): 1 = single channel,
    /// 2 = FAA redundant, 3 = NWS redundant.
    pub redundant_chan_config: i32,
    /// Test signal attenuator insertion losses for 0 dB to 103 dB of attenuation
    /// (ATTEN_TABLE(0..103), bytes 228-643), dB (index n ranges from -(n+1).00 to -(n-1).00).
    pub atten_table: [f32; 104],
    /// Path loss (PATH_LOSSES(7), bytes 668-671), vertical IF heliax to 4AT16, dB (-5.00 to 0.00).
    pub path_losses_7: f32,
    /// Path loss (PATH_LOSSES(13), bytes 692-695), 2A9A9 RF delay line, dB (-60.00 to -40.00).
    pub path_losses_13: f32,
    /// Path loss (PATH_LOSSES(28), bytes 752-755), horizontal IF heliax to 4AT17, dB (-5.00 to
    /// 0.00).
    pub path_losses_28: f32,
    /// RF pallet horizontal coupler transmitter loss (H_COUPLER_XMT_LOSS, bytes 756-759), dB
    /// (-40.00 to -20.00).
    pub h_coupler_xmt_loss: f32,
    /// Path loss (PATH_LOSSES(32), bytes 768-771), WG02 harmonic filter, dB (-0.50 to -0.05).
    pub path_losses_32: f32,
    /// Path loss (PATH_LOSSES(33), bytes 772-775), waveguide klystron to switch, dB (-1.00 to
    /// -0.01).
    pub path_losses_33: f32,
    /// Path loss (PATH_LOSSES(35), bytes 780-783), WG06 spectrum filter, dB (-0.50 to 0.00).
    pub path_losses_35: f32,
    /// Path loss (PATH_LOSSES(39), bytes 796-799), WG04 circulator, dB (-0.50 to -0.05).
    pub path_losses_39: f32,
    /// Path loss (PATH_LOSSES(40), bytes 800-803), A6 arc detector, dB (-0.50 to -0.01).
    pub path_losses_40: f32,
    /// Path loss (PATH_LOSSES(42), bytes 808-811), 1DC1 transmitter coupler coupling, dB (-40.00 to
    /// -20.00).
    pub path_losses_42: f32,
    /// Path loss (PATH_LOSSES(43), bytes 812-815), A33 pad, dB (-10.00 to 0.00).
    pub path_losses_43: f32,
    /// Path loss (PATH_LOSSES(44), bytes 816-819), coax transmitter RF sample to A33 pad, dB
    /// (-3.00 to 0.40).
    pub path_losses_44: f32,
    /// Path loss (PATH_LOSSES(45), bytes 820-823), A20J1_4 power splitter, dB (-8.00 to -4.00).
    pub path_losses_45: f32,
    /// Path loss (PATH_LOSSES(46), bytes 824-827), A20J1_3 power splitter, dB (-8.00 to -4.00).
    pub path_losses_46: f32,
    /// Path loss (PATH_LOSSES(47), bytes 828-831), A20J1_2 power splitter, dB (-8.00 to -4.00).
    pub path_losses_47: f32,
    /// RF pallet horizontal coupler test signal loss (H_COUPLER_CW_LOSS, bytes 832-835), dB
    /// (-40.00 to -20.00).
    pub h_coupler_cw_loss: f32,
    /// RF pallet vertical coupler transmitter loss (V_COUPLER_XMT_LOSS, bytes 836-839), dB
    /// (-40.00 to -20.00).
    pub v_coupler_xmt_loss: f32,
    /// AME test signal bias (AME_TS_BIAS, bytes 844-847), dB.
    pub ame_ts_bias: f32,
    /// Path loss (PATH_LOSSES(52), bytes 848-851), 1AT4 transmitter coupler pad, dB (-6.00 to
    /// 0.00).
    pub path_losses_52: f32,
    /// RF pallet vertical coupler test signal loss (V_COUPLER_CW_LOSS, bytes 852-855), dB
    /// (-40.00 to -20.00).
    pub v_coupler_cw_loss: f32,
    /// Power sense calibration offset bias (PWR_SENSE_BIAS, bytes 864-867), dB (-10.00 to 10.00).
    pub pwr_sense_bias: f32,
    /// AME noise source vertical excess noise ratio (AME_V_NOISE_ENR, bytes 868-871), dB (10.00 to
    /// 35.00).
    pub ame_v_noise_enr: f32,
    /// Path loss (PATH_LOSSES(58), bytes 872-875), 4AT17 attenuator, dB (-7.00 to 0.00).
    pub path_losses_58: f32,
    /// Path loss (PATH_LOSSES(59), bytes 876-879), IFDR IF anti-alias filter, dB (-4.00 to 0.00).
    pub path_losses_59: f32,
    /// Path loss (PATH_LOSSES(60), bytes 880-883), A20J1_5 power splitter, dB (-8.00 to -4.00).
    pub path_losses_60: f32,
    /// Path loss (PATH_LOSSES(61), bytes 884-887), AT5 50 dB attenuator, dB (-53.00 to -47.00).
    pub path_losses_61: f32,
    /// Path loss (PATH_LOSSES(63), bytes 892-895), A39 RF/IF burst mixer, dB (-16.00 to -6.00).
    pub path_losses_63: f32,
    /// Path loss (gain) (PATH_LOSSES(64), bytes 896-899), AR1 burst IF amplifier, dB (23.00 to
    /// 33.00).
    pub path_losses_64: f32,
    /// Path loss (PATH_LOSSES(65), bytes 900-903), IFDR burst anti-alias filter, dB (-4.00 to
    /// 0.00).
    pub path_losses_65: f32,
    /// Path loss (PATH_LOSSES(66), bytes 904-907), DC3 J1_3 6 dB coupler, through, dB (-3.00 to
    /// 0.00).
    pub path_losses_66: f32,
    /// Path loss (PATH_LOSSES(67), bytes 908-911), 4DC3J1 to 4A39 L, dB (-15.00 to -5.00).
    pub path_losses_67: f32,
    /// Path loss (PATH_LOSSES(68), bytes 912-915), AT2+AT3 26 dB COHO attenuator, dB (-29.00 to
    /// -23.00).
    pub path_losses_68: f32,
    /// Non-controlling channel calibration difference (CHAN_CAL_DIFF, bytes 920-923), dB (0.00 to
    /// 4.00).
    pub chan_cal_diff: f32,
    /// AME vertical test signal power (V_TS_CW, bytes 936-939), dBm (0.00 to 30.00).
    pub v_ts_cw: f32,
    /// Horizontal receiver noise normalization per elevation sector (-1.0 to -0.5 deg, -0.5 to
    /// 0.0 deg, ... 4.5 to 5.0 deg, above 5.0 deg) (H_RNSCALE(0..12), bytes 940-991), unitless
    /// (1.000 to 1.800).
    pub h_rnscale: [f32; 13],
    /// Two-way atmospheric loss per km for the elevation sectors of Table XVI (-1.0 to -0.5 deg,
    /// ... above 5.0 deg) (ATMOS(0..12), bytes 992-1043), dB/km (-0.0200 to -0.0020).
    pub atmos: [f32; 13],
    /// Bypass map generation elevation angles (EL_INDEX(0..11), bytes 1044-1091), deg (-1.000 to
    /// 45.000).
    pub el_index: [f32; 12],
    /// Transmitter frequency (TFREQ_MHZ, bytes 1092-1095), MHz (2700 to 3000).
    pub tfreq_mhz: i32,
    /// Point clutter suppression threshold (TCN) (BASE_DATA_TCN, bytes 1096-1099), dB (0.0 to
    /// 30.0).
    pub base_data_tcn: f32,
    /// Range unfolding overlay threshold (TOVER) (REFL_DATA_TOVER, bytes 1100-1103), dB (0.0 to
    /// 20.0).
    pub refl_data_tover: f32,
    /// Horizontal target system calibration (dBZ0) for long pulse (TAR_H_DBZ0_LP, bytes 1104-1107),
    /// dBZ (-65.00 to -45.00).
    pub tar_h_dbz0_lp: f32,
    /// Vertical target system calibration (dBZ0) for long pulse (TAR_V_DBZ0_LP, bytes 1108-1111),
    /// dBZ (-65.00 to -45.00).
    pub tar_v_dbz0_lp: f32,
    /// Initial system differential phase (INIT_PHI_DP, bytes 1112-1115), deg (0 to 359).
    pub init_phi_dp: i32,
    /// Normalized initial system differential phase (NORM_INIT_PHI_DP, bytes 1116-1119), deg (0 to
    /// 359).
    pub norm_init_phi_dp: i32,
    /// Matched filter loss for long pulse (LX_LP, bytes 1120-1123), dB (-3.00 to 0.00).
    pub lx_lp: f32,
    /// Matched filter loss for short pulse (LX_SP, bytes 1124-1127), dB (-3.00 to 0.00).
    pub lx_sp: f32,
    /// Hydrometeor refractivity factor |K|^2 (METEOR_PARAM, bytes 1128-1131), unitless (0.10 to
    /// 1.10).
    pub meteor_param: f32,
    /// Antenna beamwidth (BEAMWIDTH, bytes 1132-1135), deg. Spare in the Build 24.0 table; see
    /// the module documentation for the builds that carry it.
    pub beamwidth: f32,
    /// Antenna gain including radome (ANTENNA_GAIN, bytes 1136-1139), dB (43.00 to 47.00).
    pub antenna_gain: f32,
    /// Velocity check delta degrade limit (VEL_DEGRAD_LIMIT, bytes 1152-1155), m/s (0.5 to 2.0).
    pub vel_degrad_limit: f32,
    /// Spectrum width check delta degrade limit (WTH_DEGRAD_LIMIT, bytes 1156-1159), m/s (0.5 to
    /// 2.0).
    pub wth_degrad_limit: f32,
    /// Horizontal system noise temperature degrade limit (H_NOISETEMP_DGRAD_LIMIT, bytes
    /// 1160-1163), K (200.0 to 500.0).
    pub h_noisetemp_dgrad_limit: f32,
    /// Horizontal system noise temperature too-low limit (H_MIN_NOISETEMP, bytes 1164-1167), K
    /// (1 to 150).
    pub h_min_noisetemp: i32,
    /// Vertical system noise temperature degrade limit (V_NOISETEMP_DGRAD_LIMIT, bytes 1168-1171),
    /// K (200.0 to 500.0).
    pub v_noisetemp_dgrad_limit: f32,
    /// Vertical system noise temperature too-low limit (V_MIN_NOISETEMP, bytes 1172-1175), K (1 to
    /// 150).
    pub v_min_noisetemp: i32,
    /// Klystron output target consistency degrade limit (KLY_DEGRADE_LIMIT, bytes 1176-1179), dB
    /// (1.0 to 10.0).
    pub kly_degrade_limit: f32,
    /// COHO power at A1J4 (TS_COHO, bytes 1180-1183), dBm (23.00 to 29.00).
    pub ts_coho: f32,
    /// AME horizontal test signal power (H_TS_CW, bytes 1184-1187), dBm (0.00 to 30.00).
    pub h_ts_cw: f32,
    /// STALO power at A1J2 (TS_STALO, bytes 1196-1199), dBm (12.00 to 18.00).
    pub ts_stalo: f32,
    /// AME noise source horizontal excess noise ratio (AME_H_NOISE_ENR, bytes 1200-1203), dB
    /// (10.00 to 35.00).
    pub ame_h_noise_enr: f32,
    /// Maximum transmitter peak power alarm level (XMTR_PEAK_PWR_HIGH_LIMIT, bytes 1204-1207), kW
    /// (500.00 to 950.00).
    pub xmtr_peak_pwr_high_limit: f32,
    /// Minimum transmitter peak power alarm level (XMTR_PEAK_PWR_LOW_LIMIT, bytes 1208-1211), kW
    /// (200.00 to 700.00).
    pub xmtr_peak_pwr_low_limit: f32,
    /// Difference between computed and target horizontal dBZ0 limit (H_DBZ0_DELTA_LIMIT, bytes
    /// 1212-1215), dB (1.0 to 10.0).
    pub h_dbz0_delta_limit: f32,
    /// Bypass map generator noise threshold (THRESHOLD1, bytes 1216-1219), dB (0.0 to 36.0).
    pub threshold1: f32,
    /// Bypass map generator rejection ratio threshold (THRESHOLD2, bytes 1220-1223), dB (0.0 to
    /// 10.0).
    pub threshold2: f32,
    /// Clutter suppression degrade limit (CLUT_SUPP_DGRAD_LIM, bytes 1224-1227), dB (20.0 to 50.0).
    pub clut_supp_dgrad_lim: f32,
    /// True range at the start of the first range bin (RANGE0_VALUE, bytes 1232-1235), km (0.000 to
    /// 3.000).
    pub range0_value: f32,
    /// Scale factor converting transmitter power byte data to watts (XMTR_PWR_MTR_SCALE, bytes
    /// 1236-1239), the value of the LSB of the power measurement, W (0.0000100 to 0.0015000).
    pub xmtr_pwr_mtr_scale: f32,
    /// Difference between computed and target vertical dBZ0 limit (V_DBZ0_DELTA_LIMIT, bytes
    /// 1240-1243), dB (1.0 to 10.0).
    pub v_dbz0_delta_limit: f32,
    /// Horizontal target system calibration (dBZ0) for short pulse (TAR_H_DBZ0_SP, bytes
    /// 1244-1247), dBZ (-58.00 to -38.00).
    pub tar_h_dbz0_sp: f32,
    /// Vertical target system calibration (dBZ0) for short pulse (TAR_V_DBZ0_SP, bytes 1248-1251),
    /// dBZ (-58.00 to -38.00).
    pub tar_v_dbz0_sp: f32,
    /// Site PRF set (DELTAPRF, bytes 1252-1255): 1 = A, 2 = B, 3 = C, 4 = D, 5 = E.
    pub deltaprf: i32,
    /// Pulse width of the transmitter output in short pulse (TAU_SP, bytes 1264-1267), ns (1000 to
    /// 2000).
    pub tau_sp: i32,
    /// Pulse width of the transmitter output in long pulse (TAU_LP, bytes 1268-1271), ns (3000 to
    /// 6000).
    pub tau_lp: i32,
    /// Number of 1/4 km bins of corrupted data at the end of a sweep (1 to 10) (NC_DEAD_VALUE,
    /// bytes 1272-1275).
    pub nc_dead_value: i32,
    /// RF drive pulse width in short pulse (TAU_RF_SP, bytes 1276-1279), ns (500 to 2000).
    pub tau_rf_sp: i32,
    /// RF drive pulse width in long pulse (TAU_RF_LP, bytes 1280-1283), ns (3000 to 6000).
    pub tau_rf_lp: i32,
    /// Clutter map boundary elevation between segments 1 and 2 (SEG1LIM, bytes 1284-1287), deg
    /// (0.50 to 3.00).
    pub seg1lim: f32,
    /// Site latitude (SLATSEC, bytes 1288-1291), seconds (0.0000 to 59.9999).
    pub slatsec: f32,
    /// Site longitude (SLONSEC, bytes 1292-1295), seconds (0.0000 to 59.9999).
    pub slonsec: f32,
    /// Site latitude (SLATDEG, bytes 1300-1303), degrees (0 to 89).
    pub slatdeg: i32,
    /// Site latitude (SLATMIN, bytes 1304-1307), minutes (0 to 59).
    pub slatmin: i32,
    /// Site longitude (SLONDEG, bytes 1308-1311), degrees (0 to 179).
    pub slondeg: i32,
    /// Site longitude (SLONMIN, bytes 1312-1315), minutes (0 to 59).
    pub slonmin: i32,
    /// Site latitude direction (SLATDIR, bytes 1316-1319), "N" or "S".
    pub slatdir: String,
    /// Site longitude direction (SLONDIR, bytes 1320-1323), "E" or "W".
    pub slondir: String,
    /// Digital receiver clock frequency (DIG_RCVR_CLOCK_FREQ, bytes 2500-2507), MHz (50.0 to
    /// 250.0).
    pub dig_rcvr_clock_freq: f64,
    /// COHO frequency (COHO_FREQ, bytes 2508-2515), MHz (0.0 to 100.0).
    pub coho_freq: f64,
    /// Azimuth boresight correction factor (AZ_CORRECTION_FACTOR, bytes 8360-8363), deg (-1.000 to
    /// 1.000).
    pub az_correction_factor: f32,
    /// Elevation boresight correction factor (EL_CORRECTION_FACTOR, bytes 8364-8367), deg
    /// (-1.000 to 1.000).
    pub el_correction_factor: f32,
    /// Site name designation (ICAO identifier) (SITE_NAME, bytes 8368-8371).
    pub site_name: String,
    /// Minimum elevation angle as a two's complement binary angle (-7281 to 7281); multiply by
    /// 360/65536 for degrees (-39.99573 to 39.99573) (ANT_MANUAL_SETUP.IELMIN, bytes 8372-8375).
    pub ant_manual_setup_ielmin: i32,
    /// Maximum elevation angle as a binary angle (0 to 40049); multiply by 360/65536 for degrees
    /// (0.00000 to 219.99573) (ANT_MANUAL_SETUP.IELMAX, bytes 8376-8379).
    pub ant_manual_setup_ielmax: i32,
    /// Maximum azimuth velocity (ANT_MANUAL_SETUP.FAZVELMAX, bytes 8380-8383), deg/s (0 to 100).
    pub ant_manual_setup_fazvelmax: i32,
    /// Maximum elevation velocity (ANT_MANUAL_SETUP.FELVELMAX, bytes 8384-8387), deg/s (0 to 48).
    pub ant_manual_setup_felvelmax: i32,
    /// Site ground height above sea level (ANT_MANUAL_SETUP.IGND_HGT, bytes 8388-8391), m (-100 to
    /// 12000).
    pub ant_manual_setup_ignd_hgt: i32,
    /// Site radar height above ground (ANT_MANUAL_SETUP.IRAD_HGT, bytes 8392-8395), m (0 to 1000).
    pub ant_manual_setup_irad_hgt: i32,
    /// Azimuth motor positive sustaining drive (AZ_POS_SUSTAIN_DRIVE, bytes 8396-8399), unitless
    /// (0.00 to 7.00).
    pub az_pos_sustain_drive: f32,
    /// Azimuth motor negative sustaining drive (AZ_NEG_SUSTAIN_DRIVE, bytes 8400-8403), unitless
    /// (-7.00 to 0.00).
    pub az_neg_sustain_drive: f32,
    /// Initial estimate for the azimuth positive drive slope (AZ_NOM_POS_DRIVE_SLOPE, bytes
    /// 8404-8407), unitless (0.00 to 3.00).
    pub az_nom_pos_drive_slope: f32,
    /// Initial estimate for the azimuth negative drive slope (AZ_NOM_NEG_DRIVE_SLOPE, bytes
    /// 8408-8411), unitless (0.00 to 3.00).
    pub az_nom_neg_drive_slope: f32,
    /// Azimuth velocity feedback slope (AZ_FEEDBACK_SLOPE, bytes 8412-8415), unitless (0.000 to
    /// 15.000).
    pub az_feedback_slope: f32,
    /// Elevation motor positive sustaining drive (EL_POS_SUSTAIN_DRIVE, bytes 8416-8419), unitless
    /// (0.00 to 7.00).
    pub el_pos_sustain_drive: f32,
    /// Elevation motor negative sustaining drive (EL_NEG_SUSTAIN_DRIVE, bytes 8420-8423), unitless
    /// (-7.00 to 0.00).
    pub el_neg_sustain_drive: f32,
    /// Initial estimate for the elevation positive drive slope (EL_NOM_POS_DRIVE_SLOPE, bytes
    /// 8424-8427), unitless (0.00 to 3.00).
    pub el_nom_pos_drive_slope: f32,
    /// Initial estimate for the elevation negative drive slope (EL_NOM_NEG_DRIVE_SLOPE, bytes
    /// 8428-8431), unitless (0.00 to 3.00).
    pub el_nom_neg_drive_slope: f32,
    /// Elevation velocity feedback slope (EL_FEEDBACK_SLOPE, bytes 8432-8435), unitless (0.000 to
    /// 15.00).
    pub el_feedback_slope: f32,
    /// Slope for the first interval of the elevation position feedback curve (EL_FIRST_SLOPE, bytes
    /// 8436-8439), unitless (0.50 to 20.00).
    pub el_first_slope: f32,
    /// Slope for the second interval of the elevation position feedback curve (EL_SECOND_SLOPE,
    /// bytes 8440-8443), unitless (0.10 to 20.00).
    pub el_second_slope: f32,
    /// Slope for the third interval of the elevation position feedback curve (EL_THIRD_SLOPE, bytes
    /// 8444-8447), unitless (0.00 to 20.00).
    pub el_third_slope: f32,
    /// Neutral droop angle (EL_DROOP_POS, bytes 8448-8451), deg (-360.00 to 360.00).
    pub el_droop_pos: f32,
    /// 90 degree off-neutral drive (EL_OFF_NEUTRAL_DRIVE, bytes 8452-8455), unitless (-7.00 to
    /// 7.00).
    pub el_off_neutral_drive: f32,
    /// Azimuth moment of inertia (AZ_INERTIA, bytes 8456-8459), unitless (0.5 to 7.0).
    pub az_inertia: f32,
    /// Elevation moment of inertia (EL_INERTIA, bytes 8460-8463), unitless (0.5 to 7.0).
    pub el_inertia: f32,
    /// Azimuth stow angle for encoder alignment (AZ_STOW_ANGLE, bytes 8496-8499), deg (0.000 to
    /// 359.999).
    pub az_stow_angle: f32,
    /// Elevation stow angle for encoder alignment (EL_STOW_ANGLE, bytes 8500-8503), deg
    /// (-180.000 to 180.000).
    pub el_stow_angle: f32,
    /// Azimuth encoder alignment ETU angle (AZ_ENCODER_ALIGNMENT, bytes 8504-8507), deg (0.000 to
    /// 359.999).
    pub az_encoder_alignment: f32,
    /// Elevation encoder alignment ETU angle (EL_ENCODER_ALIGNMENT, bytes 8508-8511), deg
    /// (-180.000 to 180.000).
    pub el_encoder_alignment: f32,
    /// Refined park in use (REFINED_PARK, bytes 8688-8691).
    pub refined_park: Option<bool>,
    /// Waveguide length (RVP8NV.IWAVEGUIDE_LENGTH, bytes 8696-8699), m (0 to 1000).
    pub rvp8nv_iwaveguide_length: i32,
    /// Vertical receiver noise normalization for the elevation sectors of H_RNSCALE
    /// (V_RNSCALE(0..12)), unitless (1.000 to 1.800). Elements 0 to 10 are bytes 8700-8743 and
    /// elements 11 and 12 bytes 8752-8759, around VEL_DATA_TOVER and WIDTH_DATA_TOVER.
    pub v_rnscale: [f32; 13],
    /// Velocity unfolding overlay threshold (VEL_DATA_TOVER, bytes 8744-8747), dB (0.0 to 20.0).
    pub vel_data_tover: f32,
    /// Spectrum width unfolding overlay threshold (WIDTH_DATA_TOVER, bytes 8748-8751), dB (0.0 to
    /// 20.0).
    pub width_data_tover: f32,
    /// Start range for the first Doppler radial (DOPPLER_RANGE_START, bytes 8764-8767), km
    /// (-32.768 to 32.768).
    pub doppler_range_start: f32,
    /// Maximum index for the EL_INDEX parameters (0 to 11) (MAX_EL_INDEX, bytes 8768-8771).
    pub max_el_index: i32,
    /// Clutter map boundary elevation between segments 2 and 3 (SEG2LIM, bytes 8772-8775), deg
    /// (0.80 to 4.50).
    pub seg2lim: f32,
    /// Clutter map boundary elevation between segments 3 and 4 (SEG3LIM, bytes 8776-8779), deg
    /// (1.00 to 6.00).
    pub seg3lim: f32,
    /// Clutter map boundary elevation between segments 4 and 5 (SEG4LIM, bytes 8780-8783), deg
    /// (1.00 to 8.00).
    pub seg4lim: f32,
    /// Number of elevation segments in the ORDA clutter map (1 to 5) (NBR_EL_SEGMENTS, bytes
    /// 8784-8787).
    pub nbr_el_segments: i32,
    /// Horizontal receiver noise for long pulse (H_NOISE_LONG, bytes 8788-8791), dBm (-95.0 to
    /// -80.0).
    pub h_noise_long: f32,
    /// Antenna noise temperature (ANT_NOISE_TEMP, bytes 8792-8795), K (30.0 to 200.0).
    pub ant_noise_temp: f32,
    /// Horizontal receiver noise for short pulse (H_NOISE_SHORT, bytes 8796-8799), dBm (-90.0 to
    /// -75.0).
    pub h_noise_short: f32,
    /// Horizontal receiver noise tolerance (H_NOISE_TOLERANCE, bytes 8800-8803), dB (0.0 to 6.0).
    pub h_noise_tolerance: f32,
    /// Minimum horizontal dynamic range (MIN_H_DYN_RANGE, bytes 8804-8807), dB (85.0 to 95.0).
    pub min_h_dyn_range: f32,
    /// Auxiliary generator installed (FAA only) (GEN_INSTALLED, bytes 8808-8811).
    pub gen_installed: Option<bool>,
    /// Auxiliary generator automatic exercise enabled (FAA only) (GEN_EXERCISE, bytes 8812-8815).
    pub gen_exercise: Option<bool>,
    /// Vertical receiver noise tolerance (V_NOISE_TOLERANCE, bytes 8816-8819), dB (0.0 to 6.0).
    pub v_noise_tolerance: f32,
    /// Minimum vertical dynamic range (MIN_V_DYN_RANGE, bytes 8820-8823), dB (85.0 to 95.0).
    pub min_v_dyn_range: f32,
    /// System differential reflectivity offset degrade limit (ZDR_OFFSET_DGRAD_LIM, bytes
    /// 8824-8827), dB (0.0 to 10.0).
    pub zdr_offset_dgrad_lim: f32,
    /// Baseline system differential reflectivity offset (BASELINE_ZDR_OFFSET, bytes 8828-8831), dB
    /// (-10.0000 to 10.0000). The table assigns bytes 8828-8843 to this Real*4; the value is read
    /// from the first four and bytes 8832-8843 are ignored (zero in most corpus files).
    pub baseline_zdr_offset: f32,
    /// Vertical receiver noise for long pulse (V_NOISE_LONG, bytes 8844-8847), dBm (-95.0 to
    /// -80.0).
    pub v_noise_long: f32,
    /// Vertical receiver noise for short pulse (V_NOISE_SHORT, bytes 8848-8851), dBm (-90.0 to
    /// -75.0).
    pub v_noise_short: f32,
    /// ZDR unfolding overlay threshold (ZDR_DATA_TOVER, bytes 8852-8855), dB (-10.00 to 10.00).
    pub zdr_data_tover: f32,
    /// PHI unfolding overlay threshold (PHI_DATA_TOVER, bytes 8856-8859), dB (-10.00 to 10.00).
    pub phi_data_tover: f32,
    /// RHO unfolding overlay threshold (RHO_DATA_TOVER, bytes 8860-8863), dB (-10.00 to 10.00).
    pub rho_data_tover: f32,
    /// STALO power degrade limit (STALO_POWER_DGRAD_LIMIT, bytes 8864-8867), V (0.00 to 1.00).
    pub stalo_power_dgrad_limit: f32,
    /// STALO power maintenance limit (STALO_POWER_MAINT_LIMIT, bytes 8868-8871), V (0.00 to 1.00).
    pub stalo_power_maint_limit: f32,
    /// Minimum horizontal power sense (MIN_H_PWR_SENSE, bytes 8872-8875), dBm (70.00 to 90.00).
    pub min_h_pwr_sense: f32,
    /// Minimum vertical power sense (MIN_V_PWR_SENSE, bytes 8876-8879), dBm (70.00 to 90.00).
    pub min_v_pwr_sense: f32,
    /// Horizontal power sense calibration offset (H_PWR_SENSE_OFFSET, bytes 8880-8883), dB
    /// (-100.00 to -50.00).
    pub h_pwr_sense_offset: f32,
    /// Vertical power sense calibration offset (V_PWR_SENSE_OFFSET, bytes 8884-8887), dB
    /// (-100.00 to -50.00).
    pub v_pwr_sense_offset: f32,
    /// Power sense gain reference value (PS_GAIN_REF, bytes 8888-8891), dB (-40.00 to -20.00).
    pub ps_gain_ref: f32,
    /// RF pallet broadband loss (RF_PALLET_BROAD_LOSS, bytes 8892-8895), dB (-10.00 to 0.00).
    pub rf_pallet_broad_loss: f32,
    /// AME power supply tolerance (AME_PS_TOLERANCE, bytes 8960-8963), % (0.0 to 20.0).
    pub ame_ps_tolerance: f32,
    /// Maximum AME internal alarm temperature (AME_MAX_TEMP, bytes 8964-8967), deg C (0.0 to 65.0).
    pub ame_max_temp: f32,
    /// Minimum AME internal alarm temperature (AME_MIN_TEMP, bytes 8968-8971), deg C (-10.0 to
    /// 20.0).
    pub ame_min_temp: f32,
    /// Maximum AME receiver module alarm temperature (RCVR_MOD_MAX_TEMP, bytes 8972-8975), deg C
    /// (0.0 to 65.0).
    pub rcvr_mod_max_temp: f32,
    /// Minimum AME receiver module alarm temperature (RCVR_MOD_MIN_TEMP, bytes 8976-8979), deg C
    /// (-10.0 to 20.0).
    pub rcvr_mod_min_temp: f32,
    /// Maximum AME BITE module alarm temperature (BITE_MOD_MAX_TEMP, bytes 8980-8983), deg C
    /// (0.0 to 75.0).
    pub bite_mod_max_temp: f32,
    /// Minimum AME BITE module alarm temperature (BITE_MOD_MIN_TEMP, bytes 8984-8987), deg C
    /// (-10.0 to 20.0).
    pub bite_mod_min_temp: f32,
    /// Default (H+V) microwave assembly phase shifter position (0 to 60000) (DEFAULT_POLARIZATION,
    /// bytes 8988-8991).
    pub default_polarization: i32,
    /// TR limiter degrade limit (TR_LIMIT_DGRAD_LIMIT, bytes 8992-8995), V (0.00 to 1.00).
    pub tr_limit_dgrad_limit: f32,
    /// TR limiter failure limit (TR_LIMIT_FAIL_LIMIT, bytes 8996-8999), V (0.00 to 1.00).
    pub tr_limit_fail_limit: f32,
    /// Whether the RF pallet stepper motor is enabled (RFP_STEPPER_ENABLED, bytes 9000-9003).
    pub rfp_stepper_enabled: Option<bool>,
    /// AME Peltier current tolerance (AME_CURRENT_TOLERANCE, bytes 9008-9011), % (0.0 to 100.0).
    pub ame_current_tolerance: f32,
    /// Horizontal-only microwave assembly phase shifter position (0 to 60000) (H_ONLY_POLARIZATION,
    /// bytes 9012-9015).
    pub h_only_polarization: i32,
    /// Vertical-only microwave assembly phase shifter position (0 to 60000) (V_ONLY_POLARIZATION,
    /// bytes 9016-9019).
    pub v_only_polarization: i32,
    /// Sun measurement bias (SUN_BIAS, bytes 9028-9031), dB (-5.00 to 5.00).
    pub sun_bias: f32,
    /// Low equipment shelter temperature warning limit (A_MIN_SHELTER_TEMP_WARN, bytes 9032-9035),
    /// deg C (-20.00 to 20.00).
    pub a_min_shelter_temp_warn: f32,
    /// Power meter zero bias voltage (POWER_METER_ZERO, bytes 9036-9039), V (-10.00 to 10.00).
    pub power_meter_zero: f32,
    /// Expected value of the RDA transmit bias (TXB) (TXB_BASELINE, bytes 9040-9043), dB (-1.000 to
    /// 1.000).
    pub txb_baseline: f32,
    /// Threshold on the difference between a measured transmit bias and TXB_BASELINE above which
    /// the RDA sets an alarm (TXB_ALARM_THRESH, bytes 9044-9047), dB (0 to 5.000).
    pub txb_alarm_thresh: f32,
    /// Normal TPS power time (NORMAL_TPS_POWER_TIME, bytes 9048-9051), s (0 to 3600).
    pub normal_tps_power_time: i32,
}

impl RdaAdaptationData {
    /// Decode a message body (the bytes after the 16-byte message header, segments joined) in the
    /// Build 24.0 layout. Needs at least 9468 bytes.
    pub fn decode(body: &[u8]) -> Result<Self> {
        crate::require_len(body, 0, RDA_ADAPTATION_DATA_LEN, "RDA adaptation data")?;
        Ok(Self {
            adap_file_name: crate::ascii_trim(&body[0..12]),
            adap_format: crate::ascii_trim(&body[12..16]),
            adap_revision: crate::ascii_trim(&body[16..20]),
            adap_date: crate::ascii_trim(&body[20..32]),
            adap_time: crate::ascii_trim(&body[32..44]),
            lower_pre_limit: f32_at(body, 44),
            az_lat: f32_at(body, 48),
            upper_pre_limit: f32_at(body, 52),
            el_lat: f32_at(body, 56),
            parkaz: f32_at(body, 60),
            parkel: f32_at(body, 64),
            a_fuel_conv: std::array::from_fn(|index| f32_at(body, 68 + index * 4)),
            a_min_shelter_temp: f32_at(body, 112),
            a_max_shelter_temp: f32_at(body, 116),
            a_min_shelter_ac_temp_diff: f32_at(body, 120),
            a_max_xmtr_air_temp: f32_at(body, 124),
            a_max_rad_temp: f32_at(body, 128),
            a_max_rad_temp_rise: f32_at(body, 132),
            lower_dead_limit: f32_at(body, 136),
            upper_dead_limit: f32_at(body, 140),
            a_min_gen_room_temp: f32_at(body, 148),
            a_max_gen_room_temp: f32_at(body, 152),
            spip_5v_reg_lim: f32_at(body, 156),
            spip_15v_reg_lim: f32_at(body, 160),
            rpg_co_located: true_false(body, 176),
            spec_filter_installed: true_false(body, 180),
            tps_installed: true_false(body, 184),
            rms_installed: true_false(body, 188),
            a_hvdl_tst_int: i32_at(body, 192),
            a_rpg_lt_int: i32_at(body, 196),
            a_min_stab_util_pwr_time: i32_at(body, 200),
            a_gen_auto_exer_interval: i32_at(body, 204),
            a_util_pwr_sw_req_interval: i32_at(body, 208),
            a_low_fuel_level: f32_at(body, 212),
            config_chan_number: i32_at(body, 216),
            redundant_chan_config: i32_at(body, 224),
            atten_table: std::array::from_fn(|index| f32_at(body, 228 + index * 4)),
            path_losses_7: f32_at(body, 668),
            path_losses_13: f32_at(body, 692),
            path_losses_28: f32_at(body, 752),
            h_coupler_xmt_loss: f32_at(body, 756),
            path_losses_32: f32_at(body, 768),
            path_losses_33: f32_at(body, 772),
            path_losses_35: f32_at(body, 780),
            path_losses_39: f32_at(body, 796),
            path_losses_40: f32_at(body, 800),
            path_losses_42: f32_at(body, 808),
            path_losses_43: f32_at(body, 812),
            path_losses_44: f32_at(body, 816),
            path_losses_45: f32_at(body, 820),
            path_losses_46: f32_at(body, 824),
            path_losses_47: f32_at(body, 828),
            h_coupler_cw_loss: f32_at(body, 832),
            v_coupler_xmt_loss: f32_at(body, 836),
            ame_ts_bias: f32_at(body, 844),
            path_losses_52: f32_at(body, 848),
            v_coupler_cw_loss: f32_at(body, 852),
            pwr_sense_bias: f32_at(body, 864),
            ame_v_noise_enr: f32_at(body, 868),
            path_losses_58: f32_at(body, 872),
            path_losses_59: f32_at(body, 876),
            path_losses_60: f32_at(body, 880),
            path_losses_61: f32_at(body, 884),
            path_losses_63: f32_at(body, 892),
            path_losses_64: f32_at(body, 896),
            path_losses_65: f32_at(body, 900),
            path_losses_66: f32_at(body, 904),
            path_losses_67: f32_at(body, 908),
            path_losses_68: f32_at(body, 912),
            chan_cal_diff: f32_at(body, 920),
            v_ts_cw: f32_at(body, 936),
            h_rnscale: std::array::from_fn(|index| f32_at(body, 940 + index * 4)),
            atmos: std::array::from_fn(|index| f32_at(body, 992 + index * 4)),
            el_index: std::array::from_fn(|index| f32_at(body, 1044 + index * 4)),
            tfreq_mhz: i32_at(body, TFREQ_MHZ_OFFSET),
            base_data_tcn: f32_at(body, 1096),
            refl_data_tover: f32_at(body, 1100),
            tar_h_dbz0_lp: f32_at(body, 1104),
            tar_v_dbz0_lp: f32_at(body, 1108),
            init_phi_dp: i32_at(body, 1112),
            norm_init_phi_dp: i32_at(body, 1116),
            lx_lp: f32_at(body, 1120),
            lx_sp: f32_at(body, 1124),
            meteor_param: f32_at(body, 1128),
            beamwidth: f32_at(body, BEAMWIDTH_OFFSET),
            antenna_gain: f32_at(body, ANTENNA_GAIN_OFFSET),
            vel_degrad_limit: f32_at(body, 1152),
            wth_degrad_limit: f32_at(body, 1156),
            h_noisetemp_dgrad_limit: f32_at(body, 1160),
            h_min_noisetemp: i32_at(body, 1164),
            v_noisetemp_dgrad_limit: f32_at(body, 1168),
            v_min_noisetemp: i32_at(body, 1172),
            kly_degrade_limit: f32_at(body, 1176),
            ts_coho: f32_at(body, 1180),
            h_ts_cw: f32_at(body, 1184),
            ts_stalo: f32_at(body, 1196),
            ame_h_noise_enr: f32_at(body, 1200),
            xmtr_peak_pwr_high_limit: f32_at(body, 1204),
            xmtr_peak_pwr_low_limit: f32_at(body, 1208),
            h_dbz0_delta_limit: f32_at(body, 1212),
            threshold1: f32_at(body, 1216),
            threshold2: f32_at(body, 1220),
            clut_supp_dgrad_lim: f32_at(body, 1224),
            range0_value: f32_at(body, 1232),
            xmtr_pwr_mtr_scale: f32_at(body, 1236),
            v_dbz0_delta_limit: f32_at(body, 1240),
            tar_h_dbz0_sp: f32_at(body, 1244),
            tar_v_dbz0_sp: f32_at(body, 1248),
            deltaprf: i32_at(body, 1252),
            tau_sp: i32_at(body, 1264),
            tau_lp: i32_at(body, 1268),
            nc_dead_value: i32_at(body, 1272),
            tau_rf_sp: i32_at(body, 1276),
            tau_rf_lp: i32_at(body, 1280),
            seg1lim: f32_at(body, 1284),
            slatsec: f32_at(body, 1288),
            slonsec: f32_at(body, 1292),
            slatdeg: i32_at(body, 1300),
            slatmin: i32_at(body, 1304),
            slondeg: i32_at(body, 1308),
            slonmin: i32_at(body, 1312),
            slatdir: crate::ascii_trim(&body[1316..1320]),
            slondir: crate::ascii_trim(&body[1320..1324]),
            dig_rcvr_clock_freq: f64_at(body, 2500),
            coho_freq: f64_at(body, 2508),
            az_correction_factor: f32_at(body, 8360),
            el_correction_factor: f32_at(body, 8364),
            site_name: crate::ascii_trim(&body[8368..8372]),
            ant_manual_setup_ielmin: i32_at(body, 8372),
            ant_manual_setup_ielmax: i32_at(body, 8376),
            ant_manual_setup_fazvelmax: i32_at(body, 8380),
            ant_manual_setup_felvelmax: i32_at(body, 8384),
            ant_manual_setup_ignd_hgt: i32_at(body, 8388),
            ant_manual_setup_irad_hgt: i32_at(body, 8392),
            az_pos_sustain_drive: f32_at(body, 8396),
            az_neg_sustain_drive: f32_at(body, 8400),
            az_nom_pos_drive_slope: f32_at(body, 8404),
            az_nom_neg_drive_slope: f32_at(body, 8408),
            az_feedback_slope: f32_at(body, 8412),
            el_pos_sustain_drive: f32_at(body, 8416),
            el_neg_sustain_drive: f32_at(body, 8420),
            el_nom_pos_drive_slope: f32_at(body, 8424),
            el_nom_neg_drive_slope: f32_at(body, 8428),
            el_feedback_slope: f32_at(body, 8432),
            el_first_slope: f32_at(body, 8436),
            el_second_slope: f32_at(body, 8440),
            el_third_slope: f32_at(body, 8444),
            el_droop_pos: f32_at(body, 8448),
            el_off_neutral_drive: f32_at(body, 8452),
            az_inertia: f32_at(body, 8456),
            el_inertia: f32_at(body, 8460),
            az_stow_angle: f32_at(body, 8496),
            el_stow_angle: f32_at(body, 8500),
            az_encoder_alignment: f32_at(body, 8504),
            el_encoder_alignment: f32_at(body, 8508),
            refined_park: true_false(body, 8688),
            rvp8nv_iwaveguide_length: i32_at(body, 8696),
            v_rnscale: std::array::from_fn(|index| {
                let first = if index < 11 { 8700 } else { 8752 - 11 * 4 };
                f32_at(body, first + index * 4)
            }),
            vel_data_tover: f32_at(body, 8744),
            width_data_tover: f32_at(body, 8748),
            doppler_range_start: f32_at(body, 8764),
            max_el_index: i32_at(body, 8768),
            seg2lim: f32_at(body, 8772),
            seg3lim: f32_at(body, 8776),
            seg4lim: f32_at(body, 8780),
            nbr_el_segments: i32_at(body, 8784),
            h_noise_long: f32_at(body, 8788),
            ant_noise_temp: f32_at(body, 8792),
            h_noise_short: f32_at(body, 8796),
            h_noise_tolerance: f32_at(body, 8800),
            min_h_dyn_range: f32_at(body, 8804),
            gen_installed: true_false(body, 8808),
            gen_exercise: true_false(body, 8812),
            v_noise_tolerance: f32_at(body, 8816),
            min_v_dyn_range: f32_at(body, 8820),
            zdr_offset_dgrad_lim: f32_at(body, 8824),
            baseline_zdr_offset: f32_at(body, 8828),
            v_noise_long: f32_at(body, 8844),
            v_noise_short: f32_at(body, 8848),
            zdr_data_tover: f32_at(body, 8852),
            phi_data_tover: f32_at(body, 8856),
            rho_data_tover: f32_at(body, 8860),
            stalo_power_dgrad_limit: f32_at(body, 8864),
            stalo_power_maint_limit: f32_at(body, 8868),
            min_h_pwr_sense: f32_at(body, 8872),
            min_v_pwr_sense: f32_at(body, 8876),
            h_pwr_sense_offset: f32_at(body, 8880),
            v_pwr_sense_offset: f32_at(body, 8884),
            ps_gain_ref: f32_at(body, 8888),
            rf_pallet_broad_loss: f32_at(body, 8892),
            ame_ps_tolerance: f32_at(body, 8960),
            ame_max_temp: f32_at(body, 8964),
            ame_min_temp: f32_at(body, 8968),
            rcvr_mod_max_temp: f32_at(body, 8972),
            rcvr_mod_min_temp: f32_at(body, 8976),
            bite_mod_max_temp: f32_at(body, 8980),
            bite_mod_min_temp: f32_at(body, 8984),
            default_polarization: i32_at(body, 8988),
            tr_limit_dgrad_limit: f32_at(body, 8992),
            tr_limit_fail_limit: f32_at(body, 8996),
            rfp_stepper_enabled: true_false(body, 9000),
            ame_current_tolerance: f32_at(body, 9008),
            h_only_polarization: i32_at(body, 9012),
            v_only_polarization: i32_at(body, 9016),
            sun_bias: f32_at(body, 9028),
            a_min_shelter_temp_warn: f32_at(body, 9032),
            power_meter_zero: f32_at(body, 9036),
            txb_baseline: f32_at(body, 9040),
            txb_alarm_thresh: f32_at(body, 9044),
            normal_tps_power_time: i32_at(body, 9048),
        })
    }

    /// Site latitude in decimal degrees (positive north) from SLATDEG, SLATMIN, SLATSEC and
    /// SLATDIR.
    pub fn latitude(&self) -> f64 {
        let magnitude = f64::from(self.slatdeg)
            + f64::from(self.slatmin) / 60.0
            + f64::from(self.slatsec) / 3600.0;
        if self.slatdir.starts_with('S') {
            -magnitude
        } else {
            magnitude
        }
    }

    /// Site longitude in decimal degrees (positive east) from SLONDEG, SLONMIN, SLONSEC and
    /// SLONDIR.
    pub fn longitude(&self) -> f64 {
        let magnitude = f64::from(self.slondeg)
            + f64::from(self.slonmin) / 60.0
            + f64::from(self.slonsec) / 3600.0;
        if self.slondir.starts_with('W') {
            -magnitude
        } else {
            magnitude
        }
    }

    /// ANT_MANUAL_SETUP.IELMIN in degrees (Table XV note 7).
    pub fn manual_setup_min_elevation(&self) -> f64 {
        binary_angle_degrees(self.ant_manual_setup_ielmin)
    }

    /// ANT_MANUAL_SETUP.IELMAX in degrees (Table XV note 7).
    pub fn manual_setup_max_elevation(&self) -> f64 {
        binary_angle_degrees(self.ant_manual_setup_ielmax)
    }
}

/// A two's complement binary angle times 360/2^16 (Table XV note 7).
fn binary_angle_degrees(raw: i32) -> f64 {
    f64::from(raw) * 360.0 / 65536.0
}

fn f32_at(body: &[u8], offset: usize) -> f32 {
    crate::be_f32(body, offset)
}

fn i32_at(body: &[u8], offset: usize) -> i32 {
    crate::be_u32(body, offset) as i32
}

fn f64_at(body: &[u8], offset: usize) -> f64 {
    let mut bytes = [0; 8];
    bytes.copy_from_slice(&body[offset..offset + 8]);
    f64::from_be_bytes(bytes)
}

/// A 4-byte "T" or "F" string (Table XV note 15).
fn true_false(body: &[u8], offset: usize) -> Option<bool> {
    match body[offset] {
        b'T' => Some(true),
        b'F' => Some(false),
        _ => None,
    }
}

/// Walker hook: the typed body for message 18. Legacy RDA bodies are yielded
/// unparsed.
pub(crate) fn message_body<'a>(
    header: &MessageHeader,
    body: Cow<'a, [u8]>,
) -> Result<MessageBody<'a>> {
    match RdaSystem::from_channels(header.channels) {
        RdaSystem::Orda => RdaAdaptationData::decode(&body)
            .map(|decoded| MessageBody::Adaptation(Box::new(decoded))),
        RdaSystem::Legacy => Ok(MessageBody::Unparsed(body)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_angles_follow_note_7() {
        assert!((binary_angle_degrees(-7281) + 39.99573).abs() < 1e-5);
        assert!((binary_angle_degrees(40049) - 219.99573).abs() < 1e-5);
    }
}
