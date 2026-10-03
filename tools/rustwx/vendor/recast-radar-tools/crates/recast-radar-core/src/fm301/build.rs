//! Building the FM301 group tree (design note sections 1, 11, 12 and 14).

use std::borrow::Cow;
use std::sync::Arc;

use chrono::{DateTime, Datelike, Timelike, Utc};

use super::{
    ArrayRef, ExtraAttrs, FieldSource, FirstDim, Flavor, Group, Passthrough, RowOrder, Values,
    Variable, ViewError, ViewOptions, ViewWarning, VolumeView,
};
use crate::model::{
    ArrayBuf, AttrValue, ExtraVariable, Field, FieldAttrs, FieldData, FloatCoding, FloatWidth,
    GateMapping, IntCoding, LinearTransform, PackedInt, RangeCoord, RayAlignment, Scalar,
    SourceFormat, Sweep, SweepMode, Volume, floor_to_second,
};

type Attrs<'a> = Vec<(Cow<'a, str>, AttrValue)>;

/// Build the FM301 group view of `volume`. `extra` contributes
/// format-specific root and sweep attributes. Building is O(rays + fields) per
/// sweep plus one sort of ray indices when the requested ray order differs
/// from storage order; no gate data is copied.
pub fn volume_view<'a>(
    volume: &'a Volume,
    options: ViewOptions,
    extra: Option<&'a dyn ExtraAttrs>,
) -> Result<VolumeView<'a>, ViewError> {
    let mut builder = Builder {
        volume,
        options,
        extra,
        warnings: Vec::new(),
    };
    let root = builder.root()?;
    Ok(VolumeView {
        root,
        warnings: builder.warnings,
        volume,
    })
}

struct Builder<'a> {
    volume: &'a Volume,
    options: ViewOptions,
    extra: Option<&'a dyn ExtraAttrs>,
    warnings: Vec<ViewWarning>,
}

fn text(value: impl Into<Box<str>>) -> AttrValue {
    AttrValue::Text(value.into())
}

fn bool_text(value: bool) -> AttrValue {
    text(if value { "true" } else { "false" })
}

fn scalar(value: Scalar) -> AttrValue {
    AttrValue::Scalar(value)
}

fn time_string(time: DateTime<Utc>) -> String {
    let time = floor_to_second(time);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        time.year(),
        time.month(),
        time.day(),
        time.hour(),
        time.minute(),
        time.second()
    )
}

fn variable<'a>(
    name: impl Into<Cow<'a, str>>,
    dims: Vec<Cow<'a, str>>,
    values: Values<'a>,
    attrs: Attrs<'a>,
) -> Variable<'a> {
    Variable {
        name: name.into(),
        dims,
        values,
        attrs,
        source: None,
    }
}

fn scalar_variable<'a>(name: &'a str, value: Scalar, attrs: Attrs<'a>) -> Variable<'a> {
    variable(name, Vec::new(), Values::Scalar(value), attrs)
}

fn text_variable<'a>(name: &'a str, value: impl Into<Cow<'a, str>>) -> Variable<'a> {
    variable(name, Vec::new(), Values::Text(value.into()), Vec::new())
}

/// Per-ray values in view order: borrowed when the order is the identity.
trait RayArray {
    fn ray_values<'a>(values: &'a [Self], order: &RowOrder) -> Values<'a>
    where
        Self: Sized;
}

macro_rules! ray_array {
    ($t:ty, $variant:ident) => {
        impl RayArray for $t {
            fn ray_values<'a>(values: &'a [$t], order: &RowOrder) -> Values<'a> {
                match order {
                    RowOrder::Identity => Values::Borrowed(ArrayRef::$variant(values)),
                    RowOrder::Permutation(permutation) => Values::Owned(ArrayBuf::$variant(
                        permutation
                            .iter()
                            .filter_map(|row| values.get(*row as usize).copied())
                            .collect(),
                    )),
                }
            }
        }
    };
}

ray_array!(u8, U8);
ray_array!(i32, I32);
ray_array!(f32, F32);
ray_array!(f64, F64);

fn array_ref(data: &FieldData) -> ArrayRef<'_> {
    match data {
        FieldData::U8 { values, .. } => ArrayRef::U8(values),
        FieldData::U16 { values, .. } => ArrayRef::U16(values),
        FieldData::I8 { values, .. } => ArrayRef::I8(values),
        FieldData::I16 { values, .. } => ArrayRef::I16(values),
        FieldData::I32 { values, .. } => ArrayRef::I32(values),
        FieldData::F32 { values, .. } => ArrayRef::F32(values),
        FieldData::F64 { values, .. } => ArrayRef::F64(values),
    }
}

fn buf_ref(values: &ArrayBuf) -> Option<ArrayRef<'_>> {
    Some(match values {
        ArrayBuf::U8(v) => ArrayRef::U8(v),
        ArrayBuf::U16(v) => ArrayRef::U16(v),
        ArrayBuf::I8(v) => ArrayRef::I8(v),
        ArrayBuf::I16(v) => ArrayRef::I16(v),
        ArrayBuf::I32(v) => ArrayRef::I32(v),
        ArrayBuf::F32(v) => ArrayRef::F32(v),
        ArrayBuf::F64(v) => ArrayRef::F64(v),
        ArrayBuf::U32(_) | ArrayBuf::I64(_) | ArrayBuf::Text(_) => return None,
    })
}

/// A unit the view spells per flavor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Unit {
    Db,
    Dbm,
    Dbz,
    Seconds,
    Degrees,
    Unitless,
}

/// Unit of a Table 301-14a calibration entry (the unit suffixes of the
/// `RadarCalibration` fields).
fn calibration_unit(name: &str) -> Unit {
    match name {
        "pulse_width" => Unit::Seconds,
        "system_phidp" => Unit::Degrees,
        "probert_jones_correction"
        | "dielectric_factor_used"
        | "receiver_slope_hc"
        | "receiver_slope_vc"
        | "receiver_slope_hx"
        | "receiver_slope_vx" => Unit::Unitless,
        _ if name.starts_with("base_1km_") => Unit::Dbz,
        _ if name.starts_with("xmit_power_")
            || name.starts_with("noise_")
            || name.starts_with("sun_power_")
            || name.starts_with("test_power_") =>
        {
            Unit::Dbm
        }
        // Antenna and receiver gains, losses, radar constants, corrections.
        _ => Unit::Db,
    }
}

/// Unit of a calibration entry outside Table 301-14a, from its name (CfRadial
/// `r_calib_*` and DORADE entries); `None` when the name does not say.
fn extra_calibration_unit(name: &str) -> Option<Unit> {
    if name.contains("dbm") || name == "noise_power" {
        Some(Unit::Dbm)
    } else if name.contains("_db") || name.ends_with("_correction") || name.ends_with("gain") {
        Some(Unit::Db)
    } else if name == "k_squared_water" {
        Some(Unit::Unitless)
    } else {
        None
    }
}

/// Unit of a Table 301-11 monitoring variable.
fn monitoring_unit(name: &str) -> Unit {
    match name {
        "radar_measured_transmit_power_h"
        | "radar_measured_transmit_power_v"
        | "radar_measured_sky_noise"
        | "radar_measured_cold_noise"
        | "radar_measured_hot_noise" => Unit::Dbm,
        "phase_difference_transmit_hv"
        | "antenna_pointing_accuracy_elev"
        | "antenna_pointing_accuracy_az" => Unit::Degrees,
        _ => Unit::Db,
    }
}

fn format_metres(value: f64) -> String {
    if value.fract() == 0.0 && value.abs() < 1e15 {
        format!("{}", value as i64)
    } else {
        format!("{value}")
    }
}

impl<'a> Builder<'a> {
    fn wmo(&self) -> bool {
        self.options.flavor == Flavor::Wmo2022
    }

    fn all(&self) -> bool {
        self.options.passthrough == Passthrough::All
    }

    /// Items xradar keeps but FM301 does not name.
    fn xradar_items(&self) -> bool {
        !self.wmo() || self.all()
    }

    fn is_cfradial(&self) -> bool {
        matches!(
            self.volume.provenance.source_format,
            SourceFormat::CfRadial1 | SourceFormat::CfRadial2
        )
    }

    fn degrees(&self) -> AttrValue {
        text(if self.wmo() { "degree" } else { "degrees" })
    }

    fn metres(&self) -> AttrValue {
        text(if self.wmo() { "metres" } else { "meters" })
    }

    /// `unit` in the flavor's spelling: UDUNITS for Wmo2022; xradar's words
    /// (`seconds`, `degrees`) and CfRadial's empty string for a ratio for
    /// Xradar012.
    fn unit(&self, unit: Unit) -> AttrValue {
        let wmo = self.wmo();
        match unit {
            Unit::Db => text("dB"),
            Unit::Dbm => text("dBm"),
            Unit::Dbz => text("dBZ"),
            Unit::Seconds => text(if wmo { "s" } else { "seconds" }),
            Unit::Degrees => self.degrees(),
            Unit::Unitless => text(if wmo { "1" } else { "" }),
        }
    }

    fn time_units(&self) -> AttrValue {
        text(format!(
            "seconds since {}",
            time_string(self.volume.time_reference)
        ))
    }

    // -----------------------------------------------------------------------
    // Root
    // -----------------------------------------------------------------------

    fn root(&mut self) -> Result<Group<'a>, ViewError> {
        let volume = self.volume;
        let mut dims: Vec<(Cow<'a, str>, usize)> = vec![("sweep".into(), volume.sweeps.len())];
        let mut variables = vec![
            scalar_variable(
                "volume_number",
                Scalar::I32(volume.volume_number.unwrap_or(0)),
                Vec::new(),
            ),
            text_variable("platform_type", volume.platform_type.as_str()),
            text_variable("instrument_type", volume.instrument_type.as_str()),
        ];
        if let Some(axis) = volume.primary_axis {
            variables.push(text_variable("primary_axis", axis.as_str()));
        }
        if let Some(status) = &volume.status_str {
            variables.push(text_variable("status_str", status.as_str()));
        }
        let coverage = volume.time_coverage.or_else(|| volume.ray_time_extent());
        let (start, end) = coverage.map_or((volume.time_reference, volume.time_reference), |c| {
            (c.start, c.end)
        });
        variables.push(text_variable("time_coverage_start", time_string(start)));
        variables.push(text_variable("time_coverage_end", time_string(end)));
        variables.extend(self.location_variables());
        variables.push(variable(
            "sweep_group_name",
            vec!["sweep".into()],
            Values::Owned(ArrayBuf::Text(
                (0..volume.sweeps.len())
                    .map(|index| format!("sweep_{index}").into_boxed_str())
                    .collect(),
            )),
            Vec::new(),
        ));
        variables.push(variable(
            "sweep_fixed_angle",
            vec!["sweep".into()],
            Values::Owned(ArrayBuf::F32(
                volume
                    .sweeps
                    .iter()
                    .map(|sweep| sweep.fixed_angle_deg)
                    .collect(),
            )),
            vec![("units".into(), self.degrees())],
        ));
        if self.all() {
            for extra in &volume.extra_vars {
                add_dims(&mut dims, extra, None);
                variables.push(extra_variable(extra, None));
            }
        }

        let attrs = self.root_attrs();
        let mut children = Vec::new();
        children.extend(self.radar_parameters());
        children.extend(self.radar_calibration()?);
        children.extend(self.georeferencing_correction());
        for (index, sweep) in volume.sweeps.iter().enumerate() {
            children.push(self.sweep(index, sweep)?);
        }
        let mut root = Group {
            name: Cow::Borrowed(""),
            dims,
            variables,
            attrs,
            children,
        };
        if self.all() && !volume.variable_attrs.is_empty() {
            source_variable_attrs(volume, &mut root, "");
        }
        Ok(root)
    }

    fn location_variables(&self) -> Vec<Variable<'a>> {
        let location = &self.volume.location;
        let wmo = self.wmo();
        let mut out = Vec::new();
        let mut push = |name: &'a str, value: Option<f64>, mut attrs: Attrs<'a>| {
            if value.is_none() {
                attrs.push(("_FillValue".into(), scalar(Scalar::F64(f64::NAN))));
            }
            out.push(scalar_variable(
                name,
                Scalar::F64(value.unwrap_or(f64::NAN)),
                attrs,
            ));
        };
        let mut latitude: Attrs<'a> = vec![
            ("long_name".into(), text("latitude")),
            ("units".into(), text("degrees_north")),
        ];
        if !wmo {
            latitude.push(("positive".into(), text("up")));
        }
        latitude.push(("standard_name".into(), text("latitude")));
        push("latitude", location.latitude_deg, latitude);
        push(
            "longitude",
            location.longitude_deg,
            vec![
                ("long_name".into(), text("longitude")),
                ("units".into(), text("degrees_east")),
                ("standard_name".into(), text("longitude")),
            ],
        );
        let metres = if wmo { "metres" } else { "meters" };
        push(
            "altitude",
            location.altitude_m,
            vec![
                ("long_name".into(), text("altitude")),
                ("units".into(), text(metres)),
                ("positive".into(), text("up")),
                ("standard_name".into(), text("altitude")),
            ],
        );
        if location.altitude_agl_m.is_some() {
            push(
                "altitude_agl",
                location.altitude_agl_m,
                vec![
                    ("long_name".into(), text("altitude_above_ground_level")),
                    ("units".into(), text(metres)),
                    ("positive".into(), text("up")),
                ],
            );
        }
        out
    }

    fn root_attrs(&self) -> Attrs<'a> {
        let volume = self.volume;
        let global = &volume.attrs;
        let mut attrs: Attrs<'a> = Vec::new();
        let scan_id = || -> Option<AttrValue> {
            let text_id = volume
                .scan
                .definition
                .as_ref()
                .and_then(|definition| definition.scan_id_text.as_deref());
            match (text_id, volume.scan.id) {
                (Some(text_id), _) => Some(text(text_id)),
                (None, Some(id)) => Some(scalar(match i32::try_from(id) {
                    Ok(id) => Scalar::I32(id),
                    Err(_) => Scalar::I64(id),
                })),
                (None, None) => None,
            }
        };
        if self.wmo() {
            attrs.push(("Conventions".into(), text("CF-1.8, WMO CF-1.0")));
            attrs.push(("wmo__cf_profile".into(), text("FM 301-2022")));
            for (name, value) in [
                ("title", &global.title),
                ("institution", &global.institution),
                ("references", &global.references),
                ("source", &global.source),
                ("history", &global.history),
                ("comment", &global.comment),
            ] {
                if let Some(value) = value {
                    attrs.push((name.into(), text(value.as_str())));
                }
            }
            attrs.push((
                "instrument_name".into(),
                text(global.instrument_name.as_str()),
            ));
            if let Some(site) = &global.site_name {
                attrs.push(("site_name".into(), text(site.as_str())));
            }
            if let Some(name) = &volume.scan.name {
                attrs.push(("scan_name".into(), text(name.as_str())));
            }
            if let Some(id) = scan_id() {
                attrs.push(("scan_id".into(), id));
            }
            attrs.push((
                "platform_is_mobile".into(),
                bool_text(global.platform_is_mobile),
            ));
            if let Some(increase) = global.ray_times_increase {
                attrs.push(("ray_times_increase".into(), bool_text(increase)));
            }
            attrs.push(("simulated".into(), bool_text(global.simulated)));
            let wmo = &global.wmo;
            if let Some(wsi) = &wmo.wsi {
                attrs.push(("wmo__wsi".into(), text(wsi.as_str())));
            }
            if let Some(id) = &wmo.id {
                attrs.push(("wmo__id".into(), text(id.as_str())));
            }
            if let Some(centre) = wmo.originating_centre {
                attrs.push((
                    "wmo__originating_centre".into(),
                    scalar(Scalar::U16(centre)),
                ));
            }
            if let Some(centre) = wmo.originating_sub_centre {
                attrs.push((
                    "wmo__originating_sub_centre".into(),
                    scalar(Scalar::U16(centre)),
                ));
            }
            if let Some(category) = wmo.data_category {
                attrs.push(("wmo__data_category".into(), scalar(Scalar::U8(category))));
            }
            if let Some(policy) = wmo.data_policy {
                attrs.push(("wmo__data_policy".into(), text(policy.as_str())));
            }
            if let Some(sequence) = wmo.update_sequence_number {
                attrs.push((
                    "wmo__update_sequence_number".into(),
                    scalar(Scalar::U32(sequence)),
                ));
            }
            // Not an FM301 attribute: the source format's version, for
            // lossless output (a volume read from an FM301 file keeps the
            // file's own among its attributes, which wins).
            if self.all()
                && let Some(version) = &volume.provenance.source_version
                && !global
                    .other
                    .iter()
                    .any(|(name, _)| &**name == "source_version")
            {
                attrs.push(("source_version".into(), text(version.as_str())));
            }
        } else {
            let none = |value: &Option<String>| text(value.as_deref().unwrap_or("None"));
            attrs.push((
                "Conventions".into(),
                text(
                    volume
                        .provenance
                        .source_conventions
                        .as_deref()
                        .unwrap_or("None"),
                ),
            ));
            let instrument = if global.instrument_name.is_empty() {
                "None"
            } else {
                global.instrument_name.as_str()
            };
            attrs.push(("instrument_name".into(), text(instrument)));
            // xradar writes a CfRadial file's `version` and "None" for other
            // sources; `Passthrough::All` writes every source's version
            // ("AR2V0006", "H5rad 2.3").
            let version = if self.is_cfradial() || self.all() {
                volume
                    .provenance
                    .source_version
                    .as_deref()
                    .unwrap_or("None")
            } else {
                "None"
            };
            attrs.push(("version".into(), text(version)));
            attrs.push(("title".into(), none(&global.title)));
            attrs.push(("institution".into(), none(&global.institution)));
            attrs.push(("references".into(), none(&global.references)));
            attrs.push(("source".into(), none(&global.source)));
            attrs.push(("history".into(), none(&global.history)));
            attrs.push((
                "comment".into(),
                text(
                    global
                        .comment
                        .as_deref()
                        .unwrap_or("im/exported using xradar"),
                ),
            ));
            if self.is_cfradial() {
                attrs.push((
                    "platform_is_mobile".into(),
                    bool_text(global.platform_is_mobile),
                ));
            }
            if let Some(site) = &global.site_name {
                attrs.push(("site_name".into(), text(site.as_str())));
            }
            if let Some(name) = &volume.scan.name {
                attrs.push(("scan_name".into(), text(name.as_str())));
            }
            if volume.provenance.source_format != SourceFormat::NexradLevel2
                && let Some(id) = scan_id()
            {
                attrs.push(("scan_id".into(), id));
            }
            if let Some(increase) = global.ray_times_increase {
                attrs.push(("ray_times_increase".into(), bool_text(increase)));
            }
            if global.simulated {
                attrs.push(("simulated".into(), bool_text(true)));
            }
        }
        if let Some(extra) = self.extra {
            attrs.extend(extra.root_attrs(self.options.flavor));
        }
        if self.all() {
            attrs.extend(
                global
                    .other
                    .iter()
                    .map(|(name, value)| (Cow::Borrowed(&**name), value.clone())),
            );
        }
        attrs
    }

    fn radar_parameters(&self) -> Option<Group<'a>> {
        let parameters = &self.volume.radar_parameters;
        let names: [&'static str; 5] = if self.wmo() {
            [
                "antenna_gain_h",
                "antenna_gain_v",
                "beam_width_h",
                "beam_width_v",
                "receiver_bandwidth",
            ]
        } else {
            [
                "radar_antenna_gain_h",
                "radar_antenna_gain_v",
                "radar_beam_width_h",
                "radar_beam_width_v",
                "radar_receiver_bandwidth",
            ]
        };
        let units = ["dB", "dB", "degrees", "degrees", "s-1"];
        let values = [
            parameters.antenna_gain_h_db,
            parameters.antenna_gain_v_db,
            parameters.beam_width_h_deg,
            parameters.beam_width_v_deg,
            parameters.receiver_bandwidth_hz,
        ];
        let variables: Vec<Variable<'a>> = names
            .iter()
            .zip(units)
            .zip(values)
            .filter_map(|((name, units), value)| {
                let units = if units == "degrees" {
                    self.degrees()
                } else {
                    text(units)
                };
                value.map(|value| {
                    scalar_variable(name, Scalar::F32(value), vec![("units".into(), units)])
                })
            })
            .collect();
        (!variables.is_empty()).then(|| Group {
            name: "radar_parameters".into(),
            dims: Vec::new(),
            variables,
            attrs: Vec::new(),
            children: Vec::new(),
        })
    }

    fn radar_calibration(&self) -> Result<Option<Group<'a>>, ViewError> {
        let calibration = &self.volume.radar_calibration;
        if calibration.is_empty() {
            return Ok(None);
        }
        let dim = || vec![Cow::Borrowed("calib")];
        let mut variables = Vec::new();
        if calibration.iter().any(|entry| entry.calib_index.is_some()) {
            let values = if self.wmo() {
                let mut bytes = Vec::with_capacity(calibration.len());
                for entry in calibration {
                    bytes.push(match entry.calib_index {
                        Some(index) => i8::try_from(index).map_err(|_| ViewError::OutOfRange {
                            path: "radar_calibration/calib_index".to_owned(),
                            attr: "calib_index",
                        })?,
                        None => i8::MIN,
                    });
                }
                (ArrayBuf::I8(bytes), Scalar::I8(i8::MIN))
            } else {
                (
                    ArrayBuf::I32(
                        calibration
                            .iter()
                            .map(|entry| entry.calib_index.unwrap_or(-9999))
                            .collect(),
                    ),
                    Scalar::I32(-9999),
                )
            };
            variables.push(variable(
                "calib_index",
                dim(),
                Values::Owned(values.0),
                vec![("_FillValue".into(), scalar(values.1))],
            ));
        }
        if calibration.iter().any(|entry| entry.time_s.is_some()) {
            variables.push(variable(
                "time",
                dim(),
                Values::Owned(ArrayBuf::F64(
                    calibration
                        .iter()
                        .map(|entry| entry.time_s.unwrap_or(f64::NAN))
                        .collect(),
                )),
                vec![
                    ("standard_name".into(), text("time")),
                    ("units".into(), self.time_units()),
                ],
            ));
        }
        let entries: Vec<_> = calibration
            .iter()
            .map(|entry| entry.float_entries())
            .collect();
        if let Some(first) = entries.first() {
            for (column, (name, _)) in first.iter().enumerate() {
                if entries.iter().any(|row| row[column].1.is_some()) {
                    variables.push(variable(
                        *name,
                        dim(),
                        Values::Owned(ArrayBuf::F32(
                            entries
                                .iter()
                                .map(|row| row[column].1.unwrap_or(f32::NAN))
                                .collect(),
                        )),
                        vec![
                            ("units".into(), self.unit(calibration_unit(name))),
                            ("_FillValue".into(), scalar(Scalar::F32(f32::NAN))),
                        ],
                    ));
                }
            }
        }
        if self.xradar_items() {
            let mut names: Vec<&str> = Vec::new();
            for entry in calibration {
                for (name, _) in &entry.extra {
                    if !names.contains(&&**name) {
                        names.push(name);
                    }
                }
            }
            for name in names {
                let found: Vec<Option<&AttrValue>> = calibration
                    .iter()
                    .map(|entry| {
                        entry
                            .extra
                            .iter()
                            .find(|(key, _)| &**key == name)
                            .map(|(_, value)| value)
                    })
                    .collect();
                // float32 when every value is (the CfRadial and DORADE
                // entries), like the table entries; float64 otherwise.
                let single = found
                    .iter()
                    .flatten()
                    .all(|value| matches!(value, AttrValue::Scalar(Scalar::F32(_))));
                let (values, fill) = if single {
                    let values = found
                        .iter()
                        .map(|value| match value {
                            Some(AttrValue::Scalar(Scalar::F32(value))) => *value,
                            _ => f32::NAN,
                        })
                        .collect();
                    (ArrayBuf::F32(values), Scalar::F32(f32::NAN))
                } else {
                    let values = found
                        .iter()
                        .map(|value| value.and_then(AttrValue::as_f64).unwrap_or(f64::NAN))
                        .collect();
                    (ArrayBuf::F64(values), Scalar::F64(f64::NAN))
                };
                let mut attrs: Attrs<'a> = Vec::new();
                if let Some(unit) = extra_calibration_unit(name) {
                    attrs.push(("units".into(), self.unit(unit)));
                }
                attrs.push(("_FillValue".into(), scalar(fill)));
                variables.push(variable(name, dim(), Values::Owned(values), attrs));
            }
        }
        Ok(Some(Group {
            name: "radar_calibration".into(),
            dims: vec![("calib".into(), calibration.len())],
            variables,
            attrs: Vec::new(),
            children: Vec::new(),
        }))
    }

    fn georeferencing_correction(&self) -> Option<Group<'a>> {
        if !self.xradar_items() {
            return None;
        }
        let correction = self.volume.georeferencing_correction.as_ref()?;
        Some(Group {
            name: "georeferencing_correction".into(),
            dims: Vec::new(),
            variables: correction
                .entries()
                .into_iter()
                .filter_map(|(name, value)| {
                    value.map(|value| scalar_variable(name, Scalar::F32(value), Vec::new()))
                })
                .collect(),
            attrs: Vec::new(),
            children: Vec::new(),
        })
    }

    // -----------------------------------------------------------------------
    // Sweeps
    // -----------------------------------------------------------------------

    /// Ray dimension name and view order for a sweep (design note 12.1).
    fn ray_order(&mut self, index: usize, sweep: &Sweep) -> (&'static str, RowOrder) {
        let (dim, order) =
            view_ray_order(sweep, self.volume.provenance.source_format, self.options);
        if dim == "time" {
            let times = &sweep.rays.time_s;
            let strictly_increasing = (1..times.len()).all(|position| {
                let (a, b) = match &order {
                    RowOrder::Identity => (position - 1, position),
                    RowOrder::Permutation(p) => (p[position - 1] as usize, p[position] as usize),
                };
                times[a] < times[b]
            });
            if !strictly_increasing {
                self.warnings.push(ViewWarning::NonMonotonicTime {
                    sweep: index as u32,
                });
            }
        }
        (dim, order)
    }

    fn sweep(&mut self, index: usize, sweep: &'a Sweep) -> Result<Group<'a>, ViewError> {
        let (ray_dim, order) = self.ray_order(index, sweep);
        let nrays = sweep.nrays();
        let ray = || vec![Cow::Borrowed(ray_dim)];
        let frequency = &self.volume.radar_parameters.frequency_hz;
        let mut dims: Vec<(Cow<'a, str>, usize)> = vec![
            (ray_dim.into(), nrays),
            ("range".into(), sweep.range.ngates()),
        ];
        if !frequency.is_empty() {
            dims.push(("frequency".into(), frequency.len()));
        }
        let nprt = sweep
            .ray_vars
            .prt_sequence_s
            .as_ref()
            .map(|seq| seq.nprt as usize)
            .or_else(|| sweep.polarization_sequence.as_ref().map(Vec::len));
        if let Some(nprt) = nprt {
            dims.push(("prt".into(), nprt));
        }

        let mut variables = Vec::new();
        variables.push(variable(
            "time",
            ray(),
            f64::ray_values(&sweep.rays.time_s, &order),
            vec![
                ("standard_name".into(), text("time")),
                ("units".into(), self.time_units()),
            ],
        ));
        variables.push(self.range_variable(&sweep.range));
        let (azimuth_name, elevation_name) = if self.wmo() {
            (
                "sensor_to_target_azimuth_angle",
                "sensor_to_target_elevation_angle",
            )
        } else {
            ("ray_azimuth_angle", "ray_elevation_angle")
        };
        let mut azimuth_attrs: Attrs<'a> = vec![
            ("standard_name".into(), text(azimuth_name)),
            ("long_name".into(), text("azimuth_angle_from_true_north")),
            ("units".into(), self.degrees()),
        ];
        let mut elevation_attrs: Attrs<'a> = vec![
            ("standard_name".into(), text(elevation_name)),
            (
                "long_name".into(),
                text("elevation_angle_from_horizontal_plane"),
            ),
            ("units".into(), self.degrees()),
            ("positive".into(), text("up")),
        ];
        if !self.wmo() {
            azimuth_attrs.push(("axis".into(), text("radial_azimuth_coordinate")));
            elevation_attrs.push(("axis".into(), text("radial_elevation_coordinate")));
        }
        variables.push(variable(
            "azimuth",
            ray(),
            f32::ray_values(&sweep.rays.azimuth_deg, &order),
            azimuth_attrs,
        ));
        variables.push(variable(
            "elevation",
            ray(),
            f32::ray_values(&sweep.rays.elevation_deg, &order),
            elevation_attrs,
        ));
        if !frequency.is_empty() {
            let mut frequency_attrs: Attrs<'a> = Vec::new();
            if self.wmo() {
                frequency_attrs.push(("standard_name".into(), text("radiation_frequency")));
            }
            frequency_attrs.push(("units".into(), text("s-1")));
            variables.push(variable(
                "frequency",
                vec!["frequency".into()],
                Values::Borrowed(ArrayRef::F64(frequency)),
                frequency_attrs,
            ));
        }

        variables.push(scalar_variable(
            "sweep_number",
            Scalar::I32(i32::try_from(sweep.sweep_number).unwrap_or(i32::MAX)),
            Vec::new(),
        ));
        variables.push(text_variable("sweep_mode", sweep.sweep_mode.as_str()));
        variables.push(text_variable(
            "follow_mode",
            sweep
                .follow_mode
                .as_ref()
                .map_or("not_set", |mode| mode.as_str()),
        ));
        variables.push(text_variable(
            "prt_mode",
            sweep
                .prt_mode
                .as_ref()
                .map_or("not_set", |mode| mode.as_str()),
        ));
        if let Some(mode) = &sweep.polarization_mode {
            variables.push(text_variable("polarization_mode", mode.as_str()));
        }
        variables.push(scalar_variable(
            if self.wmo() {
                "fixed_angle"
            } else {
                "sweep_fixed_angle"
            },
            Scalar::F32(sweep.fixed_angle_deg),
            vec![("units".into(), self.degrees())],
        ));
        if let Some(sequence) = &sweep.polarization_sequence {
            variables.push(variable(
                "polarization_sequence",
                vec!["prt".into()],
                Values::Owned(ArrayBuf::Text(sequence.clone())),
                Vec::new(),
            ));
        }
        if let Some(indexed) = sweep.rays_are_indexed {
            variables.push(variable(
                "rays_are_indexed",
                Vec::new(),
                Values::Text(if indexed { "true" } else { "false" }.into()),
                Vec::new(),
            ));
        }
        if let Some(resolution) = sweep.rays_angle_resolution_deg {
            variables.push(scalar_variable(
                "rays_angle_resolution",
                Scalar::F32(resolution),
                vec![("units".into(), self.degrees())],
            ));
        }
        if let Some(procedures) = &sweep.qc_procedures {
            variables.push(text_variable("qc_procedures", procedures.as_str()));
        }
        if let Some(rate) = sweep.target_scan_rate_deg_per_s {
            variables.push(scalar_variable(
                "target_scan_rate",
                Scalar::F32(rate),
                vec![("units".into(), text("degrees/s"))],
            ));
        }

        self.ray_variables(sweep, ray_dim, &order, &mut variables);

        let mut children = Vec::new();
        if let Some(monitoring) = &sweep.monitoring {
            let mut monitoring_variables = Vec::new();
            for (name, values) in monitoring.variables() {
                let xradar_name = match name {
                    "radar_measured_transmit_power_h" => Some("measured_transmit_power_h"),
                    "radar_measured_transmit_power_v" => Some("measured_transmit_power_v"),
                    _ => None,
                };
                let attrs = vec![("units".into(), self.unit(monitoring_unit(name)))];
                match xradar_name {
                    Some(xradar_name) if !self.wmo() => variables.push(variable(
                        xradar_name,
                        ray(),
                        f32::ray_values(values, &order),
                        attrs,
                    )),
                    _ => monitoring_variables.push(variable(
                        name,
                        ray(),
                        f32::ray_values(values, &order),
                        attrs,
                    )),
                }
            }
            if !monitoring_variables.is_empty() {
                children.push(Group {
                    name: "monitoring".into(),
                    dims: vec![(ray_dim.into(), nrays)],
                    variables: monitoring_variables,
                    attrs: Vec::new(),
                    children: Vec::new(),
                });
            }
        }

        if self.xradar_items() {
            if let Some(track) = &sweep.platform_track {
                variables.push(variable(
                    "latitude",
                    ray(),
                    f64::ray_values(&track.latitude_deg, &order),
                    vec![("units".into(), text("degrees_north"))],
                ));
                variables.push(variable(
                    "longitude",
                    ray(),
                    f64::ray_values(&track.longitude_deg, &order),
                    vec![("units".into(), text("degrees_east"))],
                ));
                let vertical = || -> Attrs<'a> {
                    vec![
                        ("units".into(), self.metres()),
                        ("positive".into(), text("up")),
                    ]
                };
                variables.push(variable(
                    "altitude",
                    ray(),
                    f64::ray_values(&track.altitude_m, &order),
                    vertical(),
                ));
                if let Some(values) = &track.altitude_agl_m {
                    variables.push(variable(
                        "altitude_agl",
                        ray(),
                        f64::ray_values(values, &order),
                        vertical(),
                    ));
                }
                for (name, values) in [
                    ("heading", &track.heading_deg),
                    ("roll", &track.roll_deg),
                    ("pitch", &track.pitch_deg),
                    ("drift", &track.drift_deg),
                    ("rotation", &track.rotation_deg),
                    ("tilt", &track.tilt_deg),
                ] {
                    if let Some(values) = values {
                        variables.push(variable(
                            name,
                            ray(),
                            f32::ray_values(values, &order),
                            vec![("units".into(), self.degrees())],
                        ));
                    }
                }
            }
            for extra in &sweep.extra_vars {
                add_dims(&mut dims, extra, Some(ray_dim));
                variables.push(extra_variable(extra, Some((ray_dim, &order))));
            }
        }

        for (field_index, field) in sweep.fields.iter().enumerate() {
            let variable =
                self.field_variable(index, field_index, field, sweep, ray_dim, &order)?;
            if field.data.transform().is_some_and(|t| !t.is_linear()) {
                // CF cannot express a level table: the field is written decoded
                // (float32, NaN fill), with its codes beside it as
                // `<name>_level` carrying the coding and flag attributes.
                variables.push(self.decoded_field_variable(index, field, sweep, ray_dim, &order)?);
            }
            variables.push(variable);
        }

        let mut attrs: Attrs<'a> = Vec::new();
        if self.all() {
            attrs.extend(sweep.other.iter().map(|(name, value)| {
                (Cow::Borrowed(&**name), ray_attr(name, value, nrays, &order))
            }));
        }
        if let Some(extra) = self.extra {
            attrs.extend(extra.sweep_attrs(index, self.options.flavor));
        }
        Ok(Group {
            name: format!("sweep_{index}").into(),
            dims,
            variables,
            attrs,
            children,
        })
    }

    fn range_variable(&self, range: &RangeCoord) -> Variable<'a> {
        let centres = range.centers_f32();
        let mut attrs: Attrs<'a> = vec![
            ("units".into(), self.metres()),
            ("standard_name".into(), text("projection_range_coordinate")),
            ("long_name".into(), text("range_to_measurement_volume")),
            ("axis".into(), text("radial_range_coordinate")),
        ];
        match range {
            RangeCoord::Uniform {
                first_center_m,
                spacing_m,
                ..
            } => {
                attrs.push((
                    "meters_between_gates".into(),
                    scalar(Scalar::F32(*spacing_m as f32)),
                ));
                attrs.push(("spacing_is_constant".into(), bool_text(true)));
                attrs.push((
                    "meters_to_center_of_first_gate".into(),
                    scalar(Scalar::F32(*first_center_m as f32)),
                ));
            }
            RangeCoord::Explicit { centers_m } => {
                attrs.push(("spacing_is_constant".into(), bool_text(false)));
                attrs.push((
                    "meters_to_center_of_first_gate".into(),
                    scalar(Scalar::F32(centers_m.first().copied().unwrap_or(f32::NAN))),
                ));
            }
        }
        variable(
            "range",
            vec!["range".into()],
            Values::Owned(ArrayBuf::F32(centres)),
            attrs,
        )
    }

    fn ray_variables(
        &self,
        sweep: &'a Sweep,
        ray_dim: &'static str,
        order: &RowOrder,
        variables: &mut Vec<Variable<'a>>,
    ) {
        let vars = &sweep.ray_vars;
        let parameters = &self.volume.radar_parameters;
        let nrays = sweep.nrays();
        let ray = || vec![Cow::Borrowed(ray_dim)];
        let wmo = self.wmo();
        let units = |wmo_units: &'static str, xradar_units: &'static str| -> Attrs<'a> {
            vec![(
                "units".into(),
                text(if wmo { wmo_units } else { xradar_units }),
            )]
        };
        let float = |name: &'a str,
                     values: &'a Option<Vec<f32>>,
                     broadcast: Option<f32>,
                     attrs: Attrs<'a>|
         -> Option<Variable<'a>> {
            match (values, broadcast) {
                (Some(values), _) => {
                    Some(variable(name, ray(), f32::ray_values(values, order), attrs))
                }
                (None, Some(value)) => Some(variable(
                    name,
                    ray(),
                    Values::Owned(ArrayBuf::F32(vec![value; nrays])),
                    attrs,
                )),
                (None, None) => None,
            }
        };
        let mut nyquist_attrs: Attrs<'a> = units("m s-1", "m s-1");
        nyquist_attrs.insert(0, ("standard_name".into(), text("nyquist_velocity")));
        variables.extend(float(
            "nyquist_velocity",
            &vars.nyquist_velocity_mps,
            None,
            nyquist_attrs,
        ));
        variables.extend(float(
            "unambiguous_range",
            &vars.unambiguous_range_m,
            parameters.unambiguous_range_m,
            units("m", "meters"),
        ));
        variables.extend(float(
            "prt",
            &vars.prt_s,
            parameters.prt_s,
            units("s", "seconds"),
        ));
        variables.extend(float(
            "prt_ratio",
            &vars.prt_ratio,
            None,
            vec![("units".into(), self.unit(Unit::Unitless))],
        ));
        if let Some(sequence) = &vars.prt_sequence_s {
            let values = match order {
                RowOrder::Identity => Values::Borrowed(ArrayRef::F32(&sequence.values_s)),
                RowOrder::Permutation(permutation) => Values::Owned(
                    ArrayBuf::F32(sequence.values_s.clone())
                        .take_rows(sequence.nprt as usize, permutation)
                        .unwrap_or(ArrayBuf::F32(Vec::new())),
                ),
            };
            variables.push(variable(
                "prt_sequence",
                vec![Cow::Borrowed(ray_dim), "prt".into()],
                values,
                units("s", "seconds"),
            ));
        }
        if let Some(samples) = &vars.n_samples {
            variables.push(variable(
                "n_samples",
                ray(),
                i32::ray_values(samples, order),
                vec![
                    ("units".into(), self.unit(Unit::Unitless)),
                    ("_FillValue".into(), scalar(Scalar::I32(-9999))),
                ],
            ));
        }
        variables.extend(float(
            "pulse_width",
            &vars.pulse_width_s,
            parameters.pulse_width_s,
            units("s", "seconds"),
        ));
        variables.extend(float(
            "scan_rate",
            &vars.scan_rate_deg_per_s,
            None,
            units("degree s-1", "degrees/s"),
        ));
        if let Some(transition) = &vars.antenna_transition {
            variables.push(variable(
                "antenna_transition",
                ray(),
                u8::ray_values(transition, order),
                Vec::new(),
            ));
        }
        if let Some(calib_index) = &vars.calib_index {
            variables.push(variable(
                if wmo { "calib_index" } else { "r_calib_index" },
                ray(),
                i32::ray_values(calib_index, order),
                Vec::new(),
            ));
        }
        variables.extend(float(
            "rx_range_resolution",
            &vars.rx_range_resolution_m,
            None,
            units("m", "meters"),
        ));
        if self.all() {
            variables.extend(float(
                "independent_samples",
                &vars.independent_samples,
                None,
                Vec::new(),
            ));
        }
    }

    fn field_variable(
        &self,
        sweep_index: usize,
        field_index: usize,
        field: &'a Field,
        sweep: &'a Sweep,
        ray_dim: &'static str,
        order: &RowOrder,
    ) -> Result<Variable<'a>, ViewError> {
        let out_gates = sweep.range.ngates();
        let source = FieldSource {
            sweep: sweep_index as u32,
            field: field_index as u32,
        };
        let native = array_ref(&field.data);
        let identity = matches!(order, RowOrder::Identity)
            && field.gates == GateMapping::IDENTITY
            && field.ngates as usize == out_gates
            && field.nrays as usize == sweep.nrays();
        let values = if identity {
            Values::Borrowed(native)
        } else {
            Values::Mapped {
                source,
                native,
                nrays: sweep.nrays(),
                native_gates: field.ngates as usize,
                mapping: field.gates,
                out_gates,
                fill: field.data.fill_scalar(),
                rows: order.clone(),
            }
        };
        let form = if field.data.transform().is_some_and(|t| !t.is_linear()) {
            FieldForm::Codes
        } else {
            FieldForm::Encoded
        };
        Ok(Variable {
            name: match form {
                FieldForm::Codes => Cow::Owned(level_variable_name(field)),
                _ => Cow::Borrowed(field.name.as_str()),
            },
            dims: vec![Cow::Borrowed(ray_dim), "range".into()],
            values,
            attrs: self.field_attrs(sweep_index, field, sweep, order, form)?,
            source: Some(source),
        })
    }

    /// A field whose transform is not linear ([`LinearTransform::Levels`]),
    /// decoded to float32 physical values (NaN for every sentinel) in the
    /// view's ray order and on the sweep's range. Its codes are the
    /// `<name>_level` variable, named in `ancillary_variables`.
    fn decoded_field_variable(
        &self,
        sweep_index: usize,
        field: &'a Field,
        sweep: &'a Sweep,
        ray_dim: &'static str,
        order: &RowOrder,
    ) -> Result<Variable<'a>, ViewError> {
        let physical = field.to_physical();
        let values = super::layout::apply_mapping(
            ArrayRef::F32(&physical),
            sweep.nrays(),
            field.ngates as usize,
            field.gates,
            sweep.range.ngates(),
            Scalar::F32(f32::NAN),
            order,
        );
        Ok(Variable {
            name: Cow::Borrowed(field.name.as_str()),
            dims: vec![Cow::Borrowed(ray_dim), "range".into()],
            values: Values::Owned(values),
            attrs: self.field_attrs(sweep_index, field, sweep, order, FieldForm::Decoded)?,
            source: None,
        })
    }

    fn field_attrs(
        &self,
        sweep_index: usize,
        field: &'a Field,
        sweep: &Sweep,
        order: &RowOrder,
        form: FieldForm,
    ) -> Result<Attrs<'a>, ViewError> {
        let range = &sweep.range;
        let path = || format!("sweep_{sweep_index}/{}", field.name);
        let model = &field.attrs;
        let mut attrs: Attrs<'a> = Vec::new();

        if form == FieldForm::Codes {
            return self.level_attrs(field, range, path);
        }

        let info = field.name.info();
        let (standard_name, long_name, units) = if self.wmo() {
            (
                info.and_then(|info| info.standard_name),
                info.map(|info| info.long_name),
                info.map(|info| info.units),
            )
        } else {
            let xradar = info.and_then(|info| info.xradar);
            (
                xradar.map(|x| x.standard_name),
                xradar.map(|x| x.long_name),
                xradar.map(|x| x.units),
            )
        };
        for (name, model_value, table_value) in [
            ("standard_name", &model.standard_name, standard_name),
            ("long_name", &model.long_name, long_name),
            ("units", &model.units, units),
        ] {
            if let Some(value) = model_value.as_deref().or(table_value) {
                attrs.push((name.into(), text(value)));
            }
        }

        match &field.data {
            _ if form == FieldForm::Decoded => {
                attrs.push(("_FillValue".into(), scalar(Scalar::F32(f32::NAN))));
            }
            FieldData::U8 { coding, .. } => int_attrs(&mut attrs, coding, model, path)?,
            FieldData::U16 { coding, .. } => int_attrs(&mut attrs, coding, model, path)?,
            FieldData::I8 { coding, .. } => int_attrs(&mut attrs, coding, model, path)?,
            FieldData::I16 { coding, .. } => int_attrs(&mut attrs, coding, model, path)?,
            FieldData::I32 { coding, .. } => int_attrs(&mut attrs, coding, model, path)?,
            FieldData::F32 { coding, .. } => float_attrs(
                &mut attrs,
                coding,
                Scalar::F32,
                |v| ArrayBuf::F32(v.iter().map(|x| *x as f32).collect()),
                model,
            ),
            FieldData::F64 { coding, .. } => float_attrs(
                &mut attrs,
                coding,
                Scalar::F64,
                |v| ArrayBuf::F64(v.iter().map(|x| *x as f64).collect()),
                model,
            ),
        }

        if let Some(ratio) = model.sampling_ratio {
            attrs.push(("sampling_ratio".into(), scalar(Scalar::F32(ratio))));
        }
        for (name, value) in [
            ("is_discrete", model.is_discrete),
            ("field_folds", model.field_folds),
            ("is_quality_field", model.is_quality_field),
        ] {
            if let Some(value) = value {
                attrs.push((name.into(), bool_text(value)));
            }
        }
        for (name, value) in [
            ("fold_limit_lower", model.fold_limit_lower),
            ("fold_limit_upper", model.fold_limit_upper),
        ] {
            if let Some(value) = value {
                attrs.push((name.into(), scalar(Scalar::F32(value))));
            }
        }
        let level_name = (form == FieldForm::Decoded).then(|| level_variable_name(field));
        for (name, list, extra) in [
            ("qualified_variables", &model.qualified_variables, None),
            (
                "ancillary_variables",
                &model.ancillary_variables,
                level_name.as_deref(),
            ),
        ] {
            let mut joined: Vec<&str> = list.iter().map(|name| name.as_str()).collect();
            joined.extend(extra);
            if !joined.is_empty() {
                attrs.push((name.into(), text(joined.join(" "))));
            }
        }
        if let Some(xml) = &model.thresholding_xml {
            attrs.push(("thresholding_xml".into(), text(xml.as_str())));
        }
        // The Xradar flavor keeps a source's own `coordinates` (xradar leaves
        // CfRadial's in the encoding); the flavor default otherwise.
        let source_coordinates = (!self.wmo())
            .then(|| {
                model
                    .other
                    .iter()
                    .find(|(name, _)| &**name == "coordinates")
                    .map(|(_, value)| value.clone())
            })
            .flatten();
        attrs.push((
            "coordinates".into(),
            source_coordinates.unwrap_or_else(|| {
                text(if self.wmo() {
                    "elevation azimuth range"
                } else {
                    "elevation azimuth range latitude longitude altitude time"
                })
            }),
        ));
        if field.gates.stride > 1
            && let RangeCoord::Uniform { spacing_m, .. } = range
        {
            attrs.push((
                "comment".into(),
                text(format!(
                    "native gate spacing {} m; values repeated on the {} m range coordinate",
                    format_metres(spacing_m * f64::from(field.gates.stride)),
                    format_metres(*spacing_m)
                )),
            ));
        }
        // xradar 0.12 writes no ODIM plane `what`/`how` attributes, as it
        // writes no ODIM `how` for the root and the sweeps (Passthrough).
        let odim = self.volume.provenance.source_format == SourceFormat::OdimH5;
        if self.all() || (self.xradar_items() && !odim) {
            // Source attributes without a slot, verbatim; one the flavor
            // already wrote (a CfRadial `coordinates`) is not repeated.
            for (name, value) in &model.other {
                if attrs.iter().any(|(key, _)| key == &**name) {
                    continue;
                }
                attrs.push((
                    Cow::Borrowed(&**name),
                    ray_attr(name, value, sweep.nrays(), order),
                ));
            }
        }
        Ok(attrs)
    }

    /// Attributes of the `<name>_level` variable of a field with a level
    /// table: what the codes mean (`long_name`, `_FillValue`, `_Undetect`,
    /// `valid_range`, `flag_values` / `flag_masks` / `flag_meanings`, the
    /// model attributes without a slot) and the variable they encode.
    fn level_attrs(
        &self,
        field: &'a Field,
        range: &RangeCoord,
        path: impl Fn() -> String,
    ) -> Result<Attrs<'a>, ViewError> {
        let model = &field.attrs;
        let mut attrs: Attrs<'a> = Vec::new();
        let long_name = model
            .long_name
            .as_deref()
            .or_else(|| field.name.info().map(|info| info.long_name))
            .unwrap_or(field.name.as_str());
        attrs.push(("long_name".into(), text(format!("{long_name} data level"))));
        match &field.data {
            FieldData::U8 { coding, .. } => int_attrs(&mut attrs, coding, model, path)?,
            FieldData::U16 { coding, .. } => int_attrs(&mut attrs, coding, model, path)?,
            FieldData::I8 { coding, .. } => int_attrs(&mut attrs, coding, model, path)?,
            FieldData::I16 { coding, .. } => int_attrs(&mut attrs, coding, model, path)?,
            FieldData::I32 { coding, .. } => int_attrs(&mut attrs, coding, model, path)?,
            FieldData::F32 { .. } | FieldData::F64 { .. } => {}
        }
        if let Some(value) = model.is_discrete {
            attrs.push(("is_discrete".into(), bool_text(value)));
        }
        attrs.push((
            "comment".into(),
            text(format!(
                "data level codes of {}; values through the product's level table",
                field.name
            )),
        ));
        if field.gates.stride > 1
            && let RangeCoord::Uniform { spacing_m, .. } = range
        {
            attrs.push((
                "gate_comment".into(),
                text(format!(
                    "native gate spacing {} m; values repeated on the {} m range coordinate",
                    format_metres(spacing_m * f64::from(field.gates.stride)),
                    format_metres(*spacing_m)
                )),
            ));
        }
        if self.xradar_items() {
            for (name, value) in &model.other {
                if attrs.iter().any(|(key, _)| key == &**name) {
                    continue;
                }
                attrs.push((Cow::Borrowed(&**name), value.clone()));
            }
        }
        Ok(attrs)
    }
}

/// The three forms a field is written in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FieldForm {
    /// The stored codes (or floats) with their CF packing: every field with a
    /// linear transform.
    Encoded,
    /// The codes of a field with a level table, as `<name>_level`.
    Codes,
    /// The physical values of a field with a level table, as `<name>`.
    Decoded,
}

/// `<name>_level`: the codes variable of a field with a level table.
fn level_variable_name(field: &Field) -> String {
    format!("{}_level", field.name.as_str())
}

/// Add the source's own attributes of slotted variables
/// ([`Volume::variable_attrs`]) to the variables of `group` (at `path`) and
/// its children: an attribute the view already wrote keeps the view's value.
/// A variable of a sweep group without an entry of its own takes the root
/// entry of the same name (a CfRadial 1 file keeps its per-ray and per-sweep
/// variables at the root).
fn source_variable_attrs<'a>(volume: &'a Volume, group: &mut Group<'a>, path: &str) {
    let in_sweep = path.starts_with("sweep_");
    for variable in &mut group.variables {
        let entry = volume
            .variable_attrs
            .iter()
            .find(|entry| &*entry.group == path && *entry.name == *variable.name)
            .or_else(|| {
                in_sweep
                    .then(|| {
                        volume
                            .variable_attrs
                            .iter()
                            .find(|entry| entry.group.is_empty() && *entry.name == *variable.name)
                    })
                    .flatten()
            });
        let Some(entry) = entry else {
            continue;
        };
        for (name, value) in &entry.attrs {
            if variable.attrs.iter().any(|(have, _)| have == &**name) {
                continue;
            }
            variable.attrs.push((Cow::Borrowed(&**name), value.clone()));
        }
    }
    for child in &mut group.children {
        let child_path = if path.is_empty() {
            child.name.to_string()
        } else {
            format!("{path}/{}", child.name)
        };
        source_variable_attrs(volume, child, &child_path);
    }
}

/// The ray dimension name and ray order of `sweep` in a view with `options`
/// (design note 12.1): `time` in acquisition order for the WMO flavor and
/// `FirstDim::Time`; under `FirstDim::Auto`, `azimuth` order, or `elevation`
/// order for RHI sweeps of non-CfRadial sources.
pub(super) fn view_ray_order(
    sweep: &Sweep,
    source_format: SourceFormat,
    options: ViewOptions,
) -> (&'static str, RowOrder) {
    let first_dim = if options.flavor == Flavor::Wmo2022 {
        FirstDim::Time
    } else {
        options.first_dim
    };
    match first_dim {
        FirstDim::Time => {
            let times = &sweep.rays.time_s;
            let order = sort_order(
                times.len(),
                |a, b| times[a].total_cmp(&times[b]),
                |i| times[i],
            );
            ("time", order)
        }
        FirstDim::Auto => {
            let cfradial = matches!(
                source_format,
                SourceFormat::CfRadial1 | SourceFormat::CfRadial2
            );
            let rhi = sweep.sweep_mode == SweepMode::Rhi && !cfradial;
            let (dim, angles) = if rhi {
                ("elevation", &sweep.rays.elevation_deg)
            } else {
                ("azimuth", &sweep.rays.azimuth_deg)
            };
            let order = sort_order(
                angles.len(),
                |a, b| angles[a].total_cmp(&angles[b]),
                |i| f64::from(angles[i]),
            );
            (dim, order)
        }
    }
}

/// A verbatim attribute for the view: a per-ray array
/// ([`AttrValue::ray_alignment`]) in the view's ray order, so that it lines
/// up with the ray dimension; anything else as stored.
fn ray_attr(name: &str, value: &AttrValue, nrays: usize, order: &RowOrder) -> AttrValue {
    match order {
        RowOrder::Permutation(permutation)
            if value.ray_alignment(name, nrays) == RayAlignment::PerRay =>
        {
            value.in_ray_order(nrays, permutation)
        }
        _ => value.clone(),
    }
}

/// Stable sort of `0..len` by `compare`; `Identity` when already ordered.
fn sort_order(
    len: usize,
    compare: impl Fn(usize, usize) -> std::cmp::Ordering,
    key: impl Fn(usize) -> f64,
) -> RowOrder {
    let sorted = (1..len).all(|i| {
        let (a, b) = (key(i - 1), key(i));
        a <= b
    });
    if sorted {
        return RowOrder::Identity;
    }
    let mut indices: Vec<u32> = (0..len as u32).collect();
    indices.sort_by(|a, b| compare(*a as usize, *b as usize));
    RowOrder::Permutation(Arc::from(indices))
}

fn scale_attrs(attrs: &mut Attrs<'_>, transform: LinearTransform) {
    let (Some(scale), Some(offset)) = (transform.scale_factor(), transform.add_offset()) else {
        return;
    };
    let (scale_factor, add_offset) = match transform.attr_width() {
        FloatWidth::F64 => (Scalar::F64(scale), Scalar::F64(offset)),
        FloatWidth::F32 => (Scalar::F32(scale as f32), Scalar::F32(offset as f32)),
    };
    attrs.push(("scale_factor".into(), scalar(scale_factor)));
    attrs.push(("add_offset".into(), scalar(add_offset)));
}

fn int_attrs<T: PackedInt>(
    attrs: &mut Attrs<'_>,
    coding: &IntCoding<T>,
    model: &FieldAttrs,
    path: impl Fn() -> String,
) -> Result<(), ViewError> {
    if coding.transform.is_linear() {
        scale_attrs(attrs, coding.transform);
    }
    if let Some(fill) = coding.fill_value {
        attrs.push(("_FillValue".into(), scalar(fill.to_scalar())));
    }
    if let Some(undetect) = coding.undetect {
        attrs.push(("_Undetect".into(), scalar(undetect.to_scalar())));
    }
    if let Some([lo, hi]) = coding.valid_range {
        attrs.push((
            "valid_range".into(),
            AttrValue::Array(T::array(vec![lo, hi])),
        ));
    }
    let narrow = |values: &[i64], attr: &'static str| -> Result<Vec<T>, ViewError> {
        values
            .iter()
            .map(|value| {
                T::from_i64(*value).ok_or_else(|| ViewError::OutOfRange { path: path(), attr })
            })
            .collect()
    };
    let mut flag_values: Vec<T> = coding.range_folded.into_iter().collect();
    flag_values.extend(narrow(&model.flag_values, "flag_values")?);
    let mut flag_meanings: Vec<&str> = Vec::new();
    if coding.range_folded.is_some() {
        flag_meanings.push("range_folded");
    }
    flag_meanings.extend(model.flag_meanings.iter().map(|meaning| &**meaning));
    if !flag_values.is_empty() {
        attrs.push((
            "flag_values".into(),
            AttrValue::Array(T::array(flag_values)),
        ));
    }
    if !model.flag_masks.is_empty() {
        attrs.push((
            "flag_masks".into(),
            AttrValue::Array(T::array(narrow(&model.flag_masks, "flag_masks")?)),
        ));
    }
    if !flag_meanings.is_empty() {
        attrs.push(("flag_meanings".into(), text(flag_meanings.join(" "))));
    }
    Ok(())
}

fn float_attrs<T: Copy>(
    attrs: &mut Attrs<'_>,
    coding: &FloatCoding<T>,
    to_scalar: impl Fn(T) -> Scalar,
    to_array: impl Fn(&[i64]) -> ArrayBuf,
    model: &FieldAttrs,
) where
    FloatCoding<T>: FillCode<T>,
{
    if let Some(transform) = coding.transform {
        scale_attrs(attrs, transform);
    }
    attrs.push(("_FillValue".into(), scalar(to_scalar(coding.fill()))));
    if let Some(undetect) = coding.undetect {
        attrs.push(("_Undetect".into(), scalar(to_scalar(undetect))));
    }
    if !model.flag_values.is_empty() {
        attrs.push((
            "flag_values".into(),
            AttrValue::Array(to_array(&model.flag_values)),
        ));
    }
    if !model.flag_masks.is_empty() {
        attrs.push((
            "flag_masks".into(),
            AttrValue::Array(to_array(&model.flag_masks)),
        ));
    }
    if !model.flag_meanings.is_empty() {
        let meanings: Vec<&str> = model.flag_meanings.iter().map(|m| &**m).collect();
        attrs.push(("flag_meanings".into(), text(meanings.join(" "))));
    }
}

/// The fill code of a float coding (its `_FillValue`, else NaN).
trait FillCode<T> {
    fn fill(&self) -> T;
}

impl FillCode<f32> for FloatCoding<f32> {
    fn fill(&self) -> f32 {
        self.fill_code()
    }
}

impl FillCode<f64> for FloatCoding<f64> {
    fn fill(&self) -> f64 {
        self.fill_code()
    }
}

/// Declare the non-ray dimensions of an extra variable.
fn add_dims<'a>(
    dims: &mut Vec<(Cow<'a, str>, usize)>,
    extra: &'a ExtraVariable,
    ray_dim: Option<&'static str>,
) {
    for (index, (name, len)) in extra.dims.iter().zip(&extra.shape).enumerate() {
        if index == 0 && ray_dim.is_some() && &**name == "time" {
            continue;
        }
        if !dims.iter().any(|(existing, _)| existing == &**name) {
            dims.push((Cow::Borrowed(&**name), *len as usize));
        }
    }
}

fn extra_variable<'a>(
    extra: &'a ExtraVariable,
    ray: Option<(&'static str, &RowOrder)>,
) -> Variable<'a> {
    let per_ray = extra.is_per_ray();
    let dims = extra
        .dims
        .iter()
        .enumerate()
        .map(|(index, dim)| match ray {
            Some((ray_dim, _)) if index == 0 && per_ray => Cow::Borrowed(ray_dim),
            _ => Cow::Borrowed(&**dim),
        })
        .collect();
    // `None` when the shape overflows; `take_rows` refuses rows past the
    // values, so a shape that disagrees with them costs no allocation.
    let row_len = extra
        .shape
        .iter()
        .skip(1)
        .try_fold(1usize, |product, len| product.checked_mul(*len as usize));
    let values = match (ray, row_len) {
        (Some((_, RowOrder::Permutation(permutation))), Some(row_len)) if per_ray => Values::Owned(
            extra
                .values
                .take_rows(row_len.max(1), permutation)
                .unwrap_or_else(|| extra.values.clone()),
        ),
        _ => match buf_ref(&extra.values) {
            Some(array) => Values::Borrowed(array),
            None => Values::Owned(extra.values.clone()),
        },
    };
    variable(
        Cow::Borrowed(&*extra.name),
        dims,
        values,
        extra
            .attrs
            .iter()
            .map(|(name, value)| (Cow::Borrowed(&**name), value.clone()))
            .collect(),
    )
}
