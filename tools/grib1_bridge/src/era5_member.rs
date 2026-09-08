//! ERA5 EDA identity and byte-preserving single-member publication.
//! Local definitions: https://codes.ecmwf.int/grib/format/grib1/local/1/
//! and ecCodes definitions/grib1/local.98.36.def. No field is unpacked here.
use chrono::{Duration, NaiveDate};
use grib_core::grib1::Grib1File;
use std::collections::BTreeMap;
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
// ECMWF parameter numbers are local to a table. In particular table 228's
// lake fields must never complete table 128's separate field census.
type Key = (u8, u8, u8, u16, i64);
pub const SCHEMA: &str = "arwen.era5-member-selection.v1";

struct Identity {
    member: u8,
    key: Key,
    representation: Vec<u8>,
}

fn identity(bytes: &[u8]) -> Result<Identity> {
    let file = Grib1File::from_bytes(bytes)?;
    if file.messages.len() != 1 {
        return Err("ERA5 member message failed native GRIB1 parsing".into());
    }
    let message = &file.messages[0];
    let p = &message.pds;
    let n = bytes.get(8..11).ok_or("Missing ERA5 PDS")?;
    let len = ((n[0] as usize) << 16) | ((n[1] as usize) << 8) | n[2] as usize;
    let pds = bytes.get(8..8 + len).ok_or("Truncated ERA5 PDS")?;
    let supported_table = p.table_version == 128
        || (p.table_version == 228
            && matches!(p.parameter, 8 | 13 | 14)
            && p.level_type == 1
            && p.level_value == 0);
    if p.center_id != 98 || !supported_table || len < 52 {
        return Err(
            "Member binding requires ECMWF GRIB1 table 128 or table 228 lake surface parameters 8/13/14, and explicit ERA5 local metadata".into(),
        );
    }
    // ecCodes local.98.17.def (SST/sea ice) also defines the member and
    // ensemble-count bytes. Its dated-source entries occupy 40-byte blocks.
    let local17 = pds[40] == 17
        && p.table_version == 128
        && matches!(p.parameter, 31 | 34)
        && p.level_type == 1
        && p.level_value == 0
        && len >= 96
        && (len - 56) % 40 == 0
        && len >= 56 + 4 * pds[55] as usize;
    let known_layout = matches!((pds[40], len), (1, 52) | (36, 56)) || local17;
    if !known_layout
        || pds[41] != 23
        || !matches!(pds[42], 2 | 9)
        || u16::from_be_bytes([pds[43], pds[44]]) != 1030
        || !matches!(&pds[45..49], b"0001" | b"0005")
        || pds[49] > 9
        || !matches!(pds[50], 0 | 10)
    {
        return Err("Expected ERA5 EDA member 0..9, enda stream, analysis/forecast; means/spreads and unknown identity are refused".into());
    }
    let base = NaiveDate::from_ymd_opt(p.year() as i32, p.month as u32, p.day as u32)
        .and_then(|d| d.and_hms_opt(p.hour as u32, p.minute as u32, 0))
        .ok_or("Invalid ERA5 reference UTC")?;
    let unit = match p.time_unit {
        0 => 60,
        1 => 3600,
        2 => 86400,
        10 => 10800,
        11 => 21600,
        12 => 43200,
        13 => 900,
        14 => 1800,
        254 => 1,
        _ => return Err("Unsupported ERA5 GRIB1 time unit".into()),
    };
    let step = match p.time_range_indicator {
        0 => p.p1 as i64,
        1 if p.p1 == 0 && p.p2 == 0 => 0,
        10 => u16::from_be_bytes([p.p1, p.p2]) as i64,
        _ => return Err("ERA5 forcing member selection requires instantaneous fields".into()),
    };
    if pds[42] == 2 && step != 0 {
        return Err("An ERA5 EDA analysis cannot carry a forecast lead".into());
    }
    let valid = base
        .checked_add_signed(Duration::seconds(step * unit))
        .ok_or("ERA5 valid UTC overflow")?;
    if valid.and_utc().timestamp().rem_euclid(10800) != 0 {
        return Err("ERA5 EDA fields must occur at exact 3-hourly UTC times".into());
    }
    let start = 8 + len;
    let g = bytes
        .get(start..start + 3)
        .ok_or("Missing ERA5 grid section")?;
    let glen = ((g[0] as usize) << 16) | ((g[1] as usize) << 8) | g[2] as usize;
    let grid = bytes
        .get(start..start + glen)
        .ok_or("Truncated ERA5 grid section")?;
    if pds[7] & 128 == 0
        || grid.len() < 28
        || grid[5] != 0
        || u16::from_be_bytes([grid[23], grid[24]]) != 500
        || u16::from_be_bytes([grid[25], grid[26]]) != 500
    {
        return Err("ERA5 EDA requires its regular 0.5-degree grid".into());
    }
    // Decimal packing precision may differ between members. Identity,
    // forecast reference/step, local window and grid must agree exactly.
    let mut representation = pds[12..26].to_vec();
    representation.extend_from_slice(&pds[40..49]);
    representation.extend_from_slice(&pds[51..]);
    representation.extend_from_slice(grid);
    Ok(Identity {
        member: pds[49],
        key: (
            p.table_version,
            p.parameter,
            p.level_type,
            p.level_value,
            valid.and_utc().timestamp(),
        ),
        representation,
    })
}

fn census(
    input: &Path,
    member: u8,
    selected_only: bool,
    mut output: Option<&mut dyn Write>,
) -> Result<(usize, usize)> {
    if member > 9 {
        return Err("ERA5 EDA member must be 0..9".into());
    }
    let mut groups = BTreeMap::<Key, (u16, Vec<u8>)>::new();
    let mut selected = 0usize;
    let messages = super::visit_grib1_envelopes(input, |_, bytes| {
        let id = identity(bytes)?;
        if selected_only && id.member != member {
            return Err("Forcing contains another ERA5 EDA member".into());
        }
        let group = groups
            .entry(id.key)
            .or_insert_with(|| (0, id.representation.clone()));
        if group.1 != id.representation {
            return Err("EDA members disagree on field grid or reference identity".into());
        }
        let bit = 1u16 << id.member;
        if group.0 & bit != 0 {
            return Err("Duplicate ERA5 member for the same table, parameter, level and UTC".into());
        }
        group.0 |= bit;
        if id.member == member {
            selected += 1;
            if let Some(stream) = output.as_deref_mut() {
                stream.write_all(bytes)?;
            }
        }
        Ok(())
    })?;
    let mask = if selected_only { 1u16 << member } else { 1023 };
    if groups.values().any(|g| g.0 != mask) {
        return Err("ERA5 EDA field census is incomplete; every field/time/level needs all ten members before selection".into());
    }
    Ok((messages, selected))
}

pub fn check(input: &Path, member: u8) -> Result<()> {
    let (messages, _) = census(input, member, true, None)?;
    println!("{{\"schema\":\"{SCHEMA}\",\"member\":{member},\"messages\":{messages},\"selected_only\":true,\"byte_preserving\":true}}");
    Ok(())
}

struct Stage(PathBuf);
impl Drop for Stage {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

pub fn select(input: &Path, output: &Path, member: u8) -> Result<()> {
    if output.try_exists()? {
        return Err("Member selection preserves existing output files".into());
    }
    let parent = output
        .parent()
        .ok_or("Member output needs a parent directory")?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let stage = Stage(parent.join(format!(".era5-member-{}-{stamp}.part", std::process::id())));
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&stage.0)?;
    let mut writer = BufWriter::new(file);
    let (messages, selected) = census(input, member, false, Some(&mut writer))?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    drop(writer);
    // A create-only hard link publishes exactly the completely verified bytes.
    fs::hard_link(&stage.0, output)?;
    println!("{{\"schema\":\"{SCHEMA}\",\"member\":{member},\"input_messages\":{messages},\"messages\":{selected},\"selected_only\":true,\"byte_preserving\":true}}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    const REAL: &[u8] = include_bytes!("../tests/fixtures/era5-eda-ten-t2m.grib");
    fn path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("arwen-eda-{}-{name}.grib", std::process::id()))
    }
    #[test]
    fn real_cds_members_preserve_exact_encoded_bytes() {
        let input = path("all");
        fs::write(&input, REAL).unwrap();
        for member in 0..10 {
            let out = path(&format!("selected-{member}"));
            let _ = fs::remove_file(&out);
            select(&input, &out, member).unwrap();
            assert_eq!(
                fs::read(&out).unwrap(),
                REAL[member as usize * 130..(member as usize + 1) * 130]
            );
            check(&out, member).unwrap();
            assert!(check(&out, (member + 1) % 10).is_err());
            assert!(select(&input, &out, member).is_err());
            fs::remove_file(out).unwrap();
        }
        fs::remove_file(input).unwrap();
    }
    #[test]
    fn incomplete_duplicate_mean_and_foreign_stream_fail_closed() {
        let input = path("invalid");
        for data in [REAL[..1170].to_vec(), [REAL, &REAL[..130]].concat()] {
            fs::write(&input, data).unwrap();
            assert!(census(&input, 0, false, None).is_err());
        }
        for (offset, value) in [(8 + 42, 17), (8 + 41, 1), (8 + 49, 10), (8 + 44, 0)] {
            let mut data = REAL.to_vec();
            data[offset] = value;
            fs::write(&input, data).unwrap();
            assert!(census(&input, 0, false, None).is_err());
        }
        fs::remove_file(input).unwrap();
    }
    #[test]
    fn quarter_and_half_hour_units_are_not_seconds() {
        let expected = NaiveDate::from_ymd_opt(2013, 5, 31)
            .unwrap()
            .and_hms_opt(21, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp();
        for (unit, steps) in [(13, 12), (14, 6)] {
            let mut bytes = REAL[..130].to_vec();
            bytes[8 + 42] = 9;
            bytes[8 + 17] = unit;
            bytes[8 + 18] = steps;
            assert_eq!(identity(&bytes).unwrap().key.4, expected);
        }
        let mut bytes = REAL[..130].to_vec();
        bytes[8 + 17] = 254;
        bytes[8 + 18] = 13;
        assert!(identity(&bytes).is_err()); // 18:00:13 is not a native EDA UTC.
    }
    #[test]
    fn real_sst_local17_has_verified_member_octets() {
        const SST: &[u8] = include_bytes!("../tests/fixtures/era5-eda-ten-sst.grib");
        let input = path("sst");
        let output = path("sst-selected");
        let _ = fs::remove_file(&output);
        fs::write(&input, SST).unwrap();
        select(&input, &output, 7).unwrap();
        check(&output, 7).unwrap();
        let selected = fs::read(&output).unwrap();
        assert_eq!(selected[8 + 40], 17);
        assert_eq!(selected[8 + 49], 7);
        let mut expected = None;
        super::super::visit_grib1_envelopes(&input, |_, bytes| {
            if bytes[8 + 49] == 7 {
                expected = Some(bytes.to_vec());
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(selected, expected.unwrap());
        fs::remove_file(input).unwrap();
        fs::remove_file(output).unwrap();
    }

    // Header-only variants of the real CDS envelope validate routing and
    // byte preservation. Their encoded T2 values are not lake science data.
    fn table_field(table: u8, parameter: u8) -> Vec<u8> {
        let mut data = REAL.to_vec();
        for message in data.chunks_exact_mut(130) {
            message[8 + 3] = table;
            message[8 + 8] = parameter;
        }
        data
    }

    #[test]
    fn lake_fields_require_qualified_surface_identity() {
        for parameter in [8, 13, 14] {
            let data = table_field(228, parameter);
            let id = identity(&data[..130]).unwrap();
            assert_eq!(id.key.0, 228);
            assert_eq!(id.key.1, parameter);
            for (offset, value) in [(8 + 4, 7), (8 + 3, 129), (8 + 9, 100), (8 + 11, 1)] {
                let mut changed = data[..130].to_vec();
                changed[offset] = value;
                assert!(identity(&changed).is_err());
            }
        }
        for parameter in [7, 9, 10, 11, 12, 31, 34, 167] {
            assert!(identity(&table_field(228, parameter)[..130]).is_err());
        }
    }

    #[test]
    fn table_qualified_census_preserves_each_lake_field_and_member() {
        let input = path("lake-census");
        let output = path("lake-selected");
        let _ = fs::remove_file(&output);
        // Include a colliding numeric parameter in another accepted table.
        let fields = [table_field(128, 8), table_field(228, 8),
                      table_field(228, 13), table_field(228, 14)];
        let data = fields.concat();
        fs::write(&input, &data).unwrap();
        assert_eq!(census(&input, 7, false, None).unwrap(), (40, 4));
        select(&input, &output, 7).unwrap();
        let expected: Vec<u8> = fields.iter().flat_map(|field| field[7 * 130..8 * 130].iter().copied()).collect();
        assert_eq!(fs::read(&output).unwrap(), expected);
        assert_eq!(census(&output, 7, true, None).unwrap(), (4, 4));
        // A complete same-number table-128 field cannot repair table-228.
        let missing_lake_member = [&fields[0][..], &fields[1][..9 * 130],
                                   &fields[2][..], &fields[3][..]].concat();
        fs::write(&input, missing_lake_member).unwrap();
        assert!(census(&input, 7, false, None).is_err());
        let duplicate_lake_member = [data.as_slice(), &fields[1][..130]].concat();
        fs::write(&input, duplicate_lake_member).unwrap();
        assert!(census(&input, 7, false, None).is_err());
        fs::remove_file(input).unwrap();
        fs::remove_file(output).unwrap();
    }
}
