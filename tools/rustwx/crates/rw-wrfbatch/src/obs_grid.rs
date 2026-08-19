//! Native reader for `gpuwm-obs.radar-grid.v1` -- gridded radar
//! observations on a model mass grid.
//!
//! **Why this is a reader and not a writer.**  Every other input gap this
//! workspace had was closed by writing a wrfout, because the producer held
//! a full model state and a wrfout is the honest container for one.  This
//! file is the exception in both directions: it is ALREADY classic NetCDF
//! carrying `XLAT`/`XLONG` on the model mass grid, so there is nothing to
//! convert; and wrapping observations in a wrfout would make the renderer's
//! `TitleProvenance::LocalImport` declare that a model produced them.  A
//! plot that attributes radar returns to a forecast is worse than no plot.
//!
//! What is read (schema: `gpuwm/obs/radar_grid.py`):
//!
//! | variable | dims | meaning |
//! |---|---|---|
//! | `XLAT`, `XLONG` | `(south_north, west_east)` | model mass grid |
//! | `z_obs`, `z_mask` | `(level, south_north, west_east)` | reflectivity and its validity |
//! | `vr_obs`, `vr_mask` | `(radar, level, south_north, west_east)` | radial velocity per radar |
//! | `radar_lat`, `radar_lon` | `(radar,)` | site positions |
//! | `radar_id` | `(radar, nchar)` | site identifiers |
//!
//! A mask is `int8`, nonzero where the cell carries an observation.  The
//! masks are load-bearing: `z_obs` is dense and its unobserved cells hold a
//! fill value, so a reduction that ignores `z_mask` maps the fill onto the
//! colour table and paints observations where the radars saw nothing.

use std::path::Path;

use netcrust::File as NcFile;

/// One observation file, fully read.
pub struct ObsRadarGrid {
    pub ny: usize,
    pub nx: usize,
    pub levels: usize,
    pub lat_deg: Vec<f32>,
    pub lon_deg: Vec<f32>,
    /// `(level, ny, nx)` reflectivity, dBZ.
    pub z_obs: Vec<f64>,
    /// `(level, ny, nx)`, nonzero where observed.
    pub z_mask: Vec<i8>,
    /// `(radar, level, ny, nx)` radial velocity, m/s. Empty when absent.
    pub vr_obs: Vec<f64>,
    pub vr_mask: Vec<i8>,
    pub radars: Vec<RadarSite>,
    /// The file's own `provenance` global attribute, verbatim.
    pub provenance: Option<String>,
    pub valid_time: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RadarSite {
    pub id: String,
    pub lat_deg: f64,
    pub lon_deg: f64,
}

/// The schema stamp the file must declare.  A file that does not declare
/// it is refused by name rather than read hopefully: every variable below
/// is addressed by a fixed name and a fixed dimension order, and a file
/// with the right names in a different order is exactly the failure a
/// shape check cannot see.
const SCHEMA: &str = "gpuwm-obs.radar-grid.v1";

impl ObsRadarGrid {
    pub fn open(path: &Path) -> Result<Self, String> {
        let nc = netcrust::open(path)
            .map_err(|err| format!("open {}: {err}", path.display()))?;
        let declared = string_attribute(&nc, "schema")
            .or_else(|| string_attribute(&nc, "Conventions"));
        match declared.as_deref() {
            Some(value) if value.contains(SCHEMA) => {}
            Some(other) => {
                return Err(format!(
                    "{} declares schema {other:?}; this reader implements {SCHEMA}",
                    path.display()
                ));
            }
            None => {
                return Err(format!(
                    "{} declares no schema attribute; this reader implements {SCHEMA} \
                     and will not guess at an undeclared layout",
                    path.display()
                ));
            }
        }

        let (ny, nx, lat_deg, lon_deg) = read_mass_grid(&nc, path)?;
        let points = ny * nx;

        let z = read_f64(&nc, "z_obs", path)?;
        if z.len() % points != 0 || z.is_empty() {
            return Err(format!(
                "{}: z_obs carries {} value(s), which is not a whole number of \
                 {ny}x{nx} planes",
                path.display(),
                z.len()
            ));
        }
        let levels = z.len() / points;
        let z_mask = read_i8(&nc, "z_mask", path)?;
        if z_mask.len() != z.len() {
            return Err(format!(
                "{}: z_mask has {} value(s) against z_obs's {}; the mask is what \
                 separates an observation from a fill value, so a partial one \
                 cannot be used",
                path.display(),
                z_mask.len(),
                z.len()
            ));
        }

        let radars = read_radars(&nc, path)?;
        let (vr_obs, vr_mask) = match read_f64(&nc, "vr_obs", path) {
            Ok(vr) => {
                let mask = read_i8(&nc, "vr_mask", path)?;
                if mask.len() != vr.len() {
                    return Err(format!(
                        "{}: vr_mask has {} value(s) against vr_obs's {}",
                        path.display(),
                        mask.len(),
                        vr.len()
                    ));
                }
                (vr, mask)
            }
            Err(_) => (Vec::new(), Vec::new()),
        };

        Ok(Self {
            ny,
            nx,
            levels,
            lat_deg,
            lon_deg,
            z_obs: z,
            z_mask,
            vr_obs,
            vr_mask,
            radars,
            provenance: string_attribute(&nc, "provenance"),
            valid_time: string_attribute(&nc, "valid_time"),
        })
    }

    pub fn points(&self) -> usize {
        self.ny * self.nx
    }

    /// Column-max reflectivity over the OBSERVED levels only.
    ///
    /// A column with no observed level is NaN, not the fill value and not
    /// zero: "the radars did not see here" and "the radars saw nothing
    /// here" are different statements and only one of them is a forecast
    /// verification.
    pub fn z_composite(&self) -> Vec<f32> {
        let points = self.points();
        let mut out = vec![f32::NAN; points];
        for level in 0..self.levels {
            let base = level * points;
            for index in 0..points {
                if self.z_mask[base + index] == 0 {
                    continue;
                }
                let value = self.z_obs[base + index];
                if !value.is_finite() {
                    continue;
                }
                let value = value as f32;
                if out[index].is_nan() || value > out[index] {
                    out[index] = value;
                }
            }
        }
        out
    }

    /// How many levels each column has an observation on.
    pub fn coverage_depth(&self) -> Vec<f32> {
        let points = self.points();
        let mut out = vec![0.0f32; points];
        for level in 0..self.levels {
            let base = level * points;
            for index in 0..points {
                if self.z_mask[base + index] != 0 {
                    out[index] += 1.0;
                }
            }
        }
        out
    }

    /// How many DISTINCT radars contribute a radial velocity to each cell.
    ///
    /// The planning question this answers is multi-Doppler coverage, so a
    /// radar that sees a column on six levels counts once, not six times.
    pub fn radar_overlap(&self) -> Result<Vec<f32>, String> {
        if self.vr_mask.is_empty() || self.radars.is_empty() {
            return Err("this file carries no vr_mask; radar overlap is a \
                        property of the velocity volumes"
                .to_string());
        }
        let points = self.points();
        let per_radar = self.levels * points;
        if self.vr_mask.len() != self.radars.len() * per_radar {
            return Err(format!(
                "vr_mask has {} value(s); {} radar(s) x {} level(s) x {points} \
                 point(s) needs {}",
                self.vr_mask.len(),
                self.radars.len(),
                self.levels,
                self.radars.len() * per_radar
            ));
        }
        let mut out = vec![0.0f32; points];
        for radar in 0..self.radars.len() {
            let radar_base = radar * per_radar;
            for index in 0..points {
                let seen = (0..self.levels)
                    .any(|level| self.vr_mask[radar_base + level * points + index] != 0);
                if seen {
                    out[index] += 1.0;
                }
            }
        }
        Ok(out)
    }

    /// Radial velocity at the LOWEST observed level of each column, for one
    /// radar (`None` = the first radar that observes the column).
    ///
    /// The lowest observed level is the one a rotation signature lives on
    /// and the one closest to the ground a beam reaches; a mid-volume slice
    /// would show a different storm.
    pub fn vr_lowest(&self, radar_index: Option<usize>) -> Result<Vec<f32>, String> {
        if self.vr_obs.is_empty() {
            return Err("this file carries no vr_obs".to_string());
        }
        let points = self.points();
        let per_radar = self.levels * points;
        let radars = self.radars.len().max(1);
        if let Some(index) = radar_index {
            if index >= radars {
                return Err(format!(
                    "radar index {index} is out of range; the file carries {radars}"
                ));
            }
        }
        let candidates: Vec<usize> = match radar_index {
            Some(index) => vec![index],
            None => (0..radars).collect(),
        };
        let mut out = vec![f32::NAN; points];
        for index in 0..points {
            'column: for level in 0..self.levels {
                for radar in &candidates {
                    let slot = radar * per_radar + level * points + index;
                    if slot >= self.vr_mask.len() || self.vr_mask[slot] == 0 {
                        continue;
                    }
                    let value = self.vr_obs[slot];
                    if value.is_finite() {
                        out[index] = value as f32;
                        break 'column;
                    }
                }
            }
        }
        Ok(out)
    }

    /// Cells one named radar contributes a velocity to, as a 0/1 plane.
    pub fn radar_contribution(&self, radar_index: usize) -> Result<Vec<f32>, String> {
        let points = self.points();
        let per_radar = self.levels * points;
        if radar_index >= self.radars.len() {
            return Err(format!(
                "radar index {radar_index} is out of range; the file carries {}",
                self.radars.len()
            ));
        }
        let base = radar_index * per_radar;
        Ok((0..points)
            .map(|index| {
                let seen = (0..self.levels)
                    .any(|level| self.vr_mask[base + level * points + index] != 0);
                if seen { 1.0 } else { 0.0 }
            })
            .collect())
    }
}

fn read_mass_grid(
    nc: &NcFile,
    path: &Path,
) -> Result<(usize, usize, Vec<f32>, Vec<f32>), String> {
    let lat = nc
        .read_array_f64("XLAT")
        .map_err(|err| format!("{}: read XLAT: {err}", path.display()))?;
    let lon = nc
        .read_array_f64("XLONG")
        .map_err(|err| format!("{}: read XLONG: {err}", path.display()))?;
    if lat.shape().len() != 2 {
        return Err(format!(
            "{}: XLAT has {} dimension(s); the mass grid is (south_north, west_east)",
            path.display(),
            lat.shape().len()
        ));
    }
    if lat.shape() != lon.shape() {
        return Err(format!(
            "{}: XLAT is {:?} and XLONG is {:?}",
            path.display(),
            lat.shape(),
            lon.shape()
        ));
    }
    let ny = lat.shape()[0];
    let nx = lat.shape()[1];
    Ok((
        ny,
        nx,
        lat.values().iter().map(|value| *value as f32).collect(),
        lon.values().iter().map(|value| *value as f32).collect(),
    ))
}

fn read_f64(nc: &NcFile, name: &str, path: &Path) -> Result<Vec<f64>, String> {
    nc.read_array_f64(name)
        .map(netcrust::DataArray::into_values)
        .map_err(|err| format!("{}: read {name}: {err}", path.display()))
}

fn read_i8(nc: &NcFile, name: &str, path: &Path) -> Result<Vec<i8>, String> {
    Ok(read_f64(nc, name, path)?
        .into_iter()
        .map(|value| if value.abs() < 0.5 { 0i8 } else { 1i8 })
        .collect())
}

fn read_radars(nc: &NcFile, path: &Path) -> Result<Vec<RadarSite>, String> {
    let Ok(lat) = read_f64(nc, "radar_lat", path) else {
        return Ok(Vec::new());
    };
    let lon = read_f64(nc, "radar_lon", path)?;
    if lon.len() != lat.len() {
        return Err(format!(
            "{}: radar_lat has {} entries and radar_lon {}",
            path.display(),
            lat.len(),
            lon.len()
        ));
    }
    let mut identifiers = read_radar_ids(nc, &lat, &lon);
    // Second source: the site's OWN POSITION against the vendored WSR-88D
    // table.  MEASURED on a real three-radar file (KPUX/KGLD/KFTG,
    // 2021-12-30): the roster is in the `provenance` attribute exactly as
    // the reader above wants it, and netcrust returns no attribute at all
    // because the value is 15916 bytes -- the longest attribute it DID
    // return from the same file is 718.  So the file's two id-carrying
    // fields were both unreadable and every panel said `site0`/`site1`.
    //
    // Position is the identity a radar cannot misstate, and the table
    // ships inside this binary, so this needs nothing of the container.
    for (index, slot) in identifiers.iter_mut().enumerate() {
        if slot.is_none() {
            *slot = site_table_label(lat[index], lon[index]);
        }
    }
    // Whatever is STILL unlabelled is a degraded label and says so.
    let unlabelled = identifiers.iter().filter(|id| id.is_none()).count();
    if unlabelled > 0 {
        eprintln!(
            "SITE_LABEL_UNAVAILABLE\t{unlabelled}\tof {}\tthe `radar_id` \
             variable is NC_CHAR in a NETCDF4_CLASSIC container, which \
             arrives as DataType::String and is refused by every numeric \
             read; the `provenance` global attribute carrying the same \
             roster was not returned by the reader; and no vendored WSR-88D \
             site sits within {SITE_MATCH_TOLERANCE_DEG} deg of those \
             positions.  They are drawn at their radar_lat/radar_lon with \
             POSITIONAL names",
            lat.len()
        );
    }
    Ok(lat
        .into_iter()
        .zip(lon)
        .enumerate()
        .map(|(index, (lat_deg, lon_deg))| RadarSite {
            id: identifiers
                .get(index)
                .cloned()
                .flatten()
                .unwrap_or_else(|| format!("site{index}")),
            lat_deg,
            lon_deg,
        })
        .collect())
}

/// How close a vendored site must sit to a file's `radar_lat`/`radar_lon`
/// to lend it its id.
///
/// 0.02 deg is about 2 km.  The table's own coordinates are published to
/// four decimals and the volumes' Message-31 VOL blocks agree with them to
/// ~0.001 deg, so the honest fixes all land far inside this; the nearest
/// pair of WSR-88D sites in the network is two orders of magnitude further
/// apart than this, so a match inside it is unique or it is not a match.
pub const SITE_MATCH_TOLERANCE_DEG: f64 = 0.02;

/// The vendored WSR-88D id whose published position is this one, or `None`.
///
/// UNIQUENESS is required, not nearest-wins: two candidates inside the
/// tolerance mean the tolerance is wrong for this network, and labelling a
/// marker with the wrong radar is worse than leaving it positional.
fn site_table_label(lat_deg: f64, lon_deg: f64) -> Option<String> {
    let mut found: Option<&str> = None;
    for (id, _name, site_lat, site_lon, _alt) in wx_radar::sites::SITES {
        if (site_lat - lat_deg).abs() > SITE_MATCH_TOLERANCE_DEG
            || (site_lon - lon_deg).abs() > SITE_MATCH_TOLERANCE_DEG
        {
            continue;
        }
        if found.is_some() {
            eprintln!(
                "SITE_LABEL_DECLINED\t-\ttwo vendored sites sit within \
                 {SITE_MATCH_TOLERANCE_DEG} deg of ({lat_deg:.4}, \
                 {lon_deg:.4}); the position does not identify one radar"
            );
            return None;
        }
        found = Some(id);
    }
    found.map(str::to_string)
}

/// Site identifiers, from the file's own `provenance` JSON, POSITION-CHECKED.
///
/// The obvious source is the `radar_id` variable, and it is unreadable
/// here: the writer emits `NETCDF4_CLASSIC`, where `(radar, nchar)`
/// `NC_CHAR` arrives through netcrust's HDF5 path as `DataType::String`
/// and every numeric read of it is refused by type.  (A classic-container
/// char array reads fine as byte codes -- that is how the wrfout lane
/// reads `Times` -- so this is a container difference, not a schema one.)
///
/// The `provenance` global attribute is a JSON STRING, which netcrust does
/// read, and it carries the same roster with `id`, `lat_deg` and
/// `lon_deg` per radar.  Matching it to the numeric `radar_lat`/`radar_lon`
/// arrays by POSITION alone would be trusting two independently written
/// lists to be in the same order; a label on the wrong marker is worse
/// than no label, so each entry's coordinates must agree with the
/// variable's to within a tenth of a degree or that site keeps its
/// positional name.
fn read_radar_ids(nc: &NcFile, lat: &[f64], lon: &[f64]) -> Vec<Option<String>> {
    let mut out = vec![None; lat.len()];
    let Some(text) = string_attribute(nc, "provenance") else {
        return out;
    };
    let Ok(document) = serde_json::from_str::<serde_json::Value>(&text) else {
        return out;
    };
    let Some(radars) = document.get("radars").and_then(serde_json::Value::as_array) else {
        return out;
    };
    for (index, record) in radars.iter().enumerate() {
        if index >= out.len() {
            break;
        }
        let (Some(id), Some(record_lat), Some(record_lon)) = (
            record.get("id").and_then(serde_json::Value::as_str),
            record.get("lat_deg").and_then(serde_json::Value::as_f64),
            record.get("lon_deg").and_then(serde_json::Value::as_f64),
        ) else {
            continue;
        };
        if (record_lat - lat[index]).abs() > 0.1 || (record_lon - lon[index]).abs() > 0.1 {
            eprintln!(
                "SITE_LABEL_DECLINED\t{index}\tprovenance names {id} at \
                 ({record_lat:.4}, {record_lon:.4}) but radar_lat/radar_lon put \
                 site {index} at ({:.4}, {:.4})",
                lat[index], lon[index]
            );
            continue;
        }
        out[index] = Some(id.to_string());
    }
    out
}

fn string_attribute(nc: &NcFile, name: &str) -> Option<String> {
    nc.attribute(name)
        .and_then(|attribute| attribute.as_string().map(str::to_string))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe(levels: usize, ny: usize, nx: usize) -> ObsRadarGrid {
        let points = ny * nx;
        ObsRadarGrid {
            ny,
            nx,
            levels,
            lat_deg: vec![0.0; points],
            lon_deg: vec![0.0; points],
            z_obs: vec![0.0; levels * points],
            z_mask: vec![0i8; levels * points],
            vr_obs: Vec::new(),
            vr_mask: Vec::new(),
            radars: Vec::new(),
            provenance: None,
            valid_time: None,
        }
    }

    #[test]
    fn a_radar_position_carries_its_own_identity() {
        // The three radars of the MEASURED file, at the positions its
        // `radar_lat`/`radar_lon` carry (Message-31 VOL-block fixes, so
        // they differ from the vendored table in the fourth decimal).
        assert_eq!(
            site_table_label(38.459_548_950_195_31, -104.181_350_708_007_81),
            Some("KPUX".to_string())
        );
        assert_eq!(
            site_table_label(39.366_943_359_375, -101.700_279_235_839_84),
            Some("KGLD".to_string())
        );
        assert_eq!(
            site_table_label(39.786_640_167_236_33, -104.545_806_884_765_62),
            Some("KFTG".to_string())
        );
    }

    #[test]
    fn a_position_no_vendored_site_sits_at_stays_unlabelled() {
        // Mid-Pacific: a label here would be an invention, and an
        // invented id on a marker is worse than a positional one.
        assert_eq!(site_table_label(0.0, -160.0), None);
    }

    #[test]
    fn an_unobserved_column_is_blank_not_the_fill_value() {
        // The defect this exists to prevent: a dense array whose
        // unobserved cells hold -9.99e30, reduced without the mask,
        // paints a -9.99e30 composite everywhere the radars did not look.
        let mut grid = probe(2, 1, 3);
        grid.z_obs = vec![-9.99e30, 40.0, -9.99e30, -9.99e30, 45.0, -9.99e30];
        grid.z_mask = vec![0, 1, 0, 0, 1, 0];
        let composite = grid.z_composite();
        assert!(composite[0].is_nan());
        assert_eq!(composite[1], 45.0);
        assert!(composite[2].is_nan());
    }

    #[test]
    fn coverage_depth_counts_levels_not_values() {
        let mut grid = probe(3, 1, 2);
        grid.z_mask = vec![1, 0, 1, 1, 0, 0];
        assert_eq!(grid.coverage_depth(), vec![2.0, 1.0]);
    }

    #[test]
    fn radar_overlap_counts_each_radar_once_however_many_levels_it_sees() {
        let mut grid = probe(3, 1, 2);
        grid.radars = vec![
            RadarSite { id: "A".into(), lat_deg: 0.0, lon_deg: 0.0 },
            RadarSite { id: "B".into(), lat_deg: 0.0, lon_deg: 0.0 },
        ];
        // radar A sees point 0 on all three levels; radar B sees point 0
        // on one level and point 1 on one level.
        grid.vr_mask = vec![
            1, 0, 1, 0, 1, 0, // radar A
            1, 0, 0, 1, 0, 0, // radar B
        ];
        grid.vr_obs = vec![0.0; grid.vr_mask.len()];
        assert_eq!(grid.radar_overlap().unwrap(), vec![2.0, 1.0]);
    }

    #[test]
    fn vr_lowest_takes_the_lowest_OBSERVED_level_not_level_zero() {
        let mut grid = probe(3, 1, 2);
        grid.radars = vec![RadarSite { id: "A".into(), lat_deg: 0.0, lon_deg: 0.0 }];
        grid.vr_mask = vec![0, 1, 1, 0, 0, 0];
        grid.vr_obs = vec![99.0, -12.0, 7.5, 99.0, 99.0, 99.0];
        let lowest = grid.vr_lowest(None).unwrap();
        assert_eq!(lowest[0], 7.5, "level 1 is the lowest observed for point 0");
        assert_eq!(lowest[1], -12.0);
    }

    #[test]
    fn a_file_with_no_velocity_says_so_rather_than_drawing_zeros() {
        let grid = probe(1, 1, 1);
        assert!(grid.vr_lowest(None).is_err());
        assert!(grid.radar_overlap().is_err());
    }

    #[test]
    fn an_out_of_range_radar_is_named_not_clamped() {
        let mut grid = probe(1, 1, 1);
        grid.radars = vec![RadarSite { id: "A".into(), lat_deg: 0.0, lon_deg: 0.0 }];
        grid.vr_obs = vec![1.0];
        grid.vr_mask = vec![1];
        let error = grid.vr_lowest(Some(4)).unwrap_err();
        assert!(error.contains("out of range"), "{error}");
        assert!(grid.radar_contribution(4).is_err());
    }
}
