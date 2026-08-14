//! Write a decoded ODIM polar volume as a `gpuwm-obs.radar-sweeps.v3` pack.
//!
//! The bridge, and deliberately nothing more. gpuwm already has a complete
//! radar-to-LETKF path — dealias, superob, radar grid, LETKF adapter — and
//! none of it below the decoder is American. The join is at the *top*: one
//! new pack writer, so that a European radial velocity is assimilated by
//! exactly the code that assimilates an American one.
//!
//! The container is not re-implemented here. [`rw_nexrad::pack`] owns the
//! byte format — magic, 64-byte header, JSON metadata, contiguous payload —
//! and this module builds its [`PackMeta`] and hands it the same
//! [`PayloadBuilder`]. One implementation of the format, two producers, which
//! is what keeps a European pack readable by the reader that was written for
//! an American one.
//!
//! # What ODIM makes this writer say differently
//!
//! - **Gate centres.** ODIM's `rstart` is the *start* of the first bin, and
//!   the pack's `first_gate_range_m` is the *centre* of it — that is what the
//!   Python reader's `slant_range_m()` computes from. Half a bin is added
//!   here, once, at the seam. Py-ART omits it and places every gate half a
//!   bin too close; at a 250 m bin that is a 125 m beam-height bias in the
//!   same direction on every assimilated gate.
//! - **Nyquist granularity.** ODIM declares one `/datasetN/how/NI` per sweep.
//!   The pack states `nyquist_granularity: "sweep"` and omits the per-radial
//!   array rather than broadcasting one number across the radials.
//! - **Framing.** There is none: an ODIM file is HDF5, not Archive-II, so the
//!   pack's `framing` key is absent rather than filled with a placeholder.

use std::collections::BTreeMap;
use std::error::Error;

use rw_nexrad::pack::{
    ArrayEntry, AssembledEntry, AssembledMember, DecodeParams, MomentEntry, PackMeta,
    PayloadBuilder, SWEEPS_SCHEMA_ODIM, SiteEntry, SweepEntry, VolumeEntry,
};

use crate::assemble::AssembleReport;

use crate::censor;
use crate::volume::{NyquistSource, PolarVolume, Sweep};

/// What the pack's `nyquist_granularity` says for every ODIM sweep.
///
/// Not a parameter. ODIM has one `how/NI` per `/datasetN` and no per-ray
/// Nyquist anywhere in the model, so this is a property of the format.
pub const NYQUIST_GRANULARITY: &str = "sweep";

/// How a sweep was disposed of, so a thin pack is never mistaken for a thin
/// volume.
#[derive(Debug, Default, Clone, Copy)]
pub struct PackCensus {
    pub dropped_sweeps: usize,
    pub dropped_moments: usize,
    pub sweeps_without_nyquist: usize,
}

/// Options for [`build_pack`].
#[derive(Debug, Clone)]
pub struct PackOptions {
    /// Quantities to carry, ODIM spelling (`DBZH`, `VRADH`, ...). Empty means
    /// every quantity the volume holds.
    pub quantities: Vec<String>,
    /// Drop cuts above this elevation. The 90-degree birdbath a Dutch volume
    /// opens with is a calibration cut, not an observation of anything a
    /// model grid has a column for.
    pub max_elevation_deg: f64,
    /// Trim gates beyond this range.
    pub max_range_km: f64,
}

impl Default for PackOptions {
    fn default() -> Self {
        Self {
            quantities: Vec::new(),
            max_elevation_deg: 90.0,
            max_range_km: f64::INFINITY,
        }
    }
}

fn boxed(message: String) -> Box<dyn Error> {
    Box::<dyn Error>::from(message)
}

/// Build the `v3` pack metadata and payload for one decoded volume.
///
/// `file_name`, `file_bytes` and `file_sha256` describe the file on disk as
/// it arrived; they are the volume's identity and are not recomputed from the
/// decode.
///
/// `assembled` is `None` for the ordinary one-file volume. When the volume
/// came out of [`crate::assemble`] it carries the member manifest, and then
/// `file_sha256` must be the manifest digest rather than any file's, because
/// there is no single file to digest.
pub fn build_pack(
    volume: &PolarVolume,
    options: &PackOptions,
    file_name: &str,
    file_bytes: usize,
    file_sha256: &str,
    assembled: Option<&AssembleReport>,
) -> Result<(PackMeta, Vec<u8>), Box<dyn Error>> {
    let mut builder = PayloadBuilder::new();
    let mut entries: Vec<SweepEntry> = Vec::new();
    let mut census = PackCensus::default();
    let mut trimmed_gates = 0usize;

    for sweep in &volume.sweeps {
        if sweep.elevation_deg > options.max_elevation_deg {
            census.dropped_sweeps += 1;
            continue;
        }
        match build_sweep(
            sweep,
            options,
            &mut builder,
            &mut census,
            &mut trimmed_gates,
        ) {
            Ok(Some(entry)) => entries.push(entry),
            Ok(None) => census.dropped_sweeps += 1,
            Err(error) => return Err(error),
        }
    }

    if entries.is_empty() {
        return Err(boxed(format!(
            "{file_name}: every one of the volume's {} sweeps was dropped by the \
             filters (max_elevation_deg {}, quantities {:?}); a pack with no sweeps \
             would claim the volume was empty. Widen --max-elevation-deg or \
             --quantities, or check `rw_odim inspect` for what the file holds.",
            volume.sweeps.len(),
            options.max_elevation_deg,
            options.quantities
        )));
    }

    let (payload, arrays) = builder.finish();
    let meta = PackMeta {
        schema: SWEEPS_SCHEMA_ODIM.to_string(),
        status: "READY".to_string(),
        site: site_entry(volume),
        volume: volume_entry(volume, file_name, file_bytes, file_sha256, assembled)?,
        params: DecodeParams {
            moments: options.quantities.clone(),
            max_range_km: options.max_range_km,
            max_elevation_deg: options.max_elevation_deg,
            // Always true: an ODIM pack without its censor planes would have
            // thrown away every correct negative in the file, which is the
            // one thing this decoder exists to keep.
            censor_flags: true,
        },
        sweeps: entries,
        arrays: arrays as BTreeMap<String, ArrayEntry>,
        payload_bytes: payload.len(),
        content_sha256: sha256_hex(&payload),
        dropped_sweeps: census.dropped_sweeps,
        dropped_moments: census.dropped_moments,
        trimmed_gates,
    };
    Ok((meta, payload))
}

fn build_sweep(
    sweep: &Sweep,
    options: &PackOptions,
    builder: &mut PayloadBuilder,
    census: &mut PackCensus,
    trimmed_gates: &mut usize,
) -> Result<Option<SweepEntry>, Box<dyn Error>> {
    let radial_count = sweep.nrays;
    if sweep.azimuth_deg.len() != radial_count {
        return Err(boxed(format!(
            "sweep {} declares nrays {} but carries {} azimuths; the pack's \
             azimuth array is what every downstream beam vector is built from, \
             so a disagreement here is refused rather than padded",
            sweep.index,
            radial_count,
            sweep.azimuth_deg.len()
        )));
    }

    // Which ODIM quantity wins each canonical token on THIS cut, decided
    // before any of them is written. Deciding inside the loop would let the
    // canonical moment depend on the order the file happens to list its
    // quantities in, and a German volume carries DBZH and TH together.
    let mut canonical: BTreeMap<&'static str, (usize, &str)> = BTreeMap::new();
    for moment in &sweep.moments {
        if !options.quantities.is_empty()
            && !options
                .quantities
                .iter()
                .any(|wanted| wanted.eq_ignore_ascii_case(&moment.quantity))
        {
            continue;
        }
        if let Some((token, rank)) = crate::quantity::canonical_claim(&moment.quantity) {
            let better = match canonical.get(token) {
                None => true,
                Some((held, _)) => rank < *held,
            };
            if better {
                canonical.insert(token, (rank, moment.quantity.as_str()));
            }
        }
    }

    let mut moments: Vec<MomentEntry> = Vec::new();
    for moment in &sweep.moments {
        if !options.quantities.is_empty()
            && !options
                .quantities
                .iter()
                .any(|wanted| wanted.eq_ignore_ascii_case(&moment.quantity))
        {
            census.dropped_moments += 1;
            continue;
        }
        if moment.nrays != radial_count {
            return Err(boxed(format!(
                "sweep {} moment {} has {} rays against the cut's {}; the pack \
                 stores one azimuth axis per sweep, so a moment on a different \
                 ray count cannot be placed on it",
                sweep.index, moment.quantity, moment.nrays, radial_count
            )));
        }
        if moment.censor.len() != moment.values.len() {
            return Err(boxed(format!(
                "sweep {} moment {} has {} values and {} censor codes",
                sweep.index,
                moment.quantity,
                moment.values.len(),
                moment.censor.len()
            )));
        }

        // Trim the far end, keeping whole gates.
        let gates = if options.max_range_km.is_finite() {
            let limit_m = options.max_range_km * 1000.0;
            let mut keep = moment.nbins;
            while keep > 0 && gate_centre_m(sweep, keep - 1) > limit_m {
                keep -= 1;
            }
            *trimmed_gates += moment.nbins - keep;
            keep
        } else {
            moment.nbins
        };
        if gates == 0 {
            census.dropped_moments += 1;
            continue;
        }

        let mut values = Vec::with_capacity(radial_count * gates);
        let mut codes = Vec::with_capacity(radial_count * gates);
        for ray in 0..radial_count {
            let row = ray * moment.nbins;
            for bin in 0..gates {
                let value = moment.values[row + bin];
                let code = moment.censor[row + bin];
                let narrowed = value as f32;
                // The Python reader refuses a pack whose censor plane and
                // moment plane disagree about which gates are numbers, and it
                // is right to.  Catch the one way this writer could cause
                // that -- an f64 that is finite but overflows f32 -- here,
                // where the gate can still be named.
                if code == censor::MEASURED && !narrowed.is_finite() {
                    return Err(boxed(format!(
                        "sweep {} moment {} ray {ray} bin {bin} is coded measured \
                         with value {value}, which is not finite as the f32 the \
                         pack stores; the censor plane and the moment plane would \
                         contradict each other in the file",
                        sweep.index, moment.quantity
                    )));
                }
                if code != censor::MEASURED && narrowed.is_finite() {
                    return Err(boxed(format!(
                        "sweep {} moment {} ray {ray} bin {bin} is coded {} yet \
                         holds the finite value {value}; a censored gate must be \
                         NaN in the moment plane",
                        sweep.index,
                        moment.quantity,
                        censor::name(code)
                    )));
                }
                values.push(narrowed);
                codes.push(code);
            }
        }

        let shape = vec![radial_count, gates];
        let array = builder.push_f32(&values, shape.clone());
        let censor_array = builder.push_u8(&codes, shape);
        // The canonical token when this quantity won it, its own ODIM name
        // otherwise. A moment that keeps its ODIM name is in the pack and
        // readable, but no consumer reads it as an observation -- which is
        // the right answer for the uncorrected TH beside a corrected DBZH.
        let claimed = canonical
            .iter()
            .find(|(_, (_, winner))| *winner == moment.quantity)
            .map(|(token, _)| *token);
        let product = claimed.unwrap_or(moment.quantity.as_str()).to_string();
        let source_quantity = (product != moment.quantity).then(|| moment.quantity.clone());
        moments.push(MomentEntry {
            product,
            unit: moment.unit.clone(),
            gate_count: gates,
            // ODIM's rstart is the START of the first bin; the pack's field
            // is its CENTRE.  Half a bin, added once, at the seam.
            first_gate_range_m: sweep.range_start_m + 0.5 * sweep.range_scale_m,
            gate_size_m: sweep.range_scale_m,
            array,
            censor_array: Some(censor_array),
            source_quantity,
        });
    }

    if moments.is_empty() {
        return Ok(None);
    }

    let azimuth: Vec<f32> = sweep.azimuth_deg.iter().map(|a| *a as f32).collect();
    let elevation: Vec<f32> = match &sweep.ray_elevation_deg {
        // Where the writer recorded the antenna's actual elevation per ray,
        // that is what the beam vector should be built from.
        Some(per_ray) if per_ray.len() == radial_count => {
            per_ray.iter().map(|e| *e as f32).collect()
        }
        _ => vec![sweep.elevation_deg as f32; radial_count],
    };
    let azimuth_array = builder.push_f32(&azimuth, vec![radial_count]);
    let elevation_array = builder.push_f32(&elevation, vec![radial_count]);

    let nyquist = match sweep.nyquist.source {
        // A dual-PRF sweep with no declared NI is not dealiasable, and a
        // single-PRF estimate for one would be a factor of three small.  The
        // sweep still ships -- its reflectivity is perfectly good -- but it
        // carries no interval, and the dealiaser refuses velocity on it by
        // name, which is the behaviour that is load-bearing.
        NyquistSource::Unavailable => {
            census.sweeps_without_nyquist += 1;
            None
        }
        _ => sweep.nyquist.interval_ms,
    };

    Ok(Some(SweepEntry {
        sweep_index: u16::try_from(sweep.index).map_err(|_| {
            boxed(format!(
                "sweep index {} does not fit the pack's u16 sweep_index",
                sweep.index
            ))
        })?,
        elevation_number: u8::try_from(sweep.index.min(255)).unwrap_or(255),
        elevation_angle_deg: sweep.elevation_deg,
        nyquist_velocity_ms: nyquist,
        // ODIM records one interval for the cut, so its radials cannot
        // disagree.  False here is a statement about the format, not an
        // unchecked default.
        nyquist_radials_disagree: false,
        // The three NEXRAD radial-status fields have no ODIM counterpart: an
        // ODIM `/datasetN` is a whole cut by construction, with no partial
        // sector and no start/end markers to be missing.  Zero and `complete`
        // are what that means, not placeholders for something unread.
        start_status: 0,
        end_status: 0,
        cut_sector: 0,
        complete: true,
        radial_count,
        azimuth_array,
        elevation_array,
        // Omitted deliberately: see NYQUIST_GRANULARITY.
        nyquist_by_radial_array: None,
        nyquist_granularity: Some(NYQUIST_GRANULARITY.to_string()),
        moments,
    }))
}

/// Range to the centre of `bin`, metres.
fn gate_centre_m(sweep: &Sweep, bin: usize) -> f64 {
    sweep.range_start_m + (bin as f64 + 0.5) * sweep.range_scale_m
}

fn site_entry(volume: &PolarVolume) -> SiteEntry {
    let id = volume
        .source
        .nod
        .clone()
        .or_else(|| volume.source.wigos.clone())
        .or_else(|| volume.source.wmo.clone())
        .unwrap_or_else(|| "unknown".to_string());
    let name = volume
        .source
        .place
        .clone()
        .unwrap_or_else(|| id.to_uppercase());
    SiteEntry {
        id,
        name,
        lat_deg: volume.site.latitude_deg,
        lon_deg: volume.site.longitude_deg,
        // The antenna height the frozen ODIM site table has as null for all
        // 136 radars.  It is in every volume, so a pack never needs the table
        // to answer it and a superob never has to guess.
        alt_m: volume.site.height_m,
        source: format!("odim:/where + /what/source {:?}", volume.source.raw),
    }
}

fn volume_entry(
    volume: &PolarVolume,
    file_name: &str,
    file_bytes: usize,
    file_sha256: &str,
    assembled: Option<&AssembleReport>,
) -> Result<VolumeEntry, Box<dyn Error>> {
    let nominal = volume.nominal_time.ok_or_else(|| {
        boxed(format!(
            "{file_name}: the volume declares no nominal time (/what/date + \
             /what/time); every observation the pack yields would be unplaceable \
             in the assimilation window"
        ))
    })?;
    let station_id = volume
        .source
        .nod
        .clone()
        .or_else(|| volume.source.wigos.clone())
        .unwrap_or_else(|| "unknown".to_string());
    let epoch_day = nominal.timestamp().div_euclid(86_400);
    let second_of_day = nominal.timestamp().rem_euclid(86_400);
    Ok(VolumeEntry {
        file: file_name.to_string(),
        bytes: file_bytes,
        sha256: file_sha256.to_string(),
        station_id,
        valid_time: nominal.to_rfc3339(),
        volume_date: u16::try_from(epoch_day).unwrap_or(0),
        volume_time_ms: u32::try_from(second_of_day * 1000).unwrap_or(0),
        // No Archive-II framing exists for an HDF5 container; the key is
        // absent rather than invented.
        framing: None,
        assembled: assembled.map(|report| AssembledEntry {
            files: report.files,
            sweeps_merged: report.merged,
            manifest_sha256: report.manifest_sha256.clone(),
            members: report
                .members
                .iter()
                .map(|member| AssembledMember {
                    name: member.name.clone(),
                    bytes: member.bytes,
                    sha256: member.sha256.clone(),
                })
                .collect(),
        }),
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pack_places_the_first_gate_at_its_centre_not_its_start() {
        // ODIM rstart is the start of bin 0.  The pack field the Python
        // reader builds its range axis from is the centre.  This is the
        // half-bin Py-ART drops.
        let sweep = Sweep {
            index: 0,
            path: "/dataset1".to_string(),
            elevation_deg: 0.3,
            nrays: 1,
            nbins: 4,
            range_scale_m: 250.0,
            range_start_m: 0.0,
            a1gate: None,
            start_time: None,
            end_time: None,
            nyquist: crate::volume::Nyquist {
                interval_ms: Some(20.0),
                source: NyquistSource::Declared,
                high_prf_hz: None,
                low_prf_hz: None,
                dual_prf: false,
            },
            azimuth_deg: vec![0.0],
            azimuth_source: crate::volume::AzimuthSource::NominalFromRayCount { astart_deg: 0.0 },
            ray_elevation_deg: None,
            moments: Vec::new(),
        };
        assert_eq!(gate_centre_m(&sweep, 0), 125.0);
        assert_eq!(gate_centre_m(&sweep, 1), 375.0);
    }

    #[test]
    fn the_granularity_token_says_sweep_because_odim_has_no_per_ray_nyquist() {
        assert_eq!(NYQUIST_GRANULARITY, "sweep");
    }
}
