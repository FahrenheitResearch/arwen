use rustwx_core::GridShape;

use crate::error::CalcError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VolumeShape {
    pub grid: GridShape,
    pub nz: usize,
}

impl VolumeShape {
    pub fn new(grid: GridShape, nz: usize) -> Result<Self, CalcError> {
        if nz == 0 {
            return Err(CalcError::LengthMismatch {
                field: "nz",
                expected: 1,
                actual: 0,
            });
        }
        Ok(Self { grid, nz })
    }

    pub fn len2d(self) -> usize {
        self.grid.len()
    }

    pub fn len3d(self) -> usize {
        self.grid.len() * self.nz
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EcapeGridInputs<'a> {
    pub shape: VolumeShape,
    pub pressure_3d_pa: &'a [f64],
    pub temperature_3d_c: &'a [f64],
    pub qvapor_3d_kgkg: &'a [f64],
    pub height_agl_3d_m: &'a [f64],
    pub u_3d_ms: &'a [f64],
    pub v_3d_ms: &'a [f64],
    pub psfc_pa: &'a [f64],
    pub t2_k: &'a [f64],
    pub q2_kgkg: &'a [f64],
    pub u10_ms: &'a [f64],
    pub v10_ms: &'a [f64],
}

#[derive(Debug, Clone, Copy)]
pub struct EcapeVolumeInputs<'a> {
    pub pressure_pa: &'a [f64],
    pub temperature_c: &'a [f64],
    pub qvapor_kgkg: &'a [f64],
    pub height_agl_m: &'a [f64],
    pub u_ms: &'a [f64],
    pub v_ms: &'a [f64],
    pub nz: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct SurfaceInputs<'a> {
    pub psfc_pa: &'a [f64],
    pub t2_k: &'a [f64],
    pub q2_kgkg: &'a [f64],
    pub u10_ms: &'a [f64],
    pub v10_ms: &'a [f64],
}

#[derive(Debug, Clone, Copy)]
pub struct EcapeOptions<'a> {
    pub parcel_type: &'a str,
    pub storm_motion_type: &'a str,
    pub entrainment_rate: Option<f64>,
    pub pseudoadiabatic: Option<bool>,
    pub storm_motion: Option<(f64, f64)>,
}

impl Default for EcapeOptions<'_> {
    fn default() -> Self {
        Self::new("sb", "right_moving")
    }
}

impl<'a> EcapeOptions<'a> {
    pub fn new(parcel_type: &'a str, storm_motion_type: &'a str) -> Self {
        Self {
            parcel_type,
            storm_motion_type,
            entrainment_rate: None,
            pseudoadiabatic: None,
            storm_motion: None,
        }
    }

    pub fn with_entrainment_rate(mut self, entrainment_rate: f64) -> Self {
        self.entrainment_rate = Some(entrainment_rate);
        self
    }

    pub fn with_pseudoadiabatic(mut self, pseudoadiabatic: bool) -> Self {
        self.pseudoadiabatic = Some(pseudoadiabatic);
        self
    }

    pub fn with_user_storm_motion(mut self, storm_u_ms: f64, storm_v_ms: f64) -> Self {
        self.storm_motion = Some((storm_u_ms, storm_v_ms));
        self
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EcapeTripletOptions<'a> {
    pub storm_motion_type: &'a str,
    pub entrainment_rate: Option<f64>,
    pub pseudoadiabatic: Option<bool>,
    pub storm_motion: Option<(f64, f64)>,
}

impl Default for EcapeTripletOptions<'_> {
    fn default() -> Self {
        Self::new("right_moving")
    }
}

impl<'a> EcapeTripletOptions<'a> {
    pub fn new(storm_motion_type: &'a str) -> Self {
        Self {
            storm_motion_type,
            entrainment_rate: None,
            pseudoadiabatic: None,
            storm_motion: None,
        }
    }

    pub fn with_entrainment_rate(mut self, entrainment_rate: f64) -> Self {
        self.entrainment_rate = Some(entrainment_rate);
        self
    }

    pub fn with_pseudoadiabatic(mut self, pseudoadiabatic: bool) -> Self {
        self.pseudoadiabatic = Some(pseudoadiabatic);
        self
    }

    pub fn with_user_storm_motion(mut self, storm_u_ms: f64, storm_v_ms: f64) -> Self {
        self.storm_motion = Some((storm_u_ms, storm_v_ms));
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EcapeFields {
    pub ecape_jkg: Vec<f64>,
    pub ncape_jkg: Vec<f64>,
    pub cape_jkg: Vec<f64>,
    pub cin_jkg: Vec<f64>,
    pub lfc_m: Vec<f64>,
    pub el_m: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EcapeFieldsWithFailureMask {
    pub fields: EcapeFields,
    pub failure_mask: Vec<u8>,
}

impl EcapeFieldsWithFailureMask {
    pub fn failure_count(&self) -> usize {
        self.failure_mask.iter().filter(|&&flag| flag != 0).count()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EcapeTripletFieldsWithFailureMask {
    pub sb: EcapeFieldsWithFailureMask,
    pub ml: EcapeFieldsWithFailureMask,
    pub mu: EcapeFieldsWithFailureMask,
}

impl EcapeTripletFieldsWithFailureMask {
    pub fn total_failure_count(&self) -> usize {
        self.sb.failure_count() + self.ml.failure_count() + self.mu.failure_count()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EcapeTripletFields {
    pub sb: EcapeFields,
    pub ml: EcapeFields,
    pub mu: EcapeFields,
}

pub fn compute_ecape(
    inputs: EcapeGridInputs<'_>,
    options: &EcapeOptions<'_>,
) -> Result<EcapeFields, CalcError> {
    let (volume, surface) = split_inputs(inputs);
    compute_ecape_from_parts(inputs.shape.grid, volume, surface, *options)
}

pub fn compute_ecape_with_failure_mask(
    inputs: EcapeGridInputs<'_>,
    options: &EcapeOptions<'_>,
) -> Result<EcapeFieldsWithFailureMask, CalcError> {
    let (volume, surface) = split_inputs(inputs);
    compute_ecape_with_failure_mask_from_parts(inputs.shape.grid, volume, surface, *options)
}

pub fn compute_ecape_triplet_with_failure_mask(
    inputs: EcapeGridInputs<'_>,
    options: &EcapeTripletOptions<'_>,
) -> Result<EcapeTripletFieldsWithFailureMask, CalcError> {
    let (volume, surface) = split_inputs(inputs);
    compute_ecape_triplet_with_failure_mask_from_parts(inputs.shape.grid, volume, surface, *options)
}

pub fn compute_ecape_triplet(
    inputs: EcapeGridInputs<'_>,
    options: &EcapeTripletOptions<'_>,
) -> Result<EcapeTripletFields, CalcError> {
    let (volume, surface) = split_inputs(inputs);
    compute_ecape_triplet_from_parts(inputs.shape.grid, volume, surface, *options)
}

pub fn compute_ecape_from_parts(
    grid: GridShape,
    volume: EcapeVolumeInputs<'_>,
    surface: SurfaceInputs<'_>,
    options: EcapeOptions<'_>,
) -> Result<EcapeFields, CalcError> {
    validate_inputs(grid, volume, surface)?;

    let (storm_u, storm_v) = unzip_storm_motion(options);
    let (ecape, ncape, cape, cin, lfc, el) = metrust::calc::severe::grid::compute_ecape(
        volume.pressure_pa,
        volume.temperature_c,
        volume.qvapor_kgkg,
        volume.height_agl_m,
        volume.u_ms,
        volume.v_ms,
        surface.psfc_pa,
        surface.t2_k,
        surface.q2_kgkg,
        surface.u10_ms,
        surface.v10_ms,
        grid.nx,
        grid.ny,
        volume.nz,
        options.parcel_type,
        options.storm_motion_type,
        options.entrainment_rate,
        options.pseudoadiabatic,
        storm_u,
        storm_v,
    )
    .map_err(CalcError::Metrust)?;

    Ok(EcapeFields {
        ecape_jkg: ecape,
        ncape_jkg: ncape,
        cape_jkg: cape,
        cin_jkg: cin,
        lfc_m: lfc,
        el_m: el,
    })
}

pub fn compute_ecape_with_failure_mask_from_parts(
    grid: GridShape,
    volume: EcapeVolumeInputs<'_>,
    surface: SurfaceInputs<'_>,
    options: EcapeOptions<'_>,
) -> Result<EcapeFieldsWithFailureMask, CalcError> {
    validate_inputs(grid, volume, surface)?;

    let (storm_u, storm_v) = unzip_storm_motion(options);
    let (ecape, ncape, cape, cin, lfc, el, failure_mask) =
        metrust::calc::severe::grid::compute_ecape_with_failure_mask(
            volume.pressure_pa,
            volume.temperature_c,
            volume.qvapor_kgkg,
            volume.height_agl_m,
            volume.u_ms,
            volume.v_ms,
            surface.psfc_pa,
            surface.t2_k,
            surface.q2_kgkg,
            surface.u10_ms,
            surface.v10_ms,
            grid.nx,
            grid.ny,
            volume.nz,
            options.parcel_type,
            options.storm_motion_type,
            options.entrainment_rate,
            options.pseudoadiabatic,
            storm_u,
            storm_v,
        )
        .map_err(CalcError::Metrust)?;

    Ok(EcapeFieldsWithFailureMask {
        fields: EcapeFields {
            ecape_jkg: ecape,
            ncape_jkg: ncape,
            cape_jkg: cape,
            cin_jkg: cin,
            lfc_m: lfc,
            el_m: el,
        },
        failure_mask,
    })
}

pub fn compute_ecape_triplet_with_failure_mask_from_parts(
    grid: GridShape,
    volume: EcapeVolumeInputs<'_>,
    surface: SurfaceInputs<'_>,
    options: EcapeTripletOptions<'_>,
) -> Result<EcapeTripletFieldsWithFailureMask, CalcError> {
    validate_triplet_inputs(grid, volume, surface)?;

    let (storm_u, storm_v) = unzip_triplet_storm_motion(options);
    let triplet = if pressure_is_levels(volume) {
        metrust::calc::severe::grid::compute_ecape_triplet_with_failure_mask_levels(
            volume.pressure_pa,
            volume.temperature_c,
            volume.qvapor_kgkg,
            volume.height_agl_m,
            volume.u_ms,
            volume.v_ms,
            surface.psfc_pa,
            surface.t2_k,
            surface.q2_kgkg,
            surface.u10_ms,
            surface.v10_ms,
            grid.nx,
            grid.ny,
            volume.nz,
            options.storm_motion_type,
            options.entrainment_rate,
            options.pseudoadiabatic,
            storm_u,
            storm_v,
        )
    } else {
        metrust::calc::severe::grid::compute_ecape_triplet_with_failure_mask(
            volume.pressure_pa,
            volume.temperature_c,
            volume.qvapor_kgkg,
            volume.height_agl_m,
            volume.u_ms,
            volume.v_ms,
            surface.psfc_pa,
            surface.t2_k,
            surface.q2_kgkg,
            surface.u10_ms,
            surface.v10_ms,
            grid.nx,
            grid.ny,
            volume.nz,
            options.storm_motion_type,
            options.entrainment_rate,
            options.pseudoadiabatic,
            storm_u,
            storm_v,
        )
    }
    .map_err(CalcError::Metrust)?;

    Ok(EcapeTripletFieldsWithFailureMask {
        sb: EcapeFieldsWithFailureMask {
            fields: EcapeFields {
                ecape_jkg: triplet.sb.fields.ecape,
                ncape_jkg: triplet.sb.fields.ncape,
                cape_jkg: triplet.sb.fields.cape,
                cin_jkg: triplet.sb.fields.cin,
                lfc_m: triplet.sb.fields.lfc,
                el_m: triplet.sb.fields.el,
            },
            failure_mask: triplet.sb.failure_mask,
        },
        ml: EcapeFieldsWithFailureMask {
            fields: EcapeFields {
                ecape_jkg: triplet.ml.fields.ecape,
                ncape_jkg: triplet.ml.fields.ncape,
                cape_jkg: triplet.ml.fields.cape,
                cin_jkg: triplet.ml.fields.cin,
                lfc_m: triplet.ml.fields.lfc,
                el_m: triplet.ml.fields.el,
            },
            failure_mask: triplet.ml.failure_mask,
        },
        mu: EcapeFieldsWithFailureMask {
            fields: EcapeFields {
                ecape_jkg: triplet.mu.fields.ecape,
                ncape_jkg: triplet.mu.fields.ncape,
                cape_jkg: triplet.mu.fields.cape,
                cin_jkg: triplet.mu.fields.cin,
                lfc_m: triplet.mu.fields.lfc,
                el_m: triplet.mu.fields.el,
            },
            failure_mask: triplet.mu.failure_mask,
        },
    })
}

pub fn compute_analytic_ecape_triplet_with_failure_mask_from_parts(
    grid: GridShape,
    volume: EcapeVolumeInputs<'_>,
    surface: SurfaceInputs<'_>,
    options: EcapeTripletOptions<'_>,
) -> Result<EcapeTripletFieldsWithFailureMask, CalcError> {
    validate_triplet_inputs(grid, volume, surface)?;

    let (storm_u, storm_v) = unzip_triplet_storm_motion(options);
    let triplet = if pressure_is_levels(volume) {
        metrust::calc::severe::grid::compute_analytic_ecape_triplet_with_failure_mask_levels(
            volume.pressure_pa,
            volume.temperature_c,
            volume.qvapor_kgkg,
            volume.height_agl_m,
            volume.u_ms,
            volume.v_ms,
            surface.psfc_pa,
            surface.t2_k,
            surface.q2_kgkg,
            surface.u10_ms,
            surface.v10_ms,
            grid.nx,
            grid.ny,
            volume.nz,
            options.storm_motion_type,
            options.pseudoadiabatic,
            storm_u,
            storm_v,
        )
    } else {
        metrust::calc::severe::grid::compute_analytic_ecape_triplet_with_failure_mask(
            volume.pressure_pa,
            volume.temperature_c,
            volume.qvapor_kgkg,
            volume.height_agl_m,
            volume.u_ms,
            volume.v_ms,
            surface.psfc_pa,
            surface.t2_k,
            surface.q2_kgkg,
            surface.u10_ms,
            surface.v10_ms,
            grid.nx,
            grid.ny,
            volume.nz,
            options.storm_motion_type,
            options.pseudoadiabatic,
            storm_u,
            storm_v,
        )
    }
    .map_err(CalcError::Metrust)?;

    Ok(EcapeTripletFieldsWithFailureMask {
        sb: EcapeFieldsWithFailureMask {
            fields: EcapeFields {
                ecape_jkg: triplet.sb.fields.ecape,
                ncape_jkg: triplet.sb.fields.ncape,
                cape_jkg: triplet.sb.fields.cape,
                cin_jkg: triplet.sb.fields.cin,
                lfc_m: triplet.sb.fields.lfc,
                el_m: triplet.sb.fields.el,
            },
            failure_mask: triplet.sb.failure_mask,
        },
        ml: EcapeFieldsWithFailureMask {
            fields: EcapeFields {
                ecape_jkg: triplet.ml.fields.ecape,
                ncape_jkg: triplet.ml.fields.ncape,
                cape_jkg: triplet.ml.fields.cape,
                cin_jkg: triplet.ml.fields.cin,
                lfc_m: triplet.ml.fields.lfc,
                el_m: triplet.ml.fields.el,
            },
            failure_mask: triplet.ml.failure_mask,
        },
        mu: EcapeFieldsWithFailureMask {
            fields: EcapeFields {
                ecape_jkg: triplet.mu.fields.ecape,
                ncape_jkg: triplet.mu.fields.ncape,
                cape_jkg: triplet.mu.fields.cape,
                cin_jkg: triplet.mu.fields.cin,
                lfc_m: triplet.mu.fields.lfc,
                el_m: triplet.mu.fields.el,
            },
            failure_mask: triplet.mu.failure_mask,
        },
    })
}

pub fn compute_ecape_triplet_from_parts(
    grid: GridShape,
    volume: EcapeVolumeInputs<'_>,
    surface: SurfaceInputs<'_>,
    options: EcapeTripletOptions<'_>,
) -> Result<EcapeTripletFields, CalcError> {
    validate_triplet_inputs(grid, volume, surface)?;

    let (storm_u, storm_v) = unzip_triplet_storm_motion(options);
    let triplet = if pressure_is_levels(volume) {
        metrust::calc::severe::grid::compute_ecape_triplet_levels(
            volume.pressure_pa,
            volume.temperature_c,
            volume.qvapor_kgkg,
            volume.height_agl_m,
            volume.u_ms,
            volume.v_ms,
            surface.psfc_pa,
            surface.t2_k,
            surface.q2_kgkg,
            surface.u10_ms,
            surface.v10_ms,
            grid.nx,
            grid.ny,
            volume.nz,
            options.storm_motion_type,
            options.entrainment_rate,
            options.pseudoadiabatic,
            storm_u,
            storm_v,
        )
    } else {
        metrust::calc::severe::grid::compute_ecape_triplet(
            volume.pressure_pa,
            volume.temperature_c,
            volume.qvapor_kgkg,
            volume.height_agl_m,
            volume.u_ms,
            volume.v_ms,
            surface.psfc_pa,
            surface.t2_k,
            surface.q2_kgkg,
            surface.u10_ms,
            surface.v10_ms,
            grid.nx,
            grid.ny,
            volume.nz,
            options.storm_motion_type,
            options.entrainment_rate,
            options.pseudoadiabatic,
            storm_u,
            storm_v,
        )
    }
    .map_err(CalcError::Metrust)?;

    Ok(EcapeTripletFields {
        sb: EcapeFields {
            ecape_jkg: triplet.sb.ecape,
            ncape_jkg: triplet.sb.ncape,
            cape_jkg: triplet.sb.cape,
            cin_jkg: triplet.sb.cin,
            lfc_m: triplet.sb.lfc,
            el_m: triplet.sb.el,
        },
        ml: EcapeFields {
            ecape_jkg: triplet.ml.ecape,
            ncape_jkg: triplet.ml.ncape,
            cape_jkg: triplet.ml.cape,
            cin_jkg: triplet.ml.cin,
            lfc_m: triplet.ml.lfc,
            el_m: triplet.ml.el,
        },
        mu: EcapeFields {
            ecape_jkg: triplet.mu.ecape,
            ncape_jkg: triplet.mu.ncape,
            cape_jkg: triplet.mu.cape,
            cin_jkg: triplet.mu.cin,
            lfc_m: triplet.mu.lfc,
            el_m: triplet.mu.el,
        },
    })
}

pub(crate) fn validate_inputs(
    grid: GridShape,
    volume: EcapeVolumeInputs<'_>,
    surface: SurfaceInputs<'_>,
) -> Result<(), CalcError> {
    let n2d = grid.len();
    let n3d = n2d * volume.nz;

    validate_len("pressure_pa", volume.pressure_pa.len(), n3d)?;
    validate_len("temperature_c", volume.temperature_c.len(), n3d)?;
    validate_len("qvapor_kgkg", volume.qvapor_kgkg.len(), n3d)?;
    validate_len("height_agl_m", volume.height_agl_m.len(), n3d)?;
    validate_len("u_ms", volume.u_ms.len(), n3d)?;
    validate_len("v_ms", volume.v_ms.len(), n3d)?;

    validate_len("psfc_pa", surface.psfc_pa.len(), n2d)?;
    validate_len("t2_k", surface.t2_k.len(), n2d)?;
    validate_len("q2_kgkg", surface.q2_kgkg.len(), n2d)?;
    validate_len("u10_ms", surface.u10_ms.len(), n2d)?;
    validate_len("v10_ms", surface.v10_ms.len(), n2d)?;

    Ok(())
}

pub(crate) fn validate_triplet_inputs(
    grid: GridShape,
    volume: EcapeVolumeInputs<'_>,
    surface: SurfaceInputs<'_>,
) -> Result<(), CalcError> {
    let n2d = grid.len();
    let n3d = n2d * volume.nz;

    if !pressure_is_levels(volume) {
        validate_len("pressure_pa", volume.pressure_pa.len(), n3d)?;
    } else {
        validate_len("pressure_levels_pa", volume.pressure_pa.len(), volume.nz)?;
    }
    validate_len("temperature_c", volume.temperature_c.len(), n3d)?;
    validate_len("qvapor_kgkg", volume.qvapor_kgkg.len(), n3d)?;
    validate_len("height_agl_m", volume.height_agl_m.len(), n3d)?;
    validate_len("u_ms", volume.u_ms.len(), n3d)?;
    validate_len("v_ms", volume.v_ms.len(), n3d)?;

    validate_len("psfc_pa", surface.psfc_pa.len(), n2d)?;
    validate_len("t2_k", surface.t2_k.len(), n2d)?;
    validate_len("q2_kgkg", surface.q2_kgkg.len(), n2d)?;
    validate_len("u10_ms", surface.u10_ms.len(), n2d)?;
    validate_len("v10_ms", surface.v10_ms.len(), n2d)?;

    Ok(())
}

fn split_inputs(inputs: EcapeGridInputs<'_>) -> (EcapeVolumeInputs<'_>, SurfaceInputs<'_>) {
    (
        EcapeVolumeInputs {
            pressure_pa: inputs.pressure_3d_pa,
            temperature_c: inputs.temperature_3d_c,
            qvapor_kgkg: inputs.qvapor_3d_kgkg,
            height_agl_m: inputs.height_agl_3d_m,
            u_ms: inputs.u_3d_ms,
            v_ms: inputs.v_3d_ms,
            nz: inputs.shape.nz,
        },
        SurfaceInputs {
            psfc_pa: inputs.psfc_pa,
            t2_k: inputs.t2_k,
            q2_kgkg: inputs.q2_kgkg,
            u10_ms: inputs.u10_ms,
            v10_ms: inputs.v10_ms,
        },
    )
}

fn unzip_storm_motion(options: EcapeOptions<'_>) -> (Option<f64>, Option<f64>) {
    match options.storm_motion {
        Some((u, v)) => (Some(u), Some(v)),
        None => (None, None),
    }
}

fn unzip_triplet_storm_motion(options: EcapeTripletOptions<'_>) -> (Option<f64>, Option<f64>) {
    match options.storm_motion {
        Some((u, v)) => (Some(u), Some(v)),
        None => (None, None),
    }
}

fn pressure_is_levels(volume: EcapeVolumeInputs<'_>) -> bool {
    volume.pressure_pa.len() == volume.nz
}

pub(crate) fn validate_len(
    field: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), CalcError> {
    if actual == expected {
        Ok(())
    } else {
        Err(CalcError::LengthMismatch {
            field,
            expected,
            actual,
        })
    }
}

/// [`validate_len`] for inputs a kernel validates but never reads: the
/// array may be ABSENT (empty) — the store-ingest derived lane frees the
/// wind volumes after the wind-consuming kernels and still runs the
/// purely thermodynamic kernels over the same `EcapeVolumeInputs` shape —
/// but a non-empty array of the wrong length is still rejected.
pub(crate) fn validate_len_or_absent(
    field: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), CalcError> {
    if actual == 0 {
        Ok(())
    } else {
        validate_len(field, actual, expected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f64, b: f64) {
        assert!(
            (a - b).abs() < 1.0e-6,
            "values differed: left={a}, right={b}"
        );
    }

    #[test]
    fn triplet_from_parts_accepts_pressure_levels_vector() {
        let grid = GridShape::new(2, 1).unwrap();
        let height_m = [
            0.0, 250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 4000.0, 5000.0,
            6000.0, 7500.0, 9000.0, 10500.0, 12000.0, 14000.0, 16000.0,
        ];
        let pressure_pa = [
            100000.0,
            96923.32344763441,
            93941.30628134758,
            91051.03613800342,
            88249.69025845954,
            82902.91181804004,
            77880.07830714049,
            73161.56289466418,
            68728.92787909723,
            60653.06597126334,
            53526.142851899036,
            47236.65527410147,
            39160.5626676799,
            32465.24673583497,
            26914.634872918385,
            22313.016014842982,
            17377.394345044515,
            13533.52832366127,
        ];
        let temperature_k = [
            302.0, 300.2, 298.4, 296.6, 294.8, 291.2, 287.6, 284.0, 280.4, 273.2, 266.0, 258.8,
            248.0, 237.2, 226.4, 215.6, 215.6, 215.6,
        ];
        let dewpoint_k = [
            296.0, 295.625, 295.25, 294.875, 294.3, 290.7, 287.1, 283.5, 279.9, 272.7, 265.5,
            258.3, 247.5, 236.7, 225.9, 215.1, 215.1, 215.1,
        ];
        let u_wind_ms = [
            4.0, 4.625, 5.25, 5.875, 6.5, 7.75, 9.0, 10.25, 11.5, 14.0, 16.5, 19.0, 22.75, 26.5,
            30.25, 34.0, 39.0, 44.0,
        ];
        let v_wind_ms = [
            1.0, 1.375, 1.75, 2.125, 2.5, 3.25, 4.0, 4.75, 5.5, 7.0, 8.5, 10.0, 12.25, 14.5, 16.75,
            19.0, 22.0, 25.0,
        ];
        let qvapor: Vec<f64> = pressure_pa
            .iter()
            .zip(dewpoint_k)
            .map(|(&p, td)| {
                let td_c: f64 = td - 273.15_f64;
                let e_hpa = 6.112_f64 * ((td_c * 17.67_f64) / (td_c + 243.5_f64)).exp();
                0.622 * e_hpa / (p / 100.0 - e_hpa)
            })
            .collect();
        let pressure_levels = &pressure_pa[1..];
        let temperature_c_single: Vec<f64> =
            temperature_k[1..].iter().map(|t| t - 273.15).collect();
        let qvapor_single = &qvapor[1..];
        let height_single = &height_m[1..];
        let u_single = &u_wind_ms[1..];
        let v_single = &v_wind_ms[1..];

        let mut pressure_broadcast = Vec::with_capacity(pressure_levels.len() * grid.len());
        let mut temperature_c_3d = Vec::with_capacity(temperature_c_single.len() * grid.len());
        let mut qvapor_3d = Vec::with_capacity(qvapor_single.len() * grid.len());
        let mut height_agl_3d = Vec::with_capacity(height_single.len() * grid.len());
        let mut u_3d = Vec::with_capacity(u_single.len() * grid.len());
        let mut v_3d = Vec::with_capacity(v_single.len() * grid.len());
        for idx in 0..pressure_levels.len() {
            pressure_broadcast.extend([pressure_levels[idx], pressure_levels[idx]]);
            temperature_c_3d.extend([temperature_c_single[idx], temperature_c_single[idx]]);
            qvapor_3d.extend([qvapor_single[idx], qvapor_single[idx]]);
            height_agl_3d.extend([height_single[idx], height_single[idx]]);
            u_3d.extend([u_single[idx], u_single[idx]]);
            v_3d.extend([v_single[idx], v_single[idx]]);
        }

        let surface_pressure = [pressure_pa[0], pressure_pa[0]];
        let surface_t2 = [temperature_k[0], temperature_k[0]];
        let surface_q2 = [qvapor[0], qvapor[0]];
        let surface_u10 = [u_wind_ms[0], u_wind_ms[0]];
        let surface_v10 = [v_wind_ms[0], v_wind_ms[0]];

        let broadcast = compute_ecape_triplet_with_failure_mask_from_parts(
            grid,
            EcapeVolumeInputs {
                pressure_pa: &pressure_broadcast,
                temperature_c: &temperature_c_3d,
                qvapor_kgkg: &qvapor_3d,
                height_agl_m: &height_agl_3d,
                u_ms: &u_3d,
                v_ms: &v_3d,
                nz: pressure_pa.len() - 1,
            },
            SurfaceInputs {
                psfc_pa: &surface_pressure,
                t2_k: &surface_t2,
                q2_kgkg: &surface_q2,
                u10_ms: &surface_u10,
                v10_ms: &surface_v10,
            },
            EcapeTripletOptions::new("bunkers_rm")
                .with_pseudoadiabatic(true)
                .with_user_storm_motion(12.0, 6.0),
        )
        .unwrap();

        let levels = compute_ecape_triplet_with_failure_mask_from_parts(
            grid,
            EcapeVolumeInputs {
                pressure_pa: pressure_levels,
                temperature_c: &temperature_c_3d,
                qvapor_kgkg: &qvapor_3d,
                height_agl_m: &height_agl_3d,
                u_ms: &u_3d,
                v_ms: &v_3d,
                nz: pressure_pa.len() - 1,
            },
            SurfaceInputs {
                psfc_pa: &surface_pressure,
                t2_k: &surface_t2,
                q2_kgkg: &surface_q2,
                u10_ms: &surface_u10,
                v10_ms: &surface_v10,
            },
            EcapeTripletOptions::new("bunkers_rm")
                .with_pseudoadiabatic(true)
                .with_user_storm_motion(12.0, 6.0),
        )
        .unwrap();

        assert_eq!(
            levels.total_failure_count(),
            broadcast.total_failure_count()
        );
        for (lhs, rhs) in levels
            .sb
            .fields
            .ecape_jkg
            .iter()
            .zip(broadcast.sb.fields.ecape_jkg.iter())
        {
            assert_close(*lhs, *rhs);
        }
    }
}
