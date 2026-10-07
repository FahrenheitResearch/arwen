//
// Group C of the WOOF post-processor: parcels, shear, helicity and severe
// indices (SPEC section 6).  Two paths compute the same fields:
//
// - `compute_cpu`: the f64 reference, one scalar function per column
//   (parcel.rs, wind.rs, thermo.rs), rayon over columns;
// - `gpu::SevereGpu`: CUDA kernels (kernels/severe_c.cu) that carry the same
//   column algorithms statement for statement, also in f64, one thread per
//   column (and per parcel type for the parcel kernel).  No atomics, no
//   cross-thread reductions and fixed iteration counts, so a launch is
//   deterministic.
//
// Inputs are Group A's shared column state of SPEC 3.4 ([`crate::state`]),
// viewed through [`ColumnState::from_shared`]: level-major
// `[k][cell]` f32 planes, mass levels bottom to top, interfaces ground to
// top.  A non-finite input anywhere a field needs makes that cell missing
// (NaN) in that field only.

pub mod parcel;
pub mod thermo;
pub mod wind;

#[cfg(any(windows, target_os = "linux"))]
pub mod gpu;

use rayon::prelude::*;

/// Output planes, in order.  The first 21 are the catalogue fields; the
/// last is the surface parcel's LCL height (an STP input, kept for review).
pub const FIELDS: [&str; 22] = [
    "sbcape",
    "sbcin",
    "mlcape",
    "mlcin",
    "mucape",
    "mucin",
    "cape_best180",
    "cin_best180",
    "lcl_height",
    "storm_motion_u",
    "storm_motion_v",
    "srh_0_1km",
    "srh_0_3km",
    "shear_u_0_1km",
    "shear_v_0_1km",
    "shear_u_0_6km",
    "shear_v_0_6km",
    "bulk_shear_0_1km",
    "bulk_shear_0_6km",
    "stp",
    "ehi_0_1km",
    "sb_lcl_height",
];
pub const NPLANES: usize = FIELDS.len();

/// Plane index of a field name.
pub fn plane_index(name: &str) -> Option<usize> {
    FIELDS.iter().position(|f| *f == name)
}

/// Shared column state slice view (SPEC 3.4) for Group C.
#[derive(Clone, Copy, Debug)]
pub struct ColumnState<'a> {
    pub nz: usize,
    pub ncell: usize,
    /// Mass-level pressure coordinate (Pa), `[nz][ncell]`.
    pub p: &'a [f32],
    /// Mass-level temperature (K).
    pub tk: &'a [f32],
    /// Mass-level water vapour mixing ratio (kg/kg).
    pub r: &'a [f32],
    /// Mass-level height (m MSL).
    pub z_mass: &'a [f32],
    /// Destaggered mass-level wind (m/s).
    pub u: &'a [f32],
    pub v: &'a [f32],
    /// Interface pressure and height, `[nz+1][ncell]`, ground first.
    pub p_int: &'a [f32],
    pub z_int: &'a [f32],
    /// 2D: terrain height (m), surface pressure (Pa), 2 m temperature (K),
    /// 2 m mixing ratio (kg/kg), shelter pressure (Pa), 10 m wind (m/s).
    pub zsfc: &'a [f32],
    pub psfc: &'a [f32],
    pub t2: &'a [f32],
    pub q2: &'a [f32],
    pub p2: &'a [f32],
    pub u10: &'a [f32],
    pub v10: &'a [f32],
}

#[derive(Debug)]
pub struct ShapeError(pub String);

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ShapeError {}

impl<'a> ColumnState<'a> {
    /// Group C's view of Group A's shared state (pressure coordinate p,
    /// temperature, mixing ratio, mass height, winds, interface pressure and
    /// height) plus the 2D carriers: terrain, PSFC, 2 m temperature, 2 m
    /// mixing ratio, shelter pressure and the 10 m wind.
    #[allow(clippy::too_many_arguments)]
    pub fn from_shared(
        st: &'a crate::state::ColumnState,
        zsfc: &'a [f32],
        psfc: &'a [f32],
        t2: &'a [f32],
        q2: &'a [f32],
        p2: &'a [f32],
        u10: &'a [f32],
        v10: &'a [f32],
    ) -> Self {
        use crate::state::{InterfaceState as I, MassState as M};
        Self {
            nz: st.shape.nz,
            ncell: st.shape.ncell(),
            p: st.mass_slot(M::P),
            tk: st.mass_slot(M::Tk),
            r: st.mass_slot(M::R),
            z_mass: st.mass_slot(M::ZMass),
            u: st.mass_slot(M::U),
            v: st.mass_slot(M::V),
            p_int: st.interface_slot(I::PInt),
            z_int: st.interface_slot(I::ZInt),
            zsfc,
            psfc,
            t2,
            q2,
            p2,
            u10,
            v10,
        }
    }

    pub fn validate(&self) -> Result<(), ShapeError> {
        let (nz, n) = (self.nz, self.ncell);
        let check = |name: &str, len: usize, want: usize| {
            if len == want {
                Ok(())
            } else {
                Err(ShapeError(format!("{name}: {len} values, expected {want}")))
            }
        };
        if nz < 2 || n == 0 {
            return Err(ShapeError(format!("degenerate shape nz={nz} ncell={n}")));
        }
        for (name, s) in [("p", self.p), ("tk", self.tk), ("r", self.r), ("z_mass", self.z_mass), ("u", self.u), ("v", self.v)] {
            check(name, s.len(), nz * n)?;
        }
        for (name, s) in [("p_int", self.p_int), ("z_int", self.z_int)] {
            check(name, s.len(), (nz + 1) * n)?;
        }
        for (name, s) in [
            ("zsfc", self.zsfc),
            ("psfc", self.psfc),
            ("t2", self.t2),
            ("q2", self.q2),
            ("p2", self.p2),
            ("u10", self.u10),
            ("v10", self.v10),
        ] {
            check(name, s.len(), n)?;
        }
        Ok(())
    }

    fn thermo_column(&self, c: usize, col: &mut parcel::Column) -> bool {
        let (nz, n) = (self.nz, self.ncell);
        col.p.clear();
        col.tk.clear();
        col.r.clear();
        col.p_int.clear();
        col.z_int.clear();
        let mut ok = true;
        for k in 0..nz {
            let (p, t, r) = (self.p[k * n + c], self.tk[k * n + c], self.r[k * n + c]);
            ok &= p.is_finite() && t.is_finite() && r.is_finite();
            col.p.push(p as f64);
            col.tk.push(t as f64);
            col.r.push(r as f64);
        }
        for k in 0..=nz {
            let (p, z) = (self.p_int[k * n + c], self.z_int[k * n + c]);
            ok &= p.is_finite() && z.is_finite();
            col.p_int.push(p as f64);
            col.z_int.push(z as f64);
        }
        col.zsfc = self.zsfc[c] as f64;
        col.psfc = self.psfc[c] as f64;
        col.t2 = self.t2[c] as f64;
        col.q2 = self.q2[c] as f64;
        col.p2 = self.p2[c] as f64;
        ok && [col.zsfc, col.psfc, col.t2, col.q2, col.p2].iter().all(|v| v.is_finite())
    }

    fn wind_column(&self, c: usize, col: &mut wind::WindColumn) -> bool {
        let (nz, n) = (self.nz, self.ncell);
        col.z_agl.clear();
        col.u.clear();
        col.v.clear();
        let zsfc = self.zsfc[c] as f64;
        let mut ok = zsfc.is_finite();
        for k in 0..nz {
            let (z, u, v) = (self.z_mass[k * n + c], self.u[k * n + c], self.v[k * n + c]);
            ok &= z.is_finite() && u.is_finite() && v.is_finite();
            col.z_agl.push(z as f64 - zsfc);
            col.u.push(u as f64);
            col.v.push(v as f64);
        }
        col.u10 = self.u10[c] as f64;
        col.v10 = self.v10[c] as f64;
        ok && col.u10.is_finite() && col.v10.is_finite()
    }
}

/// Output planes, `[NPLANES][ncell]` f32, NaN for missing.
#[derive(Clone, Debug)]
pub struct SevereFields {
    pub ncell: usize,
    pub planes: Vec<f32>,
}

impl SevereFields {
    pub fn plane(&self, name: &str) -> Option<&[f32]> {
        let i = plane_index(name)?;
        Some(&self.planes[i * self.ncell..(i + 1) * self.ncell])
    }
}

/// All Group C values of one column, in `FIELDS` order (f64).
pub fn column_values(state: &ColumnState<'_>, c: usize, tcol: &mut parcel::Column, wcol: &mut wind::WindColumn) -> [f64; NPLANES] {
    let mut out = [f64::NAN; NPLANES];
    if state.thermo_column(c, tcol) {
        let p = parcel::parcels(tcol);
        out[..9].copy_from_slice(&p[..9]);
        out[21] = p[9];
    }
    if state.wind_column(c, wcol) {
        let w = wind::winds(wcol);
        out[9..19].copy_from_slice(&w);
    }
    let (sbcape, sbcin, srh01) = (out[0], out[1], out[11]);
    out[19] = wind::stp(sbcape, sbcin, out[21], srh01, out[18]);
    out[20] = if sbcape.is_finite() && srh01.is_finite() { wind::ehi(sbcape, srh01) } else { f64::NAN };
    out
}

/// CPU reference over a whole frame.
pub fn compute_cpu(state: &ColumnState<'_>) -> Result<SevereFields, ShapeError> {
    state.validate()?;
    let n = state.ncell;
    let per_cell: Vec<[f32; NPLANES]> = (0..n)
        .into_par_iter()
        .map_init(
            || (parcel::Column::default(), wind::WindColumn::default()),
            |(tcol, wcol), c| {
                let v = column_values(state, c, tcol, wcol);
                let mut o = [0f32; NPLANES];
                for i in 0..NPLANES {
                    o[i] = v[i] as f32;
                }
                o
            },
        )
        .collect();
    let mut planes = vec![0f32; NPLANES * n];
    for (c, vals) in per_cell.iter().enumerate() {
        for i in 0..NPLANES {
            planes[i * n + c] = vals[i];
        }
    }
    Ok(SevereFields { ncell: n, planes })
}

/// Compute Group C on the CPU, or on an opened kernel set when `gpu` is
/// given (falling back to the CPU when the frame does not fit and the
/// device is not mandatory).  Returns the fields and the device label.
pub fn compute(
    state: &ColumnState<'_>,
    #[cfg(any(windows, target_os = "linux"))] gpu: Option<&crate::gpu::GpuPost>,
    device: crate::PostDevice,
) -> Result<(SevereFields, String), Box<dyn std::error::Error>> {
    #[cfg(any(windows, target_os = "linux"))]
    if device != crate::PostDevice::Cpu {
        if let Some(g) = gpu {
            let sg = gpu::SevereGpu::new(g);
            let fits = sg.memory().map(|(free, _)| free > gpu::SevereGpu::bytes_needed(state) + (256 << 20)).unwrap_or(false);
            if fits || device == crate::PostDevice::Gpu {
                let (f, _) = sg.compute(state)?;
                return Ok((f, g.device.describe()));
            }
        } else if device == crate::PostDevice::Gpu {
            return Err("PostDevice::Gpu needs an opened GPU".into());
        }
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    if device == crate::PostDevice::Gpu {
        return Err("no CUDA support on this platform".into());
    }
    Ok((compute_cpu(state)?, "cpu".to_string()))
}
