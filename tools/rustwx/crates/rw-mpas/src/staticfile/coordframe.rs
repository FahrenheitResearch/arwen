//! WHICH COORDINATE REPRESENTATION A MESH FILE STORES, SAID IN THE FILE.
//!
//! # The quantity this is about
//!
//! Every consumer that recomputes anything from a mesh's stored Cartesian
//! coordinates -- an arc length, an edge normal, a tangent plane, a
//! reconstruction coefficient -- inherits the COORDINATE QUANTUM: the spacing
//! between representable values at the magnitude the components carry. It is
//! an ABSOLUTE error and it does not shrink with the mesh, so it is the whole
//! of what makes a fine mesh harder to store than a coarse one.
//!
//! MEASURED (2026-08-29, `evidence/dvedge-floor-20260829/`), on the published
//! `x1.40962.static.nc` and `x4.163842.static.nc` and on two graded meshes
//! this generator made, over a 16x range of quantum and cell spacings from
//! 1 km to 120 km. Three laws, `q` the quantum, `h` the cell spacing, `dv` the
//! dual-edge length:
//!
//! * stored-point orthogonality, `abs(cos(primal,dual)) = b * q / dv`,
//!   b median 0.203, p99 0.963, WORST 1.935;
//! * dual-edge length error, `a * q`, ABSOLUTE, a median 0.250, worst 1.556;
//! * TRiSK coefficient `R = w*dc/dv`, `c * q / h`, c median 0.196, worst 0.704.
//!
//! At binary32 and `sphere_radius = 6 371 229 m` the quantum is 0.5 m exactly,
//! which is what every published static carries and what pins the dycore
//! byte-identity anchor. At binary64 it is 9.31e-10 m.
//!
//! # Why a mesh may choose, and why only some may
//!
//! Coordinates are binary32 because native MPAS-A stores them that way, and
//! byte-identity against native MPAS-A v8.4.1 is the project's correctness
//! anchor. That anchor is a property of files that HAVE a native counterpart.
//! A mesh this generator produced has none -- native MPAS-A cannot produce it
//! -- so there is nothing for it to be byte-identical TO, and it is free to
//! store its points at the precision it actually knows them to.
//!
//! The line is drawn by provenance, not by resolution: a file with a native
//! counterpart (a published mesh, or a regional cull of one) keeps
//! [`CoordinateRepresentation::Binary32EarthCentred`] and keeps the anchor; a
//! file this crate GENERATED may declare
//! [`CoordinateRepresentation::Binary64EarthCentred`]. The default everywhere
//! is binary32, so a reader that finds no declaration reads exactly the bytes
//! that ship today.
//!
//! # THE BREAKAGE THE DECLARATION PREVENTS, and the one it does not
//!
//! It does NOT prevent a wrong storage tolerance. Every consumer derives that
//! from the dtype it finds -- the MPAS port's `spherical_arc_tolerance` reads
//! the dtype of `xCell`/`yCell`/`zCell` and returns `2*sqrt(3)` of its
//! spacing, 1.73 m at binary32 and 3.2e-9 m at binary64 -- so a reader is
//! already right about the quantum with no attribute at all, and every
//! published file is read correctly today. A declaration claiming to fix that
//! would be naming a breakage that does not occur.
//!
//! What it prevents is a SILENT DEMOTION. Every check downstream loosens with
//! the dtype it finds, so a fine mesh rewritten from binary64 to binary32 --
//! a rescale, a conversion through a tool that only writes floats, a
//! round-trip that promotes on read and demotes on write -- loses 5.4e8 of its
//! coordinate precision and PASSES every one of them, because they all move
//! with it. On a 115 m dual edge that is the difference between 1.4e-11 and
//! 8.4e-3 of orthogonality defect in the point set the dycore's operators are
//! built from, and nothing else in the pipeline says a word. A file that
//! carries what its producer INTENDED is the only thing that can notice.
//!
//! And an ORIGIN a reader does not honour is worse: the mesh is then placed
//! somewhere else on the Earth entirely, and terrain lookup, lateral-boundary
//! interpolation and every rendered product land at the wrong point with no
//! arithmetic anywhere disagreeing. That is why the origin is written
//! EXPLICITLY beside the representation, as three doubles a reader ADDS to the
//! stored components, even though it is (0,0,0) in both representations that
//! exist today.
//!
//! # WHAT ABSENCE MEANS, and why it is not binary32
//!
//! A file with no declaration is judged by its DTYPE, exactly as every
//! consumer already judges it. Absence is not a claim of binary32 and must
//! never be read as one: the published `x1.40962.grid.nc` and
//! `x4.163842.grid.nc` store binary64 coordinates and carry no attribute, so a
//! rule that read silence as binary32 would refuse the two meshes the whole
//! project is anchored to. The declaration binds only when it is present.

use rw_store::netcdf_classic::{NcAttr, NcAttrValue, NcType};

use crate::error::{MpasError, MpasResult};

/// Attribute naming the representation. Absent means
/// [`CoordinateRepresentation::Binary32EarthCentred`] -- what every published
/// file carries and what this crate wrote before the choice existed.
pub const REPRESENTATION_ATTR: &str = "rw_coordinate_representation";

/// Attribute carrying the origin the stored components are measured FROM, in
/// metres, as three doubles. A reader reconstructs an absolute position as
/// `stored + origin`.
pub const ORIGIN_ATTR: &str = "rw_coordinate_origin_xyz";

/// Attribute carrying the coordinate quantum in metres, so a reader takes the
/// number rather than modelling it.
pub const QUANTUM_ATTR: &str = "rw_coordinate_quantum_m";

/// Attribute carrying the sentence a human needs.
pub const NOTE_ATTR: &str = "rw_coordinate_note";

/// The worst-edge constant of the stored-point orthogonality law
/// `abs(cos(primal,dual)) = b * q / dvEdge`.
///
/// MEASURED 2026-08-29 off the published binary32 statics themselves --
/// `x4.163842.static.nc` over 491,520 edges and `x1.40962.static.nc` over
/// 122,880 -- binned by dual-edge length across six bins from 1 km to 131 km:
/// `b` medians 0.177-0.292, p99s 0.441-0.997, worst 1.935. The worst is the
/// one a floor is written against. Receipt:
/// `evidence/dvedge-floor-20260829/RECEIPT.md`.
pub const ORTHOGONALITY_WORST_CONSTANT: f64 = 1.935;

/// How a mesh file stores its Cartesian coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CoordinateRepresentation {
    /// binary32 components of the ABSOLUTE position, origin at the Earth's
    /// centre. What native MPAS-A writes, what both published statics carry,
    /// and what the dycore byte-identity anchor is a property of. Quantum
    /// 0.5 m at `sphere_radius = 6 371 229 m`.
    Binary32EarthCentred,
    /// binary64 components of the ABSOLUTE position, origin at the Earth's
    /// centre. Quantum 9.31e-10 m at the same radius: 537 million times finer,
    /// which is what takes the storage quantum out of the way of a
    /// sub-kilometre mesh. Legal only on a file with NO native counterpart.
    Binary64EarthCentred,
}

impl Default for CoordinateRepresentation {
    /// binary32 Earth-centred: what a file with no declaration is, what every
    /// published file is, and what preserves the byte-identity anchor.
    fn default() -> Self {
        CoordinateRepresentation::Binary32EarthCentred
    }
}

impl CoordinateRepresentation {
    /// The string written into [`REPRESENTATION_ATTR`].
    pub fn tag(self) -> &'static str {
        match self {
            CoordinateRepresentation::Binary32EarthCentred => "binary32_earth_centred",
            CoordinateRepresentation::Binary64EarthCentred => "binary64_earth_centred",
        }
    }

    /// Parse a declaration. `None` for a tag this build does not know, which
    /// callers turn into a refusal rather than a default -- guessing at an
    /// unknown representation is the one thing this attribute exists to stop.
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag.trim() {
            "binary32_earth_centred" => Some(CoordinateRepresentation::Binary32EarthCentred),
            "binary64_earth_centred" => Some(CoordinateRepresentation::Binary64EarthCentred),
            _ => None,
        }
    }

    /// The netCDF type the coordinate and lat/lon arrays are written in.
    ///
    /// LAT/LON MOVE WITH XYZ, and they have to. The port cross-checks the pair
    /// (`mesh.py`, `expected_xyz`) and tightens `metric_rtol` from 2.0e-5 to
    /// 5.0e-10 the moment it sees binary64 metrics; a binary64 `xCell` beside a
    /// binary32 `latCell` carries ~0.38 m of latitude quantisation into a
    /// millimetre-scale comparison and fails the load.
    pub fn nc_type(self) -> NcType {
        match self {
            CoordinateRepresentation::Binary32EarthCentred => NcType::Float,
            CoordinateRepresentation::Binary64EarthCentred => NcType::Double,
        }
    }

    /// True when coordinates go out as binary64.
    pub fn is_binary64(self) -> bool {
        matches!(self, CoordinateRepresentation::Binary64EarthCentred)
    }

    /// The coordinate quantum in metres at this sphere radius: the spacing
    /// between representable values at the largest magnitude a component
    /// carries. Computed from the representation, never tabulated, so a
    /// different radius gives the right answer without an edit.
    pub fn quantum_m(self, sphere_radius_m: f64) -> f64 {
        let r = sphere_radius_m.abs();
        match self {
            CoordinateRepresentation::Binary32EarthCentred => {
                let r32 = r as f32;
                (f32::from_bits(r32.to_bits() + 1) - r32) as f64
            }
            CoordinateRepresentation::Binary64EarthCentred => f64::from_bits(r.to_bits() + 1) - r,
        }
    }

    /// The representation a mesh THIS CRATE GENERATED is stored at.
    ///
    /// One place, because two consumers read it and they must not drift: the
    /// grid file's `rw_static_coordinate_representation` attribute, and the
    /// dual-edge floor in [`crate::mesh::validate::Limits::for_storage`],
    /// which is a multiple of this representation's quantum. A grid that
    /// declares one representation while its generator was gated against
    /// another would either refuse meshes it could store or emit meshes its
    /// static cannot represent.
    ///
    /// It is binary64 because a generated mesh has no native MPAS-A
    /// counterpart and therefore no byte-identity anchor on its storage
    /// precision. Nothing that HAS a counterpart reaches this function.
    pub fn for_generated_mesh() -> Self {
        CoordinateRepresentation::Binary64EarthCentred
    }

    /// The origin the stored components are measured from, in metres. Both
    /// representations that exist are Earth-centred, so this is (0,0,0); it is
    /// written anyway (see the module note) so a frame is never inferred.
    pub fn origin_xyz(self) -> [f64; 3] {
        [0.0, 0.0, 0.0]
    }

    /// The four attributes a file carries to declare itself.
    pub fn attributes(self, sphere_radius_m: f64) -> Vec<NcAttr> {
        let o = self.origin_xyz();
        vec![
            NcAttr::text(REPRESENTATION_ATTR, self.tag()),
            NcAttr::doubles(ORIGIN_ATTR, vec![o[0], o[1], o[2]]),
            NcAttr::doubles(QUANTUM_ATTR, vec![self.quantum_m(sphere_radius_m)]),
            NcAttr::text(
                NOTE_ATTR,
                "xCell/yCell/zCell and the lat/lon pair beside them are stored in \
                 rw_coordinate_representation. An absolute position is stored + \
                 rw_coordinate_origin_xyz (metres). rw_coordinate_quantum_m is the \
                 spacing between representable coordinate values at sphere_radius, and \
                 an absolute storage tolerance on any length recomputed from these \
                 points is 2*sqrt(3) of it. A reader that finds no declaration must \
                 read binary32_earth_centred, which is what every published file is.",
            ),
        ]
    }
}

impl CoordinateRepresentation {
    /// The attributes a file actually WRITES, which is nothing at all for the
    /// default.
    ///
    /// WHY THE DEFAULT WRITES NOTHING, and why that is not ambiguity. Every
    /// published mesh, and every static this crate has ever written, carries
    /// no declaration; the mesh registry pins a grid and its static together
    /// by byte count and SHA-256, so stamping four attributes onto a binary32
    /// static would change the bytes of every already-registered mesh --
    /// MEASURED: 78,329,032 bytes becoming 78,329,696, with every one of the
    /// 82 variables identical -- for a file whose representation was never in
    /// doubt.
    ///
    /// Absence is SAFE because an undeclared file is read by its DTYPE, which
    /// is what every consumer already does and is already correct. What the
    /// declaration adds is a record of what the PRODUCER stored, so a file
    /// rewritten at a different width is caught; a file this crate wrote at
    /// the default was stored at the width it is read at, and there is nothing
    /// for the record to catch.
    pub fn declaration_attributes(self, sphere_radius_m: f64) -> Vec<NcAttr> {
        if self == CoordinateRepresentation::default() {
            Vec::new()
        } else {
            self.attributes(sphere_radius_m)
        }
    }
}

/// What a file's own attributes say, and whether they agree with its arrays.
///
/// `declared` is the value of [`REPRESENTATION_ATTR`], absent meaning the
/// binary32 default. `stored_is_binary64` is read from the arrays themselves.
/// `origin` is [`ORIGIN_ATTR`], absent meaning the Earth's centre.
///
/// Refuses, by name, on each of the three ways a file can be ambiguous. See
/// the module note for what each one breaks.
pub fn verify_declaration(
    file_label: &str,
    declared: Option<&str>,
    stored_is_binary64: bool,
    origin: Option<[f64; 3]>,
    sphere_radius_m: f64,
) -> MpasResult<CoordinateRepresentation> {
    // An undeclared file is read by its DTYPE, which is what every consumer
    // already does and what makes the published binary64 grid files readable.
    // Absence is never a claim; see the module note.
    let by_dtype = if stored_is_binary64 {
        CoordinateRepresentation::Binary64EarthCentred
    } else {
        CoordinateRepresentation::Binary32EarthCentred
    };
    let rep = match declared {
        None => return Ok(by_dtype),
        Some(tag) => CoordinateRepresentation::from_tag(tag).ok_or_else(|| {
            MpasError::Refusal(format!(
                "{file_label}: {REPRESENTATION_ATTR} = {tag:?}, which this build does not know. \
                 The attribute records what the file's PRODUCER stored, so that a file rewritten \
                 at a different width is caught rather than read at the width it now happens to \
                 have; a tag this build cannot interpret carries no such statement and would be \
                 ignored in silence. Known tags: {:?}, {:?}",
                CoordinateRepresentation::Binary32EarthCentred.tag(),
                CoordinateRepresentation::Binary64EarthCentred.tag(),
            ))
        })?,
    };
    if rep.is_binary64() != stored_is_binary64 {
        let (says, holds) = if rep.is_binary64() {
            ("binary64", "binary32")
        } else {
            ("binary32", "binary64")
        };
        return Err(MpasError::Refusal(format!(
            "{file_label}: {REPRESENTATION_ATTR} says {says} but xCell/yCell/zCell are stored as \
             {holds}, so this file was rewritten after its producer wrote it. {} Every storage \
             check downstream derives its tolerance from the dtype it FINDS -- 2*sqrt(3) \
             coordinate ULP, 1.73 m at binary32 and 3.2e-9 m at binary64 -- so all of them move \
             with the rewrite and none of them would say a word. Rebuild the static from its \
             grid rather than editing either the attribute or the arrays",
            if rep.is_binary64() {
                format!(
                    "A demotion loses {:.2e} of the coordinate precision this mesh was generated \
                     and gated at, and the point set the dycore builds its edge normals, tangent \
                     planes and reconstruction coefficients from stops being orthogonal to that \
                     accuracy.",
                    CoordinateRepresentation::Binary32EarthCentred.quantum_m(sphere_radius_m)
                        / CoordinateRepresentation::Binary64EarthCentred
                            .quantum_m(sphere_radius_m)
                )
            } else {
                "A promotion changes no value but proves the arrays are not the bytes the \
                 producer wrote, so nothing else in the file is the bytes it wrote either."
                    .to_string()
            }
        )));
    }
    if let Some(o) = origin {
        if o != [0.0, 0.0, 0.0] {
            return Err(MpasError::Refusal(format!(
                "{file_label}: {ORIGIN_ATTR} = [{:.6e}, {:.6e}, {:.6e}] m, not the Earth's \
                 centre. Every representation this build knows is Earth-centred, so it would \
                 read these components as absolute positions and place the whole mesh {:.1} km \
                 from where the file puts it -- terrain lookup, lateral-boundary interpolation \
                 and every rendered product would land at the wrong point with no arithmetic \
                 anywhere disagreeing. A locally-framed file needs a build that adds the origin \
                 back",
                o[0],
                o[1],
                o[2],
                (o[0] * o[0] + o[1] * o[1] + o[2] * o[2]).sqrt() / 1000.0
            )));
        }
    }
    Ok(rep)
}

/// What representation a static built from this GRID file must use.
///
/// The rule is provenance, not dtype and not resolution: a grid this generator
/// wrote carries [`crate::mesh::emit::STATIC_COORDINATES_ATTR`] naming its
/// choice; every other grid -- a published NCAR mesh, a native regional cull,
/// anything with a counterpart the dycore is pinned against -- carries none
/// and gets binary32. Reading the rule from the file rather than from a flag
/// is what stops a static being built at a precision its grid never asked for.
pub fn for_static_from_grid(grid_path: &std::path::Path) -> MpasResult<CoordinateRepresentation> {
    let f = netcrust::File::open(grid_path).map_err(|e| {
        MpasError::Refusal(format!("cannot open grid {}: {e}", grid_path.display()))
    })?;
    let declared = f
        .attribute(crate::mesh::emit::STATIC_COORDINATES_ATTR)
        .and_then(|a| a.as_string().map(|s| s.to_string()));
    match declared {
        None => Ok(CoordinateRepresentation::default()),
        Some(tag) => CoordinateRepresentation::from_tag(&tag).ok_or_else(|| {
            MpasError::Refusal(format!(
                "{}: {} = {tag:?}, which this build does not know. A static built on a guess                  here would be judged downstream against a storage tolerance derived from the                  wrong coordinate quantum, in one of two silent directions: far too tight                  refuses a defect-free mesh for its own storage precision, far too loose admits                  a dvEdge corrupted by a metre. Known tags: {:?}, {:?}",
                grid_path.display(),
                crate::mesh::emit::STATIC_COORDINATES_ATTR,
                CoordinateRepresentation::Binary32EarthCentred.tag(),
                CoordinateRepresentation::Binary64EarthCentred.tag(),
            ))
        }),
    }
}

/// Pull the declaration out of a list of global attributes.
pub fn read_declaration(
    file_label: &str,
    attrs: &[NcAttr],
    stored_is_binary64: bool,
    sphere_radius_m: f64,
) -> MpasResult<CoordinateRepresentation> {
    let text = |name: &str| -> Option<String> {
        attrs
            .iter()
            .find(|a| a.name == name)
            .and_then(|a| match &a.value {
                NcAttrValue::Text(t) => Some(t.clone()),
                _ => None,
            })
    };
    let triple = |name: &str| -> Option<[f64; 3]> {
        attrs
            .iter()
            .find(|a| a.name == name)
            .and_then(|a| match &a.value {
                NcAttrValue::Doubles(v) if v.len() == 3 => Some([v[0], v[1], v[2]]),
                _ => None,
            })
    };
    verify_declaration(
        file_label,
        text(REPRESENTATION_ATTR).as_deref(),
        stored_is_binary64,
        triple(ORIGIN_ATTR),
        sphere_radius_m,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::geom::EARTH_RADIUS_M;

    #[test]
    fn quantum_matches_the_measured_readings() {
        let b32 = CoordinateRepresentation::Binary32EarthCentred.quantum_m(EARTH_RADIUS_M);
        let b64 = CoordinateRepresentation::Binary64EarthCentred.quantum_m(EARTH_RADIUS_M);
        // 0.5 m EXACTLY at 6,371,229 m -- the reading the whole floor rests on.
        assert_eq!(b32, 0.5, "binary32 quantum at earth radius");
        assert!(
            (b64 - 9.313225746154785e-10).abs() < 1e-24,
            "binary64 quantum at earth radius: {b64:e}"
        );
        // The port's absolute arc floor, both ways.
        assert!(((2.0 * 3f64.sqrt() * b32) - 1.7320508075688772).abs() < 1e-12);
    }

    /// AN UNDECLARED FILE IS READ BY ITS DTYPE, in both widths. Reading
    /// silence as binary32 would refuse the published `x1.40962.grid.nc` and
    /// `x4.163842.grid.nc`, which store binary64 coordinates and carry no
    /// attribute -- the two meshes the project is anchored to.
    #[test]
    fn an_undeclared_file_is_read_by_its_dtype_in_both_widths() {
        let f32_file = verify_declaration("f", None, false, None, EARTH_RADIUS_M).unwrap();
        assert_eq!(f32_file, CoordinateRepresentation::Binary32EarthCentred);
        assert_eq!(f32_file.nc_type(), NcType::Float);
        let f64_file = verify_declaration("published grid", None, true, None, EARTH_RADIUS_M)
            .expect("a published binary64 grid carries no declaration and must still read");
        assert_eq!(f64_file, CoordinateRepresentation::Binary64EarthCentred);
    }

    #[test]
    fn a_tag_this_build_does_not_know_is_refused_not_defaulted() {
        let err =
            verify_declaration("f", Some("binary32_local_origin"), false, None, EARTH_RADIUS_M)
                .unwrap_err()
                .to_string();
        assert!(err.contains("does not know"), "{err}");
        assert!(err.contains("binary64_earth_centred"), "{err}");
    }

    /// A DECLARATION THAT DISAGREES WITH THE ARRAYS is a rewritten file, in
    /// both directions, and the demotion direction says what was lost.
    #[test]
    fn declaration_and_dtype_must_agree_in_both_directions() {
        let promoted =
            verify_declaration("f", Some("binary32_earth_centred"), true, None, EARTH_RADIUS_M)
                .unwrap_err()
                .to_string();
        assert!(promoted.contains("rewritten after its producer wrote it"), "{promoted}");
        assert!(promoted.contains("A promotion changes no value"), "{promoted}");
        let demoted =
            verify_declaration("f", Some("binary64_earth_centred"), false, None, EARTH_RADIUS_M)
                .unwrap_err()
                .to_string();
        assert!(demoted.contains("A demotion loses"), "{demoted}");
        assert!(demoted.contains("5.37e8"), "the loss is quantified: {demoted}");
    }

    #[test]
    fn a_non_earth_centred_origin_is_refused_by_name() {
        let err = verify_declaration(
            "f",
            Some("binary64_earth_centred"),
            true,
            Some([1.0e5, 0.0, 0.0]),
            EARTH_RADIUS_M,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains(ORIGIN_ATTR), "{err}");
        assert!(err.contains("100.0 km"), "{err}");
    }

    /// The default writes NOTHING, so every already-registered static keeps
    /// its byte count and its SHA-256 -- MEASURED byte-identical on statics
    /// built from `x1.40962.grid.nc` and `x4.163842.grid.nc`. Anything else
    /// writes the full set.
    #[test]
    fn only_a_non_default_representation_stamps_attributes() {
        assert!(
            CoordinateRepresentation::Binary32EarthCentred
                .declaration_attributes(EARTH_RADIUS_M)
                .is_empty(),
            "a binary32 static must be byte-unchanged"
        );
        let stamped =
            CoordinateRepresentation::Binary64EarthCentred.declaration_attributes(EARTH_RADIUS_M);
        assert_eq!(stamped.len(), 4);
        assert!(stamped.iter().any(|a| a.name == REPRESENTATION_ATTR));
        assert!(stamped.iter().any(|a| a.name == ORIGIN_ATTR));
        assert!(stamped.iter().any(|a| a.name == QUANTUM_ATTR));
    }

    #[test]
    fn attributes_round_trip_through_read_declaration() {
        for rep in [
            CoordinateRepresentation::Binary32EarthCentred,
            CoordinateRepresentation::Binary64EarthCentred,
        ] {
            let attrs = rep.attributes(EARTH_RADIUS_M);
            let back = read_declaration("f", &attrs, rep.is_binary64(), EARTH_RADIUS_M).unwrap();
            assert_eq!(back, rep);
        }
    }
}
