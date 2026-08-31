//! PUTTING EVERY REQUESTED SPACING ON THE LADDER THE GENERATOR CAN BUILD.
//!
//! # The breakage this prevents, measured
//!
//! The hierarchical arm's only refinement operator is MIDPOINT INSERTION: a
//! Delaunay edge is split in half, so a level changes the spacing by a factor
//! of exactly two. A refined core therefore only ever sits at
//! `background / 2^k`. When a spec asks for anything else, the last rung of
//! [`crate::mesh::hierarchy::ladder`] is partial -- `previous / requested` in
//! `(1, 2]` -- and the mesh cannot deliver the request. It does not say so.
//!
//! MEASURED 2026-08-29 on this generator, `evidence/finemesh-unlock-20260829/`,
//! two specs that PASSED every gate and wrote a grid file:
//!
//! | request | background | ladder ends | delivered in the fine plateau | miss |
//! |---|---|---|---|---|
//! | 51.2 km | 480 km | 480,240,120,60,51.2 | **59.49 km** | +16.2 % |
//! | 93.75 km | 480 km | 480,240,120,93.75 | **120.02 km** | +28.0 % |
//!
//! Both delivered `background / 2^k` and neither refused: the level's delivery
//! gate is a MEDIAN over every cell, and a graded mesh's transition annulus
//! carries more cells than its core, so a core-only miss cannot move it (the
//! all-cells medians read 1.0070 and 1.0204 against a 1.0212 bound). A grid
//! file that says 51.2 km and is 59.5 km is exactly the silent resolution lie
//! the level gate's own refusal text warns about -- "the spacing on paper would
//! not be the spacing in the window it was bought for" -- reached by the one
//! route that gate cannot see.
//!
//! # What this does instead
//!
//! Every region's spacing is moved onto the background's power-of-two ladder
//! BEFORE anything is seeded, and the move is recorded. Two rules:
//!
//! * **Finer, never coarser.** `delivered = background / 2^ceil(log2(ratio))`,
//!   which is at most the request and at worst half of it. A caller always
//!   gets at least the resolution asked for; what they may get is more of it,
//!   at up to 4x the cells in the refined region, and the count is reported by
//!   the sizing line before a run is spent.
//! * **The ramp keeps its WIDTH.** `transition_cells` is counted in cells of
//!   the region's own spacing, so snapping finer would silently steepen the
//!   ramp -- 130 cells is 6.8 %/cell at 4 km and 11.9 %/cell at 2.34 km, which
//!   is the difference between a buildable spec and a surgery-locality
//!   refusal. A snapped region's ramp is converted to the equivalent
//!   `transition_km` at the REQUESTED spacing, so the physical width the
//!   caller described is what gets built.
//!
//! The background is never moved: it is the one spacing the caller can always
//! have exactly, and coarsening it would deliver less than asked for in the
//! plateau. The receipt names the background that WOULD have hit each request
//! exactly, the same way `region_attainment` prints `widest_transition_km`.
//!
//! # Why a snap rather than a refusal
//!
//! Because a refusal here would require the caller to know that the generator
//! refines by halving -- which is the generator's business, not the request's.
//! A uniform request already snaps this way: an icosahedral seed can only
//! deliver `10*(m^2+mn+n^2)+2` cells, so the count is moved to the nearest
//! achievable one and [`crate::mesh::Seeding`] records the move rather than
//! letting the delivered count quietly disagree with the request. This is the
//! same rule applied to the graded arm's own quantisation.

use serde::Serialize;

use crate::mesh::density::{MeshSpec, TransitionField};

/// How close to an integer a `log2(ratio)` has to be to count as already on
/// the ladder. Generous enough to absorb `fitted_to`'s uniform rescale, which
/// multiplies background and region spacing by the same factor and leaves the
/// ratio within an ulp; tight enough that no real request is misread (the
/// nearest thing to a power of two the campaign ever asked for was
/// `75 / 4.0 = 18.75`, which is `log2 = 4.229`).
const ON_LADDER_LOG2_EPS: f64 = 1.0e-9;

/// What one region's request had to move to become buildable.
#[derive(Debug, Clone, Serialize)]
pub struct RegionSnap {
    /// Index into `MeshSpec::regions`.
    pub region: usize,
    pub requested_spacing_km: f64,
    pub delivered_spacing_km: f64,
    /// `log2(background / delivered)`: how many halvings reach it.
    pub ladder_levels: usize,
    /// `delivered / requested - 1`. Zero when the request was already on the
    /// ladder; otherwise negative, and never below -0.5.
    pub snap_relative: f64,
    /// The cell-count cost of the move in the refined region: the delivered
    /// spacing packs `(requested/delivered)^2` times as many cells.
    pub cell_cost_factor: f64,
    /// The background spacing that would have met this region's request
    /// exactly, in km. The printed remedy for a caller who wants the number
    /// they asked for rather than a finer one.
    pub exact_background_km: f64,
    /// Set when a `transition_cells` ramp was converted to the equivalent
    /// `transition_km` at the REQUESTED spacing, so the snap does not steepen
    /// it. `None` when the spec already wrote `transition_km`, or when the
    /// region did not move.
    pub transition_held_km: Option<f64>,
}

/// What the whole spec had to move.
#[derive(Debug, Clone, Serialize)]
pub struct LadderSnap {
    pub background_km: f64,
    pub regions: Vec<RegionSnap>,
    /// True when at least one region moved.
    pub moved: bool,
    /// A single background that would have met EVERY region's request exactly,
    /// when one exists. `None` when the regions disagree, in which case each
    /// region's own `exact_background_km` is the answer for that region alone.
    pub exact_background_km: Option<f64>,
}

impl LadderSnap {
    /// One line per moved region, for the progress stream.
    pub fn progress_lines(&self) -> Vec<String> {
        self.regions
            .iter()
            .filter(|r| r.snap_relative != 0.0)
            .map(|r| {
                format!(
                    "LADDERSNAP\t{}\t{:.4}\t{:.4}\t{}\t{:+.2}%\t{:.2}x\t{:.4}",
                    r.region,
                    r.requested_spacing_km,
                    r.delivered_spacing_km,
                    r.ladder_levels,
                    r.snap_relative * 100.0,
                    r.cell_cost_factor,
                    r.exact_background_km
                )
            })
            .collect()
    }
}

/// Is this ratio already an exact power of two?
fn levels_for(ratio: f64) -> Option<usize> {
    if !(ratio.is_finite() && ratio > 1.0) {
        return None;
    }
    let x = ratio.log2();
    let l = if (x - x.round()).abs() <= ON_LADDER_LOG2_EPS {
        x.round()
    } else {
        x.ceil()
    };
    if l < 0.0 || l > 60.0 {
        return None;
    }
    Some(l as usize)
}

/// True when every refinement region already sits on the background's
/// power-of-two ladder.
pub fn is_on_ladder(spec: &MeshSpec) -> bool {
    first_off_ladder(spec).is_none()
}

/// The first region whose spacing the ladder cannot reach, with the levels it
/// would take and the spacing it would actually deliver.
pub fn first_off_ladder(spec: &MeshSpec) -> Option<(usize, f64, usize, f64)> {
    for (i, r) in spec.regions.iter().enumerate() {
        let ratio = spec.background_km / r.spacing_km;
        let Some(l) = levels_for(ratio) else {
            continue; // not a refinement at all; the field ignores it
        };
        let delivered = spec.background_km * 0.5f64.powi(l as i32);
        if (delivered / r.spacing_km - 1.0).abs() > 1.0e-9 {
            return Some((i, r.spacing_km, l, delivered));
        }
    }
    None
}

/// Move every region onto the ladder, finer never coarser, holding each ramp's
/// physical width. Returns the buildable spec and the record of what moved.
pub fn snap_to_ladder(spec: &MeshSpec) -> (MeshSpec, LadderSnap) {
    let mut out = spec.clone();
    let mut records = Vec::with_capacity(spec.regions.len());
    let mut moved = false;
    for (i, r) in spec.regions.iter().enumerate() {
        let requested = r.spacing_km;
        let ratio = spec.background_km / requested;
        let Some(l) = levels_for(ratio) else {
            // A region no finer than the background refines nothing: the
            // resolution field takes the finer of the two everywhere, so this
            // region contributes no ladder level and is left exactly as
            // written rather than being quietly pulled to the background.
            records.push(RegionSnap {
                region: i,
                requested_spacing_km: requested,
                delivered_spacing_km: requested,
                ladder_levels: 0,
                snap_relative: 0.0,
                cell_cost_factor: 1.0,
                exact_background_km: spec.background_km,
                transition_held_km: None,
            });
            continue;
        };
        let delivered = spec.background_km * 0.5f64.powi(l as i32);
        let on_ladder = (delivered / requested - 1.0).abs() <= 1.0e-9;
        let mut transition_held = None;
        if !on_ladder {
            moved = true;
            out.regions[i].spacing_km = delivered;
            // Hold the ramp's WIDTH: `transition_cells` is counted in cells of
            // the region's own spacing, so leaving it alone would steepen the
            // ramp by exactly the snap factor.
            if let TransitionField::Cells(n) = r.transition {
                let km = n * requested;
                out.regions[i].transition = TransitionField::Km(km);
                transition_held = Some(km);
            }
        }
        records.push(RegionSnap {
            region: i,
            requested_spacing_km: requested,
            delivered_spacing_km: if on_ladder { requested } else { delivered },
            ladder_levels: l,
            snap_relative: if on_ladder {
                0.0
            } else {
                delivered / requested - 1.0
            },
            cell_cost_factor: if on_ladder {
                1.0
            } else {
                (requested / delivered).powi(2)
            },
            exact_background_km: requested * 2f64.powi(l as i32),
            transition_held_km: transition_held,
        });
    }
    let exact = records.first().map(|r| r.exact_background_km).filter(|&b| {
        records
            .iter()
            .all(|r| (r.exact_background_km - b).abs() <= 1e-9 * b.max(1.0))
    });
    let snap = LadderSnap {
        background_km: spec.background_km,
        regions: records,
        moved,
        exact_background_km: exact,
    };
    (out, snap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::density::{Region, Shape};

    fn spec(bg: f64, spacings: &[f64], transition: TransitionField) -> MeshSpec {
        MeshSpec {
            background_km: bg,
            name: None,
            regions: spacings
                .iter()
                .map(|&s| Region {
                    shape: Shape::Cap {
                        center_deg: [39.0, -98.0],
                        radius_km: 800.0,
                    },
                    spacing_km: s,
                    transition,
                })
                .collect(),
        }
    }

    /// A power-of-two request is untouched, to the bit, and reports no move.
    #[test]
    fn a_request_already_on_the_ladder_does_not_move() {
        let s = spec(480.0, &[60.0, 120.0], TransitionField::Km(3000.0));
        assert!(is_on_ladder(&s));
        let (out, snap) = snap_to_ladder(&s);
        assert!(!snap.moved);
        assert_eq!(out.regions[0].spacing_km, 60.0);
        assert_eq!(out.regions[1].spacing_km, 120.0);
        for r in &snap.regions {
            assert_eq!(r.snap_relative, 0.0);
            assert_eq!(r.cell_cost_factor, 1.0);
        }
        assert_eq!(snap.regions[0].ladder_levels, 3);
        assert_eq!(snap.regions[1].ladder_levels, 2);
    }

    /// The two requests that were MEASURED to deliver a coarser mesh than they
    /// claimed now move onto the rung the generator actually builds.
    #[test]
    fn the_two_measured_silent_misses_snap_to_what_the_ladder_delivers() {
        // 480 -> 51.2 delivered 59.49 km (levels 4 reaches 30 km).
        let (out, snap) = snap_to_ladder(&spec(480.0, &[51.2], TransitionField::Km(3000.0)));
        assert_eq!(snap.regions[0].ladder_levels, 4);
        assert_eq!(out.regions[0].spacing_km, 30.0);
        assert!(snap.moved);
        assert!(snap.regions[0].snap_relative < 0.0);
        // 480 -> 93.75 delivered 120.02 km (levels 3 reaches 60 km).
        let (out, snap) = snap_to_ladder(&spec(480.0, &[93.75], TransitionField::Km(3000.0)));
        assert_eq!(snap.regions[0].ladder_levels, 3);
        assert_eq!(out.regions[0].spacing_km, 60.0);
    }

    /// FINER, NEVER COARSER, and never worse than half.
    #[test]
    fn a_snap_is_always_finer_and_never_worse_than_a_factor_of_two() {
        for bg in [40.0f64, 75.0, 120.0, 480.0] {
            for k in 1..400 {
                let want = bg / (1.0 + k as f64 * 0.37);
                let (out, snap) = snap_to_ladder(&spec(bg, &[want], TransitionField::Km(1.0)));
                let got = out.regions[0].spacing_km;
                assert!(got <= want * (1.0 + 1e-12), "bg {bg} want {want} got {got}");
                assert!(got > want * 0.5 - 1e-12, "bg {bg} want {want} got {got}");
                assert!(is_on_ladder(&out), "bg {bg} want {want} got {got}");
                assert!(snap.regions[0].cell_cost_factor <= 4.0 + 1e-9);
            }
        }
    }

    /// The ramp keeps its PHYSICAL width. A `transition_cells` ramp left alone
    /// through a snap would steepen by exactly the snap factor, which is what
    /// turned a buildable 2.34 km spec into a surgery-locality refusal.
    #[test]
    fn a_cell_counted_ramp_holds_its_width_through_a_snap() {
        let s = spec(75.0, &[2.9], TransitionField::Cells(235.0));
        let (out, snap) = snap_to_ladder(&s);
        assert_eq!(snap.regions[0].ladder_levels, 5);
        assert!((out.regions[0].spacing_km - 75.0 / 32.0).abs() < 1e-12);
        // 235 cells at the REQUESTED 2.9 km is 681.5 km; left as Cells(235) at
        // the delivered 2.34375 km it would have become 550.8 km.
        match out.regions[0].transition {
            TransitionField::Km(km) => {
                assert!((km - 235.0 * 2.9).abs() < 1e-9, "held {km} km");
            }
            other => panic!("ramp was not held: {other:?}"),
        }
        assert_eq!(snap.regions[0].transition_held_km, Some(235.0 * 2.9));
    }

    /// A `transition_km` ramp is already a physical width and is not touched.
    #[test]
    fn a_km_ramp_passes_through_a_snap_unchanged() {
        let s = spec(75.0, &[2.9], TransitionField::Km(700.0));
        let (out, snap) = snap_to_ladder(&s);
        match out.regions[0].transition {
            TransitionField::Km(km) => assert_eq!(km, 700.0),
            other => panic!("{other:?}"),
        }
        assert_eq!(snap.regions[0].transition_held_km, None);
    }

    /// The remedy is printed: the background that would have met the request
    /// exactly. Unanimous across regions or not stated at all.
    #[test]
    fn the_exact_background_remedy_is_reported() {
        let (_, snap) = snap_to_ladder(&spec(75.0, &[2.9], TransitionField::Km(700.0)));
        assert!((snap.regions[0].exact_background_km - 2.9 * 32.0).abs() < 1e-9);
        assert!((snap.exact_background_km.unwrap() - 92.8).abs() < 1e-9);
        // Two regions wanting different backgrounds: no single answer.
        let (_, snap) = snap_to_ladder(&spec(75.0, &[2.9, 11.0], TransitionField::Km(700.0)));
        assert!(snap.exact_background_km.is_none());
    }

    /// A region no finer than the background refines nothing and is left
    /// exactly as written.
    #[test]
    fn a_region_coarser_than_the_background_is_untouched() {
        let s = spec(75.0, &[75.0, 200.0], TransitionField::Km(700.0));
        let (out, snap) = snap_to_ladder(&s);
        assert!(!snap.moved);
        assert_eq!(out.regions[0].spacing_km, 75.0);
        assert_eq!(out.regions[1].spacing_km, 200.0);
    }

    /// The snap SURVIVES `fitted_to`, which rescales every spacing by one
    /// factor: a uniform scale leaves every ratio alone, so a spec that is on
    /// the ladder before fitting is on the ladder after it. This is why the
    /// snap runs before the fit rather than after.
    #[test]
    fn a_uniform_rescale_keeps_a_snapped_spec_on_the_ladder() {
        let (snapped, _) = snap_to_ladder(&spec(75.0, &[2.9, 20.0], TransitionField::Km(700.0)));
        assert!(is_on_ladder(&snapped));
        for k in [0.31_f64, 0.7, 1.0, 1.9, 4.3, 11.7] {
            assert!(is_on_ladder(&snapped.scaled(k)), "scale {k}");
        }
    }
}
