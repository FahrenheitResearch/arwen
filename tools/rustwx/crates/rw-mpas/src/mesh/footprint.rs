//! The device footprint of a full-physics MPAS run, PER CARD.
//!
//! `footprint(cells) = fixed_mib + cells * bytes_per_cell`. It is a MEMORY
//! model, not a skill or a speed claim.
//!
//! # Why the fixed term cannot be one constant
//!
//! Most of it is a single allocation: the CUDA per-context LOCAL-MEMORY
//! BACKING STORE. The driver sizes that store for the widest launched kernel
//! frame at FULL RESIDENCY and never returns it while the context lives:
//!
//! ```text
//! store_bytes = (widest_kernel_frame_bytes - 1024) * SMs * maxThreadsPerSM
//! ```
//!
//! Two of the three factors are properties of the PART. A 70 SM card and a
//! 170 SM card therefore pay different fixed terms for identical code, and a
//! fixed term quoted without naming the card says nothing.
//!
//! MEASURED, null launches, one kernel and one fresh process each, sm_120:
//!
//! | kernel             | frame B | 70 SM measured | 170 SM measured |
//! |--------------------|--------:|---------------:|----------------:|
//! | `gf_gfdrv_stage`   |  29,264 |    2,896.0 MiB |     7,034.0 MiB |
//! | `gf_deep_stage`    |  26,880 |    2,650.0 MiB |     6,438.0 MiB |
//! | `gf_shallow_stage` |  18,944 |    1,836.0 MiB |     4,462.0 MiB |
//! | `rlw_rtrn_march`   |   2,048 |      104.0 MiB |       254.0 MiB |
//! | `rlw_cldprmc`      |      64 |        0.0 MiB |               - |
//! | no local frame     |       0 |        0.0 MiB |         0.0 MiB |
//!
//! [`local_store_mib`] reproduces every one of those rows to better than
//! 0.15 % on BOTH parts, and the zero-frame row is the negative control that
//! says the meter is reading the store and not something else.
//!
//! # What is derived, and what is only measured
//!
//! Subtracting the derived store from each card's measured fixed term leaves
//! a RESIDUE -- the context, the module images, the driver's own scratch, and
//! whatever else is resident before the first cell exists:
//!
//! | card        | SMs | derived store | measured fixed | residue     |
//! |-------------|----:|--------------:|---------------:|------------:|
//! | RTX 5090    | 170 |   7,032.4 MiB |    9,798.0 MiB | 2,765.6 MiB |
//! | RTX 5070 Ti |  70 |   2,895.7 MiB |    5,384.0 MiB | 2,488.3 MiB |
//!
//! The derived store accounts for 4,138.0 MiB of the 4,410.0 MiB that
//! separates the two cards' fixed terms. The residue accounts for the rest --
//! and it is NOT the same number on the two parts and does NOT scale with the
//! SM count, so it cannot be predicted for a part nobody has run.
//!
//! That is why [`Card::measured`] is an `Option` and why an unmeasured card
//! is REFUSED here (see [`Card::unmeasured_refusal`]) instead of being handed
//! the derived store plus a borrowed residue.
//!
//! # Every number here belongs to ONE build
//!
//! The MPAS port runs ArWen's physics from a pinned checkout, and each
//! residue above was taken by subtracting the derived store from a measured
//! fixed term ON THAT BUILD. So the frame, the residues and the pin are a
//! single measurement set: see [`MEASURED_AGAINST_ARWEN_COMMIT`],
//! [`FRAME_CUT_COMMIT`] and the staleness argument on
//! [`WIDEST_KERNEL_FRAME_BYTES`]. The tree this file lives in has ALREADY cut
//! that frame; the pin has not moved, which is the only reason these numbers
//! still stand.
//!
//! # Adding a card is a row
//!
//! [`CARDS`] is a table. A new part is a row with its SM count, its
//! `maxThreadsPerSM`, and -- once somebody runs it -- a measured residue,
//! the checkout that residue was measured against, and its anchors. No new
//! code path, ever.

use serde::Serialize;

const MIB: f64 = 1_048_576.0;

/// The ArWen checkout every number in this file was measured against.
///
/// The MPAS port does not carry its own physics: it runs ArWen's, from a
/// checkout it is pinned to (`--arwen-checkout`). Every frame, store and
/// residue here therefore describes THAT build, and naming it is not
/// decoration -- see [`WIDEST_KERNEL_FRAME_BYTES`].
pub const MEASURED_AGAINST_ARWEN_COMMIT: &str = "e594dc5c5";

/// The commit that ENDS the validity of everything above, if the port's pin
/// is ever moved to it or past it.
///
/// `perf(gf): the column arrays leave the frame the driver prices at full
/// residency` moved `gf`'s column arrays out of the per-thread frame into a
/// global workspace. It is the only commit touching `gpuwm/core/kernels/gf.cu`
/// between the pin and the current tree, and it takes the frame from 22,416 B
/// to 88 B (NVRTC 13.0.48) and 72 B (13.3.33) -- both under the 1,024 B
/// default stack, so the store it prices falls to zero.
pub const FRAME_CUT_COMMIT: &str = "35d83fc8c";

/// Per-thread local frame of the widest kernel the full-physics MPAS path
/// LAUNCHES, in bytes: `gf_gfdrv_stage`, compiled float32 at
/// `nVertLevels = 55`, ON [`MEASURED_AGAINST_ARWEN_COMMIT`].
///
/// # This is NOT the frame in the tree you are reading
///
/// Say it plainly, because a bare constant that silently means "some other
/// checkout's build" is itself the defect. In the tree that contains this
/// file, `gf`'s frame is 88 B or 72 B depending on the NVRTC build --
/// [`FRAME_CUT_COMMIT`] cut it, and this crate's own branch descends from
/// that cut. 29,264 B describes the PINNED checkout the port runs against,
/// which predates it.
///
/// Pre-cut, this frame moved with the level count: 22,416 B at
/// `nVertLevels = 40` -- the figure recorded on the ArWen side, whose
/// recordings compile at that count -- rising to 29,264 B at the 55 the port
/// uses. The two numbers are the same source at two level counts and neither
/// is a correction of the other.
///
/// The driver sizes ONE store for the widest kernel actually launched, so the
/// siblings compiled beside it (`gf_deep_stage` at 26,880 B,
/// `gf_shallow_stage` at 18,944 B) do not add to it -- they are covered by it.
///
/// # It cannot move on its own
///
/// This constant and every [`Measured::residue_mib`] below are ONE
/// measurement set taken on ONE build. Each residue was obtained by
/// SUBTRACTING the store this frame derives from a measured whole-process
/// fixed term, so moving either half alone silently re-prices every card:
///
/// - frame updated, residues not: on a post-cut build the widest launched
///   frame is no longer `gf`'s, and the derived 70 SM store falls from
///   2,895.7 MiB to 841.6 MiB. Added to an unchanged 2,488.3 MiB residue that
///   gives a 3,330 MiB fixed term against the pinned build's real 5,384 --
///   2,054 MiB LOW, so the door over-sizes the mesh and the run dies at
///   step 1 on an out-of-memory that blames the model.
/// - pin advanced past the cut, frame not: the store stays priced at
///   2,895.7 MiB when the real one is near zero, the fixed term comes out
///   thousands of MiB HIGH, and the door under-sizes -- the card is wasted,
///   quietly, in the direction nobody notices.
///
/// Both have to move together, which is why the pin is a named constant and
/// why `tests/test_mpas_mesh_door.py` fails the build if the port's pin ever
/// reaches [`FRAME_CUT_COMMIT`] or if a card's residue starts naming a
/// different checkout from this frame.
///
/// # Re-measuring
///
/// On a post-cut build the widest LAUNCHED frame is whatever is widest once
/// `gf` is out of the way; on the ArWen side that was `ysu_column` at 9,232 B
/// until its own cut landed, then `kf_column` at 9,216 B. Read it, do not
/// assume it, and re-take every residue in the same session -- a frame from
/// one build beside a residue from another is the failure above.
pub const WIDEST_KERNEL_FRAME_BYTES: f64 = 29_264.0;

/// The driver backs `frame - 1024 B` per resident thread, not the whole
/// frame: a frame at or under the context's default stack is already covered
/// and reserves nothing.
///
/// 1,024 B is the CUDA default stack limit, the same value ArWen's own
/// preflight prices against (`tests/test_gf_workspace.py::DEFAULT_STACK_BYTES`).
/// Measured here: it is what makes the formula reproduce all eleven
/// null-launch rows above on two different parts, and what makes the 64 B
/// `rlw_cldprmc` row reserve nothing at all.
pub const LOCAL_FRAME_CREDIT_BYTES: f64 = 1_024.0;

/// The CUDA per-context local-memory backing store a part is forced to
/// reserve, in MiB, from its resident-thread capacity.
///
/// This is the DERIVED half of the fixed term and the reason the fixed term
/// is per card.
pub fn local_store_mib(streaming_multiprocessors: u32, max_threads_per_sm: u32) -> f64 {
    (WIDEST_KERNEL_FRAME_BYTES - LOCAL_FRAME_CREDIT_BYTES)
        * streaming_multiprocessors as f64
        * max_threads_per_sm as f64
        / MIB
}

/// One measured `(cells, whole-process MiB)` point.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Anchor {
    pub cells: usize,
    pub process_mib: f64,
    /// How close the model has to come, in MiB. Wider where the reading was
    /// taken off `nvidia-smi` against an idle baseline rather than off a
    /// per-allocation ledger.
    pub tolerance_mib: f64,
}

/// What a run on this part actually measured.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Measured {
    /// Fixed term left over once [`local_store_mib`] is subtracted, MiB.
    ///
    /// Stored rather than the fixed term itself, because the fixed term is
    /// DERIVED: the store half comes out of the card's own SM count and the
    /// residue is the only part a run has to tell us.
    ///
    /// Obtained BY SUBTRACTION against [`WIDEST_KERNEL_FRAME_BYTES`], so it
    /// is only meaningful beside that frame -- see `residue_measured_against`.
    pub residue_mib: f64,
    /// The ArWen checkout `residue_mib` was measured against.
    ///
    /// Present per row rather than assumed, so that a card re-measured on a
    /// newer build cannot sit silently beside a frame from an older one. A
    /// row naming a different checkout from
    /// [`MEASURED_AGAINST_ARWEN_COMMIT`] is a failing test, not a warning.
    pub residue_measured_against: &'static str,
    /// Bytes of device memory per cell.
    pub bytes_per_cell: f64,
    /// Where `bytes_per_cell` comes from when this card did not separate it
    /// itself, and the measurement that says it carries over.
    pub bytes_per_cell_provenance: &'static str,
    pub anchors: &'static [Anchor],
    pub measured_on: &'static str,
    pub provenance: &'static str,
}

/// One part.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Card {
    pub key: &'static str,
    pub display_name: &'static str,
    pub streaming_multiprocessors: u32,
    /// Resident threads per SM. 1,536 on every sm_86 and sm_120 part here.
    pub max_threads_per_sm: u32,
    pub compute_capability: &'static str,
    /// What `nvidia-smi` reports, MiB.
    pub device_total_mib: f64,
    /// What CUDA's own `memGetInfo` reports as TOTAL, MiB, where it has been
    /// read. Always the smaller of the two and always the one to size
    /// against; the difference is not addressable.
    pub device_addressable_mib: Option<f64>,
    pub measured: Option<Measured>,
    pub notes: &'static str,
}

impl Card {
    /// The fixed term, DERIVED: this card's own local-memory backing store
    /// plus the residue a run on this card measured.
    ///
    /// `None` when nobody has run this part -- see
    /// [`Card::unmeasured_refusal`].
    pub fn fixed_mib(&self) -> Option<f64> {
        Some(self.local_store_mib() + self.measured.as_ref()?.residue_mib)
    }

    /// The derived half of the fixed term for this part.
    pub fn local_store_mib(&self) -> f64 {
        local_store_mib(self.streaming_multiprocessors, self.max_threads_per_sm)
    }

    pub fn bytes_per_cell(&self) -> Option<f64> {
        Some(self.measured.as_ref()?.bytes_per_cell)
    }

    /// The memory a mesh of `cells` cells needs on this part, MiB.
    pub fn footprint_mib(&self, cells: usize) -> Result<f64, String> {
        let fixed = self.fixed_mib().ok_or_else(|| self.unmeasured_refusal())?;
        let per_cell = self.bytes_per_cell().expect("measured implies a slope");
        Ok(fixed + cells as f64 * per_cell / MIB)
    }

    /// The budget to size against when the caller names no explicit one:
    /// what CUDA can actually address, never what `nvidia-smi` prints.
    pub fn sizing_budget_mib(&self) -> f64 {
        self.device_addressable_mib.unwrap_or(self.device_total_mib)
    }

    /// Largest whole cell count inside `budget_mib` on this part.
    pub fn cells_that_fit(&self, budget_mib: f64) -> Result<usize, String> {
        let fixed = self.fixed_mib().ok_or_else(|| self.unmeasured_refusal())?;
        let per_cell = self.bytes_per_cell().expect("measured implies a slope");
        let spare = budget_mib - fixed;
        if spare <= 0.0 {
            return Err(format!(
                "{} ({}) pays a {fixed:.0} MiB fixed footprint before the first cell exists, and the budget given is {budget_mib:.0} MiB, so no mesh of ANY size fits.\n  \
                 where it goes: {:.0} MiB is the CUDA per-context local-memory backing store this part's {} SMs x {} resident threads force for the {:.0} B frame of gf_gfdrv_stage, and {:.0} MiB is the measured residue. Both are paid whether the mesh is one cell or a million.\n  \
                 what changes it: a card with more memory, a budget that is not already spent on something else, or a build that cuts that frame.",
                self.key,
                self.display_name,
                self.local_store_mib(),
                self.streaming_multiprocessors,
                self.max_threads_per_sm,
                WIDEST_KERNEL_FRAME_BYTES,
                self.measured.as_ref().expect("measured implies a residue").residue_mib,
            ));
        }
        Ok((spare * MIB / per_cell).floor() as usize)
    }

    /// Why this part cannot be sized, what IS known about it, and the routes
    /// that do work.
    ///
    /// The breakage this prevents is named in full because a refusal that
    /// does not name it is not a refusal: sizing a mesh against a card whose
    /// footprint nobody measured produces a cell count that is confidently
    /// wrong in an unknown direction. Both directions have happened. Too high
    /// and the run dies at step 1 on an out-of-memory that blames the model;
    /// too low and the card is wasted -- the 170 SM part's fixed term on a
    /// 70 SM part under-sizes it by 1.6x.
    pub fn unmeasured_refusal(&self) -> String {
        let measured: Vec<&str> = CARDS
            .iter()
            .filter(|c| c.measured.is_some())
            .map(|c| c.key)
            .collect();
        let residues: Vec<String> = CARDS
            .iter()
            .filter_map(|c| {
                c.measured
                    .as_ref()
                    .map(|m| format!("{:.0} MiB on {}", m.residue_mib, c.key))
            })
            .collect();
        format!(
            "{} ({}) has NO MEASURED device footprint, so no mesh will be sized for it here.\n  \
             what IS known about this part: {} SMs x {} resident threads derive a {:.0} MiB CUDA per-context local-memory backing store from the {:.0} B frame of gf_gfdrv_stage. That is the dominant term and it comes out of the card's own numbers.\n  \
             what is NOT known: the rest of the fixed term. It is measured, never derived, and it does not scale with the SM count -- {} on the parts that HAVE been run, which is why it cannot be carried across.\n  \
             the breakage this refuses: a mesh sized against a card nobody measured comes out at a cell count that is confidently wrong in a direction nobody can name. Too high and the run dies at step 1 with an out-of-memory that blames the model; too low and the card is wasted -- borrowing the 170 SM part's fixed term under-sizes a 70 SM part by about 1.6x.\n  \
             what to do instead:\n    \
             --card {}\n      size against a part that WAS measured\n    \
             --cells N\n      state the count yourself; no card model is consulted\n    \
             measure this part\n      one full-physics run at two mesh sizes separates the two terms; then this becomes a row in CARDS, not a code path\n  \
             {}",
            self.key,
            self.display_name,
            self.streaming_multiprocessors,
            self.max_threads_per_sm,
            self.local_store_mib(),
            WIDEST_KERNEL_FRAME_BYTES,
            if residues.is_empty() {
                "no part has been run at all".to_string()
            } else {
                residues.join(" against ")
            },
            if measured.is_empty() {
                "(none)".to_string()
            } else {
                measured.join(" | --card ")
            },
            self.notes,
        )
    }

    /// One line of the `--list-cards` table.
    pub fn summary_line(&self) -> String {
        match (self.fixed_mib(), self.bytes_per_cell()) {
            (Some(fixed), Some(per_cell)) => format!(
                "  {:<14} {:<32} {:>3} SM  {:>5} MiB  MEASURED   {:>8.0} MiB fixed ({:.0} store + {:.0} residue) + {:.0} B/cell   {} holds {} cells",
                self.key,
                self.display_name,
                self.streaming_multiprocessors,
                self.sizing_budget_mib().round(),
                fixed,
                self.local_store_mib(),
                self.measured.as_ref().expect("fixed implies measured").residue_mib,
                per_cell,
                if self.device_addressable_mib.is_some() { "addressable" } else { "total" },
                self.cells_that_fit(self.sizing_budget_mib())
                    .map(|n| n.to_string())
                    .unwrap_or_else(|_| "no".to_string()),
            ),
            _ => format!(
                "  {:<14} {:<32} {:>3} SM  {:>5} MiB  NOT MEASURED  derived store {:.0} MiB; residue not measured, so refused by name",
                self.key,
                self.display_name,
                self.streaming_multiprocessors,
                self.sizing_budget_mib().round(),
                self.local_store_mib(),
            ),
        }
    }
}

/// The table. One row per part; a new part is a row.
pub static CARDS: &[Card] = &[
    Card {
        key: "rtx-5070-ti",
        display_name: "NVIDIA GeForce RTX 5070 Ti",
        streaming_multiprocessors: 70,
        max_threads_per_sm: 1_536,
        compute_capability: "sm_120",
        device_total_mib: 16_303.0,
        // MEASURED: nvidia-smi says 16,303 MiB and CUDA's own memGetInfo says
        // 15,880.6. The 422 MiB gap is not addressable and sizing against the
        // larger number overruns the card by that much before a mesh exists.
        device_addressable_mib: Some(15_880.6),
        measured: Some(Measured {
            // 5,384.0 MiB measured fixed - 2,895.7 MiB derived store.
            residue_mib: 2_488.3,
            residue_measured_against: MEASURED_AGAINST_ARWEN_COMMIT,
            bytes_per_cell: 86_630.0,
            bytes_per_cell_provenance:
                "MEASURED to be the SAME as rtx-5090's. At the identical step boundary CuPy's \
                 pool.total_bytes() reads 5,404.5 MiB on both parts, to the decimal, so the \
                 per-cell half of the model is card-independent and only the fixed half moves. \
                 That is what lets one footprint anchor on this part fix its fixed term.",
            anchors: &[Anchor {
                cells: 40_962,
                process_mib: 8_768.0,
                tolerance_mib: 20.0,
            }],
            measured_on: "2026-08-21",
            provenance:
                "weather-node-1, sm_120, float32, nVertLevels 55, full physics, 6 steps, rc 0, \
                 status passed. The 8,768.0 MiB peak is the run under memory pressure: VRAM was \
                 held down in a separate process so the run saw 9,656.7 MiB free, and four \
                 separate runs across a 2.6 GiB band of held-down budgets all peaked at exactly \
                 8,768.0 MiB. Unconstrained the same run peaks at 10,514.0 MiB, which is CuPy \
                 pool opportunism and NOT a requirement -- sizing against it costs 1,745.6 MiB \
                 of pessimism. A second reading from the like-for-like subtraction against the \
                 5090 gives 5,388 MiB fixed independently, 4 MiB from this one.",
        }),
        notes:
            "The 120 km global x1.40962 mesh runs on a 10 GiB card on this part, with about a \
             GiB to spare.",
    },
    Card {
        key: "rtx-5090",
        display_name: "NVIDIA GeForce RTX 5090",
        streaming_multiprocessors: 170,
        max_threads_per_sm: 1_536,
        compute_capability: "sm_120",
        device_total_mib: 32_607.0,
        // NOT MEASURED on this part: no memGetInfo total was read, so the
        // nvidia-smi figure is the only one there is and it is optimistic by
        // whatever this part's equivalent of the 422 MiB gap is.
        device_addressable_mib: None,
        measured: Some(Measured {
            // 9,798.0 MiB measured fixed - 7,032.4 MiB derived store.
            residue_mib: 2_765.6,
            residue_measured_against: MEASURED_AGAINST_ARWEN_COMMIT,
            bytes_per_cell: 86_630.0,
            bytes_per_cell_provenance:
                "Separated on this part by subtraction between its own two mesh anchors, 40,962 \
                 and 163,842 cells, rather than assumed.",
            anchors: &[
                Anchor {
                    cells: 38_857,
                    process_mib: 13_165.0,
                    // Read off nvidia-smi (13,679 MiB peak against a 514 MiB
                    // idle baseline) rather than off the per-allocation
                    // ledger the other two come from.
                    tolerance_mib: 200.0,
                },
                Anchor {
                    cells: 40_962,
                    process_mib: 13_182.0,
                    tolerance_mib: 20.0,
                },
                Anchor {
                    cells: 163_842,
                    process_mib: 23_334.0,
                    tolerance_mib: 20.0,
                },
            ],
            measured_on: "2026-08-20",
            provenance:
                "mpas-port docs/device-memory-ledger.md, task #264: float32, nVertLevels 55, \
                 full physics, 6 steps, GFS 2026-08-12 06Z, both ledger runs rc 0. The two mesh \
                 sizes separate the fixed and per-cell terms by subtraction rather than by \
                 assumption. The 38,857-cell point is a real forecast on a mesh this crate \
                 generated.",
        }),
        notes:
            "Whole-process figures measured on an otherwise-idle card. Anything else resident on \
             the device comes out of the same memory, so pass --vram-gib with the free budget \
             when the card is shared.",
    },
    Card {
        key: "rtx-3080",
        display_name: "NVIDIA GeForce RTX 3080 (10 GiB)",
        streaming_multiprocessors: 68,
        max_threads_per_sm: 1_536,
        compute_capability: "sm_86",
        device_total_mib: 10_240.0,
        device_addressable_mib: None,
        measured: None,
        notes:
            "NOT MEASURED. A different architecture as well as a different SM count: sm_86 has \
             half the sm_120 register file per SM and a different module image, so the residue \
             is even less carryable here than between two sm_120 parts.",
    },
];

/// Look a part up by key, or refuse listing the ones that exist.
pub fn card(key: &str) -> Result<&'static Card, String> {
    CARDS.iter().find(|c| c.key == key).ok_or_else(|| {
        format!(
            "no card named \"{key}\" is in the footprint table. Known cards: {}.\n  \
             --list-cards prints each one with its measured model, its anchors and its \
             provenance. A card is a ROW: measuring a new part adds one and touches no code \
             path.",
            CARDS
                .iter()
                .map(|c| c.key)
                .collect::<Vec<_>>()
                .join(", ")
        )
    })
}

/// Every part that has been run.
pub fn measured_cards() -> impl Iterator<Item = &'static Card> {
    CARDS.iter().filter(|c| c.measured.is_some())
}

/// The refusal for a budget with no part named.
///
/// A budget says HOW MUCH memory, never WHICH card, and the fixed term is a
/// property of the card. The message carries what the same budget sizes on
/// each measured part, because the spread between those numbers IS the
/// argument.
pub fn budget_without_card_refusal(budget_mib: f64) -> String {
    let mut rows = Vec::new();
    for c in measured_cards() {
        let cells = c
            .cells_that_fit(budget_mib)
            .map(|n| format!("{n} cells"))
            .unwrap_or_else(|_| "no mesh at all".to_string());
        rows.push(format!(
            "    --card {:<14} {:>3} SM, {:>6.0} MiB fixed  ->  {cells}",
            c.key,
            c.streaming_multiprocessors,
            c.fixed_mib().unwrap_or(f64::NAN),
        ));
    }
    format!(
        "--vram-gib {:.0} says how much memory, not WHICH CARD, and this budget cannot be turned into a cell count without one.\n  \
         why: the footprint model's fixed term is a property of the PART. CUDA sizes the per-context local-memory backing store from the card's resident-thread capacity -- (frame - 1024 B) x SMs x maxThreadsPerSM -- so identical code pays a different fixed term on a 70 SM card and a 170 SM card.\n  \
         the breakage this refuses: sizing this budget against the wrong part's fixed term hands back a cell count that is confidently wrong in a direction the caller cannot see. On the measured parts the same budget reads:\n{}\n  \
         what to do instead: add --card KEY (--list-cards prints them), or state --cells N yourself.",
        budget_mib / 1024.0,
        rows.join("\n"),
    )
}

/// The whole table, printed.
pub fn list_cards() -> String {
    let mut out = String::from(
        "The device footprint model, per card. footprint(cells) = fixed_mib + cells x bytes_per_cell.\n\n\
         The fixed term is DERIVED from the part: a CUDA per-context local-memory backing store of\n\
         (widest kernel frame - 1024 B) x SMs x maxThreadsPerSM, plus a residue only a run on that\n\
         part can give.\n\n\
         Widest LAUNCHED frame, gf_gfdrv_stage at nVertLevels 55: ",
    );
    out.push_str(&format!(
        "{:.0} B.\nThat is its size at ArWen checkout {}, the build the MPAS port is pinned to --\nNOT in this tree, which has since cut it ({}). Every residue below was measured\nagainst that same checkout, by subtracting the store this frame derives.\n\n",
        WIDEST_KERNEL_FRAME_BYTES, MEASURED_AGAINST_ARWEN_COMMIT, FRAME_CUT_COMMIT
    ));
    for c in CARDS {
        out.push_str(&c.summary_line());
        out.push('\n');
    }
    out.push_str("\nWhy an unmeasured card is refused rather than given a neighbour's number: the residue\n");
    out.push_str("does not scale with the SM count (");
    let residues: Vec<String> = measured_cards()
        .map(|c| {
            format!(
                "{:.0} MiB on {}",
                c.measured.as_ref().expect("measured").residue_mib,
                c.key
            )
        })
        .collect();
    out.push_str(&residues.join(" against "));
    out.push_str("), so it cannot be predicted.\nA new part is a ROW, never a code path.\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The formula against the null-launch readings, on BOTH parts.
    ///
    /// The zero-frame row is the negative control: a meter that reserved
    /// memory for a kernel with no local frame would be measuring something
    /// else and every row above it would be a coincidence.
    #[test]
    fn the_local_store_formula_reproduces_the_null_launch_readings() {
        // (frame B, SMs, measured MiB)
        for (frame, sms, measured) in [
            (29_264.0f64, 70u32, 2_896.0f64),
            (26_880.0, 70, 2_650.0),
            (18_944.0, 70, 1_836.0),
            (2_048.0, 70, 104.0),
            (64.0, 70, 0.0),
            (0.0, 70, 0.0),
            (29_264.0, 170, 7_034.0),
            (26_880.0, 170, 6_438.0),
            (18_944.0, 170, 4_462.0),
            (2_048.0, 170, 254.0),
            (0.0, 170, 0.0),
        ] {
            let derived = (frame - LOCAL_FRAME_CREDIT_BYTES).max(0.0) * sms as f64 * 1_536.0 / MIB;
            let tolerance = (measured * 0.0015).max(1.0);
            assert!(
                (derived - measured).abs() <= tolerance,
                "a {frame:.0} B frame on {sms} SMs derives {derived:.1} MiB against a measured \
                 {measured:.1} MiB"
            );
        }
    }

    /// The point of the whole file: the two parts have DIFFERENT fixed terms
    /// and the difference is mostly derivable.
    #[test]
    fn the_two_measured_parts_have_different_fixed_terms() {
        let small = card("rtx-5070-ti").unwrap();
        let big = card("rtx-5090").unwrap();
        let (a, b) = (small.fixed_mib().unwrap(), big.fixed_mib().unwrap());
        assert!(
            (a - 5_384.0).abs() < 1.0,
            "the 70 SM part's derived fixed term is {a:.1} MiB, not the measured 5,384"
        );
        assert!(
            (b - 9_798.0).abs() < 1.0,
            "the 170 SM part's derived fixed term is {b:.1} MiB, not the measured 9,798"
        );
        // and the store, not the residue, is where the difference comes from
        let store_gap = big.local_store_mib() - small.local_store_mib();
        assert!(
            (store_gap - 4_138.0).abs() < 5.0,
            "the derived stores differ by {store_gap:.1} MiB, not the measured 4,138"
        );
        assert!(
            store_gap > 0.9 * (b - a),
            "the derived store explains only {:.0}% of the {:.0} MiB gap between the two \
             measured fixed terms; if that ever falls the model has stopped being a derivation",
            100.0 * store_gap / (b - a),
            b - a
        );
    }

    /// Every anchor, on every measured part. The instrument checked against
    /// the answers it claims.
    #[test]
    fn every_measured_card_reproduces_every_one_of_its_anchors() {
        let mut checked = 0;
        for c in measured_cards() {
            let m = c.measured.as_ref().unwrap();
            assert!(
                !m.anchors.is_empty(),
                "{} claims a measured model with no anchor behind it",
                c.key
            );
            for a in m.anchors {
                let modelled = c.footprint_mib(a.cells).unwrap();
                assert!(
                    (modelled - a.process_mib).abs() <= a.tolerance_mib,
                    "{}: the model reads {modelled:.1} MiB at {} cells against a measured \
                     {:.1} MiB",
                    c.key,
                    a.cells,
                    a.process_mib
                );
                checked += 1;
            }
        }
        assert!(checked >= 4, "only {checked} anchors are pinned");
    }

    /// The 5070 Ti's anchor is a whole-process peak inside a real budget, so
    /// it pins the answer a 10 GiB owner is given.
    #[test]
    fn the_measured_ten_gib_observation_is_reproduced() {
        let c = card("rtx-5070-ti").unwrap();
        // The observation: 40,962 cells completed inside a 9,656.7 MiB free
        // budget, peaking at 8,768.0 MiB.
        let peak = c.footprint_mib(40_962).unwrap();
        assert!(
            (peak - 8_768.0).abs() < 20.0,
            "the model says 40,962 cells cost {peak:.1} MiB against a measured 8,768.0"
        );
        assert!(
            peak < 9_656.7,
            "the model does not reproduce the run completing inside 9,656.7 MiB"
        );
        // and so a 10 GiB budget must be told it holds AT LEAST that mesh
        let cells = c.cells_that_fit(10.0 * 1024.0).unwrap();
        assert!(
            cells > 40_962,
            "a 10 GiB budget is told it holds {cells} cells, but 40,962 were measured \
             completing inside a SMALLER budget than that"
        );
    }

    /// A budget must never size a mesh the card cannot then hold, and must
    /// not leave the card idle either.
    #[test]
    fn a_card_budget_never_sizes_a_mesh_that_overruns_the_card() {
        for c in measured_cards() {
            for gib in [8.0f64, 10.0, 12.0, 16.0, 24.0, 32.0] {
                let budget = gib * 1024.0;
                let Ok(cells) = c.cells_that_fit(budget) else {
                    continue;
                };
                let needed = c.footprint_mib(cells).unwrap();
                assert!(
                    needed <= budget,
                    "{}: a {gib} GiB budget sized {cells} cells, which needs {needed:.0} MiB",
                    c.key
                );
                assert!(
                    c.footprint_mib(cells + 1).unwrap() > budget,
                    "{}: a {gib} GiB budget is leaving a whole cell of headroom unused",
                    c.key
                );
            }
        }
    }

    /// A budget below the fixed term holds no mesh, and says why.
    #[test]
    fn a_budget_under_the_fixed_term_is_refused_with_the_reason() {
        let c = card("rtx-5090").unwrap();
        let err = c.cells_that_fit(8.0 * 1024.0).unwrap_err();
        assert!(err.contains("no mesh of ANY size fits"), "{err}");
        assert!(err.contains("local-memory backing store"), "{err}");
        // the same budget is fine on the smaller part -- which is the whole
        // point of the model being per card
        assert!(card("rtx-5070-ti").unwrap().cells_that_fit(8.0 * 1024.0).is_ok());
    }

    /// An unmeasured part is refused BY NAME, names the breakage, and names
    /// the routes that work.
    #[test]
    fn an_unmeasured_card_is_refused_by_name() {
        let unmeasured: Vec<&Card> = CARDS.iter().filter(|c| c.measured.is_none()).collect();
        assert!(
            !unmeasured.is_empty(),
            "the table declares no unmeasured part, so this refusal is unreachable and the \
             binary would quietly answer for cards nobody has run"
        );
        for c in unmeasured {
            assert!(c.fixed_mib().is_none());
            assert!(c.footprint_mib(40_962).is_err());
            let err = c.cells_that_fit(10.0 * 1024.0).unwrap_err();
            assert!(err.contains(c.key), "the refusal does not name the card: {err}");
            assert!(err.contains("NO MEASURED"), "{err}");
            assert!(
                err.contains("confidently wrong"),
                "the refusal does not name the breakage it prevents: {err}"
            );
            assert!(
                err.contains("--cells N"),
                "the refusal does not name a route that works: {err}"
            );
            for m in measured_cards() {
                assert!(
                    err.contains(m.key),
                    "the refusal does not offer {}, which HAS been measured: {err}",
                    m.key
                );
            }
            // it is not a warning: nothing returns a number
            assert!(c.bytes_per_cell().is_none());
        }
    }

    #[test]
    fn an_unknown_card_is_refused_and_lists_the_known_ones() {
        let err = card("rtx-4090").unwrap_err();
        assert!(err.contains("rtx-4090"), "{err}");
        for c in CARDS {
            assert!(err.contains(c.key), "{err}");
        }
    }

    /// A budget with no card names no model.
    #[test]
    fn a_budget_with_no_card_is_refused_and_shows_the_spread() {
        let err = budget_without_card_refusal(16.0 * 1024.0);
        assert!(err.contains("not WHICH CARD"), "{err}");
        assert!(err.contains("--list-cards"), "{err}");
        for c in measured_cards() {
            assert!(err.contains(c.key), "{err}");
        }
        // the spread is the argument, so it has to be visible: the two parts
        // must not size 16 GiB the same way
        let a = card("rtx-5070-ti").unwrap().cells_that_fit(16.0 * 1024.0).unwrap();
        let b = card("rtx-5090").unwrap().cells_that_fit(16.0 * 1024.0).unwrap();
        assert!(
            a as f64 > 1.5 * b as f64,
            "the two parts size a 16 GiB budget at {a} and {b} cells; if those ever converge \
             this refusal has stopped being worth making"
        );
    }

    /// The frame and every residue are ONE measurement set, on ONE build.
    ///
    /// Each residue was obtained by subtracting the store this frame derives
    /// from a measured fixed term. A row whose residue came from a different
    /// checkout, sitting beside this frame, re-prices that card by thousands
    /// of MiB with nothing on screen to say so -- 2,054 MiB low on a 70 SM
    /// part if the frame moved to a post-cut build and the residue did not,
    /// which over-sizes the mesh and dies at step 1.
    ///
    /// The git-aware half of this gate -- that the port's pin has not
    /// advanced to or past `FRAME_CUT_COMMIT` -- lives in
    /// `tests/test_mpas_mesh_door.py`, which can read the repository this
    /// crate is only a subdirectory of.
    #[test]
    fn the_frame_and_every_residue_name_the_same_build() {
        assert!(
            !MEASURED_AGAINST_ARWEN_COMMIT.trim().is_empty(),
            "the frame names no checkout, so it silently means whichever \
             build the reader happens to assume"
        );
        assert!(!FRAME_CUT_COMMIT.trim().is_empty());
        assert_ne!(
            MEASURED_AGAINST_ARWEN_COMMIT, FRAME_CUT_COMMIT,
            "the frame cannot have been measured on the commit that cut it"
        );
        for c in measured_cards() {
            let m = c.measured.as_ref().unwrap();
            assert_eq!(
                m.residue_measured_against, MEASURED_AGAINST_ARWEN_COMMIT,
                "{}'s residue was measured against {} while the frame it was \
                 subtracted from belongs to {}; one of the two is stale and \
                 the fixed term derived from the pair is wrong by thousands \
                 of MiB",
                c.key, m.residue_measured_against, MEASURED_AGAINST_ARWEN_COMMIT
            );
        }
    }

    /// The frame is priced against the default stack, not against zero.
    ///
    /// The credit is the CUDA default stack limit: a frame at or under it is
    /// already covered and reserves nothing. Pricing from zero would charge
    /// every kernel in the build for memory the context already holds.
    #[test]
    fn the_frame_credit_is_the_default_stack() {
        assert_eq!(LOCAL_FRAME_CREDIT_BYTES, 1_024.0);
        // the negative control from the null-launch table, stated as the
        // property rather than as a row: nothing at or under the stack
        // reserves anything
        assert_eq!(
            (64.0f64 - LOCAL_FRAME_CREDIT_BYTES).max(0.0) * 70.0 * 1536.0,
            0.0
        );
        assert!(WIDEST_KERNEL_FRAME_BYTES > LOCAL_FRAME_CREDIT_BYTES);
    }

    /// The published table names the card for every number it prints.
    #[test]
    fn the_listing_names_a_card_beside_every_number() {
        let text = list_cards();
        for c in CARDS {
            assert!(text.contains(c.key), "{} is missing from --list-cards", c.key);
            assert!(text.contains(c.display_name), "{}", c.key);
        }
        assert!(text.contains("NOT MEASURED"));
        assert!(text.contains("MEASURED"));
        assert!(text.contains("maxThreadsPerSM"));
        assert!(
            text.contains(MEASURED_AGAINST_ARWEN_COMMIT),
            "the listing quotes a frame without naming the build it belongs \
             to, which is the whole defect one level down"
        );
    }

    /// Every measured row either separated its own slope from two or more of
    /// its own anchors, or says where the slope came from.
    #[test]
    fn a_measured_row_that_borrowed_its_slope_says_so() {
        for c in measured_cards() {
            let m = c.measured.as_ref().unwrap();
            if m.anchors.len() < 2 {
                assert!(
                    !m.bytes_per_cell_provenance.trim().is_empty(),
                    "{} fixes a two-term model from one anchor without saying what pinned the \
                     other term",
                    c.key
                );
                // and the slope it borrowed has to be one that was actually
                // separated somewhere
                assert!(
                    measured_cards()
                        .any(|o| o.measured.as_ref().unwrap().anchors.len() >= 2
                            && (o.bytes_per_cell().unwrap() - m.bytes_per_cell).abs() < 1.0),
                    "{}'s per-cell term matches no part that separated one",
                    c.key
                );
            }
            assert!(!m.provenance.trim().is_empty(), "{} has no provenance", c.key);
        }
    }
}
