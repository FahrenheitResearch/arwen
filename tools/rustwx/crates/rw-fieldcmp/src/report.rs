//! The metric table.
//!
//! The layout is the campaign judge's output contract, down to the column
//! padding and the four-significant-digit rendering, so a table produced here
//! can be diffed against a shipped one without a translation step.  Nothing
//! about the layout is case-specific: the arm names and the field labels are
//! the caller's, and the padding rule generalises the reference's two
//! hard-coded arm names to any pair.

use std::fmt::Write as _;

use crate::numfmt::{fixed, general};
use crate::{Comparison, CompositeOutcome, FieldOutcome, SummaryRecord};

/// Width the arm name is padded to on the per-field statistic lines.
const ARM_COLUMN: usize = 5;

/// Significant digits every summary statistic is printed with.
const SIGNIFICANT: usize = 4;

fn summary_line(record: &SummaryRecord) -> String {
    format!(
        "mean={} p10={} p50={} p99={} max={}",
        general(record.mean, SIGNIFICANT, false),
        general(record.p10, SIGNIFICANT, false),
        general(record.p50, SIGNIFICANT, false),
        general(record.p99, SIGNIFICANT, false),
        general(record.max, SIGNIFICANT, false),
    )
}

fn delta_line(record: &SummaryRecord) -> String {
    format!(
        "mean={} p10={} p50={} p99={}",
        general(record.mean, SIGNIFICANT, true),
        general(record.p10, SIGNIFICANT, true),
        general(record.p50, SIGNIFICANT, true),
        general(record.p99, SIGNIFICANT, true),
    )
}

/// Render the whole comparison as the judge's table.
pub fn table(comparison: &Comparison) -> String {
    let left = &comparison.left_label;
    let right = &comparison.right_label;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{left} files: {}  {right} files: {}",
        comparison.left_frame_count, comparison.right_frame_count
    );

    for frame in &comparison.frames {
        let _ = writeln!(out, "==== {}", frame.frame);
        for field in &frame.fields {
            match field {
                FieldOutcome::Missing {
                    label,
                    left_present,
                    right_present,
                    ..
                } => {
                    let _ = writeln!(
                        out,
                        "  {label}: MISSING ({left}={} {right}={})",
                        python_bool(*left_present),
                        python_bool(*right_present)
                    );
                }
                FieldOutcome::Present {
                    label,
                    left: left_summary,
                    right: right_summary,
                    delta,
                    ..
                } => {
                    let _ = writeln!(out, "  {label}:");
                    let _ = writeln!(
                        out,
                        "    {:<width$} {}",
                        left,
                        summary_line(left_summary),
                        width = ARM_COLUMN
                    );
                    let _ = writeln!(
                        out,
                        "    {:<width$} {}",
                        right,
                        summary_line(right_summary),
                        width = ARM_COLUMN
                    );
                    let _ = writeln!(out, "    delta {}", delta_line(delta));
                }
            }
        }

        for accumulation in &frame.accumulations {
            let _ = writeln!(
                out,
                "  {} domain-sum: {left}={} {right}={} ratio-1={}%",
                accumulation.variable,
                fixed(accumulation.left, 1, false),
                fixed(accumulation.right, 1, false),
                fixed(accumulation.ratio_percent, 1, true),
            );
        }

        match &frame.composite {
            CompositeOutcome::NotRequested => {}
            CompositeOutcome::Missing {
                variable,
                left_present,
                right_present,
            } => {
                let _ = writeln!(
                    out,
                    "  {}: {variable} missing ({left}={} {right}={})",
                    comparison.composite_label,
                    python_bool(*left_present),
                    python_bool(*right_present)
                );
            }
            CompositeOutcome::Present {
                counts,
                left_max,
                right_max,
                ..
            } => {
                for count in counts {
                    let _ = writeln!(
                        out,
                        "  {} >= {}: {left}={} {right}={} cells",
                        comparison.composite_label,
                        general(count.threshold, 6, false),
                        count.left,
                        count.right
                    );
                }
                let _ = writeln!(
                    out,
                    "  {} max: {left}={} {right}={}",
                    comparison.composite_label,
                    fixed(*left_max, 1, false),
                    fixed(*right_max, 1, false)
                );
            }
        }
    }
    out
}

fn python_bool(value: bool) -> &'static str {
    if value {
        "True"
    } else {
        "False"
    }
}
