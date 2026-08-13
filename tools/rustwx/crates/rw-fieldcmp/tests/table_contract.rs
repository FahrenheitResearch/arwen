//! The table layout is an output contract, not a style choice.
//!
//! Downstream readers parse these tables, and shipped verdicts quote them
//! line by line, so a stray space or a dropped significant digit is a
//! breaking change.  The blocks asserted here are copied verbatim out of a
//! shipped campaign verdict, which makes this test a check against the
//! consumer's expectations rather than against this crate's own opinion.

use rw_fieldcmp::{
    AccumulationOutcome, Comparison, CompositeCount, CompositeOutcome, FieldOutcome,
    FrameComparison, SummaryRecord,
};

fn summary(mean: f64, p10: f64, p50: f64, p99: f64, max: f64) -> SummaryRecord {
    SummaryRecord {
        mean,
        p10,
        p50,
        p99,
        max,
    }
}

fn field(label: &str, left: SummaryRecord, right: SummaryRecord) -> FieldOutcome {
    FieldOutcome::Present {
        variable: label.to_string(),
        label: label.to_string(),
        cells: 50_000,
        left,
        right,
        delta: summary(
            left.mean - right.mean,
            left.p10 - right.p10,
            left.p50 - right.p50,
            left.p99 - right.p99,
            left.max - right.max,
        ),
        left_nonfinite: 0,
        right_nonfinite: 0,
    }
}

fn comparison(frames: Vec<FrameComparison>) -> Comparison {
    Comparison {
        left_label: "arwen".to_string(),
        right_label: "wrf".to_string(),
        left_directory: "left".to_string(),
        right_directory: "right".to_string(),
        left_frame_count: 14,
        right_frame_count: 14,
        composite_label: "comp-dBZ".to_string(),
        frames,
    }
}

/// A frame with every row type populated, checked against the shipped
/// verdict's 13Z block.
#[test]
fn a_scored_frame_matches_the_shipped_layout() {
    let rendered = rw_fieldcmp::report::table(&comparison(vec![FrameComparison {
        frame: "wrfout_d01_1974-04-03_13_00_00".to_string(),
        fields: vec![field(
            "MYJ PBLH m",
            summary(450.3, 16.25, 327.9, 1525.6781, 4715.0),
            summary(449.266, 16.19822, 327.0387, 1526.4, 4715.0),
        )],
        accumulations: vec![AccumulationOutcome {
            variable: "RAINNC".to_string(),
            left: 223.1,
            right: 222.4,
            left_f64: 223.1,
            right_f64: 222.4,
            ratio_percent: (223.1 / 222.4 - 1.0) * 100.0,
        }],
        composite: CompositeOutcome::Present {
            variable: "REFL_10CM".to_string(),
            columns: 50_000,
            levels: 50,
            counts: vec![
                CompositeCount {
                    threshold: 20.0,
                    left: 581,
                    right: 568,
                },
                CompositeCount {
                    threshold: 40.0,
                    left: 1,
                    right: 0,
                },
            ],
            left_max: 40.04,
            right_max: 32.91,
        },
    }]));

    let expected = "\
arwen files: 14  wrf files: 14
==== wrfout_d01_1974-04-03_13_00_00
  MYJ PBLH m:
    arwen mean=450.3 p10=16.25 p50=327.9 p99=1526 max=4715
    wrf   mean=449.3 p10=16.2 p50=327 p99=1526 max=4715
    delta mean=+1.034 p10=+0.05178 p50=+0.8613 p99=-0.7219
  RAINNC domain-sum: arwen=223.1 wrf=222.4 ratio-1=+0.3%
  comp-dBZ >= 20: arwen=581 wrf=568 cells
  comp-dBZ >= 40: arwen=1 wrf=0 cells
  comp-dBZ max: arwen=40.0 wrf=32.9
";
    assert_eq!(rendered, expected);
}

/// The frame where an arm has not written reflectivity yet, and where a
/// domain sum of zero makes the ratio undefined.  Both appear in every
/// shipped verdict's first frame, so both have to render exactly.
#[test]
fn an_absent_variable_and_an_undefined_ratio_match_the_shipped_layout() {
    let rendered = rw_fieldcmp::report::table(&comparison(vec![FrameComparison {
        frame: "wrfout_d01_1974-04-03_12_00_00".to_string(),
        fields: vec![FieldOutcome::Missing {
            variable: "SNOWH".to_string(),
            label: "snow depth m".to_string(),
            left_present: false,
            right_present: true,
        }],
        accumulations: vec![AccumulationOutcome {
            variable: "RAINC".to_string(),
            left: 0.0,
            right: 0.0,
            left_f64: 0.0,
            right_f64: 0.0,
            ratio_percent: f64::NAN,
        }],
        composite: CompositeOutcome::Missing {
            variable: "REFL_10CM".to_string(),
            left_present: false,
            right_present: true,
        },
    }]));

    let expected = "\
arwen files: 14  wrf files: 14
==== wrfout_d01_1974-04-03_12_00_00
  snow depth m: MISSING (arwen=False wrf=True)
  RAINC domain-sum: arwen=0.0 wrf=0.0 ratio-1=+nan%
  comp-dBZ: REFL_10CM missing (arwen=False wrf=True)
";
    assert_eq!(rendered, expected);
}
