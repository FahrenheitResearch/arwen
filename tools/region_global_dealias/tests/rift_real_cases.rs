use region_global_dealias::solver;

struct Case {
    name: &'static str,
    rows: usize,
    gates: usize,
    changed: usize,
    first_gate_m: f32,
    gate_spacing_m: f32,
    observed: &'static [u8],
    azimuth: &'static [u8],
    nyquist: &'static [u8],
    expected: &'static [u8],
}

const CASES: &[Case] = &[
    Case {
        name: "El Reno 2013",
        rows: 25,
        gates: 37,
        changed: 137,
        first_gate_m: 53_125.0,
        gate_spacing_m: 250.0,
        observed: include_bytes!("../test/fixtures/rift/el-reno-2013/observed.f32"),
        azimuth: include_bytes!("../test/fixtures/rift/el-reno-2013/azimuth.f32"),
        nyquist: include_bytes!("../test/fixtures/rift/el-reno-2013/nyquist.f32"),
        expected: include_bytes!("../test/fixtures/rift/el-reno-2013/expected.f32"),
    },
    Case {
        name: "Tuscaloosa 2011",
        rows: 34,
        gates: 49,
        changed: 78,
        first_gate_m: 36_875.0,
        gate_spacing_m: 250.0,
        observed: include_bytes!("../test/fixtures/rift/tuscaloosa-2011/observed.f32"),
        azimuth: include_bytes!("../test/fixtures/rift/tuscaloosa-2011/azimuth.f32"),
        nyquist: include_bytes!("../test/fixtures/rift/tuscaloosa-2011/nyquist.f32"),
        expected: include_bytes!("../test/fixtures/rift/tuscaloosa-2011/expected.f32"),
    },
];

fn f32s(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn bits_equal(left: &[f32], right: &[f32]) -> bool {
    left.iter()
        .zip(right)
        .all(|(&left, &right)| left.to_bits() == right.to_bits())
}

#[test]
fn reviewed_tornado_crops_keep_the_exact_local_branch() {
    for case in CASES {
        let observed = f32s(case.observed);
        let azimuth = f32s(case.azimuth);
        let nyquist = f32s(case.nyquist);
        let expected = f32s(case.expected);
        let resolved = solver::resolve_nyquist(&nyquist, case.rows);
        let baseline_folds = solver::region_folds(
            &observed,
            &resolved,
            case.rows,
            case.gates,
            solver::sweep_wraps(&azimuth),
        );
        let mut baseline = vec![f32::NAN; observed.len()];
        for (row, &row_nyquist) in resolved.iter().take(case.rows).enumerate() {
            for gate in 0..case.gates {
                let index = row * case.gates + gate;
                let value = observed[index];
                if value.is_finite() {
                    baseline[index] = value + 2.0 * row_nyquist * baseline_folds[index] as f32;
                }
            }
        }

        let stable = solver::dealias_sweep(&observed, &nyquist, case.rows, case.gates, &azimuth);
        assert!(
            bits_equal(&stable, &baseline),
            "{} changed the stable region-only API",
            case.name
        );

        let refined = solver::dealias_sweep_rift(
            &observed,
            &nyquist,
            case.rows,
            case.gates,
            &azimuth,
            &solver::RiftContext::default(),
            solver::RiftOptions {
                first_gate_m: case.first_gate_m,
                gate_spacing_m: case.gate_spacing_m,
                automatic_single_sweep: true,
                ..solver::RiftOptions::default()
            },
        )
        .expect("valid reviewed RIFT crop");
        assert!(
            bits_equal(&refined.velocity, &expected),
            "{} did not reproduce the reviewed field",
            case.name
        );
        let changed = refined
            .velocity
            .iter()
            .zip(&baseline)
            .filter(|(refined, baseline)| refined.to_bits() != baseline.to_bits())
            .count();
        assert_eq!(changed, case.changed, "{} changed-gate count", case.name);
        assert_eq!(
            refined.stats.gates_refined as usize, case.changed,
            "{} RIFT stats",
            case.name
        );
        assert_eq!(
            refined
                .confidence
                .iter()
                .filter(|&&confidence| confidence > 0)
                .count(),
            case.changed,
            "{} confidence mask",
            case.name
        );
    }
}

#[test]
fn automatic_roi_budgets_and_confidence_threshold_abstain_safely() {
    for case in CASES {
        let observed = f32s(case.observed);
        let azimuth = f32s(case.azimuth);
        let nyquist = f32s(case.nyquist);
        let baseline = solver::dealias_sweep(&observed, &nyquist, case.rows, case.gates, &azimuth);
        let solve = |options: solver::RiftOptions| {
            solver::dealias_sweep_rift(
                &observed,
                &nyquist,
                case.rows,
                case.gates,
                &azimuth,
                &solver::RiftContext::default(),
                options,
            )
            .expect("a strict automatic option must abstain rather than error")
        };

        for (name, options) in [
            (
                "per-ROI gate cap",
                solver::RiftOptions {
                    max_roi_gates: 1,
                    first_gate_m: case.first_gate_m,
                    gate_spacing_m: case.gate_spacing_m,
                    automatic_single_sweep: true,
                    ..solver::RiftOptions::default()
                },
            ),
            (
                "total ROI gate cap",
                solver::RiftOptions {
                    max_total_roi_gates: 1,
                    first_gate_m: case.first_gate_m,
                    gate_spacing_m: case.gate_spacing_m,
                    automatic_single_sweep: true,
                    ..solver::RiftOptions::default()
                },
            ),
        ] {
            let result = solve(options);
            assert!(
                bits_equal(&result.velocity, &baseline),
                "{} changed under {name}",
                case.name
            );
            assert_eq!(result.stats.rois_accepted, 0, "{} {name}", case.name);
            assert_eq!(result.stats.gates_refined, 0, "{} {name}", case.name);
            assert_eq!(result.stats.budget_aborts, 1, "{} {name}", case.name);
            assert!(result.confidence.iter().all(|&value| value == 0));
            assert!(result.reasons.iter().any(|reason| {
                reason & solver::RIFT_REASON_BUDGET_EXCEEDED != 0
                    && reason & solver::RIFT_REASON_ABSTAINED != 0
            }));
        }

        let result = solve(solver::RiftOptions {
            min_confidence: 221,
            first_gate_m: case.first_gate_m,
            gate_spacing_m: case.gate_spacing_m,
            automatic_single_sweep: true,
            ..solver::RiftOptions::default()
        });
        assert!(
            bits_equal(&result.velocity, &baseline),
            "{} changed above the automatic confidence level",
            case.name
        );
        assert_eq!(result.stats.rois_accepted, 0, "{} confidence", case.name);
        assert_eq!(result.stats.gates_refined, 0, "{} confidence", case.name);
        assert_eq!(result.stats.budget_aborts, 0, "{} confidence", case.name);
        assert!(result.confidence.iter().all(|&value| value == 0));
        assert!(
            result
                .reasons
                .iter()
                .any(|reason| reason & solver::RIFT_REASON_ABSTAINED != 0)
        );
    }
}

#[test]
fn automatic_max_rois_is_a_strict_work_cap() {
    let case = &CASES[0];
    let source = f32s(case.observed);
    let azimuth = f32s(case.azimuth);
    let nyquist = f32s(case.nyquist);
    let second_start = case.gates + 16;
    let gates = second_start + case.gates;
    let mut observed = vec![f32::NAN; case.rows * gates];
    for row in 0..case.rows {
        let source_row = &source[row * case.gates..(row + 1) * case.gates];
        let target_row = &mut observed[row * gates..(row + 1) * gates];
        target_row[..case.gates].copy_from_slice(source_row);
        target_row[second_start..second_start + case.gates].copy_from_slice(source_row);
    }

    let result = solver::dealias_sweep_rift(
        &observed,
        &nyquist,
        case.rows,
        gates,
        &azimuth,
        &solver::RiftContext::default(),
        solver::RiftOptions {
            max_rois: 1,
            max_total_roi_gates: observed.len() as u32,
            first_gate_m: case.first_gate_m,
            gate_spacing_m: case.gate_spacing_m,
            automatic_single_sweep: true,
            ..solver::RiftOptions::default()
        },
    )
    .expect("a one-ROI automatic work cap must be valid");

    assert!(result.stats.rois_detected >= 2);
    assert!(result.stats.rois_solved <= 1);
    assert!(result.stats.rois_accepted <= 1);
    assert_eq!(result.stats.budget_aborts, 1);
    assert!(result.reasons.iter().any(|reason| {
        reason & solver::RIFT_REASON_BUDGET_EXCEEDED != 0
            && reason & solver::RIFT_REASON_ABSTAINED != 0
    }));
}

#[test]
fn caller_reference_splits_one_observed_region_at_gate_resolution() {
    const ROWS: usize = 24;
    const GATES: usize = 36;
    const NYQUIST: f32 = 20.0;
    let observed = vec![-5.0f32; ROWS * GATES];
    let nyquist = vec![NYQUIST; ROWS];
    let azimuth: Vec<f32> = (0..ROWS)
        .map(|row| row as f32 * 360.0 / ROWS as f32)
        .collect();
    let mut reference = observed.clone();
    for row in 8..16 {
        for gate in 13..23 {
            reference[row * GATES + gate] = 35.0;
        }
    }

    let stable = solver::dealias_sweep(&observed, &nyquist, ROWS, GATES, &azimuth);
    assert!(bits_equal(&stable, &observed));

    let references = [solver::ReferenceField {
        velocity: &reference,
        quality: None,
        kind: solver::ReferenceKind::Caller,
    }];
    let context = solver::RiftContext {
        references: &references,
        ..solver::RiftContext::default()
    };
    let result = solver::dealias_sweep_rift(
        &observed,
        &nyquist,
        ROWS,
        GATES,
        &azimuth,
        &context,
        solver::RiftOptions {
            max_total_roi_gates: (ROWS * GATES) as u32,
            ..solver::RiftOptions::default()
        },
    )
    .expect("valid caller-reference RIFT input");

    let mut changed = 0usize;
    for row in 0..ROWS {
        for gate in 0..GATES {
            let index = row * GATES + gate;
            let in_lobe = (8..16).contains(&row) && (13..23).contains(&gate);
            let expected = if in_lobe { 35.0 } else { -5.0 };
            assert_eq!(result.velocity[index], expected, "gate ({row}, {gate})");
            assert_eq!(result.folds[index], i8::from(in_lobe));
            if in_lobe {
                changed += 1;
                assert!(result.confidence[index] >= 160);
                assert_ne!(result.reasons[index] & solver::RIFT_REASON_CALLER_ANCHOR, 0);
                assert_ne!(
                    result.reasons[index] & solver::RIFT_REASON_FUSION_ACCEPTED,
                    0
                );
            } else {
                assert_eq!(result.confidence[index], 0);
            }
        }
    }
    assert_eq!(changed, 80);
    assert_eq!(result.stats.gates_refined, 80);
    assert_eq!(result.stats.rois_accepted, 1);
}

#[test]
fn empty_context_is_exactly_the_legacy_result_when_no_local_roi_is_authorized() {
    const ROWS: usize = 32;
    const GATES: usize = 64;
    let observed: Vec<f32> = (0..ROWS * GATES)
        .map(|index| -12.0 + (index % GATES) as f32 * 24.0 / GATES as f32)
        .collect();
    let nyquist = vec![25.0; ROWS];
    let azimuth: Vec<f32> = (0..ROWS)
        .map(|row| row as f32 * 360.0 / ROWS as f32)
        .collect();
    let legacy = solver::dealias_sweep(&observed, &nyquist, ROWS, GATES, &azimuth);
    let rift = solver::dealias_sweep_rift(
        &observed,
        &nyquist,
        ROWS,
        GATES,
        &azimuth,
        &solver::RiftContext::default(),
        solver::RiftOptions::default(),
    )
    .expect("valid no-context RIFT input");
    assert!(bits_equal(&rift.velocity, &legacy));
    assert!(rift.confidence.iter().all(|&value| value == 0));
    assert_eq!(rift.stats.gates_refined, 0);
}

#[test]
fn caller_reference_can_select_multiple_folds_and_preserves_missing_gates() {
    const ROWS: usize = 22;
    const GATES: usize = 34;
    const NYQUIST: f32 = 20.0;
    for expected_fold in [2i8, -3i8] {
        let wrapped = if expected_fold > 0 { -5.0 } else { 5.0 };
        let mut observed = vec![wrapped; ROWS * GATES];
        let nyquist = vec![NYQUIST; ROWS];
        let azimuth: Vec<f32> = (0..ROWS)
            .map(|row| row as f32 * 360.0 / ROWS as f32)
            .collect();
        let mut reference = observed.clone();
        for row in 7..15 {
            for gate in 11..23 {
                let index = row * GATES + gate;
                reference[index] = wrapped + 2.0 * NYQUIST * f32::from(expected_fold);
            }
        }
        let missing = 10 * GATES + 16;
        observed[missing] = f32::NAN;
        reference[missing] = f32::NAN;

        let references = [solver::ReferenceField {
            velocity: &reference,
            quality: None,
            kind: solver::ReferenceKind::Caller,
        }];
        let context = solver::RiftContext {
            references: &references,
            ..solver::RiftContext::default()
        };
        let result = solver::dealias_sweep_rift(
            &observed,
            &nyquist,
            ROWS,
            GATES,
            &azimuth,
            &context,
            solver::RiftOptions {
                max_total_roi_gates: (ROWS * GATES) as u32,
                ..solver::RiftOptions::default()
            },
        )
        .expect("valid multi-fold reference");

        for row in 0..ROWS {
            for gate in 0..GATES {
                let index = row * GATES + gate;
                if index == missing {
                    assert!(result.velocity[index].is_nan());
                    assert_eq!(result.folds[index], 0);
                    continue;
                }
                let in_lobe = (7..15).contains(&row) && (11..23).contains(&gate);
                assert_eq!(
                    result.folds[index],
                    if in_lobe { expected_fold } else { 0 },
                    "fold {expected_fold}, gate ({row}, {gate})"
                );
            }
        }
    }
}

#[test]
fn reference_fusion_uses_each_rays_nyquist_interval() {
    const ROWS: usize = 24;
    const GATES: usize = 40;
    let observed = vec![-4.0f32; ROWS * GATES];
    let nyquist: Vec<f32> = (0..ROWS)
        .map(|row| if row < ROWS / 2 { 20.0 } else { 26.0 })
        .collect();
    let azimuth: Vec<f32> = (0..ROWS)
        .map(|row| row as f32 * 360.0 / ROWS as f32)
        .collect();
    let mut reference = observed.clone();
    for row in 6..18 {
        for gate in 14..27 {
            reference[row * GATES + gate] = -4.0 + 2.0 * nyquist[row];
        }
    }
    let references = [solver::ReferenceField {
        velocity: &reference,
        quality: None,
        kind: solver::ReferenceKind::Temporal,
    }];
    let context = solver::RiftContext {
        references: &references,
        ..solver::RiftContext::default()
    };
    let result = solver::dealias_sweep_rift(
        &observed,
        &nyquist,
        ROWS,
        GATES,
        &azimuth,
        &context,
        solver::RiftOptions {
            max_total_roi_gates: (ROWS * GATES) as u32,
            ..solver::RiftOptions::default()
        },
    )
    .expect("valid split-Nyquist reference");

    for (row, &row_nyquist) in nyquist.iter().take(ROWS).enumerate() {
        for gate in 0..GATES {
            let index = row * GATES + gate;
            let in_lobe = (6..18).contains(&row) && (14..27).contains(&gate);
            assert_eq!(result.folds[index], i8::from(in_lobe));
            let expected = if in_lobe {
                -4.0 + 2.0 * row_nyquist
            } else {
                -4.0
            };
            assert_eq!(result.velocity[index], expected);
        }
    }
}

#[test]
fn reference_budget_exhaustion_abstains_without_changing_the_baseline() {
    const ROWS: usize = 20;
    const GATES: usize = 30;
    let observed = vec![-5.0f32; ROWS * GATES];
    let mut reference = observed.clone();
    for row in 5..15 {
        for gate in 8..22 {
            reference[row * GATES + gate] = 35.0;
        }
    }
    let nyquist = vec![20.0; ROWS];
    let azimuth: Vec<f32> = (0..ROWS)
        .map(|row| row as f32 * 360.0 / ROWS as f32)
        .collect();
    let references = [solver::ReferenceField {
        velocity: &reference,
        quality: None,
        kind: solver::ReferenceKind::Caller,
    }];
    let context = solver::RiftContext {
        references: &references,
        ..solver::RiftContext::default()
    };
    let result = solver::dealias_sweep_rift(
        &observed,
        &nyquist,
        ROWS,
        GATES,
        &azimuth,
        &context,
        solver::RiftOptions {
            max_total_roi_gates: 1,
            ..solver::RiftOptions::default()
        },
    )
    .expect("a tight budget must abstain rather than error");
    assert!(bits_equal(&result.velocity, &observed));
    assert_eq!(result.stats.gates_refined, 0);
    assert_eq!(result.stats.budget_aborts, 1);
    assert!(result.reasons.iter().any(|reason| {
        reason & solver::RIFT_REASON_BUDGET_EXCEEDED != 0
            && reason & solver::RIFT_REASON_ABSTAINED != 0
    }));
}

#[test]
fn conflicting_references_abstain_instead_of_becoming_order_dependent() {
    const ROWS: usize = 20;
    const GATES: usize = 32;
    let observed = vec![-5.0f32; ROWS * GATES];
    let nyquist = vec![20.0; ROWS];
    let azimuth: Vec<f32> = (0..ROWS)
        .map(|row| row as f32 * 360.0 / ROWS as f32)
        .collect();
    let mut positive = observed.clone();
    let mut negative = observed.clone();
    for row in 6..14 {
        for gate in 10..22 {
            let index = row * GATES + gate;
            positive[index] = 35.0;
            negative[index] = -45.0;
        }
    }

    let solve = |reverse: bool| {
        let fields = if reverse {
            [
                solver::ReferenceField {
                    velocity: &negative,
                    quality: None,
                    kind: solver::ReferenceKind::Vertical,
                },
                solver::ReferenceField {
                    velocity: &positive,
                    quality: None,
                    kind: solver::ReferenceKind::Temporal,
                },
            ]
        } else {
            [
                solver::ReferenceField {
                    velocity: &positive,
                    quality: None,
                    kind: solver::ReferenceKind::Temporal,
                },
                solver::ReferenceField {
                    velocity: &negative,
                    quality: None,
                    kind: solver::ReferenceKind::Vertical,
                },
            ]
        };
        solver::dealias_sweep_rift(
            &observed,
            &nyquist,
            ROWS,
            GATES,
            &azimuth,
            &solver::RiftContext {
                references: &fields,
                ..solver::RiftContext::default()
            },
            solver::RiftOptions {
                max_total_roi_gates: (ROWS * GATES) as u32,
                ..solver::RiftOptions::default()
            },
        )
        .expect("valid conflicting-reference input")
    };

    let forward = solve(false);
    let reversed = solve(true);
    assert!(bits_equal(&forward.velocity, &observed));
    assert!(bits_equal(&forward.velocity, &reversed.velocity));
    assert_eq!(forward.reasons, reversed.reasons);
    for row in 6..14 {
        for gate in 10..22 {
            let reason = forward.reasons[row * GATES + gate];
            assert_ne!(reason & solver::RIFT_REASON_CONFLICTING_REFERENCES, 0);
            assert_ne!(reason & solver::RIFT_REASON_ABSTAINED, 0);
        }
    }
}
