"""Synthetic coverage for the stock-WRF feedback signature hook."""

from tools.compare_feedback_signature import (
    _operator_descriptor,
    synthetic_self_test,
)


def test_feedback_signature_harness_synthetic_self_test():
    report = synthetic_self_test()
    assert report["schema"] == "gpuwm-feedback-signature-comparison-v2"
    assert report["certification_kind"] == "signature-not-bit-comparison"
    assert report["all_signatures_track"] is True
    assert report["rows"][0]["field"] == "T"
    assert report["rows"][0]["operator"] == {
        "class": "mass",
        "routine": "copy_fcn",
        "stagger": "mass",
        "stencil": "cell-average",
        "source_points_per_parent": 9,
    }


def test_feedback_signature_operator_classes_mirror_wrf_split():
    assert _operator_descriptor(
        "T", ratio=4, x_staggered=False, y_staggered=False
    )["source_points_per_parent"] == 16
    assert _operator_descriptor(
        "U", ratio=4, x_staggered=True, y_staggered=False
    ) == {
        "class": "u-face",
        "routine": "copy_fcn",
        "stagger": "x",
        "stencil": "face-average",
        "source_points_per_parent": 4,
    }
    assert _operator_descriptor(
        "V", ratio=4, x_staggered=False, y_staggered=True
    )["class"] == "v-face"
    assert _operator_descriptor(
        "TSK", ratio=4, x_staggered=False, y_staggered=False
    ) == {
        "class": "masked",
        "routine": "copy_fcnm",
        "stagger": "mass",
        "stencil": "nearest-neighbor-southwest",
        "source_points_per_parent": 1,
    }
    # Tagged WRF 4.6.1 copy_fcnm is a one-point coincident pick on odd
    # ratios too; only the chosen point changes between the parity branches.
    assert _operator_descriptor(
        "TSK", ratio=3, x_staggered=False, y_staggered=False
    )["stencil"] == "coincident-point"
