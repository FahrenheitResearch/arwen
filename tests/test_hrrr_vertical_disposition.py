from __future__ import annotations

import base64
import copy
import hashlib
from types import MappingProxyType
import zlib

import numpy as np
import pytest

from gpuwm.ingest import real
from gpuwm.ingest.preprocess_backend import resolve_preprocess_backend
from gpuwm.verify.npref import np_wrf_real_vert_interp


_BASE_PRESSURES = (
    110000.0, 100000.0, 95000.0, 94000.0,
    80000.0, 50000.0, 30000.0, 20000.0,
)


def _case(
        source_pressures, surface_pressure, target_pressures, source_level,
        *, replay_factory=None):
    source_pressure = np.asarray(
        source_pressures, dtype=np.float32)[:, None, None]
    surface_pressure = np.asarray(
        [[surface_pressure]], dtype=np.float32)
    target_pressure = np.asarray(
        target_pressures, dtype=np.float32)[:, None, None]
    source = np.zeros(source_pressure.shape, dtype=np.float32)
    source[source_level, 0, 0] = np.float32(1.0)

    if replay_factory is None:
        def replay(field):
            return np.asarray(np_wrf_real_vert_interp(
                field, np.zeros((1, 1), dtype=np.float32),
                source_pressure, surface_pressure, target_pressure,
                interp_in_logp=True, extrap="constant",
                vboundb=target_pressure.shape[0] + 1), dtype=np.float32)
    else:
        replay = replay_factory(
            source_pressure, surface_pressure, target_pressure)
    initialized = replay(source)
    evidence = real.build_hrrr_hydrometeor_vertical_disposition(
        {"QC": source}, np.arange(source.shape[0], dtype=np.int32),
        source_pressure, surface_pressure, target_pressure,
        {"QC": initialized}, operator_replay=replay)
    validation = real.validate_hrrr_hydrometeor_vertical_disposition(
        {"QC": real.array_correspondence_fingerprint(source)},
        {"QC": real.array_correspondence_fingerprint(initialized)},
        evidence)
    labels = np.frombuffer(zlib.decompress(base64.b64decode(
        evidence["species"]["QC"]["labels_base64"])), dtype=np.uint8)
    return source, initialized, evidence, validation, labels


@pytest.mark.parametrize(
    ("source_pressures", "surface_pressure", "target_pressures",
     "source_level", "expected_class"),
    (
        (_BASE_PRESSURES, 97000.0,
         (90000.0, 85000.0, 70000.0, 40000.0, 25000.0), 2,
         "WRF_FORCE_SURFACE_EXCLUDED"),
        (_BASE_PRESSURES, 99700.0,
         (90000.0, 85000.0, 70000.0, 40000.0, 25000.0), 1,
         "WRF_ZAP_CLOSE_EXCLUDED"),
        (_BASE_PRESSURES, 95400.0,
         (95000.0, 90000.0, 70000.0, 40000.0, 25000.0), 2,
         "WRF_ZAP_CLOSE_EXCLUDED"),
        (_BASE_PRESSURES, 99500.0,
         (90000.0, 85000.0, 70000.0, 40000.0, 25000.0), 1,
         "WRF_BELOW_GROUND_OUTSIDE_TARGET_SUPPORT"),
        (_BASE_PRESSURES, 95500.0,
         (95000.0, 90000.0, 70000.0, 40000.0, 25000.0), 2,
         "TARGET_INFLUENCING"),
        ((95000.0, 94600.0, 90000.0, 70000.0,
          50000.0, 30000.0, 20000.0), 95400.0,
         (95000.0, 85000.0, 70000.0, 40000.0, 25000.0), 0,
         "WRF_ZAP_CLOSE_EXCLUDED"),
        (_BASE_PRESSURES, 97000.0,
         (90000.0, 85000.0, 70000.0, 40000.0, 25000.0), 0,
         "WRF_BELOW_GROUND_OUTSIDE_TARGET_SUPPORT"),
        (_BASE_PRESSURES, 97000.0,
         (90000.0, 85000.0, 70000.0, 60000.0, 40000.0), 7,
         "WRF_ABOVE_TARGET_TOP_OUTSIDE_TARGET_SUPPORT"),
        ((110000.0, 90000.0, 80000.0, 70000.0,
          60000.0, 30000.0, 20000.0), 100000.0,
         (95000.0, 65000.0, 40000.0, 25000.0), 2,
         "WRF_NO_TARGET_STENCIL"),
        (_BASE_PRESSURES, 97000.0,
         (90000.0, 85000.0, 70000.0, 40000.0, 25000.0), 4,
         "TARGET_INFLUENCING"),
    ),
)
def test_sparse_source_disposition_matches_the_wrf_operator(
        source_pressures, surface_pressure, target_pressures, source_level,
        expected_class):
    source, initialized, evidence, validation, labels = _case(
        source_pressures, surface_pressure, target_pressures, source_level)

    code = real._HRRR_DISPOSITION_CLASSES[expected_class]
    assert labels[source_level] == code
    assert np.count_nonzero(labels) == np.count_nonzero(source) == 1
    assert evidence["species"]["QC"]["class_counts"][expected_class] == 1
    strength = validation["species"]["QC"]["strength"]
    if expected_class == "TARGET_INFLUENCING":
        assert strength == "PROVEN"
        assert np.count_nonzero(initialized) > 0
    else:
        assert strength == "WRF_EXCLUDED"
        assert np.count_nonzero(initialized) == 0
    replay = evidence["species"]["QC"]["operator_replay"]
    assert replay["source_output"] == replay["target_influencing_output"]
    assert replay["excluded_output"]["nonzero_count"] == 0


def test_production_replay_refuses_a_false_exclusion(monkeypatch):
    authority = real._hrrr_operator_geometry

    def falsely_exclude(*args, **kwargs):
        geometry = authority(*args, **kwargs)
        geometry["operator_class"][4, 0, 0] = np.uint8(0)
        return geometry

    monkeypatch.setattr(real, "_hrrr_operator_geometry", falsely_exclude)
    with pytest.raises(ValueError, match="falsely excluded or included"):
        _case(
            _BASE_PRESSURES, 97000.0,
            (90000.0, 85000.0, 70000.0, 40000.0, 25000.0), 4)


def test_disposition_refuses_negative_and_unclassified_source(monkeypatch):
    source_pressure = np.asarray(
        _BASE_PRESSURES, dtype=np.float32)[:, None, None]
    surface_pressure = np.asarray([[97000.0]], dtype=np.float32)
    target_pressure = np.asarray(
        [90000.0, 70000.0, 40000.0], dtype=np.float32)[:, None, None]
    source = np.zeros(source_pressure.shape, dtype=np.float32)
    source[4] = np.float32(-1.0)

    with pytest.raises(ValueError, match="source QC is invalid"):
        real.build_hrrr_hydrometeor_vertical_disposition(
            {"QC": source}, np.arange(source.shape[0]),
            source_pressure, surface_pressure, target_pressure,
            {"QC": np.zeros(target_pressure.shape, dtype=np.float32)},
            operator_replay=lambda value: np.zeros(
                target_pressure.shape, dtype=np.float32))

    authority = real._hrrr_operator_geometry

    def unclassified(*args, **kwargs):
        geometry = authority(*args, **kwargs)
        geometry["operator_class"][4, 0, 0] = np.uint8(255)
        return geometry

    monkeypatch.setattr(real, "_hrrr_operator_geometry", unclassified)
    with pytest.raises(AssertionError, match="geometry is unclassified"):
        _case(
            _BASE_PRESSURES, 97000.0,
            (90000.0, 85000.0, 70000.0, 40000.0, 25000.0), 4)


def test_coherent_target_to_excluded_mutation_is_refused_by_support():
    source, initialized, evidence, _validation, _labels = _case(
        _BASE_PRESSURES, 97000.0,
        (90000.0, 85000.0, 70000.0, 40000.0, 25000.0), 4)
    changed = copy.deepcopy(evidence)
    item = changed["species"]["QC"]
    labels = np.frombuffer(zlib.decompress(base64.b64decode(
        item["labels_base64"])), dtype=np.uint8).copy()
    target_code = real._HRRR_DISPOSITION_CLASSES["TARGET_INFLUENCING"]
    excluded_name = "WRF_NO_TARGET_STENCIL"
    excluded_code = real._HRRR_DISPOSITION_CLASSES[excluded_name]
    flat_index = int(np.flatnonzero(labels == target_code)[0])
    labels[flat_index] = excluded_code
    item["labels_base64"] = base64.b64encode(zlib.compress(
        labels.tobytes(), level=9)).decode("ascii")
    item["labels_sha256"] = hashlib.sha256(labels.tobytes()).hexdigest()
    for class_name, code in real._HRRR_DISPOSITION_CLASSES.items():
        mask = labels == code
        item["class_counts"][class_name] = int(np.count_nonzero(mask))
        item["class_mask_sha256"][class_name] = real._packed_mask_sha256(mask)
    moved = item["examples"]["TARGET_INFLUENCING"]["records"].pop()
    item["examples"]["TARGET_INFLUENCING"].update({
        "complete": True, "total_count": 0})
    moved["class"] = excluded_name
    moved["influencing_target_indices"] = []
    item["examples"][excluded_name]["records"] = [moved]
    item["examples"][excluded_name].update({
        "complete": True, "total_count": 1})
    item["target_influencing_source_count"] = 0
    item["wrf_excluded_source_count"] = 1
    unsigned = dict(changed)
    unsigned.pop("evidence_sha256")
    changed["evidence_sha256"] = real._canonical_receipt_sha256(unsigned)

    with pytest.raises(ValueError, match="exact production target support"):
        real.validate_hrrr_hydrometeor_vertical_disposition(
            {"QC": real.array_correspondence_fingerprint(source)},
            {"QC": real.array_correspondence_fingerprint(initialized)},
            changed)


def test_coherent_excluded_to_target_mutation_is_refused_by_support():
    source_pressure = np.asarray(
        _BASE_PRESSURES, dtype=np.float32)[:, None, None]
    surface_pressure = np.asarray([[97000.0]], dtype=np.float32)
    target_pressure = np.asarray(
        [90000.0, 85000.0, 70000.0, 40000.0, 25000.0],
        dtype=np.float32)[:, None, None]
    source = np.zeros(source_pressure.shape, dtype=np.float32)
    source[0] = np.float32(1.0)
    source[4] = np.float32(1.0)

    def replay(field):
        return np.asarray(np_wrf_real_vert_interp(
            field, np.zeros((1, 1), dtype=np.float32),
            source_pressure, surface_pressure, target_pressure,
            interp_in_logp=True, extrap="constant",
            vboundb=target_pressure.shape[0] + 1), dtype=np.float32)

    initialized = replay(source)
    evidence = real.build_hrrr_hydrometeor_vertical_disposition(
        {"QC": source}, np.arange(source.shape[0], dtype=np.int32),
        source_pressure, surface_pressure, target_pressure,
        {"QC": initialized}, operator_replay=replay)
    changed = copy.deepcopy(evidence)
    item = changed["species"]["QC"]
    labels = np.frombuffer(zlib.decompress(base64.b64decode(
        item["labels_base64"])), dtype=np.uint8).copy()
    target_name = "TARGET_INFLUENCING"
    target_code = real._HRRR_DISPOSITION_CLASSES[target_name]
    excluded_name = "WRF_BELOW_GROUND_OUTSIDE_TARGET_SUPPORT"
    excluded_code = real._HRRR_DISPOSITION_CLASSES[excluded_name]
    assert labels[0] == excluded_code
    assert labels[4] == target_code
    labels[0] = target_code
    item["labels_base64"] = base64.b64encode(zlib.compress(
        labels.tobytes(), level=9)).decode("ascii")
    item["labels_sha256"] = hashlib.sha256(labels.tobytes()).hexdigest()
    for class_name, code in real._HRRR_DISPOSITION_CLASSES.items():
        mask = labels == code
        item["class_counts"][class_name] = int(np.count_nonzero(mask))
        item["class_mask_sha256"][class_name] = \
            real._packed_mask_sha256(mask)
    moved = item["examples"][excluded_name]["records"].pop()
    item["examples"][excluded_name].update({
        "complete": True, "total_count": 0})
    moved["class"] = target_name
    moved["influencing_target_indices"] = [0]
    item["examples"][target_name]["records"].append(moved)
    item["examples"][target_name].update({
        "complete": True, "total_count": 2})
    item["target_influencing_source_count"] = 2
    item["wrf_excluded_source_count"] = 0
    replay_receipt = item["operator_replay"]
    replay_receipt["target_influencing_mask_sha256"] = \
        real._packed_mask_sha256(labels == target_code)
    replay_receipt["excluded_mask_sha256"] = real._packed_mask_sha256(
        (labels != 0) & (labels != target_code))
    unsigned = dict(changed)
    unsigned.pop("evidence_sha256")
    changed["evidence_sha256"] = real._canonical_receipt_sha256(unsigned)

    with pytest.raises(ValueError, match="exact production target support"):
        real.validate_hrrr_hydrometeor_vertical_disposition(
            {"QC": real.array_correspondence_fingerprint(source)},
            {"QC": real.array_correspondence_fingerprint(initialized)},
            changed)


def test_restored_mappingproxy_disposition_keeps_the_same_identity():
    source, initialized, evidence, validation, _labels = _case(
        _BASE_PRESSURES, 97000.0,
        (90000.0, 85000.0, 70000.0, 40000.0, 25000.0), 4)

    def freeze(value):
        if isinstance(value, dict):
            return MappingProxyType({
                key: freeze(child) for key, child in value.items()})
        if isinstance(value, list):
            return [freeze(child) for child in value]
        return value

    restored = real.validate_hrrr_hydrometeor_vertical_disposition(
        freeze({"QC": real.array_correspondence_fingerprint(source)}),
        freeze({"QC": real.array_correspondence_fingerprint(initialized)}),
        freeze(evidence))

    assert restored == validation


def test_numpy_and_production_cpu_disposition_are_identical_when_available():
    try:
        backend = resolve_preprocess_backend("cpu", workers=1)
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"native CPU preprocess bridge is not built: {exc}")

    def cpu_replay_factory(source_pressure, surface_pressure, target_pressure):
        plan = backend.prepare_wrf_vertical(
            source_pressure, surface_pressure, target_pressure)

        def replay(field):
            return plan.apply(
                field, np.zeros(surface_pressure.shape, dtype=np.float32),
                interp_in_logp=True, extrap="constant",
                vboundb=target_pressure.shape[0] + 1)

        return replay

    arguments = (
        _BASE_PRESSURES, 97000.0,
        (90000.0, 85000.0, 70000.0, 40000.0, 25000.0), 4)
    numpy_case = _case(*arguments)
    cpu_case = _case(*arguments, replay_factory=cpu_replay_factory)
    assert numpy_case[4].tobytes() == cpu_case[4].tobytes()
    assert (numpy_case[2]["species"]["QC"]["class_counts"]
            == cpu_case[2]["species"]["QC"]["class_counts"])
    assert (numpy_case[2]["geometry"]["production_target_support"]
            ["mask_sha256"]
            == cpu_case[2]["geometry"]["production_target_support"]
            ["mask_sha256"])
    np.testing.assert_allclose(
        cpu_case[1], numpy_case[1], rtol=3.0e-5, atol=5.0e-8)
