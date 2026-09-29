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


# A high-terrain pressure-level column: five isobaric levels lie below the
# source surface, and the lowest target sits one FP32 ulp from that surface.
# The only target either surface neighbour can reach is that lowest one.
_HIGH_TERRAIN_SOURCE = (
    100000.0, 97500.0, 95000.0, 92500.0, 90000.0,
    87500.0, 85000.0, 80000.0, 70000.0, 50000.0, 30000.0, 20000.0)
_HIGH_TERRAIN_SURFACE = 89023.25
_HIGH_TERRAIN_UPPER_TARGETS = (84000.0, 80000.0, 70000.0, 50000.0, 30000.0,
                               25000.0)

# A low-terrain column (every source level above the surface) whose second
# target sits one FP32 ulp above the 950 hPa source level.  The 900 hPa
# sample's only candidate target is that one: its neighbours are 950 and
# 850 hPa and no other target lies between them.
_LOW_TERRAIN_SOURCE = (
    100000.0, 95000.0, 90000.0, 85000.0, 80000.0, 70000.0, 50000.0,
    30000.0, 20000.0)
_LOW_TERRAIN_SURFACE = 101300.0
_LOW_TERRAIN_TIED_LEVEL = 1
_LOW_TERRAIN_UPPER_TARGETS = (84000.0, 70000.0, 50000.0, 30000.0, 25000.0)


def _float64_kernel(source_pressure, surface_pressure, target_pressure):
    def replay(field):
        return np.asarray(np_wrf_real_vert_interp(
            field, np.zeros(surface_pressure.shape, dtype=np.float32),
            source_pressure, surface_pressure, target_pressure,
            interp_in_logp=True, extrap="constant",
            vboundb=target_pressure.shape[0] + 1), dtype=np.float32)

    return replay


def _rounding_kernel(target_index, tie_pressure):
    """A kernel whose FP32 logarithm puts one target on a tie pressure.

    One pressure ulp is about a tenth of a logarithm ulp here, so a logf
    within one ulp of exact may return the same value for both.  On the
    linear Q path a target enters only through its logarithm (and, for the
    lowest target, the force-level search, which picks the same level
    either way here), so this kernel's output is the float64 operator's
    with that target moved onto the tie pressure.
    """

    def factory(source_pressure, surface_pressure, target_pressure):
        rounded = np.array(target_pressure, dtype=np.float32, copy=True)
        rounded[target_index] = tie_pressure(source_pressure, surface_pressure)
        return _float64_kernel(source_pressure, surface_pressure, rounded)

    return factory


def _onto_surface(source_pressure, surface_pressure):
    return surface_pressure


def _onto_tied_level(source_pressure, surface_pressure):
    return source_pressure[_LOW_TERRAIN_TIED_LEVEL]


def _surface_tie_case(direction):
    surface = np.float32(_HIGH_TERRAIN_SURFACE)
    lowest = np.nextafter(surface, np.float32(direction))
    return (_HIGH_TERRAIN_SOURCE, float(surface),
            (float(lowest), *_HIGH_TERRAIN_UPPER_TARGETS))


def _level_tie_case():
    level = np.float32(_LOW_TERRAIN_SOURCE[_LOW_TERRAIN_TIED_LEVEL])
    tied = np.nextafter(level, np.float32(-np.inf))
    return (_LOW_TERRAIN_SOURCE, _LOW_TERRAIN_SURFACE,
            (99000.0, float(tied), *_LOW_TERRAIN_UPPER_TARGETS))


@pytest.mark.parametrize(
    ("column", "kernel", "source_level", "expected_class", "written"),
    (
        (_surface_tie_case(np.inf), _float64_kernel, 4,
         "TARGET_INFLUENCING", [0]),
        (_surface_tie_case(np.inf), _rounding_kernel(0, _onto_surface), 4,
         "WRF_BELOW_GROUND_OUTSIDE_TARGET_SUPPORT", []),
        (_surface_tie_case(-np.inf), _float64_kernel, 5,
         "TARGET_INFLUENCING", [0]),
        (_surface_tie_case(-np.inf), _rounding_kernel(0, _onto_surface), 5,
         "WRF_NO_TARGET_STENCIL", []),
        (_level_tie_case(), _float64_kernel, 2, "TARGET_INFLUENCING", [1]),
        (_level_tie_case(), _rounding_kernel(1, _onto_tied_level), 2,
         "WRF_NO_TARGET_STENCIL", []),
    ),
)
def test_a_cloudy_sample_on_a_logarithm_tie_follows_the_kernel_that_ran(
        column, kernel, source_level, expected_class, written):
    source_pressures, surface_pressure, target_pressures = column

    source, initialized, evidence, validation, labels = _case(
        source_pressures, surface_pressure, target_pressures, source_level,
        replay_factory=kernel)

    code = real._HRRR_DISPOSITION_CLASSES[expected_class]
    assert labels[source_level] == code
    records = evidence["species"]["QC"]["examples"][expected_class]["records"]
    assert len(records) == 1
    assert records[0]["influencing_target_indices"] == written
    assert np.flatnonzero(initialized[:, 0, 0]).tolist() == written
    assert validation["species"]["QC"]["strength"] == (
        "PROVEN" if written else "WRF_EXCLUDED")
    partition = evidence["geometry"]["pressure_partition"]
    assert partition["logarithm_tie_sample_count"] == 1
    assert partition["logarithm_tie_supported_count"] == int(bool(written))
    assert partition["disagreeing_sample_count"] == 0


# The lowest part of a high-terrain analysis column whose lowest target
# sits nine FP32 ulps above the surface pressure.  numpy's float32 log puts
# that target strictly inside the below-ground sample's interval and the C
# library's logf puts it on the surface, so the sample at 898.9 hPa is used
# by one and not the other.
_NINE_ULP_SOURCE = (
    99889.65, 97389.65, 94889.65, 92389.65, 89889.65, 87395.96, 84905.305,
    82414.01, 79922.4, 77431.09, 74940.49, 72451.02, 69962.305)
_NINE_ULP_SURFACE = 89023.25
_NINE_ULP_TARGETS = (
    89023.32, 88433.54, 87689.52, 86758.516, 85605.125, 84193.48, 82490.61,
    80470.96)


def _cpu_kernel_or_skip():
    try:
        return resolve_preprocess_backend("cpu", workers=1)
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"native CPU preprocess bridge is not built: {exc}")


def _cpu_kernel(backend):
    def factory(source_pressure, surface_pressure, target_pressure):
        plan = backend.prepare_wrf_vertical(
            source_pressure, surface_pressure, target_pressure)

        def replay(field):
            return plan.apply(
                np.asarray(field, dtype=np.float32),
                np.zeros(surface_pressure.shape, dtype=np.float32),
                interp_in_logp=True, extrap="constant",
                vboundb=target_pressure.shape[0] + 1)

        return replay

    return factory


def test_a_target_nine_ulps_from_the_surface_prepares_on_the_cpu_kernel():
    backend = _cpu_kernel_or_skip()

    source, initialized, evidence, validation, labels = _case(
        _NINE_ULP_SOURCE, _NINE_ULP_SURFACE, _NINE_ULP_TARGETS, 4,
        replay_factory=_cpu_kernel(backend))

    # Whichever way this machine's logf rounds, the label is what the kernel
    # did, and the geometry names the sample as a tie, not a disagreement.
    influencing = real._HRRR_DISPOSITION_CLASSES["TARGET_INFLUENCING"]
    assert labels[4] in (influencing, real._HRRR_DISPOSITION_CLASSES[
        "WRF_BELOW_GROUND_OUTSIDE_TARGET_SUPPORT"])
    assert (labels[4] == influencing) == bool(np.count_nonzero(initialized))
    partition = evidence["geometry"]["pressure_partition"]
    assert partition["logarithm_tie_sample_count"] == 1
    assert partition["disagreeing_sample_count"] == 0
    assert validation["species"]["QC"]["source_nonzero_count"] == 1


def test_the_production_cpu_kernel_settles_surface_ties_when_available():
    backend = _cpu_kernel_or_skip()

    # Columns whose lowest target sits zero to four ulps either side of the
    # surface, at several surface pressures; every source level is active.
    surfaces = np.float32([89023.25, 91312.0, 84410.5, 88250.75])
    offsets = np.arange(-4, 5)
    ny, nx = surfaces.size, offsets.size
    source_pressure = np.broadcast_to(
        np.asarray(_HIGH_TERRAIN_SOURCE, dtype=np.float32)[:, None, None],
        (len(_HIGH_TERRAIN_SOURCE), ny, nx)).copy()
    surface_pressure = np.broadcast_to(surfaces[:, None], (ny, nx)).copy()
    lowest = surface_pressure.copy()
    for j in range(ny):
        for i, offset in enumerate(offsets):
            for _ in range(abs(int(offset))):
                lowest[j, i] = np.nextafter(
                    lowest[j, i], np.float32(np.sign(offset) * np.inf))
    target_pressure = np.concatenate([
        lowest[None],
        np.broadcast_to(
            np.asarray(_HIGH_TERRAIN_UPPER_TARGETS,
                       dtype=np.float32)[:, None, None],
            (len(_HIGH_TERRAIN_UPPER_TARGETS), ny, nx))]).astype(np.float32)
    replay = _cpu_kernel(backend)(
        source_pressure, surface_pressure, target_pressure)

    source = np.ones(source_pressure.shape, dtype=np.float32)
    initialized = replay(source)
    evidence = real.build_hrrr_hydrometeor_vertical_disposition(
        {"QC": source}, np.arange(source.shape[0], dtype=np.int32),
        source_pressure, surface_pressure, target_pressure,
        {"QC": initialized}, operator_replay=replay)
    real.validate_hrrr_hydrometeor_vertical_disposition(
        {"QC": real.array_correspondence_fingerprint(source)},
        {"QC": real.array_correspondence_fingerprint(initialized)},
        evidence)
    # Every difference the kernel's logf can make here lies inside the band
    # the geometry reserves for it.
    partition = evidence["geometry"]["pressure_partition"]
    assert partition["disagreeing_sample_count"] == 0
    assert partition["logarithm_tie_sample_count"] > 0


def test_a_geometry_that_misses_support_is_counted_not_refused(monkeypatch):
    authority = real._hrrr_operator_geometry

    def falsely_exclude(*args, **kwargs):
        geometry = authority(*args, **kwargs)
        geometry["operator_class"][4, 0, 0] = np.uint8(0)
        return geometry

    monkeypatch.setattr(real, "_hrrr_operator_geometry", falsely_exclude)
    source, initialized, evidence, validation, labels = _case(
        _BASE_PRESSURES, 97000.0,
        (90000.0, 85000.0, 70000.0, 40000.0, 25000.0), 4)

    assert labels[4] == real._HRRR_DISPOSITION_CLASSES["TARGET_INFLUENCING"]
    assert validation["species"]["QC"]["strength"] == "PROVEN"
    partition = evidence["geometry"]["pressure_partition"]
    assert partition["disagreeing_sample_count"] == 1
    assert partition["production_only_count"] == 1
    assert partition["geometry_only_count"] == 0
    assert partition["disagreements"][0]["raw_source_index"] == [4, 0, 0]
    assert partition["disagreements"][0]["production_support"] is True


def test_a_geometry_that_invents_support_is_counted_not_refused(monkeypatch):
    authority = real._hrrr_operator_geometry

    def falsely_include(*args, **kwargs):
        geometry = authority(*args, **kwargs)
        geometry["operator_class"][2, 0, 0] = np.uint8(1)
        return geometry

    monkeypatch.setattr(real, "_hrrr_operator_geometry", falsely_include)
    source, initialized, evidence, validation, labels = _case(
        (110000.0, 90000.0, 80000.0, 70000.0, 60000.0, 30000.0, 20000.0),
        100000.0, (95000.0, 65000.0, 40000.0, 25000.0), 2)

    assert labels[2] == real._HRRR_DISPOSITION_CLASSES[
        "WRF_NO_TARGET_STENCIL"]
    assert validation["species"]["QC"]["strength"] == "WRF_EXCLUDED"
    partition = evidence["geometry"]["pressure_partition"]
    assert partition["disagreeing_sample_count"] == 1
    assert partition["geometry_only_count"] == 1
    assert partition["disagreements"][0]["geometry_support"] is True


def test_replays_that_stop_composing_are_recorded_not_refused():
    source_pressure = np.asarray(
        _BASE_PRESSURES, dtype=np.float32)[:, None, None]
    surface_pressure = np.asarray([[97000.0]], dtype=np.float32)
    target_pressure = np.asarray(
        [90000.0, 85000.0, 70000.0, 40000.0, 25000.0],
        dtype=np.float32)[:, None, None]
    source = np.zeros(source_pressure.shape, dtype=np.float32)
    source[0] = np.float32(1.0)
    source[4] = np.float32(1.0)
    operator = _float64_kernel(
        source_pressure, surface_pressure, target_pressure)

    def leaky(field):
        # Linear for one sample at a time, not for two together.
        value = operator(field)
        if np.count_nonzero(field) > 1:
            value = value.copy()
            value[-1] += np.float32(1.0e-30)
        return value

    initialized = operator(source)
    evidence = real.build_hrrr_hydrometeor_vertical_disposition(
        {"QC": source}, np.arange(source.shape[0], dtype=np.int32),
        source_pressure, surface_pressure, target_pressure,
        {"QC": initialized}, operator_replay=leaky)
    validation = real.validate_hrrr_hydrometeor_vertical_disposition(
        {"QC": real.array_correspondence_fingerprint(source)},
        {"QC": real.array_correspondence_fingerprint(initialized)},
        evidence)

    replay = evidence["species"]["QC"]["operator_replay"]
    assert replay["source_target_outputs_byte_equal"] is False
    assert replay["excluded_output_all_exact_zero"] is True
    assert validation["species"]["QC"]["strength"] == "PARTIALLY_WRF_EXCLUDED"


def test_a_receipt_written_before_the_cross_check_still_validates():
    source, initialized, evidence, validation, _labels = _case(
        _BASE_PRESSURES, 97000.0,
        (90000.0, 85000.0, 70000.0, 40000.0, 25000.0), 4)
    older = copy.deepcopy(evidence)
    older["geometry"].pop("pressure_partition")
    unsigned = dict(older)
    unsigned.pop("evidence_sha256")
    older["evidence_sha256"] = real._canonical_receipt_sha256(unsigned)

    assert real.validate_hrrr_hydrometeor_vertical_disposition(
        {"QC": real.array_correspondence_fingerprint(source)},
        {"QC": real.array_correspondence_fingerprint(initialized)},
        older)["species"] == validation["species"]


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
