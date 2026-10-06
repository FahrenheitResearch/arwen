"""Exact native scorer parity against the replaced float64 operations."""

from pathlib import Path

import numpy as np
import pytest

from gpuwm import obs_score_bridge as bridge
from gpuwm.verify import field_metrics
from gpuwm.verify.obs import contingency, fss, stations


def _bits(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    assert left.shape == right.shape
    assert np.array_equal(left.view(np.uint64), right.view(np.uint64))


def _fraction(events, valid, radius, boundary):
    def smooth(values):
        if boundary == "edge" or radius == 0:
            return field_metrics.boxcar(values, 2 * radius + 1)
        padded = np.pad(values, ((radius, radius), (radius, radius)))
        return field_metrics.boxcar(padded, 2 * radius + 1)[
            radius:-radius, radius:-radius]
    counted = smooth((events & valid).astype(np.float64))
    denominator = smooth(valid.astype(np.float64))
    fraction = np.zeros_like(counted)
    np.divide(counted, denominator, out=fraction, where=denominator > 0)
    return fraction, denominator


def _threshold(field, valid, target):
    values = np.asarray(field, dtype=np.float64)[valid]
    if target <= 0:
        return float(np.nextafter(values.max(), np.inf))
    if target >= 1:
        return float(values.min())
    return float(np.quantile(values, 1.0 - target, method="linear"))


def _fss(model, obs, valid, scored, threshold, radius, boundary, frequency):
    oe = (obs >= threshold) & valid
    denominator = np.count_nonzero(valid & scored)
    observed_rate = np.count_nonzero(oe & scored) / denominator
    cutoff = _threshold(model, valid & scored, observed_rate) if frequency else threshold
    me = (model >= cutoff) & valid
    model_rate = np.count_nonzero(me & scored) / denominator
    mf, counts = _fraction(me, valid, radius, boundary)
    of, _ = _fraction(oe, valid, radius, boundary)
    region = scored & (counts > 0)
    difference = mf[region] - of[region]
    numerator = float(np.sum(difference * difference, dtype=np.float64))
    reference = float(np.sum(mf[region] ** 2 + of[region] ** 2,
                             dtype=np.float64))
    score = 1.0 if reference == 0 else 1.0 - numerator / reference
    return [min(1.0, max(0.0, score)), cutoff, threshold, observed_rate,
            model_rate, 0.5 + observed_rate / 2], np.count_nonzero(region)


@pytest.mark.parametrize("shape", [(1, 1), (1, 17), (17, 1), (9, 11),
                                  (131, 139), (513, 517)])
@pytest.mark.parametrize("boundary", ["zero", "edge"])
@pytest.mark.parametrize("radius", [0, 1, 6])
def test_fraction_and_fss_default_door_bits(shape, boundary, radius):
    rng = np.random.default_rng(731)
    model = rng.normal(25, 15, shape)
    obs = rng.normal(25, 15, shape)
    valid = rng.random(shape) > .2
    scored = rng.random(shape) > .1
    valid[0, 0] = scored[0, 0] = True
    expected = _fraction(model >= 30, valid, radius, boundary)
    actual = fss.masked_neighborhood_fraction(model >= 30, valid, radius,
                                            boundary=boundary)
    _bits(actual[0], expected[0])
    _bits(actual[1], expected[1])
    for frequency in [False, True]:
        expected, cells = _fss(model, obs, valid, scored, 30., radius,
                               boundary, frequency)
        result = fss.masked_fss(model, obs, valid=valid, score_mask=scored,
                                threshold=30., half_width=radius,
                                boundary=boundary,
                                frequency_matched=frequency)
        _bits([result.fss, result.threshold_model, result.threshold_obs,
               result.observed_base_rate, result.model_base_rate,
               result.fss_useful], expected)
        assert result.scored_cells == cells


@pytest.mark.parametrize("buffer", [32, 8192, 16384])
def test_reduction_bits_cross_buffer_and_pairwise_boundaries(buffer):
    original = np.getbufsize()
    np.setbufsize(buffer)
    try:
        rng = np.random.default_rng(519)
        for n in [1, 7, 8, 127, 128, 129, 8191, 8192, 8193, 40031]:
            values = rng.normal(size=n) * rng.choice([1.e-12, 1., 1.e12], n)
            actual = bridge.reduce(values)
            expected = [np.mean(values, dtype=np.float64),
                        np.sqrt(np.mean(values * values, dtype=np.float64)),
                        np.median(values)]
            _bits(actual, expected)
        shape = (171, 173)
        model, obs = rng.normal(size=(2, *shape))
        valid = scored = np.ones(shape, dtype=bool)
        expected, _ = _fss(model, obs, valid, scored, .5, 3, "zero", False)
        result = fss.masked_fss(model, obs, valid=valid, threshold=.5,
                                half_width=3)
        _bits(result.fss, expected[0])
    finally:
        np.setbufsize(original)


@pytest.mark.parametrize("values", [[-0.0], [0.0, -0.0],
                                    [-0.0, 0.0, -0.0, 0.0],
                                    [-np.inf, 0., np.inf],
                                    [np.nan, 1., 2.],
                                    [1., 1., 3., 3., 8.]])
def test_threshold_and_signed_zero_bits(values):
    field = np.asarray(values)[None]
    valid = np.ones(field.shape, dtype=bool)
    for target in [0., .1, .25, .5, .75, .9, 1.]:
        with np.errstate(invalid="ignore"):
            expected = _threshold(field, valid, target)
        actual = fss.frequency_matched_threshold(field, valid, target)
        _bits(actual, expected)
    if np.all(np.isfinite(field)):
        _bits(bridge.reduce(values), [np.mean(values),
                                     np.sqrt(np.mean(field * field)),
                                     np.median(values)])


@pytest.mark.parametrize("shape", [(1, 1), (1, 13), (13, 1), (13, 17)])
def test_station_interpolation_bits_and_ties(shape):
    rng = np.random.default_rng(121)
    field = rng.normal(size=shape)
    ny, nx = shape
    for x, y in [(0., 0.), (nx - 1., ny - 1.),
                 ((nx - 1.) / 2, (ny - 1.) / 2)]:
        position = stations.StationPosition("S", x, y)
        _bits(stations.sample_field(field, position, method="nearest"),
              field[int(round(y)), int(round(x))])
        i0 = min(int(np.floor(x)), nx - 2) if nx > 1 else 0
        j0 = min(int(np.floor(y)), ny - 2) if ny > 1 else 0
        i1 = min(i0 + 1, nx - 1)
        j1 = min(j0 + 1, ny - 1)
        tx, ty = x - i0, y - j0
        expected = (field[j0, i0] * (1 - tx) * (1 - ty)
                    + field[j0, i1] * tx * (1 - ty)
                    + field[j1, i0] * (1 - tx) * ty
                    + field[j1, i1] * tx * ty)
        _bits(stations.sample_field(field, position), expected)


def test_contingency_missing_nan_boundaries_and_none_bits():
    observed = np.array([[np.nan, -0., 30., np.inf], [31., 1., 50., -np.inf]])
    forecast = observed[:, ::-1]
    valid = np.array([[True, False, True, True], [True, True, True, True]])
    for threshold in [0., 30., np.inf, np.nan]:
        oe, fe = observed >= threshold, forecast >= threshold
        expected = [np.count_nonzero(oe & fe & valid),
                    np.count_nonzero(oe & ~fe & valid),
                    np.count_nonzero(~oe & fe & valid),
                    np.count_nonzero(~oe & ~fe & valid)]
        table = contingency.contingency_table(observed, forecast,
                                             valid=valid, threshold=threshold)
        assert [table.hits, table.misses, table.false_alarms,
                table.correct_negatives] == expected
        h, m, f, q = [int(value) for value in expected]
        total = h + m + f + q
        random = (h + m) * (h + f) / total
        ratio = lambda n, d: float(n) / float(d) if d else None
        expected = [ratio(h + m, total), ratio(h + f, total), ratio(h, h + m),
                    ratio(f, h + f), ratio(h, h + m + f), ratio(h + f, h + m),
                    ratio(h - random, h + m + f - random),
                    ratio(2. * (h * q - m * f),
                          (h + m) * (m + q) + (h + f) * (f + q))]
        actual = list(contingency.contingency_scores(table).values())[5:]
        for a, e in zip(actual, expected):
            if e is None:
                assert a is None
            else:
                _bits(a, e)


def test_real_mrms_remapped_field_bits():
    root = Path(__file__).parents[1] / "tools/rustwx/crates/obs-regrid/golden/cases"
    import json
    manifest = json.loads((root / "MANIFEST.json").read_text())
    case = next(row for row in manifest["cases"]
                if row["name"] == "real_nearest_obs_to_model")
    shape = tuple(case["destination_shape"])
    directory = root / case["name"]
    obs = np.fromfile(directory / "out_values.bin", dtype="<f8", offset=26).reshape(shape)
    valid = np.fromfile(directory / "out_valid.bin", dtype="u1", offset=26).reshape(shape).astype(bool)
    model = np.fromfile(root / "real_nearest_model_to_obs" / "values.bin",
                        dtype="<f8", offset=26).reshape(shape)
    scored = np.ones(shape, dtype=bool)
    for boundary in ["zero", "edge"]:
        for radius in [0, 3, 10]:
            expected, cells = _fss(model, obs, valid, scored, 30., radius,
                                   boundary, True)
            result = fss.masked_fss(model, obs, valid=valid, threshold=30.,
                                    half_width=radius, boundary=boundary,
                                    frequency_matched=True)
            _bits(result.fss, expected[0])
            assert result.scored_cells == cells


def test_missing_native_export_is_a_rebuild_refusal():
    with pytest.raises(bridge.ObsScoreBridgeError, match="missing.*rebuild"):
        bridge._export(object(), "gpuwm_obsscore_masked_fss", Path("old.so"))


def test_native_scoring_rejects_empty_masks_and_bad_shapes():
    field = np.zeros((3, 4))
    with pytest.raises(ValueError, match="no valid cells"):
        fss.masked_fss(field, field, valid=np.zeros(field.shape, bool),
                       threshold=1, half_width=1)
    with pytest.raises(ValueError, match="inside both"):
        fss.masked_fss(field, field, valid=np.ones(field.shape, bool),
                       score_mask=np.zeros(field.shape, bool),
                       threshold=1, half_width=1)
    with pytest.raises(ValueError, match="non-negative"):
        fss.masked_neighborhood_fraction(field, field, -1)
    with pytest.raises(ValueError, match="valid cell"):
        contingency.contingency_table(field, field, threshold=1,
                                      valid=np.zeros(field.shape, bool))


def test_report_matcher_exact_offsets_ties_missing_and_lexical_order():
    from gpuwm.verify.obs.contracts import (
        ObsProvenance, Station, StationObsSet, StationReport, parse_valid_time,
    )
    stamps = ["2024-05-21T11:50:00", "2024-05-21T12:10:00",
              "2024-05-21T11:59:59", "2024-05-21T12:00:01",
              "2024-05-22T00:00:00"]
    reports = tuple(StationReport("S", stamp, {"temperature_2m": 290.})
                    for stamp in reversed(stamps))
    observations = StationObsSet(
        stations=(Station("S", 37., -97., 300.),), reports=reports,
        provenance=ObsProvenance("TEST", "surface", "test://reports",
                                 "a" * 64, "2024-05-22T00:00:00"))
    targets = ["2024-05-21T12:00:00", "2024-05-21T11:40:00",
               "2024-05-21T12:20:00", "2024-05-22T00:10:00",
               "2024-05-22T00:10:01"]
    for tolerance in [1, 600, 3600]:
        expected = {}
        for sid, records in observations.by_station().items():
            for target in targets:
                best = None
                for report in records:
                    offset = abs((parse_valid_time(report.valid_time)
                                  - parse_valid_time(target)).total_seconds())
                    candidate = (offset, report.valid_time, report)
                    if offset <= tolerance and (best is None
                                                or candidate[:2] < best[:2]):
                        best = candidate
                if best:
                    expected[(sid, target)] = best[2]
        assert stations.match_reports(observations, targets,
                                      tolerance_seconds=tolerance) == expected


def test_surface_scores_records_exact_bits_and_missing_behavior():
    from datetime import datetime, timedelta
    from gpuwm.verify.obs.contracts import StationReport
    rng = np.random.default_rng(199)
    ids = tuple(f"S{index}" for index in range(9))
    times = [(datetime(2024, 1, 1) + timedelta(hours=i)).isoformat()
             for i in range(71)]
    frozen = stations.FrozenStationSet(ids, {}, (), {})
    matched = {(sid, time): StationReport(sid, time, {"temperature_2m": 285.})
               for sid in ids for time in times}
    model = {pair: 285. + float(rng.normal() * rng.choice([1e-7, 1., 10.]))
             for pair in matched}
    missing = (ids[0], times[3])
    model[missing] = None
    score = stations.surface_scores(
        frozen, matched=matched, valid_times=times,
        variables=["temperature_2m"],
        model_value=lambda sid, text, variable: model[(sid, text)])[
            "temperature_2m"]
    residuals, by_station, by_hour = [], {}, {}
    for sid in ids:
        for text in times:
            value = model[(sid, text)]
            if value is None:
                continue
            residual = value - 285.
            residuals.append(residual)
            by_station.setdefault(sid, []).append(residual)
            by_hour.setdefault(datetime.fromisoformat(text).hour, []).append(residual)
    rmse = lambda values: np.sqrt(np.mean(np.asarray(values) ** 2, dtype=np.float64))
    expected_station = {sid: rmse(values) for sid, values in by_station.items()}
    _bits([score.bias, score.rmse, score.median_station_rmse],
          [np.mean(residuals, dtype=np.float64), rmse(residuals),
           np.median(np.asarray(sorted(expected_station.values())))])
    for sid, value in expected_station.items():
        _bits(score.station_rmse[sid], value)
    for hour, values in by_hour.items():
        _bits(score.hourly_rmse[hour], rmse(values))
        _bits(score.hourly_bias[hour], np.mean(values, dtype=np.float64))
    assert score.sample_count == len(residuals)
    model[(ids[0], times[0])] = np.inf
    with pytest.raises(ValueError, match="residual.*non-finite"):
        stations.surface_scores(
            frozen, matched=matched, valid_times=times,
            variables=["temperature_2m"],
            model_value=lambda sid, text, variable: model[(sid, text)])
