"""The hrrr route's root static carries the terrain-drag statistics.

Breakage it prevents (QA R4 on the 2.9.0 candidate): ``gpuwm domain
--recipe hrrr_v4_gsd41 --source hrrr`` emits gwd_opt = 3 (the GSL drag
suite).  The hrrr route's static builder never asked for the large- and
small-scale orographic statistics that option reads, so ``gpuwm go``
prepared for about three minutes and then refused at the forecast seam
(``prepared_terrain_drag_missing: configured terrain drag requires VARLS,
CONLS, ...``), after check, fetch-geog and go --dry-run had all passed.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from gpuwm.static.orographic import GSL_LS_FIELDS, GSL_SS_FIELDS

GSL = GSL_LS_FIELDS + GSL_SS_FIELDS


def _run_builder(tmp_path, monkeypatch, *extra):
    from test_hrrr_native_static import _fixture
    from tools import hrrr_build_native_static as producer
    from gpuwm.static import build
    tmp_path.mkdir(parents=True, exist_ok=True)
    target, cache, receipt = _fixture(tmp_path)
    domain = tmp_path / "domain.json"
    domain.write_text(json.dumps(target.to_payload()), encoding="utf-8")
    with np.load(cache) as stored:
        arrays = {name: stored[name] for name in stored.files}
    geog = tmp_path / "geog"
    for directory in build._DEFAULT_GEOG_DIRS.values():
        (geog / directory).mkdir(parents=True, exist_ok=True)
        (geog / directory / "index").write_text("type=continuous\n")
    selections = []

    def fake_build(grid, root, *, selection, source_coverage_report):
        selections.append(selection)
        fields = dict(arrays)
        for name in selection.orographic:
            fields[name] = np.zeros((target.ny, target.nx))
        return fields
    monkeypatch.setattr(producer, "build_static", fake_build)
    out, sealed = tmp_path / "out.npz", tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", [
        "native-static", "--domain-spec", str(domain),
        "--output", str(out), "--receipt", str(sealed), *extra,
        *([] if "--static-cache" in extra else ["--geog-root", str(geog)])])
    producer.main()
    return selections, json.loads(sealed.read_text(encoding="utf-8")), (
        cache, receipt)


def test_the_hrrr_root_static_builds_the_gsl_drag_statistics(tmp_path, monkeypatch):
    (selection,), receipt, _ = _run_builder(
        tmp_path, monkeypatch, "--orographic", ",".join(GSL))
    assert set(selection.orographic) == set(GSL)
    assert set(GSL) <= set(receipt["array_sha256"])


def test_a_reused_cache_without_the_drag_statistics_is_refused_by_name(
        tmp_path, monkeypatch):
    from test_hrrr_native_static import _fixture
    target, cache, receipt = _fixture(tmp_path)
    domain = tmp_path / "domain.json"
    domain.write_text(json.dumps(target.to_payload()), encoding="utf-8")
    from tools import hrrr_build_native_static as producer
    monkeypatch.setattr(sys, "argv", [
        "native-static", "--domain-spec", str(domain),
        "--static-cache", str(cache), "--static-receipt", str(receipt),
        "--output", str(tmp_path / "o.npz"), "--receipt", str(tmp_path / "o.json"),
        "--orographic", ",".join(GSL)])
    with pytest.raises(ValueError, match="prepared_terrain_drag_missing: .*VARLS"):
        producer.main()


def test_the_strict_verifier_accepts_drag_statistics_with_their_coverage(tmp_path):
    from test_hrrr_native_static import _fixture
    from gpuwm.hrrr_native_static import _array_sha256, verify_hrrr_native_static
    target, cache, receipt_path = _fixture(tmp_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    with np.load(cache) as stored:
        fields = {name: stored[name] for name in stored.files}
    fields["VARLS"] = np.zeros((target.ny, target.nx), dtype=np.float64)
    np.savez(cache, **fields)
    from gpuwm.hrrr_native_static import sha256_file
    receipt["cache"].update(bytes=cache.stat().st_size, sha256=sha256_file(cache))
    receipt["array_sha256"]["VARLS"] = _array_sha256(fields["VARLS"])
    record = dict(receipt["geog_source_coverage"]["terrain"], field="VARLS")
    receipt["geog_source_coverage"]["orographic:VARLS"] = record
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    loaded, _ = verify_hrrr_native_static(cache, receipt_path, target)
    assert "VARLS" in loaded
    # A drag array without its coverage record is still refused.
    del receipt["geog_source_coverage"]["orographic:VARLS"]
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="source-coverage fields mismatch"):
        verify_hrrr_native_static(cache, receipt_path, target)
