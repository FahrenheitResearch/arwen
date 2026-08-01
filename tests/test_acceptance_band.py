"""The acceptance band as data: schema, coverage, derivation, provenance.

No test here asserts a metric value, and none asserts a distribution of the
coverage receipt.  A band widened until it swallowed the published table would
satisfy an all-inside assertion and prove nothing; the receipt is reported as
an observation and the report of record carries the tally.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

import certification_fixtures as fixtures
from gpuwm.certify.band import (BAND_SCHEMA_ID, BAND_SCHEMA_PATH,
                                CLASSIFICATIONS, MARGIN_RULE_PATH,
                                PROVENANCES, BandError, band_path_for_config,
                                extract_anchor_table, load_band,
                                load_margin_rule, place_value, validate_band)
from gpuwm.certify.verdict import certify

REPO_ROOT = fixtures.REPO_ROOT
ANCHOR_DOCUMENT = REPO_ROOT / "docs" / "public" / "VERIFICATION.md"


def _shipped():
    paths = fixtures.shipped_band_paths()
    assert paths, "this repository ships no acceptance band"
    return paths


# --------------------------------------------------------------------------
# AC7 -- a plain JSON reader can check the band, and every column is covered
# --------------------------------------------------------------------------

@pytest.mark.parametrize("band_path", _shipped(), ids=lambda p: p.stem[:12])
def test_the_band_validates_with_no_gpuwm_import(band_path, tmp_path):
    """An independent verifier reads the band with json + jsonschema only."""
    script = tmp_path / "independent_verifier.py"
    script.write_text(
        "import json, sys\n"
        "import jsonschema\n"
        "band = json.loads(open(sys.argv[1], encoding='utf-8').read())\n"
        "schema = json.loads(open(sys.argv[2], encoding='utf-8').read())\n"
        "jsonschema.validate(instance=band, schema=schema)\n"
        "assert 'gpuwm' not in sys.modules, sorted(sys.modules)\n"
        "print(band['schema'], band['provenance'],"
        " band['band_schema_version'])\n",
        encoding="utf-8")
    import subprocess

    completed = subprocess.run(
        [sys.executable, str(script), str(band_path), str(BAND_SCHEMA_PATH)],
        capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert BAND_SCHEMA_ID in completed.stdout


@pytest.mark.parametrize("band_path", _shipped(), ids=lambda p: p.stem[:12])
def test_metric_coverage_has_exactly_one_entry_per_comparator_column(
        band_path):
    columns = fixtures.metric_columns()
    band = json.loads(band_path.read_text(encoding="utf-8"))
    assert set(band["metric_coverage"]) == set(columns)
    assert len(band["metric_coverage"]) == len(columns)
    for name, entry in band["metric_coverage"].items():
        assert entry["classification"] in CLASSIFICATIONS, name


def test_the_comparator_declares_the_column_split_the_coverage_map_assumes():
    """Fifteen metric columns of eighteen; the other three key the row."""
    from gpuwm.certify.band import ROW_KEY_COLUMNS

    source = (REPO_ROOT / "tools"
              / "matched_wrfout_stream_compare.py").read_text(encoding="utf-8")
    columns = fixtures.metric_columns()
    assert len(columns) == 15
    assert len(ROW_KEY_COLUMNS) + len(columns) == 18
    # The row keys this package assumes are the row keys the comparator
    # writes, taken from the comparator's own declaration.
    assert f'_COLUMNS = {ROW_KEY_COLUMNS!r} + _METRIC_COLUMNS'.replace(
        "'", '"') in source.replace("'", '"')
    assert not set(ROW_KEY_COLUMNS) & set(columns)


def test_an_unrecognised_metrics_column_is_refused_by_name(tmp_path, capsys):
    paths = fixtures.matched_set(tmp_path,
                                 extra_columns={"brightness_temp_mae": "1.0"})
    from gpuwm.cli import main

    assert main(["certify",
                 "--run-capsule", str(paths["capsule"]),
                 "--metrics-csv", str(paths["metrics"]),
                 "--band", str(paths["band"]),
                 "--wrf-reference-manifest",
                 str(paths["wrf_reference"])]) != 0
    assert "brightness_temp_mae" in capsys.readouterr().err


def test_a_band_that_gates_a_column_it_does_not_classify_is_refused():
    band = fixtures.shipped_band()
    banded = next(name for name, entry in band["metric_coverage"].items()
                  if entry["classification"] == "banded")
    del band["metric_coverage"][banded]
    with pytest.raises(BandError):
        validate_band(band)


# --------------------------------------------------------------------------
# AC8 -- the band is the derivation rule's output, byte for byte
# --------------------------------------------------------------------------

@pytest.mark.parametrize("band_path", _shipped(), ids=lambda p: p.stem[:12])
def test_re_running_the_derivation_reproduces_the_committed_band(band_path,
                                                                 tmp_path):
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    try:
        import derive_acceptance_band as derivation
    finally:
        sys.path.pop(0)
    committed = json.loads(band_path.read_text(encoding="utf-8"))
    config_path = fixtures.config_path_for(committed["config_sha256"])
    rebuilt, receipt, band_text, receipt_text = derivation.derive(
        config_path, anchor_document=ANCHOR_DOCUMENT, band_dir=tmp_path)
    assert rebuilt.name == band_path.name
    assert band_text == band_path.read_text(encoding="utf-8")
    committed_receipt = band_path.with_name(
        band_path.stem + derivation.COVERAGE_SUFFIX)
    assert receipt.name == committed_receipt.name
    assert receipt_text == committed_receipt.read_text(encoding="utf-8")


@pytest.mark.parametrize("band_path", _shipped(), ids=lambda p: p.stem[:12])
def test_the_coverage_receipt_covers_every_published_triple(band_path):
    receipt_path = band_path.with_name(band_path.stem + ".coverage.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    _, header, rows = extract_anchor_table(
        ANCHOR_DOCUMENT.read_text(encoding="utf-8"),
        load_margin_rule()["anchor"]["section_heading"])
    published_metrics = len(header) - 2
    assert receipt["triple_count"] == len(rows) * published_metrics
    assert len(receipt["cells"]) == receipt["triple_count"]
    assert sum(receipt["tally"].values()) == receipt["triple_count"]
    assert set(receipt["tally"]) == {"inside", "outside", "nan"}
    # The receipt is an observation.  Its distribution is reported, never
    # asserted: a band widened to swallow the table would satisfy such an
    # assertion and prove nothing.
    assert {cell["placement"] for cell in receipt["cells"]} <= {
        "inside", "outside", "nan"}


def test_the_anchor_table_hash_moves_when_the_published_table_moves():
    """The band goes stale the moment its anchor is edited, and says so."""
    band = fixtures.shipped_band()
    text = ANCHOR_DOCUMENT.read_text(encoding="utf-8")
    heading = load_margin_rule()["anchor"]["section_heading"]
    table_text, _, _ = extract_anchor_table(text, heading)
    import hashlib

    assert band["derivation"]["anchor_table_sha256"] == hashlib.sha256(
        table_text.encode("utf-8")).hexdigest()
    edited = table_text.replace("| d01 |", "| d09 |", 1)
    assert hashlib.sha256(edited.encode("utf-8")).hexdigest() != (
        band["derivation"]["anchor_table_sha256"])


def test_the_derivation_rule_is_the_committed_one():
    band = fixtures.shipped_band()
    import hashlib

    assert band["derivation"]["margin_rule_sha256"] == hashlib.sha256(
        MARGIN_RULE_PATH.read_bytes()).hexdigest()
    rule = load_margin_rule()
    assert band["derivation"]["rule_id"] == rule["rule_id"]
    assert band["derivation"]["rule_version"] == rule["rule_version"]


def test_the_interval_endpoints_are_the_rule_applied_to_the_anchor():
    """Recompute one family's endpoints independently of the derivation."""
    band = fixtures.shipped_band()
    rule = load_margin_rule()
    checked = 0
    for domain, leads in band["intervals"].items():
        for lead, cells in leads.items():
            for column, interval in cells.items():
                if interval["nan_expected"]:
                    continue
                family = rule["families"][
                    rule["column_classification"][column]["family"]]
                anchor = interval["anchor"]
                margin = max(family["absolute_floor"],
                             family["relative_fraction"] * abs(anchor))
                lower = anchor - margin
                upper = anchor + margin
                if family["clamp_low"] is not None:
                    lower = max(lower, family["clamp_low"])
                if family["clamp_high"] is not None:
                    upper = min(upper, family["clamp_high"])
                assert interval["lower"] == pytest.approx(lower, abs=1e-9)
                assert interval["upper"] == pytest.approx(upper, abs=1e-9)
                assert place_value(anchor, interval) == "inside"
                checked += 1
    assert checked > 100, checked


def test_a_nan_expected_cell_wants_a_non_number():
    interval = {"lower": None, "upper": None, "nan_expected": True}
    assert place_value(None, interval) == "nan"
    assert place_value(float("nan"), interval) == "nan"
    assert place_value(0.0, interval) == "outside"


# --------------------------------------------------------------------------
# AC9 -- provenance, band schema version, and the internal-scope contract
# --------------------------------------------------------------------------

@pytest.mark.parametrize("band_path", _shipped(), ids=lambda p: p.stem[:12])
def test_the_band_declares_a_provenance_and_a_schema_version(band_path):
    band = json.loads(band_path.read_text(encoding="utf-8"))
    assert band["provenance"] in PROVENANCES
    assert re.fullmatch(r"\d+\.\d+\.\d+", band["band_schema_version"])


def test_the_verdict_records_the_provenance_and_version_it_certified_against(
        tmp_path):
    paths = fixtures.matched_set(tmp_path)
    verdict = certify(capsule_path=paths["capsule"],
                      metrics_csv=paths["metrics"],
                      band_path=paths["band"],
                      wrf_reference_manifest=paths["wrf_reference"])
    band = fixtures.shipped_band()
    assert verdict["band"]["provenance"] == band["provenance"]
    assert verdict["band"]["band_schema_version"] == (
        band["band_schema_version"])
    assert verdict["band"]["provenance"] in PROVENANCES


def test_an_envelope_band_must_carry_its_internally_scoped_ensemble_block():
    band = fixtures.shipped_band()
    band["provenance"] = "wrf-ensemble-envelope"
    with pytest.raises(BandError):
        validate_band(band)
    band["ensemble"] = {
        "member_count": 2,
        "member_config_digests": ["a" * 64, "b" * 64],
        "pair_score_artifact_digest": "c" * 64,
        "interval_statistic": "fixture-statistic",
        "scope": "internal",
    }
    validate_band(band)
    band["ensemble"]["scope"] = "public"
    with pytest.raises(BandError):
        validate_band(band)


_RETRIEVAL_VERBS = re.compile(
    r"\b(download|downloadable|obtainable|retrievable|available on request|"
    r"request a copy|published at|hosted at|mirror of)\b", re.IGNORECASE)

_INTERNAL_SCOPE_TOKENS = ("wrf-ensemble-envelope", "ensemble member",
                          "pair-score", "per-member config digest")


def test_no_public_document_says_the_internal_artifacts_can_be_fetched():
    offenders = []
    for path in sorted((REPO_ROOT / "docs" / "public").rglob("*.md")):
        for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1):
            lowered = line.lower()
            if not any(token in lowered for token in _INTERNAL_SCOPE_TOKENS):
                continue
            if _RETRIEVAL_VERBS.search(line):
                offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert not offenders, offenders


def test_the_band_lives_where_the_decision_put_it():
    band = fixtures.shipped_band()
    expected = band_path_for_config(band["config_sha256"])
    assert expected.exists()
    assert expected.parent == fixtures.BAND_DIR
    assert load_band(expected)["config_sha256"] == band["config_sha256"]
    # Identity, not a case name: the file name is the config digest.
    assert expected.stem == band["config_sha256"]


# A WRF-format timestamp: a date, one separator, a wall clock.  Anchored at
# both ends, so it matches a string that *is* a timestamp and never one that
# merely contains a year.
_WRF_TIMESTAMP = r"\d{4}-\d{2}-\d{2}[_ T]\d{2}[:_]\d{2}[:_]\d{2}"
_WRF_TIMESTAMP_RE = re.compile(rf"^{_WRF_TIMESTAMP}$")

# A dated frame filename: one of WRF's own stream prefixes, an optional
# domain tag, a timestamp, and nothing else.  Every character class here is
# closed -- a stream name from the list, ``d`` and two digits, digits and
# separators -- so the only token such a name can carry is the date of the
# frame it names.  ``wrfout_d01_1974-04-03_12_00_00`` and
# ``met_em.d02.1999-05-03_18:00:00.nc`` are both this shape.
_FRAME_FILENAME_RE = re.compile(
    r"^(wrfout|wrfinput|wrfbdy|wrfrst|wrflowinp|wrffdda|wrfchemi|met_em)"
    rf"([._]d\d{{2}})?[._]{_WRF_TIMESTAMP}(\.nc)?$")


def _is_measurement_provenance(value: str) -> bool:
    """True when a JSON *value* records which frame, or which instant, was
    scored -- rather than naming a case.

    A receipt exists to say what it measured.  The valid time of the frames
    it scored, and the names of those frames, are that provenance: the
    inventory has to carry the real spelling or a later statement about what
    was scored is not checkable against it.  A forecast valid time of
    1974-04-03 is what the run of record's frames *are*.

    The distinction is structural, not a list of blessed strings.  Both
    grammars are anchored and closed, so a receipt written next year passes
    with no edit here, and nothing that is not a timestamp or a frame name
    can reach the exemption by containing one -- ``the 1974-04-03 case`` and
    ``configs/real74_case.toml`` are neither shape and stay scanned.
    """
    return bool(_WRF_TIMESTAMP_RE.match(value)
                or _FRAME_FILENAME_RE.match(value))


def _scannable_text(path: Path) -> list[str]:
    """What a case name could hide in, for one file.

    A measurement is never a case name.  In the two formats whose payload is
    numeric, the scan reads keys and string values -- parsed, not grepped --
    because a random float's digits can spell a year by coincidence, while a
    case name in JSON or CSV is always an identifier or a string.  Everywhere
    else, code and schemas and prose, the whole file is scanned.

    Parsing also gives the scan a *position*, and position is what separates
    the two ways a campaign date can appear in a certification receipt.  In a
    value it is measurement provenance and is exempt when it matches the
    timestamp or frame-name grammar above.  In a key it is not: a key is a
    metric name, a group name, a schema field -- generic vocabulary -- and a
    campaign spelling there means the measurement itself was written for one
    case, which is the specialization this zone watches for.  Keys are
    therefore always scanned, including a key spelled exactly like a frame
    name.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".json":
        chunks: list[str] = []

        def walk(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    chunks.append(str(key))
                    walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)
            elif isinstance(node, str):
                if not _is_measurement_provenance(node):
                    chunks.append(node)

        walk(json.loads(text))
        return chunks
    if path.suffix == ".csv":
        import csv as csv_module
        import io

        cells: list[str] = []
        for row in csv_module.reader(io.StringIO(text)):
            for cell in row:
                try:
                    float(cell)
                except ValueError:
                    cells.append(cell)
        return cells
    return text.splitlines()


@pytest.mark.parametrize("token", ["real74", "1974", "ohio", "hrrr"])
def test_no_case_token_reaches_the_certification_data(token):
    roots = [REPO_ROOT / "gpuwm" / "certify",
             REPO_ROOT / "gpuwm" / "data" / "certification",
             REPO_ROOT / "docs" / "public" / "wrf-reference"]
    offenders = []
    scanned = 0
    for root in roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix == ".pyc":
                continue
            scanned += 1
            for chunk in _scannable_text(path):
                if token in chunk.lower():
                    offenders.append(f"{path}: {chunk[:120]}")
    assert scanned > 5, "the case-token scan found almost nothing to scan"
    assert not offenders, offenders


def test_the_case_token_scan_can_still_see_a_case_name_in_a_data_file(
        tmp_path):
    """Parsing must not blind the scan to a name a data file really carries."""
    payload = tmp_path / "carrier.json"
    payload.write_text(json.dumps({
        "source": "configs/real74_case.toml",
        "note": "the 1974-04-03 case",
        "value": -0.11974,
    }), encoding="utf-8")
    chunks = " ".join(_scannable_text(payload)).lower()
    assert "real74" in chunks
    assert "1974-04-03" in chunks

    sheet = tmp_path / "carrier.csv"
    sheet.write_text(
        "domain,label\nd01,ohio-run\nd02,-0.11974\n", encoding="utf-8")
    cells = " ".join(_scannable_text(sheet)).lower()
    assert "ohio" in cells
    assert "11974" not in cells


def test_a_receipt_may_record_the_instant_and_the_frames_it_scored(tmp_path):
    """Control, passing side: measurement provenance in a value position.

    This is the shape the published t=0 digest carries -- a valid time per
    domain and an input inventory of the frames that were staged and hashed.
    The second input is a different campaign in a different year and a
    different stream: the rule reads the grammar, not the digits, so a
    receipt written next year needs no edit here.
    """
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({
        "domains": {
            "d01": {"valid_time": "1974-04-03_12_00_00", "max_ulp": 0},
            "d02": {"valid_time": "1974-04-03_12_00_00", "max_ulp": 0},
        },
        "inputs": [
            {"name": "wrfout_d01_1974-04-03_12_00_00",
             "valid_time": "1974-04-03_12_00_00"},
            {"name": "met_em.d02.1999-05-03_18:00:00.nc",
             "valid_time": "1999-05-03_18:00:00"},
        ],
    }), encoding="utf-8")
    chunks = _scannable_text(receipt)

    for token in ["real74", "1974", "ohio", "hrrr"]:
        assert not [c for c in chunks if token in c.lower()], (token, chunks)
    # Structural, not a whitelist: the 1999 frame is exempt for the same
    # reason the 1974 one is, and neither year is written down anywhere.
    assert not [c for c in chunks if "1999" in c], chunks
    # The exemption is narrow -- it removed the timestamps and the two frame
    # names, and left every generic key in place to be scanned.
    assert set(chunks) == {"domains", "d01", "d02", "valid_time", "max_ulp",
                           "inputs", "name"}


def test_a_case_named_key_in_a_receipt_is_still_a_violation(tmp_path):
    """Control, failing side: the same file, specialized in a key.

    Watched firing: each of these fixtures puts a case token somewhere the
    value-position exemption does not reach, and each is reported.  A metric
    keyed on one campaign, a group named after one case, or a frame-shaped
    *key* would all mean the certified measurement was written for one run.
    """
    named_metric = tmp_path / "named_metric.json"
    named_metric.write_text(json.dumps({
        "domains": {"d01": {"real74_max_ulp": 0,
                            "valid_time": "1974-04-03_12_00_00"}},
    }), encoding="utf-8")
    assert [c for c in _scannable_text(named_metric)
            if "real74" in c.lower()] == ["real74_max_ulp"]

    named_group = tmp_path / "named_group.json"
    named_group.write_text(json.dumps({
        "covered_groups": ["moisture", "hrrr_surface"],
        "inputs": [{"name": "wrfout_d01_1974-04-03_12_00_00"}],
    }), encoding="utf-8")
    assert [c for c in _scannable_text(named_group)
            if "hrrr" in c.lower()] == ["hrrr_surface"]

    # A key spelled exactly like a frame name is still a key, and a schema
    # keyed on one campaign's frames is exactly the specialization the value
    # exemption must not launder.
    frame_shaped_key = tmp_path / "frame_shaped_key.json"
    frame_shaped_key.write_text(json.dumps({
        "per_frame": {"wrfout_d01_1974-04-03_12_00_00": {"max_ulp": 0}},
    }), encoding="utf-8")
    assert [c for c in _scannable_text(frame_shaped_key)
            if "1974" in c] == ["wrfout_d01_1974-04-03_12_00_00"]

    # A value that merely CONTAINS a timestamp is not one.  Both grammars are
    # anchored at both ends precisely so a case name cannot be laundered by
    # pinning a measured instant to the end of it.
    laundered = tmp_path / "laundered.json"
    laundered.write_text(json.dumps({
        "inputs": [{"name": "real74 staged at 1974-04-03_12_00_00",
                    "note": "ohio wrfout_d01_1974-04-03_12_00_00"}],
    }), encoding="utf-8")
    chunks = _scannable_text(laundered)
    assert [c for c in chunks if "real74" in c.lower()] == [
        "real74 staged at 1974-04-03_12_00_00"]
    assert [c for c in chunks if "ohio" in c.lower()] == [
        "ohio wrfout_d01_1974-04-03_12_00_00"]

    # The stream-name list is closed for the same reason: a frame name is
    # exempt because a WRF stream wrote it, not because it ends in a date.
    foreign_stream = tmp_path / "foreign_stream.json"
    foreign_stream.write_text(json.dumps({
        "inputs": [{"name": "real74_d01_1974-04-03_12_00_00"}],
    }), encoding="utf-8")
    assert [c for c in _scannable_text(foreign_stream)
            if "real74" in c.lower()] == ["real74_d01_1974-04-03_12_00_00"]
