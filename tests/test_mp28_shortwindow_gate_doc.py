"""Every number ``mp28-shortwindow-gate.md`` publishes is read back.

Same contract as ``test_mp28_matched_trajectory_doc.py``, for the successor
gate: the page may say MORE than the receipts, but every number it does
publish must be the receipt's, and the receipt directory may hold nothing
the gates below do not read.  The first pass caught four hand-typed
divergences this way; this file exists so the repeat cannot grow its own.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "public" / "validation" / "mp28-shortwindow-gate.md"
FIRST = ROOT / "docs" / "public" / "validation" / "mp28-matched-trajectory.md"
RECEIPTS = ROOT / "docs" / "public" / "receipts" / "mp28-shortwindow-gate"
GATE = RECEIPTS / "shortwindow-gate.json"

GATE_FIELDS = ("U", "V", "W", "PH", "MU", "T",
               "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP",
               "QNRAIN", "QNICE")


def _doc() -> str:
    return DOC.read_text(encoding="utf-8")


def _gate() -> dict:
    return json.loads(GATE.read_text(encoding="utf-8"))


# --- the receipt is real and is the one the design declared ----------------

def test_the_receipt_exists_and_is_the_declared_design():
    g = _gate()
    assert g["declared_in"] == \
        "docs/public/validation/mp28-shortwindow-gate.md"
    assert g["start_s"] == 1800.0 and g["frame_dt_s"] == 60.0
    assert tuple(g["gate_fields"]) == GATE_FIELDS
    assert g["g1_ratio"] == 3.0 and g["g0_tol"] == 1.0e-8

    run = json.loads((RECEIPTS / "sw-arwen-mp28-run.json").read_text())
    assert run["model"] == "gpuwm" and run["mp_physics"] == 28
    assert (run["nx"], run["ny"], run["nz"]) == (120, 120, 40)
    assert run["steps"] == 50 and run["frame_steps"] == 5
    assert "5090" in run["device"]


def test_the_declared_verdict_is_the_receipts_verdict():
    g = _gate()
    assert g["verdict"] == {"G0": True, "G1": True, "G2": True, "G3": True,
                            "control_diagnostic": False,
                            "outcome": "inconclusive"}
    page = _doc()
    assert "**Outcome: INCONCLUSIVE, by the rule in §5.**" in page, (
        "the receipt's outcome is inconclusive; the page must publish "
        "exactly that and may not soften it")
    # An inconclusive outcome moves nothing.
    assert "implemented-unverified" in page
    registry = json.loads((ROOT / "gpuwm" / "physics_registry_v2.json")
                          .read_text(encoding="utf-8"))
    option = registry["components"]["microphysics"]["options"][
        "thompson-aerosol-mp28"]
    assert option["maturity"] == "implemented-unverified"


def test_the_g1_counts_and_worst_rows_are_the_receipts():
    g, page = _gate(), _doc()

    rows = g["G1"]["rows"]
    assert g["G1"]["n_rows"] == 140 and len(rows) == 140
    failing = [r for r in rows if not r["pass"]]
    assert len(failing) == g["G1"]["n_fail"] == 0
    ranked = sorted(rows, key=lambda r: -r["ratio"])
    assert f"worst **{ranked[0]['ratio']:.3f}**" in page
    assert ranked[0]["field"] == "QNRAIN" and ranked[0]["step"] == 10
    assert f"runner-up **{ranked[1]['ratio']:.3f}**" in page
    assert ranked[1]["field"] == "QNRAIN" and ranked[1]["step"] == 2
    assert f"0 of {len(rows)} rows over" in page

    c = g["G1_control"]
    c_failing = [r for r in c["rows"] if not r["pass"]]
    assert c["n_rows"] == 140 and len(c_failing) == c["n_fail"] == 8
    assert c["pass"] is False
    assert f"8 of 140 rows over, worst **{c['worst']:.3f}**" in page
    worst = max(c["rows"], key=lambda r: r["ratio"])
    assert worst["field"] == "QRAIN" and worst["step"] == 2

    # Every failing control row the page names, with its ratio.
    for r in c_failing:
        assert f"{r['ratio']:.3f}" in page, (
            f"control failure {r['field']} step {r['step']} at "
            f"{r['ratio']:.3f} is not published")


def test_the_g0_screen_is_exactly_zero_as_published():
    g = _gate()
    assert g["G0"]["pass"] is True
    assert g["G0"]["nonzero_or_failing"] == [], (
        "the page says all 28 step-0 rows are exactly 0.0; the receipt "
        "records a nonzero one")
    assert "28/28 rows exactly 0.0" in _doc()


def test_the_dual_run_screen_and_g3_are_as_published():
    g = _gate()
    for config in ("mp08", "mp28"):
        screen = g["dual_run"][config]
        assert screen["status"] == "ok"
        assert screen["identical"] is True and screen["differing"] == []
        assert screen["frames_compared"] == 11
    assert g["G3"]["pass"] is True
    assert g["G3"]["nonfinite"] == [] and g["G3"]["bound_violations"] == []
    assert "11/11 + 11/11" in _doc()


def test_the_measurement_table_is_the_receipts_numbers():
    """Every cell of §9's table, both schemes, all five printed frames."""
    g, page = _gate(), _doc()
    frames = {m: {r["step"]: r for r in g["tested"][m]["frames"]}
              for m in ("08", "28")}
    cols = ("U", "V", "W", "PH", "MU", "T",
            "QVAPOR", "QCLOUD", "QNWFA", "QNIFA")
    for step in (0, 1, 2, 5, 10):
        for m in ("08", "28"):
            for f in cols:
                v = frames[m][step].get(f)
                if v is None:
                    continue
                want = "0" if v == 0.0 else f"{v:.3e}"
                assert want in page, (
                    f"mp{m} {f} at frame {step} = {want} is not in the page")

    # The control's published step-0 drift.
    c0 = {m: next(r for r in g["control"][m]["frames"] if r["step"] == 0)
          for m in ("08", "28")}
    assert f"{c0['08']['W']:.3e}" in page and f"{c0['28']['W']:.3e}" in page


def test_the_new_fields_claims_are_recomputed_not_transcribed():
    """§9's observation 2: the U/V/PH/MU ratio band."""
    g = _gate()
    mu = [r["ratio"] for r in g["G1"]["rows"] if r["field"] == "MU"]
    uvph = [r["ratio"] for r in g["G1"]["rows"]
            if r["field"] in ("U", "V", "PH")]
    assert len(mu) == 10 and len(uvph) == 30
    page = _doc()
    assert f"{min(mu):.3f}" in page and f"{max(mu):.3f}" in page
    assert f"worst `U`/`V`/`PH` ratio {max(uvph):.3f}" in page


def test_the_reproduction_claim_holds_against_the_first_pass_receipt():
    """§9's observation 1, both halves: shared cells match at the first
    pass's printed precision (3 significant figures everywhere, 4 for the
    ``W`` cells its table printed at 4), AND the trajectories are not
    bit-identical — the page names the QICE cell where the fourth digit
    moves, and that non-identity must stay real."""
    g = _gate()
    old = json.loads((ROOT / "docs" / "public" / "receipts" /
                      "mp28-matched-trajectory" / "shortwindow.json")
                     .read_text(encoding="utf-8"))
    new_frames = {m: {r["step"]: r for r in g["tested"][m]["frames"]}
                  for m in ("08", "28")}
    old_frames = {m: {r["step"]: r for r in old["runs"][m]["frames"]}
                  for m in ("08", "28")}
    shared_fields = ("W", "T", "QVAPOR", "QCLOUD", "QICE",
                     "QNWFA", "QNIFA")
    compared, worst_rel, worst_cell = 0, 0.0, None
    for m in ("08", "28"):
        for step in (1, 2, 5, 10):
            for f in shared_fields:
                a = old_frames[m].get(step, {}).get(f)
                b = new_frames[m].get(step, {}).get(f)
                if a is None or b is None or a == 0.0:
                    continue
                rel = abs(a - b) / abs(a)
                if rel > worst_rel:
                    worst_rel, worst_cell = rel, (m, f, step)
                if f == "W":
                    assert f"{a:.3e}" == f"{b:.3e}", (
                        f"mp{m} W frame {step} no longer matches at the "
                        "4 printed digits the page claims")
                compared += 1
    assert compared == 48, (
        f"{compared} shared cells, not the 48 the page's bound is over")

    page = _doc()
    assert f"{worst_rel:.3e}" in page, (
        f"the page's published worst relative difference is not the "
        f"recomputed {worst_rel:.3e}")
    assert worst_cell == ("08", "QICE", 10), (
        f"the worst cell moved to {worst_cell}; the page names QICE/10")
    assert worst_rel > 0.0, (
        "every shared cell is now bit-equal; the page's 'not "
        "bit-identical' disclosure is stale")

    # The named QICE cell, at the page's own digits.
    a = old_frames["08"][10]["QICE"]
    b = new_frames["08"][10]["QICE"]
    assert f"{a:.4e}" in page and f"{b:.4e}" in page, (
        "the QICE non-identity example is not the receipts' values")

    # The worst row carried across too.
    old_worst = max(old["V3_short_window"]["rows"], key=lambda r: r["ratio"])
    new_worst = max(g["G1"]["rows"], key=lambda r: r["ratio"])
    assert (old_worst["field"], old_worst["step"]) == \
        (new_worst["field"], new_worst["step"]) == ("QNRAIN", 10)
    assert f"{old_worst['ratio']:.3f}" == f"{new_worst['ratio']:.3f}"


def test_the_provenance_hashes_are_published_as_measured():
    prov = (RECEIPTS / "provenance.txt").read_text(encoding="utf-8")
    page = _doc()

    # The tarball and source pins must equal the design's requirements.
    assert ("b8ec11b240a3cf1274b2bd609700191c6ec84628e4c991d3ab562ce9dc50"
            "b5f2  WRFv4.6.1.tar.gz") in prov
    assert "fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802" \
        in prov
    assert "aa99da858be41efa579966680708d230123a7417560af0eb2e24f4c94e253688" \
        in prov
    assert "f2b8d3916560f9046f89f8ac5f32c5292a1800498fd75301e422f147c82a3dbd" \
        in prov

    # Every truncated hash the page prints must prefix a hash in the
    # provenance record (bit-identical claims are claims about these) —
    # or, for the first-pass column of §7's table, in the first pass's
    # own document, which is where those hashes were published.
    first = FIRST.read_text(encoding="utf-8")
    for stub in re.findall(r"`([0-9a-f]{8})…`", page):
        here = re.search(rf"\b{stub}[0-9a-f]{{56}}\b", prov)
        there = re.search(rf"\b{stub}[0-9a-f]{{56}}\b", first) \
            or f"`{stub}" in first
        assert here or there, (
            f"the page prints `{stub}…` but no such hash is in "
            "provenance.txt or the first pass's document")

    # The first pass's published table hashes, which §7 claims were
    # reproduced bit-for-bit, straight from the other document.
    first = FIRST.read_text(encoding="utf-8")
    for stub in ("c7a01fa7", "441bd836", "910fb31d",
                 "a054a3a4", "bcef0ef3", "5e7438b6"):
        assert stub in first and re.search(
            rf"\b{stub}[0-9a-f]{{56}}\b", prov), stub


def test_the_committed_namelists_are_the_declared_window():
    for name in ("sw-wrf-mp08-namelist.input", "sw-wrf-mp28-namelist.input",
                 "sw-wrf-novec-mp08-namelist.input",
                 "sw-wrf-novec-mp28-namelist.input"):
        text = (RECEIPTS / name).read_text(encoding="utf-8")
        expected_mp = 28 if "mp28" in name else 8
        checks = {
            "e_we": "121", "e_sn": "121", "e_vert": "41", "dx": "2000",
            "time_step": "12", "mp_physics": str(expected_mp),
            "periodic_x": ".true.", "periodic_y": ".true.",
            "run_minutes": "40", "history_interval": "1",
            "history_begin_m": "30",
        }
        for key, want in checks.items():
            found = re.search(rf"^\s*{key}\s*=\s*([^,\n]+)", text, re.M)
            assert found, f"{name} has no {key}"
            assert found.group(1).strip() == want, (
                f"{name}: {key} = {found.group(1).strip()}, not {want}")
        for off in ("ra_lw_physics", "ra_sw_physics", "sf_sfclay_physics",
                    "sf_surface_physics", "bl_pbl_physics", "cu_physics"):
            found = re.search(rf"^\s*{off}\s*=\s*([^,\n]+)", text, re.M)
            assert found and found.group(1).strip() == "0", (
                f"{name}: {off} is not 0")


def test_the_first_document_points_here_and_is_not_rewritten():
    first = FIRST.read_text(encoding="utf-8")
    assert "mp28-shortwindow-gate.md" in first
    # Its own verdict sentence must survive untouched.
    assert "**By the rule declared in §6, before any run: HOLD. V3 failed.**" \
        in first


#: Everything committed beside the document, and nothing else.
_RECEIPT_INVENTORY = {
    "shortwindow-gate.json", "provenance.txt", "SHA256SUMS-node.txt",
    "sw-arwen-mp08-run.json", "sw-arwen-mp08-b-run.json",
    "sw-arwen-mp28-run.json", "sw-arwen-mp28-b-run.json",
    "sw-wrf-mp08-namelist.input", "sw-wrf-mp28-namelist.input",
    "sw-wrf-novec-mp08-namelist.input", "sw-wrf-novec-mp28-namelist.input",
    "build-vec.sh", "build-novec.sh", "stage-sw.sh", "run-sw-wrf.sh",
    "run-sw-arwen.sh", "run-gate.sh", "provenance.sh", "chain.sh",
}


def test_the_receipt_directory_holds_nothing_the_gates_do_not_read():
    present = {p.name for p in RECEIPTS.iterdir() if p.is_file()}
    assert present == _RECEIPT_INVENTORY, (
        f"unread: {sorted(present - _RECEIPT_INVENTORY)}; "
        f"missing: {sorted(_RECEIPT_INVENTORY - present)}")


@pytest.mark.parametrize("name", sorted(_RECEIPT_INVENTORY & {
    "shortwindow-gate.json",
    "sw-arwen-mp08-run.json", "sw-arwen-mp08-b-run.json",
    "sw-arwen-mp28-run.json", "sw-arwen-mp28-b-run.json"}))
def test_every_json_receipt_is_parseable_and_non_empty(name):
    payload = json.loads((RECEIPTS / name).read_text(encoding="utf-8"))
    assert isinstance(payload, (dict, list)) and payload
