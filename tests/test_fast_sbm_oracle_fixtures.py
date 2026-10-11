"""The fast SBM Fortran oracle fixtures are intact, controlled and fit to gate on.

The float64 host port of WRF's fast spectral bin microphysics (mp_physics 30)
is gated against the fixtures in ``tests/data/fast_sbm_oracle``, which
``tools/fast_sbm_wrf480_oracle/build.sh`` produced on the reference build host from the pinned
WRF v4.8.0 source. Every later gate trusts these bytes, so this file checks
what would make them untrustworthy, without Fortran, network or GPU:

* a fixture whose bytes, dtypes, shapes or record counts drifted from the
  manifest (a hand-edited or partially regenerated oracle);
* a missing family (a wave-2 item gating against nothing);
* source pins that are not the design's (an oracle of some other WRF);
* a source transform that did more than the one deleted line and the inserted
  probe CALL lines (an oracle of an edited scheme);
* inertness and determinism controls absent or false (a harness that changed
  the answer it records);
* a KC-1 case that never exercised what it exists to exercise, or went cold
  enough to reach the ice path;
* NaN or Inf in a reference value;
* fixtures over the 20 MB budget;
* a reader that returns data from a corrupted file.

Milestone 2 (the cold half) adds: milestone 1 bytes that moved when the
harness grew, a graupel build that is more than its one hail_opt line, cold
controls absent or false, a KC-2 that missed a habit band, a KCOND regime, a
live collision call, Hallett-Mossop, a melt band, D11 or surface snow and
hail (each count recomputed from the Fortran-written summary, so a manifest
that drifted from its fixtures fails), a low-edge run that missed the
pre-pass's I = 1 and J = 1 branches, cold refusals that stop somewhere other
than their check, cold sweeps that miss the branches they exist for or drop
a carried case, and orientation cases that cannot see a swapped grid.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from tools.fast_sbm_wrf480_oracle import fixtures as fx

REPO_ROOT = Path(__file__).resolve().parents[1]
ORACLE = REPO_ROOT / "tests" / "data" / "fast_sbm_oracle"
TOOLS = REPO_ROOT / "tools" / "fast_sbm_wrf480_oracle"


def _harness_module(name):
    """Import a harness script module without keeping its ``sys.path`` entry.

    instrument_probes.py and transform_graupel.py put their own directory at
    the head of ``sys.path`` so they also run as scripts, and they are pinned
    harness files (the manifest's ``harness_sha256``), so the import is
    contained here.  Left in place, that entry made the WRF subtree
    tools/fast_sbm_wrf480_oracle/wrf importable as a namespace package
    ``wrf`` for the rest of the pytest worker: ``pytest.importorskip("wrf")``
    then passed and five tests/test_enprod_hardening.py cases ran on the
    Windows cpu leg, which builds none of the natives they need (E3
    a1388eae7).
    """
    import importlib
    import sys

    saved = list(sys.path)
    try:
        return importlib.import_module(f"tools.fast_sbm_wrf480_oracle.{name}")
    finally:
        sys.path[:] = saved

# fast-sbm-DESIGN.md, section 3.1 and appendix A.
DESIGN_SOURCE_PINS = {
    "phys/module_mp_fast_sbm.F":
        "fd72158202a98d4ad618ebb608eeadffd50852a9875f8bfd9bb8ad63c15f1a2e",
    "phys/module_mp_SBM_polar_radar.F":
        "dffddfbbca15b575aa2948482f34ce996896657b2b0809d3e3746c1c5a1bdf40",
}
DESIGN_ARCHIVE = ("25ed63ab3550aed110ae24df229faece20eafedfdf21cde01d8973826a7294bf", 1339105)
DESIGN_MEMBERS = {
    "BLKD_SDC.dat": (2280, "6992497ee5b95f0fd215b762ad4af8611079d1e13dbe3db78687fe04a128abf2"),
    "capacity33.asc": (3042, "33759dbb0e238b37403ce2ad62759cca57c03538d72d34c2cc04bb7b2547c951"),
    "masses33.asc": (3042, "f53d3acdaa0a3e2a265272fa22a2a0bd54292f51815149370bbf315e3fa1bee4"),
    "termvels33_corrected.asc":
        (3042, "bf4209175492a3019563671676787afc3befd948fddc6435db58f7a8d7f702e3"),
    "constants33.asc": (3911, "d3d7cf345ebf850fa12376a7f173ea12b169f6413910939682494bcc5874a4fa"),
    "kernLL_z33.asc": (43016, "ce4e140595ff6db2434ecd5376a16c50ea410d03ddc1700bfa95a3490c15106f"),
    "ckli_33_300mb_500mb_750mb.asc":
        (129047, "59ce82054f29df7d9f982249b915201f3f0385ae559c5ef2b7cdb1907a46f81a"),
    "cklg_33_300mb_500mb_750mb.asc":
        (43016, "f3e8149a7e1bac947706f945a1d2ed0fb81ef2737404a0ef8c8c0c682fb4290f"),
    "cklh_33_300mb_500mb_750mb.asc":
        (43016, "a2ffe0b96e4acfcd06dc1a0b10cb28f60601ebe255f8e00a70ce1797705fb50e"),
    "ckls_33_300mb_500mb_750mb.asc":
        (43016, "ae282329a306d384bd6de958f1e0bf745c496d286237e8d22c85926081e0456a"),
    "ckii_33_300mb_500mb_750mb.asc":
        (387140, "546777c064e7687bbaaaf30ab417dfc146e2decaabc41b270439e33328a75c7f"),
    "ckis_33_300mb_500mb_750mb.asc":
        (129047, "657428fd7bbc0dcf5e7810bb0e25af42645f6122a4b24313404b805cfcdec44d"),
    "cksg_33_300mb_500mb_750mb.asc":
        (43016, "85b5fd667242ad216dd1b06f0f1c5ddfe15d797ae6c14dcc45e1097743bb37b9"),
    "ckss_33_300mb_500mb_750mb.asc":
        (43016, "1946554a0e1341cd12c9ae6be2cd2c786bfc3e4622b8fe87f2e85fcc64373847"),
    "bulkdens33.asc": (3042, "a5607b3d0b9eb1fcdba30e50b714e348a81aabb64d963b0a00ff4f7b2a24caa6"),
    "bulkradii33.asc": (3512, "d432d8eed8760f5000c6c86ac1b602687feb0783f558adbf10811fe7040a9de0"),
    "coeff_p_new_33.dat":
        (107730, "0fc78f82cfed4033eab9bcaf3f5d6bf07a4f1bbbbf3aff708ede6a0b2e921e34"),
    "coeff_q_new_33.dat":
        (9396, "cbda1e0fb2c604d6c8d61ad0abc22be7b1552be93bf5daed10b5186c4ba07d9e"),
}
SWEEPS = ("polysvp", "jerrate", "jersupsat", "jerdfun", "jernewf", "onecond1",
          "nucleation", "coll_xxx", "coal_bott", "breakup", "spont", "falflux")
COLD_SWEEPS = ("freez", "melt", "ice_nucl", "jernucl01_ice", "jerrate_ice", "jertimesc_ice",
               "jerdfun_ice", "onecond2", "onecond3", "coll_cold", "modkrn", "coal_cold")
REQUIRED_M1 = sorted(
    ["tables_o64", "tables_o32", "kernels_o64", "kernels_o32", "refusals", "kc1_case",
     "kc1_summary_o64", "kc1_summary_o32", "kc1_replay_o64", "kc1_replay_o32",
     "kc1_crossreplay_o64_on_o32", "kc1_sensitivity_o64", "kc1_probe_o64", "kc1_probe_o32"]
    + [f"sweep_{f}_{b}" for f in SWEEPS for b in ("o64", "o32")])
REQUIRED_M2 = sorted(
    ["refusals_cold", "kc2_case", "kc2_crossreplay_o64_on_o32", "kc2_sensitivity_o64"]
    + [f"{n}_{b}" for n in ("kc2_summary", "kc2_replay", "kc2_probe", "kc2g_replay",
                            "kc2g_summary", "sweep_coal_cold_graupel", "kc2edge_replay",
                            "kc2edge_probe") for b in ("o64", "o32")]
    + [f"sweep_{f}_{b}" for f in COLD_SWEEPS for b in ("o64", "o32")])
REQUIRED = sorted(REQUIRED_M1 + REQUIRED_M2)
HARNESS_FILES = ("build.sh", "sources.sha256", "stubs_wrf.F90", "transform_access.py",
                 "instrument_probes.py", "fsbm_probe.F90", "run_dump_tables.F90",
                 "run_sweeps.F90", "run_column.F90", "transform_graupel.py", "run_kc2.F90")
SIZE_LIMIT = 20 * 1024 * 1024
# The SHA-256 of every milestone 1 fixture as milestone 1 packed it
# (the first packing of the warm fixtures); every milestone 1 gate was written against
# these bytes.
M1_FIXTURE_SHA256 = {
    "kc1_case":
        "6bec8bd958b1c158cad2ac736e7eb5348c37263b209c4990b31134734e0847f8",
    "kc1_crossreplay_o64_on_o32":
        "151969f2f46fe9c013b1636195d22ca8e5c0f8775977ac20985c8c79ba9451e2",
    "kc1_probe_o32":
        "7c6b812b8b2e41aee92c7688d52cd2611e6ff837092279f7eaa554c0985fcf91",
    "kc1_probe_o64":
        "d1bbf8df5251b11226f739867c189bca6fdc36794570468e7b2425603c1410bb",
    "kc1_replay_o32":
        "b79e4beaa91f5d9b530cbe82d5aee920a8d5d59ace831426100f2b672edfc1c9",
    "kc1_replay_o64":
        "23982d0f809821a7f2ea470351494f5fe57d37323bba99aab04fb2dfe0cae7e5",
    "kc1_sensitivity_o64":
        "6e139f5c2c8223401adf621b1755c0355640a1a105cbf6fc470ae64edcfe4e8c",
    "kc1_summary_o32":
        "f95c9e665ed2866992e230fac86f36d7a47a5d9a9f7256e0370bf07b534983e2",
    "kc1_summary_o64":
        "e9543270ca2705c11b69cb8459863de32bb29ea3dd13217fa6987bc05e51c0e3",
    "kernels_o32":
        "60c6471741e73f359520d8921433d97752e91a7f9efe5424a21e59fc44defe43",
    "kernels_o64":
        "804002f1001cd6842dd52ae7f7d9f7a3e5b6386bfcb0e06b5b50243212332097",
    "refusals":
        "a52adaaec0ac0788ad7557cdb65ee0ebf23c662b605adc308de88f224182cf4d",
    "sweep_breakup_o32":
        "a45cdbb93c8cdc9d259d5b3a06ae2b94f31121ff0d9e9867e48728496ae4f972",
    "sweep_breakup_o64":
        "c12efb3107013dac92399c8cb6e2ff21f81ce5cb0a59c1f8b51ecd1ebcff5506",
    "sweep_coal_bott_o32":
        "c6f0a30bd46e6d337d7bf4f8fbd48d261d327cc3f8f53c0190acffecd78b1907",
    "sweep_coal_bott_o64":
        "692b75f0a369c88b740d315bf8eb79e64e1f92764e294a24339b6fbe631b667b",
    "sweep_coll_xxx_o32":
        "36c48cdfcae6e7808fe973b5dbf24f94766b7c301c657c467029303cf288344f",
    "sweep_coll_xxx_o64":
        "07a8d45df9e8d0b0ab7f337cee1729921964f969963b4b35a7c02bc36a20ee17",
    "sweep_falflux_o32":
        "0a6c0142a88332167e8ab4e3beb296423bf55624813df92c920cc88089159c2c",
    "sweep_falflux_o64":
        "b3016daf2ba5156420d8616a078536e7417038c7c42c74e4c195d2f1ae41a705",
    "sweep_jerdfun_o32":
        "40f5d1894f3dc693750b367630890d59f32d3ed7f8a89e16353a36a5e9e1d08b",
    "sweep_jerdfun_o64":
        "0133f529f3cd3171e9d51cc17089d7abde45ae2ce68371ea5de3a3dc84017253",
    "sweep_jernewf_o32":
        "7d1fb9b9f66b32a1da69c4f146b41627d31ea3ed603e76e4e9b845f88ca6c1ec",
    "sweep_jernewf_o64":
        "40d3cad41a4632fedc119da94b463ca3826eb03fcbb9ad04dbd6641984fac86b",
    "sweep_jerrate_o32":
        "301b250088a66941962e18509d96437b5e11bb00fbe00cd9794dcbf69f826511",
    "sweep_jerrate_o64":
        "b31053ca7b9153d39a1d9dcba9c75756b238ebcf1bf957f2a0a8f91727c07650",
    "sweep_jersupsat_o32":
        "ff228db4554bf078e74936f6c6840d1c768fc97b507f71f018bb199f7543c60a",
    "sweep_jersupsat_o64":
        "c875a12803005a6155533cb61d4280cd48df4def32deb39d1d603eb1412525bd",
    "sweep_nucleation_o32":
        "6cd526de82645b11f04db70b4c42212935870fa5de2c07f612ea3e909a80753e",
    "sweep_nucleation_o64":
        "2ca0c45b5394e813832b475a835fe1e346a88eb3d636f2bf36d7b892bf961db4",
    "sweep_onecond1_o32":
        "e5c3600050f8a318c764ce0ad9545218e7f0a9d838e5a60fd494b4b5967440a7",
    "sweep_onecond1_o64":
        "99d30a7c28d0a3e83a0dfa347e085b0dbaa4c4acdc3d155e7281a07c0f5c3bc3",
    "sweep_polysvp_o32":
        "822c494f8dba960746ec8b27cf719e844cab62645a4431dd397d01e58495f4ec",
    "sweep_polysvp_o64":
        "0694f86c99cdb1cee8839ad5d66f8e9412560654ab2b70bde7d075a1ac448fe0",
    "sweep_spont_o32":
        "8a818564142b73b45adf1e911528e0455f5eb640d1a9bfe3dee9c3fac3835822",
    "sweep_spont_o64":
        "9cd2ff1b2973960c956f2ad919d40874cdfa75c45f3140372ffde9b19582d5ba",
    "tables_o32":
        "bc1579ca66b2bdaabfa0ece889230905807f3a352d8b22007a3966755c130fb2",
    "tables_o64":
        "d08c366a0fc671073283d4f13f0f4b7c4e79b9e193160e724430b885c79eaac2",
}
CALL_LINE = re.compile(r"^\s*CALL fsbm_probe_[a-z0-9_]+\(.*\)$")


@pytest.fixture(scope="module")
def manifest() -> dict:
    return fx.load_manifest()


def _raw(name: str, manifest: dict) -> bytes:
    return gzip.decompress((ORACLE / manifest["files"][name]["file"]).read_bytes())


def test_every_required_family_is_present(manifest):
    names = set(manifest["files"])
    assert sorted(set(REQUIRED) - names) == []
    assert sorted(manifest["required_families"]) == REQUIRED
    on_disk = {p.name for p in ORACLE.glob("*.bin.gz")}
    assert on_disk == {entry["file"] for entry in manifest["files"].values()}


@pytest.mark.parametrize("name", REQUIRED)
def test_each_fixture_matches_its_manifest_entry(name, manifest):
    entry = manifest["files"][name]
    raw = _raw(name, manifest)
    assert len(raw) == entry["bytes"]
    assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
    gz = (ORACLE / entry["file"]).read_bytes()
    assert len(gz) == entry["gzip_bytes"]
    assert hashlib.sha256(gz).hexdigest() == entry["gzip_sha256"]
    # gzip -9 -n form: no stored name, zero timestamp.
    assert gz[3] & 0x08 == 0 and gz[4:8] == b"\x00\x00\x00\x00"
    if entry.get("container") == "refusals":
        members = entry["members"]
        assert sum(m["bytes"] for m in members) == len(raw)
        records = 0
        for m in members:
            chunk = raw[m["offset"]:m["offset"] + m["bytes"]]
            assert hashlib.sha256(chunk).hexdigest() == m["sha256"]
            layout = fx.Layout.from_json(m["layout"])
            assert layout.nbytes == m["bytes"]
            records += len(list(fx.parse_stream(chunk, layout)))
        assert records == entry["record_count"]
        return
    layout = fx.Layout.from_json(entry["layout"])
    assert layout.nbytes == len(raw)
    seen: dict[str, int] = {}
    kinds = set()
    for rec in fx.parse_stream(raw, layout):
        seen[rec.type] = seen.get(rec.type, 0) + 1
        spec = layout.by_name()[rec.type]
        for f in spec.fields:
            value = np.asarray(rec[f.name])
            assert value.shape == f.shape, (rec.type, f.name)
            assert value.dtype == f.dtype, (rec.type, f.name)
            kinds.add(f.kind)
    assert seen == entry["record_counts"]
    assert sum(seen.values()) == entry["record_count"]
    assert sorted(kinds) == entry["field_kinds"]
    if entry["build"] == "o64":
        # -freal-4-real-8 promotes every default and KIND=4 real.
        assert "f4" not in kinds


def test_the_native_build_keeps_wrfs_float32_tables(manifest):
    # WRF declares the tables REAL(KIND=4) and the collision grids REAL(KIND=8);
    # the native fixtures must show both, the promoted ones float64 only.
    assert "f4" in manifest["files"]["tables_o32"]["field_kinds"]
    assert "f4" not in manifest["files"]["tables_o64"]["field_kinds"]
    t32 = fx.load_fixture("tables_o32")
    assert t32["read.xl"].dtype == np.float32 and t32["derived.chucm"].dtype == np.float64


def test_no_value_is_nan_or_inf(manifest):
    for name in manifest["files"]:
        if manifest["files"][name].get("container"):
            records = [r for ref in fx.iter_refusals(name=name) for r in ref.records]
        else:
            records = fx.iter_records(name)
        for rec in records:
            for key, value in rec.fields.items():
                arr = np.asarray(value)
                if arr.dtype.kind == "f":
                    assert np.all(np.isfinite(arr)), f"{name} {rec.type}.{key}"


def test_source_pins_are_the_designs(manifest):
    pins = manifest["source_pins"]
    assert {k: v["lf_sha256"] for k, v in pins["sources"].items()} == DESIGN_SOURCE_PINS
    assert (pins["archive"]["sha256"], pins["archive"]["bytes"]) == DESIGN_ARCHIVE
    members = {k.split("/", 1)[1]: (v["bytes"], v["sha256"]) for k, v in pins["members"].items()}
    assert members == DESIGN_MEMBERS
    assert pins["url"] == "https://www2.mmm.ucar.edu/wrf/src/wrf_files/SBM_input_33.tgz"
    # The pins the harness enforces are the ones recorded.
    text = (TOOLS / "sources.sha256").read_text()
    for path, digest in DESIGN_SOURCE_PINS.items():
        assert f"source {digest} - {path}" in text
    assert manifest["transforms"]["access"]["input_sha256"] == DESIGN_SOURCE_PINS[
        "phys/module_mp_fast_sbm.F"]


def test_fixtures_come_from_the_committed_harness(manifest):
    for name in HARNESS_FILES:
        digest = hashlib.sha256((TOOLS / name).read_bytes()).hexdigest()
        assert manifest["harness_sha256"][name] == digest, (
            f"{name} changed since the fixtures were built; rerun build.sh on the reference build host "
            "and pack_fixtures.py")


def test_access_transform_deletes_exactly_line_3862(manifest):
    access = manifest["transforms"]["access"]
    assert access["deleted_line_number"] == 3862
    assert access["deleted_line_content"] == " PRIVATE"
    lines = access["diff"].splitlines()
    assert [ln for ln in lines if ln.startswith("@@")] == ["@@ -3862 +3861,0 @@"]
    body = [ln for ln in lines if not ln.startswith(("---", "+++", "@@"))]
    assert body == ["- PRIVATE"]


def test_probe_transform_only_inserts_call_lines(manifest):
    probe = manifest["transforms"]["probe"]
    lines = probe["diff"].splitlines()
    body = [ln for ln in lines if not ln.startswith(("---", "+++", "@@"))]
    assert not [ln for ln in body if ln.startswith("-")]
    added = [ln[1:] for ln in body if ln.startswith("+")]
    assert added and all(CALL_LINE.match(ln) for ln in added)
    assert len(added) == probe["inserted_lines"] == len(probe["anchors"])
    assert probe["removed_lines"] == 0
    # The inserted statements are the ones the committed transform defines.
    instrument_probes = _harness_module("instrument_probes")
    assert sorted(a.strip() for a in added) == sorted(a.call for a in instrument_probes.ANCHORS)


def test_inertness_and_determinism_controls_hold_in_both_precisions(manifest):
    controls = manifest["controls"]
    for b in ("o64", "o32"):
        for key in (f"inert_access_{b}_control", f"inert_probe_{b}_control",
                    f"inert_probe_run2_{b}_control", f"determinism_pristine_{b}_control",
                    f"determinism_probe_{b}_handoff", f"selfreplay_{b}_identical",
                    f"xl_mg_module_equals_6078_expression_{b}"):
            assert controls.get(key) is True, key
        for f in SWEEPS:
            assert controls.get(f"determinism_sweep_{b}_{f}") is True, (b, f)
        for f in ("kc1_case", "kc1_summary", "kc1_replay", "kc1_control", "kc1_probe"):
            assert controls.get(f"determinism_probe_{b}_{f}_raw") is True, (b, f)
    streams = controls["kc1_control_stream_sha256"]
    for b in ("o64", "o32"):
        runs = [streams[f"{r}_{b}_kc1_{n}/kc1_control.raw"]
                for r, n in (("pristine", 1), ("pristine", 2), ("access", 1),
                             ("probe", 1), ("probe", 2))]
        assert len(set(runs)) == 1, b
    assert streams["pristine_o64_kc1_1/kc1_control.raw"] != \
        streams["pristine_o32_kc1_1/kc1_control.raw"]


def test_kc1_acceptance_counts(manifest):
    kc1 = manifest["kc1"]
    for b in ("o64", "o32"):
        acc = kc1["acceptance"][b]
        c = acc["counts"]
        assert all(t > 275.15 for t in acc["min_temperature_k"]), b
        for col in range(3):   # c1, c2, c3
            assert acc["rainnc_final_mm"][col] > 0 and c["rain_steps"][col] > 0, (b, col)
        assert c["coal_breakup"][2] > 0                          # collisional breakup in c3
        assert sum(c["nucl_cb_act"]) > 0 and sum(c["nucl_cb_calls"]) > 0
        assert sum(c["nucl_ic_act"]) > 0 and sum(c["nucl_ic_calls"]) > 0
        assert sum(c["cond1_evap"]) > 0                           # ONECOND1 KCOND 10
        assert sum(c["spont_taken"]) > 0
        assert all(acc["checks"].values())
    # The recorded counts are the ones in the summary fixtures.
    for b in ("o64", "o32"):
        data = fx.load_fixture(f"kc1_summary_{b}")
        c = kc1["acceptance"][b]["counts"]
        for key in ("nucl_cb_act", "nucl_ic_act", "cond1_evap", "coal_breakup",
                    "spont_taken", "rain_steps"):
            assert data[f"acceptance.{key}"][0].tolist() == c[key]
        tmin = np.minimum(data["acceptance.tmin_run_pre"][0], data["acceptance.tmin_run_post"][0])
        assert float(tmin.min()) > 275.15
        # Every per-step minimum temperature too, not only the run summary.
        assert float(data["step.tmin"].min()) > 275.15
        assert data["step.step"].tolist() == list(range(1, 301))


def test_kc1_case_definition_is_published(manifest):
    d = manifest["kc1"]["definition"]
    assert d["time_step_s"] == 12.0 and d["steps"] == 300
    assert d["replay_steps"] == [2, 3, 4, 6, 10, 15, 25, 40, 60, 80, 100, 130, 160, 200,
                                 250, 300]
    assert d["probe_steps"] == [10, 40, 80, 130, 200, 300]
    assert len(d["probe_levels"]) == 8
    case = fx.load_fixture("kc1_case")
    assert case["case.dz"].shape == (1, 40)
    dz = case["case.dz"][0]
    assert dz[0] == 50.0 and np.allclose(dz[1:] / dz[:-1], 1.025, rtol=0, atol=1e-12)
    assert np.allclose(case["case.t_surface"][0], [303.0, 303.0, 303.0, 296.0])
    assert np.allclose(case["case.p_surface"][0], [1.0e5, 1.005e5, 0.995e5, 0.85e5])
    assert case["case.replay_steps"][0].tolist() == d["replay_steps"]


def test_replay_and_probe_iterators(manifest):
    steps = []
    for step, pre, post in fx.iter_replay("kc1_replay_o64"):
        steps.append(step)
        assert pre["chem"].shape == (4, 40, 142) and post["chem"].shape == (4, 40, 142)
        assert pre["chem"].dtype == np.float64
    assert steps == manifest["kc1"]["definition"]["replay_steps"]
    cross = [(s, pre) for s, pre, _ in fx.iter_replay("kc1_crossreplay_o64_on_o32")]
    assert [s for s, _ in cross] == steps and all(pre is None for _, pre in cross)
    kinds = {r.type for r in fx.iter_probe("kc1_probe_o32", step=40)}
    assert {"entry", "nucl_before", "nucl_after", "cond1_before", "cond1_after",
            "coal_before", "coal_after", "diag"} <= kinds
    cb = [r for r in fx.iter_probe("kc1_probe_o64", types=["nucl_before"])
          if r["is_cloud_base"] == 1]
    assert cb, "the cloud-base activation event is not in the probe records"
    refusals = list(fx.iter_refusals())
    assert len(refusals) == 12
    assert {(r.case, r.build) for r in refusals} == {(n, b) for n in range(1, 7)
                                                     for b in ("o64", "o32")}
    for r in refusals:
        assert (r.exit_status == 99 and r.message) or (r.exit_status == 0 and not r.message)


def test_total_size_is_within_budget(manifest):
    total = sum(p.stat().st_size for p in ORACLE.glob("*.bin.gz"))
    assert total == manifest["total_gzip_bytes"]
    assert total <= SIZE_LIMIT


def test_load_fixture_round_trip_returns_named_arrays(manifest):
    data = fx.load_fixture("tables_o64")
    assert data["read.xl"].shape == (1, 33) and data["read.xl"].dtype == np.float64
    assert data["derived.ima"].dtype == np.int32 and data["derived.ima"].shape == (1, 33, 33)
    assert data["read.pkij"].shape == (1, 18, 18, 18)
    data32 = fx.load_fixture("tables_o32")
    assert data32["read.xl"].dtype == np.float32
    # Bins double in mass to the tables' five significant digits.
    xl = data["read.xl"][0]
    assert np.all(np.abs(xl[1:] / xl[:-1] - 2.0) < 1e-4)
    kern = fx.load_fixture("kernels_o64")
    assert kern["kernels.p_hpa"].tolist() == [1000., 950., 850., 750., 700., 600., 500., 400.,
                                              300., 250.]
    assert kern["kernels.cwll"].shape == (10, 33, 33)
    refusals = fx.load_fixture("refusals")
    assert int(refusals["case6_o64.exit_status"]) == 0


def test_reader_refuses_a_corrupted_byte(tmp_path, manifest):
    root = tmp_path / "oracle"
    shutil.copytree(ORACLE, root)
    name = "sweep_polysvp_o64"
    path = root / manifest["files"][name]["file"]
    raw = bytearray(gzip.decompress(path.read_bytes()))
    raw[len(raw) // 2] ^= 0x01
    path.write_bytes(gzip.compress(bytes(raw), compresslevel=9, mtime=0))
    with pytest.raises(fx.FixtureError, match="SHA-256"):
        fx.load_fixture(name, root=root)
    # A damaged compressed stream is refused too, not passed through.
    gz = bytearray((ORACLE / manifest["files"][name]["file"]).read_bytes())
    gz[len(gz) // 2] ^= 0xFF
    path.write_bytes(bytes(gz))
    with pytest.raises(fx.FixtureError):
        fx.load_fixture(name, root=root)
    # And the untouched copy still loads.
    assert fx.load_fixture("sweep_polysvp_o32", root=root)["case.es"].shape[0] > 0


def test_manifest_prose_travels_with_the_fixtures():
    text = (ORACLE / "MANIFEST.md").read_text(encoding="utf-8")
    assert "KC-1" in text and "3862" in text
    assert "—" not in text   # plain sentences, no em-dashes
    data = json.loads((ORACLE / "MANIFEST.json").read_text(encoding="utf-8"))
    assert data["schema"] == "fsbm-oracle-manifest 1"


# ======================================================================
# Milestone 2: the cold oracle
# ======================================================================

KC2_COLD = (0, 2, 4, 8)          # tile columns (i fastest) of A (2,2), D (4,2), B (3,3), C (4,4)


def _records(name, rtype):
    return [r for r in fx.iter_records(name, types=[rtype])]


def test_milestone1_fixtures_are_the_bytes_milestone1_was_built_on(manifest):
    # Every milestone 1 gate was written against these bytes; the extended
    # harness must reproduce them exactly.
    for name, digest in M1_FIXTURE_SHA256.items():
        assert manifest["files"][name]["sha256"] == digest, name
    assert manifest["controls"]["milestone1_streams_unchanged"] is True
    assert manifest["controls"]["milestone1_streams_changed"] == []


def test_graupel_transform_changes_only_hail_opt(manifest):
    for key in ("graupel", "graupel_probe"):
        tx = manifest["transforms"][key]
        lines = tx["diff"].splitlines()
        body = [ln for ln in lines if not ln.startswith(("---", "+++", "@@"))]
        assert body == ["- INTEGER,PARAMETER :: hail_opt = 1",
                        "+ INTEGER,PARAMETER :: hail_opt = 0"], key
        assert len([ln for ln in lines if ln.startswith("@@")]) == 1
    assert manifest["transforms"]["graupel"]["hunk"] == "@@ -3886 +3886 @@"
    assert manifest["transforms"]["graupel"]["input_sha256"] == DESIGN_SOURCE_PINS[
        "phys/module_mp_fast_sbm.F"]
    # The graupel probe build is the probe source with that one line changed.
    assert (manifest["transforms"]["graupel_probe"]["input_sha256"]
            == manifest["transforms"]["probe"]["output_sha256"])
    transform_graupel = _harness_module("transform_graupel")
    assert transform_graupel.OLD_LINE.strip() == "INTEGER,PARAMETER :: hail_opt = 1"


def test_graupel_transform_refuses_anything_but_its_line():
    transform_graupel = _harness_module("transform_graupel")
    refusal = transform_graupel.TransformRefusal
    fake = b"  MODULE module_mp_fast_sbm\n INTEGER,PARAMETER :: hail_opt = 1\n CONTAINS\n"
    with pytest.raises(refusal, match="SHA-256"):
        transform_graupel.graupel_transform(fake, "pristine")
    digest = hashlib.sha256(fake).hexdigest()
    out, diff, summary = transform_graupel.graupel_transform(fake, "probe", digest)
    assert out == fake.replace(b"hail_opt = 1", b"hail_opt = 0")
    assert summary["changed_line_number"] == 2
    with pytest.raises(refusal, match="recorded probe output"):
        transform_graupel.graupel_transform(fake, "probe", "0" * 64)
    twice = fake + b" INTEGER,PARAMETER :: hail_opt = 1\n"
    with pytest.raises(refusal, match="2 lines"):
        transform_graupel.graupel_transform(twice, "probe", hashlib.sha256(twice).hexdigest())


def test_cold_controls_hold_in_both_precisions(manifest):
    controls = manifest["controls"]
    for b in ("o64", "o32"):
        for key in (f"inert_access_{b}_kc2_control", f"inert_probe_{b}_kc2_control",
                    f"inert_probe_run2_{b}_kc2_control", f"determinism_pristine_{b}_kc2_control",
                    f"determinism_gpristine_{b}_kc2g_control", f"inert_gprobe_{b}_kc2g_control",
                    f"inert_gprobe_run2_{b}_kc2g_control", f"determinism_probe_{b}_kc2_handoff",
                    f"selfreplay_kc2_{b}_identical", f"selfreplay_kc2g_{b}_identical",
                    f"graupel_build_differs_from_hail_{b}",
                    f"determinism_sweep_{b}_coal_cold_graupel"):
            assert controls.get(key) is True, key
        for f in COLD_SWEEPS:
            assert controls.get(f"determinism_sweep_{b}_{f}") is True, (b, f)
        for f in ("kc2_case", "kc2_summary", "kc2_replay", "kc2_control", "kc2_probe"):
            assert controls.get(f"determinism_probe_{b}_{f}_raw") is True, (b, f)
        for f in ("kc2g_summary", "kc2g_replay", "kc2g_control", "kc2g_probe"):
            assert controls.get(f"determinism_gprobe_{b}_{f}_raw") is True, (b, f)
        for key in (f"inert_probe_{b}_kc2edge_control", f"inert_probe_run2_{b}_kc2edge_control"):
            assert controls.get(key) is True, key
        for f in ("kc2edge_replay", "kc2edge_probe"):
            for part in ("raw", "layout"):
                assert controls.get(f"determinism_probe_{b}_{f}_{part}") is True, (b, f, part)
    streams = controls["kc2_control_stream_sha256"]
    for b in ("o64", "o32"):
        runs = [streams[f"{r}_{b}_kc2_{n}/kc2_control.raw"]
                for r, n in (("pristine", 1), ("pristine", 2), ("access", 1),
                             ("probe", 1), ("probe", 2))]
        assert len(set(runs)) == 1, b
        graupel = [streams[f"{r}_{b}_kc2g_{n}/kc2g_control.raw"]
                   for r, n in (("gpristine", 1), ("gpristine", 2), ("gprobe", 1), ("gprobe", 2))]
        assert len(set(graupel)) == 1, b
        assert graupel[0] != runs[0]
    assert streams["pristine_o64_kc2_1/kc2_control.raw"] != \
        streams["pristine_o32_kc2_1/kc2_control.raw"]


def _kc2_acceptance_from_fixture(cm: str, b: str) -> dict:
    # The acceptance record the Fortran wrote into the summary fixture, in the
    # manifest's form: every integer field raveled in Fortran order, and the
    # per-column temperatures and surface totals.
    data = fx.load_fixture(f"{cm}_summary_{b}")
    counts = {}
    for key, values in data.items():
        if not key.startswith("acceptance."):
            continue
        arr = np.asarray(values[0])
        if arr.dtype.kind == "i" and key != "acceptance.column_type":
            counts[key.split(".", 1)[1]] = [int(x) for x in arr.ravel(order="F")]
    floats = {name: [float(x) for x in np.asarray(data[f"acceptance.{field}"][0]).ravel(order="F")]
              for name, field in (("min_temperature_k", "tmin_run"),
                                  ("max_temperature_k", "tmax_run"),
                                  ("rainnc_final_mm", "rainnc_final"),
                                  ("snownc_final_mm", "snownc_final"),
                                  ("graupelnc_final_mm", "graupelnc_final"))}
    assert data["step.step"].tolist() == list(range(1, 301))
    return {"counts": counts, **floats}


def _catalogue_from_counts(c: dict) -> dict:
    def total(k):
        return sum(c[k])

    return {
        "D11_levels_small_snow_deleted": total("d11_levels"),
        "D12_negative_bins_after_melt": total("melt_negative_bin"),
        "D14_homogeneous_freezing_events": total("homfrz"),
        "D15_prepass_points_below_238K": total("d15_points"),
        "D16_mixed_phase_at_or_below_233K_no_growth": total("dispatch_none_d16_cold"),
        "D16_ice_without_drops_above_minus_0.187C_no_growth": total("dispatch_none_d16_warm"),
        "S1_supsat_branches_8_and_9": total("supsat_branch8") + total("supsat_branch9"),
        "S2_supsat_branch_7": total("supsat_branch7"),
        "supsat_branches": {str(i): total(f"supsat_branch{i}") for i in range(10)},
        "kcond50_calls": total("cond3_kcond50"),
    }


def test_kc2_acceptance(manifest):
    kc2 = manifest["kc2"]
    for b in ("o64", "o32"):
        for key, cm, build in (("acceptance", "kc2", "hail"),
                               ("acceptance_graupel", "kc2g", "graupel")):
            acc = kc2[key][b]
            assert all(acc["checks"].values()), (b, key, acc["checks"])
            # Every count, temperature and surface total is the one in the
            # summary fixture the Fortran wrote, not only the manifest's copy.
            fixture = _kc2_acceptance_from_fixture(cm, b)
            assert fixture["counts"] == acc["counts"], (b, key)
            for name in ("min_temperature_k", "max_temperature_k", "rainnc_final_mm",
                         "snownc_final_mm", "graupelnc_final_mm"):
                assert fixture[name] == acc[name], (b, key, name)
            cat = _catalogue_from_counts(fixture["counts"])
            assert cat == kc2["catalogue_counts"][f"{build}_{b}"], (b, key)
            c = fixture["counts"]
            # through 0 C and below -40 C, from a 300 K surface
            assert max(fixture["max_temperature_k"]) >= 295.0
            assert min(fixture["min_temperature_k"]) < 233.15
            for band in range(1, 7):
                assert sum(c[f"icenucl_band{band}"]) > 0, (b, key, band)
            assert sum(c["homfrz"]) > 0
            for k in ("cond1_calls", "cond2_calls", "cond3_calls", "cond3_kcond30",
                      "cond3_kcond31", "cond3_kcond32", "hm_splinters", "freez_to_plates",
                      "freez_to_hail", "d11_levels"):
                assert sum(c[k]) > 0, (b, key, k)
            for band in (1, 2, 3):
                assert sum(c[f"melt_snow_band{band}"]) > 0 and sum(c[f"melt_dense_band{band}"]) > 0
            assert max(fixture["snownc_final_mm"]) > 0 and max(fixture["graupelnc_final_mm"]) > 0
            # BreakUp_Snow is never called at 33 bins, though its guard is true
            # at every point of every column.
            assert sum(c["snowbrk_call"]) == 0
            assert c["snowbrk_guard_true"] == [300*50]*9
        live = kc2["live_cold_collision_calls"]
        names = live["names"]
        hail = _kc2_acceptance_from_fixture("kc2", b)["counts"]
        graupel = _kc2_acceptance_from_fixture("kc2g", b)["counts"]
        for i in range(1, 13):
            assert (sum(hail[names[i - 1]]) > 0) == (i in live["hail"]), (b, i)
            assert (sum(graupel[names[i - 1]]) > 0) == (i in live["graupel"]), (b, i)
        # the environment columns do nothing
        for col in range(9):
            if col not in KC2_COLD:
                assert hail["forcing"][col] == 0 and hail["coal_calls"][col] == 0
        # What the catalogue rows point at: D11, D14, D15 and both D16 cases
        # occur; S2 (branch 7) is reached by real calls and S1 (branches 8 and 9)
        # never is; at 12 s no melt band drives a bin negative (D12 needs 100 s).
        for build in ("hail", "graupel"):
            cat = kc2["catalogue_counts"][f"{build}_{b}"]
            for key in ("D11_levels_small_snow_deleted", "D14_homogeneous_freezing_events",
                        "D15_prepass_points_below_238K",
                        "D16_mixed_phase_at_or_below_233K_no_growth",
                        "D16_ice_without_drops_above_minus_0.187C_no_growth",
                        "S2_supsat_branch_7"):
                assert cat[key] > 0, (build, b, key)
            assert cat["S1_supsat_branches_8_and_9"] == 0
            assert cat["D12_negative_bins_after_melt"] == 0
            assert cat["kcond50_calls"] == 0


def test_kc2edge_takes_the_low_edge_branches(manifest):
    # KC-2's tile starts at i, j = 2; the low-edge run moves it to i, j = 1..3
    # so the host pre-pass has an oracle for DERIVT_X/Y (:4422-4423,
    # :4429-4430) and the DSUPICE_XYZ differences (:4460-4492) at I = 1, J = 1.
    ke = manifest["kc2edge"]
    for b in ("o64", "o32"):
        acc = ke["acceptance"][b]
        assert all(acc["checks"].values()), (b, acc["checks"])
        case = next(fx.iter_records(f"kc2edge_replay_{b}", types=["edge_case"]))
        assert case["tile"].tolist() == [1, 3, 1, 3, 1, 50] == acc["tile"]
        assert int(np.asarray(case["column_type"])[0, 0]) == 2      # B in the corner
        assert case["replay_steps"].tolist() == [2, 4, 6] == acc["replay_steps"]
        assert [s for s, _, _ in fx.iter_replay(f"kc2edge_replay_{b}")] == [2, 4, 6]
        prepass = list(fx.iter_probe(f"kc2edge_probe_{b}", types=["prepass"]))
        assert [int(r["step"]) for r in prepass] == [2, 4, 6]
        ds = [np.asarray(r["dsupice_xyz"]) for r in prepass]         # (i, k, j) of the tile
        dtf = [np.asarray(r["dtfreez_xyz"]) for r in prepass]
        assert ds[0].shape == (3, 50, 3)
        i1 = (0, slice(None), slice(None))
        j1 = (slice(None), slice(None), 0)
        for name, arrays, cut in (("dsupice_nonzero_points_i1", ds, i1),
                                  ("dsupice_nonzero_points_j1", ds, j1),
                                  ("dtfreez_nonzero_points_i1", dtf, i1),
                                  ("dtfreez_nonzero_points_j1", dtf, j1)):
            n = sum(int(np.count_nonzero(a[cut])) for a in arrays)
            assert n > 0 and n == acc[name], (b, name)
        counts = next(fx.iter_probe(f"kc2edge_probe_{b}", types=["counts_tile"]))
        assert int(np.asarray(counts["icenucl_calls"])[0]) > 0          # ICE_NUCL in the corner


def test_kc2_case_definition_is_published(manifest):
    d = manifest["kc2"]["definition"]
    assert d["time_step_s"] == 12.0 and d["steps"] == 300
    assert d["replay_steps"] == [2, 12, 30, 60, 120, 200, 300]
    assert d["probe_steps"] == [10, 60, 150, 300] and len(d["probe_levels"]) == 8
    case = fx.load_fixture("kc2_case")
    dz = case["case.dz"][0]
    assert dz.shape == (50,) and dz[0] == 134.0
    assert np.allclose(dz[1:] / dz[:-1], 1.025, rtol=0, atol=1e-12)
    assert 12900.0 < case["case.z_face"][0][-1] < 13200.0
    assert case["case.domain"][0].tolist() == [1, 5, 1, 5, 1, 51]
    assert case["case.tile"][0].tolist() == [2, 4, 2, 4, 1, 50]
    types = case["case.column_type"][0]            # (i, j), 1-based as a[i-1, j-1]
    assert types[1, 1] == 1 and types[2, 2] == 2 and types[3, 3] == 3 and types[3, 1] == 4
    assert int((types != 0).sum()) == 4
    # u and v vary across the domain, so the pre-pass's averaged and edge winds differ.
    u = case["case.u"][0]
    assert u[0, 0, 0] != u[1, 0, 0] and u[0, 0, 0] != u[0, 1, 0]
    assert case["case.t_surface"][0].max() == 300.0
    # Only the bins, built on the table masses as each build reads them, differ.
    assert sorted(manifest["kc2"]["case_fields_that_differ_between_builds"]) == [
        "case.bins_mass_mixing_ratio", "case.xl_used_for_bins"]


def test_kc2_replay_probe_and_graupel_iterators(manifest):
    steps = []
    for step, pre, post in fx.iter_replay("kc2_replay_o64"):
        steps.append(step)
        assert pre["chem"].shape == (3, 50, 3, 142) and post["chem"].shape == (3, 50, 3, 142)
        assert pre["th"].shape == (4, 50, 4) and pre["th_old"].shape == (4, 50, 4)
        assert pre["w"].shape == (3, 51, 3)
    assert steps == manifest["kc2"]["definition"]["replay_steps"]
    cross = [s for s, pre, _ in fx.iter_replay("kc2_crossreplay_o64_on_o32")]
    assert cross == steps
    assert [s for s, _, _ in fx.iter_replay("kc2g_replay_o32")] == [2, 150, 300]
    kinds = {r.type for r in fx.iter_probe("kc2_probe_o64", step=150)}
    assert {"prepass", "entry", "nucl_ice", "icenucl_before", "icenucl_after", "cond2_before",
            "cond2_after", "cond3_before", "cond3_after", "supsat", "coal_ice_before",
            "coal_cold_before", "coal_cold_after", "rime", "freez_before", "freez_after",
            "melt_before", "melt_after", "merge", "snowbrk", "exit"} <= kinds
    assert any(r.type == "hm" for r in fx.iter_probe("kc2_probe_o64"))
    tile = next(fx.iter_probe("kc2_probe_o64", types=["tile_config"]))
    assert tile["column_recorded"].tolist() == [1, 0, 1, 0, 1, 0, 0, 0, 1]
    pre = next(fx.iter_probe("kc2_probe_o64", types=["prepass"]))
    assert pre["dsupice_xyz"].shape == (3, 50, 3) and pre["t_old"].shape == (4, 50, 4)


def test_cold_refusals(manifest):
    refusals = list(fx.iter_refusals(name="refusals_cold"))
    assert {(r.case, r.build) for r in refusals} == {(n, b) for n in range(1, 14)
                                                     for b in ("o64", "o32")}
    members = {(m["case"], m["build"]): m for m in manifest["files"]["refusals_cold"]["members"]}
    assert set(members) == {(n, b) for n in range(1, 14) for b in ("o64", "o32")}
    # Every reachable fatal check of the cold routines, ONECOND2's two
    # negative-substep checks included: :7452 on KCOND 21 (case 4) and :7535
    # on KCOND 20 (case 12).
    lines = {1: 3463, 2: 2750, 3: 7441, 4: 7452, 5: 7694, 6: 7993, 7: 8049, 8: 8239, 9: 8599,
             10: 8636, 11: 8614, 12: 7535, 13: 3463}
    for (n, b), m in members.items():
        if n == 1:
            # ICE_NUCL's bare STOP 099 through a non-finite change: exit 99 with
            # no refusal record; unreachable from a float32 grid, so the native
            # build returns.
            if b == "o64":
                assert m["stop_kind"] == "bare STOP 099" and m["exit_status"] == 99
                assert m["message"] == "" and m["wrf_line"] == 3463
            else:
                assert m["stop_kind"] == "returned" and m["exit_status"] == 3
            continue
        if n == 13:
            # The same stop on finite mixed-sign bins, in both builds.
            assert m["stop_kind"] == "bare STOP 099" and m["exit_status"] == 99, (n, b)
            assert m["message"] == "" and m["wrf_line"] == 3463
            assert m["reachable_from_fast_sbm"] is False
            continue
        assert m["stop_kind"] == "wrf_error_fatal" and m["exit_status"] == 99, (n, b)
        assert m["wrf_line"] == lines[n], (n, b)
        assert m["reachable_from_fast_sbm"] is False
    # The M1 container is untouched by the new one, and the reader returns
    # the cold container by name.
    assert len(list(fx.iter_refusals())) == 12
    arrays = fx.load_fixture("refusals_cold")
    for n in range(1, 14):
        for b in ("o64", "o32"):
            want = 3 if (n, b) == (1, "o32") else 99
            assert int(arrays[f"case{n}_{b}.exit_status"]) == want, (n, b)
    assert set(arrays) != set(fx.load_fixture("refusals"))


def test_cold_sweeps_reach_their_branches():
    ice = _records("sweep_ice_nucl_o64", "case")
    bands = {int(r["ice_nucl_band"]) for r in ice if int(r["ice_nucl_calls"]) == 1}
    assert bands == {1, 2, 3, 4, 5, 6}
    assert any(int(r["ice_nucl_calls"]) == 0 for r in ice)          # above 0 C, below -37 C
    added = [int(r["ice_nucl_added"]) for r in ice if int(r["ice_nucl_band"]) == 1]
    assert added and not any(added)                                  # HELEK1 = 0 above -5 C
    modk = _records("sweep_modkrn_o64", "case")
    unset = sorted(float(r["tt"]) for r in modk if float(r["factor_out"]) == -1.0)
    assert len(unset) == 3 and unset[0] < 233.16 and unset[-1] == 278.0   # -40 C and D6
    melt = _records("sweep_melt_o64", "case")
    assert any(np.any(r["ff_out"] < 0) for r in melt)                # D12
    assert all(int(r["grids_unchanged"]) == 1 for r in melt)
    c2 = _records("sweep_onecond2_o64", "case")
    assert {int(r["supsat_kcond"]) for r in c2} == {20, 21}
    assert {int(r["supsat_branch"]) for r in c2} == {0, 4, 5}
    c3 = _records("sweep_onecond3_o64", "case")
    assert {int(r["supsat_kcond"]) for r in c3 if int(r["supsat_calls"])} == {30, 31, 32}
    assert {2, 6, 7} <= {int(r["supsat_branch"]) for r in c3}
    assert sum(1 for r in c3 if int(r["supsat_calls"]) == 0) >= 27     # KCOND 50 and the skip
    for fam, live in (("sweep_coal_cold_o64", {1, 3, 4, 10, 11, 12}),
                      ("sweep_coal_cold_graupel_o64", {2, 3, 5, 6, 7, 12})):
        recs = _records(fam, "case")
        mask = 0
        for r in recs:
            mask |= int(r["cold_call_mask"])
        assert {i + 1 for i in range(12) if mask >> i & 1} == live, fam
        assert sum(int(r["hm_fired"]) for r in recs) > 0
        alwc = {float(r["cold_scalars"][0]) for r in recs}
        assert min(a for a in alwc if a > 0) < 0.5 < max(alwc)       # both sides of ALCR
    jn = _records("sweep_jernucl01_ice_o64", "case")
    assert len(jn) == 48 and sum(int(r["homfrz_calls"]) for r in jn) > 0


# The old port's live cases each family carries, in its set 1.
CARRIED_CASES = {"sweep_freez": 20, "sweep_melt": 20, "sweep_ice_nucl": 18,
                 "sweep_modkrn": 12, "sweep_jernucl01_ice": 48}


@pytest.mark.parametrize("build", ["o64", "o32"])
def test_cold_sweeps_carry_their_cases_and_branches(build):
    for fam, n in CARRIED_CASES.items():
        recs = _records(f"{fam}_{build}", "case")
        assert sum(1 for r in recs if int(r["set"]) == 1) == n, fam
    # FREEZ (:6560-6636): drops freeze only below 0 C, into plates (habit 2,
    # species column 2) at bins up to KRFREEZ = 21 and into hail (column 6)
    # above it; it never writes the grids.
    to_plates = to_hail = 0
    for r in _records(f"sweep_freez_{build}", "case"):
        fin, fout = np.asarray(r["ff_in"]), np.asarray(r["ff_out"])
        gained = fout > fin
        assert int(r["grids_unchanged"]) == 1
        if float(r["t_in"]) >= 273.15:
            assert np.array_equal(fin, fout), int(r["case_id"])
            continue
        assert not gained[:, [1, 3, 4, 5]].any(), int(r["case_id"])
        assert not gained[21:, 2].any() and not gained[:21, 6].any(), int(r["case_id"])
        to_plates += int(gained[:21, 2].any())
        to_hail += int(gained[21:, 6].any())
    assert to_plates > 0 and to_hail > 0
    # The melt sweep's dt = 200 s cases drive a bin negative (D12).
    assert any(float(r["dt"]) == 200.0 and np.any(np.asarray(r["ff_out"]) < 0)
               for r in _records(f"sweep_melt_{build}", "case") if int(r["set"]) == 1)
    # JERRATE_KS ice: all seven species, dense ice with fl 0 and fl > 0, and
    # each species on both sides of 273.15 K (the snow nskin switch).
    jr = _records(f"sweep_jerrate_ice_{build}", "case")
    assert {int(r["species"]) for r in jr} == set(range(1, 8))
    assert {float(np.max(r["fl"])) for r in jr} == {0.0, 0.5}
    for sp in range(1, 8):
        tps = {float(r["tp"]) for r in jr if int(r["species"]) == sp}
        assert min(tps) < 273.15 < max(tps), sp
    # JERDFUN_KS ice: the (IND, ITYPE) pairs the sweep exists for, and set 3's
    # negative bin dropped by the remap without a refusal.
    jd = _records(f"sweep_jerdfun_ice_{build}", "case")
    assert {(int(r["ind"]), int(r["itype"])) for r in jd} == {
        (3, 1), (3, 2), (3, 3), (1, 3), (1, 4), (1, 5), (2, 3)}
    s3 = [r for r in jd if int(r["set"]) == 3]
    assert len(s3) == 1 and np.min(s3[0]["fi"]) < 0 <= np.min(s3[0]["psi"])
    # Cold COAL_BOTT_NEW: colleff is recorded only where modkrn_KS set it.
    for fam in (f"sweep_coal_cold_{build}", f"sweep_coal_cold_graupel_{build}"):
        recs = _records(fam, "case")
        assert 1 in {int(r["colleff_defined"]) for r in recs}, fam
        for r in recs:
            if int(r["colleff_defined"]) == 0:
                assert float(r["colleff"]) == 0.0, fam


def test_orientation_cases_see_a_swapped_grid():
    # On y = 1.25 x each routine's output differs from the same case on y = x,
    # so a host that swaps x and y fails; on WRF's own equal grids it could not.
    recs = _records("sweep_coll_cold_o64", "orient")
    assert len(recs) == 72
    by = {(int(r["op"]), int(r["indc"]), int(r["seed"]), int(r["set"])): r for r in recs}
    assert len(by) == 72 and {k[3] for k in by} == {1, 2, 3}
    # Set 3 runs the same draws on the real tables (x = y = xl_mg, the
    # module's chucm and ima). Every call moves mass between the populations
    # and keeps their total, so a record that is not a collision fails.
    seeds = {int(d["seed"]): d for d in _records("sweep_coll_cold_o64", "draws")}
    for (op, indc, seed, grid), r in by.items():
        g_in = np.asarray(seeds[seed]["g"], dtype=np.float64)
        g_out = np.asarray(r["g_out"], dtype=np.float64)
        assert np.all(g_out >= 0.0), (op, indc, seed, grid)
        assert abs(g_out.sum() - g_in.sum()) <= 1e-12 * g_in.sum(), (op, indc, seed, grid)
        if grid == 3:
            assert not np.array_equal(g_out, g_in), (op, indc, seed)
            assert not np.array_equal(r["g_out"], by[(op, indc, seed, 2)]["g_out"])
    for (op, indc, seed, grid), r in by.items():
        if grid == 1:
            other = by[(op, indc, seed, 2)]
            assert not np.array_equal(r["g_out"], other["g_out"]), (op, indc, seed)
    draws = _records("sweep_coll_cold_o64", "draws")
    assert [int(d["seed"]) for d in draws] == [0, 71, 2701, 7753]
    ck = draws[0]["ck"]
    assert not np.array_equal(ck, ck.T)                               # a non-symmetric kernel


def test_cold_findings_are_published(manifest):
    f = manifest["findings_cold"]
    for key in ("ice_nucl_stop_099_unreachable", "modkrn_unset_at_minus_40",
                "jerdfun_negative_bin_with_growth", "deldrop_reached_only_through_the_gmin_floor",
                "snow_breakup_guard_always_true", "onecond2_sign_flip_checks_unreachable",
                "supsat_small_ratio_branches_unreachable_from_onecond3",
                "graupel_build_dead_calls", "anti_bergeron_regime"):
        assert key in f and chr(0x2014) not in f[key]      # plain sentences
    # The STOP 099 finding names both refusal cases and why FAST_SBM cannot
    # reach the stop (it passes ICE_NUCL no crystals).
    stop = f["ice_nucl_stop_099_unreachable"]
    assert "refusal case 13" in stop and "refusal case 1," in stop
    assert ":4548-4550" in stop and ":4667-4674" in stop


def test_oracle_directory_is_within_the_design_budget():
    total = sum(p.stat().st_size for p in ORACLE.iterdir() if p.is_file())
    assert total <= SIZE_LIMIT
