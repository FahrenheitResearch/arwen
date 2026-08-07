"""Generate the GFS-versus-HRRR background A/B plan, arm by arm.

The experiment is one question -- does starting WaH from a
convection-allowing background make it better -- asked three ways,
because asking it once cannot separate the three things that change when
GFS is swapped for HRRR: grid spacing, background AGE, and whether the
first guess already contains storms.

    G-gfs        GFS 0.25 deg, cycle 00Z, f004..f010, init 04Z.
                 The published configuration, argument for argument, on
                 the case the published numbers came from.  It exists to
                 REPRODUCE 0.7274 / 0.7557 / 0.7655 / 0.7376 / 0.7316 /
                 0.7239 against control 0.2360 -> 0.3410.  If it does
                 not, something moved underneath the comparison and no
                 HRRR number from this run means anything.

    H-matched    HRRR 3 km, cycle 00Z, f004..f008, init 04Z.
                 THE HEADLINE.  Same init, same forecast age, same
                 hourly forcing interval, same grid, same eta ladder,
                 same physics, same observations, same seed.  The
                 background's AGE is held fixed, so what is left is the
                 model that made it: 25 km global with explicitly zero
                 condensate against 3 km convection-allowing with
                 QC/QI/QR/QS/QG decoded natively.

    H-fresh      HRRR 3 km, cycle 04Z, f000..f004, init 04Z.
                 What an hourly cadence actually offers a user, and
                 deliberately CONFOUNDED: it is four hours fresher AND
                 it is an analysis into which NOAA has already
                 assimilated radar.  Read against H-matched, not against
                 G-gfs, or the answer credits our filter with someone
                 else's.

Everything the three arms share is shared by construction rather than by
care.  The observations are one set of files, gridded once, passed to
every arm byte-identically.  The prepared cases are proved to carry the
same domain identity and the same vertical grid BEFORE any of them is
built (``build_case_inputs.py``) and their static geography is compared
after (``check_case_parity.py``).  The perturbation, the localization,
the relaxation, the seed and the leg schedule are copied from the
published run's own recorded arguments.

Two arms cost nothing on the card and say so (``needs_gpu: false``), so
the four-and-a-bit gigabytes of GRIB2 and the scoring do not sit on a
GPU that somebody else is queued for.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PLAN_SCHEMA = "gpuwm-da.sweep-plan.v1"

#: The published run's own arguments, read off its recorded report and
#: restated here so one edit changes every arm at once.  These are the
#: things that must NOT differ between arms.
SHARED_CYCLE_ARGS = (
    "--physics-profile", "wsm6-ysu-mm5-noah-no-radiation-v1",
    "--history-interval-seconds", "900.0",
    "--leg-seconds", "900.0",
    "--free-legs", "6",
    "--save-composites",
    "--members", "10",
    "--seed", "20260805",
    "--wind-sigma-ms", "1.5",
    "--length-scale-km", "50.0",
    "--horizontal-loc-m", "12000.0",
    "--vertical-loc-m", "3000.0",
    "--rtps-alpha", "0.9",
    "--relaxation", "rtps",
    "--thin-cells", "2",
    "--err-inflation", "1.0",
    "--memory-budget-mib", "6144.0",
    "--solve-device", "cuda",
)

#: The six analysis times and the six verification times, as HHMM.
ANALYSIS_TIMES = ("0415", "0430", "0445", "0500", "0515", "0530")
VERIFY_TIMES = ("0545", "0600", "0615", "0630", "0645", "0700")

#: The published anchor this A/B lands beside.  Carried in the plan so a
#: reader of the plan alone can tell what "reproduces" would mean.
PUBLISHED_ANCHOR = {
    "run": "evidence/da-demo/live-fire-3",
    "fss30_27km_ensemble_mean": [0.7274, 0.7557, 0.7655, 0.7376, 0.7316,
                                 0.7239],
    "fss30_27km_control": [0.2360, 0.2719, 0.3173, 0.3156, 0.3205, 0.3410],
    "control_storm_columns": [519, 715, 881, 1006, 1090, 1162],
    "observed_storm_columns": [2817, 3169, 3519, 3667, 3857, 3991],
}


def gfs_pins(prepared_root: Path) -> dict[str, str]:
    """The GFS arm's three digests, READ from its own prepared case.

    Typing them would be one transcription away from binding the wrong
    case, and the wrong case would still run -- it would just answer a
    different question.  Every one of them is derivable from the tree the
    arm will actually open: the proof's file digest, and the manifest and
    content digests the cache header already carries.
    """

    import hashlib

    proof = prepared_root / "proof.json"
    digest = hashlib.sha256()
    with proof.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    header = json.loads(
        (prepared_root / "prepared-cache" / "header.json").read_text(
            encoding="utf-8"))
    return {
        "proof_sha256": digest.hexdigest(),
        "source_manifest_sha256":
            header["identity"]["bridge_manifest_sha256"],
        "prepared_content_sha256": header["content_sha256"],
    }


def hrrr_arms() -> list[dict]:
    return [
        {"key": "matched", "cycle": "2026-08-05_00:00:00",
         "forecast_start_hour": 4,
         "leads": "4,5,6,7,8", "grib": "00z",
         "why": ("the background's age held at the GFS arm's four hours, "
                 "so the only thing left is the model that made it")},
        {"key": "fresh", "cycle": "2026-08-05_04:00:00",
         "forecast_start_hour": 0,
         "leads": "0,1,2,3,4", "grib": "04z",
         "why": ("what the hourly cadence actually offers; four hours "
                 "fresher AND already radar-assimilated by NOAA, so it "
                 "is read against H-matched, never against G-gfs")},
    ]


def build(*, case_root: str, geog_root: str, run_seconds_hrrr: float,
          run_seconds_gfs: float, gfs_pins: dict) -> dict:
    # Paths are joined with pathlib rather than spelled, so the plan a
    # venue generates is the plan that venue can read.  A prepared case
    # is host-bound; a plan full of another host's separators would be
    # a second, silent way to be on the wrong machine.
    root = Path(case_root)
    authority = str(root / "go" / "authority")
    prepared = str(root / "go" / "prepared")
    obs = [str(root / "obs" / f"kdmx-live-{hhmm}.nc")
           for hhmm in ANALYSIS_TIMES]
    grids = [str(root / "go" / "run" / "wrfout"
                 / f"wrfout_d01_2026-08-05_{hhmm[:2]}_{hhmm[2:]}_00")
             for hhmm in ANALYSIS_TIMES]
    verify = [str(root / "obsverify" / f"kdmx-verify-{hhmm}.nc")
              for hhmm in VERIFY_TIMES]

    def obs_pairs() -> list[str]:
        argv: list[str] = []
        for observation, grid in zip(obs, grids):
            argv.extend(("--obs", observation, "--grid-wrfout", grid))
        return argv

    arms: list[dict] = []

    # ---- 1-2. the GRIB2, which never touches the card -------------------
    for entry in hrrr_arms():
        arms.append({
            "name": f"fetch-hrrr-{entry['grib']}",
            "needs_gpu": False,
            "what": (f"HRRR {entry['cycle'][:13]}Z leads {entry['leads']}, "
                     "field-subset by published byte range; measured at "
                     "435 MB per forecast hour, not the 1.1 GB the whole "
                     "object would cost"),
            "steps": [{
                "name": "fetch",
                "argv": ["${PYTHON}", "-m", "tools.download_hrrr_native_subset",
                         "--cycle", entry["cycle"],
                         "--forecast-hours", entry["leads"],
                         "--output-root",
                         f"${{RUN_DIR}}/hrrr/grib/{entry['grib']}"],
            }],
        })

    # ---- 3. the case inputs, and the refusal that guards the whole A/B --
    arms.append({
        "name": "case-inputs",
        "needs_gpu": False,
        "what": ("derive the HRRR target domain and namelist.input FROM "
                 "the GFS arm's own authority, and REFUSE unless the "
                 "prepared-domain identity and the eta ladder match it "
                 "exactly.  This runs on the venue host, so a venue that "
                 "would have produced a different case says so here "
                 "rather than after the GPU time is spent"),
        "steps": [{
            "name": "inputs",
            "argv": ["${PYTHON}", "-m",
                     "tools.da_background_ab.build_case_inputs",
                     "--gfs-authority", authority,
                     "--gfs-prepared-root", prepared,
                     "--name", "kdmx_da_ab_132x132x49",
                     "--valid-time", "2026-08-05_04:00:00",
                     "--run-seconds", str(run_seconds_hrrr),
                     "--history-interval-seconds", "900.0",
                     "--physics-profile",
                     "wsm6-ysu-mm5-noah-no-radiation-v1",
                     "--out-domain-spec",
                     "${RUN_DIR}/inputs/kdmx_target_domain.json",
                     "--out-namelist-input",
                     "${RUN_DIR}/inputs/kdmx_namelist.input",
                     "--out-receipt",
                     "${RUN_DIR}/inputs/case-equivalence.json"],
        }],
    })

    # ---- 4-5. the two HRRR preparations ---------------------------------
    for entry in hrrr_arms():
        key = entry["key"]
        arms.append({
            "name": f"prepare-hrrr-{key}",
            "needs_gpu": True,
            "what": (f"native HRRR preparation through the shipped front "
                     f"door (gpuwm.source_cli --source hrrr), cycle "
                     f"{entry['cycle'][:13]}Z f{entry['forecast_start_hour']:03d}; "
                     "--wps-namelist is the opt-in that publishes the "
                     "three portable authorities the cycling driver binds"),
            "steps": [{
                "name": "prepare",
                "argv": ["${PYTHON}", "-m",
                         "tools.da_background_ab.prepare_case",
                         "--manifest",
                         f"${{RUN_DIR}}/hrrr/grib/{entry['grib']}/SHA256SUMS",
                         "--",
                         "--source", "hrrr",
                         "--source-root",
                         f"${{RUN_DIR}}/hrrr/grib/{entry['grib']}",
                         "--source-sha256s",
                         f"${{RUN_DIR}}/hrrr/grib/{entry['grib']}/SHA256SUMS",
                         "--namelist-input",
                         "${RUN_DIR}/inputs/kdmx_namelist.input",
                         "--domain-spec",
                         "${RUN_DIR}/inputs/kdmx_target_domain.json",
                         "--wps-namelist",
                         str(root / "go" / "authority" / "namelist.wps"),
                         "--geog-root", geog_root,
                         "--physics-profile",
                         "wsm6-ysu-mm5-noah-no-radiation-v1",
                         "--valid-time", entry["cycle"],
                         "--forecast-start-hour",
                         str(entry["forecast_start_hour"]),
                         "--run-seconds", str(int(run_seconds_hrrr)),
                         "--history-interval-seconds", "900.0",
                         "--output-root", f"${{RUN_DIR}}/hrrr/case-{key}"],
            }],
        })

    # ---- 6. arrays, once both cases exist -------------------------------
    arms.append({
        "name": "case-parity",
        "needs_gpu": False,
        "what": ("static geography must be identical across arms and the "
                 "initial layer heights must not be; both are measured "
                 "rather than assumed, and the height difference is the "
                 "stated price of one shared observation set"),
        "steps": [{
            "name": "parity",
            "argv": ["${PYTHON}", "-m",
                     "tools.da_background_ab.check_case_parity",
                     "--reference-prepared-root", prepared,
                     "--other-prepared-root",
                     "matched=${RUN_DIR}/hrrr/case-matched",
                     "--other-prepared-root",
                     "fresh=${RUN_DIR}/hrrr/case-fresh",
                     "--out", "${RUN_DIR}/results/case-parity.json"],
        }],
    })

    # ---- 7. the GFS arm, reproducing the published anchor ---------------
    arms.append({
        "name": "G-gfs",
        "needs_gpu": True,
        "members": 10,
        "what": ("the published configuration on the published case; it "
                 "has to reproduce the anchor or every other arm is "
                 "suspect"),
        "steps": [{
            "name": "cycle",
            "argv": ["${PYTHON}", "-m", "tools.da_background_ab.run_arm", "--",
                     "--source", "gfs",
                     "--prepared-root", prepared,
                     "--authority-dir", authority,
                     "--proof-sha256", gfs_pins["proof_sha256"],
                     "--source-manifest-sha256",
                     gfs_pins["source_manifest_sha256"],
                     "--prepared-content-sha256",
                     gfs_pins["prepared_content_sha256"],
                     "--run-seconds", str(run_seconds_gfs),
                     *SHARED_CYCLE_ARGS,
                     *obs_pairs(),
                     "--out", "${RUN_DIR}/cases/G-gfs/cycle-out",
                     "--stage-dir", "${RUN_DIR}/cases/G-gfs/stage"],
        }],
    })

    # ---- 8-9. the two HRRR arms -----------------------------------------
    for entry in hrrr_arms():
        key = entry["key"]
        arms.append({
            "name": f"H-{key}",
            "needs_gpu": True,
            "members": 10,
            "what": entry["why"],
            "steps": [{
                "name": "cycle",
                "argv": ["${PYTHON}", "-m", "tools.da_background_ab.run_arm",
                         "--pins-from",
                         f"${{RUN_DIR}}/hrrr/case-{key}/"
                         "public-wrapper-result.json",
                         "--",
                         "--source", "hrrr",
                         "--prepared-root", f"${{RUN_DIR}}/hrrr/case-{key}",
                         "--authority-dir", f"${{RUN_DIR}}/hrrr/case-{key}",
                         "--run-seconds", str(run_seconds_hrrr),
                         *SHARED_CYCLE_ARGS,
                         *obs_pairs(),
                         "--out", f"${{RUN_DIR}}/cases/H-{key}/cycle-out",
                         "--stage-dir", f"${{RUN_DIR}}/cases/H-{key}/stage"],
            }],
        })

    # ---- 10. the answer --------------------------------------------------
    score_argv = ["${PYTHON}", "-m",
                  "tools.da_background_ab.score_background_ab",
                  "--arm", "gfs=${RUN_DIR}/cases/G-gfs/cycle-out",
                  "--arm", "hrrr-matched=${RUN_DIR}/cases/H-matched/cycle-out",
                  "--arm", "hrrr-fresh=${RUN_DIR}/cases/H-fresh/cycle-out",
                  "--baseline-arm", "gfs",
                  "--first-free-leg", "6", "--dx-km", "3.0"]
    for path in verify:
        score_argv.extend(("--obs", path))
    score_argv.extend(("--out", "${RUN_DIR}/results/background-ab.json"))
    arms.append({
        "name": "analyse",
        "needs_gpu": False,
        "what": ("the FSS-versus-neighborhood ladder for every arm, both "
                 "controls, the spread trajectory, the increment "
                 "magnitudes, and the falsification verdict"),
        "steps": [{"name": "score", "argv": score_argv}],
    })

    return {
        "schema": PLAN_SCHEMA,
        "experiment": "gpuwm-da.background-ab",
        "question": ("does a convection-allowing background improve WaH's "
                     "skill, and if so is it the assimilation or the "
                     "background's own forecast that improved"),
        "case": {
            "site": "KDMX",
            "init": "2026-08-05T04:00:00Z",
            "window_end": "2026-08-05T05:30:00Z",
            "cycles": 6, "cycle_seconds": 900, "free_legs": 6,
            "free_forecast_minutes": 90,
            "verification_times": [f"2026-08-05T{t[:2]}:{t[2:]}Z"
                                   for t in VERIFY_TIMES],
            "prepared_case_root": case_root,
            "physics_profile": "wsm6-ysu-mm5-noah-no-radiation-v1",
            "observations": ("one set of files, gridded once onto the GFS "
                             "arm's georeference trajectory, passed to "
                             "every arm byte-identically"),
        },
        "metric": {
            "function": "gpuwm.verify.field_metrics.fss_distance",
            "threshold_dbz": 30.0,
            "published_box_km_across": 27.0,
            "convention": ("a square SIDE LENGTH, not a radius; a rung of "
                           "half_width h scores a box (2h+1) cells across, "
                           "and the FSS smooths the truth field as well as "
                           "the forecast"),
            "ladder_half_widths": [0, 1, 2, 3, 4, 6, 8],
            "ladder_km": [3.0, 9.0, 15.0, 21.0, 27.0, 39.0, 51.0],
            "scored_field": ("ensemble MEAN column-max reflectivity, which "
                             "is what the published figure scored; the "
                             "per-member distribution at the 27 km rung is "
                             "reported beside it"),
            "ladder_method": ("the neighborhood ladder from "
                              "tools/ens_sweep/score_resolution.py, built "
                              "there for the 3 km / 1.5 km resolution "
                              "comparison, reused here"),
            "published_anchor": PUBLISHED_ANCHOR,
        },
        "falsified_if": [
            "no rung of the ladder improves -- the FSS difference between "
            "H-matched and G-gfs is <= 0 at every neighborhood from 3 to "
            "51 km, over the mean of the six verification times",
            "the never-analysed control gains at least as much as the "
            "analysed ensemble at every rung: then the background's own "
            "forecast improved and WaH's skill did not",
            "skill improves while the ensemble becomes LESS able to "
            "explain its own innovations than the GFS arm's -- a "
            "collapsing ensemble makes a smoother mean, which FSS "
            "rewards, and that is not a better forecast system",
            "skill improves and the analysis increments do NOT shrink: "
            "then the first guess was not closer to the observations and "
            "the claimed mechanism is wrong, whatever the headline says",
        ],
        "honesty": [
            "Every arm is a single draw.  No arm is repeated, so none of "
            "these numbers carries an error bar, and the standing no-ECC "
            "dual-run screen is NOT applied to any of them.",
            "H-fresh confounds three things at once (age, resolution, and "
            "NOAA's own radar assimilation in the HRRR analysis) and is "
            "reported as such.  H-matched is the attributable arm.",
            "The perturbation amplitudes were tuned against a storm-free "
            "GFS first guess and are NOT retuned here.  On a background "
            "that already contains balanced convection an unbalanced "
            "50 km wind perturbation is a cruder instrument, and it may "
            "make HRRR look worse before it looks better.",
            "The GFS arm is already 8x to 24x under-dispersive in "
            "observation space on this case (innovation_rms^2 over "
            "spread^2 + obs_error^2 runs 7.8 -> 24.5 across the six "
            "analyses).  The spread bar is not carried over from it "
            "unexamined; the ratio is reported per arm per cycle.",
            "The observations are bound to the GFS arm's georeference, so "
            "a gate sits in the layer the GFS column put it in.  "
            "check_case_parity.py measures that offset as a fraction of "
            "the local layer depth.",
            "Radial velocity is masked for aliasing risk and never "
            "dealiased, equally in every arm.",
        ],
        "arms": arms,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.da_background_ab.build_plan",
        description=__doc__.splitlines()[0])
    parser.add_argument("--case-root", required=True,
                        help="the GFS arm's case directory (host-bound)")
    parser.add_argument("--geog-root", required=True)
    parser.add_argument("--run-seconds-gfs", type=float, default=21600.0,
                        help="the GFS case's own hash-bound run length")
    parser.add_argument("--run-seconds-hrrr", type=float, default=14400.0,
                        help="the HRRR cases' run length.  The legs sum to "
                             "10800 s, so 14400 carries an hour of margin "
                             "on five forecast hours instead of seven -- "
                             "two fewer HRRR hours to fetch and store, and "
                             "not one second of different integration")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    plan = build(
        case_root=args.case_root, geog_root=args.geog_root,
        run_seconds_hrrr=args.run_seconds_hrrr,
        run_seconds_gfs=args.run_seconds_gfs,
        gfs_pins=gfs_pins(Path(args.case_root) / "go" / "prepared"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=1) + "\n",
                        encoding="utf-8", newline="\n")
    print(f"{len(plan['arms'])} arms, "
          f"{sum(len(arm['steps']) for arm in plan['arms'])} steps -> "
          f"{args.out}")
    for arm in plan["arms"]:
        print(f"  {'GPU' if arm.get('needs_gpu', True) else '   '} "
              f"{arm['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
