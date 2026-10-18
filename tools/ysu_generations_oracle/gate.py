"""Compare every column output word and state increment at three steps.

The reference is the unmodified release. Mixed BEP momentum retains the
previously declared rural drag correction; every other word must match.

The release legs replay the committed references (tests/data/oracles/
ysu_generations, written by `--fixture record` from the pinned public
sources); `--fixture verify` rebuilds them and requires the same words.
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import re
import hashlib
import json
import lzma
import subprocess

import numpy as np

from gpuwm.verify.ysu_wide_oracle import (load_wide_oracle, run_port_wide,
    LEVEL_FIELDS, SURFACE_FIELDS, TOPO_FIELDS)
from tools.ysu_wrf461_oracle.make_wide_columns import write_records


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def grade(cols, port, *, step, allow_declarations=True):
    stats, cells, states = {}, [], {}
    for c in cols:
        fields = {**LEVEL_FIELDS, **SURFACE_FIELDS, "kpbl": "kpbl"}
        if c.topo == 1:
            fields.update(TOPO_FIELDS)
        got = port[c.index]
        for name, wname in fields.items():
            want = (np.asarray([c.wrf_kpbl], np.int32) if name == "kpbl"
                    else c.wrf[wname])
            a = np.ascontiguousarray(got[name]).reshape(-1)
            b = np.ascontiguousarray(want).reshape(-1)
            assert a.dtype.itemsize == b.dtype.itemsize == 4
            different = a.view(np.uint32) != b.view(np.uint32)
            declared = bool(allow_declarations and c.topo == 2 and c.scalars["_pad"] != np.float32(1.)
                            and name in ("du", "dv"))
            row = stats.setdefault(name, {"words": 0, "different": 0,
                                          "declared": 0, "unexplained": 0})
            row["words"] += len(a)
            row["different"] += int(different.sum())
            row["declared" if declared else "unexplained"] += int(different.sum())
            for k in np.flatnonzero(different):
                cells.append({"step": step, "column": c.index, "field": name,
                              "level": int(k), "got": int(a.view(np.uint32)[k]),
                              "want": int(b.view(np.uint32)[k]), "declared": declared})
        for state, tendency in (("u", "du"), ("v", "dv"), ("theta", "dtheta"),
                                ("qv", "dqv"), ("qc", "dqc"), ("qi", "dqi")):
            dt = np.float32(c.scalars["dt"])
            ref = np.float32(c.levels[state] + np.float32(c.wrf[LEVEL_FIELDS[tendency]] * dt))
            actual = np.float32(c.levels[state] + np.float32(got[tendency] * dt))
            different = actual.view(np.uint32) != ref.view(np.uint32)
            declared = (allow_declarations and c.topo == 2 and c.scalars["_pad"] != np.float32(1.) and state in ("u", "v"))
            row = states.setdefault(state, {"words": 0, "different": 0, "unexplained": 0})
            row["words"] += len(ref)
            row["different"] += int(different.sum())
            if not declared:
                row["unexplained"] += int(different.sum())
    return stats, states, cells


def records_after(cols):
    records = []
    for c in cols:
        rec = {**c.levels, "p_interface": c.p_interface, "nz": c.nz,
               "topdown": c.topdown, "regime": c.regime, "topo": c.topo,
               "scalars": c.scalars}
        for state, tendency in (("u", "utnp"), ("v", "vtnp"), ("theta", "rthblten"),
                                ("qv", "qvtnp"), ("qc", "qctnp"), ("qi", "qitnp")):
            rec[state] = np.float32(rec[state] + np.float32(c.wrf[tendency] * c.scalars["dt"]))
        rec["tx"] = np.float32(rec["theta"] * rec["pi"])
        records.append(rec)
    return records


#: The committed references: every word the unmodified releases wrote on the
#: corpus, three steps per generation.  `--fixture record` writes them from a
#: live build, `--fixture verify` rebuilds and compares them, and the default
#: `--fixture replay` grades the port against them with no Fortran toolchain,
#: so every GPU leg runs this gate.
FIXTURE = Path(__file__).resolve().parents[2] / "tests/data/oracles/ysu_generations"
TAGS = {"wrf_480": "v4.8.0", "wrf_461": "v4.6.1"}
#: The words each step's grade reads: the primary reference (bounded-patched
#: on four-level records), the BEP declared-correction reference, and the
#: traced top-down diagnostics.
STEP_FILES = ("ysu-wide-wrf.bin", "declared-reference.bin", "trace.txt")


def xz_read(path):
    return lzma.decompress(Path(path).read_bytes())


def xz_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(lzma.compress(data, preset=9 | lzma.PRESET_EXTREME))


def live_references(root, tag, work):
    """Run the pinned unmodified-release oracles on work's inputs."""
    inputs = work / "ysu-wide-inputs.bin"
    subprocess.run([str(root / tag / "scalar/oracle"), str(inputs), str(work / "ysu-wide-wrf.bin")], check=True, stdout=subprocess.DEVNULL)
    trace = subprocess.check_output([str(root / tag / "diagnostics/oracle"), str(inputs), str(work / "trace.bin")], text=True)
    cols = load_wide_oracle(work)
    # get_pblh can read the unwritten final TKE word on four-level grids.
    # Grade those records against the separately declared bounded source.
    # Every defined record must remain unchanged by diagnostic tracing.
    ordinary = (work / "ysu-wide-wrf.bin").read_bytes()
    traced = (work / "trace.bin").read_bytes()
    subprocess.run([str(root / tag / "bounded/oracle"), str(inputs), str(work / "bounded.bin")], check=True, stdout=subprocess.DEVNULL)
    bounded = (work / "bounded.bin").read_bytes()
    patched = bytearray(ordinary)
    position = 0
    for c in cols:
        end = position + 24 + 40 * c.nz
        if c.nz == 4:
            patched[position:end] = bounded[position:end]
        else:
            assert ordinary[position:end] == traced[position:end], f"tracing changed defined outputs in column {c.index}"
        position = end
    (work / "ysu-wide-wrf.bin").write_bytes(patched)
    subprocess.run([str(root / tag / "bep_corrected/oracle"), str(inputs), str(work / "declared-reference.bin")], check=True, stdout=subprocess.DEVNULL)
    (work / "trace.txt").write_text(trace)


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="work root; for record and verify it also holds columns.bin and the build.sh oracles")
    parser.add_argument("--version", default="wrf_480", choices=tuple(TAGS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plant", action="store_true")
    parser.add_argument("--fixture", choices=("replay", "record", "verify"), default="replay",
                        help="replay: grade against the committed references; record: write them "
                             "from the live oracles; verify: rebuild them and require the committed words")
    parser.add_argument("--fixture-dir", type=Path, default=FIXTURE)
    args = parser.parse_args(argv)
    root = args.root
    tag = TAGS[args.version]
    if args.plant:
        import gpuwm.core.ysu as launcher
        from gpuwm.core.kernels import module_source, compile_noftz_module
        source = module_source("ysu_480")
        term = "real ent_eff = ysu_min(0.2f * 8.0f * tmp1, 0.4f);"
        assert source.count(term) == 1
        # Flip one exponent bit: 0x3ecccccd XOR 0x00800000 = 0x3e4ccccd.
        # A lowest mantissa bit can be absorbed by later rounding in a column;
        # the source-expression oracle measures that bit directly.
        source = source.replace(term, term.replace("0.4f", "__uint_as_float(0x3e4ccccdu)"))
        module = compile_noftz_module("ysu_480", source, "ysu-480-planted-cap")
        original = launcher.get_kernel
        launcher.get_kernel = lambda name, function: module.get_function(function) if name == "ysu_480" else original(name, function)
    from gpuwm.wrf_exact import ENABLED
    summary = {"version": args.version, "plant": args.plant, "fixture": args.fixture,
               "strict_arithmetic": bool(ENABLED), "steps": [], "pass": True}
    args.out.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    fixture = args.fixture_dir
    columns = (xz_read(fixture / "columns.bin.xz") if args.fixture == "replay"
               else (root / "columns.bin").read_bytes())
    if args.fixture == "record":
        xz_write(fixture / "columns.bin.xz", columns)
    elif args.fixture == "verify":
        assert columns == xz_read(fixture / "columns.bin.xz"), "make_columns.py no longer writes the committed corpus"
    cols = None  # the previous step's WRF columns; step 0 reads the corpus
    for step in range(3):
        work = root / f"{args.version}-step{step}"
        work.mkdir(parents=True, exist_ok=True)
        inputs = work / "ysu-wide-inputs.bin"
        if step == 0:
            inputs.write_bytes(columns)
        else:
            write_records(records_after(cols), inputs)
        committed = fixture / args.version / f"step{step}"
        input_digest = hashlib.sha256(inputs.read_bytes()).hexdigest()
        if args.fixture == "replay":
            want = (committed / "inputs.sha256").read_text().split()[0]
            assert input_digest == want, \
                f"step {step} inputs differ from the ones the committed references were written on"
            for name in STEP_FILES:
                (work / name).write_bytes(xz_read(committed / f"{name}.xz"))
        else:
            live_references(root, tag, work)
            if args.fixture == "record":
                committed.mkdir(parents=True, exist_ok=True)
                (committed / "inputs.sha256").write_text(input_digest + "\n")
                for name in STEP_FILES:
                    xz_write(committed / f"{name}.xz", (work / name).read_bytes())
            else:
                assert input_digest == (committed / "inputs.sha256").read_text().split()[0], \
                    f"step {step} inputs differ from the committed ones"
                for name in STEP_FILES:
                    assert (work / name).read_bytes() == xz_read(committed / f"{name}.xz"), \
                        f"rebuilt {name} at step {step} differs from the committed reference"
        trace = (work / "trace.txt").read_text()
        cols = load_wide_oracle(work)
        declared_columns = [c.index for c in cols if c.nz == 4]
        port = run_port_wide(cols, ysu_version=args.version)
        stats, states, cells = grade(cols, port, step=step)
        primary = (work / "ysu-wide-wrf.bin").read_bytes()
        (work / "ysu-wide-wrf.bin").write_bytes((work / "declared-reference.bin").read_bytes())
        audit_cols = load_wide_oracle(work)
        audit_fields, audit_states, audit_cells = grade(audit_cols, port, step=step, allow_declarations=False)
        (work / "ysu-wide-wrf.bin").write_bytes(primary)
        active, capped, below, diagnostic_differences = 0, 0, 0, []
        column = None
        for line in trace.splitlines():
            if line.startswith("YSUCOL "):
                column = int(line.split()[1])
            elif line.startswith("YSUTD "):
                active += 1
                bits = [int(x, 16) for x in line.split()[1:]]
                tmp1 = np.asarray(bits[:1], np.uint32).view(np.float32)[0]
                capped += int(np.float32(np.float32(.2) * np.float32(8.) * tmp1) >= np.float32(.4))
                below += int(np.float32(np.float32(.2) * np.float32(8.) * tmp1) < np.float32(.4))
                for name, word in zip(("topdown_radsum", "wstar3_2", "cloudflg"), bits[2:]):
                    actual = np.ascontiguousarray(port[column][name]).view(np.uint32)[0]
                    if actual != word:
                        diagnostic_differences.append({"column": column, "field": name,
                                                        "got": int(actual), "want": word})
        # No active cloud block means these native diagnostics are defined zeros.
        active_columns = {int(m) for m in re.findall(r"YSUCOL (\d+)\nYSUTD", trace)}
        for c in cols:
            if c.index not in active_columns:
                for name in ("topdown_radsum", "wstar3_2", "cloudflg"):
                    word = int(np.ascontiguousarray(port[c.index][name]).view(np.uint32)[0])
                    if word != 0:
                        diagnostic_differences.append({"column": c.index, "field": name, "got": word, "want": 0})
        hashes = {name: digest(np.concatenate([port[c.index][name].reshape(-1) for c in cols]))
                  for name in port[0]}
        result = {"step": step, "columns": len(cols), "fields": stats, "states": states,
                  "declared_reference_audit": {"fields": audit_fields, "states": audit_states, "different_cells": audit_cells},
                  "declared_bounded_reference_columns": declared_columns,
                  "port_hashes": hashes, "nonfinite_reference_columns": [c.index for c in cols if not c.wrf_finite]}
        unexplained = sum(r["unexplained"] for r in stats.values()) + sum(r["unexplained"] for r in states.values())
        unexplained += len(diagnostic_differences)
        unexplained += sum(r["unexplained"] for r in audit_fields.values()) + sum(r["unexplained"] for r in audit_states.values())
        result["topdown_diagnostics"] = {"active_columns": active, "capped": capped,
            "below_cap": below, "differences": diagnostic_differences}
        if args.version == "wrf_480":
            assert active > 0 and capped > 0 and below > 0, "column corpus missed a side of the top-down cap"
        result["unexplained_words"] = unexplained
        summary["pass"] &= unexplained == 0
        summary["steps"].append(result)
        (args.out / f"cells-step{step}.json").write_text(json.dumps(cells, separators=(",", ":")) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"pass": summary["pass"], "version": args.version, "fixture": args.fixture,
                      "strict_arithmetic": bool(ENABLED),
                      "unexplained_words": [s["unexplained_words"] for s in summary["steps"]]}))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
