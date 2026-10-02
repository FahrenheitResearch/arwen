"""Attribute every original metric word moved by default dycore oracle fixes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.wrf_diffopt1_oracle.capture import stats


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def attribute(data, captures, output, merged_commit):
    controls = [{key: value for key, value in np.load(captures / f"control-{level}.npz").items()}
                for level in range(5)]
    originals = {}
    with np.load(data / "diff2-baseline.npz") as stored:
        originals.update({key: value for key, value in stored.items()})
    with np.load(data / "diff2-legacy.npz") as stored:
        originals.update({"legacy_" + key: value for key, value in stored.items()})
    assert originals.keys() == controls[0].keys()
    changes = ((4, 3, "46b0a09fe", "native acoustic damping constant"),
               (3, 2, "dc226e820", "native lid and mapped boundary advection"),
               (2, 1, "0673d8c51", "native outer-row geopotential and limiter rounding"),
               (1, 0, "83fde6032+2c212f621", "native diffusion and composed dry map coupling"))
    fields = []
    totals = {row[2]: 0 for row in changes}
    original_words = moved_words = 0
    for key, old in sorted(originals.items()):
        current = controls[0][key]
        np.testing.assert_array_equal(old.view("u4"), controls[4][key].view("u4"),
                                      err_msg="unattributed original words: " + key)
        changed = np.flatnonzero(old.view("u4").reshape(-1) != current.view("u4").reshape(-1))
        original_words += old.size
        moved_words += changed.size
        effects = {}
        transitions = {}
        for before, after, commit, reason in changes:
            left = controls[before][key].view("u4").reshape(-1)
            right = controls[after][key].view("u4").reshape(-1)
            altered = left != right
            totals[commit] += int(np.count_nonzero(altered))
            effects[commit] = stats(controls[after][key], controls[before][key])
            transitions[commit] = (left, right, altered)
        word_rows = []
        for index in changed:
            reasons = [dict(commit=commit, before=f"{int(left[index]):08x}",
                            after=f"{int(right[index]):08x}")
                       for commit, (left, right, altered) in transitions.items()
                       if altered[index]]
            assert reasons, (key, int(index))
            word_rows.append(dict(index=[int(v) for v in np.unravel_index(index, old.shape)],
                                  original=f"{int(old.view('u4').reshape(-1)[index]):08x}",
                                  merged=f"{int(current.view('u4').reshape(-1)[index]):08x}",
                                  transitions=reasons))
        fields.append(dict(name=key, shape=list(old.shape), merged_vs_original=stats(current, old),
                           control_transitions=effects, moved_words=word_rows))
    baseline = {key: value for key, value in controls[0].items() if not key.startswith("legacy_")}
    legacy = {key[len("legacy_"):]: value for key, value in controls[0].items()
              if key.startswith("legacy_")}
    np.savez_compressed(output / "diff2-merged-baseline.npz", **baseline)
    np.savez_compressed(output / "diff2-merged-legacy.npz", **legacy)
    current_receipt = json.loads((captures / "control-0.json").read_text())
    receipt = dict(merged_commit=merged_commit, original_commit="a4177ebbf342252405df3f6ed8309704daee94fc",
                   original_baseline_sha256=digest(data / "diff2-baseline.npz"),
                   original_legacy_sha256=digest(data / "diff2-legacy.npz"),
                   original_checkpoints=current_receipt["legacy_checkpoints"],
                   merged_baseline_sha256=digest(output / "diff2-merged-baseline.npz"),
                   merged_legacy_sha256=digest(output / "diff2-merged-legacy.npz"),
                   runtime_sources=current_receipt["runtime_sources"],
                   controls=[json.loads((captures / f"control-{level}.json").read_text())
                             for level in range(5)],
                   words=int(original_words), moved_words=int(moved_words),
                   control_transition_words=totals,
                   control_groups=[dict(before=b, after=a, commit=c, reason=r)
                                   for b, a, c, r in changes], fields=fields)
    full_words=output / "diff2-merged-attribution-words.json"
    full_words.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    summary=dict(receipt)
    summary["word_attribution_sha256"]=digest(full_words)
    summary["fields"]=[{key:value for key,value in row.items() if key!="moved_words"}
                       for row in fields]
    summary["native_provenance"]={
        "46b0a09fe": ["tests/test_smallstep_vertical_wrf471_parity.py",
                      "tools/smallstep_wrf471_oracle"],
        "dc226e820": ["tests/test_advect_wrf471_parity.py",
                      "tools/advect_wrf471_oracle"],
        "0673d8c51": ["tests/test_bigstep_prep_wrf471_parity.py",
                      "tests/test_w_crit_cfl.py", "tools/bigstep_wrf471_oracle"],
        "83fde6032+2c212f621": ["tests/test_diff6_wrf471_parity.py",
                              "tests/test_deformation_wrf471_parity.py",
                              "tests/test_horizontal_diffusion_wrf471_parity.py",
                              "tests/test_vertical_diffusion_wrf471_parity.py",
                              "tests/test_rk_addtend_dry_map_factors.py",
                              "tools/wrf_diffusion_oracle"],
    }
    summary["mapping_note"]=("All synthetic inputs keep identity map factors and has_msf=False. "
                             "The 2c212f621 map-coupling correction is inert on this corpus; "
                             "the diffusion transition measures 83fde6032 default corrections.")
    (output / "diff2-merged-attribution.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                        encoding="utf-8")
    print(json.dumps({key: receipt[key] for key in
                      ("words", "moved_words", "control_transition_words")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("captures", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--merged-commit", required=True)
    args = parser.parse_args()
    attribute(args.data, args.captures, args.output, args.merged_commit)
