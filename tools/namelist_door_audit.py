"""Every WRF v4.7.1 namelist key, set off its default, through the importer.

    python tools/namelist_door_audit.py --out audit.json [--md audit.md]

The namelist door has one rule: a WRF namelist setting either maps
onto an implemented equivalent or is refused with the breakage named.  This
audit holds the importer to it key by key.  For every key WRF v4.7.1's
Registry declares in a group the importer reads (gpuwm/data/wrf_namelist/
registry_v4.7.1.json, plus the WRF-Chem &chem vocabulary with chemistry on),
it writes the key into a valid baseline pair at a value other than its
Registry default and imports the pair, then classifies what happened:

* ``refused``     -- the import refused, naming the key;
* ``refused-other`` -- the import refused, naming something else (the key
  changed a precondition another key checks);
* ``translated``  -- the emitted TOML differs from the baseline's;
* ``substituted`` -- the report books a substitution (a declared divergence)
  for the key: printed, never silent, but not the requested model;
* ``fixed`` / ``dropped`` -- the TOML is unchanged and the report records the
  key as validated against a pin, or consumed without a counterpart.  These
  are the rows a reviewer reads: each must say why the requested value
  changes nothing, or it is a silent drop;
* ``unrecorded``  -- imported, TOML unchanged, and the report says nothing
  about the key: a silent drop by construction.

The probe value is the default's logical negation, +1 for an integer (1 for
a zero default), x2 (or 1.0) for a real; character keys are skipped.  One
probe per key is a census, not a proof: a key whose effect depends on a
second key is judged in that second key's default context.
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path


def _probe(row: dict):
    from gpuwm.namelist_contract import _registry_value

    kind = row["type"]
    if kind == "character":
        return None
    default = _registry_value(row)
    if kind == "logical":
        return not default
    if kind == "integer":
        return default + 1 if default else 1
    if kind == "real":
        return default * 2.0 if default else 1.0
    return None


def _column(row: dict, value, max_dom: int):
    return [value] * (max_dom if row["nentries"] == "max_domains" else 1)


def _classify(section, key, report, text, base_text) -> tuple[str, str]:
    for sub in report.substitutions:
        names = str(sub.key).split("/")
        if key in names:
            return "substituted", sub.reason or "(package replacement)"
    if text != base_text:
        return "translated", ""
    for fixed in report.fixed:
        if fixed.section == section and fixed.key == key:
            return "fixed", fixed.reason
    for dropped in report.dropped:
        # A drop may name two keys decided together ("clos_choice/ishallow")
        # or one domain's entry of a column ("cudt[2]").
        names = [name.split("[")[0] for name in str(dropped.key).split("/")]
        if dropped.section == section and key in names:
            return "dropped", dropped.reason
    for default in report.defaults_applied:
        if default.key == key:
            return "fixed", default.reason
    return "unrecorded", ""


def audit(baseline: str = "rrtmg-rte-rrtmgp") -> dict:
    from gpuwm import namelist_import as ni
    from gpuwm.namelist_contract import BASELINES, _apply, _base_pair
    from gpuwm.wrf_namelist_registry import wrf_namelist_keys
    from gpuwm.explain import muted_warnings

    variant = dict(BASELINES)[baseline]
    wps, inp = _apply(_base_pair(), variant)
    options = dict(variant["options"])
    max_dom = 2

    def run(w, i):
        with muted_warnings():
            return ni.import_parsed_namelists(
                deepcopy(w), deepcopy(i), wps_path=Path("namelist.wps"),
                input_path=Path("namelist.input"), **options)

    base_text, _ = run(wps, inp)
    rows = []
    groups = set(ni.INPUT_SECTIONS)
    vocabulary = [(s, k, r) for (s, k), r in sorted(wrf_namelist_keys().items())
                  if s in groups]
    for section, key, row in vocabulary:
        value = _probe(row)
        record = {"section": section, "key": key, "default": row["default"],
                  "probe": value}
        if value is None:
            record.update(outcome="skipped", why="character key")
            rows.append(record)
            continue
        probe = deepcopy(inp)
        if key in probe.get(section, {}):
            record.update(outcome="skipped", why="baseline sets it")
            rows.append(record)
            continue
        probe.setdefault(section, {})[key] = _column(row, value, max_dom)
        try:
            text, report = run(wps, probe)
        except (ValueError, NotImplementedError) as error:
            message = str(error)
            named = (f"{key}" in message)
            record.update(outcome="refused" if named else "refused-other",
                          why=message[:600])
            rows.append(record)
            continue
        outcome, why = _classify(section, key, report, text, base_text)
        record.update(outcome=outcome, why=why)
        rows.append(record)
    counts: dict[str, int] = {}
    for record in rows:
        counts[record["outcome"]] = counts.get(record["outcome"], 0) + 1
    return {"schema": "gpuwm.namelist-door-audit.v1",
            "baseline": baseline, "counts": counts, "rows": rows}


def markdown(result: dict) -> str:
    lines = [f"# Namelist door audit (baseline {result['baseline']})", "",
             "| outcome | keys |", "|---|---:|"]
    for outcome, count in sorted(result["counts"].items()):
        lines.append(f"| {outcome} | {count} |")
    for outcome in ("unrecorded", "substituted", "dropped", "fixed",
                    "refused-other"):
        chosen = [r for r in result["rows"] if r["outcome"] == outcome]
        if not chosen:
            continue
        lines += ["", f"## {outcome} ({len(chosen)})", "",
                  "| group | key | default | probe | recorded reason |",
                  "|---|---|---|---|---|"]
        for r in chosen:
            why = str(r.get("why", "")).replace("|", "/").replace("\n", " ")
            lines.append(f"| {r['section']} | {r['key']} | {r['default']} | "
                         f"{r['probe']} | {why[:300]} |")
    return "\n".join(lines) + "\n"


def audit_every_baseline() -> dict:
    """:func:`audit` over every contract baseline that imports, merged:
    a key's rows are kept per distinct (outcome, reason), so a key that is
    inert under one suite and consumed under another shows both."""
    from gpuwm import namelist_import as ni
    from gpuwm.namelist_contract import BASELINES, _apply, _base_pair

    merged: dict[tuple, dict] = {}
    used = []
    for label, variant in BASELINES:
        wps, inp = _apply(_base_pair(), variant)
        if ni.namelist_refusals(wps, inp, **dict(variant["options"])):
            continue
        used.append(label)
        for record in audit(label)["rows"]:
            if record["outcome"] == "skipped":
                continue
            reason = str(record.get("why", ""))
            token = (record["section"], record["key"], record["outcome"],
                     reason if record["outcome"] in (
                         "dropped", "fixed", "substituted") else "")
            row = merged.setdefault(token, {**record, "baselines": []})
            row["baselines"].append(label)
    rows = sorted(merged.values(),
                  key=lambda r: (r["section"], r["key"], r["outcome"]))
    counts: dict[str, int] = {}
    for outcome in {r["outcome"] for r in rows}:
        counts[outcome] = len({(r["section"], r["key"]) for r in rows
                               if r["outcome"] == outcome})
    return {"schema": "gpuwm.namelist-door-audit.v1",
            "baseline": "every contract baseline: " + ", ".join(used),
            "counts": counts, "rows": rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--md", type=Path)
    parser.add_argument("--baseline", default="rrtmg-rte-rrtmgp")
    parser.add_argument("--every-baseline", action="store_true",
                        help="merge the audit over every contract baseline")
    args = parser.parse_args(argv)
    result = (audit_every_baseline() if args.every_baseline
              else audit(args.baseline))
    args.out.write_text(json.dumps(result, indent=1, default=str) + "\n",
                        encoding="utf-8", newline="\n")
    if args.md:
        args.md.write_text(markdown(result), encoding="utf-8", newline="\n")
    print(json.dumps(result["counts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
