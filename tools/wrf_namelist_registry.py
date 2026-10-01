"""Extract WRF's namelist vocabulary from its Registry into a data table.

    python tools/wrf_namelist_registry.py --registry WRF/Registry \
        --tag v4.7.1 --commit f52c197ed39d12e087d02c50f412d90d418f6186 \
        --out gpuwm/data/wrf_namelist/registry_v4.7.1.json

WRF declares every namelist key as an ``rconfig`` line in its Registry
(``rconfig <type> <name> namelist,<group> <nentries> <default> ...``).
This reads the files the ARW build compiles -- ``Registry.EM`` and every
file it includes, with the build's own ``ifdef`` settings (EM_CORE=1,
DA_CORE=0, NMM_CORE=0) -- and writes one row per key: its namelist group,
Fortran type, entry count (``1``, ``max_domains`` or a number) and the
Registry default exactly as written.

The table is what the namelist contract (gpuwm.namelist_contract) checks
the importer against: every key WRF can read is either consumed by the
importer or listed as refused, so the site can tell a user about a key
before a GPU box boots.  WRF is public domain; the table records the tag,
the commit and the sha256 of every file it was read from.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

#: The ARW build's preprocessor settings for Registry ``ifdef`` lines.
ARW_DEFINES = {"EM_CORE": "1", "DA_CORE": "0", "NMM_CORE": "0"}

_RCONFIG = re.compile(
    r"^rconfig\s+(?P<type>\w+)\s+(?P<name>\w+)\s+namelist,(?P<group>\w+)"
    r"\s+(?P<nentries>\S+)\s+(?P<default>\S+)")


def _read(path: Path, defines: dict[str, str], seen: list[Path],
          rows: dict[tuple[str, str], dict]) -> None:
    if not path.is_file():
        # io_boilerplate_temporary.inc is generated at compile time and
        # declares only the numbered aux streams, which the importer
        # matches by pattern.
        return
    seen.append(path)
    active = [True]
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("ifdef ") or line.startswith("ifndef "):
            word, cond = line.split(None, 1)
            name, _, value = cond.partition("=")
            truth = defines.get(name.strip(), "0") == (value.strip() or "1")
            if word == "ifndef":
                truth = not truth
            active.append(active[-1] and truth)
            continue
        if line == "endif":
            if len(active) > 1:
                active.pop()
            continue
        if not active[-1]:
            continue
        if line.startswith("include "):
            _read(path.parent / line.split(None, 1)[1].strip(), defines,
                  seen, rows)
            continue
        match = _RCONFIG.match(line)
        if match:
            key = (match["group"].lower(), match["name"].lower())
            rows.setdefault(key, {
                "section": key[0], "key": key[1],
                "type": match["type"].lower(),
                "nentries": match["nentries"].lower(),
                "default": match["default"],
            })


def build_table(registry: Path, *, tag: str, commit: str) -> dict:
    seen: list[Path] = []
    rows: dict[tuple[str, str], dict] = {}
    _read(registry / "Registry.EM", ARW_DEFINES, seen, rows)
    return {
        "schema": "gpuwm.wrf-namelist-registry.v1",
        "wrf": {"tag": tag, "commit": commit,
                "url": "https://github.com/wrf-model/WRF",
                "license": "public domain"},
        "read_from": [
            {"file": f"Registry/{path.name}",
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in seen],
        "keys": [rows[key] for key in sorted(rows)],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    table = build_table(args.registry, tag=args.tag, commit=args.commit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8",
                        newline="\n")
    print(f"{len(table['keys'])} namelist keys from {len(table['read_from'])} "
          f"Registry files -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
