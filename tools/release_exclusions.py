"""Read the exact path exclusions shared by release assembly and public checks.

Only literal paths and a trailing ``/**`` subtree rule are supported. Keep the
first matching rule so a refusal names the same exclusion the builder applied.
This module needs only the supplied repository's RELEASE-EXCLUDE.txt; publisher
scaffolding under work/ is deliberately absent from a public source snapshot.

It also carries the private-machine rule both release scans apply to the data
files a wheel carries: the snapshot and export scan over the public tree
(``work/build_release_snapshot.py``, which the cut's battery and export run)
and the artifact verifier over the built wheel
(``tools/verify_release_artifacts.py``).
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import re


def read_exclusions(repo_root: str | Path) -> list[str]:
    with (Path(repo_root) / "RELEASE-EXCLUDE.txt").open(encoding="utf-8") as stream:
        return [line for raw in stream if (line := raw.strip())
                and not line.startswith("#")]


def matches(rel: str, rules: list[str]) -> str | None:
    """Return the first rule matching a forward-slash relative path."""
    for rule in rules:
        if rule.endswith("/**"):
            root = rule[:-3]
            if rel == root or rel.startswith(root + "/"):
                return rule
        elif rel == rule:
            return rule
    return None


# ---------------------------------------------------------------------------
# Private machine names in shipped data files
# ---------------------------------------------------------------------------
# The breakage: a private machine's name published in a wheel.  A data file
# ships every byte it holds, and 2.8.1's fetch route table carried "posting
# watch" plus the lab host that watched in 20 measured rows (A154); the
# repository configs and the MPAS sizing table carried the same kind of name,
# and one NetCDF terrain file a LAN address.  Measurement records name the
# measurement, never the machine.
#
# Code is out of this rule's scope on purpose: a provenance comment in a .py or
# .cu file is read by a developer, and the path scan beside this still reads
# those files for machine paths.
#
# The patterns are assembled so this file carries no host name itself.

#: ``(pattern, kind)``: the lab hosts' names (node-1 to node-4 and their
#: weather-node-N spellings) and a private IPv4 address.  "node-wide" and
#: ``node_4`` are not host names and do not match.
PRIVATE_HOST_MARKERS = (
    (re.compile(r"(?i)(?<![A-Za-z0-9_.-])(?:weather" + "-)?node" + r"-[0-9]+(?![A-Za-z0-9])"),
     "private machine name"),
    (re.compile(r"(?<![0-9.])192" + r"\.168\.[0-9]{1,3}\.[0-9]{1,3}(?![0-9])"),
     "private network address"),
)

#: Suffixes of code, which the rule does not read (see above).
CODE_SUFFIXES = frozenset({
    ".bat", ".c", ".cc", ".cmd", ".cpp", ".cu", ".cuh", ".f", ".f90", ".h",
    ".hpp", ".ps1", ".py", ".pyi", ".rs", ".sh",
})

#: Tree subtrees every non-code file of which a wheel carries: the gpuwm
#: package's package-data (tests/test_package_data_coverage.py holds every
#: data file under gpuwm/ declared), the data companion's ``data/**/*`` and
#: the repository configs (``configs = ["**/*"]``).
WHEEL_DATA_SUBTREES = ("gpuwm/", "gpuwm-data/gpuwm_data/", "configs/")

#: The other package-data declarations, as tree patterns: the tools package's
#: ``release/*.json``, tilestream's one table and the seam document the
#: ``docs`` namespace package carries.
WHEEL_DATA_PATTERNS = ("tools/release/*.json", "tilestream/output-scaling.json",
                       "docs/mpas-seam.md")


def _is_code(name: str) -> bool:
    return PurePosixPath(name).suffix.lower() in CODE_SUFFIXES


def ships_as_wheel_data(rel: str) -> bool:
    """Is this tree path a data file one of the wheels carries?"""
    if _is_code(rel):
        return False
    if rel.startswith(WHEEL_DATA_SUBTREES):
        return True
    path = PurePosixPath(rel)
    return any(len(path.parts) == len(PurePosixPath(pattern).parts)
               and path.match(pattern) for pattern in WHEEL_DATA_PATTERNS)


def is_wheel_data_member(name: str) -> bool:
    """Is this wheel member a data file (not code, not the wheel's metadata)?"""
    top = name.split("/", 1)[0]
    return not (top.endswith((".dist-info", ".data")) or name.endswith("/")
                or _is_code(name))


#: The rule's one allowance.  A shipped file whose exact bytes a committed
#: record pins is a record, and rewording the machine out of it would falsify
#: the record, not clean it: the experiment config a published WRF reference
#: was built for, whose sha256 is that reference's ``config_sha256`` and names
#: its files.  The allowance is by digest, so any edit to the file ends it.
PINNED_RECORD_MANIFESTS = "docs/public/wrf-reference/*.manifest.json"


def pinned_record_digests(repo_root: str | Path) -> frozenset[str]:
    """The sha256 of every config a committed WRF reference manifest pins.

    A manifest that cannot be read pins nothing, so the rule stays strict.
    """
    import json

    digests = set()
    for manifest in sorted(Path(repo_root).glob(PINNED_RECORD_MANIFESTS)):
        try:
            digest = json.loads(manifest.read_text(encoding="utf-8")).get(
                "config_sha256")
        except (OSError, ValueError, AttributeError):
            continue
        if isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest):
            digests.add(digest)
    return frozenset(digests)


def private_host_kinds() -> frozenset[str]:
    return frozenset(kind for _pattern, kind in PRIVATE_HOST_MARKERS)
