"""The release packet's owner-identity gate reads every member of every asset.

The breakage it prevents: the owner's name, e-mail address or home directory
published on PyPI or a GitHub release (owner ruling 2026-09-10).  2.9.0's first
candidate wheel and sdist carried the name in five new lines that no gate read
for, and 2.9.0 cut r7 sealed with the all-capitals name twice in the
rw_mpas_static binary (both platform wheels, both bridge bundles) while the
gate, which then read only the capitalised word, reported 0 hits.  So the name
is read in every case, inside nested archives too, and the only allowance is a
named list of exact phrases (the lower-case spelling is also the past tense of
"draw").  The gate lives in the private cut tooling, which the public snapshot
does not carry, so a published tree skips this module.

Every identity string below is assembled from fragments: this file is itself
read by the snapshot scan and ships in the sdist, where the packet gate reads
it in every case.
"""

from __future__ import annotations

import gzip
import importlib.util
import io
import json
from pathlib import Path
import re
import tarfile
import zipfile

import pytest

REPO = Path(__file__).resolve().parents[1]
DRIVERS = REPO / "tools" / "release" / "cut" / "drivers"
SCANNER = DRIVERS / "owner_identity_scan.py"
ALLOWANCE = DRIVERS / "owner_identity_allow.json"

pytestmark = pytest.mark.skipif(
    not SCANNER.is_file(),
    reason="the release cut tooling is not in this tree (published snapshot)")

NAME = "Dr" + "ew"
USER = "dr" + "ew"
CAPS = USER.upper()
MAIL = "fahrenheit" + "agi" + "@example.invalid"
BS = chr(92)


def _scanner():
    spec = importlib.util.spec_from_file_location("owner_identity_scan", SCANNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _wheel(path: Path, members: dict[str, bytes]) -> Path:
    path.write_bytes(_zip_bytes(members))
    return path


def _sdist(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


def test_every_identity_shape_is_found_in_code_data_and_binaries(tmp_path):
    scan = _scanner()
    wheel = _wheel(tmp_path / "pkg-1.0-py3-none-any.whl", {
        "pkg/namelist_chem.py": ("# " + NAME + "'s ruling (2026-10-12)\n").encode(),
        "pkg/receipt.json": ('{"by": "' + MAIL + '"}\n').encode(),
        "pkg/plan.json": ('{"dir": "C:' + BS * 2 + "Users" + BS * 2 + USER
                          + BS * 2 + 'deck"}\n').encode(),
        "pkg/bin/tool": (b"\x7fELF\x00\x01/ho" + b"me/" + USER.encode()
                         + b"/.cargo/registry/src/x.rs\x00\x02"),
        "pkg/mnt.txt": ("/mnt/c/Users/" + USER + "/notes\n").encode(),
    })
    rows = scan.scan_asset(wheel, allow={})
    found = sorted((row["member"], row["kind"]) for row in rows)
    # A home directory is reported once, as a home directory, not again as the name.
    assert found == [
        ("pkg/bin/tool", "owner's home directory"),
        ("pkg/mnt.txt", "owner's home directory"),
        ("pkg/namelist_chem.py", "owner's name"),
        ("pkg/plan.json", "owner's home directory"),
        ("pkg/receipt.json", "owner's e-mail"),
    ]
    assert all(row["asset"] == wheel.name and row["where"] == "member-body" for row in rows)


def test_the_name_is_read_in_every_case(tmp_path):
    """2.9.0 cut r7: the all-capitals name in a native binary read 0 hits."""

    scan = _scanner()
    wheel = _wheel(tmp_path / "pkg-1.0-py3-none-manylinux_2_28_x86_64.whl", {
        "pkg/libexec/bridges/rw_static": (
            b"\x7fELF\x00\x00rw_native_undefined\x00EXCLUSIONS PROPOSED FOR " + CAPS.encode()
            + b": OL1 row zero; case-specific sign-off required\x00\x01PROPOSED FOR "
            + CAPS.encode() + b": case-specific sign-off required\x00"),
        "pkg/ask.py": ("# ask " + USER + " before changing this\n").encode(),
        "pkg/lane.py": ('HOLDER = "' + USER + '-gui"\n').encode(),
        "pkg/ident.py": ("x_" + USER + "_y = 1\n").encode(),
        "pkg/fine.py": ("# with" + USER + " the An" + USER + "s case\n").encode(),
    })
    rows = scan.scan_asset(wheel, allow={})
    found = sorted((row["member"], row["phrase"]) for row in rows)
    assert found == [
        ("pkg/ask.py", "ask " + USER + " before"),
        ("pkg/ident.py", "x_" + USER + "_y ="),
        ("pkg/lane.py", '= "' + USER + '-gui"'),
        ("pkg/libexec/bridges/rw_static", "FOR " + CAPS + ": OL1"),
        ("pkg/libexec/bridges/rw_static", "FOR " + CAPS + ": case-specific"),
    ]
    assert all(row["kind"] == "owner's name" for row in rows)


def test_an_allowance_forgives_only_its_exact_phrase(tmp_path):
    scan = _scanner()
    verb = "the renderer " + USER + " 43"
    allow = {"renderer " + USER + " 43": "the past tense of draw"}
    sdist = _sdist(tmp_path / "pkg-1.0.tar.gz", {
        "pkg-1.0/render.py": ("# " + verb + " pictures\n").encode(),
        "pkg-1.0/upper.py": ("# the renderer " + NAME + " 43 pictures\n").encode(),
        "pkg-1.0/other.py": ("# the renderer " + USER + " 44 pictures\n").encode(),
    })
    forgiven = []
    rows = scan.scan_asset(sdist, allow=allow, forgiven=forgiven)
    assert sorted((row["member"], row["phrase"]) for row in rows) == [
        ("pkg-1.0/other.py", "renderer " + USER + " 44"),
        ("pkg-1.0/upper.py", "renderer " + NAME + " 43"),
    ]
    assert [(row["member"], row["phrase"]) for row in forgiven] == [
        ("pkg-1.0/render.py", "renderer " + USER + " 43")]


def test_words_and_other_users_are_not_the_owner(tmp_path):
    scan = _scanner()
    sdist = _sdist(tmp_path / "pkg-1.0.tar.gz", {
        "pkg-1.0/render.py": ("# the renderer " + USER + " 43 pictures and with" + USER
                              + " none\n").encode(),
        "pkg-1.0/paths.txt": ("/ho" + "me/somebody/wrf\n/ho" + "me/" + USER
                              + "ess/x\nC:/Users/" + USER + "lab/y\n").encode(),
        "pkg-1.0/shell.py": b'NAMES = ["owner\'s case", "Kai\\u2019s case"]\n',
    })
    allow = {"renderer " + USER + " 43": "the past tense of draw"}
    assert scan.scan_asset(sdist, allow=allow) == []


def test_archives_inside_members_are_opened(tmp_path):
    """A wheel inside a tarball, an .npz and a .gz inside a wheel: their members are read."""

    scan = _scanner()
    inner = _zip_bytes({"pkg/bin/tool": b"\x00PROPOSED FOR " + CAPS.encode() + b": x\x00"})
    npz = _zip_bytes({"table.npy": b"\x93NUMPY\x00" + ("by " + NAME + " ").encode() + b"\x00"})
    wheel = _zip_bytes({"pkg/data/table.npz": npz,
                        "pkg/data/rows.csv.gz": gzip.compress(("a," + NAME + ",b\n").encode())})
    outer = _sdist(tmp_path / "Desktop-1.0-Linux-x64-pip.tar.gz", {
        "Desktop-1.0/wheels/pkg-1.0-py3-none-any.whl": inner,
        "Desktop-1.0/wheels/data-1.0-py3-none-any.whl": wheel,
    })
    rows = scan.scan_asset(outer, allow={})
    assert sorted(row["member"] for row in rows) == [
        "Desktop-1.0/wheels/data-1.0-py3-none-any.whl!pkg/data/rows.csv.gz!<gunzipped>",
        "Desktop-1.0/wheels/data-1.0-py3-none-any.whl!pkg/data/table.npz!table.npy",
        "Desktop-1.0/wheels/pkg-1.0-py3-none-any.whl!pkg/bin/tool",
    ]


def test_member_names_are_read_as_well_as_bodies(tmp_path):
    scan = _scanner()
    sdist = _sdist(tmp_path / "pkg-1.0.tar.gz", {
        "pkg-1.0/notes/" + NAME + "-ruling.md": b"text\n"})
    rows = scan.scan_asset(sdist, allow={})
    assert [(row["where"], row["kind"]) for row in rows] == [
        ("member-name", "owner's name")]


def test_the_command_line_exits_nonzero_on_a_hit(tmp_path, capsys):
    scan = _scanner()
    clean = _wheel(tmp_path / "clean-1.0-py3-none-any.whl", {"a/b.py": b"x = 1\n"})
    dirty = _wheel(tmp_path / "dirty-1.0-py3-none-any.whl",
                   {"a/b.py": ("# " + CAPS + "\n").encode()})
    assert scan.main([str(clean)]) == 0
    assert scan.main([str(clean), str(dirty)]) == 1
    assert '"hits": 1' in capsys.readouterr().out


def test_the_allowance_names_a_reason_for_every_phrase_and_no_identity():
    """Every row is a hit with a reason; none is a login, a holder, a path or the name capitalised."""

    record = json.loads(ALLOWANCE.read_text(encoding="utf-8"))
    phrases = record["phrases"]
    assert phrases, "an empty allowance would block every past tense of draw"
    scan = _scanner()
    assert scan.load_allowance() == phrases
    word = re.compile(r"(?i)(?<![A-Za-z])" + USER + r"(?![A-Za-z])")
    for phrase, why in phrases.items():
        assert why.strip(), phrase
        spellings = word.findall(phrase)
        assert spellings and all(s == USER for s in spellings), (
            f"{phrase!r}: only the lower-case spelling can be the verb; the capitalised "
            "or all-capitals word is the name and is reworded at its source")
        for shape in (USER + "@", "@" + USER, USER + "-", "/home/", "/Users/", BS + "Users"):
            assert shape not in phrase, f"{phrase!r} has an identity shape ({shape!r})"


def test_the_gate_is_wired_into_the_final_packet_verdict():
    """A hit must BLOCK the packet, not merely be recorded."""

    source = (DRIVERS / "verify_final_packet.py").read_text(encoding="utf-8")
    assert "import owner_identity_scan as owner_scan" in source
    assert "owner_rows.extend(owner_scan.scan_asset(path, forgiven=owner_forgiven))" in source
    assert "if owner_rows:\n    blockers.append({'gate': 'owner_identity'," in source


def test_the_built_wheels_are_read_with_the_same_gate():
    """packages.py reads each wheel and sdist the moment it is built, before the packet."""

    source = (REPO / "tools" / "release" / "cut" / "packages.py").read_text(encoding="utf-8")
    assert "from .drivers.owner_identity_scan import scan_asset" in source
    assert "if hits:\n        raise ValueError('built artifacts contain private identity" in source
