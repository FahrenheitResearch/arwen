"""The release packet's owner-identity gate reads every member of every asset.

The breakage it prevents: the owner's name, e-mail address or home directory
published on PyPI or a GitHub release (owner ruling 2026-09-10).  2.9.0's first
candidate wheel and sdist carried the name in five new lines that no gate read
for.  The gate lives in the private cut tooling, which the public snapshot does
not carry, so a published tree skips this module.

Every identity string below is assembled from fragments: this file is itself
read by the snapshot scan and ships in the sdist.
"""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import tarfile
import zipfile

import pytest

REPO = Path(__file__).resolve().parents[1]
DRIVERS = REPO / "tools" / "release" / "cut" / "drivers"
SCANNER = DRIVERS / "owner_identity_scan.py"

pytestmark = pytest.mark.skipif(
    not SCANNER.is_file(),
    reason="the release cut tooling is not in this tree (published snapshot)")

NAME = "Dr" + "ew"
USER = "dr" + "ew"
MAIL = "fahrenheit" + "agi" + "@example.invalid"
BS = chr(92)


def _scanner():
    spec = importlib.util.spec_from_file_location("owner_identity_scan", SCANNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wheel(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
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
    rows = scan.scan_asset(wheel)
    found = sorted((row["member"], row["kind"]) for row in rows)
    assert found == [
        ("pkg/bin/tool", "owner's home directory"),
        ("pkg/mnt.txt", "owner's home directory"),
        ("pkg/namelist_chem.py", "owner's name"),
        ("pkg/plan.json", "owner's home directory"),
        ("pkg/receipt.json", "owner's e-mail"),
    ]
    assert all(row["asset"] == wheel.name and row["where"] == "member-body" for row in rows)


def test_words_and_other_users_are_not_the_owner(tmp_path):
    scan = _scanner()
    sdist = _sdist(tmp_path / "pkg-1.0.tar.gz", {
        "pkg-1.0/render.py": b"# the renderer drew 43 pictures and withdrew none\n",
        "pkg-1.0/paths.txt": ("/ho" + "me/somebody/wrf\n/ho" + "me/" + USER
                              + "ess/x\nC:/Users/" + USER + "-lab/y\n").encode(),
        "pkg-1.0/shell.py": b'NAMES = ["owner\'s case", "Kai\\u2019s case"]\n',
    })
    assert scan.scan_asset(sdist) == []


def test_member_names_are_read_as_well_as_bodies(tmp_path):
    scan = _scanner()
    sdist = _sdist(tmp_path / "pkg-1.0.tar.gz", {
        "pkg-1.0/notes/" + NAME + "-ruling.md": b"text\n"})
    rows = scan.scan_asset(sdist)
    assert [(row["where"], row["kind"]) for row in rows] == [
        ("member-name", "owner's name")]


def test_the_command_line_exits_nonzero_on_a_hit(tmp_path, capsys):
    scan = _scanner()
    clean = _wheel(tmp_path / "clean-1.0-py3-none-any.whl", {"a/b.py": b"x = 1\n"})
    dirty = _wheel(tmp_path / "dirty-1.0-py3-none-any.whl",
                   {"a/b.py": ("# " + NAME + "\n").encode()})
    assert scan.main([str(clean)]) == 0
    assert scan.main([str(clean), str(dirty)]) == 1
    assert '"hits": 1' in capsys.readouterr().out


def test_the_gate_is_wired_into_the_final_packet_verdict():
    """A hit must BLOCK the packet, not merely be recorded."""

    source = (DRIVERS / "verify_final_packet.py").read_text(encoding="utf-8")
    assert "import owner_identity_scan as owner_scan" in source
    assert "owner_rows.extend(owner_scan.scan_asset(path))" in source
    assert "if owner_rows:\n    blockers.append({'gate': 'owner_identity'," in source
