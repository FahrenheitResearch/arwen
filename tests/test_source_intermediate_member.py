"""source-intermediate --member selects a frame without sealing an identity.

Breakage (hex TC init, 2026-10-27): ``gpuwm source-intermediate --member N`` always
refused with "member and member_identity are an atomic pair", because the door handed
the manifest author a member with no identity, so no single GEFS member could reach hex
through this door.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import gpuwm.mapped_authoring as mapped_authoring
import gpuwm.source_authorities as source_authorities
from gpuwm import source_intermediate


class _Stop(Exception):
    pass


def _args(tmp_path: Path, *extra: str):
    parser = argparse.ArgumentParser()
    source_intermediate.register_cli(parser.add_subparsers())
    argv = ["source-intermediate", "--profile", "p", "--input", str(tmp_path / "a.grib2"),
            "--valid-time", "2024-10-08T00:00:00", "--out", str(tmp_path / "MET"),
            "--work-dir", str(tmp_path / "work"), *extra]
    return parser.parse_args(argv)


@pytest.fixture
def seen(monkeypatch):
    calls = []
    monkeypatch.setattr(source_authorities, "packaged_authorities",
                        lambda profile: {"mapping": Path("m.json"), "composition": Path("c.json")})
    monkeypatch.setattr(source_authorities, "packaged_contributing_mappings", lambda profile: [])
    monkeypatch.setattr(source_authorities, "packaged_provenance_files", lambda profile: [])

    def author(path, **kwargs):
        member, identity = kwargs.get("member"), kwargs.get("member_identity")
        calls.append((member, identity))
        if (member is None) != (identity is None):
            raise ValueError("member and member_identity are an atomic pair")
        raise _Stop

    monkeypatch.setattr(mapped_authoring, "author_input_manifest", author)
    return calls


def test_member_alone_selects_the_frame_and_seals_no_identity(tmp_path, seen):
    with pytest.raises(_Stop):
        source_intermediate.main(_args(tmp_path, "--member", "3"))
    assert seen == [(None, None)]


def test_member_identity_seals_the_member_into_the_manifest(tmp_path, seen):
    with pytest.raises(_Stop):
        source_intermediate.main(_args(tmp_path, "--member", "3", "--member-identity", "grib-member"))
    assert seen == [("3", "grib-member")]
