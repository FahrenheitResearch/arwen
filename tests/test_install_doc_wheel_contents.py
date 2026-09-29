"""The install guide says what each pip artifact actually carries.

pyproject's package data puts the staged native tools
(``gpuwm/libexec/bridges``) into the wheel, and setup.py tags a wheel that
carries them for its one platform.  The guide's pip paragraph still said
a pip wheel contains no compiled Rust, beside a paragraph of the same
page saying the platform wheels carry them.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: Pages a pip user reads for what an install contains.
PAGES = ("README.md", "docs/install.md",
         *sorted(str(path.relative_to(REPO)).replace("\\", "/")
                 for path in (REPO / "docs" / "public").glob("*.md")))

#: A sentence saying something carries no compiled or native code.
NO_NATIVE = re.compile(
    r"\b(?:contains?|carr(?:y|ies)|ships?)\s+no\s+(?:compiled\s+)?"
    r"(?:Rust|native|binar)", re.IGNORECASE)

#: The artifacts that really carry none.
NO_NATIVE_ARTIFACT = re.compile(r"pure|py3-none-any|sdist|source distribution",
                                re.IGNORECASE)


def _sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.:;])\s+", re.sub(r"\s+", " ", text))


def test_platform_wheels_do_carry_the_native_tools():
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert "libexec/bridges/*" in project["tool"]["setuptools"]["package-data"]["gpuwm"]


@pytest.mark.parametrize("relative", PAGES)
def test_no_page_says_every_wheel_lacks_native_tools(relative):
    text = (REPO / relative).read_text(encoding="utf-8")
    wrong = [sentence for sentence in _sentences(text)
             if "wheel" in sentence.lower() and NO_NATIVE.search(sentence)
             and not NO_NATIVE_ARTIFACT.search(sentence)]
    assert not wrong, f"{relative}: {wrong}"


def test_the_install_guide_says_the_platform_wheels_carry_them():
    text = re.sub(r"\s+", " ", (REPO / "docs" / "install.md").read_text(encoding="utf-8"))
    assert re.search(r"platform wheels?[^.]*(?:contain|carr)[^.]*"
                     r"(?:compiled Rust|native tools)", text)
