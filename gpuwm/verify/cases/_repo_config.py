"""Locating a repository config that the wheel does not ship.

Some case modules are the *module half* of a case whose other half is a
TOML under the repository's top-level ``configs/`` directory.  That
directory is not a package and is named by neither
``[tool.setuptools.packages.find]`` nor ``[tool.setuptools.package-data]``
in ``pyproject.toml``, so it is in no wheel and in no sdist: the wheel's
only top-level entries are ``gpuwm``, ``tools`` and ``tilestream``.

A case module that resolves its config as ``Path(__file__).parents[3] /
"configs" / NAME`` therefore points at ``<site-packages>/configs/NAME``
after ``pip install gpuwm`` -- a path that has never existed on any
machine.  Reading it produced the generic loader refusal *"... does not
exist; pass the experiment .toml that `gpuwm domain` wrote"*, which is
wrong twice over: the wizard does not emit these configs, and the module
that printed it accepted no path to pass.

This module holds the one honest answer.  :func:`locate` returns the
config when it is really there, and :func:`missing_config_message` builds
the refusal that names *why* it is not there and what the reader can
actually do, given the flag the calling module offers.  Nothing here
knows any case name -- the filename is always an argument.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Environment override, for a checkout kept somewhere else.  Names a
#: DIRECTORY that holds the config files, standing in for the
#: repository's own ``configs/``.
CONFIG_ROOT_ENV = "GPUWM_CONFIGS_ROOT"


def module_name(name: str, spec) -> str:
    """The dotted module name, even under ``python -m``.

    ``__name__`` is ``"__main__"`` in exactly the invocation these case
    modules document, which would make every ``prog=`` and every refusal
    prefix read ``__main__``.  ``__spec__.name`` carries the real one.
    """

    return getattr(spec, "name", None) or name

#: Where the repository's configs live, relative to this package, when
#: this package is being imported out of a source checkout.
_CHECKOUT_ROOT = Path(__file__).resolve().parents[3]


def config_roots() -> tuple[Path, ...]:
    """Every directory a repository config is looked for, in order."""

    roots: list[Path] = []
    override = os.environ.get(CONFIG_ROOT_ENV)
    if override:
        roots.append(Path(override).expanduser())
    roots.append(_CHECKOUT_ROOT / "configs")
    return tuple(roots)


def locate(name: str) -> Path | None:
    """The readable repository config called ``name``, or ``None``."""

    for root in config_roots():
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def default_path(name: str) -> Path:
    """The path a config WOULD have, for display and for a default.

    Always the checkout-relative location, never the override, so the
    value a module exposes as its default is stable and does not change
    meaning when an environment variable is set.
    """

    return _CHECKOUT_ROOT / "configs" / name


def shipped_in_wheel() -> bool:
    """Whether ``configs/`` is present next to the installed package."""

    return (_CHECKOUT_ROOT / "configs").is_dir()


def missing_config_message(name: str, *, flag: str = "--config") -> str:
    """Why the config is not on this machine, and what to do about it.

    ``flag`` is the option the calling module accepts a path on, so the
    remedy names a door that module really has.
    """

    searched = "\n".join(f"    {root / name}" for root in config_roots())
    lines = [
        f"the repository config {name} is not on this machine.",
        "",
        "Searched:",
        searched,
        "",
    ]
    if not shipped_in_wheel():
        lines += [
            "This case is the module half of a case whose other half is a "
            "TOML under the repository's top-level `configs/` directory.  "
            "`configs/` is not a Python package and ships in no wheel and "
            "no sdist, so a `pip install gpuwm` has never had this file.",
            "",
        ]
    lines += [
        "Remedies, either one:",
        f"  * pass the file yourself:  {flag} PATH/TO/{name}",
        f"  * or point {CONFIG_ROOT_ENV} at the directory holding it, and "
        "run without the flag.",
        "",
        "Both want the `configs/` directory of a gpuwm source checkout "
        "(https://github.com/FahrenheitResearch/arwen); the file is "
        "committed there.  `gpuwm domain` does NOT emit it -- it is a "
        "ratified experiment config, not a wizard product.",
    ]
    return "\n".join(lines)
