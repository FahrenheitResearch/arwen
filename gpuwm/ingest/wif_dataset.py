"""The mp=28 WIF climatology dataset: pin, acquisition route, resolver.

``gpuwm/ingest/wif_climatology.py`` READS the dataset; this module says
WHICH bytes are admissible, WHERE they come from, and refuses -- by name,
with the command that fixes it -- when they are not there.

It is deliberately the same shape as
:mod:`gpuwm.core.thompson_aerosol_contract`'s ``CCN_ACTIVATE.BIN``
handling, because the problem is the same one: a fixed external file that
never changes, that no code path can regenerate, and whose substitution
would be a different numerical setup rather than an error.  The four parts
of that pattern are reproduced here on purpose --

1. a pinned exact size and SHA-256 (:data:`WIF_DATASET_ASSET`);
2. a documented acquisition route quoted verbatim in the refusal
   (:data:`WIF_DATASET_SOURCE`);
3. an override precedence -- explicit path, then the file environment
   override, then the directory override, then the staged user root --
   in which an override that names a missing file is an ERROR and never
   a silent fall-through to some other copy;
4. a fail-closed resolver (:func:`resolve_wif_climatology_path`) whose
   message names what is missing, what it is, where to get it, and how to
   supply it.

WHERE IT DIFFERS, and why: ``CCN_ACTIVATE.BIN`` is 35,288 bytes and gpuwm
redistributes it inside the wheel.  This file is 225,443,520 bytes -- over
twice PyPI's 100 MB per-file cap and over GitHub's 100 MiB blob limit --
so :data:`WIF_DATASET_REDISTRIBUTED` is False and stays False.  That is
exactly the situation ``gpuwm/table_assets.py`` already exists for
(``freezeH2O.dat``, 243 MiB, and ``qr_acr_qg_V4.dat``, 71 MiB), so this
dataset joins THAT command rather than growing a second downloader:
``gpuwm fetch-tables --wif`` stages it under the identical
verify-then-atomically-install contract.

WHY A MISSING FILE IS FATAL rather than defaulted.  Unlike
``CCN_ACTIVATE.BIN`` there is no prefill to be silently wrong with: the
WIF ingest is selected by ``aer_init_opt=1`` WITH ``wif_input_opt=1``, and
the alternative -- thompson_init's synthetic nwfa/nifa profile -- is a
DIFFERENT and perfectly valid configuration that ``aer_init_opt=0``
already selects.  Quietly demoting to it would mean a run whose receipt
says "monthly climatology" while its aerosol came from an exponential
profile in ``module_mp_thompson.F:493-546``.  That is the concrete
breakage this refusal prevents.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Mapping

from gpuwm.core.thompson_contract import TableAsset


#: The dataset's canonical filename, as WRF's ``METGRID.TBL``
#: ``constants_name`` names it and as every WRF tree carries it.
WIF_DATASET_FILE = "QNWFA_QNIFA_SIGMA_MONTHLY.dat"

#: The pin.  Measured 2026-08-27 on the reference host from two
#: independently staged copies (``~/.gpuwm/wif`` and the lane's own
#: staged ``wif-data`` tree), which agree byte for byte; it is the same
#: file the WRF-4.7.1 real.exe oracle run of commit 94260bf44 read, whose
#: QNWFA/QNIFA agreement to 1e-5 is the numerical claim this port makes.
#: A different file would be a different climatology and would invalidate
#: that measurement without raising anything, which is why this is a
#: hard pin and not a size sanity check.
WIF_DATASET_BYTES = 225_443_520
WIF_DATASET_SHA256 = \
    "2f828eabd96a45f3872390f901240ea2259a1e9a629247010f42ce7a31cc46be"

WIF_DATASET_ASSET = TableAsset(
    WIF_DATASET_FILE, WIF_DATASET_BYTES, WIF_DATASET_SHA256)

#: False, and it stays False: 225,443,520 bytes is over PyPI's 100 MB
#: per-file cap and over GitHub's 100 MiB blob limit, and the release
#: checklist's size gates are explicit.  Named as a constant for the same
#: reason ``thompson_aerosol_contract.AEROSOL_ASSET_REDISTRIBUTED`` is:
#: the packaging exclusion, the fetch route and this prose have to move
#: together or not at all.
WIF_DATASET_REDISTRIBUTED = False

#: Human-readable origin, quoted verbatim in the fail-closed refusal so an
#: operator can act on it without reading any source.  NCAR distributes it
#: with the WRF/WPS static input data for aerosol-aware Thompson
#: (``mp_physics=28`` with ``use_aero_icbc``); every WRF tree that has ever
#: run the scheme from climatology has a copy, and metgrid reads it through
#: ``constants_name`` exactly as ArWen does.
WIF_DATASET_SOURCE = (
    "NCAR's WRF/WPS static input data for aerosol-aware Thompson "
    "microphysics -- the global monthly water/ice-friendly aerosol "
    "climatology WPS routes through metgrid's constants_name.  Any WRF "
    "installation that runs mp_physics=28 from climatology already has "
    "the file; copy it from there.  It is fixed data: no WRF build and no "
    "gpuwm code path regenerates it.")

#: File-level override, checked first after an explicit argument.
WIF_DATASET_PATH_ENV = "GPUWM_WIF_CLIMATOLOGY"

#: Directory-level override.  Same relationship to the file override that
#: ``GPUWM_THOMPSON_TABLE_ROOT`` has to ``GPUWM_THOMPSON_CCN_ACTIVATE``.
WIF_DATASET_ROOT_ENV = "GPUWM_WIF_DATA_ROOT"


class MissingWifClimatologyDataset(FileNotFoundError):
    """The WIF climatology dataset could not be located.

    A ``FileNotFoundError`` subclass so ordinary ``except
    FileNotFoundError`` handling still applies, and a distinct type so a
    preflight can tell "the operator has not staged the 215 MiB dataset"
    apart from "the config named a path that is wrong".

    DEFINED HERE AND ONLY HERE.  ``gpuwm/ingest/wif_climatology.py``
    raises the same refusal from its own resolver and re-exports THIS
    class rather than declaring a second one: two same-named exceptions
    in two modules is an ``except`` that catches one of them and lets the
    other escape, and both are reachable from a single mp=28 run.
    """


def user_wif_data_root() -> Path:
    """``~/.gpuwm/wif`` -- where ``gpuwm fetch-tables --wif`` stages it.

    Beside ``~/.gpuwm/tables/thompson`` and ``~/.gpuwm/bridges``, and
    outside any install, for the reason ``table_assets`` records: staging
    a quarter-gigabyte inside site-packages means the next
    ``pip install --upgrade`` deletes it without saying so.

    Its own directory rather than the Thompson table root because a table
    root is validated as a COMPLETE SET; dropping a WPS intermediate file
    into it would make "complete" unanswerable for both.
    """

    return Path.home() / ".gpuwm" / "wif"


def resolve_wif_data_root(root: str | Path | None = None, *,
                          env: Mapping[str, str] | None = None) -> Path:
    """Explicit argument, then ``GPUWM_WIF_DATA_ROOT``, then the user root."""

    if root is not None:
        return Path(root)
    environ = os.environ if env is None else env
    override = environ.get(WIF_DATASET_ROOT_ENV)
    if override:
        return Path(override)
    return user_wif_data_root()


def _missing_dataset_message(candidates, *, explicit: bool) -> str:
    looked = "\n".join(f"    {candidate}" for candidate in candidates)
    named = ""
    if explicit:
        named = (
            "  the path above was NAMED (config wif_climatology_path, or "
            f"{WIF_DATASET_PATH_ENV}); a named path that does not exist is "
            "an error even if a staged copy would have worked, because "
            "silently ignoring an operator's override is how a run ends up "
            "reading a dataset nobody chose.\n")
    return (
        f"{WIF_DATASET_FILE} is required by the mp_physics=28 WIF aerosol "
        "climatology ingest (aer_init_opt=1 with wif_input_opt=1) and was "
        "not found.\n"
        f"  looked for:\n{looked}\n"
        f"{named}"
        "  what it is: WRF's GLOBAL monthly water/ice-friendly aerosol "
        "number climatology in WPS intermediate format (ungrib IFV=5), "
        "288x181 x 30 levels x 12 months x QNWFA/QNIFA/P_WIF.  gpuwm does "
        "NOT redistribute it: at "
        f"{WIF_DATASET_BYTES:,} bytes it is over PyPI's 100 MB per-file cap "
        "and over GitHub's 100 MiB blob limit, so it is staged on demand "
        "exactly as freezeH2O.dat is.\n"
        f"  where to get it: {WIF_DATASET_SOURCE}\n"
        f"    expected {WIF_DATASET_BYTES} bytes, sha256 "
        f"{WIF_DATASET_SHA256}\n"
        "  how to supply it: `gpuwm fetch-tables --wif --from DIR` stages "
        "it from the WRF/WPS tree that already has it into "
        f"{user_wif_data_root()} (exact size + SHA-256 verified BEFORE "
        "install), `gpuwm fetch-tables --wif` downloads it from the "
        "release asset base, or set "
        f"{WIF_DATASET_PATH_ENV} to its full path, or set "
        f"{WIF_DATASET_ROOT_ENV} to a directory containing it, or set "
        "wif_climatology_path in the config.\n"
        "  why this is fatal rather than defaulted: aer_init_opt=1 with "
        "wif_input_opt=1 is a REQUEST for the monthly climatology.  The "
        "only thing gpuwm could fall back to is thompson_init's synthetic "
        "exponential profile (module_mp_thompson.F:493-546) -- which is a "
        "different, perfectly valid configuration that aer_init_opt=0 "
        "already selects.  A silent demotion would produce a run whose "
        "receipt says 'monthly climatology' and whose aerosol came from an "
        "analytic profile, with nothing anywhere saying so.")


def resolve_wif_climatology_path(
        path: str | Path | None = None,
        root: str | Path | None = None, *,
        env: Mapping[str, str] | None = None) -> Path:
    """Locate the WIF dataset or fail closed with an actionable error.

    Precedence, highest first: the explicit ``path`` argument (the
    config's ``wif_climatology_path``), the ``GPUWM_WIF_CLIMATOLOGY``
    file override, then ``<resolved root>/QNWFA_QNIFA_SIGMA_MONTHLY.dat``
    where the root follows the explicit ``root`` argument,
    ``GPUWM_WIF_DATA_ROOT``, and ``~/.gpuwm/wif`` in that order.

    An explicitly named path that does not exist is an error even if the
    staged copy would have worked -- the same rule
    :func:`gpuwm.core.thompson_aerosol_contract.resolve_ccn_activation_path`
    states, for the same reason.

    A directory is accepted wherever a path is: an operator who points at
    the WRF tree rather than at the file inside it has named the right
    thing, and refusing that would be pedantry with no failure behind it.
    """

    environ = os.environ if env is None else env
    explicit = False
    candidates = []
    if path is not None and str(path):
        explicit = True
        candidates.append(Path(path))
    else:
        override = environ.get(WIF_DATASET_PATH_ENV)
        if override:
            explicit = True
            candidates.append(Path(override))
        else:
            candidates.append(
                resolve_wif_data_root(root, env=environ) / WIF_DATASET_FILE)
    looked = []
    for candidate in candidates:
        looked.append(candidate)
        if candidate.is_dir():
            candidate = candidate / WIF_DATASET_FILE
            looked.append(candidate)
        if candidate.is_file():
            return candidate
    raise MissingWifClimatologyDataset(
        _missing_dataset_message(tuple(looked), explicit=explicit))


def validate_wif_dataset(path) -> TableAsset:
    """Content-address one copy at an arbitrary location: size, then SHA-256.

    Size first because it is a ``stat`` and rejects the common mistakes
    (a truncated download, the still-compressed archive, the wrong file
    entirely) without reading 215 MiB.
    """

    source = Path(path)
    if not source.is_file():
        raise MissingWifClimatologyDataset(
            _missing_dataset_message((source,), explicit=True))
    actual_bytes = source.stat().st_size
    if actual_bytes != WIF_DATASET_BYTES:
        raise ValueError(
            f"WIF climatology dataset {source} has {actual_bytes} bytes; "
            f"expected {WIF_DATASET_BYTES} (from {WIF_DATASET_SOURCE})")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != WIF_DATASET_SHA256:
        raise ValueError(
            f"WIF climatology dataset {source} SHA-256 {actual_sha256}; "
            f"expected {WIF_DATASET_SHA256} (from {WIF_DATASET_SOURCE})")
    return WIF_DATASET_ASSET


def wif_dataset_is_staged(root=None, *, env: Mapping[str, str] | None = None):
    """Presence only -- the question a preflight owns.

    Deliberately does NOT hash: absence is one sentence and costs a
    ``stat``; a present-but-wrong file is a different sentence, raised by
    :func:`validate_wif_dataset` at load, and making every preflight pay
    215 MiB of hashing to say "yes" is not a service.
    """

    try:
        resolve_wif_climatology_path(None, root, env=env)
    except MissingWifClimatologyDataset:
        return False
    return True


__all__ = [
    "MissingWifClimatologyDataset",
    "WIF_DATASET_ASSET",
    "WIF_DATASET_BYTES",
    "WIF_DATASET_FILE",
    "WIF_DATASET_PATH_ENV",
    "WIF_DATASET_REDISTRIBUTED",
    "WIF_DATASET_ROOT_ENV",
    "WIF_DATASET_SHA256",
    "WIF_DATASET_SOURCE",
    "resolve_wif_climatology_path",
    "resolve_wif_data_root",
    "user_wif_data_root",
    "validate_wif_dataset",
    "wif_dataset_is_staged",
]
