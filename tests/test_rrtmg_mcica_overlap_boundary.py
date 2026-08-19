"""The cloud-overlap boundary: icld 4/5 exists, and nothing shipped selects it.

``gpuwm/core/rrtmg_mcica.py``'s ``_stochastic_cdf`` grows an exponential
decorrelation profile for ``overlap in (4, 5)``, and it does that through
``_expf32`` -- a Python ``for`` loop over a raveled float32 array, one
``expf`` per element.  A Python loop on a forecast's per-timestep radiation
path would be a defect of exactly the kind the Noah-MP cold start was.

It is not on a forecast path, and this module is what says so rather than
leaving it to a reader's grep.  ``overlap`` is ``int(icld)`` and ``icld`` is
``int(cldovrlp)``, so the question is entirely "can a user deliver
``cldovrlp`` 4 or 5 to the legacy RRTMG adapter", and the answer is no at
four independent gates:

* **No config key.**  ``cldovrlp`` is not a ``RunConfig`` field, so an
  experiment TOML carrying it is refused by name at load.
* **No namelist import.**  A WRF 4/4 namelist declaring ``cldovrlp = 4``
  is refused rather than imported (``gpuwm/namelist_import.py``).
* **No adapter admission.**  ``RRTMGLegacyRadiation._check_pins`` refuses
  every value but 2 before a forecast starts.
* **Not the code that runs anyway.**  The shipped adapter hands the batch
  preps the DEVICE twins, whose ``MCICA_DEVICE_ICLD`` is 2 and which fail
  closed on anything else; the NumPy generators (the only reader of
  ``_expf32``) are the paired scalar authority, reached from tests.

There is no environment variable and no flag that reaches overlap 4/5.  The
only caller is a direct Python call to
:func:`gpuwm.core.rrtmg_mcica.generate_lw_subcolumns` /
:func:`~gpuwm.core.rrtmg_mcica.generate_sw_subcolumns`, which is what the
last test here does -- so this file also proves the branch still WORKS,
i.e. the verdict is "unreachable in production", not "quietly broken".

Prose record: ``docs/rrtmg_legacy_integration.md`` section 5.  All CPU.
"""

from __future__ import annotations

import inspect
from dataclasses import fields

import numpy as np
import pytest

from gpuwm.config import RunConfig
from gpuwm.core import rrtmg_mcica as mcica
from gpuwm.core.rrtmg_legacy import RRTMGLegacyRadiation, _PINNED_OPTIONS

#: Every spelling WRF and this tree use for the subcolumn-overlap choice.
_OVERLAP_SPELLINGS = ("cldovrlp", "icld", "idcor", "overlap")

#: The two settings whose CDF walk reads ``_expf32``.
_EXPONENTIAL_OVERLAPS = (4, 5)


# ---------------------------------------------------------------------------
# Gate 1: the config surface has no such key
# ---------------------------------------------------------------------------

def test_no_run_config_field_can_carry_a_cloud_overlap_choice():
    """``RunConfig`` is the config file's whole key vocabulary.

    ``gpuwm/config.py``'s ``load_config`` builds its accepted key set from
    ``fields(RunConfig)`` and refuses anything else by name, so a field
    that does not exist here is a key no config file can deliver.
    """
    names = {field.name for field in fields(RunConfig)}
    assert names, "RunConfig must be a dataclass with fields"
    offenders = sorted(
        name for name in names
        if any(spelling in name for spelling in _OVERLAP_SPELLINGS))
    assert offenders == [], (
        "a cloud-overlap selector reached the config surface: "
        f"{offenders}; the McICA exponential-overlap walk is NumPy-only "
        "and its CDF stage runs a Python per-element expf loop")


def test_an_experiment_toml_naming_the_overlap_is_refused_by_name(tmp_path):
    """The refusal a user actually meets, through the real loader."""
    from test_namelist_import import _import_with

    from gpuwm.experiment import load_experiment

    toml_text, _report = _import_with(tmp_path)
    assert "cldovrlp" not in toml_text
    # The importer's own output loads: the refusal below is caused by the
    # added key and by nothing else.
    clean = tmp_path / "clean.toml"
    clean.write_text(toml_text)
    load_experiment(clean)

    poisoned = tmp_path / "poisoned.toml"
    poisoned.write_text(
        toml_text.replace("[shared]", "[shared]\ncldovrlp = 4", 1))
    with pytest.raises(ValueError, match="does not have a key 'cldovrlp'"):
        load_experiment(poisoned)


# ---------------------------------------------------------------------------
# Gate 2: the namelist importer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("overlap", _EXPONENTIAL_OVERLAPS)
def test_a_wrf_namelist_asking_for_exponential_overlap_is_refused(
        tmp_path, overlap):
    """A real WRF 4/4 namelist is the one route that could smuggle it in."""
    from test_namelist_import import _import_with

    with pytest.raises(ValueError, match="cldovrlp"):
        _import_with(tmp_path,
                     extra_physics=f" cldovrlp = {overlap}, {overlap},\n")

    # The same namelist with the implemented value imports, so the refusal
    # above is the overlap value and not the added key.
    toml_text, _report = _import_with(tmp_path,
                                      extra_physics=" cldovrlp = 2, 2,\n")
    assert "ra_physics = 4" in toml_text
    assert "cldovrlp" not in toml_text


# ---------------------------------------------------------------------------
# Gate 3: the adapter refuses before a forecast starts
# ---------------------------------------------------------------------------

def test_the_legacy_adapter_admits_maximum_random_overlap_and_nothing_else():
    """``_check_pins`` is evaluated on every ``__call__``."""
    from types import SimpleNamespace

    assert _PINNED_OPTIONS["cldovrlp"] == 2

    adapter = object.__new__(RRTMGLegacyRadiation)
    adapter.o3input = 2

    def cfg(cldovrlp):
        return SimpleNamespace(icloud=1, cldovrlp=cldovrlp, idcor=0,
                               ghg_input=0, aer_opt=0, swint_opt=0,
                               o3input=2)

    adapter._check_pins(cfg(2))
    for overlap in (1, 3, *_EXPONENTIAL_OVERLAPS):
        with pytest.raises(NotImplementedError, match="cldovrlp=2"):
            adapter._check_pins(cfg(overlap))


def test_the_shipped_adapter_hands_the_preps_the_device_twins():
    """What the forecast path actually calls, read off the adapter.

    The batch preps fall back to the NumPy generators when
    ``subcolumn_generator`` is None (``rrtmg_legacy_prep``), so "the
    NumPy generator is unreachable" is a property of the CALLER.  If this
    ever regresses to the NumPy entry, a shipped forecast spends ~94 s per
    251,001-column call instead of ~0.24 s -- and it becomes possible for
    a future overlap knob to land on the Python expf loop.
    """
    source = inspect.getsource(RRTMGLegacyRadiation.__call__)
    assert "_mcica.gpu_generate_lw_subcolumns" in source
    assert "_mcica.gpu_generate_sw_subcolumns" in source
    assert "_mcica.generate_lw_subcolumns" not in source
    assert "_mcica.generate_sw_subcolumns" not in source
    assert "cldovrlp=2" in source

    assert mcica.MCICA_DEVICE_ICLD == 2


@pytest.mark.parametrize("overlap", _EXPONENTIAL_OVERLAPS)
def test_the_device_twin_fails_closed_on_exponential_overlap(overlap):
    """Fail-closed BEFORE any cupy import, so this runs on any box."""
    with pytest.raises(NotImplementedError, match="icld=2"):
        mcica.gpu_generate_lw_subcolumns(
            0, 1, 1, overlap, 150, 0, None, None, None, None, None,
            None, None, None, None, None, 0, 0.0, None)


# ---------------------------------------------------------------------------
# The branch itself: live, exercised, and running the Python loop
# ---------------------------------------------------------------------------

def _column_deck(ncol=3, nlay=5):
    play = np.linspace(900.0, 300.0, nlay, dtype=np.float32)
    play = np.repeat(play[None, :], ncol, axis=0)
    cldfrac = np.full((ncol, nlay), np.float32(0.4))
    cldfrac[:, 0] = np.float32(0.0)          # a clear layer for icld=5
    hgt = np.repeat(
        np.linspace(0.0, 12000.0, nlay, dtype=np.float32)[None, :],
        ncol, axis=0)
    ones = np.full((ncol, nlay), np.float32(1.0e-3))
    return dict(
        iplon=0, ncol=ncol, nlay=nlay, permuteseed=150, irng=0,
        play=play, cldfrac=cldfrac, ciwp=ones, clwp=ones, cswp=ones,
        rei=np.full((ncol, nlay), np.float32(30.0)),
        rel=np.full((ncol, nlay), np.float32(15.0)),
        res=np.full((ncol, nlay), np.float32(50.0)),
        tauc=np.full((mcica.NBNDLW, ncol, nlay), np.float32(0.1)),
        hgt=hgt, idcor=0, juldat=100.0,
        lat=np.full(ncol, np.float32(30.0)))


@pytest.mark.parametrize("overlap", _EXPONENTIAL_OVERLAPS)
def test_the_numpy_authority_still_walks_exponential_overlap(
        monkeypatch, overlap):
    """Positive evidence, both halves.

    The branch produces a DIFFERENT subcolumn mask from maximum-random --
    an identical answer would mean the parametrisation never reached the
    code under test -- and it gets there through ``_expf32``, counted here
    so the "Python loop" half of the declaration is measured rather than
    asserted from a reading of the source.
    """
    calls = {"n": 0, "elements": 0}
    real = mcica._expf32

    def counting(values):
        calls["n"] += 1
        calls["elements"] += int(np.asarray(values).size)
        return real(values)

    monkeypatch.setattr(mcica, "_expf32", counting)

    deck = _column_deck()
    exponential = mcica.generate_lw_subcolumns(icld=overlap, **deck)
    assert calls["n"] == 1
    assert calls["elements"] == deck["ncol"] * (deck["nlay"] - 1)

    maximum_random = mcica.generate_lw_subcolumns(icld=2, **deck)
    assert calls["n"] == 1, "icld=2 must not read the exponential profile"
    assert not np.array_equal(exponential["cldfmcl"],
                              maximum_random["cldfmcl"])
