"""One declared card, one device, wherever the estimate is asked for.

``gpuwm domain --card N`` sizes a configuration, writes it, and then runs
``gpuwm check`` on the file it has just written.  Both doors price the
same file against the same declared card and the same declared budget,
and they disagreed: 13.73 GiB from the wizard against 14.72 GiB from the
check, one of them inside the 14.54 GiB budget and the other 0.18 GiB
outside it, so the wizard emitted a configuration its own check refused
with exit 4.

The whole difference is ONE term.  :func:`estimate_experiment` resolves
an absent card to the measured reference device profile before it prices
the non-pool terms and the per-scheme column workspaces, because a
declared card is a named reference device and not an unknown one.  The
shared legacy-RRTMG chunk workspace read the RAW argument instead, so
the wizard -- which declares a card rather than measuring one, and
therefore passes no profile -- priced that workspace for an unknown
device while every other number on its own page was priced for the
reference device.

An unknown device is not the conservative answer here.  The shortwave
chain has no ceiling: with no device it takes a fixed no-device width,
and the 170-SM reference device saturates WIDER than that width.  So the
unknown-device answer is the narrower workspace and the optimistic
envelope, which is the direction
:func:`card_local_memory_profile` promises never to take.
"""

from __future__ import annotations

import pytest

from gpuwm.cli import main as cli_main
from gpuwm.core import rrtmg_lw as _lw
from gpuwm.core import rrtmg_sw as _sw
from gpuwm.core.preflight import (DeviceLocalMemoryProfile,
                                  card_local_memory_profile,
                                  estimate_experiment)
from gpuwm.experiment import load_experiment

#: The card the two doors are asked about.  Not in this machine, which
#: is the point: both doors resolve the declaration the same way or they
#: do not agree about the file.
DECLARED_GIB = 16.0


def _emit(tmp_path, *, card="16gb", ladder="12-3", source="hrrr"):
    """The wizard's own emission, returned with its exit code.

    The files are written before the check runs, so an emitted
    configuration is available to price even on the exit code this test
    file exists to retire.
    """
    out = tmp_path / "area.toml"
    rc = cli_main(["domain", "--point=39.7,-96.6", "--card", card,
                   "--ladder", ladder, "--source", source,
                   "--cycle", "2026-07-28T05", "--out", str(out)])
    return rc, out


def test_the_shortwave_width_of_an_unknown_device_is_the_narrow_one():
    """The reading the rest of this file rests on.

    Longwave has a ceiling and an unknown device takes it, so longwave
    is priced HIGH with no device.  Shortwave has none: with no device
    it takes a fixed width, and a 170-SM reference device saturates
    past it.  The workspace is the max over both chains' phases, and on
    a two-domain 49-level configuration the shortwave phase is the
    maximum, so the unknown device prices the smaller workspace.
    """

    reference = card_local_memory_profile(DECLARED_GIB)
    threads = reference.resident_thread_capacity

    assert _sw.sw_batch_column_chunk(50, resident_threads=0) == 2048
    assert _sw.sw_batch_column_chunk(50, resident_threads=threads) == 2560

    ceiling = _lw.LW_BATCH_COLUMN_CHUNK_CEILING
    assert _lw.batch_column_chunk(
        _lw.NGPTLW, ceiling, resident_threads=0) == ceiling
    assert _lw.batch_column_chunk(
        _lw.NGPTLW, ceiling, resident_threads=threads) < ceiling


def test_a_declared_card_prices_one_workspace_whichever_door_asks(tmp_path):
    """The defect, at the estimator.

    The wizard calls this function with ``profile=None`` and
    ``vram_gib=16``; ``gpuwm check`` calls it with the resolved
    reference profile and the same ``vram_gib``.  Same file, same
    declared card, and before this the two calls returned workspaces
    2.89 and 3.62 GiB apart, which carried into the itemized estimate
    and then through the envelope factor into a 0.99 GiB difference in
    the answer each door printed.
    """

    _, out = _emit(tmp_path)
    exp = load_experiment(out)
    reference = card_local_memory_profile(DECLARED_GIB)

    undeclared = estimate_experiment(exp, vram_gib=DECLARED_GIB, profile=None)
    resolved = estimate_experiment(exp, vram_gib=DECLARED_GIB,
                                   profile=reference)

    assert undeclared.workspace_bytes == resolved.workspace_bytes
    assert undeclared.alloc_estimate_bytes == resolved.alloc_estimate_bytes
    assert undeclared.peak_envelope_bytes == resolved.peak_envelope_bytes


def test_a_measured_card_still_prices_its_own_workspace(tmp_path):
    """The fix resolves an ABSENT card and overrides no present one.

    A caller that measured the device it is sizing for hands that
    profile in, and a 128-SM card must not be priced on the 170-SM
    reference: that is the mixing this repair exists to stop, in the
    other direction.
    """

    _, out = _emit(tmp_path)
    exp = load_experiment(out)
    measured = DeviceLocalMemoryProfile(
        name="present card", multiprocessor_count=128,
        max_threads_per_multiprocessor=1536)

    on_card = estimate_experiment(exp, vram_gib=DECLARED_GIB,
                                  profile=measured)
    reference = estimate_experiment(
        exp, vram_gib=DECLARED_GIB,
        profile=card_local_memory_profile(DECLARED_GIB))

    assert on_card.workspace_bytes < reference.workspace_bytes
    assert on_card.local_memory_profile is measured


@pytest.mark.parametrize("source", ["hrrr", "era5"])
def test_the_wizard_emits_a_configuration_its_own_check_admits(
        tmp_path, capsys, source):
    """The door, end to end.

    ``gpuwm domain`` fits every emitted level against the budget and
    then hands the file to ``gpuwm check``.  With the two estimates
    disagreeing it wrote a file, printed 0.81 GiB of headroom, and then
    exited 4 on the same file with 0.18 GiB of overrun.  One estimate
    means the fit is sized on the number the check will read, so the
    emission either fits or is never written.
    """

    rc, out = _emit(tmp_path, source=source)
    printed = capsys.readouterr().out
    assert rc == 0, printed
    assert out.is_file()
    assert "gpuwm check FAILED" not in printed
