"""The adaptive-timestep block moves the checkpoint bytes by EXACTLY itself.

Twelve fields joining ``RunConfig`` necessarily changes the checkpoint,
because the header echoes the config.  What must be proved is that it
changes it by those twelve keys and by nothing else -- the same
construction, and the same argument, ``_digest_without_the_wif_config_keys``
makes for the pair that landed with the mp=28 aerosol work.

The restart IDENTITY is a separate question and is scoped in
``gpuwm.core.model.restart_identity_payload``: with the controller off the
whole block drops out, so an existing checkpoint still resumes.  That is
covered in ``test_adaptive_timestep_surface.py``; this file covers the
bytes.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from datetime import timedelta

import numpy as np
import pytest

import gpuwm.io.restart as restart
from gpuwm.core.model import ADAPTIVE_TIMESTEP_RUN_FIELDS
from test_restart import (_VOLATILE_CHECKPOINT_HEADER,
                                _canonical_member_digest,
                                _sealed_tree_fixture)

#: ``_canonical_member_digest`` on the tree immediately BEFORE the
#: adaptive-timestep block joined RunConfig, harvested from that tree.
_PRE_ADAPTIVE_ROOT_DIGEST = \
    "c85c921d720d92be2e138d889eae3db2ce978b5c3fc63f867340a86cf26faac1"
_PRE_ADAPTIVE_CHILD_DIGEST = \
    "fb448bfd57196090e0aa3ac025f5ec7234764e1f80f628aded9e5b991a75312d"


def _digest_without_the_adaptive_config_keys(path) -> str:
    """The canonical member digest as it would read WITHOUT the block.

    Same construction as :func:`_canonical_member_digest` with exactly
    three substitutions: the twelve keys are dropped from the config echo,
    and the two hashes the writer derives from that echo are recomputed
    from the trimmed one using the writer's own helpers.  Every array
    member is hashed unchanged.
    """
    with np.load(path, allow_pickle=False) as data:
        header = json.loads(bytes(bytearray(
            data[restart._HEADER_KEY])).decode("utf-8"))
        for name in _VOLATILE_CHECKPOINT_HEADER:
            header.pop(name, None)
        # eta_levels was appended later (80a3009c2/06c29b747).
        # Unwind that separate config addition to reach the historical
        # pre-adaptive tree; leave every array and non-config header bound.
        for key in ADAPTIVE_TIMESTEP_RUN_FIELDS + ("eta_levels",):
            header["config"].pop(key, None)
        values = {key: value for key, value in header["config"].items()
                  if key not in restart.CONFIG_RUN_LENGTH_FIELDS
                  and key not in restart.CONFIG_DIAGNOSTIC_FIELDS}
        setup = copy.deepcopy(header["physics_setup"])
        setup["configuration_sha256"] = restart._json_sha256(
            restart._json_value(values, "RunConfig"))
        header["physics_setup"] = setup
        header["physics_setup_fingerprint"] = restart._json_sha256(setup)
        digest = hashlib.sha256()
        digest.update(json.dumps(header, sort_keys=True).encode("utf-8"))
        for name in sorted(data.files):
            if name == restart._HEADER_KEY:
                continue
            host = data[name]
            digest.update(name.encode("utf-8"))
            digest.update(str(host.dtype).encode("utf-8"))
            digest.update(str(host.shape).encode("utf-8"))
            digest.update(host.tobytes(order="C"))
    return digest.hexdigest()


def _write(monkeypatch, tmp_path):
    source, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0, payload_seed=31)
    root = restart.write_tree_restart(
        tmp_path, source, start + timedelta(seconds=3600))
    child = next(p for p in tmp_path.glob("gpuwmrst_d02_*.npz"))
    return root, child


def test_removing_the_block_restores_the_pre_adaptive_digest(
        monkeypatch, tmp_path):
    root, child = _write(monkeypatch, tmp_path)
    assert (_digest_without_the_adaptive_config_keys(root)
            == _PRE_ADAPTIVE_ROOT_DIGEST)
    assert (_digest_without_the_adaptive_config_keys(child)
            == _PRE_ADAPTIVE_CHILD_DIGEST)


def test_the_keys_really_are_in_the_echo(monkeypatch, tmp_path):
    """So the reconstruction removes something rather than succeeding vacuously.

    Without this the test above would pass just as happily if the block
    had never reached the checkpoint at all -- which is the shape of gate
    that proves nothing.
    """
    root, _ = _write(monkeypatch, tmp_path)
    with np.load(root, allow_pickle=False) as data:
        echo = json.loads(bytes(bytearray(
            data[restart._HEADER_KEY])).decode("utf-8"))["config"]
    for key in ADAPTIVE_TIMESTEP_RUN_FIELDS:
        assert key in echo, key


def test_the_block_really_does_move_the_digest(monkeypatch, tmp_path):
    """The counterpart: the trimmed digest must NOT equal the plain one."""
    root, _ = _write(monkeypatch, tmp_path)
    assert (_canonical_member_digest(root)
            != _digest_without_the_adaptive_config_keys(root))


# ------------------------------------- the acoustic substep count moves

def _run_cfg(**over):
    from gpuwm.config import RunConfig

    base = dict(nx=12, ny=12, nz=8, dx=1.0e4, dy=1.0e4, ztop=1.5e4,
                dt=30.0, run_seconds=300.0)
    base.update(over)
    return RunConfig(**base)


def test_time_step_sound_is_state_not_identity_under_an_adaptive_clock():
    """``_apply`` rewrites it from the live dt, so it moves with the dt.

    Upstream zeroes ``time_step_sound`` whenever the adaptive clock is on
    (``start_em.F:966``) precisely so ``solve_em`` derives the acoustic
    substep count from the live dt, and ``adaptive_clock._apply`` does the
    same here through ``wrf_num_sound_steps``.  Bound as identity, a
    checkpoint written after dt grew past the four-substep floor -- the
    doc's 10 km case reaches dt ~ 76 s, where the count is 6 against a
    configured 4 -- could be resumed only by a run that had adapted to the
    identical step, i.e. by nothing.
    """
    live = _run_cfg(use_adaptive_time_step=True, dt=30.0, time_step_sound=4)
    grown = _run_cfg(use_adaptive_time_step=True, dt=76.0, time_step_sound=6)
    restart._require_config_match(
        dataclasses.asdict(grown), live, "checkpoint")
    assert (restart._configuration_fingerprint(live)
            == restart._configuration_fingerprint(grown))


def test_a_fixed_clock_still_binds_the_acoustic_substep_count():
    """Where it really is a setting, it is still compared to the bit."""
    live = _run_cfg(time_step_sound=4)
    grown = _run_cfg(time_step_sound=6)
    with pytest.raises(restart.RestartMismatchError,
                       match="time_step_sound"):
        restart._require_config_match(
            dataclasses.asdict(grown), live, "checkpoint")
    assert (restart._configuration_fingerprint(live)
            != restart._configuration_fingerprint(grown))
