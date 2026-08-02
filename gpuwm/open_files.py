"""The open-file budget a preparation needs, asked for before it starts.

A GFS preparation holds one descriptor per decoded array for the whole
of the bundle write, and the array count is
``forcing times x fields per time``.  That is linear in forecast length,
so on a box with the ordinary 1024 soft ``RLIMIT_NOFILE`` there is a
forecast length past which no GFS run can be preprocessed at all.
Measured on the shipped 1.4.1 build (RTX 4080, Ubuntu 24.04, soft limit
1024, hard limit 1048576), a 120 h series -- 41 forcing times, 984
arrays -- peaked at **1023** descriptors and died:

    rw-wps --source gfs: [Errno 24] Too many open files:
      '/.../gpuwm-gfs-bridge-32m6ic0i/decoded/f000'

The same command with the soft limit raised to the hard limit completed
at rc 0 and wrote its proof.  Nothing else differed.

Two things are wrong there and this module fixes the second:

1. Holding 984 descriptors at once is retention nobody needs -- the
   arrays are written and never reopened.  Closing them as they are
   written is the better fix and belongs to the bundle writer; it is
   NOT done here, because that path is hash-bound and this one is not.
2. The soft limit was never raised, though the hard limit was a
   thousand times larger and raising it needs no privileges.  A process
   that knows how many descriptors its work needs should ask for them,
   and if it cannot have them it should say so BEFORE spending the
   decode rather than discovering it partway through -- as an
   ``[Errno 24]`` from whichever unlucky call happened to be next,
   which is how this arrived: the message above names a path the run
   had already read successfully.

The ceiling is real either way.  This module removes it wherever the
hard limit allows, and where it does not, converts a mid-run EMFILE
into a sentence naming the number and the command that grants it.
"""

from __future__ import annotations

#: Descriptors a preparation holds per forcing time.  The decoded-array
#: count per time (:data:`gpuwm.gfs_direct._THREE_D` plus
#: :data:`~gpuwm.gfs_direct._TWO_D`), which is what a 41-time snapshot
#: counted (984 = 41 x 24) -- and which a 21-time run then exceeded 1024
#: anyway, so treat this as a LOWER BOUND on the requirement, never as
#: the requirement.  It is used only to decide whether a capped
#: deployment must be refused; the limit itself is raised as far as it
#: will go regardless.
DEFAULT_ARRAYS_PER_FORCING_TIME = 24

#: Descriptors reserved for everything that is not per-forcing-time:
#: stdio, the CUDA device nodes (18 on the measured box), the bridge
#: pipes, shared libraries, the receipts being written.  Measured peak
#: minus the per-time arrays was 39; this is that with room.
DESCRIPTOR_OVERHEAD = 128


def required_descriptors(forcing_times: int,
                         arrays_per_forcing_time: int
                         = DEFAULT_ARRAYS_PER_FORCING_TIME) -> int:
    """Descriptors a preparation of this many forcing times will hold."""

    return (max(int(forcing_times), 0) * int(arrays_per_forcing_time)
            + DESCRIPTOR_OVERHEAD)


def _rlimit():
    """``(resource, soft, hard)``, or ``None`` where there is no such limit.

    Windows has no ``RLIMIT_NOFILE`` and no ``resource`` module.  That is
    not a degraded case to warn about -- the limit this module raises
    does not exist there -- so the answer is "no opinion", and every
    caller treats ``None`` as "nothing to do".
    """

    try:
        import resource
    except ImportError:
        return None
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (OSError, ValueError):
        return None
    return resource, soft, hard


def ensure_descriptor_headroom(needed: int, *, what: str) -> str | None:
    """Raise the soft open-file limit to ``needed``; refusal, or ``None``.

    ``None`` means the process may proceed -- either the limit was
    already sufficient, or it has been raised and now is.  A string is a
    refusal for the caller to raise, and it is deliberately produced
    BEFORE the work rather than during it.

    The soft limit only is moved, and only upward, and never past the
    hard limit: raising your own soft limit toward your own hard limit
    is unprivileged and is what the two limits are for.  The hard limit
    is not touched at all, so a deployment that lowered it on purpose
    keeps its decision and gets told what it costs.
    """

    limits = _rlimit()
    if limits is None:
        return None
    resource, soft, hard = limits
    started_at = soft
    if soft == resource.RLIM_INFINITY:
        return None

    # UNCONDITIONALLY, and to the WHOLE hard limit -- not to the modelled
    # `needed`, and not only when `needed` exceeds the current soft
    # limit.  Both of those were tried and both were wrong, measured:
    #
    #   * 120 h / 41 forcing times: the peak was 1023 against a 1024
    #     limit, i.e. clamped, so its true requirement is unknown from
    #     above.  A model landing near it still fails.
    #   * 60 h / 21 forcing times on a 60x48 grid: ALSO EMFILE, though
    #     the per-forcing-time model puts it at 632, comfortably under
    #     1024.  So the real requirement is not the decoded-array count,
    #     and a raise gated on that model would have left this case --
    #     a two-and-a-half-day forecast on a tiny domain -- still broken.
    #
    # `needed` therefore decides only whether to REFUSE, never whether
    # to ask.  RLIMIT_NOFILE is a per-process guard rail; taking your
    # own hard limit is unprivileged and reversible, and the hard limit
    # -- someone's actual policy -- is never touched.
    for candidate in (hard, needed):
        if candidate == resource.RLIM_INFINITY:
            candidate = max(needed, 1 << 20)
        if candidate <= soft:
            continue
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (candidate, hard))
        except (OSError, ValueError):
            continue        # e.g. a platform OPEN_MAX below the hard limit
        soft = candidate
        break
    if soft >= needed:
        return None
    return (
        f"{what} needs about {needed} open file descriptors and this "
        f"process can have at most {soft} (it raised its own soft limit "
        f"from {started_at} to {soft}; the hard limit is {hard}).\n"
        f"  Refusing before the decode: past this point the shortfall "
        f"arrives as an [Errno 24] from whichever file happened to be "
        f"next, which names an innocent path.\n"
        f"  remedy: ulimit -n {needed}   # then re-run\n"
        f"  # The SOFT limit is raised automatically and needs no "
        f"action; it is the hard limit of {hard} that binds here, and "
        f"raising that is your administrator's "
        f"(/etc/security/limits.conf, or a service's LimitNOFILE).\n"
        f"  # or preprocess a shorter forecast: the requirement is "
        f"linear in the number of forcing times")


__all__ = ["DEFAULT_ARRAYS_PER_FORCING_TIME", "DESCRIPTOR_OVERHEAD",
           "ensure_descriptor_headroom", "required_descriptors"]
