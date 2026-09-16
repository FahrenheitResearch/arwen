"""RUC's admission checks, resolved by one host read instead of one per field.

Every RUC entry point validates its inputs before it launches anything: a
float field must be finite, a soil-property column must be positive, a root
count must lie inside the resolved geometry.  Each of those tests used to end
in ``bool(...)`` on a device reduction, and on the device namespace a
``bool()`` is a copy back to pageable host memory, which is a stream
synchronisation.  One land-surface call validates about 122 input fields
twice over and its leaves validate their own, so a single call spent roughly
568 synchronisations deciding that nothing was wrong -- 89.8 % of the
forecast's blocking device reads per root step, measured over 50 root steps
of the 299x299 + 282x129 pair on a 5090.

The check itself is not the cost; the synchronisation is.  A reduction that
writes its verdict into a slot of one flag block costs a kernel launch and
nothing else, and the whole block is read once.  This module is that flag
block, and the batch it fills.

Nothing here changes what is validated, what trips it, or what the refusal
says.  A batch raises the FIRST queued failure in submission order -- the
same field, with the same message, that the field-by-field code raised -- and
a caller that must refuse for some other reason flushes the batch first, so a
non-finite field queued earlier still beats a shape error found later.

**Against the two mechanisms next to it.**
:func:`gpuwm.core.health_ledger.check_finite` defers a verdict PAST the step,
is opt-in, and appends a sentence to the failure saying the run advanced
before the failure was detected; it exists so a step's launch sequence stops
being a function of the data, which is what lets a graph be captured.  This
batch resolves inside the call, is on by default, and changes no message: it
is not a deferral at all, it is one read instead of many.  The two compose --
a ledger, if one is installed, still sees every reduction this batch runs.
``gpuwm.core.mynn_pbl_gpu``'s ``_flag_mask`` is the same idea applied to
MYNN's own validation in 2.7.4, and the scan kernel here is modelled on its
``mynn_validate_batch``; folding the two onto one kernel is a follow-up that
would touch a file this lane does not own.

The batched scan itself belongs to the namespace: a device namespace that can
scan many arrays in one launch exposes ``ruc_validate_batch`` (the way
``gpuwm.core.ruc_gpu.RUC_DEVICE_ARRAYS`` exposes ``ruc_tanhf_glibc``), and a
namespace without one falls back to one reduction per array, which is still
one host read for the whole batch.  numpy is such a namespace, so the host
transcription and its oracle fixtures run through exactly this code with no
device at all.
"""

from __future__ import annotations

import numpy as _np

__all__ = [
    "RucValidationBatch",
    "VALIDATION_SCAN_GROUP",
]

#: Arrays per batched-scan launch.  The launch takes one pointer/length pair
#: per array, so the group size is what bounds both the argument list and the
#: number of compiled variants; a batch longer than this is several launches
#: and still one host read.
VALIDATION_SCAN_GROUP = 48

#: Below this many eligible arrays the batched launch is not worth its
#: compile, and one reduction each costs the same single read.
_SCAN_MINIMUM = 4


class RucValidationBatch:
    """Admission tests queued now, decided by one host read at :meth:`flush`.

    ``arrays`` is the namespace the caller's data lives in -- numpy for the
    host transcription, ``RUC_DEVICE_ARRAYS`` for a forecast.  A batch with
    no namespace is inert and every test runs immediately, which is what the
    default argument of each validation helper gives an oracle caller.
    """

    def __init__(self, arrays):
        self._arrays = arrays
        self._entries: list[tuple[str, object, object]] = []

    def __len__(self) -> int:
        return len(self._entries)

    def __enter__(self) -> "RucValidationBatch":
        return self

    def __exit__(self, kind, value, traceback):
        if kind is None:
            self.flush()
        else:
            self._entries.clear()
        return False

    # -- queueing ---------------------------------------------------------

    def finite(self, array, name: str):
        """Queue ``name must be finite`` over ``array``; return ``array``."""
        self._entries.append(("finite", array, f"{name} must be finite"))
        return array

    def finite_message(self, array, message):
        """Queue a finiteness test whose refusal is not ``name must be finite``."""
        self._entries.append(("finite", array, message))
        return array

    def refuse_if_any(self, condition, message):
        """Queue ``message`` for when any element of ``condition`` is true.

        ``message`` may be a callable, evaluated only on the failing path, so
        a refusal that names the offending value ("RUC nroot 9 is outside
        1..8") still reads the value it names and still costs nothing when
        the field is good.
        """
        self._entries.append(("any", condition, message))
        return condition

    # -- resolution -------------------------------------------------------

    def flush(self) -> None:
        """Decide every queued test with ONE host read, then raise the first
        failure in submission order."""
        entries = self._entries
        if not entries:
            return
        self._entries = []
        arrays = self._arrays
        count = len(entries)

        scan = getattr(arrays, "ruc_validate_batch", None)
        eligible = [index for index, (kind, payload, _) in enumerate(entries)
                    if kind == "finite" and _scannable(payload)]
        if scan is None or len(eligible) < _SCAN_MINIMUM:
            eligible = []
        batched = set(eligible)
        rest = [index for index in range(count) if index not in batched]
        order = eligible + rest

        flags = arrays.zeros(count, dtype=_np.int32)
        for start in range(0, len(eligible), VALIDATION_SCAN_GROUP):
            group = eligible[start:start + VALIDATION_SCAN_GROUP]
            scan([entries[index][1] for index in group],
                 flags[start:start + len(group)])
        for offset, index in enumerate(rest):
            kind, payload, _ = entries[index]
            slot = len(eligible) + offset
            # A ONE-ELEMENT SLICE, not ``flags[slot]``: element assignment
            # from a rank-0 array goes through the scalar protocol, which on
            # a device namespace is the host read this class exists to avoid.
            if kind == "finite":
                flags[slot:slot + 1] = ~arrays.all(arrays.isfinite(payload))
            else:
                flags[slot:slot + 1] = arrays.any(payload)

        # The one read this class exists for.
        verdicts = _host_list(flags)
        tripped = [False] * count
        for position, index in enumerate(order):
            tripped[index] = bool(verdicts[position])
        for index, bad in enumerate(tripped):
            if bad:
                message = entries[index][2]
                raise ValueError(message() if callable(message) else message)


def _scannable(array) -> bool:
    """True when the batched launch can read ``array`` as it stands."""
    dtype = getattr(array, "dtype", None)
    flags = getattr(array, "flags", None)
    return (dtype == _np.float32 and flags is not None
            and bool(flags.c_contiguous) and int(getattr(array, "size", 0)) > 0)


def _host_list(flags):
    """``flags`` on the host.  One synchronisation, whatever the namespace."""
    return flags.tolist()
