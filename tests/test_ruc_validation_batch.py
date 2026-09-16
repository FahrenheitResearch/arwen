"""RUC admits its inputs with one device read, and refuses exactly as before.

``gpuwm.core.ruc_validation.RucValidationBatch`` replaced a per-field
``bool(...)`` on a device reduction -- about 568 blocking reads and 568
stream synchronisations for one land-surface call -- with one flag block read
once.  Two things have to be true for that to be an optimisation rather than
a loss of a guard:

* every condition the field-by-field reads guarded still refuses, with the
  same message, naming the same field or the same value.  The list below was
  written from the code as it stood before the batch, so a refusal that were
  quietly dropped fails here;
* the reads really are gone.  ``ruc_device_read_counter`` counts them the way
  the profiler counted them on the card, and its per-site counts are pinned
  against the profile of record in that module's own docstring.

Each claim carries its falsification: a batch that never raised, and a
counter that never counted, would both pass a one-sided test.
"""

from __future__ import annotations

import numpy as np
import pytest

import ruc_device_read_counter as counter
from gpuwm.core.ruc import (RUC_DRIVER_ARW_FORCING, RUC_DRIVER_COLUMN_FORCING,
                            RUC_DRIVER_COLUMN_STATE, RUC_DRIVER_PROFILE_STATE,
                            RUC_SFCTMP_COLUMN_INPUTS,
                            RUC_SFCTMP_PROFILE_INPUTS, _horizontal_float_field,
                            _root_count_field, ruc_land_surface_step,
                            ruc_surface_temperature_step)
from gpuwm.core.ruc_runtime import (C1SN, C2SN, DEFINED_ILNB, ISNCOVR_OPT,
                                    XICE_THRESHOLD, RucRuntimeParameters)
from gpuwm.core.ruc_validation import RucValidationBatch
from test_ruc import SFCTMP_ORACLE, _sfctmp_call, _sfctmp_oracle
from test_ruc_column_batching import _columns

_PARAMS = RucRuntimeParameters()

#: Every float field the driver admits.  The refusal for each of them used to
#: be its own ``bool(np.all(np.isfinite(...)))``.
_DRIVER_FIELDS = (tuple(RUC_DRIVER_PROFILE_STATE)
                  + tuple(RUC_DRIVER_COLUMN_STATE)
                  + tuple(RUC_DRIVER_COLUMN_FORCING)
                  + tuple(RUC_DRIVER_ARW_FORCING))


def _driver_case(columns: int = 4):
    values, ivgtyp, isltyp = _columns(columns, snow=False, seed=11)
    for name in RUC_DRIVER_ARW_FORCING:
        values.setdefault(name, np.zeros(columns, np.float32))
    values["sr"] = np.zeros(columns, np.float32)
    keywords = dict(
        dt=12.0, ktau=2, zs=_PARAMS.zs, ivgtyp=ivgtyp, isltyp=isltyp,
        em_core=1, frpcpn=True, ilnb=DEFINED_ILNB, ilnb_chain=False,
        c1sn=C1SN, c2sn=C2SN, isncovr_opt=ISNCOVR_OPT, lakemodel=1,
        xice_threshold=float(XICE_THRESHOLD),
        mminlu=_PARAMS.dataset_identifier, parameters=_PARAMS.bundle,
    )
    return values, keywords


# ---------------------------------------------------------------------------
# the refusals


@pytest.mark.parametrize("name", _DRIVER_FIELDS)
def test_every_driver_input_still_refuses_by_its_own_name(name):
    values, keywords = _driver_case()
    poisoned = dict(values)
    field = np.array(values[name], dtype=np.float32, copy=True)
    field.reshape(-1)[0] = np.float32("nan")
    poisoned[name] = field
    with pytest.raises(ValueError, match=rf"^{name} must be finite$"):
        ruc_land_surface_step(poisoned, **keywords)


def test_a_clean_driver_call_refuses_nothing():
    """The falsification of the test above: the same grid, unpoisoned."""
    values, keywords = _driver_case()
    ruc_land_surface_step(values, **keywords)


@pytest.mark.parametrize(
    "name", tuple(RUC_SFCTMP_PROFILE_INPUTS) + tuple(RUC_SFCTMP_COLUMN_INPUTS))
def test_every_sfctmp_input_still_refuses_by_its_own_name(name):
    _, field = _sfctmp_oracle(SFCTMP_ORACLE)
    values, keywords = _sfctmp_call(field, 0)
    poisoned = dict(values)
    array = np.array(values[name], dtype=np.float32, copy=True)
    array.reshape(-1)[0] = np.float32("nan")
    poisoned[name] = array
    with pytest.raises(ValueError, match=rf"^{name} must be finite$"):
        ruc_surface_temperature_step(poisoned, **keywords)


def test_a_clean_sfctmp_call_refuses_nothing():
    _, field = _sfctmp_oracle(SFCTMP_ORACLE)
    values, keywords = _sfctmp_call(field, 0)
    ruc_surface_temperature_step(values, **keywords)


def test_the_driver_still_refuses_a_vegetation_category_outside_the_table():
    values, keywords = _driver_case()
    keywords = dict(keywords, ivgtyp=np.zeros(4, np.int32))
    with pytest.raises(ValueError, match=r"^RUC ivgtyp is outside 1\.\."):
        ruc_land_surface_step(values, **keywords)


def test_the_driver_still_refuses_a_soil_category_outside_the_table():
    values, keywords = _driver_case()
    keywords = dict(keywords, isltyp=np.full(4, 99, np.int32))
    with pytest.raises(ValueError, match=r"^RUC isltyp is outside 1\.\."):
        ruc_land_surface_step(values, **keywords)


def test_a_root_count_refusal_still_names_the_count_it_rejected():
    """The message reads the offending value, and only on the failing path."""
    batch = RucValidationBatch(np)
    _root_count_field(np.full(3, 9, np.int32), (3,), arrays=np, nzs=9,
                      batch=batch)
    with pytest.raises(ValueError, match=r"^RUC nroot 9 is outside 1\.\.8$"):
        batch.flush()


def test_a_root_count_inside_the_geometry_queues_nothing_to_raise():
    batch = RucValidationBatch(np)
    _root_count_field(np.full(3, 4, np.int32), (3,), arrays=np, nzs=9,
                      batch=batch)
    batch.flush()


def test_a_shape_refusal_still_loses_to_a_nonfinite_field_queued_earlier():
    """Submission order decides, exactly as the field-by-field code did.

    The old code raised at the FIRST field that failed.  A batch that let a
    later shape error jump the queue would report a different field than the
    run it replaced, which is a changed refusal, not a faster one.
    """
    batch = RucValidationBatch(np)
    _horizontal_float_field(np.full(4, np.nan, np.float32), (4,), "first",
                            arrays=np, batch=batch)
    with pytest.raises(ValueError, match=r"^first must be finite$"):
        _horizontal_float_field(np.zeros(5, np.float32), (4,), "second",
                                arrays=np, batch=batch)


def test_a_shape_refusal_stands_on_its_own_when_nothing_was_queued():
    batch = RucValidationBatch(np)
    with pytest.raises(ValueError, match=r"is not broadcastable to"):
        _horizontal_float_field(np.zeros(5, np.float32), (4,), "second",
                                arrays=np, batch=batch)


def test_the_batch_raises_the_first_failure_not_the_worst_one():
    batch = RucValidationBatch(np)
    batch.finite(np.zeros(4, np.float32), "clean")
    batch.finite(np.full(4, np.inf, np.float32), "early")
    batch.refuse_if_any(np.ones(4, bool), "late")
    with pytest.raises(ValueError, match=r"^early must be finite$"):
        batch.flush()


def test_a_flushed_batch_is_empty_and_a_second_flush_is_quiet():
    batch = RucValidationBatch(np)
    batch.finite(np.zeros(4, np.float32), "clean")
    assert len(batch) == 1
    batch.flush()
    assert len(batch) == 0
    batch.flush()


def test_a_refusal_message_can_read_the_value_it_names():
    batch = RucValidationBatch(np)
    values = np.array([1, 2, 77], np.int32)
    batch.refuse_if_any(values > 9, lambda: f"saw {int(values[values > 9][0])}")
    with pytest.raises(ValueError, match=r"^saw 77$"):
        batch.flush()


def test_a_callable_message_is_never_evaluated_on_the_clean_path():
    def explode():                       # pragma: no cover - must not run
        raise AssertionError("the refusal message was built for a clean field")

    batch = RucValidationBatch(np)
    batch.refuse_if_any(np.zeros(4, bool), explode)
    batch.flush()


def test_a_batch_longer_than_one_scan_group_still_names_every_field():
    """The batch chunks the scan; the flag block and the order do not chunk."""
    from gpuwm.core.ruc_validation import VALIDATION_SCAN_GROUP

    namespace = counter.namespace()
    batch = RucValidationBatch(namespace)
    width = VALIDATION_SCAN_GROUP * 2 + 3
    for index in range(width):
        array = np.zeros(8, np.float32)
        if index == width - 1:
            array[3] = np.float32("nan")
        batch.finite(counter.counted(array), f"field{index}")
    with pytest.raises(ValueError, match=rf"^field{width - 1} must be finite$"):
        batch.flush()


# ---------------------------------------------------------------------------
# the reads


def _counted_driver_reads(columns: int = 64):
    values, keywords = _driver_case(columns)
    namespace = counter.namespace()
    counted = {name: counter.counted(value) for name, value in values.items()}
    keywords = dict(keywords, arrays=namespace,
                    ivgtyp=counter.counted(keywords["ivgtyp"]),
                    isltyp=counter.counted(keywords["isltyp"]))
    counter.reset()
    ruc_land_surface_step(counted, **keywords)
    return counter.total()


def test_one_land_surface_call_no_longer_reads_the_device_per_field():
    """Admission used to cost one read per field; now it costs one per batch.

    On this grid -- 64 warm land columns, the shape of the production case --
    the same call made 326 reads before this lane: 122 of them in
    ``_horizontal_float_field`` alone, 157 scalar reads over the whole call,
    and 169 more in boolean-mask gathers and scatters.
    """
    reads = _counted_driver_reads()
    flushes = sum(n for (_, name), n in counter.READS.items()
                  if name == "_host_list")
    per_field = sum(n for (_, name), n in counter.READS.items()
                    if name == "_horizontal_float_field")
    # One read per BATCH, whatever the batch holds: five batches decide the
    # driver's 65 fields, sfctmp's 63, and the three small helpers'.
    assert flushes <= 8, dict(counter.READS)
    # What is left at the per-field helper is the host leaves' own admission,
    # which a forecast does not run: its leaves are the CUDA ones.
    assert per_field <= 6, dict(counter.READS)
    assert reads <= 30, dict(counter.READS)


def test_the_dispatch_decides_all_eleven_arms_with_one_read():
    """The arm census, and the conversion of the two arms that have columns.

    Eleven arms, eight distinct masks, one ``np.any`` stack read for the
    census; a warm grid then converts the two arms that run and skips the
    nine that do not.
    """
    _counted_driver_reads()
    census = counter.READS[("ruc.py:%d" % _sfctmp_line(),
                            "ruc_surface_temperature_step")]
    converted = sum(n for (site, name), n in counter.READS.items()
                    if name == "_selected")
    assert census == 2, dict(counter.READS)     # the census and the outputs
    assert converted <= 5, dict(counter.READS)


def _sfctmp_line() -> int:
    return ruc_surface_temperature_step.__code__.co_firstlineno


def test_the_counter_still_counts_what_it_is_there_to_count():
    """The falsification: a counter that counted nothing would pass above."""
    counter.reset()
    array = counter.counted(np.arange(8, dtype=np.float32))
    gathered = array[array > np.float32(3.0)]
    array[array > np.float32(3.0)] = np.float32(0.0)
    assert gathered.size == 4
    assert counter.READS[("TOTAL", "mask_gather")] == 1
    assert counter.READS[("TOTAL", "mask_scatter")] == 1
    assert bool(counter.counted(np.array(True)))
    assert counter.READS[("TOTAL", "scalar")] == 1


def test_the_batch_reads_once_however_many_fields_it_holds():
    namespace = counter.namespace()
    for width in (1, 4, 40, 120):
        batch = RucValidationBatch(namespace)
        for index in range(width):
            batch.finite(counter.counted(np.zeros(16, np.float32)),
                         f"field{index}")
        counter.reset()
        batch.flush()
        assert counter.total() == 1, (width, dict(counter.READS))
