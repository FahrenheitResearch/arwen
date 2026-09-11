"""Exact initial values for prognostic species absent from analyzed input.

These are state-allocation contracts, independent of the source name and of
radiation, turbulence, surface or cumulus presets. WRF v4.6.1 Registry.EM_COMMON
lines 3025/3031 bind Milbrandt/WDM6 qnc to QNCLOUD (line 542); NSSL
uses the distinct qndrop/QNDROP declaration at lines 521-522.
An analyzed mass inventory is validated separately by initialize_real and its correspondence receipt.
"""
from __future__ import annotations

from types import SimpleNamespace
import struct


def source_absent_microphysics(cfg):
    """Return (WRF field names, native FP32 initial values) for active extras."""
    mp = int(cfg.mp_physics)
    if mp in (0, 1, 6):
        return (), {}
    if mp == 8:
        return ("QNICE", "QNRAIN"), {"ni": 0.0, "nr": 0.0}
    if mp == 9:
        return (("QHAIL", "QNCLOUD", "QNRAIN", "QNICE", "QNSNOW",
                 "QNGRAUPEL", "QNHAIL"),
                dict.fromkeys(("qh", "nc", "nr", "ni", "ns", "ng", "nh"), 0.0))
    if mp == 10:
        return (("QNRAIN", "QNICE", "QNSNOW", "QNGRAUPEL"),
                dict.fromkeys(("nc", "nr", "ni", "ns", "ng"), 0.0))
    if mp == 16:
        return (("QNCLOUD", "QNRAIN", "QNCCN"),
                {"nc": 0.0, "nr": 0.0, "nn": float(cfg.wdm6_ccn_conc)})
    if mp == 18:
        from gpuwm.core.microphysics_transition import NSSL2_BACKGROUND_CCN_PER_KG
        values = dict.fromkeys(("qh", "qndrop", "qnr", "qni", "qns", "qng",
                                "qnh", "qnn", "qvolg", "qvolh"), 0.0)
        values["qnn"] = NSSL2_BACKGROUND_CCN_PER_KG
        return (("QHAIL", "QNDROP", "QNRAIN", "QNICE", "QNSNOW", "QNGRAUPEL",
                 "QNHAIL", "QNCCN", "QVGRAUPEL", "QVHAIL"), values)
    if mp == 50:
        return (("QNICE", "QNRAIN", "QIR", "QIB"),
                dict.fromkeys(("ni", "nr", "qir", "qib"), 0.0))
    if mp == 28:
        # Aerosol-aware Thompson (Registry.EM_COMMON:3036).  The three
        # number species a cold start owns are nc, nr and ni, and every
        # one of them starts at exact zero -- the value gpuwm/core/state.py
        # allocates and the value real.exe leaves when the analyzed input
        # carries no number moment.
        #
        # nwfa/nifa (and the two 2-D emission fields) are deliberately NOT
        # here.  Their initial condition is not "absent from the analysed
        # input, therefore zero": it is DECIDED by mp28_aerosol_source --
        # the WIF climatology ingest fills them
        # (gpuwm/ingest/real.py, real.exe's aer_init_opt=1 path) and the
        # synthetic source leaves the exact zeros thompson_init's MAXVAL
        # test reads to install its own profile
        # (module_mp_thompson.F:493/:531).  That authority publishes its
        # own receipt and refuses a nonzero field it did not write, so a
        # second contract over the same fields here would be a second
        # spelling -- and it would be WRONG for the climatology run, whose
        # nwfa is nonzero by the time any consumer of this contract looks.
        #
        # This function used to RAISE for mp=28, on the premise that no
        # source carried the aerosol boundary species.  Audit R-044
        # retired that premise (the WIF ingest is the source, on every
        # route), and the raise outlived it: it was reached from
        # tools/prepare_hrrr_wrf.py after the wizard had already reported
        # PASS -- a refusal after step 0, which the gate law forbids.  The
        # precondition that survives is the DATASET one, measured by
        # gpuwm.config.mp28_aerosol_lateral_forcing_precondition: reported
        # at plan review and raised at the run door by
        # gpuwm.config.validate_run_preparation.
        return (("QNCLOUD", "QNRAIN", "QNICE"),
                dict.fromkeys(("nc", "nr", "ni"), 0.0))
    raise ValueError(f"no native prognostic-species initialization for mp_physics={mp}")


def cold_start_contract(selection):
    """Wire-level expected FP32 values and bits, computed from active settings."""
    cfg = SimpleNamespace(**selection) if isinstance(selection, dict) else selection
    fields, values = source_absent_microphysics(cfg)
    expected = {}
    for name, value in values.items():
        packed = struct.pack("<f", value)
        expected[name] = (struct.unpack("<f", packed)[0], struct.unpack("<I", packed)[0])
    return fields, expected
