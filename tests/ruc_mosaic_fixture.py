"""Inputs recorded by the unmodified WRF RUC mosaic column oracle."""

from pathlib import Path
import csv
import numpy as np

from gpuwm.core.ruc import (RUC_DRIVER_COLUMN_FORCING, RUC_DRIVER_COLUMN_STATE,
                            RUC_DRIVER_PROFILE_STATE)
from tools.ruc_wrf461_oracle.validate_lsmruc_oracle import (
    _call_arguments, _entry, _load, _result, _single_count_sfcevp)

ORACLE = Path(__file__).parents[1] / "gpuwm/data/ruc/oracle"


def surface_cases():
    with (ORACLE / "mosaic_surface.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        n = int(row["case"])
        lf = np.zeros((21, 1), np.float32)
        sf = np.zeros((19, 1), np.float32)
        lf[[0, 11, 13], 0] = [.15, .55, .3]
        sf[[5, 7, 13], 0] = [.35, .5, .15]
        if n == 5:
            lf *= np.float32(.5)
        if n == 6:
            lf *= np.float32(1.1)
        if n in (7, 9):
            sf.fill(0)
            sf[13] = 1
        if n == 8:
            sf *= np.float32(1.1)
        if n == 9:
            lf.fill(0)
            lf[16] = 1
        if n == 12:
            lf.fill(0)
            lf[11] = 1
            sf.fill(0)
            sf[3] = 1
        inputs = [np.array([int(row[key])], np.int32) for key in ("isltyp", "ivgtyp")]
        inputs += [np.array([float(row[key])], np.float32) for key in (
            "shdmin", "shdmax", "vegfrac", "znt_before", "lai_before")]
        keywords = dict(mosaic_lu=int(row["mosaic_lu"]), mosaic_soil=int(row["mosaic_soil"]),
                        landusef=lf, soilctop=sf, rdlai2d=bool(int(row["rdlai2d"])))
        yield row, inputs, keywords


def driver_fractions(run):
    nl, crop, natural = (21, 12, 10) if run == 1 else (28, 3, 5)
    lu = np.zeros((nl, 12), np.float32)
    so = np.zeros((19, 12), np.float32)
    lu[crop - 1] = .55
    lu[natural - 1] = .3
    lu[0] = .15
    so[5] = .35
    so[7] = .5
    so[13] = .15
    lu[:, 3] = 0
    lu[0, 3] = 1
    lu[:, 4] *= np.float32(.5)
    lu[:, 5] *= np.float32(1.1)
    so[:, 6] = 0
    so[13, 6] = 1
    so[:, 7] *= np.float32(1.1)
    return lu, so


def driver_calls():
    groups, field = _load(str(ORACLE / "mosaic_driver.csv"))
    for begin in range(0, len(groups), 12):
        take = slice(begin, begin + 12)
        values = {name: _entry(field, name)[:, take].copy() for name in RUC_DRIVER_PROFILE_STATE}
        values.update({name: _entry(field, name)[0, take].copy() for name in RUC_DRIVER_COLUMN_STATE})
        values.update({name: field[name][0, take].copy() for name in RUC_DRIVER_COLUMN_FORCING})
        keywords = _call_arguments(field, begin)
        keywords.update(zs=field["zs"][:, begin].copy(),
                        ivgtyp=field["ivgtyp"][0, take].astype(np.int32),
                        isltyp=field["isltyp"][0, take].astype(np.int32),
                        ilnb=1, ilnb_chain=False,
                        # The oracle is the byte-unmodified WRF v4.6.1 module,
                        # so its irrigation and SOILPROP are that lineage's
                        # by name.
                        irrigation="wrf_461", soilprop="wrf_461")
        keywords["landusef"], keywords["soilctop"] = driver_fractions(groups[begin][0])
        expected = {name: _result(field, name)[:, take].copy() for name in RUC_DRIVER_PROFILE_STATE}
        expected.update({name: _result(field, name)[0, take].copy() for name in RUC_DRIVER_COLUMN_STATE})
        # WRF counts SFCEVP twice per land step (:1095, :1116); WOOF once.
        expected["sfcevp"] = _single_count_sfcevp(field)[take].copy()
        yield groups[begin][:2], values, keywords, expected
