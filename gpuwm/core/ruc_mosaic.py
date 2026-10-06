"""WRF v4.6.1 RUC mosaic parameter mixture and irrigation.

Transcribed from ``phys/module_sf_ruclsm.F:soilvegin`` and ``lsmruc``.
WRF is public domain; see ``licenses/LICENSE-WRF-public-domain.txt`` and
the repository NOTICE. The two namelist defaults remain zero.
"""

from __future__ import annotations

import numpy as np

from gpuwm.core.noahmp_libm import expf, logf


def mosaic_option(value, name):
    if type(value) is not int or value not in (0, 1):
        raise ValueError(
            f"RUC {name}={value!r} must be the integer 0 (dominant "
            "category) or 1 (fractional mosaic): the drivers mix by "
            "category fraction for any nonzero value, so another value "
            "would replace the dominant-category surface parameters under "
            "a selector WRF does not define")


def mosaic_fractions(value, shape, name, maximum, *, arrays=np, validate_values=True):
    """Require source fractions; inventing one-hot fractions loses the mosaic."""
    if value is None:
        raise ValueError(f"RUC mosaic requires {name} category fractions; "
                         "a dominant category cannot reconstruct mixed cover")
    result = arrays.asarray(value, dtype=arrays.float32)
    if (result.ndim != len(shape) + 1 or result.shape[1:] != tuple(shape)
            or not 1 <= result.shape[0] <= maximum):
        raise ValueError(f"RUC {name} shape {result.shape}; expected "
                         f"(1..{maximum}, {', '.join(map(str, shape))}): "
                         "the mixture weights each category's table row by "
                         "its cell fraction, so another shape would weight "
                         "the wrong cells or categories that have no row")
    if validate_values and bool(arrays.any(~arrays.isfinite(result) | (result < 0) | (result > 1))):
        raise ValueError(f"RUC {name} fractions must be finite and within 0..1: "
                         "they are the weights of the surface parameter "
                         "mixture, and a weight outside that range gives "
                         "roughness, emissivity and soil parameters no "
                         "category mixture can have")
    if validate_values and name == "landusef" and bool(arrays.any(arrays.sum(result, axis=0) <= 0)):
        raise ValueError("RUC landusef has zero area; WRF's mosaic roughness "
                         "and surface parameter normalization are undefined")
    return arrays.ascontiguousarray(result)


def surface_mixture(base, *, isltyp, shdmin, shdmax, vegfrac, znt, lai,
                    vegetation, soil, mosaic_lu, mosaic_soil, landusef,
                    soilctop, iswater, rdlai2d):
    """Replace the selected dominant outputs with WRF's ordered mixtures."""
    f = np.float32
    shape = base.pc.shape
    if mosaic_lu:
        fractions = mosaic_fractions(landusef, shape, "landusef", len(vegetation.rows))
        area = np.zeros(shape, dtype=f)
        emiss = np.zeros(shape, dtype=f)
        rough = np.zeros(shape, dtype=f)
        leaf = np.zeros(shape, dtype=f)
        pc = np.zeros(shape, dtype=f)
        span = (shdmax - shdmin).astype(f)
        ratio = ((vegfrac - shdmin).astype(f) / np.maximum(f(1), span)).astype(f)
        factor = np.where(span < f(1), f(1),
                          (f(1) - np.maximum(f(0), np.minimum(f(1), ratio))).astype(f)).astype(f)
        for k in range(fractions.shape[0]):
            row = vegetation.rows[k]
            cap = {1: .2, 2: .5, 3: .45, 4: .75, 5: .86, 7: .5}.get(row.ifor, 0)
            delta = min(f(cap), f(f(.8) * f(row.lai)))
            today_lai = np.full(shape, f(row.lai))
            today_znt = np.full(shape, f(row.z0))
            if k + 1 == iswater:
                today_znt = np.asarray(znt, dtype=f)
            else:
                today_lai = (today_lai - (delta * factor).astype(f)).astype(f)
                if row.ifor == 7:
                    today_znt = (today_znt - (f(.125) * factor).astype(f)).astype(f)
            # Scalar glibc transcriptions keep the Fortran LOG/EXP words.
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio_z = (f(5) / today_znt).astype(f)
                logarithm = np.fromiter((logf(x) for x in ratio_z.flat), f).reshape(shape)
                term = (fractions[k] / (logarithm * logarithm).astype(f)).astype(f)
            area = (area + fractions[k]).astype(f)
            emiss = (emiss + (f(row.lemi) * fractions[k]).astype(f)).astype(f)
            rough = (rough + term).astype(f)
            leaf = (leaf + (today_lai * fractions[k]).astype(f)).astype(f)
            pc = (pc + (f(row.pc) * fractions[k]).astype(f)).astype(f)
        area = np.minimum(area, f(1))
        exponent = np.sqrt((f(1) / rough).astype(f)).astype(f)
        denominator = np.fromiter((expf(x) for x in exponent.flat), f).reshape(shape)
        base.emiss[...] = (emiss / area).astype(f)
        base.pc[...] = (pc / area).astype(f)
        base.znt[...] = (f(5) / denominator).astype(f)
        if not rdlai2d:
            base.lai[...] = (leaf / area).astype(f)
    if mosaic_soil:
        fractions = mosaic_fractions(soilctop, shape, "soilctop", len(soil.rows))
        names = ("rhocs", "bclh", "dqm", "ksat", "psis", "qmin", "ref", "wilt", "qwrtz")
        sums = {name: np.zeros(shape, dtype=f) for name in names}
        area = np.zeros(shape, dtype=f)
        table = np.asarray([row.values for row in soil.rows], dtype=f)
        def properties(row):
            return (f(row[2] * f(1.e6)), row[0], f(row[3] - row[1]),
                    row[6], -row[5], row[1], row[4], row[8], row[9])
        for k in range(fractions.shape[0]):
            if k == 13:
                continue
            area = (area + fractions[k]).astype(f)
            for name, value in zip(names, properties(table[k])):
                sums[name] = (sums[name] + (value * fractions[k]).astype(f)).astype(f)
        area = np.minimum(area, f(1))
        fallback = properties(np.moveaxis(table[np.asarray(isltyp) - 1], -1, 0))
        denominator = np.where(area > 0, area, f(1))
        for name, value in zip(names, fallback):
            getattr(base, name)[...] = np.where(area > 0, (sums[name] / denominator).astype(f), value)
    return base


#: The two WRF forms of LSMRUC's post-SFCTMP irrigation, by WRF lineage.
#:
#: ``wrf_45``: WRF v4.0 to v4.5 ``phys/module_sf_ruclsm.F:970-999``, the
#: form the operational RAP/HRRR branch of WRF also carries.  A HARD FLOOR,
#: applied whatever ``mosaic_lu`` says: where the cropland fraction is
#: positive and the leaf area index exceeds 1.1, each root layer is held at
#: or above ``(1.1*WLTSMC - DRYSMC) * lufrac(crop)``; otherwise, where the
#: dominant category is the crop/natural mosaic and the leaf area index
#: exceeds 0.7, at or above ``(1.2*WLTSMC - DRYSMC) * lufrac(natural) * 0.4``.
#: The floor is scaled by the category FRACTION, so it is idempotent and
#: never lifts a cell past the share of it that is cropland.  WRF fills the
#: fractions from LANDUSEF whatever ``mosaic_lu`` says; a gpuwm run that
#: carries no LANDUSEF (``mosaic_lu = 0``) gives the dominant category the
#: whole cell, WRF's arithmetic on a one-hot LANDUSEF (the crop arm is then
#: the dominant-cropland form these lines carry commented out at :971).
#:
#: ``wrf_461``: WRF v4.6.1 ``:985-1009``, under ``mosaic_lu == 1`` only.  A
#: PER-STEP RELAXATION: where any crop or crop/natural fraction exists and
#: the greenness factor exceeds 0.75, each root layer below
#: ``newsm = cropsm*cropfr + (1-cropfr)*soilm1d`` is set to ``newsm``.  Since
#: ``soilm1d < newsm`` exactly when ``soilm1d < cropsm``, every step moves the
#: layer a fraction ``cropfr`` of the way to the FULL ``1.1*WLTSMC``, so a
#: cell with any sliver of cropland converges to 1.1 times its wilting point
#: within the first forecast hour where the rule fires.  The rule is not
#: what wets a dry top soil level everywhere in the first hour: that is
#: SOILPROP's v4.6.1 diffusivity (``ruc_soilprop``,
#: :mod:`gpuwm.core.ruc_tier`).  MEASURED on a 3 km October afternoon cut
#: over Kansas and Colorado (106,671 land cells, table leaf area, wrf_45
#: SOILPROP): the two irrigation forms differ by 0.003 m3/m3 in the
#: top-level mean and 0.13 K in 2 m dewpoint after one hour.  Kept
#: selected by default for generic runs. The operational namelist
#: importer and configuration recipe explicitly select the wrf_45 form.
IRRIGATION_FORMS = ("wrf_45", "wrf_461")
IRRIGATION_DEFAULT = "wrf_461"


def irrigation_form(value):
    """The kernel selector (0 or 1) for a named irrigation form, or refuse."""
    if type(value) is not str or value not in IRRIGATION_FORMS:
        raise ValueError(
            f"RUC ruc_irrigation={value!r} must be one of {IRRIGATION_FORMS}: "
            "'wrf_45' holds root layers at a crop-fraction-scaled floor "
            "(WRF v4.0-4.5, the operational RAP/HRRR form) and 'wrf_461' "
            "relaxes them to the full 1.1 x wilting point every step (WRF "
            "v4.6.1); an unknown name cannot select either without "
            "guessing which soil water the forecast adds")
    return IRRIGATION_FORMS.index(value)


def irrigate(soilm1d, *, landusef, vegfrac, shdmin, shdmax, wilt, qmin,
             nroot, crop, natural, active, form=IRRIGATION_DEFAULT,
             lai=None, ivgtyp=None, arrays=np):
    """WRF LSMRUC's post-SFCTMP irrigation, including its water addition.

    Only soil moisture changes. WRF does not update liquid water or fluxes
    here and does not record the added water in a budget accumulator.
    ``form`` names which WRF lineage's rule applies (:data:`IRRIGATION_FORMS`);
    ``wrf_45`` reads ``lai`` (the leaf area index the scheme is running
    with, table or 2-D) and ``ivgtyp`` (the dominant category) for its gates.
    """
    f = arrays.float32
    selector = irrigation_form(form)
    if landusef is None:
        # A run without LANDUSEF (mosaic_lu = 0): the wrf_45 floor gives the
        # dominant category the whole cell, WRF's arithmetic on a one-hot
        # LANDUSEF.  wrf_461 never reaches here without fractions.
        if selector == 1 or ivgtyp is None:
            raise ValueError("RUC irrigation without LANDUSEF needs the wrf_45 "
                             "form and the dominant category to stand in for "
                             "the crop and crop/natural fractions")
        category = arrays.asarray(ivgtyp)
        croparea = (category == crop).astype(f)
        naturalarea = (category == natural).astype(f)
    else:
        if landusef.shape[0] < max(crop, natural):
            raise ValueError("RUC landusef omits the table's crop/natural categories; "
                             "LSMRUC irrigation cannot index the source fractions")
        croparea, naturalarea = landusef[crop - 1], landusef[natural - 1]
    if selector == 1:
        factor = arrays.maximum(f(0), arrays.minimum(f(1),
            ((vegfrac - shdmin).astype(f) / arrays.maximum(f(1), (shdmax - shdmin).astype(f))).astype(f)))
        enabled = active & ((croparea > 0) | (naturalarea > 0)) & (factor > f(.75))
        cropsm = ((f(1.1) * wilt).astype(f) - qmin).astype(f)
        cropfr = arrays.minimum(f(1), (croparea + (f(.4) * naturalarea).astype(f)).astype(f))
        for k in range(soilm1d.shape[0]):
            newsm = ((cropsm * cropfr).astype(f) +
                     ((f(1) - cropfr).astype(f) * soilm1d[k]).astype(f)).astype(f)
            soilm1d[k] = arrays.where(enabled & (k < nroot) & (soilm1d[k] < newsm), newsm, soilm1d[k])
        return
    if lai is None or ivgtyp is None:
        raise ValueError("RUC irrigation 'wrf_45' gates on the leaf area index "
                         "and the dominant category; both must be supplied")
    leaf = arrays.asarray(lai, dtype=f)
    category = arrays.asarray(ivgtyp)
    # WRF v4.5.2 phys/module_sf_ruclsm.F:970-999.  ``cropsm*lufrac(crop)`` and
    # ``cropsm * lufrac(natural)*0.4`` in Fortran's left-to-right order.
    crop_arm = active & (croparea > 0) & (leaf > f(1.1))
    natural_arm = active & ~crop_arm & (category == natural) & (leaf > f(0.7))
    crop_floor = (((f(1.1) * wilt).astype(f) - qmin).astype(f) * croparea).astype(f)
    natural_floor = ((((f(1.2) * wilt).astype(f) - qmin).astype(f) * naturalarea).astype(f)
                     * f(0.4)).astype(f)
    floor = arrays.where(crop_arm, crop_floor,
                         arrays.where(natural_arm, natural_floor, f(-1.0))).astype(f)
    enabled = crop_arm | natural_arm
    for k in range(soilm1d.shape[0]):
        soilm1d[k] = arrays.where(enabled & (k < nroot) & (soilm1d[k] < floor), floor, soilm1d[k])
