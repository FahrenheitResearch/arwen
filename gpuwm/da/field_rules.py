"""EXPERIMENTAL: which analysed fields each KIND of observation may move.

Why.  The filter lets every observation update every analysed field at
every height its localization reaches.  Two kinds do damage that way:

* **Cloud water path** is a retrieval of condensate.  Through the
  ensemble's sampled covariances it also moved vapour and heat: on the
  research branch's storm-scale cycle it carried about three quarters of
  the inflow drying in one arm (audit S9; project notes 2026-09-24).  Here
  it updates condensate and wind and never ``qv`` or ``thp``.
* **Surface reports** (2 m temperature, 10 m wind) sit at the lowest model
  level.  Through sampled covariances they moved storm hydrometeors aloft.
  Here they update wind, heat and vapour and never a hydrometeor moment.

How.  Each rule is a row of :data:`KIND_FIELD_RULES` (a table, not a code
path per stream).  A batch matching a row is withheld from that row's
fields in the columns it can reach (its observed columns dilated by its
horizontal localization cutoff), through the same exact local re-solve the
radial-velocity dispersion gate uses
(:func:`gpuwm.da.velocity_dispersion.withhold`): in those columns the
withheld fields take the solve WITHOUT that batch, every other field and
column keeps the joint solve.  Gates are grouped by identical field sets
and each group is one ``withhold`` call, so a zone never withholds a batch
from a field its own rule does not name.

On by default (``RadarAssimilationConfig.kind_field_rules``); the receipt
records every gate and every re-solve.

Radar rules (DA lane 3, ``RadarAssimilationConfig.field_rules``)
---------------------------------------------------------------
DA design 2026-10-05, D1 field lists (lane 3).  The breakages this
prevents, named and measured on the 2026-10-01 19Z CONUS 9 km analysis
(32 members, every observation allowed to update all 14 analysed fields,
aerosols included):

* Radar reflectivity wrote vapour and heat over wide areas.  Member 0 gained
  three times more vapour than condensate; 36 % of it went into columns whose
  strongest echo was 0 to 15 dBZ, and theta increments reached 16.8 K.  The
  echo area doubled within the hour.  No operational centre lets 3D
  reflectivity drive vapour and heat through ensemble covariances at 9 km.
* Number concentrations and aerosols were analysed from radar (nc, nr, ni,
  nwfa, nifa all moved), distorting the drop-size distribution the
  reflectivity operator and the microphysics then read.  HRRRDAS found that
  analysing rain number did not help.

Rules (``rules="design"``):

* winds (u, v, w) and the thermodynamic fields (thp, qv) take the solve
  WITHOUT the reflectivity and clear-air batches (radial velocity and
  surface/conventional observations still move them; the velocity
  dispersion gate still applies);
* reflectivity adds to thp and qv only ``z_thermo_weight`` times its own
  contribution (the joint solve minus the no-radar solve), only in columns
  with observed precipitation echo, with ``|dqv|`` capped;
* hydrometeor masses take the solve on the reflectivity and clear-air
  batches alone (``z_hydrometeors=False`` analyses none of them, the
  KENDA-style arm);
* number concentrations and aerosols get no filter increment; rain and ice
  number are rediagnosed from the analysed mass (:func:`rediagnose_numbers`).

``rules="joint"`` is the legacy behaviour: every batch updates every field.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

#: Receipt schema.
FIELD_RULES_SCHEMA = "gpuwm-da.kind-field-rules.v1"

#: Hydrometeor moments: every physically non-negative field except vapour.
def _hydrometeor_fields() -> tuple[str, ...]:
    from gpuwm.da.positivity import NON_NEGATIVE_FIELDS

    return tuple(name for name in NON_NEGATIVE_FIELDS if name != "qv")


#: ``kind -> (how a batch name is recognised, fields it may NOT move)``.
#: Recognisers read the batch name's leading token (``<token>:<rest>``, a
#: window slot adding ``@label``).
KIND_FIELD_RULES = {
    "cloud-water-path": {
        "tokens": ("cwp",),
        "withheld": ("qv", "thp"),
        "reason": ("a condensate retrieval; through sampled covariances it "
                   "removed inflow vapour (audit S9)"),
    },
    "surface": {
        "tokens": ("temperature_2m", "wind_speed_10m", "dewpoint_2m",
                   "pressure_surface", "u10", "v10", "t2"),
        "withheld": "hydrometeors",
        "reason": ("a lowest-level report; through sampled covariances it "
                   "moved storm hydrometeors aloft (audit S9)"),
    },
}


def batch_token(name: str) -> str:
    """The leading token of a batch name: ``cwp``, ``vr``, a quantity."""

    return str(name).split("@", 1)[0].split(":", 1)[0]


def rule_for_batch(name: str):
    """``(kind, row)`` for a batch name, or ``(None, None)``."""

    token = batch_token(name)
    for kind, row in KIND_FIELD_RULES.items():
        if token in row["tokens"]:
            return kind, row
    return None, None


def withheld_fields(row) -> tuple[str, ...]:
    """Every field a rule row withholds, whether analysed or not.

    The whole row, not its intersection with one analysis, so the field set
    of a row is the same in every analysis and groups with the dispersion
    gate's (``thp``, ``qv``) exactly; ``withhold`` itself solves only the
    analysed ones.
    """

    return (_hydrometeor_fields() if row["withheld"] == "hydrometeors"
            else tuple(row["withheld"]))


def kind_gates(batches: Iterable, analysis_fields: Sequence[str], *,
               shape, dx_m: float, dy_m: float, localization):
    """``(gates, receipt)``: one :class:`DispersionGate` per batch a rule
    names that observes something and shares a field with the analysis."""

    from gpuwm.da.velocity_dispersion import (DispersionGate, _dilate,
                                              _host, _stencil)

    ny, nx = int(shape[-2]), int(shape[-1])
    gates, rows = [], []
    for batch in batches:
        kind, row = rule_for_batch(batch.name)
        if row is None:
            continue
        fields = withheld_fields(row)
        if not set(fields) & set(analysis_fields):
            continue
        mask = np.asarray(_host(batch.mask), dtype=bool)
        columns = np.zeros((ny, nx), dtype=bool)
        window = getattr(batch, "window", None)
        observed = mask.any(axis=0) if mask.ndim == 3 else mask
        if window is None:
            columns |= observed
        else:
            j0, i0 = int(window[0]), int(window[2])
            columns[j0:j0 + observed.shape[0],
                     i0:i0 + observed.shape[1]] |= observed
        if not columns.any():
            continue
        spec = getattr(batch, "localization", None) or localization
        dj, di, _ = _stencil(dx_m, dy_m, float(spec.horizontal_m))
        reach = _dilate(columns, dj, di)
        gates.append(DispersionGate(batch=batch.name, fields=fields,
                                    columns=reach))
        rows.append({"batch": batch.name, "kind": kind,
                     "fields": [name for name in analysis_fields
                                if name in fields],
                     "columns": int(np.count_nonzero(reach)),
                     "reason": row["reason"]})
    return gates, {"schema": FIELD_RULES_SCHEMA, "gates": rows}


def group_by_fields(gates) -> list[list]:
    """Gates grouped by identical (order-free) field sets, first-seen order.

    One ``withhold`` call per group keeps every zone exact: within a group
    every gate withholds the same fields.  Two groups must not share a
    field, or a later group's re-solve would overwrite the earlier one's
    withheld field with a solve that still holds that batch.
    """

    groups: dict[frozenset, list] = {}
    for gate in gates:
        groups.setdefault(frozenset(gate.fields), []).append(gate)
    keys = list(groups)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            shared = keys[a] & keys[b]
            if shared:
                raise ValueError(
                    f"two gate groups both withhold {sorted(shared)} "
                    f"({sorted(keys[a])} and {sorted(keys[b])}); their "
                    "sequential re-solves would not be exact. Give the "
                    "rows identical or disjoint field sets")
    return [groups[key] for key in keys]


# ---------------------------------------------------------------------------
# Radar rules and number rediagnosis (lane 3)
# ---------------------------------------------------------------------------

SCHEMA = "gpuwm-da.field-rules.v1"
RULES = ("design", "joint")
DEFAULT_RULES = "design"
DEFAULT_Z_THERMO_WEIGHT = 0.1
DEFAULT_Z_HYDROMETEORS = True
#: |qv| increment cap on the radar part, kg/kg (design D1 qv increment cap).
#: Off by default: the in-pass keep weight already bounds the radar's share,
#: and a cap needs the radar-free solve separately (one local re-solve over
#: the echo columns), which the in-pass route avoids.
DEFAULT_Z_QV_CAP = None
NUMBER_MODES = ("scheme", "preserve-size", "off")
DEFAULT_NUMBER_MODE = "scheme"

RADAR_BATCHES = ("z", "z0")
#: Batch tokens that observe condensate and so may move hydrometeor mass
#: under the design rules: reflectivity, clear air and cloud water path.
CONDENSATE_TOKENS = ("z", "z0", "cwp")
WINDS = ("u", "v", "w")
THERMO = ("thp", "qv")
MASSES = ("qc", "qr", "qi", "qs", "qg", "qh")
NUMBERS = ("nc", "nr", "ni", "ns", "ng", "nh")
AEROSOLS = ("nwfa", "nifa")
#: The Thompson thresholds below which a species is absent
#: (module_mp_thompson.F R1 = 1e-12 kg/kg mass).
MASS_ABSENT = 1.0e-12


class FieldRuleError(ValueError):
    """A field-rule setting this module cannot honour."""


def check_settings(rules, z_thermo_weight, qv_cap, number_mode):
    if rules not in RULES:
        raise FieldRuleError(f"field_rules must be one of {RULES}, got {rules!r}")
    w = float(z_thermo_weight)
    if not np.isfinite(w) or not 0.0 <= w <= 1.0:
        raise FieldRuleError(
            f"z_thermo_weight must be in [0, 1], got {z_thermo_weight!r}")
    if qv_cap is not None and not (np.isfinite(float(qv_cap))
                                   and float(qv_cap) > 0.0):
        raise FieldRuleError(f"z_qv_cap must be positive or None, got {qv_cap!r}")
    if number_mode not in NUMBER_MODES:
        raise FieldRuleError(
            f"number_rediagnosis must be one of {NUMBER_MODES}, got {number_mode!r}")


def echo_columns(batches, *, echo_floor: float, dilate: int) -> np.ndarray | None:
    """``(ny, nx)`` columns holding observed precipitation echo (any level
    of the reflectivity batch at or above ``echo_floor``), dilated by
    ``dilate`` cells (the horizontal thinning stride, so a thinned-out
    neighbour of an observed echo cell still counts)."""
    for batch in batches:
        if batch.name != "z":
            continue
        mask = np.asarray(batch.mask, dtype=bool)
        values = np.asarray(batch.values, dtype=np.float64)
        cols = (mask & (values >= echo_floor)).any(axis=0)
        out = cols.copy()
        for _ in range(max(0, int(dilate))):
            grown = out.copy()
            grown[1:, :] |= out[:-1, :]
            grown[:-1, :] |= out[1:, :]
            grown[:, 1:] |= out[:, :-1]
            grown[:, :-1] |= out[:, 1:]
            out = grown
        return out
    return None


def _host(array):
    if hasattr(array, "get"):
        return array.get()
    return np.asarray(array)


def combine(increments, *, no_radar, radar_only, analysis_fields,
            z_thermo_weight, z_hydrometeors, qv_cap, columns):
    """``(increments, receipt)``: the design rules applied to the joint
    increments, given the no-radar solve ``no_radar`` (winds and thermo) and
    the radar-only solve ``radar_only`` (masses; ``None`` when
    ``z_hydrometeors`` is off)."""
    out = dict(increments)
    receipt = {"schema": SCHEMA, "rules": "design",
               "z_thermo_weight": float(z_thermo_weight),
               "z_hydrometeors": bool(z_hydrometeors),
               "z_qv_cap": None if qv_cap is None else float(qv_cap),
               "echo_columns": None if columns is None else int(columns.sum()),
               "fields": {}}
    weight = None
    if columns is not None:
        weight = np.where(columns, float(z_thermo_weight), 0.0)[None, None]
    for name in analysis_fields:
        joint = _host(increments[name])
        if name in WINDS:
            out[name] = _host(no_radar[name]).astype(joint.dtype, copy=False)
            rule = "no reflectivity"
        elif name in THERMO:
            base = _host(no_radar[name]).astype(np.float64)
            radar = joint.astype(np.float64) - base
            radar = radar * (weight if weight is not None else 0.0)
            if name == "qv" and qv_cap is not None:
                radar = np.clip(radar, -float(qv_cap), float(qv_cap))
            out[name] = (base + radar).astype(joint.dtype, copy=False)
            rule = f"no reflectivity + {float(z_thermo_weight)} x reflectivity in echo"
        elif name in MASSES:
            if z_hydrometeors and radar_only is not None:
                out[name] = _host(radar_only[name]).astype(joint.dtype, copy=False)
                rule = "reflectivity and clear air only"
            else:
                out[name] = np.zeros_like(joint)
                rule = "not analysed"
        elif name in NUMBERS or name in AEROSOLS:
            out[name] = np.zeros_like(joint)
            rule = "not analysed (numbers rediagnosed)" if name in NUMBERS \
                else "not analysed"
        else:
            rule = "joint"
        receipt["fields"][name] = rule
    return out, receipt


def _temperature(pressure, inverse_density, vapour):
    from gpuwm.core import constants as c

    return (np.asarray(pressure, np.float64) * np.asarray(inverse_density, np.float64)
            / (c.RD * (1.0 + c.RVOVRD * np.asarray(vapour, np.float64))))


PAIRS = (("qc", "nc"), ("qr", "nr"), ("qi", "ni"), ("qs", "ns"),
         ("qg", "ng"), ("qh", "nh"))


def rediagnose_numbers(prior, increments, states, indices, *, mode: str,
                       mp_physics) -> tuple:
    """``(increments, receipt)`` with every analysed number moment following
    its analysed mass.

    ``mode="scheme"``: for Thompson (8, 28) rain and ice, wherever the mass
    increment is nonzero, the number is the scheme's own initialisation from
    mass (``make_RainNumber`` / ``make_IceNumber``,
    gpuwm.core.thompson_entry), zero where the analysed mass is absent.
    Every other pair, and every pair under ``"preserve-size"``, keeps the
    background mean particle mass (number scaled with mass); where the
    background had no number the number stays zero and the applier's moment
    repair (gpuwm.da.moments, the scheme's own authority) sets it.  No other
    scheme's relation is invented here.  ``"off"``: unchanged.
    """
    receipt = {"mode": mode, "species": {}}
    if mode == "off":
        return increments, receipt
    scheme_fns = {}
    if mode == "scheme" and mp_physics is not None and int(mp_physics) in (8, 28):
        from gpuwm.core.thompson_entry import make_ice_number, make_rain_number
        scheme_fns = {"nr": make_rain_number, "ni": make_ice_number}
    out = dict(increments)
    for mass, num in PAIRS:
        if mass not in increments or num not in prior or mass not in prior:
            continue
        fn = scheme_fns.get(num)
        q_inc = _host(increments[mass])
        n_prior = np.asarray(prior[num], dtype=np.float64)
        q_prior = np.asarray(prior[mass], dtype=np.float64)
        new = np.zeros(n_prior.shape, dtype=np.float64)
        changed_total = 0
        for slot, index in enumerate(indices):
            qa = np.maximum(q_prior[slot] + q_inc[slot].astype(np.float64), 0.0)
            qb = q_prior[slot]
            nb = n_prior[slot]
            changed = q_inc[slot] != 0
            changed_total += int(changed.sum())
            na = nb.copy()
            if changed.any():
                qa_c, qb_c, nb_c = qa[changed], qb[changed], nb[changed]
                ratio = np.where(qb_c > MASS_ABSENT,
                                 qa_c / np.maximum(qb_c, MASS_ABSENT), 0.0)
                kept = np.where(qa_c > MASS_ABSENT, nb_c * ratio, 0.0)
                if fn is not None:
                    state = states[index]
                    rho = 1.0 / np.asarray(state["alt"], np.float64)[changed]
                    present = qa_c > MASS_ABSENT
                    scheme = np.zeros(qa_c.shape)
                    if present.any():
                        vapour = (np.asarray(prior["qv"][slot])[changed][present]
                                  if "qv" in prior else 0.0)
                        temp = _temperature(
                            np.asarray(state["p"])[changed][present],
                            np.asarray(state["alt"])[changed][present], vapour)
                        scheme[present] = (np.asarray(
                            fn(qa_c[present] * rho[present], temp),
                            np.float64) / rho[present])
                    na[changed] = scheme
                else:
                    na[changed] = kept
            new[slot] = na - nb
        out[num] = new.astype(np.asarray(prior[num]).dtype, copy=False)
        receipt["species"][num] = {
            "from": mass, "relation": ("scheme" if fn is not None
                                       else "background size"),
            "points_rediagnosed": changed_total}
    return out, receipt


def radar_rule_gates(batches, analysis_fields, *, shape, echo, z_thermo_weight,
                     z_hydrometeors, in_pass_keep=True, keep0_thermo=None):
    """``(gates, echo_gates, receipt)``: the design rules as per-column batch
    masks for the filter's own pass (lane 0's in-pass withhold), grouped by
    the same field sets lane 0's rules use so the groups stay disjoint.

    * winds: reflectivity and clear air withheld from u, v, w everywhere;
    * theta/qv: reflectivity and clear air withheld from thp, qv in every
      column outside observed echo (everywhere when ``z_thermo_weight`` is 0);
    * hydrometeors: every batch that does not observe condensate withheld
      from lane 0's hydrometeor field set everywhere (and, with
      ``z_hydrometeors`` off, the condensate batches too).

    Inside echo the radar keeps ``z_thermo_weight`` of its theta/qv
    increment.  With ``in_pass_keep`` that is a ``keep=w`` gate the filter's
    own pass blends (lane 0's per-zone keep); precedence where a keep-0
    theta/qv gate of another kind already holds (``keep0_thermo``, the
    dispersion or cloud-water-path columns): the radar's theta/qv is fully
    withheld there too, because one zone blends one keep.  Without
    ``in_pass_keep`` the in-echo gates come back separately as
    ``echo_gates`` for the host blend (the route a qv cap needs).
    """
    from gpuwm.da.velocity_dispersion import DispersionGate

    ny, nx = int(shape[-2]), int(shape[-1])
    everywhere = np.ones((ny, nx), dtype=bool)
    echo = np.zeros((ny, nx), dtype=bool) if echo is None else np.asarray(echo, bool)
    analysed = set(analysis_fields)
    winds = tuple(f for f in WINDS if f in analysed)
    thermo = ("thp", "qv")
    hydro = withheld_fields({"withheld": "hydrometeors"})
    w = float(z_thermo_weight)
    gates, echo_gates, rows = [], [], []
    for batch in batches:
        token = batch_token(batch.name)
        radar = batch.name in RADAR_BATCHES
        condensate = radar or token in CONDENSATE_TOKENS
        if radar and winds:
            gates.append(DispersionGate(batch=batch.name, fields=winds,
                                        columns=everywhere))
            rows.append({"batch": batch.name, "fields": list(winds),
                         "columns": "all"})
        if radar and set(thermo) & analysed:
            if w == 0.0 or not echo.any():
                cols, keep_cols = everywhere, None
            else:
                blocked = (np.zeros((ny, nx), bool) if keep0_thermo is None
                           else np.asarray(keep0_thermo, bool))
                keep_cols = echo & ~blocked
                cols = ~keep_cols
            gates.append(DispersionGate(batch=batch.name, fields=thermo,
                                        columns=cols))
            rows.append({"batch": batch.name, "fields": list(thermo),
                         "columns": int(np.count_nonzero(cols)), "keep": 0.0})
            if keep_cols is not None and keep_cols.any():
                if in_pass_keep:
                    gates.append(DispersionGate(batch=batch.name,
                                                fields=thermo,
                                                columns=keep_cols, keep=w))
                    rows.append({"batch": batch.name, "fields": list(thermo),
                                 "columns": int(np.count_nonzero(keep_cols)),
                                 "keep": w})
                else:
                    echo_gates.append(DispersionGate(
                        batch=batch.name, fields=thermo, columns=keep_cols))
        if set(hydro) & analysed and (not condensate or not z_hydrometeors):
            gates.append(DispersionGate(batch=batch.name, fields=hydro,
                                        columns=everywhere))
            rows.append({"batch": batch.name, "fields": "hydrometeors",
                         "columns": "all"})
    return gates, (echo_gates or None), {
        "schema": SCHEMA, "rules": "design", "route": "in-pass gates",
        "z_thermo_weight": w, "z_hydrometeors": bool(z_hydrometeors),
        "echo_columns": int(echo.sum()), "gates": rows}


def blend_echo(increments, withheld, *, columns, weight, qv_cap):
    """Inside the echo columns: ``withheld + weight * (joint - withheld)``
    for thp and qv, the radar part of qv capped at ``qv_cap``."""
    out = dict(increments)
    col = np.asarray(columns, dtype=bool)[None, None]
    for name in ("thp", "qv"):
        if name not in increments or name not in withheld:
            continue
        joint = _host(increments[name]).astype(np.float64)
        base = _host(withheld[name]).astype(np.float64)
        radar = (joint - base) * float(weight)
        if name == "qv" and qv_cap is not None:
            radar = np.clip(radar, -float(qv_cap), float(qv_cap))
        out[name] = np.where(col, base + radar, joint).astype(
            _host(increments[name]).dtype, copy=False)
    return out


def zero_unanalysed(increments, analysis_fields):
    """Numbers and aerosols take no filter increment under the design rules
    (numbers are rediagnosed from analysed mass afterwards)."""
    out = dict(increments)
    for name in analysis_fields:
        if name in NUMBERS or name in AEROSOLS:
            out[name] = np.zeros_like(_host(increments[name]))
    return out
