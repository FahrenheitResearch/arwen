"""WRF namelist importer: namelist.wps + namelist.input -> experiment TOML.

Phase-5 Task 1 (recon G10, panel lane L1).  ``import_namelists`` reads both
WRF namelists into the typed experiment schema (gpuwm/experiment.py),
converts WRF's STAGGERED ``e_we``/``e_sn``/``e_vert`` to gpuwm mass
dimensions explicitly (one fewer point: the bundle's 251 x 201 x 50
namelist domain is gpuwm's 250 x 200 x 49), maps only implemented physics
schemes, and emits the resolved TOML text plus a structured
:class:`SubstitutionReport` -- NEVER silent: every namelist key is either
translated, recorded as a ratified substitution (mp 55 -> 10 Morrison,
RRTMG -> RTE+RRTMGP; bl_pbl 11 imports natively since the Shin-Hong
port), recorded as consumed-without-counterpart, or a hard error.

Child ``dx``/``dy`` namelist entries (the bundle hand-types
``333.333333`` for d04) are cross-checked against the exact parent-chain
rational and then DROPPED from the output: the resolved TOML never
hand-types child dx/dt -- gpuwm/experiment.py derives them exactly
(dx 12000 -> 3000 -> 1000 -> 1000/3 m; dt 60 -> 15 -> 5 -> 5/3 s for the
bundle's 1,4,3,3 ratios).  The namelist chain is authoritative; the
"500 m d04" prose value is a pinned hard error, not a rounding choice.

Omitted namelist keys resolve to their WRF v4.6.1 Registry defaults
(review F2), so an implicitly two-way namelist (omitted ``feedback`` =>
Registry default 1) enables the plainly labelled experimental feedback
path instead of silently importing as one-way; ``km_opt`` (Registry default -1 =
must-set) is a hard error when omitted.  Values gpuwm supplies WITHOUT a
Registry source (the ``ztop`` scaffold; ``time_step_sound`` auto -> 4)
are recorded as :class:`AppliedDefault` entries.  Registry defaults that
bind for the reference run and differ from gpuwm's frozen RunConfig
defaults are emitted explicitly (Phase-4 ratified native-dt baseline):
``emdiv = 0.01``, ``hypsometric_opt = 2``, ``h_sca_adv_order = 5``.

Knob-parity contract (product/knobs lane): every namelist key the model
GENUINELY HONORS maps to its experiment-TOML counterpart, and the
report classifies every consumed key into exactly one of three explicit
sections -- TRANSLATED (a TOML value came out of it, including the
ratified substitutions), FIXED BY ARWEN (the key was validated against
the single implemented value and recorded with that value and why), or
NOT IMPLEMENTED (consumed without a counterpart, each with a reason).
RunConfig-honored keys the importer previously refused as unmapped now
translate: the per-domain turbulence row ``c_s``, ``c_k``,
``mix_isotropic``, ``mix_upper_bound``, ``tke_upper_bound``,
``tke_heat_flux``, ``tke_drag_coefficient`` (every one ``max_domains`` in
the Registry and admitted per domain by
``gpuwm.experiment._DOMAIN_RUN_OVERRIDES``, so a PBL parent may carry a
PBL-off LES child that names its own closure constants), the per-domain
moist-filter switch ``moist_mix6_off`` (Registry.EM_COMMON:2889,
divergence-ledger entry L4), plus
``diff_6th_thresh``, ``no_mp_heating``,
``mp_tend_lim``, ``ysu_topdown_pblmix``, ``isftcflx``, ``iz0tlnd``,
``usemonalb``, ``rdlai2d``, ``opt_thcnd`` (each emitted only when the
namelist supplies it -- their Registry defaults equal gpuwm's frozen
RunConfig defaults, so the established imports stay byte-identical).
Keys the core pins where WRF has options are validated against the
pinned value and refused otherwise (``rk_ord = 3``,
``h_mom_adv_order = 5``, ``v_mom_adv_order = v_sca_adv_order = 3``,
``momentum_adv_opt = 1``, ``swint_opt = 0``, ``use_mp_re = 1``, the
MYNN/Noah-MP/RUC option identities, all &stoch selectors off, no FDDA
nudging) -- never silently reinterpreted.
"""

from __future__ import annotations

import math
import re
import tomllib
from dataclasses import dataclass
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path

from gpuwm.experiment import _MOVING_NEST_KEYS, build_experiment
from gpuwm.static.projection import WRF_MAP_PROJ_CODES
from gpuwm.physics_compat import (
    RRTMG_VARIANT_LEGACY,
    RRTMG_VARIANT_RTE_RRTMGP,
    WRF_RRTMG_LEGACY,
    WRF_RRTMG_TO_RTE_RRTMGP,
    require_ready_wrf_physics,
)

#: Relative tolerance for the namelist child-dx cross-check (matches
#: gpuwm.experiment._REL_TOL): the bundle's truncated ``333.333333`` vs
#: the exact 1000/3 m passes (~1e-9); a hand-typed 500 m fails hard.
_REL_TOL = 1.0e-6


# ---------------------------------------------------------------------------
# Report types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Substitution:
    """One ratified physics substitution applied by the importer."""

    key: str            # WRF namelist key
    wrf_value: object
    wrf_name: str       # WRF scheme name
    gpuwm_key: str      # resolved TOML key
    gpuwm_value: object
    gpuwm_name: str     # gpuwm scheme name


@dataclass(frozen=True)
class DroppedKey:
    """A namelist key consumed without a gpuwm TOML counterpart."""

    section: str
    key: str
    values: tuple
    reason: str


@dataclass(frozen=True)
class FixedKey:
    """A namelist key validated against the single implemented value.

    The key never reaches the emitted TOML because ArWen has exactly one
    implemented behavior for it; the importer checked the supplied value
    against that pin (any other value is a hard error, never a silent
    reinterpretation) and records the pin and its evidence here.
    """

    section: str
    key: str
    values: tuple
    fixed_value: object
    reason: str


@dataclass(frozen=True)
class TranslatedKey:
    """A namelist key that produced a value in the emitted TOML."""

    section: str
    key: str


@dataclass(frozen=True)
class AppliedDefault:
    """A resolved-TOML value the namelist did not supply.

    Omitted WRF keys default to their v4.6.1 Registry values (F2 fix)
    and are visible in the emitted TOML; entries here additionally record
    every value whose gpuwm resolution DEVIATES from (or has no) WRF
    Registry source -- the never-silent contract's last mile.
    """

    key: str
    value: object
    reason: str


@dataclass(frozen=True)
class SubstitutionReport:
    """Structured record of every non-1:1 importer decision.

    ``format`` renders the three explicit knob-parity sections --
    translated / fixed-by-ArWen / not-implemented -- plus the
    gpuwm-supplied-values footnote (the F2 never-silent last mile).
    """

    substitutions: tuple[Substitution, ...]
    dropped: tuple[DroppedKey, ...]
    defaults_applied: tuple[AppliedDefault, ...] = ()
    fixed: tuple[FixedKey, ...] = ()
    translated: tuple[TranslatedKey, ...] = ()

    def format(self) -> str:
        lines = []
        if self.translated:
            lines.append(f"Translated (namelist -> experiment TOML): "
                         f"{len(self.translated)} key(s)")
            by_section: dict[str, list[str]] = {}
            for t in self.translated:
                by_section.setdefault(t.section, []).append(t.key)
            for section, keys in by_section.items():
                lines.append(f"  &{section}: {', '.join(sorted(keys))}")
        lines.append("Physics substitutions (ratified):")
        if self.substitutions:
            for s in self.substitutions:
                lines.append(
                    f"  {s.key} {s.wrf_value} ({s.wrf_name}) -> "
                    f"{s.gpuwm_key} {s.gpuwm_value} ({s.gpuwm_name})")
        else:
            lines.append("  (none)")
        if self.fixed:
            lines.append("Fixed by ArWen (validated against the only "
                         "implemented value):")
            for f in self.fixed:
                values = ", ".join(str(v) for v in f.values)
                lines.append(f"  &{f.section} {f.key} = {values} "
                             f"[fixed: {f.fixed_value}]: {f.reason}")
        if self.defaults_applied:
            lines.append("gpuwm-supplied values without a WRF Registry "
                         "source:")
            for a in self.defaults_applied:
                lines.append(f"  {a.key} = {a.value}: {a.reason}")
        lines.append("Not implemented (namelist keys consumed without a "
                     "gpuwm counterpart):")
        for d in self.dropped:
            values = ", ".join(str(v) for v in d.values)
            lines.append(f"  &{d.section} {d.key} = {values}: {d.reason}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fortran-namelist parsing (multi-line value continuations included --
# the bundle's eta_levels span ten lines)
# ---------------------------------------------------------------------------

_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")
#: Fortran repetition constant ``N*value`` (e.g. ``3*1.0``, ``4*.true.``).
_REPEAT = re.compile(r"([1-9]\d*)\*(.+)")
#: Fortran double-precision exponent literal (``2.90D2``); Python floats
#: only accept E, so D/d maps to e before conversion.
_D_EXPONENT = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)[dD][+-]?\d+")


def _parse_scalar(tok: str):
    if _QUOTED.fullmatch(tok):
        return tok[1:-1]
    low = tok.lower()
    if low in (".true.", ".t.", "t"):
        return True
    if low in (".false.", ".f.", "f"):
        return False
    if _D_EXPONENT.fullmatch(tok):
        return float(low.replace("d", "e"))
    try:
        return int(tok)
    except ValueError:
        pass
    try:
        return float(tok)
    except ValueError:
        return tok


def _parse_token(tok: str) -> list:
    """One comma-separated token -> list of values (repetition expands)."""
    match = None if _QUOTED.fullmatch(tok) else _REPEAT.fullmatch(tok)
    if match:
        return [_parse_scalar(match.group(2).strip())] * int(match.group(1))
    return [_parse_scalar(tok)]


def _tokens(rhs: str) -> list:
    out: list = []
    for tok in rhs.strip().rstrip(",").split(","):
        tok = tok.strip()
        if tok:
            out.extend(_parse_token(tok))
    return out


def read_namelist_role(path: str | Path, role: str) -> dict[str, dict[str, list]]:
    """:func:`parse_namelist`, with an unreadable file as one sentence.

    THE shared refusal for "you named a namelist that is not there".
    ``gpuwm import-namelist`` has always produced it; ``rw-wps
    --namelist-support-report``, documented as step one of
    migrating-from-wps.md, called :func:`parse_namelist` directly and
    handed the reader a five-frame ``FileNotFoundError`` traceback for
    the identical mistake.  Two surfaces answering the same condition
    differently is the bug; one function is the fix.
    """

    try:
        return parse_namelist(path)
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read {role} {path}: {error}") from None


def parse_namelist(path: str | Path) -> dict[str, dict[str, list]]:
    """Parse a Fortran namelist FILE into {section: {key: [values]}}."""

    return parse_namelist_text(Path(path).read_text(encoding="utf-8"))


def parse_namelist_text(text: str) -> dict[str, dict[str, list]]:
    """Parse Fortran namelist TEXT into {section: {key: [values]}}.

    Handles the WRF/WPS style: ``&section`` .. ``/`` blocks,
    ``key = v1, v2,`` lines, bare continuation lines carrying further
    values for the previous key (namelist.input eta_levels), repetition
    constants (``3*1.0``), and D-exponent reals (``2.90D2``).

    Split out from :func:`parse_namelist` so a RENDERER can read back
    what it just produced without writing a file first -- an emitter
    that checks its own bytes is checking the artifact, and one that
    checks its own variables is checking its intentions.
    """
    sections: dict[str, dict[str, list]] = {}
    current: dict[str, list] | None = None
    current_key: str | None = None
    for raw in text.splitlines():
        line = raw.split("!", 1)[0].strip()
        if not line:
            continue
        if line.startswith("&"):
            name = line[1:].strip().lower()
            sections[name] = {}
            current, current_key = sections[name], None
            continue
        if line == "/":
            current, current_key = None, None
            continue
        if current is None:
            continue
        if "=" in line:
            key, rhs = line.split("=", 1)
            current_key = key.strip().lower()
            current[current_key] = _tokens(rhs)
        elif current_key is not None:
            current[current_key].extend(_tokens(line))
    return sections


# ---------------------------------------------------------------------------
# The d01-only view of a namelist that may describe a whole hierarchy
# ---------------------------------------------------------------------------

#: Sentence a single-domain consumer appends when the namelist it was
#: handed describes more than one domain.  Kept here so the diagnosis is
#: spelled the same wherever it is raised.
MULTI_DOMAIN_ROOT_VIEW_HINT = (
    "multi-domain namelist passed to a single-domain root preparer: "
    "d01 is the domain being prepared, so its own (FIRST) column value is "
    "the one that binds; the differing per-domain column(s) below belong "
    "to the nests.  Fix by correcting d01's value, or by preparing the "
    "root from a max_dom = 1 namelist")


def root_domain_namelist_view(
        sections, *, source: str = "",
) -> tuple[dict[str, dict[str, object]], tuple[str, ...]]:
    """``(d01 view, per-domain columns that differ across domains)``.

    WRF per-domain arrays are ordered d01..dNN, so the head grid's value
    is element zero of every column -- for ``&physics``, ``&dynamics``,
    ``&time_control`` and the rest alike.  A single-domain consumer handed
    a hierarchy namelist wants exactly that view, and nothing else in the
    file.

    The second return value names every column that is NOT uniform.  Those
    are the keys where reading the wrong end of the array silently
    prepares a different domain, and a consumer that refuses on one of
    them owes the user :data:`MULTI_DOMAIN_ROOT_VIEW_HINT` rather than a
    sentence blaming the value it happened to read.

    ``source`` appears in refusals about a malformed column.
    """

    view: dict[str, dict[str, object]] = {}
    differing: list[str] = []
    for section_name, entries in sections.items():
        resolved: dict[str, object] = {}
        for key, values in entries.items():
            if not isinstance(values, list) or not values:
                raise ValueError(
                    f"&{section_name}/{key}"
                    + (f" of {source}" if source else "")
                    + " has no value; a WRF namelist key must carry at "
                    "least its d01 column entry")
            resolved[key] = values[0]
            if any(value != values[0] for value in values[1:]):
                differing.append(f"&{section_name}/{key}")
        view[section_name] = resolved
    return view, tuple(sorted(differing))


# ---------------------------------------------------------------------------
# Ratified scheme maps.  Every entry is (gpuwm value, WRF scheme name,
# gpuwm scheme name); identity entries share the name.  Any WRF value
# absent from its map is a hard error -- only implemented schemes map.
# ---------------------------------------------------------------------------

_MP_MAP = {
    0: (0, "none", "none"),
    1: (1, "Kessler", "Kessler"),
    6: (6, "WSM6", "WSM6"),
    8: (8, "Thompson", "Thompson"),
    10: (10, "Morrison 2-moment", "Morrison 2-moment"),
    18: (18, "NSSL 2-moment", "NSSL 2-moment"),
    # A TRANSLATION, not a substitution: the mapped value equals the WRF
    # value, so PHYSICS.md's "exactly three ratified substitutions" claim is
    # untouched.  Mapping 28 -> 8 would have been the substitution, and it
    # would have been a lie: classic Thompson pins nc at Nt_c = 100e6 and
    # runs Cooper nucleation, while 28 carries prognostic nc/nwfa/nifa,
    # CCN activation and iceDeMott.  Silently downgrading a namelist that
    # asked for aerosol-aware physics is precisely the failure this map
    # exists to prevent.
    28: (28, "Thompson aerosol-aware", "Thompson aerosol-aware"),
    55: (10, "ISHMAEL", "Morrison 2-moment"),
}

#: WRF's aerosol-aware-Thompson namelist surface, section -> key -> what
#: ArWen does with it.  ``_Section.finish`` refuses any key it was not
#: offered, so every one of these must be consumed explicitly; the choice is
#: only ever between an INERT drop (the key changes nothing under the
#: selected scheme, in WRF either) and a hard refusal that names what is
#: missing.
#:
#: Under ``mp_physics != 28`` every key here is inert in WRF as well -- the
#: thompsonaero package is the only consumer -- so they drop with that
#: reason.  Under ``mp_physics == 28`` each one is refused by name, because
#: ArWen's established posture is to REFUSE where WRF silently overwrites:
#: WRF's ``share/module_check_a_mundo.F`` forces ``grav_settling`` to 0
#: (:2459-2474) and ``scalar_pblmix`` to 1 (:2477-2495) under mp=28 without
#: failing, and a user who asked for the other value would otherwise be
#: given a different model than the one they wrote down.
_MP28_AEROSOL_NAMELIST_KEYS: dict[str, dict[str, str]] = {
    "physics": {
        "use_aero_icbc":
            "ArWen has no aerosol IC/BC ingest: use_aero_icbc=.true. makes "
            "real.exe derive aer_init_opt=1 and read QNWFA/QNIFA from "
            "metgrid (dyn_em/module_initialize_real.F:2325-2732), and there "
            "is no such stream here",
        "use_rap_aero_icbc":
            "the same missing aerosol IC/BC ingest as use_aero_icbc, "
            "RAP-sourced variant (share/module_check_a_mundo.F:2477-2495 "
            "pairs the two)",
        "qna_update":
            "no aerosol IC/BC lane exists to update from; the knob only "
            "means anything alongside use_aero_icbc",
        "scalar_pblmix":
            "PBL scalar mixing of the aerosol scalars qnc/qnwfa/qnifa is "
            "not wired -- "
            "gpuwm/core/mynn_pbl.py passes flag_qnc/flag_qnwfa/flag_qnifa "
            "as literal False.  WRF SILENTLY forces scalar_pblmix=1 under "
            "mp_physics=28 with use_aero_icbc "
            "(share/module_check_a_mundo.F:2477-2495); ArWen refuses rather "
            "than accepting a value it would not honour",
        "grav_settling":
            "WRF SILENTLY forces grav_settling=0 under mp_physics=28 "
            "because the scheme already has gravitational fog settling "
            "(share/module_check_a_mundo.F:2459-2474); ArWen has no fog "
            "settling option at all, so it refuses the key instead of "
            "accepting a request it would overwrite",
        "wif_fire_emit":
            "no biomass-burning aerosol emission source "
            "(Registry.EM_COMMON:2657); ArWen's only nwfa2d is "
            "thompson_init's derived surface emission "
            "(phys/module_mp_thompson.F:510)",
        "wif_fire_inj":
            "no biomass-burning aerosol injection profile "
            "(Registry.EM_COMMON:2659), for the same reason.  Refused even "
            "at its WRF Registry default of 1, because with no emission "
            "source there is no honest value: 1 describes a vertical "
            "distribution ArWen never performs",
        "dust_emis":
            "no dust emission source (Registry.EM_COMMON:2591); ArWen's "
            "nifa2d is identically zero, exactly as "
            "module_mp_thompson.F leaves it",
    },
    "domains": {
        "wif_input_opt":
            "no WIF metgrid ingest.  WRF FATALs mp_physics=28 with "
            "wif_input_opt=0 "
            "(dyn_em/module_initialize_real.F:2735-2736) and ArWen runs "
            "thompson_init's synthetic profile instead, so no value of this "
            "key describes a configuration both models can produce",
        "num_wif_levels":
            "the WIF vertical axis has no consumer without the ingest "
            "(Registry/registry.new3d_wif:14-16)",
    },
}
_BL_MAP = {
    0: (0, "none", "none"),
    1: (1, "YSU", "YSU"),
    # The coupled MYNN 5/5 suite.  Two of the three legs the comment below
    # requires were already in place -- physics_compat admits the pair and
    # PHYSICS_SLOT_DISPATCH runs it, which is what the shipped MYNN fixed
    # profile forecasts with.  Only the importer leg lagged, so MYNN was
    # reachable at mp_physics=6 and nowhere else: pairing it with Thompson,
    # Morrison or NSSL had no front door at all.  Adding the map entry
    # widens no gate -- require_ready_wrf_physics still previews the whole
    # suite below and refuses every pairing it refused before.  (The
    # RUC/Noah-MP pairing refusals named in an earlier version of this
    # comment were retired by the surface-driver ownership port.)
    5: (5, "MYNN2.5", "MYNN"),
    # Native since the Shin-Hong port (certified CPU authority, max ULP 0
    # against WRF v4.6.1; see the physics registry's shinhong option).  The
    # row was (1, "Shin-Hong", "YSU") -- a ratified substitution -- until
    # the scheme itself was admitted; per the doc block below, this map row
    # widens together with the physics_compat readiness row (the WRF
    # matrix's four (11, sfclay) cells) and the PHYSICS_SLOT_DISPATCH row.
    11: (11, "Shin-Hong", "Shin-Hong"),
}
_RA_LW_MAP = {
    0: (0, "none", "none"),
    1: (1, "RRTM", "WRF RRTM"),
    4: (4, "RRTMG", "RTE+RRTMGP"),
}
_RA_SW_MAP = {
    0: (0, "none", "none"),
    1: (1, "Dudhia", "WRF Dudhia"),
    4: (4, "RRTMG", "RTE+RRTMGP"),
}
#: WRF values the importer may emit into a gpuwm configuration.  These are
#: the RUNNABLE sets, deliberately narrower than gpuwm/config.py's schema
#: tables: importing a namelist writes a file someone will later run, so a
#: value that config would accept but physics_compat still refuses must fail
#: here rather than produce an unrunnable config.  Admitting a scheme means
#: widening these together with its physics_compat row and its
#: PHYSICS_SLOT_DISPATCH row -- never one of the three alone.
_SFCLAY_ALLOWED = {0, 1, 5, 91}
_SFSFC_ALLOWED = {0, 2, 3, 4}
_CU_ALLOWED = {0, 1, 3}


def _port_receipt(**selection) -> str:
    """The fail-closed port receipt for an in-port selector, or ``""``."""
    from gpuwm.physics_compat import pending_wrf_physics_components

    request = {"mp_physics": 0, "sf_sfclay_physics": 0, "bl_pbl_physics": 0,
               "sf_surface_physics": 0, "num_soil_layers": 4}
    request.update(selection)
    blockers = pending_wrf_physics_components(**request)
    if not blockers:
        return ""
    return " Port status: " + "; ".join(
        item.format() for item in blockers)


def _err(section: str, key: str, value, why: str) -> ValueError:
    return ValueError(f"&{section} {key} = {value!r}: {why}")


def _identity_matches(value, admitted) -> bool:
    """Type-strict comparison for option-identity pins.

    Booleans must be Fortran logicals (never 0/1 integers), floats accept
    exact integer spellings (``soiltstep = 0``), and integers refuse
    fractional or logical tokens -- no coercion that could smuggle a
    different option through the pin.
    """
    if isinstance(admitted, bool):
        return isinstance(value, bool) and value == admitted
    if isinstance(admitted, float):
        return (isinstance(value, (int, float))
                and not isinstance(value, bool)
                and float(value) == admitted)
    return (isinstance(value, int) and not isinstance(value, bool)
            and value == admitted)


#: Keys the importer requires to be PRESENT (contract values are checked
#: later, one rule at a time).  Censused in one sweep up front so a
#: namelist missing several keys yields ONE report naming all of them
#: instead of one refusal per re-run.  The scattered ``required()``
#: raises stay as fail-closed backstops.
_REQUIRED_KEYS = {
    ("wps", "geogrid"): (
        "dx", "e_sn", "e_we", "i_parent_start", "j_parent_start",
        "parent_grid_ratio", "parent_id", "ref_lat", "ref_lon",
        "stand_lon", "truelat1", "truelat2"),
    ("input", "time_control"): (
        "end_day", "end_month", "end_year",
        "start_day", "start_month", "start_year"),
    ("input", "domains"): (
        "e_sn", "e_vert", "e_we", "i_parent_start", "j_parent_start",
        "parent_grid_ratio", "parent_id", "parent_time_step_ratio",
        "time_step"),
    ("input", "dynamics"): ("km_opt", "mix_full_fields"),
    ("input", "physics"): (
        "bl_pbl_physics", "mp_physics", "ra_lw_physics", "ra_sw_physics"),
}


def _census_missing_keys(wps: dict, inp: dict,
                         wps_path, input_path) -> None:
    """One report of every missing required key across both namelists."""

    parsed = {"wps": (wps, wps_path), "input": (inp, input_path)}
    lines = []
    for (role, section), keys in _REQUIRED_KEYS.items():
        document, path = parsed[role]
        entries = document.get(section, {})
        absent = sorted(key for key in keys if key not in entries)
        if absent:
            lines.append(f"  &{section} of {path}: {', '.join(absent)}")
    if lines:
        raise ValueError(
            "the namelist pair is missing required key(s):\n"
            + "\n".join(lines)
            + "\nadd every listed key and re-run (this is the complete "
            "missing-key inventory, not the first failure).")


class _Section:
    """One namelist section with consume-or-fail key accounting."""

    def __init__(self, name: str, entries: dict[str, list], source: str):
        self.name = name
        self.entries = dict(entries)
        self.source = source
        #: Keys actually popped from the namelist, in consumption order --
        #: the raw material of the report's Translated section (keys later
        #: recorded as fixed/dropped are subtracted there).
        self.consumed: list[str] = []
        moving = sorted(_MOVING_NEST_KEYS & set(self.entries))
        if moving:
            raise ValueError(
                f"moving-nest key(s) {moving} in &{name} of {source} are "
                "rejected: gpuwm implements static nests only.")

    def take(self, key: str, default=None) -> list | None:
        if key in self.entries:
            self.consumed.append(key)
            return self.entries.pop(key)
        return default

    def col(self, key: str, n: int, default=None) -> list | None:
        """Per-domain column using the importer's ratified last-value fill.

        This is not generic Fortran namelist behavior. Registry arrays whose
        omitted tail must retain its initialized default use ``registry_col``.
        """
        values = self.take(key)
        if values is None:
            return None if default is None else [default] * n
        if len(values) > n:
            values = values[:n]
        return values + [values[-1]] * (n - len(values))

    def registry_col(self, key: str, n: int, default) -> list:
        """Per-domain column whose omitted tail keeps its Registry default.

        Fortran namelist assignment does not broadcast a scalar across an
        array: elements not named by the input retain their initialized WRF
        Registry values.  Use this for Registry arrays where that distinction
        changes the resolved experiment rather than for columns WRF explicitly
        normalizes elsewhere.
        """
        values = self.take(key)
        if values is None:
            return [default] * n
        values = values[:n]
        return values + [default] * (n - len(values))

    def scalar(self, key: str, default=None):
        values = self.take(key)
        if values is None:
            return default
        return values[0]

    def required(self, key: str):
        values = self.take(key)
        if values is None:
            raise ValueError(
                f"&{self.name} of {self.source} is missing required "
                f"key {key}.")
        return values[0]

    def finish(self) -> None:
        if self.entries:
            raise ValueError(
                f"unmapped key(s) {sorted(self.entries)} in &{self.name} "
                f"of {self.source}: the importer refuses to drop namelist "
                "settings silently -- extend the ratified map or remove "
                "the key(s).")


def _uniform(section: str, key: str, values: list):
    first = values[0]
    if any(v != first for v in values[1:]):
        raise _err(section, key, values,
                   "per-domain values must be identical (gpuwm shares "
                   "this setting across domains).")
    return first


def _require_bools(section: str, key: str, values: list) -> list[bool]:
    """Fortran-logical columns must decode to real booleans -- a residual
    string (e.g. an unparsed token) must never be truth-tested (shadow
    review S2's silent false->true hazard)."""
    for v in values:
        if not isinstance(v, bool):
            raise _err(section, key, v,
                       "must be a Fortran logical (.true./.false.).")
    return [bool(v) for v in values]


def _fmt(value) -> str:
    """TOML literal for a scalar value."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat(sep="T")
    return '"' + str(value) + '"'


# ---------------------------------------------------------------------------
# The importer
# ---------------------------------------------------------------------------

def import_namelists(wps_path: str | Path, input_path: str | Path,
                     name: str | None = None,
                     rrtmg_variant: str = RRTMG_VARIANT_RTE_RRTMGP
                     ) -> tuple[str, SubstitutionReport]:
    """Translate namelist.wps + namelist.input into (TOML text, report).

    ``name`` sets ``[experiment].name`` (the CLI ``--name`` flag);
    the default is derived deterministically from the start time and
    domain count.  ``rrtmg_variant`` selects which gpuwm implementation a
    WRF RRTMG 4/4 request maps to: the default ``"rte-rrtmgp"`` keeps the
    established substitution (and its output byte-identical), while
    ``"rrtmg_legacy"`` maps to the exact legacy-RRTMG port and emits its
    own compatibility token (the resulting configuration fails closed at
    physics setup until the legacy compute kernels land).  The returned
    TOML is validated through :func:`gpuwm.experiment.build_experiment`
    before being returned, so every section-A rule (root/child flags,
    ratios, clearance, vertical identity, cadence divisibility, derived
    dt/dx chain) binds at import time.
    """
    if rrtmg_variant not in (RRTMG_VARIANT_RTE_RRTMGP,
                             RRTMG_VARIANT_LEGACY):
        raise ValueError(
            f"rrtmg_variant must be '{RRTMG_VARIANT_RTE_RRTMGP}' or "
            f"'{RRTMG_VARIANT_LEGACY}', got {rrtmg_variant!r}")
    wps_path, input_path = Path(wps_path), Path(input_path)

    wps = read_namelist_role(wps_path, "namelist.wps")
    inp = read_namelist_role(input_path, "namelist.input")
    substitutions: list[Substitution] = []
    dropped: list[DroppedKey] = []
    fixed: list[FixedKey] = []
    defaults_applied: list[AppliedDefault] = [AppliedDefault(
        key="ztop", value=20000.0,
        reason="gpuwm-only vertical scaffold height (no WRF namelist "
               "counterpart); the real path derives heights from "
               "p_top/eta_levels -- the certified reference profile "
               "value")]

    def drop(section: str, key: str, values, reason: str) -> None:
        if values is None:
            return
        if not isinstance(values, list):
            values = [values]
        dropped.append(DroppedKey(section=section, key=key,
                                  values=tuple(values), reason=reason))

    def fix(section: str, key: str, values, fixed_value, reason: str
            ) -> None:
        if values is None:
            return
        if not isinstance(values, list):
            values = [values]
        fixed.append(FixedKey(section=section, key=key,
                              values=tuple(values), fixed_value=fixed_value,
                              reason=reason))

    for section_name in ("ungrib", "metgrid"):
        if section_name in wps:
            for key, values in wps[section_name].items():
                drop(section_name, key, values,
                     "ungrib/metgrid staging is replaced by gpuwm's "
                     "direct ERA5 GRIB ingest")
    # &fdda: an ACTIVE nudging request must refuse (importing it into a
    # model that will not nudge is a silent trajectory change); disabled
    # selectors and their inert companion keys drop with a reason.
    for key, values in inp.get("fdda", {}).items():
        if key in ("grid_fdda", "grid_sfdda", "obs_nudge_opt") \
                and any(int(v) != 0 for v in values):
            raise _err("fdda", key, values,
                       "FDDA nudging is not implemented; set it to 0 (or "
                       "remove &fdda) -- gpuwm will not import an active "
                       "nudging request into a model that cannot nudge.")
    for section_name in ("fdda", "grib2", "namelist_quilt"):
        if section_name in inp:
            for key, values in inp[section_name].items():
                drop(section_name, key, values,
                     "FDDA/GRIB2/quilt-server machinery has no gpuwm "
                     "counterpart")

    known_wps = {"share", "geogrid", "ungrib", "metgrid"}
    unknown = sorted(set(wps) - known_wps)
    if unknown:
        raise ValueError(
            f"unknown section(s) {unknown} in {wps_path}; known: "
            f"{sorted(known_wps)}.")
    known_inp = {"time_control", "domains", "physics", "fdda", "dynamics",
                 "bdy_control", "grib2", "namelist_quilt", "noah_mp",
                 "stoch"}
    unknown = sorted(set(inp) - known_inp)
    if unknown:
        raise ValueError(
            f"unknown section(s) {unknown} in {input_path}; known: "
            f"{sorted(known_inp)}.")
    for required, parsed, path in (("share", wps, wps_path),
                                   ("geogrid", wps, wps_path),
                                   ("time_control", inp, input_path),
                                   ("domains", inp, input_path)):
        if required not in parsed:
            raise ValueError(f"{path} has no &{required} section.")
    _census_missing_keys(wps, inp, wps_path, input_path)

    share = _Section("share", wps["share"], str(wps_path))
    geo = _Section("geogrid", wps["geogrid"], str(wps_path))
    tc = _Section("time_control", inp["time_control"], str(input_path))
    dm = _Section("domains", inp["domains"], str(input_path))
    ph = _Section("physics", inp.get("physics", {}), str(input_path))
    dyn = _Section("dynamics", inp.get("dynamics", {}), str(input_path))
    bdy = _Section("bdy_control", inp.get("bdy_control", {}),
                   str(input_path))
    noahmp = _Section("noah_mp", inp.get("noah_mp", {}), str(input_path))
    stoch = _Section("stoch", inp.get("stoch", {}), str(input_path))

    # ---- &noah_mp: option-identity validation --------------------------
    # Every Noah-MP option is identity-pinned (gpuwm/config.py
    # NOAHMP_OPTION_IDENTITY_EVIDENCE: the ported column is validated at
    # exactly one value of each).  A key at the identity value is recorded
    # as fixed; any other value refuses with the identity's evidence.
    from gpuwm.config import (MYNN_PBL_OPTION_IDENTITY,
                              NOAHMP_OPTION_IDENTITY_EVIDENCE)
    for option, (admitted, evidence) in \
            NOAHMP_OPTION_IDENTITY_EVIDENCE.items():
        values = noahmp.take(option)
        if values is None:
            continue
        if any(not _identity_matches(value, admitted) for value in values):
            raise _err(
                "noah_mp", option, values,
                f"gpuwm's Noah-MP port implements {option} = {admitted!r} "
                f"only ({evidence}); no nearby branch is substituted for "
                "an unported one.")
        fix("noah_mp", option, values, admitted,
            f"Noah-MP option identity ({evidence})")
    noahmp.finish()

    # ---- &stoch: every stochastic scheme must be off --------------------
    # Seed/ensemble bookkeeping keys are inert once every selector is 0
    # (WRF reads iseed_* only inside an enabled scheme) and drop; any
    # nonzero selector is a hard error -- no stochastic physics is
    # implemented.
    for key in sorted(stoch.entries):
        values = stoch.take(key)
        if key.startswith("iseed") or key in ("nens",):
            drop("stoch", key, values,
                 "stochastic seed/ensemble bookkeeping is inert with "
                 "every &stoch selector off")
            continue
        if any(bool(value) for value in values):
            raise _err(
                "stoch", key, values,
                "stochastic physics (SPP/SPPT/SKEBS/rand_perturb) is not "
                "implemented; every &stoch selector must be 0/.false..")
        fix("stoch", key, values, 0,
            "no stochastic physics is implemented; validated off")
    stoch.finish()

    # ---- &share ---------------------------------------------------------
    wrf_core = share.scalar("wrf_core", "ARW")
    if str(wrf_core).upper() != "ARW":
        raise _err("share", "wrf_core", wrf_core, "only ARW is supported.")
    max_dom = dm.scalar("max_dom", 1)
    if isinstance(max_dom, bool) or not isinstance(max_dom, int) \
            or not 1 <= max_dom <= 21:
        raise _err("domains", "max_dom", max_dom,
                   "must be an integer in [1, 21] (WRF's compiled "
                   "max_domains default); refusing to expand per-domain "
                   "arrays for an implausible domain count.")
    wps_max_dom = share.scalar("max_dom")
    if wps_max_dom is not None and wps_max_dom != max_dom:
        raise ValueError(
            f"max_dom mismatch: {wps_path} says {wps_max_dom}, "
            f"{input_path} says {max_dom}.")

    # ---- &time_control: start/run length --------------------------------
    def _dt_columns(prefix: str) -> list[datetime]:
        parts: dict[str, list[int]] = {}
        for unit, default in (("year", None), ("month", None),
                              ("day", None), ("hour", 0), ("minute", 0),
                              ("second", 0)):
            col = tc.col(f"{prefix}_{unit}", max_dom,
                         default=default)
            if col is None:
                raise ValueError(
                    f"{input_path} &time_control is missing "
                    f"{prefix}_{unit}.")
            parts[unit] = list(col)
        try:
            return [
                datetime(parts["year"][index], parts["month"][index],
                         parts["day"][index], parts["hour"][index],
                         parts["minute"][index], parts["second"][index])
                for index in range(max_dom)
            ]
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{input_path} &time_control has an invalid per-domain "
                f"{prefix}_* datetime column: {error}") from None

    start_times = _dt_columns("start")
    end_times = _dt_columns("end")
    start_time = start_times[0]
    end_time = _uniform("time_control", "end_* datetime", end_times)
    run_days = int(tc.scalar("run_days", 0))
    run_hours = int(tc.scalar("run_hours", 0))
    run_minutes = int(tc.scalar("run_minutes", 0))
    run_secs = int(tc.scalar("run_seconds", 0))
    run_seconds = float(run_days * 86400 + run_hours * 3600
                        + run_minutes * 60 + run_secs)
    if run_seconds <= 0.0:
        run_seconds = (end_time - start_time).total_seconds()
    if run_seconds <= 0.0:
        raise ValueError(
            f"{input_path} &time_control yields a non-positive run "
            f"length: run_days/hours/minutes/seconds all zero and "
            f"end {end_time} <= start {start_time}.")
    expected_end = start_time + timedelta(seconds=run_seconds)
    if end_time != expected_end:
        raise ValueError(
            f"{input_path} &time_control run_* ends the root at "
            f"{expected_end}, but the uniform end_* columns declare "
            f"{end_time}.")

    wps_start = share.take("start_date")
    if wps_start is not None:
        if len(wps_start) > max_dom:
            raise ValueError(
                f"{wps_path} &share/start_date declares {len(wps_start)} "
                f"values but max_dom = {max_dom}.")
        wps_start = list(wps_start) + [wps_start[-1]] * (
            max_dom - len(wps_start))
        parsed = [
            datetime.strptime(str(value), "%Y-%m-%d_%H:%M:%S")
            for value in wps_start
        ]
        if parsed != start_times:
            raise ValueError(
                f"per-domain start mismatch: {wps_path} start_date = "
                f"{[value.isoformat() for value in parsed]} but "
                f"{input_path} start_* = "
                f"{[value.isoformat() for value in start_times]}.")
    drop("share", "end_date", share.take("end_date"),
         "run length comes from &time_control run_*/end_*")
    for section, obj, key, reason in (
            ("share", share, "interval_seconds",
             "forcing cadence is discovered and validated from the input "
             "catalog, not declared"),
            ("time_control", tc, "interval_seconds",
             "forcing cadence is discovered and validated from the input "
             "catalog, not declared"),
            ("share", share, "io_form_geogrid",
             "WPS geogrid staging is replaced by gpuwm's static builder"),
            ("share", share, "debug_level", "WPS logging control"),
            ("share", share, "nocolons",
             "gpuwm filenames are always colon-free (H_M_S)"),
            ("time_control", tc, "nocolons",
             "gpuwm filenames are always colon-free (H_M_S)"),
            ("time_control", tc, "history_begin",
             "history alarms start at t=0 (WRF before-solve position)"),
            ("time_control", tc, "frames_per_outfile",
             "gpuwm writes one frame per wrfout file"),
            ("time_control", tc, "restart",
             "resuming is the `gpuwm run --restart` flag, not a config "
             "key"),
            ("time_control", tc, "io_form_history", "WRF I/O layer key"),
            ("time_control", tc, "io_form_restart", "WRF I/O layer key"),
            ("time_control", tc, "io_form_input", "WRF I/O layer key"),
            ("time_control", tc, "io_form_boundary", "WRF I/O layer key"),
    ):
        drop(section, key, obj.take(key), reason)

    # nwp_diagnostics (Registry.EM_COMMON:2210, &time_control, default 0):
    # mapped 1:1 onto the RunConfig field of the same name (knob-parity
    # conventions -- emitted only when supplied, WRF Registry default ==
    # frozen RunConfig default).  gpuwm implements the UP_HELI_MAX member
    # of WRF's nwp_output family per step (gpuwm/core/uh_diag.py); the
    # unimplemented members (WSPD10MAX, W_UP_MAX/W_DN_MAX, W_MEAN,
    # GRPL_MAX, HAIL_MAX*) are simply absent from wrfouts, and WRF's
    # every-step diagflag forcing (solve_em.F:369) changes only which
    # instant computes diagnostics WRF then discards -- the emitted
    # instantaneous REFL_10CM at history time is the same field either
    # way, so the old scope refusal is retired.
    nwp_diagnostics_values = tc.take("nwp_diagnostics")
    nwp_diagnostics = None
    if nwp_diagnostics_values is not None:
        value = nwp_diagnostics_values[0]
        if isinstance(value, bool) or not isinstance(value, int) \
                or value not in (0, 1):
            raise _err(
                "time_control", "nwp_diagnostics", nwp_diagnostics_values,
                "must be 0 (off, the WRF default) or 1 (per-step "
                "UP_HELI_MAX running-max diagnostic).")
        nwp_diagnostics = value

    input_from_file = _require_bools(
        "time_control", "input_from_file",
        tc.col("input_from_file", max_dom, default=True))
    if not all(input_from_file):
        raise _err("time_control", "input_from_file", input_from_file,
                   "every domain must initialize from real data "
                   "(input_from_file = .true.); the parent-only "
                   "interpolation branch is reserved for idealized "
                   "nests.")
    fix("time_control", "input_from_file", input_from_file, True,
        "ERA5-direct per-domain real init IS the input_from_file=T "
        "branch (med_nest_initial, share/mediation_integrate.F:509-952)")

    history_min = tc.col("history_interval", max_dom, default=0)
    history_sec = tc.col("history_interval_s", max_dom, default=0)
    history_interval_s = [
        float(s) if s else float(m) * 60.0
        for m, s in zip(history_min, history_sec)]
    if any(v <= 0.0 for v in history_interval_s):
        raise _err("time_control", "history_interval",
                   list(zip(history_min, history_sec)),
                   "every domain needs a positive history cadence.")
    restart_interval_s = float(tc.scalar("restart_interval", 0)) * 60.0
    # Output-stream and logging keys with no gpuwm counterpart: gpuwm
    # writes its fixed per-domain wrfout product set on the history
    # cadence, so auxiliary stream declarations and I/O field overrides
    # are recorded, never silently influential.  ``adjust_output_times``
    # is inert on gpuwm's exact rational clock (history alarms land
    # exactly on the namelist cadence; WRF needs the adjustment only
    # under the adaptive time step, which is rejected above).
    drop("time_control", "debug_level", tc.take("debug_level"),
         "WRF logging control")
    drop("time_control", "adjust_output_times",
         tc.take("adjust_output_times"),
         "inert on the exact rational clock (fixed dt; alarms land "
         "exactly on the history cadence)")
    _AUX_STREAM = re.compile(r"^aux(hist|input)\d+_")
    for key in sorted(tc.entries):
        if _AUX_STREAM.match(key) or key in (
                "iofields_filename", "ignore_iofields_warning",
                "output_diagnostics"):
            drop("time_control", key, tc.take(key),
                 "auxiliary I/O stream keys are not translated: gpuwm "
                 "writes its fixed per-domain wrfout product set (one "
                 "frame per file)")
    tc.finish()
    share.finish()

    # ---- &geogrid vs &domains cross-checks -------------------------------
    def _int_col(section: _Section, key: str, required=True):
        values = section.col(key, max_dom)
        if values is None:
            if required:
                raise ValueError(
                    f"&{section.name} of {section.source} is missing "
                    f"{key}.")
            return None
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in values):
            raise _err(
                section.name, key, values,
                "must contain Fortran integer tokens; fractional, logical, "
                "and string values are rejected without coercion.")
        return list(values)

    geo_parent_id = _int_col(geo, "parent_id")
    geo_ratio = _int_col(geo, "parent_grid_ratio")
    geo_i = _int_col(geo, "i_parent_start")
    geo_j = _int_col(geo, "j_parent_start")
    geo_e_we = _int_col(geo, "e_we")
    geo_e_sn = _int_col(geo, "e_sn")
    parent_id = _int_col(dm, "parent_id")
    grid_ids = _int_col(dm, "grid_id", required=False) \
        or list(range(1, max_dom + 1))
    ratio = _int_col(dm, "parent_grid_ratio")
    tratio = _int_col(dm, "parent_time_step_ratio")
    i_start = _int_col(dm, "i_parent_start")
    j_start = _int_col(dm, "j_parent_start")
    e_we = _int_col(dm, "e_we")
    e_sn = _int_col(dm, "e_sn")
    e_vert = _int_col(dm, "e_vert")

    expected_grid_ids = list(range(1, max_dom + 1))
    if grid_ids != expected_grid_ids:
        raise ValueError(
            "WRF hierarchy grid_id values must be contiguous and listed "
            f"parent-before-child: expected {expected_grid_ids}, got "
            f"{grid_ids}.")
    if parent_id[0] != 0:
        raise ValueError(
            f"d01 must declare parent_id = 0, got {parent_id[0]}.")
    declared_parent_ids = {grid_ids[0]}
    for grid_id, declared_parent in zip(grid_ids[1:], parent_id[1:]):
        if declared_parent not in declared_parent_ids:
            raise ValueError(
                f"parent_id = {declared_parent} of grid_id = {grid_id} "
                "does not name a previously declared domain (orphan and "
                "cyclic hierarchies are rejected before geometry "
                "derivation).")
        declared_parent_ids.add(grid_id)

    wps_root = {
        "parent_id": geo_parent_id[0],
        "parent_grid_ratio": geo_ratio[0],
        "i_parent_start": geo_i[0],
        "j_parent_start": geo_j[0],
    }
    expected_wps_root = {
        "parent_id": 1,
        "parent_grid_ratio": 1,
        "i_parent_start": 1,
        "j_parent_start": 1,
    }
    if wps_root != expected_wps_root:
        raise ValueError(
            "WPS d01 root topology must use parent_id=1, "
            "parent_grid_ratio=1, i_parent_start=1, j_parent_start=1; "
            f"got {wps_root}.")

    # WPS lists d01's parent as itself (parent_id = 1); namelist.input
    # uses 0 for the head grid -- compare children only.
    for key, wps_col, inp_col in (
            ("parent_id", geo_parent_id[1:], parent_id[1:]),
            ("parent_grid_ratio", geo_ratio[1:], ratio[1:]),
            ("i_parent_start", geo_i, i_start),
            ("j_parent_start", geo_j, j_start),
            ("e_we", geo_e_we, e_we), ("e_sn", geo_e_sn, e_sn)):
        if wps_col != inp_col:
            raise ValueError(
                f"nest layout mismatch for {key}: {wps_path} says "
                f"{wps_col} but {input_path} says {inp_col}.")

    wps_dx = float(geo.required("dx"))
    wps_dy = float(geo.scalar("dy", wps_dx))
    dm_dx = dm.col("dx", max_dom)
    dm_dy = dm.col("dy", max_dom)
    root_dx = float(dm_dx[0]) if dm_dx is not None else wps_dx
    root_dy = float(dm_dy[0]) if dm_dy is not None else wps_dy
    if abs(wps_dx - root_dx) > _REL_TOL * abs(root_dx) \
            or abs(wps_dy - root_dy) > _REL_TOL * abs(root_dy):
        raise ValueError(
            f"d01 dx/dy mismatch: {wps_path} says {wps_dx}/{wps_dy} but "
            f"{input_path} says {root_dx}/{root_dy}.")
    if root_dx != root_dy:
        raise ValueError(
            f"d01 dx = {root_dx} must equal dy = {root_dy} (conformal "
            "projected grids are isotropic).")

    # Child dx/dy: cross-check the hand-typed decimals against the exact
    # chain, then DROP them -- the resolved TOML derives child dx at load.
    dx_exact = [Fraction(root_dx)]
    for n in range(1, max_dom):
        parent_index = grid_ids.index(parent_id[n])
        dx_exact.append(dx_exact[parent_index] / ratio[n])
        for key, col in (("dx", dm_dx), ("dy", dm_dy)):
            if col is None:
                continue
            supplied = float(col[n])
            derived = float(dx_exact[n])
            if abs(supplied - derived) > _REL_TOL * abs(derived):
                raise ValueError(
                    f"&domains {key}[{n + 1}] = {supplied!r} in "
                    f"{input_path} contradicts the parent chain: "
                    f"d0{grid_ids[n]} derives {key} = {dx_exact[n]} m "
                    f"(parent {dx_exact[parent_index]} m / ratio "
                    f"{ratio[n]}).  The namelist chain is authoritative "
                    "-- child dx is never hand-typed.")
    if max_dom > 1:
        for key, col in (("dx (children)", dm_dx), ("dy (children)",
                                                    dm_dy)):
            drop("domains", key,
                 [float(v) for v in col[1:]] if col else None,
                 "child dx/dy derive exactly from the parent chain at "
                 "load; the hand-typed namelist decimals are "
                 "cross-checked, not copied")

    # ---- &domains: clock, vertical grid, guards --------------------------
    time_step = int(dm.required("time_step"))
    fract_num = int(dm.scalar("time_step_fract_num", 0))
    fract_den = int(dm.scalar("time_step_fract_den", 1))
    nz = _uniform("domains", "e_vert", e_vert) - 1
    eta_levels = [float(v) for v in (dm.take("eta_levels") or [])]
    # Omitted namelist keys take their WRF v4.6.1 Registry defaults
    # (review F2 -- silently substituting gpuwm-convenient values is a
    # never-silent violation): p_top_requested 5000 Pa
    # (Registry.EM_COMMON:2275), feedback 1 (:2322), smooth_option 2
    # (:2323) -- an implicitly two-way namelist therefore selects the
    # experimental runtime path instead of importing as one-way.
    p_top = float(dm.scalar("p_top_requested", 5000))
    # hypsometric_opt is a &domains SCALAR, not a &dynamics column
    # (Registry.EM_COMMON:2283, `namelist,domains`, nentries 1;
    # run/README.namelist documents it inside &domains).  gpuwm read it
    # from &dynamics, which is a section no real WRF namelist can carry
    # it in -- wrf.exe fails the &dynamics namelist read outright -- so
    # the importer could only ever have accepted namelists gpuwm itself
    # wrote.  Ratified Registry-default binding when the namelist leaves
    # it unset (Phase-4 native-dt baseline): 2.
    hypsometric_opt = int(dm.scalar("hypsometric_opt", 2))

    _NEST_GUARD_WHY = {
        "interp_method_type":
            "only SINT (2, the WRF default per "
            "Registry.EM_COMMON:2301) is implemented.",
        "nest_interp_coord":
            "isobaric nest re-interpolation is not implemented.",
        "vert_refine_method":
            "vertical nest refinement is rejected (identical "
            "e_vert/eta_levels/p_top on all domains).",
        "input_from_hires":
            "high-resolution child terrain input is rejected "
            "(children SINT + blend the parent terrain).",
        "use_adaptive_time_step":
            "the adaptive time step is not supported (gpuwm "
            "integrates on the fixed namelist clock).",
        "smooth_cg_topo":
            "coarse-grid topography smoothing is not "
            "implemented.",
    }
    for key, ok in (("interp_method_type", 2), ("nest_interp_coord", 0),
                    ("vert_refine_method", 0), ("input_from_hires", False),
                    ("use_adaptive_time_step", False),
                    ("smooth_cg_topo", False)):
        raw = dm.take(key)
        value = ok if raw is None else raw[0]
        if value != ok:
            raise _err("domains", key, value, _NEST_GUARD_WHY[key])
        fix("domains", key, raw, ok, _NEST_GUARD_WHY[key].rstrip("."))

    # WRF process/tile decomposition layout: gpuwm's GPU decomposition is
    # internal, so these carry no science and drop with a reason.
    for key in ("numtiles", "nproc_x", "nproc_y"):
        drop("domains", key, dm.take(key),
             "WRF parallel tile/process decomposition layout; gpuwm's "
             "GPU decomposition is internal")

    # WRF automatic eta-level generation (real.exe) is not implemented:
    # with explicit eta_levels the generator is bypassed in WRF too, so
    # its tuning keys are inert and drop; without eta_levels they would
    # select level generation gpuwm cannot perform -- hard error.
    for key in ("auto_levels_opt", "max_dz", "dzbot", "dzstretch_s",
                "dzstretch_u"):
        raw = dm.take(key)
        if raw is None:
            continue
        if not eta_levels:
            raise _err(
                "domains", key, raw,
                "WRF's automatic eta-level generation is not implemented; "
                "supply explicit eta_levels (the generator's tuning keys "
                "are inert once eta_levels is explicit).")
        drop("domains", key, raw,
             "inert: explicit eta_levels bypass WRF's automatic level "
             "generation, which gpuwm does not implement")

    feedback = int(dm.scalar("feedback", 1))
    smooth_option = int(dm.scalar("smooth_option", 2))
    blend_width = int(dm.scalar("blend_width", 5))
    for key, reason in (
            ("num_metgrid_levels", "forcing level count is derived from "
                                   "the input catalog at ingest"),
            ("num_metgrid_soil_levels", "soil level count is derived "
                                        "from the input catalog"),
            ("sfcp_to_sfcp", "interpolation policy key declared in "
                             "[case_data], not imported"),
    ):
        drop("domains", key, dm.take(key), reason)

    # ---- the &domains half of the mp=28 aerosol sweep --------------------
    # It has to run HERE, before dm.finish(), and the &physics half runs
    # with the rest of the physics block far below.  WRF splits WIF's two
    # keys into &domains (Registry/registry.new3d_wif:16-17) while every
    # other aerosol knob is in &physics, and dm.finish() -- which refuses
    # any unconsumed key -- fires long before mp_physics has been mapped.
    # So mp_physics is PEEKED off the unconsumed &physics entries rather
    # than read from the mapped value; peeking cannot consume, so the
    # physics block below still validates mp_physics itself.
    # Any domain asking for 28 is enough to refuse: a mixed column is
    # rejected later by _uniform anyway, and treating the root's value as
    # the whole answer would let `mp_physics = 6, 28` drop a WIF key as
    # inert on its way to that rejection.
    _peeked_mp = ph.entries.get("mp_physics") or ()
    _selects_28 = any(
        isinstance(value, int) and not isinstance(value, bool)
        and value == 28 for value in _peeked_mp)
    for _aero_key, _why in _MP28_AEROSOL_NAMELIST_KEYS["domains"].items():
        _values = dm.take(_aero_key)
        if _values is None:
            continue
        if not _selects_28:
            drop("domains", _aero_key, _values,
                 "inert: WRF consumes this only inside the thompsonaero "
                 "package (Registry/Registry.EM_COMMON:3036, "
                 "mp_physics = 28)")
            continue
        raise _err("domains", _aero_key, _values, _why + ".")

    dm.finish()

    # ---- &geogrid projection ---------------------------------------------
    map_proj = str(geo.scalar("map_proj", "lambert")).lower()
    if map_proj not in ("lambert", "mercator", "polar"):
        blocker = (
            " Regular/rotated latitude-longitude needs angular dx/dy "
            "rather than metre spacing and WRF's global/pole polar filter; "
            "rotated grids also need pole_lat/pole_lon state and the "
            "map_proj == 6 curvature branch."
            if map_proj.replace("_", "-") in {
                "lat-lon", "latlon", "regular-ll", "rotated-lat-lon",
                "rotated-ll",
            } else "")
        raise _err("geogrid", "map_proj", map_proj,
                   "implemented projections: 'lambert', 'mercator', "
                   f"'polar'.{blocker}")
    ref_lat = float(geo.required("ref_lat"))
    ref_lon = float(geo.required("ref_lon"))
    truelat1 = float(geo.required("truelat1"))
    if map_proj == "lambert":
        truelat2 = float(geo.required("truelat2"))
        stand_lon = float(geo.required("stand_lon"))
    else:
        # WPS semantics: Mercator ignores truelat2/stand_lon and polar
        # stereographic ignores truelat2; a namelist may omit them.  The
        # emitted [projection] table always carries all six keys.
        truelat2 = float(geo.scalar("truelat2", truelat1))
        stand_lon = float(geo.scalar("stand_lon", ref_lon))
    projection = {
        "map_proj": map_proj,
        "ref_lat": ref_lat,
        "ref_lon": ref_lon,
        "truelat1": truelat1,
        "truelat2": truelat2,
        "stand_lon": stand_lon,
    }
    s_we = geo.col("s_we", max_dom, default=1)
    s_sn = geo.col("s_sn", max_dom, default=1)
    if any(int(v) != 1 for v in s_we + s_sn):
        raise _err("geogrid", "s_we/s_sn", [s_we, s_sn],
                   "sub-window starts other than 1 are not supported.")
    for key in ("ref_x", "ref_y"):
        if geo.take(key) is not None:
            raise _err("geogrid", key, "set",
                       "only the WPS default reference point "
                       "(e_we/2, e_sn/2) is supported.")
    for key, reason in (
            ("geog_data_res", "GEOG dataset/resolution selection is "
                              "static-build configuration, not imported"),
            ("geog_data_path", "geography root is declared in "
                               "[case_data] (or --geog-root), not "
                               "imported"),
            ("opt_geogrid_tbl_path", "WPS table path"),
    ):
        drop("geogrid", key, geo.take(key), reason)
    geo.finish()

    # ---- &physics: ratified scheme maps ----------------------------------
    # Inspect the complete requested suite before consuming individual keys.
    # This keeps unfinished ports fail-closed while reporting Thompson, the
    # coupled MYNN stack, and RUC together instead of stopping at the first
    # unsupported selector.  ``peek`` follows WRF's repeat-last convention but
    # does not mutate _Section, so ordinary parsing remains authoritative.
    def _peek_uniform_int(key: str, default=None) -> int | None:
        values = ph.entries.get(key)
        if values is None:
            return default
        values = list(values[:max_dom])
        values += [values[-1]] * (max_dom - len(values))
        return int(_uniform("physics", key, values))

    def _peek_col_int(key: str) -> list | None:
        """The padded per-domain column, WRF's repeat-last convention."""
        values = ph.entries.get(key)
        if values is None:
            return None
        values = list(values[:max_dom])
        values += [values[-1]] * (max_dom - len(values))
        return [int(v) for v in values]

    # bl_pbl_physics is a PER-DOMAIN column (see the _mapped call below):
    # gpuwm.experiment._DOMAIN_RUN_OVERRIDES admits it per domain so a PBL
    # parent can carry a PBL-off LES child, and the importer has to be able
    # to read back what the experiment schema can express.  The readiness
    # preview runs once per DISTINCT value so every domain's suite is
    # refused on exactly the grounds it was refused on before.
    bl_preview_col = _peek_col_int("bl_pbl_physics")
    preview = {
        "mp_physics": _peek_uniform_int("mp_physics"),
        "sf_sfclay_physics": _peek_uniform_int("sf_sfclay_physics"),
        "bl_pbl_physics": (None if bl_preview_col is None
                           else bl_preview_col[0]),
        "sf_surface_physics": _peek_uniform_int("sf_surface_physics", 0),
        # Registry.EM_COMMON:2504 defaults to 5.  The target suite supplies
        # 9 explicitly; retaining the real default here avoids inventing a
        # RUC geometry in diagnostics.
        "num_soil_layers": _peek_uniform_int("num_soil_layers", 5),
    }
    if all(preview[key] is not None for key in (
            "mp_physics", "sf_sfclay_physics", "bl_pbl_physics")):
        for _bl in dict.fromkeys(bl_preview_col):
            require_ready_wrf_physics(**{**preview, "bl_pbl_physics": _bl})

    def _mapped(key: str, table: dict, per_domain=False):
        values = ph.col(key, max_dom)
        if values is None:
            raise ValueError(
                f"{input_path} &physics is missing {key}.")
        values = [int(v) for v in values]
        if not per_domain:
            value = _uniform("physics", key, values)
            values = [value]
        out = []
        for value in values:
            if value not in table:
                raise _err("physics", key, value,
                           f"no ratified gpuwm mapping (implemented: "
                           f"{sorted(table)}).")
            out.append(table[value])
        return values, out

    mp_wrf, mp_mapped = _mapped("mp_physics", _MP_MAP)
    mp_physics, wrf_name, gp_name = mp_mapped[0]
    if mp_physics != mp_wrf[0]:
        substitutions.append(Substitution(
            key="mp_physics", wrf_value=mp_wrf[0], wrf_name=wrf_name,
            gpuwm_key="mp_physics", gpuwm_value=mp_physics,
            gpuwm_name=gp_name))
    # Scalar WRF option, Registry.EM_COMMON:2663-2666: default 1 = hail,
    # explicit 0 = graupel.  It selects AG/BG/RHOG before Morrison derives
    # CG and its gamma/exponent products (module_mp_morr_two_moment.F:
    # 337-411, :483-510).
    morr_rimed_ice = int(ph.scalar("morr_rimed_ice", 1))
    if morr_rimed_ice not in (0, 1):
        raise _err("physics", "morr_rimed_ice", morr_rimed_ice,
                   "must be 0 (graupel) or 1 (hail, WRF default).")
    # WRF Registry.EM_COMMON:2665, scalar default 0.  This is a distinct
    # WSM6 switch; do not alias Morrison's opposite-default selection.
    wsm6_hail_opt = int(ph.scalar("hail_opt", 0))
    if wsm6_hail_opt not in (0, 1):
        raise _err("physics", "hail_opt", wsm6_hail_opt,
                   "must be 0 (graupel, WRF WSM6 default) or 1 (hail).")

    # ---- RunConfig-honored &physics knobs (knob-parity lane) -----------
    # Each maps 1:1 onto an existing consumed RunConfig field and is
    # emitted only when the namelist supplies it: every WRF Registry
    # default below equals gpuwm's frozen RunConfig default, so omission
    # resolves identically on both sides and established imports stay
    # byte-identical.
    def _optional_scalar_int(key: str, allowed: tuple, why: str):
        values = ph.take(key)
        if values is None:
            return None
        value = values[0]
        if isinstance(value, bool) or not isinstance(value, int) \
                or value not in allowed:
            raise _err("physics", key, values, why)
        return value

    def _optional_scalar_bool(key: str):
        values = ph.take(key)
        if values is None:
            return None
        return _require_bools("physics", key, values)[0]

    no_mp_heating = _optional_scalar_int(
        "no_mp_heating", (0, 1),
        "must be 0 (microphysics latent heating on, the WRF default) or "
        "1 (heating off, module_big_step_utilities_em.F:5770-5782).")
    mp_tend_lim_values = ph.take("mp_tend_lim")
    mp_tend_lim = None
    if mp_tend_lim_values is not None:
        mp_tend_lim = float(mp_tend_lim_values[0])
        if not math.isfinite(mp_tend_lim) or mp_tend_lim <= 0.0:
            raise _err("physics", "mp_tend_lim", mp_tend_lim_values,
                       "must be a finite positive microphysics theta-"
                       "tendency clamp in K/s (WRF Registry default 10.0).")
    ysu_topdown_pblmix = _optional_scalar_int(
        "ysu_topdown_pblmix", (0, 1),
        "must be 0 or 1 (top-down radiation-driven PBL mixing).")
    isfflx = _optional_scalar_int(
        "isfflx", (0, 1),
        "must be 0 (surface heat/moisture fluxes off) or 1 (on); WRF "
        "value 2 also needs gpuwm's unported prescribed-flux/diffusion "
        "forcing path.")
    if isfflx == 1:
        fix(
            "physics", "isfflx", [1], 1,
            "the established surface heat/moisture-flux-on value")
        isfflx = None
    use_mp_re = _optional_scalar_int(
        "use_mp_re", (0, 1),
        "must be 0 (legacy-RRTMG calculated radii) or 1 (use the WRF "
        "microphysics scheme table).")
    if use_mp_re == 1:
        fix(
            "physics", "use_mp_re", [1], 1,
            "the established WRF microphysics effective-radius scheme-table "
            "value")
        use_mp_re = None
    isftcflx = _optional_scalar_int(
        "isftcflx", (0, 1, 2),
        "must be 0 (standard MM5 water-point roughness), 1 (Garratt), or "
        "2 (Donelan).")
    iz0tlnd = _optional_scalar_int(
        "iz0tlnd", (0, 1, 2),
        "must be 0 (standard CZIL), 1 (Chen-Zhang), or 2 (fixed CZIL "
        "0.1).")
    usemonalb = _optional_scalar_bool("usemonalb")
    rdlai2d = _optional_scalar_bool("rdlai2d")
    rdmaxalb = _optional_scalar_bool("rdmaxalb")
    seaice_albedo_default_values = ph.take("seaice_albedo_default")
    seaice_albedo_default = None
    if seaice_albedo_default_values is not None:
        seaice_albedo_default = float(seaice_albedo_default_values[0])
        if (not math.isfinite(seaice_albedo_default)
                or not 0.0 <= seaice_albedo_default <= 1.0):
            raise _err(
                "physics", "seaice_albedo_default",
                seaice_albedo_default_values,
                "must be a finite albedo fraction in [0, 1].")
    opt_thcnd = _optional_scalar_int(
        "opt_thcnd", (1, 2),
        "must be 1 (Johansen soil thermal conductivity) or 2 "
        "(McCumber-Pielke).")

    # ---- &physics keys the model pins where WRF has options ------------
    # Present keys are validated against the single implemented value and
    # recorded as fixed; any other value is a hard error, never a silent
    # reinterpretation.
    for key, pin, why in (
            ("swint_opt", 0,
             "shortwave interpolation between radt calls is not "
             "implemented; radiation is recomputed on the radt cadence"),
            ("gwd_opt", 0, "no gravity-wave-drag scheme is implemented"),
            ("sf_lake_physics", 0, "no lake model is implemented"),
            ("shcu_physics", 0,
             "no shallow-cumulus scheme is implemented"),
            ("topo_shading", 0,
             "terrain shadowing of shortwave is not implemented"),
            ("slope_rad", 0,
             "slope-dependent radiation geometry is not implemented"),
            ("kf_edrates", 0,
             "KF entrainment/detrainment rate diagnostics are not "
             "implemented"),
            ("flag_sm_adj", 0,
             "RUC option identity: gpuwm has no RUC soil ingest for the "
             "adjustment to act on (share/module_soil_pre.F:2063)"),
            ("sst_update", 0,
             "SST update cycling is not implemented (single-analysis "
             "case runs)"),
            ("sst_skin", 0,
             "skin-SST diurnal adjustment is not implemented"),
            ("tmn_update", 0,
             "deep-soil temperature update is not implemented"),
    ):
        raw = ph.take(key)
        if raw is None:
            continue
        if any(isinstance(value, bool) or not isinstance(value, int)
               or value != pin for value in raw):
            raise _err("physics", key, raw,
                       f"gpuwm implements {key} = {pin} only ({why}).")
        fix("physics", key, raw, pin, why)
    cu_rad_feedback = ph.take("cu_rad_feedback")
    if cu_rad_feedback is not None:
        if any(_require_bools("physics", "cu_rad_feedback",
                              cu_rad_feedback)):
            raise _err("physics", "cu_rad_feedback", cu_rad_feedback,
                       "cumulus cloud fraction does not feed radiation in "
                       "gpuwm (WRF Registry default .false.); only the "
                       "disabled branch is implemented.")
        fix("physics", "cu_rad_feedback", cu_rad_feedback, False,
            "cumulus-radiation feedback is not implemented (the WRF "
            "Registry default .false. branch is the implemented one)")

    # ---- MYNN option identity ------------------------------------------
    # gpuwm's MYNN port is validated at exactly one value of each option
    # (gpuwm/config.py MYNN_PBL_OPTION_IDENTITY); a present key at the
    # identity value is fixed, anything else refuses.
    for option, admitted in MYNN_PBL_OPTION_IDENTITY.items():
        values = ph.take(option)
        if values is None:
            continue
        if any(not _identity_matches(value, admitted) for value in values):
            raise _err(
                "physics", option, values,
                f"gpuwm implements {option} = {admitted!r} only (MYNN "
                "option identity, gpuwm/config.py); no nearby branch is "
                "substituted for an unported one.")
        fix("physics", option, values, admitted,
            "MYNN option identity: the ported solver is validated at "
            "exactly this value")

    # ---- NSSL parameters -------------------------------------------------
    # The mp18 port runs at the WRF v4.6.1 Registry defaults pinned by
    # gpuwm/core/nssl2_contract.py; tunable NSSL parameters are not yet
    # plumbed.  Under any other scheme the keys are inert in WRF too.
    nssl_keys = sorted(key for key in ph.entries if key.startswith("nssl_"))
    if nssl_keys:
        from gpuwm.core.nssl2_contract import (
            CONTRACT_ID as _NSSL_CONTRACT_ID,
            WRF_NAMELIST_DEFAULTS as _NSSL_DEFAULTS,
        )
        for key in nssl_keys:
            values = ph.take(key)
            if key not in _NSSL_DEFAULTS:
                raise _err(
                    "physics", key, values,
                    "no ratified NSSL parameter mapping (extend "
                    "gpuwm/core/nssl2_contract.py first).")
            if mp_physics != 18:
                drop("physics", key, values,
                     "inert: NSSL parameters are consumed only under "
                     "mp_physics = 18")
                continue
            admitted = _NSSL_DEFAULTS[key]
            if any(not _identity_matches(value, admitted)
                   for value in values):
                raise _err(
                    "physics", key, values,
                    f"the NSSL two-moment port runs at the WRF v4.6.1 "
                    f"Registry default {key} = {admitted!r} only "
                    f"(contract {_NSSL_CONTRACT_ID}); tunable NSSL "
                    "parameters are not plumbed.")
            fix("physics", key, values, admitted,
                f"NSSL parameter pinned at its Registry default by "
                f"contract {_NSSL_CONTRACT_ID}")

    # ---- aerosol-aware Thompson keys ------------------------------------
    # Same shape as the NSSL sweep above and for the same structural
    # reason: _Section.finish() refuses any unconsumed key, so every one of
    # WRF's mp=28 aerosol knobs must be answered here or an otherwise valid
    # namelist becomes unimportable.  Inert under any other scheme; refused
    # by name under 28.  See _MP28_AEROSOL_NAMELIST_KEYS for the per-key
    # citation.
    # The &domains half already ran above, before dm.finish().
    for _aero_key, _why in _MP28_AEROSOL_NAMELIST_KEYS["physics"].items():
        _values = ph.take(_aero_key)
        if _values is None:
            continue
        if mp_physics != 28:
            drop("physics", _aero_key, _values,
                 "inert: WRF consumes this only inside the "
                 "thompsonaero package "
                 "(Registry/Registry.EM_COMMON:3036, mp_physics = 28)")
            continue
        raise _err("physics", _aero_key, _values, _why + ".")
    if mp_physics == 28:
        # The never-silent last mile.  A user who imports an mp=28 namelist
        # gets a printed line saying exactly which aerosol initial state
        # their run will use and which WRF line refuses the same
        # configuration -- rather than discovering months later that their
        # ArWen and WRF runs were never comparable.
        from gpuwm.config import MP28_AEROSOL_SOURCE_DEVIATION
        defaults_applied.append(AppliedDefault(
            key="mp28 aerosol initial state",
            value="thompson_init synthetic CCN/IN profile",
            reason=MP28_AEROSOL_SOURCE_DEVIATION))

    bl_wrf, bl_mapped = _mapped("bl_pbl_physics", _BL_MAP, per_domain=True)
    bl_pbl_col = [entry[0] for entry in bl_mapped]
    bl_pbl_physics, wrf_name, gp_name = bl_mapped[0]
    if bl_pbl_physics != bl_wrf[0]:
        substitutions.append(Substitution(
            key="bl_pbl_physics", wrf_value=bl_wrf[0], wrf_name=wrf_name,
            gpuwm_key="bl_pbl_physics", gpuwm_value=bl_pbl_physics,
            gpuwm_name=gp_name))
    lw_wrf, lw_mapped = _mapped("ra_lw_physics", _RA_LW_MAP)
    sw_wrf, sw_mapped = _mapped("ra_sw_physics", _RA_SW_MAP)
    ra_lw_physics, lw_wrf_name, lw_gp_name = lw_mapped[0]
    ra_sw_physics, sw_wrf_name, sw_gp_name = sw_mapped[0]
    # Preserve the frozen aggregate representation for the already shipped
    # coupled RTE+RRTMGP configurations.  Every other WRF pair is emitted in
    # the native split schema, including RRTM LW + Dudhia SW (1/1).
    coupled_legacy = (ra_lw_physics == ra_sw_physics
                      and ra_lw_physics in (0, 4))
    ra_physics = ra_lw_physics if coupled_legacy else 0
    legacy_rrtmg = rrtmg_variant == RRTMG_VARIANT_LEGACY
    if legacy_rrtmg and (ra_lw_physics, ra_sw_physics) != (4, 4):
        raise ValueError(
            f"rrtmg_variant='{RRTMG_VARIANT_LEGACY}' requires the WRF "
            "namelist to request ra_lw_physics = ra_sw_physics = 4 "
            f"(RRTMG); got {lw_wrf[0]}/{sw_wrf[0]}")
    if use_mp_re == 0 and not legacy_rrtmg:
        raise _err(
            "physics", "use_mp_re", [use_mp_re],
            "use_mp_re=0 is implemented only by the exact legacy-RRTMG "
            "wrapper; the selected radiation adapter has no equivalent "
            "calculated-radius branch.")
    o3input = None
    if (ra_lw_physics, ra_sw_physics) == (4, 4):
        wrf_rrtmg_compatibility = (
            WRF_RRTMG_LEGACY if legacy_rrtmg else WRF_RRTMG_TO_RTE_RRTMGP)
    else:
        wrf_rrtmg_compatibility = "none"
    adapter_44 = ("WRF legacy RRTMG" if legacy_rrtmg else lw_gp_name)
    if (ra_lw_physics, ra_sw_physics) == (4, 4):
        substitutions.append(Substitution(
            key="ra_lw_physics/ra_sw_physics", wrf_value=lw_wrf[0],
            wrf_name=lw_wrf_name, gpuwm_key="ra_physics",
            gpuwm_value=ra_physics, gpuwm_name=adapter_44))
    elif ra_lw_physics != lw_wrf[0]:
        substitutions.append(Substitution(
            key="ra_lw_physics", wrf_value=lw_wrf[0],
            wrf_name=lw_wrf_name, gpuwm_key="ra_lw_physics",
            gpuwm_value=ra_lw_physics, gpuwm_name=lw_gp_name))
    if (ra_lw_physics, ra_sw_physics) != (4, 4) \
            and ra_sw_physics != sw_wrf[0]:
        substitutions.append(Substitution(
            key="ra_sw_physics", wrf_value=sw_wrf[0],
            wrf_name=sw_wrf_name, gpuwm_key="ra_sw_physics",
            gpuwm_value=ra_sw_physics, gpuwm_name=sw_gp_name))
    if ra_sw_physics == 1:
        icloud = int(ph.scalar("icloud", 1))
        swrad_scat = float(ph.scalar("swrad_scat", 1.0))
        if icloud not in (0, 1):
            raise _err("physics", "icloud", icloud, "must be 0 or 1.")
        if not math.isfinite(swrad_scat) or swrad_scat < 0.0:
            raise _err("physics", "swrad_scat", swrad_scat,
                       "must be finite and non-negative.")
    else:
        icloud = 1
        swrad_scat = 1.0
        raw_icloud = ph.take("icloud")
        adapter_label = ("legacy RRTMG"
                         if (ra_lw_physics, ra_sw_physics) == (4, 4)
                         and legacy_rrtmg else "RTE+RRTMGP")
        if raw_icloud is not None and any(int(value) != 1
                                         for value in raw_icloud):
            raise _err(
                "physics", "icloud", raw_icloud,
                f"gpuwm's {adapter_label} adapter currently has "
                "cloud-radiation coupling always on; icloud=0 cannot be "
                "silently ignored.")
        fix("physics", "icloud", raw_icloud, 1,
            f"cloud-radiation coupling is fixed on in the {adapter_label} "
            "driver")
        drop("physics", "swrad_scat", ph.take("swrad_scat"),
             "Dudhia shortwave is not selected")

    if (ra_lw_physics, ra_sw_physics) == (4, 4):
        # RRTMG-family radiation options, ratified with fail-closed ranges.
        # Both 4/4 adapters implement McICA maximum-random overlap
        # (cldovrlp=2), constant decorrelation (idcor=0), analytic
        # year-dependent well-mixed gases (ghg_input=0), and no aerosol
        # (aer_opt=0).  Legacy RRTMG additionally implements the wrapper's
        # O3DATA branch (o3input=0); both variants retain CAM climatology
        # (o3input=2).  Any other request would be a silent WRF semantic
        # change, so an explicit other value refuses instead of importing.
        # Absent keys keep the established import result byte-identical
        # (the shipped RTE+RRTMGP configurations must not change).  NOTE
        # ghg_input's WRF Registry default is 1 (time-varying GHG from a
        # CAMtr file when one is present in the run directory); the
        # campaign CPU references pin ghg_input = 0 explicitly, and a
        # mirrored run that really consumed a CAMtr file cannot be
        # reproduced here -- that limit is documented rather than guessed
        # at from an absent key.
        adapter_label = ("legacy RRTMG" if legacy_rrtmg else "RTE+RRTMGP")
        raw_o3input = ph.take("o3input")
        if raw_o3input is not None:
            value = int(_uniform("physics", "o3input", raw_o3input))
            admitted = (0, 2) if legacy_rrtmg else (2,)
            if value not in admitted:
                raise _err(
                    "physics", "o3input", raw_o3input,
                    f"gpuwm's {adapter_label} adapter implements "
                    f"o3input values {admitted}; o3input=0 belongs to the "
                    "legacy wrapper's O3DATA branch.")
            if value == 2:
                fix(
                    "physics", "o3input", raw_o3input, 2,
                    f"the established CAM-climatology value in the "
                    f"{adapter_label} adapter")
            else:
                o3input = value
        for key, supported, why in (
                ("cldovrlp", 2,
                 "McICA maximum-random overlap is the only implemented "
                 "subcolumn walk"),
                ("idcor", 0,
                 "constant 2500 m decorrelation is the only implemented "
                 "choice (inert at cldovrlp=2)"),
                ("ghg_input", 0,
                 "the analytic year-dependent well-mixed gas formulas are "
                 "the only implemented GHG source (no CAMtr file reader)"),
                ("aer_opt", 0,
                 "radiation aerosol input is not implemented")):
            raw = ph.take(key)
            if raw is None:
                continue
            if any(int(value) != supported for value in raw):
                raise _err(
                    "physics", key, raw,
                    f"gpuwm's {adapter_label} adapter only implements "
                    f"{key} = {supported} ({why}).")
            fix("physics", key, raw, supported,
                f"the only implemented value in the {adapter_label} "
                f"adapter ({why})")

    sfclay_values = ph.col("sf_sfclay_physics", max_dom)
    if sfclay_values is None:
        raise _err(
            "physics", "sf_sfclay_physics", None,
            "must-set namelist key (WRF Registry default -1 is rejected "
            "by the surface-layer driver); gpuwm will not silently "
            "disable the scheme.")
    sfclay = int(_uniform(
        "physics", "sf_sfclay_physics", sfclay_values))
    if sfclay not in _SFCLAY_ALLOWED:
        raise _err("physics", "sf_sfclay_physics", sfclay,
                   f"no gpuwm mapping (implemented: "
                   f"{sorted(_SFCLAY_ALLOWED)})."
                   + _port_receipt(sf_sfclay_physics=sfclay))
    sfsfc = int(_uniform("physics", "sf_surface_physics",
                         ph.col("sf_surface_physics", max_dom, 0)))
    if sfsfc not in _SFSFC_ALLOWED:
        raise _err("physics", "sf_surface_physics", sfsfc,
                   f"no gpuwm mapping (implemented: "
                   f"{sorted(_SFSFC_ALLOWED)})."
                   + _port_receipt(sf_surface_physics=sfsfc))
    if (seaice_albedo_default is not None
            and seaice_albedo_default != 0.65 and sfsfc != 3):
        raise _err(
            "physics", "seaice_albedo_default",
            [seaice_albedo_default],
            "a nondefault value is implemented only by RUC LSM "
            "(sf_surface_physics=3).")
    if rdmaxalb is False and sfsfc != 2:
        raise _err(
            "physics", "rdmaxalb", [rdmaxalb],
            "rdmaxalb=false is implemented by Noah LSMINIT and requires "
            "sf_surface_physics=2.")
    if isfflx == 0 and sfclay == 0:
        raise _err(
            "physics", "isfflx", [isfflx],
            "isfflx=0 has no consumer when sf_sfclay_physics=0; gpuwm "
            "implements the gate in its MM5 and MYNN surface layers.")
    from gpuwm.config import validated_soil_layer_count

    resolved_soil_layers = validated_soil_layer_count(sfsfc)
    requested_soil_layers = ph.take("num_soil_layers")
    if requested_soil_layers is not None and any(
            int(value) != resolved_soil_layers
            for value in requested_soil_layers):
        raise _err(
            "physics", "num_soil_layers", requested_soil_layers,
            f"must be {resolved_soil_layers} for sf_surface_physics="
            f"{sfsfc}; gpuwm does not silently overwrite a requested soil "
            "geometry")
    fix(
        "physics", "num_soil_layers", requested_soil_layers,
        resolved_soil_layers,
        f"validated soil geometry for sf_surface_physics={sfsfc}")
    cu_values = ph.col("cu_physics", max_dom)
    if cu_values is None:
        raise _err(
            "physics", "cu_physics", None,
            "must-set namelist key (WRF Registry default -1 is rejected "
            "by the cumulus driver); gpuwm will not silently disable the "
            "scheme.")
    cu = [int(v) for v in cu_values]
    for value in cu:
        if value not in _CU_ALLOWED:
            raise _err("physics", "cu_physics", value,
                       f"no gpuwm mapping (implemented: "
                       f"{sorted(_CU_ALLOWED)}).")
    cudt = [float(v) for v in ph.col("cudt", max_dom, 0)]
    # The two Grell-family keys, WRF v4.6.1 Registry defaults (both 0,
    # single-instance).  Read whenever present; consumed only where
    # cu_physics = 3, exactly as WRF's cumulus driver reads them only for
    # the Grell schemes.
    clos_choice = int(_uniform(
        "physics", "clos_choice", ph.col("clos_choice", max_dom, 0)))
    ishallow = int(_uniform(
        "physics", "ishallow", ph.col("ishallow", max_dom, 0)))
    if clos_choice != 0:
        raise _err(
            "physics", "clos_choice", [clos_choice],
            "only the 16-member ensemble closure (0, the Registry "
            "default) is admitted: the single-closure arms carry no GF "
            "oracle coverage.")
    if ishallow not in (0, 1):
        raise _err("physics", "ishallow", [ishallow],
                   "must be 0 or 1 (CUP_gf_sh off/on).")
    if (clos_choice or ishallow) and all(value != 3 for value in cu):
        drop("physics", "clos_choice/ishallow",
             [clos_choice, ishallow],
             "Grell-family keys with no Grell scheme selected "
             "(cu_physics has no 3); WRF's cumulus driver would not read "
             "them either")
        clos_choice = 0
        ishallow = 0
    radt = [float(v) for v in ph.col("radt", max_dom, 0)]
    bldt = float(_uniform("physics", "bldt", ph.col("bldt", max_dom, 0)))
    for key, reason in (
            ("ifsnow", "Noah snow physics is always active"),
            ("surface_input_source", "surface fields come from the "
                                     "ingest catalog"),
            ("do_radar_ref", "REFL_10CM is a gpuwm output product "
                             "(evaluated at output time, PROVENANCE D2)"),
    ):
        drop("physics", key, ph.take(key), reason)
    # ---- land-use identity keys ----------------------------------------
    # Both are ordinary keys in a WRF-Runner/WPS-generated namelist and both
    # used to reach ph.finish() unmapped, which refused the whole import
    # with "unmapped key(s) ['fractional_seaice', 'num_land_cat']" -- a
    # sentence about the importer's map, not about the run.
    #
    # num_land_cat is the geography's land-use category count.  gpuwm never
    # reads it from the namelist: the count comes from LANDUSE.TBL for the
    # MMINLU the static build stamped (gpuwm/core/landuse.py
    # load_landuse_table + the lucats bound check), and every gpuwm
    # geography and wrfout is the 21-category MODIS set
    # (gpuwm/native_wrf_contract.py:47, gpuwm/io/wrfout.py:197).  So it is
    # validated against that identity and recorded -- a namelist declaring
    # USGS's 24 describes static data gpuwm does not build, and refusing is
    # the only honest answer.
    _MODIS_LAND_CATEGORIES = 21
    num_land_cat = ph.take("num_land_cat")
    if num_land_cat is not None:
        if any(isinstance(value, bool) or not isinstance(value, int)
               or value != _MODIS_LAND_CATEGORIES for value in num_land_cat):
            raise _err(
                "physics", "num_land_cat", num_land_cat,
                f"gpuwm builds one land-use identity -- the "
                f"{_MODIS_LAND_CATEGORIES}-category "
                "MODIFIED_IGBP_MODIS_NOAH set stamped by its static builder "
                "and written into every wrfout -- so no other category count "
                "describes the geography it will actually initialize from.")
    fix("physics", "num_land_cat", num_land_cat, _MODIS_LAND_CATEGORIES,
        "the land-use category count is read from LANDUSE.TBL for the "
        "static build's MMINLU (MODIFIED_IGBP_MODIS_NOAH), never from the "
        "namelist")
    # fractional_seaice selects WRF's sea-ice land-use branch.  gpuwm does
    # not take it from the namelist either, and unlike num_land_cat the two
    # gpuwm initialization routes do not agree: the prepared-cache route
    # every HRRR/GFS/ERA5 forecast runs (gpuwm/ingest/hrrr_physics.py:154)
    # calls initialize_landuse with the FRACTIONAL branch, while the
    # case-data runtime path (gpuwm/runtime.py:702, :930) takes the default
    # XICE >= 0.5 branch.  Recording that divergence beside the value is
    # the honest report; inventing a knob that only one of the two routes
    # would honor is not.
    fractional_seaice = ph.take("fractional_seaice")
    if fractional_seaice is not None and any(
            isinstance(value, bool) or not isinstance(value, int)
            or value not in (0, 1) for value in fractional_seaice):
        raise _err(
            "physics", "fractional_seaice", fractional_seaice,
            "must be 0 (WRF's XICE >= 0.5 branch, the Registry default) or "
            "1 (the fractional branch).")
    drop("physics", "fractional_seaice", fractional_seaice,
         "gpuwm selects the sea-ice land-use branch per initialization "
         "route, not from the namelist: the prepared-cache route every "
         "HRRR/GFS/ERA5 forecast runs uses the FRACTIONAL branch "
         "(gpuwm/ingest/hrrr_physics.py -> gpuwm/core/landuse.py:264,318), "
         "the case-data runtime path uses the XICE >= 0.5 branch")
    urban = ph.take("sf_urban_physics")
    if urban is not None and any(int(v) != 0 for v in urban):
        raise _err("physics", "sf_urban_physics", urban,
                   "urban physics is not implemented.")
    fix("physics", "sf_urban_physics", urban, 0,
        "urban physics is not implemented; validated off")
    for key in ("sf_surface_mosaic", "mosaic_lu", "mosaic_soil"):
        values = ph.take(key)
        if values is not None and any(int(value) != 0 for value in values):
            raise _err(
                "physics", key, values,
                "mosaic land/soil physics is not implemented; the target "
                "suite requires this option off (0).")
        fix("physics", key, values, 0,
            "mosaic land/soil physics is not implemented; validated off")
    ph.finish()

    # ---- switches WRF's namelist cannot state -----------------------------
    # ``moist_cq`` has no WRF namelist key, and gpuwm's ``top_lid`` default
    # is deliberately not WRF's Registry default.  Both land in every
    # prepared-cache domain identity, so the importer, the shipped physics
    # profiles and the domain wizard have to give the same answer for the
    # same suite or a root sealed from a profile can never match a
    # hierarchy imported from the same namelist.  physics_compat owns that
    # answer; this is a lookup, not a second opinion.
    from gpuwm.physics_compat import implicit_runtime_switches

    implicit = implicit_runtime_switches(
        mp_physics=mp_physics, sf_sfclay_physics=sfclay,
        sf_surface_physics=sfsfc, bl_pbl_physics=bl_pbl_physics,
        cu_physics=cu[0], num_soil_layers=resolved_soil_layers,
        ra_lw_physics=ra_lw_physics, ra_sw_physics=ra_sw_physics)

    # ---- &dynamics --------------------------------------------------------
    # Omitted keys take WRF Registry defaults (F2): hybrid_opt 2
    # (registry.hyb_coord:62), damp_opt 3 (Registry.EM_COMMON:2848);
    # km_opt's Registry default is -1 = MUST-SET, so omission (or -1) is
    # a hard error rather than a silently invented scheme.
    use_theta_m_values = dyn.take("use_theta_m")
    # Registry.EM_COMMON:2860 defaults this scalar to 1.  gpuwm's
    # h_diabatic/moist-physics transcription implements only the dry-theta
    # (use_theta_m=0) branch, so omission must not silently select 0.
    use_theta_m = (1 if use_theta_m_values is None
                   else int(use_theta_m_values[0]))
    if use_theta_m != 0:
        raise _err(
            "dynamics", "use_theta_m", use_theta_m,
            "unsupported: the moist-theta branch (use_theta_m = 1, also "
            "the WRF Registry default when omitted) is not implemented; "
            "gpuwm requires use_theta_m = 0.")
    fix("dynamics", "use_theta_m", use_theta_m_values, 0,
        "gpuwm transcribes the non-moist-theta use_theta_m=0 branch")
    top_lid_values = dyn.col("top_lid", max_dom)
    if top_lid_values is None:
        # NOT WRF's Registry default.  gpuwm's open-top branch is
        # implemented and selectable, but it is not what gpuwm runs when
        # nobody says: gpuwm/config.py records the 2026-07-18 probe where
        # the open top NaN'd a two-domain real-data run within 15 sim-min,
        # and every shipped physics profile states the value its suite was
        # certified with.  Resolving an ABSENT key to WRF's default instead
        # of to that certified value is what made a hierarchy imported from
        # a profile's own namelist unable to match the root prepared from
        # it.
        top_lid = bool(implicit["top_lid"])
        defaults_applied.append(AppliedDefault(
            key="top_lid", value=top_lid,
            reason="the namelist does not declare &dynamics/top_lid; "
                   f"resolved from {implicit['source']} (WRF's Registry "
                   "default is an open top, which gpuwm implements but "
                   "does not select for you)"))
    else:
        top_lid = bool(_uniform(
            "dynamics", "top_lid",
            _require_bools("dynamics", "top_lid", top_lid_values)))
    hybrid_opt = int(dyn.scalar("hybrid_opt", 2))
    etac = float(dyn.scalar("etac", 0.2))
    w_damping = int(dyn.scalar("w_damping", 0))
    # Registry.EM_COMMON initializes every domain to 0.1.  A scalar
    # ``epssm = 0.5`` namelist assignment changes d01 only; its unassigned
    # tail remains 0.1 (as confirmed by WRF's effective namelist.output).
    # Preserve that per-domain state instead of broadcasting the scalar.
    epssm = [float(value) for value in
             dyn.registry_col("epssm", max_dom, 0.1)]
    km_opt_col = dyn.col("km_opt", max_dom)
    if km_opt_col is None:
        raise _err("dynamics", "km_opt", None,
                   "must-set namelist key (WRF Registry default -1 "
                   "refuses to run); gpuwm will not invent a mixing "
                   "scheme for you.")
    # PER-DOMAIN, for the same reason bl_pbl_physics is: km_opt is in
    # gpuwm.experiment._DOMAIN_RUN_OVERRIDES ("a PBL parent may carry a
    # PBL-off Smagorinsky child"), and a nested LES tree is exactly a
    # namelist whose km_opt column is not constant.  The root value is the
    # [shared] one; a domain that differs emits its own override.
    km_opt_col = [int(value) for value in km_opt_col]
    km_opt = km_opt_col[0]
    diff_opt = dyn.col("diff_opt", max_dom)
    if diff_opt is not None and any(int(v) != 2 for v in diff_opt):
        raise _err("dynamics", "diff_opt", diff_opt,
                   "only the diff_opt=2 full-diffusion form backs "
                   "gpuwm's km_opt selection.")
    fix("dynamics", "diff_opt", diff_opt, 2,
        "gpuwm's km_opt selection implies the diff_opt=2 mixing form")
    mix_full = dyn.col("mix_full_fields", max_dom)
    if mix_full is None:
        raise _err(
            "dynamics", "mix_full_fields", None,
            "must be explicitly true on every domain: WRF's Registry "
            "default is false when omitted, while gpuwm implements only "
            "full-field diff_opt=2 mixing.")
    if not all(_require_bools("dynamics", "mix_full_fields", mix_full)):
        raise _err("dynamics", "mix_full_fields", mix_full,
                   "gpuwm Smagorinsky mixing always acts on full fields.")
    fix("dynamics", "mix_full_fields", mix_full, True,
        "explicit WRF full-field mode matches gpuwm's implemented mixing")
    diff_6th_opt = int(_uniform("dynamics", "diff_6th_opt",
                                dyn.col("diff_6th_opt", max_dom, 0)))
    diff_6th_factor = [float(v)
                       for v in dyn.col("diff_6th_factor", max_dom, 0.12)]
    diff_6th_slopeopt = int(_uniform(
        "dynamics", "diff_6th_slopeopt",
        dyn.col("diff_6th_slopeopt", max_dom, 0)))
    # moist_mix6_off (Registry.EM_COMMON:2889, &dynamics, max_domains,
    # default .false.): WRF's own switch for taking the 6th-order filter
    # off the moist scalars (dyn_em/module_em.F:1421), mapped 1:1 onto the
    # RunConfig field of the same name and spelling.  Supplied-only, on
    # the knob-parity rule: the Registry default equals gpuwm's frozen
    # RunConfig default, so omission emits nothing and established imports
    # stay byte-identical.  Per domain on epssm's Fortran-assignment
    # semantics: an unassigned tail keeps the Registry default rather
    # than broadcasting the head value.
    moist_mix6_off_raw = dyn.take("moist_mix6_off")
    moist_mix6_off_col = None
    if moist_mix6_off_raw is not None:
        supplied = _require_bools("dynamics", "moist_mix6_off",
                                  moist_mix6_off_raw[:max_dom])
        moist_mix6_off_col = supplied + [False] * (max_dom - len(supplied))
    # The km_opt=2/3 turbulence parameter row (knob-parity lane).  Every
    # key here is `max_domains` in Registry.EM_COMMON (c_s :2862, c_k
    # :2863, mix_isotropic :2896, mix_upper_bound :2897, tke_upper_bound
    # :2899, tke_drag_coefficient :2900, tke_heat_flux :2901) and every
    # one sits in gpuwm.experiment._DOMAIN_RUN_OVERRIDES, so each is read
    # as a COLUMN for the same reason km_opt and bl_pbl_physics are: a PBL
    # parent carrying a PBL-off LES child is exactly a namelist whose
    # turbulence columns are not constant.
    #
    # Before this, only c_s was read (and it was forced uniform); c_k,
    # mix_isotropic, mix_upper_bound, tke_upper_bound, tke_heat_flux and
    # tke_drag_coefficient were not read at all.  That left NO pair of
    # inputs able to express a km_opt=2 LES child: omitting c_k resolved
    # gpuwm's RunConfig default 0.15 against an LES TOML's em_les 0.10 and
    # the prepared cache -- which binds the turbulence row per domain --
    # refused the forecast on `run.c_k`, while supplying c_k in the
    # namelist hit _Section.finish()'s unmapped-key refusal instead.  Both
    # directions were closed, so the whole km_opt=2 hierarchy gate was.
    #
    # Each Registry default equals gpuwm's frozen RunConfig default, so an
    # omitted key resolves identically and established imports stay
    # byte-identical.  The root value is the [shared] one; a domain that
    # differs emits its own override, on epssm's existing rule.
    def _real_col(key: str, ok, why: str) -> list[float] | None:
        """A supplied per-domain real column, validated element-wise."""
        raw = dyn.col(key, max_dom)
        if raw is None:
            return None
        values = [float(value) for value in raw]
        if not all(ok(value) for value in values):
            raise _err("dynamics", key, raw, why)
        return values

    c_s_col = _real_col(
        "c_s", lambda v: math.isfinite(v) and v > 0.0,
        "must be a finite positive Smagorinsky constant "
        "(WRF Registry default 0.25).")
    c_k_col = _real_col(
        "c_k", lambda v: math.isfinite(v) and v > 0.0,
        "must be a finite positive TKE-closure constant "
        "(WRF Registry default 0.15; the em_les reference sets 0.10).")
    mix_upper_bound_col = _real_col(
        "mix_upper_bound", lambda v: math.isfinite(v) and v > 0.0,
        "must be a finite positive non-dimensional K cap "
        "(WRF Registry default 0.1).")
    tke_upper_bound_col = _real_col(
        "tke_upper_bound", lambda v: math.isfinite(v) and v > 0.0,
        "must be a finite positive TKE ceiling in m2 s-2 "
        "(WRF Registry default 1000.).")
    tke_heat_flux_col = _real_col(
        "tke_heat_flux", math.isfinite,
        "must be a finite kinematic heat flux in K m s-1 "
        "(WRF Registry default 0.).")
    tke_drag_coefficient_col = _real_col(
        "tke_drag_coefficient", lambda v: math.isfinite(v) and v >= 0.0,
        "must be a finite non-negative drag coefficient "
        "(WRF Registry default 0.).")
    mix_isotropic_raw = dyn.col("mix_isotropic", max_dom)
    mix_isotropic_col = None
    if mix_isotropic_raw is not None:
        if any(isinstance(value, bool) or not isinstance(value, int)
               or value not in (0, 1) for value in mix_isotropic_raw):
            raise _err("dynamics", "mix_isotropic", mix_isotropic_raw,
                       "must be 0 (anisotropic mixing lengths) or 1 "
                       "(isotropic (dx*dy*dz)^(1/3)).")
        mix_isotropic_col = [int(value) for value in mix_isotropic_raw]
    #: The row in emission order: (TOML key, supplied column or None).
    turbulence_row = (
        ("c_s", c_s_col),
        ("c_k", c_k_col),
        ("mix_isotropic", mix_isotropic_col),
        ("mix_upper_bound", mix_upper_bound_col),
        ("tke_upper_bound", tke_upper_bound_col),
        ("tke_heat_flux", tke_heat_flux_col),
        ("tke_drag_coefficient", tke_drag_coefficient_col),
    )
    diff_6th_thresh_col = dyn.col("diff_6th_thresh", max_dom)
    diff_6th_thresh = (None if diff_6th_thresh_col is None else float(
        _uniform("dynamics", "diff_6th_thresh", diff_6th_thresh_col)))
    if diff_6th_thresh is not None and (
            not math.isfinite(diff_6th_thresh) or diff_6th_thresh <= 0.0):
        raise _err("dynamics", "diff_6th_thresh", diff_6th_thresh_col,
                   "must be a finite positive terrain slope in m/m "
                   "(WRF Registry default 0.10).")
    # &dynamics keys the dycore pins where WRF has options: validated
    # against the single implemented value, recorded as fixed.
    for key, pin, why in (
            ("rk_ord", 3,
             "the dycore integrates WRF's 3-stage RK3 only "
             "(gpuwm/core/dycore.py stage table)"),
            ("h_mom_adv_order", 5,
             "horizontal momentum advection is the WRF flux5 stencil, "
             "hardcoded (gpuwm/core/kernels/advection.cu)"),
            ("v_mom_adv_order", 3,
             "vertical momentum advection is the WRF flux3 stencil, "
             "hardcoded (gpuwm/core/kernels/advection.cu)"),
            ("v_sca_adv_order", 3,
             "vertical scalar advection is the WRF flux3 stencil, "
             "hardcoded (gpuwm/core/kernels/advection.cu)"),
            ("momentum_adv_opt", 1,
             "standard (non-positive-definite) momentum advection only"),
    ):
        raw = dyn.col(key, max_dom)
        if raw is None:
            continue
        if any(isinstance(value, bool) or not isinstance(value, int)
               or value != pin for value in raw):
            raise _err("dynamics", key, raw,
                       f"gpuwm implements {key} = {pin} only ({why}).")
        fix("dynamics", key, raw, pin, why)
    # tke_adv_opt (Registry.EM_COMMON:2880, max_domains, default 1).  The
    # old rationale here -- "inert: no prognostic-TKE mixing scheme is
    # selectable (km_opt 1 and 4 only)" -- was falsified by km_opt=2, and
    # a dropped key that DOES change the answer is exactly the silent loss
    # _Section.finish() exists to prevent.  Which of the two it is now
    # depends on the mixing scheme actually selected:
    #
    # * with a km_opt=2 domain in the tree, TKE is a prognostic carrier
    #   that gpuwm advects through the moist-scalar rows, and it takes
    #   WRF's POSITIVE-DEFINITE update -- tke_adv_opt = 1's branch
    #   (gpuwm/core/moist.py advance_tke_stage, WRF solve_em.F:2100-2119)
    #   -- unconditionally.  gpuwm implements no other transport for it,
    #   so 1 is pinned and 0/2 are refused rather than dropped;
    # * with no km_opt=2 domain there is no TKE carrier to transport, so
    #   WRF would not consume the key either and it is dropped as inert.
    tke_adv_opt_raw = dyn.take("tke_adv_opt")
    if 2 in km_opt_col:
        if tke_adv_opt_raw is not None:
            column = list(tke_adv_opt_raw[:max_dom])
            column += [column[-1]] * (max_dom - len(column))
            if any(isinstance(value, bool) or not isinstance(value, int)
                   or value != 1 for value in column):
                raise _err(
                    "dynamics", "tke_adv_opt", tke_adv_opt_raw,
                    "gpuwm implements tke_adv_opt = 1 only (the "
                    "positive-definite RK3 TKE transport of WRF "
                    "solve_em.F:2100-2119, which km_opt = 2 makes live).")
            fix("dynamics", "tke_adv_opt", tke_adv_opt_raw, 1,
                "the km_opt=2 TKE carrier takes WRF's positive-definite "
                "RK3 transport, the only one gpuwm implements")
    else:
        drop("dynamics", "tke_adv_opt", tke_adv_opt_raw,
             "inert: no domain selects a prognostic-TKE mixing scheme "
             "(km_opt = 2), so there is no TKE carrier to transport and "
             "WRF would not consume it either")
    base_temp = float(dyn.scalar("base_temp", 290.0))
    damp_opt = int(dyn.scalar("damp_opt", 3))
    zdamp = float(_uniform("dynamics", "zdamp",
                           dyn.col("zdamp", max_dom, 5000.0)))
    dampcoef = float(_uniform("dynamics", "dampcoef",
                              dyn.col("dampcoef", max_dom, 0.2)))
    khdif = float(_uniform("dynamics", "khdif",
                           dyn.col("khdif", max_dom, 0)))
    kvdif = float(_uniform("dynamics", "kvdif",
                           dyn.col("kvdif", max_dom, 0)))
    if any(value == 4 for value in km_opt_col) and (khdif > 0.0
                                                    or kvdif > 0.0):
        raise _err(
            "dynamics", "km_opt", km_opt_col,
            "selects WRF Smagorinsky mixing, but khdif/kvdif also enable "
            "the km_opt=1 constant-K operator; choose exactly one mixing "
            "scheme.")
    non_hydro = dyn.col("non_hydrostatic", max_dom)
    if non_hydro is not None and not all(_require_bools(
            "dynamics", "non_hydrostatic", non_hydro)):
        raise _err("dynamics", "non_hydrostatic", non_hydro,
                   "gpuwm is nonhydrostatic-only.")
    fix("dynamics", "non_hydrostatic", non_hydro, True,
        "gpuwm is nonhydrostatic-only")
    moist_adv_opt = int(_uniform(
        "dynamics", "moist_adv_opt",
        dyn.col("moist_adv_opt", max_dom, 1)))
    scalar_adv = dyn.col("scalar_adv_opt", max_dom, 1)
    scalar_adv_opt = int(_uniform(
        "dynamics", "scalar_adv_opt", scalar_adv))
    if moist_adv_opt != 1 or scalar_adv_opt != 1:
        raise _err(
            "dynamics", "moist_adv_opt/scalar_adv_opt",
            (moist_adv_opt, scalar_adv_opt),
            "only the matched WRF positive-definite option 1 is "
            "implemented for both moisture mass and scalar/number fields.")
    fix("dynamics", "scalar_adv_opt", scalar_adv, 1,
        "validated equal to the supported moist_adv_opt=1 path")
    time_step_sound = int(dyn.scalar("time_step_sound", 0))
    if time_step_sound == 0:
        # WRF Registry default 0 = automatic selection; WRF picks 4 for
        # these dt/dx regimes and gpuwm needs an explicit even count --
        # recorded, never silent (F2).
        time_step_sound = 4
        defaults_applied.append(AppliedDefault(
            key="time_step_sound", value=4,
            reason="WRF Registry default 0 means automatic selection; "
                   "4 is WRF's choice at conventional dt/dx and gpuwm "
                   "requires an explicit even value"))
    smdiv = float(dyn.scalar("smdiv", 0.1))
    # WRF Registry defaults the namelist leaves unset, ratified binding
    # (Phase-4 native-dt baseline): emdiv 0.01, h_sca_adv_order 5.
    # (hypsometric_opt's binding is the same, but it is read from
    # &domains, where WRF declares it -- see the &domains block above.)
    emdiv = float(dyn.scalar("emdiv", 0.01))
    h_sca_adv_order = int(dyn.scalar("h_sca_adv_order", 5))
    # The importer refuses rather than tolerates the old placement.  A
    # namelist carrying hypsometric_opt in &dynamics is one wrf.exe
    # cannot read at all, so silently honouring it here would hand a
    # gpuwm forecast a setting its mirrored WRF arm could never run;
    # generic "unmapped key" would be true but would not say where the
    # key belongs, and this refusal has one obvious repair.
    stale_hypsometric = dyn.take("hypsometric_opt")
    if stale_hypsometric is not None:
        raise _err(
            "dynamics", "hypsometric_opt", stale_hypsometric,
            "WRF declares hypsometric_opt in &domains as a scalar "
            "(Registry.EM_COMMON:2283, `namelist,domains`, nentries 1); "
            "in &dynamics it fails wrf.exe's own namelist read before "
            "the first timestep.  Move the key to &domains.")
    if h_sca_adv_order != 5:
        raise _err(
            "dynamics", "h_sca_adv_order", h_sca_adv_order,
            "only the WRF Registry default 5 imports: gpuwm's transported-"
            "scalar stencils are fixed at WRF's 5th-order horizontal/"
            "3rd-order vertical forms, and the configurable "
            "h_sca_adv_order feeds only the geopotential advection "
            "(rhs_ph) -- a non-default value here would not mean what it "
            "means in WRF.")
    dyn.finish()

    # ---- &bdy_control -----------------------------------------------------
    spec_bdy_width = int(bdy.scalar("spec_bdy_width", 5))
    spec_zone = int(bdy.scalar("spec_zone", 1))
    relax_zone = int(bdy.scalar("relax_zone", 4))
    spec_exp = float(bdy.scalar("spec_exp", 0.0))
    # WRF defaults when &bdy_control omits the flags: the head grid takes
    # external specified LBCs, every child is nested (the bundle sets
    # specified = T,F,F,F / nested = F,T,T,T explicitly).
    specified_col = bdy.col("specified", max_dom)
    specified = _require_bools("bdy_control", "specified", specified_col) \
        if specified_col is not None \
        else [True] + [False] * (max_dom - 1)
    nested_col = bdy.col("nested", max_dom)
    nested = _require_bools("bdy_control", "nested", nested_col) \
        if nested_col is not None \
        else [False] + [True] * (max_dom - 1)
    bdy.finish()

    # ---- emit the resolved TOML ------------------------------------------
    if name is None:
        name = f"wrf_{start_time:%Y%m%d%H}_{max_dom}dom"
    lines = [
        "# gpuwm experiment translated from WRF namelists by "
        "`gpuwm import-namelist`",
        f"# ({wps_path.name} + {input_path.name}).  WRF staggered "
        "e_we/e_sn/e_vert convert",
        "# to gpuwm mass dimensions nx/ny/nz (one fewer point).  Child "
        "dx/dt are NEVER",
        "# written: gpuwm/experiment.py derives them exactly from the "
        "parent chain",
        "# (the namelist chain is authoritative).",
    ]
    if substitutions:
        lines.append("# Physics substitutions (ratified, never silent):")
        for s in substitutions:
            lines.append(f"#   {s.key} {s.wrf_value} ({s.wrf_name}) -> "
                         f"{s.gpuwm_key} {s.gpuwm_value} ({s.gpuwm_name})")
    lines += [
        "",
        "[experiment]",
        f'name = "{name}"',
        f"start_time = {start_time.isoformat(sep='T')}",
        f"run_seconds = {_fmt(run_seconds)}",
        f"feedback = {feedback}",
        f"smooth_option = {smooth_option}",
        f"blend_width = {blend_width}",
        f"spec_bdy_width = {spec_bdy_width}",
        f"restart_interval_s = {_fmt(restart_interval_s)}",
        "",
        "[projection]",
        f'map_proj = "{projection["map_proj"]}"',
    ]
    for key in ("ref_lat", "ref_lon", "truelat1", "truelat2", "stand_lon"):
        lines.append(f"{key} = {_fmt(projection[key])}")
    lines += [
        "",
        "[shared]",
        f"# WRF e_vert = {nz + 1} staggered full levels -> nz = {nz} "
        "mass levels.",
        f"nz = {nz}",
        "# Nominal vertical scaffold height; the real path derives "
        "heights from",
        "# p_top/eta_levels.",
        "ztop = 20000.0",
        f"p_top = {_fmt(p_top)}",
    ]
    if eta_levels:
        lines.append("eta_levels = [")
        for i in range(0, len(eta_levels), 5):
            chunk = ", ".join(repr(v) for v in eta_levels[i:i + 5])
            lines.append(f"    {chunk},")
        lines.append("]")
    lines += [
        f"hybrid_opt = {hybrid_opt}",
        f"etac = {_fmt(etac)}",
        f"base_temp = {_fmt(base_temp)}",
        f"time_step_sound = {time_step_sound}",
        # [shared] carries d01; differing Registry-tail values are emitted as
        # explicit [[domain]] overrides below.
        f"epssm = {_fmt(epssm[0])}",
        "# WRF Registry defaults the namelist leaves unset, ratified "
        "binding for the",
        "# reference configuration (Phase-4 native-dt baseline): emdiv "
        "0.01,",
        "# hypsometric_opt 2, h_sca_adv_order 5.",
        f"emdiv = {_fmt(emdiv)}",
        f"hypsometric_opt = {hypsometric_opt}",
        f"h_sca_adv_order = {h_sca_adv_order}",
        f"smdiv = {_fmt(smdiv)}",
        f"top_lid = {_fmt(top_lid)}",
        f"moist = {_fmt(mp_physics > 0)}",
        # WRF has no moist_cq namelist key: it derives calc_cq from the
        # moist species that exist.  The importer used to answer that with
        # its own rule, `mp_physics > 0`, which contradicted every shipped
        # WSM6/Kessler/MYNN/RUC/Noah-MP profile (all moist_cq = false) and
        # so made a public HRRR hierarchy unable to bind the public root it
        # was prepared from.  physics_compat is the one authority now.
        f"moist_cq = {_fmt(bool(implicit['moist_cq']))}",
        f"mp_physics = {mp_physics}",
        f"morr_rimed_ice = {morr_rimed_ice}",
        f"moist_adv_opt = {moist_adv_opt}",
        f"km_opt = {km_opt}",
        f"diff_6th_opt = {diff_6th_opt}",
        f"diff_6th_slopeopt = {diff_6th_slopeopt}",
        f"w_damping = {w_damping}",
        f"damp_opt = {damp_opt}",
        f"zdamp = {_fmt(zdamp)}",
        f"dampcoef = {_fmt(dampcoef)}",
        f"khdif = {_fmt(khdif)}",
        f"kvdif = {_fmt(kvdif)}",
    ]
    # Supplied-only knobs (knob-parity lane): each maps 1:1 onto a
    # consumed RunConfig field whose default equals the WRF Registry
    # default, so absence emits nothing and the established imports stay
    # byte-identical.
    for _turb_key, _turb_col in turbulence_row:
        if _turb_col is not None:
            lines.append(f"{_turb_key} = {_fmt(_turb_col[0])}")
    if diff_6th_thresh is not None:
        lines.append(f"diff_6th_thresh = {_fmt(diff_6th_thresh)}")
    if moist_mix6_off_col is not None:
        lines.append(f"moist_mix6_off = {_fmt(moist_mix6_off_col[0])}")
    lines += [
        f"spec_zone = {spec_zone}",
        f"relax_zone = {relax_zone}",
        "terrain_opt = 1",
        f"map_proj = {WRF_MAP_PROJ_CODES[projection['map_proj']]}",
        f"sf_sfclay_physics = {sfclay}",
        f"sf_surface_physics = {sfsfc}",
        f"bl_pbl_physics = {bl_pbl_physics}",
        f"ra_physics = {ra_physics}",
    ]
    if resolved_soil_layers != 4:
        lines.append(f"num_soil_layers = {resolved_soil_layers}")
    for key, value in (("no_mp_heating", no_mp_heating),
                       ("mp_tend_lim", mp_tend_lim),
                       ("ysu_topdown_pblmix", ysu_topdown_pblmix),
                       ("isfflx", isfflx),
                       ("isftcflx", isftcflx),
                       ("iz0tlnd", iz0tlnd),
                       ("usemonalb", usemonalb),
                       ("rdlai2d", rdlai2d),
                       ("opt_thcnd", opt_thcnd),
                       ("o3input", o3input),
                       ("use_mp_re", use_mp_re),
                       ("seaice_albedo_default",
                        seaice_albedo_default),
                       ("rdmaxalb", rdmaxalb),
                       ("nwp_diagnostics", nwp_diagnostics)):
        if value is not None:
            lines.append(f"{key} = {_fmt(value)}")
    if wrf_rrtmg_compatibility != "none":
        lines.append(
            f'wrf_rrtmg_compatibility = "{wrf_rrtmg_compatibility}"')
    if legacy_rrtmg:
        lines.append(f'ra_rrtmg_variant = "{RRTMG_VARIANT_LEGACY}"')
    if mp_physics == 6:
        lines.append(f"wsm6_hail_opt = {wsm6_hail_opt}")
    if not coupled_legacy:
        lines += [
            f"ra_lw_physics = {ra_lw_physics}",
            f"ra_sw_physics = {ra_sw_physics}",
        ]
    if ra_sw_physics == 1:
        lines += [
            f"icloud = {icloud}",
            f"swrad_scat = {_fmt(swrad_scat)}",
        ]
    lines.append(f"bldt = {_fmt(bldt)}")
    for n in range(max_dom):
        is_root = parent_id[n] == 0
        lines += [
            "",
            "[[domain]]",
            f"# WRF e_we = {e_we[n]}, e_sn = {e_sn[n]} (staggered) -> "
            "mass dimensions.",
            f"grid_id = {grid_ids[n]}",
            f"parent_id = {parent_id[n]}",
            f"i_parent_start = {i_start[n]}",
            f"j_parent_start = {j_start[n]}",
            f"parent_grid_ratio = {ratio[n]}",
            f"parent_time_step_ratio = {tratio[n]}",
            f"nx = {e_we[n] - 1}",
            f"ny = {e_sn[n] - 1}",
        ]
        if is_root:
            lines.append(f"time_step = {time_step}")
            if fract_num:
                lines.append(f"time_step_fract_num = {fract_num}")
                lines.append(f"time_step_fract_den = {fract_den}")
            lines.append(f"dx = {_fmt(root_dx)}")
            # The exponential sponge acts only on the specified (root)
            # branch (module_bc_em.F:1320); children carry spec_exp = 0
            # by the nested-branch rule.
            lines.append(f"spec_exp = {_fmt(spec_exp)}")
        lines += [
            f"specified = {_fmt(specified[n])}",
            f"nested = {_fmt(nested[n])}",
            f"history_interval_s = {_fmt(history_interval_s[n])}",
        ]
        if start_times[n] != start_time:
            lines.append(
                f"start_time = {start_times[n].isoformat(sep='T')}")
        if epssm[n] != epssm[0]:
            lines.append(f"epssm = {_fmt(epssm[n])}")
        # The turbulence column, emitted on exactly the same rule as
        # epssm: only where a domain differs from the root, so every
        # uniform namelist keeps producing a byte-identical TOML.
        if km_opt_col[n] != km_opt_col[0]:
            lines.append(f"km_opt = {km_opt_col[n]}")
        if bl_pbl_col[n] != bl_pbl_col[0]:
            lines.append(f"bl_pbl_physics = {bl_pbl_col[n]}")
        # The km_opt=2/3 turbulence parameter row, same rule: c_k on an
        # LES child beside its parents' default is the case that opened
        # this path (see the read side above).
        for _turb_key, _turb_col in turbulence_row:
            if _turb_col is not None and _turb_col[n] != _turb_col[0]:
                lines.append(f"{_turb_key} = {_fmt(_turb_col[n])}")
        if moist_mix6_off_col is not None \
                and moist_mix6_off_col[n] != moist_mix6_off_col[0]:
            lines.append(
                f"moist_mix6_off = {_fmt(moist_mix6_off_col[n])}")
        if radt[n] > 0.0:
            lines.append(f"radt = {_fmt(radt[n])}")
        else:
            # WRF radt = 0 -> radiation every step.  RunConfig's compat
            # `radt` key cannot express that (a positive radt overrides
            # radt_minutes, and radt = 0 falls back to the 12-minute
            # radt_minutes default), so emit radt_minutes = 0.0 (review
            # F3).
            lines.append("radt_minutes = 0.0")
        lines.append(f"cu_physics = {cu[n]}")
        if cu[n] == 1:
            lines.append(f"cudt_minutes = {_fmt(cudt[n])}")
        elif cu[n] == 3:
            # GF runs on the model step (STEPCU = 1, WRF's usual GF
            # configuration); RunConfig validation enforces the same.
            lines.append("cudt_minutes = 0.0")
            lines.append(f"clos_choice = {clos_choice}")
            lines.append(f"ishallow = {ishallow}")
            if cudt[n]:
                drop("physics", f"cudt[{n + 1}]", [cudt[n]],
                     "GF carries no cudt cadence: WRF's GF runs every "
                     "model step and so does gpuwm's")
        elif cudt[n]:
            drop("physics", f"cudt[{n + 1}]", [cudt[n]],
                 "cudt is consumed only where cu_physics = 1")
        # An omitted cudt_minutes inherits RunConfig's live 5.0 while the
        # shipped physics profiles and the domain wizard write 0.0 for the
        # same cumulus-off suite.  Both spellings mean the same dead
        # switch -- gpuwm/core/clock.py builds no cumulus calendar and
        # gpuwm/core/physics.py takes no cumulus step at cu_physics = 0 --
        # so the two are reconciled where identities are COMPARED
        # (gpuwm.ingest.prepared_cache.effective_prepared_domain_config),
        # not by writing a value into a receipt this importer has always
        # reproduced byte for byte.
        lines.append(f"diff_6th_factor = {_fmt(diff_6th_factor[n])}")
    text = "\n".join(lines) + "\n"

    # Validate the emitted TOML through the schema loader: every
    # section-A rule binds at import time.
    build_experiment(
        tomllib.loads(text),
        source=f"import of {wps_path.name} + {input_path.name}")
    # Every consumed key lands in exactly one report section: keys
    # recorded above as fixed or dropped are subtracted from each
    # section's consumption trace and the remainder is what produced the
    # TOML -- the Translated section.
    handled = ({(d.section, d.key) for d in dropped}
               | {(f.section, f.key) for f in fixed})
    translated: list[TranslatedKey] = []
    for section_obj in (share, geo, tc, dm, ph, dyn, bdy, noahmp, stoch):
        for key in section_obj.consumed:
            if (section_obj.name, key) not in handled:
                translated.append(TranslatedKey(section=section_obj.name,
                                                key=key))
    report = SubstitutionReport(
        substitutions=tuple(substitutions), dropped=tuple(dropped),
        defaults_applied=tuple(defaults_applied), fixed=tuple(fixed),
        translated=tuple(translated))
    return text, report
