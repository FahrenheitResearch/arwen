"""Immutable source-family authorities shipped with the RW-WPS wheel.

A PACKAGED PROFILE is a source whose three declarative authorities -- the
``rw-wps.mapping.v1`` mapping, the composition contract, and the terrain
provenance document -- ship inside the wheel and are pinned here by
SHA-256.  Everything else about such a source is table data too: its row in
:mod:`gpuwm.source_adapters` names the profile, and the front door reads
the profile rather than a per-source function.

Adding a model to this file is three JSON documents and one row of
:data:`_PACKAGED_PROFILES`.  That is the whole point: a source whose
mapping is shipped is not a code path.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


_AUTHORITY_ROOT = Path(__file__).with_name("authorities")

#: The three roles every packaged profile declares, in the order the front
#: door passes them.
PROFILE_ROLES = ("mapping", "composition", "provenance")


def _profile(
    stem: str,
    *,
    source_format: str,
    mapping: str,
    composition: str,
    provenance: str,
    data_role: str | None = None,
    provenance_role: str | None = None,
    composition_state: str = "composed",
    contributing_mappings: Mapping[str, Mapping[str, str]] | None = None,
) -> Mapping[str, object]:
    """One packaged profile: file names, byte pins, and its two roles.

    ``data_role``/``provenance_role`` repeat what the composition document
    declares in ``supplements.terrain_height`` (or, for a cross-source
    profile, on the binding that provides terrain); they are stated here
    as well because the front door has to spell them on the command line
    BEFORE anything opens the composition, and a role the caller guesses is
    a role that silently binds the wrong file.  They are checked against
    the composition at decode time by
    :func:`gpuwm.mapped_composition.decode_composed_source`, which refuses
    on any difference -- so a wrong row here fails loudly, not quietly.

    ``composition_state`` is ``"composed"`` for a runnable profile.  An
    atmosphere-only source -- complete 3-D state, zero published land
    surface -- ships ``"pending_cross_source"`` instead: its composition
    document is an explicit PENDING declaration
    (:data:`gpuwm.mapped_composition.PENDING_COMPOSITION_SCHEMA`) that
    refuses to load by naming the state the source does not publish, it
    supplies no terrain supplement, and therefore it has no roles.  A
    pending profile decodes and inspects through the mapped route; it
    does not initialize anything until the cross-source composition
    supplying the missing state lands.

    ``contributing_mappings`` is the CROSS-SOURCE slot: a composed profile
    whose composition declares ``field_sources`` bindings ships each
    contributing source's own mapping document as an additional pinned
    authority, keyed by the binding's ``mapping_role`` --
    ``{role: {"file": name, "sha256": digest}}``.  The front door passes
    each one as ``--contributing-mapping role=path`` so the prepared
    runner can hand ``contributing_mappings`` to the composed decode; the
    composition independently pins the same digest, so a wrong row here
    fails loudly at decode, not quietly.  Only a composed profile may
    declare the slot: a pending profile has no runnable composition to
    bind anything to.  Shipping the row is what lets the front door pass
    the donor table to the prepared runner without the caller supplying
    (or being able to substitute) one.
    """

    if composition_state not in {"composed", "pending_cross_source"}:
        raise ValueError(
            f"unknown composition_state {composition_state!r} for {stem}"
        )
    roles_declared = data_role is not None and provenance_role is not None
    if composition_state == "composed" and not roles_declared:
        raise ValueError(
            f"composed profile {stem} must declare data_role and "
            "provenance_role; only a pending_cross_source profile has none"
        )
    if composition_state == "pending_cross_source" and (
        data_role is not None or provenance_role is not None
    ):
        raise ValueError(
            f"pending profile {stem} has no terrain supplement and "
            "therefore no roles to declare"
        )
    contributing: dict[str, Mapping[str, str]] = {}
    for role, pin in (contributing_mappings or {}).items():
        if composition_state != "composed":
            raise ValueError(
                f"profile {stem} declares a contributing mapping but is "
                "not composed; only a runnable composition binds donors"
            )
        if set(pin) != {"file", "sha256"}:
            raise ValueError(
                f"profile {stem} contributing mapping {role!r} must pin "
                "exactly a file name and its sha256"
            )
        contributing[str(role)] = MappingProxyType({
            "file": str(pin["file"]), "sha256": str(pin["sha256"]),
        })
    return MappingProxyType({
        "source_format": source_format,
        "files": MappingProxyType({
            "mapping": f"{stem}.mapping.json",
            "composition": f"{stem}.composition.json",
            "provenance": f"{stem}.provenance.json",
        }),
        "sha256": MappingProxyType({
            "mapping": mapping,
            "composition": composition,
            "provenance": provenance,
        }),
        "data_role": data_role,
        "provenance_role": provenance_role,
        "composition_state": composition_state,
        "contributing_mappings": MappingProxyType(contributing),
    })


_PACKAGED_PROFILES = MappingProxyType({
    "20crv3-member-grib2-v1": _profile(
        "rw-wps-20crv3-member-grib2",
        source_format="grib2",
        mapping="2e9877d51d9c993e83311c87236467b99ce9022638a343985da10ccd195efe09",
        composition="aa4f3fac03c09e8461c5e6c5e04a6bed48b5ad477babc4c75e8dd10fd92fe7b2",
        provenance="d1248e1b091f59841757a98a024cbe2868cebc25308f4eb4f9608e2c1755f3b1",
        data_role="twentycrv3_in_band_surface",
        provenance_role="twentycrv3_in_band_surface_provenance",
    ),
    "20crv3-netcdf-v1": _profile(
        "rw-wps-20crv3-netcdf",
        source_format="netcdf",
        mapping="2bb8b34843466dffbad9b37ac2de4dc10101c4310ce694e9d427ef808fc1a048",
        composition="2c243fe4c4dba1c8f47178f2be583f3a20148d54d77101ee3421d1824d10b1c5",
        provenance="8daeb53502d28483a049936262910004cfda17aa5030cc3066d3bd01413d3066",
        data_role="twentycrv3_netcdf_recovered_invariant",
        provenance_role="twentycrv3_netcdf_recovered_invariant_provenance",
    ),
    # HRRR's public pressure-level product (wrfprs), decoded through the
    # generic mapped route: the Lambert CONUS grid, grid-relative wind
    # rotation and nine-node RUC soil are all TABLE DATA in these three
    # documents.  Selectors were authored from real 2026-08-15 00Z bytes
    # through the converged grib-core inventory.
    "hrrr-prs-grib2-v1": _profile(
        "rw-wps-hrrr-prs-grib2",
        source_format="grib2",
        mapping="6e17b6edcc2230262dbf072a72fd0b38a4a70e9812a76ea61d2b90fbd4301e54",
        composition="2a2bb75714428cdb9b051303e53d91c88f3c1b48a798339bb9244a6b412e392e",
        provenance="f2aade12671166959e42cacd357bc54359af4d3034eedff81630b26646eb4b8c",
        data_role="hrrr_prs_in_band_surface",
        provenance_role="hrrr_prs_in_band_surface_provenance",
    ),
    # RAP's awip32 product (AWIPS grid 221, 32 km Lambert, all of North
    # America): the one public RAP product that carries the complete
    # state in a single file per valid time -- 39 pressure levels
    # (byte-identical ladder to HRRR wrfprs), surface/2 m/10 m fields,
    # in-band terrain, LAND/ICEC, and the nine-node RUC soil column.
    # Selectors were authored from real 2026-08-16 00Z bytes through the
    # converged grib-core inventory.  Pressure-level humidity is RH, so
    # the mapping reuses the GFS profile's declared derivation; no new
    # engine capability was needed -- HRRR's Lambert grid family, wind
    # rotation and node soil carry every RAP-specific fact as table data.
    "rap-awip32-grib2-v1": _profile(
        "rw-wps-rap-awip32-grib2",
        source_format="grib2",
        mapping="61a1041ee68dff3202eb6d37a7492b57f634899bbde7e0c7644f78679634f55e",
        composition="bae76db6052e933906713d877b9029079357a45426ab52387acee0d9a11f385f",
        provenance="9d861f738b9ba72661130b9b174d969c8e134fbf11e191183fccc092a43a6abb",
        data_role="rap_awip32_in_band_surface",
        provenance_role="rap_awip32_in_band_surface_provenance",
    ),
    # ECCC's GDPS (GEM global) 15 km regular lat-lon GRIB2 product on MSC
    # Datamart: 33 pressure levels, surface/2m/10m state, a single
    # 0-10 cm ISBA soil layer, and the once-per-cycle analysis
    # invariants (orography, land mask, ice analysis) declared through
    # the generic cycle-invariant/broadcast grammar.  Selectors were
    # authored from real 2026-08-16 00Z bytes through the converged
    # grib-core inventory.
    "gem-gdps-grib2-v1": _profile(
        "rw-wps-gem-gdps-grib2",
        source_format="grib2",
        mapping="75811eab726d9217cca4abe224162ccdfb717457f4f0e39faefbf1ef08e1c98a",
        composition="13d7e4ca06f8012cfd252c03d70d9fae69d2e18b9b51975445d70a506e1d90ed",
        provenance="823e07b38677e3c0c83da984637a4fda83d1eb09be2401cb4e0a9e433820de22",
        data_role="gdps_analysis_invariant_surface",
        provenance_role="gdps_analysis_invariant_surface_provenance",
    ),
    # ECMWF's AIFS single deterministic forecast (open data, 0.25-degree
    # GDT-0 GRIB2): the reduced AI-model field set as TABLE DATA.  The
    # north-to-south row order, the analysis-step-only invariants (land
    # mask and surface geopotential ride the 0-hour file alone, declared
    # cycle-invariant), the geopotential-to-metres terrain scale and the
    # two-layer ordinal (type 151) soil column are all rows in these three
    # documents.  Selectors were authored from real 2026-08-17 00Z bytes
    # through the converged grib-core inventory.
    "aifs-single-grib2-v1": _profile(
        "rw-wps-aifs-single-grib2",
        source_format="grib2",
        mapping="7f65b6f9879112b27abcb1b6f02c2313de0571e673e864f848e2f5d01f3ea247",
        composition="bf0687369b7888f85ea5c2aa61f7f6800c8b99bec3331f9f4d021c565a775ba8",
        provenance="8aeb0d51ef5e43c504d7d5cf9d4b98992bd5becee2b75dccbb7d8b7808579aff",
        data_role="aifs_single_in_band_surface",
        provenance_role="aifs_single_in_band_surface_provenance",
    ),
    # ECMWF's open-data IFS oper product at 0.25 degrees: a plain global
    # GDT-0 latitude/longitude GRIB2 feed with 14 pressure levels,
    # in-band surface geopotential for terrain, and the four IFS soil
    # layers addressed by ordinal on fixed-surface type 151 (declared as
    # the composition's indexed selector_depth_binding).  Selectors were
    # authored from real 2026-08-16 00Z bytes through the converged
    # grib-core inventory.
    "ecmwf-open-data-oper-grib2-v1": _profile(
        "rw-wps-ecmwf-open-data-oper-grib2",
        source_format="grib2",
        mapping="1255405235db1f456c03de0f4757750568017b13cfed1b4a5f8b27bf3ead7c47",
        composition="3bd9c9cc25c74b53169b586e9d9ff499aeef3f512ef35c8dd551fd769fdfe5c1",
        provenance="c574a07fb303620eab432eea89dad3e765ccd510d965f2b0734c4e278d435b8b",
        data_role="ecmwf_open_data_in_band_surface",
        provenance_role="ecmwf_open_data_in_band_surface_provenance",
    ),
    # NCEP's GDAS analysis-cycle pgrb2 at 0.25 degrees: record-for-record
    # the GFS pgrb2 catalogue (696/696 at f000, MEASURED), arriving at a
    # different acquisition path with analysis-cycle semantics -- hourly
    # f000..f009 only, ~+7 h latency.  The four Noah soil layers are
    # bound by their scaled type-106 depth pairs (the integer level key
    # collides between the first two layers), soil moisture is an NCEP
    # local-table row selected by octets, and the honesty facts ride the
    # provenance document: even f000 is stamped a forecast in the bytes,
    # and the one analysis-stamped product has no land surface at all.
    # Selectors were authored from real 2026-08-17 06Z bytes through the
    # converged grib-core inventory.
    "gdas-pgrb2-0p25-grib2-v1": _profile(
        "rw-wps-gdas-pgrb2-0p25-grib2",
        source_format="grib2",
        mapping="f6de7a0a42d86ba1aa7e8cd84d30db79870843c25ccb6aa7ad3a3dbe8b4c4fc5",
        composition="7d57188638a53ecf8771ce4779c7923015cee519b63318f6dcd0982798abf75d",
        provenance="13105c86f74d247d8795c075458ccc87bea1de5f663d53b805a1765aaa07c5a6",
        data_role="gdas_pgrb2_in_band_surface",
        provenance_role="gdas_pgrb2_in_band_surface_provenance",
    ),
    # NCEP's AIGFS (GraphCast-based 0.25-degree global AI forecast): the
    # barest product in the catalog, and the first ATMOSPHERE-ONLY
    # profile -- six 3-D fields on 13 pressure levels plus 2 m/10 m/MSLP
    # state, and NO land surface of any kind.  Its composition role is
    # therefore the explicit PENDING declaration (loading it refuses by
    # naming the missing state), and the profile decodes/inspects without
    # initializing until the cross-source land-surface donor lands.  The
    # acquisition identity is part of the profile: operational bytes are
    # NOMADS-only and carry subCentre 0, while an S3 bucket serves a
    # DIFFERENT experimental run under identical filenames with
    # subCentre 2 -- every selector pins subcenter=0 so the imposter
    # refuses by name.  Selectors were authored from real 2026-08-17 00Z
    # bytes through the converged grib-core inventory.
    "aigfs-nomads-grib2-v1": _profile(
        "rw-wps-aigfs-nomads-grib2",
        source_format="grib2",
        mapping="0bb2fd3721a3feebf08eab3340331c6833e61afbbf6ed18313ae1187d5d3fcb9",
        composition="39953c876827616ee0142a26729a2d9813d062ca9350500e0d232a901cd00b41",
        provenance="7199ecd0f94c06d8c1d98829ebf66f24a78905ccc6cf52643aeba6e2f2979fad",
        composition_state="pending_cross_source",
    ),
    # The AIGFS HYBRID: the same operational NOMADS atmosphere (every
    # selector still pins subcenter=0 against the S3/EAGLE imposter) made
    # RUNNABLE by borrowing the seven canonicals AIGFS does not publish --
    # terrain, land mask, skin temperature, surface pressure, 2 m
    # humidity, and the four-layer soil column -- from the SAME CYCLE's
    # GDAS 0.25-degree analysis through the cross-source composition
    # (field_sources, source_cycle_analysis_broadcast clock).  The donor
    # decodes through its own mapping, shipped here as a fourth pinned
    # authority: the checked-in GFS pressure-level table with one
    # table-data change (2 m specific humidity DIRECTLY selected, because
    # a borrowed field must be directly selected in the donor).  Proven on
    # real 2026-08-17 00Z NOMADS + GDAS bytes.
    "aigfs-gdas-hybrid-grib2-v1": _profile(
        "rw-wps-aigfs-gdas-hybrid-grib2",
        source_format="grib2",
        mapping="0427834583be3189130ff676a57aff9d000a2196b82eb0bb7c2f7215412fff4d",
        composition="85aafd9a79e3d6ac33b6e14fcaee08cfc5608b90d64e0fdcf0de9dc0874481a7",
        provenance="99aea1e8945dd56e618d12e516870943b6d18c5a452ed016eddd24edb0264256",
        data_role="physical_analysis_surface_data",
        provenance_role="physical_analysis_surface_provenance",
        contributing_mappings={
            "physical_analysis_surface_mapping": {
                "file": "rw-wps-gdas-pgrb2-donor.mapping.json",
                "sha256": (
                    "35e0d4e2895b38a2702952fabc97fb9f53594b41dbb2d511"
                    "e39a13458f72f8cf"
                ),
            },
        },
    ),
    # One MEMBER of NCEP's GEFS v12 through the generic mapped route:
    # the 0.5-degree pgrb2a + pgrb2b pair of a single member.  The two
    # products' isobaric level sets are exactly disjoint (measured zero
    # overlap on every variable), so the mapping's 31-level ladder is
    # only satisfiable by both files of the SAME member; every selector
    # pins PDT 1, which is what refuses the geavg/gespr statistic files
    # (PDT 2) and the PDT-11 accumulation twins at the byte level.  The
    # four Noah soil layers split across the pair (0-0.1 m in pgrb2a,
    # the rest in pgrb2b) under the measured 66-percent ocean bitmap;
    # terrain is in band but migrates products (analysis: pgrb2a;
    # forecast steps: pgrb2b -- measured), so the terrain supplement is
    # the same a+b pair per valid time, proven invariant across the
    # window.  Which members exist and how their
    # bytes verify is the sibling members grammar below -- this profile
    # answers only how a verified member's FIELDS decode.  Selectors
    # were authored from real 2026-08-17 00Z bytes through the converged
    # grib-core inventory.
    "gefs-ensemble-grib2-v1": _profile(
        "rw-wps-gefs-ensemble-grib2",
        source_format="grib2",
        mapping="d54a0fda85ae9f226379f419d9a6136b9597f86a031801c5c846705dd8ad9762",
        composition="f7db1f0399456334c86b4aad0a1591a3e1faf38fba190d7a439ce96fecebf529",
        provenance="5f7644c1cb68f06a347602d60b63443170c33973efa98e530c254a463f3fe154",
        data_role="gefs_member_in_band_surface",
        provenance_role="gefs_member_in_band_surface_provenance",
    ),
    # DWD open data's ICON-EU regular-lat-lon product set through the
    # generic mapped route: field-per-file bz2 GRIB2 objects, the 20-level
    # pressure ladder, once-per-cycle invariant FR_LAND/HSURF, and the
    # TERRA soil column (temperature at nine depth nodes, water as
    # column-integrated mass over the eight layers whose midpoints are
    # those interior nodes).  Selectors were authored from real
    # 2026-08-17 00Z bytes through the converged grib-core inventory.
    "icon-eu-regular-grib2-v1": _profile(
        "rw-wps-icon-eu-regular-grib2",
        source_format="grib2",
        mapping="141c635c9adc6690719fd52ff0fbcfe78d478a14e18cea8bb13df812359b3fc1",
        composition="6220800aa224b2ef8ae40d899760d8ed9100d0ce69a44fa06a80e039bdc74b2b",
        provenance="79ed94eee82ef0dc90b0e5b4ea79ce437f9006aa80d873bc297542002421fa57",
        data_role="icon_eu_invariant_surface",
        provenance_role="icon_eu_invariant_surface_provenance",
    ),
    # NCEP's operational AI global ensemble member state (Project EAGLE,
    # 0.25-degree GDT-0 GRIB2, 13 pressure levels), completed by a
    # same-cycle physical analysis through the CROSS-SOURCE composition:
    # the member product publishes no soil, no land mask, no orography,
    # no skin temperature and no 2 m humidity, so the mapping declares
    # those six gaps composition_bound and the composition binds them to
    # the packaged donor mapping under source_cycle_analysis_broadcast.
    # Every selector pins PDT 1 -- the individual-member template -- so
    # deterministic bytes and ensemble statistics refuse at the mapped
    # decode itself.  Member identity is verified separately by the
    # packaged rw-wps.members.v1 grammar (`gpuwm-member-prep`).
    # Selectors were authored from real 2026-08-17 00Z bytes through the
    # extended grib-core inventory.
    "aigefs-member-hybrid-grib2-v1": _profile(
        "rw-wps-aigefs-member-hybrid-grib2",
        source_format="grib2",
        mapping="2d5df9b66b465e72934b97d5f91904fb17bd5137ef885011831b4c698085d5bc",
        composition="9965904c92f08cd71323e2565d7a663863c8ca061f3e991b813b52ad2a6a5d10",
        provenance="a0439034d605b6215c16445c3f2459d2f027b87ac4d7228d4f1118d3cbf15d23",
        data_role="physical_analysis_surface_data",
        provenance_role="physical_analysis_surface_provenance",
        contributing_mappings={
            "physical_analysis_surface_mapping": {
                "file": "rw-wps-gdas-pgrb2-donor.mapping.json",
                "sha256": (
                    "35e0d4e2895b38a2702952fabc97fb9f53594b41dbb2d511"
                    "e39a13458f72f8cf"
                ),
            },
        },
    ),
    # RRFS -- HRRR's operational successor, flowing today on the
    # noaa-rrfs-ops-pds bucket and NOMADS rrfs/v1.0.  The 3 km CONUS grid
    # is bit-for-bit HRRR's Lambert (measured from real bytes: every
    # geolocating octet identical), so the entire HRRR wrfprs machinery
    # carries it as table data.  What is RRFS's own is also table data:
    # the 45-level pressure ladder (2 hPa top, 70 hPa where HRRR has 75,
    # no 1013.2 hPa entry) and the split of the state across a
    # prslev/2dfld file PAIR -- prslev is pure upper air, so the terrain
    # supplement and every surface/soil selector resolve in the 2dfld
    # files, which the caller passes alongside.  Selectors were authored
    # from real prototype-cycle bytes (2026-08-12 00Z) and cross-checked
    # against the live operational cycle (2026-08-17 00Z) through the
    # converged grib-core inventory: 19 of 19 fields, exactly one record
    # each per valid time, pair disjoint.
    "rrfs-prslev-2dfld-grib2-v1": _profile(
        "rw-wps-rrfs-prslev-2dfld-grib2",
        source_format="grib2",
        mapping="3139a3b93f7fea54f600e36e42115fd1e986e67e1df6462b5ef866b11f8d6c1a",
        composition="a632eae5203e92eeb4f6af5ee1fcfe2f24c1e03c7a5124f73a669fbb4e1ca6b1",
        provenance="e364f22ec917bfd991ec3bb756aa49b697d03c7cd425d40a35501dd769f279de",
        data_role="rrfs_prslev_2dfld_in_band_surface",
        provenance_role="rrfs_prslev_2dfld_in_band_surface_provenance",
    ),
})


#: Packaged ``rw-wps.members.v1`` documents: the ensemble member-
#: addressing grammars.  A members document is table data of the same
#: standing as a profile -- an ensemble source's member set, filename
#: patterns, byte-verification contract and statistic namespace are
#: rows in one JSON, pinned here by SHA-256, and adding an ensemble is
#: one document plus one row.  It is a separate table from
#: :data:`_PACKAGED_PROFILES` because the two answer different
#: questions (how to decode fields vs. which trajectory a file is), and
#: an ensemble can have a members grammar before its field mapping
#: exists -- which is exactly the state GEFS and AIGEFS ship in.
_PACKAGED_MEMBER_GRAMMARS = MappingProxyType({
    # NCEP GEFS v12, the 0.5-degree atmos pgrb2a/b (and 0.25-degree
    # pgrb2s) member files: 31 forecasts whose encoded ensemble size
    # says 30 (the octet excludes the control), whose control is flagged
    # low-resolution control (type 1), and whose mean/spread files
    # (geavg/gespr) share the member directories and filename pattern.
    # Every declared value was measured from real 2026-08-17 00Z bytes
    # through the extended grib-core inventory.
    "gefs-ensemble-grib2-members-v1": MappingProxyType({
        "file": "rw-wps-gefs-ensemble-grib2.members.json",
        "sha256": (
            "7342f58bb0c01c5c8a6b051a7d7619245a2f48743e3760cbfcff64de28f6ca7c"
        ),
    }),
    # NCEP AIGEFS (Project EAGLE), the operational AI global ensemble:
    # 31 members whose member identity is a PATH component (every leaf
    # filename is byte-identical across members), whose encoded ensemble
    # size says 31 (control included -- the opposite octet convention
    # from GEFS), and whose control carries NO control flag
    # (typeOfEnsembleForecast = 3 like every perturbed member;
    # perturbationNumber == 0 is the only discriminator).  Every
    # declared value was measured from real 2026-08-17 00Z bytes
    # through the extended grib-core inventory.
    "aigefs-ensemble-grib2-members-v1": MappingProxyType({
        "file": "rw-wps-aigefs-ensemble-grib2.members.json",
        "sha256": (
            "de8dceeb2f37efd351ae443efab6ce233540c4e78fc2c610bc93d88b656cc7ce"
        ),
    }),
})


def packaged_member_grammar_ids() -> tuple[str, ...]:
    """Every packaged members document this distribution ships, sorted."""

    return tuple(sorted(_PACKAGED_MEMBER_GRAMMARS))


def packaged_member_grammar_sha256(grammar_id: str) -> str:
    """One members document's immutable SHA-256, without touching disk."""

    return str(_member_grammar_row(grammar_id)["sha256"])


def _member_grammar_row(grammar_id: str) -> Mapping[str, object]:
    try:
        return _PACKAGED_MEMBER_GRAMMARS[grammar_id]
    except KeyError:
        raise KeyError(
            f"unknown packaged member grammar {grammar_id!r}; this "
            f"distribution ships {sorted(_PACKAGED_MEMBER_GRAMMARS)}"
        ) from None


def packaged_member_grammar(grammar_id: str) -> Path:
    """Resolve and byte-verify one packaged members document."""

    row = _member_grammar_row(grammar_id)
    path = (_AUTHORITY_ROOT / str(row["file"])).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"packaged member grammar {grammar_id} is missing: {path}")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != row["sha256"]:
        raise RuntimeError(
            f"packaged member grammar {grammar_id} hash differs: "
            f"expected {row['sha256']}, got {observed}")
    return path


def packaged_profile(profile_id: str) -> Mapping[str, object]:
    """The declaration for one packaged profile, or a useful refusal."""

    try:
        return _PACKAGED_PROFILES[profile_id]
    except KeyError:
        raise KeyError(
            f"unknown packaged source profile {profile_id!r}; this "
            f"distribution ships {sorted(_PACKAGED_PROFILES)}"
        ) from None


def packaged_profile_ids() -> tuple[str, ...]:
    """Every packaged profile this distribution ships, sorted."""

    return tuple(sorted(_PACKAGED_PROFILES))


def packaged_authorities(profile_id: str) -> Mapping[str, Path]:
    """Resolve and byte-verify one packaged profile's three authorities."""

    profile = packaged_profile(profile_id)
    names: Mapping[str, str] = profile["files"]        # type: ignore[assignment]
    expected: Mapping[str, str] = profile["sha256"]    # type: ignore[assignment]
    resolved: dict[str, Path] = {}
    for role in PROFILE_ROLES:
        path = (_AUTHORITY_ROOT / names[role]).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"packaged {profile_id} {role} authority is missing: {path}"
            )
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected[role]:
            raise RuntimeError(
                f"packaged {profile_id} {role} authority hash differs: "
                f"expected {expected[role]}, got {observed}"
            )
        resolved[role] = path
    return MappingProxyType(resolved)


def packaged_contributing_mappings(profile_id: str) -> Mapping[str, Path]:
    """Resolve and byte-verify one profile's contributing mapping documents.

    Empty for a profile whose composition declares no ``field_sources``
    bindings.  Each returned path is the packaged donor mapping the front
    door must pass as ``--contributing-mapping role=path``; the bytes are
    verified against the profile pin here, and the composition's own
    pinned digest re-verifies them at decode.
    """

    profile = packaged_profile(profile_id)
    declared: Mapping[str, Mapping[str, str]] = (
        profile["contributing_mappings"])    # type: ignore[assignment]
    resolved: dict[str, Path] = {}
    for role, pin in declared.items():
        path = (_AUTHORITY_ROOT / pin["file"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"packaged {profile_id} contributing mapping {role!r} is "
                f"missing: {path}"
            )
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != pin["sha256"]:
            raise RuntimeError(
                f"packaged {profile_id} contributing mapping {role!r} hash "
                f"differs: expected {pin['sha256']}, got {observed}"
            )
        resolved[role] = path
    return MappingProxyType(resolved)


def packaged_authority_sha256(profile_id: str) -> Mapping[str, str]:
    """One profile's immutable SHA-256 contract, without touching disk."""

    return packaged_profile(profile_id)["sha256"]   # type: ignore[return-value]


def packaged_contributing_sha256(profile_id: str) -> Mapping[str, str]:
    """The contributing-mapping pin table, without touching the filesystem."""

    profile = packaged_profile(profile_id)
    declared: Mapping[str, Mapping[str, str]] = (
        profile["contributing_mappings"])       # type: ignore[assignment]
    return MappingProxyType({
        role: str(row["sha256"]) for role, row in declared.items()
    })


def twentycrv3_authorities() -> Mapping[str, Path]:
    """Resolve and byte-verify the exact packaged 20CRv3 GRIB2 authorities."""

    return packaged_authorities("20crv3-member-grib2-v1")


def twentycrv3_authority_sha256() -> Mapping[str, str]:
    """Return the immutable SHA-256 contract without touching the filesystem."""

    return packaged_authority_sha256("20crv3-member-grib2-v1")


#: The GFS WPS Vtable `gpuwm adapt` documents as its worked example.
#:
#: It lived in `configs/`, which is not a package, so the wheel did not
#: carry it and a pip user following the documented adapt flow was told
#: to pass a file their install did not have.  It ships beside the
#: 20CRv3 authorities now, under the same recursive package-data glob
#: and the same byte contract, because it is the same kind of thing: an
#: immutable input a front door reads, not a config anyone edits.
_GFS_VTABLE_NAME = "Vtable.GFS.rw-wps"
#: Re-pinned 2026-07-30 to the committed bytes: 9e391880... -> ec8e615b...
#:
#: The old value was this file's CRLF form.  It is not a JSON file, so
#: the per-path ``gpuwm/authorities/*.json text eol=lf`` rule did not
#: cover it, and on a Windows clone (git-for-Windows defaults
#: core.autocrlf=true) it materialized with CRLF -- which is the
#: checkout the constant was taken from, and the wheel that was built
#: from it.  The same file on Linux, and in the object database, is LF,
#: so this gate could only ever have been true on one platform: the
#: byte contract it exists to enforce was itself platform-dependent.
#:
#: The repository now declares ``* -text``, so every checkout gets the
#: committed bytes and this is the one hash for all of them.  Nothing
#: was widened: the file is unchanged, the check is unchanged, and the
#: constant now names what the file actually is.
_GFS_VTABLE_SHA256 = (
    "ec8e615ba724b3ddf114c4c199a81083b3a17b4e1705055ec016f1769144090e")


def packaged_gfs_vtable() -> Path:
    """Resolve and byte-verify the packaged GFS WPS Vtable."""

    path = (_AUTHORITY_ROOT / _GFS_VTABLE_NAME).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"packaged GFS Vtable is missing: {path}")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != _GFS_VTABLE_SHA256:
        raise RuntimeError(
            f"packaged GFS Vtable hash differs: expected "
            f"{_GFS_VTABLE_SHA256}, got {observed}")
    return path


def packaged_gfs_vtable_sha256() -> str:
    """The immutable SHA-256 contract, without touching the filesystem."""

    return _GFS_VTABLE_SHA256


__all__ = [
    "PROFILE_ROLES", "packaged_authorities", "packaged_authority_sha256",
    "packaged_contributing_mappings", "packaged_contributing_sha256",
    "packaged_gfs_vtable", "packaged_gfs_vtable_sha256",
    "packaged_member_grammar", "packaged_member_grammar_ids",
    "packaged_member_grammar_sha256", "packaged_profile",
    "packaged_profile_ids", "twentycrv3_authorities",
    "twentycrv3_authority_sha256",
]
