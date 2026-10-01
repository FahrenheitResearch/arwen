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
import json
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
    input_normalizer: str | None = None,
    normalization: str | None = None,
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
    if (input_normalizer is None) != (normalization is None):
        raise ValueError(
            f"profile {stem} must declare an input normalizer NAME and the "
            "SHA-256 of the normalization document that defines it, or "
            "neither"
        )
    if input_normalizer is not None and composition_state != "composed":
        raise ValueError("an input normalizer requires a composed profile")
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
            **({"normalization": f"{stem}.normalization.json"}
               if input_normalizer else {}),
        }),
        "sha256": MappingProxyType({
            "mapping": mapping,
            "composition": composition,
            "provenance": provenance,
            **({"normalization": normalization} if input_normalizer else {}),
        }),
        "data_role": data_role,
        "provenance_role": provenance_role,
        "composition_state": composition_state,
        "contributing_mappings": MappingProxyType(contributing),
        # Absent for a profile whose source publishes bytes the mapped engine
        # already reads: their public declarations remain unchanged.
        **({"input_normalizer": input_normalizer} if input_normalizer else {}),
    })


_PACKAGED_PROFILES = MappingProxyType({
    # Native global ICON carries its coordinates in separate GDT-101
    # CLAT/CLON records. Normalize before the existing mapped authority is
    # authored, never disguise its unstructured array as an embedded grid.
    # A173 moved the mapping and normalization digests: DWD posts every
    # field hourly to f078 (listed 2026-10-01 00Z), so both now declare the
    # publisher's 1 h spacing and take its whole multiples; 3 h stays the
    # default (the registry row and the route's default_cadence).
    "icon-global-grib2-v1": _profile(
        "rw-wps-icon-global-grib2",
        source_format="grib2",
        mapping="7bf357241b1be4682df42371a2c722db1437b8df9d004d1e7851e00a61e7853f",
        composition="a75dc9deabf72d750eb5d3f333274ed85d2d190e69a09f5aaf08f5c44891604b",
        provenance="1770e4b1c4092d53db0ee5b18aa371465215c04325f4de2610f0d4f740eb0b09",
        data_role="icon_global_invariant_surface",
        provenance_role="icon_global_invariant_surface_provenance",
        input_normalizer="icon-gdt101-pressure-v1",
        normalization="d1fc9b43d7c4815cbb1f09e0a456a68097310718643a66b9b887001cc0106e07",
    ),
    # DWD's 2.2 km ICON-D2 on its native R19B07 mesh (542,040 cells), the
    # same GDT-101 normalization with its own mesh, ladder and cadence rows.
    # DWD also publishes a regular-lat-lon ICON-D2, but that product masks
    # the 17 % of its bounding box outside the model domain in every field,
    # and the mapped route validates each field over its whole extent, so
    # the native mesh is the product the route can read.  Selectors were
    # authored from real 2026-09-27 12Z bytes (grib_ls inventory).
    "icon-d2-grib2-v1": _profile(
        "rw-wps-icon-d2-grib2",
        source_format="grib2",
        mapping="26900d41048587a750f71a104d2a9d36c05c556f5871e8025a745bb0b1ff6f47",
        composition="a4b05c684fee8db595ea2b16f4cb2ba8af807415af4e494b57f948061268e8a6",
        provenance="0241a993bd971863aed47e34fba254448f3c301f1f362a7ef7228701a4d58559",
        data_role="icon_d2_invariant_surface",
        provenance_role="icon_d2_invariant_surface_provenance",
        input_normalizer="icon-d2-gdt101-model-level-v1",
        normalization="7e4d50967e93c0df585cc2f9a18f331b2d2b90217fd7891c9d6ef6a5234be150",
    ),
    "20crv3-member-grib2-v1": _profile(
        "rw-wps-20crv3-member-grib2",
        source_format="grib2",
        mapping="75089fd4973e5e9f246f2dc63f9862d7d625e28a02e6b7e56133c1753101e648",
        composition="aa4f3fac03c09e8461c5e6c5e04a6bed48b5ad477babc4c75e8dd10fd92fe7b2",
        provenance="d1248e1b091f59841757a98a024cbe2868cebc25308f4eb4f9608e2c1755f3b1",
        data_role="twentycrv3_in_band_surface",
        provenance_role="twentycrv3_in_band_surface_provenance",
    ),
    "20crv3-netcdf-v1": _profile(
        "rw-wps-20crv3-netcdf",
        source_format="netcdf",
        mapping="76520d6a6181c71135c350233c4266ce5ae756c258feb98883f5aa129caaa6e1",
        composition="2c243fe4c4dba1c8f47178f2be583f3a20148d54d77101ee3421d1824d10b1c5",
        provenance="8daeb53502d28483a049936262910004cfda17aa5030cc3066d3bd01413d3066",
        data_role="twentycrv3_netcdf_recovered_invariant",
        provenance_role="twentycrv3_netcdf_recovered_invariant_provenance",
    ),
    # HRRR's public pressure-level product (wrfprs), decoded through the
    # generic mapped route: the Lambert CONUS grid, grid-relative wind
    # rotation and nine-node RUC soil are all TABLE DATA in these three
    # documents.  Selectors were authored from real 2026-08-15 00Z bytes
    # through the converged grib-core inventory.  Cloud ice also lists
    # CICE (0/6/0), the identity HRRR published it under before HRRRv3
    # (July 2018), after CIMIXR (0/1/82), read from real 2017-01-19 00Z
    # bytes: with CIMIXR alone every earlier cycle was refused at
    # preparation for lacking cloud_ice_mixing_ratio.
    "hrrr-prs-grib2-v1": _profile(
        "rw-wps-hrrr-prs-grib2",
        source_format="grib2",
        mapping="1bb2dd3f91bb0c645d4256cc23d7827bd7f6ba17eaf8da4d4fa4caa590ac8d61",
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
        mapping="5bc58e43b2cb997c2946aa28b2211ffca0a778c5543642c45dda919a94158de0",
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
        mapping="093f2286700f539baa33b425e0d2a2f30a8622bbff3918f588c189b2c1c3f9fa",
        composition="13d7e4ca06f8012cfd252c03d70d9fae69d2e18b9b51975445d70a506e1d90ed",
        provenance="823e07b38677e3c0c83da984637a4fda83d1eb09be2401cb4e0a9e433820de22",
        data_role="gdps_analysis_invariant_surface",
        provenance_role="gdps_analysis_invariant_surface_provenance",
    ),
    # ECMWF's AIFS single deterministic forecast (open data, 0.25-degree
    # GDT-0 GRIB2): the reduced AI-model field set as TABLE DATA.  The
    # north-to-south row order, the geopotential-to-metres terrain scale
    # and the two-layer ordinal (type 151) soil column are rows in these
    # documents.  The land mask and surface geopotential ride the step-0
    # object alone (the f012 object carries neither), so the mapping
    # declares both composition_bound and the composition binds them from
    # the SAME cycle's step-0 object through a fourth pinned authority,
    # the step-0 donor mapping (this table under its own name, the land
    # mask without a time binding), on the source_cycle_analysis_broadcast
    # clock: every lead of every window reads the cycle's own statics,
    # and the route table fetches that object beside a window starting
    # later.  Selectors were authored from real 2026-08-17 00Z bytes
    # through the converged grib-core inventory.  Earlier AIFS cycles
    # publish pressure-level geopotential and no geopotential height, so
    # the height field lists those records second, scaled to metres.
    "aifs-single-grib2-v1": _profile(
        "rw-wps-aifs-single-grib2",
        source_format="grib2",
        mapping="1dec5755c55c2a527e7545bc6add7ce11beddc9fdd51c7f6ea95a8c7083d1a85",
        composition="eccaa63a5b025c91378f005d88d93b411e9fc07724c75328a2460065590abbf7",
        provenance="26e16197183f4d2b6dffc9c732b9d18933ae26118b40d61617663a1355557f2c",
        data_role="aifs_single_in_band_surface",
        provenance_role="aifs_single_in_band_surface_provenance",
        contributing_mappings={
            "aifs_single_step0_donor_mapping": {
                "file": "rw-wps-aifs-single-step0-donor.mapping.json",
                "sha256": (
                    "99adc3a8894c2d19ba37d6da6d53418a95dcf32c2b1094b86"
                    "e8ccd1feecb7b02"
                ),
            },
        },
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
        mapping="95ff9698d9db7f346f3973333936e58fda5860c9f2e4f76da76b2ed2c1284185",
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
    # local-table row selected by octets, and the accuracy facts ride the
    # provenance document: even f000 is stamped a forecast in the bytes,
    # and the one analysis-stamped product has no land surface at all.
    # Selectors were authored from real 2026-08-17 06Z bytes through the
    # converged grib-core inventory.
    "gdas-pgrb2-0p25-grib2-v1": _profile(
        "rw-wps-gdas-pgrb2-0p25-grib2",
        source_format="grib2",
        mapping="eee8342a5a9a57f17d267dd48ad5edbb1a72f6d34c8803bd2ba1010668e0268d",
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
    # RUNNABLE by borrowing the six canonicals AIGFS does not publish --
    # terrain, land mask, skin temperature, 2 m humidity, and the
    # four-layer soil column -- from the SAME CYCLE's GDAS 0.25-degree
    # analysis through the cross-source composition (field_sources,
    # source_cycle_analysis_broadcast clock).  Surface pressure, which
    # AIGFS does not publish either, is derived at every lead from its own
    # mean-sea-level pressure at the analysis terrain height, so the
    # column mass follows the forecast instead of holding the analysis
    # value.  The donor
    # decodes through its own mapping, shipped here as a fourth pinned
    # authority: the checked-in GFS pressure-level table with one
    # table-data change (2 m specific humidity DIRECTLY selected, because
    # a borrowed field must be directly selected in the donor).  Proven on
    # real 2026-08-17 00Z NOMADS + GDAS bytes.
    "aigfs-gdas-hybrid-grib2-v1": _profile(
        "rw-wps-aigfs-gdas-hybrid-grib2",
        source_format="grib2",
        mapping="64cab09660beffc3b4cd35a8fc7a711e35c3034bfe930362c72a3ef68581a320",
        composition="e11db42390a6b1f08b2edcd7d25e4574dc5be74021748e65c3c7b1115c34d07c",
        provenance="72299bca17576bce3a0bedfe846806dbde31d126781d4d3ebaafb9dbf78318ed",
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
        mapping="2445d2a0b985c24096a729e5f7b443613d5da579828627fba239356c6efbf668",
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
        mapping="ae8075c069206c4f3917830aa769a2c4ef8f39a03ae31b3d7ee8c31c63f23cea",
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
    # Surface pressure is derived at every lead from the member's own
    # mean-sea-level pressure at the analysis terrain height: the NOMADS
    # sfc product carries no surface pressure, and the record the AWS
    # mirror appends sits on the AI model's own orography, not on the
    # terrain the composition pairs it with.
    # Every selector pins PDT 1 -- the individual-member template -- so
    # deterministic bytes and ensemble statistics refuse at the mapped
    # decode itself.  Member identity is verified separately by the
    # packaged rw-wps.members.v1 grammar (`gpuwm-member-prep`).
    # Selectors were authored from real 2026-08-17 00Z bytes through the
    # extended grib-core inventory.
    "aigefs-member-hybrid-grib2-v1": _profile(
        "rw-wps-aigefs-member-hybrid-grib2",
        source_format="grib2",
        mapping="d34ff39693af3a6cef6d4f3d83a72c7bff1d710470518d986724eae368bde623",
        composition="9965904c92f08cd71323e2565d7a663863c8ca061f3e991b813b52ad2a6a5d10",
        provenance="4050afea573ffa3b590deb6ef0f0e2a86f2d6feca25423b38cd31f4e025b6bb2",
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
        mapping="6b89a9d1eed5ad8868505e5560301599736837340ca2fb8d87dd4fdf872d68cb",
        composition="a632eae5203e92eeb4f6af5ee1fcfe2f24c1e03c7a5124f73a669fbb4e1ca6b1",
        provenance="e364f22ec917bfd991ec3bb756aa49b697d03c7cd425d40a35501dd769f279de",
        data_role="rrfs_prslev_2dfld_in_band_surface",
        provenance_role="rrfs_prslev_2dfld_in_band_surface_provenance",
    ),
    # ERA5 on its NATIVE 137 hybrid sigma-pressure model levels -- the
    # first shipped profile whose vertical coordinate is not a pressure
    # ladder.  The A/B interface coefficients ride IN BAND as the GRIB2
    # Section-4 coordinate-values (pv) octets, so nothing about this
    # source needs a per-model channel: the decode reads the 276 values
    # the producer wrote, materializes p = A + B*ps, and integrates
    # geopotential height hydrostatically from the borrowed terrain.
    # The model-level product publishes the prognostic atmosphere and
    # NOTHING else -- no soil, no land mask, no orography, no skin
    # temperature, no 2 m or 10 m diagnostics, and surface pressure only
    # as lnsp (which affine-only unit transforms cannot exponentiate) --
    # so the same-hour ERA5 pressure-level/single-level analysis is the
    # cross-source donor for all ten of those fields, decoded through
    # its own pinned mapping.  It also lends the water state the
    # pressure-level route reads from the same file -- sea surface
    # temperature, sea ice and the lake model's water and ice -- so the
    # two routes assemble the same water temperature from one donor
    # instead of this one giving every lake its skin temperature, and
    # the snow water equivalent, which this route otherwise started at
    # zero everywhere.
    # Proven on real 2026-05-30 CDS bytes: both
    # engines byte-identical on air_pressure and geopotential_height
    # (max ULP 0) and a preparation to rc 0.
    "era5-model-level-l137-grib2-v1": _profile(
        "rw-wps-era5-model-level-l137-grib2",
        source_format="grib2",
        mapping="65cfca7fc9fc6be102c796bd481126fe9e22491df8342fbaa1ff54b087fb09c6",
        composition="eab1fcd5c10a746ebdaf269e910e08ea77a40d0db9789539423cbda477201e62",
        provenance="3c6477f94c1ab3428c5f3f1d6fb29c57ca9620dfecf41d600ca165df8609da7c",
        data_role="physical_analysis_surface_data",
        provenance_role="physical_analysis_surface_provenance",
        contributing_mappings={
            "physical_analysis_surface_mapping": {
                "file": "rw-wps-era5-plev-surface-donor.mapping.json",
                "sha256": (
                    "a16c6a9b615d5999030051aa3fb2170de7993467580c4f44"
                    "840eebd0f1ec5e69"
                ),
            },
        },
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
    # (typeOfEnsembleForecast like every perturbed member: 6 as NOMADS
    # serves it, 3 in the AWS mirror's rewritten copy;
    # perturbationNumber == 0 is the only discriminator), and whose
    # mirror serves part of its archive under another writer's octets
    # (declared per class as rewrites).  Every declared value was
    # measured from real bytes through the extended grib-core inventory.
    "aigefs-ensemble-grib2-members-v1": MappingProxyType({
        "file": "rw-wps-aigefs-ensemble-grib2.members.json",
        "sha256": (
            "7d8237d0ad3d5010380527beaeea9ea8f68c3b4154a37bb4394dc0e71d21795c"
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


def packaged_normalizer_ids() -> tuple[str, ...]:
    """Every input normalizer this distribution ships, sorted."""

    return tuple(sorted(
        str(row["input_normalizer"]) for row in _PACKAGED_PROFILES.values()
        if row.get("input_normalizer")))


def _normalizer_profile(name: str) -> str:
    for profile_id, row in _PACKAGED_PROFILES.items():
        if row.get("input_normalizer") == name:
            return profile_id
    raise KeyError(
        f"unknown packaged input normalizer {name!r}; this distribution "
        f"ships {list(packaged_normalizer_ids())}"
    )


def packaged_normalization(name: str) -> Path:
    """Resolve and byte-verify one packaged normalization document.

    The argument is the NORMALIZER NAME a profile declares, not a path: a
    name this distribution does not ship never becomes a file read.
    """

    profile_id = _normalizer_profile(name)
    profile = _PACKAGED_PROFILES[profile_id]
    file_name = profile["files"]["normalization"]      # type: ignore[index]
    expected = profile["sha256"]["normalization"]      # type: ignore[index]
    path = (_AUTHORITY_ROOT / str(file_name)).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"packaged {profile_id} normalization authority is missing: {path}")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise RuntimeError(
            f"packaged {profile_id} normalization authority hash differs: "
            f"expected {expected}, got {observed}")
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


#: Parsing is cached by the verified bytes, never by mutable filesystem times.
_COMPOSITIONS: dict[str, tuple[str, Mapping[str, object]]] = {}


def packaged_composition(profile_id: str) -> Mapping[str, object]:
    """Verify every authority and parse the exact verified composition bytes."""
    profile = packaged_profile(profile_id)
    names = profile["files"]
    expected = profile["sha256"]
    composition = None
    for role in PROFILE_ROLES:
        path = (_AUTHORITY_ROOT / names[role]).resolve()
        data = path.read_bytes()
        observed = hashlib.sha256(data).hexdigest()
        if observed != expected[role]:
            raise RuntimeError(f"packaged {profile_id} {role} authority hash differs: "
                               f"expected {expected[role]}, got {observed}")
        if role == "composition":
            composition = data
    digest = hashlib.sha256(composition).hexdigest()
    cached = _COMPOSITIONS.get(profile_id)
    if cached is None or cached[0] != digest:
        _COMPOSITIONS[profile_id] = (digest, MappingProxyType(json.loads(composition)))
    return _COMPOSITIONS[profile_id][1]


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


#: The mapping target key that widens ``boundary_interval_seconds`` from
#: the one spacing a series must have to the finest one it may have.
BOUNDARY_MULTIPLES_KEY = "accept_boundary_interval_multiples"


def boundary_interval_refusal(target: Mapping[str, object], seconds: int,
                              *, subject: str = "mapped cadence") -> str | None:
    """Why a mapping target refuses a uniform boundary interval, or None.

    ``boundary_interval_seconds`` names the publisher's own spacing.  A
    target that also declares :data:`BOUNDARY_MULTIPLES_KEY` takes a
    uniform series at any whole multiple of that spacing: every valid
    time in such a series is one the publisher wrote, and no mapped field
    depends on the spacing, because the mapping grammar has no
    time-window statistic to bind one to.  Without the key the declared
    spacing is the only one the target takes, which is what every
    mapping that does not declare it keeps.

    ONE function for the decode's frame check, the direct export's
    contract check and the doors that write a cadence, so a cadence one
    of them accepts cannot be refused by another.
    """

    declared = target.get("boundary_interval_seconds")
    if target.get(BOUNDARY_MULTIPLES_KEY) is True:
        declared = int(declared)
        if seconds > 0 and seconds % declared == 0:
            return None
        return (f"{subject} {seconds} seconds is not a whole multiple of "
                f"the {declared} seconds the target contract declares")
    if declared is not None and int(declared) == seconds:
        return None
    return f"{subject} {seconds} seconds differs from target contract {declared!r}"


def boundary_interval_takes(target: Mapping[str, object]) -> str:
    """The spacings a mapping target takes, as a refusal sentence says them.

    Read by every door that refuses a cadence on a preparation's behalf,
    so each says what :func:`boundary_interval_refusal` decides.
    """

    spacing_h = int(target["boundary_interval_seconds"]) / 3600
    if target.get(BOUNDARY_MULTIPLES_KEY) is True:
        return f"any whole multiple of {spacing_h:g} h"
    return f"{spacing_h:g} h and no other spacing"


#: Parsed mapping targets, keyed by profile and the verified bytes' digest.
_TARGETS: dict[str, tuple[str, Mapping[str, object]]] = {}


def packaged_mapping_target(profile_id: str) -> Mapping[str, object]:
    """One packaged profile's verified mapping ``target`` contract."""

    path = packaged_authorities(profile_id)["mapping"]
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    cached = _TARGETS.get(profile_id)
    if cached is None or cached[0] != digest:
        target = json.loads(data).get("target")
        if not isinstance(target, dict):
            raise RuntimeError(
                f"packaged {profile_id} mapping declares no target contract")
        _TARGETS[profile_id] = (digest, MappingProxyType(target))
    return _TARGETS[profile_id][1]


#: Mapping ``target`` keys that decide which requests a mapping admits and
#: nothing it decodes, each with its reason.  A preparation copies the
#: mapping it ran into its evidence and binds that document's digest
#: everywhere (manifest, composition receipt, cache identity), so when a
#: release adds or changes one of these keys every earlier preparation's
#: digest stops matching the packaged pin although its frames are the
#: same bytes (A166: A159 added the first key below to ten mappings).
#:
#: - :data:`BOUNDARY_MULTIPLES_KEY`: which uniform boundary spacings a
#:   series may have.  Frames are decoded one valid time at a time by the
#:   same ``fields`` table, and the mapping grammar has no time-window
#:   statistic, so a frame does not depend on it; the spacing a
#:   preparation was made at is still held to the packaged target by
#:   :func:`bound_mapping_refusal`.
#: - ``boundary_interval_seconds``: the spacing those multiples count
#:   from (A173 moved icon-global's from 3 h to the 1 h DWD posts).  For
#:   the same reason it decides which series are admitted and no frame:
#:   a frame is one valid time, and a preparation's own spacing is held to
#:   the packaged target, so a 3 h icon-global preparation made before
#:   A173 is still icon-global's.
#:
#: A key joins this set only with the reason it cannot change a frame.
ADMISSION_ONLY_TARGET_KEYS = frozenset({BOUNDARY_MULTIPLES_KEY,
                                        "boundary_interval_seconds"})


def mapping_decode_identity(document: Mapping[str, object]) -> str:
    """SHA-256 of a mapping document without its admission-only keys.

    Canonical JSON of the parsed document (sorted keys, no whitespace), so
    two documents with this digest equal parse to the same fields,
    derivations, grid and target, which is everything the engine decodes
    from; only :data:`ADMISSION_ONLY_TARGET_KEYS` are left out.
    """

    stripped = dict(document)
    target = stripped.get("target")
    if isinstance(target, Mapping):
        stripped["target"] = {key: value for key, value in target.items()
                              if key not in ADMISSION_ONLY_TARGET_KEYS}
    canonical = json.dumps(stripped, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


#: Packaged mappings' decode identities, keyed by profile and pinned digest.
_DECODE_IDENTITIES: dict[str, tuple[str, str]] = {}


def _packaged_mapping_decode_identity(profile_id: str) -> str:
    pinned = packaged_authority_sha256(profile_id)["mapping"]
    cached = _DECODE_IDENTITIES.get(profile_id)
    if cached is None or cached[0] != pinned:
        path = packaged_authorities(profile_id)["mapping"]
        _DECODE_IDENTITIES[profile_id] = (
            pinned, mapping_decode_identity(json.loads(path.read_bytes())))
    return _DECODE_IDENTITIES[profile_id][1]


def bound_mapping_refusal(profile_id: str, bound: bytes, *,
                          spacings_seconds=()) -> str | None:
    """Why a preparation's bound mapping is not this profile's, or None.

    ``bound`` is the mapping document a preparation copied into its
    evidence.  The packaged bytes resolve.  A document that differs from
    the packaged one only in :data:`ADMISSION_ONLY_TARGET_KEYS` resolves
    too, when every boundary spacing the preparation was made at
    (``spacings_seconds``) is one the packaged target takes: its frames
    are the packaged mapping's frames.  Anything else is refused, because
    a frame the packaged mapping would not decode is not this profile's
    preparation.
    """

    pinned = packaged_authority_sha256(profile_id)["mapping"]
    digest = hashlib.sha256(bound).hexdigest()
    if digest == pinned:
        # The packaged bytes: the decode held the series to this very
        # target when it ran.
        return None
    try:
        document = json.loads(bound)
    except (UnicodeError, ValueError):
        return f"its mapping ({digest}) is not a JSON document"
    if (not isinstance(document, dict)
            or mapping_decode_identity(document)
            != _packaged_mapping_decode_identity(profile_id)):
        return (f"its mapping ({digest}) decodes differently from the "
                f"packaged one ({pinned})")
    target = packaged_mapping_target(profile_id)
    for seconds in sorted(set(spacings_seconds)):
        reason = boundary_interval_refusal(
            target, int(seconds), subject="its boundary spacing")
        if reason is not None:
            return reason
    return None


__all__ = [
    "ADMISSION_ONLY_TARGET_KEYS", "bound_mapping_refusal",
    "mapping_decode_identity",
    "BOUNDARY_MULTIPLES_KEY", "boundary_interval_refusal",
    "boundary_interval_takes", "packaged_mapping_target",
    "PROFILE_ROLES", "packaged_authorities", "packaged_authority_sha256",
    "packaged_normalization", "packaged_normalizer_ids",
    "packaged_composition",
    "packaged_contributing_mappings", "packaged_contributing_sha256",
    "packaged_gfs_vtable", "packaged_gfs_vtable_sha256",
    "packaged_member_grammar", "packaged_member_grammar_ids",
    "packaged_member_grammar_sha256", "packaged_profile",
    "packaged_profile_ids", "twentycrv3_authorities",
    "twentycrv3_authority_sha256",
]
