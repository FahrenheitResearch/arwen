"""``gpuwm fetch-bridges``: stage the prebuilt Rust artifacts.

A PLATFORM wheel now ships the compiled Rust itself: the artifacts below
are staged into ``gpuwm/libexec/bridges`` by
``tools/stage_wheel_bridges.py`` before the wheel is built, so on a
platform a bundle exists for, ``pip install gpuwm`` is already complete
and this command has nothing left to do.

It remains the route for everyone else: the ``py3-none-any`` fallback
wheel, which pip resolves wherever no platform bundle is published,
carries no binaries at all.  Before either existed, the only way to get
the GRIB decoders, the CPU preprocessing library, the fetch backbone,
the batch renderer, the two radar front doors, the MRMS, Stage-IV,
surface, GOES and European-composite front doors, the NetCDF decoder,
the mapped decode engine and the observation remap onto a wheel install
was to clone the repository and run ``cargo build`` three times -- a
Rust toolchain, a 2.5 GB checkout and a few minutes of compiling, for
twenty-six files.
``gpuwm fetch-bridges`` is the same trade :mod:`gpuwm.table_assets`
already makes for the externalized physics tables: the artifacts are
published as versioned GitHub release assets, their exact size and
SHA-256 pins are packaged inside the wheel at release time, and every
byte is verified against those pins *before* anything is installed.

What is staged, and where
-------------------------
One bundle per platform, holding the twenty-six artifacts of
:data:`BUNDLED_ARTIFACTS`, staged into :func:`gpuwm.bridges
.default_bridge_dir` (``~/.gpuwm/bridges``) -- the last rung of the
resolution ladder every consumer already searches, so nothing else in
gpuwm needs to know this command exists.  ``--dest DIR`` stages
somewhere else; the per-artifact environment variables keep overriding
everything, and a set override is reported rather than silently
shadowed.

A bundle also carries the renderer's map assets -- the Natural Earth
and US Census shapefiles ``rw_wrfbatch`` draws coastlines, borders,
state lines and counties from -- under :data:`ASSET_ROOT`, staged
beside the binaries as ``<dest>/assets/basemap/...``.  That is not a
new lookup: ``rw_wrfbatch`` already resolves ``assets/basemap`` under
its own ancestors, so a renderer staged at ``~/.gpuwm/bridges`` finds
the shapefiles itself, with no environment variable set and no Python
in the loop.  They travel in the bundle rather than the wheel because
the wheel has no room, and that is measured rather than estimated:
as of 2026-08-17 the platform wheels are 108.70 MB (win_amd64) and
111.70 MB (manylinux_2_28_x86_64) against PyPI's 100 MB per-file cap --
already over it before a byte of basemap, which deflates to 21.1 MB.
The published pair is the pure one (91.93 MB wheel, 95.25 MB sdist);
see ``tools/stage_wheel_bridges.py``'s docstring for the full table and
what makes the platform pair uploadable.  So the binaries ship in the wheel and the
basemaps do not -- which is the one place the "arrive by one mechanism"
rule bends, and the reason a wheel-only install renders fields without
the cartographic overlay until this command runs.

Platform support is a capability check on the OS and machine
architecture -- can this box run the bytes in that bundle -- and never
an identity gate.  A platform with no published bundle is told so by
name and handed the build-from-source route, which remains the
universal one.

The staging contract
--------------------
Every artifact is verified three ways before it is installed: the exact
byte count, the SHA-256 pin packaged with this release, and (for the
decoders that declare one) the contract marker
:data:`gpuwm.bridges.BRIDGE_ABI_MARKERS` -- the same static check
``gpuwm doctor`` applies to a binary already on disk.  A staged file
that fails any of the three is deleted and refused, never installed.
Extraction reads members by their exact pinned filename, so an archive
cannot place a byte anywhere the pins did not name.

Where this deliberately differs from ``fetch-tables``: an existing file
whose bytes do not match the pin is *replaced* here rather than
refused.  A physics table is a fixed asset, so a wrong one on disk is
the operator's to explain; a bridge executable is a versioned build,
and yesterday's copy sitting in ``~/.gpuwm/bridges`` after a Python-half
upgrade is exactly the skew that made 1.1.0 preparations fail with a
message blaming the file gpuwm had just written correctly.  Replacing
it is the point of the command.  The replacement is still gated on the
new bytes passing all three checks first.

Offline and mirrors
-------------------
``--from DIR`` stages from a local directory under identical
verification: either the bundle archive itself, or the twenty-six
artifacts loose in that directory (what an air-gapped operator has
after building them on a machine that does have a toolchain).
``GPUWM_BRIDGE_ASSET_URL_BASE`` overrides the download base URL; the
bundle filename is appended to it either way.

Pins are generated at release time
----------------------------------
The bundles are built by the release workflow from the same commit as
the wheel, and ``tools/build_bridge_bundle.py`` computes the pins from
those exact bytes and writes them into :data:`PINS_RESOURCE` before the
wheel is built.  A tree that has not been through that step carries a
pins document declaring no platforms, and this command says so and
refuses rather than inventing a hash to check against.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform as platform_module
import shutil
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile

from gpuwm import bridges
from gpuwm.explain import warn

#: Schema of the packaged pins document.
PINS_SCHEMA = "gpuwm-bridge-pins-v1"

#: Schema of the manifest published beside the bundles as a release
#: asset (the human/machine-readable index of what a release shipped).
BUNDLE_MANIFEST_SCHEMA = "gpuwm-bridge-bundle-manifest-v1"

#: The packaged pins document, relative to the ``gpuwm`` package.
PINS_RESOURCE = "data/bridges/bridge-pins.json"

#: Platform keys a release publishes bundles for.  A key names an OS and
#: a machine architecture and nothing else: it answers "can this box run
#: those bytes", never "who is asking".
SUPPORTED_PLATFORMS = ("win-x86_64", "linux-x86_64")

#: Override the download base URL (private mirrors, tests).  The bundle
#: filename is appended to it.
ASSET_URL_BASE_ENV = "GPUWM_BRIDGE_ASSET_URL_BASE"

#: Subdirectory of the staging destination holding in-flight and
#: verified bundle archives (kept only under ``--keep-bundle``).
ARCHIVE_SUBDIR = ".fetch-bridges"

#: Prefix, inside a bundle and inside the staging destination, of every
#: data file the renderer reads.  ``rw_wrfbatch`` searches
#: ``assets/basemap`` under the first eight ancestors of its own
#: directory, so staging under this prefix puts the shapefiles where the
#: binary already looks -- the reason the asset half needs no new
#: resolution logic, no environment variable, and no cooperation from
#: the Python half at all.
ASSET_ROOT = "assets"

#: The asset subtrees a bundle is required to carry, relative to
#: :data:`ASSET_ROOT`.  Directories, never filenames: the file list is
#: generated from the tree at pack time and pinned by hash, so it cannot
#: drift the way a hand-maintained enumeration does.  This tuple is the
#: declaration a release is checked against.
REQUIRED_ASSET_SUBDIRS = ("basemap",)

#: Byte marker every bundled artifact's build embeds, followed by the
#: 40-hex git commit the binary was built from.  The 1.4.1 cut nearly
#: shipped bundles predating the source tip -- the two platform zips
#: were not even the same revision -- and every check on the cut path
#: hashed whatever bytes it was handed, so a stale binary pinned and
#: verified perfectly.  The stamp is what makes "built from the release
#: commit" a property of the bytes themselves: the cut extracts it and
#: refuses a mismatch without executing anything (the same static
#: string-in-the-binary convention as :data:`gpuwm.bridges
#: .BRIDGE_ABI_MARKERS` and the renderer's ``GPUWM_INITIAL_*``
#: attribute literals).  The Rust half lives in the workspace build
#: scripts (``tools/grib1_bridge/build.rs`` and the ``build.rs`` of the
#: six bundled ``tools/rustwx`` crates), which inject the checkout's
#: HEAD as ``GPUWM_BRIDGE_SOURCE_REV`` for each entry point to embed.
#:
#: Asked of every artifact whose source moves with this checkout, which
#: is every artifact gpuwm wrote.  A ``vendored`` artifact is proved
#: instead by its contract marker -- see :class:`BundledArtifact`.
SOURCE_REV_MARKER = b"GPUWM_BRIDGE_SOURCE_REV="

#: Character length of a full git commit hash, which is what the stamp
#: carries -- an abbreviation can collide, a branch name can move.
_SOURCE_REV_LENGTH = 40

_BLOCK_BYTES = 8 * 1024 * 1024
_USER_AGENT = "gpuwm-fetch-bridges/1"
_TIMEOUT_S = 120


class BridgeAssetError(RuntimeError):
    """A refusal: wrong bytes, unfetchable bundle, or unusable source."""


# ---------------------------------------------------------------------------
# What a bundle carries
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BundledArtifact:
    """One built file a bundle carries, and what it is for.

    ``kind`` is ``"executable"`` or ``"library"``: the two spellings a
    platform gives a built artifact, which is the whole reason the
    filename cannot be derived from the logical name alone.

    ``vendored`` says the crate is a verbatim copy of an upstream tree
    frozen at a recorded commit, and it changes what proves the bytes are
    not stale.  A gpuwm-authored artifact is proved by
    :data:`SOURCE_REV_MARKER`: its source moves with this checkout, so the
    release commit stamped into the binary is the statement that the two
    agree.  A vendored crate does not move with this checkout at all --
    re-vendoring is a commit that edits ``UPSTREAM_COMMIT`` and the
    directory together -- so a HEAD stamp would say nothing about it, and
    injecting one would mean editing a tree whose whole claim is that no
    file in it differs from upstream by a byte.  What is asked of a
    vendored artifact instead is its declared contract marker
    (:data:`gpuwm.bridges.BRIDGE_ABI_MARKERS`), which is a property of the
    bytes and names the ABI this release speaks, on top of the size and
    SHA-256 every artifact is pinned by.
    """

    name: str
    kind: str
    crate: str
    env_var: str
    consumer: str
    vendored: bool = False


#: The twenty-six artifacts a bundle carries, in build order: the five
#: GRIB decoders and the CPU preprocessing library from the decoder
#: workspace, then the fetch backbone, the batch renderer, the two radar
#: front doors, the five observation front doors, the NetCDF decoder,
#: the NetCDF writer, the static-field builder and the observation remap
#: from the renderer workspace, then the region-global dealiasing library
#: from its own vendored crate and the mapped decode engine from the
#: engine workspace.  The environment variables are
#: the ones the resolution ladder already honours, so a staged bundle
#: and a hand-built tree are found by exactly the same code.
#:
#: ``rw_odim`` joined for the reason ``rw_nexrad`` did, one continent
#: over.  European polar-volume ingest had a complete library path, a
#: ``gpuwm obs radar`` front door, and no way to obtain the binary that
#: path drives without a Rust toolchain and a source checkout -- which by
#: this project's rule means it was not shipped: a capability the engine
#: has and the user cannot reach does not exist.  It builds ``--locked
#: --offline`` from the same vendored closure on the same two platforms
#: as its siblings and carries the same source-revision stamp, so
#: bundling it costs an entry here and nothing else.
#:
#: ``rw_opera`` was the one door still left out when these two lanes were
#: written apart, and for a reason that stopped being true when they were
#: put together: ``crates/rw-obs`` carried no ``build.rs`` and none of its
#: entry points embedded the source-revision stamp, so a cut could not
#: prove the binary it staged.  The observation-front-door lane added that
#: ``build.rs`` and stamped three of the crate's four entry points; the
#: fourth, ``opera.rs``, did not exist on that lane.  Stamping it here is
#: the whole remaining cost, so the composite ships with its siblings
#: rather than waiting for a lane of its own.
#:
#: ``rw_nexrad`` joined this list once it was clear that leaving it out
#: made ``gpuwm doctor`` pass on boxes that could not ingest a single
#: radar observation: it is the only route into the radar-DA nowcast,
#: it has no fallback the way the fetch backbone and the renderer do,
#: and it builds ``--locked --offline`` out of the same vendored
#: closure on the same two platforms as its two workspace siblings.
#: The alternative -- telling every user to install a Rust toolchain --
#: is the one prerequisite this project otherwise never imposes.
#:
#: ``rw_mrms``, ``rw_stage4``, ``rw_asos`` and ``rw_goes`` joined for
#: the same reason, one wave later and one wave too late.  All four were
#: written, tested and committed; all four are resolved out of
#: ``~/.gpuwm/bridges`` by :mod:`gpuwm.obs.frontdoor`; and none of them
#: was in this tuple, so ``gpuwm fetch-bridges`` -- the command their
#: own refusals named -- staged a complete bundle and the refusal
#: repeated verbatim.  That is the ``rw_mpas_convert`` failure exactly:
#: committed is not shipped, and a refusal naming a command that cannot
#: help is worse than no message.  A front door that a resolver looks
#: for and a refusal names belongs in the bundle that resolver reads.
#: The release workspace already BUILT them -- ``cargo build --release
#: --locked`` at ``tools/rustwx`` builds every workspace member -- so
#: the four binaries were produced by the cut, probed by nothing, and
#: thrown away at the end of the job.
BUNDLED_ARTIFACTS: tuple[BundledArtifact, ...] = (
    BundledArtifact(
        "grib1_bridge", "executable", bridges.CRATE_RELATIVE,
        bridges.BRIDGE_ENV["grib1_bridge"],
        "ERA5 route (gpuwm check/run, rw-wps --source era5)"),
    BundledArtifact(
        "gfs_grib2_bridge", "executable", bridges.CRATE_RELATIVE,
        bridges.BRIDGE_ENV["gfs_grib2_bridge"],
        "GFS front door (rw-wps --source gfs)"),
    BundledArtifact(
        "hrrr_grib2_bridge", "executable", bridges.CRATE_RELATIVE,
        bridges.BRIDGE_ENV["hrrr_grib2_bridge"],
        "HRRR front door (rw-wps --source hrrr)"),
    BundledArtifact(
        "grib2_inventory", "executable", bridges.CRATE_RELATIVE,
        bridges.BRIDGE_ENV["grib2_inventory"],
        "20CRv3/mapped GRIB2 routes"),
    BundledArtifact(
        "grib2_dump", "executable", bridges.CRATE_RELATIVE,
        bridges.BRIDGE_ENV["grib2_dump"],
        "20CRv3/mapped GRIB2 routes"),
    BundledArtifact(
        "gpuwm_preprocess_cpu", "library", bridges.CRATE_RELATIVE,
        "GPUWM_CPU_PREPROCESS_BRIDGE",
        "--preprocess-backend cpu"),
    BundledArtifact(
        "rw_fetch", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "GPUWM_RW_FETCH", "gpuwm fetch --engine rust"),
    BundledArtifact(
        "rw_wrfbatch", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "GPUWM_RW_WRFBATCH", "gpuwm render --engine rust"),
    BundledArtifact(
        "rw_nexrad", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "GPUWM_RW_NEXRAD", "radar observation ingest (the DA nowcast)"),
    BundledArtifact(
        "rw_odim", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "GPUWM_RW_ODIM",
        "European polar volumes (gpuwm obs radar)"),
    BundledArtifact(
        "rw_mrms", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "GPUWM_RW_MRMS", "MRMS composite reflectivity (gpuwm obs mrms)"),
    BundledArtifact(
        "rw_stage4", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "GPUWM_RW_STAGE4",
        "Stage-IV precipitation (gpuwm obs stage4)"),
    BundledArtifact(
        "rw_asos", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "GPUWM_RW_ASOS",
        "ASOS/METAR surface observations (gpuwm obs asos)"),
    BundledArtifact(
        "rw_goes", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "GPUWM_RW_GOES",
        "GOES ABI cloud-water-path packs (gpuwm obs goes)"),
    BundledArtifact(
        "rw_opera", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "GPUWM_RW_OPERA",
        "European composite reflectivity (gpuwm obs opera)"),
    # The NetCDF decoder.  It joined this list the moment NetCDF decode
    # moved off `netCDF4.Dataset` and onto `netcrust`: from that commit
    # `rw-wps --source netcdf` and every mapped NetCDF route REFUSE
    # without it, by design and by name.  A bundle that omits it is a
    # bundle that cannot read the one source format a user is most
    # likely to bring of their own -- and the alternative, telling them
    # to install a Rust toolchain, is the prerequisite this project
    # otherwise never imposes.
    BundledArtifact(
        "rw_netcdf", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "GPUWM_RW_NETCDF", "NetCDF sources (rw-wps --source netcdf, "
        "gpuwm adapt, the mapped routes)"),
    # The dealiasing engine `--dealias` gets by default since 2026-08-12.
    # It joined this list the day it became the default and for the same
    # reason `rw_nexrad` did: a prerequisite of the shipped configuration
    # that a wheel install cannot satisfy is a wheel install that cannot
    # run the shipped configuration, and the alternative is telling every
    # user to install a Rust toolchain.  It is the first VENDORED entry
    # here -- a verbatim copy of an upstream crate, built the same way on
    # the same two platforms, whose staleness is proved by its contract
    # marker rather than by a stamp of this checkout's HEAD.
    # The crate path and the environment variable are spelled out rather
    # than imported from `gpuwm.obs.dealias_region`, exactly as the other
    # nine are: this module is reached by `gpuwm fetch-bridges` on a fresh
    # wheel and has no business importing the observation stack to read
    # two strings.  `tests/test_bridge_fetch.py` binds both to that
    # module's own constants.
    BundledArtifact(
        "region_global_dealias", "library", "tools/region_global_dealias",
        "GPUWM_DEALIAS_REGION_BRIDGE",
        "velocity dealiasing (--dealias, the default engine)",
        vendored=True),
    # The NetCDF WRITER.  It joined this list the moment the product tape
    # flipped onto it BY DEFAULT: from that commit `gpuwm sim` and every
    # history write REFUSE without it, by design and by name
    # (`gpuwm.io.wrfout.WrfoutWriter` names the breakage and the
    # GPUWM_WRFOUT_WRITER=python workaround), so a wheel that omits it is
    # a wheel that cannot write its own forecast.  The wrfinput/wrfbdy
    # pair then flipped onto the same library, so such a wheel now cannot
    # write its own WRF INPUT either -- `gpuwm prep` refuses the same way,
    # naming GPUWM_WRFINPUT_WRITER=python.  Environment variable spelled
    # to match gpuwm.io.nc_writer_bridge.NCWRITE_BRIDGE_ENV, and a test
    # binds the two.
    BundledArtifact(
        "netcdf_writer", "library", bridges.RUSTWX_CRATE_RELATIVE,
        "GPUWM_NCWRITE_BRIDGE",
        "wrfout history writes and the wrfinput/wrfbdy export "
        "(both default engines)"),
    # The STATIC-FIELD builder.  It joined this list the moment
    # `build_static` and the ProjectedGrid array methods flipped onto the
    # Rust crate BY DEFAULT (the static-rust-port lanes): from that
    # commit every bare static build on a wheel without it is a reported
    # WORKAROUND onto the numpy fallback -- a degradation, never the
    # shipped configuration.  Environment variable spelled to match
    # gpuwm.static.rust_bridge.STATIC_BRIDGE_ENV; a test binds the two
    # (tests/test_static_rust_parity.py::TestEstateAndDoctor).
    BundledArtifact(
        "static_fields", "library", bridges.RUSTWX_CRATE_RELATIVE,
        "GPUWM_STATIC_BRIDGE",
        "static-field builds (the default geogrid-equivalent builder)"),
    # The MAPPED DECODE ENGINE, and the plainest case on this list.
    # `gpuwm.mapped_engine_bridge.DEFAULT_ENGINE` is `rust`, so every
    # mapped source -- the route almost every registered source reaches
    # canonical frames through -- decodes in this binary on a bare run.
    # It was left out of this tuple when the route flipped, and the
    # measurement of what that costs is not a hypothetical: a fresh
    # py3.14 wheel install reports `MISSING mapped decode engine ...
    # blocks every mapped source, which is the default decode path`, and
    # no gpuwm command could supply it -- `gpuwm fetch-bridges` staged a
    # complete bundle and the gap survived, because the roster it reads
    # is this one.  A wheel user's only route was a clone and a cargo
    # build, for the DEFAULT decode path.  That is the rw_mpas_convert
    # failure exactly, one layer more expensive: committed is not
    # shipped, and the alternative -- telling every user to install a
    # Rust toolchain -- is the prerequisite this project otherwise never
    # imposes.
    #
    # It is the first entry from the engine workspace (`tools/rw_wps`),
    # which builds `--locked --offline` from its own vendored closure
    # exactly as the other two workspaces do.  The crate path and the
    # environment variable are spelled out rather than imported from
    # `gpuwm.mapped_engine_bridge` for the reason the dealiasing entry
    # gives: this module is reached by `gpuwm fetch-bridges` on a fresh
    # wheel and has no business importing the decode stack to read two
    # strings.  `tests/test_bridge_fetch.py` binds both to that module's
    # own constants.  NOT vendored: `crates/mapped-engine` is gpuwm's
    # own crate beside the donor snapshot (VENDOR.md), so it is proved
    # by the source-revision stamp its own `build.rs` injects, like
    # every other gpuwm-authored artifact here.
    BundledArtifact(
        "gpuwm_mapped_engine", "executable", "tools/rw_wps",
        "GPUWM_MAPPED_ENGINE_BIN",
        "mapped-source decode (the default decode path)"),
    # The OBSERVATION REMAP.  It joined this list the moment
    # `gpuwm.verify.obs.regrid`'s plan build and apply flipped onto the
    # Rust crate BY DEFAULT: from that commit every observation score on
    # a wheel without it is a reported WORKAROUND onto the scipy
    # fallback, whose exact-tie answers are cKDTree traversal order
    # rather than a rule.  Environment variable spelled to match
    # gpuwm.obs_regrid_bridge.OBSREGRID_BRIDGE_ENV; a test binds
    # the two (tests/test_obs_regrid_rust_parity.py).
    BundledArtifact(
        "obs_regrid", "library", bridges.RUSTWX_CRATE_RELATIVE,
        "GPUWM_OBSREGRID_BRIDGE",
        "observation remap plans (the default battery remap engine)"),
    # The four MPAS binaries of the first wave (the fifth, the
    # lateral-boundary producer, is the last entry in this tuple and
    # carries its own note), and `rw_mpas_convert` is the artifact this
    # whole tuple's comment block keeps naming as the precedent: it was
    # written, committed, resolved by name and carried by no bundle, so
    # the refusals that named `gpuwm fetch-bridges` sent readers to a
    # command that staged a complete bundle without it.  It stayed in
    # that state one wave after the observation front doors were fixed
    # for the identical reason, and `rw_mpas_mesh` -- the generator that
    # reproduces the published NCAR meshes to 1e-11 -- joined it there
    # the day it was built: no bundle row, no marker, no environment
    # variable, no resolver and no Python that called it.
    #
    # All four build `--locked --offline` out of the same vendored
    # closure on the same two platforms as their eleven `tools/rustwx`
    # siblings, and the release workspace already BUILT them -- `cargo
    # build --release --locked` at `tools/rustwx` builds every workspace
    # member -- so the binaries were produced by the cut, probed by
    # nothing, and thrown away at the end of the job.  Bundling them
    # costs four entries here and the `build.rs` that stamps the crate.
    #
    # Environment variables spelled to match gpuwm.mpas_mesh.BRIDGES; a
    # test binds each row to that module's own resolver.
    BundledArtifact(
        "rw_mpas_mesh", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "GPUWM_RW_MPAS_MESH",
        "MPAS mesh generation (gpuwm mesh)"),
    # The other half of what `gpuwm mesh` delivers, and it is not
    # optional: a grid file reaches no dycore on its own, because the
    # mesh registry pins BOTH the grid and a matching static by byte
    # count and sha256.  Shipping the generator without the static
    # builder ships a command whose output nothing accepts.
    BundledArtifact(
        "rw_mpas_static", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "GPUWM_RW_MPAS_STATIC",
        "the matching MPAS static (gpuwm mesh)"),
    BundledArtifact(
        "rw_mpas_init", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "GPUWM_RW_MPAS_INIT",
        "MPAS initial conditions from a grid and a static file"),
    BundledArtifact(
        "rw_mpas_convert", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "GPUWM_RW_MPAS_CONVERT",
        "MPAS history onto the renderer's tape"),
    # The lateral-boundary producer, and the plainest repeat of the
    # failure every paragraph above describes.  Its SOURCE has been in
    # this tree since the limited-area lane landed
    # (`tools/rustwx/crates/rw-mpas/src/bin/rw_mpas_lbc.rs`); the
    # release workspace has BUILT it at every cut since, because `cargo
    # build --release --locked` at `tools/rustwx` builds every workspace
    # member and this is a declared `[[bin]]`; and no published bundle
    # has ever carried it, because this tuple is the roster `pack`
    # walks and it was not in it.  Source presence is not shipping: the
    # binary was produced by the cut, probed by nothing, and thrown
    # away at the end of the job, four times over.
    #
    # What its absence costs is not an optional extra.  A limited-area
    # mesh is two halves: `rw_mpas_mesh --cull-parent` cuts the mesh,
    # and this produces the boundary series that mesh is driven by.
    # Shipping the cull without the producer ships a mesh nothing can
    # run -- the same shape as shipping the generator without
    # `rw_mpas_static`, which is why that one is in this tuple too.
    BundledArtifact(
        "rw_mpas_lbc", "executable", bridges.RUSTWX_CRATE_RELATIVE,
        "GPUWM_RW_MPAS_LBC",
        "MPAS lateral boundaries for a limited-area mesh"),
)


#: The ABI handshake of every ``kind == "library"`` artifact, by artifact
#: name: the symbol a loader resolves and the version that symbol must
#: answer.  Per artifact, and declared exactly once.
#:
#: Every library used to be probed with `gpuwm_preprocess_cpu`'s symbol,
#: which was true while there was one library and became false the day
#: the vendored dealiasing cdylib joined the bundle exporting
#: `bw_abi_version` instead.  The workflow's per-runner probe was taught
#: the difference; `tools/verify_release_artifacts.py` was not, so the
#: 2.1.0 prepare job refused a correct bundle after the tag was public.
#: The lesson is not "fix the second copy" but "have one": this table is
#: the single source both the release verifier and the workflow probe
#: read, so a third library cannot be added to `BUNDLED_ARTIFACTS` and
#: quietly inherit some other library's handshake.
LIBRARY_ABI: dict[str, tuple[str, int]] = {
    "gpuwm_preprocess_cpu": ("gpuwm_preprocess_cpu_abi_version", 1),
    "region_global_dealias": ("bw_abi_version", 1),
    # Matches gpuwm.io.nc_writer_bridge.NCWRITE_ABI; a test binds them.
    "netcdf_writer": ("gpuwm_ncwrite_abi_version", 1),
    # Matches gpuwm.static.rust_bridge.STATIC_ABI; a test binds them.
    "static_fields": ("gpuwm_static_abi_version", 1),
    # Matches gpuwm.obs_regrid_bridge.OBSREGRID_ABI; a test
    # binds them.
    "obs_regrid": ("gpuwm_obsregrid_abi_version", 1),
}


def library_abi_for(name: str) -> tuple[str, int]:
    """``(symbol, expected_version)`` for the library artifact ``name``.

    Fail closed.  A library artifact with no entry here is refused
    rather than probed with a neighbour's symbol: the wrong symbol
    either fails to resolve (a confusing refusal blamed on the bytes) or
    resolves against a library that happens to export the same name and
    reports a handshake nobody checked.  Adding a library therefore
    means adding its entry, and the refusal says so by name.
    """

    try:
        return LIBRARY_ABI[name]
    except KeyError:
        raise BridgeAssetError(
            f"{name}: library artifact with no declared ABI handshake in "
            "gpuwm.bridge_assets.LIBRARY_ABI, so nothing says which "
            "exported symbol proves these bytes speak this release's "
            "contract; declare it there beside BUNDLED_ARTIFACTS"
        ) from None


def artifact_filename(artifact: BundledArtifact, platform: str) -> str:
    """The filename ``artifact`` is built under on ``platform``.

    Parametric in the platform rather than in the host, because the
    release tool inspects both bundles from one machine.  On the host it
    agrees with the resolvers gpuwm already uses
    (:func:`gpuwm.bridges.executable_name` and the CPU backend's own
    library naming); a test binds the two.
    """

    if platform not in SUPPORTED_PLATFORMS:
        raise ValueError(f"unknown platform {platform!r}; known: "
                         f"{SUPPORTED_PLATFORMS}")
    windows = platform.startswith("win-")
    if artifact.kind == "library":
        return f"{artifact.name}.dll" if windows else f"lib{artifact.name}.so"
    return f"{artifact.name}.exe" if windows else artifact.name


def host_platform() -> str | None:
    """This box's platform key, or None when no bundle is published.

    A capability check: the operating system and the machine
    architecture, which together decide whether the bytes in a bundle
    can execute here.  None is not a refusal of service -- it routes to
    the build-from-source remedy, which works everywhere.
    """

    machine = platform_module.machine().lower()
    if sys.platform == "win32" and machine in ("amd64", "x86_64"):
        return "win-x86_64"
    if sys.platform.startswith("linux") and machine == "x86_64":
        return "linux-x86_64"
    return None


def host_platform_description() -> str:
    """What :func:`host_platform` looked at, for an honest refusal."""

    return f"{sys.platform}/{platform_module.machine() or 'unknown'}"


# ---------------------------------------------------------------------------
# The pins document
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BinaryPin:
    """One artifact inside a bundle, pinned by exact size and SHA-256."""

    artifact: str
    filename: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class AssetPin:
    """One data file inside a bundle, pinned by exact size and SHA-256.

    ``path`` is the archive member name and, unchanged, the path the
    file is staged to under the destination -- a relative POSIX path
    beginning with :data:`ASSET_ROOT`.  Unlike a :class:`BinaryPin` it
    names no artifact, no environment variable and no ABI marker: a
    shapefile is data, identical on every platform, and the only
    questions worth asking of it are how many bytes it is and whether
    they hash to what the release published.
    """

    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class BundlePin:
    """One platform's bundle: the archive, pinned, and its contents.

    ``assets`` defaults to empty so a pins document written before the
    asset half existed still parses; a *release* with no assets is
    refused by ``tools/build_bridge_bundle.py pin``, which is where that
    staleness would otherwise be introduced.
    """

    platform: str
    filename: str
    bytes: int
    sha256: str
    binaries: tuple[BinaryPin, ...]
    assets: tuple[AssetPin, ...] = ()


@dataclass(frozen=True)
class BridgePins:
    """The packaged pins: which release, and one bundle per platform."""

    release: str | None
    platforms: dict[str, BundlePin]

    def bundle_for(self, platform: str | None) -> BundlePin | None:
        if platform is None:
            return None
        return self.platforms.get(platform)


def packaged_pins_path() -> Path:
    """Path of the pins document inside this installation."""

    return Path(__file__).resolve().parent / Path(PINS_RESOURCE)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BridgeAssetError(f"bridge pins document is malformed: {message}")


def parse_pins(payload: object, *, origin: str = "pins") -> BridgePins:
    """Validate a pins document and return it, or refuse.

    A malformed pins document must fail loudly here rather than produce
    a bundle nobody can verify: every field this staging path relies on
    is required, and a hash is required to look like one.
    """

    _require(isinstance(payload, dict), f"{origin} is not an object")
    assert isinstance(payload, dict)  # narrowed by _require
    _require(payload.get("schema") == PINS_SCHEMA,
             f"{origin} schema is {payload.get('schema')!r}, expected "
             f"{PINS_SCHEMA!r}")
    release = payload.get("release")
    _require(release is None or (isinstance(release, str) and release),
             f"{origin} release must be a non-empty string or null")
    raw_platforms = payload.get("platforms")
    _require(isinstance(raw_platforms, dict),
             f"{origin} platforms must be an object")
    assert isinstance(raw_platforms, dict)
    platforms: dict[str, BundlePin] = {}
    for key, record in sorted(raw_platforms.items()):
        if key not in SUPPORTED_PLATFORMS:
            # A pins document from a newer release may name a platform
            # this build does not know; that entry is irrelevant to
            # this host and skipping it keeps the document usable.
            warn(f"{origin} declares platform {key!r} this build does "
                 "not know; skipping that entry")
            continue
        _require(isinstance(record, dict), f"{key}: record is not an object")
        bundle = record.get("bundle")
        _require(isinstance(bundle, dict), f"{key}: bundle is not an object")
        binaries_raw = record.get("binaries")
        _require(isinstance(binaries_raw, list) and binaries_raw,
                 f"{key}: binaries must be a non-empty list")
        binaries: list[BinaryPin] = []
        for entry in binaries_raw:
            _require(isinstance(entry, dict),
                     f"{key}: a binaries entry is not an object")
            binaries.append(BinaryPin(
                artifact=_string(entry, "artifact", key),
                filename=_string(entry, "filename", key),
                bytes=_size(entry, key),
                sha256=_digest(entry, key)))
        assets_raw = record.get("assets", [])
        _require(isinstance(assets_raw, list),
                 f"{key}: assets must be a list")
        assets: list[AssetPin] = []
        for entry in assets_raw:
            _require(isinstance(entry, dict),
                     f"{key}: an assets entry is not an object")
            assets.append(AssetPin(
                path=_asset_path(entry, key),
                bytes=_size(entry, key),
                sha256=_digest(entry, key)))
        _require(release is not None,
                 f"{key}: a platform is pinned but the release is null")
        platforms[key] = BundlePin(
            platform=key, filename=_string(bundle, "filename", key),
            bytes=_size(bundle, key), sha256=_digest(bundle, key),
            binaries=tuple(binaries), assets=tuple(assets))
    return BridgePins(release=release, platforms=platforms)


def _string(record: dict, field: str, where: str) -> str:
    value = record.get(field)
    _require(isinstance(value, str) and value,
             f"{where}: {field} must be a non-empty string")
    assert isinstance(value, str)
    return value


def _asset_path(record: dict, where: str) -> str:
    """A relative asset path under :data:`ASSET_ROOT`, or a refusal.

    The pins document decides where staging writes, so it is the place
    to refuse a path that could escape the destination.  Absolute paths,
    drive letters, backslashes and ``..`` components are all rejected
    here rather than left for the filesystem to interpret; staging then
    joins a path it already knows is contained.
    """

    value = _string(record, "path", where)
    parts = value.split("/")
    _require(
        not value.startswith("/") and "\\" not in value and ":" not in value
        and all(part not in ("", ".", "..") for part in parts)
        and parts[0] == ASSET_ROOT and len(parts) > 1,
        f"{where}: asset path {value!r} must be a relative path under "
        f"{ASSET_ROOT!r}/ with no '..' component")
    return value


def _size(record: dict, where: str) -> int:
    value = record.get("bytes")
    _require(isinstance(value, int) and not isinstance(value, bool)
             and value > 0, f"{where}: bytes must be a positive integer")
    assert isinstance(value, int)
    return value


def _digest(record: dict, where: str) -> str:
    value = record.get("sha256")
    _require(isinstance(value, str) and len(value) == 64
             and all(c in "0123456789abcdef" for c in value),
             f"{where}: sha256 must be 64 lowercase hex characters")
    assert isinstance(value, str)
    return value


def load_pins(path: Path | None = None) -> BridgePins:
    """Read and validate the packaged pins document."""

    resolved = packaged_pins_path() if path is None else Path(path)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except OSError as error:
        raise BridgeAssetError(
            f"bridge pins document {resolved} is unreadable: {error}")
    except ValueError as error:
        raise BridgeAssetError(
            f"bridge pins document {resolved} is not JSON: {error}")
    return parse_pins(payload, origin=str(resolved))


def staging_available(pins: BridgePins | None = None) -> bool:
    """Can THIS install stage a bundle for THIS platform?

    Both halves are capabilities of artifacts rather than opinions: a
    platform key for this OS/arch, and a bundle pinned for it in the
    document this wheel actually carries.  ``gpuwm doctor`` offers
    ``gpuwm fetch-bridges`` exactly when this is true, so the report
    never advertises a command that would refuse.
    """

    platform = host_platform()
    if platform is None:
        return False
    try:
        resolved = load_pins() if pins is None else pins
    except BridgeAssetError:
        return False
    return resolved.bundle_for(platform) is not None


def asset_url_base(pins: BridgePins) -> str:
    """Base URL the bundle is downloaded from (env override wins)."""

    override = os.environ.get(ASSET_URL_BASE_ENV)
    if override and override.strip():
        return override.strip().rstrip("/")
    if not pins.release:
        raise BridgeAssetError(
            "the packaged pins declare no release, so there is no URL to "
            f"download from; set {ASSET_URL_BASE_ENV} or use --from DIR")
    return f"{bridges.REPOSITORY_URL}/releases/download/{pins.release}"


def bundle_url(pins: BridgePins, bundle: BundlePin) -> str:
    return f"{asset_url_base(pins)}/{bundle.filename}"


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def verify_pinned_file(path: Path, *, expected_bytes: int,
                       expected_sha256: str, label: str) -> None:
    """Exact size and SHA-256, or a refusal naming both observations."""

    actual = path.stat().st_size
    if actual != expected_bytes:
        raise BridgeAssetError(
            f"{label}: {actual:,} bytes, expected {expected_bytes:,}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise BridgeAssetError(
            f"{label}: SHA-256 {observed}, expected {expected_sha256}")


def _is_source_revision(text: str) -> bool:
    return (len(text) == _SOURCE_REV_LENGTH
            and all(c in "0123456789abcdef" for c in text))


def embedded_source_revisions(payload: bytes) -> tuple[str, ...]:
    """Every distinct well-formed source-revision stamp in ``payload``.

    A stamp is :data:`SOURCE_REV_MARKER` immediately followed by 40
    lowercase hex characters.  Marker occurrences followed by anything
    else -- ``unknown`` from a build outside a git checkout, or
    truncated bytes -- are not revisions and are not returned; the
    verifier tells those cases apart from a missing marker.  Order of
    first appearance is preserved so a refusal can name what it saw.
    """

    found: list[str] = []
    start = 0
    while (index := payload.find(SOURCE_REV_MARKER, start)) != -1:
        begin = index + len(SOURCE_REV_MARKER)
        token = payload[begin:begin + _SOURCE_REV_LENGTH]
        text = token.decode("ascii", errors="replace")
        if _is_source_revision(text) and text not in found:
            found.append(text)
        start = begin
    return tuple(found)


def verify_source_revision(payload: bytes, *, expected: str,
                           label: str) -> None:
    """``payload`` was built from commit ``expected``, or a refusal.

    Read from the bytes, never by executing the artifact: the cut
    inspects both platforms' bundles from one machine, and a staleness
    check that needs to run the binary cannot look at the other
    platform's.  Refusals name what was found so the remedy (rebuild
    from the release checkout and repack) is unambiguous.
    """

    expected = expected.strip().lower()
    if not _is_source_revision(expected):
        raise BridgeAssetError(
            f"{label}: expected source revision {expected!r} is not a "
            "full 40-hex git commit; pass the exact commit being "
            "released, never a ref name that can move")
    marker = SOURCE_REV_MARKER.decode("ascii").rstrip("=")
    revisions = embedded_source_revisions(payload)
    if not revisions:
        if SOURCE_REV_MARKER in payload:
            raise BridgeAssetError(
                f"{label}: carries a {marker} stamp with no 40-hex "
                "revision after it -- it was built outside a git "
                "checkout, so nothing proves which source produced it; "
                "rebuild from the release checkout")
        raise BridgeAssetError(
            f"{label}: carries no {marker} stamp, so nothing proves it "
            f"was built from the source revision being released "
            f"({expected}); it is a stale build predating the stamped "
            "bridges, or not a gpuwm bridge artifact at all -- rebuild "
            "from the release checkout and repack")
    if len(revisions) > 1:
        raise BridgeAssetError(
            f"{label}: carries {len(revisions)} distinct source-revision "
            f"stamps ({', '.join(revisions)}); one binary cannot be from "
            "two commits, so this build is not trustworthy -- rebuild "
            "from the release checkout")
    if revisions[0] != expected:
        raise BridgeAssetError(
            f"{label}: was built from source revision {revisions[0]}, "
            f"not the {expected} being released -- a stale build; "
            "rebuild from the release checkout and repack")


def verify_contract_marker(artifact: str, path: Path) -> None:
    """The static contract check ``gpuwm doctor`` applies, before install.

    A bundle is built from the same commit as the wheel that carries its
    pins, so a marker mismatch here means the release process assembled
    the two from different trees.  Catching it at staging is the whole
    difference between a refusal and a preparation that dies hours later
    blaming a file gpuwm wrote correctly.
    """

    if artifact not in bridges.BRIDGE_ABI_MARKERS:
        return
    ok, evidence = bridges.bridge_abi_matches(artifact, path)
    if not ok:
        raise BridgeAssetError(f"{path.name}: {evidence}")


def matches_pin(path: Path, pin: BinaryPin) -> bool:
    """Is the file already exactly these bytes?  No exceptions raised."""

    try:
        if not path.is_file() or path.stat().st_size != pin.bytes:
            return False
        return sha256_file(path) == pin.sha256
    except OSError:
        return False


def classify_destination(dest: Path, bundle: BundlePin
                         ) -> tuple[list[BinaryPin], list[BinaryPin],
                                    list[BinaryPin]]:
    """(already pinned, present but different, absent) at ``dest``."""

    staged: list[BinaryPin] = []
    stale: list[BinaryPin] = []
    absent: list[BinaryPin] = []
    for pin in bundle.binaries:
        path = dest / pin.filename
        if not path.is_file():
            absent.append(pin)
        elif matches_pin(path, pin):
            staged.append(pin)
        else:
            stale.append(pin)
    return staged, stale, absent


def classify_assets(dest: Path, bundle: BundlePin
                    ) -> tuple[list[AssetPin], list[AssetPin],
                               list[AssetPin]]:
    """(already pinned, present but different, absent) for the assets.

    Separate from :func:`classify_destination` because the two answer
    different questions for the caller: the binaries are listed
    individually in a report, and several dozen shapefiles are counted.
    """

    staged: list[AssetPin] = []
    stale: list[AssetPin] = []
    absent: list[AssetPin] = []
    for pin in bundle.assets:
        path = dest / pin.path
        if not path.is_file():
            absent.append(pin)
        elif matches_pin(path, pin):
            staged.append(pin)
        else:
            stale.append(pin)
    return staged, stale, absent


# ---------------------------------------------------------------------------
# What a RESOLUTION is allowed to hand back
# ---------------------------------------------------------------------------

#: Policy for a staged artifact that is not the one this release pinned,
#: applied where a resolution ladder is about to hand the file to a door.
#:
#: ``refresh`` (the default) re-fetches this release's bundle and uses
#: the verified bytes; ``refuse`` never touches the network and names
#: the mismatch instead; ``allow`` runs the staged file anyway and is a
#: reported WORKAROUND, not a configuration.
STALE_POLICY_ENV = "GPUWM_BRIDGE_STALE_POLICY"

#: The accepted values of :data:`STALE_POLICY_ENV`, default first.
STALE_POLICIES = ("refresh", "refuse", "allow")

#: Socket timeout for an automatic refresh.  Shorter than
#: :data:`_TIMEOUT_S`, which serves a command the operator typed and is
#: watching: an auto-refresh happens inside somebody else's door, so an
#: unreachable network has to become the offline refusal quickly rather
#: than hold a run open for two minutes per read.
_AUTO_REFRESH_TIMEOUT_S = 30

#: ``{(path, size, mtime_ns): matches}`` for pin comparisons already
#: made in this process.  A door can resolve the same artifact many
#: times in one run and the answer cannot change under it without the
#: size or the mtime moving, so the SHA-256 is computed once.
_PIN_MEMO: dict[tuple[str, int, int], bool] = {}


def _under(path: Path, directory: Path) -> bool:
    """Is ``path`` inside ``directory``?  Never raises."""

    try:
        return path.resolve().is_relative_to(directory.resolve())
    except (OSError, ValueError):
        return False


@dataclass(frozen=True)
class StagedPinStatus:
    """One staged artifact, judged against the pins this wheel carries.

    Produced only for a file under :func:`gpuwm.bridges
    .default_bridge_dir` that this release publishes a pin for, which is
    the one rung whose contents no version of gpuwm controls: a wheel
    upgrade replaces the Python half and leaves that directory exactly
    as the previous release wrote it.
    """

    path: Path
    pin: BinaryPin
    release: str
    matches: bool
    observed_bytes: int
    observed_revisions: tuple[str, ...]

    @property
    def observed_revision(self) -> str | None:
        """The single stamp the file carries, or None if it is not one."""

        return (self.observed_revisions[0]
                if len(self.observed_revisions) == 1 else None)

    def provenance(self) -> str:
        """What the bytes on disk say about where they came from."""

        if self.observed_revision is not None:
            return f"built from source revision {self.observed_revision}"
        if not self.observed_revisions:
            marker = SOURCE_REV_MARKER.decode("ascii").rstrip("=")
            return f"carrying no {marker} stamp"
        return (f"carrying {len(self.observed_revisions)} distinct "
                "source-revision stamps ("
                + ", ".join(self.observed_revisions) + ")")

    def describe(self) -> str:
        """The mismatch in one sentence, naming both sides by number."""

        return (f"{self.path} is {self.observed_bytes:,} B "
                f"{self.provenance()}; {self.release} pins "
                f"{self.pin.bytes:,} B (SHA-256 {self.pin.sha256[:12]}...) "
                f"for {self.pin.artifact}")


def staged_pin_status(path: Path | str, *,
                      pins: BridgePins | None = None
                      ) -> StagedPinStatus | None:
    """Judge a resolved artifact against this release's pins, or None.

    ``None`` -- the question does not arise -- for every case where the
    pins have nothing to say, and each of those is a deliberate
    exemption rather than an oversight:

    * the file is not under :func:`gpuwm.bridges.default_bridge_dir`.
      An environment override is an explicit declaration, a checkout's
      ``target/release`` is a build the developer just made, and
      ``libexec`` beside the package or inside it arrived with this
      version.  None of the three is the skew this judgement exists
      for, and a cargo build never matches a release pin by
      construction, so judging them would refuse the developer path on
      every run.
    * this install carries no pins for this platform -- a source
      checkout (whose packaged document declares no release until the
      cut stamps it), or a platform no bundle is published for.
    * this release publishes no pin for a file by that name.

    Otherwise the answer is the one ``gpuwm fetch-bridges`` and ``gpuwm
    doctor`` already ask of the same file: are these the exact bytes,
    size and SHA-256, that this release published?  A mismatch reads the
    embedded ``GPUWM_BRIDGE_SOURCE_REV`` stamps as well, so a refusal
    can name which release the file on disk came from rather than only
    that it is wrong.
    """

    try:
        resolved = Path(path).resolve()
    except (OSError, ValueError):
        return None
    if not _under(resolved, bridges.default_bridge_dir()):
        return None
    if pins is None:
        try:
            pins = load_pins()
        except BridgeAssetError:
            # An unreadable pins document is doctor's line to report; it
            # is not a reason to block every door on this box.
            return None
    bundle = pins.bundle_for(host_platform())
    if bundle is None or not pins.release:
        return None
    pin = next((entry for entry in bundle.binaries
                if entry.filename == resolved.name), None)
    if pin is None:
        return None
    try:
        stat = resolved.stat()
    except OSError:
        return None
    key = (str(resolved), stat.st_size, stat.st_mtime_ns)
    matches = _PIN_MEMO.get(key)
    if matches is None:
        matches = matches_pin(resolved, pin)
        _PIN_MEMO[key] = matches
    revisions: tuple[str, ...] = ()
    if not matches:
        try:
            revisions = embedded_source_revisions(resolved.read_bytes())
        except OSError:
            revisions = ()
    return StagedPinStatus(
        path=resolved, pin=pin, release=pins.release, matches=matches,
        observed_bytes=stat.st_size, observed_revisions=revisions)


def forget_pin_memo() -> None:
    """Drop the in-process pin cache (a refresh has moved the bytes)."""

    _PIN_MEMO.clear()


def stale_policy() -> str:
    """The configured policy for a stale staged artifact, validated.

    An unrecognised value is not silently treated as the default: a
    misspelled policy would then look like it was honoured.
    """

    raw = (os.environ.get(STALE_POLICY_ENV) or "").strip().lower()
    if not raw:
        return STALE_POLICIES[0]
    if raw not in STALE_POLICIES:
        warn(f"{STALE_POLICY_ENV}={raw!r} is not one of "
             f"{', '.join(STALE_POLICIES)}; using {STALE_POLICIES[0]!r}")
        return STALE_POLICIES[0]
    return raw


def refresh_staged_bundle(*, pins: BridgePins | None = None,
                          dest: Path | None = None, progress=None,
                          urlopen_fn=None) -> list[Path]:
    """Re-stage this release's bundle over the staged directory.

    The same verified path ``gpuwm fetch-bridges`` walks -- download,
    size and SHA-256 against the packaged pins, contract marker, atomic
    replace -- reached from a resolution rather than from a command, so
    that a bare run stops using an artifact this release did not
    publish.  Every refusal on that path is a :class:`BridgeAssetError`
    and stays one here; the caller turns it into the offline arm.
    """

    resolved_pins = load_pins() if pins is None else pins
    bundle = resolved_pins.bundle_for(host_platform())
    if bundle is None:
        raise BridgeAssetError(
            "this install carries no bundle pin for "
            f"{host_platform_description()}, so there is nothing to "
            "refresh from")
    destination = bridges.default_bridge_dir() if dest is None else Path(dest)
    opener = urlopen_fn
    if opener is None:
        def opener(request):
            return urlopen(request, timeout=_AUTO_REFRESH_TIMEOUT_S)
    installed = fetch_bundle(resolved_pins, bundle, destination,
                             progress=progress or (lambda message: None),
                             urlopen_fn=opener)
    forget_pin_memo()
    return installed


# ---------------------------------------------------------------------------
# Download (resumable, restartable)
# ---------------------------------------------------------------------------

def _default_urlopen(request: Request):
    return urlopen(request, timeout=_TIMEOUT_S)


def download_bundle(url: str, dest: Path, *, expected_bytes: int,
                    progress, urlopen_fn=_default_urlopen) -> None:
    """Download ``url`` to ``dest``, resuming or restarting as needed.

    An interrupted run leaves a partial file; the next run asks the
    server to continue from its byte count.  A server that answers
    anything but ``206 Partial Content`` -- including one that has no
    idea what a range is -- restarts the transfer from zero instead of
    appending to bytes it did not extend, which is the failure mode a
    naive resume turns into a corrupt archive.  A partial longer than
    the pin is discarded outright: it cannot be a prefix of the file we
    are pinned to.
    """

    offset = dest.stat().st_size if dest.exists() else 0
    if offset > expected_bytes:
        dest.unlink()
        progress(f"gpuwm fetch-bridges: the partial download was larger "
                 f"than the pinned {expected_bytes:,} B; discarded it and "
                 "restarted")
        offset = 0
    if offset == expected_bytes:
        progress("gpuwm fetch-bridges: the partial download is already "
                 "complete; verifying it instead of re-downloading")
        return
    headers = {"User-Agent": _USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        progress(f"gpuwm fetch-bridges: resuming at {offset:,} B of "
                 f"{expected_bytes:,} B")
    request = Request(url, headers=headers)
    try:
        with urlopen_fn(request) as response:
            status = getattr(response, "status", None)
            mode = "ab" if (offset and status == 206) else "wb"
            if offset and status != 206:
                progress("gpuwm fetch-bridges: the server ignored the "
                         "resume range; restarting the download")
            with dest.open(mode) as sink:
                while block := response.read(_BLOCK_BYTES):
                    sink.write(block)
    except HTTPError as error:
        raise BridgeAssetError(
            f"HTTP {error.code} from {url}; the partial file is kept and "
            "a re-run resumes")
    except (URLError, OSError, TimeoutError) as error:
        raise BridgeAssetError(
            f"download failed from {url}: {error}; the partial file is "
            "kept and a re-run resumes")
    size = dest.stat().st_size
    if size != expected_bytes:
        dest.unlink(missing_ok=True)
        raise BridgeAssetError(
            f"downloaded {size:,} B from {url}, expected "
            f"{expected_bytes:,} B; the incomplete file was removed -- "
            "re-run to fetch it again")


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------

def _install(temp: Path, final: Path, pin: BinaryPin) -> None:
    """Verify ``temp`` three ways, then atomically install it."""

    try:
        verify_pinned_file(temp, expected_bytes=pin.bytes,
                           expected_sha256=pin.sha256, label=pin.filename)
        verify_contract_marker(pin.artifact, temp)
    except BridgeAssetError:
        temp.unlink(missing_ok=True)
        raise
    if os.name != "nt":
        # A zip carries no reliable execute bit, and an executable that
        # cannot be executed is not staged, it is merely present.
        os.chmod(temp, 0o755)
    os.replace(temp, final)


def _install_asset(temp: Path, final: Path, pin: AssetPin) -> None:
    """Verify ``temp`` by size and hash, then atomically install it.

    Two checks, not three, and no execute bit: a shapefile has no ABI
    marker to match and nothing should ever run it.  Everything else --
    verify before install, delete on failure, atomic replace -- is what
    :func:`_install` does for a binary.
    """

    try:
        verify_pinned_file(temp, expected_bytes=pin.bytes,
                           expected_sha256=pin.sha256, label=pin.path)
    except BridgeAssetError:
        temp.unlink(missing_ok=True)
        raise
    if os.name != "nt":
        os.chmod(temp, 0o644)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temp, final)


def _bundle_members(archive: Path) -> list[str]:
    try:
        with zipfile.ZipFile(archive) as zf:
            return zf.namelist()
    except (zipfile.BadZipFile, OSError) as error:
        raise BridgeAssetError(f"{archive.name} is not a readable bundle "
                               f"archive: {error}")


def stage_from_bundle(archive: Path, bundle: BundlePin, dest: Path,
                      *, progress) -> list[Path]:
    """Extract, verify and install every pinned artifact from ``archive``.

    Members are read by their exact pinned filename, never by whatever
    path the archive proposes, so nothing can be written outside
    ``dest``.  A pinned name the archive does not carry is a refusal
    that names what the archive holds instead -- which is how the other
    platform's bundle identifies itself.
    """

    present = set(_bundle_members(archive))
    missing = [pin.filename for pin in bundle.binaries
               if pin.filename not in present]
    if missing:
        listed = ", ".join(sorted(present)[:8]) or "nothing"
        raise BridgeAssetError(
            f"{archive.name} does not carry {', '.join(missing)} -- it "
            f"holds {listed}.  This is not the {bundle.platform} bundle")
    absent_assets = [pin.path for pin in bundle.assets
                     if pin.path not in present]
    if absent_assets:
        raise BridgeAssetError(
            f"{archive.name} is missing {len(absent_assets)} pinned map "
            f"asset(s), starting with {absent_assets[0]}.  A bundle whose "
            "assets do not match its pins would render geography-less "
            "plots; refusing it")
    dest.mkdir(parents=True, exist_ok=True)
    # Per PROCESS, because staging is no longer only something an
    # operator types: a resolution that finds a stale artifact refreshes
    # the estate itself, so two runs on one box can be extracting into
    # this directory at the same moment.  A shared scratch name means
    # the ``finally`` below deletes the other run's half-written files.
    # The install itself is already safe -- verify, then ``os.replace``.
    work = dest / f"{ARCHIVE_SUBDIR}-stage-{os.getpid()}"
    work.mkdir(parents=True, exist_ok=True)
    installed: list[Path] = []
    try:
        with zipfile.ZipFile(archive) as zf:
            for pin in bundle.binaries:
                temp = work / pin.filename
                with zf.open(pin.filename) as source, \
                        temp.open("wb") as sink:
                    shutil.copyfileobj(source, sink, _BLOCK_BYTES)
                final = dest / pin.filename
                replaced = final.is_file()
                _install(temp, final, pin)
                installed.append(final)
                progress(
                    f"gpuwm fetch-bridges: {'replaced' if replaced else 'staged'}"
                    f" {final} ({pin.bytes:,} B, SHA-256 {pin.sha256[:12]}...)")
            staged_assets = 0
            asset_bytes = 0
            for pin in bundle.assets:
                # The pinned path is validated relative and contained by
                # parse_pins, so the join cannot leave dest.
                temp = work / "asset.part"
                with zf.open(pin.path) as source, temp.open("wb") as sink:
                    shutil.copyfileobj(source, sink, _BLOCK_BYTES)
                final = dest / pin.path
                _install_asset(temp, final, pin)
                installed.append(final)
                staged_assets += 1
                asset_bytes += pin.bytes
            if staged_assets:
                progress(
                    f"gpuwm fetch-bridges: staged {staged_assets} map asset "
                    f"file(s) ({asset_bytes / (1024 * 1024):.1f} MiB) under "
                    f"{dest / ASSET_ROOT}; the renderer finds them there "
                    "without any environment variable")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return installed


def stage_from_loose_files(source_dir: Path, bundle: BundlePin, dest: Path,
                           *, progress) -> list[Path]:
    """Install the pinned artifacts sitting loose in ``source_dir``.

    What an air-gapped operator has after building on a machine that
    does have a toolchain: twenty-six files, no archive.  Same three
    checks, same atomic install.
    """

    absent = [pin.filename for pin in bundle.binaries
              if not (source_dir / pin.filename).is_file()]
    present = [pin for pin in bundle.binaries
               if (source_dir / pin.filename).is_file()]
    if absent and not present:
        raise BridgeAssetError(
            f"{source_dir} carries neither {bundle.filename} nor the loose "
            f"artifacts; missing {', '.join(absent)}")
    if absent:
        # The twenty-six artifacts are independent; an air-gapped operator
        # with the decoders but not the renderer gets the decoders,
        # verified, and doctor names what is still missing.
        warn(f"{source_dir} is missing {len(absent)} of "
             f"{len(bundle.binaries)} pinned artifact(s) "
             f"({', '.join(absent)}); staging the {len(present)} that "
             "are present -- gpuwm doctor reports the rest")
    dest.mkdir(parents=True, exist_ok=True)
    # Per PROCESS, because staging is no longer only something an
    # operator types: a resolution that finds a stale artifact refreshes
    # the estate itself, so two runs on one box can be extracting into
    # this directory at the same moment.  A shared scratch name means
    # the ``finally`` below deletes the other run's half-written files.
    # The install itself is already safe -- verify, then ``os.replace``.
    work = dest / f"{ARCHIVE_SUBDIR}-stage-{os.getpid()}"
    work.mkdir(parents=True, exist_ok=True)
    installed: list[Path] = []
    try:
        for pin in present:
            temp = work / pin.filename
            shutil.copyfile(source_dir / pin.filename, temp)
            final = dest / pin.filename
            replaced = final.is_file()
            _install(temp, final, pin)
            installed.append(final)
            progress(
                f"gpuwm fetch-bridges: {'replaced' if replaced else 'staged'}"
                f" {final} ({pin.bytes:,} B, SHA-256 {pin.sha256[:12]}...)")
        # The map assets keep their layout in a loose directory too, so
        # an operator who unzipped a bundle by hand and an operator who
        # copied a checkout's assets/ both land here.
        staged_assets = 0
        missing_assets: list[str] = []
        for pin in bundle.assets:
            origin = source_dir / pin.path
            if not origin.is_file():
                missing_assets.append(pin.path)
                continue
            temp = work / "asset.part"
            shutil.copyfile(origin, temp)
            final = dest / pin.path
            _install_asset(temp, final, pin)
            installed.append(final)
            staged_assets += 1
        if staged_assets:
            progress(f"gpuwm fetch-bridges: staged {staged_assets} map "
                     f"asset file(s) under {dest / ASSET_ROOT}")
        if missing_assets:
            warn(f"{source_dir} carries no {ASSET_ROOT}/ tree for "
                 f"{len(missing_assets)} pinned map asset file(s); the "
                 "renderer will draw plots without coastlines or borders "
                 "unless it finds them elsewhere")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return installed


def stage_from_dir(source_dir: Path, bundle: BundlePin, dest: Path,
                   *, progress) -> list[Path]:
    """``--from DIR``: the bundle archive if it is there, else the files."""

    archive = source_dir / bundle.filename
    if archive.is_file():
        verify_pinned_file(archive, expected_bytes=bundle.bytes,
                           expected_sha256=bundle.sha256,
                           label=bundle.filename)
        progress(f"gpuwm fetch-bridges: {archive} matches the packaged "
                 "bundle pin; staging from it")
        return stage_from_bundle(archive, bundle, dest, progress=progress)
    return stage_from_loose_files(source_dir, bundle, dest, progress=progress)


def fetch_bundle(pins: BridgePins, bundle: BundlePin, dest: Path, *,
                 keep_bundle: bool = False, progress=print,
                 urlopen_fn=_default_urlopen) -> list[Path]:
    """Download, verify and stage the platform bundle."""

    archive_dir = dest / ARCHIVE_SUBDIR
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_dir / bundle.filename
    url = bundle_url(pins, bundle)
    if archive.is_file() and archive.stat().st_size == bundle.bytes \
            and sha256_file(archive) == bundle.sha256:
        progress(f"gpuwm fetch-bridges: {bundle.filename} is already "
                 "downloaded and pin-verified; reusing it")
    else:
        progress(f"gpuwm fetch-bridges: downloading {bundle.filename} "
                 f"({bundle.bytes / (1024 * 1024):.1f} MiB) from {url}")
        download_bundle(url, archive, expected_bytes=bundle.bytes,
                        progress=progress, urlopen_fn=urlopen_fn)
        try:
            verify_pinned_file(archive, expected_bytes=bundle.bytes,
                               expected_sha256=bundle.sha256,
                               label=bundle.filename)
        except BridgeAssetError:
            archive.unlink(missing_ok=True)
            raise
    installed = stage_from_bundle(archive, bundle, dest, progress=progress)
    if keep_bundle:
        progress(f"gpuwm fetch-bridges: kept the verified bundle {archive}")
    else:
        archive.unlink(missing_ok=True)
        try:
            archive_dir.rmdir()
        except OSError:
            pass
    return installed


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------

#: What stays broken while the artifacts are absent.  Named once, in
#: routes rather than in filenames, because "nine artifacts are missing"
#: is not a consequence a reader can act on and "prep refuses" is.
_MISSING_ARTIFACT_BREAKAGE = (
    "gpuwm fetch-bridges: until the artifacts exist, the routes that run "
    "them stay unresolved -- GRIB decode, NetCDF read/write and the "
    "renderer -- so `gpuwm prep` refuses to decode source bytes and "
    "`gpuwm render` has no renderer to call.")


def _source_build_lines() -> list[str]:
    """The build route as commands, never as a pointer at a command.

    Two installs, two true answers, and the difference is on disk: a
    checkout has the crates, so the one-liner runs as printed; a wheel
    has none, so the answer is the whole bootstrap down to the clone.
    Taken straight from :mod:`gpuwm.bridges` rather than through its
    ``_bundle_first`` wrapper, so no branch of this refusal can grow a
    ``gpuwm fetch-bridges`` line at its head -- that is the loop this
    text exists to be out of.
    """

    if bridges.sources_present():
        return [f"  {bridges.cargo_build_one_liner()}",
                "  # builds every bridge this install's routes look for,"
                " offline."]
    return list(bridges.build_from_clone_hint())


def _unsupported_platform_note(platform: str | None) -> str:
    """Why no bundle can be staged here, and the commands that fix it.

    Circularity is the defect this text exists to avoid.  Both branches
    used to end "`gpuwm doctor` prints the exact steps for this
    install", and doctor's own next command for a missing bridge is
    ``gpuwm fetch-bridges`` -- so is ``gpuwm setup``, which runs it --
    so a reader who did as they were told arrived back at this same
    refusal having learned nothing and cost themselves two commands.  A
    refusal that hands over another command of ours has not named a
    remedy; the steps are printed here instead.

    The no-pins branch also has a state to name that nothing on the
    reader's disk shows.  The pins are computed from the published
    bytes by ``tools/build_bridge_bundle.py`` and written into
    :data:`PINS_RESOURCE` before the wheel is built, so a wheel
    installed from PyPI is the only artifact that carries them: a
    source checkout, and a wheel built out of one, both carry the empty
    document this branch is reading, and always will.  Saying so is the
    difference between "my install is broken" and "this install was
    never going to have them", and only the second one tells the reader
    which of the two routes below is theirs.

    A checkout is never told to ``pip install`` over itself: its
    sources are present, so the build is both shorter and true, and the
    wheel appears there as the explanation rather than as the step.
    """

    if platform is None:
        return "\n".join([
            f"gpuwm fetch-bridges: no bundle is published for "
            f"{host_platform_description()} (bundles exist for "
            f"{', '.join(SUPPORTED_PLATFORMS)}).",
            "gpuwm fetch-bridges: bundles are built per OS and "
            "architecture, so this is not something a newer install "
            "carries: no released wheel has bytes for this box, and this "
            "command has nothing it could stage here.",
            _MISSING_ARTIFACT_BREAKAGE,
            "gpuwm fetch-bridges: build the artifacts from a clone "
            "instead -- the route that works on every platform:",
            *_source_build_lines(),
        ])

    lines = [
        f"gpuwm fetch-bridges: this install carries no bundle pins for "
        f"{platform}.",
        "gpuwm fetch-bridges: the pins are computed from the published "
        f"bytes and written into {packaged_pins_path()} when the wheel is "
        "built, so a wheel installed from PyPI is the install that carries "
        "them; this one declares no platforms at all, which is what a "
        "source checkout -- or a wheel built out of one -- looks like.  "
        "This command stages only bytes it holds a SHA-256 for, so with no "
        "pins there is nothing here for it to fetch.",
        _MISSING_ARTIFACT_BREAKAGE,
    ]
    if bridges.sources_present():
        lines.append(
            "gpuwm fetch-bridges: this install has the Rust sources, so "
            "build the artifacts here:")
        lines += _source_build_lines()
        return "\n".join(lines)
    lines.append(
        "gpuwm fetch-bridges: install the published wheel, which carries "
        f"the pins for {platform} and the bundle they verify:")
    lines += [
        "  pip install --upgrade gpuwm",
        "  gpuwm fetch-bridges",
        "  # the second line stages the bundle; it refuses again only if",
        "  # the first one did not replace this install.",
        "gpuwm fetch-bridges: or build the artifacts from a clone "
        "instead, which needs no published bundle at all:",
    ]
    lines += _source_build_lines()
    return "\n".join(lines)


def _override_warnings(bundle: BundlePin) -> list[str]:
    """Environment overrides that would shadow what we just staged."""

    by_name = {artifact.name: artifact for artifact in BUNDLED_ARTIFACTS}
    notes: list[str] = []
    for pin in bundle.binaries:
        artifact = by_name.get(pin.artifact)
        if artifact is None:
            continue
        override = os.environ.get(artifact.env_var)
        if override:
            notes.append(
                f"gpuwm fetch-bridges: NOTE: {artifact.env_var} is set to "
                f"{override}, which wins over the staged copy; unset it to "
                "use what was just staged")
    return notes


def _print_listing(pins: BridgePins, bundle: BundlePin, dest: Path) -> None:
    print(f"gpuwm fetch-bridges: platform {bundle.platform} "
          f"({host_platform_description()})")
    print(f"gpuwm fetch-bridges: release {pins.release}")
    try:
        print(f"gpuwm fetch-bridges: bundle {bundle_url(pins, bundle)} "
              f"({bundle.bytes / (1024 * 1024):.1f} MiB)")
    except BridgeAssetError as error:
        print(f"gpuwm fetch-bridges: bundle URL unavailable: {error}")
    print(f"gpuwm fetch-bridges: destination {dest}")
    staged, stale, absent = classify_destination(dest, bundle)
    states: dict[str, str] = {}
    for state, group in (("staged", staged), ("differs", stale),
                         ("needed", absent)):
        for pin in group:
            states[pin.filename] = state
    for pin in bundle.binaries:
        print(f"  {states[pin.filename]:<8} {pin.filename} "
              f"({pin.bytes:,} B)")
    print(f"gpuwm fetch-bridges: {len(staged)} of "
          f"{len(bundle.binaries)} artifacts already staged and pin-valid")
    if bundle.assets:
        held, differs, needed = classify_assets(dest, bundle)
        total = sum(pin.bytes for pin in bundle.assets)
        print(f"gpuwm fetch-bridges: {len(held)} of {len(bundle.assets)} map "
              f"asset files staged and pin-valid under {dest / ASSET_ROOT} "
              f"({total / (1024 * 1024):.1f} MiB packed)"
              + (f"; {len(differs)} differ, {len(needed)} needed"
                 if (differs or needed) else ""))


def fetch_bridges_main(args) -> int:
    with bridges.inspection_only():
        return _fetch_bridges_main(args)


def _fetch_bridges_main(args) -> int:
    # Inside :func:`gpuwm.bridges.inspection_only` because this command
    # IS the remedy the stale-artifact refusal names.  A door that
    # resolves a stale binary refuses and points here; if getting here
    # could itself hit that refusal, the remedy would be unreachable
    # from the exact state it exists to repair.
    dest = (Path(args.dest) if getattr(args, "dest", None)
            else bridges.default_bridge_dir())
    platform = host_platform()
    try:
        pins = load_pins()
    except BridgeAssetError as error:
        print(f"gpuwm fetch-bridges: REFUSED: {error}")
        return 2
    bundle = pins.bundle_for(platform)
    if bundle is None:
        print(_unsupported_platform_note(platform))
        return 2

    if getattr(args, "list", False):
        _print_listing(pins, bundle, dest)
        return 0

    staged, stale, absent = classify_destination(dest, bundle)
    held_assets, stale_assets, absent_assets = classify_assets(dest, bundle)
    if not stale and not absent and not stale_assets and not absent_assets:
        print(f"gpuwm fetch-bridges: all {len(staged)} artifacts and "
              f"{len(held_assets)} map asset file(s) at {dest} verified "
              "(exact size + SHA-256); nothing to fetch")
        for note in _override_warnings(bundle):
            print(note)
        return 0
    if (stale_assets or absent_assets) and not stale and not absent:
        # The exact shape of the bug this asset half exists to close: a
        # complete set of binaries staged by an older release, and no
        # map assets beside them.  Say so, rather than reporting a
        # generic re-fetch.
        print(f"gpuwm fetch-bridges: the {len(staged)} artifacts at {dest} "
              f"are current, but {len(stale_assets) + len(absent_assets)} of "
              f"{len(bundle.assets)} map asset file(s) are missing or stale "
              "-- without them the renderer draws plots with no coastlines "
              "or borders")
    if stale:
        print(f"gpuwm fetch-bridges: {len(stale)} artifact(s) at {dest} do "
              "not match this release's pins and will be replaced once the "
              f"new bytes verify: {', '.join(p.filename for p in stale)}")

    source_dir = getattr(args, "from_dir", None)
    try:
        if source_dir is not None:
            installed = stage_from_dir(Path(source_dir), bundle, dest,
                                       progress=print)
        else:
            installed = fetch_bundle(
                pins, bundle, dest,
                keep_bundle=getattr(args, "keep_bundle", False),
                progress=print)
    except BridgeAssetError as error:
        print(f"gpuwm fetch-bridges: REFUSED: {error}")
        return 2

    remaining = [pin.filename for pin in bundle.binaries
                 if not matches_pin(dest / pin.filename, pin)]
    installed_names = {path.name for path in installed}
    broken = [name for name in remaining if name in installed_names]
    if broken:
        # Something this invocation just wrote fails its pin: that is
        # an integrity failure, and it stays a refusal.
        print("gpuwm fetch-bridges: REFUSED: staging finished but "
              f"{', '.join(broken)} still do not match their pins")
        return 2
    if remaining:
        # A partial --from DIR staging: everything written verified;
        # what the directory did not carry is reported, not fatal.
        warn(f"{len(remaining)} pinned artifact(s) are still not "
             f"staged ({', '.join(remaining)}); gpuwm doctor names "
             "the remedy for each")
    verified = len(bundle.binaries) - len(remaining)
    print(f"gpuwm fetch-bridges: {len(installed)} file(s) staged at "
          f"{dest}; {verified} of {len(bundle.binaries)} artifacts verified "
          "against the packaged pins.  `gpuwm doctor` re-checks the "
          "full estate")
    unstaged_assets = [pin.path for pin in bundle.assets
                       if not matches_pin(dest / pin.path, pin)]
    if unstaged_assets:
        warn(f"{len(unstaged_assets)} of {len(bundle.assets)} pinned map "
             "asset file(s) are still not staged; the renderer will draw "
             "plots without coastlines or borders unless it finds them "
             "elsewhere")
    elif bundle.assets:
        print(f"gpuwm fetch-bridges: {len(bundle.assets)} map asset file(s) "
              f"verified under {dest / ASSET_ROOT}")
    for note in _override_warnings(bundle):
        print(note)
    return 0


def staged_artifact_summary() -> str:
    """What a bundle carries, in one line, DERIVED from the table.

    ``--help`` used to carry a hand-written parenthesis -- "GRIB
    decoders, CPU preprocessing library, fetch backbone, batch renderer"
    -- written when those were all there was.  ``rw_nexrad`` joined the
    bundle and the line did not move, so the command that stages the
    radar front door did not say it stages the radar front door, and a
    reader looking for it had no reason to run this.  A hand-written
    inventory of a machine-readable table drifts on the first addition
    and stays wrong until someone notices; deriving it means the two
    cannot disagree.

    One phrase per artifact would be thirteen clauses, so the names are
    grouped by the workspace that builds them, in the table's own order,
    and each group names its members.  Adding an artifact adds a name
    here on the same commit, with no second edit to remember.
    """

    decoders = [a.name for a in BUNDLED_ARTIFACTS
                if a.crate == bridges.CRATE_RELATIVE]
    renderer = [a.name for a in BUNDLED_ARTIFACTS
                if a.crate != bridges.CRATE_RELATIVE]
    return (f"{', '.join(decoders)}; {', '.join(renderer)}")


def register_cli(subparsers) -> None:
    summary = staged_artifact_summary()
    parser = subparsers.add_parser(
        "fetch-bridges",
        help="download this platform's prebuilt Rust artifacts into "
             "~/.gpuwm/bridges -- the GRIB decoders and CPU preprocessing "
             "library, the mapped decode engine, the fetch backbone, the "
             "batch renderer, the dealiasing engine and the radar, MRMS, "
             "Stage-IV, surface and GOES observation front doors -- each "
             "one verified "
             "against the packaged SHA-256 pins before it is installed; "
             "idempotent when everything is already staged",
        # `description` and not `help` alone, because `gpuwm
        # fetch-bridges --help` prints the description and the audit
        # measured exactly that surface.  The artifact names are
        # DERIVED, so the day a bridge joins the bundle is the day this
        # text names it -- which is the property the hand-written
        # parenthesis did not have: it still said "GRIB decoders, CPU
        # preprocessing library, fetch backbone, batch renderer" long
        # after rw_nexrad and the dealiasing engine had joined.
        description=(
            "Download this platform's prebuilt Rust artifacts into "
            "~/.gpuwm/bridges, each verified against the packaged "
            "SHA-256 pins before it is installed.\n\n"
            f"This release's bundle carries: {summary}.\n\n"
            "Idempotent: everything already staged and pin-valid is "
            "left alone."),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--from", dest="from_dir", metavar="DIR", default=None,
        help="stage from a local directory instead of downloading "
             "(offline installs): either the bundle archive or the "
             "artifacts loose in it; verification is identical")
    parser.add_argument(
        "--dest", metavar="DIR", default=None,
        help="stage into DIR instead of ~/.gpuwm/bridges (gpuwm finds "
             "the default on its own; anywhere else needs the "
             "per-artifact environment variables)")
    parser.add_argument(
        "--keep-bundle", action="store_true",
        help=f"keep the verified archive under <dest>/{ARCHIVE_SUBDIR} "
             "after staging (default: remove it)")
    parser.add_argument(
        "--list", action="store_true",
        help="print the platform, bundle and per-artifact staged state, "
             "then exit without touching the network")
    parser.set_defaults(func=fetch_bridges_main)
    return parser


__all__ = [
    "ARCHIVE_SUBDIR", "ASSET_ROOT", "ASSET_URL_BASE_ENV",
    "BUNDLED_ARTIFACTS", "BUNDLE_MANIFEST_SCHEMA", "REQUIRED_ASSET_SUBDIRS",
    "AssetPin", "BinaryPin", "BridgeAssetError", "BridgePins",
    "BundlePin", "BundledArtifact", "LIBRARY_ABI", "library_abi_for",
    "PINS_RESOURCE", "PINS_SCHEMA",
    "SOURCE_REV_MARKER", "STALE_POLICIES", "STALE_POLICY_ENV",
    "StagedPinStatus", "forget_pin_memo", "refresh_staged_bundle",
    "staged_pin_status", "stale_policy",
    "SUPPORTED_PLATFORMS", "artifact_filename", "asset_url_base",
    "bundle_url", "classify_assets", "classify_destination",
    "download_bundle", "embedded_source_revisions", "fetch_bundle",
    "fetch_bridges_main", "host_platform", "host_platform_description",
    "load_pins", "matches_pin", "packaged_pins_path", "parse_pins",
    "register_cli", "sha256_file", "staged_artifact_summary",
    "stage_from_bundle", "stage_from_dir",
    "stage_from_loose_files", "staging_available", "verify_contract_marker",
    "verify_pinned_file", "verify_source_revision",
]
