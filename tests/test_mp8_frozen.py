"""The mp_physics=8 freeze gate -- WP-00 of the mp_physics=28 port.

This is the merge criterion for every other package on the
``feature/mp28-thompson-aerosol-aware`` branch.  Run it at the tip of each
package's work before that work is considered landable.

WHAT IT PROVES, AND WHY THAT IS ENOUGH
--------------------------------------
The mp=8 non-regression argument is a MECHANISM, not a measurement.
``gpuwm/core/kernels/__init__.py::load_module`` builds exactly one
``cupy.RawModule`` per ``.cu`` file from a single source string.  If that
string is byte-identical then the PTX is identical, therefore the register
allocation and FP contraction are identical, therefore every mp=8 result is
bit-identical.  No PTX-diff tooling is needed, and none should be relied on:
a PTX gate is a test that fails late and can be "fixed" by relaxing it.

So the primary assertion here is a digest of the *assembled* source string
-- captured by driving the REAL loader with a recording ``RawModule``, not
by re-implementing it -- for every ``.cu`` translation unit that existed at
the frozen commit.  That catches an edit to ``thompson.cu``, an edit to
``common.cuh``, a change to ``CUDA_DEFINES``, a change to ``_preamble()``,
and any loader change that is not perfectly inert.  WP-02's
``_EXTRA_HEADERS`` allow-list is exactly such a change, and this gate is
what holds it to its inertness claim.

Around the mechanism sit six receipts (R1..R6), each a real failure mode
that would otherwise produce a model that runs, stays stable and is
silently wrong -- the specific hazard of this port, where a half-converted
prognostic ``nc`` never raises anything.

R1  source identity of thompson.cu / thompson.py / every frozen module.
R2  the classic table contract (CCN_ACTIVATE.BIN must never appear in it).
R3  ``extra_moist_species`` -- Morrison's deliberate ``nc`` exclusion.
R4  the preflight allocation and scratch-arena surface.
R5  the nest-transition edge field codes (28 APPENDED, never inserted).
R6  ``acoustic`` n_mass selection plus the recorded ``_apply_thompson``
    launcher call graph, argument by argument.

Plus two fixture receipts: F1 freezes the 92 committed mp=8 oracle CSVs,
and F2 pins the four-file clean-rebuild exception together with a hermetic
witness for its cause (see :mod:`tools.mp8_freeze_receipt`).

New mp=28 files are expected and permitted everywhere: the gate pins what
existed and ignores additions.  It never passes because something was
deleted -- every pinned name must still be present.

This module does not import cupy in its own source and opens no device: the
loader capture replaces ``cupy.RawModule`` with a recorder before any
compile can happen.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import mp8_freeze_receipt as freeze          # noqa: E402

# ==========================================================================
# PINS.  RE-ANCHORED at the 1.4.1 merge, and the rule is unchanged: nothing
# on this branch may move a value below, and a diff that has to edit this
# file has by definition changed mp=8.
#
# The pins were captured from an orphan snapshot of ArWen 1.3.1
# (789f61181fb0b198ace10775f3ea184eb5e786a3), the last tree before any
# mp_physics=28 work.  That tree is 371 commits behind the release line the
# port now sits on, and ten of the pinned kernels moved on that line while
# the port was off it: eight noahmp_* translation units, rrtmg_sw, and
# thompson itself, the last by 5e4af4e3 ("the rain MVD bound belongs to
# TAU+1") and cb765336 ("the rain-presence gate is a mass concentration").
# SCRATCH_SLOT_REGISTRY_MP8 gained physics_validation_status the same way.
#
# Re-pinning a freeze is exactly how a freeze gets defeated, so the new
# values were not taken on trust.  Every one of the ten was verified to be
# BYTE-IDENTICAL to integration/release-1.4.1 -- `git show
# integration/release-1.4.1:gpuwm/core/kernels/<name>.cu | sha256sum`
# equals `git show HEAD:...` for all ten -- before its digest was written
# here.  The compiled-string digests are safe for the same reason at one
# remove: PREAMBLE_SHA256 and COMMON_CUH_SHA256 below did NOT move, and
# kernels/__init__._EXTRA_HEADERS lists only the six mp=28 translation
# units, so _extra_header_text() returns "" for every frozen module and
# their assembled sources are still exactly _preamble() + source.
#
# The freeze therefore still asserts what it always asserted: mp=28 has not
# touched mp=8.  What it no longer asserts is that mp=8 never changes --
# it does, on its own line, in its own commits, and this branch inherits
# those by merging rather than by editing.
# ==========================================================================

#: The release-line commit these pins were re-captured against.  The
#: original capture point, 789f61181fb0b198ace10775f3ea184eb5e786a3, is an
#: orphan snapshot with no parent and no descendant; it cannot be diffed
#: against the current line, which is why the anchor moved.
FROZEN_COMMIT = "b15a2558a569c661e148c0db3cfc99896f3af91a"
FROZEN_COMMIT_ORIGINAL = "789f61181fb0b198ace10775f3ea184eb5e786a3"

# -- R1 --------------------------------------------------------------------

THOMPSON_CU_SHA256 = (
    "3ca6b7e902d2d77ca9881a66eb484141df552c8ea30a0076bd07023e7255e760")
#: sha256 of ``_preamble() + thompson.cu`` -- the exact string nvrtc sees.
#: THIS is the mp=8 numerics guarantee.
THOMPSON_COMPILED_SOURCE_SHA256 = (
    "8cb23f0a78b1e48a402266fd4f841b2facdc7a3ee4b7d53e3c5482aae973775d")
THOMPSON_COMPILED_SOURCE_LEN = 340875

COMMON_CUH_SHA256 = (
    "c78b17cb02ef67a2ad24d19e06e1129d7d5bcda74b972b38470fd33a6e58ff43")
PREAMBLE_SHA256 = (
    "1888bcf077e4251398b3baf7cca53b7c15ba993ab791992b7e278c3e32bb75e8")
PREAMBLE_LEN = 768

CUDA_DEFINES_PIN = {
    "G": 9.81, "RD": 287.0, "RV": 461.6, "CP": 1004.5, "CV": 717.5,
    "P0": 100000.0, "T0": 300.0, "GAMMA": 1.4,
    "RCP": 0.2857142857142857, "RCV": 0.4,
    "RVOVRD": 1.6083624362945557, "RERADIUS": 1.5698587127158556e-07,
    "XLV": 2500000.0, "SVP1": 0.6112, "SVP2": 17.67, "SVP3": 29.65,
    "SVPT0": 273.15, "RHOWATER": 1000.0, "EP2": 0.6217504332755632,
}

THOMPSON_PY_SHA256 = (
    "ede422fe6c2acba76771e40aabd231ad0edf8b391db5923535f67ac0b85f70e8")

#: ``gpuwm/core/thompson.py::__all__`` verbatim, in declaration order.
#: mp=28 launchers live in the new ``thompson_aerosol_*.py`` modules; not
#: one name may be added here.
THOMPSON_PY_ALL = (
    "launch_cloud_freezing",
    "launch_cloud_saturation_adjust",
    "launch_cloud_sedimentation",
    "launch_classic_graupel_number_init",
    "launch_classic_graupel_number_finalize",
    "launch_cold_cloud_source_network",
    "launch_cold_rain_source_network",
    "launch_cold_rain_snow_graupel_network",
    "launch_effective_radius",
    "launch_final_phase_cleanup",
    "launch_frozen_vapor_network",
    "launch_frozen_vapor_network_from_owner",
    "launch_graupel_cloud_riming",
    "launch_graupel_fallout_column_mask",
    "launch_hydrometeor_column_mask",
    "launch_graupel_sedimentation",
    "launch_graupel_melting",
    "launch_graupel_sublimation",
    "launch_ice_autoconversion",
    "launch_ice_deposition",
    "launch_ice_nucleation",
    "launch_ice_sedimentation",
    "launch_rain_evaporation",
    "launch_rain_freezing",
    "launch_rain_graupel_collection",
    "launch_rain_ice_collection",
    "launch_rain_snow_collection",
    "launch_rain_sedimentation",
    "launch_rain_self_collection",
    "launch_snow_sublimation",
    "launch_snow_cloud_riming",
    "launch_snow_ice_collection",
    "launch_snow_melting",
    "launch_snow_rime_conversion",
    "launch_snow_sedimentation",
    "launch_snow_vapor_exchange",
    "launch_warm_autoconversion",
    "launch_warm_process_network",
    "launch_warm_frozen_source_network",
    "launch_warm_frozen_source_network_from_owner",
    "launch_warm_rain_collection",
    "launch_warm_saturation_adjust",
)

#: The constant-Nt_c inventory, MEASURED on the frozen tree.  Every line
#: here hardcodes what mp=28 must make prognostic.  Recorded so that a
#: reviewer can see the port never "fixed" one of them in place, and so
#: that the claim in the port spec is checkable rather than asserted.
#:
#: NOTE, and this corrects the spec's summary paragraph: the measured
#: counts are 13 / 6 / 2 / 3, not 12 / 7 / 3.  ``thompson.cu:343`` is the
#: one CORRECT 2730.0f -- ``calc_effectRad`` genuinely uses WRF's integer
#: ``g_ratio`` PARAMETER there -- and the 272.0f site is line 1007, not
#: 1006.
THOMPSON_CU_LITERAL_SITES = {
    "100.0e6f": [343, 895, 1012, 2094, 2903, 2970, 3187, 3793, 4018, 4141,
                 4273, 4693, 6945],
    "2730.0f": [343, 895, 1012, 4018, 4141, 4693],
    "272.0f": [901, 1020],
    "cloud_number_bin = 65": [3946, 4359, 7018],
}

#: Every ``.cu`` translation unit present at the frozen commit, as
#: ``name -> (file sha256, assembled-compile-string sha256)``.  Modules
#: added later (the mp=28 ones) are ignored by the gate; a pinned name
#: that disappears is a failure.
FROZEN_MODULE_DIGESTS = {
    'acoustic': (
        'f1cd7276428f19d6090483a01705258643fd9ea8993974f0f4010721e8c05e8c',
        'aa7b93e78bb92d9442f3d76b4f37e5d6c7e1470ac325aad1fc79684692229b74'),
    'advection': (
        '8a88c2fc0ed833e9fc5bd55bd3f3f78752fbb9e68714122c2fb68adc368d2d7e',
        '3449e3bc306ef5ba9a374e0e04b6e0f7601cbe6e3d7b0aa6ba6b8edd91c8d16e'),
    'coriolis_map': (
        '53fc37398d086dcf54c317917872e5d8d473b83d6d217697335d5ae6bc96d143',
        '7981384362444c8e782abb80efb29cbf61d0a3183cc0c316799b3bf9f3c43671'),
    'diagnostics': (
        'b1e2fb2866d73e37d98a0ae3aed791ffd3fd6242f3413e617957049903386e52',
        '76ed32fcddf6cb1bd196d447c1d10b77feec652d6c7d3c764cb8f3110c7da774'),
    'diff6': (
        '7dbcfb2d4e259ad36a3d29705e936a276b56e9ea52511c5f82054749e38302e9',
        '563febbc809cd53695782a77095b3ab01f60c64866657091b790f792f39a5394'),
    'diff6_seam': (
        '776ed7053a2dd697b0401e87600073e8dcfcdb3f7136fa4fcf73bb0b4972b464',
        '7af0e3ddeb9dc13d94992bf66ff65af583dd36e229de73236d0ed1da39f05d92'),
    'diffusion': (
        '00fb2e5d5550680fef154b4f67c7e282ad7ca1b170df59abdea89f888dad91ef',
        'f4958de3298bfcd764a5fd848aedfbb13938043ab91132db419408951c0a061e'),
    'dycore': (
        '93846074b99bdb5f03b33da51c7f70aeedc2cb0d9058d3baac43c107f8387e90',
        'a1510158f36fc8b2289e309339dba17f371f6c6f79f51f600b2be621abaadcdf'),
    'health': (
        '381575b3b81aed334bd175c98abfe7360842b2783c01e8edf7cb6e3ca38fc7e0',
        'df8a33c13788d1832995a34dc9a3259a933d1be302ea182ca9e13a4145838ae8'),
    'kessler': (
        'fecf2e8028fda0ed4cb47fccce4c602d4632048d2dcbdd163613685ded952fdc',
        '530faef7f3bc5e5600d7a5f1086c9e4d0914a3aeda735214072bed30907c05d7'),
    'kf': (
        '287b91e65affe88194822fafb2b428938db3ac84b6aef239b1a98aa4bcac956b',
        'b15a1ef99e8a0e7994e68ad4c2e34957a2c9772937ba035fa7bed63ce6381752'),
    'lbc_flow': (
        '09ad5f0ad10b75efb9207c5217aa6eba2c4a4a45d20bf111dd517c2867b92e64',
        '6383f03f152f827b1bef4cc3fb42b3e4dd0a34aa5e817dbf9323f894398de0c4'),
    'lbc_state': (
        'fadf66fea201e4eac56e8a58d72b11940325b142a4e08fcc0b8db80fd78b53ec',
        '4cd1c59322d6a800eaebee8182a5a2a25413c37ffe4681f56a164a71ebd3a47b'),
    'morrison': (
        'fb0e18e4df5c78735b9aac80e719fcd2703497765422a72087794c6183e0033f',
        '2e6c69adedf7a2c3195840d7906fb2ae5e9a637bd74fa604e03e0b15bb6fd53d'),
    'mynn_pbl': (
        'b53ab90e634e61367afadfaa77667c8f2eb2430fc061ce9976509fe0e2f4490e',
        '87f80d06cc7724fd1277eefbf91738fe8eb0e774768ed64292cb1157f19a2d84'),
    'mynn_surface': (
        'a94de3ff2da95c37e12b437123b4a3807ac1318c524b339609f6024f4d21f85b',
        '891ec5d565c720afabab57169f1a3b1aa95efc3d1ac84e87ffbd4ebb239c57fe'),
    'nest': (
        '19ebb671e2fc3d25f7d4ec571ec3bcfa4da6d137c175f02f232d865b0fb37e6a',
        '3635d3b6ae282e9f341898286ecdf769ed27003dbb713168d18c02d4df75391d'),
    'nest_microphysics': (
        '674374b70247c65fd818ab7845f375db46e859f28a6b88e977b938c70bf2e859',
        '78a85fff68e0312503549e5d34214b8dc4a89e11e5478199b90079ae5cfd1caf'),
    'noah': (
        'c3eefebad446acb74bcb3c3666f90789f278560fabfefc0b08da9d38496bf245',
        '57c25845288d4de8c66030c8a0ead986dd03355c49f1cdaabcebadfcb2835a85'),
    'noahmp_bareflux': (
        '54fb5065e95b24d4cf676e2deda29bae44b3e9305d3d98cbc1abf5ed55f444ce',
        'fbb19fc8b5668ea2edbcc1270f8ffe367475124ffa0ce99d3bc34639f3f31e9f'),
    'noahmp_driver': (
        'bd555be10ccade5a5bdddcaf4c56b7f4353dae1208fc48a5586fb7ce7d32d643',
        'f1913fe0054adb74188effa6499b799e989cfb1e96ecd676edcd43df0083030e'),
    'noahmp_energy': (
        '46c5f15a0590357144f5447093dff1fc2f6dcdd2edf98aefec63a45cd7f35090',
        '53f9a5d9d243a1445c9ad1ec5725d3393e7f08f54ea88eb2622dc8d51ae6b58c'),
    'noahmp_fluxprep': (
        'eef473608e9d1c0176574c6bd72183659249b42a022f8d63d77c611de506f4ed',
        '4ae139ef5d31234b5e9fdbc04c4cfd3c5156cd91d0326e5544b5b78f7c17fb31'),
    'noahmp_leaves': (
        '0ce9461705395dccbfebed3d9d27e87eebaeaca79896ae369eaa02ec1e77307f',
        'c0dd5d46d2cfe191d36d74c68331ddfecd6a5f0174ad6582460b399eb6afc388'),
    'noahmp_libm_slab': (
        '0144fa7d142a8d24f5f0f52bd0dade987312cd97c2a2efad9a6f33edb1a35fda',
        'c7cdc57aa7d3d507d2b57935df13e16dba87dd783384e28a7240b95a536f39a7'),
    'noahmp_radiation': (
        'c26d7a68ed86bd7182ddbea5e2005fb4761805dd5eaf13c3115e640eb2234c22',
        'a454d07fefa080f986423d2f74bf0fe1ecf85959638e02a7a447b44d68aea24a'),
    'noahmp_sflx': (
        '47f51fed351b3f720203c07ea2d5e5a8902f4162afe17b634f5196447728365f',
        'f17caed98ec58c4d9836778f3a9edea9e88cf13808dbe64d5e099fd81ffebc77'),
    'noahmp_snow': (
        'f46f850fb54acf3dadcaf7a7df9bb4c09d107be8de0c577b704536a20271b42a',
        'd375fe9595c514690fb7fe6db756b76f4c2341862dd5c3b70bae2131ccfcde22'),
    'noahmp_soilwater': (
        'a53adc6aaa1a46974b13497676e682c56587176442bf1a26003b4d377d87688e',
        'd9e8788e7bb479104faabbb3ca5d6ae698768a394a60f54b0e2840862bfbfa5a'),
    'noahmp_thermal': (
        '2598ca76b7f9c0d6de35631d5113901d1e25f24a014c8ea4e0df490158d16c86',
        'f7eef131507a29def54e2387686f34a4807851354f446684deaf5322dc226753'),
    'noahmp_vegeflux': (
        '2178b13989853a7433869bb35e7974df004ddb9f441a96946672c03730472b3c',
        '688723dc9eef069b5e82338a8023438636d61db2db1dfd583cd1ced03b923a19'),
    'noahmp_vegprecip': (
        '9ce5667599d111be0efc7cb0871e5719d70259bb95b30fbe923e01964b75c26c',
        '599e85f0fe59a28ae5fd5437e47db7f41c31abd21783979f66b987b212f1f41a'),
    'noahmp_water': (
        '4154bace0d97235503d4ca9ed6cb4877c8543762f2f384dfc8883fe3b2ed429e',
        '6cfeaef3fd00b8054d1761fd8bcfd6a920a6fb476f53986672ef300cbadd53ba'),
    'nssl2': (
        '18e828ed1d6c2d2d69c5e146c7b1d661f3d474a7246cf38daf45e6b0d49951d4',
        'a9639658a1d2f0a8127293682ec14aead9eee86735fbfa1ec5853d3e78812cbd'),
    'nssl2_diagnostics': (
        'a95ae9e0bc3dd20a13865cfa6d1148d2a78ee5d7c17c9c1bca9a0c8dbdf19868',
        '331b4a9734959260ab515216e24ee7100eac18d8e6d646b1e0e8bf21c0c23374'),
    'nssl2_driver_support': (
        '4e52a801fbf2b5b1f877e5d1b0a02cdc00eec775e55271fbbdeb4cf490edfe51',
        '43c74556040cc34ad3f3e2ea3171049cb7421e41e97aed2e21e8691703a6282d'),
    'nssl2_fused_gs': (
        '4b6bb119e54a19039f2b81e46964a1ac38d3bb835b05a9f4c16c789720fad9c8',
        '8acb6a2ac2a8c292a5f30e527c91276c941cf512593456a14986965f1c2eb593'),
    'nssl2_nucond': (
        'e7b2df8d8fd6a0dd98c464d0db595437fc701143be7d3652bf565659253a2244',
        'd4c9659486dbe9e3acb5b91efb979e89713d5e2a527aa5f8680083a7115d93f0'),
    'nssl2_qvexcess': (
        '6906dcd9f8822d73d87ff3cb6e545a1b1ddef567c16c658435d4f669f1f369dc',
        '89d27036499b7f57d780c594308766b83ef711fbc9d9c5d0ece2e79c270f6626'),
    'openbc': (
        'a929bad2ec82ae36f86dcc10e1d315460f8af947cbcfd7bdbd87696e40e57624',
        '00b901c9f26df6447906726324593fe7b476679629ed1ab6ca16537cfbc27044'),
    'pd_advection': (
        '606e396872b2c42bafcff8d46d6a4c16d0f1c4f0fc796bb1728f4ae3678c309c',
        'd9e8649915c1a8bd0b65354131baa8fc29d00f921d61849bb4fa0d47b065c9ea'),
    'refl': (
        'ff4e3c6dd532be49fc866692e829e9a7efa7ac1c65a95b4cd1beb910f39d07b6',
        '8e3843a3884edee0ed8ade032401df5750ccde3df2d396a4d17a59267d0f27a1'),
    'rrtmg_lw': (
        'd6bd85ef62136e77c083d929fbe923135b0165b3f689206778b82170879d5583',
        '0a7af901550d4ad74566f612928549377ee7c839bb5643d7ad0ba1010dbb208f'),
    'rrtmg_lw_chain': (
        'a71a779ea3a733e422414ee5d5d885cb904972fe32ac2bfe6438d6c5214c1252',
        'ee8f4c95e52248d0b641a48a8532d44d6aff0b75344dda59a416a3905ae972c1'),
    'rrtmg_lw_taugb02_10_11_12': (
        'fe5d57d1eb2649005d2748ae40f849c340b03a5f3d3a61c7fb506d0c69792485',
        'f7d4c98b0ed523e2c5912d4810925c60c3bcbb52d3e1a40d06a2708686cb632c'),
    'rrtmg_lw_taugb03_05': (
        '5de3f0c7c700cff993d5d5a3caa868bbd68d810ea30a902e6e92d652e5612cf7',
        '8a86d6cfcd46ea76bbc3027cb1f68f8842d4e37de5b011d681a5c502c1794b2e'),
    'rrtmg_lw_taugb06_09': (
        'ab677eb0de0faeb2b7a3f4a5923c6777e11128ab7950d9b63abc9cdb61125157',
        '4899dd33838614a7cf94bf3e74291039ed7bb3b35806495a22d3e1c559d01c01'),
    'rrtmg_lw_taugb13_16': (
        '104c3157451dc878e3f9d751db1cf01a4368b1a2892825678370c5159a787ff6',
        '56f315a0258d902c69f99abacbe5e53005f50ce478725c146647d3df26c71522'),
    'rrtmg_mcica_wrf': (
        'edb6bcb71a9d0763d3576b602db0afa05ec5eddc9bc63548f4991f34bd6d718c',
        '07af9ba6f5a0ed7e3c735f0b574041aace48f306e2641e7563252ae992a2cfd9'),
    'rrtmg_sw': (
        '36d2e51fee63649acb551fc3fb2d4996a0a556ddf1e09cb3346a9532eb58d8e5',
        '6b790600388493317e095e04538dc5197d12ef7c75283b190420ffbe47309825'),
    'rrtmgp_cloud': (
        '015aec6065be8a23bcec1ce5421ae28cfbc74de1d6a7713a75bc1a78d1f7bc08',
        '5976824ca813f3e40a8b6d73ccb88d39b333f110054bb502d482817a3a7c6ad7'),
    'rrtmgp_gas': (
        'e744423659529d02d2df9eb6cc9a96ced538fceeb6677ab6f33796166b643212',
        'd1da202a9a0a59bbb646659431a67d81b734c9ce0c245b27f80b1d20378ce7fb'),
    'rrtmgp_mcica': (
        '2f816f5e7adcfc7cf6ba405af9b70aa71d4271ba9b21836bba7d45a7d73b0be9',
        'd324e9c18e7656190cb4a1aafc5dcf783def58e789b307cbad4ca1a5511e2f9e'),
    'rrtmgp_rte': (
        '684184537fe8023f78c20037ddc4396f2923a6f79517835f6a0c0283b9e89388',
        '32ace441d06180613420ba4c5c4d1693cc8ddee95760e1d56ec23637300bdee9'),
    'rrtmgp_validation': (
        'c3a2554827e7c39db269d0fa1d5ad594d1623d6974be859a1ae3a044d438951f',
        '36f7a4dccc8c66be225e16fdd6088f0fafa7a357ac1aba9b4a99f98af297c370'),
    'ruc': (
        'd446b7462e4952416d3e21482b051823766a6f675163236686c7d9fab7fbbdb7',
        'f3fa5f309adf4de861d35594a884ad9ed5abcc2b252a61a171354459cc8b8028'),
    'saxpy': (
        '8637cb5cb0a6878d59a32454a6ae662a8b18c0be4d94c067fbde1e4bf5bad079',
        '7b7083065716a2b3b58d47c3ac456ea8d0c1a38ec771219897917bb0b1b79cb2'),
    'sfclay': (
        '3c8ce6512c15d480b831e76ece064f94d8ed3ee8ee2a950fc9c74b8daf14b31a',
        '223f5aaa69f4de5e434467bb517894533e64973c175f4e248609a6eddfbc0179'),
    'smag2d': (
        # Re-pinned on the 1.5 integration line: feature/les-integration's
        # verified km_opt=2/3 work edits smag2d.cu after this table was
        # frozen on the mp28 lane.  smag2d is not an mp=8 translation unit;
        # the mp=8 numerics guarantee is untouched.
        '50d824f885452ebb7806c38c1a6ed45e4976c9caea9d6c9d8338e126aa4c50da',
        '4fef25ab15bddb0847f7a79c54b54a80e6b34217d8747bedfc6fbd67bf682bae'),
    'spec_bdy': (
        'bcc7090fbbb8ea307bd6dd6c65ab9b8a3f56948c4752ae3d744127b450d20161',
        'bc03ed595bacc546d8e041fbb1d11b5bb3b3b90760ef06ea1dd1f0f18b4de931'),
    'thompson': (
        '3ca6b7e902d2d77ca9881a66eb484141df552c8ea30a0076bd07023e7255e760',
        '8cb23f0a78b1e48a402266fd4f841b2facdc7a3ee4b7d53e3c5482aae973775d'),
    'uh_diag': (
        'cbfc98e8d025a4511fd7f8a41ca4bd163c261da4a48dec22bb979ec5a496b14e',
        '9dc88c6e14b2aaaa4249a9f844dc231f105431623375c988a2894e322de2f3ea'),
    'vert_interp': (
        'ab608d651d99f882a42881f7b977b29d8879f474ea26404a79d06c77a86125d9',
        '65b0fe8a9aadda83084e0286d3f70570e251bb6d2fb1a0636a0530103b8d9d89'),
    'wsm6': (
        '0526192b79d90d3be7c733a475987216d37cc81b17f8de4f1fe3e4220a6b81d7',
        '1a6d20da0d450f235227fe609bdb12b368d96aec5ac231752074ff4dd9cc50e6'),
    'ysu': (
        '1ae69b78b0cbe9ff572da9dbe5576809ab735288dfcf0c11fc4d66b6d7fb7f91',
        '2f8256dd2793074df630b8592c1b811263d38d08645db602f0faf0325278e21d'),
}

# -- R2 --------------------------------------------------------------------

CLASSIC_TABLE_ASSETS_PIN = (
    ("qr_acr_qg_V4.dat", 74_966_480,
     "89b779855847b2acdca1b40e24c5f1bd89b0c6ed105ca91a5a076d80c2437c3f"),
    ("qr_acr_qsV2.dat", 43_764_288,
     "47350be20bd59c9f31378dd5805ce7d35fd14bebcfafb4ade56626f6eed818d7"),
    ("freezeH2O.dat", 254_944_848,
     "c235d1ce6f8750a671b2273d0e216ed3acf9a869bfd52a14676826f87aab5c02"),
    ("thompson_aux_tables.dat", 6_164_536,
     "a1bda803cdb53aedce8a2970c04c355fad19e3744398e1c9b13a876f09730547"),
)
TABLE_SET_ID_PIN = "wrf-v4.6.1-classic-thompson-mp8-gfortran13-v1"
WRF_REFERENCE_VERSION_PIN = "v4.6.1"
WRF_REFERENCE_COMMIT_PIN = "d66e442fccc04111067e29274c9f9eaccc3cef28"

# -- R3 --------------------------------------------------------------------

EXTRA_MOIST_SPECIES_MP8 = ("qi", "qs", "qg", "nr", "ni")
EXTRA_MOIST_SPECIES_MP10 = ("qi", "qs", "qg", "nr", "ni", "ns", "ng")
TRANSPORTED_NUMBER_SPECIES_PIN = ("nr", "ni", "ns", "ng")

# -- R4.  Captured for the fixed probe config nx=8 ny=6 nz=4,
# moist=True, mp_physics=8 (see freeze.PREFLIGHT_PROBE_CONFIG).
# ------------------------------------------------------------------
STATE_ARRAY_SHAPES_MP8 = {
    'al': (4, 6, 8),
    'al_pp': (4, 6, 8),
    'alb': (4,),
    'alt': (4, 6, 8),
    'c1f': (5,),
    'c1h': (4,),
    'c2f': (5,),
    'c2h': (4,),
    'c3f': (5,),
    'c3h': (4,),
    'c4f': (5,),
    'c4h': (4,),
    'cosa': (6, 8),
    'dn': (4,),
    'dnw': (4,),
    'e': (6, 8),
    'effc': (4, 6, 8),
    'effi': (4, 6, 8),
    'effs': (4, 6, 8),
    'f': (6, 8),
    'fnm': (4,),
    'fnp': (4,),
    'h_diabatic': (4, 6, 8),
    'ht': (6, 8),
    'msft': (6, 8),
    'msfu': (6, 9),
    'msfv': (7, 8),
    'mu_pp': (6, 8),
    'mub2d': (6, 8),
    'mup': (6, 8),
    'mup0': (6, 8),
    'ni': (4, 6, 8),
    'ni0': (4, 6, 8),
    'nr': (4, 6, 8),
    'nr0': (4, 6, 8),
    'p': (4, 6, 8),
    'p_pp': (4, 6, 8),
    'p_pp_old': (4, 6, 8),
    'pb': (4,),
    'ph_pp': (5, 6, 8),
    'phb': (5,),
    'php': (5, 6, 8),
    'php0': (5, 6, 8),
    'qc': (4, 6, 8),
    'qc0': (4, 6, 8),
    'qg': (4, 6, 8),
    'qg0': (4, 6, 8),
    'qi': (4, 6, 8),
    'qi0': (4, 6, 8),
    'qr': (4, 6, 8),
    'qr0': (4, 6, 8),
    'qs': (4, 6, 8),
    'qs0': (4, 6, 8),
    'qv': (4, 6, 8),
    'qv0': (4, 6, 8),
    'rdn': (4,),
    'rdnw': (4,),
    'rmu_t': (6, 8),
    'rph_t': (5, 6, 8),
    'rth_t': (4, 6, 8),
    'ru_t': (4, 6, 9),
    'rv_t': (4, 7, 8),
    'rw_t': (5, 6, 8),
    'sina': (6, 8),
    'th_pp': (4, 6, 8),
    'thb': (4,),
    'thp': (4, 6, 8),
    'thp0': (4, 6, 8),
    'u': (4, 6, 9),
    'u0': (4, 6, 9),
    'u_pp': (4, 6, 9),
    'v': (4, 7, 8),
    'v0': (4, 7, 8),
    'v_pp': (4, 7, 8),
    'w': (5, 6, 8),
    'w0': (5, 6, 8),
    'w_pp': (5, 6, 8),
    'ww_pp': (5, 6, 8),
    'znu': (4,),
    'znw': (5,),
}
SCRATCH_SLOT_REGISTRY_MP8 = {
    'acoustic_a': (5, 6, 8),
    'acoustic_alpha': (5, 6, 8),
    'acoustic_c2a': (4, 6, 8),
    'acoustic_gamma': (5, 6, 8),
    'acoustic_mu_pp_old': (6, 8),
    'acoustic_th_pp_old': (4, 6, 8),
    'adv_ru': (4, 6, 9),
    'adv_rv': (4, 7, 8),
    'adv_rw': (5, 6, 8),
    'integration_health_aux_ptr': (2048,),
    'integration_health_bounds': (1024, 2),
    'integration_health_field_ptr': (2048,),
    'integration_health_field_size': (2048,),
    'integration_health_flags': (1024,),
    'integration_health_partial': (1, 9),
    'integration_health_planes': (1024,),
    'integration_health_result': (8,),
    'integration_health_status_bits': (2048,),
    'integration_health_validation': (4,),
    'moist_pd_q0': (4, 6, 8),
    'moist_rq_t': (4, 6, 8),
    'mp_dz8w': (4, 6, 8),
    'mp_graupelnc': (6, 8),
    'mp_graupelncv': (6, 8),
    'mp_pii': (4, 6, 8),
    'mp_rainnc': (6, 8),
    'mp_rainncv': (6, 8),
    'mp_snownc': (6, 8),
    'mp_snowncv': (6, 8),
    'mp_sr': (6, 8),
    'mp_th': (4, 6, 8),
    'mp_thompson_frozen_reference_density': (4, 6, 8),
    'mp_thompson_frozen_reference_temperature': (4, 6, 8),
    'mp_thompson_graupel_melt_marker': (4, 6, 8),
    'mp_thompson_graupel_number_shadow': (4, 6, 8),
    'mp_thompson_rain_reference_density': (4, 6, 8),
    'mp_thompson_snow_melt_marker': (4, 6, 8),
    'mp_thompson_snow_velocity_boost': (4, 6, 8),
    'mp_thompson_temperature': (4, 6, 8),
    'mp_z8w': (5, 6, 8),
    'pd_fxc': (4, 6, 9),
    'pd_fxl': (4, 6, 9),
    'pd_fyc': (4, 7, 8),
    'pd_fyl': (4, 7, 8),
    'pd_fzc': (5, 6, 8),
    'pd_fzl': (5, 6, 8),
    'physics_validation_status': (1,),
    'refl_10cm': (4, 6, 8),
    'refl_t': (4, 6, 8),
    'rk_ru': (4, 6, 9),
    'rk_ru_m': (4, 6, 9),
    'rk_rv': (4, 7, 8),
    'rk_rv_m': (4, 7, 8),
    'rk_ww': (5, 6, 8),
    'rk_ww_m': (5, 6, 8),
}
NEST_FIELD_KINDS_MP8 = (
    'u', 'v', 'w', 't', 'ph', 'mu',
    'qv', 'qc', 'qr', 'qi', 'qs', 'qg', 'nr', 'ni',
)
STATE_ARRAY_SHAPES_DIGEST = (
    '46ab221ff2a84d7b0e5caa8fc575a77b448e3d443cd277b03f49cf0d2c7fa92a')
SCRATCH_SLOT_REGISTRY_DIGEST = (
    'cfa4fe7ed787889825d504ebb122e0a7042cc8de177ae33367b6cfc8f3ec6d2c')
ORACLE_FIXTURE_COUNT = 92
#: RE-PINNED with the corrected oracle, not with an edit.  The Thompson
#: oracle lane found that five committed fixtures were the output of a
#: libmvec-linked build and regenerated the whole set devectorised
#: (fix/thompson-oracle-devectorized, 576b755b/8290189d), which this branch
#: merges rather than reproduces.  See the inverted witness below.
ORACLE_FIXTURE_AGGREGATE_SHA256 = (
    '6c7f555df2a44206b1cb12013b68c50bd57ac94f4dd0967f1bf0595b7a9eac53')

# -- R5 --------------------------------------------------------------------

PORTED_MP_PHYSICS_PIN = (1, 6, 8, 10, 18)
#: The 20 pre-existing nest-edge field codes.  ``28`` must be APPENDED to
#: ``PORTED_MP_PHYSICS``; inserting it renumbers this table and silently
#: re-points the ratified mp8 -> mp18 nest edge at different fields.
EDGE_FIELD_CODES_PIN = {
    "qv": 0, "qc": 1, "qr": 2, "qi": 3, "qs": 4, "qg": 5,
    "nr": 6, "ni": 7, "ns": 8, "ng": 9,
    "qh": 10, "qndrop": 11, "qnr": 12, "qni": 13, "qns": 14,
    "qng": 15, "qnh": 16, "qnn": 17, "qvolg": 18, "qvolh": 19,
}

# -- R6.  The recorded _apply_thompson call graph.  Arguments are
# identity labels, not values: 'state.qc' or 'scratch[mp_dz8w]'.
# ------------------------------------------------------------------
ADAPTER_CALLS_NO_REFL = (
    ('save_pre_mp_theta', (
        '<_HostAdapterState>',
     ), {}),
    ('launch_classic_graupel_number_init', (
        'state.qg',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'scratch[mp_thompson_graupel_number_shadow]',
     ), {}),
    ('launch_frozen_vapor_network_from_owner', (
        'state.qi',
        'state.ni',
        'state.qs',
        'state.qg',
        'state.qr',
        'state.nr',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        '<classic-table-owner>',
        '10.0',
     ), {
        'graupel_number_shadow':
            'scratch[mp_thompson_graupel_number_shadow]',
        'qc':
            'state.qc',
        'snow_velocity_boost':
            'scratch[mp_thompson_snow_velocity_boost]',
    }),
    ('launch_warm_frozen_source_network_from_owner', (
        'state.qc',
        'state.qr',
        'state.nr',
        'state.qs',
        'state.qg',
        'scratch[mp_thompson_graupel_number_shadow]',
        'scratch[mp_thompson_graupel_melt_marker]',
        'scratch[mp_thompson_snow_melt_marker]',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        '<classic-table-owner>',
        '10.0',
     ), {}),
    ('launch_hydrometeor_column_mask', (
        'state.qr',
        'scratch[mp_rainncv]',
     ), {}),
    ('launch_hydrometeor_column_mask', (
        'state.qc',
        'scratch[mp_snowncv]',
     ), {}),
    ('launch_graupel_fallout_column_mask', (
        'scratch[mp_thompson_frozen_reference_temperature]',
        'state.qg',
        'scratch[mp_sr]',
     ), {}),
    ('launch_cloud_saturation_adjust', (
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'state.qc',
     ), {
        'reference_density':
            'scratch[mp_thompson_frozen_reference_density]',
        'reference_temperature':
            'scratch[mp_thompson_frozen_reference_temperature]',
    }),
    ('launch_rain_evaporation', (
        'state.qr',
        'state.nr',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        '10.0',
     ), {
        'graupel_melt_marker':
            'scratch[mp_thompson_graupel_melt_marker]',
        'reference_density':
            'scratch[mp_thompson_rain_reference_density]',
    }),
    ('launch_cloud_sedimentation', (
        'state.qc',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'state.w[view:(3, 1, 1)]',
        'scratch[mp_dz8w]',
        '10.0',
     ), {
        'cloud_active_columns':
            'scratch[mp_snowncv]',
        'rain_active_columns':
            'scratch[mp_rainncv]',
        'reference_density':
            'scratch[mp_thompson_frozen_reference_density]',
    }),
    ('launch_ice_sedimentation', (
        'state.qi',
        'state.ni',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'scratch[mp_dz8w]',
        'scratch[mp_rainnc]',
        'scratch[mp_rainncv]',
        'scratch[mp_snownc]',
        'scratch[mp_snowncv]',
        '10.0',
     ), {
        'reference_density':
            'scratch[mp_thompson_frozen_reference_density]',
    }),
    ('launch_snow_sedimentation', (
        'state.qs',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'scratch[mp_dz8w]',
        'scratch[mp_rainnc]',
        'scratch[mp_rainncv]',
        'scratch[mp_snownc]',
        'scratch[mp_snowncv]',
        '10.0',
     ), {
        'accumulate_surface':
            'True',
        'melt_rain_nr':
            'state.nr',
        'melt_rain_qr':
            'state.qr',
        'reference_density':
            'scratch[mp_thompson_frozen_reference_density]',
        'reference_temperature':
            'scratch[mp_thompson_frozen_reference_temperature]',
        'snow_melt_marker':
            'scratch[mp_thompson_snow_melt_marker]',
        'velocity_boost':
            'scratch[mp_thompson_snow_velocity_boost]',
    }),
    ('launch_graupel_sedimentation', (
        'state.qg',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'scratch[mp_dz8w]',
        'scratch[mp_rainnc]',
        'scratch[mp_rainncv]',
        'scratch[mp_graupelnc]',
        'scratch[mp_graupelncv]',
        '10.0',
     ), {
        'accumulate_surface':
            'True',
        'active_columns':
            'scratch[mp_sr]',
        'graupel_number_shadow':
            'scratch[mp_thompson_graupel_number_shadow]',
        'reference_density':
            'scratch[mp_thompson_frozen_reference_density]',
    }),
    ('launch_rain_sedimentation', (
        'state.qr',
        'state.nr',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'scratch[mp_dz8w]',
        'scratch[mp_rainnc]',
        'scratch[mp_rainncv]',
        '10.0',
     ), {
        'accumulate_surface':
            'True',
        'reference_density':
            'scratch[mp_thompson_rain_reference_density]',
    }),
    ('launch_final_phase_cleanup', (
        'state.qc',
        'state.qi',
        'state.ni',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
     ), {}),
    ('launch_classic_graupel_number_finalize', (
        'state.qg',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'scratch[mp_thompson_graupel_number_shadow]',
     ), {}),
    ('launch_effective_radius', (
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'state.qc',
        'state.qi',
        'state.ni',
        'state.qs',
        'state.effc',
        'state.effi',
        'state.effs',
     ), {}),
    ('moist_physics_finish', (
        '<_HostAdapterState>',
        '<SimpleNamespace>',
        'scratch[mp_th]',
        '10.0',
     ), {}),
)

ADAPTER_CALLS_WITH_REFL = (
    ('save_pre_mp_theta', (
        '<_HostAdapterState>',
     ), {}),
    ('launch_classic_graupel_number_init', (
        'state.qg',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'scratch[mp_thompson_graupel_number_shadow]',
     ), {}),
    ('launch_frozen_vapor_network_from_owner', (
        'state.qi',
        'state.ni',
        'state.qs',
        'state.qg',
        'state.qr',
        'state.nr',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        '<classic-table-owner>',
        '10.0',
     ), {
        'graupel_number_shadow':
            'scratch[mp_thompson_graupel_number_shadow]',
        'qc':
            'state.qc',
        'snow_velocity_boost':
            'scratch[mp_thompson_snow_velocity_boost]',
    }),
    ('launch_warm_frozen_source_network_from_owner', (
        'state.qc',
        'state.qr',
        'state.nr',
        'state.qs',
        'state.qg',
        'scratch[mp_thompson_graupel_number_shadow]',
        'scratch[mp_thompson_graupel_melt_marker]',
        'scratch[mp_thompson_snow_melt_marker]',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        '<classic-table-owner>',
        '10.0',
     ), {}),
    ('launch_hydrometeor_column_mask', (
        'state.qr',
        'scratch[mp_rainncv]',
     ), {}),
    ('launch_hydrometeor_column_mask', (
        'state.qc',
        'scratch[mp_snowncv]',
     ), {}),
    ('launch_graupel_fallout_column_mask', (
        'scratch[mp_thompson_frozen_reference_temperature]',
        'state.qg',
        'scratch[mp_sr]',
     ), {}),
    ('launch_cloud_saturation_adjust', (
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'state.qc',
     ), {
        'reference_density':
            'scratch[mp_thompson_frozen_reference_density]',
        'reference_temperature':
            'scratch[mp_thompson_frozen_reference_temperature]',
    }),
    ('launch_rain_evaporation', (
        'state.qr',
        'state.nr',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        '10.0',
     ), {
        'graupel_melt_marker':
            'scratch[mp_thompson_graupel_melt_marker]',
        'reference_density':
            'scratch[mp_thompson_rain_reference_density]',
    }),
    ('launch_cloud_sedimentation', (
        'state.qc',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'state.w[view:(3, 1, 1)]',
        'scratch[mp_dz8w]',
        '10.0',
     ), {
        'cloud_active_columns':
            'scratch[mp_snowncv]',
        'rain_active_columns':
            'scratch[mp_rainncv]',
        'reference_density':
            'scratch[mp_thompson_frozen_reference_density]',
    }),
    ('launch_ice_sedimentation', (
        'state.qi',
        'state.ni',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'scratch[mp_dz8w]',
        'scratch[mp_rainnc]',
        'scratch[mp_rainncv]',
        'scratch[mp_snownc]',
        'scratch[mp_snowncv]',
        '10.0',
     ), {
        'reference_density':
            'scratch[mp_thompson_frozen_reference_density]',
    }),
    ('launch_snow_sedimentation', (
        'state.qs',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'scratch[mp_dz8w]',
        'scratch[mp_rainnc]',
        'scratch[mp_rainncv]',
        'scratch[mp_snownc]',
        'scratch[mp_snowncv]',
        '10.0',
     ), {
        'accumulate_surface':
            'True',
        'melt_rain_nr':
            'state.nr',
        'melt_rain_qr':
            'state.qr',
        'reference_density':
            'scratch[mp_thompson_frozen_reference_density]',
        'reference_temperature':
            'scratch[mp_thompson_frozen_reference_temperature]',
        'snow_melt_marker':
            'scratch[mp_thompson_snow_melt_marker]',
        'velocity_boost':
            'scratch[mp_thompson_snow_velocity_boost]',
    }),
    ('launch_graupel_sedimentation', (
        'state.qg',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'scratch[mp_dz8w]',
        'scratch[mp_rainnc]',
        'scratch[mp_rainncv]',
        'scratch[mp_graupelnc]',
        'scratch[mp_graupelncv]',
        '10.0',
     ), {
        'accumulate_surface':
            'True',
        'active_columns':
            'scratch[mp_sr]',
        'graupel_number_shadow':
            'scratch[mp_thompson_graupel_number_shadow]',
        'reference_density':
            'scratch[mp_thompson_frozen_reference_density]',
    }),
    ('launch_rain_sedimentation', (
        'state.qr',
        'state.nr',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'scratch[mp_dz8w]',
        'scratch[mp_rainnc]',
        'scratch[mp_rainncv]',
        '10.0',
     ), {
        'accumulate_surface':
            'True',
        'reference_density':
            'scratch[mp_thompson_rain_reference_density]',
    }),
    ('launch_final_phase_cleanup', (
        'state.qc',
        'state.qi',
        'state.ni',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
     ), {}),
    ('launch_classic_graupel_number_finalize', (
        'state.qg',
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'scratch[mp_thompson_graupel_number_shadow]',
     ), {}),
    ('reflectivity', (
        '<_HostAdapterState>',
        '<SimpleNamespace>',
        'scratch[mp_thompson_temperature]',
        'state.p',
     ), {
        'thompson_graupel_number':
            'scratch[mp_thompson_graupel_number_shadow]',
    }),
    ('launch_effective_radius', (
        'scratch[mp_thompson_temperature]',
        'state.p',
        'state.qv',
        'state.qc',
        'state.qi',
        'state.ni',
        'state.qs',
        'state.effc',
        'state.effi',
        'state.effs',
     ), {}),
    ('moist_physics_finish', (
        '<_HostAdapterState>',
        '<SimpleNamespace>',
        'scratch[mp_th]',
        '10.0',
     ), {}),
)

ACOUSTIC_N_MASS = {
    "mp1": 3, "mp6": 6, "mp8": 6, "mp10": 6, "mp18": 7,
}

# -- F2 --------------------------------------------------------------------

ORACLE_REBUILD_EXCEPTION_FILES = frozenset({
    "warm-column.csv", "ice-column.csv",
    "mixed-column.csv", "mixed-surface.csv",
})
#: Levels (1-based) at which the three original fixtures' ``p_pa`` differs
#: by one float32 ulp from the other 43 committed fixtures.
ORACLE_PPA_DIVERGENT_LEVELS = (2, 3, 5, 6, 7, 8, 10, 14, 16, 17, 18, 19, 22)


# ==========================================================================
# R1 -- source identity
# ==========================================================================

@pytest.fixture(scope="module")
def r1():
    return freeze.receipt_r1_sources()


def test_thompson_cu_is_byte_frozen(r1):
    """The single most important assertion in the port."""
    module = r1["modules"]["thompson"]
    assert module["file_sha256"] == THOMPSON_CU_SHA256, (
        "gpuwm/core/kernels/thompson.cu was edited.  It is byte-frozen: "
        "the entire mp=8 numerics guarantee is that its compiled source "
        "string never moves.  mp=28 kernels belong in new .cu files.")


def test_thompson_compiled_source_string_is_frozen(r1):
    """Identical source string => identical PTX => identical FP results.

    Stronger than the file hash: this is what cupy actually compiles, so a
    change to ``_preamble()``, ``CUDA_DEFINES``, ``common.cuh`` or the
    loader itself fails here too.
    """
    module = r1["modules"]["thompson"]
    assert module["capture_method"] == "loader-capture", (
        "the compile string was reconstructed instead of captured from the "
        "real loader; the inertness claim is then unproven")
    assert module["compiled_source_len"] == THOMPSON_COMPILED_SOURCE_LEN
    assert (module["compiled_source_sha256"]
            == THOMPSON_COMPILED_SOURCE_SHA256)


def test_preamble_and_common_header_are_frozen(r1):
    assert r1["preamble_sha256"] == PREAMBLE_SHA256
    assert r1["preamble_len"] == PREAMBLE_LEN
    assert r1["common_cuh_sha256"] == COMMON_CUH_SHA256
    assert r1["cuda_defines"] == CUDA_DEFINES_PIN


def test_every_frozen_kernel_module_is_unchanged(r1):
    """All 65 pre-existing translation units, file AND compile string.

    New mp=28 ``.cu`` files are allowed and ignored; a MISSING pinned name
    is a failure.
    """
    modules = r1["modules"]
    missing = sorted(set(FROZEN_MODULE_DIGESTS) - set(modules))
    assert not missing, f"frozen kernel modules disappeared: {missing}"
    drift = {}
    for name, (file_sha, compiled_sha) in sorted(
            FROZEN_MODULE_DIGESTS.items()):
        got = modules[name]
        if (got["file_sha256"], got["compiled_source_sha256"]) != (
                file_sha, compiled_sha):
            drift[name] = {
                "expected": (file_sha, compiled_sha),
                "actual": (got["file_sha256"],
                           got["compiled_source_sha256"]),
            }
    assert not drift, f"kernel source drift: {drift}"


def test_loader_hook_is_inert_for_every_frozen_module(r1):
    """``load_module`` still assembles ``_preamble() + <name>.cu`` exactly.

    WP-02 adds an ``_EXTRA_HEADERS`` allow-list to the shared loader.  This
    is the assertion that holds it to "every module not named in the dict
    assembles a byte-identical source string".
    """
    not_inert = sorted(
        name for name in FROZEN_MODULE_DIGESTS
        if not r1["modules"][name]["loader_matches_preamble_plus_file"])
    assert not not_inert, (
        "the kernel loader no longer assembles _preamble() + <name>.cu for "
        f"these pre-existing modules: {not_inert}")


def test_thompson_py_has_no_new_launcher(r1):
    assert r1["thompson_py_sha256"] == THOMPSON_PY_SHA256
    assert r1["thompson_py_all"] == THOMPSON_PY_ALL
    assert (r1["thompson_py_launch_symbols"]
            == tuple(sorted(THOMPSON_PY_ALL))), (
        "gpuwm/core/thompson.py grew or lost a launch_* symbol; mp=28 "
        "launchers belong in gpuwm/core/thompson_aerosol_*.py")


def test_constant_droplet_number_inventory_is_unchanged(r1):
    """The 13 + 6 + 2 + 3 literal sites mp=28 must replace, not edit.

    If this moves, someone has been editing the frozen kernel in place --
    which the source digests would also catch, but this failure names the
    physics.
    """
    assert r1["thompson_cu_literal_sites"] == THOMPSON_CU_LITERAL_SITES


# ==========================================================================
# R2 -- classic table contract
# ==========================================================================

@pytest.fixture(scope="module")
def r2():
    return freeze.receipt_r2_tables()


def test_classic_table_assets_are_unchanged(r2):
    got = tuple((a["filename"], a["bytes"], a["sha256"])
                for a in r2["classic_table_assets"])
    assert got == CLASSIC_TABLE_ASSETS_PIN
    assert r2["table_set_id"] == TABLE_SET_ID_PIN
    assert r2["mp_physics"] == 8
    assert tuple(r2["number_species"]) == ("ni", "nr")
    assert tuple(r2["mass_species"]) == ("qv", "qc", "qr", "qi", "qs", "qg")
    assert r2["wrf_reference_version"] == WRF_REFERENCE_VERSION_PIN
    assert r2["wrf_reference_commit"] == WRF_REFERENCE_COMMIT_PIN


def test_aerosol_blob_never_enters_the_classic_contract(r2):
    """CCN_ACTIVATE.BIN is third-party parcel-model output only mp=28 reads.

    It ships with gpuwm as of 2026-08-01 and is listed in
    ``tables/MANIFEST.sha256``, which is exactly why this assertion still
    matters: that manifest is a ``sha256sum -c`` file nothing reads at run
    time, while ``CLASSIC_TABLE_ASSETS`` IS the four-asset inventory every
    mp=8 launch validates and every mp=8 restart identity binds.  Adding the
    blob there would change what a validated mp=8 table set means and make
    an mp=8 launch fail closed on a file it never reads.  WP-01 gives it its
    OWN contract and its own set id instead.
    """
    assert r2["aerosol_blob_in_classic_assets"] is False


# ==========================================================================
# R3 -- transported species
# ==========================================================================

def test_morrison_droplet_number_exclusion_survives():
    """``nc`` must not start being advected as a side effect of mp=28.

    mp_physics=10 already allocates ``state.nc`` and deliberately does not
    transport it.  A presence-based ``nc`` in
    ``TRANSPORTED_NUMBER_SPECIES`` would silently start advecting Morrison's
    diagnostic droplet number through all 8 generic dycore call sites.  The
    mp=10 probe below CARRIES ``nc``, so its absence from the result is
    proved rather than merely unexercised.
    """
    r3 = freeze.receipt_r3_species()
    assert tuple(r3["extra_moist_species_mp8"]) == EXTRA_MOIST_SPECIES_MP8
    assert tuple(r3["extra_moist_species_mp10"]) == EXTRA_MOIST_SPECIES_MP10
    assert "nc" not in r3["extra_moist_species_mp10"]
    assert "nwfa" not in r3["extra_moist_species_mp10"]
    assert "nifa" not in r3["extra_moist_species_mp10"]
    assert (tuple(r3["transported_number_species"])
            == TRANSPORTED_NUMBER_SPECIES_PIN)


# ==========================================================================
# R4 -- preflight allocation surface
# ==========================================================================

@pytest.fixture(scope="module")
def r4():
    return freeze.receipt_r4_preflight()


def test_mp8_state_allocation_list_is_unchanged(r4):
    """Every array DomainState allocates for mp=8, with its exact shape."""
    assert r4["probe_config"]["mp_physics"] == 8
    got = r4["state_array_shapes"]
    assert set(got) == set(STATE_ARRAY_SHAPES_MP8), {
        "added": sorted(set(got) - set(STATE_ARRAY_SHAPES_MP8)),
        "removed": sorted(set(STATE_ARRAY_SHAPES_MP8) - set(got)),
    }
    assert got == STATE_ARRAY_SHAPES_MP8
    assert r4["state_array_shapes_digest"] == STATE_ARRAY_SHAPES_DIGEST


def test_mp8_scratch_arena_layout_is_unchanged(r4):
    """The scratch registry IS the aliasing contract.

    ``_apply_thompson`` deliberately lifetime-aliases buffers (the graupel
    entry marker with the held-temperature buffer, RAINNCV/SNOWNCV/SR with
    column masks).  A new or resized slot moves the arena and can turn one
    of those aliases into a live conflict without any test failing on
    values.
    """
    got = r4["scratch_slot_registry"]
    assert set(got) == set(SCRATCH_SLOT_REGISTRY_MP8), {
        "added": sorted(set(got) - set(SCRATCH_SLOT_REGISTRY_MP8)),
        "removed": sorted(set(SCRATCH_SLOT_REGISTRY_MP8) - set(got)),
    }
    assert got == SCRATCH_SLOT_REGISTRY_MP8
    assert r4["scratch_slot_registry_digest"] == SCRATCH_SLOT_REGISTRY_DIGEST


def test_mp8_nest_field_kinds_are_unchanged(r4):
    assert tuple(r4["nest_field_kinds"]) == NEST_FIELD_KINDS_MP8
    assert "nc" not in r4["nest_field_kinds"]


# ==========================================================================
# R5 -- nest transition edge codes
# ==========================================================================

def test_edge_field_codes_for_the_twenty_pre_existing_names():
    """28 must be APPENDED to PORTED_MP_PHYSICS, never inserted.

    ``_EDGE_FIELD_CODES`` is ``enumerate`` over the de-duplicated union of
    every ported scheme's fields in PORTED_MP_PHYSICS order.  Inserting 28
    anywhere but the end renumbers the table, and the ratified mp8 -> mp18
    nest edge then selects different fields with no error anywhere.
    """
    r5 = freeze.receipt_r5_edge_codes()
    codes = r5["edge_field_codes"]
    for name, code in EDGE_FIELD_CODES_PIN.items():
        assert codes.get(name) == code, (
            f"nest edge field {name!r} moved from code {code} to "
            f"{codes.get(name)}")
    assert tuple(r5["ported_mp_physics"])[:5] == PORTED_MP_PHYSICS_PIN
    assert tuple(r5["all_edge_fields"])[:20] == tuple(EDGE_FIELD_CODES_PIN)


# ==========================================================================
# R6 -- acoustic selection and the adapter call graph
# ==========================================================================

def test_acoustic_moist_cq_selects_six_masses_for_mp8():
    """``calc_cq`` mass loading must not learn about aerosol numbers.

    WRF's ``calc_cq`` sums the ``moist`` Registry package; number moments
    live in the separate ``scalar`` package and are not mass loading.  mp=28
    adds nc/nwfa/nifa as scalars, so n_mass for mp=8 (and for mp=28) stays
    6.
    """
    n_mass = freeze.receipt_r6_call_graph()["acoustic_n_mass"]
    assert n_mass["mp8"] == 6
    assert n_mass == ACOUSTIC_N_MASS


def _as_tuple(calls):
    return tuple(
        (c["launcher"], tuple(c["args"]), dict(c["kwargs"])) for c in calls)


def test_apply_thompson_issues_the_identical_launcher_sequence():
    """Call-recording double over the real adapter.

    Every launcher, ``save_pre_mp_theta``, ``moist_physics_finish`` and the
    reflectivity entry point are replaced by spies, and the classic table
    owner by a sentinel, so no CUDA runs and no 380 MB table is read.  What
    is pinned is the ORDER of the calls and the IDENTITY of every argument
    -- which state field or which named scratch slot -- because that
    ordering and that aliasing are the mp=8 trajectory.

    The mp=28 adapter is a SEPARATE module
    (``gpuwm/core/thompson_aerosol.py``); nothing in this sequence may
    change to accommodate it.
    """
    recorded = _as_tuple(freeze.record_adapter_calls(refl_10cm_due=False))
    assert recorded == ADAPTER_CALLS_NO_REFL


def test_apply_thompson_reflectivity_call_graph_is_unchanged():
    recorded = _as_tuple(freeze.record_adapter_calls(refl_10cm_due=True))
    assert recorded == ADAPTER_CALLS_WITH_REFL


def test_adapter_still_feeds_cloud_sedimentation_the_lower_w_slice():
    """WRF copies ``w(i,k,j)`` into ``w1d(k)`` with no averaging.

    mp=28's ``activ_ncloud`` needs the same field and is far more sensitive
    to it, so this is the argument the aerosol adapter must match.  Pinned
    here as its own named claim because a silent switch to a mass-level
    average would still pass the whole-sequence comparison's shape checks.
    """
    calls = {c["launcher"]: c
             for c in freeze.record_adapter_calls(refl_10cm_due=False)}
    assert calls["launch_cloud_sedimentation"]["args"][4] == (
        "state.w[view:(3, 1, 1)]")


# ==========================================================================
# F1 -- the committed mp=8 oracle fixtures
# ==========================================================================

def test_committed_mp8_oracle_fixtures_are_frozen():
    """92 CSVs, byte-for-byte.

    WP-03 adds ``gpuwm/data/thompson/oracle-aero/``; it may not touch this
    directory.  Regenerating any file here moves a model-validated
    baseline.
    """
    f1 = freeze.receipt_f1_oracle_fixtures()
    assert f1["count"] == ORACLE_FIXTURE_COUNT
    assert f1["aggregate_sha256"] == ORACLE_FIXTURE_AGGREGATE_SHA256


# ==========================================================================
# F2 -- the documented four-file clean-rebuild exception
# ==========================================================================

def test_rebuild_exception_list_is_exactly_four_named_files():
    """No file may be added to the exception list to make a gate pass.

    The exception is a recorded historical fact about four fixtures, not a
    tolerance.  Everything else must rebuild byte-for-byte.
    """
    assert set(freeze.ORACLE_REBUILD_EXCEPTIONS) == (
        ORACLE_REBUILD_EXCEPTION_FILES)


def test_the_committed_fixtures_now_share_one_pressure_profile():
    """THE WITNESS, INVERTED, BECAUSE THE DEFECT IT WITNESSED WAS FIXED.

    This test used to be called
    ``test_the_three_original_fixtures_carry_a_foreign_pressure_profile``
    and it asserted the DEFECT: that the 46 committed column fixtures split
    into two ``p_pa`` profiles, 43 against {warm, mixed, ice}.  That was a
    true and carefully measured statement about a stale oracle, and the
    receipt beside it explained the split as a build-provenance difference
    -- "a different libm expf, i.e. a different machine or glibc" -- and
    recorded it rather than repairing it.

    The Thompson oracle lane found the actual mechanism and repaired it.
    It is not a different machine: from GCC 12 on, ``-O2`` implies
    ``-ftree-vectorize``, a vectorised ``exp``/``pow`` loop links glibc's
    libmvec SIMD entry points instead of the scalar routines, libmvec is
    not bit-identical to scalar libm, and whether any given loop vectorises
    is a cost-model decision that depends on how much UNRELATED source
    surrounds it.  ``run_column.F90`` was 227 lines when warm/mixed/ice
    were generated and roughly a thousand when the other 43 were, so the
    same ``-O2`` produced two different oracles.  ``build.sh`` now pins
    ``-fno-tree-vectorize``, which removes libmvec from the link entirely
    and makes the set invariant across gfortran 12/13/14/15 at -O1/-O2/-O3,
    and all five affected fixtures were regenerated
    (fix/thompson-oracle-devectorized, 576b755b and 8290189d).

    So the assertion INVERTS rather than being deleted: ONE profile, 46
    files, no minority group.  That is strictly stronger than what it
    asserted before, and a regression that reintroduces a second profile
    fails here exactly as the old form would have.

    The old docstring's remaining content is preserved because it is still
    the correction of record for the port spec's blocking unknown #2:

    HERMETIC witness for the CAUSE of the (former) exception.

    ``p_pa = p0 * exp(-z(k)/8000.0)`` (run_column.F90:219) is a pure
    function of the harness's own z grid and does not depend on the
    scenario, so all 46 committed column fixtures must print the same 24
    values.  They do not: there are exactly two profiles, and the minority
    one belongs to exactly {warm, mixed, ice} -- which
    ``gpuwm/data/thompson/PROVENANCE.md`` records as the first three columns
    ever generated.  Those three came from a different build of the harness
    (a different libm ``expf``: they differ by one float32 ulp at 13 of 24
    levels), and every other difference in those files -- including the
    ``after`` rows and mixed-surface's rainnc -- follows from it through the
    nonlinear scheme.

    This CORRECTS the port spec's blocking unknown #2, which described the
    drift as a float32-vs-float64 vapour seed with "p_pa, pii and theta
    byte-identical".  p_pa and pii are NOT byte-identical, and the k=1
    vapour coincidence the spec generalised from does not hold at k=2..24.

    Reading only committed repository bytes, this test needs no gfortran,
    no WRF tree and no rebuild -- so the exception stays verified on every
    run instead of resting on a note.
    """
    witness = freeze.fixture_provenance_witness()
    assert witness["distinct_p_pa_profiles"] == 1, (
        "the committed column fixtures split into "
        f"{witness['distinct_p_pa_profiles']} pressure profiles again; "
        "p_pa = p0*exp(-z/8000) does not depend on the scenario, so more "
        "than one profile means the set is a stratigraphy of builds rather "
        "than one oracle run.  The minority group is "
        f"{sorted(witness['minority_group'])}")
    assert witness["group_sizes"] == [46]
    assert witness["minority_group"] == []
    assert witness["p_pa_detail"] == {}, (
        "the witness is reporting a p_pa divergence again; see "
        "tests/test_thompson_oracle_provenance.py for the same property "
        "asserted from the fixture bytes directly")
    # The port spec's blocking unknown #2 claimed the drift was a
    # float32-vs-float64 vapour seed, generalising from a k=1 coincidence.
    # With the oracle corrected, even that coincidence is gone: the
    # committed k=1 qv is the float32-throughout evaluation, not the
    # float64-then-REAL(4) one.  Both halves of the spec's claim are now
    # measured false, and both are asserted so rather than described.
    assert witness["seed_k1_note"]["committed_matches_float64_path"] is False
    assert witness["seed_k1_note"]["generalises_to_other_levels"] is False


@pytest.mark.skipif(
    not os.environ.get("GPUWM_MP8_ORACLE_REBUILD_DIR"),
    reason="set GPUWM_MP8_ORACLE_REBUILD_DIR to a completed "
           "tools/thompson_wrf461_oracle/build.sh output directory")
def test_clean_oracle_rebuild_matches_except_the_four_documented_files():
    """Opt-in empirical gate: 4/4 .dat SHAs and 88/92 CSVs, exactly.

    Opt-in because it needs gfortran, the pristine WRF tree and ~380 MB of
    regenerated coefficient tables.  Measured on gfortran 13.3.0 / glibc
    2.39 / Ubuntu 24.04 during WP-00.
    """
    build_dir = Path(os.environ["GPUWM_MP8_ORACLE_REBUILD_DIR"])
    result = freeze.compare_rebuilt_oracle(build_dir)
    assert result["dat_all_match"] is True, result["dat_assets"]
    assert result["csv_missing"] == []
    assert result["csv_total"] == ORACLE_FIXTURE_COUNT
    assert set(result["csv_differing"]) == ORACLE_REBUILD_EXCEPTION_FILES
    assert result["csv_identical"] == (
        ORACLE_FIXTURE_COUNT - len(ORACLE_REBUILD_EXCEPTION_FILES))
    for name, diff in result["csv_differences"].items():
        pinned = freeze.ORACLE_REBUILD_EXCEPTIONS[name]
        assert diff["max_relative_overall"] <= (
            pinned["max_relative_overall"] * 1.001), (
            f"{name} drifted beyond the recorded deviation")
