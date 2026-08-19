# Vendored donor snapshot: rw-wps + its grib-core and netcrust

Unmodified snapshot of Drew's Rust mapping/composition engine, vendored for
the 2.5.0 mapped-engine port (Python boundary ruling, 2026-08-16).

- Donor repository: a private working checkout of
  `rusty-weather-consolidated` (not a public URL)
- Donor branch: `integration/rw-consolidated-20260817`
- Donor commit: `edf044cfdd31a8d2e7540a862c435c3e46f8e3e6`
- Snapshot date: 2026-08-17
- Contents: `crates/rw-wps` (mapping/composition/namelist engine, schema
  reconciled byte-for-byte to gpuwm's live authority files, 65/65 green at
  donor tip), `vendor/grib-core` (donor-side hardening: spatial-differencing
  missing values, minute statistical intervals, WMO-wrapped components),
  `vendor/netcrust` (+ its vendored `hdf5-reader`), which rw-wps's NetCDF
  path uses.
- Excluded: build outputs (`target/`) only.  Every tracked donor file under
  the three roots is present and byte-identical; verified with sha256 at
  vendor time (80/80 OK).

The donor repository stays read-only.  Fixes converge here and, for
grib-core, into the ONE superset crate at `tools/grib1_bridge/vendor/grib-core`
(see `docs/dev/decode-vendor-design.md`).  This snapshot's `vendor/grib-core`
is the convergence INPUT and is deleted when the superset lands; do not grow
it.

## Skeleton additions (gpuwm's, not donor modifications)

The donor files above stay byte-identical.  gpuwm added, in the
skeleton commit: the workspace `Cargo.toml` + `Cargo.lock`, the
`crates/mapped-engine` seam stub, and the two files under
`crates/rw-glm/tests/fixtures/` — the real GOES-19 GLM LCFA NetCDF4
fixture (240 KB, byte-identical to the donor's copy at the same
donor-relative path) plus its README, because the donor lib test
`real_netcdf4_fixture_is_inventoried_through_hdf5_fallback` reaches it
via `../rw-glm/tests/fixtures/` and the suite must stay 65/65 without
editing donor code.  Verified in-tree after vendoring: 60 lib + 5
integration tests green.

## Port-lane 2 additions and the two donor edits it made

gpuwm added: `crates/mapped-engine/**` (the decode engine and its
goldens), `vendor/crates-io/**` + `.cargo/config.toml` (the offline
source replacement), and `crates/rw-glm/tests/fixtures/.gitignore`.

## Release-bundle addition

gpuwm added `crates/mapped-engine/build.rs`, the same source-revision
stamp script every other gpuwm-authored bridge carries: the engine
joined the release bundle roster
(`gpuwm.bridge_assets.BUNDLED_ARTIFACTS`), and the cut refuses to pin a
bundle whose binaries do not name the commit being released.  It is on
gpuwm's crate, not the donor's -- the snapshot under `crates/rw-wps`
and `vendor/` stays byte-identical and ships nothing, so it is stamped
by nothing.

Two donor-side files changed, both recorded here because the "unmodified
snapshot" claim above is otherwise no longer true:

1. `crates/rw-wps/Cargo.toml` — the `grib-core` path dep now points at
   `../../../grib1_bridge/vendor/grib-core` (gpuwm's copy, the declared
   BASE of lane 1's superset) instead of `../../vendor/grib-core` (the
   donor snapshot).  Cargo refuses two different packages named
   `grib-core v0.1.0` in one lockfile, and the mapped engine has to
   decode through the SAME crate the shipped `grib2_dump` uses or a
   crate difference lands inside the parity verdict.  This is the
   integration path-flip of design doc §5, done early; deleting
   `vendor/grib-core` stays lane 1 / coordinator work.
2. `crates/rw-wps/src/mapping.rs` — two lines, translating gpuwm's
   `second_level_type == 255` sentinel into the donor's `Option::None`.
   Widening it to `Some(255)` instead would announce a second fixed
   surface of type "missing" and break every selector that matches on
   the absence of one.

`cargo test -p rw-wps` is 65/65 green (60 lib + 5 integration) against
gpuwm's grib-core with those two edits, offline and `--locked`.

**The GLM fixture was not actually in the skeleton commit.**  The
repo-wide `*.gitignore` rule `*.nc` (root, line 13) swallowed it, so the
donor suite was 59/60 on any fresh clone while the skeleton lane measured
65/65 against an untracked working-tree file.  Fixed by a nested
`.gitignore` in that fixtures directory carrying `!*.nc`, and the fixture
is now tracked.

Offline posture is now the shipped one: `cargo build --locked --offline`
and `cargo test --locked --offline` both pass with no network, resolving
every crates.io dependency from `vendor/crates-io` (91 packages, ~57 MB).

`serde_json` is built with the **`float_roundtrip`** feature, and that is
a correctness requirement rather than a preference: its default float
parser returned a mapping's declared unit scale one ULP off, which moved
every cell of every scaled field.  See the crate manifest for the
measurement.

## Integration: the donor grib-core copy is gone, and Cargo.toml moved

`vendor/grib-core/**` was DELETED at integration.  It was the donor half
of the convergence, and its hardening now lives in the one superset
crate at `tools/grib1_bridge/vendor/grib-core`, which BOTH members of
this workspace and all five shipped bridge binaries build against.
`tests/test_grib_core_convergence.py` holds the line that exactly one
`grib-core` manifest is tracked in the repository.

Its sha256 rows are LEFT in the file list below on purpose: they are the
donor's identity at `edf044c` and the record of what was converged FROM.
They no longer describe files on disk, and the `crates/rw-wps/*` rows
that the path flip and the accessor change touched are stale for the
same reason the lane-2 section already gives.  Re-pinning the whole list
belongs with the next donor resync, when there is a new snapshot to pin
it to; re-pinning it now would erase the only record of the input.

`crates/rw-wps/src/mapping.rs` also changed once more at integration:
the `second_level_type == 255` translation lane 2 spelled inline now
goes through `ProductDefinition::second_fixed_surface()`, the accessor
lane 1 added to the superset for exactly this call site, so the sentinel
test lives inside grib-core and cannot drift from the crate's own
reading of Code Table 4.5.

## File list (sha256, path relative to this directory)

2a2a1c232581f4814f45faef3b779dbcc2087e0b2f304f84f954179f5ce299e9 *crates/rw-wps/Cargo.toml
f8209a0637501ae1cf048ba53d1fb77a2f0542a8383de99265b5ad8d5d386256 *crates/rw-wps/src/lib.rs
4048cda22d99f90c494d985c348985c3221a12c019670a09c8ce059d3e9985f5 *crates/rw-wps/src/main.rs
987805fbe2f53cae1fdcb0cec1ecb83ea455361c2b8213ce7f7266bfeeff1302 *crates/rw-wps/src/mapping.rs
9f1014069ab52a8210efb736c98beec3ccb38f8e79827177d202ff255844264e *crates/rw-wps/src/namelist.rs
d2c9ee08e45478a64e4d2bba689e9bad1d2e97bde713477ee2a4de26e31d7ad3 *crates/rw-wps/tests/fixtures/rw-wps-era5-netcdf.mapping.json
5b0f41a7f4ddee1116ce8310dfd67827761413908d45402e1f55f32facc61d86 *crates/rw-wps/tests/fixtures/rw-wps-gfs-pressure-grib2.mapping.json
2f3bd3348236cb499f10e00a1625c4ca2ab5433392b0a6efec6f2152913058d4 *crates/rw-wps/tests/mapping_schema_parity.rs
881e30ee3b9dfd9b8f3efe4fbfc200772b84f6770084f0c74a9ad2dc49156ec6 *vendor/grib-core/Cargo.toml
6df14e9e90167aa9961694d24d0215acfdf4bfdaf48dd197f4f39a6a64508641 *vendor/grib-core/src/grib1/grid.rs
ba7ffc970dcf3afa5f1fbb13ac085625b0faa1de791a310c3a0f01c1e2f94711 *vendor/grib-core/src/grib1/mod.rs
723afc9d867133acaa4d1ff414a979eb06f904b10034b3a371c2f09f322203ea *vendor/grib-core/src/grib1/parser.rs
f7297c37bee034a8a49e3269c616778850b734917b880dce8abbc452a782be7f *vendor/grib-core/src/grib1/tables.rs
d2d57fda300fd2f6ecd78ff67663372a81cfd8c68efe515cf90f318e54aa1aae *vendor/grib-core/src/grib1/unpack.rs
bb3a85a57c5ad4f61d5d19fa9ddbcef82a089bb0a390f120ea6f703a6ea59e9c *vendor/grib-core/src/grib2/grid.rs
85b0e9a23f96d29e06aa11baa5bbfcdc775da36ffc25d06e48d91d23a28a7c8b *vendor/grib-core/src/grib2/mod.rs
1632bab43b0735785eea121d645dea777238a8105f2fa75c2295f1e71fe91ce8 *vendor/grib-core/src/grib2/parser.rs
5f27a197548f0e761995ff0235ff683aa58a479db41bad04df1f44fb22992f14 *vendor/grib-core/src/grib2/search.rs
9e396d8cc8b8b588530a74ae7f8fda9aec16066a0548912099d1403c6b93b83e *vendor/grib-core/src/grib2/tables.rs
c2db1d871409dffce2667408c69acf35de83c6de0f99f8608a85e2b3cee60207 *vendor/grib-core/src/grib2/unpack.rs
d9ab44de5f10c8193bcde3a058b07c30d4a1d4bb91fea91969e2100f8e3a6fe0 *vendor/grib-core/src/lib.rs
25dbc1b805eb6172f4879fd388f3eaec7b2deee8f99d3363cc5d039efc8d3d5f *vendor/netcrust/Cargo.toml
8e684fc8a89c34c7f7b8b5ffb386f6b5270b1116b6c258a450ba44d8c9025606 *vendor/netcrust/LICENSE-APACHE
200a8eb7ca809b2c1ef2476c7473159ef5924a23fc5f719739efccd463469cdc *vendor/netcrust/LICENSE-MIT
1ddcee32d81511aa9d27b71b2c51e92ba81ace7ac0181f7290cc297aba28cae7 *vendor/netcrust/README.md
60cc4d6afbb25af990f58d157e5582b61d51c50c2ac1599691eb630f24700302 *vendor/netcrust/src/bin/netcrust-inspect.rs
d419092e020110969e02b2486d09b9240e032b55b9393ddc03b2e6da7806d368 *vendor/netcrust/src/lib.rs
776fa944b9f96f15d459718d27688ac9bf1903a04ab70ef03971d5a7b61e2b9d *vendor/netcrust/tests/wrf_real.rs
7d4b5fa39f6b798e5e59679ad84c888579af88f77c87c6025b823b68248ef153 *vendor/netcrust/vendor/hdf5-reader/Cargo.toml
b925696b6b6ca10d944325fcc70a08dcb65ca81533b44822f13913c69010bc36 *vendor/netcrust/vendor/hdf5-reader/Cargo.toml.orig
187352faa3374b1b872f410f957d615c5dd85858822994645620a30784de0668 *vendor/netcrust/vendor/hdf5-reader/README.md
04d52bf313116a1a5c17302b1cc14f516b40b59d38379b767dfe2793d50186bb *vendor/netcrust/vendor/hdf5-reader/src/attribute_api.rs
c8a4e99dd7630f94109d4a3cec0f74da4bae3f1faf93f962f2fb2c95c1213563 *vendor/netcrust/vendor/hdf5-reader/src/btree_v1.rs
193212076be223908e6768a1c077159ec3853472bfec42aa1c0663aa2ca2f46a *vendor/netcrust/vendor/hdf5-reader/src/btree_v2.rs
fd8b115a2d9db4a86e7e0182005e93564b560dbb9e3621ccb7ecd91a8075b36f *vendor/netcrust/vendor/hdf5-reader/src/cache.rs
5cb26af29aa04f071770242a2aa3b9f00ec3e3f339f703243cfb1271771f4898 *vendor/netcrust/vendor/hdf5-reader/src/checksum.rs
4f44bbd34ede47fff4ef2f1a2018013083754509943a67993b44de1609ab5eb4 *vendor/netcrust/vendor/hdf5-reader/src/chunk_index.rs
6bfde8759754bb5622f50b7f97b00ecd7b5e351762423063fe54c9c2eb5116c9 *vendor/netcrust/vendor/hdf5-reader/src/dataset.rs
e2d85f7c9a04a33d638908d2ba5868ba9810439f1f71f36b308fc9a08bfb2e3f *vendor/netcrust/vendor/hdf5-reader/src/datatype_api.rs
8c586c937ef5fd2502bf1ad4bcd8b742c74ff35275d5b879b33a7cff4cd6511e *vendor/netcrust/vendor/hdf5-reader/src/error.rs
82a6be9c06e3e4559efbbc33c1fcaeb7272a55b5c8f4be31ac4bf99812f87f49 *vendor/netcrust/vendor/hdf5-reader/src/extensible_array.rs
113df9ff46b62fac0dfde305d2daa9d7b9948c31b55b90773335091a95a0b3d5 *vendor/netcrust/vendor/hdf5-reader/src/filters/deflate.rs
f012be489cd327da9855eccb6ea28b25e80ec3507ef0ffdd7e049778e7d64ea4 *vendor/netcrust/vendor/hdf5-reader/src/filters/fletcher32.rs
b30eb8425921962c67b7ff99375f48f2b6b70d28fafe2b278a3b919f35f06153 *vendor/netcrust/vendor/hdf5-reader/src/filters/lz4.rs
7a7e35d8a89a478b38c6986b0354de156ff3fb4a3cf8bbcfa4c731bef2e9b0d2 *vendor/netcrust/vendor/hdf5-reader/src/filters/mod.rs
19dc4c3446f039c3177f307c8d34771cfd847765fc4a618ebab96d68e7fd0858 *vendor/netcrust/vendor/hdf5-reader/src/filters/nbit.rs
2fd4ab9f9ea8494e0c87d90386fa24e09a9cbfa4920698ab4a738457fc36257f *vendor/netcrust/vendor/hdf5-reader/src/filters/scaleoffset.rs
34652831308fb6a1df54bd518465a4dd428a9d2d8a6291ba3044340f1327af60 *vendor/netcrust/vendor/hdf5-reader/src/filters/shuffle.rs
cc6b9950d1515691162e9b919716098482b1b8dcad310a9ebd57657583885319 *vendor/netcrust/vendor/hdf5-reader/src/fixed_array.rs
7247e34d8a2479e2fab74288011c78700f32f4e4df70e34c93cddd74b889f9bd *vendor/netcrust/vendor/hdf5-reader/src/fractal_heap.rs
775e7ee270323ed19cde93b25dc3c064e93d082b27d8b600b101d12609ca14f6 *vendor/netcrust/vendor/hdf5-reader/src/global_heap.rs
c071f7e8c10fd002aa00fc58529b86eff1a28301ce42c90ca896724717739b8d *vendor/netcrust/vendor/hdf5-reader/src/group.rs
2411e8bbd32fd70ed5e8dd4bb12263f08e0da41702bb996971b0738401313c58 *vendor/netcrust/vendor/hdf5-reader/src/io.rs
5f67f85fd54899dab78db521213b3dab13f96a98a68d6a545cbd1ae74acf9a7c *vendor/netcrust/vendor/hdf5-reader/src/lib.rs
0d6a1f1f521309bbcfb68d9b9a7c145e4bcf1802565fd08343644d4aecfde7de *vendor/netcrust/vendor/hdf5-reader/src/local_heap.rs
7656895e31b9bff0a93201585704d56b2207ae47b5096aaa5df10755f741cc21 *vendor/netcrust/vendor/hdf5-reader/src/messages/attribute.rs
4a34aa05296bebde95d916c03e750b7c4b3877a54f1b01c432ea5a67c5fec957 *vendor/netcrust/vendor/hdf5-reader/src/messages/attribute_info.rs
d6418a92730ad34c5a8359907ee9c370afa9ff08e2bcd4c36ab7aadce1a9198c *vendor/netcrust/vendor/hdf5-reader/src/messages/btree_k.rs
75f0c567d27650b1a3c646364499e29dad1daad52b1fca3f82f190ad786fe3e7 *vendor/netcrust/vendor/hdf5-reader/src/messages/continuation.rs
02a24036007016ce188242c9b1554a6484e2c28797efd56111a276404be841e3 *vendor/netcrust/vendor/hdf5-reader/src/messages/dataspace.rs
d22ecabb1829c870dd756d7a4cb119b8cd5d19df8f49c442c1f88f73c76368fe *vendor/netcrust/vendor/hdf5-reader/src/messages/datatype.rs
fed78dfd6c380d4e7ea1b385a64ce35a027ea16d4451a6f5e5d55bfc1facbefd *vendor/netcrust/vendor/hdf5-reader/src/messages/external_files.rs
4a5685c152835179a4f178092532155aab7e38d591d57ca51283c2a916f732a9 *vendor/netcrust/vendor/hdf5-reader/src/messages/fill_value.rs
33c1f53263152e520e5c47482b3011c843d21f9981dc91e3a81f390f36d38ea9 *vendor/netcrust/vendor/hdf5-reader/src/messages/filter_pipeline.rs
a82ad9e4d5dc3b059f8c7344f0d3286d5622b319a7a3467122b7f20c2120b44c *vendor/netcrust/vendor/hdf5-reader/src/messages/group_info.rs
e955622e0eab2572514746d536368804ac3a53466924dce9ebc4d00cb7dbef9d *vendor/netcrust/vendor/hdf5-reader/src/messages/layout.rs
ccba856313e41ffe36c0f2e6011a48e23ee159726efe7a080b7809147a4b88f9 *vendor/netcrust/vendor/hdf5-reader/src/messages/link.rs
39121aac27e9f13d4b1995c1519697df4e1e2067a0d386077153dc5aef748ae5 *vendor/netcrust/vendor/hdf5-reader/src/messages/link_info.rs
651d27b8e1e829b2c10a2526dca2e0f7778e0325dc096b86e03b703e9b838b13 *vendor/netcrust/vendor/hdf5-reader/src/messages/mod.rs
3bf87c8b740230c56aaf5bc805e60265e75dca3dccacc1d60a75bea5d9c469f8 *vendor/netcrust/vendor/hdf5-reader/src/messages/modification_time.rs
36df05369b85c9f632976a3613799911958f1f9073f3115df269c4e4846b4b65 *vendor/netcrust/vendor/hdf5-reader/src/messages/shared.rs
ae41817e0f4a14ca0c530d4331d21942209d7299260e7c3856c72dd82a4af269 *vendor/netcrust/vendor/hdf5-reader/src/messages/symbol_table_msg.rs
dd332dd4c8c10407f025e18a4d1d5fc05b2b0509c119a3be087a3ac88b7fdfa8 *vendor/netcrust/vendor/hdf5-reader/src/object_header.rs
7dd9a92cd29dedab2ba99e893725527debadf5c39551d244f86df542e4efe2f0 *vendor/netcrust/vendor/hdf5-reader/src/reference.rs
ff78c810464721df6b779e11c23c9ccae303ce2f25519cbc4ecffd6e60d6e70e *vendor/netcrust/vendor/hdf5-reader/src/storage.rs
18f95086ef18b467e6a9971d97e81d17a647890b591d91456dc69c37b57a2fe3 *vendor/netcrust/vendor/hdf5-reader/src/superblock.rs
2a7b8463d4902c81d25833da3ecf97558a10470fcd450da712a4c4e3afc72723 *vendor/netcrust/vendor/hdf5-reader/src/symbol_table.rs
c8950d584b271804ac580a4448ee641d842b6e3cdb1e18c9795f05d0a572d473 *vendor/netcrust/vendor/hdf5-reader/tests/corruption.rs
8e4cb7be54880027df02e455e96197e38fba54c9c05309fca07ebf443a3f09c4 *vendor/netcrust/vendor/hdf5-reader/tests/integration.rs
a68e27c0b4034bea360b872bb4fd9c7bbe821be8eaac90b392297a565d8a2e70 *vendor/netcrust/vendor/hdf5-reader/tests/proptest_tests.rs
