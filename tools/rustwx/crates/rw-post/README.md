# rw-post: WOOF post-processor (clean room, Apache-2.0)

The GPU-first post-processor of WOOF 2.8.7, written only from the clean-room
post specification (2026-10-05) and the public sources it cites. It replaces
the post modules held out of 2.8.6.

## One crate

| Module | Group | Products |
| --- | --- | --- |
| `state`, `history`, `hybrid` | A | the ONE shared column state (SPEC 3.4) every group consumes |
| `surface`, `thermo`, `solar`, `wind`, `snow`, `grib_coords` | A | shelter and surface fields, shared helpers |
| `plev` | B | pressure levels (height, T, Td, RH, u, v), PRMSL, MAPS SLP, membrane SLP, PWAT |
| `severe` | C | sb/ml/mu/best-180 CAPE and CIN, LCL, storm motion, SRH, shear, STP, EHI |
| `group_d` | D | composite and 1 km reflectivity, low/mid/high cloud, cloud base, top and ceiling, visibility, simulated infrared brightness temperature (`simulated_ir`, own kernel `woof_post_d_ir_v1`) |
| `group_e` | E | PBL height, gust, 80 m wind, freezing, highest-freezing, -10 C and -20 C levels |
| `math` | all | the ONE maths library (D30) |
| `gpu` | all | the ONE kernel loader (`GpuPost`) |
| `reference` | tests | binary64 forms the tests grade both paths against |

Group D was ported from the retired Python + CuPy lane (`lane/cleanroom-post-d`);
no Python remains in its data path.

## One state

Group A builds the column state once per frame (`state::build_cpu`, or
`GpuPost::build_state` on the device): theta, r, q, hydrostatic p (D1),
p_full, T, Tv, theta_v, z_mass, destaggered u and v on mass levels, and
hydrostatic p_int and z_int on interfaces. Groups B, C, D and E read its slots;
none of them rebuilds pressure, temperature or winds.

## One maths library, one answer

`src/math.rs` and its CUDA twin `kernels/woof_math.cuh` implement ln, exp,
pow, cbrt, sin, cos, atan, atan2 and asin from IEEE `+ - * /`, `sqrt` and
bit operations in the same order on both sides. No platform C library, no
`libm` crate and no CUDA libdevice transcendental is called. Rust never
contracts `a * b + c`; the kernels are built with `--fmad=false
--prec-div=true --prec-sqrt=true --ftz=false`. The CPU reference (rayon
over columns) and the GPU return the same bits, on Linux and Windows and on
every architecture.

## One kernel set

`kernels/woof_post.cu` is one translation unit holding every group's
kernels (C, E and B each in their own namespace). It is compiled into
CUBINs for the two certified Blackwell architectures
(`woof_post_sm100.cubin`, `woof_post_sm120.cubin`) plus a compute_75
`woof_post.ptx` that every other card JIT-compiles, described by
`woof_post.manifest.json` (source and artifact hashes), and loaded through
the driver API only (`cudarc =0.19.8`, dynamic loading). User machines need
no CUDA toolkit and no NVRTC. Only those three images are embedded: each
is compiled into every binary that links this crate, and the twelve-CUBIN
set put the 2.8.7 manylinux wheel over PyPI's 100,000,000-byte limit.

## Method switches

Each is one constant in Rust and its twin in the kernel; change both.
The group D switches are one Rust constant each, passed to the kernel as a
launch flag or argument, so there is no kernel twin to change.

| Switch | Default | Ruling |
| --- | --- | --- |
| `severe::wind::SHEAR06_LAYER_MEANS` / `SHEAR06_LAYER_MEANS` | false: 10 m to 6 km endpoints | RULINGS.md 3 |
| `severe::wind::STP_FLOOR_ZERO` / `STP_FLOOR_ZERO` | true: floored at 0 | RULINGS.md 4 |
| storm motion | Bunkers 2000, height-weighted 0-6 km mean, 7.5 m/s | RULINGS.md 2 |
| `group_e::column::TROP_P_MAX/MIN` / `W_TROP_P_MAX/MIN` | 500 to 50 hPa | RULINGS.md 8 |
| `group_e::GustMethod` | exporter `auto`: `MixedLayerTke` where the history carries TKE, else `Similarity` (`--gust similarity` or `--gust tke` force one) | RULINGS.md 7, settled by the ASOS gust score (2.8.7 merge-and-accept) |
| `group_d::CLOUD_FRACTION_SOURCE` | `ModelThenDiagnosed`: CLDFRA when present, else diagnosed (`ModelOnly` drops the six cloud fields without CLDFRA) | RULINGS.md 1 |
| `group_d::DIAGNOSED_CLOUD` | `XuRandall1996` (`CondensatePresence` the alternative) | PROVISIONAL: RULINGS.md 1 names no method and SPEC 7.2 cites none |
| `group_d::REFL_INTERP` | `LinearZ` (`LinearDbz` the alternative) | RULINGS.md 5, MRMS check pending |
| `group_d::VIS_CONTRAST` | `CONTRAST_MOR_5PCT` (`CONTRAST_2PCT` the alternative) | RULINGS.md 6, ASOS check pending |
| `group_d::column::IR_K_LIQUID` / `IR_K_ICE` (twins `D_IR_K_LIQUID` / `D_IR_K_ICE` in `group_d.cu`) | 0.145 and 0.272 m2 per g of condensate: the window absorption that decides where a column is opaque for `simulated_ir` | PROVISIONAL: no peer-reviewed source traced; kept by the GOES-16/18 band 13 score of the b2-ir lane (halving or doubling both still beats 2.8.5 on every score) |

## Commands

    cargo test -p rw-post                                    # CPU tests
    cargo test -p rw-post -- --include-ignored               # plus device tests (a card) and fixture tests
    WOOF_POST_FIXTURES=DIR cargo test -p rw-post --test group_e -- --include-ignored fixture
    cargo run -p rw-post --release --example groups_frame -- WRFOUT OUTDIR [--no-gpu]
    cargo run -p rw-post --release --example group_d_frame -- WRFOUT OUTDIR [--cf model|model-only|diagnosed] [--diag xu|presence] [--interp z|dbz] [--contrast 5|2] [--tile N]
    cargo run -p rw-post --release --example post_frame -- WRFOUT --out DIR --device both
    cargo run -p rw-post --release --example severe_fixtures -- FIXTURE_DIR
    cargo run -p rw-post --example rebuild_kernels           # developer only, needs NVRTC 13
    python scripts/compare_group_c.py OLD_SFC.grib2 DIR cpu  # black-box distance, values only

With a card visible (`CUDA_VISIBLE_DEVICES` set), a device test fails when
the kernels do not load; it never passes silently.
