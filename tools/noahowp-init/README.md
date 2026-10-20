# Rust Noah-OWP initialization

This production preparer reads a native Noah-OWP namelist and its Apache-2.0
SOILPARM/GENPARM/MPTABLE parameter files. It initializes every resident field
without a Fortran dependency. Rayon initializes catchments in parallel;
float32 arithmetic preserves the pinned native initialization order. Existing
native undefined fields receive zero; documented HUGE sentinels are preserved.
Forcing is supplied by the resident forcing component before the first step.

Build from this directory with `cargo build --release --offline --locked`.
Put CARGO_HOME, CARGO_TARGET_DIR and TMPDIR under the lane scratch directory.
The executable and cdylib use the normal staged bridge resolver via
GPUWM_NOAHOWP_INIT and GPUWM_NOAHOWP_INIT_LIBRARY.

The Python `prepare_initial` door uses the packaged, pinned Apache-2.0 standard
parameter tables. Its optional `parameter_dir` chooses a different default;
explicit per-row directories take precedence. Direct CLI calls preserve the
namelist's parameter directory unless `--parameters DIRECTORY` is supplied.

Run `arwen-noahowp-init REQUEST.json NEW_OUTPUT_DIRECTORY`. Example request:

```json
{"catchments":[{"id":"cat-id","namelist":"/path/to/namelist.input",
 "parameter_dir":"/path/to/parameters",
 "overrides":{"lat":40.0,"lon":-90.0,"isltyp":3,"vegtyp":5}}]}
```

Each row has its own real catchment attributes and initial profiles. Overrides
are generic namelist keys, including scalar strings and numeric vectors. Rows
must follow the routing catchment order. Relative namelist paths resolve from
the request; parameter_dir overrides also resolve from the request. Without
that override the namelist parameter_dir resolves from its containing folder.
All three table files and each namelist are hashed in the output manifest.
No calibration or spatial defaults are invented by the preparer.

Prepared initial soil temperature uses the extension
`overrides.soil_temperature_k`: exactly four finite positive Kelvin values,
ordered like the four soil entries of `dzsnso`. Rust converts these to native
REAL (float32) and writes `energy.stc(1:4)` after the native cold-start
constants. Absence preserves the pinned native 298 K initialization exactly.
The request SHA and per-column overrides seal the profile in the existing
manifest and restart identity.

Prepared snow uses three overrides together: `snow_water_equivalent_mm`,
`snow_depth_m`, and `snow_temperature_k`. The explicit
`aggregate-all-ice-divide-v1` recipe assumes zero liquid, zero age, and a uniform
initial temperature. It uses the pinned Apache Noah-OWP DIVIDE/COMBO operation
order to initialize up to three thermal layers; packs below 0.025 m remain in
native shallow aggregate storage. It updates snow ice, temperature, geometry,
old ice fraction and SWE, while preserving the soil profile. Two zero aggregates
preserve all native cold-start words. Missing keys preserve the original cold
start; a partially specified pack is refused. Negative/nonfinite mass or depth,
inconsistent mass/volume, temperature above freezing, density above ice, and
nonfinite derived native REAL values are refused because the all-ice recipe
cannot represent them. This recipe does not recover an unobserved liquid
fraction, snow age or vertical temperature profile.

The `hydro-world-init engine-packets` producer joins the sealed state and spin-up
packets to the fabric by exact catchment-ID order and epoch. It uses the declared
four soil-layer bounds, supplies all three soil profiles and the three aggregate
snow arrays, and converts SWE from metres to millimetres before native float32
rounding at the producer boundary, before JSON transport. Array hashes, units and shapes are checked before output creation.
The per-column manifest names the snow recipe. Initialization and serialization
proofs do not establish physical first/next-step continuation; that still needs
a coupled reference with the same forcing and complete state. Existing native
cold-start replay results keep their original inputs.

For a real initializer/serialization check, build and run the Rust example:
`cargo run --release --offline --locked --example profile_check -- NAMELIST PARAMETERS NEW_OUTPUT`.
It checks two distinct per-catchment profiles, all unaffected serialized
words, sealed identities and malformed-profile refusals before output creation.

Output has field-major f32/i32/f64 slabs and a sealed metadata manifest. The
realization factory `gpuwm.hydro.noahowp.factory:prepare_slab` accepts
`initial_manifest`, `fabric_sha256`, `time_origin`, optional `catchment_ids`
and optional `forcing_component`. Rust verifies payload hashes before GPU
transport; the factory binds resident arrays once. No per-step host I/O.
The factory checks every resident `domain.start_datetime` against `time_origin`
for manifest, packed-slab and complete-field inputs. Supplying another date
does not relabel an initialized packet: regenerate it for that date. ISO UTC
offsets describe the same instant; timestamps without an offset retain the
UTC convention. This checks the initialized model clock, not the provenance
or physical completeness of the initial water state.

The current CUDA allocation is 4 soil and 3 snow layers, NROOT 1..4. The parser
supports the upstream USGS/MODIS namelist tables, quoted paths, comments and
repeat counts. Indexed namelist assignments are explicitly rejected because
silently ignoring their indices would initialize the wrong layers.

Initialization metadata is generated offline from the pinned Apache upstream
by `tools/noahowp_port/init_rules.py`; it is not a runtime Fortran interpreter.
