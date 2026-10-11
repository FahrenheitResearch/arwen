# Source-pinned model presets

`gpuwm preset list` lists named native model specifications and their launch
status. The registry is `gpuwm/data/model_presets/registry.v1.json`. A model
is a metadata row; the CLI never selects a scheme by a model name.

The `rrfs-v2-hex-tempo` specification follows the public RRFS v2 development
workflow at `45ace1bf97776ea51d257249b843da34cabdebde`. Its forecast model is
MPAS `1d6e370e7b43a332074809f170532a216c2212fb`, with TEMPO
`9adf9ef2827104640125f0c322e1218fce74e94c`. The registry supplies links and
SHA-256 values for the source files used to assemble its settings.

The configuration selects the `conus3km` mesh, 59 layers with 60 specified
interfaces ending at 25878.712 m, a 20 s step with four acoustic substeps,
RK2, three dynamics splits, monotonic transport, eight CAM damping levels
with coefficient 2, and horizontal-wind Rayleigh damping starting at 16 km.
The physics are the `hrrrv5` suite: TEMPO with aerosol, hail and ML droplet
options enabled, MYNN-EDMF and its surface layer, nine-level RUC, UGWP,
legacy RRTMG on a 900 s cadence with MYNN cloud fraction, and convection off.

Inspect and export the configuration:

```sh
gpuwm preset show rrfs-v2-hex-tempo
gpuwm preset export rrfs-v2-hex-tempo --start 2026-10-03_12:00:00 --seconds 3600 --out namelist.atmosphere
gpuwm preset check rrfs-v2-hex-tempo
gpuwm preset verify-sources rrfs-v2-hex-tempo
```

The export retains every requested physics switch. The start and duration
are explicit inputs, not defaults tied to a weather event. The mesh and
vertical authority URLs are metadata; export does not download or prepare
them. `configs/rrfs-v2-hex-tempo.toml` identifies this specification and is
not an ARW RunConfig or an input to `gpuwm go`.

The preset ships as an **RRFS configuration with declared divergences**: every
runtime requirement is implemented, and what still differs from NOAA's pins is
named in `declared_divergences` with what runs instead, why and the follow-up:
the MPAS 8.4.1 dycore (kept by the 2026-08-20 ruling), the MYNN surface layer
(the pinned b66187f generation is a qualified port held for 2.8.9), MYNN-EDMF
`c942418`, the RUCLSM `8c9f1a8` MPAS code line and the UGWP `c1c893ed`
orographic drag. Its verification of record is observation skill (ASOS, MRMS).
It runs from the front door:

```sh
gpuwm preset run rrfs-v2-hex-tempo --mesh conus12km --grid GRID --static STATIC --init INIT --lbc-dir LBC --invariant INVARIANT --start-time YYYY-MM-DD_HH:MM:SS --init-source SOURCE --hours 6 --history-every-minutes 60 --card RTX-PRO-6000 --out OUTPUT
```

Run this command inside the card queue. The existing hex forecast door checks
the mesh, inputs, soil levels, source pins and device. The runner records a
measurement receipt and full-field boundary hashes. This arm earns no native
parity or registered integration anchor. The 12 km arm selects New Tiedtke,
60 s and two acoustic substeps; the 3 km arm selects convection off, 20 s and
four substeps. Calendar and source inputs remain explicit.

`gpuwm preset check rrfs-v2-hex-tempo` exits 0 and reports the declared
divergences without any GPU allocation; an unmet requirement would exit 3 and
refuse `preset run` unless `--experimental` is given. The TEMPO column proof includes native optional ML
cloud radii and pre-physics emissions, all 26 output slots over 200 columns
and 175 set-step pairs, with zero differing defined words in strict and default
arithmetic. The experimental 12 km preset completed 360 steps and seven hourly
frames through the ordinary door, with native-cell comparisons against the
NOAA MPAS fork and Rust-rendered NOAA and HRRR sheets. Shortwave aerosol optics
are enabled through `aer_opt=3` with NOAA's land/water aerosol type
(`rrtmg_aer3_type_source = "landmask"`: rural over land, maritime over water),
held word for word to the NOAA MPAS fork's Fortran on a mixed column deck. This
does not establish mixed aerosol tendencies, workflow-pinned physics interfaces
or dycore equivalence. The registry names each concrete missing behavior and its evidence
site. Implementation must remove the corresponding unmet requirement when
its breakage is fixed; changing a status label alone cannot qualify the run.

The configurable column-suite seam starts from the earlier hex integration.
The TEMPO extension carries thirteen transported scalars, excludes number and
volume moments from moisture loading, supplies the post-dynamics vertical
velocity, and adds the selected cumulus raw momentum before edge reconstruction.
The preset selects native ML cloud inputs and pre-physics aerosol emission;
the emission initializer uses the supplied cell area and lowest-layer depth.
Its first-call state and aerosol initialization status persist at transaction
and restart boundaries. Whole-model restart identity needs a separate real check.
Existing microphysics selections retain their dispatch and arithmetic.
