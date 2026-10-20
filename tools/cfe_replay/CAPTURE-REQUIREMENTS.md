CFE source at `f9182dfb3d81c407c66c5b782114173893c30cf2`: “Apache License”; “Version 2.0, January 2004” — Apache-2.0.

This is a source-based boundary review, not a second capture schema or a runtime
ABI. Referee owns the fresh fixture schema. CFE port owns the concrete device
field ordering in `FIELD_NAMES`/`OPTION_NAMES`; its current contract is
the CFE port's own CONTRACT.md. Driver owns the prepared-stage
contract, including resident cross-step state and enqueue-only launches.

At this pin, `src/bmi_cfe.c` has concrete serialization defects:

| Source location | Actual behavior | Consequence for capture |
|---|---|---|
| `var_info` / `STATE_VAR_NAME_COUNT` | Advertises 95 fields; `Get_state_var_ptrs` assigns only slots 0–91 | Slots 92–94 are not valid pointers |
| `Get_state_var_ptrs`, slot 85 | Points to `verbosity` (`int`); table says `flux_Qout_m` (`double*`) | Reading advertised eight-byte value can overread the field |
| Same function, slot 86 | Points to the Schaake constant (`double`); table says `verbosity` (`int`) | Capture truncates/mislabels the value |
| Same function, slots 87–89 | Pointer targets and table names are shifted by one | Valid doubles receive incorrect meanings |
| Same function, slot 90 | Exports **address of** `soil_reservoir.soil_layer_depths_m` | Captures process pointer bits rather than soil-layer values |
| Same function, slot 91 | Points to `max_rootzone_layer` (`int`); table says soil profile (`double`) | Wrong value, width and shape |
| `Get_state_var_sizes` | Writes string length into slot55 (`is_forcing_from_bmi`), GIUH+1 into74 (`forcing_data_time`), lateral-Nash+1 into75 (`giuh_ordinates`), GIUH+1 into76 (`nash_storage_subsurface`) | Legacy sizes cannot delimit the pointed-to arrays |

Use explicitly named typed fields and live counts, rather than the experimental
serialization table. The metadata audit reports 146 declared leaf members,
54 absent from its pointer mapping. Some are unused/uninitialized members;
this count describes source structure, **not** a required numerical word count.
Do not capture struct padding, addresses, allocation slack or unused bytes.
The referee must record which values are actually initialized/read/written at
the chosen boundary. No tolerance or exclusion is granted by this document.

The boundary must be named exactly:

- BMI `Update` includes rainfall selection/conversion (`precip/1000`),
  `time_step_fraction`, `run_cfe`, `current_time_step += 1`, then ponded-depth
  summation from the selected surface-routing queue. Entry/exit around this call
  tests the public BMI update semantics.
- `run_cfe` calls the numerical core only. A fixture captured there cannot by
  itself prove BMI forcing conversion, counters or ponded-depth output.
- BMI `Update_until` may temporarily change `time_step_size` and fraction;
  record whether the captured call is a full or fractional step. Do not infer
  timestep solely from a realization's nominal hourly interval.

State needed to reproduce the actual configured call includes the reservoir
storage/thresholds/coefficients, soil and infiltration parameters, all ET fields
used by the call, mass-balance accumulators, carried flux values (notably prior
`flux_perc_m` read by `cfe`), active scheme flags, timestep and forcing values.
For root-zone ET/SFT, retain the active profile/depth/ice/flag inputs rather than
substituting uncoupled defaults. Preserve the full initialized output state,
not just `Q_OUT` or the BMI output list. The BMI output called
`SURF_RUNOFF_SCHEME` actually points to
`infiltration_excess_params_struct.surface_water_partitioning_scheme`; the
distinct `surface_runoff_scheme` selects GIUH versus Nash. Keep these separate.

| Dynamic pointee | Native meaningful length | Evidence |
|---|---|---|
| `giuh_ordinates` | `num_giuh_ordinates` | `giuh_convolution_integral` reads indices 0..N-1 |
| `runoff_queue_m_per_timestep` | `num_giuh_ordinates + 1` | Same routine writes queue[N]=0, then shifts indices1..N |
| `nash_storage_subsurface` | `N_nash_subsurface` | `nash_cascade_subsurface` loop bounds |
| `nash_surface_params.nash_storage` | live `N_nash` when allocated | Native surface-Nash/retention-ET path; preserve NULL distinction |
| `soil_reservoir.smc_profile`, `soil_layer_depths_m`, `delta_soil_layer_depth_m` | live allocated/read root-zone lengths | Referee records initialization and configured `n_soil_layers`/root-zone mode |

The current port contract and Rust packer use **97 f64 scalar rows, 18 i32
option rows and five i32 ragged-offset rows**, in the exported tuple order.
Option row17 (zero-based), `nash_surface_params.has_storage`, preserves native
pointer presence independently of `nash_surface_params.N_nash`. It is a derived
binding flag, not an additional native scalar member. At the pinned source's
`src/bmi_cfe.c:1123-1124`, GIUH initialization explicitly sets surface-Nash
storage to NULL. The packer witnesses that case through the selected scheme
and absent storage descriptor; active surface Nash requires a complete
positive-length storage descriptor. A 17-row option matrix loses this
distinction and does not match the current ABI. Zero-length inactive storage
remains a declared checkpoint carrier.

The setup-side packer maps the captured native paths into these matrices.
Array processing and byte comparison use Rust/CUDA. No numerical host work, new device
allocation, readback, synchronization or CUDA graph belongs in a prepared step.
Fast remains default; strict proof is explicit. Replays report all differing
words with no implicit tolerance or thinning. Current real-boundary evidence
and its exact source/input scope are recorded in [README.md](README.md).
