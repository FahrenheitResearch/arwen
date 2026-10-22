# Local forecast publication

`gpuwm go CONFIG --publish-root DIR` writes a static website as live products finish.
The default is the run output's `www` directory. `--render-products none` disables
rendering and publication. Serve `DIR` with an ordinary static web server; this
door does not start a server or upload anything.

`manifest.json` uses schema `gpuwm.publish.v1`. Each `hours` entry identifies its
cycle, domain, forecast hour and valid time, and names immutable PNG URLs under
`<cycle>/fNN/dNN/<generation>/`. `index.html` reads this manifest. Generations are
installed before an atomic manifest replacement, so a reader sees a complete
published batch. Independent processes sharing a root serialize manifest updates.

The five priority products are `composite_reflectivity`, `2m_temperature`,
`10m_wind_speed_and_direction`, `qpf_1h`, and `mslp_10m_winds`. An all-products live
request draws that set first, publishes it, then draws the full request. Explicit
subsets are preserved. No precipitation window exists at f00, so its required list
omits `qpf_1h`. Missing required products leave `online_at` null.

Each entry records:

- `hour_closed_at`: UTC at history submission, before serialization/durability.
  Older producers that supply only commit events leave this null, with
  `timing_source: unknown`; commit time is never substituted for model hour close.
- `render_start` / `render_end`: UTC around the render work, excluding publication.
- `online_at`: UTC observed immediately after the manifest replacement naming
  the complete priority set; a second replacement records that observation. `online_lag_seconds` subtracts `hour_closed_at`.
- `priority_render_start` / `priority_render_end`: timings of that first set,
  retained when the full set replaces the row.
- `all_products_online_at`, `complete`, `products_count`, and the required,
  present and requested priority lists distinguish the two publication phases.
- `hydro`: the launch's requested hydro policy; `auto` does not assert activation.

Rendering stays on the existing Rust renderer through `gpuwm.rustwx`. Live workers
share CPU and memory budgets alongside stepping. The last worker, once stepping
has closed its queue, gives its next render subprocess all available cores at
normal priority (an explicit `RAYON_NUM_THREADS` still takes precedence).
History staging can be reused as soon as native serialization consumes its arrays;
fsync, validation, hash and durable history callbacks still complete in order.
The renderer currently consumes committed history files; this is not a direct
in-memory weather-field interface, and real-size timing must establish its lag.

`--publish-cmd 'COMMAND ARGUMENTS'` is an optional, disabled-by-default integration
hook. It receives the committed hour row as JSON on stdin and
`GPUWM_PUBLISH_ROOT` / `GPUWM_PUBLISH_MANIFEST` environment variables. Arguments are
parsed without a shell. Hook failures warn and retain the local publication.
Configuring a website upload command is a separate deployment decision.
