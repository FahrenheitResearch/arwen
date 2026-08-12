# Chunked forecast streaming

**This page is not [Tiling a domain that does not fit on the card](TILES.md).**
That page is the `[tiles]` table, which runs one domain out of core by
cycling it through the card a tile at a time. This page is a fetch-and-run
cadence over a live HRRR cycle. The two share no configuration and no code
path, and either can be used without the other.

`gpuwm stream PLAN.toml` follows an uploading HRRR cycle with sealed hourly
forecast legs. It uses the Rust full-file fetch path, prepares a new immutable
forcing prefix, resumes from the preceding tree checkpoint, requires initial
and final health PASS, and hashes every authority receipt and checkpoint
member into a durable chain. It never changes forcing inside a live model
process.

The experiment must contain at least two domains and set
`restart_interval_s = 3600`. `gpuwm domain --polygon ...` can fit every domain
level around a local GeoJSON Polygon, MultiPolygon, Feature, or
FeatureCollection with one buffer per level. The streaming plan also needs the
native root-domain JSON and the WPS/native/stock WRF namelists used by the
ordinary prepared-hierarchy route.

```toml
schema = "gpuwm-stream-plan-v1"

[stream]
work_root = "/forecast/stream-job"
cycle = "latest"
cycle_count = 2
target_lead = 4
poll_seconds = 30
wait_timeout_seconds = 7200

[fetch]
cache_dir = "/forecast/cache"

[prepare]
experiment_config = "/forecast/config/experiment.toml"
domain_spec = "/forecast/config/root-domain.json"
wps_namelist = "/forecast/config/namelist.wps"
namelist_input = "/forecast/config/namelist.input"
stock_wrf_namelist_input = "/forecast/config/namelist.stock.input"
geog_root = "/forecast/WPS_GEOG"
physics_profile = "thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1"
pipeline_workers = 8
prepare_workers = 8
child_workers = 8
preprocess_backend = "cpu"
preprocess_workers = 8

[run]
io_mode = "history"
health_debug = false
gpu_uuid = "GPU-01234567-89ab-cdef-0123-456789abcdef"
allow_shared_gpu = false
```

`physics_profile` was `wsm6-ysu-mm5-noah-no-radiation-v1` in this plan
through 1.8.7. Read that name carefully if you have it in a plan of your
own: it does not run "no radiation", it runs Dudhia **shortwave** with
longwave **off**, so nothing computes the downward longwave the land
surface reads and a streaming job that crosses local night is running the
1.7.1 dewpoint-collapse configuration. The template's registry entry now
says so in its own warnings. `thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1`
is the HRRR route's default and runs both radiation streams; it is what
this plan hands you now.

Run it once:

```console
gpuwm stream PLAN.toml
```

`cycle = "latest"` is resolved when the command starts. For a multi-cycle
job, each later cycle must be the exact hourly successor; the controller
refuses to leapfrog a missed cycle. `cycle_count` is bounded, so a completed
plan is idempotent. Use a new work root for a new bounded job.

`target_lead` defaults to 1. Set a larger value deliberately: every hourly
lead retains a complete root preparation, prepared hierarchy, forecast run,
and tree-checkpoint generation. A target of 18 therefore retains 18 such
generations; an extended-cycle target of 48 retains 48. The controller never
deletes older generations.

`gpu_uuid` selects the physical GPU by UUID, independently of CUDA ordinal
ordering. With the default `allow_shared_gpu = false`, the controller holds
the supervisor UUID lock and refuses unrelated compute processes on that GPU.
Set `allow_shared_gpu = true` only when deliberate GPU sharing is acceptable.

## What is incremental

The root preparation is a real one-hour extension. For fNNN after f001 it
decodes only the terminal overlap hour and the newly available hour, proves
the old source and bridge manifests are unchanged prefixes, then hardlinks
the immutable prepared-cache arrays, bridge payloads, and static artifacts
from the preceding root. This requires predecessor and successor roots on the
same filesystem; the command refuses instead of silently copying them.

The nested hierarchy is currently rebuilt in full for each hourly leg. Each
forecast is also a new sealed process: it restores the preceding tree
checkpoint and integrates to the new endpoint. No running process observes a
forcing-file mutation.

When `cache_dir` is set, the full forcing objects are retained twice: once in
the cycle source tree and once in the fetch cache. The initial
`disk-capacity.json` prices every source hour, the cache copy, retained
forecast generations including t=0 and configured history cadence, and a
fixed 2 GiB margin. Before each fetch, `disk-headroom.json` rechecks the next
source/cache writes, one generation, and a 512 MiB emergency margin. A shared
work/cache filesystem is charged one combined requirement against one free
space observation; split filesystems are gated independently and their free
space is never summed. Headroom has three fail-closed source-prefix states: a
strictly verified requested prefix has zero source/cache write demand; a
strictly verified predecessor prefix prices only the new terminal hour; and
any corrupt, partial, or otherwise unverified predecessor prices replacement
of the complete f000..fNNN prefix (using every sealed HEAD size, with the
enforced envelope as fallback). Generation and the emergency margin remain
gated in every state.

The enforced full-object reservation envelope is 8 GiB per source hour for
the combined atmosphere and soil products. An observation above that envelope
refuses before download so a future object cannot invalidate the initial
capacity receipt.

The work root contains:

- `stream-summary.json`: the cross-cycle hash chain.
- `cycles/<cycle>/chain-summary.json`: the per-cycle PASS timeline.
- `cycles/<cycle>/legs/fNNN/chain-link.json`: the availability observation,
  stage timestamps, prior-checkpoint hashes, output-checkpoint hashes, health,
  and authority-artifact hashes for one sealed leg.
- `active-cycle.json`: the atomic crash-resume marker.

Rerunning the same plan verifies every retained link and artifact before it
adopts completed work. It also recomputes the outer cycle count and chain tail,
requires status/count agreement, and proves every cycle is the exact hourly
successor of the preceding cycle. Incomplete create-only outputs are moved to an
`interrupted` quarantine name and rebuilt; verified fetch files remain in the
production source directory and are reused by the fetcher's normal digest and
inventory bars.

## Availability and timeline semantics

The S3 watcher requires both forcing objects (`wrfnat` and `wrfprs`) and both
of their `.idx` indexes. It records URL, Content-Length, ETag when supplied,
Last-Modified, and first-observed time for each. After fetch it repeats the
object/index observations and refuses identity drift. The fetch manifest then
binds the effective URLs and transports to downloaded byte counts, sizes, and
SHA-256 digests.

The timeline reference is `forcing_set_first_observed_at`: when this running
controller first observed the complete four-resource set. It is not a
fabricated producer upload/completion timestamp. `remote_ready_last_modified_at`
is the maximum serving-endpoint HTTP Last-Modified value and is labelled with
that narrower meaning.

Each timeline row reports distinct root and hierarchy preparation intervals:
`root_preparation_started_at`, `root_preparation_completed_at`,
`root_preparation_seconds`, `hierarchy_preparation_started_at`,
`hierarchy_preparation_completed_at`, and
`hierarchy_preparation_seconds`. `preparation_completed_at` is retained as a
compatibility alias for `hierarchy_preparation_completed_at`. Fetch and
forecast intervals and availability-to-completion durations are reported
separately.

Local deterministic tests prove controller restart, manifest integrity,
public root-prefix extension, production hierarchy artifact/preflight
contracts, and restart-set compatibility. They are not evidence of live HRRR
latency or GPU forecast advancement. Only an unattended run against a real
uploading cycle, using the shipped hierarchy and forecast commands, supports
the operational timeline and per-leg validity claim.
