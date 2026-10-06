# Physics parameter sets

Registered constants can be selected for a forecast without changing its
prepared inputs. With no set, kernel source, parameter tables and public
configuration retain their default bytes. No parameter identity or output
attribute is added. No tuned set is a new default.

A set file contains:

```toml
[physics_params]
name = "parameter-member"
values = { "mynn.prandtl" = 0.8, "ruc.z0.short" = 0.6 }
```

An experiment may carry the same table, or reference a relative set file
with `[physics_params] set = "member.toml"`. Unknown constants, invalid
values, conflicting set sources and constants whose schemes run on no
domain refuse before the forecast. Ranges and exact CUDA source sites live
in `gpuwm/physics_params_registry_v1.json`. CASE(1) mixing-length parameters
do not change CASE(2). Roughness rows exclude seasonal crops whose roughness
receives a subtractive seasonal adjustment.

For members sharing a preparation, keep the experiment file and prepared
root unchanged. Launch each forecast in its own process with
`GPUWM_PHYSICS_PARAMS` naming the absolute set-file path. Set this variable
only for the member forecast, after preparation. An absent variable selects
the ordinary default member. Parsing resolves the file through
`gpuwm.physics_params.environment_set()`; programmatic validation uses
`parse_table()` or `make_set()`. Values are canonical float32 values, and
`document()` supplies the name, values, registry digest and set digest.

Direct Python `run_experiment()` and model construction bind the validated
set before preparation. Execution of an already-built tree retains its
original set and refuses a first selection or a switch after construction.

Future ensemble recipe support needs an optional member set-file reference.
The recipe resolves the path relative to its own file, validates the set,
checks its schemes against the member experiment, and records `document()`
in the member receipt. It passes the absolute path through the child
process environment. This member option does not enter the preparation
cache key: all registered edits are forecast-time constants. The launch
must sanitize an inherited `GPUWM_PHYSICS_PARAMS` for default members and
must not compile kernels in a shared parent process before setting a
member's environment. Members use separate output and checkpoint roots.

The set digest binds the experiment identity used by checkpoints. Outputs
carry `GPUWM_PHYSICS_PARAMS` and `GPUWM_PHYSICS_PARAMS_SHA256`; edited RUC
bundles retain pinned input-table hashes and record each changed cell.
CUDA source edits preserve every byte outside the named literal spans.
RUC table arithmetic runs on the GPU after pinned-table validation. One
process cannot switch to a different set after compiling kernels or
issuing tables; a new member requires a new process.
