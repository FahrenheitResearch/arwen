# ArWen 2.7 integration kit

Build a client around the versioned CLI documents and durable run files. The desktop GUI and Rust TUI use the same forecast engine. This kit describes those interfaces and includes a small Python subprocess client; it does not introduce a separate Python simulation API.

Start with `integration-guide.md`, then run `python examples/client.py --help`. `prototyping-brief.md` is a ready-to-use brief for exploring new interfaces and visual directions.

The JSON catalogs are snapshots from the candidate. Query the installed runtime again when constructing a real plan. Availability depends on the installed native renderer, the source, requested time, available inputs, and the selected computer.

The exporter substitutes `<USER_HOME>` for the build computer's home directory. Credential availability in a snapshot describes that isolated query, so a client must re-query the user's runtime.

## Files

- `integration-guide.md`: client boundaries, discovery, review, execution, reconnect, maps, and profiles.
- `examples/client.py`: JSON discovery, plan review, and durable run inspection without starting a forecast.
- `examples/prepared-plan.json`: envelope for an existing experiment TOML; replace its paths.
- `examples/test_client.py`: checks for torn event tails, schema validation, and run identity changes.
- `sources.json`, `physics-profiles.json`, and `catalog.json`: generated release catalogs when the corresponding capability is available.
- `prototyping-brief.md`: interface and visual prototyping instructions.

The portable Windows package's interpreter is `runtime/python.exe`. An ordinary installation can use its own `python -m gpuwm.cli` or the `gpuwm` console command. Keep credentials and SSH keys in the user's configured environment.
