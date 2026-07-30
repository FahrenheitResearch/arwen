# ecape-rs

`ecape-rs` is a standalone Rust crate for ECAPE parcel calculations. It was split out of the JavaScript rewrite work and is now tuned for direct parity against the original Python `ecape-parcel` package.

## Status

- Standalone Rust crate
- Depends on `metrust` for the sensitive meteorological primitives
- ECAPE-specific parcel logic implemented in this crate
- Parity-checked directly against `ecape-parcel` on real-world soundings

## Verification

Focused real-world checks currently show near-parity against the reference implementations:

- `OUN 2024-05-06 00Z`: max parcel-temperature diff `0.000129 K` pseudoadiabatic, `0.0000023 K` irreversible
- `LBF 2024-06-20 00Z`: max parcel-temperature diff `0.0000145 K` pseudoadiabatic, `0.0000066 K` irreversible
- `BMX 2024-03-14 00Z`: max parcel-temperature diff `0.000394 K` pseudoadiabatic, `0.0000230 K` irreversible

Typical speed from the same checks:

- `ecape-js`: roughly `58 ms` to `159 ms`
- `ecape-rs`: roughly `0.41 ms` to `1.73 ms`

Detailed numbers are in [verification_summary.json](./verification_summary.json).

## Build

```bash
cargo build --release
```

## Notes

- This crate currently targets direct parity with `ecape-parcel`, with `ecape-js` retained as an additional cross-check.
- It uses Cartesian wind components internally: `u > 0` eastward, `v > 0` northward.
- Supported storm-motion modes are `right_moving` (Bunkers RM), `left_moving` (Bunkers LM), `mean_wind`, and `user_defined`.

## Acknowledgements

This work is derived from the original Python `ecape-parcel` package and the later parity work in `ecape-js`.

## License

MIT
