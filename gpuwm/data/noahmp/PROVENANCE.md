# Noah-MP v4.6.1 parameter assets

These are the unmodified Git blobs from NCAR/noahmp commit
`848f54ad3d28c4303151fe5ad83724e232694422`, the submodule revision pinned by
WRF v4.6.1 commit `d66e442fccc04111067e29274c9f9eaccc3cef28`.

| Asset | Bytes | SHA-256 |
|---|---:|---|
| `MPTABLE.TBL` | 56,140 | `7fae6a77660c90ad80845565ecfb057093c100de41f35f25a7ffa63f41c19e5d` |
| `SOILPARM.TBL` | 6,557 | `1e2275a32d8cd3b48ca693d22c0816df0013f83b6594ac632716361db337d58f` |
| `GENPARM.TBL` | 261 | `9c02832a0e4a2ecaf47fcee485539aad95cd732c379c5c258161a88eb3d25ea2` |

The files intentionally retain canonical LF line endings. A checkout expanded
to CRLF is different input and is rejected by `load_noahmp_parameters`.
Packaging and parsing these assets does not by itself admit
`sf_surface_physics=4`; the Noah-MP column solver and integration gates remain
required.
