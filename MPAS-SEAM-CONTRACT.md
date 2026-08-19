# MPAS column-batch seam contract

The frozen references an external MPAS CUDA core builds against.
FORWARD COMMITS ONLY on `lane/mpas-column-batch`; published shas are
never rebased.

## Contract v2 (glacier + classification), 2026-08-11

Frozen commits, forward on the v1 freeze `fa35bfbde`:

- `342f8780d` -- the implementation (`feat(glacier)`: the port, the
  threshold, the XLAND law).
- `e5a644002` -- the composed-unit loader fix plus the node-5 gate
  receipts (`evidence/noahmp-glacier-gate/`); build against this or any
  later tip of `lane/mpas-column-batch`.

Contract surface hash (the two contract-bearing files, concatenated in
this order and SHA-256'd):

```
sha256( gpuwm/core/mpas_column_batch.py + docs/mpas-seam.md )
  = 9405bc70cfbe60bdbd61057fe0880129ebe8f79c0dd4939f70636ce1ba5602e4
git blob gpuwm/core/mpas_column_batch.py   ce1a19482d2037f2e728c9ce657378b04a7747f5
git blob docs/mpas-seam.md                 2baf5fc70d4ab172475a3ae13d3f9aee0437b539
git blob gpuwm/core/noahmp_glacier.py      4dee338803efa25e34f3ef68701ef2873e045ff5
git blob gpuwm/core/kernels/noahmp_glacier.cu
                                           c6dd4740d288ffd581b35d4d878331bb9efd9c50
gpuwm subtree at 342f8780d                 066528c2e08ac7a22c23a440f567a112e274dc2d
```

Verify:

```
git cat-file blob 342f8780d:gpuwm/core/mpas_column_batch.py > a
git cat-file blob 342f8780d:docs/mpas-seam.md > b
cat a b | sha256sum
```

What v2 adds over v1 (full semantics in `docs/mpas-seam.md`):

- Glacier columns run. `IVGTYP == ISICE_TABLE` active land dispatches
  to the ported NOAHMP_GLACIER column (`module_sf_noahmp_glacier.F`,
  byte-frozen WRF v4.6.1 tree `d66e442fc`, file sha256
  `bf94f3522c3b9c2c9cfbb34fa7e485ff58519106db434520968793409a520579`),
  never to NOAHMP_SFLX. The guard that used to refuse remains as the
  disabled-path backstop only.
- `xice_threshold=` constructor input (default WRF Registry 0.5; the
  collaborator case runs 0.02). Part of the seam restart identity.
- `xland=` constructor input: the caller's classification wins
  VERBATIM; derivation from landmask is only the documented fallback
  when no xland is passed. `seam.surface_classification` names which
  source decided and counts every class;
  `seam.last_noahmp_census` reports per-step counts plus the glacier
  execution provenance.

## Contract v1 (the physics seam), 2026-08-10

Frozen at `fa35bfbde` (seam commit `7d914d016`, proofs commit
`ee3f99a1d`). Node-5 parity proofs and their provenance:
`evidence/mpas-seam-proofs/` (receipt sha256
`1496d033ac282ddeaa2807e9886288ae98d9bab204cfb73447784c247328acc4`).
