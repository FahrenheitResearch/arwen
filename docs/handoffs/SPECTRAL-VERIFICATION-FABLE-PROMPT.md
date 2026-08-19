# Prompt for the main Arwen Fable agent

You are integrating the additive spectral-verification v2 package into
`FahrenheitResearch/arwen`.

Base audited by the handoff: commit
`891a07ea248e6afe3fa6e180cc2ac58a8992a936` (`ArWen 2.4.1`). The live branch may
have advanced; inspect rather than blindly overwrite.

Read first:

```text
docs/handoffs/SPECTRAL-VERIFICATION-FABLE.md
docs/public/SPECTRAL_VERIFICATION.md
```

Then apply/reconcile:

```text
patches/0001-spectral-verification-v2.patch
patches/0002-spectral-stage1.patch
```

Hard constraints:

- Do not edit or re-pin `gpuwm/verify/spectral.py`.
- Its hash must remain
  `f3d1d17f80742c7114be8984cdfb55c10db117ecaa6e926b3ccc27414834062e`.
- Keep `gpuwm spectral` CPU-only; do not add a CuPy capability requirement.
- Registration must be published before score opens model output.
- Missing/nonfinite/unresolved evidence never becomes pass.
- Do not invent spectral gate thresholds. Calibrate only from a predeclared
  known-good receipt population and preserve its receipt hashes.
- Do not place regridding inside the scorer.

Run at minimum:

```bash
pytest -q tests/test_spectral_compare_v2.py tests/test_spectral_receipt_v2.py
pytest -q tests/test_spectral.py tests/test_chaos_envelope.py
pytest -q tests/test_stage1_manifest.py
pytest -q tests/test_cli.py tests/test_cli_capability_refusals.py
```

Also run `gpuwm spectral pins` and `gpuwm spectral --help` in a genuinely
no-CuPy install, because the publish environment—not a local environment
variable on a CuPy-equipped box—is the capability control.

Use the supplied `demo/` as the end-to-end fixture. Then create one real,
pre-registered matched Arwen/WRF campaign for T, U/V, and W on one domain and
lead. Keep the resulting registration, receipt, plots manifest, source hashes,
and evaluator commit together.

Report back with:

1. exact live base and resulting commit;
2. files changed;
3. every test command and result;
4. no-CuPy CLI proof;
5. synthetic receipt/plot hashes;
6. real campaign registration hash;
7. real campaign receipt hash and interpretation by wavelength band; and
8. any unresolved geometry/field-authority issue without guessing.
