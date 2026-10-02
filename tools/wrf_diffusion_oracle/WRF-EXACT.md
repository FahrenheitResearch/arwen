# Active diffusion verification

The retained fourteen initial/evolved-wind operator fixtures come from unchanged compiled WRF v4.7.1. Their source and build hashes are in `tests/data/wrf471_diffusion/deformation-build-receipt.json` and the horizontal manifest. The two fixture manifests pin every input/reference archive by SHA-256.

Run the complete active km_opt=4 comparison on an authorized GPU:

```sh
GPUWM_WRF_EXACT=1 GPUWM_WRF_EXACT_DIFFUSION=1 python -m pytest -q tests/test_wrf_exact_diffusion.py
```

The test runs the ordinary engine launchers on actual fixture state arrays. It supplies no saved geometry to the CUDA kernel. Its diagnostic entry point only exposes the existing production tensor helpers. It compares every output word, signed zeros included, across all 2,722,902 words. No tolerance accepts a changed result.

The optional mode includes separate verification controls for the reproduced tensor donor and N2 outer-row defects. Those controls are named defects and do not become deliberate physics improvements. Default kernel bodies retain their original behavior. The default-on corrections belong to the diffusion defect patch and require a separate integration step.

The result qualifies the retained active horizontal mixing corpus. It does not qualify km_opt=2/3, the PBL-off vertical/TKE package, an entire forecast or observed forecast skill.
