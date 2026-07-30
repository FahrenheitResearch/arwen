# Morrison two-moment provenance

The constants in `constants.toml` and the equations in
`gpuwm/core/kernels/morrison.cu` / `gpuwm/verify/npref.py` were transcribed
from the local requirements-authority source:

- WRF release: v4.6.1
- upstream repository: https://github.com/wrf-model/WRF
- source commit: `d66e442fccc04111067e29274c9f9eaccc3cef28`
- source file: `phys/module_mp_morr_two_moment.F`
- source-file SHA-256:
  `d4313264b63dc1e1703555f9f6b357fc73e1325ed9dea9ffbf7f521b91a14916`
- canonical source path: `phys/module_mp_morr_two_moment.F`

Machine-local staging paths are intentionally excluded from this
distributable authority.

Relevant source ranges are constants and switches at lines 247-546, the
WRF wrapper at 563-925, the column algorithm at 929-4062, and the saturation
vapor-pressure function at 4066-4149.  No runtime lookup table or downloaded
binary data is needed by this scheme.  `LICENSE.txt` is copied from the WRF
v4.6.1 source root (with repository CRLF normalized to LF by Git) as required
by the Phase 4 data-provenance rule.
