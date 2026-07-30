# RW-WPS release checklist

No artifact may be called a community release until every blocking item is
complete and the machine-readable support matrix matches the documentation.

- [x] Owner selected and approved Apache-2.0 for the project.
- [ ] RW-WPS has a model-independent Python package boundary.
- [ ] Redistributable fixtures replace private/local case dependencies.
- [ ] Clean Linux x86-64 CPU install and end-to-end source gate pass.
- [ ] Clean Linux x86-64 CUDA 12.x install and end-to-end source gate pass.
- [ ] Second-machine archive extraction reproduces integrity/runtime checks.
- [ ] Current mapping/composition hashes own current stock-WRF gates.
- [ ] Sdist, wheel, runtime archive, dependency lock, SBOM, checksums, and
      signatures are reproducible and retained.
- [ ] Wheel/archive scans contain no credentials, node addresses, personal
      paths, private evidence, or unreviewed redistributable data.
- [ ] Threat review covers malformed inputs, resource bounds, path traversal,
      symlinks, atomic publication, cancellation, and no-clobber behavior.
- [ ] README, install, CLI, examples, compatibility matrix, changelog,
      contribution, security, and release notes agree.
- [ ] Every claimed source/domain/vertical/physics/backend cell names its exact
      evidence receipt and unchanged-WRF result where applicable.

`PUBLIC_RELEASE_ACCEPTANCE.md` is the detailed acceptance authority.
