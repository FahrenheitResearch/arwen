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

## Per-cut: the assets a wheel's pins point at

Two classes of file are published as release assets rather than shipped
in the wheel, and both are verified against pins the wheel carries.  The
externalized physics tables (`gpuwm fetch-tables`) are the same bytes at
every release, so their pins live in `gpuwm/core/thompson_contract.py`
and change only when a table does.  The prebuilt Rust bundles
(`gpuwm fetch-bridges`) are **built from the commit being released**, so
their pins are generated during the cut and must land in the package
*before* the wheel is built.

`.github/workflows/publish.yml` does this from one manual dispatch on the
exact tag while its GitHub release is still a draft. The final privilege-
separated job publishes the draft only after the immutable native assets and
the prevalidated PyPI distributions land. Making a release public is not a
second workflow trigger. The steps are listed here because a cut driven by
hand has to reproduce them, not because their proofs are optional:

- [ ] The tag and `[project].version` in `pyproject.toml` agree
      (`vX.Y.Z` and `X.Y.Z`).  The workflow refuses the cut otherwise: a
      bundle named for one while the wheel says the other is a download
      nobody can find. This public workflow accepts stable numeric `X.Y.Z`
      versions only; prerelease cuts require a separately defined contract.
- [ ] The selected workflow ref and required `release_tag` input name that
      same tag, and a non-prerelease draft GitHub release already exists for
      it. Stable PyPI versions are never promoted as GitHub prereleases.
- [ ] GitHub **release immutability** is enabled in repository or organization
      settings **before the draft release is created** (the setting applies
      only to future releases), and the required confirmation input is true.
      The final job refuses success unless GitHub reports the published
      release as immutable, which locks its tag and attached assets.
- [ ] The committed tag has `release: null` and `platforms: {}` in
      `gpuwm/data/bridges/bridge-pins.json`. Official wheel/sdist pins are
      generated later from the exact bundles; GitHub's automatic source
      archives stay honestly unpinned and require a source build.
- [ ] `cargo build --release --locked` in `tools/grib1_bridge` and in
      `tools/rustwx`, once per published platform
      (`gpuwm.bridge_assets.SUPPORTED_PLATFORMS`).
- [ ] `python tools/build_bridge_bundle.py pack --release <tag>
      --platform <platform> --search <each target/release> --out
      dist-bridges` on each of those platforms.
- [ ] Upload every bundle to the release **before** the wheel is
      published, so no published wheel names an asset that is not there.
      Every job must preserve the captured draft release ID as well as its
      tag. A retry deletes only an interrupted GitHub `starter` asset, accepts
      only an uploaded asset whose size and SHA-256 match, and refuses changed
      or unexpected bytes.
- [ ] `python tools/build_bridge_bundle.py pin --release <tag> --bundle
      <each bundle> --out gpuwm/data/bridges/bridge-pins.json
      --manifest dist-bridges/bridge-bundle-manifest.json`, then upload
      the manifest beside the bundles.
- [ ] Build the wheel/sdist *after* that write, from a tree with no
      `build/` directory left over from an earlier build.  `bdist_wheel`
      zips whatever is in `build/lib`, and `build_py` adds to that
      directory rather than reconciling it, so a `build/lib` populated
      before an entry was added to `[tool.setuptools.exclude-package-data]`
      keeps shipping that entry with no line in the log saying it was
      copied from the source tree.  With the two externalized Thompson
      tables that is a 328 MB wheel PyPI would refuse, built from a tree
      whose exclusions are correct.  A fresh checkout (what the workflow
      builds from) or `git archive HEAD` into an empty directory both
      satisfy this; check the wheel's size against the last release's
      before uploading anything.
- [ ] Confirm the built package reads its own pins: the release matches
      the tag, every supported platform is pinned, and each bundle names
      every artifact in `gpuwm.bridge_assets.BUNDLED_ARTIFACTS`.
- [ ] Confirm the built wheel stamps the release version.  Install it
      into a scratch virtual environment and read it back --
      `python -c "import gpuwm; print(gpuwm.__version__)"` -- because
      `gpuwm.__version__` is the installed distribution's metadata, and a
      source tree beside a stale `site-packages` reports the stale
      number.
- [ ] Reconcile the target PyPI version before upload. Exact existing files
      are retained, only missing wheel/sdist files are staged for Trusted
      Publishing, and any filename/size/SHA-256 mismatch refuses. Prove the
      final PyPI version contains exactly those two artifacts before making
      the GitHub draft public; this makes a one-file partial upload retryable.
- [ ] After the release is public, one live smoke against the published
      URL: `GPUWM_NETWORK_TESTS=1 python -m pytest -q -m network
      tests/test_bridge_fetch.py`.
- [ ] Confirm the workflow's final job, rather than an operator racing it,
      changed the proven draft to a public release. A lost PATCH response is
      retry-safe by rerunning the failed final job in the same workflow run:
      the same already-public release ID/tag is success, and a post-PATCH read
      proves `draft=false` and `immutable=true`. Do not dispatch a new full run
      after publication; its draft-authority gate correctly refuses.

A cut that skips the pin step ships a wheel whose pins declare no
platform.  That is not a corrupt release -- `gpuwm fetch-bridges` says
so and `gpuwm doctor` falls back to the build-from-source remedy -- but
it is a release that did not deliver what it built.
