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

`.github/workflows/publish.yml` has two ingresses and one set of proofs.
**Publishing the GitHub release** triggers it and the tag comes from the
release payload; that is the motion every release through v1.4.0 used. A
**manual dispatch** on the exact tag triggers it too, names the tag in the
required `release_tag` input, and expects a draft release that the final
privilege-separated job promotes only after the native assets and the
prevalidated PyPI distributions land. The ingress changes exactly one thing:
the release state each job expects to see. Every byte proof is identical on
both paths. The steps are listed here because a cut driven by hand has to
reproduce them, not because their proofs are optional:

- [ ] **Stage 1 is green at the stamped tip, and it ran BEFORE the tag was
      created.**  Not the stamp guards, not the suites the version bump
      selects: the whole of `tools/battery/stage1_files.txt`, at the commit
      the tag will point at.  This item is first because it is the only one
      whose cost depends on the order: everything below is recoverable at
      any time, and a tag is not.  It has been skipped twice and cost a
      release number both times.  On the 1.8.8 line two imports added to a
      module the RW-WPS wheel stages pointed at modules it does not carry;
      the publish pipeline was the first thing to say so, the tag was spent,
      and the release became 1.8.9.  On 2.3.0 the same class recurred with
      four imports, and that release became 2.3.1.  Both were covered by a
      stage-1 entry that no lane runs.
      `tests/test_native_wrf_distribution.py` has since been promoted to
      `tools/battery/always_files.txt`, so a lane now runs the specific gate
      those two cuts tripped over -- but the promotion closes one hole, not
      the class.  Stage 1 is the list that catches the next one.
- [ ] **The publish workflow's own `test` job list, run with NO GPU extra
      installed, BEFORE the tag.**  A second item about order, for a second
      reason: stage 1 above proves the tip on the box it runs on, and that
      box has CuPy.  `GPUWM_NO_LOCAL_GPU=1` does not uninstall it, so
      `gpuwm.capabilities.is_installed("cupy")` stays true and every refusal
      that fires only where CuPy is ABSENT is invisible locally -- while
      being the state of every CI runner and every user who has not
      installed a GPU extra.  That is how v2.4.0 was tagged and then refused
      by its own `test` job on two rows of `tests/test_cli.py`, and the tag
      was spent.  The reproduction is a venv with `pip install -e ".[dev]"`
      and nothing else, which is exactly what the workflow installs:

          python -m venv <scratch>/venv-ci
          <scratch>/venv-ci/Scripts/python -m pip install -e ".[dev]"
          cd <worktree> && PYTHONNOUSERSITE=1 <scratch>/venv-ci/Scripts/python               -m pytest -q -m "not gpu and not slow and not network" <the list>

      The list is the one in `.github/workflows/publish.yml`; its files are
      on `tools/battery/stage1_files.txt` so a lane runs them too, but the
      list alone does not close this -- running them WITHOUT CuPy is the
      half that catches the class.
- [ ] **The committed public snapshot tree equals the verified snapshot,
      file for file, BEFORE the tag.**  Compare `git ls-tree -r HEAD
      --name-only` in the public repo against `find . -type f` in the
      snapshot directory, both directions, and require an empty diff.
      The breakage this prevents: the public repo's `.gitignore` carries
      `tools/rustwx/vendor/**/target/` for cargo build directories, and
      the vendored cc crate has a SOURCE directory named `cc/src/target/`.
      Its four files are tracked, so the rule is inert -- until an
      assembly that runs `git rm -r .` before re-adding drops them from
      the index, at which point `git add -A` skips them as freshly
      ignorable and the commit silently loses them.  That is how the
      v2.5.5 tag shipped a tree whose `cargo build --release --locked`
      could not verify the vendored cc checksum, the bridges job failed,
      and the number was spent.  Assemble by overlay copy onto the
      existing tree (no `git rm`), or `git add -Af`; either way this
      compare is the gate, because the next ignore rule to swallow a
      source directory will not announce itself either.
- [ ] The tag and `[project].version` in `pyproject.toml` agree
      (`vX.Y.Z` and `X.Y.Z`).  That equality is a refusal: a bundle named
      for one while the wheel says the other is a download nobody can find.
      Whether the version must additionally be a *stable* `X.Y.Z` is the
      operator's call. Leave the optional `stable_release_expected` input
      unticked -- the default -- and a non-stable version publishes with a
      workflow warning naming it; tick it and the cut refuses anything but
      `X.Y.Z`. It reads this way by owner ruling of 2026-08-03: the
      stable-only refusal was added on 2026-08-01 with no ruling behind it,
      and it blocked a motion that had always worked.

      Two more spellings of that same version must agree, and
      `tests/test_companion_distribution.py` is what enforces it:
      `gpuwm-data/gpuwm_data/VERSION`, and the `gpuwm-data==X.Y.Z` pin in
      `[project].dependencies`.  Bump all three together.  They are
      restatements rather than one dynamic lookup because
      `[project].version` has to stay a static literal --
      `gpuwm.provenance` reads it straight out of the file for the
      code-version receipt.
- [ ] The selected workflow ref and the tag name each other, and **exactly
      one** GitHub release carries that tag, in the state the ingress implies
      (a draft for a dispatch, an already-public release for the release
      event). Both are refusals: two releases on one tag means the cut cannot
      know which one it is publishing, and the wrong state means the motion
      is not the one the trigger claimed. A release marked **prerelease**
      warns and continues, and its prerelease state is carried through the
      cut unchanged rather than silently cleared -- owner ruling 2026-08-03,
      the same demotion as above.
- [ ] Optionally, GitHub **release immutability**: enable it in repository or
      organization settings **before the release is created** (the setting
      applies only to future releases) and tick `immutable_releases_enabled`.
      The final job then refuses success unless GitHub reports the published
      release as immutable, which locks its tag and attached assets. Left
      unticked -- the default, and how every release through v1.4.0 shipped
      -- the cut proceeds and reports the release's actual immutable state.
- [ ] The committed tag has `release: null` and `platforms: {}` in
      `gpuwm/data/bridges/bridge-pins.json`. Official wheel/sdist pins are
      generated later from the exact bundles; GitHub's automatic source
      archives stay honestly unpinned and require a source build.
      `tools/verify_source_bridge_pins.py` is a hard gate and stays one by
      owner ruling 2026-08-03: the failure mode is a source archive
      impersonating pinned release bytes.
- [ ] `cargo build --release --locked` in `tools/grib1_bridge`, in
      `tools/rustwx`, in `tools/region_global_dealias` and in
      `tools/rw_wps` (the mapped decode engine, which every mapped
      source runs on by default), once per published platform
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
- [ ] `python tools/build_bridge_bundle.py pin --release <tag>
      --source-rev <tag commit> --bundle <each bundle>
      --out gpuwm/data/bridges/bridge-pins.json
      --manifest dist-bridges/bridge-bundle-manifest.json`, then upload
      the manifest beside the bundles.  `--source-rev` is the full
      40-hex commit the tag peels to: every binary in every bundle
      embeds `GPUWM_BRIDGE_SOURCE_REV=<commit>` at build time, and pin
      refuses a bundle whose stamp is absent or names any other commit.
      This is the check the 1.4.1 preparation lacked when it nearly
      shipped platform zips from two different source revisions -- a
      stale binary hashes perfectly, so only the stamp can catch it.
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
- [ ] **TWO distributions, one commit.**  `gpuwm-data` is built from the
      same tree, at the same version, and uploaded in the same act:

          python -m build --wheel            # gpuwm, per-platform
          python -m build --wheel gpuwm-data # the companion, py3-none-any

      The companion is pure data, so ONE `py3-none-any` wheel serves every
      platform and it is built once per cut, not once per platform.  Build
      it after `rm -rf gpuwm-data/build` for exactly the reason above.

      Neither may be uploaded without the other, and the order is
      companion first.  `gpuwm` pins `gpuwm-data==<this version>`, so a
      `gpuwm` on PyPI whose companion is not there yet is an install that
      resolves to nothing -- every `pip install gpuwm` in the window fails
      at resolution, including the quickstart.  The reverse gap is
      invisible: a companion nobody depends on yet is inert.

      Measure both against the 100 MiB per-file cap before uploading
      anything.  At 2.5.0 they were 39.41 MiB (win_amd64, bridges staged)
      and 64.22 MiB.  The split happened because the single wheel had
      reached 103.62 MiB; if either half approaches the cap again, the
      remedy is another directory into
      `gpuwm.data_assets.COMPANION_TREES`, not a dropped decoder.
- [ ] Confirm the built package reads its own pins: the release matches
      the tag, every supported platform is pinned, and each bundle names
      every artifact in `gpuwm.bridge_assets.BUNDLED_ARTIFACTS`.
      `python tools/verify_release_artifacts.py --dry-run --wheel ... --sdist
      ... --pins ... --manifest ... --bundles ... --release <tag> --source-rev
      <commit> --receipt <path>` runs every document, wheel, sdist, and bundle
      assertion the cut runs, against staged artifacts and before any of it is
      public. It skips only what needs a wheel installed outside the checkout
      and executable target-native binaries, and its receipt names those
      skips, so a dry-run receipt is never mistaken for the cut's.
- [ ] BEFORE THE TAG, run the one skipped leg that can be run locally:
      `python tools/probe_library_abi.py --receipt <path>`, after
      `cargo build --release --locked` in `tools/grib1_bridge` and in
      `tools/region_global_dealias` (both build offline).  It loads every
      `kind == "library"` artifact this host can build and resolves the
      symbol `gpuwm.bridge_assets.LIBRARY_ABI` declares for that artifact,
      refusing a library that declares none.  The dry run cannot do this:
      loading a library needs target-native bytes.  That gap is what cost
      2.1.0 its number -- the verifier's staging leg asked every library
      for `gpuwm_preprocess_cpu_abi_version`, the vendored dealiasing
      cdylib exports `bw_abi_version`, and the refusal surfaced in the
      prepare job with the tag already public.  A full local non-dry-run
      leg is genuinely unavailable pre-tag: that path needs the release
      wheel installed outside the checkout AND a pinned bundle for every
      supported platform, and the other platform's bundle cannot be built
      on this host.  This probe is the narrowest leg that is nonetheless
      real, and it exercises the same table and the same code the cut and
      the workflow runners use.
- [ ] Confirm the built wheel stamps the release version.  Install it
      into a scratch virtual environment and read it back --
      `python -c "import gpuwm; print(gpuwm.__version__)"` -- because
      `gpuwm.__version__` is the installed distribution's metadata, and a
      source tree beside a stale `site-packages` reports the stale
      number.
- [ ] In that same scratch environment, read a moved table back through
      the resolver rather than trusting the file list:

          python -c "import hashlib; from gpuwm import data_assets; \
            p = data_assets.data_path('rrtmgp/rrtmgp-gas-lw-g256.nc'); \
            print(p, hashlib.sha256(p.read_bytes()).hexdigest())"

      It must land in `site-packages/gpuwm_data/data/` and match the
      digest in `tests/test_companion_distribution.py`.  A wheel whose
      file list is right and whose resolver points at the checkout is a
      release that only works on the machine that built it.
- [ ] Reconcile the target PyPI version before upload. Exact existing files
      are retained, only missing wheel/sdist files are staged for Trusted
      Publishing, and any filename/size/SHA-256 mismatch refuses. Prove the
      final PyPI version contains exactly those two artifacts before making
      the GitHub draft public; this makes a one-file partial upload retryable.
      Reconcile the `gpuwm-data` project at the same version the same way,
      and prove it public FIRST -- `gpuwm`'s `==` pin makes the companion a
      precondition of every install, not a follow-up.
- [ ] After the release is public, one live smoke against the published
      URL: `GPUWM_NETWORK_TESTS=1 python -m pytest -q -m network
      tests/test_bridge_fetch.py`.
- [ ] On the dispatch ingress, confirm the workflow's final job, rather than
      an operator racing it, changed the proven draft to a public release. A
      lost PATCH response is retry-safe by rerunning the failed final job in
      the same workflow run: the same already-public release ID/tag is
      success, and a post-PATCH read proves `draft=false` and the captured
      prerelease state (plus `immutable=true` when immutability was opted
      into). Do not *dispatch* a new full run after publication; that ingress
      expects a draft and correctly refuses. On the release-event ingress
      there is nothing to promote -- the release was already public when the
      event fired, and the final job proves it and exits.

A cut that skips the pin step can no longer build a wheel at all:
`setup.py` refuses `bdist_wheel` while
`gpuwm/data/bridges/bridge-pins.json` declares no release and no
platforms (the 2.5.0 candidate shipped exactly that state, and a clean
home answered `pip install gpuwm && gpuwm setup` with FAILED bridges --
no GRIB decoder, no NetCDF decoder, renderer rw_wrfbatch not built).
`tests/test_wheel_pin_gate.py` proves the refusal in every lane and in
stage 1; `GPUWM_ALLOW_UNPINNED_WHEEL=1` is the explicit dev override
for a wheel that never leaves the machine that built it.
