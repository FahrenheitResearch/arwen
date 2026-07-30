# Design note: platform wheels that carry the Rust artifacts

**Status: a design, not an implementation.**  Nothing described here is
built.  `gpuwm fetch-bridges` (shipped) closes the same gap with one
command after `pip install`; this note is about closing it with *no*
command, and about what that would cost.

The wish it answers, in the product owner's words: "i would really
prefer if just `pip install gpuwm` was enough to have everything."

## What stands between here and there

A wheel is a zip that pip unpacks into `site-packages`.  It can carry
any file, including an executable -- the thing it cannot carry is a file
that means different bytes on different machines.  gpuwm needs eight
such files: five GRIB decoders, the CPU preprocessing library, the
`rw_fetch` backbone and the `rw_wrfbatch` renderer.  A single wheel
cannot hold the Windows and Linux builds of all eight and still be
honest about what it is, which is why the current wheel holds neither.

The packaging answer is a *platform wheel*: one artifact per platform,
tagged so pip installs the right one and refuses the wrong one.  The
work is not in the idea, it is in the four consequences below.

## 1. Size, and what PyPI allows

Measured on this tree (2026-07-30), Windows x86-64, the artifacts a
release build produces:

| artifact | bytes |
|---|---|
| `grib1_bridge.exe` | 245,760 |
| `gfs_grib2_bridge.exe` | 385,024 |
| `hrrr_grib2_bridge.exe` | 439,296 |
| `grib2_inventory.exe` | 279,552 |
| `grib2_dump.exe` | 284,160 |
| `gpuwm_preprocess_cpu.dll` | 169,472 |
| `rw_fetch.exe` | 2,867,712 |
| `rw_wrfbatch.exe` | 8,807,424 |
| **total, uncompressed** | **13,478,400 (12.9 MiB)** |
| **the same eight, deflated into one zip** | **5,321,160 (5.1 MiB)** |

The Linux x86-64 set has never been built on this machine, so its sizes
are unmeasured; the crates are the same and the artifacts should be
comparable, but that sentence is an expectation, not a measurement.

Against the wheel: the v1.0.0 wheel was 77,295,377 B (73.7 MiB), and
the package data behind it is 110.5 MiB uncompressed today, so the
artifacts add roughly **5 MiB** to a platform wheel -- about 7%.  The
relevant limits:

- **100 MB per file.**  A platform wheel lands near 79 MiB.  That fits,
  but the headroom drops from about 26 MB to about 21 MB, and it is the
  same headroom the two externalized Thompson tables were pushed out of
  the wheel to create.  Anything that grows the packaged data by 20 MB
  puts platform wheels over the cap before it puts the current wheel
  over it.
- **10 GB per project, by default.**  Each release would publish an
  sdist plus one wheel per platform instead of one wheel: roughly
  240 MB per release at two platforms, against about 80 MB today.  That
  is on the order of 40 releases inside the default quota rather than
  120.  A quota increase is a request form, not a blocker, but it is a
  thing someone has to do before the quota bites, not after.

## 2. It must not become *harder* to install anywhere

If a release publishes only `win_amd64` and `manylinux_x_y_x86_64`
wheels, then on macOS, on aarch64 Linux, and on any platform not in the
matrix, `pip install gpuwm` fails outright with "no matching
distribution" -- a strictly worse outcome than today, where the install
succeeds and `gpuwm doctor` explains what is missing.

So a platform-wheel release publishes **three or more** artifacts: the
platform wheels *and* the existing `py3-none-any` wheel.  pip prefers
the most specific match, so a Windows user gets the platform wheel and
everyone else gets the pure-Python one plus `gpuwm fetch-bridges` or a
source build.  This also keeps the sdist honest: an sdist carries no
compiled artifacts by definition, so `pip install --no-binary :all:`
lands in exactly today's world and must keep working.

**`gpuwm fetch-bridges` is therefore not replaced by this design.**  It
remains the route for every platform without a wheel, for sdist
installs, for air-gapped staging (`--from DIR`), and for repairing a
corrupted or stale staged copy.  Whatever is built here, that command
stays.

## 3. Where the files go, and how resolution stays one mechanism

`gpuwm.bridges.artifact_candidates` is the single resolution ladder for
every built artifact.  Today it is: the per-artifact environment
variable, a source checkout's `target/{release,debug}`,
`<site-packages>/libexec/bridges`, then `~/.gpuwm/bridges`.

A platform wheel needs one new rung, and it must be *inside* the
package: `gpuwm/libexec/`.  The existing `libexec` rung sits beside the
package (`<site-packages>/libexec/bridges`), which is the sealed
runtime archive's layout; a wheel cannot write there without shipping a
top-level `libexec` directory into `site-packages`, which is exactly
the kind of namespace pollution a library must not do.

The proposed order, and the argument for each position:

1. the per-artifact environment variable -- unchanged, still wins, still
   a hard error when it names a missing file;
2. a source checkout's own build -- unchanged: a developer who just ran
   `cargo build` means that binary;
3. **`gpuwm/libexec/` inside the installed package** (new) -- versioned
   with the wheel, so it always speaks the contract this Python half
   expects;
4. `<site-packages>/libexec/bridges` -- unchanged;
5. `~/.gpuwm/bridges` -- unchanged, but now *below* the packaged copy.

Position 3 above position 5 is the load-bearing choice.  The staged
directory is not versioned: a user who ran `gpuwm fetch-bridges` at
1.1.3 and upgraded to 1.2.0 has 1.1.3 binaries sitting in it, and that
skew is precisely the failure this project has already been bitten by
(the 1.1.0 GFS series-file incident).  A wheel that carries its own
matching artifacts should use them, and the environment variable
remains for anyone who means something else.

`gpuwm doctor` needs no new concept: it probe-executes whatever the
ladder resolved and checks the contract marker, wherever the file came
from.  It should gain one honest line -- that the artifact came from
the wheel rather than from a staging directory -- because "where did
this binary come from" is the first question when one misbehaves.  The
packaged pins keep their value too: doctor could verify the
wheel-carried artifacts against them and catch a truncated install,
which is a check no execution probe performs.

## 4. Building them: what the tools actually offer

**maturin** builds a wheel from a Rust crate, and is the right tool when
the Rust *is* the Python extension (PyO3/cffi).  It is the wrong tool
here: gpuwm is a large setuptools project whose Rust half is two
separate cargo workspaces producing seven standalone executables and one
cdylib loaded through `ctypes` -- no Python bindings anywhere.  Adopting
maturin would mean restructuring the build backend of the whole project
around a crate that is not the project.

**cibuildwheel** exists to build the *CPython-ABI* matrix -- one wheel
per (Python version, platform) -- inside manylinux containers, and to
run `auditwheel`/`delocate` on the result.  gpuwm's wheel has no Python
extension module: it is ABI-independent, so the entire per-Python-version
axis of that matrix is wasted work, and `auditwheel` has no extension
module to inspect.  What is left of cibuildwheel's value is the
manylinux container itself, which a single matrix job can use directly.

**The route that fits** is the one the release workflow is already
shaped like: the per-platform build jobs that produce the bundles today
would additionally copy their artifacts into `gpuwm/libexec/` and build
a wheel with a forced platform tag (setuptools takes `--plat-name`;
a pure-Python project needs the standard `has_ext_modules` shim to stop
`bdist_wheel` tagging it `py3-none-any`).  The Linux job would run in a
manylinux container so the glibc the binaries link against matches the
tag.

That last point is the real risk and deserves to be stated plainly:
**with no extension module, `auditwheel` cannot verify the tag.**  The
manylinux tag on such a wheel is an assertion by whoever wrote the
workflow, and a wrong one produces a wheel that installs cleanly and
fails at the first decode with a glibc symbol error -- on the user's
machine, at the worst moment.  Two ways to make the assertion true
rather than hopeful: build the Linux artifacts against
`x86_64-unknown-linux-musl` for a fully static binary (then the tag is
conservative by construction), or build inside the manylinux image whose
glibc the tag names and check the artifacts' `NEEDED`/version symbols in
CI.  Either is a decision to take deliberately; neither is free.

## 5. Migration path

Each step is shippable on its own and none of them removes a route.

1. **Shipped now.**  `gpuwm fetch-bridges`, SHA-256-pinned release
   bundles, one command after `pip install`.  The pins, the bundle
   format, the per-platform build jobs and the release tooling are all
   reusable by every step below -- a platform wheel would carry the same
   artifacts, verified against the same pins.
2. **The ladder rung.**  Add `gpuwm/libexec/` to
   `artifact_candidates`, with the ordering argued above, and teach
   doctor to name which rung answered.  This is a small, testable change
   that does nothing visible until step 3 -- and it can be tested by
   staging a bundle into that directory by hand.
3. **One platform wheel, as an experiment.**  Publish `win_amd64`
   alongside the existing pure-Python wheel for one release, and measure:
   the artifact size on PyPI, the install on a clean machine, whether
   `gpuwm doctor` reports zero Rust gaps straight out of
   `pip install gpuwm`.  Windows first because it is the platform where
   "install a Rust toolchain" costs a user the most.
4. **Linux, once the tag question is answered.**  Only after the
   musl-versus-manylinux decision of section 4 is made and checked in
   CI.
5. **Never remove the fallback.**  `gpuwm fetch-bridges` and the
   clone-and-build remedy both stay, forever, for the platforms no wheel
   covers.

## What would make this not worth doing

Three things, any one of which is a reason to stop at what is shipped:

- **The quota.**  If release cadence stays high, tripling per-release
  storage buys a saved command at the price of a conversation with PyPI
  every year.
- **The tag assertion.**  If nobody wants to own the glibc claim in
  section 4, a Linux platform wheel is a promise the project cannot
  check, and a wrong promise is worse than an extra command.
- **The command is already small.**  `gpuwm fetch-bridges` is one line,
  it is verified, it is re-runnable, and `gpuwm doctor` prints it.  The
  distance between "one command" and "no command" is real but it is
  smaller than the distance this project just closed, which was between
  "clone a repository and install a compiler" and "one command".
