# Where rendered products land

`gpuwm render` files every picture it draws at a path you can compute
before it exists:

```
<--out>/<run folder>/<domain>/<product>/<valid-day>/<filename>.png
```

This is the default. There is no flag to turn it on.

The `<run folder>` level is one render invocation's own timestamped
directory (`run-20260817-041233Z_i202605171800Z`), so two renders into
one `--out` never overwrite each other. It is documented on its own page
— [run-output-folders.md](run-output-folders.md) — and everything below
is about the three segments underneath it, which are unchanged. The
examples in this page are written relative to the run folder.

```
out/myarea/png/
  d02-3km/
    composite_reflectivity/
      1974-04-03/  arwen_wrf_19740403_22z_f000.png
                   arwen_wrf_19740403_22z_f001.png
      1974-04-04/  arwen_wrf_19740403_22z_f002.png
    2m_temperature/
      1974-04-03/  ...
  d04-100m/
    composite_reflectivity/
      ...
```

## The three segments

| segment | what it is | values |
| --- | --- | --- |
| `<domain>` | the nest, with its grid spacing | `d02-3km`, `d04-100m`, `d05-111m`; `native_grid` when the input file proves no domain identity (no `GRID_ID`, no `wrfout_dNN` name) |
| `<product>` | the chart | rust engine (the default, and the only one `--engine auto` selects): the catalog slug, e.g. `composite_reflectivity`, `2m_temperature`, `total_qpf`, `sbcape`. The `--engine matplotlib` workaround, asked for by name: `refl`, `t2`, `wind10`, `precip`, `olr` |
| `<valid-day>` | `YYYY-MM-DD` of the time the frame is VALID | `1974-04-04`; `undated` when the file carries no readable valid time |

`--out` itself is the case folder. Every front door already sets it per
case (`gpuwm go` uses `<case>/png`), so nothing in the renderer invents
a case name.

Two details worth knowing before you write a path by hand:

* The day is the day the frame is **valid**, not the day its run was
  initialised. A 22Z cycle at f+02 files under the next morning. This is
  why the example above has a `1974-04-04` folder under a run that
  started on the 3rd.
## The filename carries what the folders do not

A delivered filename is

```
arwen_<model>_<YYYYMMDD>_<H>z_f<NNN>[_valid_<...>_lead_<...>].png
```

— the frame's own identity: model, cycle date, cycle hour, forecast
hour, and, for a sub-hourly frame, the exact valid time and lead that
`f<NNN>` cannot express. The domain and the product are **not** in it,
because the two folders directly above it are exactly those, and a name
that repeated them cost real deliveries the Windows path ceiling: a
measured tree reached 310 characters, at which point Explorer, `tar`,
and the readers a recipient's own script imports all refuse to open a
picture that is otherwise filed correctly.

Nothing is lost. Folder plus filename together carry the same facts the
v2.4.1 name did, so frames stay distinct across members, valid times and
sub-hourly cadences; `gpuwm.render_layout.engine_name(name, domain=...,
product=...)` rebuilds the v2.4.1 filename exactly if a consumer of
yours wants it. `--layout flat` — where there are no folders to carry
anything — writes the v2.4.1 name byte for byte.

* The `<domain>` folder is spelled exactly as the domain token the
  frames in it were rendered for, so the folder and the files agree.

## Predicting a path from a script

Given a domain token, a product slug and a valid time you can build the
directory with no globbing and no directory listing:

```python
from pathlib import Path

def product_dir(out, domain, product, valid_time):
    """Where `gpuwm render` will put this product's frames."""
    return Path(out) / domain / product / valid_time.strftime("%Y-%m-%d")
```

That is the whole contract. `gpuwm.render_layout` is the in-tree
implementation of it (`place`, `product_dir`, `valid_day`) if you would
rather import than transcribe, and `gpuwm render` prints the layout it
is about to write **before** it draws anything:

```
render: engine rust (.../rw_wrfbatch.exe)
render: run folder run-20260817-041233Z_i202605171800Z under out/myarea/png
render: layout nested -- out/myarea/png/run-20260817-041233Z_i202605171800Z/<domain>/<product>/<valid-day>/<file>.png (domain as d02-3km / d05-111m / native_grid, valid-day as YYYY-MM-DD)
```

To watch for one frame becoming readable, watch that one path. A
picture appears at its final name atomically: both engines write
elsewhere and rename onto the published name, so a reader sees a
complete PNG or no PNG, never a partial one.

## Reading a whole render directory

Recurse; do not glob one level. In the shell:

```sh
find out/myarea/png -name '*.png'          # POSIX
Get-ChildItem out/myarea/png -Recurse -Filter *.png   # PowerShell
```

Skip dot-prefixed directories while a run is still going. `gpuwm go`'s
early render works in `.first-products-scratch` beside the pictures and
publishes out of it; files under a dot-directory are not published yet.
In Python, `gpuwm.render_layout.iter_rendered` does both of these and is
what every in-tree reader uses.

### On Windows: read long paths

Three folders plus a filename is around seventy characters, so a deep
enough case root still passes the classic 260-character ceiling. ArWen
writes those frames into the layout anyway rather than dropping them
flat at the root, using the extended-length spelling
(`gpuwm.render_layout.fs_path`); `iter_rendered` reads them back.

A reader of your own needs the same. In Python, open
`render_layout.fs_path(path)` rather than `path`. In PowerShell, prefix
the absolute path with `\\?\`. Or enable long paths once for the whole
machine (`HKLM\SYSTEM\CurrentControlSet\Control\FileSystem`,
`LongPathsEnabled = 1`) and nothing needs the prefix.

## `--layout flat`, and why you probably do not want it

```sh
gpuwm render out/myarea/wrfout_d0* --out out/myarea/png --layout flat
```

writes every picture directly into the run folder, which is what
releases up to v2.4.1 did inside `--out`. It exists for one reason: a
consumer written against the old single directory needs somewhere to
stand while it is updated. It is a compatibility escape hatch, not a
supported alternative, and it is what produced the report this layout
answers -- a real run leaves thousands of frames of every product and
every valid time in one folder, and the only way to find one is to read
filenames.

Adding `--run-stamp off` drops the run folder too, which is the v2.4.1
directory byte for byte.

The two layouts hold the same pictures under the same filenames. On a
two-nest, three-product, three-frame render measured with the real rust
engine:

| | PNGs | directories | busiest directory |
| --- | --- | --- | --- |
| `--layout flat` | 18 | 1 | 18 files |
| `--layout nested` (default) | 18 | 12 | 2 files |

Byte-identical output, both ways.

## What already understands the layout

* `gpuwm render --pair A_DIR B_DIR` reads both directories recursively
  and pairs on the filename, so a nested directory pairs against a flat
  one -- which is the case during a migration.
* `gpuwm go`'s early render publishes into the same tree its finalize
  stage renders into, and its `first-products.json` receipt names each
  picture by its path **relative to the render directory** (posix
  spelling), so `render_dir / entry["name"]` finds it on any platform.

## The delivered tree holds products only

The Rust renderer needs a working store for the intermediate hour files
it builds before it draws.  That store is scratch, and it does **not**
live in the tree you are delivered: it lives in a sibling directory
named after the render directory with `.render-scratch` appended
(`png.render-scratch/rwstore-xxxxxxxx`), created for one renderer
invocation and removed when it finishes.

It matters because the removal is best-effort by nature -- a
memory-mapped hour file can hold its handle past the renderer's exit --
and because scratch is present *during* the render whether or not it is
removed cleanly afterwards.  Both showed up on real deliveries:
leftover scratch directories sitting among the products, with paths long
enough to break a Windows directory listing, and a `tar` of a tree being
rendered into dying with `File removed before we read it`.  Copy, tar,
sync or scan a render directory at any moment and you get pictures.
