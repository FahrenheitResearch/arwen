"""Dual-write verification for the wrfout writer flip.

2.5.0 flips :class:`gpuwm.io.wrfout.WrfoutWriter` onto the Rust classic
writer BY DEFAULT.  The verification that justifies the flip is spelled
here, once, and used from two places:

* ``tests/test_wrfout_dual_write.py`` runs it on small fixtures spanning
  the surface (a small-GFS-shaped real case, a 20CRv3-demo-shaped case,
  a nested case) on every test run;
* this file's CLI runs the FULL-SIZE version on real wrfout files once,
  manually, with receipts::

      python tools/wrfout_dual_write.py CASE_NAME WRFOUT [WRFOUT ...] \
          --workdir DIR [--render] [--receipt OUT.json]

The claim being verified, for one identical stream of frames written
through BOTH engines:

1. STRUCTURE: same dimensions (name, size, unlimitedness), same global
   attribute inventory with bitwise-equal values, same variables in the
   same order with the same declared dtype, dimensions and attributes;
2. VALUES: every variable's payload bitwise-equal (NaN payloads and
   signed zeros included) -- read raw, masking and scaling off;
3. THE ARTIFACT: ``rw_wrfbatch`` renders of the two files are
   byte-identical, product by product.

The CONTAINER is expected to differ and is reported, not compared:
the Python engine writes HDF5 ``NETCDF4_CLASSIC``, the Rust engine
writes CDF-2 ``NETCDF3_64BIT_OFFSET``.  Everything a reader can ask of
the file must be equal; the envelope is the one thing the flip changes.

Replay mode (the CLI) takes an EXISTING wrfout, reads its frames and
global attributes raw, and writes them through ``WrfoutWriter`` twice --
``engine="python"`` and ``engine="rust"`` -- so both engines receive
byte-identical inputs regardless of GPU determinism.
"""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import netCDF4

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gpuwm.io.wrfout import WrfoutWriter  # noqa: E402

#: Dimensions the writer owns; everything else in a source file is
#: carried by its variables and re-derived from their shapes.
_WRITER_DIMS = ("Time", "DateStrLen", "west_east", "south_north",
                "bottom_top", "west_east_stag", "south_north_stag",
                "bottom_top_stag", "soil_layers_stag")


def _raw(variable):
    variable.set_auto_maskandscale(False)
    if hasattr(variable, "set_auto_chartostring"):
        variable.set_auto_chartostring(False)
    return variable


def semantic_diff(path_a, path_b, *, max_reported: int = 25) -> list[str]:
    """Every semantic difference between two wrfout files, named.

    Empty list = the files are identical in structure and values.  The
    container format itself is deliberately not compared.  Global
    attributes are compared as an inventory (name set) with bitwise
    values; variables are compared in ORDER, because creation order is
    part of what both engines promise to preserve.
    """
    problems: list[str] = []

    def note(text: str) -> None:
        if len(problems) < max_reported:
            problems.append(text)
        elif len(problems) == max_reported:
            problems.append("... further differences suppressed")

    with netCDF4.Dataset(path_a) as a, netCDF4.Dataset(path_b) as b:
        dims_a = {name: (len(dim), dim.isunlimited())
                  for name, dim in a.dimensions.items()}
        dims_b = {name: (len(dim), dim.isunlimited())
                  for name, dim in b.dimensions.items()}
        if dims_a != dims_b:
            note(f"dimensions differ: {dims_a} != {dims_b}")

        names_a, names_b = set(a.ncattrs()), set(b.ncattrs())
        if names_a != names_b:
            note("global attribute inventories differ: "
                 f"only-a={sorted(names_a - names_b)}, "
                 f"only-b={sorted(names_b - names_a)}")
        for name in sorted(names_a & names_b):
            va, vb = a.getncattr(name), b.getncattr(name)
            if isinstance(va, str) or isinstance(vb, str):
                if va != vb:
                    note(f"global attr {name}: {va!r} != {vb!r}")
                continue
            va, vb = np.asarray(va), np.asarray(vb)
            if va.dtype != vb.dtype or va.shape != vb.shape \
                    or va.tobytes() != vb.tobytes():
                note(f"global attr {name}: {va.dtype}{va!r} != "
                     f"{vb.dtype}{vb!r}")

        order_a, order_b = list(a.variables), list(b.variables)
        if order_a != order_b:
            note(f"variable inventory/order differs: {order_a} != {order_b}")
        for name in [v for v in order_a if v in b.variables]:
            var_a, var_b = _raw(a.variables[name]), _raw(b.variables[name])
            if var_a.dtype != var_b.dtype:
                note(f"{name}: dtype {var_a.dtype} != {var_b.dtype}")
                continue
            if var_a.dimensions != var_b.dimensions:
                note(f"{name}: dimensions {var_a.dimensions} != "
                     f"{var_b.dimensions}")
                continue
            attrs_a = {k: var_a.getncattr(k) for k in var_a.ncattrs()}
            attrs_b = {k: var_b.getncattr(k) for k in var_b.ncattrs()}
            if set(attrs_a) != set(attrs_b):
                note(f"{name}: attribute inventories differ: "
                     f"{sorted(attrs_a)} != {sorted(attrs_b)}")
            for key in sorted(set(attrs_a) & set(attrs_b)):
                aa, bb = attrs_a[key], attrs_b[key]
                same = (aa == bb if isinstance(aa, str) or isinstance(bb, str)
                        else (np.asarray(aa).dtype == np.asarray(bb).dtype
                              and np.asarray(aa).tobytes()
                              == np.asarray(bb).tobytes()))
                if not same:
                    note(f"{name}.{key}: {aa!r} != {bb!r}")
            data_a = np.asarray(var_a[:])
            data_b = np.asarray(var_b[:])
            if data_a.shape != data_b.shape:
                note(f"{name}: data shape {data_a.shape} != {data_b.shape}")
            elif data_a.tobytes() != data_b.tobytes():
                both = np.isfinite(data_a) & np.isfinite(data_b) \
                    if data_a.dtype.kind == "f" else np.ones_like(
                        data_a, dtype=bool)
                worst = float(np.max(np.abs(
                    data_a[both].astype("f8") - data_b[both].astype("f8")))) \
                    if data_a.dtype.kind in "fiu" and both.any() else float("nan")
                note(f"{name}: payload differs bitwise "
                     f"(max |a-b| over mutually finite: {worst})")
    return problems


def replay(source, out_path, *, engine: str) -> Path:
    """Rewrite ``source``'s frames through ``WrfoutWriter`` on ``engine``.

    Both engines are handed byte-identical inputs: the source's raw
    per-record payloads, its global attributes verbatim, and its own
    geometry.  What the writer stamps on its own (variable metadata,
    XTIME semantics, the completion attribute) it stamps for both.
    """
    source = Path(source)
    out_path = Path(out_path)
    with netCDF4.Dataset(source) as ds:
        nx = len(ds.dimensions["west_east"])
        ny = len(ds.dimensions["south_north"])
        nz = len(ds.dimensions["bottom_top"])
        soil = (len(ds.dimensions["soil_layers_stag"])
                if "soil_layers_stag" in ds.dimensions else None)
        global_attrs = {name: ds.getncattr(name) for name in ds.ncattrs()}
        times_var = _raw(ds.variables["Times"])
        stamps = [row.tobytes().decode("ascii").rstrip("\x00")
                  for row in times_var[:]]
        data_names = [name for name in ds.variables if name != "Times"]
        frames = []
        for t in range(len(stamps)):
            frames.append({
                name: np.asarray(_raw(ds.variables[name])[t])
                for name in data_names})
    writer = WrfoutWriter(
        out_path, nx=nx, ny=ny, nz=nz,
        dx=float(global_attrs.get("DX", 1000.0)),
        dy=float(global_attrs.get("DY", 1000.0)),
        title=str(global_attrs.get("TITLE", "gpuwm")),
        global_attrs=global_attrs, soil_layers=soil,
        field_schema=frames[0], engine=engine)
    try:
        for stamp, frame in zip(stamps, frames):
            writer.write_frame(stamp, frame)
    except BaseException:
        writer.abort()
        raise
    writer.close()
    return out_path


def render_products(wrfout, out_root, *, width: int = 1200,
                    height: int = 900) -> dict[str, Path]:
    """Render every renderable product ``rw_wrfbatch`` offers for ``wrfout``."""
    from gpuwm import rustwx

    renderer = rustwx.find_renderer()
    if renderer is None:
        raise SystemExit(
            "rw_wrfbatch is not resolvable; build it or set "
            f"{rustwx.RENDERER_ENV}")
    out_root = Path(out_root)
    store_root = out_root / "store"
    out_dir = out_root / "png"
    store_root.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, _summary = rustwx.list_products(
        renderer, Path(wrfout), store_root=store_root)
    slugs = [slug for slug, _kind, status, _detail in rows
             if status == "renderable"]
    if not slugs:
        raise SystemExit(f"{wrfout}: no renderable products in the catalog")
    _written, failures, _skipped = rustwx.run_renderer(
        renderer, Path(wrfout), store_root=store_root, out_dir=out_dir,
        products=",".join(slugs), frames="all", width=width, height=height)
    if failures:
        raise SystemExit(f"rw_wrfbatch failed on {wrfout}: {failures}")
    return {path.relative_to(out_dir).as_posix(): path
            for path in sorted(out_dir.rglob("*.png"))}


def compare_renders(products_a, products_b) -> tuple[int, list[str]]:
    problems = []
    if set(products_a) != set(products_b):
        problems.append(
            f"product sets differ: only-a="
            f"{sorted(set(products_a) - set(products_b))}, only-b="
            f"{sorted(set(products_b) - set(products_a))}")
    matched = sorted(set(products_a) & set(products_b))
    for key in matched:
        if not filecmp.cmp(products_a[key], products_b[key], shallow=False):
            problems.append(f"render differs: {key}")
    return len(matched), problems


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def dual_write_case(case: str, sources, workdir, *, render: bool) -> dict:
    """Replay every source through both engines; compare; optionally render."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    receipt = {"schema": "gpuwm-wrfout-dual-write-receipt-v1",
               "case": case, "files": [], "ok": True}
    for source in sources:
        source = Path(source)
        stem = source.name
        py_path = workdir / f"{stem}.python.nc"
        rs_path = workdir / f"{stem}.rust.nc"
        replay(source, py_path, engine="python")
        replay(source, rs_path, engine="rust")
        problems = semantic_diff(py_path, rs_path)
        with netCDF4.Dataset(py_path) as ds:
            fmt_py, nvars = ds.file_format, len(ds.variables)
        with netCDF4.Dataset(rs_path) as ds:
            fmt_rs = ds.file_format
        entry = {
            "source": str(source),
            "python": {"path": str(py_path), "format": fmt_py,
                       "bytes": py_path.stat().st_size,
                       "sha256": _sha256(py_path)},
            "rust": {"path": str(rs_path), "format": fmt_rs,
                     "bytes": rs_path.stat().st_size,
                     "sha256": _sha256(rs_path)},
            "variables": nvars,
            "semantic_differences": problems,
        }
        if render:
            products_py = render_products(py_path, workdir / f"{stem}.render.python")
            products_rs = render_products(rs_path, workdir / f"{stem}.render.rust")
            matched, render_problems = compare_renders(products_py, products_rs)
            entry["renders_compared"] = matched
            entry["render_differences"] = render_problems
            if render_problems:
                receipt["ok"] = False
        if problems:
            receipt["ok"] = False
        receipt["files"].append(entry)
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case", help="case label for the receipt")
    parser.add_argument("wrfouts", nargs="+", type=Path)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--render", action="store_true",
                        help="also render both files with rw_wrfbatch and "
                             "compare the PNGs byte for byte")
    parser.add_argument("--receipt", type=Path, default=None)
    args = parser.parse_args(argv)

    receipt = dual_write_case(args.case, args.wrfouts, args.workdir,
                              render=args.render)
    rendered = json.dumps(receipt, indent=1)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
