"""Read-only plot metadata for the terminal picker; no forecast defaults change.

This module owns two small tables and one question.

The tables are the curated plot PRESETS (``plot-presets.json``) and the
packaged record of what the wrfout history lane cannot draw and why
(``research-diagnostics.json``).  The question is the one a reader has
before a forecast rather than after it: *of the products I just chose,
which will this install actually draw?*

The record is deliberately NOT a second catalog.  The renderer owns the
product vocabulary and answers availability per store; the packaged JSON
carries only two things the renderer cannot state on its own -- the
REASON a product is unserved on this lane, written for a reader, and the
WINDOW a product needs before it means anything.  A row's absence from
it therefore says nothing at all, and nothing here refuses on absence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

#: The packaged lane record.  One file, read by both doors that ask this
#: question -- the research recipe validator and the preset picker -- so
#: the two cannot disagree about one configuration.
#:
#: NAMING DEBT, recorded rather than paid: the file is still called
#: ``research-diagnostics.json`` and is no longer research-specific.
#: Renaming it would move the sha256 authority block it carries, its
#: ``MANIFEST.in`` row and ``tools/build_rw_wps_release.py``, so the name
#: stays and this note says why.
DIAGNOSTICS_PATH = (Path(__file__).parent / "data" / "tui" /
                    "research-diagnostics.json")


def presets() -> dict:
    """Small curated requests, separate from the renderer-owned catalog."""
    return json.loads((Path(__file__).parent / "data" / "tui" /
                       "plot-presets.json").read_text(encoding="utf-8"))


def lane_capabilities(path: Path | None = None) -> dict:
    """The packaged lane record: unserved reasons and window requirements.

    THE reader for that file.  ``products`` carries the window a
    recorded product needs (``minimum_hours``) and the basis it was
    recorded from; ``unavailable`` carries, per product, the concrete
    breakage that stops this lane drawing it and what to do instead.
    Both are read straight out of the package, with the file's own
    digest, so a caller can say which bytes it decided on.

    A product in neither map is not a refusal and not an error.  It is a
    product this record says nothing about, and the caller runs it.
    """

    path = DIAGNOSTICS_PATH if path is None else Path(path)

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        payload = path.read_bytes()
        document = json.loads(payload.decode("utf-8"),
                              object_pairs_hook=unique_object)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Cannot read packaged lane diagnostic capabilities at {path}: "
            f"{error}. Restore the matching ArWen package data, then retry."
        ) from error
    try:
        if not isinstance(document, dict):
            raise ValueError("the document must be a JSON object")
        if document.get("schema") != "arwen.research.diagnostics.v1":
            raise ValueError("expected schema arwen.research.diagnostics.v1")
        if not isinstance(document["products"], dict) or not document["products"]:
            raise ValueError("products must be a nonempty object")
        if not isinstance(document["unavailable"], dict):
            raise ValueError("unavailable must be an object")
        for name, entry in document["products"].items():
            minimum = entry["minimum_hours"]
            if (not isinstance(name, str) or not isinstance(entry["kind"], str)
                    or isinstance(minimum, bool)
                    or not isinstance(minimum, int) or minimum < 0):
                raise ValueError(f"invalid diagnostic capability {name}")
        for name, reason in document["unavailable"].items():
            if not isinstance(name, str) or not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"unavailable {name} records no reason")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid research diagnostic capabilities at {path}: {error}. "
            "Restore the matching ArWen package data, then retry.") from error
    return {**document, "sha256": hashlib.sha256(payload).hexdigest()}


def preset_availability(capabilities=None, requirements=None) -> dict:
    """Per preset, the products this lane will not draw, and why.

    Two sources, one statement.  The packaged record names the products
    the wrfout lane cannot serve AND the concrete reason, in the words a
    reader needs; the renderer's own fileless requirement rows name the
    store selector the import does not write, for this build.  Whichever
    has something to say says it, and when both do the reader gets both
    sentences rather than a choice between them.

    It states, never refuses.  A preset is a curated request, not a
    promise about one install, and narrowing the curated list to what
    this box happens to draw would hide the finding instead of reporting
    it.  When the renderer cannot be asked at all, the block still comes
    back, carrying what the record knows and saying so.
    """

    capabilities = lane_capabilities() if capabilities is None else capabilities
    recorded = dict(capabilities.get("unavailable") or {})
    document = {}
    for preset in presets()["presets"]:
        rows = {}
        for slug in preset["products"]:
            reason = recorded.get(slug)
            if requirements is not None:
                from gpuwm.rustwx import undrawable

                measured = undrawable([slug], requirements=requirements).get(slug)
                if measured:
                    reason = f"{reason} {measured}" if reason else measured
            if reason:
                rows[slug] = reason
        document[preset["id"]] = rows
    return document


def catalog_document() -> dict:
    """Ask the installed renderer without building, staging, or probing a GPU."""
    from gpuwm import bridges
    from gpuwm.runplan import render_catalog

    with bridges.inspection_only():
        catalog = render_catalog()
        requirements, basis = _requirements()
    capabilities = lane_capabilities()
    return {**catalog, "presets": presets(),
            "preset_availability": preset_availability(capabilities, requirements),
            "preset_availability_basis": basis,
            "lane_record_sha256": capabilities["sha256"],
            "lanes": other_lanes()}


def other_lanes() -> dict:
    """Product lanes beside the wrfout catalog, listed rather than hidden.

    The observation-grid engine draws five products from an observation
    volume, and nothing in this tree listed them anywhere a reader
    looks.  They deliberately do NOT join the wrfout catalog: that one
    takes forecast frames and would refuse all five on every file, which
    is the menu-nothing-serves failure its own door exists to prevent.
    They are their own lane, named, with what they take.
    """

    from gpuwm import rustwx_lanes

    return {
        "observation-grid": {
            "products": list(rustwx_lanes.OBSGRID_PRODUCTS),
            "input": "a gpuwm-obs.radar-grid.v1 observation volume "
                     "(--obs FILE.nc), not a wrfout frame",
            "engine": rustwx_lanes.OBSGRID_NAME,
        },
    }


def _requirements():
    """``(the renderer's requirement rows or None, the basis sentence)``."""

    try:
        from gpuwm.rustwx import REQUIREMENTS_BASIS, catalog_requirements

        answer = catalog_requirements()
    except Exception as error:  # the picker states this, it never dies of it
        return None, ("the installed renderer could not be asked which "
                      f"products it can draw ({error}); what follows is the "
                      "packaged lane record only, so a product this install "
                      "cannot draw for another reason is not named here")
    if answer is None:
        return None, ("no renderer was resolvable to ask which products it "
                      "can draw; what follows is the packaged lane record "
                      "only, so a product this install cannot draw for "
                      "another reason is not named here")
    return answer, (f"the packaged lane record, plus {REQUIREMENTS_BASIS}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", action="store_true",
                        help="return the installed renderer catalog as JSON")
    args = parser.parse_args(argv)
    print(json.dumps(catalog_document() if args.catalog else presets()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
