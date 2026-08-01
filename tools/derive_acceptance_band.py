"""Derive a committed acceptance band, and its coverage receipt, from data.

    python tools/derive_acceptance_band.py --config CONFIG.toml

The band is the deterministic output of ``gpuwm/data/certification/
margin_rule.json`` applied to the published comparison table.  Re-running this
command on an unchanged tree rewrites the committed files byte for byte; the
band test re-runs the same production derivation and asserts exactly that, so
a band edited by hand is caught by the tool that made it.

The band is addressed by the SHA-256 of the configuration it certifies.  No
case name enters the band, the receipt, or the file names.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpuwm.certify.band import (  # noqa: E402
    BAND_DIR, MARGIN_RULE_PATH, band_path_for_config, derive_band,
    derive_coverage_receipt, dumps_certification_json, load_margin_rule,
    sha256_file, write_certification_json)

#: Where the coverage receipt for a band lands, beside the band itself.
COVERAGE_SUFFIX = ".coverage.json"


def derive(config_path: Path, *, anchor_document: Path,
           band_dir: Path = BAND_DIR) -> tuple[Path, Path, str, str]:
    """Write the band and its coverage receipt; return both paths and texts."""
    margin_rule = load_margin_rule()
    anchor_text = anchor_document.read_text(encoding="utf-8")
    band = derive_band(
        config_sha256=sha256_file(config_path),
        anchor_document_text=anchor_text,
        margin_rule=margin_rule,
        margin_rule_sha256=sha256_file(MARGIN_RULE_PATH),
        anchor_document_name=anchor_document.relative_to(
            REPO_ROOT).as_posix())
    receipt = derive_coverage_receipt(
        band, anchor_document_text=anchor_text, margin_rule=margin_rule)
    band_path = band_path_for_config(band["config_sha256"],
                                     directory=band_dir)
    receipt_path = band_path.with_name(band_path.stem + COVERAGE_SUFFIX)
    band_text = dumps_certification_json(band)
    receipt_text = dumps_certification_json(receipt)
    write_certification_json(band_path, band)
    write_certification_json(receipt_path, receipt)
    return band_path, receipt_path, band_text, receipt_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, metavar="TOML",
                        help="the configuration this band certifies; its "
                             "SHA-256 becomes the band's identity")
    parser.add_argument("--anchor-document", type=Path,
                        default=REPO_ROOT / "docs" / "public"
                        / "VERIFICATION.md", metavar="MD",
                        help="document carrying the published comparison "
                             "table the margin rule anchors on")
    parser.add_argument("--band-dir", type=Path, default=BAND_DIR,
                        metavar="DIR")
    args = parser.parse_args(argv)
    band_path, receipt_path, _, receipt_text = derive(
        args.config, anchor_document=args.anchor_document,
        band_dir=args.band_dir)
    import json

    tally = json.loads(receipt_text)["tally"]
    print(f"band:     {band_path}")
    print(f"coverage: {receipt_path}")
    print(f"tally:    {tally}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
