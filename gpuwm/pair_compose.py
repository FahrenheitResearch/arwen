"""Compose matching product PNGs from two runs into labeled pair sheets.

``gpuwm render --pair A_DIR B_DIR`` takes two directories of rendered
product PNGs -- either engine's output -- and writes one side-by-side
sheet per product the two runs have in common, plus a ``manifest.tsv``
recording exactly which files each sheet was composed from.  This is
the campaign's paired-comparison flow (its compositor is preserved
byte-for-byte in the campaign receipts) generalized to any two runs:
labels and title come from the caller or default to the directory
names, never to any particular experiment's wording.

Matching key: everything after the rust engine's ``_fNNN_`` lead marker
-- the domain token and the product slug together -- else the whole file
stem (the matplotlib engine's naming, where identical product/domain/
valid-time stems pair naturally).  The domain stays *in* the key on
purpose: a directory can now hold several nests of one run, and pairing
a 3 km panel against a 333 m one would be a comparison of nothing.

Pillow is the only dependency -- it ships with the ``[render]`` extra
(matplotlib depends on it), and no science is performed here: pixels
are pasted, never recomputed.
"""

from __future__ import annotations

import re
from pathlib import Path

from gpuwm.render_layout import fs_path, iter_rendered

#: The rust renderer's forecast-hour marker.  Everything before it is the
#: run's identity (model, init date, cycle), which two compared runs are
#: expected to differ in; everything after it -- domain token and product
#: slug -- is what has to match.  Both brand spellings key identically:
#: ``arwen_`` is what the wheel ships since the output-filename rebrand
#: (:func:`gpuwm.render._rebrand_engine_output`), ``rustwx_`` is what
#: the vendored engine writes natively and what every directory rendered
#: before the rebrand still holds -- so an old render pairs against a
#: new one.
LEAD_MARKER = re.compile(r"^(?:arwen|rustwx)_.+?_f\d{3}_")

#: The domain token a pairing key starts with (``d02-3km_``,
#: ``d05-111m_``).  ``native_grid`` deliberately does not match: a
#: pre-token name carries no domain to strip.
_DOMAIN_TOKEN = re.compile(r"^d\d{2}-[^_]+_")


def product_name(path: Path) -> str:
    """Pairing key for one rendered PNG."""

    name = path.stem
    match = LEAD_MARKER.match(name)
    return name[match.end():] if match else name


def _load_font(size: int, *, bold: bool = False):
    from PIL import ImageFont

    candidates = []
    windows = Path("C:/Windows/Fonts")
    candidates.extend([
        windows / ("segoeuib.ttf" if bold else "segoeui.ttf"),
        windows / ("arialbd.ttf" if bold else "arial.ttf"),
    ])
    dejavu = Path("/usr/share/fonts/truetype/dejavu")
    candidates.append(
        dejavu / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"))
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _fit(image, width: int):
    from PIL import Image

    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def compose_pairs(left_dir: Path, right_dir: Path, out_dir: Path, *,
                  title: str, subtitle: str = "",
                  left_label: str | None = None,
                  right_label: str | None = None,
                  panel_width: int = 900) -> list[Path]:
    """Write one pair sheet per common product; return the sheet paths.

    Raises ``ValueError`` when the directories share no product PNGs --
    an empty comparison is a user-facing refusal, not a silent success.
    """

    from PIL import Image, ImageDraw

    left_dir, right_dir = Path(left_dir), Path(right_dir)
    # Through the layout walker, not `glob("*.png")`: since 2.5.0 the
    # render directory is a tree (domain/product/valid-day) and a
    # top-level glob finds nothing in one, which would have turned
    # `--pair` into the "no matching product PNGs" refusal on every run
    # rendered by the shipped default.  The pairing KEY is unchanged --
    # it is read from the filename, which did not move.
    left = {product_name(p): p for p in iter_rendered(left_dir)}
    right = {product_name(p): p for p in iter_rendered(right_dir)}
    common = sorted(left.keys() & right.keys())
    pairs: dict[str, tuple[Path, Path]] = {
        key: (left[key], right[key]) for key in common}
    # Second pass, for the comparison DOWNSCALE.md documents: a parent
    # directory against its offline child's is two DIFFERENT domains by
    # definition, so the domain-bearing keys above share nothing and the
    # documented command refused (walked 2026-08-17).  Keys the exact
    # pass left unmatched pair across the domain token -- but only where
    # the stripped key is unambiguous on its own side, so two nests of
    # one run in one directory still never cross-pair (that pin
    # predates this pass and stands).
    def _stripped_unmatched(mapping, matched):
        candidates: dict[str, list[tuple[str, Path]]] = {}
        for key, path in mapping.items():
            if key in matched:
                continue
            token = _DOMAIN_TOKEN.match(key)
            if token is None:
                continue
            candidates.setdefault(key[token.end():], []).append((key, path))
        return {stripped: group[0] for stripped, group in
                candidates.items() if len(group) == 1}

    left_stripped = _stripped_unmatched(left, set(common))
    right_stripped = _stripped_unmatched(right, set(common))
    for stripped in sorted(set(left_stripped) & set(right_stripped)):
        (left_key, left_path) = left_stripped[stripped]
        (right_key, right_path) = right_stripped[stripped]
        pairs[f"{left_key}_vs_{right_key}"] = (left_path, right_path)
    if not pairs:
        raise ValueError(
            f"no matching product PNGs between {left_dir} and {right_dir}")

    left_label = left_label or left_dir.name or str(left_dir)
    right_label = right_label or right_dir.name or str(right_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    title_font = _load_font(34, bold=True)
    subtitle_font = _load_font(22)
    label_font = _load_font(28, bold=True)
    product_font = _load_font(24)
    sheets: list[Path] = []
    manifest: list[str] = []

    for product in sorted(pairs):
        left_path, right_path = pairs[product]
        # Opened through the layout's own spelling: `iter_rendered`
        # walks as deep as the placement can write, and on Windows that
        # is past the ordinary API's ceiling.  A composer that listed a
        # frame and then could not open it would fail a --pair run on
        # exactly the deep case roots the layout exists to organise.
        panel_l = _fit(Image.open(fs_path(left_path)).convert("RGB"),
                       panel_width)
        panel_r = _fit(Image.open(fs_path(right_path)).convert("RGB"),
                       panel_width)
        header = 128 if subtitle else 96
        label_height = 48
        body_height = max(panel_l.height, panel_r.height)
        canvas = Image.new(
            "RGB",
            (panel_width * 2 + 24, header + label_height + body_height),
            "#f4f6f8",
        )
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 14), title, fill="#17212b", font=title_font)
        pretty = product.replace("_", " ").title()
        title_width = draw.textbbox((0, 0), pretty, font=product_font)[2]
        draw.text(
            (canvas.width - title_width - 24, 22),
            pretty,
            fill="#334155",
            font=product_font,
        )
        if subtitle:
            draw.text((24, 60), subtitle, fill="#52606d",
                      font=subtitle_font)
        draw.rectangle((0, header, panel_width, header + label_height),
                       fill="#e3edf7")
        draw.rectangle(
            (panel_width + 24, header, canvas.width, header + label_height),
            fill="#e7f5ed",
        )
        draw.text((24, header + 7), left_label, fill="#174a72",
                  font=label_font)
        draw.text(
            (panel_width + 48, header + 7),
            right_label,
            fill="#17623b",
            font=label_font,
        )
        canvas.paste(panel_l, (0, header + label_height))
        canvas.paste(panel_r, (panel_width + 24, header + label_height))
        output = out_dir / f"{product}-pair.png"
        canvas.save(output, optimize=True)
        sheets.append(output)
        manifest.append(
            f"{product}\t{left_path}\t{right_path}\t{output}")

    (out_dir / "manifest.tsv").write_text(
        "product\tleft\tright\tpair\n" + "\n".join(manifest) + "\n",
        encoding="utf-8",
    )
    return sheets


__all__ = ["LEAD_MARKER", "compose_pairs", "product_name"]
