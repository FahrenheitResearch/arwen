"""Numerical-generation metadata and TITLE provenance for classic Thompson."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class ClassicGeneration:
    releases: tuple[str, ...]
    backend: str
    source_sha256: str


CLASSIC_GENERATIONS = {
    "wrf_461": ClassicGeneration(("4.6.1",), "legacy", "fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802"),
    "wrf_44": ClassicGeneration(("4.4", "4.4.0"), "column", "c89bd2ec24a2e6bc95a6ec6d47ad7239a1656c1eca3cc1e70f5089345c6211cb"),
}
RELEASE_GENERATIONS = {release: key for key, row in CLASSIC_GENERATIONS.items() for release in row.releases}
TITLE_RELEASE = re.compile(r"\bV(\d+\.\d+(?:\.\d+)?)\b", re.IGNORECASE)


def title_generation(title):
    match = TITLE_RELEASE.search(str(title))
    return RELEASE_GENERATIONS.get(match[1], "wrf_461") if match else "wrf_461"


def select_classic_generation(metadata, *, runtime_titles=(), override=None):
    classic = [item for item in metadata.values() if item.mp_physics == 8]
    if not classic:
        if override is not None:
            raise ValueError("a classic Thompson generation override requires mp_physics=8; another scheme has different state and would ignore that override")
        return None, {"source": "scheme has no classic Thompson state"}
    if override is not None:
        if override not in CLASSIC_GENERATIONS:
            raise ValueError(f"unknown classic Thompson generation {override!r}; an unknown name cannot identify its constants or lookup tables")
        return override, {"source": "explicit override", "generation": override}
    titles = list(runtime_titles) or [str(item.global_attributes.get("TITLE", "")) for item in classic]
    generations = {title_generation(title) for title in titles}
    if len(generations) != 1:
        raise ValueError("WRF files declare different Thompson generations; one run cannot silently use one generation for both source identities")
    generation = generations.pop()
    return generation, {"source": "runtime TITLE" if runtime_titles else "input TITLE", "titles": titles, "generation": generation}


def matching_runtime_titles(directory, metadata):
    """A matching model restart is stronger evidence than a REAL_EM stamp.

    Preprocessing can precede the integrating WRF release. Only same-grid,
    same-simulation model restarts qualify; unrelated old restarts do not.
    """
    from gpuwm.ingest.wrfinput import read_wrfinput_metadata
    root = metadata[min(metadata)]
    if root.mp_physics != 8:
        return ()
    titles = []
    for path in sorted(Path(directory).glob("wrfrst_d01_*")):
        candidate = read_wrfinput_metadata(path)
        attrs = candidate.global_attributes
        if (candidate.mp_physics != root.mp_physics
                or (candidate.nx,candidate.ny,candidate.nz,candidate.soil_layers) != (root.nx,root.ny,root.nz,root.soil_layers)
                or str(attrs.get("SIMULATION_START_DATE", "")) != str(root.start_date)):
            continue
        title = str(attrs.get("TITLE", ""))
        if "MODEL" in title.upper() and TITLE_RELEASE.search(title):
            titles.append(title)
    return tuple(titles)
