"""Build the storm wiki's seed store from open records.

    python tools/wiki_seed/build_seed.py --cache DIR [--out gpuwm/gui/seed/wiki-seed.json]

The seed fills the wiki until the event atlas writes its own cards.  Every
fact it writes names the record it came from: an IBTrACS row, an SPC
tornado row, a Census county code, a Natural Earth outline, a glossary
entry, or a statistic this script computed from those tables (the
computed record names its inputs and its method).  Nothing is typed in by
hand except the list of record ids below and the rules this script applies.

Inputs are downloaded into ``--cache`` when missing; the output is one
``gpuwm.wiki.v1`` document (docs/dev/WIKI.md).  Standard library only.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sys
import urllib.request

RETRIEVED = datetime.now(timezone.utc).strftime("%Y-%m-%d")
BUILDER = "tools/wiki_seed/build_seed.py"

# ------------------------------------------------------------------ what to seed

#: IBTrACS storm ids (SID).  Chosen to span every basin the archive covers.
TC_SIDS = [
    "2005236N23285", "2015293N13266", "2013306N07162", "2019063S18038", "2008117N11090",
    "2007151N14072", "2004086S29318", "2016041S14170", "2017280N32321", "2023294N09264",
    "2023036S12117",
]
#: SPC tornado rows as (year, om).  om is unique within a year.
TORNADO_ROWS = [
    ("2011", "1105221634-01"), ("2013", "1305201356-01"), ("1999", "466"),
    ("2021", "2112102054-01"), ("2023", "2303241857-01"), ("1972", "114"),
]

# ------------------------------------------------------------------ the records

DATASETS = {
    "ibtracs": {
        "file": "ibtracs.since1980.list.v04r01.csv",
        "url": "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
               "v04r01/access/csv/ibtracs.since1980.list.v04r01.csv",
        "title": "IBTrACS v04r01, storms since 1980 (CSV)",
        "publisher": "NOAA National Centers for Environmental Information",
        "licence": "US Government work, no copyright in the US. NCEI asks users to cite the paper and the dataset.",
        "citation": "Knapp, K. R., M. C. Kruk, D. H. Levinson, H. J. Diamond, and C. J. Neumann, 2010: The "
                    "International Best Track Archive for Climate Stewardship (IBTrACS). BAMS 91, 363-376, "
                    "doi:10.1175/2009BAMS2755.1. Gahtan, J., et al., 2024: IBTrACS Project, Version 4r01, "
                    "doi:10.25921/82ty-9e16.",
    },
    "ibtracs-doc": {
        "url": "https://www.ncei.noaa.gov/sites/default/files/2021-07/IBTrACS_v04_column_documentation.pdf",
        "title": "IBTrACS v04 column documentation",
        "publisher": "NOAA National Centers for Environmental Information",
        "licence": "US Government work, no copyright in the US.",
    },
    "spc": {
        "file": "spc_1950-2025_actual_tornadoes.csv",
        "url": "https://www.spc.noaa.gov/wcm/data/1950-2025_actual_tornadoes.csv",
        "title": "SPC tornado database, 1950 to 2025 (CSV)",
        "publisher": "NOAA Storm Prediction Center",
        "licence": "US Government work, no copyright in the US.",
    },
    "spc-doc": {
        "file": "spc_desc.pdf",
        "url": "https://www.spc.noaa.gov/wcm/data/SPC_severe_database_description.pdf",
        "title": "SPC severe weather database description",
        "publisher": "NOAA Storm Prediction Center",
        "licence": "US Government work, no copyright in the US.",
    },
    "census-counties": {
        "file": "national_county2020.txt",
        "url": "https://www2.census.gov/geo/docs/reference/codes2020/national_county2020.txt",
        "title": "2020 national county codes (FIPS)",
        "publisher": "US Census Bureau",
        "licence": "US Government work, no copyright in the US.",
    },
    "ne-countries": {
        "file": "ne_110m_admin_0_countries.geojson",
        "url": "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
               "ne_110m_admin_0_countries.geojson",
        "title": "Natural Earth 1:110m countries",
        "publisher": "Natural Earth",
        "licence": "Public domain.",
    },
    "ne-states": {
        "file": "ne_110m_admin_1_states_provinces.geojson",
        "url": "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
               "ne_110m_admin_1_states_provinces.geojson",
        "title": "Natural Earth 1:110m states and provinces",
        "publisher": "Natural Earth",
        "licence": "Public domain.",
    },
    "nws-glossary": {
        "url": "https://forecast.weather.gov/glossary.php",
        "title": "National Weather Service glossary",
        "publisher": "NOAA National Weather Service",
        "licence": "US Government work, no copyright in the US.",
    },
    "era5-doc": {
        "url": "https://doi.org/10.1002/qj.3803",
        "title": "Hersbach et al., 2020: The ERA5 global reanalysis. QJRMS 146, 1999-2049",
        "publisher": "Royal Meteorological Society (open access)",
        "licence": "CC BY 4.0 article.",
    },
    "era5-back-doc": {
        "url": "https://doi.org/10.1002/qj.4174",
        "title": "Bell et al., 2021: The ERA5 global reanalysis, preliminary extension to 1950. QJRMS 147, 4186-4227",
        "publisher": "Royal Meteorological Society (open access)",
        "licence": "CC BY 4.0 article.",
    },
}

#: Glossary entries quoted word for word from the NWS glossary.
GLOSSARY = {
    "tornado": ("Tornado", "A violently rotating column of air, usually pendant to a cumulonimbus, with circulation "
                "reaching the ground. It nearly always starts as a funnel cloud and may be accompanied by a loud "
                "roaring noise. On a local scale, it is the most destructive of all atmospheric phenomena."),
    "tropical-cyclone": ("Tropical Cyclone", "A warm-core, non-frontal synoptic-scale cyclone, originating over "
                         "tropical or subtropical waters with organized deep convection and a closed surface wind "
                         "circulation about a well-defined center."),
}

#: IBTrACS BASIN codes, as the column documentation spells them.
BASINS = {"NA": "North Atlantic", "EP": "Eastern North Pacific", "WP": "Western North Pacific",
          "NI": "North Indian", "SI": "South Indian", "SP": "Southern Pacific", "SA": "South Atlantic"}
#: USA_SSHS codes, as the column documentation spells them.
SSHS = {-5: "Unknown", -4: "Post-tropical", -3: "Disturbance", -2: "Subtropical", -1: "Tropical depression",
        0: "Tropical storm", 1: "Category 1", 2: "Category 2", 3: "Category 3", 4: "Category 4", 5: "Category 5"}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
SEASONS = {12: "dec-feb", 1: "dec-feb", 2: "dec-feb", 3: "mar-may", 4: "mar-may", 5: "mar-may",
           6: "jun-aug", 7: "jun-aug", 8: "jun-aug", 9: "sep-nov", 10: "sep-nov", 11: "sep-nov"}
SEASON_WORDS = {"dec-feb": "December to February", "mar-may": "March to May", "jun-aug": "June to August",
                "sep-nov": "September to November"}
#: The date SPC began rating on the Enhanced Fujita scale (SPC database description).
EF_START = "2007-02-01"
KT_KMH = 1.852
#: GFS cycles are kept on the public archive the engine reads from this date on; older recipes start from ERA5.
GFS_FROM = "2021-03-01T00"


# ------------------------------------------------------------------ small tools

def fetch(cache: Path, key: str) -> Path:
    info = DATASETS[key]
    path = cache / info["file"]
    if not path.is_file():
        print(f"downloading {info['url']}", file=sys.stderr)
        with urllib.request.urlopen(info["url"], timeout=600) as response, path.open("wb") as out:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
    return path


def url_ok(url: str) -> bool:
    try:
        request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "gpuwm-wiki-seed"})
        with urllib.request.urlopen(request, timeout=30) as response:
            return 200 <= response.status < 400
    except Exception:  # noqa: BLE001 - a link that does not answer is left out
        return False


def ring_contains(ring: list[list[float]], x: float, y: float) -> bool:
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


class Outlines:
    """Natural Earth polygons with bounding boxes, for point-in-polygon lookups."""

    def __init__(self, path: Path, key) -> None:
        document = json.loads(path.read_text(encoding="utf-8"))
        self.items = []
        for feature in document["features"]:
            geometry = feature["geometry"]
            polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
            xs = [p[0] for poly in polygons for p in poly[0]]
            ys = [p[1] for poly in polygons for p in poly[0]]
            self.items.append((key(feature["properties"]), feature["properties"], polygons,
                               (min(xs), max(xs), min(ys), max(ys))))

    def find(self, lon: float, lat: float):
        for ident, props, polygons, (x0, x1, y0, y1) in self.items:
            if not (x0 <= lon <= x1 and y0 <= lat <= y1):
                continue
            for poly in polygons:
                if ring_contains(poly[0], lon, lat) and not any(ring_contains(hole, lon, lat) for hole in poly[1:]):
                    return ident, props
        return None, None


def num(value: str) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%MZ")


def day_words(moment: datetime) -> str:
    return f"{moment.day} {MONTHS[moment.month - 1]} {moment.year}"


def plural(count: int, word: str, many: str | None = None) -> str:
    return f"{count:,} {word if count == 1 else (many or word + 's')}"


def recipe_source(cycle: str) -> str:
    return "gfs" if cycle >= GFS_FROM else "era5"


# ------------------------------------------------------------------ the store being built

class Store:
    def __init__(self) -> None:
        self.sources: dict[str, dict] = {}
        self.places: dict[str, dict] = {}
        self.events: list[dict] = []
        for key, info in DATASETS.items():
            self.sources[f"dataset-{key}"] = {
                "id": f"dataset-{key}", "kind": "dataset", "title": info["title"], "publisher": info["publisher"],
                "url": info["url"], "licence": info["licence"], "retrieved": RETRIEVED,
                **({"citation": info["citation"]} if "citation" in info else {}),
            }

    def source(self, ident: str, **record) -> str:
        record.setdefault("retrieved", RETRIEVED)
        self.sources[ident] = {"id": ident, **record}
        return ident

    def place(self, ident: str, title: str, kind: str, cite: list[str], parent: str | None = None) -> str:
        if ident not in self.places:
            self.places[ident] = {"id": ident, "title": title, "kind": kind, "cite": cite,
                                  **({"parent": parent} if parent else {}), "seed": True}
        return ident


#: IBTrACS USA_AGENCY codes as the IBTrACS column documentation names them (dataset-ibtracs-doc).
AGENCIES = {
    "hurdat_atl": "NHC best track (HURDAT2, Atlantic)",
    "hurdat_epa": "NHC best track (HURDAT2, eastern Pacific)",
    "cphc": "CPHC best track (central Pacific)",
    "jtwc_wp": "JTWC best track (western Pacific)",
    "jtwc_io": "JTWC best track (Indian Ocean)",
    "jtwc_sh": "JTWC best track (Southern Hemisphere)",
    "jtwc_cp": "JTWC best track (central Pacific)",
    "jtwc_ep": "JTWC best track (eastern Pacific)",
    "atcf": "ATCF best track (JTWC working file)",
    "tcvitals": "TC vitals (operational messages)",
}


def fact(ident: str, label: str, text: str, cite: list[str], value=None, unit: str | None = None,
         infobox: bool = True) -> dict:
    out = {"id": ident, "label": label, "text": text, "cite": cite, "infobox": infobox}
    if value is not None:
        out["value"] = value
    if unit:
        out["unit"] = unit
    return out


def era_quality(year: int) -> dict:
    if year >= 1979:
        return {"flag": "satellite-era", "text": "ERA5 has satellite data for this year (1979 on).",
                "cite": ["dataset-era5-doc"]}
    if year >= 1950:
        return {"flag": "before-satellites", "text": "Before 1979 ERA5 has no satellite radiances; it rests on "
                "surface and upper-air stations, so small or ocean systems are weaker in it.",
                "cite": ["dataset-era5-back-doc"]}
    return {"flag": "early", "text": "ERA5 before 1950 is its back extension with few observations.",
            "cite": ["dataset-era5-back-doc"]}


# ------------------------------------------------------------------ tropical cyclones

def tropical_cyclones(store: Store, cache: Path) -> None:
    path = fetch(cache, "ibtracs")
    countries = Outlines(fetch(cache, "ne-countries"), lambda p: p["ADM0_A3"])
    storms: dict[str, dict] = {}
    rows: dict[str, list[dict]] = {sid: [] for sid in TC_SIDS}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        next(reader)  # the units row
        for row in reader:
            sid = row["SID"]
            if row["TRACK_TYPE"] != "main":
                continue
            s = storms.get(sid)
            if s is None:
                s = storms[sid] = {"basin": row["BASIN"], "season": int(row["SEASON"]), "wind": None,
                                   "peak_month": None, "land": set()}
            wind = num(row["USA_WIND"])
            if wind is not None and (s["wind"] is None or wind > s["wind"]):
                s["wind"] = wind
                s["peak_month"] = int(row["ISO_TIME"][5:7])
            if row["DIST2LAND"].strip() == "0":
                lat, lon = num(row["LAT"]), num(row["LON"])
                if lat is not None and lon is not None:
                    s["land_pts"] = s.get("land_pts", [])
                    s["land_pts"].append((lon, lat))
            if sid in rows:
                rows[sid].append(row)
    # Countries each storm's track touched over land, for every storm (the places' counts need them all).
    names: dict[str, str] = {}
    for sid, s in storms.items():
        for lon, lat in s.pop("land_pts", []):
            code, props = countries.find(((lon + 180) % 360) - 180, lat)
            if code:
                s["land"].add(code)
                names[code] = props["NAME"]
    first_year = min(s["season"] for s in storms.values())
    last_year = max(s["season"] for s in storms.values())

    for sid in TC_SIDS:
        track = rows[sid]
        if not track:
            raise SystemExit(f"IBTrACS has no storm {sid}")
        s = storms[sid]
        first, last = track[0], track[-1]
        name = first["NAME"].strip()
        season = int(first["SEASON"])
        basin = first["BASIN"]
        winds = [(num(r["USA_WIND"]), i) for i, r in enumerate(track) if num(r["USA_WIND"]) is not None]
        peak_i = max(winds)[1] if winds else 0
        peak = track[peak_i]
        pressures = [(num(r["USA_PRES"]), i) for i, r in enumerate(track) if num(r["USA_PRES"]) is not None]
        low = track[min(pressures)[1]] if pressures else None
        t0 = datetime.strptime(first["ISO_TIME"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        t1 = datetime.strptime(last["ISO_TIME"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        tp = datetime.strptime(peak["ISO_TIME"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        keep = ("SID", "SEASON", "BASIN", "NAME", "ISO_TIME", "NATURE", "LAT", "LON", "WMO_WIND", "WMO_PRES",
                "WMO_AGENCY", "DIST2LAND", "USA_AGENCY", "USA_ATCF_ID", "USA_WIND", "USA_PRES", "USA_SSHS")

        def row_source(row: dict, what: str) -> str:
            ident = f"ibtracs-{sid}-{row['ISO_TIME'][:13].replace(' ', 'T').replace(':', '')}"
            return store.source(ident, kind="record", dataset="dataset-ibtracs",
                                title=f"IBTrACS row: storm {sid}, {row['ISO_TIME'][:16]} UTC ({what})",
                                publisher=DATASETS["ibtracs"]["publisher"], url=DATASETS["ibtracs"]["url"],
                                licence=DATASETS["ibtracs"]["licence"],
                                row={k: row[k].strip() for k in keep})

        src_first = row_source(first, "first row")
        src_last = row_source(last, "last row")
        src_peak = row_source(peak, "highest wind")
        src_low = row_source(low, "lowest pressure") if low is not None else None
        track_src = store.source(f"ibtracs-{sid}-track", kind="record", dataset="dataset-ibtracs",
                                 title=f"IBTrACS track: storm {sid}, {len(track)} rows",
                                 publisher=DATASETS["ibtracs"]["publisher"], url=DATASETS["ibtracs"]["url"],
                                 licence=DATASETS["ibtracs"]["licence"],
                                 row={"SID": sid, "rows": str(len(track)), "first": first["ISO_TIME"],
                                      "last": last["ISO_TIME"]})

        title_name = name.title() if name not in ("UNNAMED", "NOT_NAMED") else f"Unnamed {BASINS[basin]} cyclone"
        title = f"{title_name} ({season})"
        basin_place = store.place(f"basin-{basin.lower()}", BASINS[basin], "basin", ["dataset-ibtracs-doc"])
        touched = sorted(s["land"], key=lambda code: next(
            (i for i, r in enumerate(track) if r["DIST2LAND"].strip() == "0"
             and countries.find(((num(r["LON"]) + 180) % 360) - 180, num(r["LAT"]))[0] == code), 0))
        country_places = [store.place(f"country-{code.lower()}", names[code], "country", ["dataset-ne-countries"])
                          for code in touched]
        land_src = store.source(f"computed-{sid}-land", kind="computed",
                                title=f"Countries the track of {sid} touched over land",
                                method="Each IBTrACS main-track row of this storm with DIST2LAND equal to 0 is "
                                       "placed in a Natural Earth 1:110m country outline.",
                                code=f"{BUILDER}: tropical_cyclones", inputs=[track_src, "dataset-ne-countries"],
                                result={"countries": [names[c] for c in touched]})

        facts = [
            fact("name", "Name in the record", name, [src_first], infobox=False),
            fact("sid", "IBTrACS id", sid, [src_first]),
            fact("basin", "Basin", BASINS[basin], [src_first, "dataset-ibtracs-doc"], value=basin),
            fact("start", "First track point", f"{t0:%Y-%m-%d %H:%M} UTC", [src_first], value=iso(t0)),
            fact("end", "Last track point", f"{t1:%Y-%m-%d %H:%M} UTC", [src_last], value=iso(t1)),
        ]
        if winds:
            w = max(winds)[0]
            facts.append(fact("peak_wind", "Highest wind", f"{w:.0f} kt ({w * KT_KMH:.0f} km/h), 1-minute mean",
                              [src_peak, "dataset-ibtracs-doc"], value=w, unit="kt"))
            agency = peak["USA_AGENCY"].strip()
            facts.append(fact("wind_agency", "Wind record from", AGENCIES.get(agency, agency),
                              [src_peak, "dataset-ibtracs-doc"], value=agency))
            facts.append(fact("peak_time", "Highest wind at", f"{tp:%Y-%m-%d %H:%M} UTC", [src_peak], value=iso(tp)))
            code = int(num(peak["USA_SSHS"]) or -5)
            facts.append(fact("peak_class", "Class at peak", SSHS.get(code, str(code)), [src_peak, "dataset-ibtracs-doc"],
                              value=code))
        if low is not None:
            p = min(pressures)[0]
            facts.append(fact("min_pressure", "Lowest pressure", f"{p:.0f} hPa", [src_low], value=p, unit="hPa"))
        if touched:
            facts.append(fact("land", "Track over land in", ", ".join(names[c] for c in touched), [land_src]))

        # rarity: storms of this basin since 1980 at least this strong, peaking within a month of this one
        rarity = None
        if winds and s["wind"] is not None:
            m = s["peak_month"]
            near = {((m - 2) % 12) + 1, m, (m % 12) + 1}
            basin_storms = [x for x in storms.values() if x["basin"] == basin and x["wind"] is not None]
            stronger_season = [x for x in basin_storms if x["wind"] >= s["wind"] and x["peak_month"] in near]
            stronger_all = [x for x in basin_storms if x["wind"] >= s["wind"]]
            months = [MONTHS[(m - 2) % 12], MONTHS[(m % 12)]]
            rare_src = store.source(
                f"computed-{sid}-rarity", kind="computed",
                title=f"Rarity of {sid} in the {BASINS[basin]} basin",
                method=(f"Counted IBTrACS main-track storms of basin {basin}, seasons {first_year} to {last_year}, "
                        f"whose highest USA_WIND is at least {s['wind']:.0f} kt, all year and with that highest "
                        f"wind in {months[0]} to {months[1]}."),
                code=f"{BUILDER}: tropical_cyclones", inputs=["dataset-ibtracs"],
                result={"at_least_this_strong_in_season": len(stronger_season),
                        "at_least_this_strong_all_year": len(stronger_all),
                        "storms_with_wind": len(basin_storms), "years": [first_year, last_year]})
            rarity = {
                "count": len(stronger_season), "of": len(basin_storms), "place": basin_place,
                "text": (f"{plural(len(stronger_season), 'storm')} in the {BASINS[basin]} basin reached {s['wind']:.0f} kt or "
                         f"more with their peak in {months[0]} to {months[1]}, {first_year} to {last_year} "
                         f"({len(stronger_all):,} in any month, of {len(basin_storms):,} storms)."),
                "cite": [rare_src],
            }
            facts.append(fact("rarity", "At least this strong here", f"{len(stronger_season):,} of {len(basin_storms):,} "
                              f"storms, {months[0][:3]} to {months[1][:3]}, {first_year} to {last_year}", [rare_src],
                              value=len(stronger_season)))
        for code in touched:
            touching = [x for x in storms.values() if code in x["land"]]
            count_src = store.source(
                f"computed-land-{code.lower()}", kind="computed",
                title=f"Storm tracks that touched {names[code]}",
                method=(f"Counted IBTrACS main-track storms, seasons {first_year} to {last_year}, with at least one "
                        f"row over land (DIST2LAND 0) inside the Natural Earth 1:110m outline of {names[code]}."),
                code=f"{BUILDER}: tropical_cyclones", inputs=["dataset-ibtracs", "dataset-ne-countries"],
                result={"storms": len(touching), "years": [first_year, last_year]})
            store.places[f"country-{code.lower()}"].setdefault("counts", {})["tropical-cyclone"] = {
                "count": len(touching), "cite": [count_src],
                "text": f"{len(touching)} storm tracks in IBTrACS touched land here, {first_year} to {last_year}."}

        # the track, six-hourly plus the peak row
        points = []
        for i, r in enumerate(track):
            if r["ISO_TIME"][11:13] not in ("00", "06", "12", "18") and i != peak_i:
                continue
            if r["ISO_TIME"][14:16] != "00" and i != peak_i:
                continue
            points.append([num(r["LON"]), num(r["LAT"]), num(r["USA_WIND"]), r["ISO_TIME"][:16],
                           int(num(r["USA_SSHS"]) if num(r["USA_SSHS"]) is not None else -5)])

        cycle_time = (tp - timedelta(hours=24)).replace(hour=(tp.hour // 12) * 12, minute=0)
        cycle = cycle_time.strftime("%Y-%m-%dT%H")
        recipe_src = store.source(
            f"rule-recipe-tc", kind="rule", title="Seed recipe for a tropical cyclone",
            method=("Box 1500 by 1500 km centred on the row with the highest wind; a 12 km grid; start at the "
                    "00Z or 12Z cycle 24 hours before that row; 48 hours long; GFS for cycles from 2021-03-01 on, "
                    "ERA5 before."),
            code=f"{BUILDER}: tropical_cyclones", inputs=[])
        observed = [{"label": "IBTrACS storm page", "url": f"https://ncics.org/ibtracs/index.php?name=v04r01-{sid}",
                     "cite": src_first}]
        atcf = peak["USA_ATCF_ID"].strip()
        if atcf[:2] in ("AL", "EP") and name not in ("UNNAMED", "NOT_NAMED"):
            tcr = f"https://www.nhc.noaa.gov/data/tcr/{atcf}_{name.title()}.pdf"
            if url_ok(tcr):
                observed.append({"label": "NHC Tropical Cyclone Report (PDF)", "url": tcr, "cite": src_peak})

        month = tp.month
        summary = [
            {"text": f"{title_name} was a tropical cyclone in the "}, {"fact": "basin"},
            {"text": " basin. Its track runs from "}, {"fact": "start"}, {"text": " to "}, {"fact": "end"},
            {"text": ". "},
        ]
        if winds:
            summary += [{"text": "Its highest wind in the record is "}, {"fact": "peak_wind"},
                        {"text": ", at "}, {"fact": "peak_time"}, {"text": ". "}]
        if low is not None:
            summary += [{"text": "Its lowest pressure is "}, {"fact": "min_pressure"}, {"text": ". "}]
        if touched:
            summary += [{"text": "Its track crossed land in "}, {"fact": "land"}, {"text": "."}]

        store.events.append({
            "id": f"tc-{sid.lower()}", "type": "tropical-cyclone", "title": title, "seed": True,
            "added": RETRIEVED,
            "when": {"start": iso(t0), "end": iso(t1), "peak": iso(tp)},
            "where": {"lat": num(peak["LAT"]), "lon": num(peak["LON"]), "label": BASINS[basin]},
            "region": BASINS[basin], "decade": season // 10 * 10, "season": SEASONS[month],
            "places": [basin_place, *country_places],
            "rarity": rarity, "facts": facts,
            "summary": {"parts": summary, "generated": "seed-builder"},
            "geometry": {"track": points, "track_cite": track_src},
            "observed": observed,
            "era5": {"statistics": None, "quality": era_quality(season)},
            "recipe": {"source": recipe_source(cycle), "cycle": cycle, "hours": 48, "lat": num(peak["LAT"]),
                       "lon": ((num(peak["LON"]) + 180) % 360) - 180, "width_km": 1500, "height_km": 1500,
                       "dx_km": 12, "cite": [recipe_src]},
        })


# ------------------------------------------------------------------ tornadoes

def tornadoes(store: Store, cache: Path) -> None:
    path = fetch(cache, "spc")
    states = Outlines(fetch(cache, "ne-states"), lambda p: p.get("postal") or p.get("iso_3166_2"))
    state_names = {props.get("postal"): props["name"] for _, props, _, _ in states.items
                   if props.get("iso_a2") == "US" and props.get("postal")}
    counties = {}
    with fetch(cache, "census-counties").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="|"):
            counties[(row["STATEFP"], row["COUNTYFP"])] = row["COUNTYNAME"]
    fetch(cache, "spc-doc")
    all_rows = []
    wanted = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            all_rows.append(row)
            key = (row["yr"], row["om"])
            if key in TORNADO_ROWS:
                wanted[key] = row
    first_year = min(int(r["yr"]) for r in all_rows)
    last_year = max(int(r["yr"]) for r in all_rows)
    keep = ("om", "yr", "mo", "dy", "date", "time", "tz", "st", "stf", "mag", "inj", "fat", "slat", "slon", "elat",
            "elon", "len", "wid", "ns", "sn", "f1", "f2", "f3", "f4")
    country = store.place("country-usa", "United States of America", "country", ["dataset-ne-countries"])

    for key in TORNADO_ROWS:
        row = wanted.get(key)
        if row is None:
            raise SystemExit(f"SPC has no tornado {key}")
        yr, om = key
        src = store.source(f"spc-{yr}-{om}", kind="record", dataset="dataset-spc",
                           title=f"SPC tornado row: {row['date']} {row['time']}, {row['st']}, om {om}",
                           publisher=DATASETS["spc"]["publisher"], url=DATASETS["spc"]["url"],
                           licence=DATASETS["spc"]["licence"], row={k: row[k] for k in keep})
        st = row["st"]
        state_name = state_names.get(st, st)
        state_place = store.place(f"us-{st.lower()}", state_name, "state", ["dataset-ne-states"], parent=country)
        mag = int(row["mag"])
        scale = "EF" if row["date"] >= EF_START else "F"
        # tz 3 is Central Standard Time (SPC database description): six hours behind UTC
        tz_hours = 6 if row["tz"] == "3" else 0
        local = datetime.strptime(f"{row['date']} {row['time']}", "%Y-%m-%d %H:%M:%S")
        start = (local + timedelta(hours=tz_hours)).replace(tzinfo=timezone.utc)
        end = None
        if row.get("edat") and row.get("etime"):
            end = (datetime.strptime(f"{row['edat']} {row['etime']}", "%Y-%m-%d %H:%M:%S")
                   + timedelta(hours=tz_hours)).replace(tzinfo=timezone.utc)
        codes = []
        for k in ("f1", "f2", "f3", "f4"):
            code = row[k].strip().zfill(3)
            if code != "000" and code not in codes and (row["stf"].zfill(2), code) in counties:
                codes.append(code)
        names = [counties[(row["stf"].zfill(2), code)] for code in codes]
        county = (names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]) if names else None
        slat, slon, elat, elon = (float(row[k]) for k in ("slat", "slon", "elat", "elon"))
        has_end = elat != 0.0 and elon != 0.0
        end_state = None
        end_src = None
        if int(row["ns"]) > 1 and has_end:
            code, props = states.find(elon, elat)
            if code:
                end_state = props["name"]
                end_src = store.source(f"computed-spc-{yr}-{om}-end-state", kind="computed",
                                       title=f"State of the end point of SPC tornado {om}",
                                       method="The row's end point (elat, elon) placed in a Natural Earth 1:110m "
                                              "state outline.",
                                       code=f"{BUILDER}: tornadoes", inputs=[src, "dataset-ne-states"],
                                       result={"state": end_state})
                store.place(f"us-{code.lower()}", end_state, "state", ["dataset-ne-states"], parent=country)
        rating = f"{scale}{mag}"
        where_words = f"{county}, {state_name}" if county and len(codes) <= 2 else (f"{state_name} to {end_state}" if end_state else state_name)
        title = f"{rating} tornado, {where_words}, {day_words(local)}"
        county_srcs = [store.source(f"census-{row['stf'].zfill(2)}{code}", kind="record",
                                    dataset="dataset-census-counties",
                                    title=f"Census county code {row['stf'].zfill(2)}{code}",
                                    publisher=DATASETS["census-counties"]["publisher"],
                                    url=DATASETS["census-counties"]["url"],
                                    licence=DATASETS["census-counties"]["licence"],
                                    row={"STATEFP": row["stf"].zfill(2), "COUNTYFP": code,
                                         "COUNTYNAME": counties[(row["stf"].zfill(2), code)]})
                       for code in codes]
        facts = [
            fact("rating", "Rating", rating, [src, "dataset-spc-doc"], value=mag),
            fact("start", "Start", f"{start:%Y-%m-%d %H:%M} UTC ({row['time'][:5]} CST)", [src, "dataset-spc-doc"],
                 value=iso(start)),
        ]
        if end is not None:
            facts.append(fact("end", "End", f"{end:%Y-%m-%d %H:%M} UTC", [src, "dataset-spc-doc"], value=iso(end)))
        facts.append(fact("state", "Starts in", state_name, [src, "dataset-ne-states"], value=st))
        if county:
            facts.append(fact("county", "County" if len(codes) == 1 else "Counties", county, [src, *county_srcs]))
        if end_state:
            facts.append(fact("end_state", "Ends in", end_state, [end_src]))
        length_km = float(row["len"]) * 1.609344
        width_m = float(row["wid"]) * 0.9144
        facts += [
            fact("length", "Path length", f"{length_km:.1f} km ({float(row['len']):.1f} mi)", [src], value=length_km,
                 unit="km"),
            fact("width", "Path width", f"{width_m:.0f} m ({int(float(row['wid']))} yd)", [src], value=width_m,
                 unit="m"),
            fact("fatalities", "Deaths", row["fat"], [src], value=int(row["fat"])),
            fact("injuries", "Injuries", row["inj"], [src], value=int(row["inj"])),
            fact("start_point", "Start point", f"{slat:.2f}, {slon:.2f}", [src], infobox=False),
        ]

        season = SEASONS[local.month]
        same = [r for r in all_rows if r["st"] == st and int(r["mag"]) >= mag and SEASONS[int(r["mo"])] == season]
        any_season = [r for r in all_rows if r["st"] == st and int(r["mag"]) >= mag]
        in_state = [r for r in all_rows if r["st"] == st]
        rare_src = store.source(
            f"computed-spc-{yr}-{om}-rarity", kind="computed", title=f"Rarity of SPC tornado {om} in {state_name}",
            method=(f"Counted SPC tornado rows, {first_year} to {last_year}, starting in {state_name} (st {st}) with "
                    f"mag at least {mag}, all year and in {SEASON_WORDS[season]}. Unknown ratings (mag -9) are left out."),
            code=f"{BUILDER}: tornadoes", inputs=["dataset-spc"],
            result={"at_least_this_strong_in_season": len(same), "at_least_this_strong_all_year": len(any_season),
                    "tornadoes_in_state": len(in_state), "years": [first_year, last_year]})
        facts.append(fact("rarity", "At least this strong here",
                          f"{len(same):,} of {len(in_state):,} tornadoes, {SEASON_WORDS[season]}, {first_year} to "
                          f"{last_year}", [rare_src], value=len(same)))
        counts = store.places[state_place].setdefault("counts", {})
        if "tornado" not in counts:
            count_src = store.source(f"computed-spc-count-{st.lower()}", kind="computed",
                                     title=f"SPC tornadoes starting in {state_name}",
                                     method=f"Counted SPC tornado rows starting in {state_name}, {first_year} to "
                                            f"{last_year}.",
                                     code=f"{BUILDER}: tornadoes", inputs=["dataset-spc"],
                                     result={"tornadoes": len(in_state)})
            counts["tornado"] = {"count": len(in_state), "cite": [count_src],
                                 "text": f"{len(in_state):,} tornadoes in the SPC table start here, {first_year} to "
                                         f"{last_year}."}

        cycle_time = start - timedelta(hours=6)
        cycle_time = cycle_time.replace(hour=(cycle_time.hour // 6) * 6, minute=0, second=0)
        finish = end or start
        hours = max(12, math.ceil((finish - cycle_time).total_seconds() / 3600) + 3)
        clat = (slat + elat) / 2 if has_end else slat
        clon = (slon + elon) / 2 if has_end else slon
        span_km = 0.0
        if has_end:
            span_km = max(abs(elat - slat) * 111.32, abs(elon - slon) * 111.32 * math.cos(math.radians(clat)))
        box = int(max(600, math.ceil((span_km + 300) / 100) * 100))
        cycle = cycle_time.strftime("%Y-%m-%dT%H")
        recipe_src = store.source(
            "rule-recipe-tornado", kind="rule", title="Seed recipe for a tornado",
            method=("A square box centred on the middle of the path, 600 km or the path's extent plus 300 km, "
                    "whichever is larger; a 3 km grid; start at the 00Z, 06Z, 12Z or 18Z cycle at least 6 hours "
                    "before the tornado; long enough to pass its end by 3 hours, at least 12 hours; GFS for cycles "
                    "from 2021-03-01 on, ERA5 before."),
            code=f"{BUILDER}: tornadoes", inputs=[])
        fips = f"{int(row['stf'])}%2C{state_name.upper().replace(' ', '+')}"
        stormevents = ("https://www.ncdc.noaa.gov/stormevents/listevents.jsp?eventType=%28C%29+Tornado"
                       f"&beginDate_mm={local:%m}&beginDate_dd={local:%d}&beginDate_yyyy={local:%Y}"
                       f"&endDate_mm={local:%m}&endDate_dd={local:%d}&endDate_yyyy={local:%Y}"
                       f"&county=ALL&hailfilter=0.00&tornfilter=0&windfilter=000&sort=DT&submitbutton=Search"
                       f"&statefips={fips}")
        observed = [{"label": "SPC tornado table (CSV)", "url": DATASETS["spc"]["url"], "cite": src},
                    {"label": "NCEI Storm Events Database, this day and state", "url": stormevents, "cite": src}]
        summary = [{"text": "This tornado was rated "}, {"fact": "rating"}, {"text": ". It started in "},
                   {"fact": "county" if county else "state"}]
        if end_state:
            summary += [{"text": " and ended in "}, {"fact": "end_state"}]
        summary += [{"text": " at "}, {"fact": "start"}, {"text": ". Its path was "}, {"fact": "length"},
                    {"text": " long and "}, {"fact": "width"}, {"text": " wide. Deaths in the record: "},
                    {"fact": "fatalities"}, {"text": "; injuries: "}, {"fact": "injuries"}, {"text": "."}]
        places = [state_place]
        if end_state:
            places.append(f"us-{[c for c, n in state_names.items() if n == end_state][0].lower()}")
        places.append(country)
        store.events.append({
            "id": f"tornado-{yr}-{om.lower()}", "type": "tornado", "title": title, "seed": True, "added": RETRIEVED,
            "when": {"start": iso(start), "end": iso(end) if end else iso(start), "peak": iso(start)},
            "where": {"lat": slat, "lon": slon, "label": where_words},
            "region": state_name, "decade": int(yr) // 10 * 10, "season": season,
            "places": places,
            "rarity": {"count": len(same), "of": len(in_state), "place": state_place,
                       "text": (f"{plural(len(same), 'tornado', 'tornadoes')} rated {scale}{mag} or higher started in {state_name} in "
                                f"{SEASON_WORDS[season]}, {first_year} to {last_year} ({len(any_season):,} in any "
                                f"month, of {len(in_state):,} tornadoes)."),
                       "cite": [rare_src]},
            "facts": facts,
            "summary": {"parts": summary, "generated": "seed-builder"},
            "geometry": {"path": [[slon, slat], [elon, elat]] if has_end else [[slon, slat]], "path_cite": src},
            "observed": observed,
            "era5": {"statistics": None, "quality": era_quality(int(yr))},
            "recipe": {"source": recipe_source(cycle), "cycle": cycle, "hours": hours, "lat": round(clat, 2),
                       "lon": round(clon, 2), "width_km": box, "height_km": box, "dx_km": 3, "cite": [recipe_src]},
        })


# ------------------------------------------------------------------ phenomena

def phenomena(store: Store) -> list[dict]:
    out = []
    for ident, (word, text) in GLOSSARY.items():
        src = store.source(f"nws-glossary-{ident}", kind="record", dataset="dataset-nws-glossary",
                           title=f"NWS glossary: {word}",
                           publisher=DATASETS["nws-glossary"]["publisher"],
                           url=f"https://forecast.weather.gov/glossary.php?word={word.replace(' ', '+')}",
                           licence=DATASETS["nws-glossary"]["licence"], row={"word": word, "definition": text})
        if ident == "tornado":
            rule = store.source("rule-select-tornado", kind="rule", title="How tornadoes are chosen for the Weather Library",
                                method="Rows of the SPC tornado table picked one by one when the Weather Library was built. Rarity counts "
                                       "tornadoes that start in the same state, in the same three-month season, "
                                       "rated at least as high.",
                                code=f"{BUILDER}: tornadoes", inputs=["dataset-spc"])
            detect = "These pages come from rows of the SPC tornado table, the United States' record of every tornado since 1950. More will come from storm setups found in ERA5."
        else:
            rule = store.source("rule-select-tc", kind="rule", title="How tropical cyclones are chosen for the Weather Library",
                                method="Storms of IBTrACS since 1980 picked one by one when the Weather Library was built, one or more "
                                       "from each basin. Rarity counts storms of the same basin whose highest wind "
                                       "is at least as high, peaking within a month of the same time of year.",
                                code=f"{BUILDER}: tropical_cyclones", inputs=["dataset-ibtracs"])
            detect = "These pages come from IBTrACS, the world's collection of cyclone best tracks. More will come from cyclones found in ERA5."
        out.append({"id": ident, "title": word.capitalize() if ident == "tornado" else "Tropical cyclone",
                    "plural": "Tornadoes" if ident == "tornado" else "Tropical cyclones",
                    "what": {"quote": text, "cite": [src]},
                    "detect": {"text": detect, "cite": [rule]}, "seed": True})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, required=True, help="where the downloaded records are kept")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[2] / "gpuwm" / "gui" / "seed" / "wiki-seed.json")
    args = parser.parse_args(argv)
    args.cache.mkdir(parents=True, exist_ok=True)
    store = Store()
    tropical_cyclones(store, args.cache)
    tornadoes(store, args.cache)
    kinds = phenomena(store)
    document = {
        "schema": "gpuwm.wiki.v1",
        "origin": "seed",
        "about": "Seed pages built from open records by " + BUILDER + ". Every fact names its record.",
        "built": RETRIEVED,
        "builder": BUILDER,
        "sources": dict(sorted(store.sources.items())),
        "phenomena": kinds,
        "places": sorted(store.places.values(), key=lambda p: p["id"]),
        "events": store.events,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
                        encoding="utf-8", newline="\n")
    print(f"{args.out}: {len(store.events)} events, {len(store.places)} places, {len(store.sources)} sources")
    return 0


if __name__ == "__main__":
    sys.exit(main())
