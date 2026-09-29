"""The landfall each seed cyclone's recipe is timed around, from the IBTrACS rows the seed was built from.

Rule (table data, no storm is special-cased): the key landfall is the first IBTrACS main-track row at or
after six hours before the peak whose centre is within CORE_KM of land (DIST2LAND; the wiki's "Track over
land in" fact uses DIST2LAND 0, and a centre passing 20 km off a coast puts the eyewall ashore), or whose
USA_RECORD is L (a US agency's landfall row). A storm with no such row has none.

Usage: python landfall_table.py <ibtracs.since1980.list.v04r01.csv> <wiki-seed.json> > landfall.json
"""
import csv, json, sys
from datetime import datetime, timedelta

BEFORE_PEAK_H = 6
#: km from the centre to land within which the storm core is counted ashore.
CORE_KM = 25
URL = ("https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
       "v04r01/access/csv/ibtracs.since1980.list.v04r01.csv")
ROW_KEYS = ("SID", "SEASON", "BASIN", "NAME", "ISO_TIME", "NATURE", "LAT", "LON", "DIST2LAND", "USA_RECORD",
            "USA_AGENCY", "USA_WIND", "USA_PRES", "USA_SSHS")


def main(csv_path, seed_path):
    seed = json.load(open(seed_path, encoding="utf-8"))
    tcs = {e["id"].split("-")[1].upper(): e for e in seed["events"] if e["type"] == "tropical-cyclone"}
    rows = {k: [] for k in tcs}
    with open(csv_path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        next(r)
        for x in r:
            if x["SID"] in rows and x["TRACK_TYPE"] == "main":
                rows[x["SID"]].append(x)
    out = {}
    for sid, e in sorted(tcs.items()):
        peak = datetime.strptime(e["when"]["peak"][:16], "%Y-%m-%dT%H:%M")
        first = None
        for x in rows[sid]:
            when = datetime.strptime(x["ISO_TIME"][:16], "%Y-%m-%d %H:%M")
            if when < peak - timedelta(hours=BEFORE_PEAK_H):
                continue
            d2l = int(x["DIST2LAND"]) if x["DIST2LAND"].strip() else 9999
            if d2l <= CORE_KM or x["USA_RECORD"].strip() == "L":
                first = {"time": when.strftime("%Y-%m-%dT%H:%MZ"), "lat": float(x["LAT"]), "lon": float(x["LON"]),
                         "row": f"DIST2LAND {d2l} km" + (", USA_RECORD L" if x["USA_RECORD"].strip() == "L" else ""),
                         "cite": f"ibtracs-{sid}-{when.strftime('%Y-%m-%dT%H')}",
                         # the same record shape the seed builder writes for a cited IBTrACS row
                         "source": {"kind": "record", "dataset": "dataset-ibtracs",
                                    "title": f"IBTrACS row: storm {sid}, {x['ISO_TIME'][:16]} UTC (key landfall)",
                                    "publisher": "NOAA National Centers for Environmental Information",
                                    "licence": "US Government work, no copyright in the US. NCEI asks users to "
                                               "cite the paper and the dataset.",
                                    "retrieved": "2026-09-25", "url": URL,
                                    "row": {k: x[k] for k in ROW_KEYS}}}
                break
        out[e["id"]] = {"sid": sid, "peak": e["when"]["peak"], "landfall": first}
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
