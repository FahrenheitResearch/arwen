"""Source rows, saved HTTP indexes, and real Rust decode/remap of a tiny tape."""
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm import chem_table, chem_source_netcdf as source, obs_regrid_bridge as bridge
from gpuwm.io import nc_writer_bridge as writer
from gpuwm import netcdf_bridge
from gpuwm.verify.cell_sum_ref import ulp_table


def test_rows_load_through_catalog():
    catalog = chem_table.catalog()
    for name in ("rave-3km", "rave-13km"):
        row = catalog.sources[name]
        assert row.kind == "emission" and row.credential is None
        assert row.variables["PM25"]["selector"] == "PM25"


def test_saved_listing_resolution():
    row = chem_table.catalog().sources["rave-3km"]
    text = (Path(__file__).parent / "data/chem/source-listing.html").read_text()
    entries = source.parse_listing(text, row.grid["listing"])
    hour = datetime(2025,1,8,6,tzinfo=timezone.utc)
    assert source.resolve_hour(row,hour,entries).endswith("c202501080802200.nc")
    assert max(entries) == hour
    with pytest.raises(source.EmissionSourceError, match="missing observed_hourly file"):
        source.resolve_hour(row,hour.replace(hour=7),entries)


def test_listing_newest_creation_and_untrusted_links():
    row = chem_table.catalog().sources["rave-3km"]
    stem = "RAVE-HrlyEmiss-3km_v2r0_blend_s202501080600000_e202501080659590_"
    older, newer = stem+"c202501080802200.nc", stem+"c202501080902200.nc"
    text = "".join(f'<a href="{name}">file</a>' for name in (older,newer,"https://invalid/"+newer,"../"+newer))
    assert list(source.parse_listing(text,row.grid["listing"]).values()) == [newer]


def test_newest_posted_month_boundary(monkeypatch):
    row = chem_table.catalog().sources["rave-3km"]
    last = datetime(2025,1,31,23,tzinfo=timezone.utc)
    def listing(row,hour,cache,refresh):
        return "https://invalid/", {last:"file.nc"} if hour.month == 1 else {}
    monkeypatch.setattr(source,"listing",listing)
    assert source.newest_posted_hour(row,"unused",now=datetime(2025,2,1,tzinfo=timezone.utc)) == last


def lambert_grid(dx, ny, nx, lat0=30.005, lon0=100.003):
    """Spherical tangent Lambert, inverse at mass centres, test geometry only."""
    radius = 6370000.0
    phi = np.deg2rad(lat0)
    cone = np.sin(phi)
    factor = np.cos(phi)*np.tan(np.pi/4+phi/2)**cone/cone
    rho0 = radius*factor/np.tan(np.pi/4+phi/2)**cone
    y,x = np.meshgrid((np.arange(ny)-(ny-1)/2)*dx,
                       (np.arange(nx)-(nx-1)/2)*dx,indexing="ij")
    rho = np.hypot(x,rho0-y)
    lat = 2*np.arctan((radius*factor/rho)**(1/cone))-np.pi/2
    lon = np.deg2rad(lon0)+np.arctan2(x,rho0-y)/cone
    mapfac = cone*rho/(radius*np.cos(lat))
    return np.rad2deg(lat),np.rad2deg(lon),mapfac


def test_synthetic_netcdf_mask_and_unreachable_mass(tmp_path):
    try:
        bridge._sum_library()
        writer.load()
        netcdf_bridge.resolve_netcdf_bin()
    except (RuntimeError, OSError) as error:
        pytest.skip(f"Rust NetCDF writer/decoder and cell-sum library required: {error}")
    schema = writer.ClassicSchema()
    dims = [schema.def_dim("y",2),schema.def_dim("x",3)]
    arrays = {}
    lat,lon = np.meshgrid(30+np.arange(2)*0.01,100+np.arange(3)*0.01,indexing="ij")
    for name,values,units in [("lat",lat,"degrees_north"),("lon",lon,"degrees_east"),
        ("mass",np.array([[5.,3.,2.],[7.,-1.,11.]]),"kg"),
        ("quality",np.array([[3.,1.,3.],[3.,-1.,3.]]),"1")]:
        var = schema.def_var(name,"f8",dims)
        schema.put_var_attr(var,"units",units)
        arrays[var] = values
    path = tmp_path/"synthetic.nc"
    with schema.create(path) as tape:
        for var,values in arrays.items():
            tape.write_var(var,values)
    row = SimpleNamespace(kind="emission",format="netcdf",remap="cell_sum_split",
        grid=dict(latitude="lat",longitude="lon"),time=dict(cadence_s=3600),
        variables=dict(mass=dict(selector="mass",units="kg",no_fire_sentinel=-1,
            masks=[dict(selector="quality",minimum=2,maximum=3,nonzero_only=True)])))
    dlat,dlon,mapfac = lambert_grid(750,7,3)
    fields,receipts = source.ingest(row,path,["mass"],latitude=dlat,longitude=dlon,
        map_factors=mapfac,dx_m=750,max_distance_m=600)
    receipt = receipts["mass"]
    assert receipt["split_n"] == 2
    assert receipt["total_source_mass"] == 28.0
    assert receipt["masked_mass"] == 3.0
    assert receipt["unreachable_mass"] == 13.0
    assert receipt["total_remapped_mass"] == 12.0
    totals = np.array([receipt[name] for name in
                      ("total_source_mass","total_remapped_mass","unreachable_mass","masked_mass")])
    assert ulp_table(totals,np.array([28.,12.,13.,3.])) == dict(max_ulp=0,nonzero=0,count=4)
    assert fields["mass"].sum() == 12.0
    assert receipt["source_sha256"] == source.file_sha256(path)
    assert not receipt["valid"].all()


def test_window_keeps_a_model_edge_west_of_the_first_source_column():
    # The model's west edge (99.995 E) lies just west of the source's first
    # column (100.0 E).  Unwrapping around the source's minimum longitude put
    # that edge at 459.995 and disabled the window; around the span's middle
    # it stays at 99.995.
    lat, lon = np.meshgrid(30 + np.arange(2) * 0.01, 100 + np.arange(3) * 0.01,
                           indexing="ij")
    dlat = np.linspace(29.98, 30.03, 21).reshape(7, 3)
    dlon = np.tile([99.995, 100.003, 100.011], (7, 1))
    assert source._window(lat, lon, dlat, dlon, 600) == (slice(0, 2), slice(0, 3))


def test_window_finds_a_west_longitude_model_on_a_0_to_360_source():
    # RAVE's grid runs 144.975 to 332.145 degrees east; a model given in
    # -180..180 (here 118.5 W) must land on the columns near 241.5 E.
    lat, lon = np.meshgrid(81.785 - np.arange(2610) * 0.03,
                           144.975 + np.arange(6240) * 0.03, indexing="ij")
    dlat = np.array([[33.0, 33.0], [35.0, 35.0]])
    dlon = np.array([[-120.0, -117.0], [-120.0, -117.0]])
    rows, cols = source._window(lat, lon, dlat, dlon, 3000.0)
    assert lat[rows, 0].min() < 33.0 < 35.0 < lat[rows, 0].max()
    assert lon[0, cols].min() < 240.0 < 243.0 < lon[0, cols].max()
    assert cols.stop - cols.start < 200 and rows.stop - rows.start < 200


def test_window_is_empty_for_a_model_the_source_cannot_reach():
    lat, lon = np.meshgrid(30 + np.arange(4) * 0.03, 100 + np.arange(4) * 0.03,
                           indexing="ij")
    window = source._window(lat, lon, np.array([[-40.0]]), np.array([[20.0]]),
                            3000.0)
    assert window == (slice(0, 0), slice(0, 0))


def _frames(ingested):
    from gpuwm.chem_emission_frames import EmissionFrames
    table = chem_table.load_sets(("smoke",), ("rave-3km",))
    def fetch(row, hour, cache):
        return Path(f"{row.name}-{hour:%Y%m%d%H}.nc"), {}
    def ingest(row, path, fields, **_):
        ingested.append((path.name, tuple(fields)))
        return ({name: np.full((2, 3), float(len(ingested)))
                 for name in fields}, {name: {} for name in fields})
    return EmissionFrames(table, latitude=np.zeros((2, 3)),
                          longitude=np.zeros((2, 3)),
                          map_factors=np.ones((2, 3)), dx_m=3000.0,
                          reference_time=datetime(2025, 1, 8, 6,
                                                  tzinfo=timezone.utc),
                          fetch=fetch, ingest=ingest)


def test_an_hour_is_ingested_once_for_its_emission_and_fire_power():
    # The fire process asks for PM25, then its fire power FRP_MEAN, then
    # PM25 again every step; the file is decoded and remapped once.
    ingested = []
    frames = _frames(ingested)
    hour = datetime(2025, 1, 8, 7, 30, tzinfo=timezone.utc)
    for _ in range(3):
        frames.at("rave-3km", "PM25", hour)
        frames.at("rave-3km", "FRP_MEAN", hour)
    assert ingested == [("rave-3km-2025010807.nc", ("FRP_MEAN", "PM25"))]
    frames.at("rave-3km", "PM25", hour.replace(hour=8))
    assert len(ingested) == 2


def test_a_source_left_out_of_chem_sources_is_refused_by_the_frames():
    from gpuwm.chem_emission_frames import EmissionFrameError, EmissionFrames
    table = chem_table.load_sets(("smoke",), ())
    frames = EmissionFrames(table, latitude=np.zeros((1, 1)),
                            longitude=np.zeros((1, 1)),
                            map_factors=np.ones((1, 1)), dx_m=3000.0,
                            reference_time=datetime(2025, 1, 8, 6,
                                                    tzinfo=timezone.utc))
    with pytest.raises(EmissionFrameError, match="not enabled in chem_sources"):
        frames.at("rave-3km", "PM25", datetime(2025, 1, 8, 6,
                                              tzinfo=timezone.utc))


def test_the_next_posted_hour_is_read_ahead_and_served_from_the_read():
    ingested = []
    frames = _frames(ingested)
    frames.end_time = datetime(2025, 1, 8, 9, tzinfo=timezone.utc)
    posted = {datetime(2025, 1, 8, h, tzinfo=timezone.utc) for h in (6, 7, 8, 9, 10)}
    frames._posted["rave-3km"] = frozenset(posted)
    first = frames.at("rave-3km", "PM25", datetime(2025, 1, 8, 6, tzinfo=timezone.utc))
    pending = frames._pending[("rave-3km", datetime(2025, 1, 8, 7, tzinfo=timezone.utc))]
    pending.result(timeout=30)
    assert [name for name, _ in ingested] == ["rave-3km-2025010806.nc", "rave-3km-2025010807.nc"]
    second = frames.at("rave-3km", "FRP_MEAN", datetime(2025, 1, 8, 7, 59, tzinfo=timezone.utc))
    # Served from the read-ahead (value 2: the second ingest), not read again.
    assert float(first[0, 0]) == 1.0 and float(second[0, 0]) == 2.0
    assert ("rave-3km", datetime(2025, 1, 8, 8, tzinfo=timezone.utc)) in frames._pending
    frames._pending[("rave-3km", datetime(2025, 1, 8, 8, tzinfo=timezone.utc))].result(timeout=30)
    frames.at("rave-3km", "PM25", datetime(2025, 1, 8, 8, tzinfo=timezone.utc))
    frames._pending[("rave-3km", datetime(2025, 1, 8, 9, tzinfo=timezone.utc))].result(timeout=30)
    frames.at("rave-3km", "PM25", datetime(2025, 1, 8, 9, tzinfo=timezone.utc))
    # 10 Z is posted but past the run's end: never read.
    assert len(ingested) == 4 and not frames._pending


def test_a_failed_read_ahead_is_read_again_on_request():
    ingested = []
    frames = _frames(ingested)
    frames.end_time = datetime(2025, 1, 8, 9, tzinfo=timezone.utc)
    frames._posted["rave-3km"] = frozenset(
        datetime(2025, 1, 8, h, tzinfo=timezone.utc) for h in (6, 7))
    good = frames._ingest
    calls = []
    def flaky(row, path, fields, **kw):
        calls.append(path.name)
        if path.name.endswith("07.nc") and calls.count(path.name) == 1:
            raise OSError("transient")
        return good(row, path, fields, **kw)
    frames._ingest = flaky
    frames.at("rave-3km", "PM25", datetime(2025, 1, 8, 6, tzinfo=timezone.utc))
    frames._pending[("rave-3km", datetime(2025, 1, 8, 7, tzinfo=timezone.utc))].exception(timeout=30)
    frames.at("rave-3km", "PM25", datetime(2025, 1, 8, 7, tzinfo=timezone.utc))
    assert calls == ["rave-3km-2025010806.nc", "rave-3km-2025010807.nc", "rave-3km-2025010807.nc"]
