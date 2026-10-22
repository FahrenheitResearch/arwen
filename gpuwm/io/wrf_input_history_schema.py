"""Native WRF input records carried into history without arithmetic.

Rows are pinned to WRF 4.6.1 Registry declarations; dimensions distinguish
input scalars from named soil, mass and staggered coordinate axes.
"""


def _history_row(description, units, registry, dimensions=(), *,
                 dtype="f4", stagger="", input_dimensions=None):
    """Give each row its own lists, including when the two axes agree."""
    return {
        "dtype": dtype,
        "stagger": stagger,
        "description": description,
        "units": units,
        "registry": registry,
        "input_dimensions": list(dimensions if input_dimensions is None
                                 else input_dimensions),
        "history_dimensions": list(dimensions),
    }


WRF_INPUT_HISTORY_ROWS = {
    'SHDMAX': _history_row(
        'ANNUAL MAX VEG FRACTION', '', 'Registry.EM_COMMON:768',
        ('south_north', 'west_east')),
    'SHDMIN': _history_row(
        'ANNUAL MIN VEG FRACTION', '', 'Registry.EM_COMMON:769',
        ('south_north', 'west_east')),
    'LAKEMASK': _history_row(
        'LAKE MASK (1 FOR LAKE, 0 FOR NON-LAKE)', '', 'Registry.tladj:10',
        ('south_north', 'west_east')),
    'ZS': _history_row(
        'DEPTHS OF CENTERS OF SOIL LAYERS', 'm', 'Registry.EM_COMMON:68',
        ('soil_layers_stag',), stagger='Z'),
    'DZS': _history_row(
        'THICKNESSES OF SOIL LAYERS', 'm', 'Registry.EM_COMMON:69',
        ('soil_layers_stag',), stagger='Z'),
    'VAR_SSO': _history_row(
        'variance of subgrid-scale orography', 'm2', 'Registry.EM_COMMON:91',
        ('south_north', 'west_east')),
    'BATHYMETRY_FLAG': _history_row(
        'Flag for bathymetry in the global attributes for metgrid data', '', 'Registry.EM_COMMON:148',
        dtype='i4'),
    'FNM': _history_row(
        'upper weight for vertical stretching', '', 'Registry.EM_COMMON:349',
        ('bottom_top',)),
    'FNP': _history_row(
        'lower weight for vertical stretching', '', 'Registry.EM_COMMON:350',
        ('bottom_top',)),
    'RDNW': _history_row(
        'inverse d(eta) values between full (w) levels', '', 'Registry.EM_COMMON:351',
        ('bottom_top',)),
    'RDN': _history_row(
        'inverse d(eta) values between half (mass) levels', '', 'Registry.EM_COMMON:352',
        ('bottom_top',)),
    'DNW': _history_row(
        'd(eta) values between full (w) levels', '', 'Registry.EM_COMMON:353',
        ('bottom_top',)),
    'DN': _history_row(
        'd(eta) values between half (mass) levels', '', 'Registry.EM_COMMON:354',
        ('bottom_top',)),
    'CFN': _history_row(
        'extrapolation constant', '', 'Registry.EM_COMMON:358'),
    'CFN1': _history_row(
        'extrapolation constant', '', 'Registry.EM_COMMON:359'),
    'THIS_IS_AN_IDEAL_RUN': _history_row(
        'T/F flag: this is an ARW ideal simulation', '', 'Registry.EM_COMMON:363',
        dtype='i4'),
    'RDX': _history_row(
        'INVERSE X GRID LENGTH', 'm-1', 'Registry.EM_COMMON:397'),
    'RDY': _history_row(
        'INVERSE Y GRID LENGTH', 'm-1', 'Registry.EM_COMMON:398'),
    'ZETATOP': _history_row(
        'ZETA AT MODEL TOP', '', 'Registry.EM_COMMON:404'),
    'CF1': _history_row(
        '2nd order extrapolation constant', '', 'Registry.EM_COMMON:405'),
    'CF2': _history_row(
        '2nd order extrapolation constant', '', 'Registry.EM_COMMON:406'),
    'CF3': _history_row(
        '2nd order extrapolation constant', '', 'Registry.EM_COMMON:407'),
    'SHDAVG': _history_row(
        'ANNUAL AVG VEG FRACTION', '', 'Registry.EM_COMMON:770',
        ('south_north', 'west_east')),
    'WATER_DEPTH': _history_row(
        'global water depth', 'm', 'Registry.EM_COMMON:882',
        ('south_north', 'west_east')),
    'VAR': _history_row(
        'STANDARD DEVIATION OF SUBGRID-SCALE OROGRAPHY', 'm', 'Registry.EM_COMMON:1163',
        ('south_north', 'west_east')),
    'MAPFAC_MX': _history_row(
        'Map scale factor on mass grid, x direction', '', 'Registry.EM_COMMON:1395',
        ('south_north', 'west_east')),
    'MAPFAC_MY': _history_row(
        'Map scale factor on mass grid, y direction', '', 'Registry.EM_COMMON:1396',
        ('south_north', 'west_east')),
    'MAPFAC_UX': _history_row(
        'Map scale factor on u-grid, x direction', '', 'Registry.EM_COMMON:1397',
        ('south_north', 'west_east_stag'), stagger='X'),
    'MAPFAC_UY': _history_row(
        'Map scale factor on u-grid, y direction', '', 'Registry.EM_COMMON:1398',
        ('south_north', 'west_east_stag'), stagger='X'),
    'MAPFAC_VX': _history_row(
        'Map scale factor on v-grid, x direction', '', 'Registry.EM_COMMON:1399',
        ('south_north_stag', 'west_east'), stagger='Y'),
    'MF_VX_INV': _history_row(
        'Inverse map scale factor on v-grid, x direction', '', 'Registry.EM_COMMON:1400',
        ('south_north_stag', 'west_east'), stagger='Y'),
    'MAPFAC_VY': _history_row(
        'Map scale factor on v-grid, y direction', '', 'Registry.EM_COMMON:1401',
        ('south_north_stag', 'west_east'), stagger='Y'),
    'GOT_VAR_SSO': _history_row(
        'whether VAR_SSO was included in WPS output (beginning V3.4)', '', 'Registry.EM_COMMON:1439',
        dtype='i4'),
    'T00': _history_row(
        'BASE STATE TEMPERATURE   ', 'K', 'Registry.EM_COMMON:1477'),
    'P00': _history_row(
        'BASE STATE PRESSURE', 'Pa', 'Registry.EM_COMMON:1478'),
    'TLP': _history_row(
        'BASE STATE LAPSE RATE    ', '', 'Registry.EM_COMMON:1479'),
    'TISO': _history_row(
        'TEMP AT WHICH THE BASE T TURNS CONST', 'K', 'Registry.EM_COMMON:1480'),
    'TLP_STRAT': _history_row(
        'BASE STATE LAPSE RATE (DT/D(LN(P)) IN STRATOSPHERE', 'K', 'Registry.EM_COMMON:1481'),
    'P_STRAT': _history_row(
        'BASE STATE PRESSURE AT BOTTOM OF STRATOSPHERE', 'Pa', 'Registry.EM_COMMON:1482'),
    'CLAT': _history_row(
        'COMPUTATIONAL GRID LATITUDE, SOUTH IS NEGATIVE', 'degree_north', 'Registry.EM_COMMON:1846',
        ('south_north', 'west_east')),
    'C1H': _history_row(
        'half levels, c1h = d bf / d eta, using znw', 'Dimensionless', 'registry.hyb_coord:43',
        ('bottom_top',)),
    'C2H': _history_row(
        'half levels, c2h = (1-c1h)*(p0-pt)', 'Pa', 'registry.hyb_coord:44',
        ('bottom_top',)),
    'C1F': _history_row(
        'full levels, c1f = d bf / d eta, using znu', 'Dimensionless', 'registry.hyb_coord:46',
        ('bottom_top_stag',), stagger='Z'),
    'C2F': _history_row(
        'full levels, c2f = (1-c1f)*(p0-pt)', 'Pa', 'registry.hyb_coord:47',
        ('bottom_top_stag',), stagger='Z'),
    'C3H': _history_row(
        'half levels, c3h = bh', 'Dimensionless', 'registry.hyb_coord:49',
        ('bottom_top',)),
    'C4H': _history_row(
        'half levels, c4h = (eta-bh)*(p0-pt), using znu', 'Pa', 'registry.hyb_coord:50',
        ('bottom_top',)),
    'C3F': _history_row(
        'full levels, c3f = bf', 'Dimensionless', 'registry.hyb_coord:52',
        ('bottom_top_stag',), stagger='Z'),
    'C4F': _history_row(
        'full levels, c4f = (eta-bf)*(p0-pt), using znw', 'Pa', 'registry.hyb_coord:53',
        ('bottom_top_stag',), stagger='Z'),
    'PCB': _history_row(
        'base state dry air mass in column', 'Pa', 'registry.hyb_coord:55',
        ('south_north', 'west_east')),
}

WRF_INITIALIZED_HISTORY_ROWS = {
    'NEST_POS': _history_row(
        '', '', 'Registry.EM_COMMON:306',
        ('south_north', 'west_east'), input_dimensions=()),
    'AREA2D': _history_row(
        'Horizontal grid cell area, using dx, dy, and map factors', 'm2', 'Registry.EM_COMMON:399',
        ('south_north', 'west_east'), input_dimensions=()),
    'DX2D': _history_row(
        'Horizontal grid distance: sqrt(area2d)', 'm', 'Registry.EM_COMMON:400',
        ('south_north', 'west_east'), input_dimensions=()),
    'SNOALB': _history_row(
        'ANNUAL MAX SNOW ALBEDO IN FRACTION', '', 'Registry.EM_COMMON:771',
        ('south_north', 'west_east'), input_dimensions=()),
    'MAX_MSFTX': _history_row(
        'Max map factor in domain', '', 'Registry.EM_COMMON:1483'),
    'MAX_MSFTY': _history_row(
        'Max map factor in domain', '', 'Registry.EM_COMMON:1484'),
    'SAVE_TOPO_FROM_REAL': _history_row(
        '1=original topo from real/0=topo modified by WRF', 'flag', 'Registry.EM_COMMON:2007',
        dtype='i4'),
    'SST_INPUT': _history_row(
        'SEA SURFACE TEMPERATURE FROM WRFLOWINPUT FILE', 'K', 'Registry.EM:16',
        ('south_north', 'west_east'), input_dimensions=()),
    'ISEEDARR_SPPT': _history_row(
        'Array to hold seed for restart, SPPT', '', 'registry.stoch:60',
        ('seed_dim_stag',), dtype='i4', stagger='Z', input_dimensions=()),
    'ISEEDARR_SKEBS': _history_row(
        'Array to hold seed for restart, SKEBS', '', 'registry.stoch:61',
        ('seed_dim_stag',), dtype='i4', stagger='Z', input_dimensions=()),
    'ISEEDARR_RAND_PERTURB': _history_row(
        'Array to hold seed for restart, RAND_PERT', '', 'registry.stoch:62',
        ('seed_dim_stag',), dtype='i4', stagger='Z', input_dimensions=()),
    'ISEEDARRAY_SPP_CONV': _history_row(
        'Array to hold seed for restart, RAND_PERT2', '', 'registry.stoch:63',
        ('seed_dim_stag',), dtype='i4', stagger='Z', input_dimensions=()),
    'ISEEDARRAY_SPP_PBL': _history_row(
        'Array to hold seed for restart, RAND_PERT3', '', 'registry.stoch:64',
        ('seed_dim_stag',), dtype='i4', stagger='Z', input_dimensions=()),
    'ISEEDARRAY_SPP_LSM': _history_row(
        'Array to hold seed for restart, RAND_PERT4', '', 'registry.stoch:65',
        ('seed_dim_stag',), dtype='i4', stagger='Z', input_dimensions=()),
}


# GNU Fortran intrinsic RANDOM_SEED(SIZE=) used by stock WRF 4.6.1.
WRF_GNU_RANDOM_SEED_WORDS = 8
