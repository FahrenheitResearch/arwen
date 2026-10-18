"""Native WRF input records carried into history without arithmetic.

Rows are pinned to WRF 4.6.1 Registry declarations; dimensions distinguish
input scalars from named soil, mass and staggered coordinate axes.
"""
import json

WRF_INPUT_HISTORY_ROWS = json.loads(r'''
{
  "SHDMAX": {
    "dtype": "f4",
    "stagger": "",
    "description": "ANNUAL MAX VEG FRACTION",
    "units": "",
    "registry": "Registry.EM_COMMON:768",
    "input_dimensions": [
      "south_north",
      "west_east"
    ],
    "history_dimensions": [
      "south_north",
      "west_east"
    ]
  },
  "SHDMIN": {
    "dtype": "f4",
    "stagger": "",
    "description": "ANNUAL MIN VEG FRACTION",
    "units": "",
    "registry": "Registry.EM_COMMON:769",
    "input_dimensions": [
      "south_north",
      "west_east"
    ],
    "history_dimensions": [
      "south_north",
      "west_east"
    ]
  },
  "LAKEMASK": {
    "dtype": "f4",
    "stagger": "",
    "description": "LAKE MASK (1 FOR LAKE, 0 FOR NON-LAKE)",
    "units": "",
    "registry": "Registry.tladj:10",
    "input_dimensions": [
      "south_north",
      "west_east"
    ],
    "history_dimensions": [
      "south_north",
      "west_east"
    ]
  },
  "ZS": {
    "dtype": "f4",
    "stagger": "Z",
    "description": "DEPTHS OF CENTERS OF SOIL LAYERS",
    "units": "m",
    "registry": "Registry.EM_COMMON:68",
    "input_dimensions": [
      "soil_layers_stag"
    ],
    "history_dimensions": [
      "soil_layers_stag"
    ]
  },
  "DZS": {
    "dtype": "f4",
    "stagger": "Z",
    "description": "THICKNESSES OF SOIL LAYERS",
    "units": "m",
    "registry": "Registry.EM_COMMON:69",
    "input_dimensions": [
      "soil_layers_stag"
    ],
    "history_dimensions": [
      "soil_layers_stag"
    ]
  },
  "VAR_SSO": {
    "dtype": "f4",
    "stagger": "",
    "description": "variance of subgrid-scale orography",
    "units": "m2",
    "registry": "Registry.EM_COMMON:91",
    "input_dimensions": [
      "south_north",
      "west_east"
    ],
    "history_dimensions": [
      "south_north",
      "west_east"
    ]
  },
  "BATHYMETRY_FLAG": {
    "dtype": "i4",
    "stagger": "",
    "description": "Flag for bathymetry in the global attributes for metgrid data",
    "units": "",
    "registry": "Registry.EM_COMMON:148",
    "input_dimensions": [],
    "history_dimensions": []
  },
  "FNM": {
    "dtype": "f4",
    "stagger": "",
    "description": "upper weight for vertical stretching",
    "units": "",
    "registry": "Registry.EM_COMMON:349",
    "input_dimensions": [
      "bottom_top"
    ],
    "history_dimensions": [
      "bottom_top"
    ]
  },
  "FNP": {
    "dtype": "f4",
    "stagger": "",
    "description": "lower weight for vertical stretching",
    "units": "",
    "registry": "Registry.EM_COMMON:350",
    "input_dimensions": [
      "bottom_top"
    ],
    "history_dimensions": [
      "bottom_top"
    ]
  },
  "RDNW": {
    "dtype": "f4",
    "stagger": "",
    "description": "inverse d(eta) values between full (w) levels",
    "units": "",
    "registry": "Registry.EM_COMMON:351",
    "input_dimensions": [
      "bottom_top"
    ],
    "history_dimensions": [
      "bottom_top"
    ]
  },
  "RDN": {
    "dtype": "f4",
    "stagger": "",
    "description": "inverse d(eta) values between half (mass) levels",
    "units": "",
    "registry": "Registry.EM_COMMON:352",
    "input_dimensions": [
      "bottom_top"
    ],
    "history_dimensions": [
      "bottom_top"
    ]
  },
  "DNW": {
    "dtype": "f4",
    "stagger": "",
    "description": "d(eta) values between full (w) levels",
    "units": "",
    "registry": "Registry.EM_COMMON:353",
    "input_dimensions": [
      "bottom_top"
    ],
    "history_dimensions": [
      "bottom_top"
    ]
  },
  "DN": {
    "dtype": "f4",
    "stagger": "",
    "description": "d(eta) values between half (mass) levels",
    "units": "",
    "registry": "Registry.EM_COMMON:354",
    "input_dimensions": [
      "bottom_top"
    ],
    "history_dimensions": [
      "bottom_top"
    ]
  },
  "CFN": {
    "dtype": "f4",
    "stagger": "",
    "description": "extrapolation constant",
    "units": "",
    "registry": "Registry.EM_COMMON:358",
    "input_dimensions": [],
    "history_dimensions": []
  },
  "CFN1": {
    "dtype": "f4",
    "stagger": "",
    "description": "extrapolation constant",
    "units": "",
    "registry": "Registry.EM_COMMON:359",
    "input_dimensions": [],
    "history_dimensions": []
  },
  "THIS_IS_AN_IDEAL_RUN": {
    "dtype": "i4",
    "stagger": "",
    "description": "T/F flag: this is an ARW ideal simulation",
    "units": "",
    "registry": "Registry.EM_COMMON:363",
    "input_dimensions": [],
    "history_dimensions": []
  },
  "RDX": {
    "dtype": "f4",
    "stagger": "",
    "description": "INVERSE X GRID LENGTH",
    "units": "m-1",
    "registry": "Registry.EM_COMMON:397",
    "input_dimensions": [],
    "history_dimensions": []
  },
  "RDY": {
    "dtype": "f4",
    "stagger": "",
    "description": "INVERSE Y GRID LENGTH",
    "units": "m-1",
    "registry": "Registry.EM_COMMON:398",
    "input_dimensions": [],
    "history_dimensions": []
  },
  "ZETATOP": {
    "dtype": "f4",
    "stagger": "",
    "description": "ZETA AT MODEL TOP",
    "units": "",
    "registry": "Registry.EM_COMMON:404",
    "input_dimensions": [],
    "history_dimensions": []
  },
  "CF1": {
    "dtype": "f4",
    "stagger": "",
    "description": "2nd order extrapolation constant",
    "units": "",
    "registry": "Registry.EM_COMMON:405",
    "input_dimensions": [],
    "history_dimensions": []
  },
  "CF2": {
    "dtype": "f4",
    "stagger": "",
    "description": "2nd order extrapolation constant",
    "units": "",
    "registry": "Registry.EM_COMMON:406",
    "input_dimensions": [],
    "history_dimensions": []
  },
  "CF3": {
    "dtype": "f4",
    "stagger": "",
    "description": "2nd order extrapolation constant",
    "units": "",
    "registry": "Registry.EM_COMMON:407",
    "input_dimensions": [],
    "history_dimensions": []
  },
  "SHDAVG": {
    "dtype": "f4",
    "stagger": "",
    "description": "ANNUAL AVG VEG FRACTION",
    "units": "",
    "registry": "Registry.EM_COMMON:770",
    "input_dimensions": [
      "south_north",
      "west_east"
    ],
    "history_dimensions": [
      "south_north",
      "west_east"
    ]
  },
  "WATER_DEPTH": {
    "dtype": "f4",
    "stagger": "",
    "description": "global water depth",
    "units": "m",
    "registry": "Registry.EM_COMMON:882",
    "input_dimensions": [
      "south_north",
      "west_east"
    ],
    "history_dimensions": [
      "south_north",
      "west_east"
    ]
  },
  "VAR": {
    "dtype": "f4",
    "stagger": "",
    "description": "STANDARD DEVIATION OF SUBGRID-SCALE OROGRAPHY",
    "units": "m",
    "registry": "Registry.EM_COMMON:1163",
    "input_dimensions": [
      "south_north",
      "west_east"
    ],
    "history_dimensions": [
      "south_north",
      "west_east"
    ]
  },
  "MAPFAC_MX": {
    "dtype": "f4",
    "stagger": "",
    "description": "Map scale factor on mass grid, x direction",
    "units": "",
    "registry": "Registry.EM_COMMON:1395",
    "input_dimensions": [
      "south_north",
      "west_east"
    ],
    "history_dimensions": [
      "south_north",
      "west_east"
    ]
  },
  "MAPFAC_MY": {
    "dtype": "f4",
    "stagger": "",
    "description": "Map scale factor on mass grid, y direction",
    "units": "",
    "registry": "Registry.EM_COMMON:1396",
    "input_dimensions": [
      "south_north",
      "west_east"
    ],
    "history_dimensions": [
      "south_north",
      "west_east"
    ]
  },
  "MAPFAC_UX": {
    "dtype": "f4",
    "stagger": "X",
    "description": "Map scale factor on u-grid, x direction",
    "units": "",
    "registry": "Registry.EM_COMMON:1397",
    "input_dimensions": [
      "south_north",
      "west_east_stag"
    ],
    "history_dimensions": [
      "south_north",
      "west_east_stag"
    ]
  },
  "MAPFAC_UY": {
    "dtype": "f4",
    "stagger": "X",
    "description": "Map scale factor on u-grid, y direction",
    "units": "",
    "registry": "Registry.EM_COMMON:1398",
    "input_dimensions": [
      "south_north",
      "west_east_stag"
    ],
    "history_dimensions": [
      "south_north",
      "west_east_stag"
    ]
  },
  "MAPFAC_VX": {
    "dtype": "f4",
    "stagger": "Y",
    "description": "Map scale factor on v-grid, x direction",
    "units": "",
    "registry": "Registry.EM_COMMON:1399",
    "input_dimensions": [
      "south_north_stag",
      "west_east"
    ],
    "history_dimensions": [
      "south_north_stag",
      "west_east"
    ]
  },
  "MF_VX_INV": {
    "dtype": "f4",
    "stagger": "Y",
    "description": "Inverse map scale factor on v-grid, x direction",
    "units": "",
    "registry": "Registry.EM_COMMON:1400",
    "input_dimensions": [
      "south_north_stag",
      "west_east"
    ],
    "history_dimensions": [
      "south_north_stag",
      "west_east"
    ]
  },
  "MAPFAC_VY": {
    "dtype": "f4",
    "stagger": "Y",
    "description": "Map scale factor on v-grid, y direction",
    "units": "",
    "registry": "Registry.EM_COMMON:1401",
    "input_dimensions": [
      "south_north_stag",
      "west_east"
    ],
    "history_dimensions": [
      "south_north_stag",
      "west_east"
    ]
  },
  "GOT_VAR_SSO": {
    "dtype": "i4",
    "stagger": "",
    "description": "whether VAR_SSO was included in WPS output (beginning V3.4)",
    "units": "",
    "registry": "Registry.EM_COMMON:1439",
    "input_dimensions": [],
    "history_dimensions": []
  },
  "T00": {
    "dtype": "f4",
    "stagger": "",
    "description": "BASE STATE TEMPERATURE   ",
    "units": "K",
    "registry": "Registry.EM_COMMON:1477",
    "input_dimensions": [],
    "history_dimensions": []
  },
  "P00": {
    "dtype": "f4",
    "stagger": "",
    "description": "BASE STATE PRESSURE",
    "units": "Pa",
    "registry": "Registry.EM_COMMON:1478",
    "input_dimensions": [],
    "history_dimensions": []
  },
  "TLP": {
    "dtype": "f4",
    "stagger": "",
    "description": "BASE STATE LAPSE RATE    ",
    "units": "",
    "registry": "Registry.EM_COMMON:1479",
    "input_dimensions": [],
    "history_dimensions": []
  },
  "TISO": {
    "dtype": "f4",
    "stagger": "",
    "description": "TEMP AT WHICH THE BASE T TURNS CONST",
    "units": "K",
    "registry": "Registry.EM_COMMON:1480",
    "input_dimensions": [],
    "history_dimensions": []
  },
  "TLP_STRAT": {
    "dtype": "f4",
    "stagger": "",
    "description": "BASE STATE LAPSE RATE (DT/D(LN(P)) IN STRATOSPHERE",
    "units": "K",
    "registry": "Registry.EM_COMMON:1481",
    "input_dimensions": [],
    "history_dimensions": []
  },
  "P_STRAT": {
    "dtype": "f4",
    "stagger": "",
    "description": "BASE STATE PRESSURE AT BOTTOM OF STRATOSPHERE",
    "units": "Pa",
    "registry": "Registry.EM_COMMON:1482",
    "input_dimensions": [],
    "history_dimensions": []
  },
  "CLAT": {
    "dtype": "f4",
    "stagger": "",
    "description": "COMPUTATIONAL GRID LATITUDE, SOUTH IS NEGATIVE",
    "units": "degree_north",
    "registry": "Registry.EM_COMMON:1846",
    "input_dimensions": [
      "south_north",
      "west_east"
    ],
    "history_dimensions": [
      "south_north",
      "west_east"
    ]
  },
  "C1H": {
    "dtype": "f4",
    "stagger": "",
    "description": "half levels, c1h = d bf / d eta, using znw",
    "units": "Dimensionless",
    "registry": "registry.hyb_coord:43",
    "input_dimensions": [
      "bottom_top"
    ],
    "history_dimensions": [
      "bottom_top"
    ]
  },
  "C2H": {
    "dtype": "f4",
    "stagger": "",
    "description": "half levels, c2h = (1-c1h)*(p0-pt)",
    "units": "Pa",
    "registry": "registry.hyb_coord:44",
    "input_dimensions": [
      "bottom_top"
    ],
    "history_dimensions": [
      "bottom_top"
    ]
  },
  "C1F": {
    "dtype": "f4",
    "stagger": "Z",
    "description": "full levels, c1f = d bf / d eta, using znu",
    "units": "Dimensionless",
    "registry": "registry.hyb_coord:46",
    "input_dimensions": [
      "bottom_top_stag"
    ],
    "history_dimensions": [
      "bottom_top_stag"
    ]
  },
  "C2F": {
    "dtype": "f4",
    "stagger": "Z",
    "description": "full levels, c2f = (1-c1f)*(p0-pt)",
    "units": "Pa",
    "registry": "registry.hyb_coord:47",
    "input_dimensions": [
      "bottom_top_stag"
    ],
    "history_dimensions": [
      "bottom_top_stag"
    ]
  },
  "C3H": {
    "dtype": "f4",
    "stagger": "",
    "description": "half levels, c3h = bh",
    "units": "Dimensionless",
    "registry": "registry.hyb_coord:49",
    "input_dimensions": [
      "bottom_top"
    ],
    "history_dimensions": [
      "bottom_top"
    ]
  },
  "C4H": {
    "dtype": "f4",
    "stagger": "",
    "description": "half levels, c4h = (eta-bh)*(p0-pt), using znu",
    "units": "Pa",
    "registry": "registry.hyb_coord:50",
    "input_dimensions": [
      "bottom_top"
    ],
    "history_dimensions": [
      "bottom_top"
    ]
  },
  "C3F": {
    "dtype": "f4",
    "stagger": "Z",
    "description": "full levels, c3f = bf",
    "units": "Dimensionless",
    "registry": "registry.hyb_coord:52",
    "input_dimensions": [
      "bottom_top_stag"
    ],
    "history_dimensions": [
      "bottom_top_stag"
    ]
  },
  "C4F": {
    "dtype": "f4",
    "stagger": "Z",
    "description": "full levels, c4f = (eta-bf)*(p0-pt), using znw",
    "units": "Pa",
    "registry": "registry.hyb_coord:53",
    "input_dimensions": [
      "bottom_top_stag"
    ],
    "history_dimensions": [
      "bottom_top_stag"
    ]
  },
  "PCB": {
    "dtype": "f4",
    "stagger": "",
    "description": "base state dry air mass in column",
    "units": "Pa",
    "registry": "registry.hyb_coord:55",
    "input_dimensions": [
      "south_north",
      "west_east"
    ],
    "history_dimensions": [
      "south_north",
      "west_east"
    ]
  }
}
''')

WRF_INITIALIZED_HISTORY_ROWS = json.loads(r'''
{
  "NEST_POS": {
    "dtype": "f4",
    "stagger": "",
    "description": "",
    "units": "",
    "registry": "Registry.EM_COMMON:306",
    "input_dimensions": [],
    "history_dimensions": [
      "south_north",
      "west_east"
    ]
  },
  "AREA2D": {
    "dtype": "f4",
    "stagger": "",
    "description": "Horizontal grid cell area, using dx, dy, and map factors",
    "units": "m2",
    "registry": "Registry.EM_COMMON:399",
    "input_dimensions": [],
    "history_dimensions": [
      "south_north",
      "west_east"
    ]
  },
  "DX2D": {
    "dtype": "f4",
    "stagger": "",
    "description": "Horizontal grid distance: sqrt(area2d)",
    "units": "m",
    "registry": "Registry.EM_COMMON:400",
    "input_dimensions": [],
    "history_dimensions": [
      "south_north",
      "west_east"
    ]
  },
  "SNOALB": {
    "dtype": "f4",
    "stagger": "",
    "description": "ANNUAL MAX SNOW ALBEDO IN FRACTION",
    "units": "",
    "registry": "Registry.EM_COMMON:771",
    "input_dimensions": [],
    "history_dimensions": [
      "south_north",
      "west_east"
    ]
  },
  "MAX_MSFTX": {
    "dtype": "f4",
    "stagger": "",
    "description": "Max map factor in domain",
    "units": "",
    "registry": "Registry.EM_COMMON:1483",
    "input_dimensions": [],
    "history_dimensions": []
  },
  "MAX_MSFTY": {
    "dtype": "f4",
    "stagger": "",
    "description": "Max map factor in domain",
    "units": "",
    "registry": "Registry.EM_COMMON:1484",
    "input_dimensions": [],
    "history_dimensions": []
  },
  "SAVE_TOPO_FROM_REAL": {
    "dtype": "i4",
    "stagger": "",
    "description": "1=original topo from real/0=topo modified by WRF",
    "units": "flag",
    "registry": "Registry.EM_COMMON:2007",
    "input_dimensions": [],
    "history_dimensions": []
  },
  "SST_INPUT": {
    "dtype": "f4",
    "stagger": "",
    "description": "SEA SURFACE TEMPERATURE FROM WRFLOWINPUT FILE",
    "units": "K",
    "registry": "Registry.EM:16",
    "input_dimensions": [],
    "history_dimensions": [
      "south_north",
      "west_east"
    ]
  },
  "ISEEDARR_SPPT": {
    "dtype": "i4",
    "stagger": "Z",
    "description": "Array to hold seed for restart, SPPT",
    "units": "",
    "registry": "registry.stoch:60",
    "input_dimensions": [],
    "history_dimensions": [
      "seed_dim_stag"
    ]
  },
  "ISEEDARR_SKEBS": {
    "dtype": "i4",
    "stagger": "Z",
    "description": "Array to hold seed for restart, SKEBS",
    "units": "",
    "registry": "registry.stoch:61",
    "input_dimensions": [],
    "history_dimensions": [
      "seed_dim_stag"
    ]
  },
  "ISEEDARR_RAND_PERTURB": {
    "dtype": "i4",
    "stagger": "Z",
    "description": "Array to hold seed for restart, RAND_PERT",
    "units": "",
    "registry": "registry.stoch:62",
    "input_dimensions": [],
    "history_dimensions": [
      "seed_dim_stag"
    ]
  },
  "ISEEDARRAY_SPP_CONV": {
    "dtype": "i4",
    "stagger": "Z",
    "description": "Array to hold seed for restart, RAND_PERT2",
    "units": "",
    "registry": "registry.stoch:63",
    "input_dimensions": [],
    "history_dimensions": [
      "seed_dim_stag"
    ]
  },
  "ISEEDARRAY_SPP_PBL": {
    "dtype": "i4",
    "stagger": "Z",
    "description": "Array to hold seed for restart, RAND_PERT3",
    "units": "",
    "registry": "registry.stoch:64",
    "input_dimensions": [],
    "history_dimensions": [
      "seed_dim_stag"
    ]
  },
  "ISEEDARRAY_SPP_LSM": {
    "dtype": "i4",
    "stagger": "Z",
    "description": "Array to hold seed for restart, RAND_PERT4",
    "units": "",
    "registry": "registry.stoch:65",
    "input_dimensions": [],
    "history_dimensions": [
      "seed_dim_stag"
    ]
  }
}
''')

# GNU Fortran intrinsic RANDOM_SEED(SIZE=) used by stock WRF 4.6.1.
WRF_GNU_RANDOM_SEED_WORDS = 8
