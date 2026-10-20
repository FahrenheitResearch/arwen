"""Pinned native BMI output inventory; conversions execute in CUDA."""

# name, numeric model field, units, optional native get_value conversion.
OUTPUTS = (
    ("QINSUR", "water.qinsur", "m/s", None),
    ("ETRAN", "water.etran", "mm", "dt"),
    ("QSEVA", "water.qseva", "mm/s", "1000"),
    ("EVAPOTRANS", "water.evapotrans", "m/s", None),
    ("TG", "energy.tg", "K", None),
    ("SNEQV", "water.sneqv", "mm", None),
    ("TGS", "energy.tgs", "K", None),
    ("ACSNOM", "water.acsnom", "mm", None),
    ("SNOWT_AVG", "energy.snowt_avg", "K", None),
    ("ISNOW", "water.isnow", "unitless", None),
    ("QRAIN", "water.qrain", "mm/s", None),
    ("FSNO", "water.fsno", "unitless", None),
    ("SNOWH", "water.snowh", "m", None),
    ("SNLIQ", "water.snliq", "mm", None),
    ("QSNOW", "water.qsnow", "mm/s", None),
    ("ECAN", "water.ecan", "mm", "dt"),
    ("GH", "energy.gh", "W/m2", None),
    ("TRAD", "energy.trad", "K", None),
    ("FSA", "energy.fsa", "W/m2", None),
    ("CMC", "water.cmc", "mm", None),
    ("LH", "energy.lh", "W/m2", None),
    ("FIRA", "energy.fira", "W/m2", None),
    ("FSH", "energy.fsh", "W/m2", None),
)
