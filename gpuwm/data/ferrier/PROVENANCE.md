# Ferrier-Aligo lookup table provenance

`tables/ETAMPNEW_DATA.expanded_rain` is WRF v4.6.1's own
`run/ETAMPNEW_DATA.expanded_rain` (tag v4.6.1, commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`), copied byte for byte: 43,400
bytes, SHA-256
`28e60e6c83eb55889c4ab65eaa33e9fc3b557e5ab45c3dde7ee7153c2ef95a13`.

It is the table `fer_hires_init` reads at
`phys/module_mp_fer_hires.F:2969-2983`: twelve big-endian Fortran
unformatted records (VENTR1, VENTR2, ACCRR, MASSR, VRAIN, RRATE over mean
rain diameters 50-1000 um; VENTI1, VENTI2, ACCRI, MASSI, VSNOWI over mean
ice diameters 50-1000 um; VEL_RF over (2:9, 0:40)), then a thirteenth
35-word record whose READ is commented out at :2984 and which WRF never
uses.  `gpuwm.core.fer_hires.load_table` refuses any other bytes and parses
the twelve records.  On 2026-10-08 the parsed arrays were compared word for
word with the module arrays WRF's own `fer_hires_init` filled (gfortran
13.3.0, -O0); none differed.

WRF is NCAR/UCAR public domain and is attributed in `NOTICE`; the notice
WRF asks to travel with copies is `LICENSE-WRF.txt` beside this file.
