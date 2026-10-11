"""Host float64 verification reference for WRF fast spectral bin microphysics.

Scope
-----
This package is a transcription of WRF's fast spectral bin microphysics
(FSBM-2, ``mp_physics = 30``) into float64 numpy, used as the verification
authority for the device kernels that later milestones add.  It is not a
runtime backend: no model configuration selects it, nothing in the engine
imports it, and it holds no device code (no module under this package imports
cupy).  Its job is to reproduce WRF's own Fortran, compiled with every REAL
promoted to double precision, closely enough that the Fortran oracle can gate
it, and then to serve as the reference the CUDA port is gated against.

Milestone 1 covers the bin state and unit layer (``state``), the packaged
lookup tables (``tables``), the scalar constants and the behaviour presets
(``constants``), and the warm-rain processes that later items add beside
them.  Milestone 2, the ice phase, starts from a contract its items share:
the cold-half behaviour fields and the named dense-ice option
(``constants``), the tile's cold inputs and the call's cold outputs
(``column``), and the routine map of the pinned WRF file with the findings
catalogue (``catalogue``).  Device kernels and the engine wiring follow in
later milestones.

Source pins
-----------
* WRF v4.8.0 ``phys/module_mp_fast_sbm.F``, 9,053 lines, SHA-256 of the LF
  content ``fd72158202a98d4ad618ebb608eeadffd50852a9875f8bfd9bb8ad63c15f1a2e``.
  The file is identical in WRF v4.6.1 and v4.7.1 (last upstream change
  2024-01-11, f81f98252b), so the engine's WRF v4.6.1 pin covers it.  Every
  line number cited in this package refers to that file; a CRLF checkout has
  the same line numbering.
* NCAR's ``SBM_input_33.tgz`` table archive, SHA-256
  ``25ed63ab3550aed110ae24df229faece20eafedfdf21cde01d8973826a7294bf``
  (1,339,105 bytes).  The 18 members the scheme opens are packaged byte for
  byte under ``gpuwm/data/fast_sbm/SBM_input_33/``; ``PROVENANCE.md`` beside
  them records the per-file pins and the members left out.

Attribution
-----------
WRF is developed at the National Center for Atmospheric Research (NCAR),
operated by the University Corporation for Atmospheric Research (UCAR), and
placed in the public domain; UCAR's notice travels in the repository
``NOTICE``, in ``licenses/LICENSE-WRF-public-domain.txt`` and beside the
packaged tables in ``gpuwm/data/fast_sbm/LICENSE-WRF.txt``.

The spectral bin scheme comes from the Hebrew University Cloud Model,
formulated and coded by Alexander Khain.  The first WRF fast SBM was coded by
Barry Lynn.  FSBM-2, the version transcribed here, was coded and is
maintained by Jacob Shpund, with contributions the WRF header credits to
Jiwen Fan, Alexander Ryzhkov, Jeffery Snyder, Jimy Dudhia and Dave Gill.  The
WRF header asks users of FSBM-2 to cite

    Shpund, J., A. Khain, B. Lynn, J. Fan, B. Han, A. Ryzhkov, J. Snyder,
    J. Dudhia, and D. Gill, 2019: Simulating a mesoscale convective system
    using WRF with a new spectral bin microphysics: 1: Hail vs graupel.
    J. Geophys. Res. Atmos.

The lookup tables are credited to NCAR's ``SBM_input_33`` distribution.
Every module of the port names the WRF routine and the lines it transcribes.

Importing this package has no side effects: it does not read the tables,
which ``tables.load_tables`` does on request.
"""
