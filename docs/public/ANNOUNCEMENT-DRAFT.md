# Announcement draft (for Drew to adapt; ~520 words)

<!-- Venue-neutral. First person singular; adjust to taste. -->

## ArWen 1.0: a WRF-ARW-class regional model, GPU-native, verified against WRF v4.6.1

I'm releasing ArWen, an independent GPU-native implementation of a
WRF-ARW-class regional atmospheric model (not affiliated with or
endorsed by NCAR/UCAR). It runs the whole pipeline -- data acquisition,
initialization, integration, products -- on one machine with one
consumer NVIDIA GPU.

Measured on an RTX 5090, from a first-time-user acceptance transcript:
a 6 hour forecast on a 250x200x49 12 km domain with the full physics
suite (Morrison two-moment microphysics, RTE+RRTMGP radiation, YSU
boundary layer, Noah land surface, Kain-Fritsch) completed in 3.6
minutes of wall time using about 6.3 GiB of VRAM. Getting the GFS input
data took 9.7 seconds; rendering 16 forecast product images took 2.6
seconds. A four-domain nest ladder down to 500 m grid spacing fits on a
24 GiB card; the built-in wizard sizes the grids to your GPU from a
single command.

The part I care most about is the verification story. Every physics
scheme is a transcription of WRF v4.6.1 source, gated three ways:
bit-level kernel oracles against unmodified WRF Fortran (several
components are bit-identical; the rest carry measured ULP distances,
published per option), t=0 initialization parity (the model opens the
WRF initial state at the FP32 floor on all four domains of the
reference case), and matched-run forecasts scored frame by frame
against a 48-rank WRF reference. On the reference case -- 3 April
1974, four nests to 500 m -- the 3 km domain at +3 hours matches WRF
to composite-reflectivity correlation 0.985, with >=20 dBZ echo
coverage within 3 pixels of WRF's 14,227. The docs publish the full
decay tables including the unflattering late-lead numbers at 1 km and
500 m, the measured evidence that this collapse is a property of
pixel-overlap scoring at convective scale rather than of this model
(the same run scores 0.86 storm-pixel overlap at 3 km while 1 km
decorrelates), and an explicit list of what is *not* claimed: no end-to-end bit-exactness,
one deeply validated case, FP32, no data assimilation. The
verification page ends with the config, commit, and commands to
reproduce the comparison.

Why build this: convective-scale NWP has been gated on institutional
clusters, so most of the world -- including most of the tornado- and
flood-prone world -- runs on 10-25 km global guidance. A verified
1 km limited-area model that runs on a gaming GPU changes who gets to
do this kind of simulation: researchers without allocations, students,
forecast hobbyists, anyone in a region no convection-permitting model
covers. To be clear about what it is for: ArWen is a research and
educational tool, never a substitute for official warnings from your
national meteorological service.

ArWen was designed and directed by its author and implemented with
substantial use of AI coding agents (Anthropic's Claude, including
Claude Fable 5, with auditing by OpenAI models); all model code was
gated by the verification program above rather than accepted on
generation, and the methodology is documented in VERIFICATION.md.

Apache-2.0. The same preprocessor also feeds unchanged stock WRF
(wrfinput/wrfbdy directly, no WPS, no real.exe) -- boundaries of that
claim documented. Repository, first-light walkthrough, and the full
verification dossier: [link].
