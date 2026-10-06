The namelist is the retained effective clone input from the 2026-09-30
configuration import. It has the clone's explicitly recorded substitutions
for options that were unsupported in that build. This tests generation
selection, not an exact operational configuration.

The WPS fixture keeps that clone's geometry and dates. Its geography token
is `default`, which the current importer supports, so a geography-source
refusal does not prevent the physics-generation test. No physics value in
the retained input namelist was changed for this fixture.

Both `hrrr_wrf.nl` and the clone's suffixed `hrrr_wrf.nl.c18c` filename
declare the source generation. An ordinary `namelist.input` filename does
not declare it.
