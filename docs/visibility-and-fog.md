# Visibility and low-visibility probability

`gpuwm render WRFOUT --products visibility,fog_flag --out MAPS` imports
native forecast hydrometeors and thermodynamics through Rust and draws metres.
The default visibility colorbar emphasizes 0–2000 m, including labelled 400 m
and 1000 m limits; larger values use its clear-air extension color. Stored values
retain their full diagnostic range. The flag uses category labels 0 and 1.
Both products are included by `--products all`, the default product selection.

The diagnostic combines the Stoelinga-Warner extinction relationships as
published in the NOAA GSD algorithm description with its RH-based clear-air
limit. It is an independently written Rust/CUDA implementation of the stated
mathematical relationships, with no UPP source code included or linked.

For each species take the maximum dry-air mixing ratio over the lowest three
mass levels (or all available levels if fewer). Use lowest-level pressure,
temperature and water vapor to convert these ratios to concentrations C in
g/m³. The extinction coefficients, in km⁻¹, are:

| Species | Extinction |
| --- | --- |
| Cloud liquid | 144.7 C^0.88 |
| Rain | 2.24 C^0.75 |
| Cloud ice | 327.8 C |
| Snow | 10.36 C^0.7776 |
| Graupel | 8 C^0.75 |

Sum extinction and add 10⁻¹⁰ km⁻¹ to avoid division by zero. The hydrometeor
visibility is 1000 × −ln(0.02)/extinction metres. The RH component is
90000 × exp(−2.5 × clamp(RH/100 − 0.15, 0, 0.8)) metres, using the maximum
water-relative RH of the lowest two mass levels. The reported visibility is
the minimum of these components and 90000 m. This selects the GSD 2017 RH
base of 90 km. It retains the stated fixed hydrometeor coefficients and 2%
contrast; solar-angle, wind-shear, aerosol and blowing-snow adjustments from
other algorithm versions are not included. RH alone cannot produce dense fog.

QVAPOR and QCLOUD are required. Missing optional rain, ice, snow and graupel
variables mean those species are absent. Present invalid values produce
unknown visibility, never clear air. Negative hydrometeor ratios are clipped
to zero. This near-surface diagnostic samples model mass levels, whose actual
heights depend on the model grid; it is not a direct 2 m visibility measurement.

`fog_flag` is 1 exactly when diagnosed visibility is **strictly below 1000 m**,
0 otherwise, and unknown when visibility is unknown. It flags low visibility
without identifying the obscurant: rain and snow can also set this flag. It
must not be interpreted as a smoke, dust or aerosol visibility forecast.

Ordinary ensemble output includes visibility mean, spread, extrema, finite
member count and **P(visibility < 1000 m)** and **P(visibility < 400 m)** by
default. Threshold equality is excluded. Every original member is required;
unknown values mask the cell and failed members never reduce the denominator.
The CUDA diagnostic uses the same equations on each member's native fields.
`gpuwm enprod --field visibility --products prob --engine rust` and the alias
`--field fog_prob` produce these two thresholds from retained member histories.

The existing `rw_asos stations/fetch/decode` route now requests METAR `vsby`,
converts statute miles to metres (1 SM = 1609.344 m), and retains it in the
surface observation seam as `visibility`. Reports at or above 10 SM are
right-censored to 16093.44 m and flagged `metar_visibility_ge_10sm`. The
`rw_ensemble_match` quantity `visibility` computes the native diagnostic,
uses the frozen station/time matching plan and bilinear spatial sampling,
then caps model values at the same 10 SM for continuous scores. The two low
visibility event thresholds are below this cap and remain unambiguous.
The matching receipt includes Brier scores and hit/miss/false-alarm counts for
both strict thresholds, using a 0.5 probability cutoff for contingency counts.
Unknown model visibility stays unknown. A capped score cannot measure skill
in distinguishing clear-air visibility above 10 SM. Preserve native member
histories (`--keep-member-files`) when using this verification route. A surface-only
matching plan can omit precipitation geometry; no unrelated radar data is required.

Method references:

- [NOAA GSD algorithm description and coefficient table](https://noaa-emc.github.io/UPP/upp-srw-v2.2.0/CALVIS__GSD_8f_source.html).
- [NOAA RUC/MAPS visibility verification paper](https://rapidrefresh.noaa.gov/internal/pdfs/avx00.smirnova.fogvis.pdf).
- [NOAA RAP diagnostic description](https://rapidrefresh.noaa.gov/RAP_var_diagnosis.html).

The current upstream UPP repository has an LGPL-3.0 license. No UPP code is
redistributed here; numerical constants and equations are implemented in
original code under this project's license. NOAA/NWS observational products
are public domain subject to the [NWS use notice](https://www.weather.gov/disclaimer).
