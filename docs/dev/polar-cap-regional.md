# Regional polar caps

The regional wizard accepts points and drawings enclosing the selected pole
on `polar` stereographic. Automatic projection selection already chooses this
projection at high latitudes. Lambert keeps its cone-tip clearance: longitude
is singular there, and an enclosing domain must use polar stereographic.

A cap's forcing request contains the pole latitude and the complete longitude
band. This is selected from the source coverage metadata. Regional sources
still have to cover every target point; choosing polar does not extend their
physical coverage.

A global regular forcing grid has two coordinate boundaries to handle:

* Where a longitude cut cannot clear all target stencils, keep three periodic
  donor columns on each side of the stored ring.
* Where a target stencil reaches beyond a pole row, reflect the donor latitude
  and shift its longitude by 180 degrees. Negate both earth-relative horizontal
  wind components on that chart transition. Scalars retain their values.

`global_source_cut` makes this decision before reading weather fields. GFS,
mapped-source metadata, atmospheric windowing, and the download memory estimate
consume that geometry. The added donor selection runs in Rust through
`gpuwm_static_global_donors_f64`, with Rayon row parallelism. The existing
interpolation and wind-rotation operators then use the extended regular axes.
A source with an odd number of longitude columns in a revolution cannot use
exact opposite-meridian reflection and receives a concrete refusal.

The Rust static sampler keeps a full longitude period and its interpolation
halos for a cap. This prevents its ordinary antimeridian window heuristic from
dropping donors on the other side of the pole. Projection transforms, map
factors, and rotation formulas remain the WRF/WPS oracle-gated formulas.

Covering tests are in `test_domain_wizard.py`,
`test_domain_wizard_point_extent_cap.py`, `test_mapped_dateline.py`,
`test_atmospheric_window.py`, `test_download_budget.py`, and
`test_projection_oracle.py`. The pole donor and static-window tests also run
under the `static-fields` Rust library tests. Whole-run evidence is delivered
separately by the capability lane; unit tests do not establish forecast skill.
