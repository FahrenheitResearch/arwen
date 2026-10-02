# Surface vertical velocity at nonperiodic edges

The kinematic lower boundary follows WRF v4.7.1
`dyn_em/module_bc_em.F:1246-1279`: each outside terrain donor clamps to
the current edge cell. Its terrain difference is zero. Periodic axes
retain their wrapped donors. The three-level face-wind interpolation,
map scaling and all interior slopes are unchanged.

The former fused and eager paths copied the interior slope into the
outside half-term. With equal face winds this doubled the component
normal to a sloped edge. The correction applies by default to both paths;
it is a defect fix rather than a verification arithmetic option.

`tests/test_surface_w_fused.py::test_sloped_edge_surface_matches_wrf_clamped_donors`
compares both paths against the independent WRF clamped-index reference
used for native cold-start W. Sloped terrain and nonuniform winds cover
all four edges and their corners, both boundary flags, dyadic map factors
and unchanged upper W levels. Periodic controls preserve the old wrap.

The 192-second native-input comparison uses unchanged WRF O2/O3 outputs.
At 12 seconds the former north-edge surface error, 0.0263336 m/s at
`[0,149,6]`, falls to 0.00000819 m/s. The remaining full-domain maximum,
0.00256455 m/s, equals the rim-excluded maximum; the WRF compiler-control
maximum is 0.00238875 m/s. The U boundary error remains at 12 and 192
seconds, so this correction does not establish its cause.

Other boundary copies follow separate contracts. WRF extends east/north
terrain metrics through `set_physical_bc3d` and copies deformation
tensors at physical boundaries. Those copies remain. Acoustic terrain
donors and the native cold-start W initializer already clamp their
outside indexes.
