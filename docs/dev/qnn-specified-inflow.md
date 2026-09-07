# QNN transport and specified inflow

The scalar stage now applies WRF v4.6.1's QNN boundary operation. WDM6's
`nn` receives `wdm6_ccn_conc` at inflow; NSSL's `qnn` receives the resolved
`nssl_cccn / 1.225` value, exactly 408163264 in FP32 under the admitted
NSSL parameter identity. Outflow copies the first interior row or column.
Calm flow takes the inflow branch, and Y boundaries own the corners.

Authority is stock WRF v4.6.1 commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`:

- `dyn_em/start_em.F:1750-1760` resolves the scheme-specific grid CCN value.
- `dyn_em/solve_em.F:2893-2903` selects QNN before ordinary scalar boundaries.
- `share/module_bc.F:2460-2583` defines the four boundary operations.
- `Registry/Registry.EM_COMMON:3031` declares WDM6's transported
  `qnn/qnc/qnr` package.

The stage test exposed a second missing connection: WDM6 allocated `nn` and
`nc` but the generic transported-species lookup only selected its `nr`.
WDM6 now selects its existing complete number tuple using the scheme's unique
`nn` carrier. Morrison's diagnostic `nc` remains outside that lookup.

`tests/test_qnn_specified_inflow.py` checks stock loops at all four edges and
corners, exact FP32 CUDA words, ordinary zero inflow, configured WDM6 and
resolved NSSL inflow through both scalar-stage forms, and interior transport
of WDM6 CCN/cloud number against the already transported rain-number field.
The two WDM6 stage controls failed before the transport connection was added.
The measured result is 4 CPU controls and 10 GPU controls passing; the
adjacent lateral-boundary CPU suite also passes 21 controls.

General `have_bcs_scalar` forcing remains a separate unsupported input
operation. This change does not introduce external ordinary-number tables
or change the supported NSSL parameter identity.
