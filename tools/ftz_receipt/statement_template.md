<!-- FROZEN TEMPLATE.  Every fact below arrives as a substitution from
     tools/ftz_receipt/receipt/receipt.json and receipt/route_inventory.json.
     No outcome word may be typed into this file: tests/test_ftz_render.py
     greps it for each member of the receipt's verdict vocabulary and fails
     if one appears.  Prose here may frame a measurement; it may never state
     one, and it may never quantify over loaders, kernels or routes -- the
     enumerated substitutions exist so that it does not have to. -->

[[block:provenance-compile-policy]]
- **Rounding and subnormals**: ordinary FP32/FP64 operations and conversions
  use round-to-nearest-even.  FP32 subnormal handling is not one policy: it
  differs between the compile routes this codebase uses, so it is recorded
  per route, as measured on {{device_name}} (compute capability
  {{compute_capability}}, driver {{driver_version}}, NVRTC {{nvrtc_version}},
  CuPy {{cupy_version}}) by `tools/ftz_receipt/probe.py` over
  {{mechanism_count}} arithmetic mechanisms:
{{route_bullets}}
  The option tuple each row names is the tuple NVRTC received, captured by
  wrapping the compiler entry point rather than re-deriving it: CuPy appends
  `{{append_flag}}` to whatever the caller passed, at
  `{{append_module}}` line {{append_line}} (`{{append_text}}`), after the
  caller's options, and NVRTC honours the last occurrence.
  {{tuple_lead_in}}
{{tuple_bullets}}
  {{shared_object_note}}
  Evidence: {{evidence}}.
[[end:provenance-compile-policy]]

[[block:hardware-fp32-subnormals]]
- FP32 subnormal handling was measured on this machine's GPU
  ({{device_name}}, compute capability {{compute_capability}}, driver
  {{driver_version}}) across the {{route_count}} compile routes the model
  uses, crossed with {{mechanism_count}} arithmetic mechanisms.  The answer
  depends on the route:
{{route_bullets}}
  {{shared_object_note}}
  {{control_note}}
  The consequences reach physics: each known instance is recorded in the
  physics registry ([PHYSICS.md](PHYSICS.md)), and the radiation preparation
  path routes one subnormal-sensitive block through the host by design.
  Evidence: {{evidence}}.
[[end:hardware-fp32-subnormals]]
