# Provenance

The analysis, options, tables, API helper and renderer modules were copied from Drew's website sounding service on 2026-10-17 with Drew's explicit instruction to make the window available in the public engine. Engine-owned service code is provided under the engine's Apache-2.0 licence. The accompanying rustwx-sounding bridge is included under the same grant. No assimilation logic was copied.

The window uses the existing sharppyrs/sharprs implementations, copied with their BSD-3-Clause / MIT notices. Their runtime sources are preserved; unrelated GUI example/snapshot tooling was omitted. Website-specific branding, identifiers and filesystem discovery were replaced with generic public engine identifiers and a plain-run inventory. The new input and batch modules adapt ordinary WRF/rw-store output, establish physical valid times, and emit the window, JSON and a daily HTML sequence.

The existing store reader was adapted to this engine's rw-store API. Data interpretation follows its documented volume and surface-variable contracts. Nothing launches or depends on the private service, and the public command contains no assimilation/cycling-layout discovery.

The engine port joins its render workers before GPU shutdown and uses production GPU adapter selection rather than the snapshot harness's software preference. In the copied widget, diagnostic headers, streamwiseness labels and locator backgrounds follow the selected theme so light-theme text and map boundaries remain visible. Profile analysis and diagnostic formulas are unchanged by those display fixes.

New registry dependencies are vendored with Cargo package checksums and their original notices. The GUI snapshot comparison dependency (MPL-licensed colored via dify) is excluded; it is not needed to render the window.
