# Named configuration arms on real source trajectories

The existing `gpuwm ensemble CONFIG` member-roster recipe accepts an optional
`[ensemble].trajectories` list with exactly one entry per named member arm.
Entry order binds the source, cycle and optional source member to the arm at
the same index. Without this list every arm retains the original base source.
The existing `--recipe member-roster --trajectories FILE` spelling uses the
same planner and preparation chain. An explicit roster flag retains a list
already configured in the file.

Every trajectory must cover the complete forecast window with its published
valid-time knots. A window that ends between a source's published knots (a 2 h
run on 3-hourly GFS) acquires the knots that enclose it (f0/f3) and keeps its
own end, exactly as the single-run door does: native preparation interpolates
the boundary tendency between them. Every recipe kind brackets this way. Each member still
uses ordinary source preparation and its own full physics memory admission.
The land and fixed surface selectors, duplicate-arm checks, stochastic
admission, original runner and packing policy are unchanged.

Microphysics and boundary-layer arms remain unavailable in named rosters.
This extension does not qualify a forecast or claim source-fork identity.
