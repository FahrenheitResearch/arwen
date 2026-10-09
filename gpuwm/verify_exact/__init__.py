"""`gpuwm verify-exact`: replay recorded WRF runs through WOOF and score them bitwise.

The package is split so the comparison can be tested and rerun without a card:

- :mod:`.compare` -- field digests, run-against-run scoring, ULP detail.  CPU, numpy only.
- :mod:`.recording` -- the recording layout (stock WRF combo recordings and WOOF's own replays,
  which are written in the same layout).
- :mod:`.fixtures` -- per-scheme column fixtures: a scheme's oracle inputs and outputs, dropped
  into a directory and scored by the same comparison.
- :mod:`.replay` -- the WOOF arm: one card, many runs in one process.
- :mod:`.cli` -- the command.

The format of both directories is documented in ``docs/dev/verify-exact.md``.
"""
