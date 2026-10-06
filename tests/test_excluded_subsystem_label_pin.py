"""The display-label allowance, tested where no checkout is needed.

``tests/test_excluded_subsystem_absent.py`` skips every test it holds when
there is no git checkout, because its scans read ``git ls-files`` and the
branch history.  This one test reads neither: it asks the masking rule
about three literal lines.  Kept there, it skipped on every run from a
release snapshot, so the pin that keeps the display-label allowance
narrow went unchecked exactly where the snapshot ships.  The helpers are
imported, not copied, so the rule tested is the rule the scans apply.
"""

from __future__ import annotations

from test_excluded_subsystem_absent import _radar_writer_masked, _tier1_match


def test_the_display_label_allowance_admits_only_the_exact_label():
    """The pin stays narrow: one phrase, in its files, and nothing else."""

    label = "Re" + "cast" + " WOOF"
    theme = "tools/rustwx/crates/rustwx-render/themes/woof-light.json"
    assert _tier1_match(_radar_writer_masked(theme, f'"source_label": "{label}"')) is None
    # The prefix by itself, another case, or the label in an unpinned file.
    assert _tier1_match(_radar_writer_masked(theme, "Re" + "cast")) is not None
    assert _tier1_match(_radar_writer_masked(theme, label.lower())) is not None
    assert _tier1_match(_radar_writer_masked("gpuwm/config.py", label)) is not None
