import re
from importlib import metadata
from pathlib import Path

import gpuwm


def test_version_is_the_installed_distribution_version():
    """The package reports the release it IS, not a literal someone typed.

    The constant this replaces read ``0.1.1`` on a 1.1.1 install, which
    is not merely cosmetic: the prepared-cache provenance refusal and
    ``rw-wps --version`` both quote this string to tell a user which
    release is speaking to them.
    """

    assert gpuwm.__version__ == metadata.version(gpuwm.DISTRIBUTION_NAME)
    assert gpuwm.DISTRIBUTION_NAME == "gpuwm"


def test_the_version_is_not_a_literal_in_the_package():
    """No release-shaped literal is assigned to ``__version__`` again.

    Pinning it to the metadata alone cannot catch the regression: a
    hardcoded constant that happens to match today's install passes that
    assertion and then rots at the next cut, which is exactly what
    happened for four releases.
    """

    source = Path(gpuwm.__file__).read_text(encoding="utf-8")
    assignments = re.findall(r"^\s*__version__\s*=\s*(.+)$", source, re.M)
    assert assignments, "__version__ must still be defined"
    assert not any(re.fullmatch(r"""["']\d+\.\d+.*["']""", value.strip())
                   for value in assignments), assignments
