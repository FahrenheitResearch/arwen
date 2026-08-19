"""The GPU-marking discipline of ``tests/conftest.py``, for this tree too.

WHY THIS FILE EXISTS
--------------------
pytest conftests are directory-scoped, so for as long as ``tilestream/`` sat
outside ``testpaths`` the GPU discipline never applied here: a
``pytest tilestream/`` run under ``GPUWM_NO_LOCAL_GPU=1`` -- the stage-1
leg's own definition -- imported cupy, opened the local card, and ran every
device round trip on the machine whose owner had asked that no GPU work run
there.  MEASURED on the 2.5.0 gating pass: the full tilestream pytest
surface ran 289 tests on the local 5090 with the ban variable set.

The rules are DELEGATED, not copied.  ``tests/conftest.py`` is the single
authority on what "GPU-bound" means in this repository (AST detection of
cupy imports, module-level vs function-level granularity, the
``CUDA_VISIBLE_DEVICES=-1`` backstop), and a second implementation here
would drift from it silently.  This file loads that module by path and
re-invokes its collection hook.  When the whole repository is collected at
once the hook runs twice over these items; ``add_marker`` is idempotent in
effect (a duplicate marker changes no verdict), and the skip decision is
identical both times.

The path reach-over is legitimate exactly because this file is a CHECKOUT
artifact: pytest only loads conftests for trees it collects, and the only
supported way to collect tilestream's suites is from the repository root
(the battery lists and testpaths both say so).  An installed wheel carries
``tilestream/`` but nobody collects tests out of site-packages; if someone
does, failing loudly here -- naming the layout -- beats silently running
device work on a banned card.
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import pathlib

_TESTS_CONFTEST = (pathlib.Path(__file__).resolve().parents[1]
                   / "tests" / "conftest.py")

if not _TESTS_CONFTEST.is_file():
    raise ImportError(
        f"tilestream/conftest.py delegates the GPU-marking discipline to "
        f"{_TESTS_CONFTEST}, which does not exist.  These suites are "
        f"collected from a repository checkout (see tools/battery/); "
        f"collecting an installed tilestream out of site-packages is not a "
        f"supported test surface.")

_spec = importlib.util.spec_from_file_location(
    "_gpuwm_tests_conftest_for_tilestream", _TESTS_CONFTEST)
_authority = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_authority)          # also plants the -1 backstop

#: Re-exported so a tilestream suite can spell an explicit skip the same way
#: tests/ do (``from tilestream.conftest import requires_gpu``).
HAS_GPU = _authority.HAS_GPU
NO_LOCAL_GPU = _authority.NO_LOCAL_GPU
requires_gpu = _authority.requires_gpu


def pytest_collection_modifyitems(config, items):
    _authority.pytest_collection_modifyitems(config, items)


@functools.lru_cache(maxsize=None)
def _has_pytest_tests(path: str) -> bool:
    """Whether a ``test_*.py`` module defines any top-level ``test_`` function.

    tilestream/ carries ~29 SCRIPT gates beside its 13 pytest suites -- the
    ``python -m tilestream.test_gate`` class tools/battery/tiles_gates.txt
    runs, whole-forecast comparisons with a ``main()`` and no ``test_``
    functions.  pytest would import every one of them at collection to
    collect nothing, and one (``test_share.py``) imports cupy at module
    scope, so a box without cupy could not collect this tree at all.  The
    split is derived from each module's own AST rather than from a name
    list, so a script gate that grows a real test function is collected the
    moment it does.
    """
    try:
        tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return True                      # let pytest report the real error
    # Top-level test functions, or a Test* class (whose tests are methods):
    # either shape makes the module a pytest suite.  No tilestream module
    # uses the class shape today, but silently dropping the first one that
    # does would be this hook reproducing the very gap it exists to close.
    return any(
        (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
         and node.name.startswith("test_"))
        or (isinstance(node, ast.ClassDef) and node.name.startswith("Test"))
        for node in tree.body)


def pytest_ignore_collect(collection_path, config):
    # SCOPED to this directory by hand: once a conftest is loaded its hooks
    # fire for every path in the session, and tests/ legitimately carries
    # test files whose tests are class methods -- the top-level-def probe
    # below would silently drop those.  Only tilestream's own script gates
    # are this hook's business.
    path = pathlib.Path(collection_path)
    here = pathlib.Path(__file__).resolve().parent
    if path.suffix == ".py" and path.name.startswith("test_") \
            and path.resolve().parent == here \
            and not _has_pytest_tests(str(path)):
        return True
    return None
