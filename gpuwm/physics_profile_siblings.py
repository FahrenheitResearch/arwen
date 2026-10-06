"""Named profile siblings used by prepared runner menus."""
from __future__ import annotations

PROFILE_SIBLINGS = (
    (
        "thompson-mp8-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1",
        "thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1",
    ),
    (
        "thompson-mp8-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1",
        "thompson-mp8-mynn-mynn-ruc-monthly-rrtmg-legacy-v1",
    ),
    (
        "thompson-mp8-mynn-mynn-ruc-monthly-rrtmg-legacy-v1",
        "thompson-mp8-mynn-mynn-ruc-monthly-solar-rrtmg-legacy-v1",
    ),
)


def with_profile_siblings(profiles, *, siblings=PROFILE_SIBLINGS):
    """Insert siblings beside their bases without creating a first offer.

    The table order handles siblings whose base is another sibling. Profiles
    outside the sibling table retain their order, and generation is stable.
    """
    result = list(profiles)
    for base, sibling in siblings:
        if base not in result:
            continue
        if sibling in result:
            result.remove(sibling)
        result.insert(result.index(base) + 1, sibling)
    return tuple(result)
