"""Pure configuration dependency for nested CAM ozone consumers."""
from __future__ import annotations


def cam_ozone_domain_ids(exp):
    """Domains carrying root CAM ozone for an actual nested consumer."""
    from gpuwm.config import radiation_scheme_ids
    from gpuwm.physics_compat import RRTMG_VARIANT_LEGACY, rrtmg_variant
    domains = {domain.grid_id: domain for domain in exp.domains}
    required = set()
    for domain in exp.domains:
        cfg = domain.run
        if (domain.parent_id == 0 or 4 not in radiation_scheme_ids(cfg)
                or rrtmg_variant(cfg) != RRTMG_VARIANT_LEGACY or cfg.o3input != 2):
            continue
        current = domain
        chain = set()
        while current is not None:
            if current.grid_id in chain:
                raise ValueError("CAM ozone dependency contains a parent cycle")
            chain.add(current.grid_id)
            required.add(current.grid_id)
            current = None if current.parent_id == 0 else domains[current.parent_id]
    return frozenset(required)
