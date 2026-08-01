"""Record what the production paths actually say about a run's physics.

The finding this receipt answers is that the registry's own maturity labels
were being read as statements about the radiation ENGINE, while the engine
is chosen by config fields the registry never sees.  So the receipt is
produced by running the production resolvers -- the registry loader,
``gpuwm.config.radiation_scheme_ids`` and
``gpuwm.io.wrfout.wrf_physics_selector_attrs`` -- and writing down whatever
they return, for the default template and for the legacy-RRTMG run of
record.  It asserts nothing; a test regenerates it and fails on drift.

Usage
-----
    python tools/report_registry_ground_truth.py --out <path>
"""
from __future__ import annotations

import argparse
import pathlib
import sys

MODEL = pathlib.Path(__file__).resolve().parents[1]
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

from gpuwm.physics_registry import canonical_json  # noqa: E402

RECEIPT_PATH = MODEL / "docs" / "public" / "receipts" / "F2-ground-truth.json"

#: The run of record whose radiation engine the registry label cannot see.
#: Named as a config path (data), never as a case token in an identifier.
LEGACY_RRTMG_CONFIG = (
    "configs/real74_thompson_1218z_rrtmg_legacy_4dom.toml"
)


def _template_view(registry: dict, template_id: str) -> dict[str, object]:
    template = registry["templates"][template_id]
    components = registry.get("components", {})
    selected = template.get("components", {})
    return {
        "template_id": template_id,
        "maturity": template.get("maturity"),
        "components": {
            component_id: {
                "option_id": selected[component_id],
                "maturity": (
                    components.get(component_id, {})
                    .get("options", {})
                    .get(selected[component_id], {})
                    .get("maturity")
                ),
                "scientific_evidence": (
                    components.get(component_id, {})
                    .get("options", {})
                    .get(selected[component_id], {})
                    .get("scientific_evidence")
                ),
            }
            for component_id in sorted(selected)
        },
        "radiation_engine_parameters": {
            key: template.get("parameters", {}).get(key)
            for key in ("ra_rrtmg_variant", "wrf_rrtmg_compatibility")
        },
    }


def per_domain_physics_tuple(config_path: pathlib.Path) -> list[dict]:
    """Resolve each domain's emitted physics tuple through production code.

    A four-domain run config is an EXPERIMENT document carrying a
    ``[case_data]`` table, so its production loader is
    :func:`gpuwm.case_data.load_experiment_case`, not
    :func:`gpuwm.config.load_config` -- the latter refuses the document
    (``unknown table(s) ['experiment', 'projection', 'shared', 'domain',
    'case_data']``).  ``load_experiment_case`` builds each domain's
    ``RunConfig`` with the same ``validate_run_config`` battery, and those
    are the objects the selector resolvers consume, so the tuple below is
    the one a run of this config actually writes.
    """

    from gpuwm.case_data import load_experiment_case
    from gpuwm.config import radiation_scheme_ids
    from gpuwm.io.wrfout import wrf_physics_selector_attrs

    experiment, _case_data = load_experiment_case(config_path)
    rows = []
    for domain in experiment.domains:
        run = domain.run
        lw, sw = radiation_scheme_ids(run)
        attrs = wrf_physics_selector_attrs(run)
        rows.append({
            "grid_id": domain.grid_id,
            "radiation_scheme_ids": {"lw": int(lw), "sw": int(sw)},
            "wrf_physics_selector_attrs": {
                name: int(value) for name, value in sorted(attrs.items())
            },
            "radiation_engine_fields": {
                field: getattr(run, field, None)
                for field in ("ra_rrtmg_variant", "wrf_rrtmg_compatibility")
            },
        })
    return rows


def _config_view(config_path: pathlib.Path) -> dict[str, object]:
    return {
        "config_path": config_path.relative_to(MODEL).as_posix(),
        "loader": "gpuwm.case_data.load_experiment_case",
        "per_domain": per_domain_physics_tuple(config_path),
    }


def build(registry: dict) -> dict[str, object]:
    from gpuwm.physics_registry import DEFAULT_TEMPLATE_ID, registry_sha256

    return {
        "schema": "gpuwm-f2-ground-truth-v1",
        "registry_version": registry.get("registry_version"),
        "registry_sha256": registry_sha256(registry),
        "default_template": _template_view(registry, DEFAULT_TEMPLATE_ID),
        "run_of_record": _config_view(MODEL / LEGACY_RRTMG_CONFIG),
        "note": (
            "radiation_scheme_ids and wrf_physics_selector_attrs resolve the "
            "WRF selector integers; neither carries the RRTMG engine variant, "
            "so a registry maturity label and a selector integer together "
            "still do not name which radiation code ran."),
    }


def render(report: dict) -> bytes:
    return (canonical_json(report) + "\n").encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=pathlib.Path, default=RECEIPT_PATH)
    args = parser.parse_args(argv)

    from gpuwm.physics_registry import physics_registry

    report = build(physics_registry())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(render(report))
    print("registry_sha256:", report["registry_sha256"], "|", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
