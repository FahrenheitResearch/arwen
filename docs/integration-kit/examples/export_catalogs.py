"""Export installed catalogs with the local home directory replaced by a placeholder."""
import argparse
import json
from pathlib import Path
import sys
from client import query


def portable(value):
    if isinstance(value, str):
        return value.replace(str(Path.home()), "<USER_HOME>")
    if isinstance(value, list):
        return [portable(item) for item in value]
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, arguments, schema in [
        ("sources", ["sources", "--json"], "gpuwm.run-plan.sources.v1"),
        ("physics-profiles", ["run-plan", "--physics-profiles"], "gpuwm.run-plan.physics-profiles.v1"),
        ("catalog", ["run-plan", "--catalog"], "gpuwm.run-plan.catalog.v1"),
    ]:
        document = query(args.python, arguments, schema, cwd=args.cwd)
        (args.output / f"{name}.json").write_text(json.dumps(portable(document), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
