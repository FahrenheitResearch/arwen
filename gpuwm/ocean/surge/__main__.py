"""Permanent command line door for the surge package."""
import argparse
import json
from pathlib import Path
from .driver import prepare, run, finish


def main():
    parser = argparse.ArgumentParser(description="GPU storm surge, wetting/drying and coastal inundation")
    parser.add_argument("command", choices=["prepare", "run", "finish"])
    parser.add_argument("config", type=Path, help="surge input contract JSON, documented in README.md")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-render", action="store_true", help="skip map creation for numerical checks")
    args = parser.parse_args()
    result = prepare(args.config, args.out) if args.command == "prepare" else (finish if args.command == "finish" else run)(
        args.config, args.out, render=not args.no_render)
    print(json.dumps(result, default=str, indent=2))


if __name__ == "__main__":
    main()
