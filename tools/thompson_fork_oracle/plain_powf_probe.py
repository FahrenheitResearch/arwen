"""Retired entry point for the former two-site plain-powf measurement."""
import sys


def main():
    print("plain_powf_probe is retired: every mp=28 site uses WOOF's own words. "
          "Run test_no_plain_powf_survives_in_any_mp28_arm and the column oracle "
          "instead. The former two-site receipt no longer applies.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
