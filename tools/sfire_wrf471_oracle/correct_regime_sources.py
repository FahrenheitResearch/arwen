"""Create separate corrected reference sources without changing the WRF clone."""
from pathlib import Path
import argparse
import difflib
import hashlib
import json

from tools.sfire_wrf471_oracle.apply_corrected_tg import correct as correct_tg


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"WRF 4.7.1 correction anchor differs: {old!r}")
    return text.replace(old, new, 1)


def correct_phys(text):
    text = once(text, "        etas = 0.174* se(k)**(-0.19)",
        "        ! Corrected: extinction cannot produce negative spread or intensity.\n"
        "        if(fp%fmc_g(i,j).ge.fuelmce(k))then\n"
        "            etam=0.\n"
        "        else\n"
        "            etam=max(etam,0.)\n"
        "        endif\n"
        "        etas = 0.174* se(k)**(-0.19)")
    text = once(text, "real:: umid, phis, phiw, spdms, umidm, excess",
        "real:: umid, phis, phiw, spdms, umidm, excess, denominator, available")
    text = once(text, "    ros_back=.03333", "    ros_base=0.\n    ros_back=.03333")
    text = once(text,
        "    ros_wind = ros_wind - excess*ros_wind/(ros_wind+ros_slope)\n"
        "    ros_slope = ros_slope - excess*ros_slope/(ros_wind+ros_slope)",
        "    denominator=ros_wind+ros_slope\n"
        "    if(denominator.gt.0.)then\n"
        "        ros_base=min(ros_base,ros_max)\n"
        "        available=max(ros_max-ros_base,0.)\n"
        "        ros_wind=available*(ros_wind/denominator)\n"
        "        ros_slope=available-ros_wind\n"
        "    else\n"
        "        ros_base=ros_max\n"
        "    endif")
    return text


def correct_core(text):
    text = once(text, "start_x    = ignition_line%start_x",
                "ignited=0\nstart_x    = ignition_line%start_x")
    text = once(text, "              case(0)  ! none\n                  grad=",
                "              case(0)  ! corrected missing normal components\n"
                "                  diff2x=diffCx\n                  diff2y=diffCy\n                  grad=")
    text = once(text, "              case(4) ! Sethian - twice stronger pushdown of bumps\n                  grad=",
                "              case(4) ! corrected missing normal components\n"
                "                  diff2x=sign(sqrt(max(diffLx,0.)**2+min(diffRx,0.)**2),diffCx)\n"
                "                  diff2y=sign(sqrt(max(diffLy,0.)**2+min(diffRy,0.)**2),diffCy)\n                  grad=")
    return once(text, "min(advx,0.)*diffRy", "min(advx,0.)*diffRx")


def correct_model(text):
    # This is the existing declared WOOF correction, independently tested
    # against the compiled one-ULP recovery capture in driver-burn fixtures.
    # Preserve NaN, infinity and signed-zero behavior while preventing fuel
    # recovery from producing negative heat and smoke source terms.
    return once(text,
        "            fuel_frac_burnt(i,j)=fuel_frac(i,j)-fuel_frac_end(i,j)",
        "            if(abs(fuel_frac_end(i,j)).le.huge(1.).and. &\n"
        "               fuel_frac_end(i,j).gt.fuel_frac(i,j))then\n"
        "                fuel_frac_end(i,j)=fuel_frac(i,j)\n"
        "            endif\n"
        "            fuel_frac_burnt(i,j)=fuel_frac(i,j)-fuel_frac_end(i,j)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    pins = {}
    for component, correct in (("phys", correct_phys), ("core", correct_core),
                               ("model", correct_model), ("atm", correct_tg)):
        name = f"module_fr_fire_{component}.F"
        original = (args.source / "phys" / name).read_text()
        corrected = correct(original)
        output = args.destination / name
        output.write_text(corrected)
        (args.destination / (name + ".patch")).write_text("".join(difflib.unified_diff(
            original.splitlines(True), corrected.splitlines(True),
            fromfile="WRF-v4.7.1/"+name, tofile="corrected-regimes/"+name)))
        pins[name] = dict(original_sha256=hashlib.sha256(original.encode()).hexdigest(),
                          corrected_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    (args.destination / "corrections.json").write_text(json.dumps(pins, indent=2)+"\n")


if __name__ == "__main__":
    main()
