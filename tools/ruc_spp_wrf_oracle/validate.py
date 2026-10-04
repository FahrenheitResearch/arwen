#!/usr/bin/env python3
"""Replay whole warm/snow stages against the pinned native hydraulic overlay."""
from __future__ import annotations

import argparse
import csv
from dataclasses import fields
import json
from pathlib import Path

import numpy as np

from gpuwm.core.fp32_ulp import fp32_ulp_distance
from gpuwm.core.ruc import (RUC_SOIL_STEP_COLUMN_INPUTS,
                            RUC_SNOW_SOIL_COLUMN_INPUTS,
                            ruc_soil_step, ruc_snow_soil_step)


def read_csv(path):
    with Path(path).open(encoding="ascii", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ncase = len(set(row["case"] for row in rows))
    return {name: np.asarray([float(row[name]) for row in rows], dtype=np.float32)
            .reshape(ncase, 9).T.copy() for name in rows[0] if name != "case"}


def replay(stage, oracle, mode=1, *, gpu=False, order=None):
    selection = np.arange(oracle["k"].shape[1]) if order is None else np.asarray(order)
    data = {name: value[:, selection].copy() for name, value in oracle.items()}
    names = RUC_SOIL_STEP_COLUMN_INPUTS if stage == "soil" else RUC_SNOW_SOIL_COLUMN_INPUTS
    values = {name: data[name + "_before"].copy()
              for name in ("soilmois", "tso", "smfrkeep", "keepfr")}
    values.update({name: data.get(name + "_before", data.get(name))[0].copy() for name in names})
    pattern = data["rstochcol"].copy()
    diagnostic = np.zeros_like(pattern)
    iland = data["iland"][0].astype(np.int32)
    kwargs = dict(nroot=data["nroot"][0].astype(np.int32),
                  delt=float(data["delt"][0, 0]), conflx=float(data["conflx"][0, 0]),
                  spp_lsm=mode, rstochcol=pattern, fieldcol_sf=diagnostic)
    function = ruc_soil_step if stage == "soil" else ruc_snow_soil_step
    if stage == "snowsoil":
        kwargs.update(ilnb=data["ilnb_before"][0].astype(np.int32),
                      cw=float(data["cw"][0, 0]))
    if gpu:
        import cupy as cp
        from gpuwm.core.ruc_gpu import ruc_soil_step_cuda, ruc_snow_soil_step_cuda
        function = ruc_soil_step_cuda if stage == "soil" else ruc_snow_soil_step_cuda
        values = {name: cp.asarray(value) for name, value in values.items()}
        iland = cp.asarray(iland)
        kwargs = {name: cp.asarray(value) if isinstance(value, np.ndarray) else value
                  for name, value in kwargs.items()}
        diagnostic = kwargs["fieldcol_sf"]
    result = function(values, iland, **kwargs)
    actual = {field.name: getattr(result, field.name) for field in fields(result)}
    actual["fieldcol_sf"] = diagnostic
    if gpu:
        actual = {name: cp.asnumpy(value) for name, value in actual.items()}
    return actual


def assess(stage, oracle, actual, *, require_exact=False, order=None):
    result = {}
    for name, value in actual.items():
        expected = oracle.get(name + "_after", oracle.get(name))
        if order is not None:
            expected = expected[:, order]
        if value.ndim == 1:
            expected = expected[0]
        if value.dtype.kind in "iu":
            np.testing.assert_array_equal(value, expected.astype(value.dtype), err_msg=name)
            result[name] = 0
            continue
        distance = int(np.max(fp32_ulp_distance(value, expected)))
        signed_zero = int(np.count_nonzero((value == 0) & (expected == 0)
                                          & (np.signbit(value) != np.signbit(expected))))
        tolerance = 0 if require_exact or stage != "soil" or name not in (
            "edir1", "eeta", "qfx", "evapl") else 2
        if distance > tolerance or signed_zero:
            raise AssertionError(f"{stage}/{name}: ULP={distance}, signed zeros={signed_zero}")
        result[name] = distance
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("oracle", type=Path)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.oracle / "manifest.json").read_text(encoding="utf-8"))
    summary = []
    for record in manifest["records"]:
        stage, label = record["stage"], record["label"]
        if record["variant"] != "hydraulic_overlay":
            continue
        oracle = read_csv(args.oracle / record["file"])
        actual = replay(stage, oracle, record["mode"], gpu=args.gpu)
        distances = assess(stage, oracle, actual)
        reordered = replay(stage, oracle, record["mode"], gpu=args.gpu, order=[3, 0, 2, 1])
        assess(stage, oracle, reordered, order=[3, 0, 2, 1])
        for name in actual:
            np.testing.assert_array_equal(reordered[name], actual[name][..., [3, 0, 2, 1]], err_msg=name)
        baseline = read_csv(args.oracle / "current" / f"{stage}-off.csv")
        if label in ("off", "zero"):
            for name, value in oracle.items():
                if name != "rstochcol":
                    np.testing.assert_array_equal(value.view(np.uint32), baseline[name].view(np.uint32), err_msg=name)
        else:
            if not np.any(oracle["soilmois_after"] != baseline["soilmois_after"]):
                raise AssertionError(f"{stage}/{label}: hydraulic pattern did not change moisture")
            if not np.any(oracle["fieldcol_sf"] != 0):
                raise AssertionError(f"{stage}/{label}: missing conductivity diagnostic")
        summary.append({**record, "outputs": len(distances), "max_ulp": max(distances.values()),
                        "ulp_by_field": distances,
                        "changed_moisture_cells": int(np.count_nonzero(oracle["soilmois_after"] != baseline["soilmois_after"])),
                        "column_permutation_exact": True})
    report = {"gpu": args.gpu, "whole_stage_checks": len(summary), "results": summary}
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
