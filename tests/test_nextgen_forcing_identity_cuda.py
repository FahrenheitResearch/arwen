"""Actual forcing/PET factories and CFE admission; synthetic two-row geometry."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import json
import unittest

import cupy as cp

from gpuwm.hydro.forcing import FIELDS, PreparedForcingSeries, prepare_parent
from gpuwm.hydro.pet import PARAMETER_NAMES, prepare as prepare_pet
from gpuwm.hydro.cfe import prepare as prepare_cfe

IDS = ["cat-a", "cat-b"]
ORIGIN = "2023-01-05T00:00:00Z"
PARAMETERS = dict(zip(PARAMETER_NAMES, (10., 2., .5, .1, .03, .003,
                                      .98, .23, 70., 20., 0., 25.)))


def forcing(kind, *, ids=IDS, origin=ORIGIN):
    values = (298., .008, 3., 1., 100000., 310., 450., .01)
    if kind == "series":
        frames = cp.empty((2, 8, 2), cp.float32)
        for index, value in enumerate(values):
            frames[:, index, :].fill(value)
        return PreparedForcingSeries(frames, interval_seconds=3600,
            time_origin=origin, identity={"catchment_ids": list(ids),
                                         "scope": "synthetic identity regression"})
    fields = {key: cp.full((1, 2), value, cp.float32)
              for key, value in zip(FIELDS, values)}
    fabric = SimpleNamespace(catchment_ids=tuple(ids), arrays={
        "weights_offsets": cp.asarray([0, 1, 2], cp.uint64),
        "weights_indices": cp.asarray([0, 1], cp.uint32),
        "weights": cp.ones(2, cp.float64)},
        restart_identity={"scope": "synthetic sealed-fabric interface"})
    # Only the setup fabric reader is substituted. Both production CUDA
    # preparation factories and their arrays/kernels remain real.
    with patch("gpuwm.hydrofabric.load_cuda", return_value=fabric):
        return prepare_parent(settings={}, options={
            "hydrofabric_manifest": "synthetic-metadata-interface",
            "parent_fields": {key: "components.fields." + key for key in FIELDS},
            "time_origin": origin}, state=None, physics=None, device_id=0,
            mode="coupled", components={"fields": fields})


def pet(source, *, heterogeneous=False):
    options = {"forcing_component": "forcing", "parameters": PARAMETERS, "method": 5}
    if heterogeneous:
        options = {"forcing_component": "forcing", "catchment_parameters": [
            {"catchment_id": key, "method": method, "parameters": PARAMETERS}
            for key, method in zip(IDS, (1, 5))]}
    return prepare_pet(settings={}, options=options, state=None, physics=None,
                       device_id=0, mode="standalone", components={"forcing": source})


class ForcingIdentity(unittest.TestCase):
    def test_parent_heterogeneous_pet(self):
        source = forcing("parent")
        component = pet(source, heterogeneous=True)
        self.assertEqual(component.restart_identity["identity"]["catchment_ids"], IDS)
        source.advance(3600)
        component.advance(3600)
        self.assertTrue(bool(cp.isfinite(component.fields["potential_et_m_s"]).all()))

    def test_pet_exposes_order_and_epoch(self):
        for kind in ("parent", "series"):
            with self.subTest(kind=kind):
                source = forcing(kind)
                identity = pet(source).restart_identity["identity"]
                self.assertEqual(identity["catchment_ids"], IDS)
                self.assertEqual(identity["time_origin"], ORIGIN)
                self.assertEqual(identity["forcing"], source.restart_identity)

    def mismatch(self, kind, stage, field):
        changes = {"ids": list(reversed(IDS))} if field == "order" else {
            "origin": "2023-01-05T01:00:00Z"}
        good = forcing(kind)
        changed = forcing(kind, **changes)
        precipitation = changed if stage == "precipitation" else good
        evaporation = pet(changed if stage == "PET" else good)
        with TemporaryDirectory() as directory:
            manifest = Path(directory) / "capture.json"
            # A minimal capture header suffices: admission must refuse before
            # opening any payload, resolving its Rust bridge, or advancing CFE.
            manifest.write_text(json.dumps({"schema": "cfe.capture.pack/v1"}))
            with self.assertRaisesRegex(ValueError, "CFE .* differs from the prepared " + stage):
                prepare_cfe(settings={}, options={"manifest": str(manifest),
                    "forcing_component": "forcing", "pet_component": "pet",
                    "initial_step": 0, "time_step_fraction": 1.,
                    "identity": {"catchment_ids": IDS, "time_origin": ORIGIN}},
                    state=None, physics=None, device_id=0, mode="standalone",
                    components={"forcing": precipitation, "pet": evaporation})


def _case(kind, stage, field):
    def test(self):
        self.mismatch(kind, stage, field)
    return test


for _kind in ("parent", "series"):
    for _stage in ("precipitation", "PET"):
        for _field in ("order", "epoch"):
            setattr(ForcingIdentity, f"test_{_kind}_{_stage.lower()}_{_field}",
                    _case(_kind, _stage, _field))
