"""Shared mapped preparation admits only its already captured source plan."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from gpuwm.ensemble.mapped_posted_reuse import shared_input_plan
from gpuwm.ensemble.physical_store import digest_file
from gpuwm.ensemble.recipes import SourceTrajectory
from gpuwm.ingest.boundary_stream import input_plan


@pytest.fixture
def mapped_request(tmp_path):
    mapping, composition, provenance = (tmp_path / name for name in ("mapping.json", "composition.json", "provenance.json"))
    for path in (mapping, composition, provenance):
        path.write_text("{}")
    raw = tmp_path / "not-yet-posted.grib2"
    manifest = {"schema": "gpuwm-mapped-composition-inputs-v1",
        "mapping_sha256": digest_file(mapping), "composition_sha256": digest_file(composition),
        "primary_files": [{"path": raw.name, "bytes": None, "sha256": None}],
        "supplements": {"surface": [{"path": raw.name, "bytes": None, "sha256": None}]},
        "provenance": {"surface": {"path": provenance.name, "bytes": 2, "sha256": digest_file(provenance)}},
        "decoders": {"gpuwm_mapped_engine": {"path": "absent-decoder", "bytes": 123, "sha256": "a" * 64}}}
    plan = input_plan(manifest, lead_role_prefix="", route_table_sha256="b" * 64, fixed_rows=[])
    start = datetime(2024, 5, 21, 12, tzinfo=timezone.utc)
    selected = SourceTrajectory("gefs", start, "p19")
    times = (start, start + timedelta(hours=3))
    context = SimpleNamespace(verify=lambda: None, source_plan=plan, trajectory=selected,
        physical_stream=SimpleNamespace(times=times),
        prepared_head={"basis": {"as_posted": {"forcing_leads": [0, 3], "fixed_rows": []}}})
    posted = SimpleNamespace(physical_trajectory=lambda: selected, leads=(0, 3),
        valid_times=tuple(value.replace(tzinfo=None) for value in times), fixed=(),
        input_manifest=tmp_path / "inputs.json", posted=SimpleNamespace(route_table_sha256=lambda: "b" * 64))
    kwargs = dict(mapping=mapping, composition=composition, primary=(raw,),
        supplements={"surface": (raw,)}, provenance={"surface": provenance}, contributing={})
    return context, posted, kwargs


def test_shared_plan_needs_no_decoder_or_future_raw_files(mapped_request):
    context, posted, kwargs = mapped_request
    assert not kwargs["primary"][0].exists()
    result = shared_input_plan(context, posted, **kwargs)
    assert result["plan"] == context.source_plan
    result["plan"]["manifest"]["decoders"].clear()
    assert context.source_plan["manifest"]["decoders"]


@pytest.mark.parametrize("mutation", ["mapping", "primary", "supplement", "provenance", "member", "lead", "fixed"])
def test_shared_plan_rejects_changed_native_input_authority(mapped_request, mutation):
    context, posted, kwargs = mapped_request
    if mutation == "mapping": kwargs["mapping"].write_text('{"changed":true}')
    if mutation == "primary": kwargs["primary"] = (kwargs["primary"][0].with_name("other.grib2"),)
    if mutation == "supplement": kwargs["supplements"] = {}
    if mutation == "provenance": kwargs["provenance"]["surface"].write_text("changed")
    if mutation == "member": posted.physical_trajectory = lambda: SourceTrajectory("gefs", context.trajectory.cycle, "p18")
    if mutation == "lead": posted.leads = (0, 6)
    if mutation == "fixed": posted.fixed = kwargs["primary"]
    with pytest.raises(ValueError):
        shared_input_plan(context, posted, **kwargs)
