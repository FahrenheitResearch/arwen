"""Concrete source preparation preserves posted handoffs and member identity."""
from dataclasses import replace
from datetime import datetime, timezone
import json

import pytest

from gpuwm.ensemble.source_preparation import PostedPreparationFactory, PostedSourcePreparation
from gpuwm.ensemble.recipes import SourceTrajectory
from gpuwm.fetch_routes import PREP_ARGUMENTS_SCHEMA
from gpuwm.source_posting import SCHEDULE_SCHEMA
from test_ensemble_posted_physical import bridge, portable_provider
from test_posted_prep_handoff import _handoff


def _spec(tmp_path):
    acquisition = tmp_path / "fetch"
    acquisition.mkdir()
    _handoff(acquisition)
    cfg = tmp_path / "experiment.toml"
    cfg.write_text("[run]\nnx = 3\n")
    trajectory = SourceTrajectory("gefs", datetime(2024, 5, 21, 12, tzinfo=timezone.utc), "p01")
    return PostedSourcePreparation.from_acquisition(trajectory,
        acquisition_root=acquisition, prepared_root=tmp_path / "ordinary",
        physical_root=tmp_path / "physical", native_arguments=(
            "--experiment-config", str(cfg), "--preprocess-backend", "cpu"))


def test_native_source_request_does_not_require_future_payloads(tmp_path):
    specification = _spec(tmp_path)
    arguments = specification.source_arguments()
    assert arguments[arguments.index("--source") + 1] == "gefs"
    assert arguments[arguments.index("--physical-output-store") + 1] == str(specification.physical_root)
    assert not (specification.acquisition_root / "p01.f003.grib2").exists()


@pytest.mark.parametrize("change", ["config", "handoff"])
def test_native_source_request_refuses_configuration_or_handoff_drift(tmp_path, change):
    specification = _spec(tmp_path)
    if change == "config":
        (tmp_path / "experiment.toml").write_text("[run]\nnx = 4\n")
    else:
        path = specification.acquisition_root / "prep-arguments.json"
        document = json.loads(path.read_text())
        document["argv"] += ["--p-top-pa", "10000"]
        path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="changed"):
        specification.source_arguments()


@pytest.mark.parametrize("option", ["--source", "--source=gfs", "--physical-member-index", "--as-posted"])
def test_native_controls_cannot_replace_recipe_acquisition(tmp_path, option):
    specification = _spec(tmp_path)
    with pytest.raises(ValueError, match="cannot replace"):
        PostedSourcePreparation.from_acquisition(specification.trajectory,
            acquisition_root=specification.acquisition_root,
            prepared_root=specification.prepared_root, physical_root=specification.physical_root,
            native_arguments=(option, "1"))


@pytest.fixture
def factory(tmp_path, portable_provider):
    provider, *_ = portable_provider
    trajectory = provider.recipe.members[0].trajectory
    acquisition = tmp_path / "acquisition"
    acquisition.mkdir()
    posting = acquisition / "posting"
    posting.mkdir()
    (posting / "schedule.json").write_text(json.dumps({"schema": SCHEDULE_SCHEMA,
        "source": trajectory.source, "cycle": trajectory.cycle.isoformat(), "member": None,
        "leads": [{"lead": value} for value in range(3)]}))
    (acquisition / "prep-arguments.json").write_text(json.dumps({"schema": PREP_ARGUMENTS_SCHEMA,
        "source": trajectory.source, "cycle": trajectory.cycle.isoformat(), "member": None,
        "as_posted": True, "posting": str(posting),
        "argv": ["--source", trajectory.source, "--gfs-series", str(acquisition / "series.tsv")]}))
    configuration = tmp_path / "configuration.toml"
    configuration.write_text("[run]\nnx = 3\n")
    specification = PostedSourcePreparation.from_acquisition(trajectory,
        acquisition_root=acquisition, prepared_root=provider.prepared_roots[trajectory.identity],
        physical_root=provider.streams[trajectory.identity].root,
        native_arguments=("--experiment-config", str(configuration)))
    return PostedPreparationFactory.publish(provider.recipe, {trajectory.identity: specification},
        root=tmp_path / "factory", cpu_bridge=provider.cpu_bridge)


def test_factory_reopens_native_plan_and_keeps_original_member(factory, tmp_path, bridge):
    reopened = PostedPreparationFactory.open(factory.provider.root, cpu_bridge=bridge)
    arguments = reopened.member_arguments(17, output_root=tmp_path / "member")
    assert arguments[arguments.index("--physical-member-index") + 1] == "17"
    assert "--physical-output-store" not in arguments
    assert arguments[arguments.index("--physical-input-provider") + 1] == str(factory.provider.root)
    source = next(iter(factory.sources.values()))
    assert arguments[:len(source.preparation_arguments)] == list(source.preparation_arguments)
    with pytest.raises(ValueError, match="original recipe index"):
        reopened.member_arguments(0, output_root=tmp_path / "member")


def test_factory_cannot_overwrite_bound_source_or_provider(factory):
    source = next(iter(factory.sources.values()))
    for output in (source.prepared_root, source.prepared_root / "member",
                   source.physical_root, source.acquisition_root, factory.provider.root):
        with pytest.raises(ValueError, match="overwrite"):
            factory.member_arguments(17, output_root=output)


def test_factory_rejects_replaced_configuration_after_serialization(factory, bridge):
    source = next(iter(factory.sources.values()))
    record = source.configuration_files["--experiment-config"]
    from pathlib import Path
    Path(record["path"]).write_text("changed native controls\n")
    with pytest.raises(ValueError, match="frozen preparation"):
        PostedPreparationFactory.open(factory.provider.root, cpu_bridge=bridge)


def test_factory_does_not_accept_another_ordinary_source_root(factory, tmp_path):
    identity, source = next(iter(factory.sources.items()))
    with pytest.raises(ValueError, match="ordinary source root"):
        PostedPreparationFactory(factory.provider, {identity: replace(source, prepared_root=tmp_path / "other")})


def test_factory_does_not_accept_another_captured_physical_root(factory, tmp_path):
    identity, source = next(iter(factory.sources.items()))
    with pytest.raises(ValueError, match="captured physical root"):
        PostedPreparationFactory(factory.provider, {identity: replace(source, physical_root=tmp_path / "other")})
    with pytest.raises(ValueError, match="overwrite"):
        factory.member_arguments(17, output_root=factory.provider.streams[identity].root)


def test_only_captured_raw_decoder_can_be_absent_for_a_consumer(tmp_path):
    from gpuwm.ensemble.source_preparation import _config_files
    from gpuwm.ensemble.physical_store import digest_file
    decoder = tmp_path / "raw-decoder"
    initializer = tmp_path / "native-initializer"
    config = tmp_path / "configuration.toml"
    decoder.write_bytes(b"pinned ordinary raw decoder")
    initializer.write_bytes(b"live native initialization")
    config.write_text("[run]\n")
    args = ("--bridge", str(decoder), "--cpu-preprocess-bridge", str(initializer),
            "--experiment-config", str(config))
    saved = _config_files(args)
    authority = (digest_file(decoder),)
    decoder.unlink()
    with pytest.raises(FileNotFoundError):
        _config_files(args)
    with pytest.raises(FileNotFoundError):
        _config_files(args, captured=saved, native_decoder_sha256s=("a" * 64,))
    assert _config_files(args, captured=saved, native_decoder_sha256s=authority) == saved
    decoder.write_bytes(b"changed executable")
    assert _config_files(args, captured=saved, native_decoder_sha256s=authority) != saved
    decoder.unlink()
    initializer.unlink()
    with pytest.raises(FileNotFoundError):
        _config_files(args, captured=saved, native_decoder_sha256s=authority)


def test_factory_owned_verification_retains_consumer_mode_but_producer_is_strict(tmp_path):
    from gpuwm.ensemble.source_preparation import _config_files
    from gpuwm.ensemble.physical_store import digest_file
    specification = _spec(tmp_path)
    decoder = tmp_path / "captured-decoder"
    decoder.write_bytes(b"verified producer decoder")
    arguments = (*specification.native_arguments, "--bridge", str(decoder))
    consumer = replace(specification, native_arguments=arguments,
        configuration_files=_config_files(specification.preparation_arguments + arguments),
        _consumer_decoder_sha256s=frozenset((digest_file(decoder),)))
    decoder.unlink()
    assert consumer.verify() is consumer
    assert consumer.describe()["configuration_files"] == consumer.configuration_files
    with pytest.raises(FileNotFoundError):
        consumer.source_arguments()


def test_factory_reopens_after_owned_tree_moves_behind_alias(factory, tmp_path, bridge):
    """A storage move keeps immutable source plans and exact source bytes."""
    old_root = tmp_path
    moved_root = tmp_path.with_name(tmp_path.name + "-relocated")
    factory_relative = factory.provider.root.relative_to(old_root)
    old_document = (factory.provider.root / "native-preparation.json").read_bytes()
    old_plan = (factory.provider.root / "provider-head.json").read_bytes()
    old_root.rename(moved_root)
    try:
        try:
            old_root.symlink_to(moved_root, target_is_directory=True)
        except OSError as error:
            pytest.skip(f"directory aliases unavailable: {error}")
        reopened = PostedPreparationFactory.open(moved_root / factory_relative, cpu_bridge=bridge)
        assert reopened.plan_sha256 == factory.plan_sha256
        assert (reopened.provider.root / "native-preparation.json").read_bytes() == old_document
        assert (reopened.provider.root / "provider-head.json").read_bytes() == old_plan
        assert all(path.is_relative_to(moved_root) for path in reopened.provider.prepared_roots.values())
        assert "17" in reopened.member_arguments(17, output_root=moved_root / "member")
        configuration = next(iter(reopened.sources.values())).configuration_files["--experiment-config"]
        from pathlib import Path
        Path(configuration["path"]).write_text("changed configuration after relocation\n")
        with pytest.raises(ValueError, match="frozen preparation"):
            PostedPreparationFactory.open(moved_root / factory_relative, cpu_bridge=bridge)
    finally:
        if old_root.is_symlink():
            old_root.unlink()
        moved_root.rename(old_root)


def test_captured_absent_decoder_accepts_same_alias_but_not_another_path(tmp_path):
    from pathlib import Path
    from gpuwm.ensemble.source_preparation import _config_files
    old = tmp_path / "original"
    old.mkdir()
    decoder = old / "decoder"
    decoder.write_bytes(b"captured decoder")
    saved = _config_files(("--bridge", str(decoder)))
    digest = saved["--bridge"]["sha256"]
    decoder.unlink()
    moved = tmp_path / "relocated"
    old.rename(moved)
    try:
        old.symlink_to(moved, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory aliases unavailable: {error}")
    assert _config_files(("--bridge", str(moved / "decoder")),
        captured=saved, native_decoder_sha256s=(digest,)) == saved
    with pytest.raises(FileNotFoundError):
        _config_files(("--bridge", str(moved / "other-decoder")),
            captured=saved, native_decoder_sha256s=(digest,))
    Path(old / "decoder").write_bytes(b"different installed decoder")
    assert _config_files(("--bridge", str(moved / "decoder")),
        captured=saved, native_decoder_sha256s=(digest,)) != saved
