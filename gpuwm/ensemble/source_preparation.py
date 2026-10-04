"""Concrete native preparation of posted source recipes and member boundaries.

The ordinary source CLI owns decoding, native initialization and streaming.
This module binds its source handoffs and arguments once, then supplies the
selected physical provider at that same native initialization boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
import json
from pathlib import Path
import subprocess
import sys

from gpuwm.ensemble.physical_store import digest_file
from gpuwm.ensemble.posted_physical import PostedPhysicalProvider, PostedPhysicalStream
from gpuwm.ensemble.recipes import SourceRecipe, SourceTrajectory
from gpuwm.prep_handoff import posted_preparation_arguments_from_directory

_OWNED = {"--source", "--as-posted", "--output-root", "--physical-input-store",
          "--physical-input-provider", "--physical-member-index",
          "--physical-output-store", "--physical-base-prepared"}
_CONFIG_FILES = {"--experiment-config", "--wps-namelist", "--namelist-input",
                 "--stock-wrf-namelist-input", "--domain-spec", "--static-input",
                 "--static-cache", "--static-receipt", "--cpu-preprocess-bridge",
                 "--bridge", "--grib2-inventory", "--grib2-dump"}
_RAW_DECODER_FILES = {"--bridge", "--grib2-inventory", "--grib2-dump"}


def _flags(arguments):
    return {value.split("=", 1)[0] for value in arguments if value.startswith("--")}


@lru_cache(maxsize=1)
def _path_flags():
    # The ordinary parser defines which argument values are filesystem paths.
    # Other controls, source names and numeric values retain exact equality.
    from gpuwm.source_cli import _parser
    return frozenset(option for action in _parser()._actions if action.type is Path
                     for option in action.option_strings)


def _resolved_arguments(arguments):
    result, path_value = [], False
    flags = _path_flags()
    for value in arguments:
        if path_value:
            result.append(str(Path(value).resolve()))
            path_value = False
        elif value.split("=", 1)[0] in flags:
            if "=" in value:
                flag, path = value.split("=", 1)
                result.append(flag + "=" + str(Path(path).resolve()))
            else:
                result.append(value)
                path_value = True
        else:
            result.append(value)
    return result


def _resolved_configuration(records):
    return {flag: {**record, "path": str(Path(record["path"]).resolve())}
            for flag, record in records.items()}


def _resolved_description(record):
    """Compare live filesystem aliases without rewriting frozen authorities."""
    return {**record,
            **{key: str(Path(record[key]).resolve())
               for key in ("acquisition_root", "prepared_root", "physical_root")},
            **{key: _resolved_arguments(record[key])
               for key in ("native_arguments", "preparation_arguments")},
            "configuration_files": _resolved_configuration(record["configuration_files"])}


def _config_files(arguments, *, captured=None, native_decoder_sha256s=()):
    result = {}
    for flag in _CONFIG_FILES & _flags(arguments):
        values = []
        for index, value in enumerate(arguments):
            if value == flag:
                if index + 1 == len(arguments):
                    raise ValueError(f"native preparation argument {flag} has no path")
                values.append(arguments[index + 1])
            elif value.startswith(flag + "="):
                values.append(value.split("=", 1)[1])
        if len(values) != 1:
            raise ValueError(f"native preparation needs one configuration authority for {flag}")
        path = Path(values[0]).resolve()
        if not path.is_file():
            saved = (captured or {}).get(flag)
            if (flag not in _RAW_DECODER_FILES or saved is None
                    or not saved.get("path") or Path(saved["path"]).resolve() != path
                    or saved.get("sha256") not in native_decoder_sha256s):
                raise FileNotFoundError(path)
            # This exact executable's digest is in the checked physical
            # source head. Consumers initialize captured fields and do not
            # decode again. Config/static/native-initializer files stay live.
            result[flag] = dict(saved)
        else:
            result[flag] = {"path": str(path), "sha256": digest_file(path)}
    return result


def _source_decoder_authority(provider, identity):
    context = provider._source_context(identity).verify()
    return frozenset(value for key, value in context.physical_stream.head["field_contract"]["evidence"].items()
                     if key == "native_decoder" or key.startswith("native_decoder_"))


@dataclass(frozen=True)
class PostedSourcePreparation:
    """One ordinary source capture, from its real acquisition handoff."""
    trajectory: SourceTrajectory
    acquisition_root: Path
    prepared_root: Path
    physical_root: Path
    native_arguments: tuple[str, ...]
    preparation_arguments: tuple[str, ...]
    configuration_files: dict
    _consumer_decoder_sha256s: frozenset[str] = field(default_factory=frozenset, repr=False, compare=False)

    @classmethod
    def from_acquisition(cls, trajectory, *, acquisition_root, prepared_root,
                         physical_root, native_arguments, _captured_configuration=None,
                         _native_decoder_sha256s=()):
        if not isinstance(trajectory, SourceTrajectory):
            raise ValueError("native posted preparation requires a source trajectory")
        arguments = tuple(str(value) for value in native_arguments)
        if _flags(arguments) & _OWNED:
            raise ValueError("native source controls cannot replace the acquisition or physical provider")
        acquisition = Path(acquisition_root).resolve(strict=True)
        handoff = tuple(posted_preparation_arguments_from_directory(acquisition, trajectory=trajectory))
        if _flags(arguments) & _flags(handoff):
            raise ValueError("native preparation arguments repeat an acquisition-owned source option")
        return cls(trajectory, acquisition, Path(prepared_root).resolve(),
                   Path(physical_root).resolve(), arguments, handoff,
                   _config_files(handoff + arguments, captured=_captured_configuration,
                                 native_decoder_sha256s=_native_decoder_sha256s))

    def verify(self, *, _native_decoder_sha256s=None):
        if _native_decoder_sha256s is None:
            _native_decoder_sha256s = self._consumer_decoder_sha256s
        current = tuple(posted_preparation_arguments_from_directory(
            self.acquisition_root, trajectory=self.trajectory))
        if _resolved_arguments(current) != _resolved_arguments(self.preparation_arguments):
            raise ValueError("native posted source handoff changed after the recipe was bound")
        if _resolved_configuration(_config_files(current + self.native_arguments,
                captured=self.configuration_files, native_decoder_sha256s=_native_decoder_sha256s)
                ) != _resolved_configuration(self.configuration_files):
            raise ValueError("native posted source configuration or static authority changed")
        return self

    def source_arguments(self):
        """Arguments for the ordinary source producer, before any future seal."""
        self.verify(_native_decoder_sha256s=())
        return [*self.preparation_arguments, *self.native_arguments,
                "--output-root", str(self.prepared_root),
                "--physical-output-store", str(self.physical_root)]

    def describe(self, *, _native_decoder_sha256s=None):
        self.verify(_native_decoder_sha256s=_native_decoder_sha256s)
        return {"trajectory": {"source": self.trajectory.source,
                               "cycle": self.trajectory.cycle.isoformat(),
                               "member": self.trajectory.member},
                "acquisition_root": str(self.acquisition_root),
                "prepared_root": str(self.prepared_root), "physical_root": str(self.physical_root),
                "native_arguments": list(self.native_arguments),
                "preparation_arguments": list(self.preparation_arguments),
                "configuration_files": self.configuration_files}


class PostedPreparationFactory:
    """Bind ordinary source heads and produce real member CLI preparations.

    Source producers may still be waiting for future fields. Their first
    physical frame and ordinary prepared head must exist before publication.
    The original process scheduler owns how many preparations run at once.
    """
    SCHEMA = "gpuwm-ensemble-posted-native-preparation.v1"

    def __init__(self, provider, sources):
        self.provider, self.sources = provider, dict(sources)
        required = {value.identity for value in provider.recipe.acquisitions()}
        if set(self.sources) != required:
            raise ValueError("native source factory must retain every recipe acquisition")
        for identity, specification in self.sources.items():
            if specification.trajectory.identity != identity:
                raise ValueError("native source factory key differs from its trajectory")
            # Factory-owned specifications describe captured sources. Their
            # existing verify() protocol is also used by the runtime planner.
            # Producer argv always performs the stricter live-decoder check.
            specification = replace(specification,
                _consumer_decoder_sha256s=_source_decoder_authority(provider, identity))
            self.sources[identity] = specification
            specification.verify()
            if specification.prepared_root.resolve() != provider.prepared_roots[identity].resolve():
                raise ValueError("native source factory changed the provider's ordinary source root")
            if specification.physical_root.resolve() != provider.streams[identity].root.resolve():
                raise ValueError("native source factory changed the provider's captured physical root")
            from gpuwm.ingest.boundary_stream import read_head
            captured = read_head(specification.prepared_root)["basis"]["cache"]["identity"]
            namelist_digest = captured.get("namelist_sha256")
            if namelist_digest is not None:
                controls = {record["sha256"] for flag, record in specification.configuration_files.items()
                            if flag in {"--experiment-config", "--namelist-input"}}
                if namelist_digest not in controls:
                    raise ValueError("native source factory controls differ from the captured source initialization")

    @classmethod
    def publish(cls, recipe: SourceRecipe, sources, *, root, amplitude=1.,
                cpu_bridge=None, workers=1):
        sources = dict(sources)
        for specification in sources.values():
            specification.verify(_native_decoder_sha256s=())
        provider = PostedPhysicalProvider(recipe, {
            identity: PostedPhysicalStream(specification.physical_root)
            for identity, specification in sources.items()},
            amplitude=amplitude, cpu_bridge=cpu_bridge, workers=workers)
        provider.write_plan(root, prepared_roots={
            identity: value.prepared_root for identity, value in sources.items()})
        result = cls(provider, sources)
        from gpuwm.ensemble.posted_physical import _publish_json
        _publish_json(provider.root / "native-preparation.json", {
            "schema": cls.SCHEMA, "provider_plan_sha256": result.plan_sha256,
            "sources": {key: value.describe(_native_decoder_sha256s=()) for key, value in sorted(sources.items())}})
        return result

    @property
    def plan_sha256(self):
        from gpuwm.ensemble.physical_boundary import digest
        return digest(self.provider.plan)

    @classmethod
    def open(cls, root, *, cpu_bridge=None, workers=1, on_wait=None):
        provider = PostedPhysicalProvider.open(root, cpu_bridge=cpu_bridge,
                                               workers=workers, on_wait=on_wait)
        document = json.loads((Path(root) / "native-preparation.json").read_bytes())
        if document.get("schema") != cls.SCHEMA:
            raise ValueError("native posted preparation has an unsupported schema")
        from datetime import datetime
        sources = {}
        for identity, record in document["sources"].items():
            source = record["trajectory"]
            trajectory = SourceTrajectory(source["source"], datetime.fromisoformat(source["cycle"]), source["member"])
            decoder_authority = _source_decoder_authority(provider, identity)
            specification = PostedSourcePreparation.from_acquisition(trajectory,
                acquisition_root=record["acquisition_root"], prepared_root=record["prepared_root"],
                physical_root=record["physical_root"], native_arguments=record["native_arguments"],
                _captured_configuration=record["configuration_files"],
                _native_decoder_sha256s=decoder_authority)
            if _resolved_description(specification.describe(
                    _native_decoder_sha256s=decoder_authority)) != _resolved_description(record):
                raise ValueError("native source factory differs from its frozen preparation arguments")
            sources[identity] = specification
        result = cls(provider, sources)
        if result.plan_sha256 != document.get("provider_plan_sha256"):
            raise ValueError("native source factory differs from its original provider plan")
        return result

    def member_arguments(self, member_index, *, output_root):
        """Use the base route for recentering and the selected route otherwise."""
        members = {value.index: value for value in self.provider.recipe.members}
        if type(member_index) is not int or member_index not in members:
            raise ValueError("native member preparation requires an original recipe index")
        recipe = self.provider.recipe
        trajectory = recipe.base if recipe.kind == "recentered" else members[member_index].trajectory
        specification = self.sources[trajectory.identity].verify(
            _native_decoder_sha256s=_source_decoder_authority(self.provider, trajectory.identity))
        output = Path(output_root).resolve()
        protected = [self.provider.root]
        for source in self.sources.values():
            protected.extend((source.prepared_root, source.physical_root, source.acquisition_root))
        if any(output == path or path.is_relative_to(output) or output.is_relative_to(path)
               for path in protected):
            raise ValueError("member output must not overwrite or contain a bound source or provider tree")
        return [*specification.preparation_arguments, *specification.native_arguments,
                "--output-root", str(output), "--physical-input-provider", str(self.provider.root),
                "--physical-member-index", str(member_index)]

    def prepare_member(self, member_index, *, output_root, forecast=None, observer=None):
        """Run the ordinary subprocess and, optionally, its start-first lifecycle."""
        arguments = self.member_arguments(member_index, output_root=output_root)
        command = [sys.executable, "-m", "gpuwm.source_cli", *arguments]
        def prepare():
            subprocess.run(command, check=True)
            from gpuwm.ingest.boundary_stream import read_head, verify_seal
            head = read_head(output_root)
            verify_seal(output_root, head=head)
            physical = head["basis"].get("ensemble_physical")
            if (physical is None or physical["member_index"] != member_index
                    or physical["provider_plan"] != self.provider.plan):
                raise ValueError("native member result lost its exact source recipe authority")
            return {"prepared_root": str(Path(output_root).resolve()),
                    "head_sha256": head["head_sha256"], "member_index": member_index,
                    "provider_plan_sha256": self.plan_sha256}
        if forecast is None:
            return prepare()
        from gpuwm.ingest.boundary_stream import run_chained
        return run_chained(prepared_root=output_root, prepare=prepare,
                           forecast=forecast, observer=observer)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", type=Path, required=True)
    parser.add_argument("--member-index", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    factory = PostedPreparationFactory.open(args.provider)
    result = factory.prepare_member(args.member_index, output_root=args.output_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
