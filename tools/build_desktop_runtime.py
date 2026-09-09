"""Assemble a relocatable Windows runtime from verified CPython and wheel inputs.

No installed Python prefix, user site, registry lookup, or external package path
is retained. Wheel console scripts are intentionally not generated: the desktop
uses runtime/python.exe -m gpuwm.cli, and pip remains available through -m pip.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
import re
import sys
import zipfile


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_parts(name: str) -> tuple[str, ...]:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(p in ('.', '..') or ':' in p or '\\' in p for p in path.parts):
        raise ValueError(f'Unsafe archive member: {name!r}')
    return path.parts


def unpack_python(archive: Path, root: Path, expected: str) -> None:
    if digest(archive) != expected:
        raise ValueError('CPython archive does not match the locked official archive')
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            parts = safe_parts(item.filename)
            if item.is_dir():
                continue
            if len(parts) != 1:
                raise ValueError('Expected flat CPython embedded archive')
            (root / parts[0]).write_bytes(z.read(item))
    (root / 'python313._pth').write_text('python313.zip\n.\nLib/site-packages\nimport site\n', encoding='utf-8')
    if not (root / 'python.exe').is_file() or not (root / 'LICENSE.txt').is_file():
        raise ValueError('Incomplete CPython archive')


def unpack_wheel(wheel: Path, root: Path, expected: str | None = None) -> dict:
    actual = digest(wheel)
    if expected is not None and actual != expected:
        raise ValueError(f'Wheel hash mismatch: {wheel.name}')
    site = root / 'Lib/site-packages'
    site.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError(f'Duplicate wheel members: {wheel.name}')
        records = [n for n in names if n.endswith('.dist-info/RECORD')]
        if len(records) != 1:
            raise ValueError(f'Expected one wheel RECORD: {wheel.name}')
        record_name = records[0]
        recorded = {}
        for name, hash_value, size in csv.reader(io.StringIO(z.read(record_name).decode('utf-8'))):
            if name in recorded:
                raise ValueError(f'Duplicate RECORD entry: {name}')
            recorded[name] = (hash_value, size)
        files = {item.filename for item in z.infolist() if not item.is_dir()}
        if files != set(recorded):
            raise ValueError(f'Wheel RECORD member set mismatch: {wheel.name}')
        installed = []
        licenses = []
        for item in z.infolist():
            name = item.filename
            parts = safe_parts(name)
            if item.is_dir():
                continue
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f'Wheel symlink is not allowed: {name}')
            data = z.read(item)
            encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')
            hash_value, size = recorded[name]
            if name != record_name and (hash_value != 'sha256=' + encoded or size != str(len(data))):
                raise ValueError(f'Wheel RECORD hash/size mismatch: {name}')
            if name == record_name:
                continue
            target = site.joinpath(*parts)
            if parts[0].endswith('.data'):
                if len(parts) < 3:
                    raise ValueError(f'Invalid wheel data member: {name}')
                if parts[1] in ('purelib', 'platlib'):
                    target = site.joinpath(*parts[2:])
                elif parts[1] == 'data':
                    target = root.joinpath(*parts[2:])
                else:
                    raise ValueError(f'Unsupported wheel data scheme: {name}')
            if target.exists() and target.read_bytes() != data:
                raise ValueError(f'Conflicting installed member: {target.relative_to(root)}')
            if target.suffix == '.pth':
                raise ValueError(f'Wheel contains a path hook requiring explicit review: {name}')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            relative = target.relative_to(root).as_posix()
            # RECORD paths are relative to site-packages, even for shared data.
            record_relative = relative[len('Lib/site-packages/'):] if relative.startswith('Lib/site-packages/') else '../../' + relative
            installed.append((record_relative, 'sha256=' + encoded, str(len(data))))
            if any(word in name.lower() for word in ('license', 'copying', 'notice')):
                licenses.append(relative)
        stream = io.StringIO(newline='')
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerows(sorted(installed))
        writer.writerow((record_name, '', ''))
        (site / record_name).write_text(stream.getvalue(), encoding='utf-8', newline='')
        return {'filename': wheel.name, 'sha256': actual, 'bytes': wheel.stat().st_size,
                'installed_files': len(installed) + 1, 'license_files': sorted(licenses)}


def validate_engine_inputs(engine: Path, data: Path, bundle: Path, revision: str) -> list[dict]:
    # Reuse the engine's source-stamp and vendored-ABI contract. This import is
    # for assembly only; its source path is never written into the runtime.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gpuwm import bridge_assets, bridges
    metadata = []
    for path, name in ((engine, 'gpuwm'), (data, 'gpuwm-data')):
        with zipfile.ZipFile(path) as z:
            entries = [n for n in z.namelist() if n.endswith('.dist-info/METADATA')]
            if len(entries) != 1:
                raise ValueError(f'Expected one distribution METADATA: {path.name}')
            m = BytesParser().parsebytes(z.read(entries[0]))
            if m['Name'] != name:
                raise ValueError(f'Unexpected distribution in {path.name}')
            metadata.append(m)
    if metadata[0]['Version'] != metadata[1]['Version']:
        raise ValueError('Engine/data package versions differ')
    with zipfile.ZipFile(engine) as z:
        prefix = 'gpuwm/libexec/bridges/'
        manifest = json.loads(z.read(prefix + 'BUNDLE.json'))
        pins = json.loads(z.read('gpuwm/data/bridges/bridge-pins.json'))['platforms']['win-x86_64']
        expected = {a.name: a for a in bridge_assets.BUNDLED_ARTIFACTS}
        rows = manifest['artifacts']
        if manifest['platform'] != 'win-x86_64' or len(rows) != len(expected) or {r['artifact'] for r in rows} != set(expected):
            raise ValueError('Incomplete Windows engine native inventory')
        for row in rows:
            payload = z.read(prefix + row['filename'])
            if len(payload) != row['bytes'] or hashlib.sha256(payload).hexdigest() != row['sha256']:
                raise ValueError(f'Native payload does not match BUNDLE: {row["artifact"]}')
            artifact = expected[row['artifact']]
            if not artifact.vendored:
                bridge_assets.verify_source_revision(payload, expected=revision, label=row['artifact'])
            marker = bridges.BRIDGE_ABI_MARKERS.get(row['artifact'])
            if marker and marker not in payload:
                raise ValueError(f'Native ABI marker missing: {row["artifact"]}')
        pin = pins['bundle']
        if bundle.name != pin['filename'] or bundle.stat().st_size != pin['bytes'] or digest(bundle) != pin['sha256']:
            raise ValueError('Native/map bundle does not match the exact engine wheel pins')
        with zipfile.ZipFile(bundle) as native:
            assets = pins['assets']
            expected_names = {r['filename'] for r in rows} | {r['path'] for r in assets}
            if len(native.namelist()) != len(expected_names) or set(native.namelist()) != expected_names:
                raise ValueError('Native/map bundle member set differs from engine pins')
            for row in rows:
                if native.read(row['filename']) != z.read(prefix + row['filename']):
                    raise ValueError(f'Native/map bundle and wheel differ: {row["filename"]}')
            for row in assets:
                safe_parts(row['path'])
                payload = native.read(row['path'])
                if len(payload) != row['bytes'] or hashlib.sha256(payload).hexdigest() != row['sha256']:
                    raise ValueError(f'Basemap asset does not match engine pin: {row["path"]}')
        return assets


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--python-archive', type=Path, required=True)
    p.add_argument('--wheelhouse', type=Path, required=True)
    p.add_argument('--lock', type=Path, default=Path(__file__).parent / 'release/desktop-runtime-windows-cp313.lock.json')
    p.add_argument('--engine-wheel', type=Path)
    p.add_argument('--data-wheel', type=Path)
    p.add_argument('--bridge-bundle', type=Path)
    p.add_argument('--source-revision')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if bool(a.engine_wheel) != bool(a.data_wheel):
        p.error('Engine and data wheels must be supplied together')
    if bool(a.engine_wheel) != bool(a.bridge_bundle):
        p.error('The matching native/map bundle is required with the engine/data wheels')
    if a.engine_wheel and not re.fullmatch(r'[0-9a-f]{40}', a.source_revision or ''):
        p.error('A complete source revision is required with the engine/data wheels')
    if a.output.exists():
        p.error('Use a new output directory; existing artifacts are retained')
    lock = json.loads(a.lock.read_text(encoding='utf-8'))
    if lock['schema'] != 'arwen.desktop-runtime-inputs.v1':
        p.error('Unsupported runtime lock schema')
    # Validate every input before creating a partially populated runtime.
    inputs = [(a.wheelhouse / row['filename'], row['sha256']) for row in lock['dependencies']]
    inputs.append((a.python_archive, lock['python']['sha256']))
    for path, expected in inputs:
        if not path.is_file() or digest(path) != expected:
            p.error(f'Locked runtime input missing or changed: {path.name}')
    assets = validate_engine_inputs(a.engine_wheel, a.data_wheel, a.bridge_bundle, a.source_revision) if a.engine_wheel else []
    a.output.mkdir(parents=True)
    unpack_python(a.python_archive, a.output, lock['python']['sha256'])
    rows = [unpack_wheel(a.wheelhouse / row['filename'], a.output, row['sha256']) for row in lock['dependencies']]
    if a.engine_wheel:
        rows += [unpack_wheel(a.engine_wheel, a.output), unpack_wheel(a.data_wheel, a.output)]
        with zipfile.ZipFile(a.bridge_bundle) as native:
            for row in assets:
                target = a.output / 'Lib/site-packages/gpuwm/libexec/bridges' / row['path']
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(native.read(row['path']))
    manifest = {'schema': 'arwen.desktop-runtime.v1', 'status': 'ASSEMBLED_REQUIRES_RELOCATION_CHECK',
                'python': lock['python'], 'input_lock_sha256': digest(a.lock),
                'source_revision': a.source_revision, 'contains_engine': bool(a.engine_wheel),
                'basemap_assets': assets,
                'entrypoint': 'python.exe -m gpuwm.cli', 'wheels': rows,
                'cuda': 'Bundled CUDA 13 libraries and headers; NVIDIA driver remains a host prerequisite.',
                'path_policy': 'Embedded interpreter and relative local site-packages only; no external path hooks or generated absolute console launchers.'}
    (a.output / 'ARWEN-RUNTIME.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': manifest['status'], 'output': str(a.output.resolve()), 'wheels': len(rows)}))


if __name__ == '__main__':
    main()
