#!/usr/bin/env python3
"""Append/refresh the coupled hydro output library's dependency notices in both bundles.

librw_nextgen_output (tools/nextgen_output) joined the bridge roster at 2.9.4 as an on-demand
artifact, so its locked dependency closure (parquet, thrift, snap, the project's netcdf-writer and
netcdf-reader, ...) now reaches users in binary form inside the bridge bundle. The binary-form notice
inventories the rustwx/grib1_bridge/rw_wps vendor walk, the TUI and the sounding window; this section
covers the output library's closure the same way tools/update_soundings_license_notice.py covers
rw-soundings. Breakage it prevents: the 2.9.4 bundle shipped librw_nextgen_output with no licence
text for parquet, thrift, crc32fast or the other crates it links.

Output depends only on tools/nextgen_output/Cargo.lock and the notice files on disk (its crates.io
sources resolve to tools/rustwx/vendor/crates-io through tools/nextgen_output/.cargo/config.toml).
Notice files are ordered by case-sensitive path parts, named with forward slashes, read and written
as UTF-8 bytes. Running it twice changes nothing; ``--check`` exits 1 when either copy is stale.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
MARKER = '\nCOUPLED HYDRO OUTPUT LIBRARY (rw_nextgen_output) — LOCKED DEPENDENCY NOTICES\n'
END = '\nEND COUPLED HYDRO OUTPUT LIBRARY DEPENDENCY NOTICES\n'
COPIES = ('licenses/THIRD-PARTY-LICENSES-bridge-binaries.txt',
          'tools/rustwx/assets/basemap/THIRD-PARTY-LICENSES.txt')
WORKSPACE = 'tools/nextgen_output'
VENDOR = 'tools/rustwx/vendor/crates-io'
ROOT_PACKAGE = 'rw-nextgen-output'
PREFIXES = ('LICENSE', 'LICENCE', 'COPYING', 'COPYRIGHT', 'NOTICE')
#: Build output and local caches never hold a notice of the crate itself.
SKIP_PARTS = {'target', '.cargo', 'tests'}


def in_tree(path, root):
    """``path`` named through the tree (a linked tools/<workspace> maps back through its link)."""
    if path.is_relative_to(root):
        return path
    for entry in sorted((root / 'tools').iterdir()):
        if entry.is_symlink() and path.is_relative_to(entry.resolve()):
            return entry / path.relative_to(entry.resolve())
    if path.is_relative_to(root.resolve()):
        return root / path.relative_to(root.resolve())
    raise RuntimeError(f'{path} is outside the tree {root}')


def _notice_files(directory):
    found = []
    for path in directory.rglob('*'):
        parts = path.relative_to(directory).parts
        if SKIP_PARTS.intersection(parts[:-1]) or not path.is_file():
            continue
        if path.name.upper().startswith(PREFIXES) or path.name.upper().endswith(('-NOTICE.TXT', '-NOTICE')):
            found.append(path)
    return sorted(found, key=lambda p: p.relative_to(directory).parts)


def section_text(root=ROOT):
    """The generated section (from MARKER's heading line, without END)."""
    workspace = root / WORKSPACE
    metadata = json.loads(subprocess.check_output(
        ['cargo', 'metadata', '--offline', '--locked', '--format-version', '1'], cwd=workspace))
    packages = {p['id']: p for p in metadata['packages']}
    nodes = {p['id']: p for p in metadata['resolve']['nodes']}
    pending = [p['id'] for p in packages.values() if p['name'] == ROOT_PACKAGE]
    if not pending:
        raise RuntimeError(f'{ROOT_PACKAGE} is not in the {WORKSPACE} lock')
    selected = set()
    while pending:
        key = pending.pop()
        if key in selected:
            continue
        selected.add(key)
        pending.extend(nodes[key]['dependencies'])
    vendor = {}
    for directory in sorted((root / VENDOR).iterdir()):
        manifest = directory / 'Cargo.toml'
        if manifest.is_file():
            package = tomllib.loads(manifest.read_bytes().decode('utf-8'))['package']
            vendor[package['name'], package['version']] = directory
    texts, inventory = {}, []
    for key in sorted(selected, key=lambda k: (packages[k]['name'], packages[k]['version'],
                                               packages[k].get('source') or '')):
        package = packages[key]
        source = package.get('source') or ''
        if source.startswith('registry+'):
            directory = vendor[package['name'], package['version']]
        else:
            directory = in_tree(Path(package['manifest_path']).parent, root)
        files = _notice_files(directory)
        grant = package.get('license') or ''
        if not files:
            if 'Apache-2.0' in grant:
                files = [root / 'LICENSE']  # The permissive Apache alternative.
            elif 'MIT' in grant and not source:
                files = [root / 'tools/rustwx/LICENSE']  # A rustwx path crate (MIT workspace).
            else:
                raise RuntimeError(f"missing notices: {package['name']} ({grant or 'no licence declared'})")
            inventory.append(f"  {package['name']}: package declares {grant}; no separate notice file; "
                             f"authors: {', '.join(package.get('authors', []))}; grant text below.")
        refs = []
        for path in files:
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            entry = texts.setdefault(digest, {'id': f'HYDRO-OUTPUT-{len(texts)+1:03}',
                                              'text': raw.decode('utf-8-sig'), 'paths': []})
            entry['paths'].append(path.relative_to(root).as_posix())
            refs.append(entry['id'])
        inventory.append(f"{package['name']} {package['version']} | {grant or 'path crate'} | {','.join(refs)}")
    lines = [MARKER.strip(), '', 'Generated by tools/update_nextgen_output_license_notice.py.',
             'The locked dependency closure of librw_nextgen_output / rw_nextgen_output.dll, all platforms.',
             'Missing package notice files use the declared SPDX grant; declared authors are listed.',
             '', *inventory]
    for entry in texts.values():
        lines.extend(['', '=' * 78, entry['id'], *entry['paths'], '',
                      '\n'.join(line.rstrip() for line in entry['text'].splitlines()).rstrip()])
    return '\n'.join(lines) + '\n', len(selected), len(texts)


def compose(old, section):
    """``old`` with its hydro output section replaced by ``section`` (appended if absent)."""
    before, marker, rest = old.partition(MARKER)
    after = ''
    if marker:
        _, end, after = rest.partition(END)
        if not end:
            raise RuntimeError('hydro output notices are missing their end marker')
    return before.rstrip() + '\n' + MARKER + section.split('\n', 1)[1] + END + after


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--check', action='store_true',
                        help='write nothing; exit 1 when either copy is stale')
    args = parser.parse_args(argv)
    section, dependencies, notices = section_text()
    changed = []
    for name in COPIES:
        path = ROOT / name
        old = path.read_bytes().decode('utf-8')
        new = compose(old, section)
        if new != old:
            changed.append(name)
            if not args.check:
                path.write_bytes(new.encode('utf-8'))
    if args.check:
        if changed:
            print('stale hydro output notices (run python tools/update_nextgen_output_license_notice.py): '
                  + ', '.join(changed), file=sys.stderr)
            return 1
        print(f'{dependencies} hydro output dependencies; {notices} notice texts; both copies current')
        return 0
    print(f'{dependencies} hydro output dependencies; {notices} notice texts in both binary notices'
          f" ({'updated: ' + ', '.join(changed) if changed else 'unchanged'})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
