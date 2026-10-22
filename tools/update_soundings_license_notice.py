#!/usr/bin/env python3
"""Append/refresh the public sounding binary's dependency notices in both bundles.

The output depends only on the locked dependency closure of ``rw-soundings`` and
the notice files it ships, never on the platform: notice files are ordered by
their case-sensitive path parts and named with forward slashes, and every file
is read and written as UTF-8 bytes. Running it twice changes nothing
(``--check`` exits 1 when either copy is stale; tests/test_licence_notices_ship.py
asserts it). Origin: same-name same-version crates (a path crate and its git twin)
were ordered by Python's per-process set hashing; on Windows the tool sorted notice files case-insensitively,
wrote backslash paths and CRLF line ends, and crashed on the cp1252 codec; and
the committed section lacked cudarc, which rustwx-render gained in 529053024f
without a refresh, so one run rewrote ~3,466 lines of both notices.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
MARKER = '\nPUBLIC SOUNDING WINDOW — LOCKED DEPENDENCY NOTICES\n'
END = '\nEND PUBLIC SOUNDING WINDOW DEPENDENCY NOTICES\n'
COPIES = ('licenses/THIRD-PARTY-LICENSES-bridge-binaries.txt',
          'tools/rustwx/assets/basemap/THIRD-PARTY-LICENSES.txt')


def in_tree(path, root=ROOT):
    """``path`` named through the tree, never through where a linked workspace's bytes live.

    cargo metadata reports canonical manifest paths. When a tools/<workspace> folder is a symlink
    (the R7b hosted-runner emulation links every native workspace to a key-stable snapshot outside
    the clone), a path crate such as ecape-rs came back outside ``root`` and ``relative_to`` raised
    (2.9.4 rolling pass 7c12bbf2, route B: "vendor/ecape-rs/LICENSE is not in the subpath of
    /work/src"). Mapping it back through the link names the same file the plain tree names, so the
    section stays byte-identical whatever the checkout layout.
    """
    if path.is_relative_to(root):
        return path
    for entry in sorted((root / 'tools').iterdir()):
        if entry.is_symlink() and path.is_relative_to(entry.resolve()):
            return entry / path.relative_to(entry.resolve())
    if path.is_relative_to(root.resolve()):
        return root / path.relative_to(root.resolve())
    raise RuntimeError(f'{path} is outside the tree {root}')


def section_text(root=ROOT):
    """The generated section (from MARKER's heading line, without END)."""
    workspace = root / 'tools/rustwx'
    metadata = json.loads(subprocess.check_output(
        ['cargo', 'metadata', '--offline', '--locked', '--format-version', '1'], cwd=workspace))
    packages = {p['id']: p for p in metadata['packages']}
    nodes = {p['id']: p for p in metadata['resolve']['nodes']}
    pending = [p['id'] for p in packages.values() if p['name'] == 'rw-soundings']
    selected = set()
    while pending:
        key = pending.pop()
        if key in selected:
            continue
        selected.add(key)
        pending.extend(nodes[key]['dependencies'])
    vendor = {}
    for directory in sorted((workspace / 'vendor/crates-io').iterdir()):  # filesystem order differs by machine
        manifest = directory / 'Cargo.toml'
        if manifest.is_file():
            package = tomllib.loads(manifest.read_bytes().decode('utf-8'))['package']
            vendor[package['name'], package['version']] = directory
    texts, inventory = {}, []
    # A path crate and its git twin share name and version (wx-field 0.3.6); without the
    # source in the key their order, and so every later SOUNDING id, followed set hashing.
    for key in sorted(selected, key=lambda k: (packages[k]['name'], packages[k]['version'],
                                               packages[k].get('source') or '')):
        package = packages[key]
        if (package.get('source') or '').startswith('registry+'):
            directory = vendor[package['name'], package['version']]
        else:
            directory = in_tree(Path(package['manifest_path']).parent, root)
        # Case-sensitive path parts: a Windows path sorts case-insensitively.
        files = sorted((p for p in directory.rglob('*') if p.is_file() and
                        p.name.upper().startswith(('LICENSE', 'LICENCE', 'COPYING', 'COPYRIGHT', 'NOTICE', 'OFL.'))),
                       key=lambda p: p.relative_to(directory).parts)
        if not files:
            grant = package.get('license') or 'MIT'
            if 'Apache-2.0' in grant:
                files = [root / 'LICENSE']  # Select the permissive Apache alternative.
            elif grant == 'CC0-1.0':
                files = [root / 'licenses/soundings/LICENSE-CC0-1.0.txt']
            elif grant == 'BSL-1.0':
                files = [root / 'licenses/soundings/LICENSE-clipboard-win-BSL-1.0.txt']
            elif grant == 'MIT':
                if 'objc2' in (package.get('repository') or ''):
                    files = [root / 'licenses/soundings/LICENSE-objc2-MIT.txt']
                else:
                    files = [workspace / 'LICENSE']
            else:
                raise RuntimeError(f"missing notices: {package['name']} ({grant})")
            inventory.append(f"  {package['name']}: package declares {grant}; no separate notice file; "
                             f"authors: {', '.join(package.get('authors', []))}; grant text below.")
        refs = []
        for path in files:
            raw = path.read_bytes()
            if path == workspace / 'LICENSE' and package.get('source'):
                authors = ', '.join(package.get('authors', [])) or package['name'] + ' contributors'
                raw = raw.replace(b'Copyright (c) 2026 rustwx contributors',
                                  ('Declared package authors: ' + authors).encode())
            digest = hashlib.sha256(raw).hexdigest()
            entry = texts.setdefault(digest, {'id': f'SOUNDING-{len(texts)+1:03}',
                                              'text': raw.decode('utf-8-sig'), 'paths': []})
            entry['paths'].append(path.relative_to(root).as_posix())
            refs.append(entry['id'])
        inventory.append(f"{package['name']} {package['version']} | {package.get('license')} | {','.join(refs)}")
    lines = [MARKER.strip(), '', 'Generated by tools/update_soundings_license_notice.py.',
             'Includes the renderer dependency closure, build dependencies and all platform variants.',
             'Missing package notice files use the declared SPDX grant; declared authors are listed.',
             'The engine-owned adapter uses Apache-2.0; external grants and notices follow.', '', *inventory]
    for entry in texts.values():
        lines.extend(['', '='*78, entry['id'], *entry['paths'], '', '\n'.join(line.rstrip() for line in entry['text'].splitlines()).rstrip()])
    return '\n'.join(lines) + '\n', len(selected), len(texts)


def compose(old, section):
    """``old`` with its sounding section replaced by ``section`` (appended if absent)."""
    before, marker, rest = old.partition(MARKER)
    after = ''
    if marker:
        _, end, after = rest.partition(END)
        if not end:
            raise RuntimeError('sounding notices are missing their end marker')
    return before.rstrip() + '\n' + MARKER + section.split('\n', 1)[1] + END + after


def stale(root=ROOT):
    """The copies (relative names) whose sounding section differs from a fresh run."""
    section, _, _ = section_text(root)
    return [name for name in COPIES
            if compose((root / name).read_bytes().decode('utf-8'), section)
            != (root / name).read_bytes().decode('utf-8')]


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
            print('stale sounding notices (run python tools/update_soundings_license_notice.py): '
                  + ', '.join(changed), file=sys.stderr)
            return 1
        print(f'{dependencies} sounding dependencies; {notices} notice texts; both copies current')
        return 0
    print(f'{dependencies} sounding dependencies; {notices} notice texts in both binary notices'
          f" ({'updated: ' + ', '.join(changed) if changed else 'unchanged'})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
