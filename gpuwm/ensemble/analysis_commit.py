"""Durable analysis decisions, independent of the outer cycle manifest.

An intent contains the exact roster, file digests and prospective receipt.
It is published before any member rename and retained after completion. A
retry completes that decision; it does not call the analysis method again.
The caller must hold the run's single-writer lock for the whole operation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Callable

from gpuwm.filesystem_paths import publish_new

INTENT_NAME = 'analysis-publication.json'
COMMIT_NAME = 'analysis-commit.json'
SCHEMA = 'gpuwm-da.analysis-commit.v1'


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path) -> str:
    from gpuwm.output_identity import file_record
    return file_record(path)['sha256']


def sync_directory(path) -> None:
    # Python exposes directory fsync on POSIX. Windows durability is limited
    # to the platform's link/rename guarantees and the flushed file itself.
    if os.name == 'posix':
        fd = os.open(Path(path), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def write_record(path, value) -> dict:
    """Create a sealed record without overwriting an existing identity."""
    path = Path(path)
    body = {key: val for key, val in value.items() if key != 'self_sha256'}
    body['self_sha256'] = digest(body)
    payload = canonical(body) + b'\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f'{path}: a record cannot be a symbolic link')
    fd, name = tempfile.mkstemp(prefix='.record-', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            publish_new(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError(f'{path}: refusing to replace a published identity') from None
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return body


def read_record(path) -> dict:
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f'{path}: a record cannot be a symbolic link')
    document = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(document, dict):
        raise ValueError(f'{path}: expected a record object')
    if document.get('schema') == 'gpuwm-da-analysis-publication.v1':
        raise ValueError(f'{path}: this legacy publication lacks byte identities and its method receipt. Preserve the analyses and restore their original complete receipt before recovery')
    unsigned = {k: v for k, v in document.items() if k != 'self_sha256'}
    if digest(unsigned) != document.get('self_sha256'):
        raise ValueError(f'{path}: record digest does not match its contents')
    return document


def _path(root: Path, relative: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute() or '..' in raw.parts:
        raise ValueError(f'{relative!r}: analysis path must remain inside its leg')
    candidate = root / raw
    current = root
    for part in raw.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f'{current}: analysis paths cannot follow symbolic links')
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'{candidate}: analysis path escaped its leg')
    return candidate


def _validate(root: Path, document: dict) -> list[tuple[int, Path, Path, str]]:
    if document.get('schema') != SCHEMA:
        raise ValueError(f'{root}: unsupported analysis commit schema')
    members = document.get('members')
    count = document.get('member_count')
    if type(count) is not int or count < 1 or not isinstance(members, list):
        raise ValueError(f'{root}: invalid analysis commit roster')
    indices = [row.get('member') for row in members if isinstance(row, dict)]
    if len(indices) != len(members) or any(type(i) is not int for i in indices) or indices != list(range(count)):
        raise ValueError(f'{root}: analysis commit does not contain the exact member roster')
    from gpuwm.ensemble.manifest import member_directory_name
    observed = sorted(int(path.name[7:]) for path in root.iterdir()
                      if path.is_dir() and path.name.startswith('member_')
                      and path.name[7:].isascii() and path.name[7:].isdigit()
                      and member_directory_name(int(path.name[7:])) == path.name)
    if observed != list(range(count)):
        raise ValueError(f'{root}: the declared publication omits or adds member directories; restore the complete roster before recovery')
    analysis_name = document.get('analysis_name')
    if not isinstance(analysis_name, str) or not analysis_name or Path(analysis_name).name != analysis_name or analysis_name in ('.', '..'):
        raise ValueError(f'{root}: invalid analysis filename')
    suffix = document.get('staged_suffix')
    if not isinstance(suffix, str) or not suffix or any(c in suffix for c in '/\\'):
        raise ValueError(f'{root}: invalid staged suffix')
    result = []
    for row in members:
        member = row['member']
        prefix = f'{member_directory_name(member)}/{analysis_name}'
        if row.get('analysis') != prefix or row.get('staged') != prefix + suffix:
            raise ValueError(f'{root}: member {member} has a noncanonical analysis path')
        expected = row.get('sha256', '')
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
            raise ValueError(f'{root}: invalid member digest')
        result.append((member, _path(root, row['staged']), _path(root, row['analysis']), expected))
    if not isinstance(document.get('receipt'), dict) or not isinstance(document.get('context'), dict):
        raise ValueError(f'{root}: analysis intent is missing its receipt or binding')
    context = document['context']
    if context.get('leg_root') != str(root.resolve()):
        raise ValueError(f'{root}: analysis decision belongs to another leg; restore the matching record')
    if context.get('members') != list(range(count)):
        raise ValueError(f'{root}: analysis decision context has a different member roster')
    if not isinstance(context.get('assets'), list):
        raise ValueError(f'{root}: analysis decision is missing its input identities')
    if document['receipt'].get('member_count') != count or document['receipt'].get('cycle') != context.get('cycle'):
        raise ValueError(f'{root}: analysis receipt disagrees with its publication roster or clock')
    _validate_decision_agreement(root, document, count)
    return result


def _agree(root, label, observed, expected):
    if canonical(observed) != canonical(expected):
        raise ValueError(f'{root}: analysis {label} contradicts its bound decision. '
                         'Preserve the member files and restore the original matching decision before recovery.')


def _validate_decision_agreement(root, document, count):
    """One consistency check before either door can publish a member."""
    context, receipt = document['context'], document['receipt']
    _agree(root, 'receipt status', receipt.get('status'), 'APPLIED')
    for reported, bound in (('positivity_policy', 'positivity'),
                            ('moment_policy', 'moment_policy'),
                            ('moment_repair', 'moment_repair')):
        if bound not in context or reported not in receipt:
            raise ValueError(f'{root}: the analysis decision is missing {bound}. '
                             'Restore its original policy receipt before recovery.')
        _agree(root, f'receipt {reported}', receipt[reported], context[bound])
    method, reported_method = context.get('method'), receipt.get('method')
    if not isinstance(method, dict) or not isinstance(reported_method, dict):
        raise ValueError(f'{root}: the analysis decision is missing its method receipt. '
                         'Restore the original method record before recovery.')
    _agree(root, 'receipt method callable', reported_method.get('callable'), method.get('callable'))
    declaration = reported_method.get('declared_by')
    if declaration is None:
        _agree(root, 'receipt undeclared method provenance', reported_method.get('provenance'), None)
    elif declaration not in ('caller', 'assimilation-callable') or not isinstance(reported_method.get('provenance'), dict):
        raise ValueError(f'{root}: the receipt method declaration contradicts its provenance. '
                         'Restore the original method receipt before recovery.')
    if declaration == 'caller':
        _agree(root, 'declared method receipt', reported_method.get('provenance'),
               context.get('declared_method'))
    receipts = receipt.get('receipts')
    if not isinstance(receipts, list) or any(not isinstance(row, dict) for row in receipts):
        raise ValueError(f'{root}: the analysis decision is missing its member receipts. '
                         'Restore the original complete receipt before recovery.')
    _agree(root, 'receipt member roster', [row.get('member') for row in receipts], list(range(count)))

    binding = context.get('run_binding')
    if binding is None:
        # A direct caller can have no enclosing cycle. The filesystem check
        # in verify_context_assets must also confirm that no authority exists.
        return
    if not isinstance(binding, dict):
        raise ValueError(f'{root}: the analysis run binding is not a record. '
                         'Restore the original cycle binding before recovery.')
    if 'n_members' in binding:
        _agree(root, 'run member count', count, binding['n_members'])
    if 'positivity' in binding:
        _agree(root, 'run positivity policy', context['positivity'], binding['positivity'])
    analysis = binding.get('analysis')
    if analysis is None:
        return
    if not isinstance(analysis, dict):
        raise ValueError(f'{root}: the analysis method binding is not a record. '
                         'Restore the original cycle binding before recovery.')
    if 'enabled' in analysis:
        _agree(root, 'run analysis mode', analysis['enabled'], True)
    for key in ('method', 'declared_method', 'moment_policy', 'moment_repair', 'mp_physics'):
        if key in analysis:
            _agree(root, f'run {key}', context.get(key), analysis[key])


def verify_context_assets(context):
    from gpuwm.output_identity import file_record
    for asset in context['assets']:
        if not isinstance(asset, dict) or not isinstance(asset.get('path'), str):
            raise ValueError('The analysis decision contains an invalid input identity; restore its original receipt')
        actual = file_record(asset['path'])
        if actual != asset:
            raise ValueError(f"{asset['path']}: analysis input bytes changed; restore the original inputs before recovery")
    from gpuwm.ensemble.manifest import CYCLE_MANIFEST_NAME, CYCLE_MANIFEST_SCHEMA, read_manifest
    path = Path(context['leg_root']).parent / CYCLE_MANIFEST_NAME
    if context.get('run_binding') is None:
        if path.exists() or path.is_symlink():
            raise ValueError(f'{path}: the analysis decision omits its existing enclosing run binding. '
                             'Restore the original matching decision before completing its members.')
        return
    manifest = read_manifest(path, schema=CYCLE_MANIFEST_SCHEMA)
    recorded = manifest.get('cycle_binding')
    expected = dict(context['run_binding'])
    # Older complete manifests did not record method identities. Their
    # existing binding still has to agree with every fact it did record.
    if isinstance(recorded, dict) and 'analysis' not in recorded:
        expected.pop('analysis', None)
    if recorded != expected:
        raise ValueError(f'{path}: the analysis belongs to another run binding; restore the matching cycle manifest')


def begin(root, *, context: dict, receipt: dict, pairs, staged_suffix: str, analysis_name: str) -> dict:
    """Flush staged bytes and bind the decision before the first rename."""
    root = Path(root).resolve()
    if (root / INTENT_NAME).exists() or (root / COMMIT_NAME).exists():
        raise ValueError(f'{root}: an analysis decision already exists; recover it')
    rows = []
    for member, staged, analysis in sorted(pairs):
        staged, analysis = Path(staged), Path(analysis)
        if analysis.exists():
            raise ValueError(f'{analysis}: refusing to replace an earlier analysis')
        rows.append(dict(member=member, staged=staged.resolve().relative_to(root).as_posix(),
                         analysis=analysis.resolve().relative_to(root).as_posix(), sha256=file_sha(staged)))
    document = dict(schema=SCHEMA, member_count=len(rows), members=rows, staged_suffix=staged_suffix, analysis_name=analysis_name,
                    context=context, receipt=receipt)
    checked = _validate(root, document)
    verify_context_assets(context)
    from gpuwm.supervisor import fsync_file
    for _, staged, _, _ in checked:
        fsync_file(staged)
        sync_directory(staged.parent)
    return write_record(root / INTENT_NAME, document)


def recover(root, *, publish: Callable, context: dict | None = None) -> dict | None:
    """Verify all members first, complete the decision, and return its receipt."""
    root = Path(root).resolve()
    intent, commit = root / INTENT_NAME, root / COMMIT_NAME
    if not intent.exists():
        if commit.exists():
            raise ValueError(f'{root}: commit exists without its analysis intent')
        return None
    document = read_record(intent)
    members = _validate(root, document)
    if context is not None and document['context'] != context:
        raise ValueError(f'{root}: analysis intent belongs to different inputs or policies')
    verify_context_assets(document['context'])
    committed = read_record(commit) if commit.exists() else None
    expected_commit = dict(schema=SCHEMA, intent_sha256=document['self_sha256'],
                           receipt=document['receipt'])
    if committed is not None and {k: v for k, v in committed.items() if k != 'self_sha256'} != expected_commit:
        raise ValueError(f'{root}: commit does not identify its analysis intent')
    # No rename is attempted until every surviving copy has been verified.
    for member, staged, analysis, expected in members:
        candidates = [p for p in (staged, analysis) if p.is_file()]
        if not candidates or (committed is not None and not analysis.is_file()):
            raise ValueError(f'{root}: member {member} has lost its analysis bytes. Restore its original staged or published file before completing the roster')
        for candidate in candidates:
            if file_sha(candidate) != expected:
                raise ValueError(f'{candidate}: analysis digest differs from the committed decision')
    if committed is None:
        for _, staged, analysis, _ in members:
            if not analysis.is_file():
                publish(staged, analysis)
            elif staged.is_file():
                staged.unlink()
            sync_directory(analysis.parent)
        # The owner remains responsible for the actual member publication.
        for _, _, analysis, expected in members:
            if file_sha(analysis) != expected:
                raise ValueError(f'{analysis}: publisher did not preserve staged bytes')
        write_record(commit, expected_commit)
    return document['receipt']
