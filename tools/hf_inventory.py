#!/usr/bin/env python3
"""Read-only, anonymous HF Earth-2 inventory. Requires only requests + stdlib.

Repo discovery, loader requirements and licence policy are table data. No model is
imported, no weights are downloaded, and no HF SDK or credential store is used.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import datetime as dt
import difflib
import fnmatch
import hashlib
import json
from pathlib import Path
import re
import threading
import time
from urllib.parse import quote, unquote, urljoin, urlsplit

import requests

HOST = 'https://huggingface.co'
LOCAL = threading.local()
RECEIPTS: list[dict] = []
LOCK = threading.Lock()
WEIGHT_SUFFIXES = ('.onnx', '.ckpt', '.mdlus', '.safetensors', '.pt', '.pth',
                   '.checkpoint', '.bin', '.tar')


def session():
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
        # Disables implicit .netrc authentication and environmental credentials.
        LOCAL.session.trust_env = False
        LOCAL.session.headers['User-Agent'] = 'gpuwm-hf-inventory/1 (anonymous metadata only)'
    return LOCAL.session


def request(method, url, **kwargs):
    """At most one connection per worker; never log signed redirect locations."""
    if urlsplit(url).netloc != 'huggingface.co':
        raise ValueError('Metadata requests must stay on huggingface.co')
    for attempt in range(6):
        start = time.monotonic()
        try:
            response = session().request(method, url, timeout=(15, 90),
                                         allow_redirects=False, **kwargs)
            status = response.status_code
            error = None
        except requests.RequestException as exc:
            response, status, error = None, None, type(exc).__name__
        with LOCK:
            RECEIPTS.append({'method': method, 'url': response.url if response is not None else url,
                             'status': status, 'attempt': attempt + 1,
                             'seconds': round(time.monotonic() - start, 3), 'error': error})
        if status not in (None, 429, 500, 502, 503, 504) or attempt == 5:
            if response is None:
                raise RuntimeError(f'{method} {url}: {error}')
            return response
        retry = response.headers.get('Retry-After', '') if response is not None else ''
        delay = min(120, max(2 ** attempt, float(retry) if retry.isdigit() else 0))
        if response is not None:
            response.close()
        time.sleep(delay)
    raise AssertionError('retry loop exhausted')


def json_get(url, **kwargs):
    with request('GET', url, **kwargs) as r:
        try:
            body = r.json()
        except ValueError:
            body = {'error': f'Non-JSON HTTP {r.status_code}'}
        return r.status_code, body, r.links.get('next', {}).get('url')


def api_path(typ, repo, rev=None):
    path = f'{HOST}/api/{"datasets" if typ == "dataset" else "models"}/{repo}'
    if rev is not None:
        path += '/revision/' + quote(unquote(rev), safe='')
    return path


def resolve_url(typ, repo, rev, path):
    prefix = 'datasets/' if typ == 'dataset' else ''
    return f'{HOST}/{prefix}{repo}/resolve/{quote(unquote(rev), safe="")}/{quote(path, safe="/")}'


def search(typ, term, author):
    url = f'{HOST}/api/{"datasets" if typ == "dataset" else "models"}'
    params = {'author': author, 'search': term, 'limit': 1000}
    ids, pages, errors = [], 0, []
    while url:
        status, body, nxt = json_get(url, params=params)
        pages += 1
        if status != 200 or not isinstance(body, list):
            errors.append({'url': url, 'status': status, 'error': body})
            break
        ids.extend(x['id'] for x in body)
        url, params = nxt, None
    return {'type': typ, 'term': term, 'author': author, 'pages': pages,
            'repositories': sorted(set(ids)), 'errors': errors}


def normalize_files(siblings):
    return sorted([{'path': x.get('rfilename', x.get('path')),
                    'size': x.get('size'),
                    'sha256': (x.get('lfs') or {}).get('sha256', (x.get('lfs') or {}).get('oid')),
                    'git_blob_id': x.get('blobId', x.get('oid'))}
                   for x in siblings if x.get('type', 'file') == 'file'], key=lambda x: x['path'])


def metadata(typ, repo, requested_revision=None):
    url = api_path(typ, repo, requested_revision)
    status, body, _ = json_get(url, params={'blobs': 'true'})
    out = {'requested_revision': requested_revision or 'main', 'api_url': url,
           'api_status': status, 'revision': body.get('sha') if isinstance(body, dict) else None}
    if status != 200 or not out['revision']:
        out.update(error=body.get('error', 'No commit SHA returned') if isinstance(body, dict) else body,
                   files=[], total_size_bytes=None, size_complete=False, gated=None, card_data={})
        return out
    files = normalize_files(body.get('siblings', []))
    if not files or any(x['size'] is None for x in files):
        tree, params = [], {'recursive': 'true', 'limit': 1000}
        tree_url = (f'{HOST}/api/{"datasets" if typ == "dataset" else "models"}/{repo}'
                    f'/tree/{out["revision"]}')
        while tree_url:
            code, page, nxt = json_get(tree_url, params=params)
            if code != 200 or not isinstance(page, list):
                out['tree_error'] = {'status': code, 'error': page}
                break
            tree.extend(page)
            tree_url, params = nxt, None
        if tree:
            files = normalize_files(tree)
    out.update(files=files, total_size_bytes=sum(x['size'] or 0 for x in files),
               size_complete=all(x['size'] is not None for x in files) and 'tree_error' not in out,
               gated=body.get('gated'), card_data=body.get('cardData') or {},
               licence_files=[x['path'] for x in files
                              if re.search(r'(^|/)(licen[cs]e|copying|notice)([._-]|$)', x['path'], re.I)])
    return out


def small_text(typ, repo, rev, file, max_bytes=524288):
    """Only known small card/config/licence files, never model or data payloads."""
    if file['size'] is None or file['size'] > max_bytes:
        return None
    # raw serves Git text without following weight-storage redirects.
    prefix = 'datasets/' if typ == 'dataset' else ''
    url = f'{HOST}/{prefix}{repo}/raw/{rev}/{quote(file["path"], safe="/")}'
    with request('GET', url, stream=True) as response:
        if response.status_code != 200:
            return None
        chunks, size = [], 0
        for chunk in response.iter_content(16384):
            size += len(chunk)
            if size > max_bytes:
                return None
            chunks.append(chunk)
        return b''.join(chunks).decode('utf-8', errors='replace')


def needed(typ, repo, info, loader):
    files = {x['path']: x for x in info['files']}
    prefix = loader.get('prefix', '')
    patterns = [(prefix + '/' if prefix else '') + p for p in loader['selectors']]
    selected = {path for path in files if any(fnmatch.fnmatchcase(path, p) for p in patterns)}
    # Expand package descriptor file references, including nested registry entries.
    descriptors = []
    queue = sorted(selected)
    visited = set()
    while queue:
        path = queue.pop(0)
        if path in visited:
            continue
        visited.add(path)
        if not path.endswith(('.json', '.yaml', '.yml')):
            continue
        text = small_text(typ, repo, info['revision'], files[path])
        if text is None:
            continue
        descriptors.append({'path': path, 'sha256': hashlib.sha256(text.encode()).hexdigest(),
                            'text': text})
        strings = set(re.findall(r'[A-Za-z0-9_./+%-]+\.(?:mdlus|ckpt|onnx|npy|npz|nc|h5|tar|zip|pt|pth|checkpoint|json|yaml|yml)', text))
        for ref in strings:
            for candidate in (ref, str(Path(path).parent / ref), Path(ref).name):
                if candidate in files and candidate not in selected:
                    selected.add(candidate)
                    queue.append(candidate)
    optional = {(prefix + '/' if prefix else '') + p for p in loader.get('optional_selectors', [])}
    missing = [p for p in patterns if p not in optional and not any(fnmatch.fnmatchcase(path, p) for path in files)]
    return {'loader': loader['key'], 'classes': loader['classes'],
            'source': loader['source'], 'source_line': loader['default_package_line'],
            'requested_revision': loader['requested_revision'], 'revision': info['revision'],
            'scope': loader['scope'], 'selectors': patterns, 'optional_selectors': sorted(optional),
            'files': [files[p] for p in sorted(selected)],
            'unmatched_selectors': missing, 'descriptors': descriptors}


def licence_class(repo, card, texts, policy):
    fields = {k: v for k, v in card.items() if k.startswith('license')}
    evidence = json.dumps(fields).lower() + '\n' + '\n'.join(texts).lower()
    if repo.lower() in {x.lower() for x in policy['research_only_repositories']}:
        return 'research-only', 'source restriction or unlicensed mixed mirror', fields
    if any(x in evidence for x in policy['restricted_markers']):
        return 'research-only', 'NC/ND or non-commercial terms', fields
    if any(x in evidence for x in policy['permissive_aliases']):
        return 'permissive-ships', 'recognised permissive licence', fields
    return 'research-only', 'missing or unrecognised licence', fields


def probe(typ, repo, info, loaders):
    actual = info['files']
    candidates = [x['path'] for x in actual if x['path'].lower().endswith(WEIGHT_SUFFIXES)]
    if not candidates:
        candidates = [x['path'] for x in actual
                      if x.get('sha256') and not x['path'].lower().endswith(('.md', '.json', '.yaml'))]
    if not candidates and actual:
        candidates = [x['path'] for x in actual if x['path'] not in ('.gitattributes', 'README.md')]
    known = bool(candidates)
    if not candidates:
        candidates = [(l.get('prefix', '') + '/' if l.get('prefix') else '') + p
                      for l in loaders for p in l['selectors']
                      if not any(c in p for c in '*?[') and p.lower().endswith(WEIGHT_SUFFIXES)]
    if not candidates:
        # A descriptor is useful denial evidence, but is explicitly not a weight.
        candidates = [(l.get('prefix', '') + '/' if l.get('prefix') else '') + p
                      for l in loaders for p in l['selectors'] if not any(c in p for c in '*?[')]
    if not candidates:
        return {'method': None, 'status': None, 'access': 'no payload listed', 'file': None}
    path = sorted(candidates)[0]
    rev = info['revision'] or (loaders[0]['requested_revision'] if loaders else 'main')
    url = resolve_url(typ, repo, rev, path)
    with request('HEAD', url) as response:
        status = response.status_code
        denial = {k: response.headers[k] for k in ('X-Error-Code', 'X-Error-Message') if k in response.headers}
    access = 'public' if status in (200, 302) else 'denied' if status in (401, 403) else 'unconfirmed'
    return {'method': 'HEAD', 'url': url, 'file': path, 'revision': rev,
            'status': status, 'access': access,
            'file_evidence': 'API listing' if known else 'source-only; existence not confirmed by API',
            'is_weight': path.lower().endswith(WEIGHT_SUFFIXES), 'error_headers': denial}


def inventory_row(seed, config):
    typ, repo = seed['type'], seed['repository']
    loaders = [x for x in config['loaders'] if x['repository'].lower() == repo.lower()] if typ == 'model' else []
    current = metadata(typ, repo)
    variants = {}
    # Explicit pinned API query confirms the snapshot behind the current listing.
    if current['revision']:
        pinned = metadata(typ, repo, current['revision'])
        if pinned['revision'] == current['revision']:
            current = pinned
        else:
            current['pin_error'] = pinned
    for l in loaders:
        rev = l['requested_revision']
        if rev not in variants:
            info = current if rev in ('main', current['revision']) else metadata(typ, repo, rev)
            variants[rev] = {'metadata': info, 'requirements': []}
        variants[rev]['requirements'].append(needed(typ, repo, variants[rev]['metadata'], l))
    texts = []
    for file in current['files']:
        if file['path'] in current.get('licence_files', []):
            content = small_text(typ, repo, current['revision'], file)
            if content is not None:
                texts.append({'path': file['path'], 'text': content,
                              'sha256': hashlib.sha256(content.encode()).hexdigest()})
    cls, reason, fields = licence_class(repo, current['card_data'], [x['text'] for x in texts], config['licence_policy'])
    return {**seed, **current, 'default_packages': variants, 'licence': current['card_data'].get('license'),
            'licence_fields': fields, 'licence_texts': texts, 'licence_class': cls,
            'licence_class_reason': reason, 'anonymous_probe': probe(typ, repo, current, loaders)}


def revision_patch(table_path, rows, config, output):
    original = table_path.read_text()
    table = json.loads(original)
    new = copy.deepcopy(table)
    blocks, errors, changes = {}, [], []
    by_repo = {(r['type'], r['repository'].lower()): r for r in rows}
    loaders = {l['key']: l for l in config['loaders']}
    for key, model in table['models'].items():
        weights = model.get('weights', {})
        if not weights.get('host', '').startswith('huggingface'):
            continue
        loader_key = config['ingest_bindings'].get(key)
        if not loader_key:
            errors.append({'model': key, 'error': 'No table binding for declared HF model'})
            continue
        loader = loaders[loader_key]
        row = by_repo.get(('model', loader['repository'].lower()))
        variant = row['default_packages'][loader['requested_revision']]
        info = variant['metadata']
        requirement = next(x for x in variant['requirements'] if x['loader'] == loader_key)
        if not info['revision'] or not requirement['files']:
            errors.append({'model': key, 'error': 'Default package unresolved', 'api_status': info['api_status']})
            continue
        include = config.get('ingest_selectors', {}).get(key)
        files = [{k: f[k] for k in ('path', 'size', 'sha256')} for f in requirement['files']
                 if include is None or any(fnmatch.fnmatchcase(f['path'], p) for p in include)]
        blocks[key] = {'repository': loader['repository'], 'revision': info['revision'], 'files': files}
        new['models'][key]['weights'].update(blocks[key])
        if weights.get('repository') != loader['repository']:
            changes.append({'model': key, 'before': weights.get('repository'), 'after': loader['repository'],
                            'source': loader['source'], 'line': loader['default_package_line']})
    # Replace only each original weights object, preserving all other table formatting.
    amended = original
    for key, block in blocks.items():
        old_weights = table['models'][key]['weights']
        anchor = '"weights": ' + json.dumps(old_weights)
        if amended.count(anchor) != 1:
            raise ValueError(f'Cannot uniquely locate weights for {key}')
        amended = amended.replace(anchor, '"weights": ' + json.dumps(new['models'][key]['weights']), 1)
    patch = ''.join(difflib.unified_diff(original.splitlines(True), amended.splitlines(True),
                    fromfile='a/gpuwm/authorities/earth2-models.v1.json',
                    tofile='b/gpuwm/authorities/earth2-models.v1.json'))
    (output / 'earth2-revisions.json').write_text(json.dumps(blocks, indent=2) + '\n')
    (output / 'earth2-models-revisions.patch').write_text(patch)
    return {'models': len(blocks), 'errors': errors, 'repository_corrections': changes,
            'input_sha256': hashlib.sha256(original.encode()).hexdigest()}


def markdown(rows, summary):
    lines = ['# Anonymous HF Earth-2 inventory', '', 'Local draft. Snapshot metadata and source-file requirements only.', '',
             'Sizes are logical bytes across all listed files at each current repo revision. Shared blobs and repositories count separately.',
             'Default-package revisions, complete file lists, API errors, source expressions and HTTP receipts are in inventory.json.',
             'Requirements are a union across selectable class variants, with descriptor references expanded. Unmatched selectors are explicit in JSON.',
             'A denied API cannot establish whether a repository is gated, private, absent or unreleased. A 401 alone does not distinguish them.', '',
             '```json', json.dumps(summary, indent=2), '```', '',
             '| repo | type | revision | size (bytes) | licence | class | gated + anonymous HTTP | files needed |',
             '|---|---|---|---:|---|---|---|---|']
    for row in rows:
        need = []
        for rev, variant in row['default_packages'].items():
            for req in variant['requirements']:
                paths = ', '.join(x['path'] for x in req['files']) or ', '.join(req['selectors'])
                need.append(f'{req["loader"]}@{(req["revision"] or rev)[:12]}: {paths}')
        if not need:
            need = ['No default HF package in reviewed 0.19.0 source (discovery or mirror only)']
        p = row['anonymous_probe']
        cells = [row['repository'], row['type'], (row['revision'] or 'UNRESOLVED')[:12],
                 str(row['total_size_bytes']) if row['total_size_bytes'] is not None else 'unknown',
                 str(row['licence_fields'] or 'none'), row['licence_class'],
                 f'API gated={json.dumps(row["gated"])}; API {row["api_status"]}; HEAD {p["status"]}; {p["access"]}', '; '.join(need)]
        lines.append('| ' + ' | '.join(x.replace('|', '\\|').replace('\n', ' ') for x in cells) + ' |')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'gpuwm/authorities/hf-earth2-inventory.v1.json')
    parser.add_argument('--ingest-table', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=16, choices=range(1, 17))
    args = parser.parse_args()
    started = time.monotonic()
    config = json.loads(args.config.read_text())
    seeds = {(r['type'], r['repository'].lower()): copy.deepcopy(r) for r in config['seeds']}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(search, typ, term, config['search']['author'])
                for typ in ('model', 'dataset') for term in config['search']['terms']]
        searches = [f.result() for f in jobs]
        for result in searches:
            for repo in result['repositories']:
                key = (result['type'], repo.lower())
                seed = seeds.setdefault(key, {'repository': repo, 'type': result['type'], 'evidence': []})
                seed['evidence'].append(f'anonymous {result["type"]} search author={result["author"]} term={result["term"]}')
        jobs = {pool.submit(inventory_row, seed, config): seed for seed in seeds.values()}
        rows = []
        for future in concurrent.futures.as_completed(jobs):
            row = future.result()
            rows.append(row)
            print(f'{row["type"]} {row["repository"]} api={row["api_status"]} sha={row["revision"]} HEAD={row["anonymous_probe"]["status"]}', flush=True)
    rows.sort(key=lambda x: (x['type'], x['repository'].lower()))
    args.out.mkdir(parents=True, exist_ok=True)
    patch_receipt = revision_patch(args.ingest_table, rows, config, args.out)
    denied = [{'repository': r['repository'], 'type': r['type'], 'api_gated': r['gated'],
               'api_status': r['api_status'], 'head_status': r['anonymous_probe']['status'],
               'file': r['anonymous_probe']['file']} for r in rows
              if r['gated'] or r['anonymous_probe']['status'] in (401, 403)]
    default_errors = [{'repository': r['repository'], 'requested_revision': rev,
                       'api_status': v['metadata']['api_status'], 'error': v['metadata'].get('error')}
                      for r in rows for rev, v in r['default_packages'].items()
                      if not v['metadata']['revision']]
    summary = {'rows': len(rows), 'models': sum(r['type'] == 'model' for r in rows),
               'datasets': sum(r['type'] == 'dataset' for r in rows),
               'resolved': sum(bool(r['revision']) for r in rows),
               'unresolved': [r['repository'] for r in rows if not r['revision']],
               'gated_or_denied_with_exact_status': denied,
               'total_bytes': sum(r['total_size_bytes'] or 0 for r in rows),
               'size_incomplete': [r['repository'] for r in rows if not r['size_complete']],
               'default_package_revision_errors': default_errors,
               'search_errors': sum(len(s['errors']) for s in searches),
               'requests': len(RECEIPTS), 'parallelism': args.workers,
               'earth2_revision_blocks': patch_receipt['models'],
               'wall_seconds': round(time.monotonic() - started, 3)}
    artifact = {'schema': 'gpuwm-hf-earth2-inventory-result-v1',
                'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                'anonymous': True, 'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
                'earth2studio': config['earth2studio'], 'summary': summary,
                'searches': searches, 'repositories': rows, 'ingest_patch': patch_receipt,
                'http_receipts': RECEIPTS}
    (args.out / 'inventory.json').write_text(json.dumps(artifact, indent=2) + '\n')
    (args.out / 'INVENTORY.md').write_text(markdown(rows, summary))
    print('SUMMARY ' + json.dumps(summary, sort_keys=True), flush=True)
    # Inaccessible rows stay in the inventory. They are evidence, not invented SHAs.
    return 1 if summary['search_errors'] or patch_receipt['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
