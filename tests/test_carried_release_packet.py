"""Carried-source packet emission is bound to actual old/new Git objects."""
from pathlib import Path
import importlib.util
import json
import subprocess

import pytest

from test_prepared_publication import publication, manifest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('carried_packet_under_test', ROOT / 'tools/release/prepare_carried_release.py')
packet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packet)


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def repository(root, version, contents):
    root.mkdir()
    git(root, 'init', '-q')
    git(root, 'config', 'user.name', 'Fixture')
    git(root, 'config', 'user.email', 'fixture@example.invalid')
    git(root, 'config', 'core.autocrlf', 'false')
    (root / 'pyproject.toml').write_text(f'[project]\nname="gpuwm"\nversion="{version}"\n')
    source = root / 'gpuwm/core/x.py'
    source.parent.mkdir(parents=True)
    source.write_bytes(contents)
    git(root, 'add', '.')
    git(root, 'commit', '-qm', 'fixture endpoint')
    return git(root, 'rev-parse', 'HEAD')


@pytest.fixture
def emitted(tmp_path):
    old_repo, new_repo = tmp_path / 'old-public', tmp_path / 'new-candidate'
    old = repository(old_repo, '2.7.3', b'old\r\n')
    new = repository(new_repo, '2.7.4', b'new\n')
    receiver = tmp_path / 'receiver.py'
    receiver.write_text('''from pathlib import Path
import hashlib, difflib
def channel_scope():
    return {'schema':'arwen.carried-scope.v1','engine_root':'gpuwm','units':[
        {'engine':'gpuwm/core/x.py','carried':'core/x.py','kind':'file'}]}
def channel_protocol():
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    return {'schema':'arwen.carried-receiver.v1','byte_api':'named-raw-sides.v1',
        'receiver_sha256':sha(__file__),'rewiring_sha256':sha(__file__),
        'difflib_sha256':sha(difflib.__file__)}
hunks_from_bytes=rejoin_rows=divergence_document=key=lambda *a,**kw:None
''')
    value = packet.emit(new_repo, old, new, receiver, tmp_path / 'emitted', old_repo=old_repo)
    return dict(**value, old_repo=old_repo, repo=new_repo, old=old, new=new)


def test_separate_public_endpoint_is_verified_and_attached_exactly(emitted, tmp_path):
    artifact, receipt = emitted['release'], emitted['verification']
    data = packet.channel.read_json(artifact)
    assert data['old']['commit'] == emitted['old'] and data['new']['commit'] == emitted['new']
    assert packet.channel.unpack_release(data)[0]['gpuwm/core/x.py'] == b'old\r\n'
    stage = tmp_path / 'packet'
    stage.mkdir()
    fragment = packet.attach_to_packet(artifact, receipt, stage, revision=emitted['new'], version='2.7.4',
        repo=emitted['repo'], old_repo=emitted['old_repo'])
    assert fragment['release'] == packet.row(stage / artifact.name)
    assert fragment['verification'] == packet.row(stage / receipt.name)
    assert {p.name for p in stage.iterdir()} == {artifact.name, receipt.name}
    with pytest.raises(FileExistsError):
        packet.attach_to_packet(artifact, receipt, stage, revision=emitted['new'], version='2.7.4',
            repo=emitted['repo'], old_repo=emitted['old_repo'])


def test_packet_cannot_relabel_the_new_revision(emitted):
    with pytest.raises(ValueError, match='revision/version'):
        packet.verify_asset(emitted['release'], emitted['verification'], revision='a'*40, version='2.7.4')


def test_resealed_wrong_git_bytes_still_fail_endpoint_verification(emitted):
    channel = packet.channel
    artifact, receipt = emitted['release'], emitted['verification']
    original = channel.read_json(artifact)
    before, after = channel.unpack_release(original)
    after['gpuwm/core/x.py'] = b'not the committed engine'
    forged = channel.build_release(original['scope'], before, after, original['old'], original['new'])
    artifact.write_text(json.dumps(forged))
    proof = channel.read_json(receipt)
    proof.update(packet.summary(forged), asset=packet.row(artifact))
    receipt.write_text(json.dumps(channel.seal(proof)))
    with pytest.raises(ValueError, match='named Git'):
        packet.verify_asset(artifact, receipt, revision=emitted['new'], version='2.7.4',
            repo=emitted['repo'], old_repo=emitted['old_repo'])


def test_changed_sidecar_cannot_ride_original_publication_pin(emitted, publication):
    fragment = packet.verify_asset(emitted['release'], emitted['verification'], revision=emitted['new'], version='2.7.4')
    plan = dict(commit=emitted['new'], version='2.7.4', carried_physics=fragment)
    assert publication.verify_carried_release_assets(emitted['release'].parent, plan) == fragment
    proof = packet.channel.read_json(emitted['verification'])
    proof['receiver_protocol']['rewiring_sha256'] = '0'*64
    emitted['verification'].write_text(json.dumps(packet.channel.seal(proof)))
    with pytest.raises(publication.PublicationError, match='pinned publication'):
        publication.verify_carried_release_assets(emitted['release'].parent, plan)


def test_corrective_release_requires_channel_and_verification_assets(publication, manifest):
    missing = dict(manifest)
    missing.pop('carried_physics')
    with pytest.raises(publication.PublicationError, match='missing the carried-physics'):
        publication.load_manifest(missing, 'v9.8.7', 'a'*40, 'FahrenheitResearch/arwen')
    manifest['github']['assets'] = [r for r in manifest['github']['assets'] if not r['filename'].endswith('.verification.json')]
    with pytest.raises(publication.PublicationError, match='exact hashed GitHub assets'):
        publication.load_manifest(manifest, 'v9.8.7', 'a'*40, 'FahrenheitResearch/arwen')
