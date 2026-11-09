"""Metadata-only readiness tests. No model imports, network, GPU or weight reads."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ltx_metadata_preflight', ROOT / 'tools/ltx25_native_preflight.py')
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def test_missing_snapshot_stays_blocked_and_command_returns_two(capsys):
    result = preflight.report()
    assert result['native_launch_permitted'] is False
    assert result['snapshot']['weights_verified'] is False
    assert any('native bridge is absent' in item for item in result['blockers'])
    assert preflight.main([]) == 2
    assert json.loads(capsys.readouterr().out)['native_launch_permitted'] is False


def test_metadata_cannot_authorize_or_open_shards(tmp_path, monkeypatch):
    for name in preflight.METADATA:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'_class_name': 'LTX2Pipeline', 'native_launch_permitted': True}))
    index = tmp_path / 'transformer/weights.safetensors.index.json'
    index.write_text(json.dumps({'weight_map': {'a': 'part.safetensors', 'b': 'missing.safetensors'}}))
    (index.parent / 'part.safetensors').write_bytes(b'fixture only')
    original = Path.read_bytes
    def read(path):
        assert path.suffix == '.json', 'weight payload must never be opened'
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', read)
    result = preflight.report(tmp_path)
    assert result['native_launch_permitted'] is False
    assert not result['snapshot']['weights_verified']
    assert result['snapshot']['indices'][0]['present_shards'] == ['part.safetensors']
    assert result['snapshot']['indices'][0]['absent_shards'] == ['missing.safetensors']


def test_bad_index_references_are_rejected(tmp_path):
    path = tmp_path / 'transformer/weights.safetensors.index.json'
    path.parent.mkdir()
    path.write_text(json.dumps({'weight_map': {'a': '../outside.safetensors', 'b': '/outside.safetensors',
                                               'c': {'invalid': True}, 'd': 'text.txt'}}))
    result = preflight.inspect_snapshot(tmp_path)
    assert len(result['indices'][0]['rejected_references']) == 4
    assert not result['weights_verified']


def test_metadata_size_bound_prevents_payload_read(tmp_path, monkeypatch):
    path = tmp_path / 'model_index.json'
    with path.open('wb') as output:
        output.truncate(preflight.MAX_JSON_BYTES + 1)
    monkeypatch.setattr(Path, 'read_bytes', lambda path: (_ for _ in ()).throw(AssertionError('oversize read')))
    assert preflight.read_metadata(path)['status'] == 'refused'


def test_invalid_json_is_not_completeness_evidence(tmp_path):
    path = tmp_path / 'model_index.json'
    path.write_text('not JSON')
    assert preflight.read_metadata(path)['status'] == 'invalid'
    assert preflight.inspect_snapshot(tmp_path)['completeness'].startswith('unknown')


def test_existing_framewise_critic_is_order_blind_cpu_evidence():
    import torch
    from particle_sliders_ltx25_dummy_preview.binding import VideoFieldCritic
    with torch.random.fork_rng():
        torch.manual_seed(17)
        critic = VideoFieldCritic(3, 2, (3, 2, 2), width=4).double()
        value = torch.randn(2, 3, 3, 2, 2, dtype=torch.float64)
        condition = torch.randn(2, 2, dtype=torch.float64)
        expected = critic(value, condition)
        reordered = critic(value[:, :, [2, 0, 1]], condition)
        torch.testing.assert_close(expected, reordered, rtol=1e-12, atol=1e-12)
