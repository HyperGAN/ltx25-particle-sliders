#!/usr/bin/env python3
"""Read bounded local metadata only. This tool can never authorize a native launch."""
import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, distribution
import json
from pathlib import Path, PurePosixPath

PRODUCT_BASE = '53b97f596b4ae0954a1f96267949d99f209e2364'
CORE = 'a794e3b4a8813b012f2c4faa3da82b78d4a90214'
MODEL = 'Lightricks/LTX-2.5-Diffusers'
MAX_JSON_BYTES = 1024 * 1024
COMPONENTS = ('transformer', 'text_encoder', 'connectors', 'vae', 'audio_vae', 'vocoder')
METADATA = ('model_index.json', 'scheduler/scheduler_config.json',
            'tokenizer/tokenizer_config.json') + tuple(name + '/config.json' for name in COMPONENTS)


def read_metadata(path):
    if path.suffix != '.json' or not path.is_file():
        return {'status': 'missing'}
    if path.stat().st_size > MAX_JSON_BYTES:
        return {'status': 'refused', 'reason': 'metadata size limit exceeded'}
    try:
        raw = path.read_bytes()
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError('metadata must be a JSON object')
        return {'status': 'read', 'sha256': hashlib.sha256(raw).hexdigest(), 'data': parsed}
    except (ValueError, OSError) as error:
        return {'status': 'invalid', 'reason': type(error).__name__}


def runtime_metadata():
    result = {}
    for name in ('diffusers', 'transformers', 'peft', 'torch', 'particle-sliders-core', 'hndl', 'hypergan'):
        try:
            item = distribution(name)
            result[name] = {'installed': True, 'version': item.version}
            if name == 'particle-sliders-core':
                origin = item.read_text('direct_url.json')
                result[name]['git_commit'] = json.loads(origin).get('vcs_info', {}).get('commit_id') if origin else None
        except PackageNotFoundError:
            result[name] = {'installed': False}
    return result


def inspect_snapshot(snapshot):
    if snapshot is None:
        return {'provided': False, 'weights_verified': False, 'completeness': 'unknown'}
    root = Path(snapshot)
    records = {name: read_metadata(root / name) for name in METADATA}
    indices = []
    for component in COMPONENTS:
        for index in sorted((root / component).glob('*.safetensors.index.json')):
            record = read_metadata(index)
            present, absent, rejected = [], [], []
            if record['status'] == 'read':
                mapping = record['data'].get('weight_map')
                if not isinstance(mapping, dict) or not mapping:
                    record['status'] = 'invalid'
                else:
                    for name in sorted(mapping.values(), key=str):
                        if not isinstance(name, str) or not name or PurePosixPath(name).suffix != '.safetensors' or PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:
                            rejected.append(str(name))
                            continue
                        shard = index.parent / name
                        # Stat only: never open or deserialize a weight payload.
                        (present if shard.is_file() and shard.stat().st_size > 0 else absent).append(name)
            indices.append({'index': str(index.relative_to(root)), 'status': record['status'],
                            'present_shards': present, 'absent_shards': absent, 'rejected_references': rejected})
    return {'provided': True, 'metadata': records, 'indices': indices,
            'weights_verified': False, 'completeness': 'unknown; metadata and shard presence are not weight verification'}


def report(snapshot=None):
    runtime = runtime_metadata()
    blockers = [
        'Modern native bridge is absent; the current factory explicitly admits only the dummy fixture.',
        'No immutable native model revision, complete artifact manifest, or verified weight identity is admitted.',
        'A pinned Diffusers LTX-2.5 loader, text encoder API and native configuration must be qualified.',
        'Native unit-patch packing, video/audio axes, dtype/device ownership, masks and scaled times are unqualified.',
        'The current critic averages frame scores and cannot qualify temporal order or motion.',
        'The working embedding teacher needs a reviewed native token-response/mask/domain contract; it cannot be recast as an image.',
        'Root must approve the native cohort and route code/tests through Claude actual-diff and return review before execution.',
    ]
    if not runtime['diffusers']['installed']:
        blockers.append('Diffusers is not installed in the inspected CPU environment.')
    if runtime['particle-sliders-core'].get('git_commit') != CORE:
        blockers.append('The inspected core origin is not the exact reviewed target.')
    return {'schema': 'ltx25-metadata-only-readiness-v1', 'product_base': PRODUCT_BASE,
            'core': CORE, 'model': MODEL, 'variant': 'distilled transformer only for any first native proposal',
            'native_launch_permitted': False, 'qualification': 'metadata inspection only; no GPU or native model qualification',
            'runtime': runtime, 'snapshot': inspect_snapshot(snapshot), 'blockers': blockers}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, help='Existing local snapshot; metadata and shard stat only.')
    arguments = parser.parse_args(argv)
    print(json.dumps(report(arguments.snapshot), indent=2, sort_keys=True))
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
