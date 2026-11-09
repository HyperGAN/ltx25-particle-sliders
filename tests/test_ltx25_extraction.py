"""Source-only extraction provenance checks; no umbrella checkout is required."""
import ast
import hashlib
import json
from pathlib import Path

import pytest
from torch import nn

ROOT=Path(__file__).resolve().parents[1]


def test_allowlist_source_hashes_and_no_private_or_unrelated_files():
    allow=json.loads((ROOT/'provenance/ALLOWLIST.json').read_text())
    provenance=json.loads((ROOT/'provenance/source-map.json').read_text())
    assert provenance['source_commit']=='2b4b88e3f4ba62ccdfaecebff288a78cd6ba847b'
    permitted=set(allow['whole_files']) | {allow['bounded_fragment']['destination']} | set(allow['generated_files'])
    actual={str(path.relative_to(ROOT)) for path in ROOT.rglob('*') if path.is_file()
            and not any(part in {'.git','__pycache__','.pytest_cache'} for part in path.relative_to(ROOT).parts)}
    assert actual==permitted
    for item in provenance['entries']:
        assert hashlib.sha256((ROOT/item['destination_path']).read_bytes()).hexdigest()==item['destination_sha256']
        assert len(item['source_blob'])==40 and len(item['source_sha256'])==64
        assert item['source_repository']=='https://github.com/HyperGAN/particle-sliders'
    assert not any(path.endswith(('.pt','.safetensors','.mp4','.wav','.env','.log')) for path in actual)
    assert not any(path.startswith(('.github/','models/','evidence/')) for path in actual)


def test_only_exact_required_scaling_functions_exported():
    source=(ROOT/'conceptmod/textsliders/slider_targets.py').read_text()
    tree=ast.parse(source);lines=source.splitlines(keepends=True)
    functions={node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
    assert set(functions)=={'_is_lora_delta_layer','apply_continuous_lora_scale'}
    provenance=json.loads((ROOT/'provenance/source-map.json').read_text())
    entry=next(item for item in provenance['entries'] if item['destination_path'].endswith('/slider_targets.py'))
    for original in entry['source_ranges']:
        node=functions[original['function']]
        fragment=''.join(lines[node.lineno-1:node.end_lineno]).encode()
        assert hashlib.sha256(fragment).hexdigest()==original['sha256']


def test_historical_scaler_keeps_off_signed_and_raw_scale_semantics():
    from conceptmod.textsliders.slider_targets import apply_continuous_lora_scale
    class Leaf(nn.Module):
        def __init__(self):
            super().__init__();self.lora_A=nn.Linear(2,2,bias=False)
            self.scaling={'default':2.};self.multiplier=1.;self.lora_scale=1.
    layer=Leaf()
    for scale in (0.,.5,1.,-1.,0.):
        assert apply_continuous_lora_scale(layer,scale)==3
        assert layer.scaling['default']==2.*scale
        assert layer.multiplier==scale and layer.lora_scale==scale
