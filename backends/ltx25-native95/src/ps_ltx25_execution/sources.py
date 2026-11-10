"""Portable Source identities; no model, framework or resource work at import."""
from pathlib import Path
import hashlib
import importlib
import json

ROOT = Path(__file__).absolute().parent
BOUNDARY_NATIVE_SOURCE = dict(bytes=86690, sha256="0343c128880ac5f4ae092ff09d0b527ab2d94f163112bad9af640148a50b30d7")
BOUNDARY_GENERATED_SOURCE = dict(bytes=41192, sha256="d90c6fdede8f7d6497d635295f48c4f735b514478eee718c7c70ebc9d5c79235")


def record(path):
    p = Path(path).absolute()
    h = hashlib.sha256()
    with p.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b""):
            h.update(chunk)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=h.hexdigest())


def pinned(pin):
    if (not isinstance(pin, dict) or set(pin) != {"path", "bytes", "sha256"}
            or not Path(pin["path"]).is_absolute() or record(pin["path"]) != pin):
        raise ValueError("exact complete file identity required")
    return Path(pin["path"])


def read_json(pin):
    return json.loads(pinned(pin).read_text())


def validate_factory():
    value = json.loads((ROOT / "factory-sources.json").read_text())
    root = ROOT.parent / "ps_ltx25_native95"
    for name, expected in value["files"].items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or record(root / path)["sha256"] != expected:
            raise ValueError("captured native factory bytes differ: " + name)
    return value


def storage_sources():
    validate_factory()
    root = ROOT.parent / "ps_ltx25_native95"
    names = {"component_loader": "component_loader.py", "native": "native.py",
             "lifecycle": "lifecycle.py", "adapter": "adapter.py"}
    result = {key: record(root / name) for key, name in names.items()}
    for key, module, digest in (
            ("official_transformer", "diffusers.models.transformers.transformer_ltx2", "3a06d5043886efeb0d58f8284e7d7dc6559a71344568ccdb1230da9882727fad"),
            ("official_attention", "diffusers.models.attention", "1c387653405148e18f0d7972542d7ced68dd14f626ad8d5a433d91a9bcdf74e8")):
        pin = record(importlib.import_module(module).__file__)
        if pin["sha256"] != digest:
            raise ValueError("exact original native implementation required: " + module)
        result[key] = pin
    result["execution_frozen_storage"] = record(ROOT / "frozen_storage_official.py")
    if result["execution_frozen_storage"]["sha256"] != "ce0547cd98ace714c6091e8a0720e21d7cdb67a8f64f31902bc7a380d6ea02bf":
        raise ValueError("captured official-hook storage helper differs")
    result["execution_Torch_companion"] = record(ROOT / "storage-runtime-sources.json")
    result["original_checkpoint_helper"] = record(ROOT / "checkpointing.py")
    return result
