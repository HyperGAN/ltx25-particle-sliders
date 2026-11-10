"""Fresh-only scoped initialization, before any optimizer; original restore lineage."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from particlegan.init import Uniform, KEEP
from particle_sliders.initialization import ComponentInitialization, InitializationReceipt, initialize_fresh_components
from .contracts import (CORE_REVISION, SOURCE_PROFILE, INITIALIZATION_POLICY, digest, file_hash,
    PG_VERSION, PG_SOURCE_PROFILE, PG_SOURCE_MANIFEST_SHA256, PG_SOURCES_SHA256)


@dataclass(frozen=True)
class Construction:
    mode: str
    cohort: str
    options_sha256: str
    initialization_json: str | None = None


def source_identity():
    return {"core_revision": CORE_REVISION, "source_profile": SOURCE_PROFILE,
            "policy": INITIALIZATION_POLICY, "pg_source_profile": PG_SOURCE_PROFILE, "product_initialization_source_sha256": file_hash(__file__)}


def parameter_seed(seed, name):
    return int.from_bytes(hashlib.sha256(f"{INITIALIZATION_POLICY}:{seed}:{name}".encode()).digest()[:8], "big") % 2**63


def policies(adapter, critic, seed):
    scopes = {}
    # Down retains the ordinary bias-free Linear Kaiming(a=sqrt5) law;
    # zero-output UP leaves stay bit-exact zero under KEEP.
    for index, branch in enumerate(adapter.branches):
        for node in ("down", "up"):
            leaf = getattr(branch, node)
            if type(leaf) is not nn.Linear or leaf.bias is not None:
                raise ValueError("exact ordinary LoRA Linear leaves required")
            name = f"adapter.branch{index}.{node}"
            bound = 1 / math.sqrt(leaf.in_features)
            law = Uniform(-bound, bound) if node == "down" else KEEP
            scopes[name] = ComponentInitialization(leaf, "deterministic_orthogonal",
                {"weight": parameter_seed(seed, name + ".weight")}, distributions={"weight": law})
    for name, leaf in critic.named_modules():
        if not any(p.requires_grad for p in leaf.parameters(recurse=False)):
            continue
        if type(leaf) not in (nn.Linear, nn.Conv1d, nn.Conv3d) or any(leaf.children()):
            raise ValueError("undeclared temporal critic trainable leaf")
        key = "critic." + name
        scopes[key] = ComponentInitialization(leaf, "deterministic_orthogonal",
            {n: parameter_seed(seed, key + "." + n) for n, p in leaf.named_parameters() if p.requires_grad and p.numel()})
    actual = {id(p) for root in (adapter, critic) for p in root.parameters() if p.requires_grad}
    owned = {id(p) for scope in scopes.values() for p in scope.module.parameters() if p.requires_grad}
    if owned != actual:
        raise ValueError("explicit fresh scopes must cover exactly actual adapter/critic leaves")
    return scopes


def validate_receipt(value, cohort):
    import particle_sliders.initialization as shared
    if (not isinstance(value, dict) or value.get("schema") != "particle-sliders-scoped-fresh-initialization-v1"
            or not isinstance(cohort, str) or not cohort or value.get("cohort_id") != cohort
            or type(value.get("completed_steps")) is not int or value["completed_steps"] != 0
            or value.get("restored_from") is not None or value.get("source_identity") != source_identity()
            or value.get("helper_source") != {"file": Path(shared.__file__).name,
                "bytes": Path(shared.__file__).stat().st_size, "sha256": file_hash(shared.__file__)}
            or value.get("particlegan", {}).get("version") != PG_VERSION
            or value.get("particlegan", {}).get("source_profile") != PG_SOURCE_PROFILE
            or value.get("particlegan", {}).get("source_manifest_sha256") != PG_SOURCE_MANIFEST_SHA256
            or value.get("particlegan", {}).get("pg_sources_sha256") != PG_SOURCES_SHA256
            or value.get("particlegan", {}).get("unreleased") is not True
            or value.get("particlegan", {}).get("initializer_source", {}).get("sha256") !=
               "3534ff23efa413cdf8b4125b91e0d30671d873b3e2958547d4e744b750454b9b"):
        raise ValueError("receipt is not this new LTX2.5 fresh-initialization lineage")
    return InitializationReceipt(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))


def initialize_owned(adapter, critic, construction, options):
    if (not isinstance(construction, Construction) or construction.mode not in ("fresh", "resume")
            or construction.options_sha256 != digest(options)):
        raise ValueError("explicit fresh/resume construction required before optimizers")
    scopes = policies(adapter, critic, options.get("seed", 7))
    if construction.mode == "fresh":
        if construction.initialization_json is not None:
            raise ValueError("fresh cannot import a restore receipt")
        return initialize_fresh_components(scopes, cohort_id=construction.cohort, source_identity=source_identity(),
                                           completed_steps=0, restored_from=None, optimizers=(), pg_source_profile=PG_SOURCE_PROFILE)
    if construction.initialization_json is None:
        raise ValueError("resume must retain its original fresh receipt")
    value = json.loads(construction.initialization_json)
    receipt = validate_receipt(value, construction.cohort)
    if set(value["components"]) != set(scopes):
        raise ValueError("saved initialization scopes differ")
    for name, scope in scopes.items():
        saved = value["components"][name]
        parameters = {n: p for n, p in scope.module.named_parameters() if p.requires_grad and p.numel()}
        if (saved["method"] != scope.method or saved["gain"] != scope.gain
                or set(saved["parameters"]) != set(parameters)):
            raise ValueError("saved initialization method/parameters differ")
        for key, parameter in parameters.items():
            entry = saved["parameters"][key]
            if (entry["shape"] != list(parameter.shape) or entry["dtype"] != str(parameter.dtype)
                    or entry["seed"] != scope.parameter_seeds[key]):
                raise ValueError("saved initialization parameter identity differs")
    return receipt  # No call to initialize_fresh_components on restore.


def construction_for(config, packet):
    if packet is None:
        return Construction("fresh", config.cohort, digest(config.options))
    from particle_sliders.lifecycle import _check_packet
    from particle_sliders.response import TRAINING
    _check_packet(packet)
    if packet["schema"] != TRAINING or packet["manifest"]["config"] != config.as_dict():
        raise ValueError("resume requires exact native response FULL packet/config, no migration")
    game = packet["payload"]["game"]
    if (game["config"].get("pg_version") != PG_VERSION
            or game["config"].get("pg_source_profile") != PG_SOURCE_PROFILE):
        raise ValueError("historical PG packets retain their original loaders")
    receipt = validate_receipt(deepcopy(game["config"]["backend"]["fresh_initialization"]), config.cohort)
    return Construction("resume", config.cohort, digest(config.options), receipt.canonical_json)
