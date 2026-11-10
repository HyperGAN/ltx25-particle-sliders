"""Ordinary additive LoRA; no base registration, merge, PEFT or particle bank.

Host scope follows the historical LTX TE-last4/video-connector declaration.
The real Gemma4 attention_k_eq_v implementation may omit k_proj; enumerate
actual ordinary Linear leaves rather than inventing an absent projection.
"""
from contextlib import contextmanager
import math
import re
import torch
from torch import nn


def hosts(text_encoder, connectors):
    found = []
    for name, module in text_encoder.named_modules():
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
        if match and ".self_attn." in name and name.rsplit(".", 1)[-1] in {"q_proj", "k_proj", "v_proj", "o_proj"}:
            if type(module) is not nn.Linear:
                raise ValueError("native TE adapter hosts must be exact ordinary Linear leaves")
            found.append((int(match[1]), "te." + name, module))
    layers = sorted({layer for layer, _, _ in found})
    if len(layers) < 4:
        raise ValueError("native TE must expose at least four real attention layers")
    result = [(name, module) for layer, name, module in found if layer in layers[-4:]]
    for name, module in connectors.named_modules():
        if name.startswith("video_connector.") and type(module) is nn.Linear and (
                name.rsplit(".", 1)[-1] in {"to_q", "to_k", "to_v"} or name.endswith(".to_out.0")):
            result.append(("conn." + name, module))
    if not any(name.startswith("conn.") for name, _ in result) or len({id(m) for _, m in result}) != len(result):
        raise ValueError("actual unique video connector attention hosts required; audio hosts forbidden")
    return result


class Branch(nn.Module):
    def __init__(self, host):
        super().__init__()
        self.down = nn.Linear(host.in_features, 4, bias=False, device=host.weight.device, dtype=torch.float32)
        self.up = nn.Linear(4, host.out_features, bias=False, device=host.weight.device, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, value):
        return self.up(self.down(value.float()))  # alpha/rank=4/4=1, applied once.


class TextConnectorLoRA(nn.Module):
    rank, alpha = 4, 4.

    def __init__(self, text_encoder, connectors):
        super().__init__()
        target = hosts(text_encoder, connectors)
        object.__setattr__(self, "_hosts", tuple(target))
        self.target_names = tuple(name for name, _ in target)
        self.branches = nn.ModuleList([Branch(module) for _, module in target])
        self.names = tuple(name for name, _ in self.named_parameters())
        self.scale, self._handles = 1., []

    def attach(self):
        if self._handles:
            raise ValueError("adapter already attached")
        try:
            for (_, host), branch in zip(self._hosts, self.branches):
                def hook(module, args, output, branch=branch):
                    if self.scale == 0:
                        return output
                    return output + (branch(args[0]) * self.scale).to(output)
                self._handles.append(host.register_forward_hook(hook))
        except BaseException:
            self.close()
            raise
        return self

    @contextmanager
    def scaled(self, value):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("finite adapter strength required")
        old = self.scale
        self.scale = value
        try:
            yield self
        finally:
            self.scale = old

    def close(self):
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        object.__setattr__(self, "_hosts", ())  # Do not retain an owned12B TE after close.
