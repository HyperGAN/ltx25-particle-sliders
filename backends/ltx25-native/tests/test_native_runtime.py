"""Exercise the installed native factory on a small in-memory LTX runtime."""
import json

import torch
from torch import nn

from particle_sliders.backends import PreparedResponse, ResponseRow
from particle_sliders.lifecycle import SessionConfig, read_packet
from ps_ltx25_native.cache import POLICY, write_row
from ps_ltx25_native.contracts import BACKEND_NAME, PROTOCOL, file_hash, geometry, response_shape
from ps_ltx25_native.lifecycle import open_training


class TinyLTX:
    def __init__(self):
        with torch.random.fork_rng():
            torch.manual_seed(11)
            self.text_encoder = nn.Module()
            self.text_encoder.layers = nn.ModuleList()
            for _ in range(4):
                layer = nn.Module()
                layer.self_attn = nn.Module()
                layer.self_attn.q_proj = nn.Linear(4, 4)
                self.text_encoder.layers.append(layer)
            self.connectors = nn.Module()
            self.connectors.video_connector = nn.Module()
            self.connectors.video_connector.to_q = nn.Linear(4, 4)
            self.transformer = nn.Linear(4, 4)
        for module in (self.text_encoder, self.connectors, self.transformer):
            module.requires_grad_(False)
        self.device = self.dit_device = torch.device("cpu")
        self.dtype = torch.float32
        self.identity = {"test_model": "tiny-ltx"}
        self.pipe = self

    def encode(self, tokens):
        video = tokens.float()
        for layer in self.text_encoder.layers:
            video = torch.tanh(layer.self_attn.q_proj(video))
        video = self.connectors.video_connector.to_q(video)
        return video, None, torch.ones(1, 1, dtype=torch.bool)

    @staticmethod
    def _pack_latents(latent, _patch, _temporal_patch):
        return latent

    @staticmethod
    def _unpack_latents(latent, _frames, _height, _width, _patch, _temporal_patch):
        return latent

    @staticmethod
    def call(*, hidden_states, encoder_hidden_states, **_native):
        return (hidden_states + encoder_hidden_states.mean(),)


def write_cache(root, runtime, options):
    g = geometry(options)
    shape = response_shape(g)
    native = dict(
        audio_hidden_states=torch.zeros(1, 1),
        audio_encoder_hidden_states=torch.zeros(1, 1),
        encoder_attention_mask=torch.ones(1, 1, dtype=torch.bool),
        audio_encoder_attention_mask=torch.ones(1, 1, dtype=torch.bool),
        num_frames=shape[2], height=shape[3], width=shape[4], fps=g["fps"],
        audio_num_frames=1, video_coords=torch.zeros(1, 1), audio_coords=torch.zeros(1, 1),
        isolate_modalities=False, spatio_temporal_guidance_blocks=None, perturbation_mask=None,
        use_cross_timestep=True, attention_kwargs=None,
    )
    rows = []
    for index in range(2):
        tokens = torch.full((1, 1, 4), float(index + 1))
        neutral_video, _, _ = runtime.encode(tokens)
        a = dict(tokens=tokens, native=native, critic_video_neutral=neutral_video)
        b = dict(a, video_context=neutral_video + .2 * (index + 1))
        latent = torch.zeros(shape)
        row = ResponseRow(a, b, None, latent, timestep=torch.tensor([.1]))
        rows.append(write_row(root, index, PreparedResponse(
            row, latent + b["video_context"].mean(), latent + neutral_video.mean(),
        )))
    path = root / "index.json"
    path.write_text(json.dumps(dict(
        protocol=PROTOCOL, policy=POLICY, model=runtime.identity, geometry=g,
        rows=rows, response_shape=list(shape),
    )))
    return file_hash(path)


def test_native_factory_full_replay_and_raw(tmp_path):
    runtime = TinyLTX()
    cache = tmp_path / "cache"
    cache.mkdir()
    options = dict(
        model_dir=str(tmp_path), model_lock_sha256="0" * 64, cache_dir=str(cache),
        height=64, width=64, frames=9, steps=3, critic_width=2, ema_decay=None,
        base_precision="float32",
    )
    options["cache_index_sha256"] = write_cache(cache, runtime, options)
    config = SessionConfig(BACKEND_NAME, "tiny-native-replay", **options)
    with open_training(config, runtime=runtime) as session:
        groups = [group["dualnorm_convolution"] for group in session.game.opt_d.param_groups
                  if "dualnorm_convolution" in group]
        assert sorted(tuple(group["kernel_size"]) for group in groups) == [(1, 3), (9, 3), (9, 3)]
        assert torch.count_nonzero(session.game.network.branches[0].up.weight) == 0
        session.step()
        before = session.checkpoint(tmp_path / "full.pt")
        expected_metrics = session.step()
        expected_full = session.checkpoint()
        expected_raw = session.export_inference(selection="raw")
    assert read_packet(tmp_path / "full.pt")["sha256"] == before["sha256"]

    with open_training(config, runtime=TinyLTX(), resume=tmp_path / "full.pt") as resumed:
        assert resumed.step() == expected_metrics
        assert resumed.checkpoint()["sha256"] == expected_full["sha256"]
        assert resumed.export_inference(selection="raw")["sha256"] == expected_raw["sha256"]
