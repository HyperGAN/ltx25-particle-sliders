"""One frozen neutral native AV trajectory; positive VIDEO teacher per row.

No second neutral teacher call, DiT projection, token sketch or cache rebuild
during training. Original jointly evolved audio is frozen only across the
matched teacher/student response evaluations of each recorded native row.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import torch
from particle_sliders.pg09_backends import ResponseRow, PreparedResponse
from .contracts import PROTOCOL, digest, file_hash, geometry, response_shape
from .native import map_tensors, held_positive_features
from .render import observation, pipeline_arguments, DISTILLED_SIGMAS

POLICY = "full5D-video-native-token/mask/audio/coords/time;teacher-preconnector-nonconcept-hold-v1"
FIRST_ROWS_POLICY = "two-independent-native-initial-rows-seed7-retained-seed19-new;full5D-joint-inputs-v1"


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, dict):
        return {key: cpu(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu(child) for child in value)
    return value


def identity(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        # Singleton expanded/scalar views can be logically contiguous while
        # retaining stride0. A NEW one-dimensional allocation has stride1.
        # Same-dtype copy preserves logical element bits, including BF16/ints.
        flat = torch.empty((value.numel(),), dtype=value.dtype, device="cpu")
        flat.copy_(value.reshape(-1))
        return {"shape": list(value.shape), "dtype": str(value.dtype),
                "sha256": hashlib.sha256(flat.view(torch.uint8).numpy().tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {key: identity(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [identity(child) for child in value]
    return value


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".pending")
    with temp.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def context_vector(row):
    """Full recorded context only; learned reductions live INSIDE the critic."""
    values = []
    def visit(value):
        if isinstance(value, torch.Tensor):
            if not torch.isfinite(value).all():
                raise ValueError("nonfinite native context")
            values.append(value.detach().cpu().float().reshape(-1))
        elif isinstance(value, dict):
            for key in sorted(value):
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
        elif isinstance(value, (int, float)):
            values.append(torch.tensor([value], dtype=torch.float32))
        elif value is not None:
            raise ValueError("unsupported native condition coordinate")
    visit(row.neutral)
    visit(row.positive["video_context"])
    visit(row.timestep)
    return torch.cat(values)


def write_row(root, index, prepared):
    stored = cpu({"row": asdict(prepared.row), "target": prepared.target, "neutral": prepared.neutral})
    path = root / f"row-{index:05d}.pt"
    temp = path.with_suffix(".pending")
    if path.exists():
        raise FileExistsError("completed teacher rows cannot be replayed or overwritten")
    with temp.open("xb") as stream:
        torch.save(stored, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    fd = os.open(root, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return {"path": path.name, "sha256": file_hash(path), "bytes": path.stat().st_size,
            "row_identity": identity(stored), "condition_sha256": identity(context_vector(prepared.row))["sha256"]}


def validate_row(prepared, shape):
    row = prepared.row
    if row.schedule is not None or (row.scale, row.strength, row.guidance) != (1., 1., 1.):
        raise ValueError("unipolar frozen positive raw-velocity native row required")
    for value in (row.latent, prepared.target, prepared.neutral):
        if tuple(value.shape) != tuple(shape) or value.dtype != torch.float32 or value.requires_grad or not torch.isfinite(value).all():
            raise ValueError("all full5D native response coordinates must be frozen finite FP32")
    if row.timestep.numel() != 1 or row.timestep.requires_grad or not torch.isfinite(row.timestep).all():
        raise ValueError("one original native timestep required")
    n, p = row.neutral, row.positive
    if (set(n) != {"tokens", "native", "critic_video_neutral"} or set(p) != set(n) | {"video_context"}
            or identity(n["native"]) != identity(p["native"]) or identity(n["tokens"]) != identity(p["tokens"])):
        raise ValueError("teacher/student must share exact neutral audio/latents/coords/masks and student tokens")
    native = n["native"]
    required = {"audio_hidden_states", "audio_encoder_hidden_states", "encoder_attention_mask", "audio_encoder_attention_mask",
                "num_frames", "height", "width", "fps", "audio_num_frames", "video_coords", "audio_coords",
                "isolate_modalities", "spatio_temporal_guidance_blocks", "perturbation_mask", "use_cross_timestep", "attention_kwargs"}
    if set(native) != required or native["isolate_modalities"] or native["use_cross_timestep"] is not True:
        raise ValueError("full unmodified distilled joint-native call contract required")
    if (native["spatio_temporal_guidance_blocks"] is not None or native["perturbation_mask"] is not None
            or native["attention_kwargs"] is not None):
        raise ValueError("STG/reference/attention modifications are outside this native profile")
    _, _, t, h, w = shape
    if (native["num_frames"], native["height"], native["width"]) != (t, h, w):
        raise ValueError("native latent geometry differs from full video field")
    context_vector(row)


class CachedResponses:
    def __init__(self, directory, runtime, index_sha256, g):
        self.root, self.runtime = Path(directory).resolve(), runtime
        self.index_sha256, self.g = index_sha256, deepcopy(g)
        self.response_shape = response_shape(g)
        if file_hash(self.root / "index.json") != index_sha256:
            raise ValueError("frozen native cache index differs")
        self.index = json.loads((self.root / "index.json").read_text())
        if (self.index.get("protocol") != PROTOCOL or self.index.get("policy") not in (POLICY, FIRST_ROWS_POLICY) or
                self.index.get("model") != runtime.identity or self.index.get("geometry") != g or
                self.index.get("response_shape") != list(self.response_shape)):
            raise ValueError("native model/source/geometry/audio/teacher identity differs")
        if self.index["policy"] == FIRST_ROWS_POLICY:
            origins = self.index.get("teacher_origins")
            if (self.index.get("seeds") != [7, 19] or len(self.index.get("rows", [])) != 2
                    or not isinstance(origins, list) or len(origins) != 2
                    or [o.get("seed") for o in origins] != [7, 19]
                    or [o.get("native_step") for o in origins] != [0, 0]
                    or [o.get("kind") for o in origins] != ["retained_original_whole_row", "new_native_initial_pair"]
                    or [o.get("row") for o in origins] != self.index["rows"]
                    or self.index.get("original_eight_row_trajectory_complete") is not False
                    or self.index.get("old_teacher_forwards_replayed") != 0):
                raise ValueError("explicit two independent first-row provenance required; no old trajectory retag")
        self.records = deepcopy(self.index["rows"])
        self._identity = deepcopy(self.index)
        if not 2 <= len(self.records) <= 4096:
            raise ValueError("requires2..4096 completed frozen native rows")
        widths = {context_vector(self._load(i, move=False).row).numel() for i in range(len(self.records))}
        if len(widths) != 1:
            raise ValueError("all native conditioning coordinates must have one declared layout")
        self.condition_width = widths.pop()

    def identity(self):
        if (file_hash(self.root / "index.json") != self.index_sha256 or self.index != self._identity or
                self.records != self._identity["rows"] or self.runtime.identity != self._identity["model"]):
            raise ValueError("frozen teacher authority changed")
        return deepcopy(self._identity)

    def __len__(self):
        return len(self.records)

    def _load(self, index, *, move):
        expected = self.records[index]
        relative = Path(expected["path"])
        path = (self.root / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(self.root):
            raise ValueError("cache row must remain inside immutable cache")
        raw = path.read_bytes()
        if len(raw) != expected["bytes"] or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
            raise ValueError("cached native row bytes differ")
        stored = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        if set(stored) != {"row", "target", "neutral"} or identity(stored) != expected["row_identity"]:
            raise ValueError("full original row identity differs")
        prepared = PreparedResponse(ResponseRow(**stored["row"]), stored["target"], stored["neutral"])
        validate_row(prepared, self.response_shape)
        if identity(context_vector(prepared.row))["sha256"] != expected["condition_sha256"]:
            raise ValueError("critic full native conditioning differs")
        if move:
            stored = map_tensors(stored, self.runtime.device)
            prepared = PreparedResponse(ResponseRow(**stored["row"]), stored["target"], stored["neutral"])
        return prepared

    def __getitem__(self, index):
        self.identity()
        return self._load(index, move=True)

    def conditions(self, indices):
        self.identity()
        return torch.stack([context_vector(self._load(int(i), move=False).row) for i in indices]).to(self.runtime.device)


@torch.no_grad()
def prepare_cache(runtime, adapter, pairs, directory, *, g, seeds=(7, 19)):
    """Official complete neutral rollout records real rows before each step.

    Its OFF forward already supplies neutral velocity; only the frozen-plus
    VIDEO teacher is evaluated additionally. The official sampler continues
    its original JOINT audio/video update. No fake random audio placeholder.
    """
    if geometry(g) != g or not pairs or not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("exact geometry, prompt pairs and original seeds required")
    if any(not isinstance(pair, dict) or set(pair) != {"neutral", "positive"} or
           any(not isinstance(v, str) or not v.strip() for v in pair.values()) or pair["neutral"] == pair["positive"] for pair in pairs):
        raise ValueError("distinct neutral/positive pairs required")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    rows, trajectories = [], []
    shape = response_shape(g)
    with observation(runtime), adapter.scaled(0.):
        for pair in pairs:
            neutral_tokens = runtime.tokenize(pair["neutral"], g["text_tokens"])
            plus_tokens = runtime.tokenize(pair["positive"], g["text_tokens"])
            nf, pf = runtime.features(neutral_tokens), runtime.features(plus_tokens)
            pf = held_positive_features(pf, nf, plus_tokens, neutral_tokens)
            plus_video, _, plus_mask = runtime.encode(plus_tokens, features=pf)
            for seed in seeds:
                start, native_calls = len(rows), []
                with runtime.native_pipeline_placement():
                    original = runtime.transformer.forward
                    def forward(*args, **kwargs):
                        if args or kwargs.get("return_dict") is not False:
                            raise ValueError("only exact official keyword native tuple call supported")
                        native = {k: cpu(v) for k, v in kwargs.items() if k not in
                                  {"hidden_states", "encoder_hidden_states", "timestep", "sigma", "return_dict"}}
                        if not torch.equal(plus_mask.cpu(), native["encoder_attention_mask"].cpu()):
                            raise ValueError("matched positive/neutral connector masks differ")
                        packed = kwargs["hidden_states"]
                        latent = runtime.pipe._unpack_latents(packed, *shape[2:], 1, 1).detach().float()
                        cond = {"tokens": cpu(neutral_tokens), "native": native,
                                "critic_video_neutral": cpu(kwargs["encoder_hidden_states"])}
                        positive = dict(cond, video_context=cpu(plus_video))
                        row = ResponseRow(cond, positive, cond, latent, timestep=cpu(kwargs["timestep"]).float())
                        result = original(**kwargs)  # ONE original OFF forward.
                        # Bypass this recording wrapper for the one extra
                        # positive teacher. Same placement bridge, same audio,
                        # coordinates and original timestep; video context only.
                        teacher_args = dict(kwargs, encoder_hidden_states=plus_video)
                        teacher = original(**teacher_args)[0]
                        target = runtime.pipe._unpack_latents(teacher, *shape[2:], 1, 1).detach().float()
                        neutral = runtime.pipe._unpack_latents(result[0], *shape[2:], 1, 1).detach().float()
                        prepared = PreparedResponse(row, target.detach(), neutral)
                        validate_row(prepared, shape)
                        rows.append(write_row(root, len(rows), prepared))
                        native_calls.append(float(row.timestep))
                        atomic_json(root / "progress.json", {"status": "partial", "completed_rows": len(rows),
                            "rows": rows, "protocol": PROTOCOL, "no_completed_row_replay": True})
                        return result
                    runtime.transformer.forward = forward
                    try:
                        runtime.pipe(prompt=None, prompt_embeds=nf, prompt_attention_mask=neutral_tokens["attention_mask"],
                            generator=torch.Generator(device=runtime.device).manual_seed(seed),
                            **pipeline_arguments(g, seed=seed, output_type="latent"))
                    finally:
                        runtime.transformer.forward = original
                if len(rows) - start != len(DISTILLED_SIGMAS):
                    raise ValueError("actual official native call count differs from distilled schedule")
                trajectories.append({"pair": pair, "seed": seed, "first_row": start, "rows": len(rows) - start,
                                     "native_timesteps": native_calls})
    index = {"protocol": PROTOCOL, "policy": POLICY, "model": runtime.identity, "geometry": g,
             "response_shape": list(shape), "pairs": pairs, "seeds": list(seeds), "trajectories": trajectories,
             "rows": rows, "audio": "original joint neutral trajectory, identical per teacher/student response row"}
    atomic_json(root / "index.json", index)
    return file_hash(root / "index.json")
