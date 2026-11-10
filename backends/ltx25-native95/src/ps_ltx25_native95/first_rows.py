"""Explicit NEW two-row input cohort; never resume the lost eight-row rollout.

Retain seed7/native-step0 WHOLE bytes and original Source; compute only one NEW
neutral/positive native pair at seed19/native-step0. No work occurs on import.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import torch
from particle_sliders.pg09_backends import ResponseRow, PreparedResponse
from .cache import (FIRST_ROWS_POLICY, atomic_json, context_vector, cpu, identity,
                    validate_row, write_row)
from .contracts import PROTOCOL, file_hash, geometry, record, response_shape
from .native import held_positive_features
from .render import observation, pipeline_arguments


def require(value, message):
    if not value:
        raise ValueError(message)


def pinjson(pin):
    require(record(pin["path"]) == pin, "actual recovery metadata changed")
    return json.loads(Path(pin["path"]).read_text())


def validate_recovery(runtime, g, recovery):
    old = pinjson(recovery["old_observation"])
    inspection = pinjson(recovery["CPU_inspection"])
    require(old["schema"] == "Root-LTX25-actual-retained-first-teacher-row-observation-v1"
            and old["accepted"] is True and old["actual_tool"] == 79271
            and old["original_row_bytes_unchanged"] is True
            and old["training_updates"] == 0 and old["fresh_PG_initializer_called"] is False,
            "actual failed original cohort and complete retained first row required")
    require(inspection["schema"] == "LTX25-first-row-serializer-and-authenticated-inspection-v1"
            and inspection["status"] == "complete" and inspection["actual_check_count"] == 1
            and inspection["row"] == old["row"]
            and inspection["old_observation"] == recovery["old_observation"]
            and inspection["actual_old_runtime_identity"] == old["actual_old_runtime_identity"]
            and inspection["response_shape"] == list(response_shape(g))
            and inspection["all_tensor_values_finite"] is True
            and inspection["all_tensor_leaves_CPU"] is True
            and inspection["native_audio_output_present"] is False,
            "genuine one CPU serializer/weights_only inspection, no invented trajectory state")
    original = old["actual_old_runtime_identity"]
    require(set(original) == set(runtime.identity), "same complete original native identity layout")
    for key in original:
        if key not in ("source_profile", "product_sources"):
            require(original[key] == runtime.identity[key], "unchanged base/native/numerics required: " + key)
    for name, sha in original["product_sources"].items():
        if name not in ("cache.py", "contracts.py"):
            require(runtime.identity["product_sources"][name] == sha,
                    "every original native/forward/render/init/PG binding body stays exact: " + name)
    require(recovery["base_product_commit"] == "c899504f071ff4c7b53a89722e60980021cf593f"
            and recovery["old_seed"] == 7 and recovery["new_seed"] == 19
            and recovery["native_step"] == 0,
            "explicit new two-seed initial-row scientific input, not exact continuation")
    return old, inspection


def save_new_tensor(path, value):
    """Durable first tensor publication; an existing first is never overwritten."""
    path = Path(path)
    require(not path.exists(), "new joint-result sidecar only, never replay")
    pending = path.with_name(path.name + ".pending")
    with pending.open("xb") as stream:
        torch.save(cpu(value), stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(pending, path)
    pending.unlink()
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return record(path)


@torch.no_grad()
def prepare_first_rows(runtime, adapter, pairs, directory, *, g, recovery):
    require(geometry(g) == g and len(pairs) == 1 and set(pairs[0]) == {"neutral", "positive"}
            and all(isinstance(v, str) and v.strip() for v in pairs[0].values())
            and pairs[0]["neutral"] != pairs[0]["positive"], "one original full-native matched pair")
    old, inspection = validate_recovery(runtime, g, recovery)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    source, retained = Path(old["row"]["path"]), root / "row-00000.pt"
    require(record(source) == old["row"], "retained row WHOLE bytes, never re-evaluated")
    with source.open("rb") as incoming, retained.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    require(file_hash(retained) == old["row"]["sha256"] and retained.stat().st_size == old["row"]["bytes"],
            "original whole row copied byte-exact, not retagged or reserialized")
    fd = os.open(root, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    first = dict(path=retained.name, bytes=old["row"]["bytes"], sha256=old["row"]["sha256"],
                 row_identity=inspection["row_identity"], condition_sha256=inspection["condition_sha256"])
    origins = [dict(kind="retained_original_whole_row", seed=7, native_step=0, row=first,
        actual_original_whole_file=old["row"], actual_old_runtime_identity=old["actual_old_runtime_identity"],
        old_observation=recovery["old_observation"], CPU_inspection=recovery["CPU_inspection"],
        original_closed_terminal=old["actual_closed_terminal"], original_native_teacher_calls=2,
        teacher_calls_replayed=0, neutral_audio_output_retained=False)]
    rows = [first]
    atomic_json(root / "reused-row-origin.json", origins[0])
    shape, pair = response_shape(g), pairs[0]
    class FirstRowCompleted(Exception):
        pass
    completed = FirstRowCompleted("owned new seed19 row durably saved before next sampler forward")
    with observation(runtime), adapter.scaled(0.):
        neutral_tokens = runtime.tokenize(pair["neutral"], g["text_tokens"])
        plus_tokens = runtime.tokenize(pair["positive"], g["text_tokens"])
        nf, pf = runtime.features(neutral_tokens), runtime.features(plus_tokens)
        pf = held_positive_features(pf, nf, plus_tokens, neutral_tokens)
        plus_video, _, plus_mask = runtime.encode(plus_tokens, features=pf)
        with runtime.native_pipeline_placement():
            original = runtime.transformer.forward
            def forward(*args, **kwargs):
                require(len(rows) == 1 and not args and kwargs.get("return_dict") is False,
                        "only the genuinely new seed19 initial native pair; no next forward")
                native = {k: cpu(v) for k, v in kwargs.items() if k not in
                          {"hidden_states", "encoder_hidden_states", "timestep", "sigma", "return_dict"}}
                require(torch.equal(plus_mask.cpu(), native["encoder_attention_mask"].cpu()),
                        "same original positive/neutral masks")
                require(kwargs["timestep"].numel() == 1
                        and float(kwargs["timestep"]) == inspection["native_timestep"],
                        "same original native initial timestep, no redivision or guessed sigma")
                latent = runtime.pipe._unpack_latents(kwargs["hidden_states"], *shape[2:], 1, 1).detach().float()
                cond = {"tokens": cpu(neutral_tokens), "native": native,
                        "critic_video_neutral": cpu(kwargs["encoder_hidden_states"])}
                positive = dict(cond, video_context=cpu(plus_video))
                row = ResponseRow(cond, positive, cond, latent, timestep=cpu(kwargs["timestep"]).float())
                atomic_json(root / "new-seed19-CALL.json", dict(seed=19, native_step=0,
                    actual_runtime_identity=runtime.identity, before_new_teacher_calls=True,
                    exact_new_call_budget=2, automatic_retry=False))
                pending_row = save_new_tensor(root / "new-seed19-original-input-row.pt", {"row": asdict(row)})
                result = original(**kwargs)
                neutral_joint = save_new_tensor(root / "new-seed19-neutral-joint-result.pt",
                    dict(schema="LTX25-original-joint-native-first-result-v1", seed=19, native_step=0,
                         kind="neutral", result=result))
                teacher_args = dict(kwargs, encoder_hidden_states=plus_video)
                teacher_result = original(**teacher_args)
                positive_joint = save_new_tensor(root / "new-seed19-positive-joint-result.pt",
                    dict(schema="LTX25-original-joint-native-first-result-v1", seed=19, native_step=0,
                         kind="positive", result=teacher_result))
                target = runtime.pipe._unpack_latents(teacher_result[0], *shape[2:], 1, 1).detach().float()
                neutral = runtime.pipe._unpack_latents(result[0], *shape[2:], 1, 1).detach().float()
                prepared = PreparedResponse(row, target.detach(), neutral)
                validate_row(prepared, shape)
                second = write_row(root, 1, prepared)
                rows.append(second)
                origins.append(dict(kind="new_native_initial_pair", seed=19, native_step=0, row=second,
                    actual_runtime_identity=deepcopy(runtime.identity), original_native_teacher_calls=2,
                    teacher_calls_replayed=0, original_input_row=pending_row,
                    joint_results_sidecars=dict(neutral=neutral_joint, positive=positive_joint),
                    neutral_audio_output_retained=True, actual_cachebuilder_source=record(__file__)))
                atomic_json(root / "new-seed19-COMPLETE.json", origins[1])
                raise completed  # No return to scheduler/no subsequent native forward.
            runtime.transformer.forward = forward
            try:
                runtime.pipe(prompt=None, prompt_embeds=nf, prompt_attention_mask=neutral_tokens["attention_mask"],
                    generator=torch.Generator(device=runtime.device).manual_seed(19),
                    **pipeline_arguments(g, seed=19, output_type="latent"))
                raise RuntimeError("official pipeline returned without the owned first-row completion")
            except FirstRowCompleted as exc:
                if exc is not completed:
                    raise
            finally:
                runtime.transformer.forward = original
    require(len(rows) == len(origins) == 2, "both independent native first rows durably retained")
    index = dict(protocol=PROTOCOL, policy=FIRST_ROWS_POLICY, model=runtime.identity, geometry=g,
        response_shape=list(shape), pairs=pairs, seeds=[7, 19], rows=rows, teacher_origins=origins,
        trajectories=[dict(seed=seed, native_step=0, first_row=i, rows=1, complete=False)
                      for i, seed in enumerate((7, 19))],
        original_eight_row_trajectory_complete=False, old_teacher_forwards_replayed=0,
        old_original_teacher_calls_retained=2, new_native_teacher_calls=2,
        audio="original joint native INITIAL inputs, exact per-row teacher/student; two independent seeds",
        actual_cachebuilder_source=record(__file__))
    atomic_json(root / "index.json", index)
    return file_hash(root / "index.json")
