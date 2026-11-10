"""Execution-only official LTX checkpointing, AFTER exact original session entry restore.

No framework import or work occurs at import. The caller owns admission and execution.
Each step returns only after exact runtime-field restoration; write FULL/RAW
and inspect identities outside this helper. This is not a checkpoint migration.
"""
from copy import deepcopy
import hashlib
import inspect
from pathlib import Path


OFFICIAL_SOURCES = {
    "LTX2VideoTransformer3DModel": "3a06d5043886efeb0d58f8284e7d7dc6559a71344568ccdb1230da9882727fad",
    "ModelMixin": "1dddcfbdfbbb44b45d096ee0866521d95688360ca9400d12b00f0781ea08e702",
    "Torch_checkpoint": "431b229da53972d79bed7e5dbfb6ace2922fff30192ea103208f484e59490f23",
}
ROUTE_SOURCES = {
    "TextConnectorLoRA": "f25708c7c7ab2dddaf490c8eb70608e46e8ccb72d605908c8ee78bcbdd5b1029",
    "NativeVideoResponse": "b12c0cdd75ef0db5f6f7a6d212edce0ddda9f9ee8fd70893a04ef8325b570022",
    "FrozenResponseAdapter": "28db035a4c6c7f17ae14cbd2c1fa6ec404465322e0e7829149150826ee22a7ad",
}
_FIELDS = ("gradient_checkpointing", "_gradient_checkpointing_func")
_MISSING = object()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def source_record(value):
    path = Path(value if isinstance(value, (str, Path)) else inspect.getsourcefile(inspect.unwrap(value))).absolute()
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def function_record(value):
    function = getattr(value, "__func__", value)
    return dict(actual_function_id=id(function), module=getattr(function, "__module__", None),
                qualname=getattr(function, "__qualname__", None), source=source_record(function))


def _fields(module):
    return {key: module.__dict__.get(key, _MISSING) for key in _FIELDS}


def _parameter_identity(module):
    # No values copied or weights hashed. Original frozen Parameter objects and
    # storage/device/dtype/version must survive the temporary execution scope.
    return tuple((name, id(p), p.data_ptr(), str(p.device), str(p.dtype), p.requires_grad, p._version)
                 for name, p in module.named_parameters())


class OfficialLTXCheckpointing:
    """One session entry-authenticated profile, used once per missing complete update.

    profile = OfficialLTXCheckpointing(session, runtime.transformer,
                                       entry_packet=packet)
    metrics = profile.step()       # whole session.step()/backward, then restore
    committed(metrics)             # original FULL/RAW callback OUTSIDE scope

    Construction authenticates the entire already-restored in-memory session entry via
    the unchanged core checksum and current session.checkpoint(). It never
    reads, rewrites, restores or retags a packet. After any failed attempt this
    object refuses another step; The caller handles genuine partial work separately.
    """
    def __init__(self, session, transformer, *, entry_packet, record=None):
        from particle_sliders.lifecycle import _check_packet
        from diffusers import LTX2VideoTransformer3DModel
        from diffusers.models.modeling_utils import ModelMixin
        from particle_sliders.pg09_backends import FrozenResponseAdapter
        from ps_ltx25_native95.adapter import TextConnectorLoRA
        from ps_ltx25_native95.native import NativeVideoResponse
        import torch.utils.checkpoint as checkpoint

        _check_packet(entry_packet)
        entry_step=entry_packet["payload"]["game"]["completed_steps"]
        require(type(entry_step)is int and entry_step>=0,"authentic nonnegative entry step")
        require(not session.closed and not session.inference_only, "original live training session required")
        require(entry_packet["schema"] == session.training_schema
                and entry_packet["payload"]["game"]["completed_steps"] == entry_step
                and session.game.completed_steps == entry_step
                and session.last_metrics == entry_packet["payload"]["last_metrics"]
                and ((entry_step==0 and session.last_metrics is None)or(isinstance(session.last_metrics,dict)and session.last_metrics.get("step")==entry_step))
                and entry_packet["payload"]["ema"] == dict(decay=None, values=None),
                "only the genuine original committed session entry, no prefix replay")
        require(session.manifest == entry_packet["manifest"]
                and session.game.signature == entry_packet["payload"]["game"]["config"]
                and session.checkpoint()["sha256"] == entry_packet["sha256"],
                "strict original session entry manifest, game, both optimizers and RNG equality BEFORE enabling")
        require(type(transformer) is LTX2VideoTransformer3DModel
                and transformer is session.game.backend.runtime.transformer,
                "exact original native LTX DiT owned by restored backend")
        require(getattr(transformer.enable_gradient_checkpointing, "__func__", None)
                is ModelMixin.enable_gradient_checkpointing,
                "unchanged official enable API; no monkeypatch")
        self.sources = {
            "LTX2VideoTransformer3DModel": source_record(LTX2VideoTransformer3DModel),
            "ModelMixin": source_record(ModelMixin),
            # checkpoint() is officially decorated by torch._disable_dynamo;
            # bind the real module leaf, not the decorator wrapper's file.
            "Torch_checkpoint": source_record(checkpoint.__file__),
        }
        require(all(self.sources[key]["sha256"] == digest for key, digest in OFFICIAL_SOURCES.items()),
                "exact reviewed official checkpointing implementation")
        route_sources = {"TextConnectorLoRA": source_record(TextConnectorLoRA),
                         "NativeVideoResponse": source_record(NativeVideoResponse),
                         "FrozenResponseAdapter": source_record(FrozenResponseAdapter)}
        require(all(route_sources[key]["sha256"] == digest for key, digest in ROUTE_SOURCES.items())
                and type(session.game.network) is TextConnectorLoRA
                and type(session.game.backend) is NativeVideoResponse,
                "exact original TE/video-connector-only LoRA response path")
        require(all(id(host) not in {id(module) for module in transformer.modules()}
                    for _, host in session.game.network._hosts),
                "no dynamic adapter host inside a recomputed DiT block")
        self.session, self.transformer = session, transformer
        self._manifest, self._signature = session.manifest, session.game.signature
        self._modules = tuple(transformer.named_modules())
        self._original = tuple((name, module, _fields(module)) for name, module in self._modules)
        require(any(hasattr(module, "gradient_checkpointing") for _, module in self._modules)
                and all(getattr(module, "gradient_checkpointing", False) is False for _, module in self._modules),
                "original flags must remain disabled through strict restore")
        require(all(not module.training for _, module in self._modules)
                and all(not p.requires_grad for p in transformer.parameters()),
                "original frozen eval DiT; do not train or unfreeze it")
        self._parameters = _parameter_identity(transformer)
        self._methods = {key: transformer.__dict__.get(key, _MISSING) for key in
                         ("enable_gradient_checkpointing", "disable_gradient_checkpointing", "_set_gradient_checkpointing", "forward")}
        self._next, self._active, self._failed = entry_step+1, False, False
        self.events, self._record = [], record
        self.identity = dict(schema="LTX25-official-scoped-nonreentrant-authenticated-entry-execution-v1",
            execution_source=source_record(__file__), original_entry_packet_sha256=entry_packet["sha256"],
            official_sources=self.sources, use_reentrant=False, preserve_rng_state=True,
            route_sources=route_sources,
            routing_scope="TE/video-connector hooks run before DiT; original differentiable conditions are block inputs; no host inside replay",
            weight_ema=None, native_math_geometry_precision_changed=False,
            additional_internal_block_recomputation=True, bitwise_CUDA_equivalence_measured=False,
            RNG_scope="Torch CPU plus tensor-argument device IDs, captured autocast; no new internal device",
            determinism_limit="official saved-tensor metadata checks are not a bitwise gradient proof")

    def _emit(self, value):
        self.events.append(deepcopy(value))
        if self._record is not None:
            self._record(deepcopy(value))

    def _restored(self):
        require(all(all(module.__dict__.get(key, _MISSING) is value for key, value in fields.items())
                    for _, module, fields in self._original), "exact prior flag/function attribute presence and identity")
        require(all(self.transformer.__dict__.get(key, _MISSING) is value for key, value in self._methods.items()),
                "original official method/forward attribute presence retained")
        require(all(not module.training for _, module in self._modules)
                and _parameter_identity(self.transformer) == self._parameters,
                "original frozen eval Parameter objects/storage/dtype/device unchanged")

    def step(self):
        require(not self._active and not self._failed, "no nesting or retry after a failed/unknown update")
        require(self.session.game.completed_steps == self._next - 1
                and self.session.manifest == self._manifest and self.session.game.signature == self._signature,
                "same exact session and ordered missing updates only")
        self._restored()
        self._active = True
        metrics, primary = None, None
        try:
            # The pinned official default sets use_reentrant=False and keeps
            # Torch's preserve_rng_state=True; no alternate function is passed.
            self.transformer.enable_gradient_checkpointing()
            actual = [dict(module=name, gradient_checkpointing=module.gradient_checkpointing,
                           checkpoint_function=function_record(module._gradient_checkpointing_func))
                      for name, module in self._modules if hasattr(module, "gradient_checkpointing")]
            require(actual and all(item["gradient_checkpointing"] is True for item in actual),
                    "official enable really selected original checkpointed blocks")
            self._emit(dict(status="enabled_before_complete_step", step=self._next,
                            requested=dict(use_reentrant=False, preserve_rng_state=True), actual=actual,
                            execution_source=self.identity["execution_source"]))
            metrics = self.session.step()  # includes all VJPs/backward/official optimizer calls
            require(type(metrics.get("step")) is int and metrics["step"] == self._next
                    and self.session.game.completed_steps == self._next, "one complete ordered native update")
        except BaseException as error:
            primary, self._failed = error, True
            raise
        finally:
            try:
                # disable() alone overwrites the function and leaves a new
                # attribute. Restore BOTH exact presence and original objects.
                for _, module, fields in self._original:
                    for key, value in fields.items():
                        if value is _MISSING:
                            module.__dict__.pop(key, None)
                        else:
                            setattr(module, key, value)
                for key, value in self._methods.items():
                    if value is _MISSING:
                        self.transformer.__dict__.pop(key, None)
                    else:
                        setattr(self.transformer, key, value)
                self._restored()
                self._emit(dict(status="restored_before_checkpoint_or_export", step=self._next,
                    complete_update_returned=metrics is not None and primary is None,
                    primary_error_type=None if primary is None else type(primary).__name__,
                    exact_original_attributes_restored=True, actual_completed_steps=self.session.game.completed_steps))
            except BaseException:
                self._failed = True
                raise
            finally:
                self._active = False
        self._next += 1
        return metrics
