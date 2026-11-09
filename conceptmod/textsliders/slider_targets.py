"""Exact historical LTX scaling compatibility excerpt; see provenance/source-map.json."""

def _is_lora_delta_layer(module) -> bool:
    return any(
        hasattr(module, name)
        for name in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
    )



def apply_continuous_lora_scale(module, scale: float) -> int:
    """Multiply LoRA delta by ``scale`` (continuous 0..1).

    PEFT ``PeftModel.set_adapter_scale`` often no-ops or only toggles,
    which made live smile grids at 0.25 / 0.5 / 1.0 byte-identical.
    This writes the layer ``scaling`` dict (and ``lora_scale`` /
    ``multiplier`` when present) the same way PEFT
    ``rescale_adapter_scale`` does: ``scaling[name] = base[name] * scale``.
    ``base`` is snapshotted on first call (the init ``alpha/r``).
    Works on ``get_peft_model`` wrappers and on duck-typed mocks.
    """
    scale = float(scale)
    updated = 0
    for child in module.modules():
        is_lora = _is_lora_delta_layer(child)
        scaling = getattr(child, "scaling", None)
        if is_lora and isinstance(scaling, dict) and scaling:
            base = getattr(child, "_krea_base_scaling", None)
            if not isinstance(base, dict) or any(key not in base for key in scaling):
                child._krea_base_scaling = {k: float(v) for k, v in scaling.items()}
                base = child._krea_base_scaling
            for key in scaling:
                scaling[key] = base[key] * scale
            updated += 1
        if is_lora and hasattr(child, "lora_scale") and not callable(child.lora_scale):
            child.lora_scale = scale
            updated += 1
        if is_lora and hasattr(child, "multiplier") and not callable(child.multiplier):
            child.multiplier = scale
            updated += 1
    return updated
