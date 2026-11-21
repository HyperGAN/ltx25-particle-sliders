"""Explicit native cache/train/RAW render commands; check/help allocate nothing."""
import argparse
import json
from pathlib import Path
from .contracts import BACKEND_NAME, CORE_REVISION, PROTOCOL, SOURCE_PROFILE, INITIALIZATION_POLICY, PG_VERSION


def main(argv=None):
    parser = argparse.ArgumentParser(description="LTX2.5 native-video Source candidate; no executed native qualification claimed")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="allocation-free supported scope and exact pins")
    cache = commands.add_parser("cache", help="one frozen native joint trajectory and positive video teacher per row")
    render = commands.add_parser("render", help="RAW-only native video+audio; does not build a teacher cache or load a game")
    for command in (cache, render):
        for name in ("model-dir", "model-lock-sha256", "output"):
            command.add_argument("--" + name, required=True)
        for name in ("text-device", "dit-device"):
            command.add_argument("--" + name, choices=("cpu", "cuda:0", "cuda:1"), default="cpu")
        command.add_argument("--base-precision", choices=("float32", "bfloat16"), default="bfloat16")
    for name, default in (("height", 512), ("width", 768), ("frames", 17), ("text-tokens", 128)):
        cache.add_argument("--" + name, type=int, default=default)
    cache.add_argument("--fps", type=float, default=24.)
    cache.add_argument("--pairs", required=True)
    cache.add_argument("--seeds", type=int, nargs="+", default=[7, 19])
    train = commands.add_parser("train", help="shared PG game with durable FULL/RAW commit after each actual update")
    train.add_argument("--config", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--resume")
    train.add_argument("--stop-after", type=int)
    render.add_argument("--artifact", required=True)
    render.add_argument("--prompt", required=True)
    render.add_argument("--seed", type=int, default=7)
    render.add_argument("--strength", type=float, default=1.)
    args = parser.parse_args(argv)
    if args.command == "check":
        print(json.dumps({"backend": BACKEND_NAME, "core": CORE_REVISION, "pg": PG_VERSION, "protocol": PROTOCOL,
            "source_profile": SOURCE_PROFILE, "initialization_policy": INITIALIZATION_POLICY,
            "ema_decay": None, "particle_bank": None, "status": "Source only; native/CPU/GPU/pretrained/quality unexecuted",
            "native": "full5D video velocity; real Gemma4/LTX2; jointly generated native video+audio"}, indent=2))
        return
    if args.command == "cache":
        import torch
        from .contracts import geometry, validate_core
        from .native import NativeRuntime
        from .adapter import TextConnectorLoRA
        from .cache import prepare_cache
        from particle_sliders.restoration import _restore_all
        validate_core()
        runtime, adapter = None, None
        try:
            runtime = NativeRuntime(args.model_dir, model_lock_sha256=args.model_lock_sha256,
                text_device=args.text_device, dit_device=args.dit_device, precision=args.base_precision)
            devices = sorted({d.index for d in (runtime.device, runtime.dit_device) if d.type == "cuda"})
            with torch.random.fork_rng(devices=devices):
                adapter = TextConnectorLoRA(runtime.text_encoder, runtime.connectors).attach()
            print(prepare_cache(runtime, adapter, json.loads(Path(args.pairs).read_text()), args.output,
                                g=geometry(vars(args)), seeds=args.seeds))
        finally:
            _restore_all([*([] if adapter is None else [adapter.close]), *([] if runtime is None else [runtime.close])])
    elif args.command == "train":
        from particle_sliders.lifecycle import SessionConfig
        from .lifecycle import open_training
        from .cache import atomic_json
        from .contracts import record
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        config = SessionConfig.from_dict(json.loads(Path(args.config).read_text()))
        with open_training(config, resume=args.resume) as session:
            def committed(metrics):
                step = metrics["step"]
                full, raw = output / f"FULL-{step:06d}.pt", output / f"RAW-{step:06d}.pt"
                session.checkpoint(full)
                session.export_inference(raw, selection="raw")
                receipt = {"completed_steps": step, "FULL": record(full), "RAW": record(raw), "ema_decay": None}
                atomic_json(output / f"COMMIT-{step:06d}.json", receipt)
                atomic_json(output / "LATEST.json", receipt)
            session.train(stop_after=args.stop_after, on_step=committed)
    else:
        from particle_sliders.inference import open_inference
        from .lifecycle import NATIVE_BACKEND
        from .render import export_video
        from .cache import atomic_json
        from .contracts import record
        root = Path(args.output)
        root.mkdir(parents=True, exist_ok=False)
        with open_inference(args.artifact, backend=NATIVE_BACKEND, strength=args.strength, output_type="pil",
                native_request={"prompt": args.prompt}, options={"model_dir": args.model_dir,
                "model_lock_sha256": args.model_lock_sha256, "text_device": args.text_device,
                "dit_device": args.dit_device, "base_precision": args.base_precision}) as session:
            result = session.render(prompt=args.prompt, strength=args.strength, seed=args.seed, output_type="pil")
            path = root / "native-video-audio.mp4"
            export_video(result, path)
            atomic_json(root / "finished-video.json", {"artifact": record(args.artifact), "media": record(path),
                "prompt": args.prompt, "strength": args.strength, "seed": args.seed, "native_joint_AV": True,
                "audio_invariant": False, "effect_measured": False, "catastrophe_assessed": False})


if __name__ == "__main__":
    main()
