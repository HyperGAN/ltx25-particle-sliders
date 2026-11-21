"""Console adapter for an externally owned runner; no admission is created here."""
from pathlib import Path
import argparse
import json


def main(argv=None, *, admission=None):
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Source scope only; imports no framework")
    cache = commands.add_parser("cache", help="new declared generic complete-trajectory cache")
    train = commands.add_parser("train", help="fresh or strict FULL; commits every complete update")
    latent = commands.add_parser("sample-latents", help="RAW original8 joint steps; no decoder or MP4")
    decode = commands.add_parser("decode", help="original3 decoder tail over saved prebranch AV only")
    for command in (cache, train, latent, decode):
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
    cache.add_argument("--pairs", required=True, type=Path)
    cache.add_argument("--seeds", type=int, nargs="+", default=[7, 19])
    train.add_argument("--resume", type=Path)
    train.add_argument("--stop-after", type=int, required=True)
    latent.add_argument("--raw", type=Path, required=True)
    latent.add_argument("--prompt", required=True)
    latent.add_argument("--seed", type=int, default=7)
    latent.add_argument("--strength", type=float, default=1.)
    decode.add_argument("--phase-record", type=Path, required=True,
                        help="JSON FileRecord for completed latent-phase.json, not a tensor filename")
    a = p.parse_args(argv)
    if a.command == "check":
        from .sources import validate_factory
        print(json.dumps(dict(captured_factory=validate_factory(), owner_injected=True,
                              public_native_software_validation=None, new_public_pretrained_or_quality_proof=False), indent=2))
        return
    if admission is None:
        p.error("runner must call main(argv, admission=actual_owned_admission); console creates no owner")
    from .admission import STORAGE, DECODER, checked
    checked(admission, DECODER if a.command == "decode" else STORAGE)()
    value = json.loads(a.config.read_text())
    if a.command == "train":
        from particle_sliders.lifecycle import SessionConfig
        from .api import train as call
        result = call(SessionConfig.from_dict(value), a.output, admission=admission,
                      resume=a.resume, stop_after=a.stop_after)
    else:
        from .api import cache as make_cache, render_latents
        options = value["options"] if set(value) == {"backend", "cohort", "options"} else value
        if a.command == "cache":
            result = make_cache(options, json.loads(a.pairs.read_text()), a.output, admission=admission, seeds=a.seeds)
        elif a.command == "sample-latents":
            local = {k:options[k] for k in ("model_dir", "model_lock_sha256", "text_device", "dit_device", "base_precision")}
            result = render_latents(a.raw, options=local, output=a.output, admission=admission,
                                    prompt=a.prompt, seed=a.seed, strength=a.strength)
        else:
            from .decode import decode_saved_boundary
            result = decode_saved_boundary(json.loads(a.phase_record.read_text()), options=options,
                                           output=a.output, admission=admission)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
