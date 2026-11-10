"""NEW software boundaries only; no native numerical or pretrained proof replay."""
import builtins
import pytest


def _forbid_framework(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "diffusers", "transformers", "particlegan", "particle_sliders"}:
            raise AssertionError("framework import happened before owner refusal: " + name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)


def test_no_owner_refuses_all_heavy_APIs_before_framework_or_files(monkeypatch):
    _forbid_framework(monkeypatch)
    from ps_ltx25_execution.api import open_training, train, cache, render_latents
    from ps_ltx25_execution.decode import decode_saved_boundary
    calls = [lambda: train(None, None, admission=None, stop_after=32),
             lambda: cache(None, None, None, admission=None),
             lambda: render_latents(None, admission=None, options=None, prompt="x", output=None),
             lambda: decode_saved_boundary(None, admission=None, options=None, output=None)]
    for call in calls:
        with pytest.raises(ValueError, match="runner must supply"):
            call()
    with pytest.raises(ValueError, match="runner must supply"):
        with open_training(None, admission=None):
            raise AssertionError("unreachable")


def test_bare_installed_console_refuses_before_reading_config_or_framework(monkeypatch):
    _forbid_framework(monkeypatch)
    from ps_ltx25_execution.cli import main
    with pytest.raises(SystemExit) as error:
        main(["sample-latents", "--config", "/nonexistent/config.json", "--output", "/nonexistent/output",
              "--raw", "/nonexistent/raw.pt", "--prompt", "a teapot"])
    assert error.value.code == 2


def test_packaged_original_factory_bytes_and_no_proxy_registrations():
    from importlib.metadata import distribution
    from ps_ltx25_execution.sources import validate_factory
    value = validate_factory()
    assert len([x for x in value["files"] if x.endswith(".py")]) == 12
    assert len([x for x in value["files"] if x.endswith(".json")]) == 10
    entries = distribution("particle-sliders-ltx25-native95").entry_points
    assert next(e.value for e in entries if e.group == "console_scripts" and e.name == "ltx25-native95") == "ps_ltx25_execution.cli:main"
    for group, target in (("particle_sliders.response_backends", "BACKEND"),
                          ("particle_sliders.inference_backends", "NATIVE_BACKEND")):
        assert next(e.value for e in entries if e.group == group and e.name == "ltx25_native_video95") == "ps_ltx25_native95.lifecycle:" + target
