"""Owner-injected LTX execution; importing this namespace imports no framework."""
from .admission import OwnedAdmission, STORAGE, DECODER


def open_training(*args, **kwargs):
    from .api import open_training as call
    return call(*args, **kwargs)


def train(*args, **kwargs):
    from .api import train as call
    return call(*args, **kwargs)


def cache(*args, **kwargs):
    from .api import cache as call
    return call(*args, **kwargs)


def render_latents(*args, **kwargs):
    from .api import render_latents as call
    return call(*args, **kwargs)


def decode_saved_boundary(*args, **kwargs):
    from .decode import decode_saved_boundary as call
    return call(*args, **kwargs)
