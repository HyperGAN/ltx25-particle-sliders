"""Synchronous frozen storage inside the captured checkpoint callable.

Original Parameter objects stay registered. Only their backing storage travels;
native LTX block math, dtypes and trainable PG ownership are unchanged. This is an
explicit new execution profile, not the historical resident placement.
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import sys
import torch


def validate_runtime():
    """This private hook-composition API is admitted ONLY for pinned Torch2.9.1."""
    pins = json.loads(Path(__file__).with_name('storage-runtime-sources.json').read_text())
    if torch.__version__ != pins['torch_version']:
        raise ValueError('exact declared Torch storage/checkpoint runtime required')
    root = Path(torch.__file__).absolute().parent
    for relative, pin in pins['files'].items():
        raw = (root / relative).read_bytes()
        if len(raw) != pin['bytes'] or hashlib.sha256(raw).hexdigest() != pin['sha256']:
            raise ValueError('storage/checkpoint operative Torch Source differs: ' + relative)
    if not callable(getattr(torch._C._autograd, '_top_saved_tensors_default_hooks', None)):
        raise ValueError('pinned active saved-tensor hook composition API unavailable')


@contextmanager
def stable_frozen_saves():
    """Compose, rather than replace, the active nonreentrant checkpoint hooks.

    A frozen Parameter saved directly must not follow Parameter.data back to
    CPU. Its detached Tensor view owns the exact onload storage independently.
    The ORIGINAL checkpoint pack still receives every save in original order,
    returns its own opaque holder and owns discard/recompute/early-stop. Its
    original unpack is untouched; differentiable tensors are passed unchanged.
    """
    pack, unpack = torch._C._autograd._top_saved_tensors_default_hooks(False)
    expected = ('_checkpoint_hook.__init__.<locals>.', '_recomputation_hook.__init__.<locals>.')
    if not any(getattr(pack, '__qualname__', '') == name + 'pack_hook'
               and getattr(unpack, '__qualname__', '') == name + 'unpack_hook' for name in expected):
        raise RuntimeError('exact active nonreentrant checkpoint or recomputation hook pair required')
    def pack_independent(value):
        return pack(value.detach() if not value.requires_grad else value)
    with torch.autograd.graph.saved_tensors_hooks(pack_independent, unpack):
        yield


def _storage(value):
    return (value.device, value.dtype, tuple(value.shape), tuple(value.stride()),
            value.storage_offset(), value.untyped_storage().data_ptr())


class OfficialLTXFrozenBlockStorage:
    """One complete frozen block, CPU between calls, original CUDA math.

    Install BEFORE the scoped official LTX API enables checkpointing. Its
    block callable includes onload/finally-offload on initial execution AND
    nonreentrant recomputation. No prefetch, async stream, dtype cast, trainable
    relocation or copy of potentially changed CUDA values back to CPU occurs.
    """
    def __init__(self, module, device, *, transformer, label):
        validate_runtime()
        import weakref
        self._transformer = weakref.ref(transformer)
        self.module, self.device, self.label = module, torch.device(device), label
        self.active = self.closed = False
        self.calls = self.completed_calls = self.unwound_calls = 0
        self._had_forward = 'forward' in vars(module)
        self._previous_forward = vars(module).get('forward')
        self.original = module.forward
        if getattr(transformer, 'gradient_checkpointing', False):
            raise ValueError('install storage before official LTX checkpoint scope')
        self.parameters = tuple(module.named_parameters())
        self.buffers = tuple(module.named_buffers())
        if not self.parameters or any(p.requires_grad for _, p in self.parameters):
            raise ValueError('only a complete frozen pretrained/fixed-Pika block may be stored')
        if any(value.device.type != 'cpu' for _, value in self.parameters + self.buffers):
            raise ValueError('capture original CPU storage before any resident CUDA placement')
        self.values = tuple((name, value, value.data, _storage(value), value._version)
                            for name, value in self.parameters + self.buffers)
        def forward(*args, **kwargs):
            return self._call(*args, **kwargs)
        self.forward = forward
        module.forward = forward

    def _identity(self):
        current = tuple(self.module.named_parameters()) + tuple(self.module.named_buffers())
        expected = self.parameters + self.buffers
        if (len(current) != len(expected)
                or any(name != old_name or value is not old_value
                       for (name, value), (old_name, old_value) in zip(current, expected))
                or any(value.requires_grad for _, value in self.parameters)):
            raise RuntimeError('frozen block Parameter/buffer ownership or freezing changed')

    def assert_offloaded(self):
        self._identity()
        if self.active or any(_storage(value) != saved or value._version != version
                              for _, value, _, saved, version in self.values):
            raise RuntimeError('original frozen CPU storage or tensor version changed')

    def _restore(self):
        # These are original CPU storages, not an extra full frozen-model copy.
        # Saved CUDA tensors keep their own references until backward needs them.
        for _, value, cpu, _, _ in self.values:
            value.data = cpu
        self.active = False
        self.assert_offloaded()

    def _call(self, *args, **kwargs):
        if self.closed:
            raise RuntimeError('closed frozen block storage')
        self.assert_offloaded()
        root = self._transformer()
        if root is None:
            raise RuntimeError('original official LTX transformer owner lost')
        if torch.is_grad_enabled() and getattr(root, 'gradient_checkpointing', False) is not True:
            raise RuntimeError('differentiable LTX streaming requires its scoped official checkpoint API')
        self.calls += 1
        self.active = True
        try:
            # copy=True exercises separate backing storage in CPU software cases.
            # Production onload is CUDA; no CPU block arithmetic is substituted.
            for _, value, cpu, _, _ in self.values:
                value.data = cpu.to(device=self.device, non_blocking=False, copy=True)
            if torch.is_grad_enabled():
                with stable_frozen_saves():
                    output = self.original(*args, **kwargs)
            else:
                output = self.original(*args, **kwargs)
            self.completed_calls += 1
            return output
        except BaseException:
            # Checkpoint's early-stop exception also passes through unchanged.
            self.unwound_calls += 1
            raise
        finally:
            primary = sys.exception()
            try:
                self._restore()
            except BaseException as cleanup:
                if primary is not None:
                    raise BaseExceptionGroup('frozen call and storage restoration failed', [primary, cleanup])
                raise

    def close(self):
        if self.closed:
            return
        self.assert_offloaded()
        root = self._transformer()
        if root is not None and getattr(root, 'gradient_checkpointing', False):
            raise RuntimeError('restore official LTX checkpoint scope before storage close')
        if vars(self.module).get('forward') is not self.forward:
            raise RuntimeError('frozen storage forward ownership changed')
        if self._had_forward:
            self.module.forward = self._previous_forward
        else:
            vars(self.module).pop('forward', None)
        self.closed = True
        self.original = self.forward = self.module = None
        self.parameters = self.buffers = self.values = ()
