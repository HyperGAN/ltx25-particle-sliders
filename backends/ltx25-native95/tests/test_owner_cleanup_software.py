"""One NEW pure software cleanup boundary, authored without execution."""
import builtins
import pytest


def test_cleanup_fault_retains_handles_and_retries_only_missing_cleanup(monkeypatch):
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "diffusers", "transformers", "particlegan", "particle_sliders"}:
            raise AssertionError("cleanup boundary must not import a framework: " + name)
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    from ps_ltx25_execution.api import OwnedTraining

    calls = []
    class Owner:
        def __init__(self, name, fault=None):
            self.name, self.fault = name, fault
        def close(self):
            calls.append(self.name)
            if self.fault is not None:
                fault, self.fault = self.fault, None
                raise fault
    def no_work():
        raise AssertionError("cleanup/retry must not check resources or perform ML")

    session = Owner("session")
    storage_error = RuntimeError("injected storage cleanup fault")
    storage = Owner("storage", storage_error)
    checkpoint = object()
    owned = OwnedTraining(session, checkpoint, storage, no_work, {})
    with pytest.raises(BaseExceptionGroup) as failed:
        owned.close()
    assert failed.value.exceptions == (storage_error,)
    assert calls == ["session", "storage"]
    assert owned.failed and owned.cleanup_pending and not owned.closed
    assert (owned._session, owned._checkpoint, owned._storage) == (session, checkpoint, storage)
    assert owned._cleanup_completed == {"session"}
    assert owned.cleanup_errors == (storage_error,)
    with pytest.raises(RuntimeError, match="cannot be retried"):
        owned.step()
    with pytest.raises(RuntimeError, match="failed/unknown"):
        owned.commit(None)
    owned.close()
    assert calls == ["session", "storage", "storage"]
    assert owned.closed and not owned.cleanup_pending and owned.failed
    assert owned._session is owned._checkpoint is owned._storage is None
    assert owned.cleanup_errors == (storage_error,)  # retain the real failure
    owned.close()
    assert calls == ["session", "storage", "storage"]

    # Both failing owners still run in order, and their actual errors survive.
    calls.clear()
    session_error, storage_error = ValueError("session fault"), OSError("storage fault")
    owned = OwnedTraining(Owner("session", session_error), object(), Owner("storage", storage_error), no_work, {})
    with pytest.raises(BaseExceptionGroup) as both:
        owned.close()
    assert both.value.exceptions == (session_error, storage_error)
    assert owned._cleanup_completed == set() and not owned.closed
    owned.close()
    assert calls == ["session", "storage", "session", "storage"]
    assert owned.closed and not owned.cleanup_pending and owned.failed
    assert owned.cleanup_errors == (session_error, storage_error)
