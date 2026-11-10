"""Supported owner-injected API; framework imports follow an actual owner check."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import gc
import threading

from .admission import STORAGE, checked
from .io import atomic_json
from .sources import record, validate_factory

_OWNER_LOCK = threading.RLock()
GEOMETRY = dict(height=512, width=768, frames=17, text_tokens=128, fps=24.)


def runtime_policy(owner, profile=STORAGE):
    check = checked(owner, profile)  # MUST precede even importing torch.
    import torch
    from .admission import DECODER
    count = 1 if profile == DECODER else 2
    if (torch.__version__ != "2.9.1+cu128" or torch.cuda.device_count() != count
            or torch.get_default_dtype() is not torch.float32
            or not torch.are_deterministic_algorithms_enabled()
            or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
        raise ValueError("runner must establish the exact captured runtime/numerics and visible device count")
    validate_factory()
    return check, torch


def local_options(options):
    if (options.get("text_device") != "cuda:0" or options.get("dit_device") != "cuda:1"
            or options.get("base_precision") != "bfloat16"):
        raise ValueError("captured full native BF16 text0/DiT1 declaration required")


def _close_all(actions):
    errors = []
    for call in actions:
        if call is not None:
            try:
                call()
            except BaseException as error:
                errors.append(error)
    if errors:
        raise BaseExceptionGroup("owned LTX cleanup failures", errors)


class OwnedTraining:
    """Complete steps, then restored checkpoint flags, then caller commits."""
    def __init__(self, session, checkpointing, storage, check, entry):
        self._session, self._checkpoint, self._storage = session, checkpointing, storage
        self._check, self.entry = check, entry
        self.closed = self.failed = False
        self.cleanup_pending = False
        self.cleanup_errors = ()
        self._cleanup_completed = set()

    @property
    def completed_steps(self):
        return self._session.game.completed_steps

    @property
    def configuration(self):
        return self._session.config

    def step(self):
        if self.closed or self.failed or self.cleanup_pending:
            raise RuntimeError("closed/failed attempt cannot be retried automatically")
        try:
            self._check()
            self._storage.placement.assert_resting()
            self._storage.assert_inactive()
            metrics = self._checkpoint.step()
            # Official scope has fully exited, including backward, BEFORE save.
            self._storage.placement.assert_resting()
            self._check()
            return metrics
        except BaseException:
            self.failed = True
            raise

    def commit(self, output):
        if self.closed or self.failed or self.cleanup_pending:
            raise RuntimeError("do not claim a failed/unknown update as a commit")
        self._check()
        self._checkpoint._restored()
        step = self.completed_steps
        output = Path(output)
        full, raw = output / f"FULL-{step:06d}.pt", output / f"RAW-{step:06d}.pt"
        commit = output / f"COMMIT-{step:06d}.json"
        if any(p.exists() for p in (full, raw, commit)):
            raise FileExistsError("new commit paths only; no completed-state overwrite")
        self._session.checkpoint(full)
        self._session.export_inference(raw, selection="raw")
        value = dict(schema="LTX25-owned-exact-state-commit-v1", completed_steps=step,
                     FULL=record(full), RAW=record(raw), selection="raw", ema_decay=None,
                     actual_optimizer_rates=dict(G=[x["lr"] for x in self._session.game.opt_g.param_groups],
                                                 D=[x["lr"] for x in self._session.game.opt_d.param_groups]),
                     initialization=self._session.game.backend.fresh_initialization)
        atomic_json(commit, value)
        atomic_json(output / "LATEST.json", dict(completed_steps=step, COMMIT=record(commit)), replace=True)
        self._check()
        return value

    def close(self):
        if self.closed:
            return
        self.cleanup_pending = True
        errors = []
        for name, owner in (("session", self._session), ("storage", self._storage)):
            if name in self._cleanup_completed:
                continue
            try:
                owner.close()
            except BaseException as error:
                errors.append(error)
            else:
                self._cleanup_completed.add(name)
        if errors:
            self.failed = True
            self.cleanup_errors += tuple(errors)
            # Keep every owner and the checkpoint profile available for cleanup.
            # The failed attempt remains forbidden from stepping or committing.
            raise BaseExceptionGroup("owned LTX cleanup failures", errors)
        self.cleanup_pending = False
        self.closed = True
        self._checkpoint = self._session = self._storage = None


@contextmanager
def open_training(config, *, admission, resume=None, execution_profile=STORAGE):
    check, _ = runtime_policy(admission, execution_profile)
    if execution_profile != STORAGE:
        raise ValueError("explicit captured storage execution profile required")
    from particle_sliders.lifecycle import read_packet
    from ps_ltx25_native95 import lifecycle
    from ps_ltx25_native95.native import NativeRuntime
    from ps_ltx25_native95.contracts import geometry
    from .checkpointing import OfficialLTXCheckpointing
    from .storage import ScopedOriginalLoader
    local_options(config.options)
    if geometry(config.options) != GEOMETRY:
        raise ValueError("this execution profile retains the captured full17-frame768x512 geometry")
    if config.options.get("ema_decay", "missing") is not None:
        raise ValueError("explicit ema_decay=None required")
    packet = deepcopy(resume) if isinstance(resume, dict) else read_packet(resume) if resume is not None else None
    runtime = storage = session = owned = None
    primary = None
    with _OWNER_LOCK:
        try:
            check()
            storage = ScopedOriginalLoader(config.options["model_dir"])
            with storage:
                runtime = NativeRuntime(config.options["model_dir"], model_lock_sha256=config.options["model_lock_sha256"],
                                        text_device="cuda:0", dit_device="cuda:1", precision="bfloat16")
            # Original Construction binds fresh/restore BEFORE owned optimizer creation.
            session = lifecycle.open_training(config, resume=packet, runtime=runtime)
            entry = session.checkpoint() if packet is None else packet
            if (session.manifest != entry["manifest"]
                    or session.game.signature != entry["payload"]["game"]["config"]
                    or session.checkpoint()["sha256"] != entry["sha256"]):
                raise RuntimeError("entire original typed state/optimizers/streams/RNG entry equality required")
            profile = OfficialLTXCheckpointing(session, runtime.transformer, entry_packet=entry)
            witness = dict(owner_receipt=admission.owner_receipt, execution_plan=deepcopy(admission.plan),
                           actual_construction=storage.witness(runtime), original_packet_sha256=entry["sha256"],
                           inherited_steps=session.game.completed_steps, fresh=packet is None,
                           factory_identity_preserved=True, fit_or_bitwise_equivalence_claim=False)
            owned = OwnedTraining(session, profile, storage, check, witness)
            check()
            yield owned
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                _close_all([owned.close if owned is not None else session.close if session is not None else None,
                            storage.close if storage is not None else None,
                            runtime.close if runtime is not None else None])
            except BaseException as cleanup:
                if primary is not None:
                    raise BaseExceptionGroup("owned training and cleanup failed", [primary, cleanup])
                raise
            finally:
                owned = session = storage = runtime = None
                gc.collect()


def train(config, output, *, admission, resume=None, stop_after):
    checked(admission, STORAGE)()
    if type(stop_after) is not int or not 0 < stop_after <= config.options["steps"]:
        raise ValueError("explicit stopping budget inside the frozen horizon required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with open_training(config, admission=admission, resume=resume) as session:
        inherited = session.completed_steps
        if type(stop_after) is not int or not inherited < stop_after <= config.options["steps"]:
            raise ValueError("explicit missing budget inside frozen horizon required")
        atomic_json(output / "execution-entry.json", session.entry)
        commits = [session.commit(output)]  # genuine FULL0 or exact restored entry
        while session.completed_steps < stop_after:
            session.step()
            commits.append(session.commit(output))
        result = dict(completed_steps=session.completed_steps, inherited_steps=inherited,
                      new_updates=session.completed_steps-inherited, commits=commits,
                      automatic_continuation=False, selection="raw", ema_decay=None,
                      effect_score=None, catastrophic_status="unmeasured", current_eligibility=None)
    atomic_json(output / "training-result.json", result)
    return result


def render_latents(artifact, *, admission, options, prompt, seed=7, strength=1., output):
    check, _ = runtime_policy(admission)
    from particle_sliders.inference import open_inference, read_inference
    from ps_ltx25_native95.lifecycle import NATIVE_BACKEND
    from ps_ltx25_native95.native import NativeRuntime
    from .boundary import capture_before_native_decode
    from .sources import BOUNDARY_NATIVE_SOURCE
    from .storage import ScopedOriginalLoader
    local_options(options)
    raw_identity = record(artifact)
    packet = read_inference(artifact)
    if packet["specification"]["configuration"]["geometry"] != GEOMETRY:
        raise ValueError("the captured full17-frame768x512 RAW geometry is required before construction")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    storage = session = runtime = None
    primary = None
    with _OWNER_LOCK:
        try:
            check()
            storage = ScopedOriginalLoader(options["model_dir"])
            with storage:
                session = open_inference(packet, backend=NATIVE_BACKEND, options=options,
                                         strength=strength, operation="render", output_type="latent",
                                         native_request=dict(prompt=prompt))
            values = [cell.cell_contents for cell in (session.binding.close.__closure__ or ())]
            owners = [value for value in values if type(value) is NativeRuntime]
            if len(owners) != 1 or session.selection != "raw":
                raise RuntimeError("one original strict RAW factory owner required")
            runtime = owners[0]
            construction = storage.witness(runtime)
            def completed_boundary(receipt):
                check()
                storage.placement.assert_resting()
                storage.placement.close()
                storage.placement = None
            with capture_before_native_decode(runtime, output / "boundary", BOUNDARY_NATIVE_SOURCE,
                                              check, after_capture=completed_boundary) as calls:
                returned = session.render(prompt=prompt, seed=seed, strength=strength, output_type="latent")
                del returned  # denormalized video return is NEVER the saved decoder input
            native_receipt = record(output / "boundary/complete-native-AV-before-decode.json")
            if len(calls) != 1:
                raise RuntimeError("one complete native boundary required")
            result = dict(schema="LTX25-owned-public-latent-phase-v1", sampling_output_type="latent",
                          future_decoder_output_type="pil", complete_native_joint_sampling=True,
                          complete_finished_media=False, native_calls=1, decoder_calls=0,
                          RAW=raw_identity, native_receipt=native_receipt,
                          artifact=calls[0]["artifact"], model_identity=deepcopy(runtime.identity),
                          request=dict(prompt=prompt, seed=seed, strength=strength, geometry=GEOMETRY),
                          execution_plan=deepcopy(admission.plan), owner_receipt=admission.owner_receipt,
                          producer_source=record(__file__), captured_factory=validate_factory(),
                          actual_construction=construction, new_training_or_teacher_calls=0,
                          returned_latents_used_as_decode_input=False, fit_or_bitwise_equivalence_claim=False)
            check()
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                _close_all([session.close if session is not None else None,
                            storage.close if storage is not None else None])
            except BaseException as cleanup:
                if primary is not None:
                    raise BaseExceptionGroup("native latent call and cleanup failed", [primary, cleanup])
                raise
            finally:
                session = storage = runtime = None
                gc.collect()
    if record(artifact) != raw_identity:
        raise RuntimeError("original RAW artifact changed during sampling")
    result["native_owner_closed"] = True
    atomic_json(output / "latent-phase.json", result)
    return result


def cache(options, pairs, output, *, admission, seeds=(7, 19)):
    """Original generic full-trajectory cache, a distinct recipe from two initial rows."""
    check, torch = runtime_policy(admission)
    from ps_ltx25_native95.native import NativeRuntime
    from ps_ltx25_native95.adapter import TextConnectorLoRA
    from ps_ltx25_native95.cache import prepare_cache
    from ps_ltx25_native95.contracts import validate_core
    from .storage import ScopedOriginalLoader
    local_options(options)
    validate_core()
    runtime = storage = adapter = None
    primary = None
    with _OWNER_LOCK:
        try:
            check()
            storage = ScopedOriginalLoader(options["model_dir"])
            with storage:
                runtime = NativeRuntime(options["model_dir"], model_lock_sha256=options["model_lock_sha256"],
                                        text_device="cuda:0", dit_device="cuda:1", precision="bfloat16")
            with torch.random.fork_rng(devices=[0, 1]):
                adapter = TextConnectorLoRA(runtime.text_encoder, runtime.connectors).attach()
            check()
            index_sha256 = prepare_cache(runtime, adapter, pairs, output, g=GEOMETRY, seeds=seeds)
            check()
            return dict(cache_index=record(Path(output) / "index.json"), cache_index_sha256=index_sha256,
                        policy="original generic complete8-step trajectory cache, not measured two-initial-row cohort",
                        completed_teacher_rows_replayed=False, automatic_retry=False)
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                _close_all([adapter.close if adapter is not None else None,
                            storage.close if storage is not None else None,
                            runtime.close if runtime is not None else None])
            except BaseException as cleanup:
                if primary is not None:
                    raise BaseExceptionGroup("cache and cleanup failed", [primary, cleanup])
                raise
            finally:
                adapter = runtime = storage = None
                gc.collect()
