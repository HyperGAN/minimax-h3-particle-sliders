"""Separately named runner API; original H3 factories and source identities stay exact.

No model operation, factory, initialization or optimizer is replaced. This API
does not automatically activate archived execution helpers or collect teachers.
"""
from contextlib import AbstractContextManager
import os
import sys
import threading
import uuid

from .ownership import RunnerAdmission, OwnerIdentity, UUIDS, require

_PROCESS_SCOPE = threading.Lock()
_FLAG = 'H3_PHASE_TWO_GPU_ADMITTED'


class Runner(AbstractContextManager):
    """External runner retains leases, policy monitoring and process ownership.

    Enter once in a fresh worker before framework imports. Failed cleanup keeps
    this scope and its marker pending; repair only foreign replacements, then
    retry close(). Do not release the runner's leases while cleanup is pending.
    """
    def __init__(self, admission):
        require(type(admission) is RunnerAdmission, 'RunnerAdmission required')
        self.admission = admission
        self._identity = None
        self._sessions = []
        self._entered = self._closing = self._closed = False
        self._owns_lock = False
        self._cuda_checked = False

    def __enter__(self):
        require(not self._entered and not self._closed and not self._closing,
                'A runner scope is single-use')
        require(_PROCESS_SCOPE.acquire(blocking=False), 'Another H3 runner scope is active')
        self._owns_lock = True
        try:
            require('torch' not in sys.modules, 'Enter the actual runner before framework imports')
            require(_FLAG not in os.environ, 'A preexisting admission marker is not borrowed authority')
            self.admission.check()
            self._identity = OwnerIdentity()
            os.environ[_FLAG] = '1'
            self._entered = True
            return self
        except BaseException:
            self._owns_lock = False
            _PROCESS_SCOPE.release()
            raise

    def _owned(self):
        require(self._entered and not self._closed, 'Runner is not active')
        self._identity.check()
        require(os.environ.get(_FLAG) == '1'
                and os.environ.get('CUDA_VISIBLE_DEVICES') == ','.join(UUIDS),
                'Foreign admission environment retained without mutation')
        self.admission.structural_check()

    def check(self):
        self._owned()
        require(not self._closing, 'Runner cleanup is pending')
        self.admission.check()

    def _framework(self):
        self.check()
        if self._cuda_checked:
            return
        import torch
        require(torch.cuda.device_count() == 2, 'Exactly two admitted visible CUDA devices required')
        def canonical(value):
            if isinstance(value, str):
                return 'GPU-' + str(uuid.UUID(value[4:] if value.startswith('GPU-') else value))
            return 'GPU-' + str(uuid.UUID(bytes=bytes(value.bytes)))
        require(tuple(canonical(torch.cuda.get_device_properties(i).uuid) for i in (0, 1)) == UUIDS,
                'Actual logical CUDA devices do not map to the captured physical UUIDs')
        self._cuda_checked = True

    def open_training(self, config, *, resume=None):
        self._framework()
        require(not self._sessions, 'One complete DiT/game owner at a time')
        from minimax_h3_phase95.lifecycle import open_training
        session = open_training(config, resume=resume)
        wrapped = _Training(self, session)
        self._sessions.append(wrapped)
        try:
            self.check()
        except BaseException as primary:
            try:
                wrapped.close()
            except BaseException as cleanup:
                raise BaseExceptionGroup('H3 prepare and cleanup failed', [primary, cleanup])
            raise
        return wrapped

    def open_inference(self, packet, *, options, strength=1., operation='render',
                       output_type='native_av_latents', native_request=None):
        self._framework()
        require(not self._sessions, 'Close the training/DiT owner before native inference')
        require(options.get('device', 'cuda:0') == 'cuda:0'
                and output_type == 'native_av_latents', 'Captured GPU0 entry and internal AV latent output required')
        from minimax_h3_phase95.lifecycle import NATIVE_BACKEND
        from particle_sliders.inference import open_inference
        session = open_inference(packet, backend=NATIVE_BACKEND, operation=operation,
                                 options=options, strength=strength, output_type=output_type,
                                 native_request=native_request)
        wrapped = _Inference(self, session)
        self._sessions.append(wrapped)
        try:
            self.check()
        except BaseException as primary:
            try:
                wrapped.close()
            except BaseException as cleanup:
                raise BaseExceptionGroup('H3 inference prepare and cleanup failed', [primary, cleanup])
            raise
        return wrapped

    def decode_latents(self, snapshot, latent_result, *, device='cuda:0'):
        self._framework()
        require(not self._sessions and device in ('cuda:0', 'cuda:1'),
                'Close the complete DiT owner before GPU-only original VAE phases')
        from minimax_h3_phase95.native import decode_latents
        result = decode_latents(snapshot, latent_result, device=device, check=self.check)
        self.check()
        return result

    def close(self):
        if self._closed:
            return
        self._owned()  # Refuse foreign marker/lifetime before any borrowed mutation.
        self._closing = True
        errors = []
        for session in list(reversed(self._sessions)):
            try:
                session.close()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise BaseExceptionGroup('H3 runner cleanup pending; retain leases and retry', errors)
        # Marker LAST. Floors may have failed; structural ownership, not live
        # capacity, authorizes only-own cleanup. No external lease is released.
        self._owned()
        del os.environ[_FLAG]
        self._closed, self._closing = True, False
        self._owns_lock = False
        _PROCESS_SCOPE.release()

    def __exit__(self, kind, primary, traceback):
        try:
            self.close()
        except BaseException as cleanup:
            if primary is not None:
                raise BaseExceptionGroup('H3 runner work and cleanup failed', [primary, cleanup])
            raise


class _OwnedSession(AbstractContextManager):
    def __init__(self, runner, session):
        self._runner, self._session, self._closed = runner, session, False

    def _call(self, name, *args, **kwargs):
        require(not self._closed, 'Session closed')
        self._runner.check()
        value = getattr(self._session, name)(*args, **kwargs)
        self._runner.check()
        return value

    @property
    def manifest(self):
        self._runner.check()
        return self._session.manifest

    def close(self):
        if self._closed:
            return
        self._runner._owned()
        self._session.close()  # Failure retains exact pending session and marker.
        self._closed = True
        self._runner._sessions.remove(self)

    def __enter__(self):
        self._runner.check()
        require(not self._closed, 'Session closed')
        return self

    def __exit__(self, kind, primary, traceback):
        try:
            self.close()
        except BaseException as cleanup:
            if primary is not None:
                raise BaseExceptionGroup('H3 session work and cleanup failed', [primary, cleanup])
            raise


class _Training(_OwnedSession):
    def step(self, **kwargs):
        return self._call('step', **kwargs)

    def train(self, *, stop_after=None, on_step=None):
        def checked_step(metrics):
            self._runner.check()
            if on_step is not None:
                on_step(metrics)
            self._runner.check()
        return self._call('train', stop_after=stop_after, on_step=checked_step)

    def checkpoint(self, path=None):
        return self._call('checkpoint', path)

    def export_inference(self, path=None, *, selection='raw'):
        require(selection == 'raw', 'RAW/noEMA only')
        return self._call('export_inference', path, selection=selection)


class _Inference(_OwnedSession):
    def response(self, **kwargs):
        return self._call('response', **kwargs)

    def render(self, **kwargs):
        return self._call('render', **kwargs)
