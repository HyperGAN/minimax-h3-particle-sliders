"""Explicit original H3 module placement; no automatic map or changed math."""
from contextlib import contextmanager
from dataclasses import dataclass
import os
import torch

PROFILE = 'h3-full-native-phase-owned-CPU-TE-DiT24-26-PG0-FP32-VAE-v1'
EXECUTION_PROFILE = 'h3-native-allCUDA-DiT24-26-refiner1-1-PG0-group5-sync-unwind-FF512-v3'
# PROFILE above remains the immutable historical ENCODING identity. Its full
# four prompt returns are reused byte-for-byte; no encoding is retagged.
UUIDS = ('GPU-72c1b506-891d-b8bc-b353-e020585e1c47',
         'GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69')


@dataclass(frozen=True)
class Placement:
    profile: str = EXECUTION_PROFILE
    text_device: str = 'cpu'
    entry_device: str = 'cuda:0'
    game_device: str = 'cuda:0'
    first_device: str = 'cuda:0'
    second_device: str = 'cuda:1'
    main_boundary: int = 24
    refiner_boundary: int = 1
    outer_group_size: int = 5
    native_main_blocks: int = 50
    native_refiners: int = 2

    def validate(self):
        if self != Placement():
            raise ValueError('only exact full native resident DiT24-26/refiner1-1/game0/group5 profile is declared')
        if (os.environ.get('H3_PHASE_TWO_GPU_ADMITTED') != '1'
                or os.environ.get('CUDA_VISIBLE_DEVICES') != ','.join(UUIDS)):
            raise ValueError('Root two-device admission and exact owned physical GPU mask required before CUDA')
        # This is not resource admission; the actual Root envelope owns real
        # leases, CURRENT/caps, pre-framework capture and owned cleanup.
        return self

    @classmethod
    def _cpu_fixture(cls):
        """Private software fixture only; public validate() rejects it."""
        return cls(entry_device='cpu', game_device='cpu', first_device='cpu', second_device='cpu')


def tensors_to(value, device):
    """Device-only differentiable copies, preserving native dtype/layout."""
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, tuple):
        return tuple(tensors_to(item, device) for item in value)
    if isinstance(value, list):
        return [tensors_to(item, device) for item in value]
    if isinstance(value, dict):
        return {key: tensors_to(item, device) for key, item in value.items()}
    return value


@contextmanager
def construction_rng(devices=()):
    """A component load/constructor is not an extra game/native RNG draw."""
    with torch.random.fork_rng(devices=list(devices)):
        yield


class OriginalDiTPlacement:
    """Retain original blocks/Parameters and full original forward operations.

    Only module input devices change. In particular the final norm returns to
    entry0, so original full modality index_select tensors remain on entry0.
    No sequence/dimension sharding, attention replacement or dtype cast occurs.
    """
    def __init__(self, transformer, plan):
        if (len(transformer.transformer_blocks) != plan.native_main_blocks
                or len(transformer.token_refiner.refiner_blocks) != plan.native_refiners):
            raise ValueError('complete original50 main blocks and2 refiners required')
        self.plan, self._handles = plan, []
        self._parameters = None
        try:
            for name in ('proj_in', 'audio_proj_in', 'context_embedder', 'time_proj',
                         'time_embedder', 'rope', 'norm_out', 'proj_out', 'audio_proj_out'):
                transformer.get_submodule(name).to(device=plan.first_device)
            for index, block in enumerate(transformer.token_refiner.refiner_blocks):
                device = plan.first_device if index < plan.refiner_boundary else plan.second_device
                block.to(device=device)
                self._route(block, device)
            transformer.token_refiner.final_norm.to(device=plan.first_device)
            self._route(transformer.token_refiner.final_norm, plan.first_device)
            for index, block in enumerate(transformer.transformer_blocks):
                device = plan.first_device if index < plan.main_boundary else plan.second_device
                block.to(device=device)
                self._route(block, device)
            self._route(transformer.norm_out, plan.first_device)
            self._parameters = tuple((name, parameter) for name, parameter in transformer.named_parameters())
            self._transformer = transformer
        except BaseException:
            self.close()
            raise

    def _route(self, module, device):
        def route(owner, args, kwargs):
            return tensors_to(args, device), tensors_to(kwargs, device)
        handle = module.register_forward_pre_hook(route, with_kwargs=True)
        self._handles.append((module, handle, route))

    def assert_parameters(self):
        current = tuple(self._transformer.named_parameters())
        if (len(current) != len(self._parameters)
                or any(name != saved_name or parameter is not saved
                       for (name, parameter), (saved_name, saved) in zip(current, self._parameters))):
            raise RuntimeError('original placed DiT Parameter ownership changed')

    def close(self):
        errors = []
        for index in range(len(self._handles)-1, -1, -1):
            module, handle, route = self._handles[index]
            if module._forward_pre_hooks.get(handle.id) is not route:
                errors.append(RuntimeError('owned native device-route hook changed'))
                continue
            handle.remove()
            del self._handles[index]
        if not self._handles:
            self._parameters = self._transformer = None
        if errors:
            raise ExceptionGroup('native device-route cleanup incomplete', errors)
