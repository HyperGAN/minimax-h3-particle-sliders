"""Verified shared activation steering beside the captured H3 FP4 owner."""

from dataclasses import asdict
from hashlib import sha256
from sys import modules
from threading import Lock
from types import ModuleType
from zipfile import ZipFile

import torch


CORE_REVISION = "b1f780decc516808cfd3019d5e5b674457161f86"
CORE_MEMBER = "particle_sliders/adaptive_steering.py"
CORE_SHA256 = "5fc2a5256a935eca9cecc365d9ac62e5e94cdc375d2ab33c5dc26eb344241814"
_MODULE_NAME = f"{__name__}._core_{CORE_REVISION}"
_LOAD_LOCK = Lock()


def gate_core(core_wheel):
    """Load the exact #269 equations without changing the installed FP4 runtime."""
    with ZipFile(core_wheel) as archive:
        source = archive.read(CORE_MEMBER)
    if sha256(source).hexdigest() != CORE_SHA256:
        raise ValueError("H3 FP4 requires the exact adaptive steering source from core PR #269")
    with _LOAD_LOCK:
        if _MODULE_NAME not in modules:
            module = ModuleType(_MODULE_NAME)
            modules[_MODULE_NAME] = module
            try:
                exec(compile(source, CORE_MEMBER, "exec"), module.__dict__)
            except BaseException:
                del modules[_MODULE_NAME]
                raise
        return modules[_MODULE_NAME]


def coupled_latent_readout(video, audio):
    """Concatenate FP32 row means, packed video first and packed audio second."""
    if any(not isinstance(value, torch.Tensor) or not value.is_floating_point()
           or value.ndim != 3 or value.shape[0] != 1 or not all(value.shape)
           for value in (video, audio)) or video.device != audio.device:
        raise ValueError("Expected one native packed video/audio latent pair on the same device")
    return torch.cat((video.mean(dim=(0, 1), dtype=torch.float32),
                      audio.mean(dim=(0, 1), dtype=torch.float32)))


class AVGatePlan:
    """Explicit measured direction and budget calibration for the H3 AV readout."""

    def __init__(self, *, core_wheel, method, parameter, calibration_sha256,
                 direction=None, seed=None, clabs=None):
        if (not isinstance(calibration_sha256, str) or len(calibration_sha256) != 64
                or any(c not in "0123456789abcdef" for c in calibration_sha256)):
            raise ValueError("Provide the SHA-256 of the H3 AV gate calibration record")
        if type(parameter) not in (int, float):
            raise ValueError("Provide an explicit numeric H3 AV gate threshold, bias, or rate")
        self._core = gate_core(core_wheel)
        self._clabs = self._core.CLABSParameters(**clabs) if clabs is not None else None
        self._method, self._parameter, self._seed = method, float(parameter), seed
        self._calibration_sha256 = calibration_sha256
        self._direction = None
        if method == "random":
            if direction is not None:
                raise ValueError("The matched random control does not use an H3 AV direction")
        else:
            if direction is None:
                raise ValueError("CFP and CLABS require a fitted joint H3 AV direction")
            direction = torch.as_tensor(direction, dtype=torch.float32).detach().clone()
            if (direction.ndim != 1 or not torch.isfinite(direction).all()
                    or not torch.isclose(direction.norm(), direction.new_tensor(1.0))):
                raise ValueError("H3 AV direction must be a finite unit vector")
            self._direction = direction
        self._new_gate()

    def _new_gate(self):
        return self._core.ActivationGate(self._method, self._parameter, seed=self._seed, clabs=self._clabs)

    def _score(self, readout, strength):
        if self._direction is None:
            return None
        return float(self._core.activation_projection(readout, self._direction, strength=strength))

    def _receipt(self):
        return {
            "core_revision": CORE_REVISION,
            "core_source_sha256": CORE_SHA256,
            "readout": "fp32_mean_packed_video_then_audio_before_adapter",
            "calibration_sha256": self._calibration_sha256,
            "method": self._method,
            "parameter": self._parameter,
            "seed": self._seed,
            "clabs": asdict(self._clabs) if self._clabs is not None else None,
            "direction": self._direction.tolist() if self._direction is not None else None,
        }


def check_solver_interval(schedulers, position, inputs, video, audio):
    """Verify both native scheduler cursors and the packed row timesteps."""
    expected_cursor = position if position else None
    timesteps = inputs.get("timestep")
    row_map = inputs.get("timestep_indices")
    if (not isinstance(timesteps, torch.Tensor) or timesteps.ndim != 1
            or not isinstance(row_map, torch.Tensor) or row_map.ndim != 1
            or row_map.dtype != torch.long or row_map.device != timesteps.device):
        raise RuntimeError("H3 AV requires the native packed row timestep map")
    for modality, scheduler, latent in zip(("video", "audio"), schedulers, (video, audio), strict=True):
        rows = inputs.get(f"{modality}_indices")
        if (scheduler.step_index != expected_cursor or scheduler.begin_index is not None
                or not isinstance(rows, torch.Tensor) or rows.shape != (latent.shape[1],)
                or rows.dtype != torch.long or rows.device != timesteps.device):
            raise RuntimeError("H3 AV lost native solver interval alignment")
        expected = scheduler.timesteps[position].to(device=timesteps.device)
        if not torch.all(timesteps[row_map[rows]] == expected):
            raise RuntimeError("H3 AV packed rows do not match the current modality timestep")
