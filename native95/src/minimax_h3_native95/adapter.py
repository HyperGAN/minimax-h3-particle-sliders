"""H3-owned original HNDL native AV response, bound to current continuous PG095.
Adapted from the repository's immutable pika_h3_flex_session.py (MIT).
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from .packing import LORA_ATTN_CLASS,LORA_LINEAR_NAMES,PackedLayout
from .native import _cleanup_preserving_primary,_raise_cleanup_errors
from .builder import checked_builder as _builder
ROOT=Path(__file__).with_name("data")
def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class HNDLProjection(nn.Module):
    """A native attention hook whose two learned projections execute in HNDL."""
    def __init__(self, name, host, rank, alpha, build, identity):
        super().__init__()
        self.lora_name, self.multiplier, self.scale = name, 0.0, float(alpha) / rank
        source = (ROOT / "pika-h3-flex-slider.hndl").read_text()
        self.graph = build(source, input_shape={"value": ["B", host.in_features]},
                           output_shape=["B", host.out_features],
                           parameters={"rank": rank, "output_width": host.out_features})
        self.graph.to(device=host.weight.device, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.graph["down"].weight, a=5 ** 0.5)
        nn.init.zeros_(self.graph["up"].weight)
        self._identity = dict(identity, h3_projection_sha256=_sha(ROOT / "pika-h3-flex-slider.hndl"))
        self.org_forward = host.forward
        object.__setattr__(self, "_host", host)
        self._had_forward, self._previous_forward = "forward" in vars(host), vars(host).get("forward")
        self._closed = False
        host.forward = self.forward
        self._installed_forward = host.forward

    @property
    def lora_down(self):
        return self.graph["down"]

    @property
    def lora_up(self):
        return self.graph["up"]

    def extra_repr(self):
        return json.dumps(self._identity, sort_keys=True)

    def forward(self, value):
        base = self.org_forward(value)
        if self.multiplier == 0:
            return base
        flat = value.to(self.lora_down.weight).reshape(-1, self.lora_down.in_features)
        delta = self.graph(value=flat).reshape(*value.shape[:-1], self.lora_up.out_features)
        return base + delta.to(base) * (self.multiplier * self.scale)

    def close(self):
        if self._closed:
            return
        if self._host.forward is not self._installed_forward:
            raise RuntimeError("H3 HNDL forward ownership changed; cleanup remains pending")
        if self._had_forward:
            self._host.forward = self._previous_forward
        else:
            vars(self._host).pop("forward", None)
        self._closed = True


class HNDLSlider(nn.Module):
    """Adapter-only H3 arm; no invented routed particle bank."""
    def __init__(self, transformer, rank=8, alpha=8):
        super().__init__()
        build, identity = _builder()
        self.loras = []
        try:
            for name, module in tuple(transformer.named_modules()):
                if module.__class__.__name__ != LORA_ATTN_CLASS:
                    continue
                for child in LORA_LINEAR_NAMES:
                    host = module.get_submodule(child)
                    if not isinstance(getattr(host, "base_layer", host), nn.Linear):
                        raise ValueError("H3 HNDL requires dense native attention hosts")
                    key = ("lora_h3-" + name + "-" + child).replace(".", "-")
                    branch = HNDLProjection(key, host, rank, alpha, build, identity)
                    self.add_module(key, branch)
                    self.loras.append(branch)
        except BaseException as error:
            _cleanup_preserving_primary(error, self.close)
            raise

    def close(self):
        # Keep unresolved handles; do not restore unrelated owners after failure.
        errors = []
        for index in range(len(self.loras) - 1, -1, -1):
            try:
                self.loras[index].close()
            except BaseException as error:
                errors.append((self.loras[index].lora_name, error))
            else:
                branch = self.loras[index]
                if self._modules.get(branch.lora_name) is branch:
                    del self._modules[branch.lora_name]
                del self.loras[index]
        _raise_cleanup_errors(errors)


class HNDLCritic(nn.Module):
    score_bound = 0

    def __init__(self, response_width, condition_width):
        super().__init__()
        build, identity = _builder()
        source = (ROOT / "pika-h3-flex-critic.hndl").read_text()
        self.graph = build(source, input_shape={"response": ["B", response_width], "condition": ["B", condition_width]},
                           output_shape=["B", 1], parameters={"hidden": 16})
        self._identity = dict(identity, h3_critic_sha256=_sha(ROOT / "pika-h3-flex-critic.hndl"))

    def extra_repr(self):
        return json.dumps(self._identity, sort_keys=True)

    def forward(self, response, condition):
        return self.graph(response=response, condition=condition).squeeze(-1)


@dataclass(frozen=True)
class H3Pair:
    neutral: PackedLayout
    positive: PackedLayout
    scale: float = 1.0


@dataclass(frozen=True)
class PreparedPair:
    row: H3Pair
    target: torch.Tensor
    neutral: torch.Tensor


class H3ResponseAdapter:
    def __init__(self, runtime, shape, *, model_identity, checkpointing=False):
        from particle_sliders.pg09 import validate_pg09_domain, PG_SOURCE_PROFILE
        from particle_sliders.pg09_backends import AdapterParameters, checkpoint_with_adapter_scale
        self.domain_gate = validate_pg09_domain(shape, response_domain="continuous_vector",
                                                pg_source_profile=PG_SOURCE_PROFILE, penalty_input_domain="continuous")
        self.runtime, self.response_shape = runtime, tuple(shape)
        self.model_identity = deepcopy(model_identity)
        self.network = AdapterParameters(runtime.network.named_parameters(), runtime.scaled)
        self._handles = []
        if any(getattr(m, "gradient_checkpointing", False) for m in runtime.transformer.modules()):
            raise ValueError("native checkpointing must be off; the shared capture wrapper owns recomputation")
        try:
            if checkpointing:
                modules = tuple(runtime.transformer.transformer_blocks) + tuple(runtime.transformer.token_refiner.refiner_blocks)
                for module in modules:
                    self._handles.append(checkpoint_with_adapter_scale(module, self.network))
        except BaseException as error:
            _cleanup_preserving_primary(error, self.close)
            raise

    def _response(self, packed, scale):
        with self.network.scaled(scale):
            output = self.runtime.velocity(packed)
        value = torch.cat((output.sample.float().reshape(1, -1), output.audio_sample.float().reshape(1, -1)), dim=1)
        if tuple(value.shape) != self.response_shape or not torch.isfinite(value).all():
            raise ValueError("H3 must preserve the complete finite native video-plus-audio velocity vector")
        return value

    def prepare(self, pair):
        pair = deepcopy(pair)
        if pair.scale != 1.0:
            raise ValueError("H3 paired teacher training is unipolar at slider+1")
        for name in ("hidden_states", "audio_hidden_states", "timestep"):
            if not torch.equal(getattr(pair.neutral, name), getattr(pair.positive, name)):
                raise ValueError("H3 positive/neutral teachers must share native noise and times")
        with torch.no_grad():
            return PreparedPair(pair, self._response(pair.positive, 0).detach().clone(),
                                self._response(pair.neutral, 0).detach().clone())

    def generated(self, prepared):
        return self._response(prepared.row.neutral, prepared.row.scale)

    @torch.no_grad()
    def sample(self, prepared, *, scale=1.0):
        return self._response(prepared.row.neutral, scale).detach().clone()

    def metadata(self):
        return dict(protocol="pika-h3-flex-packed-av-velocity-v1", model_identity=self.model_identity,
                    coordinates="complete packed video velocity followed by audio velocity; no projection or field reshape",
                    teacher="fixed base+Pika positive at slider0", student="neutral at slider+1",
                    generator_arm="adapter_gan", adapter_names=list(self.network.names),
                    pg_domain_gate=self.domain_gate, native_quality_qualified=False,
                    frame_metadata=deepcopy(self.model_identity.get("frame_metadata")),
                    fresh_initialization=deepcopy(getattr(self,"fresh_initialization",None)),
                    native_boundary_sources={name: _sha(Path(__file__).with_name(name))
                                             for name in ("native.py", "packing.py")})

    def close(self):
        errors = []
        for index in range(len(self._handles) - 1, -1, -1):
            try:
                self._handles[index].close()
            except BaseException as error:
                errors.append((f"checkpoint handle {index}", error))
            else:
                del self._handles[index]
        _raise_cleanup_errors(errors)


@contextmanager
def observe_native(runtime):
    """Preserve modes and native buffers; never copy pretrained parameters."""
    modules = tuple(runtime.transformer.modules())
    modes = [(m, m.training) for m in modules]
    buffers = [(m, dict(m._buffers), set(m._non_persistent_buffers_set)) for m in modules]
    values = {id(v): (v, v.detach().clone()) for _, registered, _ in buffers for v in registered.values() if v is not None}
    try:
        yield
    finally:
        with torch.no_grad():
            for value, previous in values.values():
                if (value.shape, value.dtype, value.device) == (previous.shape, previous.dtype, previous.device):
                    value.copy_(previous)
                else:
                    value.data = previous.clone()
            for module, registered, nonpersistent in buffers:
                module._buffers.clear()
                module._buffers.update(registered)
                module._non_persistent_buffers_set.clear()
                module._non_persistent_buffers_set.update(nonpersistent)
            for module, mode in modes:
                module.training = mode
