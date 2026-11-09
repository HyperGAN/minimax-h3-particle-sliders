"""H3-owned bindings to the exact shared response lifecycle and HG/HNDL builder.

Native full-width training remains refused by official ParticleGAN release/auto.
No GAN loss, regularizer, optimizer or training loop is implemented here.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import re

import torch
from torch import nn

from .minimax_h3_backend import LORA_ATTN_CLASS, LORA_LINEAR_NAMES, PackedLayout
from .pika_h3_flex import (
    PikaH3FlexBackend, MODEL_ID, MODEL_REVISION, RELEASE, request_frames,
    prepared_frame_count, _frame_details, _attempt_cleanups, _cleanup_preserving_primary, _raise_cleanup_errors,
)

ROOT = Path(__file__).with_name("data")
LOCK = ROOT / "pika-h3-flex-session.json"
NAME = "pika-h3-flex"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_shared_runtime():
    """Check an installed reviewed core; never edit sys.path or install code."""
    import particle_sliders
    lock = json.loads(LOCK.read_text())
    try:
        core_version = version("particle-sliders-core")
    except PackageNotFoundError as error:
        raise RuntimeError("H3 shared sessions require installed particle-sliders-core distribution metadata") from error
    if core_version != lock["core_version"]:
        raise RuntimeError("H3 shared sessions require the separately installed reviewed core " + lock["core_revision"])
    root = Path(particle_sliders.__file__).resolve().parent
    actual = {str(path.relative_to(root)): _sha(path) for path in root.rglob("*")
              if path.is_file() and "__pycache__" not in path.relative_to(root).parts
              and path.suffix != ".pyc"}
    if actual != lock["core_sources"]:
        raise RuntimeError("H3 shared core source closure differs; use its original reviewed cohort " + lock["core_revision"])
    return lock


def _builder():
    require_shared_runtime()
    # Reuse the common verified builder gate, currently owned by this shared
    # module. It verifies HG build provenance and the complete HNDL source set.
    from particle_sliders.fibo_hndl import verified_builder
    return verified_builder()


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


class HNDLH3FlexBackend(PikaH3FlexBackend):
    def _build_slider(self, *, rank, alpha):
        return HNDLSlider(self.transformer, rank=rank, alpha=alpha)


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
                                                pg_source_profile=PG_SOURCE_PROFILE, penalty_input_domain="auto")
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
                    native_boundary_sources={name: _sha(Path(__file__).with_name(name))
                                             for name in ("pika_h3_flex.py", "minimax_h3_backend.py")})

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


def bind(runtime, pairs, *, options, model_identity, shape):
    """Construct the real shared game; this contains no alternate objective."""
    from particle_sliders.pg09 import ConditionalPG09
    from particle_sliders.lifecycle import Binding
    adapter = None
    closed = False
    def close():
        nonlocal closed
        if not closed:
            try:
                actions = [("HNDL forwards", runtime.network.close)]
                if adapter is not None:
                    actions.insert(0, ("checkpoint handles", adapter.close))
                _attempt_cleanups(*actions)
            except BaseException as error:
                error.retry_cleanup = close
                raise
            closed = True
    try:
        adapter = H3ResponseAdapter(runtime, shape, model_identity=model_identity,
                                    checkpointing=options.get("checkpointing", False))
        rows = tuple(adapter.prepare(pair) for pair in pairs)
        conditions = torch.eye(len(rows), device=runtime.device)
        critic = HNDLCritic(shape[1], len(rows)).to(runtime.device)
        game = ConditionalPG09(adapter, rows, critic, conditions=conditions, particles=None,
            response_domain="continuous_vector", preset=options.get("preset", "bispf_optimized"),
            batch_size=options.get("batch", 2), total_steps=options.get("steps", 4), seed=options.get("seed", 7))
        return Binding(game, close, observation=lambda: observe_native(runtime))
    except BaseException as error:
        _cleanup_preserving_primary(error, close)
        raise


def native_geometry(options):
    requested = request_frames(options.get("duration", 5))
    temporal, packed = RELEASE["temporal_contract"], RELEASE["packed_geometry"]
    frames = prepared_frame_count(requested)
    height, width = packed["canvases"][options.get("resolution", "480p")]
    latent_frames = ((frames - temporal["latents_per_chunk"]) // temporal["frames_per_chunk"]
                     * temporal["latents_per_chunk"] + temporal["latent_frame_bias"])
    video_rows = latent_frames * (height // packed["row_spatial_multiple"]) * (width // packed["row_spatial_multiple"])
    audio_rows = round(frames / temporal["fps"] * temporal["audio_latents_per_second"]) * temporal["audio_channels"]
    return dict(frame_metadata=_frame_details(requested, frames), height=height, width=width,
                video_rows=video_rows, audio_rows=audio_rows,
                response_shape=(1, video_rows * packed["video_features"] + audio_rows * packed["audio_features"]))


def native_response_shape(options):
    return native_geometry(options)["response_shape"]


def validate(options):
    allowed = {"snapshot", "rows", "device", "duration", "resolution", "steps", "batch", "seed", "ema_decay", "preset", "checkpointing", "base_precision"}
    if set(options) - allowed:
        raise ValueError("unknown H3 session options; shape/profile overrides and fixture substitutions are forbidden")
    if not isinstance(options.get("snapshot"), str) or not options["snapshot"]:
        raise ValueError("H3 requires a locally staged pinned snapshot")
    if options.get("resolution", "480p") not in ("480p", "768p"):
        raise ValueError("unsupported H3 native resolution")
    native_response_shape(options)
    if type(options.get("checkpointing", False)) is not bool:
        raise ValueError("checkpointing must be boolean")
    if options.get("base_precision", "bfloat16") != "bfloat16":
        raise ValueError("native H3 base precision is bfloat16 with released float32 islands")
    if options.get("preset", "bispf_optimized") not in ("bispf", "bispf_optimized"):
        raise ValueError("unsupported official PG preset")
    device = options.get("device", "cpu")
    if not isinstance(device, str) or (device != "cpu" and not re.fullmatch(r"cuda:[0-9]+", device)):
        raise ValueError("device must be cpu or an explicit cuda index")
    rows = options.get("rows")
    if not isinstance(rows, list) or not 2 <= len(rows) <= 128:
        raise ValueError("2..128 explicit positive/neutral prompt rows required")
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {"neutral", "positive"}
                or any(not isinstance(v, str) or not v.strip() or not 1 <= len(v) <= 8000 for v in row.values())
                or row["neutral"] == row["positive"]):
            raise ValueError("distinct complete positive/neutral H3 prompts required")


def prepare_native(options):
    torch.manual_seed(options.get("seed", 7))
    runtime = HNDLH3FlexBackend(options["snapshot"], device=options.get("device", "cpu"))
    try:
        pairs = []
        for index, row in enumerate(options["rows"]):
            kwargs = dict(duration=options.get("duration", 5), resolution=options.get("resolution", "480p"), seed=options.get("seed", 7) + index)
            pairs.append(H3Pair(runtime.prepare_t2va(row["neutral"], **kwargs), runtime.prepare_t2va(row["positive"], **kwargs)))
        geometry = native_geometry(options)
        if any(packed.frame_metadata != geometry["frame_metadata"] for pair in pairs for packed in (pair.neutral, pair.positive)):
            raise ValueError("native prepared frame metadata differs from the declared model identity")
        identity = dict(model=MODEL_ID, revision=MODEL_REVISION, published_files=RELEASE["local_snapshot_files"],
                        adapter_sha256=runtime.pipe._pika_verified_adapter_sha256, model_quality_qualified=False,
                        frame_metadata=geometry["frame_metadata"], packed_geometry=geometry)
        return bind(runtime, pairs, options=options, model_identity=identity, shape=geometry["response_shape"])
    except BaseException as error:
        _cleanup_preserving_primary(error, runtime.network.close)
        raise


def response_backend():
    require_shared_runtime()
    from particle_sliders.response import ResponseBackend, ResponseCapabilities
    from particle_sliders.pg09 import PG_VERSION, PG_SOURCE_PROFILE
    def capabilities(options):
        return ResponseCapabilities("adapter_gan", "video_audio", "native_packed_av_velocity", "continuous_vector",
            native_response_shape(options), PG_VERSION, PG_SOURCE_PROFILE, base_precisions=("bfloat16",),
            qualification="H3 binding source only; official wide-vector gate refuses native execution; pretrained/AV/GPU/quant unqualified")
    return ResponseBackend(NAME, capabilities({}), prepare_native, validate, configure_capabilities=capabilities)


def open_session(config, *, resume=None, artifact=None, operation="train"):
    """Delegate packet validation, session ownership and restore to the shared core."""
    from particle_sliders.lifecycle import _open_registered
    from particle_sliders.response import TRAINING, INFERENCE
    backend = response_backend()
    return _open_registered(config, resume=resume, artifact=artifact, operation=operation, request={},
        resolve=lambda name: backend, training_schema=TRAINING, inference_schema=INFERENCE)
