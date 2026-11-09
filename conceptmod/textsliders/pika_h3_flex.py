"""Opt-in, local-only Pika H3 Flex loader and adapter qualification.

This is a native backend boundary, not a new training objective. The historical
H3 UNI trainer remains separate. No Hub download runs here, including on import.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import warnings

import torch

from .minimax_h3_backend import AttnLoRANetwork, PackedLayout

PROVENANCE = Path(__file__).with_name("data") / "pika-h3-flex-release.json"
RELEASE = json.loads(PROVENANCE.read_text())
MODEL_ID = RELEASE["model_id"]
MODEL_REVISION = RELEASE["model_revision"]
DIFFUSERS_REVISION = RELEASE["diffusers_revision"]
SCHEDULE = (1000, 750, 500, 250, 100, 40, 0)
PIKA_ADAPTER = "pika_release"


class H3FlexUnavailable(RuntimeError):
    """The native runtime or a requested operation has not been qualified."""


def _cleanup_note(error, note):
    # Standalone conceptmod also runs outside the core's Python >=3.11 lane.
    add_note = getattr(error, "add_note", None)
    if callable(add_note) and note not in getattr(error, "__notes__", ()):
        add_note(note)


def _raise_cleanup_errors(errors):
    if errors:
        first = errors[0][1]
        for label, error in errors:
            _cleanup_note(first, f"{label} cleanup failed: {error!r}")
        first.cleanup_failures = tuple(errors)
        raise first


def _attempt_cleanups(*actions):
    errors = []
    for label, action in actions:
        try:
            action()
        except BaseException as error:
            errors.append((label, error))
    _raise_cleanup_errors(errors)


def _cleanup_preserving_primary(primary, cleanup):
    try:
        cleanup()
    except BaseException as error:
        _cleanup_note(primary, f"Cleanup also failed: {error!r}; retry_cleanup retains the pending owner.")
        primary.cleanup_error = error
        pending = getattr(primary, "pending_cleanups", ()) + (cleanup,)
        primary.pending_cleanups = pending

        def retry_cleanup():
            _attempt_cleanups(*((f"pending owner {index}", action)
                                for index, action in enumerate(pending)))

        primary.retry_cleanup = retry_cleanup


@dataclass
class PreparedH3Flex(PackedLayout):
    """Native packed inputs plus duration metadata, never a transformer kwarg."""
    frame_metadata: dict


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_base_files(root):
    """Fail closed against published hashes of the official Pika packaging.

    LFS objects publish SHA-256; ordinary Git files publish Git blob SHA-1
    (including its blob header). No network or weight download is performed.
    """
    files = RELEASE.get("local_snapshot_files", {}).get("files")
    if not files or not any(path.endswith(".safetensors") for path in files):
        raise H3FlexUnavailable("published base-file provenance is missing")
    # An unlisted monolithic checkpoint or tokenizer sidecar could take
    # precedence over a verified index; reject alternative component files.
    components = {name.split("/")[0] for name in files if "/" in name}
    for component in components:
        for path in (root / component).rglob("*"):
            if path.is_file() and str(path.relative_to(root)) not in files:
                raise H3FlexUnavailable("unlisted local base component file: " + str(path.relative_to(root)))
    for name, expected in files.items():
        path = root / name
        if not path.is_file() or path.stat().st_size != expected["size"]:
            raise H3FlexUnavailable("missing or wrong-size pinned base file: " + name)
        if expected["algorithm"] == "sha256":
            actual = _sha256(path)
        elif expected["algorithm"] == "git-blob-sha1":
            digest = hashlib.sha1()
            digest.update(f"blob {expected['size']}\0".encode())
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual = digest.hexdigest()
        else:
            raise H3FlexUnavailable("unqualified base hash algorithm: " + name)
        if actual != expected["digest"]:
            raise H3FlexUnavailable("pinned base file checksum differs: " + name)


def request_frames(duration):
    temporal = RELEASE["temporal_contract"]
    if (type(duration) is not int
            or not temporal["min_requested_frames"] <= duration * temporal["fps"] <= temporal["max_requested_frames"]):
        raise ValueError("integer duration from 5 to 15 seconds required")
    return duration * temporal["fps"]


def prepared_frame_count(requested_frames):
    temporal = RELEASE["temporal_contract"]
    if (type(requested_frames) is not int
            or not temporal["min_requested_frames"] <= requested_frames <= temporal["max_requested_frames"]):
        raise ValueError("native requested frame count must be an integer from 120 to 360")
    frames = requested_frames
    while frames % temporal["frames_per_chunk"] != temporal["latents_per_chunk"]:
        frames += 1
    return frames


def validate_aligned_frames(num_frames, *, requested_frames=None):
    """Validate a prepared frame count, not a pre-alignment native request."""
    temporal = RELEASE["temporal_contract"]
    if (type(num_frames) is not int or not temporal["min_aligned_frames"] <= num_frames <= temporal["max_aligned_frames"]
            or num_frames % temporal["frames_per_chunk"] != temporal["latents_per_chunk"]):
        raise H3FlexUnavailable("prepared H3 frames must follow the released 17*n+5 grid, 124–362")
    if requested_frames is not None:
        expected = prepared_frame_count(requested_frames)
        if num_frames != expected:
            raise H3FlexUnavailable("prepared H3 frame count differs from the native request alignment")


def _frame_details(requested_frames, prepared_frames):
    temporal = RELEASE["temporal_contract"]
    if (type(requested_frames) is not int
            or not temporal["min_requested_frames"] <= requested_frames <= temporal["max_requested_frames"]):
        raise ValueError("native requested frame count must be an integer from 120 to 360")
    validate_aligned_frames(prepared_frames, requested_frames=requested_frames)
    return {
        "requested_frame_count": requested_frames,
        "prepared_frame_count": prepared_frames,
        "requested_duration_seconds": requested_frames / temporal["fps"],
        "actual_duration_seconds": prepared_frames / temporal["fps"],
        "fps": temporal["fps"],
        "alignment_changed": prepared_frames != requested_frames,
    }


def frame_metadata(requested_frames, prepared_frames):
    result = _frame_details(requested_frames, prepared_frames)
    if result["alignment_changed"]:
        warnings.warn(
            f"Pika H3 Flex aligned {requested_frames} requested frames ({result['requested_duration_seconds']:g}s) "
            f"to {prepared_frames} frames ({result['actual_duration_seconds']:.6f}s); duration changed.",
            UserWarning, stacklevel=2,
        )
    return result


def validate_snapshot(snapshot, *, workflow="t2va"):
    """Validate release metadata and all pinned local base/adapter bytes."""
    if workflow != "t2va":
        raise H3FlexUnavailable("this slider boundary qualifies t2va only; FL2VA/reference conditioning is separate")
    root = Path(snapshot).resolve(strict=True)
    manifest_path = root / "fl2va/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = RELEASE["fl2va"]
    for key in ("release_id", "rank", "alpha", "trunk", "numerics_policy", "base", "sampling", "target_modules", "sha256"):
        if manifest.get(key) != expected[key]:
            raise H3FlexUnavailable("Pika FL2VA release manifest differs: " + key)
    index = json.loads((root / "modular_model_index.json").read_text())
    if index.get("_class_name") != "PikaH3FlexModularPipeline":
        raise H3FlexUnavailable("expected the Pika modular pipeline index, not the base H3 index")
    for name, shift in (("scheduler", 12.0), ("audio_scheduler", 3.0)):
        config = json.loads((root / name / "scheduler_config.json").read_text())
        if (config.get("_class_name") != "MiniMaxH3Scheduler" or config.get("shift") != shift
                or config.get("shift_provided_sigmas") is not True or config.get("stochastic_sampling") is not True):
            raise H3FlexUnavailable("Pika scheduler contract differs: " + name)
    for component in ("transformer", "text_encoder", "tokenizer", "processor", "vae", "audio_vae"):
        if not (root / component).is_dir():
            raise H3FlexUnavailable("missing locally staged component: " + component)
    if _sha256(root / "fl2va/adapter.safetensors") != expected["sha256"]:
        raise H3FlexUnavailable("Pika adapter checksum differs from the pinned release")
    validate_base_files(root)
    return root


def _verified_adapter(root):
    """Parse the same immutable byte payload whose release hash was checked."""
    from safetensors.torch import load
    payload = (root / "fl2va/adapter.safetensors").read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != RELEASE["fl2va"]["sha256"]:
        raise H3FlexUnavailable("Pika adapter checksum changed before loading")
    return load(payload), digest


def require_native_runtime():
    """Check the exact reviewed native contract sources, not a dev version string."""
    import diffusers

    root = Path(diffusers.__file__).resolve().parent
    for path, expected in RELEASE["diffusers_contract_sources"].items():
        file = root / path
        if not file.is_file() or _sha256(file) != expected:
            raise H3FlexUnavailable("install the reviewed Diffusers commit " + DIFFUSERS_REVISION + "; source differs: " + path)


def admit_shared_gan(response_shape):
    """Use the installed shared core's exact native continuous-domain gate.

    A packed audio/video velocity is a vector. Do not manufacture an image,
    truncate the response, or opt into a categorical profile to admit it.
    """
    from particle_sliders.pg09 import PG_SOURCE_PROFILE, validate_pg09_domain

    return validate_pg09_domain(tuple(response_shape), response_domain="continuous_vector",
                                pg_source_profile=PG_SOURCE_PROFILE, penalty_input_domain="auto")


def load_pipeline(snapshot, *, workflow="t2va", device="cpu"):
    root = validate_snapshot(snapshot, workflow=workflow)
    require_native_runtime()
    adapter_state, adapter_digest = _verified_adapter(root)
    from diffusers import ModularPipeline

    pipe = ModularPipeline.from_pretrained(str(root), workflow=workflow, local_files_only=True)
    # Override every component's source so an incomplete local snapshot can
    # never fall through to the unpinned Hub identifiers in the shipped index.
    pipe.load_components(pretrained_model_name_or_path=str(root), local_files_only=True,
                         dtype={"vae": torch.float32, "audio_vae": torch.float32, "default": torch.bfloat16})
    missing = [name for name in ("transformer", "text_encoder", "tokenizer", "processor", "vae", "audio_vae", "scheduler", "audio_scheduler")
               if getattr(pipe, name, None) is None]
    if missing:
        raise H3FlexUnavailable("native component loading failed: " + ", ".join(missing))
    pipe.load_lora_weights(adapter_state, adapter_name=PIKA_ADAPTER)
    pipe._pika_verified_adapter_sha256 = adapter_digest
    pipe.set_adapters([PIKA_ADAPTER], adapter_weights=[1.0])
    pipe.register_to_config(denoising_step_list=list(SCHEDULE))
    for module in pipe.components.values():
        if isinstance(module, torch.nn.Module):
            module.requires_grad_(False).eval()
    # Device errors must propagate; an ignored OOM is not a loaded runtime.
    pipe.to(device)
    return pipe


class PikaH3FlexBackend:
    """Frozen base plus fixed Pika adapter, with a separate float32 slider.

    Scale zero means Pika H3 Flex, including its release adapter at strength 1.
    The explicit context must enclose backward when gradient checkpointing is
    active. Native render and prompt preparation are deliberately no-grad.
    """
    def __init__(self, snapshot, *, device="cpu", rank=8, alpha=8.0):
        if type(rank) is not int or rank < 1 or not math.isfinite(float(alpha)) or float(alpha) <= 0:
            raise ValueError("positive integer rank and finite positive alpha required")
        self.pipe = load_pipeline(snapshot, device=device)
        self.transformer = self.pipe.transformer
        self.device = torch.device(device)
        self._attach_slider(rank=rank, alpha=alpha)

    def _attach_slider(self, *, rank, alpha):
        # PEFT operations can toggle requires_grad. Freeze before creating the
        # independent float32 slider, and after any permitted adapter roundtrip.
        modules = {id(m): m for root in self.pipe.components.values() if isinstance(root, torch.nn.Module) for m in root.modules()}
        parameters = {id(p): p for root in self.pipe.components.values() if isinstance(root, torch.nn.Module) for p in root.parameters()}
        self._attach_state = ([(m, m.training) for m in modules.values()], [(p, p.requires_grad) for p in parameters.values()])
        self._guard_handles = []
        try:
            self._freeze_base()
            self.network = self._build_slider(rank=rank, alpha=alpha)
            self.network.float()
            if not self.network.loras:
                raise H3FlexUnavailable("no native H3 attention slider targets found")
            for layer in self.network.loras:
                layer.multiplier = 0.0
            self._release_layers = {name: layer for name, layer in self.transformer.named_modules() if hasattr(layer, "lora_A")}
            if not self._release_layers:
                raise H3FlexUnavailable("the fixed Pika PEFT adapter is missing")
            self._install_lifecycle_guards()
            self._check_fixed_pika()
        except BaseException as error:
            _cleanup_preserving_primary(error, self._rollback_attachment)
            raise
        self._attach_state = None

    def _rollback_attachment(self):
        actions = [("lifecycle guards", self._restore_guards)]
        if hasattr(self, "network"):
            actions.append(("slider forwards", self.network.close))
        _attempt_cleanups(*actions)
        if self._attach_state is not None:
            modes, flags = self._attach_state
            for module, mode in modes:
                module.training = mode
            for parameter, flag in flags:
                parameter.requires_grad_(flag)
            self._attach_state = None

    def _guard_method(self, owner, name, replacement):
        had_value, previous = name in vars(owner), vars(owner).get(name)
        setattr(owner, name, replacement)
        self._guard_handles.append((owner, name, had_value, previous, replacement))

    def _restore_guards(self):
        errors = []
        for index in range(len(self._guard_handles) - 1, -1, -1):
            owner, name, had_value, previous, installed = self._guard_handles[index]
            try:
                if getattr(owner, name) is not installed:
                    raise RuntimeError("H3 lifecycle guard ownership changed: " + name)
                if had_value:
                    setattr(owner, name, previous)
                else:
                    vars(owner).pop(name, None)
            except BaseException as error:
                errors.append((name, error))
            else:
                del self._guard_handles[index]
        _raise_cleanup_errors(errors)

    def _build_slider(self, *, rank, alpha):
        """The standalone native boundary retains its original LoRA leaves."""
        return AttnLoRANetwork(self.transformer, rank=rank, alpha=alpha, up_init_std=0.0)

    def _freeze_base(self):
        for module in self.pipe.components.values():
            if isinstance(module, torch.nn.Module):
                module.requires_grad_(False).eval()

    def _check_fixed_pika(self):
        if self.pipe.transformer is not self.transformer:
            raise H3FlexUnavailable("fixed Pika transformer was replaced")
        actual = {name: layer for name, layer in self.transformer.named_modules() if hasattr(layer, "lora_A")}
        if actual.keys() != self._release_layers.keys() or any(actual[name] is not layer for name, layer in self._release_layers.items()):
            raise H3FlexUnavailable("fixed Pika adapter topology changed")
        for layer in actual.values():
            if (list(layer.active_adapters) != [PIKA_ADAPTER] or set(layer.lora_A) != {PIKA_ADAPTER}
                    or layer.disable_adapters or layer.merged_adapters or layer.scaling.get(PIKA_ADAPTER) != 1.0):
                raise H3FlexUnavailable("fixed Pika adapter must remain unmerged, enabled and scaled at 1")
        if any(p.requires_grad for module in self.pipe.components.values() if isinstance(module, torch.nn.Module)
               for p in module.parameters()):
            raise H3FlexUnavailable("base or Pika release parameters are trainable")

    def _install_lifecycle_guards(self):
        def blocked(*_args, **_kwargs):
            raise H3FlexUnavailable("adapter mutation is unsupported while the H3 Flex slider is attached")

        # Public high-level destructive operations are rejected before mutation.
        # Directly editing native internals is unsupported; the next operation
        # also checks topology, active adapters, merge state, scale and freezing.
        for owner in (self.pipe, self.transformer):
            for name in ("fuse_lora", "unfuse_lora", "unload_lora", "unload_lora_weights", "disable_lora",
                         "enable_lora", "disable_adapters", "enable_adapters", "delete_adapters", "add_adapter",
                         "load_lora_weights", "load_lora_adapter"):
                if hasattr(owner, name):
                    self._guard_method(owner, name, blocked)

        def guard_setter(original):
            def set_fixed(adapter_names, weights=None, **kwargs):
                names = [adapter_names] if isinstance(adapter_names, str) else adapter_names
                if kwargs:
                    if set(kwargs) != {"adapter_weights"} or weights is not None:
                        raise H3FlexUnavailable("only the fixed Pika adapter at unit weight may be selected")
                    weights = kwargs["adapter_weights"]
                if names != [PIKA_ADAPTER] or weights not in (None, 1.0, [1.0], [None]):
                    raise H3FlexUnavailable("only the fixed Pika adapter at unit weight may be selected")
                try:
                    return original(adapter_names, weights)
                finally:
                    self._freeze_base()
                    self._check_fixed_pika()
            return set_fixed
        self._guard_method(self.pipe, "set_adapters", guard_setter(self.pipe.set_adapters))
        self._guard_method(self.transformer, "set_adapters", guard_setter(self.transformer.set_adapters))

    @contextmanager
    def scaled(self, strength):
        self._check_fixed_pika()
        strength = float(strength)
        if not math.isfinite(strength):
            raise ValueError("finite slider strength required")
        previous = [layer.multiplier for layer in self.network.loras]
        try:
            for layer in self.network.loras:
                layer.multiplier = strength
            yield
        finally:
            for layer, value in zip(self.network.loras, previous):
                layer.multiplier = value

    def trainable_parameters(self):
        return list(self.network.parameters())

    def prepare_t2va(self, prompt, *, duration=5, resolution="480p", seed=7):
        """Execute the actual native conditioner, packing and timestep blocks.

        No synthetic rows, hand-built rotary positions or generic Euler loop.
        This prepares the first of the six released evaluation points.
        """
        if not isinstance(prompt, str) or not 1 <= len(prompt) <= 8000:
            raise ValueError("prompt must contain 1–8000 characters")
        num_frames = request_frames(duration)
        self._check_fixed_pika()
        if resolution not in ("480p", "768p"):
            raise ValueError("native resolution must be 480p or 768p")
        from diffusers.modular_pipelines import SequentialPipelineBlocks

        names = ("text_encoder", "denoise.no_keyframe_anchors", "denoise.prepare_layout",
                 "denoise.prepare_latents", "denoise.set_timesteps")
        blocks = SequentialPipelineBlocks.from_blocks_dict({name: self.pipe.blocks.sub_blocks[name] for name in names})
        prep = blocks.init_pipeline()
        prep.update_components(**{name: component for name, component in self.pipe.components.items() if name in prep.components})
        prep.register_to_config(denoising_step_list=list(SCHEDULE))
        height, width = RELEASE["packed_geometry"]["canvases"][resolution]
        state = prep(prompt=prompt, height=height, width=width, num_frames=num_frames,
                     generator=torch.Generator(device="cpu").manual_seed(seed))
        metadata = frame_metadata(num_frames, state.get("num_frames"))
        fields = {name: state.get(name) for name in ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices")}
        times, indices = state.get("row_timestep_plan")[0]
        return PreparedH3Flex(frame_metadata=metadata, hidden_states=state.get("latents")[None],
                            audio_hidden_states=state.get("audio_latents")[None],
                            encoder_hidden_states=state.get("prompt_embeds"), timestep=times,
                            timestep_indices=indices, **fields)

    def velocity(self, packed):
        """Call the differentiable native transformer directly, outside the pipeline."""
        self._check_fixed_pika()
        return self.transformer(**{name: getattr(packed, name) for name in PackedLayout.__dataclass_fields__}, return_dict=True)

    @torch.no_grad()
    def render(self, prompt, *, strength=0.0, duration=5, resolution="480p", seed=7):
        num_frames = request_frames(duration)
        if resolution not in ("480p", "768p"):
            raise ValueError("native resolution must be 480p or 768p")
        prepared_frames = prepared_frame_count(num_frames)
        # Warn before starting generation, then verify the actual returned count.
        metadata = frame_metadata(num_frames, prepared_frames)
        height, width = RELEASE["packed_geometry"]["canvases"][resolution]
        with self.scaled(strength):
            # The native pipeline owns all six re-noising steps and both shifts.
            result = self.pipe(prompt=prompt, height=height, width=width, num_frames=num_frames,
                               generator=torch.Generator(device="cpu").manual_seed(seed),
                               output=["videos", "audio", "sampling_rate"])
        if not isinstance(result, dict) or not result.get("videos") or len(result["videos"][0]) != prepared_frames:
            raise H3FlexUnavailable("rendered frame count differs from the native request alignment")
        return {**result, "frame_metadata": metadata}


def gradient_probe(backend, packed):
    """One gradient-only qualification, no optimizer or model training.

    The scalar is a diagnostic; it is not a slider objective. autograd.grad
    avoids populating or clearing any existing parameter .grad buffers.
    """
    leaves = backend.trainable_parameters()
    if not leaves or any(not p.requires_grad or p.dtype != torch.float32 for p in leaves):
        raise H3FlexUnavailable("slider leaves must be nonempty trainable float32 tensors")
    if any(p.requires_grad for p in backend.transformer.parameters()):
        raise H3FlexUnavailable("base or Pika release parameters are trainable")
    with torch.enable_grad(), backend.scaled(1.0):
        output = backend.velocity(packed)
        scalar = output.sample.float().square().mean() + output.audio_sample.float().square().mean()
        gradients = torch.autograd.grad(scalar, leaves, allow_unused=False)
    norms = [float(g.detach().float().norm()) for g in gradients]
    if not all(math.isfinite(value) for value in norms) or not any(value > 0 for value in norms):
        raise H3FlexUnavailable("no finite nonzero native slider gradient")
    receipt = {"kind": "gradient-only diagnostic", "updated": False, "native_quality_validated": False,
               "slider_leaves": len(leaves), "nonzero_gradient_leaves": sum(v > 0 for v in norms),
               "gradient_norms": norms}
    if isinstance(packed, PreparedH3Flex):
        receipt["frame_metadata"] = dict(packed.frame_metadata)
    return receipt


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("provenance", "admit-gan", "probe"))
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prompt", default="A red fox walking through a snowy pine forest.")
    parser.add_argument("--response-width", type=int)
    args = parser.parse_args(argv)
    if args.operation == "provenance":
        print(json.dumps(RELEASE, indent=2))
    elif args.operation == "admit-gan":
        if args.response_width is None or args.response_width < 170:
            parser.error("declare the real packed response width (at least 170); no compressed toy response")
        print(json.dumps(admit_shared_gan((1, args.response_width)), indent=2))
    else:
        if args.snapshot is None:
            parser.error("probe requires a locally staged --snapshot")
        backend = PikaH3FlexBackend(args.snapshot, device=args.device)
        print(json.dumps(gradient_probe(backend, backend.prepare_t2va(args.prompt)), indent=2))


if __name__ == "__main__":
    main()
