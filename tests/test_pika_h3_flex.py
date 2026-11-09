"""Tiny/synthetic CPU fixtures only; no pretrained or quantized qualification."""
import copy
import hashlib
import json
from pathlib import Path
import types
import warnings

import pytest
import torch

from conceptmod.textsliders import pika_h3_flex as flex
from conceptmod.textsliders.minimax_h3_backend import AttnLoRANetwork, MiniMaxH3Backend, PackedLayout


def snapshot(tmp_path, monkeypatch):
    from safetensors.torch import save
    release = copy.deepcopy(flex.RELEASE)
    adapter_bytes = save({"fixture.lora_A.weight": torch.zeros(2, 2)})
    release["fl2va"]["sha256"] = hashlib.sha256(adapter_bytes).hexdigest()
    monkeypatch.setattr(flex, "RELEASE", release)
    for component in ("fl2va", "transformer", "text_encoder", "tokenizer", "processor", "vae", "audio_vae", "scheduler", "audio_scheduler"):
        (tmp_path / component).mkdir()
    (tmp_path / "fl2va/adapter.safetensors").write_bytes(adapter_bytes)
    (tmp_path / "fl2va/manifest.json").write_text(json.dumps(release["fl2va"]))
    (tmp_path / "modular_model_index.json").write_text(json.dumps({"_class_name": "PikaH3FlexModularPipeline"}))
    for name, shift in (("scheduler", 12.0), ("audio_scheduler", 3.0)):
        (tmp_path / name / "scheduler_config.json").write_text(json.dumps({"_class_name": "MiniMaxH3Scheduler", "shift": shift,
                                                                         "shift_provided_sigmas": True, "stochastic_sampling": True}))
    (tmp_path / "transformer/model.safetensors").write_bytes(b"base-v1")
    (tmp_path / "text_encoder/config.json").write_bytes(b"{}")
    files = {}
    for path in tmp_path.rglob("*"):
        if not path.is_file() or "fl2va" in path.parts:
            continue
        content = path.read_bytes()
        lfs = path.suffix == ".safetensors"
        digest = hashlib.sha256(content).hexdigest() if lfs else hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        files[str(path.relative_to(tmp_path))] = {"size": len(content), "algorithm": "sha256" if lfs else "git-blob-sha1", "digest": digest}
    release["local_snapshot_files"]["files"] = files
    return tmp_path


def test_release_gate_rejects_wrong_adapter_and_base(tmp_path, monkeypatch):
    root = snapshot(tmp_path, monkeypatch)
    assert flex.validate_snapshot(root) == root
    (root / "fl2va/adapter.safetensors").write_bytes(b"wrong")
    with pytest.raises(flex.H3FlexUnavailable, match="checksum"):
        flex.validate_snapshot(root)
    manifest = copy.deepcopy(flex.RELEASE["fl2va"])
    manifest["base"]["revision"] = "wrong"
    (root / "fl2va/manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(flex.H3FlexUnavailable, match="base"):
        flex.validate_snapshot(root)


@pytest.mark.parametrize("shift", [3.0, 1.0])
def test_release_gate_rejects_wrong_video_schedule(tmp_path, monkeypatch, shift):
    root = snapshot(tmp_path, monkeypatch)
    (root / "scheduler/scheduler_config.json").write_text(json.dumps({"_class_name": "MiniMaxH3Scheduler", "shift": shift}))
    with pytest.raises(flex.H3FlexUnavailable, match="scheduler"):
        flex.validate_snapshot(root)


def test_ref2va_not_silently_routed_to_fl2va():
    with pytest.raises(flex.H3FlexUnavailable, match="t2va only"):
        flex.validate_snapshot("does-not-exist", workflow="ref2va")


def test_base_bytes_and_missing_provenance_fail_closed(tmp_path, monkeypatch):
    root = snapshot(tmp_path, monkeypatch)
    (root / "transformer/model.safetensors").write_bytes(b"base-v2")
    with pytest.raises(flex.H3FlexUnavailable, match="base file checksum"):
        flex.validate_snapshot(root)
    (root / "transformer/model.safetensors").write_bytes(b"base-v1")
    (root / "text_encoder/config.json").write_bytes(b"[]")
    with pytest.raises(flex.H3FlexUnavailable, match="base file checksum"):
        flex.validate_snapshot(root)
    flex.RELEASE["local_snapshot_files"]["files"] = {}
    with pytest.raises(flex.H3FlexUnavailable, match="provenance is missing"):
        flex.validate_snapshot(root)


def test_unverified_alternative_checkpoint_is_rejected(tmp_path, monkeypatch):
    root = snapshot(tmp_path, monkeypatch)
    (root / "transformer/pytorch_model.bin").write_bytes(b"unverified")
    with pytest.raises(flex.H3FlexUnavailable, match="unlisted local base"):
        flex.validate_snapshot(root)


@pytest.mark.parametrize("count", [120, 121, 125, 129, 361, 363, 0, True, 124.0])
def test_prepared_temporal_grid_rejects_invalid_counts(count):
    # 121/125/129 are 4*k+1; that is not H3's prepared frame grid.
    with pytest.raises(flex.H3FlexUnavailable, match="17\\*n\\+5"):
        flex.validate_aligned_frames(count)


@pytest.mark.parametrize("duration", [0, 4, 16, 5.0, True])
def test_request_duration_rejects_invalid_input(duration):
    with pytest.raises(ValueError, match="integer duration"):
        flex.request_frames(duration)


def test_actual_native_temporal_alignment():
    pytest.importorskip("diffusers")
    from diffusers.modular_pipelines.pika_h3_flex.modular_pipeline import align_num_frames
    for duration, expected in ((5, 124), (10, 243), (15, 362)):
        assert align_num_frames(flex.request_frames(duration), 17, 5) == expected
        flex.validate_aligned_frames(expected, requested_frames=flex.request_frames(duration))
    with pytest.raises(flex.H3FlexUnavailable, match="request alignment"):
        flex.validate_aligned_frames(141, requested_frames=120)


@pytest.mark.parametrize("duration,prepared", [(5, 124), (8, 192), (10, 243), (15, 362)])
def test_frame_metadata_reports_duration_and_alignment_warning(duration, prepared):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metadata = flex.frame_metadata(flex.request_frames(duration), prepared)
    changed = prepared != duration * 24
    assert metadata == {
        "requested_frame_count": duration * 24,
        "prepared_frame_count": prepared,
        "requested_duration_seconds": float(duration),
        "actual_duration_seconds": prepared / 24,
        "fps": 24,
        "alignment_changed": changed,
    }
    assert len(caught) == int(changed)
    if changed:
        assert "duration changed" in str(caught[0].message)


def test_legacy_loader_cannot_silently_load_pika_as_base(tmp_path):
    from conceptmod.textsliders.minimax_h3_backend import _load_minimax_h3_modular, ArchitectureMismatch
    with pytest.raises(ArchitectureMismatch, match="fixed release adapter"):
        _load_minimax_h3_modular(flex.MODEL_ID)
    (tmp_path / "modular_model_index.json").write_text(json.dumps({"_class_name": "PikaH3FlexModularPipeline"}))
    with pytest.raises(ArchitectureMismatch, match="fixed release adapter"):
        _load_minimax_h3_modular(str(tmp_path))


def test_local_loader_keeps_fixed_pika_and_precision(tmp_path, monkeypatch):
    import sys
    root = snapshot(tmp_path, monkeypatch)
    calls = {}
    class Pipe:
        def __init__(self):
            for name in ("transformer", "text_encoder", "vae", "audio_vae"):
                setattr(self, name, torch.nn.Linear(2, 2))
            for name in ("tokenizer", "processor", "scheduler", "audio_scheduler"):
                setattr(self, name, object())
            self.components = {name: getattr(self, name) for name in ("transformer", "text_encoder", "vae", "audio_vae")}
        def load_components(self, **kwargs):
            calls["components"] = kwargs
            # A concurrent path replacement after capture cannot change the
            # verified in-memory adapter supplied to the actual loader.
            (root / "fl2va/adapter.safetensors").write_bytes(b"changed after capture")
        def load_lora_weights(self, path, **kwargs): calls["lora"] = (path, kwargs)
        def set_adapters(self, names, **kwargs): calls["adapters"] = (names, kwargs)
        def register_to_config(self, **kwargs): calls["config"] = kwargs
        def to(self, device): calls["device"] = device
    class ModularPipeline:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["pipeline"] = (path, kwargs)
            return Pipe()
    monkeypatch.setitem(sys.modules, "diffusers", types.SimpleNamespace(ModularPipeline=ModularPipeline))
    monkeypatch.setattr(flex, "require_native_runtime", lambda: None)
    pipe = flex.load_pipeline(root)
    assert calls["pipeline"] == (str(root), {"workflow": "t2va", "local_files_only": True})
    assert calls["components"]["pretrained_model_name_or_path"] == str(root)
    assert calls["components"]["local_files_only"] is True
    assert calls["components"]["dtype"]["vae"] == torch.float32
    assert calls["components"]["dtype"]["audio_vae"] == torch.float32
    assert calls["adapters"] == ([flex.PIKA_ADAPTER], {"adapter_weights": [1.0]})
    assert calls["config"]["denoising_step_list"] == list(flex.SCHEDULE)
    assert isinstance(calls["lora"][0], dict)
    assert torch.equal(calls["lora"][0]["fixture.lora_A.weight"], torch.zeros(2, 2))
    assert pipe._pika_verified_adapter_sha256 == flex.RELEASE["fl2va"]["sha256"]
    assert not any(p.requires_grad for module in pipe.components.values() for p in module.parameters())


def test_adapter_change_between_validation_and_capture_is_rejected(tmp_path, monkeypatch):
    root = snapshot(tmp_path, monkeypatch)
    monkeypatch.setattr(flex, "require_native_runtime", lambda: (root / "fl2va/adapter.safetensors").write_bytes(b"changed"))
    with pytest.raises(flex.H3FlexUnavailable, match="changed before loading"):
        flex.load_pipeline(root)


def test_plain_linear_selection_and_initialization_match_historical_uni():
    from collections import OrderedDict
    from conceptmod.textsliders.minimax_h3_backend import DummyOmniTransformer, _AttnLoRA, LORA_ATTN_CLASS, LORA_LINEAR_NAMES
    transformer = DummyOmniTransformer()
    attention = transformer.transformer_blocks[0].attn
    attention._modules = OrderedDict((name, attention._modules[name]) for name in ("to_v", "to_out", "to_q", "to_k"))
    historical = copy.deepcopy(transformer)
    reference = torch.nn.Module()
    selected = []
    torch.manual_seed(19)
    # Exact pre-PR algorithm: named_modules traversal, plain Linear filter.
    for name, module in historical.named_modules():
        if module.__class__.__name__ == LORA_ATTN_CLASS:
            for child_name, child in module.named_modules():
                if isinstance(child, torch.nn.Linear) and child_name in LORA_LINEAR_NAMES:
                    key = (f"lora_h3-{name}-{child_name}").replace(".", "-")
                    selected.append(key)
                    reference.add_module(key, _AttnLoRA(key, child, 2, 2, up_init_std=0.02))
    torch.manual_seed(19)
    current = AttnLoRANetwork(transformer, rank=2, alpha=2, up_init_std=0.02)
    assert [layer.lora_name for layer in current.loras] == selected
    assert list(current.state_dict()) == list(reference.state_dict())
    assert all(torch.equal(value, reference.state_dict()[name]) for name, value in current.state_dict().items())


def test_partial_legacy_network_construction_unwinds_forwards_and_registrations(monkeypatch):
    from conceptmod.textsliders import minimax_h3_backend as legacy
    transformer = legacy.DummyOmniTransformer()
    originals = [(module, "forward" in vars(module), vars(module).get("forward")) for module in transformer.modules()]
    original_branch = legacy._AttnLoRA
    calls = []
    def fail_third(*args, **kwargs):
        calls.append(True)
        if len(calls) == 3:
            raise ValueError("partial construction failure")
        return original_branch(*args, **kwargs)
    monkeypatch.setattr(legacy, "_AttnLoRA", fail_third)
    network = legacy.AttnLoRANetwork.__new__(legacy.AttnLoRANetwork)
    with pytest.raises(ValueError, match="partial construction failure"):
        network.__init__(transformer, rank=2, alpha=2)
    assert network.loras == [] and not network.state_dict()
    assert list(network.parameters()) == []
    for module, had_forward, forward in originals:
        assert ("forward" in vars(module)) == had_forward
        assert vars(module).get("forward") is forward
    network.close()


def test_cleanup_notes_are_optional_and_do_not_accumulate_on_retry():
    class Pre311Error(RuntimeError):
        add_note = None
    primary = Pre311Error("primary")
    def fail():
        raise RuntimeError("secondary")
    flex._cleanup_preserving_primary(primary, fail)
    assert isinstance(primary.cleanup_error, RuntimeError)
    current = ValueError("primary")
    flex._cleanup_preserving_primary(current, fail)
    flex._cleanup_preserving_primary(current, fail)
    assert len(current.__notes__) == 1


def test_constructor_failure_unwinds_owned_forwards_guards_and_flags(monkeypatch):
    model = tiny_native_backend().stock_reference
    from diffusers import PikaH3FlexModularPipeline
    model.train().requires_grad_(True)
    pipe = PikaH3FlexModularPipeline(workflow="t2va")
    pipe.update_components(transformer=model)
    original_forwards = {id(m): (m, "forward" in vars(m), vars(m).get("forward")) for m in model.modules()}
    original_flags = [(p, p.requires_grad) for p in model.parameters()]
    monkeypatch.setattr(flex, "load_pipeline", lambda *_a, **_k: pipe)
    def fail_after_guards(_self):
        raise ValueError("post-build constructor failure")
    monkeypatch.setattr(flex.PikaH3FlexBackend, "_check_fixed_pika", fail_after_guards)
    with pytest.raises(ValueError, match="post-build constructor failure"):
        flex.PikaH3FlexBackend("not-read")
    for module, had_value, value in original_forwards.values():
        assert ("forward" in vars(module)) == had_value
        assert vars(module).get("forward") is value
    assert all(p.requires_grad == flag for p, flag in original_flags)
    assert all(m.training for m in model.modules())
    for owner in (pipe, model):
        assert "set_adapters" not in vars(owner)
        assert "fuse_lora" not in vars(owner)


def test_constructor_preserves_primary_and_retains_failed_cleanup(monkeypatch):
    model = tiny_native_backend().stock_reference
    from diffusers import PikaH3FlexModularPipeline
    pipe = PikaH3FlexModularPipeline(workflow="t2va")
    pipe.update_components(transformer=model)
    monkeypatch.setattr(flex, "load_pipeline", lambda *_a, **_k: pipe)
    restore = flex.PikaH3FlexBackend._restore_guards
    def primary(_self):
        raise ValueError("constructor primary")
    def cleanup(_self):
        raise RuntimeError("guard cleanup secondary")
    monkeypatch.setattr(flex.PikaH3FlexBackend, "_check_fixed_pika", primary)
    monkeypatch.setattr(flex.PikaH3FlexBackend, "_restore_guards", cleanup)
    with pytest.raises(ValueError, match="constructor primary") as caught:
        flex.PikaH3FlexBackend("not-read")
    assert "guard cleanup secondary" in repr(caught.value.cleanup_error)
    assert any("Cleanup also failed" in note for note in caught.value.__notes__)
    monkeypatch.setattr(flex.PikaH3FlexBackend, "_restore_guards", restore)
    caught.value.retry_cleanup()
    assert "set_adapters" not in vars(pipe)


def test_nested_cleanup_failures_retain_every_pending_owner():
    primary = ValueError("primary")
    calls = []
    failing = True

    def cleanup(name):
        calls.append(name)
        if failing:
            raise RuntimeError(name)

    flex._cleanup_preserving_primary(primary, lambda: cleanup("inner"))
    flex._cleanup_preserving_primary(primary, lambda: cleanup("outer"))
    assert str(primary) == "primary"
    assert len(primary.pending_cleanups) == 2
    calls.clear()
    with pytest.raises(RuntimeError, match="inner"):
        primary.retry_cleanup()
    assert calls == ["inner", "outer"]
    failing = False
    calls.clear()
    primary.retry_cleanup()
    assert calls == ["inner", "outer"]


def test_shared_admission_forwards_native_shape_without_bypass(monkeypatch):
    import sys
    calls = []
    def deny(shape, **kwargs):
        calls.append((shape, kwargs))
        raise ValueError("released PG09 cannot admit flat continuous vectors with width >=170")
    fake = types.ModuleType("particle_sliders.pg09")
    fake.PG_SOURCE_PROFILE = "release-0.9.3"
    fake.validate_pg09_domain = deny
    monkeypatch.setitem(sys.modules, "particle_sliders.pg09", fake)
    with pytest.raises(ValueError, match="width >=170"):
        flex.admit_shared_gan((1, 123456))
    assert calls == [((1, 123456), {"response_domain": "continuous_vector", "pg_source_profile": "release-0.9.3", "penalty_input_domain": "auto"})]


def tiny_native_backend():
    diffusers = pytest.importorskip("diffusers")
    pytest.importorskip("peft")
    from peft import LoraConfig
    if not hasattr(diffusers, "PikaH3FlexModularPipeline"):
        pytest.skip("requires the pinned Pika-native Diffusers revision")
    flex.require_native_runtime()
    model = diffusers.MiniMaxH3Transformer3DModel(
        num_attention_heads=2, attention_head_dim=16, hidden_size=32, num_layers=1,
        num_refiner_layers=1, ffn_dim=64, in_channels=24, audio_in_channels=32,
        patch_size=(1, 2, 2), text_dim=8, freq_dim=8, time_embed_hidden_dim=32,
        time_embed_dim=16, rope_freq_dim=2)
    model.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=list(flex.RELEASE["fl2va"]["target_modules"])), adapter_name=flex.PIKA_ADAPTER)
    # A nonzero fixed release-like adapter lets scale-zero preservation detect
    # accidentally bypassing the PEFT wrapper. These are random fixture weights.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.03)
    model.requires_grad_(False).eval()
    reference = copy.deepcopy(model)
    pipe = diffusers.PikaH3FlexModularPipeline(workflow="t2va")
    pipe.update_components(transformer=model)
    backend = flex.PikaH3FlexBackend.__new__(flex.PikaH3FlexBackend)
    backend.pipe = pipe
    backend.transformer = model
    backend.device = torch.device("cpu")
    backend._attach_slider(rank=2, alpha=2.0)
    backend.stock_reference = reference
    return backend


@pytest.mark.parametrize("checkpointing", [False, True])
def test_native_peft_slider_gradient_and_frozen_parameters(checkpointing):
    backend = tiny_native_backend()
    if checkpointing:
        backend.transformer.enable_gradient_checkpointing()
    dummy = MiniMaxH3Backend(dummy=True)
    packed = dummy.pack_t2va(dummy.encode_text("a fox"))
    packed.hidden_states = torch.randn(1, 2, 96)
    packed.audio_hidden_states = torch.randn(1, 2, 32)
    before = {name: p.detach().clone() for name, p in backend.transformer.named_parameters()}
    with backend.scaled(0):
        zero = backend.velocity(packed).sample.detach().clone()
        stock = backend.stock_reference(**{name: getattr(packed, name) for name in packed.__dataclass_fields__}).sample
        assert torch.equal(zero, stock)
        # Verify scale zero includes the fixed Pika contribution, rather than
        # merely leaving its name in the active-adapter metadata.
        peft_layers = [m for m in backend.stock_reference.modules() if hasattr(m, "scaling")]
        try:
            for module in peft_layers:
                module.scaling[flex.PIKA_ADAPTER] = 0.0
            unadapted = backend.stock_reference(**{name: getattr(packed, name) for name in packed.__dataclass_fields__}).sample.detach().clone()
        finally:
            for module in peft_layers:
                module.scaling[flex.PIKA_ADAPTER] = 1.0
        assert not torch.equal(zero, unadapted)
    result = flex.gradient_probe(backend, packed)
    assert result["nonzero_gradient_leaves"] > 0
    assert result["updated"] is False
    assert len(backend.network.loras) == 8  # native block plus token refiner
    assert all(p.grad is None for p in backend.transformer.parameters())
    assert all(torch.equal(before[name], p) for name, p in backend.transformer.named_parameters())
    assert all(p.requires_grad and p.dtype == torch.float32 for p in backend.trainable_parameters())
    expected_hosts = {
        f"{prefix}.{leaf}": backend.transformer.get_submodule(f"{prefix}.{leaf}")
        for prefix in ("token_refiner.refiner_blocks.0.attn", "transformer_blocks.0.attn")
        for leaf in ("to_q", "to_k", "to_v", "to_out.0")
    }
    assert {layer.lora_name for layer in backend.network.loras} == {
        ("lora_h3-" + path).replace(".", "-") for path in expected_hosts
    }
    assert {id(layer.org_forward.__self__) for layer in backend.network.loras} == {id(host) for host in expected_hosts.values()}
    assert all("lora_A" not in layer.lora_name and "lora_B" not in layer.lora_name for layer in backend.network.loras)
    with torch.no_grad():
        for p in backend.network.parameters():
            p.add_(0.05)
    with backend.scaled(1):
        on = backend.velocity(packed).sample.detach().clone()
        with backend.scaled(0):
            assert torch.equal(zero, backend.velocity(packed).sample)
        assert all(layer.multiplier == 1 for layer in backend.network.loras)
    assert not torch.equal(on, zero)
    for owner in (backend.pipe, backend.transformer):
        owner.set_adapters([flex.PIKA_ADAPTER], [1.0])
        with backend.scaled(0):
            assert torch.equal(zero, backend.velocity(packed).sample)
        assert all(not p.requires_grad for p in backend.transformer.parameters())
        assert all(p.requires_grad and p.dtype == torch.float32 for p in backend.trainable_parameters())
    # The release PEFT scaling is never zeroed along with the custom slider.
    for module in backend.transformer.modules():
        if hasattr(module, "scaling"):
            assert module.scaling[flex.PIKA_ADAPTER] == 1.0


@pytest.mark.parametrize("owner_name,operation", [
    ("pipe", "fuse_lora"), ("pipe", "unload_lora_weights"), ("pipe", "disable_lora"),
    ("transformer", "fuse_lora"), ("transformer", "unload_lora"), ("transformer", "disable_adapters"),
])
def test_destructive_adapter_operations_rejected_before_mutation(owner_name, operation):
    backend = tiny_native_backend()
    before = {name: p.detach().clone() for name, p in backend.transformer.named_parameters()}
    with pytest.raises(flex.H3FlexUnavailable, match="mutation is unsupported"):
        getattr(getattr(backend, owner_name), operation)()
    assert all(torch.equal(before[name], p) for name, p in backend.transformer.named_parameters())
    backend._check_fixed_pika()


def test_nonunit_adapter_selection_and_direct_disable_fail_closed():
    backend = tiny_native_backend()
    with pytest.raises(flex.H3FlexUnavailable, match="unit weight"):
        backend.pipe.set_adapters([flex.PIKA_ADAPTER], adapter_weights=[0.0])
    with pytest.raises(flex.H3FlexUnavailable, match="unit weight"):
        backend.transformer.set_adapters(["other"], [1.0])
    layer = next(iter(backend._release_layers.values()))
    layer.enable_adapters(False)  # unsupported direct native-internal mutation
    with pytest.raises(flex.H3FlexUnavailable, match="unmerged, enabled"):
        with backend.scaled(0):
            pass


def test_current_shared_generic_peft_scaling_is_rejected(monkeypatch):
    """Execute the actual unmodified helper; never bind it as H3's scaler."""
    module = pytest.importorskip("particle_sliders.pg09_backends",
                                 reason="generic scaling hazard requires the installed shared dependency")
    backend = tiny_native_backend()
    generic = module.lora_scale_context([("transformer", backend.transformer)])
    with generic(0):
        assert all(layer.scaling[flex.PIKA_ADAPTER] == 0 for layer in backend._release_layers.values())
        with pytest.raises(flex.H3FlexUnavailable, match="unmerged, enabled"):
            with backend.scaled(0):
                pass
    # The generic context restores its dictionary; the independent H3 scaler
    # keeps the companion adapter at 1 at zero and signed slider strengths.
    for strength in (0, 0.5, -1, 1):
        with backend.scaled(strength):
            assert all(layer.scaling[flex.PIKA_ADAPTER] == 1 for layer in backend._release_layers.values())
            assert all(layer.multiplier == strength for layer in backend.network.loras)


@pytest.mark.parametrize("duration,prepared", [(5, 124), (8, 192), (10, 243), (15, 362)])
def test_render_metadata_matches_returned_frames(monkeypatch, duration, prepared):
    backend = tiny_native_backend()
    calls = []
    def fake_render(_pipe, **kwargs):
        calls.append(kwargs)
        return {"videos": [[None] * prepared], "audio": [torch.zeros(2, 32)], "sampling_rate": 32000}
    monkeypatch.setattr(type(backend.pipe), "__call__", fake_render)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = backend.render("a fox", duration=duration, strength=0.5)
    assert calls[0]["num_frames"] == duration * 24  # original request reaches native alignment
    assert result["frame_metadata"]["requested_frame_count"] == duration * 24
    assert result["frame_metadata"]["prepared_frame_count"] == len(result["videos"][0])
    assert result["frame_metadata"]["actual_duration_seconds"] == prepared / 24
    assert len(caught) == int(prepared != duration * 24)
    backend._check_fixed_pika()


def test_render_rejects_incorrect_returned_frame_count(monkeypatch):
    backend = tiny_native_backend()
    monkeypatch.setattr(type(backend.pipe), "__call__", lambda *_a, **_k: {"videos": [[None] * 123]})
    with pytest.warns(UserWarning, match="duration changed"), pytest.raises(flex.H3FlexUnavailable, match="rendered frame count"):
        backend.render("a fox", duration=5)


def test_native_packing_uses_both_shifted_timesteps():
    backend = tiny_native_backend()
    from diffusers import PikaH3FlexBlocks, MiniMaxH3Scheduler
    from diffusers.modular_pipelines import SequentialPipelineBlocks
    workflow = PikaH3FlexBlocks().get_workflow("t2va")
    names = ("denoise.no_keyframe_anchors", "denoise.prepare_layout", "denoise.prepare_latents", "denoise.set_timesteps")
    prep = SequentialPipelineBlocks.from_blocks_dict({name: workflow.sub_blocks[name] for name in names}).init_pipeline()
    prep.update_components(scheduler=MiniMaxH3Scheduler(shift=12.0, shift_provided_sigmas=True, stochastic_sampling=True),
                           audio_scheduler=MiniMaxH3Scheduler(shift=3.0, shift_provided_sigmas=True, stochastic_sampling=True))
    prep.register_to_config(denoising_step_list=list(flex.SCHEDULE))
    state = prep(text_token_tags=torch.ones(3, dtype=torch.long), height=64, width=64, num_frames=120,
                 generator=torch.Generator().manual_seed(7))
    assert len(state.get("row_timestep_plan")) == 6
    assert state.get("num_frames") == 124
    fields = {name: state.get(name) for name in ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices")}
    assert state.get("latents").shape[-1] == 96
    assert state.get("audio_latents").shape[-1] == 32
    # A subsequent evaluation has different video/audio times due to shifts.
    times, indices = state.get("row_timestep_plan")[1]
    assert not torch.equal(times[indices[fields["video_indices"]]][:1], times[indices[fields["audio_indices"]]][:1])
    embeddings = torch.randn(1, 3, 8)
    with pytest.warns(UserWarning, match="duration changed"):
        metadata = flex.frame_metadata(120, state.get("num_frames"))
    packed = flex.PreparedH3Flex(frame_metadata=metadata, hidden_states=state.get("latents")[None],
                          audio_hidden_states=state.get("audio_latents")[None],
                          encoder_hidden_states=embeddings, timestep=times, timestep_indices=indices, **fields)
    receipt = flex.gradient_probe(backend, packed)
    assert receipt["nonzero_gradient_leaves"] > 0
    assert receipt["frame_metadata"] == metadata
    # Run the real native denoising block over that packed fixture, without
    # loading weights or substituting a custom six-step Euler approximation.
    state.set("prompt_embeds", embeddings)
    denoiser = workflow.sub_blocks["denoise.denoise"].init_pipeline()
    denoiser.update_components(transformer=backend.transformer, scheduler=prep.scheduler, audio_scheduler=prep.audio_scheduler)
    calls = []
    hook = backend.transformer.register_forward_hook(lambda *_: calls.append(1))
    try:
        with backend.scaled(0):
            output = denoiser(state=state)
    finally:
        hook.remove()
    assert len(calls) == 6
    assert torch.isfinite(output.get("latents")).all()
