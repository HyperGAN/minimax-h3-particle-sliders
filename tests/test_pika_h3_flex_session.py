"""Source/CPU software fixtures; never a pretrained H3 or audiovisual run.

Requires the installed exact core0.12 candidate, not this checkout's core0.11.
Run with ``-o pythonpath=.`` in the separately qualified shared environment.
"""
from copy import deepcopy
from dataclasses import replace
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("particle_sliders", reason="shared H3 fixture requires particle_sliders package")
try:
    core_version = version("particle-sliders-core")
except PackageNotFoundError:
    pytest.skip("shared H3 fixture requires installed particle-sliders-core distribution metadata", allow_module_level=True)
if core_version != "0.12.0":
    pytest.skip("shared H3 fixture requires reviewed core0.12.0; installed version is " + core_version, allow_module_level=True)
diffusers = pytest.importorskip("diffusers", reason="shared H3 fixture requires Pika-native Diffusers")
pytest.importorskip("peft", reason="shared H3 fixture requires PEFT for its frozen companion adapter")
if not hasattr(diffusers, "PikaH3FlexModularPipeline"):
    pytest.skip("shared H3 fixture requires pinned PikaH3FlexModularPipeline support", allow_module_level=True)

from conceptmod.textsliders import pika_h3_flex_session as h3
from conceptmod.textsliders.pika_h3_flex import PIKA_ADAPTER
from conceptmod.textsliders.minimax_h3_backend import PackedLayout
from particle_sliders.lifecycle import SessionConfig, Binding, _open_registered, CheckpointError
from particle_sliders.response import ResponseBackend, ResponseCapabilities, TRAINING, INFERENCE
from particle_sliders.pg09 import _hash, PG_VERSION, PG_SOURCE_PROFILE


def fixture_runtime(seed=7):
    from diffusers import MiniMaxH3Transformer3DModel, PikaH3FlexModularPipeline
    from peft import LoraConfig
    torch.manual_seed(seed)
    model = MiniMaxH3Transformer3DModel(num_attention_heads=2, attention_head_dim=8, hidden_size=16,
        num_layers=1, num_refiner_layers=1, ffn_dim=32, in_channels=1, audio_in_channels=2,
        patch_size=(1, 1, 1), text_dim=8, freq_dim=8, time_embed_hidden_dim=16, time_embed_dim=8, rope_freq_dim=1)
    model.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=list(h3.RELEASE["fl2va"]["target_modules"])), adapter_name=PIKA_ADAPTER)
    with torch.no_grad():
        for name, value in model.named_parameters():
            if "lora_B" in name:
                value.fill_(0.03)
    model.requires_grad_(False).eval()
    identity = {"qualification": "random miniature native H3 architecture, not released H3 geometry or weights",
                "frozen_base_sha256": _hash(model.state_dict())}
    pipe = PikaH3FlexModularPipeline(workflow="t2va")
    pipe.update_components(transformer=model)
    runtime = h3.HNDLH3FlexBackend.__new__(h3.HNDLH3FlexBackend)
    runtime.pipe, runtime.transformer, runtime.device = pipe, model, torch.device("cpu")
    runtime._attach_slider(rank=8, alpha=8)
    pairs = []
    for index in range(2):
        generator = torch.Generator().manual_seed(seed + index + 100)
        layout = PackedLayout(hidden_states=torch.randn(1, 2, 1, generator=generator),
            audio_hidden_states=torch.randn(1, 2, 2, generator=generator),
            encoder_hidden_states=torch.randn(1, 2, 8, generator=generator),
            timestep=torch.tensor([0.2, 0.4]), timestep_indices=torch.tensor([0, 0, 0, 0, 1, 1]),
            token_tags=torch.tensor([1, 1, 0, 0, 2, 2]), position_ids=torch.zeros(6, 3),
            text_indices=torch.tensor([0, 1]), video_indices=torch.tensor([2, 3]), audio_indices=torch.tensor([4, 5]))
        positive = deepcopy(layout)
        positive.encoder_hidden_states = positive.encoder_hidden_states + 0.5
        pairs.append(h3.H3Pair(layout, positive))
    return runtime, pairs, identity


def fixture_prepare(options):
    runtime, pairs, identity = fixture_runtime(options.get("seed", 7))
    return h3.bind(runtime, pairs, options=options, model_identity=identity, shape=(1, 6))


FIXTURE = ResponseBackend("h3-random-architecture-fixture",
    ResponseCapabilities("adapter_gan", "video_audio_fixture", "random_packed_velocity", "continuous_vector", (1, 6),
        PG_VERSION, PG_SOURCE_PROFILE, qualification="six-coordinate random architecture fixture; not H3 release qualification"),
    fixture_prepare, lambda options: None)


def open_fixture(config, **kwargs):
    return _open_registered(config, resume=kwargs.get("resume"), artifact=kwargs.get("artifact"),
        operation=kwargs.get("operation", "train"), request={}, resolve=lambda name: FIXTURE,
        training_schema=TRAINING, inference_schema=INFERENCE)


def test_runtime_reports_missing_distribution_metadata(monkeypatch):
    def missing(_name):
        raise PackageNotFoundError("particle-sliders-core")
    monkeypatch.setattr(h3, "version", missing)
    with pytest.raises(RuntimeError, match="installed particle-sliders-core distribution metadata"):
        h3.require_shared_runtime()


def test_native_preflight_uses_actual_shape_and_refuses_before_loader(monkeypatch):
    called = []
    monkeypatch.setattr(h3, "HNDLH3FlexBackend", lambda *_a, **_k: called.append(True))
    options = dict(snapshot="/never-loaded", rows=[{"neutral": "a fox", "positive": "a red fox"},
                                                  {"neutral": "a room", "positive": "a bright room"}])
    assert h3.native_response_shape(options) == (1, 1398528)
    config = SessionConfig(h3.NAME, "unqualified-native", **options)
    with pytest.raises(ValueError, match="released PG09 cannot admit flat continuous vectors with width >=170"):
        h3.open_session(config)
    assert called == []
    for override in ("width", "response_shape", "dummy", "penalty_input_domain", "pg_source_profile"):
        with pytest.raises(ValueError, match="overrides"):
            h3.validate(dict(options, **{override: 6}))


@pytest.mark.parametrize("resolution,shape,video_rows", [("480p", (1, 1398528), 14430), ("768p", (1, 3593664), 37296)])
def test_geometry_matches_actual_native_packing_at_both_tiers(resolution, shape, video_rows):
    from diffusers import PikaH3FlexBlocks, MiniMaxH3Scheduler
    from diffusers.modular_pipelines import SequentialPipelineBlocks
    geometry = h3.native_geometry({"resolution": resolution, "duration": 5})
    workflow = PikaH3FlexBlocks().get_workflow("t2va")
    names = ("denoise.no_keyframe_anchors", "denoise.prepare_layout", "denoise.prepare_latents", "denoise.set_timesteps")
    prep = SequentialPipelineBlocks.from_blocks_dict({n: workflow.sub_blocks[n] for n in names}).init_pipeline()
    prep.update_components(scheduler=MiniMaxH3Scheduler(shift=12, shift_provided_sigmas=True, stochastic_sampling=True),
                           audio_scheduler=MiniMaxH3Scheduler(shift=3, shift_provided_sigmas=True, stochastic_sampling=True))
    state = prep(text_token_tags=torch.ones(3, dtype=torch.long), height=geometry["height"], width=geometry["width"],
                 num_frames=120, generator=torch.Generator().manual_seed(7))
    video, audio = state.get("latents"), state.get("audio_latents")
    assert video.shape == (video_rows, h3.RELEASE["packed_geometry"]["video_features"])
    assert audio.shape == (414, h3.RELEASE["packed_geometry"]["audio_features"])
    assert geometry["response_shape"] == shape == (1, video.numel() + audio.numel())
    assert geometry["frame_metadata"]["requested_frame_count"] == 120
    assert geometry["frame_metadata"]["prepared_frame_count"] == state.get("num_frames") == 124


def test_frame_details_are_bound_into_model_and_adapter_metadata():
    runtime, pairs, identity = fixture_runtime()
    identity["frame_metadata"] = h3.native_geometry({})["frame_metadata"]
    identity["qualification"] = "synthetic metadata fixture, not native frames"
    adapter = h3.H3ResponseAdapter(runtime, (1, 6), model_identity=identity)
    try:
        assert adapter.metadata()["frame_metadata"] == identity["frame_metadata"]
        assert adapter.metadata()["model_identity"]["frame_metadata"]["prepared_frame_count"] == 124
    finally:
        adapter.close()
        runtime.network.close()


def test_bind_attempts_independent_cleanup_and_retains_retry(monkeypatch):
    runtime, pairs, identity = fixture_runtime()
    binding = h3.bind(runtime, pairs, options={"checkpointing": True}, model_identity=identity, shape=(1, 6))
    adapter = binding.game.backend
    branches = tuple(runtime.network.loras)
    original_close = adapter.close
    def fail():
        raise RuntimeError("checkpoint cleanup failure")
    monkeypatch.setattr(adapter, "close", fail)
    with pytest.raises(RuntimeError, match="checkpoint cleanup failure") as caught:
        binding.close()
    assert runtime.network.loras == []  # independent attention forwards were removed
    assert all(("forward" in vars(b._host)) == b._had_forward for b in branches)
    assert adapter._handles  # failed checkpoint owner remains available for retry
    monkeypatch.setattr(adapter, "close", original_close)
    caught.value.retry_cleanup()
    assert adapter._handles == []
    binding.close()  # idempotent after both cleanup owners finish


def test_direct_bind_adapter_refusal_removes_registered_hndl_forwards():
    runtime, pairs, identity = fixture_runtime()
    branches = tuple(runtime.network.loras)
    with pytest.raises(ValueError, match="width >=170"):
        h3.bind(runtime, pairs, options={}, model_identity=identity, shape=(1, 1398528))
    assert runtime.network.loras == []
    assert list(runtime.network.parameters()) == []
    assert not runtime.network.state_dict()
    for branch in branches:
        assert ("forward" in vars(branch._host)) == branch._had_forward
        branch.close()  # direct projection cleanup is idempotent too


def test_hndl_close_retains_all_failures_and_only_failed_registrations():
    runtime, _, _ = fixture_runtime()
    branches = tuple(runtime.network.loras[:2])
    installed = [branch._host.forward for branch in branches]
    for branch in branches:
        branch._host.forward = lambda value: value
    with pytest.raises(RuntimeError) as caught:
        runtime.network.close()
    assert len(caught.value.cleanup_failures) == 2
    assert runtime.network.loras == list(branches)
    assert set(runtime.network._modules) == {branch.lora_name for branch in branches}
    for branch, forward in zip(branches, installed):
        branch._host.forward = forward
    runtime.network.close()
    assert not runtime.network.state_dict()


def test_bind_cleanup_does_not_hide_primary_error(monkeypatch):
    runtime, pairs, identity = fixture_runtime()
    original_close = runtime.network.close
    def fail_critic(*_args):
        raise ValueError("critic construction primary")
    def fail_cleanup():
        raise RuntimeError("graph cleanup secondary")
    monkeypatch.setattr(h3, "HNDLCritic", fail_critic)
    monkeypatch.setattr(runtime.network, "close", fail_cleanup)
    with pytest.raises(ValueError, match="critic construction primary") as caught:
        h3.bind(runtime, pairs, options={}, model_identity=identity, shape=(1, 6))
    assert "graph cleanup secondary" in repr(caught.value.cleanup_error)
    assert any("Cleanup also failed" in note for note in caught.value.__notes__)
    monkeypatch.setattr(runtime.network, "close", original_close)
    caught.value.retry_cleanup()


def test_prepare_cleanup_does_not_hide_native_primary(monkeypatch):
    from types import SimpleNamespace
    def fail_prepare(*_a, **_k):
        raise ValueError("native preparation primary")
    def fail_close():
        raise RuntimeError("native cleanup secondary")
    runtime = SimpleNamespace(prepare_t2va=fail_prepare, network=SimpleNamespace(close=fail_close))
    monkeypatch.setattr(h3, "HNDLH3FlexBackend", lambda *_a, **_k: runtime)
    with pytest.raises(ValueError, match="native preparation primary") as caught:
        h3.prepare_native({"snapshot": "not-read", "rows": [{"neutral": "a fox", "positive": "red fox"}]})
    assert "native cleanup secondary" in repr(caught.value.cleanup_error)


def test_prepare_native_refuses_frame_metadata_mismatch_before_bind(monkeypatch):
    from types import SimpleNamespace
    closed = []
    metadata = h3.native_geometry({})["frame_metadata"]
    metadata["prepared_frame_count"] = 120
    runtime = SimpleNamespace(prepare_t2va=lambda *_a, **_k: SimpleNamespace(frame_metadata=metadata),
                              network=SimpleNamespace(close=lambda: closed.append(True)))
    monkeypatch.setattr(h3, "HNDLH3FlexBackend", lambda *_a, **_k: runtime)
    monkeypatch.setattr(h3, "bind", lambda *_a, **_k: pytest.fail("mismatch reached bind"))
    with pytest.raises(ValueError, match="frame metadata differs"):
        h3.prepare_native({"snapshot": "not-read", "rows": [{"neutral": "fox", "positive": "red fox"}]})
    assert closed == [True]


def assert_atomic_refusal(session, refuse):
    from particlegan.bispf import _LAST_THERMAL_SCALE
    packet = _hash(session.checkpoint())
    rng = torch.get_rng_state().clone()
    parameters = tuple(session.game.network.parameters()) + tuple(session.game.critic.parameters())
    ids = tuple(map(id, parameters))
    gradients = [(p.grad, None if p.grad is None else p.grad.clone(), p.requires_grad) for p in parameters]
    thermal = deepcopy(_LAST_THERMAL_SCALE)
    refuse()
    assert _hash(session.checkpoint()) == packet
    assert torch.equal(torch.get_rng_state(), rng)
    assert ids == tuple(map(id, tuple(session.game.network.parameters()) + tuple(session.game.critic.parameters())))
    assert _LAST_THERMAL_SCALE == thermal
    for p, (reference, value, flag) in zip(parameters, gradients):
        assert p.grad is reference and p.requires_grad == flag
        if value is not None:
            assert torch.equal(p.grad, value)


@pytest.mark.parametrize("field,value", [("frozen_base_sha256", "0" * 64),
                                         ("frame_metadata", {"requested_frame_count": 120, "prepared_frame_count": 120})])
def test_changed_model_identity_refuses_restore_atomically(field, value):
    config = SessionConfig(FIXTURE.name, "identity-refusal-v1", seed=7, steps=2, batch=2)
    with open_fixture(config) as source:
        source.train(stop_after=1)
        packet = source.checkpoint()
    def other_identity(options):
        runtime, pairs, identity = fixture_runtime(options.get("seed", 7))
        identity[field] = value
        return h3.bind(runtime, pairs, options=options, model_identity=identity, shape=(1, 6))
    receiver_backend = replace(FIXTURE, prepare=other_identity)
    with receiver_backend.open(config) as receiver:
        def refuse():
            with pytest.raises(CheckpointError, match="source"):
                receiver.restore(packet)
        assert_atomic_refusal(receiver, refuse)


def test_foreign_pg_profile_refuses_restore_atomically():
    config = SessionConfig(FIXTURE.name, "profile-refusal-v1", seed=7, steps=2, batch=2)
    with open_fixture(config) as receiver:
        receiver.train(stop_after=1)
        packet = receiver.checkpoint()
        packet["payload"]["game"]["config"]["pg_source_profile"] = "release-0.9.3"
        packet["sha256"] = _hash({k: v for k, v in packet.items() if k != "sha256"})
        def refuse():
            with pytest.raises(CheckpointError) as caught:
                receiver.restore(packet)
            assert "PG source/cohort mismatch" in str(caught.value.__cause__)
        assert_atomic_refusal(receiver, refuse)


def test_hndl_projection_matches_native_lora_values_and_gradients():
    runtime, pairs, _ = fixture_runtime()
    branch = runtime.network.loras[0]
    value = torch.randn(1, 2, branch.lora_down.in_features, requires_grad=True)
    with torch.no_grad():
        branch.lora_up.weight.fill_(0.02)
    branch.multiplier = 0.5
    actual = branch(value)
    reference = branch.org_forward(value) + 0.5 * torch.nn.functional.linear(
        torch.nn.functional.linear(value, branch.lora_down.weight), branch.lora_up.weight)
    assert torch.equal(actual, reference)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), value, retain_graph=True)[0],
                               torch.autograd.grad(reference.sum(), value)[0], rtol=0, atol=0)
    assert type(branch.graph).__module__.startswith("hndl")
    assert branch._identity["hypergan_commit"] == "5f368a3770cf5bdae0448995c9f76e29af8049f7"
    runtime.network.close()


def test_bfloat16_native_host_keeps_hndl_leaves_float32():
    runtime, _, _ = fixture_runtime()
    runtime.transformer.to(dtype=torch.bfloat16)
    branch = runtime.network.loras[0]
    with torch.no_grad():
        branch.lora_up.weight.fill_(0.02)
    value = torch.randn(1, 2, branch.lora_down.in_features, dtype=torch.bfloat16, requires_grad=True)
    with runtime.scaled(0.5):
        output = branch(value)
        grads = torch.autograd.grad(output.float().square().mean(), tuple(branch.parameters()))
    assert output.dtype == torch.bfloat16
    assert all(p.dtype == torch.float32 for p in branch.parameters())
    assert all(g.dtype == torch.float32 and torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
    runtime._check_fixed_pika()
    runtime.network.close()


@pytest.mark.parametrize("checkpointing", [False, True])
def test_real_shared_game_updates_hndl_leaves_only_and_preserves_pika(checkpointing):
    config = SessionConfig(FIXTURE.name, "random-software-v1", seed=7, steps=2, batch=2, checkpointing=checkpointing, ema_decay=0.5)
    with open_fixture(config) as session:
        runtime = session.game.backend.runtime
        base = {name: value.detach().clone() for name, value in runtime.transformer.named_parameters()}
        before = [p.detach().clone() for p in session.game.network.parameters()]
        critic = session.game.critic
        assert isinstance(critic, h3.HNDLCritic)
        assert type(critic.graph).__module__.startswith("hndl")
        session.train()
        assert session.game.completed_steps == 2
        assert any(not torch.equal(a, b) for a, b in zip(before, session.game.network.parameters()))
        assert all(torch.equal(base[name], value) for name, value in runtime.transformer.named_parameters())
        for strength in (0, 0.5, -1, 1):
            with runtime.scaled(strength):
                assert all(layer.scaling[PIKA_ADAPTER] == 1 for layer in runtime._release_layers.values())
        runtime._check_fixed_pika()
        assert session.manifest["game_signature"]["pg_version"] == "0.9.4"
        assert session.manifest["capabilities"]["generator_arm"] == "adapter_gan"


@pytest.mark.parametrize("checkpointing", [False, True])
def test_shared_checkpoint_resume_export_and_observation_are_exact(tmp_path, checkpointing):
    config = SessionConfig(FIXTURE.name, "random-exact-resume-v1", seed=7, steps=4, batch=2, ema_decay=0.5, checkpointing=checkpointing)
    checkpoint = tmp_path / "training.pt"
    artifact = tmp_path / "adapter.pt"
    with open_fixture(config) as uninterrupted:
        uninterrupted.train()
        expected = deepcopy(uninterrupted.game.network.state_dict())
    with open_fixture(config) as first:
        first.train(stop_after=2)
        state = _hash(first.game.state_dict())
        first.evaluate(0, seed=19, strength=0.5)
        assert _hash(first.game.state_dict()) == state
        first.checkpoint(checkpoint)
    with open_fixture(config, resume=checkpoint) as second:
        second.train()
        assert all(torch.equal(expected[name], value) for name, value in second.game.network.state_dict().items())
        second.export(artifact, selection="raw")
        result = second.evaluate(0, seed=19)
    with open_fixture(config, artifact=artifact, operation="response") as inference:
        torch.testing.assert_close(inference.evaluate(0, seed=19), result, rtol=0, atol=0)
    wrong = SessionConfig(FIXTURE.name, "different-cohort", seed=7, steps=4, batch=2, ema_decay=0.5, checkpointing=checkpointing)
    with pytest.raises(CheckpointError, match="cohort"):
        open_fixture(wrong, resume=checkpoint)
