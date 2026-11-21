"""Checks using synthetic latents and the pinned native Diffusers solver."""

from contextlib import contextmanager
from copy import deepcopy
from os import environ
from pathlib import Path
from sys import modules
from types import ModuleType, SimpleNamespace
from zipfile import ZipFile

import pytest
import torch

from minimax_h3_scheduling import render_latents
from minimax_h3_scheduling.activation_gate import AVGatePlan, CORE_MEMBER, coupled_latent_readout, gate_core


@pytest.fixture(scope="module")
def core_wheel():
    supplied = environ.get("H3_STEERING_CORE_WHEEL")
    if not supplied:
        pytest.skip("set H3_STEERING_CORE_WHEEL to the reviewed core #269 wheel")
    return Path(supplied)


@pytest.fixture
def native_raw(monkeypatch):
    from diffusers.modular_pipelines.pika_h3_flex.before_denoise import PikaH3FlexSetTimestepsStep
    from diffusers.modular_pipelines.pika_h3_flex.denoise import PikaH3FlexLoopDenoiser, PikaH3FlexLoopSchedulerStep
    from diffusers.schedulers import MiniMaxH3Scheduler

    class Transformer(torch.nn.Module):
        strength = 0.0
        failure = False

        def forward(self, *, hidden_states, audio_hidden_states, encoder_hidden_states, timestep,
                    timestep_indices, attention_kwargs, return_dict, video_indices, audio_indices):
            runtime.readouts.append(coupled_latent_readout(hidden_states, audio_hidden_states))
            if self.failure:
                raise RuntimeError("fixture forward failed")
            return hidden_states * (1 + self.strength), audio_hidden_states * (1 + self.strength)

    class FP4H3Backend:
        identity = "fixture-only"
        fault = None

        def __init__(self):
            self.transformer = Transformer()
            self.pipe = SimpleNamespace(
                transformer=self.transformer,
                scheduler=MiniMaxH3Scheduler(shift=12.0, stochastic_sampling=True, shift_provided_sigmas=True),
                audio_scheduler=MiniMaxH3Scheduler(shift=3.0, stochastic_sampling=True, shift_provided_sigmas=True),
            )

        @contextmanager
        def scaled(self, value):
            previous, self.transformer.strength = self.transformer.strength, value
            try:
                yield
            finally:
                self.transformer.strength = previous

        def render(self, *, prompt, seed, strength):
            self.readouts = []
            denoiser, solver = PikaH3FlexLoopDenoiser(), PikaH3FlexLoopSchedulerStep()
            original = self.pipe
            self.pipe = SimpleNamespace(transformer=self.transformer, scheduler=deepcopy(original.scheduler),
                                        audio_scheduler=deepcopy(original.audio_scheduler))
            video_rows, audio_rows = torch.tensor([0, 1, 2]), torch.tensor([3, 4])
            layout = dict(video_indices=video_rows, audio_indices=audio_rows)
            try:
                with self.scaled(strength):
                    for scheduler in (self.pipe.scheduler, self.pipe.audio_scheduler):
                        scheduler.set_timesteps(sigmas=[1., .75, .5, .25, .1, .04, 0.])
                    times = self.pipe.scheduler.timesteps
                    audio_times = self.pipe.audio_scheduler.timesteps
                    plan = [PikaH3FlexSetTimestepsStep.build_row_timesteps(
                        video_rows, audio_rows, 0, 0, 1, float(v), float(a), 1., 1.,
                    ) for v, a in zip(times, audio_times, strict=True)]
                    state = SimpleNamespace(
                        latents=torch.ones((3, 2)), audio_latents=torch.ones((2, 3)),
                        prompt_embeds=torch.zeros((1, 1, 2)), denoiser_input_fields=layout, attention_kwargs=None,
                        row_timestep_plan=plan, audio_timesteps=audio_times,
                        num_condition_video_rows=0, num_condition_audio_rows=0,
                        generator=torch.Generator().manual_seed(seed),
                    )
                    for i, time in enumerate(times):
                        if self.fault == "wrong_audio_time" and i == 1:
                            plan[i] = (plan[i][0], torch.zeros_like(plan[i][1]))
                        denoiser(self.pipe, state, i, time)
                        if self.fault == "repeat_interval" and i == 0:
                            denoiser(self.pipe, state, i, time)
                        if self.fault == "unfinished" and i == len(times) - 1:
                            break
                        solver(self.pipe, state, i, time)
                    return state.latents, state.audio_latents
            finally:
                self.pipe = original

    fixture_module = ModuleType("minimax_h3_fp4.native")
    fixture_module.FP4H3Backend = FP4H3Backend
    monkeypatch.setitem(modules, "minimax_h3_fp4.native", fixture_module)
    runtime = FP4H3Backend()

    def close():
        return runtime

    session = SimpleNamespace(_open=lambda: None, selection="raw", render=runtime.render,
                              binding=SimpleNamespace(close=close, render=True, identity={"model": runtime.identity}))
    return runtime, session


def plan(core_wheel, **settings):
    return AVGatePlan(core_wheel=core_wheel, calibration_sha256="a" * 64, **settings)


def render(session, **kwargs):
    return render_latents(session, prompt="synthetic fixture", seed=7, strength=.5, **kwargs)


@pytest.mark.parametrize("rate", [0., 1.])
def test_random_endpoints_match_unmodified_native_outputs(core_wheel, native_raw, rate):
    runtime, session = native_raw
    original = runtime.render(prompt="synthetic fixture", seed=7, strength=.5 * rate)
    result, trace = render(session, gate_plan=plan(core_wheel, method="random", parameter=rate, seed=3))
    assert all(torch.equal(a, b) for a, b in zip(result, original, strict=True))
    assert trace["effective_strengths"] == [.5 * rate] * 6
    assert trace["gate_rate"] == rate
    assert trace["video_sigma_integrated_dose"] == .5 * rate
    assert trace["audio_sigma_integrated_dose"] == .5 * rate
    assert trace["video_sigmas"] != trace["audio_sigmas"]
    assert runtime.transformer.strength == 0
    assert runtime.pipe.scheduler.step_index is None
    assert runtime.pipe.audio_scheduler.step_index is None


@pytest.mark.parametrize("method", ["cfp", "clabs"])
def test_live_joint_projection_and_fixed_replay(core_wheel, native_raw, method):
    runtime, session = native_raw
    direction = torch.tensor([1., 0., 1., 0., 0.])
    direction = direction / direction.norm()
    if method == "cfp":
        runtime.render(prompt="synthetic fixture", seed=7, strength=0.)
        initial, second = (float(readout @ direction) for readout in runtime.readouts[:2])
        assert initial > second
        settings = dict(parameter=second)
    else:
        settings = dict(parameter=.5, seed=3, clabs=dict(mean=0., std=1., epsilon=1e-6, setpoint=1.,
                                                      proportional_gain=.1, integral_gain=.1, beta=.2))
    gate = plan(core_wheel, method=method, direction=direction, **settings)
    result, trace = render(session, gate_plan=gate)
    assert 0 < trace["fired"] < trace["eligible"] == 6
    scores = [float(readout @ direction) for readout in runtime.readouts]
    assert [record["projection"] for record in trace["decisions"]] == scores
    assert [record["interval"] for record in trace["decisions"]] == list(range(6))
    actual_replay, _ = render(session, step_gains=[v / .5 for v in trace["effective_strengths"]])
    assert all(torch.equal(a, b) for a, b in zip(result, actual_replay, strict=True))
    rerun, repeat_trace = render(session, gate_plan=gate)
    assert trace == repeat_trace
    assert all(torch.equal(a, b) for a, b in zip(result, rerun, strict=True))


@pytest.mark.parametrize("fault, message", [
    ("wrong_audio_time", "packed rows"), ("repeat_interval", "interval alignment"),
    ("unfinished", "both|Both"), ("forward_failure", "fixture forward failed"),
])
def test_gate_cleanup_after_native_failure(core_wheel, native_raw, fault, message):
    runtime, session = native_raw
    runtime.fault = fault
    runtime.transformer.failure = fault == "forward_failure"
    with pytest.raises(RuntimeError, match=message):
        render(session, gate_plan=plan(core_wheel, method="random", parameter=1., seed=3))
    assert runtime.transformer.strength == 0.0
    assert not runtime.transformer._forward_pre_hooks
    assert not runtime.transformer._forward_hooks
    assert runtime.pipe.scheduler.step_index is None and runtime.pipe.audio_scheduler.step_index is None


def test_unverified_wheel_refused_even_after_verified_wheel_loaded(core_wheel, tmp_path):
    gate_core(core_wheel)
    altered = tmp_path / "altered.whl"
    with ZipFile(altered, "w") as archive:
        archive.writestr(CORE_MEMBER, "raise AssertionError('changed source must never execute')")
    with pytest.raises(ValueError, match="exact adaptive steering source"):
        gate_core(altered)
