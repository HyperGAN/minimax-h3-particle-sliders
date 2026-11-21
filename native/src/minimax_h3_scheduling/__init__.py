"""Opt-in AV scheduling on an existing MiniMax H3 FP4 RAW session."""

from contextlib import ExitStack
from itertools import pairwise
from math import fsum, isfinite
from sys import float_info


NATIVE_STEPS = 6


def _sample(runtime, complete_denoise, *, strengths=None, strength=None, gate_plan=None):
    position = 0
    steps = len(strengths) if gate_plan is None else NATIVE_STEPS
    timesteps = None
    schedulers = None
    sigma_grids = None
    decisions = []
    applied = []
    scope = ExitStack()
    if gate_plan is not None:
        from .activation_gate import check_solver_interval, coupled_latent_readout

        gate = gate_plan._new_gate()

    def before(_transformer, _args, kwargs):
        nonlocal position, timesteps, schedulers, sigma_grids
        if position >= steps:
            raise RuntimeError("H3 FP4 made extra native denoiser evaluations")
        current = (runtime.pipe.scheduler, runtime.pipe.audio_scheduler)
        if timesteps is None:
            schedulers = current
            timesteps = {
                "video_timesteps": current[0].timesteps.tolist(),
                "audio_timesteps": current[1].timesteps.tolist(),
            }
            if any(len(values) != steps for values in timesteps.values()):
                raise RuntimeError("H3 FP4 native audio/video solver grids differ from the request")
            if gate_plan is not None:
                sigma_grids = {
                    "video_sigmas": current[0].sigmas.tolist(),
                    "audio_sigmas": current[1].sigmas.tolist(),
                }
                if any(len(grid) != steps + 1 or any(not isfinite(a) or not isfinite(b) or a <= b
                       for a, b in pairwise(grid)) for grid in sigma_grids.values()):
                    raise RuntimeError("H3 FP4 expected two complete native sigma grids")
        if gate_plan is None:
            value = strengths[position]
        else:
            if any(scheduler is not saved for scheduler, saved in zip(current, schedulers, strict=True)):
                raise RuntimeError("H3 FP4 replaced a scheduler during the coupled native trajectory")
            video, audio = kwargs.get("hidden_states"), kwargs.get("audio_hidden_states")
            readout = coupled_latent_readout(video, audio)
            check_solver_interval(schedulers, position, kwargs, video, audio)
            score = gate_plan._score(readout, strength)
            decision = gate.step(score)
            value = strength if decision.fired else 0.0
            decisions.append({"interval": position, "projection": score, "fired": decision.fired,
                              "probability": decision.probability, "effective_strength": value})
        scope.enter_context(runtime.scaled(value))
        applied.append(value)
        position += 1

    def after(_transformer, _args, _output):
        scope.close()

    with ExitStack() as request:
        request.callback(runtime.transformer.register_forward_pre_hook(before, with_kwargs=True).remove)
        request.callback(runtime.transformer.register_forward_hook(after, always_call=True).remove)
        request.callback(scope.close)
        result = complete_denoise()
    if position != steps:
        raise RuntimeError("H3 FP4 finished before evaluating every requested native interval")
    receipt = {**timesteps, "effective_strengths": applied}
    if gate_plan is not None:
        if any(scheduler.step_index != steps for scheduler in schedulers):
            raise RuntimeError("Both H3 FP4 schedulers must complete every gated interval")
        receipt.update(sigma_grids)
        for modality in ("video", "audio"):
            receipt[f"{modality}_sigma_integrated_dose"] = fsum(
                gain * (start - end) for gain, (start, end)
                in zip(applied, pairwise(sigma_grids[f"{modality}_sigmas"]), strict=True)
            )
        receipt.update(activation_gate=gate_plan._receipt(), decisions=decisions, nominal_strength=strength,
                       fired=gate.usage.fired, eligible=gate.usage.eligible, gate_rate=gate.usage.rate)
    return result, receipt


def render_latents(session, *, prompt, seed, step_gains=None, strength=1.0, gate_plan=None):
    """Return original coupled latents and the fixed or gated native step trace."""
    from minimax_h3_fp4.native import FP4H3Backend

    session._open()
    closure = getattr(session.binding.close, "__closure__", None) or ()
    runtimes = [owner for cell in closure if isinstance(owner := cell.cell_contents, FP4H3Backend)]
    if (session.selection != "raw" or session.binding.render is None or len(runtimes) != 1
            or session.binding.identity["model"] != runtimes[0].identity):
        raise ValueError("an already-opened strict FP4 H3 AV RAW session is required")
    runtime = runtimes[0]
    if type(strength) not in (float, int) or not 0 <= strength <= 1:
        raise ValueError("nominal H3 FP4 strength must be finite in [0,1]")

    def denoise():
        return session.render(prompt=prompt, seed=seed, strength=strength)

    if gate_plan is not None:
        from .activation_gate import AVGatePlan

        if step_gains is not None or not isinstance(gate_plan, AVGatePlan):
            raise ValueError("Choose fixed H3 step gains or one calibrated AVGatePlan")
        if strength == 0:
            raise ValueError("Gated H3 strength must be positive; use the original OFF render at zero")
        return _sample(runtime, denoise, strength=strength, gate_plan=gate_plan)

    if not isinstance(step_gains, (list, tuple)) or len(step_gains) != NATIVE_STEPS:
        raise ValueError("one gain is required for each of the six native H3 AV steps")
    if any(type(gain) not in (float, int) or not 0 <= gain <= float_info.max for gain in step_gains):
        raise ValueError("H3 FP4 step gains must be finite and nonnegative")
    strengths = [float(strength * gain) for gain in step_gains]
    if any(not 0 <= value <= 1 for value in strengths):
        raise ValueError("effective H3 FP4 strengths must be finite in [0,1]")
    return _sample(runtime, denoise, strengths=strengths)
