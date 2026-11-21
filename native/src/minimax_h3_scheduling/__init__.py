"""Opt-in AV scheduling on an existing MiniMax H3 FP4 RAW session."""

from contextlib import ExitStack
from sys import float_info


def _sample(runtime, complete_denoise, *, strengths):
    position = 0
    timesteps = None
    scope = ExitStack()

    def before(_transformer, _args):
        nonlocal position, timesteps
        if position >= len(strengths):
            raise RuntimeError("H3 FP4 made extra native denoiser evaluations")
        if timesteps is None:
            timesteps = {
                "video_timesteps": runtime.pipe.scheduler.timesteps.tolist(),
                "audio_timesteps": runtime.pipe.audio_scheduler.timesteps.tolist(),
            }
            if any(len(values) != len(strengths) for values in timesteps.values()):
                raise RuntimeError("H3 FP4 native audio/video solver grids differ from the request")
        scope.enter_context(runtime.scaled(strengths[position]))
        position += 1

    def after(_transformer, _args, _output):
        scope.close()

    with ExitStack() as request:
        request.callback(runtime.transformer.register_forward_pre_hook(before).remove)
        request.callback(runtime.transformer.register_forward_hook(after, always_call=True).remove)
        request.callback(scope.close)
        result = complete_denoise()
    if position != len(strengths):
        raise RuntimeError("H3 FP4 finished before evaluating every requested native interval")
    return result, {**timesteps, "effective_strengths": list(strengths)}


def render_latents(session, *, prompt, seed, step_gains, strength=1.0):
    """Return original coupled latents and the scheduled native step trace."""
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
    if not isinstance(step_gains, (list, tuple)) or len(step_gains) != 6:
        raise ValueError("one gain is required for each of the six native H3 AV steps")
    if any(type(gain) not in (float, int) or not 0 <= gain <= float_info.max for gain in step_gains):
        raise ValueError("H3 FP4 step gains must be finite and nonnegative")
    strengths = [float(strength * gain) for gain in step_gains]
    if any(not 0 <= value <= 1 for value in strengths):
        raise ValueError("effective H3 FP4 strengths must be finite in [0,1]")
    return _sample(runtime, lambda: session.render(prompt=prompt, seed=seed, strength=strength),
                   strengths=strengths)
