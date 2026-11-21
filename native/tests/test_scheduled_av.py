from contextlib import contextmanager
from types import SimpleNamespace

import torch

from minimax_h3_scheduling import _sample


def test_shared_native_strength_for_both_av_velocities():
    class Transformer(torch.nn.Module):
        strength = 0.0

        def forward(self, video, audio):
            return video * (1 + self.strength), audio + self.strength

    class Runtime:
        def __init__(self):
            self.transformer = Transformer()
            self.pipe = SimpleNamespace(
                scheduler=SimpleNamespace(timesteps=torch.linspace(0, 1, 6)),
                audio_scheduler=SimpleNamespace(timesteps=torch.linspace(0, 0.75, 6)),
            )

        @contextmanager
        def scaled(self, value):
            previous, self.transformer.strength = self.transformer.strength, value
            try:
                yield
            finally:
                self.transformer.strength = previous

        def denoise(self):
            video, audio = torch.tensor(1.0), torch.tensor(0.0)
            for index in range(6):
                video, audio = self.transformer(video + index, audio * (1 + index))
            return video, audio

    runtime = Runtime()
    with runtime.scaled(0.5):
        expected = runtime.denoise()
    constant, trace = _sample(runtime, runtime.denoise, strengths=[0.5] * 6)
    assert all(torch.equal(left, right) for left, right in zip(constant, expected))
    assert trace["video_timesteps"] == runtime.pipe.scheduler.timesteps.tolist()
    assert trace["audio_timesteps"] == runtime.pipe.audio_scheduler.timesteps.tolist()
    assert trace["effective_strengths"] == [0.5] * 6

    values = [0.75, 0.75, 0.75, 0.25, 0.25, 0.25]
    varied, trace = _sample(runtime, runtime.denoise, strengths=values)
    assert not torch.equal(varied[0], expected[0])
    assert not torch.equal(varied[1], expected[1])
    assert trace["effective_strengths"] == values
    assert runtime.transformer.strength == 0.0
