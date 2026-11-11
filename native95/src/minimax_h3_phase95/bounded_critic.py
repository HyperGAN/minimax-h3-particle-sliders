"""Exact original full-response critic with one explicit HNDL shape bound.

The native 1,398,528-value AV response plus its two conditions needs a
1,398,530-wide concat.  HNDL's default 1,048,576 typo guard refuses it.
Only this exact graph receives that finite max_dimension override.  Registry,
template, default FP32 host construction and subsequent CUDA placement retain
the original pinned HyperGAN/HNDL constructor law.  No CPU model arithmetic,
global HNDL changes, response projection or reduced critic is introduced.
"""
from pathlib import Path
from torch import nn
from .adapter import HNDLCritic as OriginalHNDLCritic, ROOT, _sha
from .builder import checked_builder

PROFILE = "h3-full-native-critic-exact-maxdimension1398530-v1"
RESPONSE_WIDTH = 1_398_528
CONDITION_WIDTH = 2
HIDDEN_WIDTH = 16
LIMITS = {"max_dimension": RESPONSE_WIDTH + CONDITION_WIDTH}
GRAPH_SHA256 = "db58a4e586cb5521eef362f0302f9821eeb646fcc5c1fcc04762b97996ed3764"


class HNDLCritic(OriginalHNDLCritic):
    """Original forward/Parameters/host law; declared full-native shape limit."""

    def __init__(self, response_width, condition_width):
        if (type(response_width) is not int or response_width != RESPONSE_WIDTH
                or type(condition_width) is not int or condition_width != CONDITION_WIDTH):
            raise ValueError("bounded H3 critic requires the exact full native response and two-row condition")
        nn.Module.__init__(self)
        _, identity = checked_builder()  # Authenticate every original HNDL/builder Source first.
        from hndl import Registry
        from hndl.torch import network
        from hypergan.network_config import render_source
        from hypergan.hndl_augmentation import register_augmentation
        from hypergan.pretrained_providers import register_providers
        path = ROOT / "pika-h3-flex-critic.hndl"
        if _sha(path) != GRAPH_SHA256:
            raise ValueError("original full-response H3 critic graph differs")
        source = render_source(path.read_text(), {"hidden": HIDDEN_WIDTH})
        registry = register_augmentation(Registry.builtins())
        registry = register_providers(registry, None)
        self.graph = network(source,
            input_shape={"response": ["B", response_width], "condition": ["B", condition_width]},
            output_shape=["B", 1], device="cpu", registry=registry, input_dtype=None,
            limits=dict(LIMITS))
        self._identity = dict(identity, h3_critic_sha256=GRAPH_SHA256,
            critic_construction_profile=PROFILE, exact_resource_limits=dict(LIMITS),
            critic_construction_source_sha256=_sha(Path(__file__)),
            original_constructor_device="cpu", native_game_device="cuda:0",
            constructor_only_no_CPU_forward_backward=True)
