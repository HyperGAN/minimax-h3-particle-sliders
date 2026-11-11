"""Owner-scoped original H3 tokenwise FF chunking; full native attention stays intact.

The pinned official helper runs unchanged FeedForward calls on token slices.
No Module/Parameter replacement, backend change, dtype cast or RNG draw occurs.
This is a declared execution profile, not a built-in H3 chunking API or bitwise
GEMM/full-model fit claim. All model arithmetic remains on the native CUDA owner.
"""
from pathlib import Path
import hashlib
import torch
from torch import nn

OFFICIAL_SOURCE_SHA256 = {'diffusers.models.attention': '1c387653405148e18f0d7972542d7ced68dd14f626ad8d5a433d91a9bcdf74e8', 'diffusers.models.activations': '3fc5f6ad6b1e8055d7f81a5d8bda9da38bff98449b57b994183240065435386e', 'diffusers.models.transformers.transformer_minimax_h3': '92d665e9fa10b1417088341dce5977663a180f9e6fa404fb9330062f353fe7ef'}
PROFILE = 'h3-original-SwiGLU-FF-token512-exact-remainder-owner-v1'


def _require(value, message):
    if not value:
        raise RuntimeError(message)


def _verify_official():
    from diffusers.models import attention, activations
    from diffusers.models.transformers import transformer_minimax_h3
    modules = (attention, activations, transformer_minimax_h3)
    _require({m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
              for m in modules} == OFFICIAL_SOURCE_SHA256, 'Exact original FF/activation/H3 Source required')
    return attention.FeedForward, activations.SwiGLU, attention._chunked_feed_forward


class OriginalFeedForwardChunks:
    """Install only per-instance ``forward`` dispatch on all50+2 original FFs.

    Original children, names, Parameters and PEFT adapters remain owned by their
    original modules. Closure references are discarded on successful close.
    A changed foreign dispatch is refused rather than overwritten.
    """
    def __init__(self, transformer, *, chunk_tokens=512):
        _require(type(chunk_tokens) is int and chunk_tokens == 512, 'Exact declared512 token bound')
        FeedForward, SwiGLU, official_chunked = _verify_official()
        main = tuple(transformer.transformer_blocks)
        refiners = tuple(transformer.token_refiner.refiner_blocks)
        _require(len(main) == 50 and len(refiners) == 2, 'All original50 main+2 refiner FFs required')
        self.chunk_tokens = chunk_tokens
        self._records = []
        self._official_chunked = official_chunked
        self._closed = False
        try:
            for name, block in [(f'transformer_blocks.{i}.ff', b) for i,b in enumerate(main)] + [
                    (f'token_refiner.refiner_blocks.{i}.ff', b) for i,b in enumerate(refiners)]:
                ff = block.ff
                _require(type(ff) is FeedForward and len(ff.net) == 3 and type(ff.net[0]) is SwiGLU
                         and type(ff.net[1]) is nn.Dropout and ff.net[1].p == 0,
                         'Exact original tokenwise SwiGLU/dropout0/linear FF required')
                _require('forward' not in vars(ff), 'Previously owned FF dispatch must not be replaced')
                parameters = tuple(ff.named_parameters())
                _require(parameters and all(p.device.type == 'cuda' for _,p in parameters),
                         'All original FF/PEFT Parameters remain CUDA-resident')
                original = ff.forward
                def dispatch(hidden_states, *args, _original=original, **kwargs):
                    _require(hidden_states.ndim == 3 and hidden_states.device.type == 'cuda'
                             and hidden_states.dtype == torch.bfloat16, 'Exact native BF16 CUDA FF rows')
                    # This passes only sequence slices to the ORIGINAL callable.
                    # The official helper needs divisibility; a final remainder
                    # is evaluated unchanged instead of padding/dropping tokens.
                    length = hidden_states.shape[1]
                    if length <= chunk_tokens:
                        return _original(hidden_states, *args, **kwargs)
                    prefix = (length // chunk_tokens) * chunk_tokens
                    call = lambda value: _original(value, *args, **kwargs)
                    value = official_chunked(call, hidden_states[:, :prefix], 1, chunk_tokens)
                    if prefix != length:
                        value = torch.cat((value, call(hidden_states[:, prefix:])), dim=1)
                    return value
                ff.forward = dispatch
                self._records.append(dict(name=name, module=ff, original=original, owned=dispatch,
                                          parameters=parameters))
        except BaseException as primary:
            try:
                self.close()
            except BaseException as cleanup:
                raise BaseExceptionGroup('FF chunk installation and owned unwind failed', [primary, cleanup]) from None
            raise

    def assert_owned(self):
        _require(not self._closed and len(self._records) == 52, 'Live complete52 FF owner required')
        for row in self._records:
            _require(row['module'].forward is row['owned'], 'Original FF dispatch ownership changed')
            current = tuple(row['module'].named_parameters())
            _require(len(current) == len(row['parameters']) and all(a == b and p is q
                     for (a,p),(b,q) in zip(current,row['parameters'])), 'Original FF Parameter names/identities changed')

    def witness(self):
        self.assert_owned()
        return dict(schema='H3-original-FF512-owner-execution-v1', profile=PROFILE,
                    original_feed_forwards=52, main_blocks=50, refiners=2, chunk_dimension=1,
                    maximum_chunk_tokens=self.chunk_tokens, exact_remainder=True,
                    full_attention_unchanged=True, original_module_and_Parameter_identities=True,
                    original_PEFT_scales_and_dtype_islands_unchanged=True, no_CPU_model_arithmetic=True,
                    official_Source_SHA256=OFFICIAL_SOURCE_SHA256,
                    bitwise_GEMM_equivalence_claimed=False, full_model_fit_qualified=False)

    def close(self):
        errors = []
        for index in range(len(self._records)-1,-1,-1):
            row = self._records[index]
            if row['module'].forward is not row['owned']:
                errors.append(RuntimeError('Owned FF dispatch changed: '+row['name']))
                continue
            vars(row['module']).pop('forward')
            del self._records[index]
        if errors:
            raise ExceptionGroup('FF dispatch cleanup incomplete; foreign owners retained', errors)
        self._official_chunked = None
        self._closed = True
