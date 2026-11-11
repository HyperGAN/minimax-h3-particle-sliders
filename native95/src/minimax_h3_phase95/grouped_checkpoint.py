"""Scoped ALL-CUDA nested checkpoint dispatch, with original Parameters.

The profile-owned inner scale-aware checkpoint owns every native block's saved
operations. Short outer groups own the inner checkpoints' Tensor inputs, using
Torch's pinned nested-checkpoint rule, so the full fifty input states are not
all retained. No saved-tensor hook, weight move, CPU arithmetic or model global
is installed here. All frozen base/Pika weights remain resident on CUDA.
"""
from pathlib import Path
from types import MethodType
import hashlib
import json
import sys
import torch
from .grouped_forward import grouped_native_forward
from .placement import tensors_to


def synchronous_checkpoint(function, *args, context_fn, **kwargs):
    """Exact pinned nonreentrant generator, with owned synchronous teardown.

    Stock checkpoint drives next(gen), function(), next(gen). If function
    raises, its suspended generator may stay in the exception traceback with
    saved-tensor/scale contexts still entered. This profile owns that same
    generator and always closes it, including a failure of either next(gen).
    No default, RNG, early-stop, determinism or tensor operation is changed.
    """
    from torch.utils.checkpoint import _checkpoint_without_reentrant_generator
    generator = _checkpoint_without_reentrant_generator(
        function, True, context_fn, 'default', False, True, *args, **kwargs)
    primary = None
    try:
        next(generator)
        result = function(*args, **kwargs)
        try:
            next(generator)
        except StopIteration:
            return result
        raise RuntimeError('pinned nonreentrant checkpoint yielded an extra phase')
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            generator.close()
        except BaseException as cleanup:
            if primary is not None:
                raise BaseExceptionGroup('checkpoint primary and synchronous teardown failed', [primary, cleanup])
            raise


def checkpoint_with_synchronous_scale_unwind(module, network):
    """Profile-owned exceptional-unwind derivative of the pinned core wrapper.

    Reuse original repeatable scale contexts and original undo handle. Both
    installed attributes, captured callable and scale semantics are identical;
    only the owned checkpoint invocation deterministically closes its generator.
    Captured core/Torch files and global/class functions remain untouched.
    """
    from particle_sliders.pg09_backends import (
        AdapterScaleCheckpoint, _checkpoint_contexts, _restore_checkpoint_attributes,
    )
    if getattr(module, '_pg09_scale_checkpoint', False):
        raise ValueError('block already checkpointed with adapter scale')
    original = module.forward
    saved = {name: vars(module)[name] for name in ('forward', '_pg09_scale_checkpoint')
             if name in vars(module)}
    def forward(*args, **kwargs):
        scale = network.scale
        return synchronous_checkpoint(original, *args,
            context_fn=lambda: _checkpoint_contexts(network, scale), **kwargs)
    try:
        module.forward = forward
        module._pg09_scale_checkpoint = True
        return AdapterScaleCheckpoint(module, forward, saved)
    except BaseException:
        _restore_checkpoint_attributes(module, saved)
        raise


def validate_runtime():
    pins = json.loads(Path(__file__).with_name('grouped-runtime-sources.json').read_text())
    if pins.get('checkpoint_lifetime') != 'owned-nonreentrant-generator-finally-close-v2':
        raise ValueError('explicit owned synchronous-checkpoint lifetime profile required')
    if torch.__version__ != pins['torch_version']:
        raise ValueError('exact declared nested-checkpoint Torch runtime required')
    import diffusers
    from particle_sliders import pg09_backends
    roots = {'torch': Path(torch.__file__).absolute().parent,
             'diffusers': Path(diffusers.__file__).absolute().parent}
    for name, pin in pins['files'].items():
        package, relative = name.split('/', 1)
        raw = (roots[package] / relative).read_bytes()
        if len(raw) != pin['bytes'] or hashlib.sha256(raw).hexdigest() != pin['sha256']:
            raise ValueError('native/nested-checkpoint operative Source differs: ' + name)
    if hashlib.sha256(Path(pg09_backends.__file__).read_bytes()).hexdigest() != pins['core_scale_wrapper_sha256']:
        raise ValueError('exact original scale-aware checkpoint implementation required')


def _metadata(named):
    return tuple((name, value, value.device, value.dtype, value.requires_grad)
                 for name, value in named)


def _same_metadata(actual, saved):
    return len(actual) == len(saved) and all(
        name == old_name and value is old and value.device == device
        and value.dtype == dtype and value.requires_grad == requires_grad
        for (name, value), (old_name, old, device, dtype, requires_grad) in zip(actual, saved))


class _Group:
    """An unregistered callable owner; it never registers/copies any Parameter."""
    def __init__(self, blocks, device, indices, network):
        self.blocks, self.device, self.indices = tuple(blocks), torch.device(device), tuple(indices)
        self.active = 0
        self.calls = 0
        self.closed = False
        self.inner = tuple(vars(block).get('forward') for block in self.blocks)
        if not self.blocks or len(self.blocks) > 5 or any(
                vars(block).get('_pg09_scale_checkpoint') is not True
                or not callable(forward) for block, forward in zip(self.blocks, self.inner)):
            raise ValueError('outer groups require <=5 profile-owned inner scale-aware checkpoints')
        self.checkpoint = checkpoint_with_synchronous_scale_unwind(self, network)

    def forward(self, hidden, temb, indices, cos, sin, attention_mask):
        if self.closed:
            raise RuntimeError('closed native checkpoint group')
        tensors = (hidden, temb, indices, cos, sin)
        if any(not torch.is_tensor(value) or value.device != self.device for value in tensors):
            raise ValueError('all group Tensor arguments must explicitly share its CUDA device')
        if attention_mask is not None and (not torch.is_tensor(attention_mask)
                                          or attention_mask.device != self.device):
            raise ValueError('group attention mask must share its declared CUDA device')
        if any(vars(block).get('forward') is not forward
               or vars(block).get('_pg09_scale_checkpoint') is not True
               for block, forward in zip(self.blocks, self.inner)):
            raise RuntimeError('original inner checkpoint ownership changed')
        self.calls += 1
        self.active += 1
        try:
            for block in self.blocks:
                # Exact original positional native call, including its full
                # rotary tuple and attention mask. Device-only route prehooks
                # see already-local tensors and therefore reuse these aliases.
                hidden = block(hidden, temb, indices, (cos, sin), attention_mask)
            return hidden
        finally:
            self.active -= 1

    def close(self):
        if self.closed:
            return
        if self.active:
            raise RuntimeError('cannot close an active original native group')
        self.checkpoint.close()
        self.blocks = self.inner = ()
        self.checkpoint = None
        self.closed = True


class GroupedCheckpointDispatch:
    """Own one instance-only native loop derivative until all backward ends.

Install AFTER the profile-owned inner block wrappers; close BEFORE their handles.
Tensor arguments (including both rotary tensors) are explicit checkpoint
inputs. Each bridge is outside its outer checkpoint and remains differentiable.
Core's original repeatable scale contexts and Torch's original CPU/CUDA RNG
save/restore apply at BOTH levels. No autocast mode or RNG stream is replaced.
"""
    def __init__(self, transformer, network, plan):
        validate_runtime()
        from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
        if type(transformer) is not MiniMaxH3Transformer3DModel:
            raise ValueError('exact original native H3 transformer required')
        if (len(transformer.transformer_blocks) != plan.native_main_blocks
                or len(transformer.token_refiner.refiner_blocks) != plan.native_refiners
                or plan.outer_group_size != 5
                or plan.first_device != 'cuda:0' or plan.second_device != 'cuda:1'
                or plan.game_device != plan.first_device):
            raise ValueError('exact all-CUDA grouped placement declaration required')
        if any(getattr(module, 'gradient_checkpointing', False) for module in transformer.modules()):
            raise ValueError('native checkpoint flags must remain off; original scale wrappers own both levels')
        if any(value.device.type != 'cuda' for value in transformer.parameters()):
            raise ValueError('every original base/Pika Parameter must remain resident on CUDA')
        if any(value.device != torch.device(plan.game_device) or value.dtype != torch.float32
               for value in network.parameters()):
            raise ValueError('all original PG trainables must stay FP32 on game0')
        values = vars(transformer)
        if 'forward' in values or '_h3_grouped_dispatch' in values:
            raise ValueError('refuse an existing native instance forward/dispatch owner')
        self.model, self.network, self.plan = transformer, network, plan
        self.parameters = _metadata(transformer.named_parameters())
        self.trainables = _metadata(network.named_parameters())
        self.groups = []
        self.active = 0
        self.closed = False
        self.installed = False
        self.forward = MethodType(grouped_native_forward, transformer)
        self.dispatch = self.run
        try:
            for first, end, device in ((0, plan.main_boundary, plan.first_device),
                                       (plan.main_boundary, plan.native_main_blocks, plan.second_device)):
                for start in range(first, end, plan.outer_group_size):
                    indices = tuple(range(start, min(start + plan.outer_group_size, end)))
                    self.groups.append(_Group(tuple(transformer.transformer_blocks[i] for i in indices),
                                              device, indices, network))
            transformer.forward = self.forward
            transformer._h3_grouped_dispatch = self.dispatch
            self.installed = True
        except BaseException as primary:
            self._close_groups(primary)
            raise

    def _identity(self):
        if (not _same_metadata(tuple(self.model.named_parameters()), self.parameters)
                or not _same_metadata(tuple(self.network.named_parameters()), self.trainables)):
            raise RuntimeError('resident native/PG Parameter identities, devices, dtypes or freezing changed')

    def run(self, hidden, temb, indices, rotary, attention_mask):
        if self.closed or not self.installed:
            raise RuntimeError('closed or incomplete grouped native dispatcher')
        if (vars(self.model).get('forward') is not self.forward
                or vars(self.model).get('_h3_grouped_dispatch') is not self.dispatch):
            raise RuntimeError('owned native group dispatcher changed')
        self._identity()
        if not isinstance(rotary, tuple) or len(rotary) != 2:
            raise ValueError('exact original full rotary cosine/sine pair required')
        cos, sin = rotary
        self.active += 1
        try:
            for group in self.groups:
                # Explicit Tensor args are managed by the parent checkpoint.
                # There is no closure retaining a full hidden state. Metadata
                # and rotary aliases remain local and unchanged inside groups.
                local = tensors_to((hidden, temb, indices, cos, sin, attention_mask), group.device)
                hidden = group.forward(*local)
            return hidden
        finally:
            self.active -= 1

    def _close_groups(self, primary=None):
        errors = []
        for index in range(len(self.groups) - 1, -1, -1):
            try:
                self.groups[index].close()
            except BaseException as error:
                errors.append(error)
            else:
                del self.groups[index]
        if errors:
            raise BaseExceptionGroup('native group cleanup failed', ([primary] if primary else []) + errors)

    def close(self):
        if self.closed:
            return
        if self.active or any(group.active for group in self.groups):
            raise RuntimeError('cannot close active native checkpoint scopes')
        self._identity()
        values = vars(self.model)
        if (values.get('forward') is not self.forward
                or values.get('_h3_grouped_dispatch') is not self.dispatch):
            raise RuntimeError('foreign native dispatcher retained; owned cleanup refused')
        self._close_groups()
        # Original native forward was class-defined. Restore descriptor
        # ownership by deleting only our exact instance attributes.
        values.pop('forward')
        values.pop('_h3_grouped_dispatch')
        self.installed = False
        self.closed = True
        self.model = self.network = self.forward = self.dispatch = None
        self.parameters = self.trainables = ()
