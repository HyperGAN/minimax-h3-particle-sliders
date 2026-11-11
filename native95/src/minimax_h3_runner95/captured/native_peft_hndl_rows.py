"""Run-owned H3 original PEFT + HNDL row512 execution; factory/weights exact.

The captured native PEFT dense base and frozen vanilla Pika A/B are tokenwise.
Their original forward runs on <=512 flattened rows BEFORE the unchanged HNDL
row512 path. The native result is concatenated to full original output geometry.
The original two bias-free HNDL linears are also tokenwise. Only flattened rows
are partitioned. Each original FP32 graph result is cast to the native base
output device/dtype and multiplied in the ORIGINAL order before concatenation.
No full-token FP32 up result is constructed. GEMM/gradient accumulation grouping
can differ: this is an explicit execution profile, never a bitwise claim.
"""
from contextlib import AbstractContextManager
import hashlib
import math
from pathlib import Path
import torch

PROFILE = 'h3-original-PEFT-base-Pika-HNDL-flat-row512-native-cast-scaled-cat-owner-v1'
ADAPTER_SHA256 = '43793d7cf3745e954bc7eeda917ce934cf0b7e320a53219f6e35ae1f617c9f5a'
GRAPH_SHA256 = 'e32094eb55b9178e8c42d7bf483a5915593d5fe6ae3cbca7478559cf80f8dc4c'
PEFT_SHA256 = '9a2688af69fd16cef94fdb08ab0e973b37d7b2a672e64ba2057a01df7e98b7d8'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def parameters(module):
    return tuple((name, value, value.device, value.dtype, value.requires_grad)
                 for name, value in module.named_parameters())


def unchanged(module, saved):
    current = tuple(module.named_parameters())
    return len(current) == len(saved) and all(
        name == old_name and value is old and value.device == device
        and value.dtype == dtype and value.requires_grad == grad
        for (name, value), (old_name, old, device, dtype, grad) in zip(current, saved))


def native_signature(host):
    """Admit frozen plain dense hosts or the captured vanilla Pika PEFT path.

    Snapshot non-Parameter execution state as well as module identities. No
    native state, dropout mode, scaling, adapter or Parameter is changed here.
    """
    from peft.tuners.lora.layer import Linear
    if type(host) is torch.nn.Linear:
        saved = parameters(host)
        require(not host.training and saved and host.weight.dtype in (torch.bfloat16, torch.float32)
                and all(device == host.weight.device and device.type == 'cuda'
                        and dtype == host.weight.dtype and not grad
                        for _, _, device, dtype, grad in saved),
                'Original frozen resident plain native Linear required')
        return ('plain_linear', tuple((key, module, module.training)
                                      for key, module in host.named_modules()))
    require(type(host) is Linear and type(host.base_layer) is torch.nn.Linear,
            'Captured dense PEFT Linear required')
    require(not host.disable_adapters and not host.merged and not host.lora_variant
            and tuple(host.active_adapters) == ('pika_release',)
            and set(host.lora_A) == set(host.lora_B) == {'pika_release'},
            'One active unmerged original vanilla Pika adapter required')
    name = 'pika_release'
    require(type(host.lora_A[name]) is type(host.lora_B[name]) is torch.nn.Linear
            and host.lora_A[name].bias is None and host.lora_B[name].bias is None
            and host.lora_A[name].out_features == host.lora_B[name].in_features == 128,
            'Original frozen rank128 bias-free Pika A/B required')
    require(type(host.lora_dropout[name]) is torch.nn.Identity
            and not host.training and all(not module.training for module in host.modules()),
            'Original deterministic frozen native eval path required')
    saved = parameters(host)
    require(saved and all(device == host.weight.device and device.type == 'cuda'
            and dtype == torch.bfloat16 and not grad for _, _, device, dtype, grad in saved),
            'Original frozen resident BF16 base/Pika Parameters required')
    scaling = host.scaling[name]
    require(type(scaling) in (int, float) and math.isfinite(scaling),
            'Original finite native scalar scaling required')
    return ('vanilla_pika', tuple(host.active_adapters), tuple(host.merged_adapters), host.disable_adapters,
            tuple((key, module, module.training) for key, module in host.named_modules()),
            tuple(host.r.items()), tuple(host.lora_alpha.items()), tuple(host.scaling.items()),
            host.fan_in_fan_out, getattr(host, 'cast_input_dtype_enabled', None))


class OriginalPEFTHNDLRowChunks(AbstractContextManager):
    """Own only host.forward and its original HNDL cleanup identity field.

    Enter after original strict FULL restoration, keep live through complete
    backward, exit BEFORE FULL/RAW/state serialization. Original Session.step
    may close a failed session before this context exits; that legitimate branch
    close is recognized explicitly. Foreign dispatch replacement always refuses.
    """
    def __init__(self, network, *, chunk_rows=512):
        from minimax_h3_phase95 import adapter
        from peft.tuners.lora import layer as peft_layer
        require(hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest() == ADAPTER_SHA256,
                'Exact captured H3 HNDL branch Source required')
        require(hashlib.sha256((adapter.ROOT / 'pika-h3-flex-slider.hndl').read_bytes()).hexdigest() == GRAPH_SHA256,
                'Original two tokenwise bias-free graph lines required')
        require(hashlib.sha256(Path(peft_layer.__file__).read_bytes()).hexdigest() == PEFT_SHA256,
                'Exact captured PEFT dense forward Source required')
        require(type(chunk_rows) is int and chunk_rows == 512, 'Declared row512 bound only')
        branches = tuple(network.loras)
        require(branches and all(type(x) is adapter.HNDLProjection for x in branches),
                'Original live HNDL branches required')
        self.rows, self.closed, self.active = [], False, 0
        self.chunk_rows, self.calls, self.maximum_graph_rows = chunk_rows, 0, 0
        self.native_calls, self.maximum_native_rows = 0, 0
        try:
            for branch in branches:
                require(not branch._closed and branch._host.forward is branch._installed_forward,
                        'Original branch host and cleanup dispatch ownership required')
                require(branch.lora_down.bias is None and branch.lora_up.bias is None
                        and branch.lora_down.out_features == branch.lora_up.in_features == 8,
                        'Original rank8 down/up graph required')
                saved = parameters(branch)
                require(saved and all(device == torch.device('cuda:0') and dtype == torch.float32
                        for _, _, device, dtype, _ in saved), 'Original FP32 game0 Parameters only')
                native_forward = (torch.nn.Linear.forward if type(branch._host) is torch.nn.Linear
                                  else peft_layer.Linear.forward)
                require(getattr(branch.org_forward, '__self__', None) is branch._host
                        and getattr(branch.org_forward, '__func__', None) is native_forward,
                        'Original bound dense/PEFT forward, without another native owner, required')
                row = dict(branch=branch, host=branch._host, previous=branch._installed_forward,
                           parameters=saved, graph=branch.graph, org_forward=branch.org_forward,
                           native_had_forward=branch._had_forward,
                           native_previous_forward=branch._previous_forward,
                           native_parameters=parameters(branch._host),
                           native_signature=native_signature(branch._host))
                def dispatch(value, _row=row):
                    return self._forward(_row, value)
                row['owned'] = dispatch
                # HNDLProjection.close checks this exact identity. Updating it
                # lets original session exception cleanup restore its own host.
                branch._host.forward = dispatch
                branch._installed_forward = dispatch
                self.rows.append(row)
        except BaseException as primary:
            try:
                self.close()
            except BaseException as cleanup:
                raise BaseExceptionGroup('HNDL row owner installation/cleanup failed', [primary, cleanup]) from None
            raise

    def _native_forward(self, row, value):
        host = row['host']
        require(value.device == host.weight.device and value.shape[-1] == host.in_features,
                'Original resident native input device/width required')
        require(unchanged(host, row['native_parameters'])
                and native_signature(host) == row['native_signature'],
                'Original frozen native Parameters and execution state must remain exact')
        flat = value.reshape(-1, host.in_features)
        require(flat.shape[0] > 0, 'Nonempty full native token geometry required')
        chunks = []
        for start in range(0, flat.shape[0], self.chunk_rows):
            part = flat[start:start + self.chunk_rows]
            # Do not reimplement PEFT, merge A/B, change dtype, scale early or
            # substitute a functional linear. Its exact bound forward performs
            # base -> original cast/dropout/A/B -> scalar multiply -> native add
            # -> original output cast for each slice, with all original objects.
            result = row['org_forward'](part)
            require(result.shape == (part.shape[0], host.out_features)
                    and result.device == value.device and result.dtype == host.weight.dtype,
                    'Original native PEFT output per row slice required')
            chunks.append(result)
            self.native_calls += 1
            self.maximum_native_rows = max(self.maximum_native_rows, part.shape[0])
        result = torch.cat(chunks, dim=0).reshape(*value.shape[:-1], host.out_features)
        del chunks
        return result

    def _forward(self, row, value):
        branch = row['branch']
        require(not self.closed and not branch._closed
                and row['host'].forward is row['owned']
                and branch._installed_forward is row['owned'], 'Live owned HNDL row dispatch required')
        require(value.device.type == 'cuda' and value.ndim >= 2,
                'Original full native CUDA input geometry required')
        self.active += 1
        try:
            base = self._native_forward(row, value)
            if branch.multiplier == 0:
                return base
            require(base.device.type == 'cuda' and base.dtype in (torch.bfloat16, torch.float32)
                    and base.shape[:-1] == value.shape[:-1]
                    and base.shape[-1] == branch.lora_up.out_features,
                    'Original native base output dtype/device/full shape required')
            # Reshape BEFORE converting; only each <=512 input slice is moved
            # to original FP32 game0. No detach/copy of learned weights or RNG.
            flat = value.reshape(-1, branch.lora_down.in_features)
            require(flat.shape[0] > 0, 'Nonempty original native rows required')
            chunks = []
            scale = branch.multiplier * branch.scale
            for start in range(0, flat.shape[0], self.chunk_rows):
                part = flat[start:start + self.chunk_rows].to(branch.lora_down.weight)
                delta = row['graph'](value=part)
                self.maximum_graph_rows = max(self.maximum_graph_rows, part.shape[0])
                chunks.append(delta.to(base) * scale)
            delta = torch.cat(chunks, dim=0).reshape(*value.shape[:-1], branch.lora_up.out_features)
            del chunks
            self.calls += 1
            return base + delta
        finally:
            self.active -= 1

    def assert_owned(self):
        require(not self.closed and self.rows, 'Live HNDL row owner required')
        for row in self.rows:
            branch = row['branch']
            require(not branch._closed and row['host'].forward is row['owned']
                    and branch._installed_forward is row['owned']
                    and branch.graph is row['graph'] and branch.org_forward is row['org_forward']
                    and unchanged(branch, row['parameters'])
                    and unchanged(row['host'], row['native_parameters'])
                    and native_signature(row['host']) == row['native_signature'],
                    'Original native PEFT/HNDL objects/Parameters/dispatch changed')

    def witness(self):
        self.assert_owned()
        return dict(schema='H3-original-PEFT-HNDL-row512-execution-v1', profile=PROFILE,
                    original_branches=len(self.rows), maximum_chunk_rows=512,
                    actual_largest_graph_rows=self.maximum_graph_rows, actual_nonzero_branch_calls=self.calls,
                    original_graph_Source_sha256=GRAPH_SHA256, captured_adapter_Source_sha256=ADAPTER_SHA256,
                    captured_PEFT_Source_sha256=PEFT_SHA256, actual_largest_native_rows=self.maximum_native_rows,
                    actual_native_chunk_calls=self.native_calls, original_native_PEFT_Parameter_objects_preserved=True,
                    original_native_PEFT_forward_per_slice=True, original_full_native_PEFT_call_partitioned=True,
                    native_PEFT_branches=sum(row['native_signature'][0] == 'vanilla_pika' for row in self.rows),
                    native_plain_Linear_branches=sum(row['native_signature'][0] == 'plain_linear' for row in self.rows),
                    original_Parameter_names_objects_devices_dtypes_preserved=True,
                    native_base_operators_and_full_output_geometry_preserved=True, CPU_model_arithmetic=False,
                    cast_then_scale_then_add_preserved=True, full_FP32_up_output_materialized=False,
                    checkpoint_recompute_uses_same_live_dispatch=True,
                    bitwise_equivalence_claimed=False, full_model_fit_qualified=False)

    def close(self):
        if self.closed:
            return
        require(not self.active, 'Cannot restore an active HNDL graph call')
        errors = []
        for index in range(len(self.rows) - 1, -1, -1):
            row = self.rows[index]
            branch, host = row['branch'], row['host']
            try:
                require(branch.graph is row['graph'] and branch.org_forward is row['org_forward']
                        and unchanged(branch, row['parameters'])
                        and unchanged(host, row['native_parameters'])
                        and native_signature(host) == row['native_signature'],
                        'Original native PEFT/HNDL Parameters/graph changed')
                require(branch._installed_forward is row['owned'], 'HNDL cleanup identity changed')
                if branch._closed:
                    # Original Session.step exception cleanup already closed
                    # this exact branch. Do not reinstall a closed adapter.
                    require(('forward' in vars(host)) == row['native_had_forward']
                            and (not row['native_had_forward']
                                 or vars(host)['forward'] is row['native_previous_forward']),
                            'Original closed branch host was not restored exactly')
                else:
                    require(host.forward is row['owned'], 'Foreign host owner retained; cleanup pending')
                    host.forward = row['previous']
                branch._installed_forward = row['previous']
            except BaseException as error:
                errors.append(error)
            else:
                del self.rows[index]
        if errors:
            raise BaseExceptionGroup('HNDL row dispatch restoration incomplete', errors)
        self.closed = True

    def __exit__(self, kind, primary, traceback):
        try:
            self.close()
        except BaseException as cleanup:
            if primary is not None:
                raise BaseExceptionGroup('Original step and HNDL row restoration failed', [primary, cleanup]) from None
            raise
        return False
