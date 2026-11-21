"""Run-owned H3 HNDL row512 execution; captured factory and weights stay exact.

The original two bias-free linears are tokenwise. Only their batch of flattened
rows is partitioned. Each original FP32 graph result is cast to the native base
output device/dtype and multiplied in the ORIGINAL order before concatenation.
No full-token FP32 up result is constructed. GEMM/gradient accumulation grouping
can differ: this is an explicit execution profile, never a bitwise claim.
"""
from contextlib import AbstractContextManager
import hashlib
from pathlib import Path
import torch

PROFILE = 'h3-original-HNDL-flat-row512-native-cast-scaled-cat-owner-v1'
ADAPTER_SHA256 = '43793d7cf3745e954bc7eeda917ce934cf0b7e320a53219f6e35ae1f617c9f5a'
GRAPH_SHA256 = 'e32094eb55b9178e8c42d7bf483a5915593d5fe6ae3cbca7478559cf80f8dc4c'


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


class OriginalHNDLRowChunks(AbstractContextManager):
    """Own only host.forward and its original HNDL cleanup identity field.

    Enter after original strict FULL restoration, keep live through complete
    backward, exit BEFORE FULL/RAW/state serialization. Original Session.step
    may close a failed session before this context exits; that legitimate branch
    close is recognized explicitly. Foreign dispatch replacement always refuses.
    """
    def __init__(self, network, *, chunk_rows=512):
        from minimax_h3_phase95 import adapter
        require(hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest() == ADAPTER_SHA256,
                'Exact captured H3 HNDL branch Source required')
        require(hashlib.sha256((adapter.ROOT / 'pika-h3-flex-slider.hndl').read_bytes()).hexdigest() == GRAPH_SHA256,
                'Original two tokenwise bias-free graph lines required')
        require(type(chunk_rows) is int and chunk_rows == 512, 'Declared row512 bound only')
        branches = tuple(network.loras)
        require(branches and all(type(x) is adapter.HNDLProjection for x in branches),
                'Original live HNDL branches required')
        self.rows, self.closed, self.active = [], False, 0
        self.chunk_rows, self.calls, self.maximum_graph_rows = chunk_rows, 0, 0
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
                row = dict(branch=branch, host=branch._host, previous=branch._installed_forward,
                           parameters=saved, graph=branch.graph, org_forward=branch.org_forward,
                           native_had_forward=branch._had_forward,
                           native_previous_forward=branch._previous_forward)
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

    def _forward(self, row, value):
        branch = row['branch']
        require(not self.closed and not branch._closed
                and row['host'].forward is row['owned']
                and branch._installed_forward is row['owned'], 'Live owned HNDL row dispatch required')
        require(value.device.type == 'cuda' and value.ndim >= 2,
                'Original full native CUDA input geometry required')
        self.active += 1
        try:
            base = row['org_forward'](value)
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
                    and unchanged(branch, row['parameters']), 'Original HNDL objects/Parameters/dispatch changed')

    def witness(self):
        self.assert_owned()
        return dict(schema='H3-original-HNDL-row512-execution-v1', profile=PROFILE,
                    original_branches=len(self.rows), maximum_chunk_rows=512,
                    actual_largest_graph_rows=self.maximum_graph_rows, actual_nonzero_branch_calls=self.calls,
                    original_graph_Source_sha256=GRAPH_SHA256, captured_adapter_Source_sha256=ADAPTER_SHA256,
                    original_Parameter_names_objects_devices_dtypes_preserved=True,
                    native_base_call_and_full_geometry_preserved=True, CPU_model_arithmetic=False,
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
                        and unchanged(branch, row['parameters']), 'Original HNDL Parameters/graph changed')
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
