"""Owner-only synchronous H3 diagnostic metadata; no model operation replacement.

CUDA_LAUNCH_BLOCKING must be set before framework initialization by the envelope.
Hooks preserve inputs/outputs and synchronize both admitted devices. The first
operation/fence failure is saved before poisoned-context cleanup can obscure it.
Normal nonreentrant early-stop unwinds are control flow, never called failures.
"""
from contextlib import AbstractContextManager
from pathlib import Path
import os
import sys
import torch

PROFILE = 'H3-original-FULL0-row512-launchblocking1-owned-sync-metadata-step1-v1'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def tensor_metadata(value):
    if torch.is_tensor(value):
        return dict(shape=list(value.shape), stride=list(value.stride()),
                    dtype=str(value.dtype), device=str(value.device), requires_grad=value.requires_grad)
    if isinstance(value, (tuple, list)):
        return [tensor_metadata(x) for x in value]
    if isinstance(value, dict):
        return {str(k): tensor_metadata(v) for k, v in value.items()}
    if value is None or type(value) in (bool, int, float, str):
        return value
    return dict(python_type=type(value).__module__ + '.' + type(value).__qualname__)


def parameter_metadata(module):
    return {name: dict(shape=list(value.shape), stride=list(value.stride()),
                       dtype=str(value.dtype), device=str(value.device), requires_grad=value.requires_grad)
            for name, value in module.named_parameters(recurse=False)}


class OwnedSynchronousDiagnostic(AbstractContextManager):
    """Observe existing block/attention/HNDL-host/PEFT-base/A/B modules only.

    It neither changes a forward callable nor adds a saved-tensor hook. Original
    placement, Pika, row execution and checkpoint contexts continue untouched.
    Always-call hooks balance ordinary checkpoint early-stop unwinds without
    issuing CUDA work during exception unwinding. Cleanup removes only exact
    owned hook IDs and performs no CUDA operation.
    """
    def __init__(self, runtime, output, *, write_json):
        require(os.environ.get('CUDA_LAUNCH_BLOCKING') == '1', 'Pre-framework launch blocking required')
        require(torch.cuda.device_count() == 2, 'Only exact two admitted CUDA devices')
        self.root, self.write = Path(output), write_json
        self.root.mkdir(exist_ok=False)
        self.records, self.closed, self.failed = [], False, False
        self.events, self.first_failure, self.last_completed = 0, None, None
        self.branches = {id(branch._host): branch for branch in runtime.network.loras}
        selected, seen = [], set()
        def add(name, module):
            if id(module) not in seen:
                seen.add(id(module)); selected.append((name, module))
        for name, module in runtime.transformer.named_modules():
            parts = name.split('.')
            main = parts[0] == 'transformer_blocks'
            refiner = parts[:2] == ['token_refiner', 'refiner_blocks']
            block = (main and len(parts) == 2 and parts[1].isdigit()) or (
                refiner and len(parts) == 3 and parts[2].isdigit())
            attention = (main and len(parts) == 3 and parts[1].isdigit() and parts[2] == 'attn') or (
                refiner and len(parts) == 4 and parts[2].isdigit() and parts[3] == 'attn')
            if block or attention:
                add(name, module)
        for branch in runtime.network.loras:
            name, host = branch.lora_name, branch._host
            add(name + '.HNDL_host_before_org_forward', host)
            base = getattr(host, 'base_layer', host)
            add(name + '.Pika_base_layer', base)
            lora_a, lora_b = getattr(host, 'lora_A', {}), getattr(host, 'lora_B', {})
            for adapter in getattr(host, 'active_adapters', ()):
                if adapter in lora_a:
                    add(name + '.Pika_lora_A.' + adapter, lora_a[adapter])
                if adapter in lora_b:
                    add(name + '.Pika_lora_B.' + adapter, lora_b[adapter])
        try:
            for name, module in selected:
                def before(owner, args, kwargs, _name=name):
                    self._before(_name, owner, args, kwargs)
                def after(owner, args, kwargs, output, _name=name):
                    self._after(_name, owner, output)
                pre = module.register_forward_pre_hook(before, with_kwargs=True)
                post = module.register_forward_hook(after, with_kwargs=True, always_call=True)
                self.records.append((module, pre, before, post, after))
            self.write(self.root / 'Source-scope.json', dict(schema='H3-owned-sync-diagnostic-scope-v1',
                profile=PROFILE, CUDA_LAUNCH_BLOCKING='1', selected_modules=[name for name, _ in selected],
                observed_modules=len(selected), tensors_copied=False, forwards_replaced=False,
                saved_tensor_hooks_added=False, model_weights_or_parameters_modified=False,
                original_checkpoint_determinism_early_stop_RNG_unchanged=True,
                synchronization_can_mask_timing_race=True, full_model_fit_claim=False))
        except BaseException as primary:
            try:
                self.close()
            except BaseException as cleanup:
                raise BaseExceptionGroup('Diagnostic installation/owned cleanup failed', [primary, cleanup]) from None
            raise

    def _sync(self):
        for device in (0, 1):
            torch.cuda.synchronize(device)

    def _streams(self):
        return dict(current_device=torch.cuda.current_device(),
                    current_streams=[dict(device=i, cuda_stream=int(torch.cuda.current_stream(i).cuda_stream))
                                     for i in (0, 1)])

    def _fail(self, name, stage, error, metadata):
        if self.failed:
            return
        self.failed = True
        self.first_failure = dict(schema='H3-first-synchronous-operation-failure-v1',
            event=self.events, module=name, stage=stage,
            error=dict(type=type(error).__module__ + '.' + type(error).__qualname__, message=str(error)[:8192]),
            metadata=metadata, last_completed=self.last_completed,
            original_attempt_not_normalized=True, no_update_or_output_claim=True)
        # CPU-only atomic metadata write. Never query/synchronize CUDA again
        # here: its context can now be poisoned and secondary errors misleading.
        self.write(self.root / 'first-failure.json', self.first_failure)

    def _before(self, name, owner, args, kwargs):
        if self.failed:
            return
        self.events += 1
        metadata = dict(inputs=tensor_metadata((args, kwargs)), direct_parameters=parameter_metadata(owner))
        branch = self.branches.get(id(owner))
        if branch is not None:
            adapters = tuple(getattr(owner, 'active_adapters', ()))
            lora_a, lora_b = getattr(owner, 'lora_A', {}), getattr(owner, 'lora_B', {})
            metadata.update(HNDL_branch=branch.lora_name, HNDL_down=parameter_metadata(branch.lora_down),
                            HNDL_up=parameter_metadata(branch.lora_up),
                            Pika_base=parameter_metadata(getattr(owner, 'base_layer', owner)),
                            active_Pika_adapters=list(adapters),
                            Pika_A={key: parameter_metadata(lora_a[key]) for key in adapters if key in lora_a},
                            Pika_B={key: parameter_metadata(lora_b[key]) for key in adapters if key in lora_b})
        self.write(self.root / 'last-event.json', dict(event=self.events, module=name,
                   stage='before_both_device_sync', metadata=metadata), replace=True)
        try:
            self._sync()
            metadata.update(self._streams())
            self.write(self.root / 'last-event.json', dict(event=self.events, module=name,
                       stage='enter_original_call_after_successful_sync', metadata=metadata), replace=True)
        except BaseException as error:
            self._fail(name, 'before_original_call_or_stream_metadata', error, metadata)
            raise

    def _after(self, name, owner, output):
        if self.failed:
            return
        primary = sys.exception()
        if output is None and primary is not None:
            if type(primary).__module__ == 'torch.utils.checkpoint' and type(primary).__name__ == '_StopRecomputationError':
                return  # Original nonreentrant early-stop; no new CUDA call.
            self._fail(name, 'original_call_raised_before_return', primary, dict(direct_parameters=parameter_metadata(owner)))
            return
        try:
            self._sync()
            self.last_completed = dict(event=self.events, module=name, output=tensor_metadata(output), **self._streams())
            self.write(self.root / 'last-completed.json', self.last_completed, replace=True)
        except BaseException as error:
            self._fail(name, 'after_original_call_both_device_sync', error, dict(output=tensor_metadata(output)))
            raise

    def witness(self):
        require(not self.closed and not self.failed, 'Successful live synchronous diagnostic required')
        return dict(schema='H3-owned-synchronous-step1-diagnostic-execution-v1', profile=PROFILE,
                    CUDA_LAUNCH_BLOCKING='1', observed_modules=len(self.records), hook_events=self.events,
                    last_completed=self.last_completed, first_failure=None,
                    tensors_or_weights_copied=False, model_forward_arithmetic_replaced=False,
                    original_attention_backend_and_checkpoint_determinism_RNG_unchanged=True,
                    no_CPU_model_arithmetic=True, full_model_fit_claim=False)

    def close(self):
        if self.closed:
            return
        errors = []
        for index in range(len(self.records) - 1, -1, -1):
            module, pre, before, post, after = self.records[index]
            if module._forward_pre_hooks.get(pre.id) is not before or module._forward_hooks.get(post.id) is not after:
                errors.append(RuntimeError('Foreign diagnostic hook ownership change; retained'))
                continue
            post.remove(); pre.remove(); del self.records[index]
        if errors:
            raise BaseExceptionGroup('Synchronous diagnostic cleanup incomplete', errors)
        self.branches = None
        self.closed = True

    def __exit__(self, kind, primary, traceback):
        try:
            self.close()
        except BaseException as cleanup:
            if primary is not None:
                raise BaseExceptionGroup('Original step and diagnostic cleanup failed', [primary, cleanup]) from None
            raise
        return False
