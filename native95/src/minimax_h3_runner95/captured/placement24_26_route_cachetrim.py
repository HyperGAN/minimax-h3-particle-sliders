"""Execution-only original H3 24/26 placement after closed23/27 attempt51519 and outer-group override.

The runtime's captured plan and all artifact signatures remain historical24/26.
Only the two exact original constructor calls receive a private execution plan.
No original file, class method, Parameter, optimizer, RNG or model math changes.
"""
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import time
from types import MethodType
from contextlib import contextmanager

PROFILE='h3-execution-only-main24-26-refiner1-1-game0-group5-route-cachetrim-v1'
ACTUAL_PLAN_PROFILE='h3-native-allCUDA-DiT24-26-refiner1-1-PG0-group5-sync-unwind-FF512-route-cachetrim-execution-v1'
HISTORICAL='h3-native-allCUDA-DiT24-26-refiner1-1-PG0-group5-sync-unwind-FF512-v3'
MARKER='_h3_owned_rebalanced24_route_cachetrim_execution'

def require(value,message):
    if not value:raise RuntimeError(message)

def exact_source(module,pin):
    path=Path(module.__file__).absolute();raw=path.read_bytes()
    require(str(path)==pin['path'] and len(raw)==pin['bytes']
        and hashlib.sha256(raw).hexdigest()==pin['sha256'],'Exact original constructor Source required')

class OwnedRebalancedExecution:
    """Own only two module-symbol overrides; actual instances stay original.

    Keep no strong instance/model references. The original runtime/adapter own
    native route hooks and checkpoint dispatch and close those original owners.
    Foreign symbol/marker replacement is refused without overwriting it.
    """
    def __init__(self,native,placement,grouped,*,sources,publish,publish_allocator):
        self.closed=self.closing=False;self._pending=[];self._markers=[]
        self._records=[];self.publish=publish;self.publish_allocator=publish_allocator;self._allocator_records=[]
        require(callable(publish_allocator),'Explicit owned allocator metadata writer required')
        require(callable(publish),'Explicit owned metadata writer required')
        for name,module in (('native',native),('placement',placement),('grouped',grouped)):
            exact_source(module,sources[name])
            require(not hasattr(module,MARKER),'Foreign rebalance owner already present')
        self.native,self.placement,self.grouped=native,placement,grouped
        original_placement=placement.OriginalDiTPlacement
        original_grouped=grouped.GroupedCheckpointDispatch
        require(native.OriginalDiTPlacement is original_placement,'Original native placement alias changed')
        self.historical=placement.Placement().validate()
        require(self.historical.profile==HISTORICAL and self.historical.main_boundary==24
            and self.historical.refiner_boundary==1 and self.historical.outer_group_size==5
            and self.historical.native_main_blocks==50 and self.historical.native_refiners==2
            and self.historical.game_device==self.historical.entry_device==self.historical.first_device=='cuda:0'
            and self.historical.second_device=='cuda:1','Exact historical full50+2 two-CUDA plan required')
        self.execution=replace(self.historical,main_boundary=24,profile=ACTUAL_PLAN_PROFILE)
        def checked_plan(plan):
            self.assert_owned()
            require(type(plan)is type(self.historical) and asdict(plan)==asdict(self.historical),
                'Captured runtime plan must remain historical24/26, never retag artifacts')
            return self.execution
        def placement_call(transformer,plan):
            execution=checked_plan(plan)
            instance=original_placement.__new__(original_placement)
            try:
                with self._owned_instance_route(instance,original_placement,transformer):
                    original_placement.__init__(instance,transformer,execution)
            except BaseException as primary:
                # The original constructor already cleans its partial hooks.
                # Also close if constructor success was followed by owned-route
                # restoration failure; no returned instance may escape unowned.
                if getattr(instance,'_handles',None):
                    try:instance.close()
                    except BaseException as cleanup:raise BaseExceptionGroup('Original placement construction and cleanup failed',[primary,cleanup])
                raise
            # Original constructor routes every complete native module; no
            # extra tensor move/copy or CPU arithmetic is introduced here.
            try:
                counts=[]
                for index,block in enumerate(transformer.transformer_blocks):
                    devices={str(value.device)for value in block.parameters()}
                    expected='cuda:0'if index<24 else'cuda:1'
                    require(devices=={expected},'Actual native main-block route differs')
                    counts.append(dict(index=index,device=expected))
                for index,block in enumerate(transformer.token_refiner.refiner_blocks):
                    require({str(value.device)for value in block.parameters()}=={'cuda:'+str(index)},
                        'Both original refiners must remain one per CUDA device')
                self._record('original_placement_constructor_return',dict(main_blocks=counts,
                    actual_main_counts=[24,26],actual_refiner_counts=[1,1],original_class_instance=True))
            except BaseException as primary:
                try:instance.close()
                except BaseException as cleanup:raise BaseExceptionGroup('Original placement result and cleanup failed',[primary,cleanup])
                raise
            return instance
        def grouped_call(transformer,network,plan):
            execution=checked_plan(plan)
            instance=original_grouped(transformer,network,execution)
            try:
                groups=[dict(indices=list(group.indices),device=str(group.device))for group in instance.groups]
                require([i for group in groups for i in group['indices']]==list(range(50))
                    and all(1<=len(group['indices'])<=5 for group in groups)
                    and all(all(('cuda:0'if i<24 else'cuda:1')==group['device']for i in group['indices'])for group in groups),
                    'Actual groups must partition all50 blocks entirely within actual24/26 stages')
                self._record('original_grouped_constructor_return',dict(actual_groups=groups,
                    stage_bridges_outside_outer_groups=True,original_class_instance=True))
            except BaseException as primary:
                try:instance.close()
                except BaseException as cleanup:raise BaseExceptionGroup('Original grouped result and cleanup failed',[primary,cleanup])
                raise
            return instance
        self._functions=(placement_call,grouped_call)
        try:
            for module in (native,placement,grouped):
                setattr(module,MARKER,self);self._markers.append(module)
            for module,name,old,new in ((native,'OriginalDiTPlacement',original_placement,placement_call),
                    (grouped,'GroupedCheckpointDispatch',original_grouped,grouped_call)):
                require(getattr(module,name)is old,'Original constructor binding changed')
                setattr(module,name,new);self._pending.append((module,name,old,new))
            self.assert_owned()
        except BaseException as primary:
            try:self.close()
            except BaseException as cleanup:raise BaseExceptionGroup('Rebalance installation and cleanup failed',[primary,cleanup])
            raise
    def _allocator_snapshot(self):
        # These are sequential process-allocator/driver metadata queries, not
        # tensor reads or a simultaneous cross-device/foreign-memory envelope.
        cuda=self.placement.torch.cuda;start=time.monotonic_ns();rows=[]
        for index in(0,1):
            before=time.monotonic_ns()
            allocated=cuda.memory_allocated(index);reserved=cuda.memory_reserved(index)
            free,total=cuda.mem_get_info(index)
            rows.append(dict(device='cuda:'+str(index),monotonic_start_ns=before,
                allocated_bytes=int(allocated),reserved_bytes=int(reserved),
                driver_free_bytes=int(free),driver_total_bytes=int(total),monotonic_end_ns=time.monotonic_ns()))
        return dict(monotonic_start_ns=start,devices=rows,monotonic_end_ns=time.monotonic_ns(),
            queries_are_sequential_not_atomic=True,intrinsic_capacity_or_foreign_budget_inferred=False)
    def _allocator_record(self,event,route_index,route_kind,route_name,device,**details):
        value=dict(schema='H3-original-placement-route-allocator-observation-v1',profile=PROFILE,
            event=event,route_index=route_index,route_kind=route_kind,route_name=route_name,
            original_route_target_device=str(device),allocator=self._allocator_snapshot(),
            math_Parameters_dtypes_and_RNG_unchanged=True,whole_model_fit_claim=False,**details)
        self._allocator_records.append(value)
        self.publish_allocator(len(self._allocator_records),deepcopy(value))
    @contextmanager
    def _owned_instance_route(self,instance,original,transformer):
        # Exact original class/constructor/Parameter moves, one private instance
        # attribute only; never patch a class or native math/kernel operation.
        had='_route'in vars(instance);saved=vars(instance).get('_route')
        require(not had,'Preexisting instance route override refused')
        original_route=original._route
        routes={id(block):('main','transformer_blocks.'+str(i))for i,block in enumerate(transformer.transformer_blocks)}
        routes.update({id(block):('refiner','token_refiner.refiner_blocks.'+str(i))for i,block in enumerate(transformer.token_refiner.refiner_blocks)})
        routes[id(transformer.token_refiner.final_norm)]=('norm','token_refiner.final_norm')
        routes[id(transformer.norm_out)]=('norm','norm_out')
        ordinal=0
        def routed(bound,module,device):
            nonlocal ordinal
            self.assert_owned();require(bound is instance,'Exact original placement instance required')
            result=original_route(instance,module,device)  # Exactly once, after original block.to returned.
            require(id(module)in routes,'Unknown original constructor route boundary')
            kind,name=routes[id(module)];ordinal+=1
            self._allocator_record('after_original_route_before_empty_cache',ordinal,kind,name,device)
            cuda=self.placement.torch.cuda;current=cuda.current_device()
            try:
                # Empty only this worker's unused allocator cache on both owned
                # devices. Active tensors/weights are not freed or transformed.
                for index in(0,1):
                    with cuda.device(index):cuda.empty_cache()
            except BaseException as error:
                self._allocator_record('empty_cache_failed',ordinal,kind,name,device,
                    error=dict(type=type(error).__name__,message=str(error)))
                raise
            require(cuda.current_device()==current,'CUDA device context must be restored')
            self._allocator_record('after_empty_cache',ordinal,kind,name,device,
                empty_cache_device_indices=[0,1],device_context_restored=True)
            return result
        wrapper=MethodType(routed,instance);setattr(instance,'_route',wrapper)
        def restore():
            require(vars(instance).get('_route')is wrapper,'Foreign instance route retained')
            if had:setattr(instance,'_route',saved)
            else:delattr(instance,'_route')
        try:yield
        except BaseException as primary:
            try:restore()
            except BaseException as cleanup:raise BaseExceptionGroup('Original placement work and instance route cleanup failed',[primary,cleanup])
            raise
        else:restore()
    def _record(self,event,details):
        record=dict(schema='H3-original-constructor-rebalanced-execution-observation-v1',profile=PROFILE,
            event=event,historical_runtime_plan=asdict(self.historical),actual_execution_plan=asdict(self.execution),
            runtime_factory_FULL_cache_initialization_identity_rewritten=False,**details)
        self._records.append(record);self.publish(len(self._records),deepcopy(record))
    def assert_owned(self):
        require(not self.closed and not self.closing,'Rebalance constructor owner closed/cleanup pending')
        require(all(getattr(module,name)is new for module,name,old,new in self._pending)
            and all(vars(module).get(MARKER)is self for module in self._markers),
            'Execution constructor/marker ownership changed')
    def witness(self):
        self.assert_owned()
        return dict(profile=PROFILE,historical_runtime_plan=asdict(self.historical),
            actual_execution_plan=asdict(self.execution),observations=deepcopy(self._records),
            actual_group_partitions_changed=False,
            changed_from_previous23_27_execution=True,stage_bridges_outside_outer_groups=True,
            allocator_route_observations=deepcopy(self._allocator_records),
            only_unused_process_allocator_cache_trimmed=True,
            original_math_Parameters_dtypes_initializer_and_RNG_unchanged=True,
            original_factory_and_FULL_signature_not_retagged=True,whole_model_fit_claim=False)
    def close(self):
        if self.closed:return
        self.closing=True;errors=[];failed=[]
        for module,name,old,new in reversed(self._pending):
            if getattr(module,name)is not new:
                errors.append(RuntimeError('Foreign constructor binding retained:'+module.__name__+'.'+name));failed.append((module,name,old,new));continue
            setattr(module,name,old)
        self._pending=list(reversed(failed))
        if not self._pending:
            failed=[]
            for module in self._markers:
                if vars(module).get(MARKER)is not self:
                    errors.append(RuntimeError('Foreign rebalance marker retained:'+module.__name__));failed.append(module);continue
                delattr(module,MARKER)
            self._markers=failed
        if errors:raise BaseExceptionGroup('Rebalance constructor cleanup pending',errors)
        self.closed,self.closing=True,False
    def __enter__(self):self.assert_owned();return self
    def __exit__(self,kind,primary,traceback):
        try:self.close()
        except BaseException as cleanup:
            if primary is not None:raise BaseExceptionGroup('H3 rebalance work and cleanup failed',[primary,cleanup])
            raise
