"""Run-owned original LTX construction/storage overlay; no work at import.

The captured factory/manifest identity remains historical input authority.
Actual CPU storage and effective loader callable are recorded separately.
Caller must authenticate the actual fresh or strict FULL session entry before steps.
"""
from contextlib import AbstractContextManager
import gc,hashlib,inspect,importlib.util,json,sys,threading
from pathlib import Path

ROOT=Path(__file__).absolute().parent
PROFILE='LTX25-original48-DiT-CUDA1-CPU-block16-31-and-inactive-decoders-storage-v1'
INDICES=tuple(range(16,32))
INACTIVE=('vae','audio_vae','vocoder')
BLOCK_PARAMETERS=386674880



def require(value,message):
    if not value:raise RuntimeError(message)


def source_record(path):
    p=Path(path).absolute();raw=p.read_bytes()
    return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def validate_sources():
    from .sources import storage_sources
    return storage_sources()


def storage_type():
    # External helper loading never assumes caller cwd/sys.path. The frozen
    # helper's own relative Torch companion is authenticated before execution.
    pins=validate_sources();pin=pins['execution_frozen_storage']
    name='_LTX25_source_bound_official_frozen_storage_v1'
    module=sys.modules.get(name)
    if module is None:
        spec=importlib.util.spec_from_file_location(name,pin['path'])
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module
        try:spec.loader.exec_module(module)
        except BaseException:sys.modules.pop(name,None);raise
    require(source_record(module.__file__)==pin,'Exact run-owned frozen storage module required')
    return module.OfficialLTXFrozenBlockStorage


class FrozenDiTPlacement:
    """All original math on CUDA1; selected frozen storage remains CPU at rest."""
    def __init__(self,transformer,*,device='cuda:1',_fixture_indices=None):
        import torch
        from diffusers import LTX2VideoTransformer3DModel
        OfficialLTXFrozenBlockStorage=storage_type()
        require(type(transformer)is LTX2VideoTransformer3DModel,'Exact original native LTX class required')
        require(getattr(transformer,'gradient_checkpointing',False)is False,'Install before official checkpoint enable/strict restore')
        indices=INDICES if _fixture_indices is None else tuple(_fixture_indices)
        require(indices and len(indices)==len(set(indices))and all(type(i)is int and 0<=i<len(transformer.transformer_blocks)for i in indices),'Explicit original whole block indices only')
        if _fixture_indices is None:
            require(len(transformer.transformer_blocks)==48 and str(device)=='cuda:1','Full original48 blocks/CUDA1 compute required')
            require(all(sum(p.numel()for p in transformer.transformer_blocks[i].parameters())==BLOCK_PARAMETERS for i in range(48)),
                'Source-derived full original per-block size must match before placement; no smaller substitute')
            require(all(p.dtype is torch.bfloat16 for p in transformer.parameters()),'Original whole BF16 DiT precision required')
        require(not transformer.training and all(not p.requires_grad and p.device.type=='cpu'for p in transformer.parameters()),'Only original frozen CPU construction may enter storage placement')
        require(all(v.device.type=='cpu'for v in transformer.buffers()),'Capture original CPU buffers before device placement')
        self.transformer,self.device,self.indices=transformer,torch.device(device),indices
        self._storage=[];self.closed=False
        parameters=tuple(transformer.named_parameters());buffers=tuple(transformer.named_buffers())
        stored_ids=set()
        try:
            for index in indices:
                block=transformer.transformer_blocks[index]
                values=tuple(block.parameters())+tuple(block.buffers())
                require(not any(id(v)in stored_ids for v in values),'No cross-block shared storage ownership')
                stored_ids.update(id(v)for v in values)
                self._storage.append(OfficialLTXFrozenBlockStorage(block,self.device,transformer=transformer,label=f'LTX.original.block.{index}'))
            # Includes root Parameters/buffers outside registered children. No
            # module.to() is allowed to move selected blocks after attachment.
            for _,value in parameters+buffers:
                if id(value)not in stored_ids:value.data=value.data.to(device=self.device,non_blocking=False)
            self._baseline=tuple((name,value,id(value),str(value.device),str(value.dtype),value._version)
                for name,value in parameters+buffers)
            self.assert_resting()
        except BaseException as primary:
            try:self.close()
            except BaseException as cleanup:raise BaseExceptionGroup('original LTX placement and cleanup failed',[primary,cleanup])
            raise

    def assert_resting(self):
        require(not self.closed,'Closed frozen LTX placement')
        current=tuple(self.transformer.named_parameters())+tuple(self.transformer.named_buffers())
        require(len(current)==len(self._baseline)and all(name==old_name and value is old and id(value)==identity
            and str(value.device)==device and str(value.dtype)==dtype and value._version==version
            for(name,value),(old_name,old,identity,device,dtype,version)in zip(current,self._baseline)),
            'Original frozen LTX Parameter/buffer identity/device/dtype/version changed at rest')
        for owner in self._storage:owner.assert_offloaded()

    def witness(self):
        self.assert_resting()
        return dict(profile=PROFILE,actual_compute_device=str(self.device),actual_CPU_stored_original_indices=list(self.indices),
            original_block_order_unchanged=True,original_parameters_replaced=False,precision_changed=False,
            stored_parameters=sum(v.numel()for owner in self._storage for _,v in owner.parameters),
            stored_payload_bytes=sum(v.numel()*v.element_size()for owner in self._storage for _,v in owner.parameters+owner.buffers),
            block_calls=[dict(label=x.label,calls=x.calls,completed=x.completed_calls,unwound=x.unwound_calls)for x in self._storage],
            hook_composition_source=source_record(validate_sources()['execution_frozen_storage']['path']),checkpoint_mode='official ModelMixin nonreentrant per-block checkpoint')

    def close(self):
        if self.closed:return
        errors=[]
        for index in range(len(self._storage)-1,-1,-1):
            try:self._storage[index].close()
            except BaseException as error:errors.append(error)
            else:del self._storage[index]
        if errors:raise BaseExceptionGroup('original LTX storage cleanup remains pending',errors)
        self.transformer=None;self._baseline=();self.closed=True


class ScopedOriginalLoader(AbstractContextManager):
    """One actual NativeRuntime constructor, then restore exact loader attribute.

    Usage: with owner: runtime=NativeRuntime(original options).
    Keep owner.placement alive through original fresh/strict FULL entry and complete steps,
    close session first, then owner.close(), then runtime.close().
    No automatic retry or second constructor is permitted.
    """
    def __init__(self,model_dir,*,record=None):
        self.model_dir=str(Path(model_dir).absolute())
        self._record=record;self.calls=0;self.placement=None;self._active=False;self._restored=False
        self._thread=threading.get_ident();self.events=[];self.inactive={};self._inactive_baseline=()

    def emit(self,value):
        self.events.append(value)
        if self._record is not None:self._record(value)

    def __enter__(self):
        require(not self._active and self.calls==0,'One nonnested original constructor only')
        self.sources=validate_sources()
        import ps_ltx25_native95.component_loader as loader
        require(source_record(loader.__file__)==self.sources['component_loader'],'Captured loader module Source required')
        self.loader=loader;self.original=loader.load_pipeline
        require(self.original.__module__==loader.__name__ and self.original.__name__=='load_pipeline'
            and source_record(inspect.getsourcefile(self.original))==self.sources['component_loader'],
            'Exact original captured callable before explicit scoped replacement')
        self.installed=self._construct;loader.load_pipeline=self.installed;self._active=True
        try:
            self.emit(dict(status='new_execution_loader_installed_for_one_original_constructor',source=source_record(__file__),
                captured_factory_source=self.sources['component_loader'],captured_nominal_loading_profile=loader.LOAD_PROFILE,
                actual_execution_profile=PROFILE,unchanged_execution_claim=False,strict_session_entry_authenticated=None))
        except BaseException:
            if loader.load_pipeline is self.installed:loader.load_pipeline=self.original
            self._active=False;self._restored=loader.load_pipeline is self.original
            raise
        return self

    def _construct(self,directory,*,dtype,text_device,dit_device):
        import torch
        require(self._active and threading.get_ident()==self._thread and self.calls==0,'Exact one synchronous owned constructor only')
        require(self.loader.load_pipeline is self.installed,'Run-owned loader replacement interference')
        require(str(Path(directory).absolute())==self.model_dir and dtype is torch.bfloat16
            and str(text_device)=='cuda:0'and str(dit_device)=='cuda:1','Original full model/BF16/text0/DiT1 options unchanged')
        self.calls+=1;root=Path(directory);classes=self.loader.component_classes();placed={};module=None
        try:
            for name in self.loader.MODEL_COMPONENTS:
                options=dict(local_files_only=True,low_cpu_mem_usage=True,device_map=None,use_safetensors=True)
                options['dtype'if name=='text_encoder'else'torch_dtype']=dtype
                module=classes[name].from_pretrained(str(root/name),**options)
                module.requires_grad_(False).eval()
                if name=='transformer':
                    # Same original dtype conversion, before any CUDA block
                    # residency. No full-transformer CUDA .to() transient.
                    module.to(device='cpu',dtype=dtype)
                    self.placement=FrozenDiTPlacement(module)
                elif name in INACTIVE:
                    # Same original module/dtype; inactive decoder objects stay
                    # CPU from construction. No decoder computation is allowed.
                    original_ids=tuple(id(p)for p in module.parameters())
                    module.to(device='cpu',dtype=dtype)
                    require(original_ids==tuple(id(p)for p in module.parameters()),'Original inactive Parameter objects retained')
                    self.inactive[name]=module
                else:module.to(device=text_device,dtype=dtype)
                placed[name]=module;module=None;gc.collect()
            self._inactive_baseline=tuple((name+'.'+key,value,str(value.device),str(value.dtype),value._version)
                for name,module in self.inactive.items()for key,value in tuple(module.named_parameters())+tuple(module.named_buffers()))
            self.assert_inactive()
            placed['scheduler']=classes['scheduler'].from_pretrained(str(root/'scheduler'),local_files_only=True)
            placed['tokenizer']=classes['tokenizer'].from_pretrained(str(root/'tokenizer'),local_files_only=True)
            pipe=classes['pipeline'](**placed,prompt_enhancer=None,processor=None,duration_head=None)
            require(all(getattr(pipe,name)is owner for name,owner in placed.items()),'Complete original component ownership retained')
            self.emit(dict(status='original_full_components_constructed_with_explicit_CPU_storage',captured_original_factory_source=self.sources['component_loader'],
                actual_loader_source=source_record(__file__),actual_storage=self.placement.witness(),
                captured_identity_is_not_an_unchanged_execution_claim=True,strict_session_entry_authenticated=None))
            return pipe
        except BaseException:
            module=None;placed.clear();gc.collect();raise

    def __exit__(self,kind,value,traceback):
        interference=self.loader.load_pipeline is not self.installed
        if not interference:self.loader.load_pipeline=self.original
        self._active=False;self._restored=not interference and self.loader.load_pipeline is self.original
        errors=[]if kind is None else[value]
        try:
            self.emit(dict(status='exact_captured_loader_callable_restored_before_strict_session_entry',restored=self._restored,
                original_callable_identity_restored=self._restored,interference=interference,constructor_calls=self.calls,
                constructor_returned=kind is None,actual_execution_profile=PROFILE))
        except BaseException as error:errors.append(error)
        if kind is not None or errors or not self._restored:
            try:self.close()
            except BaseException as cleanup:errors.append(cleanup)
        if not self._restored:errors.append(RuntimeError('Run-owned loader restoration failed; no checkpoint/state identity claim'))
        if len(errors)>1:raise BaseExceptionGroup('original constructor/record/storage restoration errors retained',errors)
        if errors and kind is None:raise errors[0]
        return False

    def assert_inactive(self):
        require(tuple(self.inactive)==INACTIVE,'All and only original inactive decoder objects required')
        current=tuple((name+'.'+key,value)for name,module in self.inactive.items()
            for key,value in tuple(module.named_parameters())+tuple(module.named_buffers()))
        require(len(current)==len(self._inactive_baseline)and all(name==oldname and value is old
            and str(value.device)==device=='cpu'and str(value.dtype)==dtype and value._version==version
            for(name,value),(oldname,old,device,dtype,version)in zip(current,self._inactive_baseline)),
            'Inactive original Parameter/buffer identity/device/dtype/version changed')
        require(all(not module.training and all(not p.requires_grad for p in module.parameters())for module in self.inactive.values()),
            'Original inactive decoders remain frozen/eval')

    def inactive_witness(self):
        self.assert_inactive()
        return {name:dict(original_class=type(module).__module__+'.'+type(module).__qualname__,
            actual_device='cpu',dtype_policy='original constructor bfloat16, no CPU arithmetic',
            parameter_bytes=sum(p.numel()*p.element_size()for p in module.parameters()),
            buffer_bytes=sum(v.numel()*v.element_size()for v in module.buffers()),
            parameter_objects=[dict(name=n,object_id=id(p),dtype=str(p.dtype),device=str(p.device))for n,p in module.named_parameters()],
            active_during_training=False,decoder_forward_calls_authorized=0)
            for name,module in self.inactive.items()}

    def witness(self,runtime):
        require(self._restored and self.calls==1 and self.placement is not None
            and runtime.transformer is self.placement.transformer,'Original constructor owner and exact loader restoration required')
        self.assert_inactive()
        return dict(schema='LTX25-explicit-original-factory-frozen-storage-execution-v1',source=source_record(__file__),
            captured_factory_source=self.sources['component_loader'],captured_native_source=self.sources['native'],
            captured_manifest_loading_profile=runtime.identity['loading_profile'],
            actual_execution_profile=PROFILE,actual_storage=self.placement.witness(),
            actual_inactive_CPU_components=self.inactive_witness(),actual_CPU_arithmetic=False,
            captured_factory_identity_preserved=True,actual_execution_is_historically_identical=False,
            exact_loader_attribute_restored=True,strict_original_session_entry_before_update_required=True,
            strict_session_equality_not_implied_by_this_witness=True,native_geometry_dtype_math_or_optimizer_changed=False,
            storage_helper_teacher_initializer_or_sampling_calls=0,fit_or_bitwise_CUDA_claim=False)

    def close(self):
        require(not self._active,'Restore constructor-scoped loader before owner close')
        if self.placement is not None:self.placement.close();self.placement=None
        self.inactive.clear();self._inactive_baseline=()
