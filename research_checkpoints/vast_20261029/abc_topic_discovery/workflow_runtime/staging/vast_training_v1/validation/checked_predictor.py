"""Draft CPU inference wrapper that detects common stateful predictors.

This is not a sandbox and cannot prevent a model from modifying globals, files,
or network state and then hiding it. Production remains disabled until exact
model/checkpoint/code binding and process isolation have independent review.
"""
from __future__ import annotations
import hashlib
import json
import math
from .two_stage import CELLS


class PredictorError(ValueError): pass


def _freeze_value(value):
    if value is None or type(value) in (str,int,bool): return value
    if type(value) is float and math.isfinite(value): return value
    if isinstance(value,(tuple,list)):
        return [_freeze_value(item) for item in value]
    if isinstance(value,(set,frozenset)):
        return sorted(_freeze_value(item) for item in value)
    if isinstance(value,dict):
        if not all(type(key) is str for key in value): raise PredictorError('unsafe module attribute key')
        return {key:_freeze_value(item) for key,item in sorted(value.items())}
    raise PredictorError('unsupported stateful module attribute or hook')


def _model_digest(model):
    h=hashlib.sha256()
    for name,module in model.named_modules():
        if module.training: raise PredictorError('model must be in eval mode')
        attributes={key:_freeze_value(value) for key,value in vars(module).items()
                    if key not in ('_parameters','_buffers','_modules')}
        h.update(json.dumps([name,attributes],sort_keys=True,separators=(',',':'),allow_nan=False).encode())
    for name,tensor in sorted(model.state_dict().items()):
        if tensor.device.type != 'cpu': raise PredictorError('inference state must be CPU only')
        value=tensor.detach().contiguous()
        h.update(json.dumps([name,str(value.dtype),list(value.shape)],separators=(',',':')).encode())
        h.update(value.numpy().tobytes(order='C'))
    return h.hexdigest()


class CheckedCPUPredictor:
    """Run six already independently verified models on one past context.

    Detects mutations to registered tensors, serializable module attributes and
    Torch CPU RNG. Does not make untrusted Python code stateless or isolated.
    """
    def __init__(self, models, *, torch):
        if type(models) is not dict or set(models) != set(CELLS):
            raise PredictorError('exact six fixed cells required')
        self._models=dict(models); self._torch=torch
        self._model_digests={cell:_model_digest(model) for cell,model in models.items()}

    def __call__(self, context):
        if type(context) is not bytes or len(context) != 400*4*4:
            raise PredictorError('expected one 400-row float32 context')
        torch=self._torch
        rng_before=torch.get_rng_state().clone()
        x=torch.frombuffer(bytearray(context),dtype=torch.float32).reshape(1,400,4)
        result={}
        with torch.inference_mode():
            for cell in CELLS:
                model=self._models[cell]
                if _model_digest(model) != self._model_digests[cell]: raise PredictorError('model state changed before inference')
                prediction=model(x)
                if tuple(prediction.shape) != (1,20,4): raise PredictorError('model output shape differs from endpoint contract')
                result[cell]=tuple(float(value) for value in prediction[0,19].tolist())
                if _model_digest(model) != self._model_digests[cell]: raise PredictorError('model state changed during inference')
            if not torch.equal(rng_before,torch.get_rng_state()): raise PredictorError('predictor changed Torch CPU RNG')
        return result
