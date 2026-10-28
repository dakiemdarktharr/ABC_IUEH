"""Stateless zero/constant-angular-velocity references; no data fitting."""
import numpy as np
from .geometry import exp_so3, inverse_wxyz, log_so3, multiply_wxyz, normalize_wxyz


def predict_constant_velocity(context, bandwidth=1):
    context = np.asarray(context)
    if context.shape != (1, 400, 4) or context.dtype != np.float32:
        raise ValueError("context must be float32 [1,400,4]")
    if bandwidth not in (0, 1, 5, 20):
        raise ValueError("fixed expert bank uses bandwidths 0,1,5,20")
    q = normalize_wxyz(context[0])
    if bandwidth == 0:
        velocity = np.zeros(3)
    else:
        relative = multiply_wxyz(q[-1], inverse_wxyz(q[-1 - bandwidth]))
        velocity = log_so3(relative) / bandwidth
    prediction = multiply_wxyz(exp_so3(np.arange(1, 21)[:, None] * velocity), q[-1])
    return normalize_wxyz(prediction).astype(np.float32)[None]
