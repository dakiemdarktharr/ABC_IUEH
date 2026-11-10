"""Direct relative-SO(3) convolutional BOXRR adaptation, without an expert bank.

Only the supplied observed prefix enters inference. There is no learned user
state, external control, target smoothing, autoregression or fitted scaler.
Rotation vectors use full physical angles, in radians, in the newest pose frame.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .boxrr_models import (
    conjugate_wxyz,
    exp_so3,
    log_so3,
    multiply_wxyz,
    normalize_quaternion,
)


class RelativeSO3Features(nn.Module):
    """Six rotation columns and three common-frame adjacent increments.

``r_t = inverse(q_last) * q_t`` maps each observed pose to the newest
frame. ``Log(r_t * inverse(r_(t-1)))`` is a spatial increment in that same
frame, rather than an increment in a different body frame at every sample.
The convolution receives radians/second at the fixed 100 Hz sampling rate;
the separate linear branch receives radians per observed interval.
"""

    def forward(self, context):
        if context.ndim != 3 or context.shape[1:] != (400, 4):
            raise ValueError("context must have shape [B,400,4]")
        q = normalize_quaternion(context)
        # A zero quaternion has no orientation; use identity for a finite fallback.
        identity = torch.zeros_like(q)
        identity[..., 0] = 1.
        q = torch.where(q.abs().sum(-1, keepdim=True) > 0, q, identity)
        last = q[:, -1]
        relative = multiply_wxyz(conjugate_wxyz(last[:, None]), q)
        w, x, y, z = relative.unbind(-1)
        rotation6d = torch.stack((
            1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y),
            2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x),
        ), -1)
        increments = log_so3(multiply_wxyz(
            relative[:, 1:], conjugate_wxyz(relative[:, :-1])))
        increments = F.pad(increments, (0, 0, 1, 0))
        tokens = torch.cat((rotation6d, 100*increments), -1).transpose(1, 2)
        return last, tokens, increments


class CausalDepthwiseBlock(nn.Module):
    """Residual kernel-7 depthwise convolution with pointwise channel mixing."""

    def __init__(self, width, dilation):
        super().__init__()
        self.left_padding = 6*dilation
        self.depthwise = nn.Conv1d(width, width, 7, dilation=dilation, groups=width)
        self.pointwise = nn.Sequential(
            nn.Conv1d(width, 2*width, 1), nn.SiLU(),
            nn.Conv1d(2*width, width, 1),
        )

    def forward(self, tokens):
        local = F.silu(self.depthwise(F.pad(tokens, (self.left_padding, 0))))
        return tokens + self.pointwise(local)


class DirectSO3Conv(nn.Module):
    """Stateless [B,400,4] -> [B,20,4] direct unit-wxyz forecaster.

The learned time readout sees all 400 observed positions. Both nonlinear and
linear branches predict displacement directly, without horizon rescaling or
clipping. Early heads remain connected to the shared trunk for joint path loss.
The linear branch starts as CAV1 but every one of its lag weights is trainable.
Small nonzero channel-head weights permit trunk gradients on the first update.
"""

    def __init__(self, width=64):
        super().__init__()
        if not isinstance(width, int) or width not in (32, 64):
            raise ValueError("width must be 32 or 64")
        self.width = width
        self.features = RelativeSO3Features()
        self.stem = nn.Conv1d(9, width, 1)
        self.blocks = nn.Sequential(*(
            CausalDepthwiseBlock(width, dilation) for dilation in (1, 2, 4)))
        self.time_head = nn.Linear(400, 20)
        self.early_head = nn.Linear(width, 3)
        self.endpoint_head = nn.Linear(width, 3)
        self.linear_branch = nn.Linear(400, 20, bias=False)
        nn.init.zeros_(self.time_head.bias)
        for head in (self.early_head, self.endpoint_head):
            nn.init.normal_(head.weight, mean=0., std=1e-3)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.linear_branch.weight)
        with torch.no_grad():
            self.linear_branch.weight[:, -1].copy_(torch.arange(1., 21.))

    def forward(self, context):
        last, tokens, increments = self.features(context)
        hidden = self.blocks(F.silu(self.stem(tokens)))
        horizon = self.time_head(hidden).transpose(1, 2)
        nonlinear = torch.cat((self.early_head(horizon[:, :19]),
                               self.endpoint_head(horizon[:, 19:])), 1)
        linear = self.linear_branch(increments.transpose(1, 2)).transpose(1, 2)
        return normalize_quaternion(multiply_wxyz(
            last[:, None], exp_so3(nonlinear+linear)))

    def get_config(self):
        return {
            "name": "direct_so3_conv", "width": self.width,
            "input_shape": [400, 4], "output_shape": [20, 4],
            "quaternion_order": "wxyz", "sampling_hz": 100,
            "features": "relative_rotation_6d_and_newest_frame_increments",
            "conv_increment_units": "radians_per_second",
            "linear_increment_units": "radians_per_interval",
            "kernel_size": 7, "dilations": [1, 2, 4],
            "early_heads_detached": False, "linear_initialization": "CAV1",
            "target_filtering": False, "stateful": False,
        }
