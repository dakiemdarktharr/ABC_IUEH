"""Count executed Conv/Linear and fixed-einsum MACs for JEPA-TCN width 64.

This is a declared subset of arithmetic, not a target-device latency measure.
"""
import argparse
import json
import math

import torch

from run_inference_v2 import load_model


def count_macs(model_dir):
    handle = load_model(model_dir, device='cpu')
    if (handle.metadata['model'] != 'jepa_v2' or
            handle.metadata['architecture'] != {'width': 64, 'latent_dim': 128, 'use_gravity': True}):
        raise ValueError('This calculation is scoped to the recorded JEPA-TCN configuration')
    model = handle.model.eval()
    if tuple(model.features.fit_weights.shape) != (3, 2, 32) or tuple(model.features.desp_slope_weights.shape) != (3, 64):
        raise ValueError('Fixed-einsum dimensions differ from the documented calculation')
    totals = {'conv1d_macs': 0, 'linear_macs': 0}
    hooks = []

    def count(module, inputs, output):
        if isinstance(module, torch.nn.Conv1d):
            totals['conv1d_macs'] += output.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)
        else:
            totals['linear_macs'] += output.numel() * module.in_features

    for module in model.modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)):
            hooks.append(module.register_forward_hook(count))
    try:
        context = torch.zeros(1, 400, 4, dtype=torch.float32)
        context[..., 0] = 1.0
        with torch.inference_mode():
            prediction = model(context)
        if prediction.shape != (1, 20, 4) or not torch.isfinite(prediction).all():
            raise ValueError('Unexpected model output')
    finally:
        for hook in hooks:
            hook.remove()
    totals['einsum_macs'] = 3 * 2 * 32 * 3 + 3 * 64 * 3
    totals['conv_linear_einsum_macs'] = sum(totals.values())
    totals['scope'] = ('Executed Conv1d, Linear and two fixed Einsum contractions; '
                       'excludes bias, elementwise operations, normalization, pooling, '
                       'other reductions and nonlinear/trigonometric operators.')
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_dir', required=True)
    args = parser.parse_args()
    print(json.dumps(count_macs(args.model_dir), indent=2))


if __name__ == '__main__':
    main()
