"""Export a hash-verified private BOXRR V2 model to a static FP32 ONNX graph.

Example: python tools/boxrr_v2/export_onnx.py --model_dir /private/deployment/jepa_v2_w64 --output /private/model_fp32.onnx
The output contains trained weights. Keep it private and follow dataset terms.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from run_inference_v2 import _aligned_unit, load_model


def export_onnx(model_dir: str, output: str, *, tolerance_deg: float = 0.002) -> dict:
    """Export one static 400→20 graph and verify synthetic CPU numerical parity."""
    destination = Path(output)
    if destination.suffix.lower() != '.onnx':
        raise ValueError('Output must use the .onnx suffix')
    temporary = destination.with_name(destination.name + '.partial')
    if destination.exists() or temporary.exists():
        raise FileExistsError('Output or temporary export already exists')
    if not destination.parent.is_dir():
        raise ValueError('Output directory must already exist')
    if not np.isfinite(tolerance_deg) or tolerance_deg <= 0:
        raise ValueError('Parity tolerance must be positive and finite')

    handle = load_model(model_dir, device='cpu')
    model = handle.model.eval()
    context = np.zeros((1, 400, 4), dtype=np.float32)
    context[..., 0] = 1.0
    try:
        torch.onnx.export(model, (torch.from_numpy(context),), str(temporary),
                          input_names=['context'], output_names=['prediction'],
                          export_params=True, opset_version=17,
                          do_constant_folding=True, dynamo=False)
        graph = onnx.load(temporary)
        onnx.checker.check_model(graph, full_check=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(temporary), options, providers=['CPUExecutionProvider'])
        if (session.get_inputs()[0].shape != [1, 400, 4] or
                session.get_outputs()[0].shape != [1, 20, 4] or
                session.get_inputs()[0].type != 'tensor(float)' or
                session.get_outputs()[0].type != 'tensor(float)'):
            raise ValueError('Unexpected ONNX interface')

        generator = np.random.default_rng(20261110)
        max_deg = 0.0
        for i in range(12):
            if i == 0:
                q = context[0]
            else:
                # Sample small, normalized quaternion perturbations. This checks
                # the interface only; it is not human motion or an accuracy test.
                q = np.zeros((400, 4), dtype=np.float32)
                q[:, 0] = 1.0
                q[:, 1:] = np.cumsum(generator.normal(0, 0.001, (400, 3)), axis=0)
                q /= np.linalg.norm(q, axis=-1, keepdims=True)
            observed = _aligned_unit(q)[None]
            with torch.inference_mode():
                expected = model(torch.from_numpy(observed)).numpy()
            actual = session.run(['prediction'], {'context': observed})[0]
            if not np.isfinite(actual).all() or not np.isfinite(expected).all():
                raise ValueError('ONNX output is nonfinite')
            a, b = actual.astype(np.float64), expected.astype(np.float64)
            norm_a = np.linalg.norm(a, axis=-1, keepdims=True)
            norm_b = np.linalg.norm(b, axis=-1, keepdims=True)
            if np.any(norm_a < 1e-8) or np.any(norm_b < 1e-8):
                raise ValueError('Model produced a zero quaternion')
            a /= norm_a
            b /= norm_b
            distance = np.minimum(np.linalg.norm(a - b, axis=-1), np.linalg.norm(a + b, axis=-1))
            angle = np.degrees(4 * np.arcsin(np.minimum(1, distance / 2)))
            max_deg = max(max_deg, float(angle.max()))
        if max_deg > tolerance_deg:
            raise ValueError(f'ONNX parity {max_deg}° exceeds {tolerance_deg}°')
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        size = temporary.stat().st_size
        # Avoid a silent overwrite when another process creates the destination.
        with destination.open('xb') as target, temporary.open('rb') as source:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
        return {'output': str(destination), 'sha256': digest, 'bytes': size,
                'opset': 17, 'synthetic_contexts': 12,
                'max_so3_disagreement_deg': max_deg, 'tolerance_deg': tolerance_deg,
                'target_xr_latency_measured': False}
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(export_onnx(args.model_dir, args.output), indent=2))


if __name__ == '__main__':
    main()
