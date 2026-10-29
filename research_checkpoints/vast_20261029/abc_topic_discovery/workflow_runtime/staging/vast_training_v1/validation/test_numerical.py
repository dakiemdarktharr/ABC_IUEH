"""Generated-only numerical tests; import torch/numpy only when explicitly run."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from . import evaluator

class NumericalTests(unittest.TestCase):
    def test_quaternion_sign_and_right_angle(self):
        import torch
        target = torch.zeros((2, 20, 4), dtype=torch.float32); target[:,:,0] = 1
        pred = target.clone(); pred[0,:,0] = -1
        pred[1,:,0] = 2 ** -0.5; pred[1,:,1] = 2 ** -0.5
        values = evaluator.endpoint20_errors(pred, target, torch=torch)
        self.assertAlmostEqual(values[0].item(), 0, places=8)
        self.assertAlmostEqual(values[1].item(), 90, places=4)

    def test_synthetic_full_user_scope_and_output(self):
        import numpy as np
        import torch
        class Model:
            def to(self, device): return self
            def __call__(self, context): return context[:, -1:, :].repeat(1, 20, 1)
        class Segments:
            fold_id=0; role=evaluator.ROLE; expected_origin_counts=(1,1,1,1,1)
            origin_identity_sha256='a'*64; read_receipt=None
            def __iter__(self):
                for i in range(5):
                    poses=np.zeros((420,4), dtype=np.float32); poses[:,0]=1
                    yield i, i, poses
                self.read_receipt={'synthetic': True}
        models = {(arm,seed):Model() for arm in evaluator.ARMS for seed in evaluator.SEEDS}
        with tempfile.TemporaryDirectory() as tmp, patch.object(evaluator, 'validate_checkpoints', return_value=(models, {})), patch.object(evaluator, 'FixedValidationSegments', Segments):
            output=Path(tmp)/'report.json'
            report=evaluator.evaluate_fold(fold_id=0,checkpoints={},fit_identities={},segments=Segments(),output_path=output)
            self.assertEqual(len(report['users']),5)
            self.assertEqual(report['pooled_origins'][f'{evaluator.ARMS[0]}:{evaluator.SEEDS[0]}']['mean_degrees'],0)
            self.assertTrue(output.is_file())
            self.assertNotIn('context', output.read_text())

if __name__ == '__main__': unittest.main()
