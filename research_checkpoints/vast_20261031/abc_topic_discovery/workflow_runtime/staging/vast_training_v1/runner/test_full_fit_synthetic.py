"""Exercise the actual fit loop and resume using only synthetic rotations."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
import trainer
from fit_api import PilotRecipe
from checkpoint import load_checkpoint
from test_checkpoint_resume import _assert_tree_equal


class Lease:
    def assert_live(self):
        return None


class Bundle:
    total_origins = 1
    identity_sha256 = 'a' * 64
    identity_summary = {'fold_id': 0, 'role': 'outer_fit10',
                        'purpose': 'ordered_motion_exploratory_fit_only_micro_pilot'}
    binding = Lease()
    def assert_active(self): self.binding.assert_live()
    def origin_for_rank(self, rank): return (0, 0)
    def sample_window(self, segment, origin):
        return struct.pack('<4f', 1, 0, 0, 0) * 400, struct.pack('<4f', 1, 0, 0, 0) * 20
    def close(self): pass
    def abort_and_close(self): pass


class Controller:
    capability_id = 'synthetic-test'
    def __init__(self): self.commits = []
    def authorize_training(self, **kwargs): pass
    def checkpoint_committed(self, **kwargs): self.commits.append(kwargs)
    def resume_restored(self, *, receipt_path, checkpoint_sha256, completed_updates):
        receipt = json.loads(Path(receipt_path).read_text())
        if receipt['checkpoint_sha256'] != checkpoint_sha256 or receipt['loaded_global_step'] != completed_updates:
            raise AssertionError('resume receipt mismatch')


class Gate:
    budget_policy = {'soft_training_stop_usd': 4.0, 'final_charge_reserve_usd': 12.0, 'campaign_total_usd': 16.0}
    def __init__(self, allowed): self.allowed = allowed
    def before_update(self, *, remaining_updates):
        if self.allowed <= 0: return False
        self.allowed -= 1
        return True


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.75))
    def forward(self, context):
        return (context[:, -1:, :] * self.scale).expand(-1, 20, -1)


class ToyLoss:
    @staticmethod
    def endpoint20_geodesic_loss(prediction, target):
        return ((prediction - target) ** 2).mean()


class FullFitTests(unittest.TestCase):
    def test_actual_loop_resumes_without_losing_optimizer_or_sampler(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(trainer, 'require_ram_floor'), \
             patch.object(trainer, '_runtime_binding', side_effect=lambda torch, np, device: {
                 'test_fixture_only': True, 'device_type': device.type,
                 'torch': str(torch.__version__), 'numpy': np.__version__}), \
             patch.object(trainer, '_load_model', side_effect=lambda *args: (ToyModel(), ToyLoss(), {})):
            root = Path(tmp)
            def run(directory, count, resume=None, digest=None, receipt=None):
                controller = Controller()
                with self.assertRaises(trainer.BudgetStop):
                    trainer.run_fit(bundle=Bundle(), root_controller=controller, budget_gate=Gate(count),
                        recipe=PilotRecipe(), checkpoint_directory=directory, resume_checkpoint=resume,
                        resume_checkpoint_sha256=digest, resume_receipt_path=receipt,
                        device_name='cpu', require_cuda=False, expected_run_id='a'*32)
                self.assertEqual(len(controller.commits), 1)
                return controller.commits[0]
            uninterrupted = run(root / 'full', 4)
            partial = run(root / 'partial', 2)
            with self.assertRaises(trainer.CheckpointMismatchError):
                trainer.run_fit(bundle=Bundle(), root_controller=Controller(), budget_gate=Gate(2),
                    recipe=PilotRecipe(), checkpoint_directory=root / 'partial',
                    resume_checkpoint=Path(partial['checkpoint_path']),
                    resume_checkpoint_sha256=partial['checkpoint_sha256'],
                    resume_receipt_path=root / 'receipt' / 'wrong-id.json',
                    device_name='cpu', require_cuda=False, expected_run_id='b'*32)
            resumed = run(root / 'partial', 2, Path(partial['checkpoint_path']),
                          partial['checkpoint_sha256'], root / 'receipt' / 'resume.json')
            full = load_checkpoint(Path(uninterrupted['checkpoint_path']),
                                   expected_sha256=uninterrupted['checkpoint_sha256'], torch=torch)
            final = load_checkpoint(Path(resumed['checkpoint_path']),
                                    expected_sha256=resumed['checkpoint_sha256'], torch=torch)
            for key in ('completed_updates', 'sampler_cursor_draws', 'sampler_draw_hash',
                        'sampler_pcg64_state', 'model_state_dict', 'optimizer_state_dict'):
                _assert_tree_equal(self, full[key], final[key])


if __name__ == '__main__':
    unittest.main()
