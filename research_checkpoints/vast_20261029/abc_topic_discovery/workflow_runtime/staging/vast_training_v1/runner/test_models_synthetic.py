"""CPU-only generated rotations: load each pinned arm and take one update."""
import unittest

import torch

import trainer


class ModelSmokeTests(unittest.TestCase):
    def test_three_pinned_models_have_finite_forward_and_backward(self):
        torch.set_num_threads(1)
        angles = torch.arange(420, dtype=torch.float32) * 0.0005
        rotations = torch.zeros((1, 420, 4), dtype=torch.float32)
        rotations[0, :, 0] = torch.cos(angles / 2)
        rotations[0, :, 1] = torch.sin(angles / 2)
        context, target = rotations[:, :400], rotations[:, 400:]
        for arm in trainer.SCREEN_ARMS:
            with self.subTest(arm=arm):
                model, loss_module, _ = trainer._load_model(arm, 20261026)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                prediction = model(context)
                self.assertEqual(tuple(prediction.shape), (1, 20, 4))
                self.assertTrue(torch.isfinite(prediction).all().item())
                loss = loss_module.endpoint20_geodesic_loss(prediction, target)
                self.assertTrue(torch.isfinite(loss).item())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(g).all().item() for g in gradients))
                optimizer.step()


if __name__ == "__main__":
    unittest.main()
