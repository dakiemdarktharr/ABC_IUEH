"""Check checkpoint timing thresholds without data or CUDA."""
import unittest
import trainer


class CheckpointPolicyTests(unittest.TestCase):
    def test_steps_time_and_final_update(self):
        due = trainer._checkpoint_due
        self.assertFalse(due(999, 0, 299.9, 0.0, 5000))
        self.assertTrue(due(1000, 0, 1.0, 0.0, 5000))
        self.assertTrue(due(10, 0, 300.0, 0.0, 5000))
        self.assertTrue(due(5000, 4999, 1.0, 0.0, 5000))
        self.assertFalse(due(1001, 2, 1.0, 0.0, 5000))
        self.assertTrue(due(1002, 2, 1.0, 0.0, 5000))


if __name__ == '__main__':
    unittest.main()
