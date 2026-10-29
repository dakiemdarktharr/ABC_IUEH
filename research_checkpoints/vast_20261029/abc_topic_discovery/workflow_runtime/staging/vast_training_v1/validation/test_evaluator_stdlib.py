"""Stdlib-only synthetic checkpoint scope checks; no data or torch imports."""
import hashlib
import json
import unittest
from types import SimpleNamespace
from . import evaluator

class FakeModel:
    def load_state_dict(self, value, strict): self.weights = value
    def eval(self): pass

class FakeRecipe:
    def __init__(self, fold_id, training_seed, arm_id): self.data = dict(fold_id=fold_id, training_seed=training_seed, arm_id=arm_id)
    def as_dict(self): return self.data

class CheckpointTests(unittest.TestCase):
    def test_all_completed_checkpoint_cells_required(self):
        canonical = lambda obj: json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()
        trainer = SimpleNamespace(ScreenRecipe=FakeRecipe, _canonical_json=canonical,
            _model_binding=lambda arm: ({}, 'model-' + arm), _runner_sha256=lambda: 'runner',
            THREE_ARM_DRAFT_SHA256='draft', _load_model=lambda arm, seed: (FakeModel(), None, None))
        cells = {(f, arm, seed): (f'{f},{arm},{seed}', 'digest')
                 for f in range(3) for arm in evaluator.ARMS for seed in evaluator.SEEDS}
        class FakeCheckpoint:
            partial = False
            @staticmethod
            def load_checkpoint(path, expected_sha256, torch):
                f, arm, seed = str(path).split(',')
                f, seed = int(f), int(seed)
                recipe = FakeRecipe(f, seed, arm).as_dict()
                return dict(recipe=recipe, recipe_sha256=hashlib.sha256(canonical(recipe)).hexdigest(),
                    draft_sha256='draft', model_sha256='model-'+arm, runner_sha256='runner',
                    completed_updates=4999 if FakeCheckpoint.partial and f == 2 else 5000,
                    sampler_cursor_draws=640000, data_identity_sha256=f'fold-{f}', model_state_dict={})
        models, hashes = evaluator.validate_checkpoints(0, cells, fit_identities={0:'fold-0',1:'fold-1',2:'fold-2'}, torch=None, trainer=trainer, checkpoint=FakeCheckpoint)
        self.assertEqual(len(models), 6)
        self.assertEqual(len(hashes), 6)
        with self.assertRaises(ValueError): evaluator.validate_checkpoints(0, dict(list(cells.items())[:-1]), fit_identities={0:'fold-0',1:'fold-1',2:'fold-2'}, torch=None, trainer=trainer, checkpoint=FakeCheckpoint)
        with self.assertRaises(ValueError):
            evaluator.validate_checkpoints(0, cells, fit_identities={0:'wrong',1:'fold-1',2:'fold-2'}, torch=None, trainer=trainer, checkpoint=FakeCheckpoint)
        FakeCheckpoint.partial = True
        with self.assertRaises(ValueError): evaluator.validate_checkpoints(0, cells, fit_identities={0:'fold-0',1:'fold-1',2:'fold-2'}, torch=None, trainer=trainer, checkpoint=FakeCheckpoint)

if __name__ == '__main__': unittest.main()
