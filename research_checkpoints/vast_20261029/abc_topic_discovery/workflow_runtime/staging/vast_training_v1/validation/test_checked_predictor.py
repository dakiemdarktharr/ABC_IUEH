"""Synthetic CPU model mutation checks; no data, CUDA, or network."""
import struct
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
try:
    import torch
except ImportError:
    torch=None
from .checked_predictor import CheckedCPUPredictor, PredictorError
from . import two_stage as stage
from .test_two_stage_stdlib import FakeSource


@unittest.skipIf(torch is None,'PyTorch unavailable')
class PredictorTests(unittest.TestCase):
    def test_eval_fixed_state_and_adversarial_counter(self):
        class Stable(torch.nn.Module):
            def __init__(self): super().__init__(); self.counter=0
            def forward(self,x): return x[:,-1:,:].repeat(1,20,1)
        class Stateful(Stable):
            def forward(self,x): self.counter+=1; return super().forward(x)
        data=struct.pack('<4f',1,0,0,0)*400
        models={cell:Stable().eval() for cell in stage.CELLS}
        safe=CheckedCPUPredictor(models,torch=torch)
        self.assertEqual(safe(data),safe(data))
        models[stage.CELLS[0]]=Stateful().eval()
        unsafe=CheckedCPUPredictor(models,torch=torch)
        with self.assertRaises(PredictorError): unsafe(data)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp); source=FakeSource(root/'pred')
            with patch.object(stage,'_available_ram_bytes',return_value=4*1024**3), self.assertRaises(PredictorError):
                stage.freeze_predictions(source=source,predict=unsafe,prediction_path=root/'pred',lock_path=root/'lock',
                    checkpoint_hashes={cell:'a'*64 for cell in stage.CELLS},predictor_code_sha256='b'*64,ram_guard=lambda:None)
            self.assertFalse(source.target_requested); self.assertFalse((root/'lock').exists())


if __name__ == '__main__': unittest.main()
