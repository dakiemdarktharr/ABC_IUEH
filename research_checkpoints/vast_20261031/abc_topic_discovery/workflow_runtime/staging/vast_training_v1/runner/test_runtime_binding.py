"""Synthetic runtime compatibility: no Torch, CUDA, network, or recordings."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from . import trainer


class RuntimeBindingTests(unittest.TestCase):
    def torch(self, capability=(8,6), version='2.9.1+cu128', vram=24):
        cuda=SimpleNamespace(get_device_capability=lambda device: capability,
            get_device_properties=lambda device: SimpleNamespace(total_memory=vram,name='synthetic'))
        return SimpleNamespace(__version__=version, version=SimpleNamespace(cuda='12.8'),
            cuda=cuda,backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda:90100)),
            are_deterministic_algorithms_enabled=lambda:True)

    def test_compatible_hosts_and_incompatible_version_or_capability(self):
        numpy=SimpleNamespace(__version__='2.4.6'); device=SimpleNamespace(type='cuda')
        with patch.object(trainer.platform,'python_version_tuple',return_value=('3','11','9')), \
             patch.object(trainer.platform,'python_version',return_value='3.11.9'), \
             patch.object(trainer.platform,'python_implementation',return_value='CPython'), \
             patch.object(trainer.platform,'system',return_value='Linux'), \
             patch.object(trainer.platform,'machine',return_value='x86_64'), \
             patch.object(trainer.platform,'platform',side_effect=AssertionError('kernel must not be pinned')), \
             patch.object(trainer.platform,'release',side_effect=['kernel-a','kernel-b']), \
             patch.object(trainer,'package_version',return_value='3.16.0'), \
             patch.object(trainer,'sha256_file',return_value='a'*64):
            self.assertEqual(trainer.platform.release(), 'kernel-a')
            first=trainer._runtime_binding(self.torch(vram=24),numpy,device)
            self.assertEqual(trainer.platform.release(), 'kernel-b')
            second=trainer._runtime_binding(self.torch(vram=16),numpy,device)
            self.assertEqual(trainer._runtime_sha256(first),trainer._runtime_sha256(second))
            other=trainer._runtime_binding(self.torch(capability=(8,9)),numpy,device)
            self.assertNotEqual(trainer._runtime_sha256(first),trainer._runtime_sha256(other))
            with self.assertRaises(trainer.TrainingError):
                trainer._runtime_binding(self.torch(version='2.10.0'),numpy,device)
            self.assertNotIn('device_total_memory',first)


if __name__ == '__main__': unittest.main()
