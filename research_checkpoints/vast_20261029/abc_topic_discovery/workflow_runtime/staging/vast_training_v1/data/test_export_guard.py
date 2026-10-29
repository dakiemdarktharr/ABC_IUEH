"""Synthetic monitor checks; no child process or human data is opened."""
import unittest
from unittest.mock import patch

from . import export_guard


class FakeChild:
    def __init__(self, return_value=None):
        self.value = return_value
        self.killed = False

    def poll(self):
        return self.value

    def kill(self):
        self.killed = True
        self.value = -9

    def wait(self):
        return self.value


class MonitorTests(unittest.TestCase):
    def test_does_not_start_below_floor(self):
        with self.assertRaises(MemoryError):
            export_guard.monitor(["not-started"], read_available=lambda: 0,
                                 popen=lambda *args, **kwargs: self.fail("started"))

    def test_kills_child_if_floor_is_crossed(self):
        child = FakeChild()
        readings = iter((export_guard.MIN_AVAILABLE + 1, export_guard.MIN_AVAILABLE - 1))
        with self.assertRaises(MemoryError):
            export_guard.monitor(["synthetic"], read_available=lambda: next(readings),
                                 popen=lambda *args, **kwargs: child)
        self.assertTrue(child.killed)

    def test_child_exit_and_memory_probe_failure(self):
        child = FakeChild(0)
        self.assertEqual(export_guard.monitor(["synthetic"], read_available=lambda: export_guard.MIN_AVAILABLE,
                                             popen=lambda *args, **kwargs: child), 0)
        child = FakeChild()
        with patch.object(export_guard.time, "sleep", side_effect=RuntimeError("probe failed")):
            with self.assertRaises(RuntimeError):
                export_guard.monitor(["synthetic"], read_available=lambda: export_guard.MIN_AVAILABLE,
                                     popen=lambda *args, **kwargs: child)
        self.assertTrue(child.killed)


if __name__ == "__main__":
    unittest.main()
