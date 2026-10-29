import unittest
from .source import FixedValidationSegments

class DenyTests(unittest.TestCase):
    def test_source_cannot_read_before_prediction_lock_implementation(self):
        source = object.__new__(FixedValidationSegments)
        with self.assertRaises(PermissionError):
            iter(source)
