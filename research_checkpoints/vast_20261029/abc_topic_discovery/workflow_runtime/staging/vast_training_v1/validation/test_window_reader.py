"""Synthetic-only validation slice mapping, with no HDF5 or source recordings."""
import struct
import unittest
from .window_reader import iter_segment_contexts, iter_segment_endpoints


class WindowReaderTests(unittest.TestCase):
    def test_context_and_endpoint_ranges_cover_fixed_origins(self):
        rows=[struct.pack('<4f',1.,0.,0.,0.) for _ in range(422)]; reads=[]; guards=[]
        def read(start,stop):
            reads.append((start,stop)); return b''.join(rows[start:stop])
        contexts=list(iter_segment_contexts(user_index=2,global_offset=9,origin_count=3,
            sample_count=422,read_rows=read,check_guard=lambda:guards.append(True)))
        self.assertEqual(reads,[(0,400),(1,401),(2,402)])
        self.assertEqual([(u,i) for u,i,_ in contexts],[(2,9),(2,10),(2,11)])
        reads.clear()
        endpoints=list(iter_segment_endpoints(user_index=2,global_offset=9,origin_count=3,
            sample_count=422,read_rows=read,check_guard=lambda:guards.append(True)))
        self.assertEqual(reads,[(419,420),(420,421),(421,422)])
        self.assertEqual([(u,i) for u,i,_ in endpoints],[(2,9),(2,10),(2,11)])
        self.assertEqual(len(guards),6)

    def test_invalid_scope_or_values_fail_closed(self):
        read=lambda a,b:struct.pack('<4f',0.,0.,0.,0.)*(b-a)
        with self.assertRaises(ValueError): list(iter_segment_contexts(user_index=0,global_offset=0,
            origin_count=1,sample_count=421,read_rows=read,check_guard=lambda:None))
        with self.assertRaises(ValueError): list(iter_segment_contexts(user_index=0,global_offset=0,
            origin_count=1,sample_count=420,read_rows=read,check_guard=lambda:None))


if __name__ == '__main__': unittest.main()
