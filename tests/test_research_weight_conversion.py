"""Check the affine packed-weight convention before cloud model loading."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from convert_research_weights import expand


class WeightConversionTests(unittest.TestCase):
    def test_low_nibble_first_and_independent_groups(self):
        packed=np.array([[0x76543210,0xfedcba98]*8],dtype=np.uint32)
        scales=np.array([[2,.5]],dtype=np.float16)
        biases=np.array([[-1,3]],dtype=np.float16)
        decoded=expand(packed,scales,biases)
        self.assertEqual(decoded.dtype,np.float16)
        self.assertEqual(decoded.shape,(1,128))
        np.testing.assert_array_equal(decoded[0,:64],np.tile(np.arange(16)*2-1,4))
        np.testing.assert_array_equal(decoded[0,64:],np.tile(np.arange(16)*.5+3,4))


if __name__=='__main__':
    unittest.main()
