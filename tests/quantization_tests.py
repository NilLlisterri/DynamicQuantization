import unittest

from quantization import scale_weight, descale_weight, get_scale_range

class TestWeightScaling(unittest.TestCase):

    def test_scale_and_descale_identity(self):
        min_w = -1.0
        max_w = 1.0
        scaled_bits = 8
        weights = [-1.0, -0.5, 0.0, 0.5, 1.0]

        for weight in weights:
            scaled = scale_weight(min_w, max_w, weight, scaled_bits)
            descaled = descale_weight(min_w, max_w, scaled, scaled_bits)
            self.assertAlmostEqual(weight, descaled, places=2)

    def test_scale_range(self):
        a, b = get_scale_range(8)
        self.assertEqual(a, 0.0)
        self.assertEqual(b, 255.0)

    def test_scale_boundaries(self):
        min_w = 0.0
        max_w = 10.0
        scaled_bits = 4  # So scale range is 0..15
        self.assertEqual(scale_weight(min_w, max_w, 0.0, scaled_bits), 0)
        self.assertEqual(scale_weight(min_w, max_w, 10.0, scaled_bits), 15)

    def test_descale_boundaries(self):
        min_w = 0.0
        max_w = 10.0
        scaled_bits = 4
        self.assertAlmostEqual(descale_weight(min_w, max_w, 0, scaled_bits), 0.0)
        self.assertAlmostEqual(descale_weight(min_w, max_w, 15, scaled_bits), 10.0)

if __name__ == '__main__':
    unittest.main()
