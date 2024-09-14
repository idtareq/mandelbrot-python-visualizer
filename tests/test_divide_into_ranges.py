import unittest
from util import divide_into_ranges


class TestDivideIntoRanges(unittest.TestCase):
    def test_equal_division(self):
        N = 20
        K = 4
        expected = [[0, 4], [5, 9], [10, 14], [15, 19]]
        result = divide_into_ranges(N, K)
        self.assertEqual(result, expected)

    def test_unequal_division(self):
        N = 17
        K = 3
        expected = [[0, 5], [6, 11], [12, 16]]
        result = divide_into_ranges(N, K)
        self.assertEqual(result, expected)

    def test_N_less_than_K(self):
        N = 3
        K = 5
        expected = [[0, 0], [1, 1], [2, 2], [3, 2], [3, 2]]
        result = divide_into_ranges(N, K)
        self.assertEqual(result, expected)

    def test_zero_N(self):
        N = 0
        K = 3
        expected = []
        result = divide_into_ranges(N, K)
        self.assertEqual(result, expected)

    def test_one_N(self):
        N = 1
        K = 1
        expected = [[0, 0]]
        result = divide_into_ranges(N, K)
        self.assertEqual(result, expected)

    def test_zero_K(self):
        N = 10
        K = 0
        with self.assertRaises(ValueError):
            divide_into_ranges(N, K)
