import unittest

import numpy as np

from utils.math import upper_discrete, lower_discrete, approximate


class TestMathUtils(unittest.TestCase):
    def test_upper_discrete(self):

        array = np.array([0.0, 0.15, 0.19, 0.30, 0.34, 0.37, 0.39])
        interval = np.array([0, 0.19, 0.34, 0.39])

        upper = upper_discrete(array, interval)

        expected = np.array([0.19, 0.19, 0.19, 0.34, 0.34, 0.39, 0.39])

        self.assertEqual(np.allclose(upper, expected), True)
        return

    def test_lower_discrete(self):

        array = np.array([0.0, 0.15, 0.19, 0.30, 0.34, 0.37, 0.39])
        interval = np.array([0.0, 0.19, 0.34, 0.39])

        lower = lower_discrete(array, interval)

        expected = np.array([0.0, 0.0, 0.0, 0.19, 0.19, 0.34, 0.34])

        self.assertEqual(np.allclose(lower, expected), True)
        return

    def test_approximate(self):

        array1 = np.array([0.15, 0.30])
        interval1 = np.array([[0.0, 0.19, 0.20, 0.21], [0.0, 0.15, 0.30, 1.0]])
        approximated1 = approximate(array1, interval1)
        expected1 = np.array([0.19, 0.30])

        array2 = np.array([0.10, 0.80])
        interval2 = np.array([[0.0, 0.19, 0.20, 0.21]])
        approximated2 = approximate(array2, interval2)
        expected2 = np.array([0.19, 0.21])

        self.assertEqual(np.allclose(approximated1, expected1), True)
        self.assertEqual(np.allclose(approximated2, expected2), True)
        return


if __name__ == "__main__":
    unittest.main()
