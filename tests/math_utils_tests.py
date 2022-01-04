import unittest

import numpy as np

from utils.math import upper_discrete, lower_discrete


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


if __name__ == "__main__":
    unittest.main()
