import unittest
import pdb

import numpy as np

from power_system_manager import PowerSystemManager
from orpd.penalty_functions import (
    taps_sinusoidal_penalty,
    shunts_sinusoidal_penalty,
)


class TestPenaltyFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = PowerSystemManager(
            system="14",
            tap_step=0.00625,
        )

        return super().setUp()

    def test_taps_sinusoidal_penalty(self):

        test_array = np.array([0.9, 1.025, 1.1, 0.903, 1.014, 1.0666])
        penalty = taps_sinusoidal_penalty(test_array, s=0.00625)
        self.assertEqual(np.allclose(penalty, 2.243099899136704), True)

    def test_shunts_sinusoidal_penalty(self):
        test_array = np.array([[0.0, 0.19, 0.34, 0.39, 0.15, 0.21, 0.35]])
        penalty = shunts_sinusoidal_penalty(
            test_array, np.array([[0.0, 0.19, 0.34, 0.39]])
        )
        self.assertEqual(np.allclose(penalty, 0.8881834506314976), True)
