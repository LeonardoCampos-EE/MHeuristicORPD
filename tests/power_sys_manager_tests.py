import unittest
import pdb

import numpy as np

from power_system_manager import PowerSystemManager


class TestPowerSystemManager(unittest.TestCase):
    def setUp(self):

        self.manager = PowerSystemManager(system="14")

        return

    def test_shunt_values(self):

        expected = np.array([[0.0, 0.19, 0.34, 0.39]])
        shunt_values = self.manager.shunt_values

        self.assertEqual(np.allclose(shunt_values, expected), True)
        return

    def test_first_agent(self):

        expected = np.array([1.045, 1.01, 1.07, 1.09, 1.022, 1.031, 1.068, 0.19])
        first_agent = self.manager.first_agent

        self.assertEqual(np.allclose(first_agent, expected), True)
        return
