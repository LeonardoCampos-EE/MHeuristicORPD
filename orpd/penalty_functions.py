import pdb

import pandas as pd
import numpy as np

from utils.math import upper_discrete, lower_discrete


def voltage_static_penalty(network: pd.DataFrame) -> float:

    """
    Calcualtes the static penalty function for the voltage deviations
    of the network


    The static penalty function is given by:

    penalty = sum(v - v_lim)**2
    Where:
        v_lim = v_lim_max, if v > v_lim_max
        v_lim = v, if v_lim_min <= v <= v_lim_max
        v_lim = v_lim_min, if v < v_lim_min
    """

    # Limit of the maximum possible voltage on the network
    v_lim_max = network.bus.max_vm_pu.to_numpy()
    v_lim_min = network.bus.min_vm_pu.to_numpy()

    v = network.res_bus.vm_pu.to_numpy()

    # Superior limit violation (v > v_lim_max)
    v_sup = v - v_lim_max
    v_sup = v_sup[np.greater(v_sup, 0.0)]

    # Inferior limit violation (v < v_lim_min)
    v_inf = v - v_lim_min
    v_inf = v_inf[np.less(v_inf, 0.0)]

    penalty = np.squeeze(np.sum(v_sup ** 2) + np.sum(v_inf ** 2))

    return penalty


def taps_sinusoidal_penalty(taps: np.ndarray, s: float) -> np.ndarray:

    """
    Calculates the sinusoidal penalty function for the discrete violation
    of the network transformer taps.

    The penalty function is given by:
    penalty = sum (sin²(taps * pi/s))

    Where s is the step between the possible discrete values the taps can assume
    """

    penalty = np.sin(taps * np.pi / s) ** 2

    # For numerical reasons, we may consider 0.0 any value below 10^-12
    penalty = np.where(penalty < 1e-12, 0.0, penalty)
    penalty = np.sum(penalty, axis=0)

    return penalty


def shunts_sinusoidal_penalty(
    shunts: np.ndarray, shunt_values: np.ndarray
) -> np.ndarray:

    shunts_upper = np.zeros_like(shunts)
    shunts_lower = np.zeros_like(shunts)

    for index, list_of_values in enumerate(shunt_values):
        shunts_upper[index] = upper_discrete(
            shunts[index].copy(), list_of_values.copy()
        )
        shunts_lower[index] = lower_discrete(
            shunts[index].copy(), list_of_values.copy()
        )

    delta = np.abs(shunts_upper - shunts_lower)

    alpha = np.pi * (
        np.ceil(np.abs(shunts_lower) / delta) - np.abs(shunts_lower) / delta
    )

    penalty = np.sin(alpha + np.pi * (shunts / delta))
    penalty = np.square(penalty)
    penalty = np.where(penalty < 1e-5, 0.0, penalty)
    penalty = np.sum(penalty, axis=0)

    return penalty
