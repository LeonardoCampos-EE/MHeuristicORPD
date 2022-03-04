import pandas as pd
import numpy as np


def objective_function(network, conductance_matrix: np.ndarray) -> float:
    """
    Computes the objective function for the ORPD problem using a PandaPower network
    containing the parameters for a search agent and the conductance matrix.

    The object function is equivalent to the activate power loss on the system
    it's given by:
    f = sum g_km * [v_k^2 + v_m^2 - 2*v_k*v_m*cos(theta_k - theta_m)]

    Note: the * operator denotes here the element-wise multiplication of matrices
    """

    # Voltage array for all the system buses
    v_k = np.expand_dims(network.res_bus.vm_pu.to_numpy(), axis=0)

    # Transposed voltage array
    v_m = v_k.T

    # Voltage angle array for all the system buses
    theta_k = np.radians(np.expand_dims(network.res_bus.va_degree.to_numpy(), axis=0))

    # Voltage angle array transposed
    theta_m = theta_k.T

    # Objective function calculation
    f = (
        (v_k ** 2)
        + (v_m ** 2)
        - 2 * np.multiply(np.multiply(v_k, v_m), np.cos(theta_k - theta_m))
    )
    f = np.multiply(f, conductance_matrix)
    f = np.squeeze(np.sum(f))

    return f
