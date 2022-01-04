import numpy as np


def upper_discrete(array: np.ndarray, interval: np.ndarray) -> np.ndarray:

    """
    Returns the upper discrete values from @interval closest to the @array values

    Inputs:
        -> array = contains the values which the algorithm will approximate to the closest upper discrete values
        -> interval = contains the set of discrete values that the values from @array admit

    Ouputs:
        -> upper = contains the upper discrete value closest to the values from @array values

    Example:
        array = [-3.5, 0.8, 0.45, 3.0, 1.6]
        interval = [0.0, 1.0, 2.0]

        Returns:
        upper = [0.0, 1.0, 1.0, 2.0, 2.0]
    """

    # Clip the values from @array to be between the minimum and maximum values from @interval
    array -= 1e-3
    clipped = np.clip(a=array, a_min=min(interval) + 1e-3, a_max=max(interval) - 1e-3)

    # Search the indices from @interval that correspond to the upper discrete values from @array
    indices = np.searchsorted(a=interval, v=clipped, side="right")

    # Get the values from @interval in the same order as @array
    upper = np.take(interval, indices)

    return upper


def lower_discrete(array: np.ndarray, interval: np.ndarray) -> np.ndarray:

    """
    Returns the lower discrete values from @interval closest to the @array values

    Inputs:
        -> array = contains the values which the algorithm will approximate to the closest lower discrete values
        -> interval = contains the set of discrete values that the values from @array admit

    Ouputs:
        -> lower = contains the lower discrete value closest to the values from @array values

    Example:
        array = [-3.5, 0.8, 0.45, 3.0, 1.6]
        interval = [0.0, 1.0, 2.0]

        Returns:
        lower = [0.0, 0.0, 0.0, 2.0, 1.0]
    """

    # Clip the values from @array to be between the minimum and maximum values from @interval
    array -= 1e-3
    clipped = np.clip(a=array, a_min=min(interval) + 1e-3, a_max=max(interval) - 1e-3)

    # Search the indices from @interval that correspond to the lower discrete values from @array
    indices = np.searchsorted(a=interval, v=clipped, side="left") - 1

    # Get the values from @interval in the same order as @array
    lower = np.take(interval, indices)

    return lower
