from heuristics.gwo import GWO
from benchmark.functions import Rosenbrock
import numpy as np

if __name__ == "__main__":
    upper_bound = np.expand_dims(np.array([1.5, 2.5]), axis=-1)
    lower_bound = np.expandlower_bound = np.expand_dims(np.array([-1.5, -0.5]), axis=-1)

    rosen = Rosenbrock(lower_bound=lower_bound, upper_bound=upper_bound)

    # rosen.visualize_surface(opacity=1.0, color="Thermal")

    solution = 1.0
    gwo = GWO(
        population_size=20,
        dim=2,
        objective_function=rosen.function,
        lower_bounds=lower_bound,
        upper_bounds=upper_bound,
    )
    while solution > 1e-3:
        gwo.optimize(iterations=50, is_orpd=False)
        solution = gwo.best_fitness[-1]

    rosen.visualize_search(
        pop_array=gwo.pop_list,
        best_agents_array=gwo.best_agents_list,
    )
