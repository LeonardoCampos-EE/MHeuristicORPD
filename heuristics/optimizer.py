import numpy as np
import matplotlib.pyplot as plt


class Optimizer:
    def __init__(
        self,
        population_size,
        dim,
        objective_function,
        constraints=None,
        upper_bounds=None,
        lower_bounds=None,
    ):

        """
        This function initializes the population array that will be optimized
        by the chosen meta-heuristic algorithm.

            ----------------- Arguments -----------------
            * population_size (int):
                Number of search agents of the algorithm

            * dim (int):
                Number of variables of the optimization problem
                (dimensionality of the problem)

            * constraints (dict):
                Dict with the contraint functions of the optimization problem
                $ {"voltage": voltage_function}

            * upper_bounds (numpy.ndarray - float32):
                Numpy array containing the variables upper limits
                $ (dim, 1)

            * lower_bounds (numpy.ndarray - float32):
                Numpy array containing the variables lower limits
                $ (dim, 1)

            ----------------- Returns -----------------
            * pop_array (numpy.ndarray - float32):
                Population array with the following dimensionality:
                $ (dim, population_size)

            * constraint_arrays (dict):
                Dictionary containing the values of each constraint for
                each search agent
                $ constraint_arrays = {
                    "voltage": np.array() -> shape = (population_size, )
                }

            * objective_array (numpy.ndarray - float32):
                Array containing the values of the objective function
                for each search agent
                $ (population_size,)

            * fitness_array (numpy.ndarray - float32):
                Array containing the values of the fitness function
                for each search agent
                $ (population_size, )
        """

        # Save the input parameters as class properties
        self.dim = dim
        self.constraints = constraints
        self.population_size = population_size
        self.upper_bounds = upper_bounds
        self.lower_bounds = lower_bounds
        self.objective_function = objective_function

        # Random number generator
        self.random_gen = np.random.default_rng()

        # Randomly intializes the population array on the interval [lower_bounds, upper_bounds)
        self.pop_array = (upper_bounds - lower_bounds) * self.random_gen.random(
            size=(dim, population_size), dtype=np.float32
        ) + lower_bounds

        # Initialize the constraints function array
        if constraints is not None:
            self.constraint_arrays = {
                constraint_name: np.zeros(shape=(population_size,))
                for constraint_name in self.constraints.keys()
            }
        else:
            self.constraint_array = None

        # Initialize the objective function array
        self.objective_array = np.ones(shape=(1, population_size)) * np.inf

        # Initialize the fitness function array
        self.fitness_array = np.ones(shape=(1, population_size)) * np.inf

        return

    def optimize(self, iterations):
        raise NotImplementedError

    def update_agents(self):
        raise NotImplementedError

    def calculate_contraints(self):
        raise NotImplementedError

    def calculate_objective(self):
        raise NotImplementedError

    def calculate_fitness(self):
        raise NotImplementedError

    def visualize_benchmark(self):
        raise NotImplementedError

    def get_fitness_chart(self, title):

        fig, ax = plt.subplots(1, facecolor="#293952")
        ax.set_facecolor("#293952")

        x = np.arange(0, len(self.best_fitness), step=1)
        y = self.best_fitness

        plt.plot(x, y, marker="o", markersize=4, color="#FDAC53", linewidth=1.5)

        ax.tick_params(axis="both", colors="w")
        plt.xticks(np.arange(0, np.max(x), step=5))
        plt.yticks(np.arange(0, np.max(y), step=5))
        plt.xlabel("Iterations", color="w", fontsize=16)
        plt.ylabel("Fitness Function", color="w", fontsize=16)
        plt.title(title, color="w", fontsize=16)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_color("w")
        ax.spines["bottom"].set_color("w")

        ax.set_axisbelow(True)
        ax.yaxis.grid(color="#FDAC53", linestyle="dashed", alpha=0.5)
        plt.show()
