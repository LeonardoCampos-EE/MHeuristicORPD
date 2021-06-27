import numpy as np

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

            * constraints (list):
                List with the contraint functions of the optimization problem

            * upper_bounds (numpy.ndarray - float64):
                Numpy array containing the variables upper limits
                $ (dim, 1)

            * lower_bounds (numpy.ndarray - float64):
                Numpy array containing the variables lower limits
                $ (dim, 1)

            ----------------- Returns -----------------
            * pop_array (numpy.ndarray - float64):
                Population array with the following dimensionality:
                $ (dim, population_size)

            * constraint_array (numpy.ndarray - float64):
                Array containing the values of each constraint for
                each search agent
                $ (len(constraints), population_size)

            * objective_array (numpy.ndarray - float64):
                Array containing the values of the objective function
                for each search agent
                $ (1, population_size)

            * fitness_array (numpy.ndarray - float64):
                Array containing the values of the fitness function
                for each search agent
                $ (1, population_size)
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
            size=(dim, population_size), dtype=np.float64
        ) + lower_bounds

        # Initialize the constraints function array
        if constraints is not None:
            self.constraint_array = np.zeros(shape=(len(constraints), population_size))
        else:
            self.constraint_array = np.zeros(shape=(population_size,))

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
