from heuristics.optimizer import *
import time
from tqdm import tqdm


class GWO(Optimizer):
    def __init__(
        self,
        population_size,
        dim,
        objective_function,
        constraints=None,
        upper_bounds=None,
        lower_bounds=None,
    ):
        super().__init__(
            population_size,
            dim,
            objective_function,
            constraints,
            upper_bounds,
            lower_bounds,
        )

    def optimize(self, iterations):

        # Keep all the agents througout the iterations for demonstrantion
        # Shape: (tmax, dim + 1, pop_size)
        self.pop_list = []
        self.best_agents_list = []

        # Initialize the parameters of the best agent
        self.best_objective = []
        self.best_constraint = []
        self.best_fitness = []

        # Initialize the best solutions, i. e., the alpha, beta and delta wolves
        self.best_agents = np.zeros(shape=(self.dim, 3), dtype=np.float64)

        # Main loop
        start_time = time.time()
        with tqdm(
            total=iterations,
            desc=f"Best fitness: {np.inf}",
            bar_format="{desc}",
        ) as tq:
            for t in tqdm(range(iterations)):

                # a(t) -> goes from 2 to 0 over the iterations
                a = 2 - (t * (2.0 / iterations))

                # r1 and r2 -> random numbers between 0 and 1
                r1 = self.random_gen.random(
                    size=(self.dim, self.population_size), dtype=np.float64
                )
                r2 = self.random_gen.random(
                    size=(self.dim, self.population_size), dtype=np.float64
                )

                # A(t) -> controls the step size of each wolf in the search space
                A_t = 2 * a * r1 - a

                # C(t) -> controls the movement of each wolf towards the best solutions
                C_t = 2 * r2

                # Calculate the objective function
                self.objective_array = self.objective_function(self.pop_array)

                # Calculate the constraints
                if self.constraints is not None:
                    for idx, constraint in enumerate(self.constraints):
                        self.constraint_array[idx, :] = constraint(self.pop_array)

                # Calculate the fitness function
                self.fitness_array = (
                    self.objective_array + self.constraint_array.sum(axis=1)
                    if self.constraints
                    else self.objective_array
                )

                # Get the indexes that would sort the fitness_array to get the best solutions of this iteration
                sort_indexes = np.argsort(self.fitness_array)

                # Sort each array with respect to sort_indexes to match the alfa, beta, delta, omega wolves format
                self.objective_array = np.take_along_axis(
                    self.objective_array, sort_indexes, axis=0
                )
                self.constraint_array = np.take_along_axis(
                    self.constraint_array, sort_indexes, axis=0
                )
                self.fitness_array = np.take_along_axis(
                    self.fitness_array, sort_indexes, axis=0
                )

                # sort_indexes = np.expand_dims(sort_indexes, axis=0)
                sort_indexes = np.reshape(sort_indexes, (1, -1))
                self.pop_array = np.take_along_axis(
                    self.pop_array, sort_indexes, axis=1
                )

                # Keep track of alpha's objective, penalty and fitness functions
                self.best_objective.append(self.objective_array[0])
                if self.constraints:
                    self.best_constraint.append(self.constraint_array[0])
                else:
                    self.best_constraint.append([0])
                self.best_fitness.append(self.fitness_array[0])

                tq.set_description(f"Best fitness at t = {t}: {self.fitness_array[0]}")

                # Save the best agents for the current iteration
                self.best_agents = self.pop_array[:, :3]

                # Save the population and the best agents for demonstrantion
                self.pop_list.append(
                    np.vstack([self.pop_array.copy(), self.fitness_array])
                )
                self.best_agents_list.append(
                    np.vstack([self.best_agents.copy(), self.fitness_array[:3]])
                )

                # Calculate D_alpha, D_beta, D_delta
                D_alpha = np.abs(
                    C_t * np.reshape(self.best_agents[:, 0], (-1, 1)) - self.pop_array
                )
                D_beta = np.abs(
                    C_t * np.reshape(self.best_agents[:, 1], (-1, 1)) - self.pop_array
                )
                D_delta = np.abs(
                    C_t * np.reshape(self.best_agents[:, 2], (-1, 1)) - self.pop_array
                )

                # Calculate X_alpha, X_beta, X_delta
                X_alpha = np.reshape(self.best_agents[:, 0], (-1, 1)) - A_t * D_alpha
                X_beta = np.reshape(self.best_agents[:, 1], (-1, 1)) - A_t * D_beta
                X_delta = np.reshape(self.best_agents[:, 2], (-1, 1)) - A_t * D_delta

                # Update population
                self.pop_array = (X_alpha + X_beta + X_delta) / 3.0

                # Clip the variables to the bounds
                np.clip(
                    a=self.pop_array,
                    a_min=self.lower_bounds,
                    a_max=self.upper_bounds,
                    out=self.pop_array,
                )

                # time.sleep(0.5)
            # End main loop

        # Execution time
        end_time = time.time()
        execution_time = end_time - start_time

        print("---------------- Solution ----------------")
        print(
            f"Fitness = {self.fitness_array[0]}; Objective = {self.objective_array[0]}; Constraint = {self.constraint_array[0]}"
        )
        print(f"Best agent = {self.best_agents[:, 0]}")
        print(f"Execution time = {execution_time}")

        return


if __name__ == "__main__":
    pass
