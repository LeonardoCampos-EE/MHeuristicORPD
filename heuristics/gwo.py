import time
from tqdm import tqdm
import pdb

import numpy as np

from heuristics.optimizer import Optimizer


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

        # Main loop
        start_time = time.time()
        with tqdm(
            total=iterations,
            desc=f"Best fitness: {np.inf}",
            bar_format="{desc}",
        ) as tq:
            for t in tqdm(range(iterations)):

                # a(t) -> goes from 2 to 0 over the iterations
                self.a = 2 - (t * (2.0 / iterations))

                # r1 and r2 -> random numbers between 0 and 1
                self.r1 = self.random_gen.random(
                    size=(self.dim, self.population_size), dtype=np.float64
                )
                self.r2 = self.random_gen.random(
                    size=(self.dim, self.population_size), dtype=np.float64
                )

                # A(t) -> controls the step size of each wolf in the search space
                self.A_t = 2 * self.a * self.r1 - self.a

                # C(t) -> controls the movement of each wolf towards the best solutions
                self.C_t = 2 * self.r2

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

                self.alpha_index = sort_indexes[0]
                self.beta_index = sort_indexes[1]
                self.delta_index = sort_indexes[2]

                # Keep track of alpha's objective, penalty and fitness functions
                self.best_objective.append(self.objective_array[self.alpha_index])
                if self.constraints:
                    self.best_constraint.append(self.constraint_array[self.alpha_index])
                else:
                    self.best_constraint.append([0])
                self.best_fitness.append(self.fitness_array[self.alpha_index])

                tq.set_description(
                    f"Best fitness at t = {t}: {self.fitness_array[self.alpha_index]}"
                )

                # Update the agents positions
                self.update_agents()

                # Save the population and the best agents for demonstrantion
                self.pop_list.append(
                    np.vstack([self.pop_array.copy(), self.fitness_array])
                )
                self.best_agents_list.append(
                    np.vstack(
                        [
                            self.pop_array[
                                :, [self.alpha_index, self.beta_index, self.delta_index]
                            ],
                            self.fitness_array[
                                [self.alpha_index, self.beta_index, self.delta_index]
                            ],
                        ]
                    )
                )

        # Execution time
        end_time = time.time()
        execution_time = end_time - start_time

        print("---------------- Solution ----------------")
        print(
            f"Fitness = {self.fitness_array[self.alpha_index]}; Objective = {self.objective_array[self.alpha_index]}; Constraint = {self.constraint_array[self.alpha_index]}"
        )
        print(f"Best agent = {self.pop_array[:, self.alpha_index]}")
        print(f"Execution time = {execution_time}")

        return

    def update_agents(self):

        """
        Update the search agents positions

        """
        # self.best_agents = self.pop_array[:, :3]

        alpha_wolf = np.expand_dims(self.pop_array[:, self.alpha_index], axis=-1)
        beta_wolf = np.expand_dims(self.pop_array[:, self.beta_index], axis=-1)
        delta_wolf = np.expand_dims(self.pop_array[:, self.delta_index], axis=-1)

        # Calculate D_alpha, D_beta, D_delta
        D_alpha = np.abs(self.C_t * alpha_wolf - self.pop_array)
        D_beta = np.abs(self.C_t * beta_wolf - self.pop_array)
        D_delta = np.abs(self.C_t * delta_wolf - self.pop_array)

        # Calculate X_alpha, X_beta, X_delta
        X_alpha = alpha_wolf - self.A_t * D_alpha
        X_beta = beta_wolf - self.A_t * D_beta
        X_delta = delta_wolf - self.A_t * D_delta

        # Update population
        self.pop_array = (X_alpha + X_beta + X_delta) / 3.0

        # Clip the variables to the bounds
        np.clip(
            a=self.pop_array,
            a_min=self.lower_bounds,
            a_max=self.upper_bounds,
            out=self.pop_array,
        )

        return


if __name__ == "__main__":
    pass
