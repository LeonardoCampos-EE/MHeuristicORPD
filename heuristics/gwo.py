import time
from tqdm import tqdm
import pdb

import numpy as np

from heuristics.optimizer import Optimizer


class GWO(Optimizer):
    def __init__(
        self,
        population_size: int,
        dim: int,
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

    def optimize(self, iterations, **kwargs):

        # Keep all the agents througout the iterations for demonstrantion
        # Shape: (tmax, dim + 1, pop_size)
        self.pop_list = []
        self.best_agents_list = []
        self.alpha_list = []

        self.alpha_fitness = np.inf
        self.beta_fitness = np.inf
        self.delta_fitness = np.inf

        self.alpha_objective = None
        self.alpha_constraints = {}
        self.iterations = iterations

        # Initialize the parameters of the best agent
        self.best_objective = []
        if self.constraints is not None:
            self.best_constraint = {
                contraint_name: [] for contraint_name in self.constraints
            }
        self.best_fitness = []
        if "run_dc_power_flow" in kwargs.keys():
            run_dc_power_flow = kwargs["run_dc_power_flow"]

        np.clip(
            a=self.pop_array,
            a_min=self.lower_bounds,
            a_max=self.upper_bounds,
            out=self.pop_array,
        )

        # Main loop
        start_time = time.time()
        with tqdm(
            total=iterations,
            desc=f"Best fitness: {np.inf}",
            bar_format="{desc}",
        ) as tq:
            for t in tqdm(range(iterations)):

                # a(t) -> goes from 2 to 0 over the iterations
                if (kwargs["mod"]):
                    self.a = 2 * (1 - (t ** 2 / iterations ** 2))
                else:
                    self.a = 2 - (t * (2.0 / iterations))
                    

                # r1 and r2 -> random numbers between 0 and 1
                self.r1 = np.random.random_sample(size=(self.dim, self.population_size))
                self.r2 = np.random.random_sample(size=(self.dim, self.population_size))

                # A(t) -> controls the step size of each wolf in the search space
                self.A = (2 * self.a * self.r1) - self.a

                # C(t) -> controls the movement of each wolf towards the best solutions
                self.C = 2 * self.r2

                # Calculate the objective function
                if not kwargs["is_orpd"]:
                    self.objective_array = self.objective_function(self.pop_array)
                else:
                    # ORPD case
                    if run_dc_power_flow and t <= iterations / 2.0 and t > 0:
                        kwargs["run_dc_power_flow"] = True
                    else:
                        kwargs["run_dc_power_flow"] = False

                    self.objective_array = self.objective_function(
                        self.pop_array,
                        self.constraint_arrays,
                        self.constraints,
                        **kwargs,
                    )

                # Calculate the constraints
                if self.constraints is not None and not kwargs["is_orpd"]:
                    for contraint_name, constraint_function in self.constraints.items():
                        self.constraint_arrays[contraint_name] = constraint_function(
                            self.pop_array
                        )

                # Calculate the fitness function
                elif self.constraints is not None and kwargs["is_orpd"]:
                    self.fitness_array = self.objective_array.copy()
                    for constraint_name, constraint in self.constraint_arrays.items():
                        self.fitness_array += constraint
                else:
                    self.fitness_array = self.objective_array.copy()

                # Get the indexes that would sort the fitness_array to get the best solutions of this iteration
                sort_indexes = np.argsort(self.fitness_array)

                self.alpha_index = sort_indexes[0]
                self.beta_index = sort_indexes[1]
                self.delta_index = sort_indexes[2]

                if self.fitness_array[self.alpha_index] <= self.alpha_fitness:
                    self.alpha_wolf = np.expand_dims(
                        self.pop_array[:, self.alpha_index], axis=-1
                    )
                    self.alpha_fitness = self.fitness_array[self.alpha_index]
                    self.alpha_objective = self.objective_array[self.alpha_index]
                    if self.constraints is not None:
                        for (
                            constraint_name,
                            constraint_array,
                        ) in self.constraint_arrays.items():
                            self.alpha_constraints[constraint_name] = constraint_array[
                                self.alpha_index
                            ]

                if self.fitness_array[self.beta_index] <= self.beta_fitness:
                    self.beta_wolf = np.expand_dims(
                        self.pop_array[:, self.beta_index], axis=-1
                    )
                    self.beta_fitness = self.fitness_array[self.beta_index]

                if self.fitness_array[self.delta_index] <= self.delta_fitness:
                    self.delta_wolf = np.expand_dims(
                        self.pop_array[:, self.delta_index], axis=-1
                    )
                    self.delta_fitness = self.fitness_array[self.delta_index]

                # Keep track of alpha's objective, penalty and fitness functions
                self.best_fitness.append(self.alpha_fitness)
                self.best_objective.append(self.alpha_objective)
                if self.constraints is not None:
                    for (
                        constraint_name,
                        constraint_array,
                    ) in self.constraint_arrays.items():
                        self.best_constraint[constraint_name].append(
                            self.alpha_constraints[constraint_name]
                        )

                tq.set_description(f"Best fitness at t = {t}: {self.alpha_fitness}")

                # Update the agents positions
                self.update_agents()

                # Save the population and the best agents for demonstrantion
                if not kwargs["is_orpd"]:
                    self.pop_list.append(
                        np.vstack([self.pop_array.copy(), self.fitness_array])
                    )

                    self.best_agents_list.append(
                        np.vstack(
                            [
                                np.hstack(
                                    [
                                        self.alpha_wolf,
                                        self.beta_wolf,
                                        self.delta_wolf,
                                    ]
                                ),
                                np.array(
                                    [
                                        [
                                            self.alpha_fitness,
                                            self.beta_fitness,
                                            self.delta_fitness,
                                        ]
                                    ]
                                ),
                            ]
                        )
                    )
                self.alpha_list.append(np.squeeze(self.alpha_wolf, axis=-1))

        # Execution time
        end_time = time.time()
        self.time = end_time - start_time

        best_solution_iteration = np.argmin(self.best_fitness)
        self.solution = self.alpha_list[best_solution_iteration]
        self.solution_fitness = self.best_fitness[best_solution_iteration]
        self.solution_objective = self.best_objective[best_solution_iteration]
        self.solution_constraints = {}
        if self.constraints is not None:
            for constraint_name in self.constraints.keys():
                self.solution_constraints[constraint_name] = self.best_constraint[
                    constraint_name
                ][best_solution_iteration]

        print(f"Execution time = {self.time}")

        return

    def update_agents(self):

        """
        Update the search agents positions
        """

        # Calculate D_alpha, D_beta, D_delta
        D_alpha = np.abs(np.multiply(self.C, self.alpha_wolf) - self.pop_array)
        D_beta = np.abs(np.multiply(self.C, self.beta_wolf) - self.pop_array)
        D_delta = np.abs(np.multiply(self.C, self.delta_wolf) - self.pop_array)

        # Calculate X_alpha, X_beta, X_delta
        X_alpha = self.alpha_wolf - np.multiply(self.A, D_alpha)
        X_beta = self.beta_wolf - np.multiply(self.A, D_beta)
        X_delta = self.delta_wolf - np.multiply(self.A, D_delta)

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
