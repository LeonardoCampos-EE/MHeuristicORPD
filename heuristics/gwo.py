from heuristics.optimizer import *
import time

class GWO(Optimizer):

    def __init__(self, population_size, dim, objective_function, constraints = None, upper_bounds = None, lower_bounds = None):
        super().__init__(population_size, dim, objective_function, constraints, upper_bounds, lower_bounds)

    def optimize(self, iterations):
        
        # Initialize the parameters of the best agent
        self.best_objective = []
        self.best_constraint = []
        self.best_fitness = []

        # Initialize the best solutions, i. e., the alpha, beta and delta wolves
        self.best_agents = np.zeros(shape = (self.dim, 3), dtype = np.float64)

        # Main loop
        start_time = time.time()
        for t in tqdm(range(iterations)):

            # a(t) -> goes from 2 to 0 over the iterations
            a = 2 - ( t * (2./iterations))

            # r1 and r2 -> random numbers between 0 and 1
            r1 = self.random_gen.random(size = (self.dim, self.population_size), dtype = np.float64)
            r2 = self.random_gen.random(size = (self.dim, self.population_size), dtype = np.float64)

            # A(t) -> controls the step size of each wolf in the search space
            A_t = 2*a*r1 - a

            # C(t) -> controls the movement of each wolf towards the best solutions
            C_t = 2*r2

            # Calculate the objective function
            self.objective_array = self.objective_function(self.pop_array)
            
            # Calculate the constraints
            if self.constraints is not None:
                for idx, constraint in enumerate(self.constraints):
                    self.constraint_array[idx, :] = constraint(self.pop_array)

            # Calculate the fitness function 
            self.fitness_array = self.objective_array + self.constraint_array.sum(axis = 1)

            # Get the indexes that would sort the fitness_array to get the best solutions of this iteration
            sort_indexes = np.argsort(self.fitness_array)

            # Sort each array to with sort_indexes to match the alfa, beta, delta, omega wolves format
            self.objective_array = np.take_along_axis(self.objective_array, sort_indexes, axis = 1)
            self.constraint_array = np.take_along_axis(self.constraint_array, sort_indexes, axis = 1)
            self.fitness_array = np.take_along_axis(self.fitness_array, sort_indexes, axis = 1)

            self.pop_array = np.take_along_axis(self.pop_array, sort_indexes, axis = 1)

            # Keep track of alpha's objective, penalty and fitness functions
            self.best_objective.append(self.objective_array[:, 0])
            self.best_constraint.append(self.constraint_array[:, 0])
            self.best_fitness.append(self.fitness_array[:, 0])
            
            # Save the best agents for the current iteration
            self.best_agents = self.pop_array[:, :3]

            # Calculate D_alpha, D_beta, D_delta 
            D_alpha = np.abs(C_t*self.best_agents[:, 0] - self.pop_array)
            D_beta = np.abs(C_t*self.best_agents[:, 1] - self.pop_array)
            D_delta = np.abs(C_t*self.best_agents[:, 2] - self.pop_array)

            # Calculate X_alpha, X_beta, X_delta
            X_alpha = self.best_agents[:, 0] - A_t*D_alpha
            X_beta = self.best_agents[:, 1] - A_t*D_beta
            X_delta = self.best_agents[:, 2] - A_t*D_delta

            # Update population
            self.pop_array = (X_alpha + X_beta + X_delta)/3.

            # Clip the variables to the bounds
            np.clip(a = self.pop_array, a_min = self.lower_bounds, 
            a_max = self.upper_bounds, out = self.pop_array)
        # End main loop

        # Execution time
        end_time = time.time()
        execution_time = end_time - start_time

        print('---------------- Solution ----------------')
        print(f'Fitness = {self.fitness_array[:, 0]}; Objective = {self.objective_array[:, 0]}; Constraint = {self.constraint_array[:, 0]}')
        print(f'Execution time = {execution_time}')

        return

if __name__ == '__main__':
    pass
