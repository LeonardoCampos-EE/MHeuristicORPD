import os
import pdb
import yaml
import pickle

import numpy as np
import pandas as pd

from heuristics.gwo import GWO
from power_system_manager.manager import PowerSystemManager
from orpd.penalty_functions import (
    voltage_static_penalty,
    taps_sinusoidal_penalty,
    shunts_sinusoidal_penalty,
)
from utils.visualization import *


class ExperimentRunner:
    def __init__(self) -> None:

        with open("experiment_parameters.yaml", "r") as stream:
            self.experiment_parameters = yaml.safe_load(stream)

        self.runs = self.experiment_parameters["runs"]
        self.name = self.experiment_parameters["name"]

        # Create the folder where the experiments results will be stored
        self.results_dir = os.path.join("results", self.name)
        os.makedirs(self.results_dir, exist_ok=True)

        self.manager = PowerSystemManager(
            system=str(self.experiment_parameters["system"]["num_buses"]),
            tap_step=float(self.experiment_parameters["system"]["tap_step"]),
        )

        self.gwo = GWO(
            population_size=int(self.experiment_parameters["optimizer"]["pop_size"]),
            dim=self.manager.ng + self.manager.nt + self.manager.ns,
            objective_function=self.manager.orpd_objective_function,
            constraints=self.build_penalty_functions_dict(),
            upper_bounds=self.manager.get_upper_bounds(),
            lower_bounds=self.manager.get_lower_bounds(),
        )

        self.gwo.pop_array = self.manager.initialize_agents(self.gwo.pop_array)
        self.gwo.pop_array[:, 0] = self.manager.first_agent

        return

    def build_penalty_functions_dict(self) -> dict:

        penalty_fun_dict = {
            "voltage": voltage_static_penalty,
            "taps": taps_sinusoidal_penalty,
            "shunts": shunts_sinusoidal_penalty,
        }

        return penalty_fun_dict

    def build_solutions_dictionary(self) -> dict:

        solutions_dict = {
            "solution": {
                "bus_voltages": self.bus_voltages,
                "bus_angles": self.bus_angles,
                "taps": self.taps,
                "shunts": self.shunts,
                "fitness": self.solution_fitness,
                "objective": self.solution_objective,
                "constraints": self.solution_constraints,
                "fitness_array": self.fitness_array,
                "objective_array": self.objective_array,
                "constraint_arrays": self.constraint_arrays,
            },
            "runs": {
                "number": self.runs,
                "fitness": self.runs_fitness,
                "objective": self.runs_objective,
            },
        }

        with open(os.path.join(self.results_dir, "solutions.pkl"), "wb") as out:
            pickle.dump(solutions_dict, out, protocol=pickle.HIGHEST_PROTOCOL)

        self.solutions_dict = solutions_dict

        return

    def get_solution(self, agent):
        (
            self.bus_voltages,
            self.bus_angles,
            self.taps,
            self.shunts,
        ) = self.manager.get_solution(agent)
        self.solution_fitness = self.gwo.solution_fitness
        self.solution_objective = self.gwo.solution_objective
        self.solution_constraints = self.gwo.solution_constraints
        self.fitness_array = self.gwo.best_fitness
        self.objective_array = self.gwo.best_objective
        self.constraint_arrays = self.gwo.best_constraint

        return

    def visualize(self):

        visualize_fitness(self.fitness_array, self.results_dir)
        visualize_objective(self.objective_array, self.results_dir)
        for constraint_name, constraint_array in self.constraint_arrays.items():
            visualize_constraint(constraint_array, constraint_name, self.results_dir)

        visualize_voltages(self.bus_voltages, self.results_dir)
        visualize_voltage_angles(self.bus_angles, self.results_dir)
        summary(self.solutions_dict)

    def load_solution(self, directory):

        with open(directory, "solutions.pkl", "rb") as inp:
            solutions_dict = pickle.loads(inp)

        self.bus_voltages = solutions_dict["solution"]["bus_voltages"]
        self.bus_angles = solutions_dict["solution"]["bus_angles"]
        self.taps = solutions_dict["solution"]["taps"]
        self.shunts = solutions_dict["solution"]["shunts"]
        self.solution_fitness = solutions_dict["solution"]["solution_fitness"]
        self.solution_objective = solutions_dict["solution"]["solution_objective"]
        self.solution_constraints = solutions_dict["solution"]["solution_constraints"]
        self.fitness_array = solutions_dict["solution"]["fitness_array"]
        self.objective_array = solutions_dict["solution"]["objective_array"]
        self.constraint_arrays = solutions_dict["solution"]["constraint_arrays"]
        self.runs = solutions_dict["runs"]["number"]
        self.runs_fitness = solutions_dict["runs"]["runs_fitness"]
        self.runs_objective = solutions_dict["runs"]["runs_objective"]

        self.visualize()

        return

    def run(self):

        best_fitness = np.inf
        self.runs_fitness = []
        self.runs_objective = []

        for run_number in range(self.runs):
            print(f"------ Run #{run_number} ------")
            self.gwo.optimize(
                self.experiment_parameters["optimizer"]["iterations"],
                is_orpd=True,
                run_dc_power_flow=self.experiment_parameters["orpd"][
                    "run_dc_power_flow"
                ],
                voltage_penalty_lambda=self.experiment_parameters["orpd"][
                    "voltage_penalty_lambda"
                ],
                taps_penalty_lambda=self.experiment_parameters["orpd"][
                    "taps_penalty_lambda"
                ],
                shunts_penalty_lambda=self.experiment_parameters["orpd"][
                    "shunts_penalty_lambda"
                ],
            )

            self.runs_fitness.append(self.gwo.solution_fitness)
            self.runs_objective.append(self.gwo.solution_objective)

            if self.gwo.solution_fitness <= best_fitness:
                best_fitness = self.gwo.solution_fitness
                self.get_solution(self.gwo.solution)

            print(f"------------------")
        self.build_solutions_dictionary()
        self.visualize()
        return


if __name__ == "__main__":
    experiment_runner = ExperimentRunner()
    experiment_runner.run()
