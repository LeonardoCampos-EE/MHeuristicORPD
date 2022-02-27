import os
import pdb
import yaml

import numpy as np
import pandas as pd

from heuristics.gwo import GWO
from power_system_manager.manager import PowerSystemManager
from orpd.penalty_functions import (
    voltage_static_penalty,
    taps_sinusoidal_penalty,
    shunts_sinusoidal_penalty,
)


class ExperimentRunner:
    def __init__(self) -> None:

        with open("experiment_parameters.yaml", "r") as stream:
            self.experiment_parameters = yaml.safe_load(stream)

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

        self.gwo.pop_array[:, 0] = self.manager.first_agent

        return

    def build_penalty_functions_dict(self) -> dict:

        penalty_fun_dict = {
            "voltage": voltage_static_penalty,
            "taps": taps_sinusoidal_penalty,
            "shunts": shunts_sinusoidal_penalty,
        }

        return penalty_fun_dict

    def run(self):

        self.gwo.optimize(
            self.experiment_parameters["optimizer"]["iterations"],
            is_orpd=True,
            run_dc_power_flow=self.experiment_parameters["orpd"]["run_dc_power_flow"],
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
        self.gwo.get_fitness_chart(self.name)

        return


if __name__ == "__main__":

    experiment_runner = ExperimentRunner()
    experiment_runner.run()
