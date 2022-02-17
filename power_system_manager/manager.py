import pdb

import pandapower as pp
import numpy as np


class PowerSystemManager:
    def __init__(self, system: str) -> None:

        self.system = system

        if system == "14":
            from pandapower.networks import case14

            self.network = case14()
        elif system == "30":
            from pandapower.networks import case_ieee30

            self.network = case_ieee30()
        elif system == "57":
            from pandapower.networks import case57

            self.network = case57()

        self.get_network_parameters()

        return

    def get_network_parameters(self):

        """
        This function get the following network paramters:
        * nb: number of buses;
        * nt: number of trafos with controlled taps;
        * ns: number of active shunts;
        * ng: number of generators;
        * lines: the network lines;

        """

        # Sort the network indices
        self.network.bus = self.network.bus.sort_index()
        self.network.res_bus = self.network.res_bus.sort_index()
        self.network.gen = self.network.gen.sort_index()
        self.network.line = self.network.line.sort_index()
        self.network.shunt = self.network.shunt.sort_index()
        self.network.trafo = self.network.trafo.sort_index()

        # Fix issue where some networks are initialized with negative taps
        self.network.trafo.tap_pos = np.abs(self.network.trafo.tap_pos)

        # Get the trafos with tap control
        self.network.trafo = self.network.trafo.sort_values("tap_pos")
        self.nt = self.network.trafo.tap_pos.count()

        # Get the number of active buses
        self.nb = self.network.bus.name.count()

        # Get the number of active shunts
        self.ns = self.network.shunt.in_service.count()

        # Get the number of generators in the system
        self.ng = self.network.gen.in_service.count()

        # Get the network lines parameters
        self.get_line_parameters()

        # Get the conductance matrix
        self.get_conductance_matrix()

        # Get the allowed shunt values for the given system
        self.get_shunt_values()

        # Get the shunt masks
        self.get_shunt_masks()

        # Get the first search agent position
        self.get_first_agent()

        # Get the trafo taps values
        self.get_tap_values()

        return

    def get_line_parameters(self):

        self.lines = {}

        # Get the indices of the buses from which the networks are originated
        self.lines["start"] = self.network.line.from_bus.to_numpy().astype(int)

        # Get the indices of the buses to which the networks are connected
        self.lines["end"] = self.network.line.to_bus.to_numpy().astype(int)

        # Get the network voltage in kilo volts
        v_network = self.network.bus.vn_kv.to_numpy()

        # Calculate the network base impedance
        # Z = V²/100M
        z_base = ((v_network * 1000) ** 2) / 100e6

        # Get the lines resistance in pu
        # r_pu = r_ohm / z_base
        self.lines["r_pu"] = np.zeros(shape=(self.network.line.index[-1] + 1,))
        for i in range(self.network.line.index[-1] + 1):
            self.lines["r_pu"][i] = (
                self.network.line.r_ohm_per_km.iloc[i] / z_base[self.lines["start"][i]]
            )

        # Get the lines reactances in pu
        # x_pu = x_ohm / z_base
        self.lines["x_pu"] = np.zeros(shape=(self.network.line.index[-1] + 1,))
        for i in range(self.network.line.index[-1] + 1):
            self.lines["x_pu"][i] = (
                self.network.line.x_ohm_per_km.iloc[i] / z_base[self.lines["start"][i]]
            )

        return

    def get_conductance_matrix(self):

        conductances = np.array(
            [self.lines["r_pu"] / self.lines["r_pu"] ** 2 + self.lines["x_pu"] ** 2]
        )

        # Get the nodal conductance matrix, it's equivalent to the real part of the nodal admitance matrix
        self.conductance_matrix = np.zeros(shape=(self.nb, self.nb))
        self.conductance_matrix[self.lines["start"], self.lines["end"]] = conductances

        return

    def get_shunt_values(self):

        if self.system == "14":
            self.shunt_values = np.array([[0.0, 0.19, 0.34, 0.39]])
        elif self.system == "30":
            self.shunt_values = np.array(
                [[0.0, 0.19, 0.34, 0.39], [0.0, 0.0, 0.05, 0.09]]
            )
        elif self.system == "57":
            self.shunt_values = np.array(
                [
                    [0.0, 0.12, 0.22, 0.27],
                    [0.0, 0.04, 0.07, 0.09],
                    [0.0, 0.0, 0.10, 0.165],
                ]
            )
        elif self.system == "118":
            self.shunt_values = np.array(
                [
                    [-0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.06, 0.07, 0.13, 0.14, 0.2],
                    [-0.25, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.15],
                    [0.0, 0.0, 0.0, 0.08, 0.12, 0.2],
                    [0.0, 0.0, 0.0, 0.0, 0.1, 0.2],
                    [0.0, 0.0, 0.0, 0.0, 0.1, 0.2],
                    [0.0, 0.0, 0.0, 0.0, 0.1, 0.2],
                    [0.0, 0.0, 0.0, 0.0, 0.1, 0.2],
                    [0.0, 0.06, 0.07, 0.13, 0.14, 0.2],
                    [0.0, 0.06, 0.07, 0.13, 0.14, 0.2],
                ]
            )
        elif self.system == "300":
            self.shunt_values = np.array(
                [
                    [0.0, 2.0, 3.5, 4.5],
                    [0.0, 0.25, 0.44, 0.59],
                    [0.0, 0.19, 0.34, 0.39],
                    [-4.5, 0.0, 0.0, 0.0],
                    [-4.5, 0.0, 0.0, 0.0],
                    [0.0, 0.25, 0.44, 0.59],
                    [0.0, 0.25, 0.44, 0.59],
                    [-2.5, 0.0, 0.0, 0.0],
                    [-4.5, 0.0, 0.0, 0.0],
                    [-4.5, 0.0, 0.0, 0.0],
                    [-1.5, 0.0, 0.0, 0.0],
                    [0.0, 0.25, 0.44, 0.59],
                    [0.0, 0.0, 0.0, 0.15],
                    [0.0, 0.0, 0.0, 0.15],
                ]
            )

        return

    def get_shunt_masks(self):

        if self.system == "14":
            self.shunt_masks = np.array([[0.25, 0.25, 0.25, 0.25]])
        elif self.system == "30":
            self.shunt_masks = np.array(
                [[0.25, 0.25, 0.25, 0.25], [0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]]
            )
        elif self.system == "57":
            self.shunt_masks = np.array(
                [
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                ]
            )
        elif self.system == "118":
            self.shunt_masks = np.array(
                [
                    [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                    [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0],
                    [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
                    [0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
                    [0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
                    [0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
                    [0.0, 0.0, 0.25, 0.25, 0.25, 0.25],
                    [0.0, 0.0, 0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                    [0.0, 0.0, 0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                    [0.0, 0.0, 0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                    [0.0, 0.0, 0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                    [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0],
                    [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0],
                ]
            )
        elif self.system == "300":
            self.shunt_masks = np.array(
                [
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.5, 0.5, 0.0, 0.0],
                    [0.5, 0.5, 0.0, 0.0],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.5, 0.5, 0.0, 0.0],
                    [0.5, 0.5, 0.0, 0.0],
                    [0.5, 0.5, 0.0, 0.0],
                    [0.5, 0.5, 0.0, 0.0],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.0, 0.0, 0.5, 0.5],
                    [0.0, 0.0, 0.5, 0.5],
                ]
            )
        return

    def get_tap_values(self, tap_step=0.00625):

        self.tap_values = np.arange(start=0.9, stop=1.1, step=tap_step)

        return

    def get_first_agent(self):

        v = self.network.gen.vm_pu.to_numpy(dtype=np.float32)
        taps = 1 + (
            (
                self.network.trafo.tap_pos.to_numpy(dtype=np.float32)[0 : self.nt]
                + self.network.trafo.tap_neutral.to_numpy(dtype=np.float32)[0 : self.nt]
            )
            * (
                self.network.trafo.tap_step_percent.to_numpy(dtype=np.float32)[
                    0 : self.nt
                ]
                / 100
            )
        )
        shunts = -self.network.shunt.q_mvar.to_numpy(dtype=np.float32) / 100

        self.first_agent = {}
        self.first_agent["v"] = v
        self.first_agent["taps"] = taps
        self.first_agent["shunts"] = shunts

        return

    def insert_voltages_from_agent(self, agent: np.ndarray):
        self.network.gen.vm_pu = agent[: self.ng]
        return

    def insert_taps_from_agent(self, agent: np.ndarray):
        """
        The transformer taps should be inserted as position values, instead of
        their pu values. To convert from pu to position:
            tap_pos = [(tap_pu - 1)*100]/tap_step_percent] + tap_neutral
        """
        self.network.trafo.tap_pos[: self.nt] = self.network.trafo.tap_neutral[
            : self.nt
        ] + (
            (agent[self.ng : self.ng + self.nt] - 1.0)
            * (100 / self.network.trafo.tap_step_percent[: self.nt])
        )
        return

    def insert_shunts_from_agent(self, agent: np.ndarray):
        """
        The shunt unit on the network is MVAr and it has a negative value
        To convert from pu to MVAr negative:
            mvar = pu * -100

        """
        self.network.shunt.q_mvar = agent[self.ng + self.nt :] * (-100)

        return

    def run_ac_power_flow(self, algorithm="nr", use_numba=True, enforce_q_lims=True):

        pp.runpp(
            self.network,
            algorithm=algorithm,
            numba=use_numba,
            enforce_q_lims=enforce_q_lims,
        )

        return
