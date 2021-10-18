class PowerSystemInterface:
    def __init__(self, implementation: str, system: str):

        if implementation == "pandapower":
            from power_systems.implementations import PandaPowerImplementation

            self.power_systems_impl = PandaPowerImplementation(system)

        elif implementation == "pypsa":
            from power_systems.implementations import PyPsaImplementation

            self.power_systems_impl = PyPsaImplementation(system)

        self.get_system_variables = self.power_systems_lib.get_system_variables

        self.insert_voltages = self.power_systems_lib.insert_voltages
        self.insert_taps = self.power_systems_impl.insert_taps
        self.insert_shunts = self.power_systems_impl.insert_shunts

        self.run_ac_power_flow = self.power_systems_impl.run_ac_power_flow
        self.run_dc_power_flow = self.power_systems_impl.run_dc_power_flow

        return
