class IPowerSystemManager:
    def __init__(self, implementation: str, system: str):

        if implementation == "pandapower":
            from power_system_manager.implementations import PowerSystemManager

            self.manager = PowerSystemManager(system)

        elif implementation == "pypsa":
            from power_system_manager.implementations import PowerSystemManager

            self.manager = PowerSystemManager(system)

        return
