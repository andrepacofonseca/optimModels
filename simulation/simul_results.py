from optimModels.utils.constantes import solverStatus

class SimulationResult():
    """
    Represents the result of a metabolic model simulation at steady-state.
    """

    def __init__(self, modelId, solverStatus, ssFluxesDistrib,
                 overrideSimulProblem=None):
        """
        Create a Simulationresult instance.

        Args:
            modelId (str): Identification of metabolic model
            solverStatus (int): Simulation result (OPTIMAL = 0, UNKNOWN = 1, ERROR = 2).
            ssFluxesDistrib (dict): Fluxes distribution achieved in steady state.
            overrideSimulProblem (OverrideSimulProblem): Modifications over the metabolic model.
        """
        self.modelId = modelId
        self.solverStatus = solverStatus
        self.ssFluxesDistrib = ssFluxesDistrib
        self.overrideSimulProblem = overrideSimulProblem

    def get_solver_status(self):
        """
        Returns  the solver status result. (see optimModels.utils.constants.solverStatus)
        """
        return self.solverStatus

    def get_override_simul_problem(self):
        """
        Gets the override simulation problem.
        """
        return self.overrideSimulProblem

    def get_fluxes_distribution(self):
        """
        Gets the steady-state flux distribution {reactionId: fluxValue}.
        """
        return self.ssFluxesDistrib

class StoicSimulationResult(SimulationResult):

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def print(self):
        print("Phenotype Simulation")
        print("------------------------")
        print("model id: " + self.modelId)
        print("status: " + solverStatus.get_status_str(self.solverStatus))
        print("fluxes: ")
        for k, v in self.ssFluxesDistrib.items():
            print("     " + k + " = " + str(v))

        if self.overrideSimulProblem:
            print("mofifications:")
            for k, v in self.overrideSimulProblem.get_modifications().items():
                print("     " + k + " = " + str(v))
        print("------------------------")

class GeckoSimulationResult(SimulationResult):

    def __init__(self, modelId, solverStatus, ssFluxesDistrib=None, protConcentrations=None,
                 overrideSimulProblem=None):
        super().__init__(modelId, solverStatus, ssFluxesDistrib, overrideSimulProblem)
        self.protConcentrations = protConcentrations


    def get_protein_concentrations(self):
        """
        Gets the protein concentrations in steady-state {proteinId: concentration value}.
        """
        return self.protConcentrations

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


    def print(self):
        print("Phenotype Simulation")
        print("------------------------")
        print("model id: " + self.modelId)
        print("status: " + solverStatus.get_status_str(self.solverStatus))
        print("fluxes: ")
        for k,v in self.ssFluxesDistrib.items():
            print("     " + k + " = " + str(v))

        print("protein concentrations: ")
        for k, v in self.protConcentrations.items():
            print("     " + k + " = " + str(v))

        if self.overrideSimulProblem:
            print("mofifications:")
            for k,v in self.overrideSimulProblem.get_modifications().items():
                print("     " + k + " = " + str(v))
        print("------------------------")




class kineticSimulationResult(SimulationResult):
    """ Represents the result of a dynamic metabolic model simulation on steady-state.

    Args:
        modelId (str): identification of metabolic model
        solverStatus (int): simulation result (OPTIMAL = 0, UNKNOWN = 1, ERROR = 2).
        ssFluxesDistrib (dict): fluxes distribution achieved in steady state.
        ssConcentrations (dict): metabolites concentration in steady state.
        overrideSimulProblem (overrideKineticSimulProblem): modifications over the metabolic model.
        timeConc (dict): metabolites concentration at each time point
    """

    def __init__(self, modelId, solverStatus, ssFluxesDistrib, ssConcentrations=None,
                 overrideSimulProblem=None, timeConc=None, timeFlux=None):
        self.ssConcentrations = ssConcentrations
        self.timeConc = timeConc
        self.timeFlux = timeFlux
        super().__init__(modelId, solverStatus,ssFluxesDistrib, overrideSimulProblem)


    def get_steady_state_concentrations(self):
        """
        Gets the metabolite concentrations in steady-state {metaboliteId: concentration value}.
        """
        return self.ssConcentrations

    def get_steady_state_fluxes(self):
        """
        Gets the reaction fluxes in steady-state {fluxId: flux value}.
        """
        return self.ssFluxesDistrib

    def get_over_time_conc(self):
        """
        Gets the metabolite concentrations over time {metaboliteId: concentration value over time points}.
        """
        return self.timeConc

    def get_over_time_flux(self):
        """
        Gets the over time flux {fluxId: flux value over time points}
        """
        return self.timeFlux


    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


    def print(self):
        print("Phenotype Simulation")
        print("------------------------")
        print("model id: " + self.modelId)
        print("status: " + solverStatus.get_status_str(self.solverStatus))
        print("steady state fluxes: ")
        for k,v in self.ssFluxesDistrib.items():
            print("     " + k + " = " + str(v))

        print("steady state concentrations: ")
        for k, v in self.ssConcentrations.items():
            print("     " + k + " = " + str(v))

        if self.overrideSimulProblem:
            print("modifications:")
            for k,v in self.overrideSimulProblem.get_modifications().items():
                print("     " + k + " = " + str(v))
        print("------------------------")
