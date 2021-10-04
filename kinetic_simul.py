import os
import optimModels
from optimModels.model.kineticModel import load_kinetic_model
from optimModels.simulation.override_simul_problem import OverrideKineticSimulProblem
from optimModels.simulation.simul_problems import KineticSimulationProblem
from optimModels.utils.configurations import KineticConfigurations


def kinetic_simulation(kinet_model, factors, **kwargs):
    """
    default template function to load models
    :param load_kinetic_model() kinet_model:
    :param dict factors:
    :return: simulation result
    """
    time = kwargs.get("time", KineticConfigurations.STEADY_STATE_TIME)
    override = OverrideKineticSimulProblem(factors=factors)
    simulProblem = KineticSimulationProblem(kinet_model, tSteps=[0, time])
    res = simulProblem.simulate(overrideSimulProblem=override)
    return res


if __name__ == "__main__":
    # First Step:
    # Loading the model
    optimmodels_path = os.path.dirname(optimModels.__file__)
    example_chassagnole_path = os.path.join(optimmodels_path, "examples", "models", "chassagnole2002.xml")
    ex_kinetic_model = load_kinetic_model(filename = example_chassagnole_path, kmap = {})

    # Second Step
    # Simulation  
    ex_factors = {}  # in the format {'parameter id': factor, ...} factors should be base 2, or 0 for KO
    result = kinetic_simulation(kinet_model = ex_kinetic_model, factors = ex_factors)
    result.print()
