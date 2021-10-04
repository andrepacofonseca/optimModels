import os
import cobra
import framed
import optimModels
from cobra.io import read_sbml_model
from framed.io.sbml import load_cbmodel
from optimModels.utils.configurations import StoicConfigurations
from optimModels.simulation.simul_problems import StoicSimulationProblem
from optimModels.optimization.decoders import DecoderReacUnderOverExpression


def cobra_or_framed(model):
    """
    aux function to decide model interpreter in optimization options
    :param model: model object
    :return boolean:
    """
    if isinstance(model, cobra.core.model.Model):
        return True
    elif isinstance(model, framed.model.cbmodel.CBModel):
        return False
    else:
        raise Exception("Model type not supported")


def stoic_simulation(model, **kwargs):
    """
    default template function  to simulate stoic models
    :param model: cobra or framed model object
    :param kwargs: all of the optional arguments
    :return StoicSimulationResult: simulation result object
    """
    uo_reacs = kwargs.get("uo", None)
    stoic_obj = kwargs.get("objective", {})
    cobra_flag = cobra_or_framed(model)
    min_flag = kwargs.get("minimize", False)
    stoic_constraints = kwargs.get("constraints", {})
    stoic_method = kwargs.get("method", StoicConfigurations.SOLVER_METHOD)
    stoic_uo_levels = kwargs.get(
        "levels", [2 ** -5, 2 ** -4, 2 ** -3, 2 ** -2, 2 ** -1, 2 ** 1, 2 ** 2, 2 ** 3, 2 ** 4, 2 ** 5])

    simulProblem = StoicSimulationProblem(
        model = model,
        objective = stoic_obj,
        minimize = min_flag,
        method = stoic_method,
        constraints = stoic_constraints,
        withCobraPy = cobra_flag
        )

    reactions = [x for x in simulProblem.get_internal_reactions() if x not in simulProblem.objective.keys()]

    res = simulProblem.simulate()
    if uo_reacs:
        simulProblem.wt_fluxes = res
        decoder = DecoderReacUnderOverExpression(ids = reactions, levels = stoic_uo_levels)
        candidate = decoder.decode_candidate_ids_to_index(identifiers = uo_reacs)
        override = decoder.get_override_simul_problem(candidate = candidate, simulProblem = simulProblem)
        res = simulProblem.simulate(overrideSimulProblem = override)

    return res


if __name__ == "__main__":
    # First Step
    # Load the Model
    optimmodels_path = os.path.dirname(optimModels.__file__)
    ecoli_model = os.path.join(optimmodels_path, "examples", "models", "Ec_iAF1260.xml")  # path to the model file
    # Option 1: Cobra
    cobra_model = read_sbml_model(filename = ecoli_model)

    # Option 2: framed
    framed_model = load_cbmodel(filename = ecoli_model, flavor = "cobra")

    # Second Step
    # Simulation
    model_biomass_reac = "R_Ec_biomass_iAF1260_core_59p81M"  # biomass reaction id
    model_glucose_reac = "R_EX_glc_e"  # example reaction id to use as constraint
    simul_constraints = {model_glucose_reac: (-10, 0)}  # format {"reac id str": (lower bound, upper bound), ...}

    framed_simulation = stoic_simulation(
        model = framed_model,
        objective = {model_biomass_reac: 1},  # format {"reac id str": 1}
        constraints = simul_constraints,  # constraints to the model
        method = "FBA",  # simulation method "FBA", "pFBA", "MOMA", and more; only with framed
        uo = {"R_PPC": 4}  # to simulate under and over expressions, format {"reac id str": uo level, ...}
        )

    print(framed_simulation.ssFluxesDistrib[model_biomass_reac])
    print(framed_simulation.ssFluxesDistrib[model_glucose_reac])
