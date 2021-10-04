import os
import optimModels
from framed import load_cbmodel
from optimModels.utils.constantes import optimType
from optimModels.optimization.run import cbm_strain_optim
from optimModels.unittests.stoic_simul import cobra_or_framed
from optimModels.utils.configurations import StoicConfigurations
from optimModels.simulation.simul_problems import StoicSimulationProblem
from optimModels.optimization.evaluation_functions import build_evaluation_function


def stoic_optimization(model, optim_type, eval_func, **kwargs):
    cand_size = kwargs.get("size", None)
    stoic_obj = kwargs.get("objective", {})
    withCobraPy = cobra_or_framed(model)
    min_flag = kwargs.get("minimize", False)
    multi_thread = kwargs.get("isMultiProc", False)
    stoic_constraints = kwargs.get("constraints", {})
    output_file = kwargs.get("output_file", "results.csv")
    stoic_method = kwargs.get("method", StoicConfigurations.SOLVER_METHOD)
    stoic_uo_levels = kwargs.get(
        "levels", [2 ** -5, 2 ** -4, 2 ** -3, 2 ** -2, 2 ** -1, 2 ** 1, 2 ** 2, 2 ** 3, 2 ** 4, 2 ** 5])

    types_dict = {
        "KO": optimType.REACTION_KO,
        "UO": optimType.REACTION_UO,
        "MEDKO": optimType.MEDIUM,
        "MEDUO": optimType.MEDIUM_LEVELS,
        "KO+MEDKO": optimType.MEDIUM_REACTION_KO
        }

    stoic_simul_problem = StoicSimulationProblem(
        model = model,
        objective = stoic_obj,
        minimize = min_flag,
        constraints = stoic_constraints,
        method = stoic_method,
        withCobraPy = withCobraPy)

    critical_proteins = kwargs.get("critical_proteins", [])
    if critical_proteins == "auto":
        critical_proteins = stoic_simul_problem.find_essential_drains()

    res = cbm_strain_optim(
        simulProblem = stoic_simul_problem,
        evaluationFunc = eval_func,
        levels = stoic_uo_levels,
        type = types_dict[optim_type],
        criticalReacs = critical_proteins,
        isMultiProc = multi_thread,
        candidateSize = cand_size,
        resultFile = output_file
        )

    return res


if __name__ == "__main__":
    # First Step
    # Load the Model: Cobra or Framed
    optimmodels_path = os.path.dirname(optimModels.__file__)
    ecoli_model = os.path.join(optimmodels_path, "examples", "models", "Ec_iAF1260.xml")  # path to the model file
    framed_model = load_cbmodel(filename = ecoli_model, flavor = "cobra")

    # Second Step
    # Evaluating Function         # different functions require different arguments
    ec_stoic_biomass = "R_Ec_biomass_iAF1260_core_59p81M"
    ec_stoic_succinate = "R_EX_succ_e"
    stoic_eval_function = build_evaluation_function(
        "WYIELD",   # evaluating function id
        ec_stoic_biomass,   # biomass reaction id
        ec_stoic_succinate,   # target reaction id
        alpha = 0.3,                         # percentage of maximum target considered
        minBiomassValue = 0.03135  # minimum biomass for viable solution
        )

    # Third Step
    # Run Optimization
    result = stoic_optimization(
        model = framed_model,
        optim_type = "KO",  # "KO", "UO", "MEDKO", "MEDUO", "KO+MEDKO"
        eval_func = stoic_eval_function,
        constraints = {},  # contraints format: {"reac_id": (lb, up), ...}
        critical_proteins = "auto",  # can also use a list of reactions
        isMultiProc = False,  # replace with True to use multi-processing
        size = 10,  # chose max number of alterations, None uses default
        output_file = "optimization_results.csv"   # chose path of the results file
        )
    print(result)
