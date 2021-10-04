import os.path
import geckopy
import pandas as pd
from geckopy import GeckoModel
from cobra.io import read_sbml_model
from optimModels.simulation.simul_problems import GeckoSimulationProblem
from optimModels.optimization.decoders import DecoderProtUnderOverExpression
from optimModels.simulation.override_simul_problem import OverrideStoicSimulProblem


def convert_mmol_to_g(filename):
    """
    Aux function to convert measured protein values

    :param str filename: file path to protein measures
    :return pd.Series:
    """
    from geckopy.data import PROTEIN_PROPERTIES
    df = pd.read_csv(filename, header = None, index_col = 0)
    mmol_series = df.iloc[:, 0]

    grams = []
    for ind, value in mmol_series.iteritems():
        gram = value/1000*PROTEIN_PROPERTIES.loc[ind, 'mw']
        grams.append(gram)

    g_series = pd.Series(data = grams, index = mmol_series.index)

    return g_series


def loading_yeast_gecko(prot_measure_fractions = None, prot_measure_ggdw = None):
    """
    Loads the provided yeast gecko
    :param pd.Series prot_measure_fractions: measured fraction of proteins
    :param pd.Series prot_measure_ggdw: measured ggdw of proteins
    :return GeckoModel:
    """
    if prot_measure_fractions is None and prot_measure_ggdw is None:
        model = GeckoModel("single-pool")
    else:
        model = GeckoModel("multi-pool")
        if prot_measure_fractions:
            model.limit_proteins(fractions = prot_measure_fractions)
        else:
            model.limit_proteins(ggdw = prot_measure_ggdw)
    return model


def loading_any_gecko(path, biomass, protein = None, carbs = None):
    """
    Default templeate function to load geckos
    :param str path: string path to the sbml file
    :param str biomass: biomass function id
    :param str protein: protein reaction id
    :param str carbs: carbohydrate function id
    :return GeckoModel:
    """
    if not protein:
        protein = biomass
    if not carbs:
        carbs = biomass

    any_sbml_model = read_sbml_model(path)
    any_gecko = GeckoModel(model = any_sbml_model,
                           biomass_reaction_id = biomass,
                           protein_reaction_id = protein,
                           carbohydrate_reaction_id = carbs)
    return any_gecko


def gecko_simulation_ko(model, ko_proteins, **kwargs):
    """
    default template function to simulate protein knockouts
    :param GeckoModel model: GeckoModel object from geckopy
    :param list ko_proteins: list of str of protein ids to knockout
    :return GeckoSimulationResult: simulation result object
    """
    constraints = kwargs.get("constraints", {})
    ko_simul_problem = GeckoSimulationProblem(model, constraints = constraints)

    dic_prot = {}
    for ko_prot in ko_proteins:
        if "draw_prot_" + ko_prot in model.reactions:
            dic_prot["draw_prot_" + ko_prot] = (0, 0)
        else:
            dic_prot["prot_" + ko_prot + '_exchange'] = (0, 0)

    override_ko_simul = OverrideStoicSimulProblem(dic_prot)
    res = ko_simul_problem.simulate(overrideSimulProblem = override_ko_simul)

    return res


def gecko_simulation_uo(model, uo_proteins, **kwargs):
    constraints = kwargs.get("constraints", {})
    uo_simul_problem = GeckoSimulationProblem(model, constraints = constraints)

    gecko_wt_fluxes = uo_simul_problem.simulate()
    uo_simul_problem.wt_concentrations = gecko_wt_fluxes.get_protein_concentrations()

    gecko_uo_levels = kwargs.get(
        "levels", [2 ** -5, 2 ** -4, 2 ** -3, 2 ** -2, 2 ** -1, 2 ** 1, 2 ** 2, 2 ** 3, 2 ** 4, 2 ** 5])

    all_proteins = [x for x in uo_simul_problem.model.proteins if x not in uo_simul_problem.objective.keys()]
    all_proteins.sort()
    uo_decoder = DecoderProtUnderOverExpression(ids = all_proteins, levels = gecko_uo_levels)

    solution = set(uo_decoder.decode_candidate_ids_to_index(identifiers = uo_proteins))
    override_uo_simul = uo_decoder.get_override_simul_problem(candidate = solution, simulProblem = uo_simul_problem)

    res = uo_simul_problem.simulate(overrideSimulProblem = override_uo_simul)

    return res


if __name__ == "__main__":
    # First Step:
    # Loading the model
    # Using the provided yeast gecko model:
    yeast_single_pool = loading_yeast_gecko()
    yeast_single_pool2 = loading_yeast_gecko()
    # # # WARNING: Never use the same model for consecutive GeckoSimulation Problems # # #

    # Using the multi-pool variant:
    gecko_path = os.path.dirname(geckopy.__file__)
    protein_ggdw_path = os.path.join(gecko_path, "data_files", "sanchez-mmol_gdw.csv")
    protein_ggdw = convert_mmol_to_g(protein_ggdw_path)

    yeast_multi_pool = loading_yeast_gecko(prot_measure_ggdw = protein_ggdw)

    # Using an outside model:
    # path_to_any_gecko = "fake_path/any_gecko.xml"
    # any_gecko_biomass = "any_biomass"
    # any_gecko_single_pool = loading_any_gecko(path = path_to_any_gecko,
    #                                           biomass = any_gecko_biomass)
    # # turn into multi_pool using limit_proteins (also possible to use franctions)
    # path_to_any_ggdw = "fake_path/any_ggdw.csv"
    # any_protein_ggdw = convert_mmol_to_g(path_to_any_ggdw)
    # any_gecko_multi_pool = any_gecko_single_pool.limit_proteins(ggdw = any_protein_ggdw)

    # Second setp:
    # Simulate
    # KO Option
    print("ko")
    simul_result_ko = gecko_simulation_ko(
        model = yeast_single_pool,
        ko_proteins = [],  # list of protein_ids as str
        constraints = {}  # contraints format: {"reac_id": (lb, up), ...}
        )

    # UO Option
    print("uo")
    simul_result_uo = gecko_simulation_uo(
        model = yeast_single_pool2,
        uo_proteins = {},  # contraints format: {"protein_id": (0, expression level), ...}
        constraints = {}  # contraints format: {"reac_id": (lb, up), ...}
        )

    # Third Step:
    # Access the flux values
    simul_flux_values = simul_result_ko.ssFluxesDistrib
    print("KO", simul_flux_values["r_2111"])  # swap str for desired reaction
    simul_flux_values2 = simul_result_uo.ssFluxesDistrib
    print("UO", simul_flux_values2["r_2111"])  # swap str for desired reaction
