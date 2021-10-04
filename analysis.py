import framed.cobra.simulation as fr
from optimModels.utils.configurations import GeckoConfigurations
from geckopy import GeckoModel


def FBA(model, objective = None, minimize = False, constraints = None, withCobraPy = False):
    """
    A FBA wrapper

    returns a fluxes dictionary
    """
    if isinstance(model, GeckoModel) or withCobraPy:
        with model:
            if constraints:
                for rId in list(constraints.keys()):
                    reac = model.reactions.get_by_id(rId)
                    reac.bounds = (constraints.get(rId)[0] * GeckoConfigurations.SCALE_CONSTANT,
                                   constraints.get(rId)[1] * GeckoConfigurations.SCALE_CONSTANT)
            if objective:
                model.objective = next(iter(objective))
                if minimize:
                    model.objective.direction = 'min'
                else:
                    model.objective.direction = 'max'
            solution = model.optimize()
            fluxes = solution.fluxes.to_dict()
            # remove the scalling factor
            fluxes = {k: v / GeckoConfigurations.SCALE_CONSTANT for k, v in fluxes.items()}

    else:
        # framed
        solution = fr.FBA(model, objective = objective, minimize = minimize, constraints = constraints)
        fluxes = solution.values
    return fluxes
