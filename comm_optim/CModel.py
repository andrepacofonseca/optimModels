from optimModels.comm_optim import ea_knockout
from optimModels.comm_optim.ea_setup import *
from copy import deepcopy
from framed import load_cbmodel, FBA, Community, Environment


class CModel:
    """
    Class to incorporate framed models into framed community models along with
        more accessible information
    """

    def __init__(self, community_id, models, extracellular_compartment_id = "e", empty_flag = False):
        """
        Creates the community model and auxiliary dictionaries to make
            some information more accessible
        :param community_id: str, id of the community model
        :param models: list of framed model objects, will become a framed community model
        :param extracellular_compartment_id: str, every model in models must have this
            as the id of the external compartment
        """
        self.community_id = community_id
        self.models = models
        self.extracellular_compartment_id = extracellular_compartment_id
        self.empty_flag = empty_flag
        self.model_dic = self.__model_dic()
        self.cmodel = self.__create_cmodel()
        self.objectives = self.cmodel.get_objective()
        self.pex_cons = self.__get_pex_constraints()
        self.biomass_reactions = self.__biomass_reactions()

    def __model_dic(self):
        """
        creates a dictionary with the model ids as keys and iModel objects as values
        :return: dict, a dictionary with the model ids as keys and iModel objects as values
        """
        model_dic = {}
        for model in self.models:
            model_dic[model.id] = IModel(model)
        return model_dic

    def __create_cmodel(self):
        """
        creates the community model itself, with a complete medium provided by
        the framed functions Community.merge and Environment.complete
        :return: framed model object, a community model merging every model
            present in self.models
        """
        community = Community(
            community_id = self.community_id,
            models = self.models,
            extracellular_compartment_id = self.extracellular_compartment_id,
            create_biomass = False,
            interacting = True
            )
        c_model = community.merged

        if not self.empty_flag:
            Environment.complete(c_model, inplace = True)
        elif self.empty_flag:
            Environment.empty(c_model, inplace = True)
        return c_model

    def __get_pex_constraints(self):
        """
        this function aggregates pseudo exchange (pex) reactions by model of origin,
        these are the original exchange reactions from the single models that have to be
        constrained in order to knockout an organism
        :return: dict, a dictionary with the compartments as keys
            and the respective pex reactions as values
        """
        exch = self.cmodel.get_exchange_reactions()
        ext_comp = [i for i in self.cmodel.get_reaction_compartments(exch[0])][0]
        exch_metas = []
        for reac in exch:
            exch_metas += \
                self.cmodel.reactions[reac].get_substrates() + \
                self.cmodel.reactions[reac].get_products()
        pex_reacs = []
        for meta in exch_metas:
            pex_reacs += self.cmodel.get_metabolite_reactions(meta)
        pex_per_comp = {}
        for pex in pex_reacs:
            comps = self.cmodel.get_reaction_compartments(pex)
            for comp in comps:
                if comp != ext_comp:
                    if comp not in pex_per_comp:
                        pex_per_comp[comp] = [pex]
                    elif comp in pex_per_comp:
                        pex_per_comp[comp].append(pex)

        for model_name in list(self.model_dic.keys()):
            for two_comp_reac in self.cmodel.reactions:
                check_endswith = [compart.endswith(model_name) for
                                  compart in self.cmodel.get_reaction_compartments(two_comp_reac)]
                if sum(check_endswith) == len(check_endswith):
                    if two_comp_reac not in pex_per_comp[self.extracellular_compartment_id + "_" + model_name]:
                        pex_per_comp[self.extracellular_compartment_id + "_" + model_name].append(two_comp_reac)

        pex_constraints = {}
        for comp in pex_per_comp:
            pex_constraints[comp] = create_constraints(pex_per_comp[comp])
        return pex_constraints

    def __biomass_reactions(self):
        """
        creates a dictionary with the biomass reactions as keys and the
        respective model ids as values
        :return: dict, a dictionary with the biomass reaction as keys
            and the iModel object as values
        """
        biomass_reactions = {}
        for model in self.model_dic:
            biomass_reactions[self.model_dic[model].info["obj"]] = model
        return biomass_reactions

    def fba(self, objective_list = None, constraints = None):
        """
        performs an fba from framed while assuring that when the biomass reaction
        of a model is not in the objectives or constraints, that reactions must be at least
        10% of its original value when simulated as a single model
        :param objective_list: list of str, reactions to be used as the objective
            instead of default
        :param constraints: dict, dictionary with str reaction ids as keys and tuples
            with two numeric values, lower bound and upper bound as values
        :return: Solution object, the result of the fba
        """
        if not objective_list:
            objective_list = []
        if not constraints:
            constraints = {}
        for reac in self.biomass_reactions:
            if (reac not in objective_list) and (reac not in constraints):
                constraint = {reac: (0.1*self.model_dic[self.biomass_reactions[reac]].info["fobj"], 10000)}
                constraints.update(constraint)
        if objective_list:
            objective = {key: 0.0 for key in self.objectives}
            for i in objective_list:
                objective[i] = 1.0
            res = FBA(model = self.cmodel, objective = objective, constraints = constraints)
        else:
            res = FBA(model = self.cmodel, constraints = constraints)
        return res

    def knockout(self, list_of_model_ids, objective_list = None, constraints = None):
        """
        this functions will run an fba while knocking out the desired models from the
        community model by constraining the respective pex reactions to be 0 (zero),
        fba function used is the one in the class, not the framed one directly
        :param list_of_model_ids: list of str, list of model ids from the models that
            are to be knocked out
        :param objective_list: list of str, reactions to be used as the objective
            instead of default
        :param constraints: dict, dictionary with str reaction ids as keys and tuples
            with two numeric values, lower bound and upper bound as values
        :return: Solution object, the result of the fba
        """
        if not objective_list:
            objective_list = []
        if not constraints:
            constraints = {}
        # for model in list_of_model_ids:
        #     for reac in self.biomass_reactions:
        #         if reac.endswith(model):
        #             constraints[reac] = (0, 0)
        for model in list_of_model_ids:
            constraints.update(self.pex_cons[self.extracellular_compartment_id + "_" + model])
            biom_reac = self.model_dic[model].info["obj"]
            self.cmodel.reactions[biom_reac].objective = 0
        res = self.fba(objective_list = objective_list, constraints = constraints)
        return res

    def ea(self, options, quantity = 0, obj_list = None, cons = None, fitness = None, goal = None):
        """
        optimization of the composition of the cmodel through evolutionary algorithms
        :param options: options for the ea, see ea_knockout
        :param quantity: number of organisms to choose, 0 to use any number
        :param obj_list: list of reaction ids to use as objective
        :param cons: dict of constraints to be considered
        :param fitness: exchange reaction to use as fitness
        :param goal: goal to stop run when met
        :return: the representation of the winning candidate and its score
        """
        if not obj_list:
            obj_list = []
        if not cons:
            cons = {}
        return ea_knockout.ea_run(options, cmodel = self, obj_list = obj_list, cons = cons,
                                  quantity = quantity, fitness = fitness, goal = goal)


class IModel:
    """
    Auxiliary class to represent the individual models in the community model from the
    CModel class
    """

    def __init__(self, model):
        """
        parameters are automatically incorporated when called by the model_dic from
        the CModel class
        :param model: framed model object
        """

        self.model = model
        self.fix_none_bounds(10, 813)
        self.info = {
            "obj": "{}_{}".format([reac for reac in self.model.get_objective()][0], self.model.id),
            "fobj": FBA(self.model).fobj
            }

    def fix_none_bounds(self, exch = 25, regular = 1000):
        """
        alters the values of the lower and upper bounds that are None, allows different
        values for exchange reactions
        :return: nothing
        """
        for reac in self.model.reactions:
            if reac in self.model.get_exchange_reactions():
                if not self.model.reactions[reac].lb:
                    self.model.reactions[reac].lb = -exch
                if not self.model.reactions[reac].ub:
                    self.model.reactions[reac].ub = exch
            else:
                if not self.model.reactions[reac].lb:
                    self.model.reactions[reac].lb = -regular
                if not self.model.reactions[reac].ub:
                    self.model.reactions[reac].ub = regular


if __name__ == "__main__":
    import os
    import optimModels
    optimmodels_path = os.path.dirname(optimModels.__file__)
    models_path = os.path.abspath(os.path.join(optimmodels_path, os.pardir, os.pardir, "examples", "models"))

    model1 = "Yokenella_regensburgei_ATCC_43003.xml"
    model2 = "Acinetobacter_junii_SH205.xml"
    model3 = "Clostridiales_sp_1_7_47FAA.xml"
    model4 = "Achromobacter_xylosoxidans_A8.xml"
    model5 = "Achromobacter_xylosoxidans_NBRC_15126.xml"
    model6 = "Acidaminococcus_intestini_RyC_MR95.xml"
    model7 = "Acidaminococcus_sp_D21.xml"
    model8 = "Acinetobacter_calcoaceticus_PHEA_2.xml"
    model9 = "Acinetobacter_lwoffii_WJ10621.xml"
    model10 = "Actinobacillus_pleuropneumoniae_L20.xml"
    model_list = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10]

    list_models = [load_cbmodel(filename = str(os.path.join(models_path, model)),
                                flavor = "cobra:other") for model in model_list]
    fit = "R_EX_M_taur__91__e__93___e_pool"
    test_reac = "R_EX_M_succ__91__e__93___e_pool"

    create_cm = CModel("testing", list_models)
    run_options = [("random", [1, 0, 1, 0, 0, 0, 0, 0, 0]), ("tourn", 1)]
    ea_run = create_cm.ea(options = run_options, quantity = 3, obj_list = [], fitness = None, goal = None)
    print(ea_run)
