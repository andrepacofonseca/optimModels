import time
from optimModels.comm_optim.CModel import *
from optimModels.comm_optim.ea_setup import *
from copy import deepcopy
from framed import load_cbmodel


def generate_ids(ids_direc = None, namelist = None, sep = "_"):
    if (ids_direc is None) & (namelist is None):
        raise Exception("Please insert either a directory (str) or a list of names (list).")
    elif (ids_direc is not None) & (namelist is not None):
        raise Exception("Please insert either a directory (str) or a list of names (list), not both.")
    else:
        pass

    if namelist:
        pass
    else:
        # fetching names from directory
        directory = os.fsencode(dir)
        namelist = []
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            namelist.append(filename)

    # creating different tags for each file
    namedic = {}
    count = 0
    for file in namelist:
        splitname = file.split(sep)
        newname = ""
        for i in splitname:
            newname += i[:1]
        if newname not in namedic.values():
            namedic[file] = newname
        else:
            namedic[file] = newname + "_{}".format(str(count))
            count += 1

    return namedic


def generate_first_population_hub(option, option2 = None):
    """
    function that chooses the function to be used to generate the inicial population
    :param option: the option chosen
    :param option2: complementary for some options
    :return: a generated list of candidates according to the input options
    """
    if (option == 0) or (option == "random"):
        return generate_random_popu()
    if (option == 1) or (option == "headstart"):
        if not option2:
            raise Exception("Headstart option needs sample.")
        return generate_headstart_popu(sample = option2)
    raise Exception("Input options are incoherent.")


def generate_new_population_hub(popu, option, option2 = None):
    """
    function that chooses the function to be used to generate the new populations
    warning: changeworst should not be used when an objective has been given
    and a fitness reaction has not been given
    :param popu: a list of candidate objects
    :param option: the option chosen
    :param option2: complementary for some options
    :return:
    """
    if (option == 0) or (option == "tourn"):
        return new_popu_tourn(popu)
    elif (option == 1) or (option == "changeworst"):
        if (type(option2) is not int) or (option2 is None):
            raise Exception("Changeworst option needs quantity")
        return new_popu_changeworst(popu, quantity = option2)
    elif (option == 2) or (option == "keep"):
        if (type(option2) is not list) or (option2 is None):
            raise Exception("Keep option needs headstart sample")
        return new_popu_keep_headstart_tourn(popu, sample = option2)
    raise Exception("Input options are incoherent.")


def evaluate(cmodel, popu, obj_list, cons, quantity = 0, fit_reacs = None):
    """
    runs an fba of the community model for every candidate representation present
    in the population list and attributes a score to each respective candidate
    :param cmodel: CModel object
    :param popu: list of candidate objects
    :param obj_list: list of str reactions ids
    :param cons: dict of constraints
    :param quantity: int, flag for type of rep
    :param fit_reacs: reactions to be used for fitness
    :return: nothing
    """
    if not obj_list:
        objs = deepcopy(cmodel.cmodel.get_objective())
    else:
        objs = {}
    for cand in popu:
        if not cand.score:
            if str(cand.rep) in config.scoredic:        # caso o score já tenha sido calculado
                cand.update()
            elif str(cand.rep) not in config.scoredic:  # caso o score ainda não tenha sido calculado
                cons_copy = deepcopy(cons)
                if not quantity:
                    indexes = binary_to_int_rep(cand.rep)
                else:
                    indexes = inverse_int_rep(cand.rep)
                model_ko = cmodel.knockout(
                    list_of_model_ids = [cmodel.models[indx].id for indx in indexes],
                    objective_list = obj_list,
                    constraints = cons_copy
                    )

                val = model_ko.values

                fit_list = []
                indexes2 = inverse_int_rep(indexes)
                min_biom = 0

                if not val:
                    fit_list = [0 for _ in indexes2]
                    val = []
                    score = 0
                    cand.set_cand_values(fit_list, val, score)
                    continue

                score_flag = False
                if fit_reacs:
                    fit_list = get_fit_reac_values(cmodel, val, fit_reacs, indexes2)
                elif not obj_list:
                    for indx in indexes2:
                        fit_list.append(val[cmodel.model_dic[cmodel.models[indx].id].info["obj"]])
                        min_biom += 0.1 * cmodel.model_dic[cmodel.models[indx].id].info["fobj"]
                elif obj_list:
                    score_flag = True

                if config.rep_type == 0 and not score_flag:
                    fit_list_rep_0 = []
                    for ind in range(config.cand_size):
                        if cand.rep[ind] == 1:
                            fit_list_rep_0.append(fit_list.pop(0))
                        else:
                            fit_list_rep_0.append(0)
                    fit_list = fit_list_rep_0

                # score = sum(fit_list) if not score_flag else model_ko.fobj
                score = (sum(fit_list), model_ko.fobj)[score_flag]
                if not fit_reacs and not obj_list:
                    if score < min_biom:
                        score = 0
                if not score:
                    score = 0

                cand.set_cand_values(fit_list, val, score)

                for reac in objs:
                    cmodel.cmodel.reactions[reac].objective = objs[reac]


def ea_run(
        option_list, list_of_models = None, cmodel = None,
        obj_list = None, cons = None, quantity = 0, fitness = None, goal = None
        ):
    """
    created a CModel out of the models introduced, creates a population of
    representations, evaluates each one, generates a new populations from the using
    the previous generation's best candidates, until a max number of gens is reached
    :param option_list: options for the various hub functions
    :param list_of_models: list of framed model objects
    :param cmodel: CModel object
    :param obj_list: list of str reaction ids
    :param cons: dict of constraints
    :param quantity: number of models to be selected out of the list
    :param fitness: reaction to be used for fitness
    :param goal: numeric value to stop the algorithm is a solution reaches it
    :return: candidate object, the candidate that best performed in the last generation
    """
    reset_config()

    if not obj_list:
        obj_list = []
    if not cons:
        cons = {}

    maxval = config.max_cand_value

    if list_of_models and cmodel:
        raise Exception("Either a list of models or a cmodel has to be included, not both")
    if not list_of_models and not cmodel:
        raise Exception("Either a list of models or a cmodel has to be included")

    if list_of_models:
        if quantity >= len(list_of_models):
            raise Exception("Quantity needs to be less than the number of models")

        ids = generate_ids(namelist = [model.id for model in list_of_models])
        cmodel_id = "cmodel"
        for model_id in ids.values():
            cmodel_id += "_" + model_id

        cmodel = CModel(cmodel_id, list_of_models)
        maxval = len(list_of_models)

    if cmodel:
        if quantity > len(cmodel.models):
            raise Exception("Quantity needs to be less than the number of models")
        maxval = len(cmodel.models)

    if quantity:
        change_config(cand = quantity, rep = 1, max_val = maxval - 1)
    else:
        change_config(cand = maxval, rep = 0)

    sample_size_check(option_list, quantity)

    if fitness:
        fit_reacs = get_predecessor_reacs(cmodel.cmodel, fitness)
    else:
        fit_reacs = None

    popu = generate_first_population_hub(option_list[0][0], option_list[0][1])
    gens = 0
    exit_flag = False
    while not exit_flag:
        evaluate(cmodel, popu, obj_list, cons, quantity, fit_reacs)
        best_score = max([candi.score for candi in popu])
        if goal and (best_score > goal):
            exit_flag = True
        if not exit_flag:
            if gens == config.max_gen:
                exit_flag = True
            else:
                popu = generate_new_population_hub(popu, option_list[1][0], option_list[1][1])
                gens += 1

    bestcand = []
    bestcandscore = -9999
    for indx in range(len(popu)):
        if popu[indx].score > bestcandscore:
            bestcandscore = popu[indx].score
            bestcand = popu[indx]
    print("Final Population:")
    for cand in popu:
        print(cand)
    return bestcand


if __name__ == "__main__":
    pass
