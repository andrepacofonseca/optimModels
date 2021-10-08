import os
import optimModels
from framed import load_cbmodel
from optimModels.comm_optim.CModel import CModel
from optimModels.comm_optim.ea_setup import change_config


def create_options(first = "random", new = "tourn", **kwargs):
    """
    auxilliary function to create the options parameter for ea
    :param str first: "random" or "headtstart"
    :param str new: "tourn", "keep", or "changeworst"
    :param kwargs: aditional parameters
    :return:
    """
    sample = kwargs.get("sample", [])
    quantity = kwargs.get("quantity", 1)

    if first == "random":
        fp_tuple = (first, None)
    elif first == "headstart":
        fp_tuple = (first, sample)
    else:
        raise Exception("First Population option not supported.")

    if new == "tourn":
        np_tuple = (new, None)
    elif new == "keep":
        np_tuple = (new, sample)
    elif new == "changeworst":
        np_tuple = (new, quantity)
    else:
        raise Exception("New Population option not supported.")

    return [fp_tuple, np_tuple]


if __name__ == "__main__":
    # Step 1
    # Load any ammount of models in framed objects
    # All of the models should be loaded in the same notation style
    # The 10 models loaded below were arbitrarily chosen
    optimmodels_path = os.path.dirname(optimModels.__file__)
    models_path = os.path.abspath(os.path.join(optimmodels_path, "examples", "models"))

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

    # Step 2
    # Create the Community Model
    comm_model = CModel(
        community_id = "model_id",
        models = list_models,
        empty_flag = False  # This creates a complete medium, to crete an empty medium input True
        )                                       # That way you can open any medium through constraints

    # Step 3
    # Run the EA Optimization
    # 3.1 Configure the optimization parameters
    change_config(  # Leave parameters as None to use defaults
        pop_size = None,  # changes the size of the populations of candidates
        max_gen = None,  # changes the max number of generations
        mut = None,  # changes the mutation rate when creating new solutions, 0 < x < 1
        cross = None  # changes the crossover rate when creating new solutions, 0 < x < 1
        )

    # 3.2 Configure the type of optimization
    ea_optim_options = create_options(
        first = "random",
        new = "tourn"
        )
    # this function defines the method of creating the first and subsequent populations
    # on first: there is "random" and "headstart"
    # on new: there is "tourn", "keep", and "changeworst"
    # changeworst requires the aditional argument quantity (int),
    # indicating the number of least performing organisms to be swaped out each gen
    # headstart and keep, require the aditional argument sample (list)
    # sample format depends on the quantity argument of the comm_model.ea() function
    # if quantity = 0, sample is a list of 1s and 0s indicating the organisms present with 1s
    # example =  [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    # elif quantity > 0, sample is a list of ints indicating the position of the organisms
    # example: [1, 0, 1, 0, 0, 0, 0, 0, 0, 0] == [0, 2]
    # below are some examples

    options_1 = create_options()  # default settings, completely random
    options_2a = create_options(  # keeps organisms in all solutions, when quantity = 0
        first = "headstart",
        new = "keep",
        sample = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    options_2b = create_options(  # keeps organisms in all solutions, when quantity > 0
        first = "headstart",
        new = "keep",
        sample = [0, 2])
    options_3 = create_options(
        new = "changeworst",
        quantity = 2)  # changes the 2 worst performing
    # organisms of each candidate every generation

    # 3.3 Run the Optimization
    ea_optim_result = comm_model.ea(
        options = options_1,
        quantity = 3,
        obj_list = [],  # list of objective reactions str ids
        cons = {},  # dictionary of constraints, format {"reac_id" : (lower bound, upper bound), ...}
        fitness = None,  # reactino id str to use as fitness instead of the objective
        goal = None  # goal to stop algorithm before the max gens if desired value is reached
        )

    print("Best Solution: ", ea_optim_result)
    print("Organisms present in solution:")
    for indx in ea_optim_result.rep:
        print("\t", model_list[indx])
