from copy import deepcopy
from random import randint, uniform, choice


class EAConfig:
    """
    This class sets up the parameters for EA
    """
    def __init__(self,
                 mut = 0.30,
                 cross = 0.30,
                 cand_size = 10,
                 max_cand_value = 8,    # for real and int representation
                 rep = 0,                       # 0 = binary, 1 = integer, else = real
                 pop_size = 10,           # min = 2
                 max_gen = 3
                 ):
        """

        :param mut: float, rate at which a mutation occurs
        :param cross: float, rate at which crossover occurs
        :param cand_size: int, size of the candidate representation
        :param max_cand_value: int or float, the max value that a value from the
            representation can take, for int or real representations only
        :param rep: int, defines the type of representation, 0 for binary, 1 for integer
            and anything else for real
        :param pop_size: int, defines the size of a population in the EA algorithm
        :param max_gen: int, defines the max number of generations in the EA
        """
        self.mutation_rate = mut
        self.crossover_rate = cross
        self.cand_size = cand_size
        self.max_cand_value = max_cand_value
        self.rep_type = rep
        self.pop_size = pop_size
        self.tourn_size = 2 + round(self.pop_size*0.05)
        self.max_gen = max_gen
        self.scoredic = {}
        self.val_dic = {}
        self.fit_dic = {}

    def __str__(self):
        configdic = {
            "mut": self.mutation_rate,
            "cross": self.crossover_rate,
            "cand size": self.cand_size,
            "max_cand": self.max_cand_value,
            "rep type": self.rep_type,
            "pop size": self.pop_size,
            "tourn size": self.tourn_size,
            "max_gen": self.max_gen,
            # "scoredic": self.scoredic,
            }
        return str(configdic)


class Candidate:
    """
    Class to represent each candidate in a population, is filled with a
    representation upon being generated a population and the score for
    that representation after being evaluated
    """
    def __init__(self, rep):
        """

        :param rep: list of int or list of float, depending on the rep_type
        """
        self.rep = rep                 # candidate representation
        self.score = None           # filled during candidate evaluation
        self.values = None          # filled during candidate evaluation
        self.fit_list = None      # filled during candidate evaluation

    def __str__(self):
        return str("{}: {}".format(self.rep, self.score))

    def update(self):
        """
        updates the candidate information, used when the candidate has already
        been evaluated
        :return: nothing
        """
        self.score = config.scoredic[str(self.rep)]
        self.fit_list = config.fit_dic[str(self.rep)]
        self.values = config.val_dic[str(self.rep)]

    def set_cand_values(self, fit_list, val, score):
        """
        sets the values of a candidate object
        :param fit_list: list of values relative to the fitness reaction
        :param val: list of all values
        :param score: evaluation score for fitness
        :return: nothing
        """
        config.fit_dic[str(self.rep)] = fit_list
        self.fit_list = config.fit_dic[str(self.rep)]
        config.val_dic[str(self.rep)] = val
        self.values = config.val_dic[str(self.rep)]
        config.scoredic[str(self.rep)] = score
        self.score = config.scoredic[str(self.rep)]


config = EAConfig()


def change_config(
        pop_size = None,
        max_gen = None,
        cand = None,
        rep = None,
        max_val = None,
        mut = None,
        cross = None
        ):
    """
    used to change the values of the EA parameters
    :param pop_size: int, defines the size of a population in the EA algorithm
    :param max_gen: int, defines the max number of generations in the EA
    :param cand: int, size of the candidate representation
    :param rep: int, defines the type of representation, 0 for binary, 1 for integer
            and anything else for real
    :param max_val: int or float, the max value that a value from the
            representation can take, for int or real representations only
    :param mut: float, rate at which a mutation occurs
    :param cross: float, rate at which crossover occurs
    :return: nothing
    """
    if pop_size:
        config.pop_size = pop_size
    if max_gen:
        config.max_gen = max_gen
    if cand:
        config.cand_size = cand
    if rep or rep == 0:
        config.rep_type = rep
    if max_val:
        config.max_cand_value = max_val
    if mut or mut == 0:
        config.mutation_rate = mut
    if cross or cross == 0:
        config.crossover_rate = cross


def reset_config():
    """
    resets (empties) the dictionary parameters of the EAConfig object
    :return: nothing
    """
    config.scoredic = {}
    config.val_dic = {}
    config.fit_dic = {}


def binary_representation():
    """
    creates a binary representation
    :return: a list of random binary values, with at least one 1
    """
    rep = [randint(0, 1) for _ in range(config.cand_size)]
    if sum(rep) == 0:
        rep = binary_representation()
    return rep


def int_representation():
    """
    creates an integer representation
    :return: a list of sorted non repeated random integers
    """
    int_rep = sorted([randint(0, config.max_cand_value) for _ in range(config.cand_size)])
    if len(int_rep) == len(set(int_rep)):
        return int_rep
    else:
        return int_representation()


def real_representation():
    """
    creates a real representation
    :return: a list of random real values
    """
    return sorted([uniform(0, config.max_cand_value) for _ in range(config.cand_size)])


def binary_to_int_rep(rep):
    """
    converts binary representations to integer format by creating a list
    of the indexes of the zeros in the original representation
    :param rep: list of binary values, representation in binary
    :return: representation in integer
    """
    return [i for i in range(len(rep)) if rep[i] == 0]


def inverse_int_rep(int_rep):
    """
    converts an integer representation with the values possible that are not
     present in the original
    :param int_rep: list of integers
    :return: list of integers, the inverse representation of the one introduced
    """
    value = 0
    if config.rep_type == 0:
        value = config.cand_size
    elif config.rep_type == 1:
        value = config.max_cand_value + 1
    new_rep = []
    for ind in range(value):
        if ind not in int_rep:
            new_rep.append(ind)
    return new_rep


def bit_flip_mutation_binary(candidate, pos = None):
    """
    alters a random or selected  binary value in the representation of a candidate
    :param candidate: candidate object
    :param pos: mutation index, autogenerated if not present
    :return: candidate object, mutated
    """
    rep = candidate.rep.copy()
    if (not pos) and (pos != 0):
        pos = randint(0, len(rep)-1)
    if rep[pos] == 0:
        rep[pos] = 1
    elif rep[pos] == 1:
        rep[pos] = 0
    if sum(rep) == 0:
        return candidate
    return Candidate(rep)


def bit_flip_mutation_int(candidate, pos = None):
    """
    alters a random or selected integer value in the representation of a candidate,
    if the result has duplicate values, returns the original
    :param candidate: candidate object
    :param pos: mutation index, autogenerated if not present
    :return: candidate object, mutated
    """
    rep = candidate.rep.copy()
    if (not pos) and (pos != 0):
        pos = randint(0, len(rep)-1)
    rep[pos] = randint(0, config.max_cand_value)
    if len(rep) == len(set(rep)):
        return Candidate(sorted(rep))
    else:
        return candidate


def bit_flip_mutation_real(candidate, pos = None):
    """
    alters a random or selected real value in the representation of a candidate
    :param candidate: candidate object
    :param pos: mutation index, autogenerated if not present
    :return: candidate object
    """
    rep = candidate.rep.copy()
    if (not pos) and (pos != 0):
        pos = randint(0, len(rep)-1)
    rep[pos] = uniform(0, config.max_cand_value)
    return Candidate(sorted(rep))


def one_point_crossover(par1, par2):
    """
    parts the representation of two parent candidates at a random point and
    creates two new candidates with the beginning of one and the end of another parent
    if one of the new candidate representations has duplicate values (for non binary
    rep_type) it returns the original candidates instead
    :param par1: candidate object
    :param par2: candidate object
    :return: two candidate objects
    """
    pos = randint(0, len(par1.rep)-1)
    rep_child1 = par1.rep[:pos]+par2.rep[pos:]
    rep_child2 = par2.rep[:pos]+par1.rep[pos:]
    if config.rep_type == 0:
        return Candidate(rep_child1), Candidate(rep_child2)

    elif config.rep_type != 0:
        if (len(rep_child1) == len(set(rep_child1))) and (len(rep_child2) == len(set(rep_child2))):
            return Candidate(sorted(rep_child1)), Candidate(sorted(rep_child2))

        else:
            return par1, par2


def uniform_crossover(par1, par2):
    """
    it creates two new candidates, for every index of the candidate representations,
    it will randomly assign the value to one of the new candidates
    if the new candidates have duplicate values, it returns the original candidates
    :param par1: candidate object
    :param par2: candidate object
    :return: two candidate objects
    """
    rep_child1 = []
    rep_child2 = []
    for i in range(len(par1.rep)):
        j = randint(0, 1)
        if j == 0:
            rep_child1.append(par1.rep[i])
            rep_child2.append(par2.rep[i])
        if j == 1:
            rep_child1.append(par2.rep[i])
            rep_child2.append(par1.rep[i])
    if config.rep_type == 0:
        return Candidate(rep_child1), Candidate(rep_child2)

    elif config.rep_type != 0:
        if (len(rep_child1) == len(set(rep_child1))) and (len(rep_child2) == len(set(rep_child2))):
            return Candidate(sorted(rep_child1)), Candidate(sorted(rep_child2))

        else:
            return par1, par2


def generate_random_popu():
    """
    creates a list of candidates with random representations according to the
    previously defined rep_type
    :return: a list of candidate objects
    """
    populist = []
    for _ in range(config.pop_size):
        if config.rep_type == 0:
            rep = binary_representation()
        elif config.rep_type == 1:
            rep = int_representation()
        else:
            rep = real_representation()
        cand = Candidate(rep)
        populist.append(cand)
    return populist


def generate_headstart_popu(sample):
    """
    generates a semi-random population given a sample to start from.
    all the generated candidates will have, at least, the indexes present in the sample
    :param sample: a candidate representation, smaller than the cand_size variable
    :return: a list of candidate objects
    """
    populist = []
    for _ in range(config.pop_size):
        if config.rep_type == 0:
            rep = deepcopy(sample)
            for i in range(len(rep)):
                if rep[i] == 0:
                    rep[i] = randint(0, 1)
        elif config.rep_type == 1:
            rep = deepcopy(sample)
            while len(set(rep)) < config.cand_size:
                rep.append(randint(0, config.max_cand_value))
            rep = sorted(list(set(rep)))
        else:
            rep = deepcopy(sample)
            while len(set(rep)) > config.cand_size:
                rep.append(uniform(0, config.max_cand_value))
            rep = sorted(list(set(rep)))
        cand = Candidate(rep)
        populist.append(cand)
    return populist


def new_popu_tourn(old_popu):
    """
    repeatedly selects the best candidate out of 15 randomly chosen two times
    to create two new candidates, that are added to a new list, until it reaches
    the desired size
    :param old_popu: list of candidate objects
    :return: list of candidate objects
    """
    new_popu = []
    keep_best = 0
    best_cand = None
    for cand in old_popu:
        if cand.score > keep_best:
            keep_best = cand.score
            best_cand = cand
    if best_cand:
        new_popu.append(best_cand)
    while len(new_popu) < config.pop_size:
        par1 = select_candidate(old_popu)
        par2 = select_candidate(old_popu)
        sib1, sib2 = maybe_crossover(par1, par2)
        sib1 = maybe_mutate(sib1)
        sib2 = maybe_mutate(sib2)
        new_popu.append(sib1)
        new_popu.append(sib2)
    new_popu = new_popu[:config.pop_size]
    return new_popu


def new_popu_changeworst(old_popu, quantity):
    """
    generates a new list of candidates by changing the x number of members
    that least contribute to the overall fitness
    :param old_popu: a list of candidate objects
    :param quantity: the number of members to be changed in each candidate
    :return: a list of candidates
    """
    new_popu = []
    keep_best = 0
    for cand in old_popu:
        if cand.score > keep_best:
            keep_best = cand.score
    for cand in old_popu:
        worst_quantity = quantity
        new_cand = deepcopy(cand)
        if new_cand.score == keep_best:
            pass
        else:
            if config.rep_type == 0:  # when change worst quantity is bigger
                if sum(cand.rep) - 1 <= quantity:  # than the number of organisms in the candidate
                    worst_quantity = sum(cand.rep) - 1  # this changes the value to one less than the cand size
            worst = find_worst(cand.fit_list, worst_quantity)
            for i in worst:
                if config.rep_type == 0:
                    new_cand = bit_flip_mutation_binary(new_cand, i)
                elif config.rep_type == 1:
                    new_cand = bit_flip_mutation_int(new_cand, i)
                else:
                    new_cand = bit_flip_mutation_real(new_cand, i)

            reverse_worst = inverse_int_rep(worst)
            keep_mut_rate = deepcopy(config.mutation_rate)
            change_config(mut = 0.15)
            if config.rep_type == 0:
                for i in reverse_worst:
                    new_cand = maybe_mutate(cand, i)
            change_config(mut = keep_mut_rate)

        new_popu.append(new_cand)
    return new_popu


def find_worst(list_of_values, quantity):
    """
    auxiliary function to changeworst,
    finds the indexes of the worst performing members
    :param list_of_values: list of values relative to the members of the candidate
        used to determine which is the worst performing ones
    :param quantity: the quantity of worst members
    :return: a list with indexes of the worst candidates, to be eliminated
    """
    if len(list_of_values) < quantity:
        raise Exception("Quantity should be lower than the number of models present.")
    worst_list = sorted([i for i in list_of_values if i])[:quantity]
    worst_ind = []
    for worst in worst_list:
        for i in range(len(list_of_values)):
            if list_of_values[i] == worst:
                worst_ind.append(i)
    return list(set(worst_ind))


def new_popu_keep_headstart_tourn(old_popu, sample):
    """
    generates a new population by tournament and after alters the candidates
    to include specific members
    :param old_popu: a list of candidate objects
    :param sample: a candidate representation, smaller than cand_size
    :return: a list of candidate objects
    """
    new_popu = new_popu_tourn(old_popu)
    for cand in new_popu:
        if config.rep_type == 0:
            for i in range(len(sample)):
                if sample[i] == 1:
                    cand.rep[i] = 1
        else:
            to_choose_from = [i for i in cand.rep if i not in sample]
            new_rep = deepcopy(sample)
            while len(set(new_rep)) < config.cand_size:
                new_rep.append(choice(to_choose_from))
            cand.rep = sorted(list(set(new_rep)))
    return new_popu


def select_candidate(popu):
    """
    selects a number of random candidates and returns the one from those with
    the best score
    :param popu: a list of candidate objects
    :return: candidate object, the candidate with the best score
    """
    cands = [randint(0, config.pop_size - 1) for _ in range(config.tourn_size)]
    bestcand = []
    bestcandscore = -99999
    for i in cands:
        if popu[i].score > bestcandscore:
            bestcandscore = popu[i].score
            bestcand = popu[i]
    return bestcand


def maybe_crossover(par1, par2):
    """
    determines randomly whether and which crossover occurs
    :param par1: candidate object
    :param par2:candidate object
    :return: two candidate objects
    """
    randval = uniform(0, 1)
    if randval > config.crossover_rate:
        return par1, par2
    if randval < config.crossover_rate:
        return uniform_crossover(par1, par2)
    else:
        return one_point_crossover(par1, par2)


def maybe_mutate(cand, pos = None):
    """
    determines randomly whether mutation occurs
    :param cand: candidate object
    :param pos: index position if necessary
    :return: candidate object
    """
    randval = uniform(0, 1)
    if randval < config.mutation_rate:
        return cand
    else:
        if config.rep_type == 0:
            return bit_flip_mutation_binary(cand, pos)
        elif config.rep_type == 1:
            return bit_flip_mutation_int(cand, pos)
        else:
            return bit_flip_mutation_real(cand, pos)


def sample_size_check(option_list, quantity):
    """
    raises errors if the size of the sample is incoherent with the chosen options
    :param option_list: option list parameter used in ea_run
    :param quantity: quantity parameter used in ea_run
    :return: nothing, raises errors when detected
    """
    if (option_list[0][0] == "headstart") or (option_list[0][0] == 1):
        if quantity == 0:
            if len(option_list[0][1]) != config.cand_size:
                raise Exception("Sample must have same length as candidate size.")
        if quantity > 0:
            if len(option_list[0][1]) > quantity:
                raise Exception("Sample length must not be lower than quantity.")
    if (option_list[1][0] == "keep") or (option_list[1][0] == 2):
        if quantity == 0:
            if len(option_list[1][1]) != config.cand_size:
                raise Exception("Sample must have same length as candidate size.")
        if quantity > 0:
            if len(option_list[1][1]) > quantity:
                raise Exception("Sample length must not be lower than quantity.")


def create_constraints(reac_list, lb = 0, up = 0):
    """
    creates a dictionary of constraints ready to be used on other functions that use fba
    :param reac_list: list of str, list of reaction ids to be constrained
    :param lb: int or float, value of the lower bound
    :param up: int or float, value of the upper bound
    :return: dict, a dictionary with reaction ids as keys, and tuples of lower and upper
        bounds as values
    """
    if lb > up:
        raise Exception("Lower bound must be lower than upper bound")
    cons_dic = {}
    for reac in reac_list:
        cons_dic[reac] = (lb, up)
    return cons_dic


def get_predecessor_reacs(model, reac_id):
    """
    recovers the reactions that produce the metabolites used in the input reaction
    :param model: framed model object
    :param reac_id: str, reaction id
    :return: list of str reaction ids
    """
    res_list = []
    target_reac = model.reactions[reac_id]
    subs = target_reac.get_substrates()[0]
    for reaction in model.reactions:
        products = model.reactions[reaction].get_products()
        if subs in products:
            res_list.append(reaction)
    return res_list


def get_fit_reac_values(cmodel, val, fit_reacs, indexes):
    """
    this function takes a CModel object, a list its respective flux values,
    and a list of reactions of which the values are to be retrieved
    and returns the values in a list
    :param cmodel: cmodel object
    :param val: values parameter from solution object
    :param fit_reacs: reactions related with the fitness evaluation
    :param indexes: indexes of the individuals present
    :return: a list of the values related to the reactions in fit_reacs
    """
    relevant_fit_values = []
    target_ids = [cmodel.models[i].id for i in indexes]
    relevant_fit_reacs = [0 for _ in target_ids]
    for ind in range(len(target_ids)):
        for fit_reac in fit_reacs:
            if fit_reac.endswith(target_ids[ind]):
                relevant_fit_reacs[ind] = fit_reac

    for reac in relevant_fit_reacs:
        if reac:
            relevant_fit_values.append(val[reac])
        else:
            relevant_fit_values.append(0)
    return relevant_fit_values


if __name__ == '__main__':
    print(binary_representation())
