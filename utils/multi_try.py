# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 16:44:09 2019

@author: BiSBII
"""

import math
import logging
import time
from optimModels.utils.constantes import solverStatus
from optimModels.utils.utils import MyPool

from geckopy import GeckoModel
from optimModels.simulation.simul_problems import GeckoSimulationProblem
from optimModels.optimization.evolutionary_computation import OptimProblemConfiguration, EAConfigurations
from optimModels.utils.constantes import optimType
from optimModels.optimization.decoders import DecoderProtKnockouts
from optimModels.utils.configurations import GeckoConfigurations
from optimModels.optimization.evaluation_functions import build_evaluation_function


try:
    import cPickle as pickle
except ImportError:
    import pickle


def evaluator(candidates, args):
    """
    This function allows the evaluation of candidate solutions.

    Args:
        candidates (list): A list of candidate solutions
        args (dict): A dictionary of keyword arguments

    Returns:
        list of floats: a list of fitness values
    """
    config = args["configuration"]
    decoder = config.get_decoder()
    simulProblem = config.get_simulation_problem()
    fitness = []
    # start_time = time.time()
    for candidate in candidates:
        # print(candidate)
        overrideProblem = decoder.get_override_simul_problem(candidate, simulProblem)

        fitInd = -1.0
        try:
            res = simulProblem.simulate(overrideProblem)
            # print("--- %s seconds ---" % (time.time() - start_time), 'end simul')
            if res.get_solver_status() == solverStatus.OPTIMAL or res.get_solver_status() == solverStatus.SUBOPTIMAL:
                fitInd = config.get_evaluation_function().get_fitness(res, candidate)
                if math.isnan(fitInd):
                    fitInd = -1.0
        except Exception as error:
            print("Oops! Solver problems.  ", error)
            logging.getLogger('optimModels').warning("Oops! Solver problems." + str(error))
        fitness.append(fitInd)
        # print("--- %s seconds ---" % (time.time() - start_time), 'end')

    return fitness


# change the original function to start a non-demoniac workers
def parallel_evaluation_mp(candidates, args):
    """
    Evaluate the candidates in parallel using ``multiprocessing``.

    This function allows parallel evaluation of candidate solutions.
    It uses the standard multiprocessing library to accomplish the
    parallelization. The function assigns the evaluation of each
    candidate to its own job, all of which are then distributed to the
    available processing units.

    Args:
        candidates: list the candidate solutions
        args: a dictionary of keyword arguments

    Returns:

    Notes:
    All arguments to the evaluation function must be pickleable.
    Those that are not will not be sent through the ``args`` variable and will be unavailable to your function.
    Required keyword arguments in args:
    - *mp_evaluator* -- actual evaluation function to be used (This function
      should have the same signature as any other inspyred evaluation function.)

    Optional keyword arguments in args:

    - *mp_nprocs* -- number of processors that will be used (default machine
      cpu count)
    """
    logger = logging.getLogger('optimModels')

    try:
        evaluator = args['mp_evaluator']
    except KeyError:
        logger.error('parallel_evaluation_mp requires \'mp_evaluator\' be defined in the keyword arguments list.')
        raise
    try:
        nprocs = args['mp_nprocs']
    except KeyError:
        logger.error('parallel_evaluation_mp requires \'mp_nprocs\' be defined in the keyword arguments list.')
        raise
    start = time.time()
    pickled_args = {}
    for key in args:
        try:
            pickle.dumps(args[key])
            pickled_args[key] = args[key]
        except (TypeError, pickle.PickleError, pickle.PicklingError):
            logger.debug('unable to pickle args parameter {0} in parallel_evaluation_mp'.format(key))
            pass
    # print("--- %s seconds ---" % (time.time() - start), 'end_pickled')

    try:
        pool = MyPool(processes=nprocs)
        results = [pool.apply_async(evaluator, ([c], pickled_args)) for c in candidates]
        pool.close()
        pool.join()
    except (OSError, RuntimeError) as e:
        logger.error('failed parallel_evaluation_mp: {0}'.format(str(e)))
        raise
    else:
        end = time.time()
        print('completed parallel_evaluation_mp in {0} seconds'.format(end - start))
        logger.debug('completed parallel_evaluation_mp in {0} seconds'.format(end - start))
        # print("--- %s seconds ---" % (time.time() - start), 'end_pop')
        return [r.get()[0] for r in results]


model = GeckoModel('single-pool')
const = {'r_1714_REV': (0, 1)}
simulProblem = GeckoSimulationProblem(model, const)
ids = [x for x in simulProblem.model.proteins if x not in simulProblem.objective.keys()]
decoder = DecoderProtKnockouts(ids)
evalFunc = build_evaluation_function("targetFlux", ["r_2056"])

eaConfig = EAConfigurations()
optimProbConf = OptimProblemConfiguration(simulProblem, type=optimType.PROTEIN_KO,
                                          decoder=decoder, evaluationFunc=evalFunc, EAConfig=eaConfig,
                                          scaleProblem=GeckoConfigurations.SCALE_CONSTANT)

candidates = [{321, 51}, {65, 130, 259, 146, 660, 91}, {488, 266, 235, 558, 119, 220}, {81, 431}, {160, 425, 162, 606},
              {97, 330, 654, 212, 671}, {419, 589, 398, 302, 275, 214}, {64, 648, 275, 221, 574}, {11, 308, 21, 614},
              {18, 77, 222}, {426, 60}, {110, 667, 358}, {290, 387, 362, 119, 223}, {353, 482, 680, 563, 604},
              {428, 373, 14}, {56, 483, 108, 341}, {359, 18, 435, 404, 93}, {640, 97, 498, 629},
              {353, 557, 52, 20, 638}, {298, 295}, {280, 634, 63}, {235}, {577, 644, 382}, {2, 201, 361, 306, 403},
              {290, 324, 520, 457, 243}, {314, 459, 606}, {27, 21, 190, 47}, {337, 363, 372, 71},
              {546, 344, 361, 503, 504}, {596}, {335}, {99, 396, 574, 79}, {652, 559, 656, 276, 277, 414}, {680}, {92},
              {316, 62, 399}, {427}, {97, 420, 37, 101, 270, 433}, {512, 345}, {316, 500}, {368, 633, 618, 351},
              {416, 668, 131, 43, 81, 572}, {641, 133, 513, 71}, {613, 639, 597, 223}, {480, 106, 431, 278, 374, 58},
              {249, 241, 79, 199}, {85, 486}, {544, 39, 296, 406, 279}, {679}, {336, 430, 253, 86}, {504},
              {429, 654, 400, 405, 55}, {227, 230, 584, 659, 216, 573}, {476}, {503, 6, 439}, {224, 649, 586, 434, 311},
              {556, 275, 596, 216, 569}, {578, 276}, {35, 94}, {472, 498}, {457}, {491, 591}, {674, 498, 165, 71},
              {480, 425, 553, 14, 25}, {656, 664, 564, 149}, {60}, {141, 567}, {456, 290, 324}, {345}, {282, 243},
              {323, 410, 347}, {509, 132, 623, 58, 603, 125}, {26, 484, 586, 247}, {10, 258, 346, 621}, {236}, {353},
              {104, 107, 11, 239, 176, 630}, {264, 106}, {448, 131, 652}, {132, 134, 487, 51, 56},
              {450, 293, 527, 158, 222}, {602, 236, 279}, {424, 387, 189, 199}, {616, 299, 182, 216, 667, 159},
              {496, 389}, {608, 72, 653, 594, 347}, {388, 76, 303, 49, 280}, {360, 663, 215}, {233, 202, 453, 126},
              {608, 641, 388, 264, 658}, {121, 501, 358, 645}, {296, 622, 562, 669, 575}, {417, 69, 663}, {613, 110},
              {537}, {259, 509}, {477, 313, 170, 586}, {300, 510}, {459}, {330}]


def evaluation(candidates, args):
    simulProblem = args["simulProblem"]
    decoder = args["decoder"]
    fitness = []
    for candidate in candidates:
        override = decoder.get_override_simul_problem(candidate, simulProblem)
        res = simulProblem.simulate(override)
        fit = res.ssFluxesDistrib['r_2056']
        fitness.append(fit)
    return fitness


def parallel_evolution(candidates, args):
    logger = logging.getLogger('optimModels')
    pickled_args = {}
    for key in args:
        try:
            pickle.dumps(args[key])
            pickled_args[key] = args[key]
        except (TypeError, pickle.PickleError, pickle.PicklingError):
            logger.debug('unable to pickle args parameter {0} in parallel_evaluation_mp'.format(key))
            pass

    try:
        nprocs = 2
        pool = MyPool(processes=nprocs)
        results = [pool.apply_async(evaluator, ([c], pickled_args)) for c in candidates]
        pool.close()
        pool.join()
    except (OSError, RuntimeError) as e:
        logger.error('failed parallel_evaluation_mp: {0}'.format(str(e)))
        raise
    else:
        end = time.time()
        print('completed parallel_evaluation_mp in {0} seconds'.format(end - start))
        logger.debug('completed parallel_evaluation_mp in {0} seconds'.format(end - start))
        # print("--- %s seconds ---" % (time.time() - start), 'end_pop')
        return [r.get()[0] for r in results]


if __name__ == '__main__':
    args = {'max_generations': 1, 'candidate_max_size': 6, 'num_elites': 1, 'num_selected': 50,
            'crossover_rate': 0.9, 'mutation_rate': 0.4, 'new_candidates_rate': 0.4,
            'configuration': optimProbConf,
            'results_file': 'C:/Users/BiSBII/Results/UO_newtry/KOx_newlevels_Gecko_SUCC_max6.csv',
            'tournament_size': 3, '_ec': None, 'mp_nprocs': 4, 'mp_evaluator': evaluator}
    start = time.time()
    parallel_evaluation_mp(candidates, args)
    end = time.time()
    print('completed evaluation in {0} seconds'.format(end - start))
