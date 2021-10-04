from abc import ABCMeta, abstractmethod
from itertools import chain
from optimModels.utils.configurations import StoicConfigurations
import pandas as pd

class EvaluationFunction:
    """
    This abstract class should be extended by all evaluation functions classes.

    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_fitness(self, simulationResult, candidate):
        return

    @abstractmethod
    def method_str(self):
        return

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

class MinCandSizeAndMaxTarget(EvaluationFunction):
    """
    This evaluation function finds the solution with the minimum candidate size and levels considering the maximization
    of the target flux.

    Args:
        maxCandidateSize (int): maximum of candidate size
        maxTargetFlux (str): reaction id to maximize

    """
    def __init__(self, maxCandidateSize, maxTargetFlux):
        self.maxCandidateSize = maxCandidateSize
        self.objective = maxTargetFlux

    def get_fitness(self, simulResult, candidate):
        fluxes = simulResult.get_fluxes_distribution()
        numModifications = len(list(chain.from_iterable(candidate)))
        sumObj=0
        for rId, ub in self.objective.items():
            ub = StoicConfigurations.DEFAULT_UB if ub is None else ub
            f = 1 if fluxes[rId]>=ub else 1-((ub-fluxes[rId])/ub)
            sumObj = sumObj + f
        objFactor=sumObj/len(self.objective)

        return  objFactor/numModifications


    def method_str(self):
        return "Minimize the number of modifications while maximize the target flux."

    @staticmethod
    def get_id():
        return "MinNumberReacAndMaxFlux"

    @staticmethod
    def get_name():
        return "Minimize the number of modifications while maximize the target flux."

    @staticmethod
    def get_parameters_ids():
        return ["Maximum of modifications allowed", "Target reactions"]

class MinCandSizeWithLevelsAndMaxTarget(EvaluationFunction):
    #TODO: Validar o que acontece em caso do ub do target ser 0 ou seja o fluxes[rId] é negativo (ver se há hipotese de isto acontecer)

    def __init__(self, maxCandidateSize, levels, maxTargetFlux):
        self.maxCandidateSize = maxCandidateSize
        self.levels = levels
        self.objective = maxTargetFlux

    def get_fitness(self, simulResult, candidate):
        fluxes = simulResult.get_fluxes_distribution()
        maxUptake = len(candidate) * self.levels[-1]
        sumUptake = 0
        sumObj = 0

        for rId in candidate.keys():
            sumUptake = sumUptake + candidate[rId]

        for rId, ub in self.objective.items():
            ub = StoicConfigurations.DEFAULT_UB if ub is None else ub
            f = 1 if fluxes[rId]>=ub else 1-((ub-fluxes[rId])/ub)
            sumObj = sumObj + f
        objFactor=sumObj/len(self.objective)

        # best solution when factors are close to 1
        upFactor = sumUptake/maxUptake
        lenFactor = len(candidate)/ self.maxCandidateSize

        return  objFactor/(upFactor * lenFactor)


    def method_str(self):
        return "Minimize the number and the fluxes of candidate while maximize the target flux."

    @staticmethod
    def get_id():
        return "MinCandSizeWithLevelsAndMaxTarget"

    @staticmethod
    def get_name():
        return "Minimize the number and the fluxes of candidate while maximize the target flux."

    @staticmethod
    def get_parameters_ids():
        return ["Maximum number of modifications allowed", "Levels", "Target reactions"]

class MinCandSize(EvaluationFunction):
    """
    This class implements the "minimization of number of reactions" objective function. The fitness is given by
    1 - size(candidate)/ max_candidate_size, where the max_candidate_size is the maximum size that a candidate can have
    during optimization.

    Args:
        maxCandidateSize(int): Maximum size allowed for candidate
        minFluxes (dict): Minimal value for fluxes to consider fitness different of 0 (key: reaction id, value: minimum of flux).

    """
    def __init__(self, maxCandidateSize, minFluxes):
        self.maxCandidateSize = maxCandidateSize
        self.minFluxes = minFluxes

    def get_fitness(self, simulResult, candidate):
        fluxes = simulResult.get_fluxes_distribution()
        for rId, flux in self.minFluxes.items():
            if fluxes[rId]< flux:
                return 0
        return 1 - len(candidate)/(self.maxCandidateSize + 1)

    def method_str(self):
        return "Minimum number of active reactions."

    @staticmethod
    def get_id():
        return "MinNumberReac"

    @staticmethod
    def get_name():
        return "Minimum number of active reactions."

    @staticmethod
    def get_parameters_ids():
        return ["Number maximum of modification allowed", "Minimum of targets flux values."]


class TargetFlux(EvaluationFunction):
    """
    This class implements the "target flux" objective function. The fitness is given by the flux value of the target reaction.

    Args:
        targetReactionId (str): Reaction identifier of the target compound production.

    """
    def __init__(self, targetReactionId):
        #TODO: take only the first element
        self.targetReactionId = targetReactionId[0]

    def get_fitness(self, simulResult, candidate):
        fluxes = simulResult.get_fluxes_distribution()
        if self.targetReactionId not in list(fluxes.keys()):
            raise ValueError("Reaction id is not present in the fluxes distribution.")
        return fluxes[self.targetReactionId]

    def method_str(self):
        return "Target Flux: " + self.targetReactionId

    @staticmethod
    def get_id():
        return "targetFlux"

    @staticmethod
    def get_name():
        return "Target Flux"

    @staticmethod
    def get_parameters_ids():
        return ["Target reaction id"]

class BPCY (EvaluationFunction):
    """
    This class implements the "Biomass-Product Coupled Yield" objective function. The fitness is given by the equation:
    (biomass_flux * product_flux)/ uptake_flux

    Args:
        biomassId (str): Biomass reaction identifier
        productId (str): Target product reaction identifier
        uptakeId (str): Reaction of uptake

        """
    def __init__(self, biomassId, productId, uptakeId):
        self.biomassId = biomassId
        self.productId = productId
        self.uptakeId = uptakeId

    def get_fitness(self, simulResult, candidate):
        ssFluxes= simulResult.get_fluxes_distribution()
        ids = list(ssFluxes.keys())
        if self.biomassId not in ids or self.productId not in ids or self.uptakeId not in ids:
            raise ValueError("Reaction ids is not present in the fluxes distribution. Please check id objective function is correct.")
        if abs(ssFluxes[self.uptakeId])==0:
            return 0
        return (ssFluxes[self.biomassId] * ssFluxes[self.productId])/abs(ssFluxes[self.uptakeId])

    def method_str(self):
        return "BPCY =  (" + self.biomassId +  " * " + self.productId +") / " + self.uptakeId

    @staticmethod
    def get_id():
        return "BPCY"

    @staticmethod
    def get_name():
        return "Biomass-Product Coupled Yield"

    @staticmethod
    def get_parameters_ids():
        return ["Biomass id", "Product id", "Uptake id"]

class BP_MinModifications (EvaluationFunction):
    """
        This class is based the "Biomass-Product Coupled Yield" objective function but considering the candidate size. The fitness is given by the equation:
        (biomass_flux * product_flux)/ candidate_size)

        Args:
            biomassId (str): biomass reaction identifier
            productId (str): target product reaction identifier

        """
    def __init__(self, biomassId, productId):
        self.biomassId = biomassId
        self.productId = productId

    def get_fitness(self, simulResult, candidate):
        ssFluxes= simulResult.get_fluxes_distribution()
        ids = list(ssFluxes.keys())
        if self.biomassId not in ids or self.productId not in ids:
            raise ValueError("Reaction ids is not present in the fluxes distribution. Please check id objective function is correct.")
        size = len(candidate)
        # optimization of medium and KO .. the candidate is a tuple of lists
        if (isinstance(candidate[0], list)):
            size = len(candidate[0]) + len(candidate[1])
        print(candidate, str(ssFluxes[self.biomassId]), str(ssFluxes[self.productId]))
        return (ssFluxes[self.biomassId] * ssFluxes[self.productId])/size

    def method_str(self):
        return "BP_MinModifications=  (" + self.biomassId +  " * " + self.productId +") / candidate_size "

    @staticmethod
    def get_id():
        return "BP_MinModifications"

    @staticmethod
    def get_name():
        return "Biomass-Product with minimun of modifications"

    @staticmethod
    def get_parameters_ids():
        return ["Biomass id", "Product id"]

class FitExperimentalDataSSConc(EvaluationFunction):
    """
    This class implements the fit to experimental data (steady state concentration of metabolites) objective function. The fitness is given by the difference between experimental and simulated data.

    Args:
        data_location (str): Location of the experimental data

    """
    def __init__(self, data_location):
        if type(data_location[0])!=str:
            raise ValueError("Data location not properly inserted, needs to be a string.")
        self.data_location = data_location[0]

    def get_fitness(self, simulResult, candidate):
        ss_conc_sim = simulResult.get_steady_state_concentrations()
        ss_conc_exp = self.read_data()
        fit = self.objfunc_met_ss(ss_conc_sim,ss_conc_exp)
        return (1/fit)

    def objfunc_met_ss(self,ss_conc_sim,ss_conc_exp):
        res_conc_ss=0
        for i in ss_conc_exp.keys():
            soma=0
            for j in ss_conc_exp[i]["Metabolite"]:
                if len(j.split("+"))==1:
                    for k in ss_conc_sim.keys():
                        if k==j:
                            exp=ss_conc_exp[i].loc[ss_conc_exp[i]["Metabolite"] == j, 'Concentration (mM)'].iloc[0]
                            sim=ss_conc_sim[k]
                            if exp!=0:
                                soma+=(abs(exp-sim))/exp
                                break
                            else:
                                soma+=abs(sim)
                elif len(j.split("+"))>1:
                    sim=0
                    for l in j.split("+"):
                        for k in ss_conc_sim.keys():
                            if k==l:
                                sim+=ss_conc_sim[l]
                    exp=ss_conc_exp[i].loc[ss_conc_exp[i]["Metabolite"] == j, 'Concentration (mM)'].iloc[0]
                    if exp!=0:
                        soma+=(abs(exp-sim))/exp
                    else:
                        soma+=abs(sim)
            res_conc_ss+=soma
        return(res_conc_ss)

    def read_data(self):
        metabolomics_ss=pd.read_excel(self.data_location,sheet_name=None)
        ss_conc_exp={}
        for i in metabolomics_ss.keys():
            ss_conc_exp[i]=metabolomics_ss[i].dropna()
            ss_conc_exp[i] = ss_conc_exp[i][ss_conc_exp[i]['Metabolite'].astype(str).str.startswith('M_')]
        return(ss_conc_exp)

    def method_str(self):
        return ("FitExperimentalDataSSConc: " + self.data_location[0])

    @staticmethod
    def get_id():
        return "FitExperimentalDataSSConc"

    @staticmethod
    def get_name():
        return "Fit experimental data to simulation data of steady state metabolites concentration"

    @staticmethod
    def get_parameters_ids():
        return ["Data Location"]

class FitExperimentalDataSSFlux(EvaluationFunction):
    """
    This class implements the fit to experimental data (steady state reactions flux) objective function. The fitness is given by the difference between experimental and simulated data.

    Args:
        data_location (str): Location of the experimental data

    """
    def __init__(self, data_location):
        if type(data_location[0])!=str:
            raise ValueError("Data location not properly inserted, needs to be a string.")
        self.data_location = data_location[0]

    def get_fitness(self, simulResult, candidate):
        ss_flux_sim = simulResult.get_steady_state_fluxes()
        ss_flux_exp = self.read_data()
        fit = self.objfunc_flux_ss(ss_flux_sim,ss_flux_exp)
        return (1/fit)

    def objfunc_flux_ss(self,ss_flux_sim,ss_flux_exp):
        res_flux_ss=0
        for i in ss_flux_exp.keys():
            soma=0
            for j in ss_flux_exp[i]["Reaction"]:
                if len(j.split("+"))==1:
                    for k in ss_flux_sim.keys():
                        if k==j:
                            if "Flux (mM/s)" in ss_flux_exp[i]:
                                exp=ss_flux_exp[i].loc[ss_flux_exp[i]["Reaction"] == j, "Flux (mM/s)"].iloc[0]
                                sim=ss_flux_sim[k]
                                if exp!=0:
                                    soma+=(abs(exp-sim))/abs(exp)
                                else:
                                    soma+=abs(sim)
                                break
                            elif "Flux (normalized)" in ss_flux_exp[i]:
                                normal=ss_flux_sim["R_PTS"]
                                exp=ss_flux_exp[i].loc[ss_flux_exp[i]["Reaction"] == j, "Flux (normalized)"].iloc[0]
                                sim=ss_flux_sim[k]
                                if exp!=0:
                                    soma+=(abs(exp-sim/normal))/abs(exp)
                                else:
                                    soma+=abs(sim/normal)
                                break
                elif len(j.split("+"))>1:
                    sim=0
                    if "Flux (mM/s)" in ss_flux_exp[i]:
                        exp=ss_flux_exp[i].loc[ss_flux_exp[i]["Reaction"] == j, "Flux (mM/s)"].iloc[0]
                        for l in j.split("+"):
                            for k in ss_flux_sim.keys():
                                if k==l:
                                    sim+=ss_flux_sim[k]
                        if exp!=0:
                            soma+=(abs(exp-sim))/abs(exp)
                        else:
                            soma+=abs(sim)
                    elif "Flux (normalized)" in ss_flux_exp[i]:
                        normal=ss_flux_sim["R_PTS"]
                        exp=ss_flux_exp[i].loc[ss_flux_exp[i]["Reaction"] == j, "Flux (normalized)"].iloc[0]
                        for l in j.split("+"):
                            for k in ss_flux_sim.keys():
                                if k==l:                
                                    sim+=ss_flux_sim[k]
                        if exp!=0:
                            soma+=(abs(exp-sim/normal))/abs(exp)
                        else:
                            soma+=abs(sim/normal)
            res_flux_ss+=soma
        return(res_flux_ss)

    def read_data(self):
        fluxomics_ss=pd.read_excel(self.data_location,sheet_name=None)
        ss_flux_exp={}
        for i in fluxomics_ss.keys():
            ss_flux_exp[i] = fluxomics_ss[i].dropna()
            ss_flux_exp[i] = ss_flux_exp[i][ss_flux_exp[i]['Reaction'].astype(str).str.startswith('R_')]
        return(ss_flux_exp)

    def method_str(self):
        return ("FitExperimentalDataSSFlux: " + self.data_location[0])

    @staticmethod
    def get_id():
        return "FitExperimentalDataSSFlux"

    @staticmethod
    def get_name():
        return "Fit experimental data to simulation data of steady state reactions flux value"

    @staticmethod
    def get_parameters_ids():
        return ["Data Location"]

class FitExperimentalDataTimeConc(EvaluationFunction):
    """
    This class implements the fit to experimental data (over time metabolite concentration) objective function. The fitness is given by the difference between experimental and simulated data.


    Args:
        data_location (str): Location of the experimental data

    """
    def __init__(self, data_location):
        if type(data_location[0])!=str:
            raise ValueError("Data location not properly inserted, needs to be a string.")
        self.data_location = data_location[0]

    def get_fitness(self, simulResult, candidate):
        time_conc_sim = simulResult.get_over_time_conc()
        time_conc_exp = self.read_data()
        fit = self.objfunc_time_conc(time_conc_sim,time_conc_exp)
        return(1/fit)

    def objfunc_time_conc(self,time_conc_sim,time_conc_exp):
        """
        A dict is created for the purpose of finding the closest simulated time points to the experimental ones,
        since it is harder to simulate every single time point for every dataset simultaneously.
        The fit is then done by comparing experimental and closest (in time) simulated data.
        """
        res_conc_time=0
        for i in time_conc_exp.keys():
            total_diff=0
            time_points={}
            for j in time_conc_exp[i].index:
                diff_saved=999999
                for k in time_conc_sim.index:
                    diff=abs(j-k)
                    if diff<diff_saved:
                        diff_saved=diff
                        prev_k=k
                    elif diff>diff_saved:
                        break
                time_points[j]=prev_k
            for l in time_conc_exp[i]:
                soma=0
                if len(l.split("+"))==1:
                    for p in time_points.keys():
                        if time_conc_exp[i][l].loc[p]==None:
                            soma+=0
                        else:
                            exp = time_conc_exp[i][l].loc[p]
                            sim = time_conc_sim[l].loc[time_points[p]]
                            if exp!=0:
                                soma+=(abs(exp-sim))/exp
                            else:
                                soma+=abs(sim)
                else:
                    sim=0
                    for p in time_points.keys():
                        if time_conc_exp[i][l].loc[p]==None:
                            soma+=0
                        else:
                            exp = time_conc_exp[i][l].loc[p]
                            for a in l.split("+"):
                                sim+=time_conc_sim[a].loc[time_points[p]]
                            if exp!=0:
                                soma+=(abs(exp-sim))/exp
                            else:
                                soma+=abs(sim)
                total_diff+=soma
            res_conc_time+=total_diff
        return (res_conc_time)

    def read_data(self):
        metabolomics_time=pd.read_excel(self.data_location,sheet_name=None)
        time_conc_exp={}
        for l in metabolomics_time.keys():
            time_conc_exp[l] = metabolomics_time[l].filter(regex='^Time|M_',axis=1)
            time_conc_exp[l] = time_conc_exp[l].dropna(thresh=2)
            time_conc_exp[l] = time_conc_exp[l].set_index("Time (s)")
            time_conc_exp[l] = time_conc_exp[l].where(pd.notnull(time_conc_exp[l]), None)
        return(time_conc_exp)

    def method_str(self):
        return ("FitExperimentalDataTimeConc: " + self.data_location[0])

    @staticmethod
    def get_id():
        return "FitExperimentalDataTimeConc"

    @staticmethod
    def get_name():
        return "Fit experimental data to simulation data of over time metabolites concentration"

    @staticmethod
    def get_parameters_ids():
        return ["Data Location"]

def build_evaluation_function(id, *args):
    """
    Function to return an evaluation function instance.

    Args:
        id (str): Name of the objective function. The implemented objective functions should be registed in constants.objFunctions class
        *args (list of str): The number of arguments depends of the objective function chosen by user.
    Returns:
        EvaluationFunction: return an evaluation function instance.
    """

    if id == BPCY.get_id():
        objFunc = BPCY(args[0],args[1],args[2])
    elif id == TargetFlux.get_id():
        objFunc = TargetFlux(args[0])
    elif id == MinCandSize.get_id():
        objFunc = MinCandSize(args[0], args[1])
    elif id ==  BP_MinModifications.get_id():
        objFunc = BP_MinModifications(args[0], args[1])
    elif id == MinCandSizeWithLevelsAndMaxTarget.get_id():
        objFunc = MinCandSizeWithLevelsAndMaxTarget (args[0], args[1], args[2])
    elif id == MinCandSizeAndMaxTarget.get_id():
        objFunc = MinCandSizeAndMaxTarget(args[0], args[1])
    elif id == FitExperimentalDataSSConc.get_id():
        objFunc = FitExperimentalDataSSConc(args[0])
    elif id == FitExperimentalDataSSFlux.get_id():
        objFunc = FitExperimentalDataSSFlux(args[0])
    elif id == FitExperimentalDataTimeConc.get_id():
        objFunc = FitExperimentalDataTimeConc(args[0])
    else:
        raise Exception("Wrong objective function!")

    return objFunc

