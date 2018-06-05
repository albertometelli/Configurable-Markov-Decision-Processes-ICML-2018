import numpy as np
from utils import evaluator
from utils.tabular import TabularModel

from utils.tabular_operations import model_mean_tv_distance, model_sup_tv_distance


class ModelChooser(object):
    def __init__(self, nS, nA):
        self.nS = nS
        self.nA = nA

    def choose(self, model, delta_mu_P, U):
        pass


class GreedyModelChooser(ModelChooser):
    def choose(self, model, delta_mu, U):
        # GREEDY POLICY COMPUTATION
        target_model_rep = self.greedy_model(U)
        # instantiation of a target policy object
        target_model = TabularModel(target_model_rep, self.nS, self.nA)

        # EXPECTED RELATIVE ADVANTAGE COMPUTATION
        er_advantage = evaluator.compute_model_er_advantage(target_model, model, U, delta_mu)

        # POLICY DISTANCE COMPUTATIONS
        distance_sup = model_sup_tv_distance(target_model, model)
        distance_mean = model_mean_tv_distance(target_model, model, delta_mu)

        return er_advantage, distance_sup, distance_mean, target_model

    def greedy_model(self, U, tol=0.0):
        greedy_model_rep = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        # loop to give maximum probability to the greedy action,
        # if more than one is greedy then uniform on the greedy actions
        for s in range(self.nS):
            for a in range(self.nA):
                sa = s * self.nA + a
                u_array = U[sa]
                probabilities = np.zeros(self.nS)

                # uniform if more than one greedy
                max = np.max(u_array)
                s1 = np.argwhere(np.abs(u_array - max) <= tol).flatten()
                probabilities[s1] = 1. / len(s1)
                greedy_model_rep[s][a] = zip(probabilities, range(self.nS))

        return greedy_model_rep


class SetModelChooser(ModelChooser):
    def __init__(self, model_set, nS, nA):
        self.model_set = model_set
        self.n_models = len(self.model_set)
        super(SetModelChooser, self).__init__(nS, nA)

    def choose(self, model, delta_mu, U):
        er_advantages = np.zeros(self.n_models)

        for i in range(self.n_models):
            er_advantages[i] = evaluator.compute_model_er_advantage(self.model_set[i], model, U, delta_mu)

        index = np.argmax(er_advantages)
        target_model = self.model_set[index]
        er_advantage = er_advantages[index]

        # POLICY DISTANCE COMPUTATIONS
        distance_sup = model_sup_tv_distance(target_model, model)
        distance_mean = model_mean_tv_distance(target_model, model, delta_mu)

        return er_advantage, distance_sup, distance_mean, target_model

    def set(self, model, delta_mu, U):
        er_advantages = np.zeros(self.n_models)
        sup_distances = np.zeros(self.n_models)
        mean_distances = np.zeros(self.n_models)

        # advantages and distances computations
        for i in range(self.n_models):
            er_advantages[i] = evaluator.compute_model_er_advantage(self.model_set[i], model, U, delta_mu)
            sup_distances[i] = model_sup_tv_distance(self.model_set[i], model)
            mean_distances[i] = model_mean_tv_distance(self.model_set[i], model, delta_mu)

        return er_advantages, sup_distances, mean_distances


class DoNotCreateTransitionsGreedyModelChooser(ModelChooser):
    def __init__(self, original_model, nS, nA):
        self.original_model = original_model
        super(DoNotCreateTransitionsGreedyModelChooser, self).__init__(nS, nA)

    def choose(self, model, delta_mu, U):
        target_model_rep = self.dnct_greedy_model(U)
        target_model = TabularModel(target_model_rep, self.nS, self.nA)

        er_advantage = evaluator.compute_model_er_advantage(target_model, model, U, delta_mu)

        distance_sup = model_sup_tv_distance(target_model, model)
        distance_mean = model_mean_tv_distance(target_model, model,
                                                 delta_mu)

        return er_advantage, distance_sup, distance_mean, target_model

    def dnct_greedy_model(self, U, tol=0.0):
        greedy_model_rep = {s: {a: [] for a in range(self.nA)} for s in
                            range(self.nS)}

        for s in range(self.nS):
            for a in range(self.nA):
                sa = s * self.nA + a
                u_array = np.copy(U[sa])

                li = self.original_model[s][a]
                for elem in li:
                    if elem[0] == 0.:
                        u_array[elem[1]] = -np.inf

                probabilities = np.zeros(self.nS)

                # uniform if more than one greedy
                max = np.max(u_array)

                s1 = np.argwhere(np.abs(u_array - max) <= tol).flatten()
                probabilities[s1] = 1. / len(s1)
                greedy_model_rep[s][a] = zip(probabilities, range(self.nS))

        return greedy_model_rep