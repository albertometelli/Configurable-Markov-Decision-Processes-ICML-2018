import numpy as np
from gym.utils import seeding

class TabularPolicy(object):
    """
    Class to instantiate a policy with given policy representation
    """

    def __init__(self, rep, nS, nA):
        """
        The constructor instantiate a policy as a dictionary over states
        in which each state point to an array of probability over actions
        :param rep: policy representation as a dictionary over states
        """
        self.policy = rep
        self.nS, self.nA = nS, nA
        self.policy_matrix = np.zeros((self.nS, self.nS * self.nA))
        for state, actions in rep.iteritems():
            self.policy_matrix[state, state*nA: (state+1)*nA] = actions

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def draw_action(self, state, done):
        actions = self.policy[np.asscalar(state)]
        return np.random.choice(range(self.nA), p=actions)
        return i

    def get_rep(self):
        return self.policy

    def get_matrix(self):
        return self.policy_matrix

class TabularModel(object):

    def __init__(self, rep, nS, nA):

        self.model = rep
        self.nS, self.nA = nS, nA
        self.model_matrix =  np.zeros((self.nS * self.nA, self.nS))
        for state, actions in rep.iteritems():
            for action, elem in actions.iteritems():
                for next_state in elem:
                    self.model_matrix[state*nA+action, next_state[1]] = next_state[0]
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def draw_action(self, state, action, done):
        next_states = self.model_matrix[np.asscalar(state*self.nA+action)]
        return np.random.choice(range(self.nS), p=next_states)
        return i

    def get_rep(self):
        return self.model

    def get_matrix(self):
        return self.model_matrix

class TabularReward(object):

    def __init__(self, rep, nS, nA):

        self.reward = rep
        self.nS, self.nA = nS, nA
        self.reward_matrix =  np.zeros((self.nS * self.nA, self.nS))
        for state, actions in rep.iteritems():
            for action, elem in actions.iteritems():
                for next_state in elem:
                    self.reward_matrix[state*nA+action, next_state[1]] = next_state[2]

    def get_rep(self):
        return self.model

    def get_matrix(self):
        return self.reward_matrix