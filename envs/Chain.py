import numpy as np
from gym import spaces
from gym.utils import seeding
from envs import discrete
from utils.matrix_builders import *
import copy

class NChainEnv(discrete.DiscreteEnv):
    """n-Chain environment
    This game presents moves along a linear chain of states, with two actions:
     0) forward, which moves along the chain but returns no reward
     1) backward, which returns to the beginning and has a small reward
    The end of the chain, however, presents a large reward, and by moving
    'forward' at the end of the chain this large reward can be repeated.
    At each action, there is a small probability that the agent 'slips' and the
    opposite transition is instead taken.
    The observed state is the current state in the chain (0 to n-1).
    This environment is described in section 6.1 of:
    A Bayesian Framework for Reinforcement Learning by Malcolm Strens (2000)
    http://ceit.aut.ac.ir/~shiry/lecture/machine-learning/papers/BRL-2000.pdf
    """
    def __init__(self, n=2, slip=0.2, small=2, large1=10, large2=8, max_steps=500):
        assert n == 2
        self.n = n
        self.slip = slip  # probability of 'slipping' an action
        self.small = small  # payout for 'backwards' action
        self.large1 = large1  # payout at end of chain for 'forwards' action
        self.large2 = large2
        self.state = 0  # Start at beginning of the chain
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(self.n)
        self.seed()
        self.max_steps = max_steps
        self.steps = 0
        self.param = 0.5

        self.nS = n
        self.nA = 2

        self.gamma = 0.99
        self.horizon = max_steps

        self.isd = np.array([1., 0])
        self.mu = self.isd

        self.P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        self.P_slip1 = copy.deepcopy(self.P)
        self.P_slip0 = copy.deepcopy(self.P)

        ll = [(self.P, self.slip), (self.P_slip0, 0.), (self.P_slip1, 1.)]
        for P, slip in ll:
            P[0][0] = [(slip, 0, small, False), (1-slip, 1, 0., False)]
            P[0][1] = [(1-self.param*slip, 0, small, False), (self.param*slip, 1, 0., False)]
            P[1][0] = [(slip, 0, small, False), (1 - slip, 1, large1, False)]
            P[1][1] = [(1-self.param*slip, 0, small, False), (self.param*slip, 1, large2, False)]

        initial_configuration = [0.8, 0.2]
        self.model_vector = np.array(initial_configuration)

        self.P_sas = p_sas(self.P, self.nS, self.nA)
        self.P_sa = p_sa(self.P_sas, self.nS, self.nA)
        self.R_sas = r_sas(self.P, self.nS, self.nA)
        self.R = r_sa(self.R_sas, self.nS, self.nA)

        super(NChainEnv, self).__init__(self.nS, self.nA, self.P, self.isd)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        state, reward, done, _ = self._step(action)

        self.steps += 1
        if self.steps >= self.max_steps:
            done = True

        return self.state, reward, done, {}

    def reset(self):
        self.state = 0
        self.steps = 0
        return self.state

    def get_state(self):
        return self.s

    def set_initial_configuration(self, model):
        self.set_model(model)
        initial_configuration = [0.8, 0.2]
        self.model_vector = np.array(initial_configuration)

    def set_model(self, model):
        self.P = copy.deepcopy(model)

        self.P_sas = p_sas(self.P, self.nS, self.nA)
        self.P_sa = p_sa(self.P_sas, self.nS, self.nA)
        self.R_sas = r_sas(self.P, self.nS, self.nA)
        self.R = r_sa(self.R_sas, self.nS, self.nA)

    def get_valid_actions(self, state):
        return [0, 1]