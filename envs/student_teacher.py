import copy
import itertools

from utils.matrix_builders import *

from envs import discrete
from utils.tabular import TabularPolicy


class TeacherStudentEnv(discrete.DiscreteEnv):
    def __init__(self, n_literals=2, max_value=1, max_update=1, max_literals_in_examples=3, horizon=10):

        '''
        Constructor
        :param n_literals: number of literals considered in the proble
        :param max_value: literal values are integers ranging in [0, max_value]
        :param max_update: maximum sum of the absolute differences between
                           two consecutive states
        :param max_literals_in_examples: maximum number of literals in an example
        '''

        '''
        The ACTION is an assignment. Eg,
            [5, 4, 1, 0, 1]
        The STATE a triple whose first component is the set of the indexes
        of the literals involved in the sum, the second is the value of the sum
        and the third is an assignment.
        Eg,
            ({0, 1, 3}, 4, [5, 4, 1, 0, 1])
        meaning that L_0 + L_1 + L_3 = 4
        '''

        self.n_literals = n_literals
        self.max_value = max_value
        self.max_update = max_update
        self.max_literals_in_examples = max_literals_in_examples

        self.nA = self.n_literals ** (self.max_value + 1)

        self.gamma = 0.99
        self.horizon = horizon

        states_encoded = self._get_all_states()
        self.nS = len(states_encoded)
        self.encoded_to_index_dict = dict(zip(states_encoded, range(self.nS)))
        self.index_to_encoded_dict = dict(zip(range(self.nS), states_encoded))

        self.isd = np.ones(self.nS) / self.nS
        self.mu = self.isd

        self.P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        self._build_P()

        self.P_sas = p_sas(self.P, self.nS, self.nA)
        self.P_sa = p_sa(self.P_sas, self.nS, self.nA)
        self.R_sas = r_sas(self.P, self.nS, self.nA)
        self.R = r_sa(self.R_sas, self.nS, self.nA)


        super(TeacherStudentEnv, self).__init__(self.nS, self.nA, self.P, self.isd)

    def set_model(self, model):
        self.P = copy.deepcopy(model)

        self.P_sas = p_sas(self.P, self.nS, self.nA)
        self.P_sa = p_sa(self.P_sas, self.nS, self.nA)
        self.R_sas = r_sas(self.P, self.nS, self.nA)
        self.R = r_sa(self.R_sas, self.nS, self.nA)

    def get_valid_actions(self, s):
        return range(self.nA)

    def _encode_action(self, action):
        index = 0
        for l in action:
            index = index * (self.max_value + 1) + l

        return index

    def _decode_action(self, index):
        action = []
        for i in range(self.n_literals):
            action.append(index % (self.max_value + 1))
            index /= (self.max_value + 1)

        action.reverse()
        return action

    def _encode_state_no_index(self, state):
        index = 0
        for l in state[0]:
            index += 2 ** l
        index = index * self.n_literals * (self.max_value + 1) + state[1]
        for l in state[2]:
            index = index * (self.max_value + 1) + l
        return index

    def _encode_state(self, state):
        return self.encoded_to_index_dict[self._encode_action_no_index(state)]

    def _decode_state(self, index):
        index = self.index_to_encoded_dict[index]
        assignmnt = []
        for i in range(self.n_literals):
            assignmnt.append(index % (self.max_value + 1))
            index /= (self.max_value + 1)
        assignmnt.reverse()
        state = (set(), index % (self.n_literals * (self.max_value + 1)), assignmnt)
        index /= (self.n_literals * (self.max_value + 1))
        for l in range(self.n_literals-1, -1, -1):
            if index % 2 == 1:
                state[0].add(l)
            index /= 2

        return state

    def _allowed_action(self, a, s):
        a = self._decode_action(a)
        s = self._decode_state(s)
        diff = sum(map(lambda x, y: abs(x - y), a, s[2]))

        return  diff <= self.max_update

    def _consistent(self, a, s):
        a = self._decode_action(a)
        s = self._decode_state(s)

        return sum([a[i] for i in s[0]]) == s[1]

    def _get_all_assignments(self):

        all_assignments = []
        def generate_assignment(i, assign):
            if i == self.n_literals:
                all_assignments.append(list(assign))
            else:
                for l in range(self.max_value + 1):
                    assign[i] = l
                    generate_assignment(i+1, assign)

        generate_assignment(0, [None] * self.n_literals)
        return all_assignments

    def _get_all_states(self):
        all_assignments = self._get_all_assignments()
        print(all_assignments)
        action_indexes = []
        for p in range(2, self.max_literals_in_examples + 1):
            literals_combs = map(set, itertools.combinations(range(self.n_literals), p))
            for literals_comb in literals_combs:
                for value in range(0, p * self.max_value + 1):
                    for assign in all_assignments:
                        action = (literals_comb, value, assign)
                        action_indexes.append(self._encode_state_no_index(action))
        return action_indexes

    def _build_P(self):

        for s in range(self.nS):
            print("%s / %s" % (s, self.nS))
            for a in range(self.nA):
                l = self.P[s][a]
                sum_prob = 0.

                if self._consistent(a, s):
                    reward = 1.
                else:
                    reward = 0.

                for s1 in range(self.nS):

                    if self._decode_state(s1)[2] == self._decode_action(a):
                        sum_prob += 1.
                        l.append((1., s1, reward, False))
                    else:
                        l.append((0., s1, 0., False))

                for i in range(len(l)):
                    l[i] = (l[i][0] / sum_prob, l[i][1], l[i][2], l[i][3])

    # method to reset the MDP state to an initial one
    def reset(self):
        s = discrete.categorical_sample(self.isd, self.np_random)
        self.s = np.array([s]).ravel()
        return self.s

    # method to get the current MDP state
    def get_state(self):
        return self.s

    def get_uniform_policy(self):
        policy = {s : [] for s in range(self.nS)}
        for s in range(self.nS):
            li = policy[s]
            sum_prob = 0.
            for a in range(self.nA):
                if self._allowed_action(a, s):
                    prob = 1.
                else:
                    prob = 0.

                sum_prob += prob
                li.append(prob)
            for a in range(self.nA):
                li[a] /= sum_prob
        return TabularPolicy(policy, self.nS, self.nA)



