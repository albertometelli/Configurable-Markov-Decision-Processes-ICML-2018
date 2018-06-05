from tabular import *

def policy_from_matrix(policy_matrix, nS=None, nA=None):
    if nS is None:
        nS = policy_matrix.shape[0]
    if nA is None:
        nA = int(policy_matrix.shape[1] / nS)

    policy_rep = {s: [] for s in range(nS)}

    for s in range(nS):
        for a in range(nA):
            sa = s * nA + a
            policy_rep[s].append(policy_matrix[s, sa])

    return TabularPolicy(policy_rep, nS, nA)


def model_from_matrix(model_matrix, original_model, nS=None, nA=None):
    '''
    CANNOT CREATE NEW TRANSITIONS! JUST CHANGE PROBABILITIES
    '''
    if nS is None:
        nS = model_matrix.shape[1]
    if nA is None:
        nA = int(model_matrix.shape[0] / nS)

    model_rep = {s: {a: [] for a in range(nA)} for s in range(nS)}

    for s in range(nS):
        for a in range(nA):
            sa = s * nA + a
            li = original_model[s][a]
            prob_sum = 0.
            temp = []
            for elem in li:
                s1 = elem[1]
                prob = model_matrix[sa, s1]
                prob_sum += prob
                temp.append((prob, s1, elem[2], elem[3]))
            for elem in temp:
                model_rep[s][a].append((elem[0] / prob_sum, elem[1], elem[2], elem[3]))

    return TabularModel(model_rep, nS, nA)
