import numpy as np

# method to populate the P_sas
def p_sas(P, nS, nA):

    # instantiation of an SxAxS matrix to collect the probabilities
    P_sas = np.zeros(shape=(nS, nA, nS))

    # loop to fill the probability values
    for s in range(nS):
        for a in range(nA):
            list = P[s][a]
            for s1 in range(nS):
                prob_sum = 0
                prob_count = 0
                for elem in list:
                    if elem[1] == s1:
                        prob_sum = prob_sum + elem[0]
                        prob_count = prob_count + 1
                if prob_count != 0:
                    p = prob_sum
                    P_sas[s][a][s1] = p

    return P_sas

def r_sas(P, nS, nA):

    # instantiation of an SxAxS matrix to collect the probabilities
    R_sas = np.zeros(shape=(nS, nA, nS))

    # loop to fill the probability values
    for s in range(nS):
        for a in range(nA):
            list = P[s][a]
            for elem in list:
                R_sas[s][a][elem[1]] = elem[2]

    return R_sas

def r_sa(R_sas, nS, nA):

    # instantiation of an SxAxS matrix to collect the probabilities
    R_sa = np.zeros(shape=(nS * nA, nS))

    a = 0
    s = 0
    for sa in range(nS * nA):
        if a == nA:
            a = 0
            s = s + 1
        R_sa[sa] = R_sas[s][a]
        a = a + 1

    return R_sa

# method to populate the P_sa
def p_sa(P_sas, nS, nA):

    P_sa = np.zeros(shape=(nS * nA, nS))
    a = 0
    s = 0
    for sa in range(nS * nA):
        if a == nA:
            a = 0
            s = s + 1
        P_sa[sa] = P_sas[s][a]
        a = a + 1

    return P_sa