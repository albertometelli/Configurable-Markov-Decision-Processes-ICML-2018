import numpy as np
import numpy.linalg as la

def compute_q_function(policy, model, reward, gamma, nS=None, nA=None, horizon=None):
    R_sas = reward.get_matrix()
    P = model.get_matrix()
    pi = policy.get_matrix()

    if nS is None:
        nS = pi.shape[0]
    if nA is None:
        nA = int(pi.shape[1] / pi.shape[0])
    nSA = nS * nA


    R_sa = (R_sas * P).sum(axis=1)
    if horizon is None:
        Q = la.solve(np.eye(nSA) - gamma * np.dot(P, pi), R_sa)
    else:
        Q = np.zeros(nS * nA)
        P_pi = np.dot(P, pi)
        for h in range(horizon):
            Q = R_sa + gamma * np.dot(P_pi, Q)
    return Q

def compute_v_function(policy, model, reward, gamma, nS=None, nA=None, horizon=None):

    R_sas = reward.get_matrix()
    P = model.get_matrix()
    pi = policy.get_matrix()

    if nS is None:
        nS = pi.shape[0]

    R_sa = (R_sas * P).sum(axis=1)
    R_s = np.dot(pi, R_sa)
    if horizon is None:
        V = la.solve(np.eye(nS) - gamma * np.dot(pi, P), R_s)
    else:
        V = np.zeros(nS)
        pi_P = np.dot(pi, P)
        for h in range(horizon):
            V = R_s + gamma * np.dot(pi_P, V)
    return V

def compute_u_function(policy, model, reward, gamma, nS=None, nA=None, horizon=None):
    pi = policy.get_matrix()
    if nS is None:
        nS = pi.shape[0]
    if nA is None:
        nA = int(pi.shape[1] / pi.shape[0])

    R_sas = reward.get_matrix()
    V = compute_v_function(policy, model, reward, gamma, nS, nA, horizon-1)
    U = R_sas + gamma * np.repeat([V], nS * nA, axis=0)
    return U

def compute_performance(mu, reward, policy, model, gamma, horizon=None, nS=None, nA=None):
    V = compute_v_function(policy, model, reward, gamma, nS, nA, horizon)
    return np.dot(mu, V)

# method to exactly compute the d_mu_P
def compute_discounted_s_distribution(mu, policy, model, gamma, horizon=None, nS=None, nA=None):
    pi = policy.get_matrix()
    P = model.get_matrix()
    if nS is None:
        nS = pi.shape[0]
    if nA is None:
        nA = int(pi.shape[1] / pi.shape[0])

    if horizon is None:
        d_mu = la.solve(np.eye(nS) - gamma * np.dot(pi, P), mu)
        d_mu = d_mu * (1 - gamma)
    else:
        d_mu = np.zeros(nS)
        pi_P = np.dot(pi, P)
        for h in range(horizon):
            d_mu = mu + gamma * np.dot(d_mu, pi_P)
        d_mu = d_mu * (1 - gamma) / (1 - gamma ** horizon)

    return d_mu

def compute_discounted_sa_distribution(mu, policy, model, gamma, horizon=None, nS=None, nA=None, d_mu=None):

    pi = policy.get_matrix()

    # d_mu computation
    if d_mu is None:  # for efficiency reasons you can avoid recompunting d_mu
        d_mu = compute_discounted_s_distribution(mu, policy, model, gamma, horizon, nS, nA)

    delta_mu = np.dot(d_mu, pi)
    return delta_mu

# method to exactly compute the expected relative advantage
def compute_policy_er_advantage(target, policy, Q, d_mu_pi):

    target_matrix = target.get_matrix()
    policy_matrix = policy.get_matrix()

    A = np.dot(target_matrix - policy_matrix, Q)
    er_advantage = np.dot(d_mu_pi, A)

    return er_advantage


# method to exactly compute the expected relative advantage
def compute_model_er_advantage(target, model, U, delta_mu_pi):
    target_matrix = target.get_matrix()
    model_matrix = model.get_matrix()

    # relative advantage as array of states
    A = np.sum((target_matrix - model_matrix) * U, axis=1)

    # computation of the expected relative advantage
    er_advantage = np.dot(delta_mu_pi, A)

    return er_advantage
