
from algorithm.model_chooser import *
from utils.evaluator import *
from utils.tabular import *
from algorithm.logger import Logger


from algorithm.policy_chooser import *
from utils.tabular_operations import policy_convex_combination, model_convex_combination, policy_equiv_check, model_equiv_check


class SPMI(object):

    def __init__(self,
                 conf_mdp,
                 eps,
                 policy_chooser=None,
                 model_chooser=None,
                 max_iter=10000,
                 delta_q=None,
                 persistent=True):
        '''
        This class allows to instantiate a Safe Policy Model Iterator object, i.e., an object exposing
        the methods to perform policy-model learning on a given Conf-MDP. This class implements the
        Safe Policy-Model Iteration along with its derivatives.

        :param conf_mdp: the Configurable Markov Decision Process object
        :param eps: threshold to be use to stop iterations
        :param policy_chooser: an object implementing the choice of the target policy
        :param model_chooser: an object implementing the choice of the target model
        :param max_iter: maximum number of iterations to be performed
        :param delta_q: the value of DeltaQ, if None 1/(1-gamma) is used
        :param persistent: whether to adopt the persistent target selection instead of the simple greedy
        '''

        # --------------------------------------
        # ----- ATTRIBUTES INITIALIZATIONS -----
        # --------------------------------------
        self.mdp = conf_mdp
        self.gamma = conf_mdp.gamma
        self.horizon = conf_mdp.horizon
        self.eps = eps
        self.iteration_horizon = max_iter
        self.persistent = persistent
        # default delta_q = (1-gamma^H)/(1-gamma)
        if delta_q is None:
            self.delta_q = (1. - self.gamma ** self.horizon) / (1 - self.gamma)
        else:
            self.delta_q = delta_q
        # default policy_chooser instantiation if none
        if policy_chooser is None:
            self.policy_chooser = GreedyPolicyChooser(conf_mdp.nS, conf_mdp.nA)
        else:
            self.policy_chooser = policy_chooser
        # default model_chooser instantiation if none
        if model_chooser is None:
            self.model_chooser = GreedyModelChooser(conf_mdp.nS, conf_mdp.nA)
        else:
            self.model_chooser = model_chooser

        # LOGGER INSTANTIATION
        self.logger = Logger(self.mdp, self.model_chooser, self.policy_chooser)

    # -------------------------------------
    # ----- ALGORITHMS IMPLEMENTATION -----
    # -------------------------------------

    # implementation of Safe Policy-Model Iteration
    def spmi(self, initial_policy, initial_model):

        # initializations
        gamma = self.gamma
        mu = self.mdp.mu
        nS, nA = self.mdp.nS, self.mdp.nA
        horizon = self.horizon
        eps = self.eps
        iteration_horizon = self.iteration_horizon
        # instantiation of the matrix-form reward
        reward = TabularReward(self.mdp.P, self.mdp.nS, self.mdp.nA)
        # reset of the logging attributes
        self.logger.reset()

        policy = initial_policy
        model = initial_model

        # choose a target policy
        Q = evaluator.compute_q_function(policy, model, reward, gamma, horizon=horizon)
        d_mu = evaluator.compute_discounted_s_distribution(mu, policy, model, gamma, horizon, nS, nA)
        p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)
        target_policy_old = target_policy

        # choose a target model
        U = evaluator.compute_u_function(policy, model, reward, gamma, horizon=horizon)
        delta_mu = evaluator.compute_discounted_sa_distribution(mu, policy, model, gamma, horizon, nS, nA, d_mu)
        m_er_adv, m_dist_sup, m_dist_mean, target_model = self.model_chooser.choose(model, delta_mu, U)
        target_model_old = target_model

        # convergence threshold
        convergence = eps / (1 - gamma)
        # POLICY-MODEL UPDATE LOOP
        # the policy and the model are continuously updated until the iteration_horizon is reached or
        # the relative advantages fall below the convergence threshold
        while (p_er_adv > convergence or m_er_adv > convergence) and self.logger.iteration < iteration_horizon:

            target_policies = [(target_policy, p_er_adv, p_dist_sup, p_dist_mean)]
            if self.persistent:
                if not policy_equiv_check(target_policy, target_policy_old) and not policy_equiv_check(policy, target_policy_old):
                    er_adv_old = evaluator.compute_policy_er_advantage(target_policy_old, policy, Q, d_mu)
                    dist_sup_old = policy_sup_tv_distance(target_policy_old, policy)
                    dist_mean_old = policy_mean_tv_distance(target_policy_old, policy, d_mu)
                    target_policies.append((target_policy_old, er_adv_old, dist_sup_old, dist_mean_old))

            target_models = [(target_model, m_er_adv, m_dist_sup, m_dist_mean)]
            if self.persistent:
                if not model_equiv_check(target_model, target_model_old) and not model_equiv_check(model, target_model_old):
                    er_adv_old = evaluator.compute_model_er_advantage(target_model_old, model, U, delta_mu)
                    dist_sup_old = policy_sup_tv_distance(target_model_old, model)
                    dist_mean_old = policy_mean_tv_distance(target_model_old, model, delta_mu)
                    target_models.append((target_model_old, er_adv_old, dist_sup_old, dist_mean_old))

            # initializations of the update variables
            bound_star = 0.
            alpha_star = 0.
            beta_star = 0.
            target_policy_star = target_policy
            target_model_star = target_model
            p_er_adv_star, p_dist_sup_star, p_dist_mean_star = None, None, None
            m_er_adv_star, m_dist_sup_star, m_dist_mean_star = None, None, None

            # loop to select the update yielding the maximum bound value
            for target_policy, p_er_adv, p_dist_sup, p_dist_mean in target_policies:
                for target_model, m_er_adv, m_dist_sup, m_dist_mean in target_models:

                    alpha0 = ((1 - gamma) * p_er_adv) / (self.delta_q * gamma *
                                p_dist_sup * p_dist_mean + 1e-24)
                    alpha1 = ((1 - gamma) * p_er_adv) / (self.delta_q * gamma *
                                p_dist_sup * p_dist_mean + 1e-24) - .5 * \
                                (m_dist_mean / (p_dist_mean + 1e-24) + m_dist_sup / (p_dist_sup + 1e-24))
                    beta0 = ((1 - gamma) * m_er_adv) / (self.delta_q * (gamma ** 2)
                                * m_dist_sup * m_dist_mean)
                    beta1 = ((1 - gamma) * m_er_adv) / (self.delta_q * (gamma ** 2)
                                * m_dist_sup * m_dist_mean) - .5 / gamma * \
                                (p_dist_mean / (m_dist_mean + 1e-24) + p_dist_sup / (m_dist_sup + 1e-24))

                    alpha0 = np.clip(alpha0, 0., 1.)
                    alpha1 = np.clip(alpha1, 0., 1.)
                    beta0 = np.clip(beta0, 0., 1.)
                    beta1 = np.clip(beta1, 0., 1.)

                    for alpha, beta in [(alpha0, 0.), (0., beta0), (alpha1, 1.), (1., beta1)]:

                        bound = alpha * p_er_adv + beta * m_er_adv - \
                                (gamma / (1 - gamma) * self.delta_q / 2) * \
                                ((alpha ** 2) * p_dist_sup * p_dist_mean +
                                 gamma * (beta ** 2) * m_dist_sup * m_dist_mean +
                                 alpha * beta * p_dist_sup * m_dist_mean + alpha * beta * p_dist_mean * m_dist_sup)

                        if bound > bound_star:
                            bound_star = bound
                            alpha_star = alpha
                            beta_star = beta
                            target_policy_star = target_policy
                            target_model_star = target_model
                            p_er_adv_star, p_dist_sup_star, p_dist_mean_star = p_er_adv, p_dist_sup, p_dist_mean
                            m_er_adv_star, m_dist_sup_star, m_dist_mean_star = m_er_adv, m_dist_sup, m_dist_mean

            # policy and model update
            if alpha_star > 0:
                policy = self.policy_combination(alpha_star, target_policy_star, policy)
            if beta_star > 0:
                model = self.model_combination(beta_star, target_model_star, model)

            # performance evaluation
            Q = evaluator.compute_q_function(policy, model, reward, gamma, horizon=horizon)

            d_stationary = evaluator.compute_stationary_distribution(policy, model, nS, nA)
            average_return = evaluator.compute_average_return(policy, model,reward, d_stationary)

            print(policy.get_matrix())
            print(model.get_matrix())
            print(d_stationary)

            J_p_m = evaluator.compute_performance(mu, reward, policy, model, gamma, horizon, nS, nA)

            self.logger.update(J_p_m, alpha_star, beta_star, p_er_adv_star, m_er_adv_star,
                                p_dist_sup_star, p_dist_mean_star, m_dist_sup_star, m_dist_mean_star,
                                target_policy_star, target_policy_old, target_model_star, target_model_old,
                                convergence, bound_star, average_return)

            # choose the next target policy
            target_policy_old = target_policy_star
            d_mu = evaluator.compute_discounted_s_distribution(mu, policy, model, gamma, horizon, nS, nA)
            p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)

            # choose the next target model
            target_model_old = target_model_star
            U = evaluator.compute_u_function(policy, model, reward, gamma, horizon=horizon)
            delta_mu = evaluator.compute_discounted_sa_distribution(mu, policy, model, gamma, horizon, nS, nA, d_mu)
            m_er_adv, m_dist_sup, m_dist_mean, target_model = self.model_chooser.choose(model, delta_mu, U)

        return policy, model


    # implementation of SPMI-sup,
    # version of SPMI adopting a looser lower bound on performance improvement
    # in which the mean distances are replaced by sup distances
    def spmi_sup(self, initial_policy, initial_model):

        # initializations
        gamma = self.gamma
        mu = self.mdp.mu
        nS, nA = self.mdp.nS, self.mdp.nA
        horizon = self.horizon
        eps = self.eps
        iteration_horizon = self.iteration_horizon
        # instantiation of the matrix-form reward
        reward = TabularReward(self.mdp.P, self.mdp.nS, self.mdp.nA)
        # reset of the logging attributes
        self.logger.reset()

        policy = initial_policy
        model = initial_model

        # choose a target policy
        Q = evaluator.compute_q_function(policy, model, reward, gamma, horizon=horizon)
        d_mu = evaluator.compute_discounted_s_distribution(mu, policy, model, gamma, horizon, nS, nA)
        p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)
        target_policy_old = target_policy

        # choose a target model
        U = evaluator.compute_u_function(policy, model, reward, gamma, horizon=horizon)
        delta_mu = evaluator.compute_discounted_sa_distribution(mu, policy, model, gamma, horizon, nS, nA, d_mu)
        m_er_adv, m_dist_sup, m_dist_mean, target_model = self.model_chooser.choose(model, delta_mu, U)
        target_model_old = target_model

        # convergence threshold
        convergence = eps / (1 - gamma)
        # POLICY-MODEL UPDATE LOOP
        # the policy and the model are continuously updated until the iteration_horizon is reached or
        # the relative advantages fall below the convergence threshold
        while ((p_er_adv + m_er_adv) > convergence) and self.logger.iteration < iteration_horizon:

            target_policies = [(target_policy, p_er_adv, p_dist_sup, p_dist_mean)]
            if self.persistent:
                if not policy_equiv_check(target_policy, target_policy_old) and not policy_equiv_check(policy, target_policy_old):
                    er_adv_old = evaluator.compute_policy_er_advantage(target_policy_old, policy, Q, d_mu)
                    dist_sup_old = policy_sup_tv_distance(target_policy_old, policy)
                    dist_mean_old = policy_mean_tv_distance(target_policy_old, policy, d_mu)
                    target_policies.append((target_policy_old, er_adv_old, dist_sup_old, dist_mean_old))

            target_models = [(target_model, m_er_adv, m_dist_sup, m_dist_mean)]
            if self.persistent:
                if not model_equiv_check(target_model, target_model_old) and not model_equiv_check(model, target_model_old):
                    er_adv_old = evaluator.compute_model_er_advantage(target_model_old, model, U, delta_mu)
                    dist_sup_old = policy_sup_tv_distance(target_model_old, model)
                    dist_mean_old = policy_mean_tv_distance(target_model_old, model, delta_mu)
                    target_models.append((target_model_old, er_adv_old, dist_sup_old, dist_mean_old))

            # initializations of the update variables
            bound_star = 0.
            alpha_star = 0.
            beta_star = 0.
            target_policy_star = target_policy
            target_model_star = target_model
            p_er_adv_star, p_dist_sup_star, p_dist_mean_star = None, None, None
            m_er_adv_star, m_dist_sup_star, m_dist_mean_star = None, None, None

            # loop to select the update yielding the maximum bound value
            for target_policy, p_er_adv, p_dist_sup, p_dist_mean in target_policies:
                for target_model, m_er_adv, m_dist_sup, m_dist_mean in target_models:

                    alpha0 = ((1 - gamma) * p_er_adv) / (self.delta_q * gamma *
                                p_dist_sup * p_dist_sup + 1e-24)
                    alpha1 = ((1 - gamma) * p_er_adv) / (self.delta_q * gamma *
                                p_dist_sup * p_dist_sup + 1e-24) - \
                                m_dist_sup / (p_dist_sup + 1e-24)
                    beta0 = ((1 - gamma) * m_er_adv) / (self.delta_q * (gamma ** 2)
                                * m_dist_sup * m_dist_sup)
                    beta1 = ((1 - gamma) * m_er_adv) / (self.delta_q * (gamma ** 2)
                                * m_dist_sup * m_dist_sup) - 1. / gamma * \
                                p_dist_sup / (m_dist_sup + 1e-24)

                    alpha0 = np.clip(alpha0, 0., 1.)
                    alpha1 = np.clip(alpha1, 0., 1.)
                    beta0 = np.clip(beta0, 0., 1.)
                    beta1 = np.clip(beta1, 0., 1.)

                    for alpha, beta in [(alpha0, 0.), (0., beta0), (alpha1, 1.), (1., beta1)]:

                        bound = alpha * p_er_adv + beta * m_er_adv - \
                                (gamma / (1 - gamma) * self.delta_q / 2) * \
                                ((alpha ** 2) * p_dist_sup ** 2 +
                                 gamma * (beta ** 2) * m_dist_sup ** 2 +
                                 2 * alpha * beta * p_dist_sup * m_dist_sup)

                        if bound > bound_star:
                            bound_star = bound
                            alpha_star = alpha
                            beta_star = beta
                            target_policy_star = target_policy
                            target_model_star = target_model
                            p_er_adv_star, p_dist_sup_star, p_dist_mean_star = p_er_adv, p_dist_sup, p_dist_mean
                            m_er_adv_star, m_dist_sup_star, m_dist_mean_star = m_er_adv, m_dist_sup, m_dist_mean

            # policy and model update
            if alpha_star > 0:
                policy = self.policy_combination(alpha_star, target_policy_star, policy)
            if beta_star > 0:
                model = self.model_combination(beta_star, target_model_star, model)

            # performance evaluation
            Q = evaluator.compute_q_function(policy, model, reward, gamma, horizon=horizon)
            J_p_m = evaluator.compute_performance(mu, reward, policy, model, gamma, horizon, nS, nA)

            d_stationary = evaluator.compute_stationary_distribution(policy, model, nS, nA)
            average_return = evaluator.compute_average_return(policy, model, reward, d_stationary)

            self.logger.update(J_p_m, alpha_star, beta_star, p_er_adv_star, m_er_adv_star,
                                p_dist_sup_star, p_dist_mean_star, m_dist_sup_star, m_dist_mean_star,
                                target_policy_star, target_policy_old, target_model_star, target_model_old,
                                convergence, bound_star, average_return)

            # choose the next target policy
            target_policy_old = target_policy_star
            d_mu = evaluator.compute_discounted_s_distribution(mu, policy, model, gamma, horizon, nS, nA)
            p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)

            # choose the next target model
            target_model_old = target_model_star
            U = evaluator.compute_u_function(policy, model, reward, gamma, horizon=horizon)
            delta_mu = evaluator.compute_discounted_sa_distribution(mu, policy, model, gamma, horizon, nS, nA, d_mu)
            m_er_adv, m_dist_sup, m_dist_mean, target_model = self.model_chooser.choose(model, delta_mu, U)

        return policy, model


    # implementation of SPMI-alt,
    # version of SPMI executing alternated improvement of the policy and the model
    # regardless of the value of the bound
    def spmi_alt(self, initial_policy, initial_model):

        # initializations
        gamma = self.gamma
        mu = self.mdp.mu
        nS, nA = self.mdp.nS, self.mdp.nA
        horizon = self.horizon
        eps = self.eps
        iteration_horizon = self.iteration_horizon
        # instantiation of the matrix-form reward
        reward = TabularReward(self.mdp.P, self.mdp.nS, self.mdp.nA)
        # reset of the logging attributes
        self.logger.reset()

        policy = initial_policy
        model = initial_model

        # choose a target policy
        Q = evaluator.compute_q_function(policy, model, reward, gamma, horizon=horizon)
        d_mu = evaluator.compute_discounted_s_distribution(mu, policy, model, gamma, horizon, nS, nA)
        p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)
        target_policy_old = target_policy

        # choose a target model
        U = evaluator.compute_u_function(policy, model, reward, gamma, horizon=horizon)
        delta_mu = evaluator.compute_discounted_sa_distribution(mu, policy, model, gamma, horizon, nS, nA, d_mu)
        m_er_adv, m_dist_sup, m_dist_mean, target_model = self.model_chooser.choose(
            model, delta_mu, U)
        target_model_old = target_model

        # convergence threshold
        convergence = eps / (1 - gamma)
        # POLICY-MODEL UPDATE LOOP
        # the policy and the model are continuously updated until the iteration_horizon is reached or
        # the relative advantages fall below the convergence threshold
        while (p_er_adv > convergence or m_er_adv > convergence) and self.logger.iteration < iteration_horizon:

            target_policies = [(target_policy, p_er_adv, p_dist_sup, p_dist_mean)]
            if self.persistent:
                if not policy_equiv_check(target_policy, target_policy_old) and not policy_equiv_check(policy, target_policy_old):
                    er_adv_old = evaluator.compute_policy_er_advantage(target_policy_old, policy, Q, d_mu)
                    dist_sup_old = policy_sup_tv_distance(target_policy_old, policy)
                    dist_mean_old = policy_mean_tv_distance(target_policy_old, policy, d_mu)
                    target_policies.append((target_policy_old, er_adv_old, dist_sup_old, dist_mean_old))

            target_models = [(target_model, m_er_adv, m_dist_sup, m_dist_mean)]
            if self.persistent:
                if not model_equiv_check(target_model, target_model_old) and not model_equiv_check(model, target_model_old):
                    er_adv_old = evaluator.compute_model_er_advantage(target_model_old, model, U, delta_mu)
                    dist_sup_old = policy_sup_tv_distance(target_model_old, model)
                    dist_mean_old = policy_mean_tv_distance(target_model_old, model, delta_mu)
                    target_models.append((target_model_old, er_adv_old, dist_sup_old, dist_mean_old))

            # initializations of the update variables
            bound_star = -1.
            alpha_star = 0.
            beta_star = 0.
            target_policy_star = target_policy
            target_model_star = target_model
            p_er_adv_star, p_dist_sup_star, p_dist_mean_star = None, None, None
            m_er_adv_star, m_dist_sup_star, m_dist_mean_star = None, None, None

            # loop selecting the update of the policy and the model alternately
            for target_policy, p_er_adv, p_dist_sup, p_dist_mean in target_policies:
                for target_model, m_er_adv, m_dist_sup, m_dist_mean in target_models:

                    alpha0 = ((1 - gamma) * p_er_adv) / (self.delta_q * gamma *
                                p_dist_sup * p_dist_mean + 1e-24)
                    beta0 = ((1 - gamma) * m_er_adv) / (self.delta_q * (gamma ** 2)
                                * m_dist_sup * m_dist_mean + 1e-24)

                    alpha0 = np.clip(alpha0, 0., 1.)
                    beta0 = np.clip(beta0, 0., 1.)

                    if self.logger.iteration % 2 == 0:
                        li = [(alpha0, 0.)]
                    else:
                        li = [(0., beta0)]

                    for alpha, beta in li:

                        bound = alpha * p_er_adv + beta * m_er_adv - \
                                (gamma / (1 - gamma) * self.delta_q / 2) * \
                                ((alpha ** 2) * p_dist_sup * p_dist_mean +
                                 gamma * (beta ** 2) * m_dist_sup * m_dist_mean +
                                 alpha * beta * p_dist_sup * m_dist_mean + alpha * beta * p_dist_mean * m_dist_sup)

                        if bound > bound_star:
                            bound_star = bound
                            alpha_star = alpha
                            beta_star = beta
                            target_policy_star = target_policy
                            target_model_star = target_model
                            p_er_adv_star, p_dist_sup_star, p_dist_mean_star = p_er_adv, p_dist_sup, p_dist_mean
                            m_er_adv_star, m_dist_sup_star, m_dist_mean_star = m_er_adv, m_dist_sup, m_dist_mean

            # policy and model update
            if alpha_star > 0:
                policy = self.policy_combination(alpha_star, target_policy_star, policy)
            if beta_star > 0:
                model = self.model_combination(beta_star, target_model_star, model)

            # performance evaluation
            Q = evaluator.compute_q_function(policy, model, reward, gamma, horizon=horizon)
            J_p_m = evaluator.compute_performance(mu, reward, policy, model, gamma, horizon, nS, nA)

            d_stationary = evaluator.compute_stationary_distribution(policy, model, nS, nA)
            average_return = evaluator.compute_average_return(policy, model, reward, d_stationary)

            self.logger.update(J_p_m, alpha_star, beta_star, p_er_adv_star, m_er_adv_star,
                                p_dist_sup_star, p_dist_mean_star, m_dist_sup_star, m_dist_mean_star,
                                target_policy_star, target_policy_old, target_model_star, target_model_old,
                                convergence, bound_star, average_return)

            # choose the next target policy
            target_policy_old = target_policy_star
            d_mu = evaluator.compute_discounted_s_distribution(mu, policy, model, gamma, horizon, nS, nA)
            p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)

            # choose the next target model
            target_model_old = target_model_star
            U = evaluator.compute_u_function(policy, model, reward, gamma, horizon=horizon)
            delta_mu = evaluator.compute_discounted_sa_distribution(mu, policy, model, gamma, horizon, nS, nA, d_mu)
            m_er_adv, m_dist_sup, m_dist_mean, target_model = self.model_chooser.choose(model, delta_mu, U)

        return policy, model


    # implementation of SPMI-nofull,
    # version of SPMI that prevents the selection of the "full step" updates,
    # i.e., the update (alfa_star,1) and (1,beta_star).
    # At each iteration, only one between the policy and the model is updated
    def spmi_no_full(self, initial_policy, initial_model):

        # initializations
        gamma = self.gamma
        mu = self.mdp.mu
        nS, nA = self.mdp.nS, self.mdp.nA
        horizon = self.horizon
        eps = self.eps
        iteration_horizon = self.iteration_horizon
        # instantiation of the matrix-form reward
        reward = TabularReward(self.mdp.P, self.mdp.nS, self.mdp.nA)
        # reset of the logging attributes
        self.logger.reset()

        policy = initial_policy
        model = initial_model

        # choose a target policy
        Q = evaluator.compute_q_function(policy, model, reward, gamma, horizon=horizon)
        d_mu = evaluator.compute_discounted_s_distribution(mu, policy, model, gamma, horizon, nS, nA)
        p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)
        target_policy_old = target_policy

        # choose a target model
        U = evaluator.compute_u_function(policy, model, reward, gamma, horizon=horizon)
        delta_mu = evaluator.compute_discounted_sa_distribution(mu, policy, model, gamma, horizon, nS, nA, d_mu)
        m_er_adv, m_dist_sup, m_dist_mean, target_model = self.model_chooser.choose(
            model, delta_mu, U)
        target_model_old = target_model

        # convergence threshold
        convergence = eps / (1 - gamma)
        # POLICY-MODEL UPDATE LOOP
        # the policy and the model are continuously updated until the iteration_horizon is reached or
        # the relative advantages fall below the convergence threshold
        while ((p_er_adv + m_er_adv) > convergence) and self.logger.iteration < iteration_horizon:

            target_policies = [(target_policy, p_er_adv, p_dist_sup, p_dist_mean)]
            if self.persistent:
                if not policy_equiv_check(target_policy, target_policy_old) and not policy_equiv_check(policy, target_policy_old):
                    er_adv_old = evaluator.compute_policy_er_advantage(target_policy_old, policy, Q, d_mu)
                    dist_sup_old = policy_sup_tv_distance(target_policy_old, policy)
                    dist_mean_old = policy_mean_tv_distance(target_policy_old, policy, d_mu)
                    target_policies.append((target_policy_old, er_adv_old, dist_sup_old, dist_mean_old))

            target_models = [(target_model, m_er_adv, m_dist_sup, m_dist_mean)]
            if self.persistent:
                if not model_equiv_check(target_model, target_model_old) and not model_equiv_check(model, target_model_old):
                    er_adv_old = evaluator.compute_model_er_advantage(target_model_old, model, U, delta_mu)
                    dist_sup_old = policy_sup_tv_distance(target_model_old, model)
                    dist_mean_old = policy_mean_tv_distance(target_model_old, model, delta_mu)
                    target_models.append((target_model_old, er_adv_old, dist_sup_old, dist_mean_old))

            # initializations of the update variables
            bound_star = 0.
            alpha_star = 0.
            beta_star = 0.
            target_policy_star = target_policy
            target_model_star = target_model
            p_er_adv_star, p_dist_sup_star, p_dist_mean_star = None, None, None
            m_er_adv_star, m_dist_sup_star, m_dist_mean_star = None, None, None

            # loop to select the update yielding the maximum bound value
            for target_policy, p_er_adv, p_dist_sup, p_dist_mean in target_policies:
                for target_model, m_er_adv, m_dist_sup, m_dist_mean in target_models:

                    alpha0 = ((1 - gamma) * p_er_adv) / (self.delta_q * gamma *
                                p_dist_sup * p_dist_mean + 1e-24)
                    beta0 = ((1 - gamma) * m_er_adv) / (self.delta_q * (gamma ** 2)
                                * m_dist_sup * m_dist_mean)

                    alpha0 = np.clip(alpha0, 0., 1.)
                    beta0 = np.clip(beta0, 0., 1.)

                    for alpha, beta in [(alpha0, 0.), (0., beta0)]:

                        bound = alpha * p_er_adv + beta * m_er_adv - \
                                (gamma / (1 - gamma) * self.delta_q / 2) * \
                                ((alpha ** 2) * p_dist_sup * p_dist_mean +
                                 gamma * (beta ** 2) * m_dist_sup * m_dist_mean +
                                 alpha * beta * p_dist_sup * m_dist_mean + alpha * beta * p_dist_mean * m_dist_sup)

                        if bound > bound_star:
                            bound_star = bound
                            alpha_star = alpha
                            beta_star = beta
                            target_policy_star = target_policy
                            target_model_star = target_model
                            p_er_adv_star, p_dist_sup_star, p_dist_mean_star = p_er_adv, p_dist_sup, p_dist_mean
                            m_er_adv_star, m_dist_sup_star, m_dist_mean_star = m_er_adv, m_dist_sup, m_dist_mean

            # policy and model update
            if alpha_star > 0:
                policy = self.policy_combination(alpha_star, target_policy_star, policy)
            if beta_star > 0:
                model = self.model_combination(beta_star, target_model_star, model)

            # performance evaluation
            Q = evaluator.compute_q_function(policy, model, reward, gamma, horizon=horizon)
            J_p_m = evaluator.compute_performance(mu, reward, policy, model, gamma, horizon, nS, nA)

            d_stationary = evaluator.compute_stationary_distribution(policy, model, nS, nA)
            average_return = evaluator.compute_average_return(policy, model, reward, d_stationary)

            self.logger.update(J_p_m, alpha_star, beta_star, p_er_adv_star, m_er_adv_star,
                                p_dist_sup_star, p_dist_mean_star, m_dist_sup_star, m_dist_mean_star,
                                target_policy_star, target_policy_old, target_model_star, target_model_old,
                                convergence, bound_star, average_return)

            # choose the next target policy
            target_policy_old = target_policy_star
            d_mu = evaluator.compute_discounted_s_distribution(mu, policy, model, gamma, horizon, nS, nA)
            p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)

            # choose the next target model
            target_model_old = target_model_star
            U = evaluator.compute_u_function(policy, model, reward, gamma, horizon=horizon)
            delta_mu = evaluator.compute_discounted_sa_distribution(mu, policy, model, gamma, horizon, nS, nA, d_mu)
            m_er_adv, m_dist_sup, m_dist_mean, target_model = self.model_chooser.choose(model, delta_mu, U)

        return policy, model


    # implementation of SPI+SMI,
    # performing a complete execution of Safe Policy Iteration (SPI)
    # followed by a complete execution of Safe Model Iteration (SMI)
    def spi_smi(self, initial_policy, initial_model):

        # initializations
        gamma = self.gamma
        mu = self.mdp.mu
        nS, nA = self.mdp.nS, self.mdp.nA
        horizon = self.horizon
        eps = self.eps
        iteration_horizon = self.iteration_horizon
        # instantiation of the matrix-form reward
        reward = TabularReward(self.mdp.P, self.mdp.nS, self.mdp.nA)
        # reset of the logging attributes
        self.logger.reset()

        policy = initial_policy
        model = initial_model

        # ------ SPI ------

        # choose a target policy
        Q = evaluator.compute_q_function(policy, model, reward, gamma, horizon=horizon)
        d_mu = evaluator.compute_discounted_s_distribution(mu, policy, model, gamma, horizon, nS, nA)
        p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)
        target_policy_old = target_policy

        # convergence threshold
        convergence = eps / (1 - gamma)
        # POLICY UPDATE LOOP
        # the policy is continuously updated until the iteration_horizon is reached or
        # the relative advantage falls below the convergence threshold
        while p_er_adv > convergence and self.logger.iteration < iteration_horizon:

            target_policies = [(target_policy, p_er_adv, p_dist_sup, p_dist_mean)]
            if self.persistent:
                if not policy_equiv_check(target_policy, target_policy_old) and not policy_equiv_check(policy, target_policy_old):
                    er_adv_old = evaluator.compute_policy_er_advantage(target_policy_old, policy, Q, d_mu)
                    dist_sup_old = policy_sup_tv_distance(target_policy_old, policy)
                    dist_mean_old = policy_mean_tv_distance(target_policy_old, policy, d_mu)
                    target_policies.append((target_policy_old, er_adv_old, dist_sup_old, dist_mean_old))

            # initializations of the update variables
            bound_star = 0.
            alpha_star = 0.
            target_policy_star = target_policy
            p_er_adv_star, p_dist_sup_star, p_dist_mean_star = None, None, None

            # selection of the update yielding the maximum bound value
            for target_policy, p_er_adv, p_dist_sup, p_dist_mean in target_policies:

                alpha = ((1 - gamma) * p_er_adv) / (self.delta_q * gamma *
                            p_dist_sup * p_dist_mean + 1e-24)

                alpha = np.clip(alpha, 0., 1.)

                bound = alpha * p_er_adv - \
                        (gamma / (1 - gamma) * self.delta_q / 2) * \
                        ((alpha ** 2) * p_dist_sup * p_dist_mean)

                if bound > bound_star:
                    bound_star = bound
                    alpha_star = alpha
                    target_policy_star = target_policy
                    p_er_adv_star, p_dist_sup_star, p_dist_mean_star = p_er_adv, p_dist_sup, p_dist_mean

            # policy update
            if alpha_star > 0:
                policy = self.policy_combination(alpha_star, target_policy_star, policy)

            # performance evaluation
            Q = evaluator.compute_q_function(policy, model, reward, gamma, horizon=horizon)
            J_p_m = evaluator.compute_performance(mu, reward, policy, model, gamma, horizon, nS, nA)

            d_stationary = evaluator.compute_stationary_distribution(policy, model, nS, nA)
            average_return = evaluator.compute_average_return(policy, model, reward, d_stationary)

            self.logger.update(J_p_m, alpha_star, np.nan, p_er_adv_star, np.nan,
                       p_dist_sup_star, p_dist_mean_star, np.nan, np.nan,
                       target_policy_star, target_policy_old, np.nan,
                       np.nan, convergence, bound_star, average_return)

            # choose the next target policy
            target_policy_old = target_policy_star
            d_mu = evaluator.compute_discounted_s_distribution(mu, policy, model, gamma, horizon, nS, nA)
            p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)

        # ------ SMI ------

        # choose a target model
        U = evaluator.compute_u_function(policy, model, reward, gamma, horizon=horizon)
        delta_mu = evaluator.compute_discounted_sa_distribution(mu, policy, model, gamma, horizon, nS, nA)
        m_er_adv, m_dist_sup, m_dist_mean, target_model = self.model_chooser.choose(model, delta_mu, U)
        target_model_old = target_model

        # convergence threshold
        convergence = eps / (1 - gamma)
        # MODEL UPDATE LOOP
        # the model is continuously updated until the iteration_horizon is reached or
        # the relative advantage falls below the convergence threshold
        while m_er_adv > convergence and self.logger.iteration < iteration_horizon:

            target_models = [(target_model, m_er_adv, m_dist_sup, m_dist_mean)]
            if self.persistent:
                if not model_equiv_check(target_model, target_model_old) and not model_equiv_check(model, target_model_old):
                    er_adv_old = evaluator.compute_model_er_advantage(target_model_old, model, U, delta_mu)
                    dist_sup_old = policy_sup_tv_distance(target_model_old, model)
                    dist_mean_old = policy_mean_tv_distance(target_model_old, model, delta_mu)
                    target_models.append((target_model_old, er_adv_old, dist_sup_old, dist_mean_old))

            # initializations of the update variables
            bound_star = 0.
            beta_star = 0.
            target_model_star = target_model
            m_er_adv_star, m_dist_sup_star, m_dist_mean_star = None, None, None

            # selection of the update yielding the maximum bound value
            for target_model, m_er_adv, m_dist_sup, m_dist_mean in target_models:

                beta = ((1 - gamma) * m_er_adv) / (self.delta_q * (gamma ** 2)
                            * m_dist_sup * m_dist_mean)

                beta = np.clip(beta, 0., 1.)

                bound = beta * m_er_adv - \
                        (gamma / (1 - gamma) * self.delta_q / 2) * \
                         gamma * (beta ** 2) * m_dist_sup * m_dist_mean

                if bound > bound_star:
                    bound_star = bound
                    beta_star = beta
                    target_model_star = target_model
                    m_er_adv_star, m_dist_sup_star, m_dist_mean_star = m_er_adv, m_dist_sup, m_dist_mean

            # model update
            if beta_star > 0:
                model = self.model_combination(beta_star, target_model_star, model)

            # performance evaluation
            J_p_m = evaluator.compute_performance(mu, reward, policy, model, gamma, horizon, nS, nA)

            d_stationary = evaluator.compute_stationary_distribution(policy, model, nS, nA)
            average_return = evaluator.compute_average_return(policy, model, reward, d_stationary)

            self.logger.update(J_p_m, np.nan, beta_star, np.nan, m_er_adv_star,
                       np.nan, np.nan, m_dist_sup_star, m_dist_mean_star,
                       np.nan, np.nan, target_model_star,
                       target_model_old, convergence, bound_star, average_return)

            # choose the next target model
            target_model_old = target_model_star
            U = evaluator.compute_u_function(policy, model, reward, gamma, horizon=horizon)
            delta_mu = evaluator.compute_discounted_sa_distribution(mu, policy, model, gamma, horizon, nS, nA)
            m_er_adv, m_dist_sup, m_dist_mean, target_model = self.model_chooser.choose(model, delta_mu, U)

        return policy, model

    # implementation of SMI+SPI,
    # performing a complete execution of Safe Model Iteration (SMI)
    # followed by a complete execution of Safe Policy Iteration (SPI)
    def smi_spi(self, initial_policy, initial_model):

        # initializations
        gamma = self.gamma
        mu = self.mdp.mu
        nS, nA = self.mdp.nS, self.mdp.nA
        horizon = self.horizon
        eps = self.eps
        iteration_horizon = self.iteration_horizon
        # instantiation of the matrix-form reward
        reward = TabularReward(self.mdp.P, self.mdp.nS, self.mdp.nA)
        # reset of the logging attributes
        self.logger.reset()

        policy = initial_policy
        model = initial_model

        # ------ SMI ------

        # choose a target model
        U = evaluator.compute_u_function(policy, model, reward, gamma, horizon=horizon)
        delta_mu = evaluator.compute_discounted_sa_distribution(mu, policy, model, gamma, horizon, nS, nA)
        m_er_adv, m_dist_sup, m_dist_mean, target_model = self.model_chooser.choose(model, delta_mu, U)
        target_model_old = target_model

        # convergence threshold
        convergence = eps / (1 - gamma)
        # MODEL UPDATE LOOP
        # the model is continuously updated until the iteration_horizon is reached or
        # the relative advantage falls below the convergence threshold
        while m_er_adv > convergence and self.logger.iteration < iteration_horizon:

            target_models = [(target_model, m_er_adv, m_dist_sup, m_dist_mean)]
            if self.persistent:
                if not model_equiv_check(target_model, target_model_old) and not model_equiv_check(model,
                                                                                                             target_model_old):
                    er_adv_old = evaluator.compute_model_er_advantage(target_model_old, model, U, delta_mu)
                    dist_sup_old = policy_sup_tv_distance(target_model_old, model)
                    dist_mean_old = policy_mean_tv_distance(target_model_old, model, delta_mu)
                    target_models.append((target_model_old, er_adv_old, dist_sup_old, dist_mean_old))

            # initializations of the update variables
            bound_star = 0.
            beta_star = 0.
            target_model_star = target_model
            m_er_adv_star, m_dist_sup_star, m_dist_mean_star = None, None, None

            # selection of the update yielding the maximum bound value
            for target_model, m_er_adv, m_dist_sup, m_dist_mean in target_models:

                beta = ((1 - gamma) * m_er_adv) / (self.delta_q * (gamma ** 2)
                                                   * m_dist_sup * m_dist_mean)

                beta = np.clip(beta, 0., 1.)

                bound = beta * m_er_adv - \
                        (gamma / (1 - gamma) * self.delta_q / 2) * \
                        gamma * (beta ** 2) * m_dist_sup * m_dist_mean

                if bound > bound_star:
                    bound_star = bound
                    beta_star = beta
                    target_model_star = target_model
                    m_er_adv_star, m_dist_sup_star, m_dist_mean_star = m_er_adv, m_dist_sup, m_dist_mean

            # model update
            if beta_star > 0:
                model = self.model_combination(beta_star, target_model_star, model)

            # performance evaluation
            J_p_m = evaluator.compute_performance(mu, reward, policy, model, gamma, horizon, nS, nA)

            d_stationary = evaluator.compute_stationary_distribution(policy, model, nS, nA)
            average_return = evaluator.compute_average_return(policy, model, reward, d_stationary)

            self.logger.update(J_p_m, np.nan, beta_star, np.nan, m_er_adv_star,
                       np.nan, np.nan, m_dist_sup_star, m_dist_mean_star,
                       np.nan, np.nan, target_model_star,
                       target_model_old, convergence, bound_star, average_return)

            # choose the next target model
            target_model_old = target_model_star
            U = evaluator.compute_u_function(policy, model, reward, gamma, horizon=horizon)
            delta_mu = evaluator.compute_discounted_sa_distribution(mu, policy, model, gamma, horizon, nS, nA)
            m_er_adv, m_dist_sup, m_dist_mean, target_model = self.model_chooser.choose(model, delta_mu, U)

        # ------ SPI ------

        # choose a target policy
        Q = evaluator.compute_q_function(policy, model, reward, gamma, horizon=horizon)
        d_mu = evaluator.compute_discounted_s_distribution(mu, policy, model, gamma, horizon, nS, nA)
        p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)
        target_policy_old = target_policy

        # convergence threshold
        convergence = eps / (1 - gamma)
        # POLICY UPDATE LOOP
        # the policy is continuously updated until the iteration_horizon is reached or
        # the relative advantage falls below the convergence threshold
        while p_er_adv > convergence and self.logger.iteration < iteration_horizon:

            target_policies = [(target_policy, p_er_adv, p_dist_sup, p_dist_mean)]
            if self.persistent:
                if not policy_equiv_check(target_policy, target_policy_old) and not policy_equiv_check(policy,
                                                                                                            target_policy_old):
                    er_adv_old = evaluator.compute_policy_er_advantage(target_policy_old, policy, Q, d_mu)
                    dist_sup_old = policy_sup_tv_distance(target_policy_old, policy)
                    dist_mean_old = policy_mean_tv_distance(target_policy_old, policy, d_mu)
                    target_policies.append((target_policy_old, er_adv_old, dist_sup_old, dist_mean_old))

            # initializations of the update variables
            bound_star = 0.
            alpha_star = 0.
            target_policy_star = target_policy
            p_er_adv_star, p_dist_sup_star, p_dist_mean_star = None, None, None

            # selection of the update yielding the maximum bound value
            for target_policy, p_er_adv, p_dist_sup, p_dist_mean in target_policies:

                alpha = ((1 - gamma) * p_er_adv) / (self.delta_q * gamma *
                                                    p_dist_sup * p_dist_mean + 1e-24)

                alpha = np.clip(alpha, 0., 1.)

                bound = alpha * p_er_adv - \
                        (gamma / (1 - gamma) * self.delta_q / 2) * \
                        ((alpha ** 2) * p_dist_sup * p_dist_mean)

                if bound > bound_star:
                    bound_star = bound
                    alpha_star = alpha
                    target_policy_star = target_policy
                    p_er_adv_star, p_dist_sup_star, p_dist_mean_star = p_er_adv, p_dist_sup, p_dist_mean

            # policy update
            if alpha_star > 0:
                policy = self.policy_combination(alpha_star, target_policy_star, policy)

            # performance evaluation
            Q = evaluator.compute_q_function(policy, model, reward, gamma, horizon=horizon)
            J_p_m = evaluator.compute_performance(mu, reward, policy, model, gamma, horizon, nS, nA)

            d_stationary = evaluator.compute_stationary_distribution(policy, model, nS, nA)
            average_return = evaluator.compute_average_return(policy, model, reward, d_stationary)

            self.logger.update(J_p_m, alpha_star, np.nan, p_er_adv_star, np.nan,
                       p_dist_sup_star, p_dist_mean_star, np.nan, np.nan,
                       target_policy_star, target_policy_old, np.nan,
                       np.nan, convergence, bound_star, average_return)

            # choose the next target policy
            target_policy_old = target_policy_star
            d_mu = evaluator.compute_discounted_s_distribution(mu, policy, model, gamma, horizon, nS, nA)
            p_er_adv, p_dist_sup, p_dist_mean, target_policy = self.policy_chooser.choose(policy, d_mu, Q)

        return policy, model

    # ---------------------------
    # ----- SUPPORT METHODS -----
    # ---------------------------

    # method to linearly combine target and current policy with coefficient alfa
    def policy_combination(self, alfa, target, current):

        new_policy = policy_convex_combination(target, current, alfa)

        return new_policy

    # method to linearly combine target and current model with coefficient beta
    # along with the update of the model coefficients in the mdp representation
    def model_combination(self, beta, target, current):

        new_model = model_convex_combination(self.mdp.P, target, current, beta)
        self.mdp.set_model(new_model.get_rep())

        return new_model
