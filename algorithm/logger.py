
from utils.tabular import *
from algorithm.model_chooser import *
from algorithm.policy_chooser import *
from utils.tabular_operations import policy_equiv_check, model_equiv_check


class Logger(object):

    def __init__(self, mdp, model_chooser, policy_chooser=None):

        self.mdp = mdp
        self.model_chooser = model_chooser
        self.policy_chooser = policy_chooser

        # LOGGING ATTRIBUTES
        self.count = 0
        self.iteration = 0
        self.iterations = list()
        self.evaluations = list()
        self.p_advantages = list()
        self.m_advantages = list()
        self.p_dist_sup = list()
        self.p_dist_mean = list()
        self.m_dist_sup = list()
        self.m_dist_mean = list()
        self.alfas = list()
        self.betas = list()
        self.w_target = list()
        self.w_current = list()
        self.theta_target = list()
        self.theta_current = list()
        self.p_change = list()
        self.m_change = list()
        self.bound = list()
        self.average_rewards = list()
        self.policy_vector = None

    # method to collect the execution data and to print the log trace,
    # it also updates the model_vector in the mdp representation for parametric model spaces
    def update(self, J_p_m, alfa_star, beta_star, p_er_adv, m_er_adv,
                       p_dist_sup, p_dist_mean, m_dist_sup, m_dist_mean,
                       target_policy, target_policy_old, target_model,
                       target_model_old, convergence, bound, average_reward=None):
        # data collections
        self.iterations.append(self.iteration)
        self.evaluations.append(J_p_m)
        self.alfas.append(alfa_star)
        self.betas.append(beta_star)
        self.p_advantages.append(p_er_adv)
        self.m_advantages.append(m_er_adv)
        self.p_dist_sup.append(p_dist_sup)
        self.p_dist_mean.append(p_dist_mean)
        self.m_dist_sup.append(m_dist_sup)
        self.m_dist_mean.append(m_dist_mean)
        self.bound.append(bound)
        if average_reward is not None:
            self.average_rewards.append(average_reward)

        # target policy change check
        if isinstance(target_policy, TabularPolicy):
            p_check_target = policy_equiv_check(target_policy, target_policy_old)
        else:
            p_check_target = np.nan
        self.p_change.append(p_check_target)
        # target model change check
        if isinstance(target_model, TabularModel):
            m_check_target = model_equiv_check(target_model, target_model_old)
        else:
            m_check_target = np.nan
        self.m_change.append(m_check_target)

        # trace print
        print('----------------------')
        if average_reward is not None:
            print('average reward: {0}'.format(average_reward))
        print('performance: {0}'.format(J_p_m))
        print('alfa/beta: {0}/{1}'.format(alfa_star, beta_star))
        print('bound: {0}'.format(bound))
        print('iteration: {0}'.format(self.iteration))
        print('condition: {0}\n'.format(convergence))

        print('policy advantage: {0}'.format(p_er_adv))
        print('alfa star: {0}'.format(alfa_star))
        print('policy dist sup: {0}'.format(p_dist_sup))
        print('policy dist mean: {0}'.format(p_dist_mean))

        print('model advantage: {0}'.format(m_er_adv))
        print('beta star: {0}'.format(beta_star))
        print('model dist sup: {0}'.format(m_dist_sup))
        print('model dist mean: {0}'.format(m_dist_mean))

        # model vector coefficients computation and print
        if isinstance(self.model_chooser, SetModelChooser):
            model_vector = self.mdp.model_vector
            model_set = self.model_chooser.model_set
            n_models = len(model_vector)
            if isinstance(target_model, TabularModel):
                for i in range(n_models):
                    if model_equiv_check(model_set[i], target_model):
                        target_index = i
                        break
                target_vector = np.zeros(n_models)
                target_vector[target_index] = 1
                new_model_vector = beta_star * target_vector + (1 - beta_star) * model_vector
                self.mdp.model_vector = new_model_vector

                print('\ntarget_model: {0}'.format(target_vector))
                print('current_model: {0}'.format(new_model_vector))
                self.w_current.append(new_model_vector)
                self.w_target.append(target_vector)
            else:
                self.w_current.append(model_vector)
                target_vector = np.empty(n_models)
                target_vector[:] = np.nan
                self.w_target.append(target_vector)

        # policy vector coefficients computation and print
        if isinstance(self.policy_chooser, SetPolicyChooser) and len(self.policy_chooser.policy_set) == 2:
            policy_vector = self.policy_vector
            policy_set = self.policy_chooser.policy_set
            n_policies = len(policy_vector)
            if isinstance(target_policy, TabularPolicy):
                for i in range(n_policies):
                    if policy_equiv_check(policy_set[i], target_policy):
                        target_index = i
                        break
                target_vector = np.zeros(n_policies)
                target_vector[target_index] = 1
                new_policy_vector = alfa_star * target_vector + (1 - alfa_star) * policy_vector
                self.policy_vector = new_policy_vector

                print('\ntarget_policy: {0}'.format(target_vector))
                print('current_policy: {0}'.format(new_policy_vector))
                self.theta_current.append(new_policy_vector)
                self.theta_target.append(target_vector)
            else:
                self.theta_current.append(policy_vector)
                target_vector = np.empty(n_policies)
                target_vector[:] = np.nan
                self.theta_target.append(target_vector)

        # iteration update
        self.iteration = self.iteration + 1

    # logger method to reset all the logging attributes
    def reset(self):
        self.count = 0
        self.iteration = 0
        self.iterations = list()
        self.evaluations = list()
        self.p_advantages = list()
        self.m_advantages = list()
        self.p_dist_sup = list()
        self.p_dist_mean = list()
        self.m_dist_sup = list()
        self.m_dist_mean = list()
        self.alfas = list()
        self.betas = list()
        self.p_change = list()
        self.m_change = list()
        self.w_target = list()
        self.w_current = list()
        self.theta_target = list()
        self.theta_current = list()
        self.bound = list()
        self.average_rewards = list()

    # logger method to save the execution data into
    # a csv file (directory path as parameter)
    def save(self, dir_path, file_name, entries=None):
        header_string = 'iterations;evaluations;p_advantages;m_advantages;' \
                        'p_dist_sup;p_dist_mean;m_dist_sup;m_dist_mean;alfa;beta;p_change;m_change;bound'

        # if coefficients not empty we are using a parametric model
        execution_data = [self.iterations, self.evaluations,
                          self.p_advantages, self.m_advantages,
                          self.p_dist_sup, self.p_dist_mean,
                          self.m_dist_sup, self.m_dist_mean,
                          self.alfas, self.betas, self.p_change,
                          self.m_change, self.bound]

        if len(self.average_rewards) > 0:
            header_string = header_string + ';average_reward'
            execution_data.append(self.average_rewards)

        if isinstance(self.model_chooser, SetModelChooser):

            if len(self.model_chooser.model_set) == 2:
                header_string = header_string + ';w_current[0];w_current[1];w_target[0];w_target[1]'

                current = np.array(self.w_current)
                target = np.array(self.w_target)

                execution_data += [
                                  current[:, 0], current[:, 1],
                                  target[:, 0], target[:, 1]]

            if len(self.model_chooser.model_set) == 4:
                header_string = header_string + ';w_current[0];w_current[1];w_current[2];w_current[3]' \
                                                ';w_target[0];w_target[1];w_target[2];w_target[3]'

                current = np.array(self.w_current)
                target = np.array(self.w_target)

                execution_data += [
                                  current[:, 0], current[:, 1],
                                  current[:, 2], current[:, 3],
                                  target[:, 0], target[:, 1],
                                  target[:, 2], target[:, 3]]

        if self.policy_chooser is not None and isinstance(self.policy_chooser, SetPolicyChooser):

            if len(self.policy_chooser.policy_set) == 2:
                header_string = header_string + ';theta_current[0];theta_current[1];theta_target[0];theta_target[1]'

                current = np.array(self.theta_current)
                target = np.array(self.theta_target)

                execution_data += [
                                  current[:, 0], current[:, 1],
                                  target[:, 0], target[:, 1]]

        print(execution_data)
        execution_data = np.array(execution_data).T

        if entries is not None:
            filter = np.arange(0, len(execution_data), len(execution_data) / entries)
            execution_data = execution_data[filter]

        np.savetxt(dir_path + '/' + file_name, execution_data,
                   delimiter=';', header=header_string, fmt='%.30e')