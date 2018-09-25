import copy
import os
import sys
sys.path.append('../utils')
print(sys.path)


from algorithm.model_chooser import *
from algorithm.spmi import SPMI
from utils.uniform_policy import UniformPolicy

from algorithm.policy_chooser import *
from envs.Chain import NChainEnv
from utils.tabular import TabularModel
from utils import tabular_factory


mdp = NChainEnv()

simulation_name = ''

dir_path = "./data/chain_" + simulation_name

if not os.path.exists(dir_path):
    os.makedirs(dir_path)

uniform_policy = UniformPolicy(mdp)
original_model = copy.deepcopy(mdp.P)

initial_model = TabularModel(mdp.P, mdp.nS, mdp.nA)
initial_policy = TabularPolicy(uniform_policy.get_rep(), mdp.nS, mdp.nA)

pi1 = tabular_factory.policy_from_matrix(np.array([[1., 0., 0., 0.],[0., 0., 1., 0.]]))
pi2 = tabular_factory.policy_from_matrix(np.array([[0., 1., 0., 0.],[0., 0., 0., 1.]]))

policy_set = [pi1, pi2]
policy_chooser = SetPolicyChooser(policy_set, mdp.nS, mdp.nA)
#policy_chooser = GreedyPolicyChooser(mdp.nS, mdp.nA)


model_set = [TabularModel(mdp.P_slip1, mdp.nS, mdp.nA),
             TabularModel(mdp.P_slip0, mdp.nS, mdp.nA)]

model_chooser = SetModelChooser(model_set, mdp.nS, mdp.nA)

eps = 0.0
spmi = SPMI(mdp, eps, policy_chooser, model_chooser, max_iter=10000, persistent=True)

#-------------------------------------------------------------------------------
#SPMI
spmi.spmi(initial_policy, initial_model)

spmi.logger.save(dir_path, 'spmi.csv')

#-------------------------------------------------------------------------------
#SPMI-sup
mdp.set_initial_configuration(original_model)
spmi.spmi_sup(initial_policy, initial_model)

spmi.logger.save(dir_path, 'spmi_sup.csv')

#-------------------------------------------------------------------------------
#SPMI-alt
mdp.set_initial_configuration(original_model)
spmi.spmi_alt(initial_policy, initial_model)

spmi.logger.save(dir_path, 'spmi_alt.csv')

#-------------------------------------------------------------------------------
#SPI+SMI
mdp.set_initial_configuration(original_model)
spmi.spi_smi(initial_policy, initial_model)

spmi.logger.save(dir_path, 'spi_smi.csv')

#-------------------------------------------------------------------------------
#SMI+SPI
mdp.set_initial_configuration(original_model)
spmi.smi_spi(initial_policy, initial_model)

spmi.logger.save(dir_path, 'smi_spi.csv')
