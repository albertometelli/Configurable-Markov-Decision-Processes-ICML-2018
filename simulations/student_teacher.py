import copy
import os

from algorithm.model_chooser import *
from algorithm.spmi import SPMI
from utils.uniform_policy import UniformPolicy

from algorithm.policy_chooser import *
from envs.student_teacher import TeacherStudentEnv
from utils.tabular import TabularModel


mdp = TeacherStudentEnv(n_literals=2,
                    max_value=1,
                    max_update=1,
                    max_literals_in_examples=2,
                    horizon=10)

simulation_name = '%s-%s-%s-%s' % (mdp.n_literals,
                              mdp.max_value,
                              mdp.max_update,
                              mdp.max_literals_in_examples)

dir_path = "./data/student_teacher_" + simulation_name

if not os.path.exists(dir_path):
    os.makedirs(dir_path)

uniform_policy = UniformPolicy(mdp)
original_model = copy.deepcopy(mdp.P)

initial_model = TabularModel(mdp.P, mdp.nS, mdp.nA)
initial_policy = TabularPolicy(uniform_policy.get_rep(), mdp.nS, mdp.nA)

policy_chooser = GreedyPolicyChooser(mdp.nS, mdp.nA)
model_chooser = DoNotCreateTransitionsGreedyModelChooser(mdp.P, mdp.nS, mdp.nA)

eps = 0.0
spmi = SPMI(mdp, eps, policy_chooser, model_chooser, max_iter=30000, persistent=True)

#-------------------------------------------------------------------------------
#SPMI
spmi.spmi(initial_policy, initial_model)

spmi.logger.save(dir_path, 'spmi.csv')

#-------------------------------------------------------------------------------
#SPMI-sup
mdp.set_model(original_model)
spmi.spmi_sup(initial_policy, initial_model)

spmi.logger.save(dir_path, 'spmi_sup.csv')

#-------------------------------------------------------------------------------
#SPMI-alt
mdp.set_model(original_model)
spmi.spmi_alt(initial_policy, initial_model)

spmi.logger.save(dir_path, 'spmi_alt.csv')

#-------------------------------------------------------------------------------
#SPI+SMI
mdp.set_model(original_model)
spmi.spi_smi(initial_policy, initial_model)

spmi.logger.save(dir_path, 'spi_smi.csv')

#-------------------------------------------------------------------------------
#SMI+SPI
mdp.set_model(original_model)
spmi.smi_spi(initial_policy, initial_model)

spmi.logger.save(dir_path, 'smi_spi.csv')