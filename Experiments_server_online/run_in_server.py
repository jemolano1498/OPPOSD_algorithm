
# %%
# Pytorch and tools
import sys

import torch as th
import numpy as np

from RunningEnv import EnvWrapper
from Experiments import ActorCriticExperiment
from Experiments import BatchActorCriticExperiment
from Learners import ReinforceLearner
from Learners import BatchReinforceLearner
from Learners import ActorCriticLearner
from Learners import OffpolicyActorCriticLearner
from Learners import PPOLearner
from Learners import BatchOffpolicyActorCriticLearner, BatchPPOLearner, OPPOSDLearner
# Reinforcement learning
import gym

pref_pace = 181
target_pace = pref_pace * 1.1

def default_params():
    """ These are the default parameters used int eh framework. """
    return {  # Debugging outputs and plotting during training
        'plot_frequency': 10,  # plots a debug message avery n steps
        'plot_train_samples': True,  # whether the x-axis is env.steps (True) or episodes (False)
        'print_when_plot': True,  # prints debug message if True
        'print_dots': False,  # prints dots for every gradient update
        # Environment parameters
        'env': 'CartPole-v0',  # the environment the agent is learning in
        'run_steps': 0,  # samples whole episodes if run_steps <= 0
        'max_episode_length': 500,  # maximum number of steps per episode
        # Runner parameters
        'max_episodes': int(1E6),  # experiment stops after this many episodes
        'max_batch_episodes': int(1E6),  # experiment stops after this many batch
        'max_steps': int(1E9),  # experiment stops after this many steps
        'multi_runner': False,  # uses multiple runners if True
        'parallel_environments': 4,  # number of parallel runners  (only if multi_runner==True)
        # Exploration parameters
        'epsilon_anneal_time': int(5E3),  # exploration anneals epsilon over these many steps
        'epsilon_finish': 0.1,  # annealing stops at (and keeps) this epsilon
        'epsilon_start': 1,  # annealing starts at this epsilon
        # Optimization parameters
        'lr': 1E-4,  # 5E-4,                       # learning rate of optimizer
        'gamma': 0.99,  # discount factor gamma
        'batch_size': 2048,  # number of transitions in a mini-batch
        'grad_norm_clip': 1,  # gradent clipping if grad norm is larger than this
        # DQN parameters
        'replay_buffer_size': int(1E5),  # the number of transitions in the replay buffer
        'use_last_episode': True,  # whether the last episode is always sampled from the buffer
        'target_model': True,  # whether a target model is used in DQN
        'target_update': 'soft',  # 'soft' target update or hard update by regular 'copy'
        'target_update_interval': 10,  # interval for the 'copy' target update
        'soft_target_update_param': 0.01,  # update parameter for the 'soft' target update
        'double_q': True,  # whether DQN uses double Q-learning
        'grad_repeats': 1,  # how many gradient updates / runner call
        # Image input parameters
        'pixel_observations': False,  # use pixel observations (we will not use this feature here)
        'pixel_resolution': (78, 78),  # scale image to this resoluton
        'pixel_grayscale': True,  # convert image into grayscale
        'pixel_add_last_obs': True,  # stacks 2 observations
        'pixel_last_obs_delay': 3,  # delay between the two stacked observations

        # Runners env
        'pref_pace': 181,  # Athlete's preferred pace
        'target_pace': pref_pace * 1.1,  # Athlete's target pace
        'states_shape': (1,),  # Amount of states
        'num_actions': 5,  # Possible actions
    }


def save_values(experiment, name):
    values = experiment['experiment'].episode_returns
    # name = experiment['model']
    np.savetxt("%s.csv"%name, values, delimiter=",")

def run_experiment (name,
                    algorithm,
                   max_batch_episodes=int(1000),
                    offpolicy_iterations=128,
                   model=None):
    return_dict = {}
    params = default_params()
    params['offpolicy_iterations'] = offpolicy_iterations
    params['plot_frequency'] = 100
    params['max_batch_episodes'] = max_batch_episodes
    params['plot_train_samples'] = False
    if algorithm == 'RL' or algorithm == 'AC':
        params['epsilon_anneal_time'] = 100000
        params['epsilon_finish'] = 0.05

    n_actions, state_dim = params.get('num_actions'),  int(params.get('states_shape')[0])
    if model==None:
        model_actor = th.nn.Sequential(th.nn.Linear(state_dim, 128), th.nn.ReLU(),
                                       th.nn.Linear(128, 512), th.nn.ReLU(),
                                       th.nn.Linear(512, 128), th.nn.ReLU(),
                                       th.nn.Linear(128, n_actions + 1))
        model_critic = th.nn.Sequential(th.nn.Linear(state_dim, 128), th.nn.ReLU(),
                                        th.nn.Linear(128, 1))
        model_w = th.nn.Sequential(th.nn.Linear(state_dim, 128), th.nn.ReLU(),
                                   th.nn.Linear(128, 1), th.nn.Softplus())
        model = [model_actor, model_critic, model_w]

    if algorithm == 'RL':
        experiment = ActorCriticExperiment(params, model, learner=ReinforceLearner(model, params=params))
    elif algorithm == 'AC':
        experiment = ActorCriticExperiment(params, model, learner=ActorCriticLearner(model, params=params))
    elif algorithm == 'OFFPAC':
        experiment = ActorCriticExperiment(params, model, learner=OffpolicyActorCriticLearner(model, params=params))
    elif algorithm == 'PPO':
        experiment = ActorCriticExperiment(params, model, learner=PPOLearner(model, params=params))
    else:
        print('No algortihm specified')
        return

    try:
        experiment.run()
    except KeyboardInterrupt:
        experiment.close()

    return_dict.update({'model': name,
                        'experiment': experiment})

    return return_dict

# RL, AC, OFFPAC, OPPOSD
algorithm = sys.argv[1]

# max_batch_episodes
max_batch_episodes = int(sys.argv[2])

run_number = sys.argv[3]

model=None
offpolicy_iterations = 64

if algorithm == 'RL' or algorithm == 'AC':
    offpolicy_iterations = 0

params = default_params()

env = EnvWrapper(params.get('pref_pace'), params.get('target_pace'))
n_actions, state_dim = params.get('num_actions'), int(params.get('states_shape')[0])

model_actor = th.nn.Sequential(th.nn.Linear(state_dim, 128), th.nn.ReLU(),
                         th.nn.Linear(128, 512), th.nn.ReLU(),
                         th.nn.Linear(512, 128), th.nn.ReLU(),
                         th.nn.Linear(128, n_actions + 1))
model_critic = th.nn.Sequential(th.nn.Linear(state_dim, 128), th.nn.ReLU(),
                         th.nn.Linear(128, 1))
model_w = th.nn.Sequential(th.nn.Linear(state_dim, 128), th.nn.ReLU(),
                         th.nn.Linear(128, 1), th.nn.Softplus())
model = [model_actor, model_critic, model_w]

name = ('runningenv_%s_%s')%(algorithm, run_number)

result = run_experiment(name,
                    algorithm,
                    max_batch_episodes=int(max_batch_episodes),
                    offpolicy_iterations=offpolicy_iterations,
                    model=model)

save_values(result, name)