# Pytorch and tools
import torch as th
import numpy as np
import sys
import matplotlib.pyplot as plt
from RunningEnv import EnvWrapper
from Experiments import BatchActorCriticExperiment
from Learners import BatchOffpolicyActorCriticLearner, OPPOSDLearner
#%%
pref_pace = 181
target_pace = pref_pace * 1.1
# DEVICE = "cuda" if th.cuda.is_available() else "cpu"
#%%
def default_params():
    """ These are the default parameters used int eh framework. """
    return {  # Debugging outputs and plotting during training
        'plot_frequency': 10,  # plots a debug message avery n steps
        'plot_train_samples': False,  # whether the x-axis is env.steps (True) or episodes (False)
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
        'epsilon_finish': 0.05,  # annealing stops at (and keeps) this epsilon
        'epsilon_start': 0.05,  # annealing starts at this epsilon
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

def run_experiment(name,
                   batch_size=int(1e5),
                   mini_batch_size=1000,
                   offpolicy_iterations=10,
                   plot_frequency=4,
                   max_batch_episodes=int(200),
                   opposd=False,
                   experiments_batch=False,
                   opposd_iterations=50,
                   model=None,
                   train_batch=None):
    return_dict = {}
    params = default_params()
    params['offpolicy_iterations'] = offpolicy_iterations
    params['plot_frequency'] = plot_frequency
    params['max_batch_episodes'] = max_batch_episodes
    params['batch_size'] = batch_size
    params['mini_batch_size'] = mini_batch_size
    params['opposd'] = opposd
    params['experiments_batch'] = experiments_batch
    params['opposd_iterations'] = opposd_iterations
    # params['data_folder_path'] = "~/Documents/THESIS/Project_Juan/"
    params['data_folder_path'] = "/home/nfs/jmolano/THESIS/Project_Juan/"

    env = EnvWrapper(params.get('pref_pace'), params.get('target_pace'))
    n_actions, state_dim = params.get('num_actions'), params.get('states_shape')[0]
    if model==None:
        model = th.nn.Sequential(th.nn.Linear(state_dim, 128), th.nn.ReLU(),
                                 th.nn.Linear(128, 512), th.nn.ReLU(),
                                 th.nn.Linear(512, 128), th.nn.ReLU(),
                                 th.nn.Linear(128, n_actions + 1))

    if opposd:
        experiment = BatchActorCriticExperiment(params, model, learner=OPPOSDLearner(model, params=params))
    else:
        experiment = BatchActorCriticExperiment(params, model,
                                                         learner=BatchOffpolicyActorCriticLearner(model, params=params))
    if train_batch==None:
        train_batch = experiment.get_transition_batch()

    try:
        experiment.run(train_batch['buffer'])
    except KeyboardInterrupt:
        experiment.close()

    plt.show()

    return_dict.update({'model': name,
                        'experiment': experiment})

    return return_dict

algorithm = sys.argv[1]
offpolicy_data = sys.argv[2]
batch_size = int(sys.argv[3])
iterations = sys.argv[4]
run_number = sys.argv[5]

max_batch_episodes=int(iterations)
opposd=False
experiments_batch=False
model=None
train_batch=None
mini_batch_size = 1000
if len(sys.argv)>=6:
    mini_batch_size = int(sys.argv[5])


if batch_size < 2000:
    mini_batch_size = 500

params = default_params()
env = EnvWrapper(params.get('pref_pace'), params.get('target_pace'))
n_actions, state_dim = params.get('num_actions'), params.get('states_shape')[0]
params['epsilon_start'] = 0

model_actor = th.nn.Sequential(th.nn.Linear(state_dim, 128), th.nn.ReLU(),
                         th.nn.Linear(128, n_actions))
model_critic = th.nn.Sequential(th.nn.Linear(state_dim, 128), th.nn.ReLU(),
                         th.nn.Linear(128, 1))
model_w = th.nn.Sequential(th.nn.Linear(state_dim, 128), th.nn.ReLU(),
                         th.nn.Linear(128, 1), th.nn.Softplus())
model = [model_actor, model_critic, model_w]

if offpolicy_data=='experiment':
    experiments_batch = True

if algorithm=='opposd':
    opposd = True

name = ('%s_%s_%d_%s')%(offpolicy_data, algorithm, batch_size, run_number)

result = run_experiment(name,
                    max_batch_episodes=int(iterations),
                    opposd=opposd,
                    experiments_batch=experiments_batch,
                    model=model,
                    batch_size=batch_size,
                    mini_batch_size=mini_batch_size,
                    train_batch=train_batch)


save_values(result, name)