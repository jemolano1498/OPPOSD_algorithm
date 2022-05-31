#%%
# Pytorch and tools
import torch as th
import numpy as np
import matplotlib.pyplot as plt
from RunningEnv import EnvWrapper
from Experiments import ActorCriticExperiment, BatchActorCriticExperiment, BatchHeuristicActorCriticExperiment, ActorCriticExperimentRunning
from Learners import ReinforceLearner
from Learners import OffpolicyActorCriticLearner, PPOLearner, OPPOSDLearner
#%%
pref_pace = 181
target_pace = pref_pace * 1.1
#%%
batch_episodes = 600
start_epsilon = 0.05
file_name = 'OFFPAC_heuristic'
experiments=[]
#%%
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
#%%
values = np.empty(0)
def save_values(experiment):
    global values
    if values.size == 0:
        values = np.append(values, experiment['experiment'].episode_returns)
    else:
        values = np.stack((values, experiment['experiment'].episode_returns))

    np.savetxt("%s.csv"%file_name, values, delimiter=",")
def test_in_environment(experiment, env):
    state = env.reset()
    try_scores = []

    for _ in range(50):
        env.reset()
        state = env.step(0)[0]
        done = False
        score = 0
        while not done:
            action = experiment.controller.choose(state, increase_counter=False).detach().item()
            new_state, reward, done = env.step(action)
            score += reward
            state = new_state
        try_scores.append(score)
    print(np.array(try_scores).mean())

    # Print one episode

    env.reset()
    state = env.step(0)[0]
    done = False

    while not done:
        action = experiment.controller.choose(state, increase_counter=False).detach().item()
        new_state, reward, done = env.step(action)

        # if action == 0:
        #     print(env.steps, action)
        # if reward < 0:
        #     print(action, state, new_state, reward)
        # if (action != 5):
        # #     # print(action, (state+1)*pref_pace, (new_state+1)*pref_pace, reward)
        #     print(action, state, new_state, reward)
        state = new_state

    x = np.linspace(0, len(env.env_pacing), len(env.env_pacing))
    plt.figure()
    plt.scatter(x[np.array(env.env_pacing) == 1], np.array(env.pace)[np.array(env.env_pacing) == 1], marker="x",
                label='Paced steps')
    plt.scatter(x[np.array(env.env_pacing) == 0], np.array(env.pace)[np.array(env.env_pacing) == 0], marker="x",
                label='Not-paced steps')

    # plt.scatter(x[np.array(env_pacing)==1], np.array(pace)[np.array(env_pacing)==1], marker="x", label='Paced steps')
    # plt.scatter(x[np.array(env_pacing)==0], np.array(pace)[np.array(env_pacing)==0], marker="x", label='Not-paced steps')

    # plt.scatter(x[np.array(pacing)==1], np.array(pacing)[np.array(pacing)==1]*181, color='r', marker="x")
    plt.axhline(y=target_pace, color='k', linestyle='--', label='Target Pace')

    plt.plot(x, env.state_traj, 'r-', linewidth=2)
    plt.legend()
    plt.show()

    print(np.sum(env.rewards))

#%% md
### Heuristic Policy
#%%
params = default_params()
env = EnvWrapper(params.get('pref_pace'), params.get('target_pace'))
n_actions, state_dim = params.get('num_actions'), params.get('states_shape')[0]
params['batch_size'] = int(1e5)
params['mini_batch_size'] = 200
params['epsilon_start'] = start_epsilon

# The model has n_action policy heads and one value head
model = th.nn.Sequential(th.nn.Linear(state_dim, 128), th.nn.ReLU(),
                         th.nn.Linear(128, 512), th.nn.ReLU(),
                         th.nn.Linear(512, 128), th.nn.ReLU(),
                         th.nn.Linear(128, n_actions + 1))
experiment = BatchHeuristicActorCriticExperiment(params, model, learner=OPPOSDLearner(model, params=params))
experiments_batch = experiment.get_transition_batch()
#%%
return_dict = {}
params = default_params()
params['offpolicy_iterations'] = 10
params['plot_train_samples'] = False
params['plot_frequency'] = 4
params['max_batch_episodes'] = int(batch_episodes)
params['batch_size'] = int(1e5)
params['mini_batch_size'] = 200
params['opposd'] = False
params['heuristic'] = False
params['opposd_iterations'] = 50
params['epsilon_start'] = start_epsilon
env = EnvWrapper(params.get('pref_pace'), params.get('target_pace'))
n_actions, state_dim = params.get('num_actions'), params.get('states_shape')[0]

# The model has n_action policy heads and one value head
model = th.nn.Sequential(th.nn.Linear(state_dim, 128), th.nn.ReLU(),
                         th.nn.Linear(128, 512), th.nn.ReLU(),
                         th.nn.Linear(512, 128), th.nn.ReLU(),
                         th.nn.Linear(128, n_actions + 1))
experiment = BatchHeuristicActorCriticExperiment(params, model, learner=OffpolicyActorCriticLearner(model, params=params))
try:
    experiment.run(experiments_batch)
except KeyboardInterrupt:
    experiment.close()

return_dict.update({'model' : 'OFFPAC heuristic',
                            'experiment': experiment})
experiments = np.append(experiments, return_dict)
save_values(return_dict)
#%%
# return_dict = {}
# params = default_params()
# params['offpolicy_iterations'] = 10
# params['plot_train_samples'] = False
# params['plot_frequency'] = 4
# params['max_batch_episodes'] = int(batch_episodes)
# params['batch_size'] = int(1e5)
# params['mini_batch_size'] = 200
# params['opposd'] = False
# params['heuristic'] = False
# params['opposd_iterations'] = 50
# params['epsilon_start'] = start_epsilon
# env = EnvWrapper(params.get('pref_pace'), params.get('target_pace'))
# n_actions, state_dim = params.get('num_actions'), params.get('states_shape')[0]
#
# # The model has n_action policy heads and one value head
# model = th.nn.Sequential(th.nn.Linear(state_dim, 128), th.nn.ReLU(),
#                          th.nn.Linear(128, 512), th.nn.ReLU(),
#                          th.nn.Linear(512, 128), th.nn.ReLU(),
#                          th.nn.Linear(128, n_actions + 1))
# experiment = BatchHeuristicActorCriticExperiment(params, model, learner=OPPOSDLearner(model, params=params))
# try:
#     experiment.run(experiments_batch)
# except KeyboardInterrupt:
#     experiment.close()
#
# return_dict.update({'model' : 'OPPOSD heuristic',
#                             'experiment': experiment})
# experiments = np.append(experiments, return_dict)
# save_values(return_dict)

# #%% md
# ## From Experiments
# #%%

# params = default_params()
# env = EnvWrapper(params.get('pref_pace'), params.get('target_pace'))
# n_actions, state_dim = params.get('num_actions'), params.get('states_shape')[0]
# params['batch_size'] = int(4500)
# params['mini_batch_size'] = 200
# params['epsilon_start'] = start_epsilon
#
# # The model has n_action policy heads and one value head
# model = th.nn.Sequential(th.nn.Linear(state_dim, 128), th.nn.ReLU(),
#                          th.nn.Linear(128, 512), th.nn.ReLU(),
#                          th.nn.Linear(512, 128), th.nn.ReLU(),
#                          th.nn.Linear(128, n_actions + 1))
# experiment = BatchActorCriticExperiment(params, model, learner=OPPOSDLearner(model, params=params))
# experiments_batch = experiment.get_transition_batch()
# #%%
# return_dict = {}
# params = default_params()
# params['offpolicy_iterations'] = 10
# params['plot_train_samples'] = False
# params['plot_frequency'] = 4
# params['max_batch_episodes'] = int(batch_episodes)
# params['batch_size'] = int(1e5)
# params['mini_batch_size'] = 200
# params['opposd'] = True
# params['opposd_iterations'] = 50
# params['epsilon_start'] = start_epsilon
# env = EnvWrapper(params.get('pref_pace'), params.get('target_pace'))
# n_actions, state_dim = params.get('num_actions'), params.get('states_shape')[0]
#
# # The model has n_action policy heads and one value head
# model = th.nn.Sequential(th.nn.Linear(state_dim, 128), th.nn.ReLU(),
#                          th.nn.Linear(128, 512), th.nn.ReLU(),
#                          th.nn.Linear(512, 128), th.nn.ReLU(),
#                          th.nn.Linear(128, n_actions + 1))
# experiment = BatchActorCriticExperiment(params, model, learner=OPPOSDLearner(model, params=params))
# try:
#     experiment.run(experiments_batch)
# except KeyboardInterrupt:
#     experiment.close()
#
# return_dict.update({'model' : 'OPPOSD',
#                             'experiment': experiment})
# experiments = np.append(experiments, return_dict)
# save_values(return_dict)
# #%%
# return_dict = {}
# params = default_params()
# params['offpolicy_iterations'] = 10
# params['plot_train_samples'] = False
# params['plot_frequency'] = 4
# params['max_batch_episodes'] = int(batch_episodes)
# params['batch_size'] = int(1e5)
# params['mini_batch_size'] = 200
# params['opposd'] = False
# params['opposd_iterations'] = 50
# params['epsilon_start'] = start_epsilon
# env = EnvWrapper(params.get('pref_pace'), params.get('target_pace'))
# n_actions, state_dim = params.get('num_actions'), params.get('states_shape')[0]
#
# # The model has n_action policy heads and one value head
# model = th.nn.Sequential(th.nn.Linear(state_dim, 128), th.nn.ReLU(),
#                          th.nn.Linear(128, 512), th.nn.ReLU(),
#                          th.nn.Linear(512, 128), th.nn.ReLU(),
#                          th.nn.Linear(128, n_actions + 1))
# experiment = BatchActorCriticExperiment(params, model, learner=OffpolicyActorCriticLearner(model, params=params))
# try:
#     experiment.run(experiments_batch)
# except KeyboardInterrupt:
#     experiment.close()
#
# return_dict.update({'model' : 'OFFPAC',
#                             'experiment': experiment})
# experiments = np.append(experiments, return_dict)
# save_values(return_dict)
# #%%
# return_dict = {}
# params = default_params()
# params['offpolicy_iterations'] = 10
# params['plot_train_samples'] = False
# params['plot_frequency'] = 4
# params['max_batch_episodes'] = int(batch_episodes)
# params['batch_size'] = int(1e5)
# params['mini_batch_size'] = 200
# params['opposd'] = False
# params['opposd_iterations'] = 50
# params['epsilon_start'] = start_epsilon
# env = EnvWrapper(params.get('pref_pace'), params.get('target_pace'))
# n_actions, state_dim = params.get('num_actions'), params.get('states_shape')[0]
#
# # The model has n_action policy heads and one value head
# model = th.nn.Sequential(th.nn.Linear(state_dim, 128), th.nn.ReLU(),
#                          th.nn.Linear(128, 512), th.nn.ReLU(),
#                          th.nn.Linear(512, 128), th.nn.ReLU(),
#                          th.nn.Linear(128, n_actions + 1))
# experiment = BatchActorCriticExperiment(params, model, learner=PPOLearner(model, params=params))
# try:
#     experiment.run(experiments_batch)
# except KeyboardInterrupt:
#     experiment.close()
#
# return_dict.update({'model' : 'PPO',
#                             'experiment': experiment})
# experiments = np.append(experiments, return_dict)
# save_values(return_dict)


