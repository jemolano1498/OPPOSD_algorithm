#%%
# Pytorch and tools
import torch as th
import numpy as np
import matplotlib.pyplot as plt
from RunningEnv import EnvWrapper
from Experiments import ActorCriticExperiment
from Experiments import BatchActorCriticExperiment
from Learners import ReinforceLearner
from Learners import BatchReinforceLearner
from Learners import ActorCriticLearner
from Learners import OffpolicyActorCriticLearner
from Learners import PPOLearner
from Learners import BatchOffpolicyActorCriticLearner, BatchPPOLearner, OPPOSDLearner
import pickle
#%%
pref_pace = 181
target_pace = pref_pace * 1.1
batch_number = 300
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
def test_in_environment(experiment, env):
    done = False
    env.reset()
    state = env.step(0)[0]
    try_scores = []

    for _ in range(50):
        env.reset()
        state = env.step(0)[0]
        done = False
        score = 0
        time_step = 0
        while env.steps < 500:
            pred = th.nn.functional.softmax(model_actor(th.tensor(state, dtype=th.float32).unsqueeze(dim=-1))[:, :n_actions], dim=-1)
            action = th.distributions.Categorical(probs=pred).sample()
            new_state, reward, done = env.step(action)
            score += reward
            state = new_state
            if action > 0:
                time_step = time_step + env.times[action % 5]
            else:
                time_step = time_step + 1
        try_scores.append(score)
    print(np.array(try_scores).mean())

    # Print one episode

    env.reset()
    state = env.step(0)[0]
    rewards = 0

    while env.steps < 500:
        pred = th.nn.functional.softmax(model_actor(th.tensor(state, dtype=th.float32).unsqueeze(dim=-1))[:, :n_actions], dim=-1)
        action = th.distributions.Categorical(probs=pred).sample()
        new_state, reward, done = env.step(action)
        rewards += reward
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

    print(rewards)
#%%
def plot_experiments(experiments, names):
    # sns.set()
    colors = ['b', 'g', 'r']
    plt.figure(figsize=(8, 6), dpi=80)
    i = 0
    for exp in experiments:
        # Smooth curves
        window = max(int(len(exp.episode_returns) / 50), 10)
        print(window)
        # if len(exp.episode_losses) < window + 2: return
        returns = np.convolve(exp.episode_returns, np.ones(window) / window, 'valid')
        # Determine x-axis based on samples or episodes
        x_returns = [i + window for i in range(len(returns))]
        plt.plot(x_returns, returns, colors[i], label=names[i])
        plt.xlabel('environment steps' if exp.plot_train_samples else 'batch trainings')
        plt.ylabel('episode return')
        i+=1
    plt.legend()

batch_experiments = []

dbfile = open('experiments_simulator_batch_steps_pickle_3e3', 'rb')
batch = pickle.load(dbfile)
dbfile.close()
#%%

params = default_params()
params['plot_train_samples'] = False
params['plot_frequency'] = 4
params['batch_size'] = int(3200)
params['offpolicy_iterations'] = 10
params['opposd'] = True
params['max_batch_episodes'] = int(batch_number)
params['mini_batch_size'] = int(500)

env = EnvWrapper(params.get('pref_pace'), params.get('target_pace'))
n_actions, state_dim = params.get('num_actions'), params.get('states_shape')[0]

# The model has n_action policy heads and one value head
model_actor = th.nn.Sequential(th.nn.Linear(state_dim, 128), th.nn.ReLU(),
                         th.nn.Linear(128, n_actions))
model_critic = th.nn.Sequential(th.nn.Linear(state_dim, 128), th.nn.ReLU(),
                         th.nn.Linear(128, 1))
model_w = th.nn.Sequential(th.nn.Linear(state_dim, 128), th.nn.ReLU(),
                         th.nn.Linear(128, 1), th.nn.Softplus())
model = [model_actor, model_critic, model_w]

experiment = BatchActorCriticExperiment(params, model, learner=OPPOSDLearner(model, params=params))

try:
    experiment.run(batch['buffer'])
except KeyboardInterrupt:
    experiment.close()
return_dict = {}
return_dict.update({'model' : 'Experimental Batch OPPOSD',
                            'experiment': experiment})
batch_experiments = np.append(batch_experiments, return_dict)


#%%
params = default_params()
params['plot_train_samples'] = False
params['plot_frequency'] = 4
params['batch_size'] = int(1e5)
params['offpolicy_iterations'] = 10
params['max_batch_episodes'] = int(batch_number)
params['mini_batch_size'] = int(500)

env = EnvWrapper(params.get('pref_pace'), params.get('target_pace'))
n_actions, state_dim = params.get('num_actions'), params.get('states_shape')[0]
# The model has n_action policy heads and one value head
model_actor = th.nn.Sequential(th.nn.Linear(state_dim, 128), th.nn.ReLU(),
                         th.nn.Linear(128, n_actions))
model_critic = th.nn.Sequential(th.nn.Linear(state_dim, 128), th.nn.ReLU(),
                         th.nn.Linear(128, 1))
model_w = th.nn.Sequential(th.nn.Linear(state_dim, 128), th.nn.ReLU(),
                         th.nn.Linear(128, 1), th.nn.Softplus())
model = [model_actor, model_critic, model_w]
experiment = BatchActorCriticExperiment(params, model, learner=BatchOffpolicyActorCriticLearner(model, params=params))

try:
    experiment.run(batch['buffer'])
except KeyboardInterrupt:
    experiment.close()

return_dict = {}
return_dict.update({'model' : 'Experimental Batch OFFPAC',
                            'experiment': experiment})
batch_experiments = np.append(batch_experiments, return_dict)

#%%
def plot_experiments(experiments, name):
    # sns.set()
    colors = ['r', 'b', 'r:', 'b:', 'c', 'm']
    plt.figure(figsize=(8, 6), dpi=80)
    i = 0
    for exp in experiments:
        # Smooth curves
        window = max(int(len(exp['experiment'].episode_returns) / 5), 1)
        # if len(exp.episode_losses) < window + 2: return
        returns = np.convolve(exp['experiment'].episode_returns, np.ones(window) / window, 'valid')
        # Determine x-axis based on samples or episodes
        x_returns = [i + window for i in range(len(returns))]
        plt.plot(x_returns, returns, colors[i], label=exp['model'])
        plt.xlabel('Policy gradient step')
        plt.ylabel('Episode return')
        i+=1
    plt.legend()
    plt.title('Running environment')
    plt.savefig('running_comp_3000_exp_batch_%s.pdf'%(name))
#%%
plot_experiments(batch_experiments, '1')

for exp in batch_experiments:
    # Smooth curves
    np.savetxt("%s_EXPERIMENT.csv"%(exp['model']), exp['experiment'].episode_returns, delimiter=",")

dbfile = open('random_simulator_batch_steps_pickle_3e3', 'rb')
# dbfile = open('random_simulator_batch_steps_pickle_5e2', 'rb') # Do not work
batch = pickle.load(dbfile)
dbfile.close()
#%%

params = default_params()
params['plot_train_samples'] = False
params['plot_frequency'] = 4
params['batch_size'] = int(3200)
params['offpolicy_iterations'] = 10
params['opposd'] = True
params['max_batch_episodes'] = int(batch_number)
params['mini_batch_size'] = int(500)

env = EnvWrapper(params.get('pref_pace'), params.get('target_pace'))
n_actions, state_dim = params.get('num_actions'), params.get('states_shape')[0]

# The model has n_action policy heads and one value head
model_actor = th.nn.Sequential(th.nn.Linear(state_dim, 128), th.nn.ReLU(),
                         th.nn.Linear(128, n_actions))
model_critic = th.nn.Sequential(th.nn.Linear(state_dim, 128), th.nn.ReLU(),
                         th.nn.Linear(128, 1))
model_w = th.nn.Sequential(th.nn.Linear(state_dim, 128), th.nn.ReLU(),
                         th.nn.Linear(128, 1), th.nn.Softplus())
model = [model_actor, model_critic, model_w]

experiment = BatchActorCriticExperiment(params, model, learner=OPPOSDLearner(model, params=params))

try:
    experiment.run(batch['buffer'])
except KeyboardInterrupt:
    experiment.close()
return_dict = {}
return_dict.update({'model' : 'Random Batch OPPOSD',
                            'experiment': experiment})
batch_experiments = np.append(batch_experiments, return_dict)


#%%
params = default_params()
params['plot_train_samples'] = False
params['plot_frequency'] = 4
params['batch_size'] = int(1e5)
params['offpolicy_iterations'] = 10
params['max_batch_episodes'] = int(batch_number)
params['mini_batch_size'] = int(500)

env = EnvWrapper(params.get('pref_pace'), params.get('target_pace'))
n_actions, state_dim = params.get('num_actions'), params.get('states_shape')[0]
# The model has n_action policy heads and one value head
model_actor = th.nn.Sequential(th.nn.Linear(state_dim, 128), th.nn.ReLU(),
                         th.nn.Linear(128, n_actions))
model_critic = th.nn.Sequential(th.nn.Linear(state_dim, 128), th.nn.ReLU(),
                         th.nn.Linear(128, 1))
model_w = th.nn.Sequential(th.nn.Linear(state_dim, 128), th.nn.ReLU(),
                         th.nn.Linear(128, 1), th.nn.Softplus())
model = [model_actor, model_critic, model_w]
experiment = BatchActorCriticExperiment(params, model, learner=BatchOffpolicyActorCriticLearner(model, params=params))

try:
    experiment.run(batch['buffer'])
except KeyboardInterrupt:
    experiment.close()

return_dict = {}
return_dict.update({'model' : 'Random Batch OFFPAC',
                            'experiment': experiment})
batch_experiments = np.append(batch_experiments, return_dict)

plot_experiments(batch_experiments, '2')

for exp in batch_experiments:
    # Smooth curves
    np.savetxt("%s_EXPERIMENT.csv"%(exp['model']), exp['experiment'].episode_returns, delimiter=",")