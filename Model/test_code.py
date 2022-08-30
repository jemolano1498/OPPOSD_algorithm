#%%
# Pytorch and tools
import torch as th
import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt
from RunningEnv import EnvWrapper
from Experiments import ActorCriticExperiment, BatchActorCriticExperiment, BatchHeuristicActorCriticExperiment, ActorCriticExperimentRunning
from Learners import ReinforceLearner
from Learners import OffpolicyActorCriticLearner, PPOLearner, OPPOSDLearner
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
#%%
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
        # print(state, action, reward)
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
#%%
def plot_experiments(experiments, names):
    colors = ['b', 'g', 'r']
    plt.figure(figsize=(8, 6), dpi=80)
    i = 0
    for exp in experiments:
        # Smooth curves
        window = max(int(len(exp.episode_returns) / 50), 10)
        if len(exp.episode_losses) < window + 2: return
        returns = np.convolve(exp.episode_returns, np.ones(window) / window, 'valid')
        # Determine x-axis based on samples or episodes
        x_returns = [i + window for i in range(len(returns))]
        plt.plot(x_returns, returns, colors[i], label=names[i])
        plt.xlabel('environment steps' if exp.plot_train_samples else 'batch trainings')
        plt.ylabel('episode return')
        i+=1
    plt.legend()
#%%

def get_offpolicy_batch(type):
    params = default_params()
    n_actions, state_dim = params.get('num_actions'), params.get('states_shape')[0]
    params['mini_batch_size'] = 200
    # The model has n_action policy heads and one value head
    model = th.nn.Sequential(th.nn.Linear(state_dim, 128), th.nn.ReLU(),
                             th.nn.Linear(128, n_actions + 1))

    if type=='heuristic':
        params['batch_size'] = int(1e5)
        experiment = BatchHeuristicActorCriticExperiment(params, model, learner=OPPOSDLearner(model, params=params))
    else:
        params['batch_size'] = int(4500)
        experiment = BatchActorCriticExperiment(params, model, learner=OPPOSDLearner(model, params=params))

    return experiment.get_transition_batch()

def save_values(experiment):
    values = experiment['experiment'].episode_returns
    name = experiment['model']
    np.savetxt("%s.csv"%name, values, delimiter=",")

def run_experiment(name,
                   batch_size=int(1e5),
                   mini_batch_size=200,
                   offpolicy_iterations=10,
                   plot_frequency=4,
                   max_batch_episodes=int(200),
                   opposd=False,
                   ppo=False,
                   heuristic=False,
                   opposd_iterations=50,
                   epsilon_start=0.05,
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
    params['heuristic'] = heuristic
    params['opposd_iterations'] = opposd_iterations
    params['epsilon_start'] = epsilon_start

    env = EnvWrapper(params.get('pref_pace'), params.get('target_pace'))
    n_actions, state_dim = params.get('num_actions'), params.get('states_shape')[0]
    if model==None:
        model = th.nn.Sequential(th.nn.Linear(state_dim, 128), th.nn.ReLU(),
                                 th.nn.Linear(128, 512), th.nn.ReLU(),
                                 th.nn.Linear(512, 128), th.nn.ReLU(),
                                 th.nn.Linear(128, n_actions + 1))
    if heuristic:
        if opposd:
            experiment = BatchHeuristicActorCriticExperiment(params, model, learner=OPPOSDLearner(model, params=params))
        elif ppo:
            experiment = BatchHeuristicActorCriticExperiment(params, model, learner=PPOLearner(model, params=params))
        else:
            experiment = BatchHeuristicActorCriticExperiment(params, model, learner=OffpolicyActorCriticLearner(model, params=params))
    else:
        if opposd:
            experiment = BatchActorCriticExperiment(params, model, learner=OPPOSDLearner(model, params=params))
        elif ppo:
            experiment = BatchActorCriticExperiment(params, model, learner=PPOLearner(model, params=params))
        else:
            experiment = BatchActorCriticExperiment(params, model,
                                                             learner=OffpolicyActorCriticLearner(model, params=params))
    if train_batch==None:
        train_batch = experiment.get_transition_batch()

    try:
        experiment.run(train_batch)
    except KeyboardInterrupt:
        experiment.close()

    experiment.plot_training()
    plt.show()

    return_dict.update({'model': name,
                        'experiment': experiment})

    return return_dict

def train_model(model, experiment, n_actions):

    init_optimizer = th.optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
    init_loss_func = th.nn.CrossEntropyLoss()
    #%%
    number = 3000
    # state 0
    states = np.random.randint(-8, high=28, size=(number,))
    # state 1
    states = np.append(states, np.random.randint(29, high=47, size=int(number/2,)))
    states = np.append(states, np.random.randint(-27, high=-9, size=int(number/2,)))
    # state 2
    states = np.append(states, np.random.randint(48, high=60, size=int(number/2,)))
    states = np.append(states, np.random.randint(-40, high=-28, size=int(number/2,)))
    # state 3
    states = np.append(states, np.random.randint(61, high=70, size=int(number/2,)))
    states = np.append(states, np.random.randint(-50, high=-41, size=int(number/2,)))
    # state 4
    states = np.append(states, np.random.randint(71, high=110, size=int(number/2,)))
    states = np.append(states, np.random.randint(-110, high=-51, size=int(number/2,)))

    states = states / 100
    states = th.tensor(states, dtype=th.float32).unsqueeze(dim=-1)

    y = experiment.runner.get_probabilities(states)

    data = th.cat((states, y), -1)

    trainloader = th.utils.data.DataLoader(data, batch_size=50,
                                              shuffle=True)

    # losses = []
    for epoch in range (20):
        # loss_ep = []
        for x in trainloader:
            init_optimizer.zero_grad()
            y = x[:,-5:]
            mx = model(x[:, 0].unsqueeze(dim=-1))[:, :n_actions]
            pred = th.nn.functional.softmax(mx, dim=-1)
            loss = init_loss_func(pred, y)
            # loss_ep = np.append(loss_ep, loss.detach().cpu())
            loss.backward()
            init_optimizer.step()

        # losses = np.append(losses, np.mean(loss_ep))
        # losses = loss.cpu()
        # if (epoch + 1)% 5 == 0:
        #     print('Pre-train Epoch: %s, Loss: %f'% (epoch, losses[-1:]))


algorithm = sys.argv[1]
offpolicy_data = sys.argv[2]
model_type = sys.argv[3]
exp_name = sys.argv[4]
iterations = sys.argv[5]

max_batch_episodes=int(iterations)
opposd=False
ppo=False
heuristic=False
model=None
train_batch=None

params = default_params()
env = EnvWrapper(params.get('pref_pace'), params.get('target_pace'))
n_actions, state_dim = params.get('num_actions'), params.get('states_shape')[0]
params['epsilon_start'] = 0

# The model has n_action policy heads and one value head
model = th.nn.Sequential(th.nn.Linear(state_dim, 128), th.nn.ReLU(),
                         th.nn.Linear(128, 512), th.nn.ReLU(),
                         th.nn.Linear(512, 128), th.nn.ReLU(),
                         th.nn.Linear(128, n_actions + 1))
experiment = BatchHeuristicActorCriticExperiment(params, model, learner=OPPOSDLearner(model, params=params))

if model_type=='trained':
    train_model(model, experiment, n_actions)
# test_in_environment(experiment, env)

if offpolicy_data=='heuristic':
    # train_batch = get_offpolicy_batch('heuristic')
    # dbfile = open('HeuristicPickle', 'ab')
    # pickle.dump(train_batch, dbfile)
    # dbfile.close()
    dbfile = open('HeuristicPickle', 'rb')
    train_batch = pickle.load(dbfile)
    dbfile.close()
    heuristic = True
else:
    # train_batch = get_offpolicy_batch('Exp')
    # dbfile = open('ExperimentsPickle', 'ab')
    # pickle.dump(train_batch, dbfile)
    # dbfile.close()
    dbfile = open('ExperimentsPickle', 'rb')
    train_batch = pickle.load(dbfile)
    dbfile.close()

if algorithm=='ppo':
    ppo = True
elif algorithm=='opposd':
    opposd = True

result = run_experiment(exp_name,
                    max_batch_episodes=int(iterations),
                    opposd=opposd,
                    ppo=ppo,
                    heuristic=heuristic,
                    model=model,
                    train_batch=train_batch)

save_values(result)