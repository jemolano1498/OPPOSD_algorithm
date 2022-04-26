import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

from data_analysis.RunningEnv import EnvWrapper
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from torch.autograd import Variable

#discount factor for future utilities
DISCOUNT_FACTOR = 0.99

#number of episodes to run
NUM_EPISODES = 1000

#max steps per episode
MAX_STEPS = 10000

#device to run model on
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FRAMES = 500
pref_pace = 181
target_pace = pref_pace*1.1

# Using a neural network to learn our policy parameters
class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)  # lstm
        self.fc_1 = nn.Linear(hidden_size, 128)  # fully connected 1
        self.fc = nn.Linear(128, num_classes)  # fully connected last layer

        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out)  # first Dense
        out = self.relu(out)  # relu
        actions = self.fc(out)  # Final Output

        # #get softmax for a probability distribution
        action_probs = F.softmax(actions, dim=1)

        return action_probs


def select_action(network, state):
    ''' Selects an action given current state
    Args:
    - network (Torch NN): network to process state
    - state (Array): Array of action space in an environment

    Return:
    - (int): action that is selected
    - (float): log probability of selecting that action given state and network
    '''

    # convert state to float tensor, add 1 dimension, allocate tensor on device
    state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
    state = torch.reshape(state, (state.shape[0], 1, state.shape[1]))

    # use network to predict action probabilities
    action_probs = network(state)

    # sample an action using the probability distribution
    m = Categorical(action_probs)
    action = m.sample()

    # return action
    return action.item(), m.log_prob(action)

#Make environment
env = EnvWrapper(pref_pace, target_pace)

#Init network
# network = PolicyNetwork(1, 10).to(DEVICE)

input_size = 1 #number of features
hidden_size = 2 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers

num_classes = 10 #number of output classes

network = LSTM1(num_classes, input_size, hidden_size, num_layers, 1).to(DEVICE)

#Init optimizer
optimizer = optim.Adam(network.parameters(), lr=1e-2)

state = env.step(0)[0]
print(select_action(network, state))