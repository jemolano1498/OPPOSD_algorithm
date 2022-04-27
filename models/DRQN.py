import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable
from data_analysis.RunningEnv import EnvWrapper

FRAMES = 500
pref_pace = 181
target_pace = pref_pace*1.1

class model(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(model, self).__init__()
        self.name = "LSTM"
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

    def init_hidden(self, batch_size):
        return [torch.zeros(self.num_layers, batch_size, self.hidden_size), Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))]

    def forward(self, x, hidden):
        # h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # hidden state
        # c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # internal state

        # Propagate input through LSTM
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1]))
        output, (hn, cn) = self.lstm(x, (hidden[0], hidden[1]))  # lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out)  # first Dense
        out = self.relu(out)  # relu
        actions = self.fc(out)  # Final Output

        return actions

        # #get softmax for a probability distribution
        # action_probs = F.softmax(actions, dim=1)
        #
        # return action_probs

env = EnvWrapper(pref_pace, target_pace)
env.reset()


# Network variables
input_size = 1 #number of features
hidden_size = 2 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers

num_classes = 5 #number of output classes

policy=model(num_classes, input_size, hidden_size, num_layers, 1)
target_net=model(num_classes, input_size, hidden_size, num_layers, 1)
target_net.load_state_dict(policy.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy.parameters())
criterion = F.smooth_l1_loss

memory=10000
store=[[dict()] for i in range(memory)]
gamma=0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200


def addEpisode(ind,prev,curr,reward,act):
    if len(store[ind]) ==0:
        store[ind][0]={'prev':prev,'curr':curr,'reward':reward,'action':act}
    else:
        store[ind].append({'prev':prev,'curr':curr,'reward':reward,'action':act})

def trainNet(total_episodes):
    if total_episodes==0:
        return
    ep=random.randint(0,total_episodes-1)
    if len(store[ep]) < 8:
        return
    else:
        start=random.randint(1,len(store[ep])-1)
        length=len(store[ep])
        inp=[]
        target=[]
        rew=torch.Tensor(1,length-start)
        actions=torch.Tensor(1,length-start)

        for i in range(start,length,1):
            inp.append((store[ep][i]).get('prev'))
            target.append((store[ep][i]).get('curr'))
            rew[0][i-start]=store[ep][i].get('reward')
            actions[0][i-start]=store[ep][i].get('action')
        targets = torch.Tensor(target[0].shape[0],target[0].shape[1])
        torch.cat(target, out=targets)
        ccs=torch.Tensor(inp[0].shape[0],inp[0].shape[1])
        torch.cat(inp, out=ccs)
        hidden = policy.init_hidden(length-start)
        qvals= target_net(targets,hidden)
        actions=actions.type('torch.LongTensor')
        actions=actions.reshape(length-start,1)
        hidden = policy.init_hidden(length-start)
        inps=policy(ccs,hidden).gather(1,actions)
        p1,p2=qvals.detach().max(1)
        targ = torch.Tensor(1,p1.shape[0])
        for num in range(start,length,1):
            if num==len(store[ep])-1:
                targ[0][num-start]=rew[0][num-start]
            else:
                targ[0][num-start]=rew[0][num-start]+gamma*p1[num-start]
        optimizer.zero_grad()
        inps=inps.reshape(1,length-start)
        loss = criterion(inps,targ)
        loss.backward()
        for param in policy.parameters():
            param.grad.data.clamp(-1,1)
        optimizer.step()

def trainDRQN(episodes):
    steps_done=0
    for i in range(0,episodes,1):
        print("Episode",i)
        env.reset()
        prev=env.step(0)[0]
        prev = torch.from_numpy(prev)
        prev = prev.type('torch.FloatTensor')
        done=False
        steps=0
        rew=0
        while env.steps < FRAMES:
            # eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            # math.exp(-1. * steps_done / EPS_DECAY)
            # print(steps,end=" ")
            steps+=1
            hidden = policy.init_hidden(1)
            output=policy(prev.unsqueeze(0), hidden)
            action=(output.argmax()).item()
            rand= random.uniform(0,1)
            if rand < 0.05:
                action=random.randint(0,5)

            sc,reward,done = env.step(action)

            sc = torch.from_numpy(sc)
            sc = sc.type('torch.FloatTensor')
            reward = torch.from_numpy(reward)
            reward = reward.type('torch.FloatTensor')
            done = torch.from_numpy(done)
            done = done.type('torch.FloatTensor')

            rew=rew+reward
            # if steps>200:
            #     terminal = torch.zeros(prev.shape[0],prev.shape[1],prev.shape[2])
            #     addEpisode(i,prev.unsqueeze(0),terminal.unsqueeze(0),-10,action)
            #     f=0
            #     break
            addEpisode(i,prev.unsqueeze(0),sc.unsqueeze(0),reward,action)
            trainNet(i)
            prev=sc
            steps_done+=1
        # terminal = torch.zeros(prev.shape[0],prev.shape[1],prev.shape[2])
        print(rew)
        # addEpisode(i,prev.unsqueeze(0),terminal.unsqueeze(0),-10,action)
        if i%10==0:
            target_net.load_state_dict(policy.state_dict())

trainDRQN(2000)