from PaceSimulator import PaceSimulator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

class Timer:
    def __init__(self, time_limit=30):
        self.time_step = 0
        self.time_limit = time_limit

    def tick(self):
        self.time_step = self.time_step + 1
        if self.time_step == self.time_limit:
            self.time_step = 0

    def timer_on(self):
        if self.time_step == 0:
            return 0
        else:
            return 1



class EwmaBiasState:
    def __init__(self, rho=0.95):
        self.rho = rho # Rho value for smoothing
        self.s_prev = 0 # Initial value ewma value
        self.timestep = 0
        self.s_cur = 0
        self.s_cur_bc = 0

    def get_next_state (self, input):
        self.s_cur = self.rho * self.s_prev + (1-self.rho) * input
        self.s_cur_bc = (self.s_cur)/(1-math.pow(self.rho,self.timestep+1))
        self.s_prev = self.s_cur
        self.timestep = self.timestep + 1

        return self.s_cur_bc

class RunningEnv:
    # Beeps should be active for 30 timesteps (10 sec)
    def __init__(self, pref_pace, time_limit=30):
        self.state = EwmaBiasState()
        self.simulator = PaceSimulator()
        self.pace_active = 0
        self.pref_pace = pref_pace
        self.time_limit = time_limit
        self.current_timer = Timer(self.time_limit)

    def step(self, action, target_pace):
        if action == 1:
            self.current_timer = Timer(self.time_limit)
            self.current_timer.tick()
            self.pace_active = 1
            pacing_value = 1
        else:
            self.pace_active = 0
            pacing_value = 0
            if self.current_timer.timer_on():
                pacing_value = 1
                self.current_timer.tick()

        current_pace = self.simulator.predict(pacing_value, self.pref_pace, target_pace)
        new_state = self.state.get_next_state(current_pace)

        reward = self.get_distance_reward(target_pace, current_pace) + self.get_pacing_reward()

        return current_pace, new_state, reward, pacing_value

    def get_distance_reward(self, target_pace, current_pace):
        return -abs(target_pace - current_pace) + 1

    def get_pacing_reward(self):
        if self.current_timer.timer_on():
            return -1
        else:
            return 0

env = RunningEnv(181)
state = np.empty(0)
pace = np.empty(0)
rewards = np.empty(0)
env_pacing = np.empty(0)
pacing = []

for y in range(100):
    value = 0
    current_pace, new_state, reward, real_pacing = env.step(value, 181)
    state = np.append(state, new_state)
    pace = np.append(pace, current_pace)
    rewards = np.append(rewards, reward)
    env_pacing = np.append(env_pacing, real_pacing)
    pacing.append(value)

for y in range(50):
    value = 1
    current_pace, new_state, reward, real_pacing = env.step(value, 181)
    state = np.append(state, new_state)
    pace = np.append(pace, current_pace)
    rewards = np.append(rewards, reward)
    env_pacing = np.append(env_pacing, real_pacing)
    pacing.append(value)

for y in range(100):
    value = 0
    current_pace, new_state, reward, real_pacing = env.step(value, 181)
    state = np.append(state, new_state)
    pace = np.append(pace, current_pace)
    rewards = np.append(rewards, reward)
    env_pacing = np.append(env_pacing, real_pacing)
    pacing.append(value)

x = np.linspace(0, len(pacing), len(pacing))
plt.scatter(x[np.array(env_pacing)==1], np.array(pace)[np.array(env_pacing)==1], marker="x", label='Paced steps')
plt.scatter(x[np.array(env_pacing)==0], np.array(pace)[np.array(env_pacing)==0], marker="x", label='Not-paced steps')

plt.scatter(x[np.array(pacing)==1], np.array(pacing)[np.array(pacing)==1]*181, color='r', marker="x")
plt.axhline(y=181, color='k', linestyle='--', label='Target Pace')

plt.plot(x, state, 'r-', linewidth=2)
plt.show()

print(rewards.mean())

##

plt.axis([0, 10, 0, 1])

for i in range(10):
    y = np.random.random()
    plt.scatter(i, y)
    plt.pause(0.05)

plt.show()