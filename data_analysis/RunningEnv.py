from PaceSimulator import PaceSimulator
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

    def get_remaining_percentage (self):
        return 1-((self.time_limit - self.time_step)/self.time_limit)



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
        done = 0
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
        avg_pace = self.state.get_next_state(current_pace)
        new_state = [(avg_pace/self.pref_pace) - 1, (2 * self.current_timer.get_remaining_percentage()) - 1]

        reward = self.get_distance_reward(target_pace, avg_pace) - pacing_value

        return np.array([current_pace]), np.array(new_state), np.array([reward], dtype=np.float), np.array([pacing_value]), np.array([done])
        # return np.array([current_pace]), np.array(new_state), np.array([reward], dtype=np.float), np.array([pacing_value]), np.array([done])

    def get_distance_reward(self, target_pace, current_pace):
        reward = 0
        if abs(target_pace - current_pace) > 4:
            reward = -5
        if abs(target_pace - current_pace) < 3:
            reward = 1
        if abs(target_pace - current_pace) < 2:
            reward = 2
        return reward

    def reset (self):
        self.state = EwmaBiasState()
        self.simulator = PaceSimulator()
        self.pace_active = 0
        self.current_timer = Timer(self.time_limit)


# env = RunningEnv(181)
# state = np.empty(0)
# pace = np.empty(0)
# rewards = np.empty(0)
# env_pacing = np.empty(0)
# pacing = []
#
# for y in range(100):
#     value = 0
#     current_pace, new_state, reward, real_pacing = env.step(value, 181)
#     state = np.append(state, new_state)
#     pace = np.append(pace, current_pace)
#     rewards = np.append(rewards, reward)
#     env_pacing = np.append(env_pacing, real_pacing)
#     pacing.append(value)
#
# for y in range(50):
#     value = 1
#     current_pace, new_state, reward, real_pacing = env.step(value, 181)
#     state = np.append(state, new_state)
#     pace = np.append(pace, current_pace)
#     rewards = np.append(rewards, reward)
#     env_pacing = np.append(env_pacing, real_pacing)
#     pacing.append(value)
#
# for y in range(100):
#     value = 0
#     current_pace, new_state, reward, real_pacing = env.step(value, 181)
#     state = np.append(state, new_state)
#     pace = np.append(pace, current_pace)
#     rewards = np.append(rewards, reward)
#     env_pacing = np.append(env_pacing, real_pacing)
#     pacing.append(value)
#
# x = np.linspace(0, len(pacing), len(pacing))
# plt.scatter(x[np.array(env_pacing)==1], np.array(pace)[np.array(env_pacing)==1], marker="x", label='Paced steps')
# plt.scatter(x[np.array(env_pacing)==0], np.array(pace)[np.array(env_pacing)==0], marker="x", label='Not-paced steps')
#
# plt.scatter(x[np.array(pacing)==1], np.array(pacing)[np.array(pacing)==1]*181, color='r', marker="x")
# plt.axhline(y=181, color='k', linestyle='--', label='Target Pace')
#
# plt.plot(x, state, 'r-', linewidth=2)
# plt.annotate("Hola", xy=(0.5,0.5), xycoords='figure points')
# plt.show()
#
# print(rewards.mean())

##
