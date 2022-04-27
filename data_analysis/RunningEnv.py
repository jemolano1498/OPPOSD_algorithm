from data_analysis.PaceSimulator import PaceSimulator
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

    def get_remaining_percentage(self):
        return 1 - ((self.time_limit - self.time_step) / self.time_limit)


class EwmaBiasState:
    def __init__(self, rho=0.95):
        self.rho = rho  # Rho value for smoothing
        self.s_prev = 0  # Initial value ewma value
        self.timestep = 0
        self.s_cur = 0
        self.s_cur_bc = 0

    def get_next_state(self, input):
        self.s_cur = self.rho * self.s_prev + (1 - self.rho) * input
        self.s_cur_bc = (self.s_cur) / (1 - math.pow(self.rho, self.timestep + 1))
        self.s_prev = self.s_cur
        self.timestep = self.timestep + 1

        return self.s_cur_bc


class RunningEnv:
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
        new_state = (avg_pace / target_pace) - 1

        reward = self.get_distance_reward(target_pace, avg_pace) - pacing_value

        return np.array([current_pace]), np.array([new_state], dtype=np.float), np.array([reward],
                                                                                         dtype=np.float), np.array(
            [pacing_value]), np.array([done])

    def get_distance_reward(self, target_pace, current_pace):
        reward = 0
        if abs(target_pace - current_pace) > 5:
            reward = -10 * (abs(target_pace - current_pace) - 5)
        if abs(target_pace - current_pace) < 4:
            reward = -5
        if abs(target_pace - current_pace) < 3:
            reward = 5
        if abs(target_pace - current_pace) < 2:
            reward = 10
        if abs(target_pace - current_pace) < 1:
            reward = 20
        return reward

    def reset(self):
        self.state = EwmaBiasState()
        self.simulator = PaceSimulator()
        self.pace_active = 0
        self.current_timer = Timer(self.time_limit)


class EnvWrapper:
    def __init__(self, pref_pace, target_pace):
        self.times = [0, 20, 25, 30, 40]
        self.pref_pace = pref_pace
        self.target_pace = target_pace
        self.running_env = RunningEnv(pref_pace, 1)

        self.state_traj = np.empty(0)
        self.pace = np.empty(0)
        self.rewards = np.empty(0)
        self.env_pacing = np.empty(0)

        self.steps = 0

    def step(self, action):
        reward = 0
        for i in range(self.times[action]):
            current_pace, new_state, _, real_pacing, _ = self.running_env.step(1, self.target_pace)
            self.state_traj = np.append(self.state_traj, (new_state[0] + 1) * self.target_pace)
            self.pace = np.append(self.pace, current_pace)
            self.rewards = np.append(self.rewards, -1)
            self.env_pacing = np.append(self.env_pacing, real_pacing)
            self.steps = self.steps + 1
            reward = reward - 1
        if action == 0:
            current_pace, new_state, reward, real_pacing, done = self.running_env.step(0, self.target_pace)
            self.state_traj = np.append(self.state_traj, (new_state[0] + 1) * self.target_pace)
            self.pace = np.append(self.pace, current_pace)
            self.rewards = np.append(self.rewards, reward)
            self.env_pacing = np.append(self.env_pacing, real_pacing)
            self.steps = self.steps + 1
            return new_state, reward, done

        return new_state, np.array([round(reward/10)], dtype=np.float), np.array([0])

    def reset(self):
        self.running_env.reset()
        self.state_traj = np.empty(0)
        self.pace = np.empty(0)
        self.rewards = np.empty(0)
        self.env_pacing = np.empty(0)
        self.steps = 0
