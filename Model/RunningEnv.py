from PaceSimulator import PaceSimulator
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

    def step(self, action, target_pace, extra_pace=0):
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
        current_pace = self.simulator.predict(pacing_value, self.pref_pace, target_pace, extra_pace)
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


class EnvWrapper_2:
    def __init__(self, pref_pace, target_pace):
        self.times = [0, 20, 25, 30, 40]
        self.pref_pace = pref_pace
        self.target_pace = target_pace
        self.running_env = RunningEnv(pref_pace, 1)

        self.max_steps = 500

        self.state_traj = np.empty(0)
        self.pace = np.empty(0)
        self.rewards = np.empty(0)
        self.env_pacing = np.empty(0)

        self.steps = 0

    def step(self, action):
        reward = 0
        done = 0
        timelapse = action % len(self.times)
        extra_pace = 0 if action < 5 else 0.01
        new_state = []
        for i in range(self.times[timelapse]):
            current_pace, new_state, _, real_pacing, _ = self.running_env.step(1, self.target_pace, extra_pace)
            self.state_traj = np.append(self.state_traj, (new_state[0] + 1) * self.target_pace)
            self.pace = np.append(self.pace, current_pace)
            self.rewards = np.append(self.rewards, -1)
            self.env_pacing = np.append(self.env_pacing, real_pacing)
            self.steps = self.steps + 1
            reward = reward - 1
        if timelapse == 0:
            current_pace, new_state, reward, real_pacing, done = self.running_env.step(0, self.target_pace)
            self.state_traj = np.append(self.state_traj, (new_state[0] + 1) * self.target_pace)
            self.pace = np.append(self.pace, current_pace)
            self.rewards = np.append(self.rewards, reward)
            self.env_pacing = np.append(self.env_pacing, real_pacing)
            self.steps = self.steps + 1
            if self.steps > self.max_steps:
                done = 1
            # Normalize output state values
            new_state = (new_state - (-0.010204584316159032)) / (0.05290357993014821)
            return new_state, reward, done
        if self.steps > self.max_steps:
            done = 1
        # Normalize output state values
        new_state = (new_state - (-0.010204584316159032)) / (0.05290357993014821)
        return new_state, np.array([round(reward)], dtype=np.float), np.array([done])

    def reset(self):
        self.running_env.reset()
        self.state_traj = np.empty(0)
        self.pace = np.empty(0)
        self.rewards = np.empty(0)
        self.env_pacing = np.empty(0)
        self.steps = 0
        return (self.running_env.step(0, self.target_pace)[1] - (-0.010204584316159032)) / (0.05290357993014821)

    def close(self):
        pass

class EnvWrapper:
    def __init__(self, pref_pace, target_pace):
        self.times = [0, 20, 25, 30, 40]
        self.pref_pace = pref_pace
        self.target_pace = target_pace
        self.running_env = RunningEnv(pref_pace, 1)

        self.tresshold = 0.0225

        self.max_steps = 500
        self.max_transitions = 16

        self.state_traj = np.empty(0)
        self.pace = np.empty(0)
        self.rewards = np.empty(0)
        self.env_pacing = np.empty(0)

        self.steps = 0
        self.transitions = 0

    def step(self, action):
        reward = 1
        done = 0
        extra_pace = 0
        new_state = []
        prev_states = []
        for i in range(self.times[action]):
            current_pace, new_state, _, real_pacing, _ = self.running_env.step(1, self.target_pace, extra_pace)
            prev_states = np.append(prev_states, new_state)
            self.state_traj = np.append(self.state_traj, (new_state[0] + 1) * self.target_pace)
            self.pace = np.append(self.pace, current_pace)
            self.rewards = np.append(self.rewards, -1)
            self.env_pacing = np.append(self.env_pacing, real_pacing)
            self.steps = self.steps + 1
            # reward = reward - 1
        if action == 0:
            for i in range(20):
                current_pace, new_state, _, real_pacing, _ = self.running_env.step(0, self.target_pace)
                self.state_traj = np.append(self.state_traj, (new_state[0] + 1) * self.target_pace)
                self.pace = np.append(self.pace, current_pace)
                self.env_pacing = np.append(self.env_pacing, real_pacing)
                self.steps = self.steps + 1
            if self.transitions >= self.max_transitions:
                done = 1

            if abs(new_state[0]) > 0.015:
                reward = -1
            elif abs(new_state[0]) < 0.002:
                reward = 1
            else:
                reward = 0

            self.rewards = np.append(self.rewards, reward)
            # Normalize output state values
            new_state = (new_state - (-0.010204584316159032)) / (0.05290357993014821)
            self.transitions += 1
            return new_state, np.array([round(reward)], dtype=np.float), done
        if self.transitions >= self.max_transitions:
            done = 1
        if abs(new_state[0]) > 0.0225:
            reward = -1
        elif math.sqrt((new_state[0] - (prev_states[0]))**2) < 0.005:
            reward = -1
        else:
            reward = 2
        # Normalize output state values
        new_state = (new_state - (-0.010204584316159032)) / (0.05290357993014821)
        self.transitions += 1
        return new_state, np.array([round(reward)], dtype=np.float), np.array([done])

    def reset(self):
        self.running_env.reset()
        self.state_traj = np.empty(0)
        self.pace = np.empty(0)
        self.rewards = np.empty(0)
        self.env_pacing = np.empty(0)
        self.steps = 0
        self.transitions = 0
        # return (self.running_env.step(0, self.target_pace)[1] - (-0.010204584316159032)) / (0.05290357993014821)
        return (self.step(0)[0])

    def close(self):
        pass
