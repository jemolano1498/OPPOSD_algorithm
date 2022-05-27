# Pytorch and tools
import torch as th
import numpy as np
import numbers
import pandas as pd
from RunningEnv import EnvWrapper
from RunningEnv import RunningEnv
from RunningEnv import EwmaBiasState
from TransitionBatch import TransitionBatch

class Runner:
    """ Implements a simple single-thread runner class. """

    def __init__(self, controller, params={}, exploration_step=1):
        self.env = EnvWrapper(params.get('pref_pace'), params.get('target_pace'))
        self.cont_actions = False
        self.controller = controller
        self.epi_len = params.get('max_episode_length', 500)
        self.gamma = params.get('gamma', 0.99)
        self.use_pixels = params.get('pixel_observations', False)

        # MANUALLY SET
        self.state_shape = params.get('states_shape')
        # Set up current state and time step
        self.sum_rewards = 0
        self.state = None
        self.time = 0
        self._next_step()

    def close(self):
        """ Closes the underlying environment. Should always when ending an experiment. """
        self.env.close()

    def transition_format(self):
        """ Returns the format of transtions: a dictionary of (shape, dtype) entries for each key. """
        return {'actions': ((1,), th.long),
                'states': (self.state_shape, th.float32),
                'next_states': (self.state_shape, th.float32),
                'rewards': ((1,), th.float32),
                'dones': ((1,), th.bool),
                'returns': ((1,), th.float32)}

    def _wrap_transition(self, s, a, r, ns, d):
        """ Takes a transition and returns a corresponding dictionary. """
        trans = {}
        form = self.transition_format()
        for key, val in [('states', s), ('actions', a), ('rewards', r), ('next_states', ns), ('dones', d)]:
            if not isinstance(val, th.Tensor):
                if isinstance(val, numbers.Number) or isinstance(val, bool): val = [val]
                val = th.tensor(val, dtype=form[key][1])
            if len(val.shape) < len(form[key][0]) + 1: val = val.unsqueeze(dim=0)
            trans[key] = val
        return trans

    def _run_step(self, a):
        """ Make a step in the environment (and update internal bookeeping) """
        ns, r, d = self.env.step(a.item())
        self.sum_rewards += r
        return r, ns, d

    def _next_step(self, done=True, next_state=None):
        """ Switch to the next time-step (and update internal bookeeping) """
        self.time = 0 if done else self.time + 1
        if done:
            self.sum_rewards = 0
            self.state = self.env.reset()
        else:
            self.state = next_state

    def run(self, n_steps, transition_buffer=None, trim=True, return_dict=None):
        """ Runs n_steps in the environment and stores them in the trainsition_buffer (newly created if None).
            If n_steps <= 0, stops at the end of an episode and optionally trims the transition_buffer.
            Returns a dictionary containing the transition_buffer and episode statstics. """
        my_transition_buffer = TransitionBatch(n_steps if n_steps > 0 else self.epi_len, self.transition_format())
        time, episode_start, episode_lengths, episode_rewards = 0, 0, [], []
        max_steps = n_steps if n_steps > 0 else self.epi_len
        for t in range(max_steps):
            # One step in the envionment
            a = self.controller.choose(self.state)
            r, ns, d = self._run_step(a)
            terminal = d and self.time < self.epi_len - 1
            my_transition_buffer.add(self._wrap_transition(self.state, a, r, ns, terminal))
            if t == self.epi_len - 1: d = True
            # Compute discounted returns if episode has ended or max_steps has been reached
            if d or t == (max_steps - 1):
                my_transition_buffer['returns'][t] = my_transition_buffer['rewards'][t]
                for i in range(t - 1, episode_start - 1, -1):
                    my_transition_buffer['returns'][i] = my_transition_buffer['rewards'][i] \
                                                         + self.gamma * my_transition_buffer['returns'][i + 1]
                episode_start = t + 1
            # Remember statistics and advance (potentially initializing a new episode)
            if d:
                episode_lengths.append(self.time + 1)
                episode_rewards.append(self.sum_rewards)
            self._next_step(done=d, next_state=ns)
            time += 1
            # If n_steps <= 0, we return after one episode (trimmed if specified)
            if d and n_steps <= 0:
                my_transition_buffer.trim()
                break
        # Add the sampled transitions to the given transition buffer
        transition_buffer = my_transition_buffer if transition_buffer is None \
            else transition_buffer.add(my_transition_buffer)
        if trim: transition_buffer.trim()
        # Return statistics (mean reward, mean length and environment steps)
        if return_dict is None: return_dict = {}
        return_dict.update({'buffer': transition_buffer,
                            'episode_reward': None if len(episode_rewards) == 0 else np.mean(episode_rewards),
                            'episode_length': None if len(episode_lengths) == 0 else np.mean(episode_lengths),
                            'episodes_amount' : len(episode_rewards),
                            'env_steps': time})
        return return_dict

    def run_episode(self, transition_buffer=None, trim=True, return_dict=None):
        """ Runs one episode in the environemnt.
            Returns a dictionary containing the transition_buffer and episode statstics. """
        return self.run(0, transition_buffer, trim, return_dict)

class Experiments_runner(Runner):
    def __init__(self, controller, params={}, exploration_step=1):
        super().__init__(controller, params, exploration_step)
        self.experiments_transition_buffer = TransitionBatch(int(5000), self.transition_format())
        self.fill_transition_buffer()

    def fill_transition_buffer(self):
        time, episode_start, episode_lengths, episode_rewards = 0, 0, [], []
        max_steps = 5000

        data_folder_path = "~/Documents/THESIS/Project_Juan/"
        exp_batch = [16, 10]
        tests = [['PNPpref', 'PNPfast'], ['CP103', 'IP103', 'CP110', 'IP110']]
        t = 0

        for exp in range(len(exp_batch)):
            for participant in range(1, exp_batch[exp]):
                participant_number = str(participant)
                for i in range(len(tests[exp])):
                    calculated_values = pd.read_csv(
                        data_folder_path + ('calculated_variables/%s_R_%s_calculated.csv') % (
                            tests[exp][i], participant_number))

                    target_pace = calculated_values[calculated_values['pacing_frequency'].notna()][
                        'pacing_frequency'].mean()
                    calculated_values_norm = calculated_values.copy()
                    calculated_values_norm['norm_step_frequency'] = calculated_values_norm[
                                                                        'step_frequency'] / target_pace

                    env = RunningEnv(target_pace)
                    wrapper = EnvWrapper(target_pace, target_pace)
                    state_func = EwmaBiasState()
                    timestep = 0
                    state = 0
                    action = 0
                    reward = 0
                    n_state = 0
                    finish_leap = False
                    skip_steps = 0
                    total_reward = 0
                    # Add first value
                    add_zero = 1

                    for row in calculated_values_norm.to_numpy():

                        if skip_steps > 0:
                            if skip_steps == 1:
                                finish_leap = True
                            skip_steps -= 1
                            timestep += 1
                            continue
                        else:
                            avg_pace = state_func.get_next_state(row[2])
                            n_state = (avg_pace / target_pace) - 1

                            if timestep == 0:
                                state = n_state
                                timestep += 1
                                continue

                            if finish_leap:
                                # print(timestep, state, n_state, action, reward, avg_pace, target_pace)
                                self.experiments_transition_buffer.add(self._wrap_transition(state, action, reward, n_state, 0))
                                t += 1
                                total_reward += reward
                                # Add random small 0's
                                add_zero = np.random.randint(0,10)

                                finish_leap = False

                            if (not pd.isna(row[3]) and add_zero == 0):
                                action = np.random.randint(1, len(wrapper.times))
                                reward = - wrapper.times[action]
                                skip_steps = wrapper.times[action] + 1
                            else:
                                if add_zero > 0:
                                    add_zero -= 1
                                action = 0
                                reward = env.get_distance_reward(target_pace, avg_pace)
                                # print(timestep, state, n_state, action, reward, avg_pace, target_pace)
                                self.experiments_transition_buffer.add(self._wrap_transition(state, action, reward, n_state, 0))
                                t += 1
                                total_reward += reward

                        state = n_state
                        timestep += 1
                    # results[((participant-1)*4)+(i)][j] = total_reward
                    if t >= max_steps:
                        t = max_steps - 1
                    self.experiments_transition_buffer['returns'][t] = self.experiments_transition_buffer['rewards'][t]
                    for i2 in range(t - 1, episode_start - 1, -1):
                        self.experiments_transition_buffer['returns'][i2] = self.experiments_transition_buffer['rewards'][i2] \
                                                              + self.gamma * self.experiments_transition_buffer['returns'][i2 + 1]
                    episode_start = t + 1
                    # print("Runner %s - %s R: %d" % (participant_number, test_name[exp][i], total_reward))

                    episode_lengths.append(self.time + 1)
                    episode_rewards.append(self.sum_rewards)
                    time += 1


    def run(self, n_steps, transition_buffer=None, trim=True, return_dict=None):
        """ Runs n_steps in the environment and stores them in the trainsition_buffer (newly created if None).
            If n_steps <= 0, stops at the end of an episode and optionally trims the transition_buffer.
            Returns a dictionary containing the transition_buffer and episode statstics. """

        self.experiments_transition_buffer.sample()

        # Add the sampled transitions to the given transition buffer
        transition_buffer = self.experiments_transition_buffer if transition_buffer is None \
            else transition_buffer.add(self.experiments_transition_buffer)
        if trim: transition_buffer.trim()
        # Return statistics (mean reward, mean length and environment steps)
        if return_dict is None: return_dict = {}
        return_dict.update({'buffer': transition_buffer})
        return return_dict

class Heuristic_runner(Runner):
    def __init__(self, controller, params={}, exploration_step=1):
        super().__init__(controller, params, exploration_step)
        self.initial_completed = False
        self.initial_wait = 10


    def run(self, n_steps, transition_buffer=None, trim=True, return_dict=None):
        """ Runs n_steps in the environment and stores them in the trainsition_buffer (newly created if None).
            If n_steps <= 0, stops at the end of an episode and optionally trims the transition_buffer.
            Returns a dictionary containing the transition_buffer and episode statstics. """

        my_transition_buffer = TransitionBatch(n_steps if n_steps > 0 else self.epi_len, self.transition_format())
        time, episode_start, episode_lengths, episode_rewards = 0, 0, [], []
        max_steps = n_steps if n_steps > 0 else self.epi_len
        for t in range(max_steps):
            # One step in the envionment with heuristic
            a = self.select_action(self.state)
            r, ns, d = self._run_step(a)
            terminal = d and self.time < self.epi_len - 1
            my_transition_buffer.add(self._wrap_transition(self.state, a, r, ns, terminal))
            if t == self.epi_len - 1: d = True
            # Compute discounted returns if episode has ended or max_steps has been reached
            if d or t == (max_steps - 1):
                my_transition_buffer['returns'][t] = my_transition_buffer['rewards'][t]
                for i in range(t - 1, episode_start - 1, -1):
                    my_transition_buffer['returns'][i] = my_transition_buffer['rewards'][i] \
                                                         + self.gamma * my_transition_buffer['returns'][i + 1]
                episode_start = t + 1
                self.initial_completed = False
                self.initial_wait = 10
            # Remember statistics and advance (potentially initializing a new episode)
            if d:
                episode_lengths.append(self.time + 1)
                episode_rewards.append(self.sum_rewards)
            self._next_step(done=d, next_state=ns)
            time += 1
            # If n_steps <= 0, we return after one episode (trimmed if specified)
            if d and n_steps <= 0:
                my_transition_buffer.trim()
                break

        # Add the sampled transitions to the given transition buffer
        transition_buffer = my_transition_buffer if transition_buffer is None \
            else transition_buffer.add(my_transition_buffer)
        if trim: transition_buffer.trim()
        # Return statistics (mean reward, mean length and environment steps)
        if return_dict is None: return_dict = {}

        return_dict.update({'buffer': transition_buffer,
                            'episode_reward': None if len(episode_rewards) == 0 else np.mean(episode_rewards),
                            'episode_length': None if len(episode_lengths) == 0 else np.mean(episode_lengths),
                            'episodes_amount' : len(episode_rewards),
                            'env_steps': time})
        return return_dict

    def _run_step(self, a):
        """ Make a step in the environment (and update internal bookeeping) """
        ns, r, d = self.env.step(a)
        self.sum_rewards += r
        return r, ns, d

    def select_action(self, state):
        avg_pace = state[0]
        action = 0

        # Initial waiting time
        if not (self.initial_completed):
            self.initial_wait = self.initial_wait - 1
            if self.initial_wait == 0:
                self.initial_completed = True
            return action

        if abs(avg_pace) > 27e-3:
            action = 4

        elif abs(avg_pace) > 22e-3:
            action = 3

        elif abs(avg_pace) > 15e-3:
            action = 2

        elif abs(avg_pace) > 11e-3:
            action = 1

        return action

    def get_probabilities(self, state):
        actions_prob = th.empty(0)
        for row in state:
            if abs(row) > 27e-3:
                actions_prob = th.cat([actions_prob, th.tensor([[0, 0, 0, 0, 1]])], dim=0)

            elif abs(row) > 22e-3:
                actions_prob = th.cat([actions_prob, th.tensor([[0, 0, 0, 1, 0]])], dim=0)

            elif abs(row) > 15e-3:
                actions_prob = th.cat([actions_prob, th.tensor([[0, 0, 1, 0, 0]])], dim=0)

            elif abs(row) > 11e-3:
                actions_prob = th.cat([actions_prob, th.tensor([[0, 1, 0, 0, 0]])], dim=0)

            else:
                actions_prob = th.cat([actions_prob, th.tensor([[1, 0, 0, 0, 0]])], dim=0)

        return actions_prob