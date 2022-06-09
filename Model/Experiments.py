import numpy as np
from datetime import datetime
from Controllers import QController, Experiments_controller
from Controllers import EpsilonGreedyController
from Controllers import ACController
from Runner import Runner, Experiments_runner, Heuristic_runner
from Learners import QLearner, BatchReinforceLearner
from Learners import ReinforceLearner
from TransitionBatch import TransitionBatch
# Plotting
from IPython import display
import matplotlib.pyplot as plt
import pylab as pl
import torch as th


class Experiment:
    """ Abstract class of an experiment. Contains logging and plotting functionality."""

    def __init__(self, params, model, **kwargs):
        self.params = params
        self.plot_frequency = params.get('plot_frequency', 10)
        self.plot_train_samples = params.get('plot_train_samples', True)
        self.print_when_plot = params.get('print_when_plot', False)
        self.print_dots = params.get('print_dots', False)
        self.episode_returns = []
        self.episode_lengths = []
        self.episode_losses = []
        self.env_steps = []
        self.total_run_time = 0.0

    def plot_training(self, update=False):
        """ Plots logged training results. Use "update=True" if the plot is continuously updated
            or use "update=False" if this is the final call (otherwise there will be double plotting). """
        # Smooth curves
        window = max(int(len(self.episode_returns) / 3), 1)
        returns = np.convolve(self.episode_returns, np.ones(window) / window, 'valid')
        # Determine x-axis based on samples or episodes
        x_returns = [(i + window) for i in range(len(returns))]
        # Create plot
        colors = ['b', 'g', 'r']
        # fig.set_size_inches(16, 4)
        plt.clf()
        # Plot the losses in the left subplot
        # pl.subplot(1, 3, 1)
        plt.plot(x_returns, returns, colors[0])
        plt.xlabel('Environment steps' if self.plot_train_samples else 'Policy gradient steps')
        plt.ylabel('Episode return')
        display.clear_output(wait=True)
        if update:
            display.display(pl.gcf())

    def close(self):
        """ Frees all allocated runtime ressources, but allows to continue the experiment later.
            Calling the run() method after close must be able to pick up the experiment where it was. """
        pass

    def run(self):
        """ Starts (or continues) the experiment. """
        assert False, "You need to extend the Expeirment class and override the method run(). "

class QLearningExperiment(Experiment):
    """ Experiment that perfoms DQN. You can provide your own learner. """

    def __init__(self, params, model, learner=None, **kwargs):
        super().__init__(params, model, **kwargs)
        self.max_episodes = params.get('max_episodes', int(100))
        self.max_steps = params.get('max_steps', int(100))
        self.run_steps = params.get('run_steps', 0)
        self.grad_repeats = params.get('grad_repeats', 1)
        self.controller = QController(model, num_actions=params.get('num_actions'), params=params)
        self.controller = EpsilonGreedyController(controller=self.controller, params=params)
        self.runner = Runner(self.controller, params=params)
        self.learner = QLearner(model, params=params) if learner is None else learner

    def close(self):
        """ Overrides Experiment.close(). """
        self.runner.close()

    def _learn_from_episode(self, episode):
        """ This function uses the episode to train.
            Although not implemented, one could also add the episode to a replay buffer here.
            Returns the training loss for logging or None if train() was not called. """
        # Call train (params['grad_repeats']) times
        total_loss = 0
        for i in range(self.grad_repeats):
            total_loss += self.learner.train(episode['buffer'])
        return total_loss / self.grad_repeats

    def run(self):
        """ Starts (or continues) the experiment. """
        # Plot previous results if they exist
        if self.plot_frequency is not None and len(self.episode_losses) > 2:
            self.plot_training(update=True)
        # Start (or continue experiment)
        env_steps = 0 if len(self.env_steps) == 0 else self.env_steps[-1]
        for e in range(self.max_episodes):
            begin_time = datetime.now()
            # Run an episode (or parts of it)
            if self.run_steps > 0:
                episode = self.runner.run(n_steps=self.run_steps, trim=False)
            else:
                episode = self.runner.run_episode()
            # Log the results
            env_steps += episode['env_steps']
            if episode['episode_length'] is not None:
                self.episode_lengths.append(episode['episode_length'])
                self.episode_returns.append(episode['episode_reward'])
                self.env_steps.append(env_steps)
            # Make one (or more) learning steps with the episode
            loss = self._learn_from_episode(episode)
            if loss is not None: self.episode_losses.append(loss)
            self.total_run_time += (datetime.now() - begin_time).total_seconds()
            # Quit if maximal number of environment steps is reached
            if env_steps >= self.max_steps:
                break
            # Show intermediate results
            if self.print_dots:
                print('.', end='')
            if self.plot_frequency is not None and (e + 1) % self.plot_frequency == 0 \
                    and len(self.episode_losses) > 2:
                self.plot_training(update=True)
                if self.print_when_plot:
                    print('Update %u, 10-epi-return %.4g +- %.3g, length %u, loss %g, run-time %g sec.' %
                          (len(self.episode_returns), np.mean(self.episode_returns[-100:]),
                           np.std(self.episode_returns[-100:]), np.mean(self.episode_lengths[-100:]),
                           np.mean(self.episode_losses[-100:]), self.total_run_time))

class ActorCriticExperiment(Experiment):
    def __init__(self, params, model, learner=None, **kwargs):
        super().__init__(params, model, **kwargs)
        self.max_episodes = params.get('max_episodes', int(1E6))
        self.max_batch_episodes = params.get('max_batch_episodes', int(1E6))
        self.max_steps = params.get('max_steps', int(1E9))
        self.grad_repeats = params.get('grad_repeats', 1)
        self.batch_size = params.get('batch_size', 1024)
        self.controller = ACController(model, num_actions=params.get('num_actions'), params=params)
        self.controller = EpsilonGreedyController(controller=self.controller, params=params)
        self.runner = Runner(self.controller, params=params)
        self.learner = ReinforceLearner(model, params=params) if learner is None else learner
        self.learner.set_controller(self.controller)

    def close(self):
        """ Overrides Experiment.close() """
        self.runner.close()

    def run(self):
        """ Overrides Experiment.run() """
        # Plot past results if available
        if self.plot_frequency is not None and len(self.episode_losses) > 2:
            self.plot_training(update=True)
        # Run the experiment
        transition_buffer = TransitionBatch(self.batch_size, self.runner.transition_format(), self.batch_size)
        env_steps = 0 if len(self.env_steps) == 0 else self.env_steps[-1]
        interacted_episodes = 0
        for e in range(self.max_batch_episodes):
            # Run the policy for batch_size steps
            batch = self.runner.run(self.batch_size, transition_buffer)
            env_steps += batch['env_steps']
            batch_episodes = 0
            if batch['episode_length'] is not None:
                self.env_steps.append(env_steps)
                self.episode_lengths.append(batch['episode_length'])
                # self.episode_returns.append(np.mean(batch['episode_reward']))
                self.episode_returns.append(batch['episode_reward'])
                batch_episodes = batch['episodes_amount']
                # Make a gradient update step
            loss = self.learner.train(batch['buffer'])
            self.episode_losses.append(loss)
            # Quit if maximal number of environment steps is reached
            if env_steps >= self.max_steps:
                print('Steps limit reached')
                break

            # interacted_episodes += batch_episodes
            # if interacted_episodes >= self.max_episodes:
            #     print('Environment interaction limit reached')
            #     break

            # Show intermediate results
            if self.print_dots:
                print('.', end='')
            if self.plot_frequency is not None and (e + 1) % self.plot_frequency == 0 \
                    and len(self.episode_losses) > 2:
                self.plot_training(update=True)
                if self.print_when_plot:
                    print('Episode %u, 100-epi-return %.4g +- %.3g, length %u, loss %g, positives %d, 0-positives %d' %
                          (len(self.episode_returns), np.mean(self.episode_returns[-100:]),
                           np.std(self.episode_returns[-100:]), np.mean(self.episode_lengths[-100:]),
                           np.mean(self.episode_losses[-100:]), (batch['buffer']['actions'][batch['buffer']['rewards']>0]).sum(),
                           (batch['buffer']['actions'][batch['buffer']['rewards']>0]==0).sum()))

class BatchActorCriticExperiment(Experiment):
    def __init__(self, params, model, learner=None, **kwargs):
        super().__init__(params, model, **kwargs)
        self.models = model
        self.max_episodes = params.get('max_episodes', int(1E6))
        self.max_batch_episodes = params.get('max_batch_episodes', int(1E6))
        self.max_steps = params.get('max_steps', int(1E9))
        self.grad_repeats = params.get('grad_repeats', 1)
        self.batch_size = params.get('batch_size', 1e5)
        self.mini_batch_size = params.get('mini_batch_size', 200)
        self.controller = ACController(model, num_actions=params.get('num_actions'), params=params)
        # self.controller = EpsilonGreedyController(controller=self.controller, params=params)
        # self.runner = Experiments_runner(self.controller, params=params)
        self.runner = Runner(self.controller, params=params)
        self.learner = BatchReinforceLearner(model, params=params) if learner is None else learner
        self.learner.set_controller(self.controller)
        self.opposd = params.get('opposd', False)
        self.opposd_iterations = params.get('opposd_iterations', 50)
        self.experiments_batch = params.get('experiments_batch', False)
        self.data_folder_path = params.get('data_folder_path', "~/Documents/THESIS/Project_Juan/")
        # self.data_folder_path = "/home/nfs/jmolano/THESIS/Project_Juan/"

    def get_transition_batch(self):
        transition_buffer = TransitionBatch(self.batch_size, self.runner.transition_format(), self.mini_batch_size)
        # batch = self.runner.experiments_transition_buffer
        if self.experiments_batch:
            batch = self.runner.fill_transition_buffer(transition_buffer, self.data_folder_path)
        else:
            batch = self.runner.run(self.batch_size, transition_buffer)
        return batch

    def close(self):
        """ Overrides Experiment.close() """
        self.runner.close()

    def run(self, batch=None):
        """ Overrides Experiment.run() """
        # Plot past results if available
        if self.plot_frequency is not None and len(self.episode_losses) > 2:
            self.plot_training(update=True)
        # Run the experiment
        if not batch:
            batch = self.get_transition_batch()
        for e in range(self.max_batch_episodes):
            # Make a gradient update step
            self.learner.train(batch)

            for _ in range(10):
                partial_result = self.test_in_env()
                self.episode_returns.append(partial_result)

            if self.plot_frequency is not None and (e + 1) % self.plot_frequency == 0:
                self.plot_training(update=True)
                if self.print_when_plot:
                    print('Batch %u, epi-return %.4g +- %.3g' %
                          (len(self.episode_returns), np.mean(self.episode_returns[-10:]),
                           np.std(self.episode_returns[-10:])))

    def test_in_env(self):
        rewards = []

        for _ in range(10):
        # for _ in range(self.plot_frequency):
            state = self.runner.env.reset()
            done = False
            score = 0
            while not done:
                action = self.controller.choose(state, increase_counter=False).detach().item()
                # action = 1
                new_state, reward, done = self.runner.env.step(action)
                score += reward
                state = new_state
            rewards.append(score)

        return np.mean(score)

class BatchHeuristicActorCriticExperiment(Experiment):
    def __init__(self, params, model, learner=None, **kwargs):
        super().__init__(params, model, **kwargs)
        self.max_episodes = params.get('max_episodes', int(1E6))
        self.max_batch_episodes = params.get('max_batch_episodes', int(1E6))
        self.max_steps = params.get('max_steps', int(1E9))
        self.grad_repeats = params.get('grad_repeats', 1)
        self.batch_size = params.get('batch_size', 1e5)
        self.mini_batch_size = params.get('mini_batch_size', 200)
        self.controller = ACController(model, num_actions=params.get('num_actions'), params=params)
        self.controller = EpsilonGreedyController(controller=self.controller, params=params)
        self.runner = Heuristic_runner(self.controller, params=params)
        self.learner = BatchReinforceLearner(model, params=params) if learner is None else learner
        self.learner.set_controller(self.controller)
        self.opposd = params.get('opposd', False)
        self.opposd_iterations = params.get('opposd_iterations', 50)

    def get_transition_batch(self):
        transition_buffer = TransitionBatch(self.batch_size, self.runner.transition_format(), self.mini_batch_size)
        batch = self.runner.run(self.batch_size, transition_buffer)
        return batch

    def close(self):
        """ Overrides Experiment.close() """
        self.runner.close()

    def run(self, batch=None):
        """ Overrides Experiment.run() """
        # Plot past results if available
        if self.plot_frequency is not None and len(self.episode_losses) > 2:
            self.plot_training(update=True)
        # Run the experiment
        if not batch:
            batch = self.get_transition_batch()
        for e in range(self.max_batch_episodes):
            # Make a gradient update step
            self.learner.train(batch["buffer"])

            for _ in range(10):
                partial_result = self.test_in_env()
                self.episode_returns.append(partial_result)

            if self.plot_frequency is not None and (e + 1) % self.plot_frequency == 0:
                # self.plot_training(update=True)
                if self.print_when_plot:
                    print('Batch %u, epi-return %.4g +- %.3g' %
                          (len(self.episode_returns), np.mean(self.episode_returns[-10:]),
                           np.std(self.episode_returns[-10:])))

    def test_in_env(self):
        rewards = []

        for _ in range(10):
        # for _ in range(self.plot_frequency):
            state = self.runner.env.reset()
            done = False
            score = 0
            while not done:
                action = self.controller.choose(state, increase_counter=False).detach().item()
                # action = 1
                new_state, reward, done = self.runner.env.step(action)
                score += reward
                state = new_state
            rewards.append(score)

        return np.mean(score)

class ActorCriticExperimentRunning(Experiment):
    def __init__(self, params, model, learner=None, **kwargs):
        super().__init__(params, model, **kwargs)
        self.max_episodes = params.get('max_episodes', int(1E6))
        self.max_batch_episodes = params.get('max_batch_episodes', int(1E6))
        self.max_steps = params.get('max_steps', int(1E9))
        self.grad_repeats = params.get('grad_repeats', 1)
        self.batch_size = params.get('batch_size', 1024)
        self.controller = ACController(model, num_actions=params.get('num_actions'), params=params)
        self.controller = EpsilonGreedyController(controller=self.controller, params=params)
        # self.controller = Experiments_controller(controller=self.controller, params=params)
        self.runner = Experiments_runner(self.controller, params=params)
        self.learner = ReinforceLearner(model, params=params) if learner is None else learner
        self.learner.set_controller(self.controller)
        self.opposd = params.get('opposd', False)

    def close(self):
        """ Overrides Experiment.close() """
        self.runner.close()

    def run(self):
        """ Overrides Experiment.run() """
        # Run the experiment
        transition_buffer = TransitionBatch(self.batch_size, self.runner.transition_format(), self.batch_size)
        env_steps = 0 if len(self.env_steps) == 0 else self.env_steps[-1]
        interacted_episodes = 0
        for e in range(self.max_batch_episodes):
            # Run the policy for batch_size steps
            batch = self.runner.run(self.batch_size, transition_buffer)
            if self.opposd:
                for _ in range(50):
                    batch_w = self.runner.run(self.batch_size, transition_buffer)
                    self.learner.update_policy_distribution(batch_w['buffer'])
            # Make a gradient update step
            loss = self.learner.train(batch['buffer'])
            self.episode_losses.append(loss)
            # Quit if maximal number of environment steps is reached
            if env_steps >= self.max_steps:
                print('Steps limit reached')
                break

            self.episode_returns = np.append(self.episode_returns, np.mean(self.test_in_env()))
            self.plot_training(update=True)
            print('Batch %d %.4g +- %.3g' % (e, np.mean(self.episode_returns[-5:]),
                                             np.std(self.episode_returns[-5:])))

        if self.max_batch_episodes == 1:
            self.episode_returns = self.test_in_env()

    def plot_training(self, update=False):
        """ Plots logged training results. Use "update=True" if the plot is continuously updated
            or use "update=False" if this is the final call (otherwise there will be double plotting). """
        # Smooth curves
        # window = max(int(len(self.episode_returns) / 50), 10)
        window = max(int(len(self.episode_returns) / 20), 2)

        returns = np.convolve(self.episode_returns, np.ones(window) / window, 'valid')

        # Determine x-axis based on samples or episodes
        x_returns = [i + window for i in range(len(returns))]

        # Create plot
        colors = ['b', 'g', 'r']
        fig = plt.gcf()
        fig.set_size_inches(16, 4)
        plt.clf()

        pl.plot(x_returns, returns, colors[0])
        pl.xlabel('environment steps' if self.plot_train_samples else 'batch trainings')
        pl.ylabel('episode return')

        # dynamic plot update
        display.clear_output(wait=True)
        if update:
            display.display(pl.gcf())

    def test_in_env(self):
        rewards = []

        for _ in range(50):
        # for _ in range(self.plot_frequency):
            state = self.runner.env.reset()
            done = False
            score = 0
            while not done:
                action = self.controller.choose(state, increase_counter=False).detach().item()
                # action = 1
                new_state, reward, done = self.runner.env.step(action)
                score += reward
                state = new_state
            rewards.append(score)

        return rewards

class ActorCriticExperimentHeuristic(Experiment):
    def __init__(self, params, model, learner=None, **kwargs):
        super().__init__(params, model, **kwargs)
        self.max_episodes = params.get('max_episodes', int(1E6))
        self.max_batch_episodes = params.get('max_batch_episodes', int(1E6))
        self.max_steps = params.get('max_steps', int(1E9))
        self.grad_repeats = params.get('grad_repeats', 1)
        self.batch_size = params.get('batch_size', 1024)
        self.controller = ACController(model, num_actions=params.get('num_actions'), params=params)
        self.controller = EpsilonGreedyController(controller=self.controller, params=params)
        # self.controller = Experiments_controller(controller=self.controller, params=params)
        self.runner = Heuristic_runner(self.controller, params=params)
        self.learner = ReinforceLearner(model, params=params) if learner is None else learner
        self.learner.set_controller(self.controller)

    def run(self):
        """ Overrides Experiment.run() """
        # Plot past results if available
        if self.plot_frequency is not None and len(self.episode_losses) > 2:
            self.plot_training(update=True)
        # Run the experiment
        transition_buffer = TransitionBatch(self.batch_size, self.runner.transition_format(), self.batch_size)
        env_steps = 0 if len(self.env_steps) == 0 else self.env_steps[-1]
        interacted_episodes = 0
        for e in range(self.max_batch_episodes):
            # Run the policy for batch_size steps
            batch = self.runner.run(self.batch_size, transition_buffer)
            batch_episodes = 0
            loss = self.learner.train(batch['buffer'])
            self.episode_losses.append(loss)

            interacted_episodes += batch_episodes
            if interacted_episodes >= self.max_episodes:
                print('Environment interaction limit reached')
                break

            # Show intermediate results
            self.episode_returns = np.append(self.episode_returns, np.mean(self.test_in_env()))
            self.plot_training(update=True)
            print('Batch %d %.4g +- %.3g' % (e, np.mean(self.episode_returns[-5:]),
                                             np.std(self.episode_returns[-5:])))
        if self.max_batch_episodes == 1:
            self.episode_returns = self.test_in_env()

    def test_in_env(self):
        rewards = []

        for _ in range(50):
        # for _ in range(self.plot_frequency):
            state = self.runner.env.reset()
            done = False
            score = 0
            while not done:
                action = self.controller.choose(state, increase_counter=False).detach().item()
                # action = 1
                new_state, reward, done = self.runner.env.step(action)
                score += reward
                state = new_state
            rewards.append(score)

        return rewards

    def plot_training(self, update=False):
        """ Plots logged training results. Use "update=True" if the plot is continuously updated
            or use "update=False" if this is the final call (otherwise there will be double plotting). """
        # Smooth curves
        window = max(int(len(self.episode_returns) / 3), 1)

        returns = np.convolve(self.episode_returns, np.ones(window) / window, 'valid')

        # Determine x-axis based on samples or episodes
        x_returns = [i + window for i in range(len(returns))]

        # Create plot
        colors = ['b', 'g', 'r']
        fig = plt.gcf()
        fig.set_size_inches(16, 4)
        plt.clf()

        pl.plot(x_returns, returns, colors[0])
        pl.xlabel('environment steps' if self.plot_train_samples else 'batch trainings')
        pl.ylabel('episode return')

        # dynamic plot update
        display.clear_output(wait=True)
        if update:
            display.display(pl.gcf())