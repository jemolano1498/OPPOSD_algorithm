import torch as th


class QLearner:
    """ A basic learner class that performs Q-learning train() steps. """

    def __init__(self, model, params={}):
        self.model = model
        self.all_parameters = list(model.parameters())
        self.gamma = params.get('gamma', 0.99)
        self.optimizer = th.optim.Adam(self.all_parameters, lr=params.get('lr', 5E-4))
        self.criterion = th.nn.MSELoss()
        self.grad_norm_clip = params.get('grad_norm_clip', 10)
        self.target_model = None  # Target models are not yet implemented!

    def target_model_update(self):
        """ This function updates the target network. No target network is implemented yet. """
        pass

    def q_values(self, states, target=False):
        """ Reutrns the Q-values of the given "states".
            I supposed to use the target network if "target=True", but this is not implemented here. """
        return self.model(states)

    def _current_values(self, batch):
        """ Computes the Q-values of the 'states' and 'actions' of the given "batch". """
        qvalues = self.q_values(batch['states'])
        return qvalues.gather(dim=-1, index=batch['actions'])

    def _next_state_values(self, batch):
        """ Computes the Q-values of the 'next_states' of the given "batch".
            Is greedy w.r.t. to current Q-network or target-network, depending on parameters. """
        with th.no_grad():  # Next state values do not need gradients in DQN
            # Compute the next states values (with target or current network)
            qvalues = self.q_values(batch['next_states'], target=True)
            # Compute the maximum over Q-values
            return qvalues.max(dim=-1, keepdim=True)[0]

    def train(self, batch):
        """ Performs one gradient decent step of DQN. """
        self.model.train(True)
        # Compute TD-loss
        targets = batch['rewards'] + self.gamma * (~batch['dones'] * self._next_state_values(batch))
        loss = self.criterion(self._current_values(batch), targets.detach())
        # Backpropagate loss
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.all_parameters, self.grad_norm_clip)
        self.optimizer.step()
        # Update target network (if specified) and return loss
        self.target_model_update()
        return loss.item()

class ReinforceLearner:
    """ A learner that performs a version of REINFORCE. """

    def __init__(self, model, controller=None, params={}):
        self.model = model
        self.controller = controller
        self.value_loss_param = params.get('value_loss_param', 1)
        self.offpolicy_iterations = params.get('offpolicy_iterations', 0)
        self.all_parameters = list(model.parameters())
        self.optimizer = th.optim.Adam(self.all_parameters, lr=params.get('lr', 5E-4))
        self.grad_norm_clip = params.get('grad_norm_clip', 10)
        self.compute_next_val = False  # whether the next state's value is computed
        self.old_pi = None  # this variable can be used for your PPO implementation

    def set_controller(self, controller):
        """ This function is called in the experiment to set the controller. """
        self.controller = controller

    def _advantages(self, batch, values=None, next_values=None):
        """ Computes the advantages, Q-values or returns for the policy loss. """
        return batch['returns']

    def _value_loss(self, batch, values=None, next_values=None):
        """ Computes the value loss (if there is one). """
        return 0

    def _policy_loss(self, pi, advantages):
        """ Computes the policy loss. """
        return -(advantages.detach() * pi.log()).mean()

    def train(self, batch):
        assert self.controller is not None, "Before train() is called, a controller must be specified. "
        self.model.train(True)
        self.old_pi, loss_sum = None, 0.0
        for _ in range(1 + self.offpolicy_iterations):
            # Compute the model-output for given batch
            out = self.model(batch['states'])  # compute both policy and values
            val = out[:, -1].unsqueeze(dim=-1)  # last entry are the values
            next_val = self.model(batch['next_states'])[:, -1].unsqueeze(dim=-1) if self.compute_next_val else None
            pi = self.controller.probabilities(out[:, :-1], precomputed=True).gather(dim=-1, index=batch['actions'])
            # Combine policy and value loss
            loss = self._policy_loss(pi, self._advantages(batch, val, next_val)) \
                   + self.value_loss_param * self._value_loss(batch, val, next_val)
            # Backpropagate loss
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.all_parameters, self.grad_norm_clip)
            self.optimizer.step()
            loss_sum += loss.item()
        return loss_sum

class BatchReinforceLearner:
    """ A learner that performs a version of REINFORCE. """

    def __init__(self, model, controller=None, params={}):
        self.learner = None
        self.model = model
        self.controller = controller
        self.value_loss_param = params.get('value_loss_param', 1)
        self.offpolicy_iterations = params.get('offpolicy_iterations', 10)
        self.all_parameters = list(model.parameters())
        self.optimizer = th.optim.Adam(self.all_parameters, lr=params.get('lr', 5E-4))
        self.grad_norm_clip = params.get('grad_norm_clip', 10)
        self.compute_next_val = False  # whether the next state's value is computed
        self.opposd = params.get('opposd', False)
        self.heuristic = params.get('heuristic', False)
        self.num_actions = params.get('num_actions', 5)
        self.old_pi = th.ones(1, 1) / self.num_actions
        self.pi_0 = None
        self.w_model = None

    def set_controller(self, controller):
        """ This function is called in the experiment to set the controller. """
        self.controller = controller

    def _advantages(self, batch, values=None, next_values=None):
        """ Computes the advantages, Q-values or returns for the policy loss. """
        return batch['returns']

    def _value_loss(self, batch, values=None, next_values=None):
        """ Computes the value loss (if there is one). """
        return 0

    def _policy_loss(self, pi, advantages):
        """ Computes the policy loss. """
        return -(advantages.detach() * pi.log()).mean()

    def update_policy_distribution(self, batch, ratios):
        pass

    def train(self, batch):
        assert self.controller is not None, "Before train() is called, a controller must be specified. "
        self.model.train(True)
        loss_sum = 0.0
        for _ in range(1 + self.offpolicy_iterations):

            if self.opposd:
                for _ in range(50):
                    # batch_w = self.runner.run(self.batch_size, transition_buffer)
                    batch_w = batch.sample(200)
                    self.pi_0 = self.old_pi + 0 * batch_w['returns']
                    if self.heuristic:
                        self.pi_0 = self.get_probabilities(batch_w['states']).gather(dim=-1, index=batch_w['actions']).detach()
                    # Compute the model-output for given batch
                    out = self.model(batch_w['states'])  # compute both policy and values
                    pi = self.controller.probabilities(out[:, :-1], precomputed=True).gather(dim=-1,
                                                                                             index=batch_w['actions'])
                    self.update_policy_distribution(batch_w, pi.detach()/self.pi_0)

            batch_ac = batch.sample(int(5e3))

            out = self.model(batch_ac['states'])  # compute both policy and values
            val = out[:, -1].unsqueeze(dim=-1)  # last entry are the values
            next_val = self.model(batch_ac['next_states'])[:, -1].unsqueeze(
                dim=-1) if self.compute_next_val else None
            pi = self.controller.probabilities(out[:, :-1], precomputed=True).gather(dim=-1,
                                                                                     index=batch_ac['actions'])

            self.pi_0 = self.old_pi + 0 * batch_ac['returns']
            if self.heuristic:
                self.pi_0 = self.get_probabilities(batch_ac['states']).gather(dim=-1, index=batch_ac['actions']).detach()
            if self.opposd:
                w = self.w_model(batch_ac['states']).detach()
                w /= th.mean(w)

                pi = w * pi.detach()/self.pi_0 * pi.log()

            # Combine policy and value loss
            loss = self._policy_loss(pi, self._advantages(batch_ac, val, next_val)) \
                   + self.value_loss_param * self._value_loss(batch_ac, val, next_val)
            # Backpropagate loss
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.all_parameters, self.grad_norm_clip)
            self.optimizer.step()
            loss_sum += loss.item()
        return loss_sum

    def get_probabilities(self, state):
        actions_prob = th.empty(0)
        for row in state:
            if abs(row) > 27e-3:
                actions_prob = th.cat([actions_prob, th.tensor([[0.01, 0.01, 0.01, 0.01, 0.96]])], dim=0)

            elif abs(row) > 22e-3:
                actions_prob = th.cat([actions_prob, th.tensor([[0.01, 0.01, 0.01, 0.96, 0.01]])], dim=0)

            elif abs(row) > 15e-3:
                actions_prob = th.cat([actions_prob, th.tensor([[0.01, 0.01, 0.96, 0.01, 0.01]])], dim=0)

            elif abs(row) > 11e-3:
                actions_prob = th.cat([actions_prob, th.tensor([[0.01, 0.96, 0.01, 0.01, 0.01]])], dim=0)

            else:
                actions_prob = th.cat([actions_prob, th.tensor([[0.96, 0.01, 0.01, 0.01, 0.01]])], dim=0)

        return actions_prob

class BiasedReinforceLearner(BatchReinforceLearner):
    def __init__(self, model, controller=None, params={}):
        super().__init__(model=model, controller=controller, params=params)
        self.value_criterion = th.nn.MSELoss()
        self.advantage_bias = params.get('advantage_bias', True)
        self.value_targets = params.get('value_targets', 'returns')
        self.gamma = params.get('gamma')
        self.compute_next_val = (self.value_targets == 'td')

    def _advantages(self, batch, values=None, next_values=None):
        """ Computes the advantages, Q-values or returns for the policy loss. """
        advantages = batch['returns']
        if self.advantage_bias:
            advantages -= values
        return advantages

    def _value_loss(self, batch, values=None, next_values=None):
        """ Computes the value loss (if there is one). """
        targets = None
        if self.value_targets == 'returns':
            targets = batch['returns']
        elif self.value_targets == 'td':
            targets = batch['rewards'] + self.gamma * (~batch['dones'] * next_values)
        return self.value_criterion(values, targets.detach())

class ActorCriticLearner(BiasedReinforceLearner):
    def __init__(self, model, controller=None, params={}):
        super().__init__(model=model, controller=controller, params=params)
        self.advantage_bootstrap = params.get('advantage_bootstrap', True)
        self.compute_next_val = self.compute_next_val or self.advantage_bootstrap

    def _advantages(self, batch, values=None, next_values=None):
        """ Computes the advantages, Q-values or returns for the policy loss. """
        advantages = None
        if self.advantage_bootstrap:
            advantages = batch['rewards'] + self.gamma * (~batch['dones'] * next_values)
        else:
            advantages = batch['returns']
        if self.advantage_bias:
            advantages = advantages - values
        return advantages

class OffpolicyActorCriticLearner(ActorCriticLearner):
    def __init__(self, model, controller=None, params={}):
        super().__init__(model=model, controller=controller, params=params)

    def _policy_loss(self, pi, advantages):
        """ Computes the policy loss. """
        if self.old_pi is None:
            self.old_pi = pi  # remember on-policy probabilities for off-policy losses
            # Return the defaul on-policy loss
            return super()._policy_loss(pi, advantages)
        else:
            # The loss for off-policy data
            ratios = pi / self.pi_0.detach()
            return -(advantages.detach() * ratios).mean()

class PPOLearner(OffpolicyActorCriticLearner):
    def __init__(self, model, controller=None, params={}):
        super().__init__(model=model, controller=controller, params=params)
        self.ppo_clipping = params.get('ppo_clipping', False)
        self.ppo_clip_eps = params.get('ppo_clip_eps', 0.2)

    def _policy_loss(self, pi, advantages):
        """ Computes the policy loss. """
        if self.old_pi is None:
            # The loss for on-policy data does not change
            return super()._policy_loss(pi, advantages)
        else:
            # The loss for off-policy data
            ratios = pi / self.pi_0.detach()
            loss = advantages.detach() * ratios
            if self.ppo_clipping:
                # off-policy loss with PPO clipping
                ppo_loss = th.clamp(ratios, 1 - self.ppo_clip_eps, 1 + self.ppo_clip_eps) * advantages.detach()
                loss = th.min(loss, ppo_loss)
            return -loss.mean()

class OPPOSDLearner(OffpolicyActorCriticLearner):
    def __init__(self, model, controller=None, params={}):
        super().__init__(model=model, controller=controller, params=params)
        self.num_actions = params.get('num_actions', 5)
        self.batch_size = params.get('batch_size')
        self.states_shape = params.get('states_shape')
        self.w_grad_norm_clip = params.get('grad_norm_clip', 10)
        self.w_model = th.nn.Sequential(th.nn.Linear(self.states_shape[0], 128), th.nn.ReLU(),
                                        th.nn.Linear(128, 512), th.nn.ReLU(),
                                        th.nn.Linear(512, 128), th.nn.ReLU(),
                                        th.nn.Linear(128, 1))
        self.w_parameters = list(self.w_model.parameters())
        self.w_optimizer = th.optim.Adam(self.w_parameters, lr=params.get('lr', 5E-4))

    def _policy_loss(self, pi, advantages):
        # The loss for off-policy data
        loss = advantages.detach() * pi
        return -loss.mean()

    def reset_w_net(self):
        self.w_model = th.nn.Sequential(th.nn.Linear(self.states_shape, 128), th.nn.ReLU(),
                                        th.nn.Linear(128, 512), th.nn.ReLU(),
                                        th.nn.Linear(512, 128), th.nn.ReLU(),
                                        th.nn.Linear(128, 1))
        self.w_parameters = list(self.w_model.parameters())
        self.w_optimizer = th.optim.Adam(self.w_parameters, lr=params.get('lr', 5E-4))

    def update_policy_distribution(self, batch, ratios):
        self.w_model.train(True)
        batch_size = batch.size

        next_states = batch['next_states']
        with th.autograd.set_detect_anomaly(True):
            w = self.w_model(batch['states'])
            w_ = self.w_model(batch['next_states'])

            w = w / th.mean(w)
            w_ = w_ / th.mean(w_)

            d = w * ratios - w_

            k = th.zeros(batch_size, batch_size, self.states_shape[0])
            for i in range(self.states_shape[0]):
                k[:, :, i] = next_states[:, i].view(1, -1) - next_states[:, i].view(-1, 1)

            k = (th.linalg.norm(k, dim=-1) < 1).float()
            prod = th.matmul(d, d.transpose(0, 1))

            # n_lm = 3
            # y = th.randn(n_lm, 4)
            # dist_gt = th.zeros(n_lm, n_lm, 4)
            # dist_gt[:,:,0] = y[:,0].view(1,-1) - y[:,0].view(-1,1)
            # dist_gt[:,:,1] = y[:,1].view(1,-1) - y[:,1].view(-1,1)
            # dist_gt[:,:,2] = y[:,2].view(1,-1) - y[:,2].view(-1,1)
            # dist_gt[:,:,3] = y[:,3].view(1,-1) - y[:,3].view(-1,1)
            # th.linalg.norm(dist_gt, dim=-1)
            # k = (th.linalg.norm(dist_gt, dim=-1)<1).float()

            D = th.sum(prod * k) / batch_size

            self.w_optimizer.zero_grad()
            D.backward()
            self.w_optimizer.step()


