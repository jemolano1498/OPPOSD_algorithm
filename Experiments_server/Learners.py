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
        self.model_actor = model[0]
        self.model_critic = model[1]
        self.model_w = model[2]
        self.controller = controller
        self.value_loss_param = params.get('value_loss_param', 1)
        self.offpolicy_iterations = params.get('offpolicy_iterations', 0)
        self.all_parameters = list(self.model_actor.parameters())
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
        self.model_actor.train(True)
        self.old_pi, loss_sum = None, 0.0
        for _ in range(1 + self.offpolicy_iterations):
            # Compute the model-output for given batch
            out = self.model_actor(batch['states'])  # compute both policy and values
            val = out[:, -1].unsqueeze(dim=-1)  # last entry are the values
            next_val = self.model_actor(batch['next_states'])[:, -1].unsqueeze(dim=-1) if self.compute_next_val else None
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
        self.model_actor = model[0]
        self.model_critic = model[1]
        self.model_w = model[2]

        self.controller = controller
        self.value_loss_param = params.get('value_loss_param', 1)
        self.offpolicy_iterations = params.get('offpolicy_iterations', 10)

        self.all_parameters_actor = list(self.model_actor.parameters())
        self.optimizer_actor = th.optim.Adam(self.all_parameters_actor, lr=params.get('lr', 1E-3))

        self.all_parameters_critic = list(self.model_critic.parameters())
        self.optimizer_critic = th.optim.Adam(self.all_parameters_critic, lr=params.get('lr', 1E-3))

        self.gamma = params.get('gamma')

        self.grad_norm_clip = params.get('grad_norm_clip', 10)
        self.compute_next_val = False  # whether the next state's value is computed
        self.opposd = params.get('opposd', False)
        self.num_actions = params.get('num_actions', 5)
        self.old_pi = th.ones(1, 1) / self.num_actions
        self.pi_0 = None

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

        # model = [model_actor, model_critic, model_w]
        loss_sum = 0.0
        for _ in range(1 + self.offpolicy_iterations):
# HERE START
            if self.opposd:
                for _ in range(50):
                    # batch_w = self.runner.run(self.batch_size, transition_buffer)
                    batch_w = batch.sample(200)
                    self.pi_0 = self.old_pi + 0 * batch_w['returns']
                    # Compute the model-output for given batch
                    pi = th.nn.functional.softmax(self.model_actor(batch_w['states']), dim=-1).gather(dim=-1, index=batch_w['actions'])
                    self.update_policy_distribution(batch_w, pi.detach()/self.pi_0)

            for _ in range(10):
                batch_c = batch.sample(int(5e3))

                val = self.model_critic(batch_c['states'])
                next_val = self.model_critic(batch_c['next_states'])
                pi = th.nn.functional.softmax(self.model_actor(batch_c['states']), dim=-1).gather(dim=-1, index=batch_c['actions'])
                pi.detach()
                self.pi_0 = self.old_pi + 0 * batch_c['returns']

                # value_loss = self._value_loss(batch_c, val, next_val)

                targets = batch_c['returns']

                # loss_fn = th.nn.MSELoss()
                # value_loss = loss_fn(val, targets)

                value_loss = th.mean((pi/self.pi_0) * (targets - val)**2)

                self.optimizer_critic.zero_grad()
                value_loss.backward()
                th.nn.utils.clip_grad_norm_(self.all_parameters_critic, self.grad_norm_clip)
                self.optimizer_critic.step()

            batch_a = batch.sample(int(5e3))

            pi = th.nn.functional.softmax(self.model_actor(batch_a['states']), dim=-1).gather(dim=-1, index=batch_a['actions'])

            val = self.model_critic(batch_a['states'])
            next_val = self.model_critic(batch_a['next_states'])
            self.pi_0 = self.old_pi + 0 * batch_a['returns']

            Q = self._advantages(batch_a, val, next_val)
            ratios = pi / self.pi_0.detach()
            if self.opposd:
                w = self.model_w(batch_a['states']).detach()
                w /= th.mean(w)
                ratios = w * ratios
            policy_loss = -(Q.detach() * ratios).mean()

            self.optimizer_actor.zero_grad()
            policy_loss.backward()
            th.nn.utils.clip_grad_norm_(self.all_parameters_actor, self.grad_norm_clip)
            self.optimizer_actor.step()

            # Combine policy and value loss
            loss = policy_loss.detach().item()

            loss_sum += loss
        return loss_sum

    def get_probabilities(self, state):
        actions_prob = th.empty(0)
        for row in state:
            val = row - 0.1
            if abs(val) > 0.6032526790300863:
                actions_prob = th.cat([actions_prob, th.tensor([[0.01, 0.01, 0.01, 0.01, 0.96]])], dim=0)

            elif abs(val) > 0.5087411165497815:
                actions_prob = th.cat([actions_prob, th.tensor([[0.01, 0.01, 0.01, 0.96, 0.01]])], dim=0)

            elif abs(val) > 0.37642492907735484:
                actions_prob = th.cat([actions_prob, th.tensor([[0.01, 0.01, 0.96, 0.01, 0.01]])], dim=0)

            elif abs(val) > 0.1883564663640196:
                actions_prob = th.cat([actions_prob, th.tensor([[0.01, 0.96, 0.01, 0.01, 0.01]])], dim=0)

            else:
                actions_prob = th.cat([actions_prob, th.tensor([[0.96, 0.01, 0.01, 0.01, 0.01]])], dim=0)

        return actions_prob

class BatchBiasedReinforceLearner(BatchReinforceLearner):
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

class BiasedReinforceLearner(ReinforceLearner):
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
            advantages = batch['rewards']+ self.gamma * (~batch['dones']* next_values)
        else:
            advantages = batch['returns']
        if self.advantage_bias:
            advantages = advantages - values
        return advantages

class BatchActorCriticLearner(BatchBiasedReinforceLearner):
    def __init__(self, model, controller=None, params={}):
        super().__init__(model=model, controller=controller, params=params)
        self.advantage_bootstrap = params.get('advantage_bootstrap', True)
        self.compute_next_val = self.compute_next_val or self.advantage_bootstrap

    def _advantages(self, batch, values=None, next_values=None):
        """ Computes the advantages, Q-values or returns for the policy loss. """
        advantages = None
        if self.advantage_bootstrap:
            advantages = batch['rewards']+ self.gamma * (~batch['dones']* next_values)
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
            # ratios = pi / self.pi_0.detach()
            ratios = pi / self.old_pi.detach()
            return -(advantages.detach() * ratios).mean()

class BatchOffpolicyActorCriticLearner(BatchActorCriticLearner):
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
            # ratios = pi / self.pi_0.detach()
            ratios = pi / self.old_pi.detach()
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
            # ratios = pi / self.pi_0.detach()
            ratios = pi / self.old_pi.detach()
            loss = advantages.detach() * ratios
            if self.ppo_clipping:
                # off-policy loss with PPO clipping
                ppo_loss = th.clamp(ratios, 1 - self.ppo_clip_eps, 1 + self.ppo_clip_eps) * advantages.detach()
                loss = th.min(loss, ppo_loss)
            return -loss.mean()

class BatchPPOLearner(BatchOffpolicyActorCriticLearner):
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
            # ratios = pi / self.pi_0.detach()
            ratios = pi / self.old_pi.detach()
            loss = advantages.detach() * ratios
            if self.ppo_clipping:
                # off-policy loss with PPO clipping
                ppo_loss = th.clamp(ratios, 1 - self.ppo_clip_eps, 1 + self.ppo_clip_eps) * advantages.detach()
                loss = th.min(loss, ppo_loss)
            return -loss.mean()

class OPPOSDLearner(BatchOffpolicyActorCriticLearner):
    def __init__(self, model, controller=None, params={}):
        super().__init__(model=model, controller=controller, params=params)
        self.num_actions = params.get('num_actions', 5)
        self.batch_size = params.get('batch_size')
        self.states_shape = params.get('states_shape')
        self.w_grad_norm_clip = params.get('grad_norm_clip', 10)
        self.parameters_w = list(self.model_w.parameters())
        self.optimizer_w = th.optim.Adam(self.parameters_w, lr=1E-3)

    def _policy_loss(self, pi, advantages):
        # The loss for off-policy data
        loss = advantages.detach() * pi
        return -loss.mean()

    def reset_w_net(self):
        pass
        # self.w_model = th.nn.Sequential(th.nn.Linear(self.states_shape, 128), th.nn.ReLU(),
        #                                 th.nn.Linear(128, 512), th.nn.ReLU(),
        #                                 th.nn.Linear(512, 128), th.nn.ReLU(),
        #                                 th.nn.Linear(128, 1))
        # self.w_parameters = list(self.w_model.parameters())
        # self.w_optimizer = th.optim.Adam(self.w_parameters, lr=params.get('lr', 5E-4))

    def update_policy_distribution(self, batch, ratios):
        self.model_w.train(True)
        batch_size = batch.size

        next_states = batch['next_states']
        with th.autograd.set_detect_anomaly(True):
            w = self.model_w(batch['states'])
            w_ = self.model_w(batch['next_states'])

            w = w / th.mean(w)
            w_ = w_ / th.mean(w_)

            d = w * ratios - w_

            k = th.zeros(batch_size, batch_size, self.states_shape[0])
            for i in range(self.states_shape[0]):
                k[:, :, i] = next_states[:, i].view(1, -1) - next_states[:, i].view(-1, 1)

            k = th.exp(-th.linalg.norm(k, dim=-1)/2)
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

            self.optimizer_w.zero_grad()
            D.backward()
            th.nn.utils.clip_grad_norm_(self.parameters_w, self.w_grad_norm_clip)
            self.optimizer_w.step()


