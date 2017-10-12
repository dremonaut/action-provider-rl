from __future__ import division

from _ast import Not
from collections import deque
import numpy as np
import keras.backend as K
from rl.core import Agent
from rl.policy import EpsGreedyQPolicy
from rl.util import *
import timeit
import math
from random import shuffle
import copyreg
import types


def mean_q(y_true, y_pred):
    # TODO: Refer to that method of the dqn.py module.
    return K.mean(K.max(y_pred, axis=-1))


class GDQNAgent(Agent):
    """
        A generalized version of the DQNAgent using an action provider.
    """

    def __init__(self, model, memory, window_length=1, policy=EpsGreedyQPolicy(),
                 gamma=.99, batch_size=32, nb_steps_warmup=1000, train_interval=1, memory_interval=1,
                 target_model_update=1000, delta_range=(-np.inf, np.inf), enable_double_dqn=True,
                 custom_model_objects={}, processor=None, **kwargs):
        """
        :param model:
            The (keras-) model is asserted to take inputs of the following form:
                [[inputs concerning the observation],[inputs concerning the actions]]
             It is asserted to have exactly one output neuron.
        :param memory:
        :param window_length:
        :param policy:
        :param gamma:
        :param batch_size:
        :param nb_steps_warmup:
        :param train_interval: Interval of steps, in which the neural net is trained with a mini-batch.
        :param memory_interval:
        :param target_model_update:
        :param delta_range:
        :param enable_double_dqn:
        :param custom_model_objects:
        :param processor:
        """

        if hasattr(model.output, '__len__') and len(model.output) > 1:
            raise ValueError(
                'Model "{}" has more than one output. GDQN expects a model that has a single output.'.format(model))
        if not hasattr(model.input, '__len__') or len(model.input) != 2:
            raise ValueError(
                'Model "{}" does not have enough inputs. The model must have at exactly two inputs, one for the action '
                'and one for the observation.'.format(
                    model))

        super(GDQNAgent, self).__init__(**kwargs)

        # Soft vs hard target model updates.
        if target_model_update < 0:
            raise ValueError('`target_model_update` must be >= 0.')
        elif target_model_update >= 1:
            # Hard update every `target_model_update` steps.
            target_model_update = int(target_model_update)
        else:
            # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
            target_model_update = float(target_model_update)

        # Parameters.
        self.window_length = window_length
        self.gamma = gamma
        self.batch_size = batch_size
        self.nb_steps_warmup = nb_steps_warmup
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.target_model_update = target_model_update
        self.delta_range = delta_range
        self.enable_double_dqn = enable_double_dqn
        self.custom_model_objects = custom_model_objects

        # Related objects.
        self.model = model
        self.recent_action = None
        self.recent_observations = None
        self.memory = memory
        self.policy = policy
        self.policy._set_agent(self)
        self.processor = processor

        # State.
        self.compiled = False
        self.reset_states()

    def select_action(self, state):
        """"
        Selects an action to execute.
         :param state: a list of recent observations - say an n-dimensional array
         :return an action
        """
        raise NotImplementedError

    def max_q_batch(self, state_batch):
        """
        An array of maximum qs for the states in state_batch.
        :param state_batch:
        :return:
        """
        raise NotImplementedError

    def train_on_batch(self, state0_batch, action_batch, targets):
        """
        :param state0_batch:
        :param action_batch:
        :param targets:
        :return:
        """
        raise NotImplementedError

    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]  # register default metrics

        # We never train the target model, hence we can set the optimizer and loss arbitrarily.
        self.target_model = clone_model(self.model, self.custom_model_objects)
        self.target_model.compile(optimizer='sgd', loss='mse')

        # Compile model.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            updates = get_soft_target_model_updates(self.target_model, self.model, self.target_model_update)
            optimizer = AdditionalUpdatesOptimizer(optimizer, updates)

        self.model.compile(optimizer=optimizer, loss=clipped_mse, metrics=metrics)

        self.compiled = True

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
        self.update_target_model_hard()

    def save_weights(self, filepath, overwrite=False):
        self.model.save_weights(filepath, overwrite=overwrite)

    def reset_states(self):
        self.recent_action = None
        self.recent_observations = deque(maxlen=self.window_length)

    def update_target_model_hard(self):
        self.target_model.set_weights(self.model.get_weights())

    def process_state_batch(self, batch):
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def forward(self, observation):
        """
        Chooses an action to execute in response to a given observation.
        :param observation: source observation.
        :return: action to execute.
        """

        if self.processor is not None:
            observation = self.processor.process_observation(observation)

        # Select an action.
        while len(self.recent_observations) < self.recent_observations.maxlen:
            # Not enough data, fill the recent_observations queue with copies of the current input.
            # This allows us to immediately perform a policy action instead of falling back to random
            # actions.
            self.recent_observations.append(np.copy(observation))
        state = np.array(list(self.recent_observations)[1:] + [observation])
        assert len(state) == self.window_length
        action = self.select_action(state)

        # Book-keeping.
        self.recent_observations.append(observation)
        self.recent_action = action

        return action

    def backward(self, reward, terminal):
        """

        :param reward:
        :param terminal:
        :return:
        """

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        if self.processor is not None:
            reward = self.processor.process_reward(reward)

        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            # TODO modify memory to cope with plain observations.
            self.memory.append(self.recent_observations[-1], self.recent_action, reward, terminal)

        # Train the network on a single stochastic batch.
        if self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:
            experiences = self.memory.sample(self.batch_size)
            assert len(experiences) == self.batch_size

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            state1_batch = []
            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)

            # Prepare and validate parameters.
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)
            assert reward_batch.shape == (self.batch_size,)
            assert terminal_batch.shape == reward_batch.shape
            assert len(action_batch) == len(reward_batch)

            # Compute Q values for mini-batch update.
            q_batch = self.max_q_batch(state1_batch)
            assert q_batch.shape == (self.batch_size,)

            discounted_reward_batch = self.gamma * q_batch
            discounted_reward_batch *= terminal_batch

            assert discounted_reward_batch.shape == reward_batch.shape
            targets = reward_batch + discounted_reward_batch

            # Finally, perform a single update on the entire batch.
            metrics = self.train_on_batch(state0_batch, action_batch, targets)

            # TODO: Generalize this to metrics of the ActionProvider.
            # TODO: Generalize for different model kinds.
            metrics += self.policy.metrics

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_model_hard()

        return metrics

    def add_experience(self, experience):
        raise NotImplementedError


def clipped_mse(y_true, y_pred):
    delta = K.clip(y_true - y_pred, -np.inf, np.inf)
    return K.mean(K.square(delta), axis=-1)
