from builtins import NotImplementedError, staticmethod

from rl.callbacks import TrainIntervalLogger, CallbackList
from rl.core import Agent
import sys
import os
import keras.models
from rl.policy import EpsGreedyQPolicy
import numpy as np
from keras.models import Model
from learning_agents.gdqn import clipped_mse, mean_q
from learning_agents.adqn import ADQNAgent


class AgentPersistenceManager(object):

    def save_agent(self, agent: Agent, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # At the moment, we exclusively consider adqn agents.
        assert isinstance(agent, ADQNAgent)
        return self.save_adqn_agent(folder_path, agent)

    @staticmethod
    def save_adqn_agent(folder_path, adqn_agent: ADQNAgent):
        adqn_agent.model.save(folder_path + '/model.h5f')

    @staticmethod
    def load_adqn_agent(folder_path, memory, action_provider, processor,
                        nb_steps_warmup, target_model_update, policy):
        # load model
        model = keras.models.load_model(filepath=folder_path + '/model.h5f',
                                        custom_objects={'clipped_mse': clipped_mse, 'mean_q': mean_q})
        # TODO save parameters and set them as default.
        adqn_agent = ADQNAgent(model=model, memory=memory, action_provider=action_provider,
                               window_length=1, policy=policy, gamma=.99, batch_size=32,
                               nb_steps_warmup=nb_steps_warmup, train_interval=1, memory_interval=1,
                               target_model_update=target_model_update,
                               delta_range=(-np.inf, np.inf), enable_double_dqn=True, custom_model_objects={},
                               processor=processor)
        return adqn_agent


class Observation(object):

    def serialize(self):
        """
        Serializes the observation to list of floats describing its features.
        We need this for feeding a neural network with features.

        :return:
        """
        raise NotImplementedError


class RemoteAdqn(object):

    def __init__(self, agent: ADQNAgent, training_steps, log_interval, folder_path,
                 agent_persistence_manager=AgentPersistenceManager(), agent_pool_size=1, callbacks=None):

        # Prepare Callbacks
        callbacks = [] if not callbacks else callbacks[:]
        callbacks += [TrainIntervalLogger(interval=log_interval)]
        self.callbacks = CallbackList(callbacks)
        if hasattr(self.callbacks, 'set_model'):
            self.callbacks.set_model(agent)
        else:
            self.callbacks._set_model(agent)
        params = {
            'nb_steps': training_steps,
        }
        if hasattr(self.callbacks, 'set_params'):
            self.callbacks.set_params(params)
        else:
            self.callbacks._set_params(params)

        self.callbacks.on_train_begin()
        self.no_training_steps = training_steps
        self.agent_persistence_manager = agent_persistence_manager

        # Create needed directories if not done yet
        self.folder_path = folder_path
        checkpoint_path = folder_path + '/checkpoints'
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        # Prepare Agent
        self.agent = agent
        self.agent.step = 0
        agent.training = True
        self.agent._on_train_begin()
        # Other parameters
        self.episode_step = 0
        self.episode = 0
        self.episode_reward = 0

        # create agent-pool
        self.agent_pool = [self.agent]
        if agent_pool_size > 1:
            for i in range(agent_pool_size-1):
                aux_agent = ADQNAgent(model=self.agent.model, policy=self.agent.policy,
                                      action_provider=self.agent.action_provider, memory=self.agent.memory,
                                      processor=self.agent.processor, nb_steps_warmup=10, gamma=.99,
                                      delta_range=(-1., 1.), target_model_update=100, train_interval=4,
                                      window_length=self.agent.window_length)
                self.agent_pool.append(aux_agent)

    def train_move(self, observation, reward, done):
        return self.single_agent_move(observation, reward, done, agent=self.agent)

    def test_move(self, observation):
        self.agent.training = False
        return self.agent.forward(observation)

    def save(self):
        self.agent_persistence_manager.save_agent(self.agent, self.folder_path)
        # TODO save config
        # TODO access model params
        # TODO access
        # TODO save metadata of model
        # TODO save a training history

    def single_agent_move(self, observation, reward, done, agent: Agent):
        self.agent.training = True
        if self.episode_step == 0:
            self.callbacks.on_episode_begin(self.episode)
        self.callbacks.on_step_begin(self.episode_step)

        # Is training ended yet?
        if agent.step >= self.no_training_steps:
            self.save()
            # We are done here.
            self.callbacks.on_train_end()
            sys.exit(0)

        # audit latest step
        if reward != -100 and len(agent.recent_observations) > 0:
            metrics = agent.backward(reward=reward, terminal=done)
            step_logs = {
                'action': agent.recent_action,
                'observation': agent.recent_observations[-1],
                'reward': reward,
                'metrics': metrics,
                'episode': self.episode,  # We may count episodes and steps globally, as the agents share a state.
                'info': {},
            }
            self.episode_reward += reward
            self.callbacks.on_step_end(self.episode_step, step_logs)
        # perform next step
        if done:
            # report
            episode_logs = {
                'episode_reward': self.episode_reward,
                'nb_episode_steps': self.episode_step
            }
            self.callbacks.on_episode_end(self.episode, episode_logs)

            self.episode += 1
            self.episode_step = 0
            self.episode_reward = 0

            return

        else:
            self.episode_step += 1
        action = agent.forward(observation)
        agent.step += 1

        return action
