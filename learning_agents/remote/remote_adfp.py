from builtins import NotImplementedError

from rl.callbacks import TrainIntervalLogger, CallbackList
import sys
import os
from learning_agents.adfp import ADFPAgent, Observation


class Processor(object):

    def process_observation(self, observation):
        raise NotImplementedError

    def process_measurement(self, measurement):
        raise NotImplementedError


class RemoteAdfp(object):

    def __init__(self, agent: ADFPAgent, training_steps, log_interval, folder_path, callbacks=None, mode='train',
                 processor=None):

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

        # Create needed directories if not done yet
        self.folder_path = folder_path
        checkpoint_path = folder_path + '/checkpoints'
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        # Parameters
        self.agent = agent
        self.episode_step = 0
        self.episode = 0
        self.episode_reward = 0
        self.step = 0
        self.recent_action = None
        self.recent_observation = None
        self.mode = mode
        self.processor = processor

    def train_move(self, raw_observation, measurement, goal_params, done):

        self.update()
        self.callbacks.on_step_begin(self.episode_step)

        if self.processor is not None:
            raw_observation = self.processor.process_observation(observation=raw_observation)
            measurement = self.processor.process_measurement(measurement)

        reward = self.agent.goal.immediate_reward_function(measurement, goal_params[-1])

        if self.step > 0:
            metrics = self.agent.backward(measurements=measurement, terminal=done)
            step_logs = {
                'action': self.recent_action,
                'observation': self.recent_observation,
                'reward': reward,
                'metrics': metrics,
                'episode': self.episode,
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
            self.step += 1

        action = self.agent.forward(observation=Observation(raw_features=raw_observation, measurements=measurement),
                                    goal_params=goal_params)

        # Update params for next backprop
        self.recent_observation = raw_observation
        self.recent_action = action
        
        return action

    def test_move(self, raw_observation, measurement, goal_params):

        if self.processor is not None:
            raw_observation = self.processor.process_observation(observation=raw_observation)
            measurement = self.processor.process_measurement(measurement)

        return self.agent.forward(Observation(raw_features=raw_observation, measurements=measurement),
                                  goal_params=goal_params)

    def save(self):
        self.agent.save(self.folder_path)

    def update(self):
        if self.episode_step == 0:
            self.callbacks.on_episode_begin(self.episode)

        # Is training ended yet?
        if self.step >= self.no_training_steps:
            self.save()
            # We are done here.
            self.callbacks.on_train_end()
            sys.exit(0)
