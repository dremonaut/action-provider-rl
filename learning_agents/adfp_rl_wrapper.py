import numpy as np
import rl.agents.dqn
from rl.core import Agent

from learning_agents.adfp import ADFPAgent, Observation


class ADFPWrapperAgent(Agent):

    def __init__(self, adfp_agent:ADFPAgent, processor, goal_params, **kwargs):
        super(ADFPWrapperAgent, self).__init__(**kwargs)
        self.adfp_agent = adfp_agent
        self.compiled = False
        self.processor = processor
        self.goal_params = goal_params

        self.aggregatedReward = 0

    def forward(self, observation):
        augmented_observation = Observation(raw_features=observation, measurements=np.array([self.aggregatedReward]))
        # append measurement
        return self.adfp_agent.forward(observation=augmented_observation, goal_params=self.goal_params)

    def backward(self, reward, terminal):
        self.aggregatedReward += reward
        metrics = self.adfp_agent.backward(measurements=[self.aggregatedReward], terminal=terminal)

        # Reset
        if terminal:
            self.aggregatedReward = 0

        return metrics

    def compile(self, optimizer, metrics=[]):
        self.adfp_agent.compile(optimizer, metrics)
        self.compiled = True

    def load_weights(self, filepath):
        raise NotImplementedError

    def save_weights(self, filepath, overwrite=False):
        raise NotImplementedError

    @property
    def metrics_names(self):
        return self.adfp_agent.metrics_names
