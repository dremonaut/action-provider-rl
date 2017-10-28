import gym
import numpy as np
from learning_agents.adfp_rl_wrapper import ADFPWrapperAgent
from keras.layers import Dense, Flatten, Input, concatenate
from keras.models import Model
from keras.optimizers import Adam
from rl.memory import SequentialMemory

from learning_agents.adfp import ADFPAgent, DecreasingEpsilonGreedyPolicy, DefaultMemory, Goal
from examples.cartpole.cartpole_utils import CartPoleProcessor, CartPoleActionProvider, CartPoleActionProviderV2

ENV_NAME = 'CartPole-v0'
WINDOW_LENGTH = 1

NB_TEMPORAL_OFFSETS = 4


def immediate_reward_function(measurement_diff, goal_params):
    reward = goal_params[0] * measurement_diff[0]
    return reward
goal = Goal(nb_temporal_offsets=NB_TEMPORAL_OFFSETS, immediate_reward_function=immediate_reward_function)

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)

inputs_observation = Input(shape=(5,))
inputs_action = Input(shape=(1,))
inputs_goal = Input(shape=(4, 1,))
flatten_goal = Flatten()(inputs_goal)
# hidden layers
merged = concatenate([inputs_observation, inputs_action, flatten_goal])
hidden_1 = Dense(128, activation='relu')(merged)
hidden_2 = Dense(128, activation='relu')(hidden_1)
hidden_3 = Dense(64, activation='relu')(hidden_2)
# output layer
output = Dense(NB_TEMPORAL_OFFSETS, activation='linear')(hidden_3)

model = Model(inputs=[inputs_observation, inputs_action, inputs_goal], outputs=output)

# Init meta params
memory = SequentialMemory(limit=2000, window_length=WINDOW_LENGTH)
optimizer = Adam(lr=1e-3)
metrics = ['mae']

adfp_agent = ADFPAgent(policy=DecreasingEpsilonGreedyPolicy(start_eps=1, end_eps=0, steps=1000),
                       model=model, action_provider=CartPoleActionProviderV2(),
                       memory=DefaultMemory(max_length=1000), goal=goal,
                       temporal_offsets=[5, 10, 20, 30], default_measurements=[0])
wrapper_agent = ADFPWrapperAgent(adfp_agent=adfp_agent, processor=CartPoleProcessor(),
                                 goal_params=[[0.2], [0.2], [0.5], [1]])
print('[ai_trainer] compiling adfp...');
wrapper_agent.compile(optimizer=optimizer, metrics=metrics)


wrapper_agent.fit(env, nb_steps=50000, visualize=False, verbose=2)





