import gym
import numpy as np
from keras.layers import Dense, Activation, Flatten, Input, concatenate
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy

from learning_agents.adqn import ADQNAgent
from examples.cartpole.cartpole_utils import CartPoleActionProvider, CartPoleProcessor

"""
 Used to assess the performance of our ADQN algorithm on the (in fact rather simple) cart pole problem.
"""


ENV_NAME = 'CartPole-v0'
WINDOW_LENGTH = 1


def init_dqn():
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=warmup,
                   target_model_update=target_model_update, policy=policy)
    dqn.compile(optimizer, metrics=metrics)
    return dqn


def init_adqn():
    inputs_state = Input(shape=(WINDOW_LENGTH,) + env.observation_space.shape)
    flatten_state = Flatten()(inputs_state)
    inputs_action = Input(shape=(1,))
    merged = concatenate([flatten_state, inputs_action])
    hidden_1 = Dense(32, activation='relu')(merged)
    hidden_2 = Dense(16, activation='relu')(hidden_1)
    hidden_3 = Dense(16, activation='relu')(hidden_2)
    output = Dense(1, activation='linear')(hidden_3)
    model = Model(inputs=[inputs_state, inputs_action], outputs=output)

    adqn_agent = ADQNAgent(model=model, memory=memory, action_provider=CartPoleActionProvider(),
                           processor=CartPoleProcessor(), policy=policy,
                           nb_steps_warmup=warmup, target_model_update=target_model_update,
                           enable_double_dqn=True)

    print('[ai_trainer] compiling dqn...');
    adqn_agent.compile(optimizer, metrics=metrics)
    return adqn_agent


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n
print('nb_actions: ' + str(nb_actions))

# Init meta params
memory = SequentialMemory(limit=50000, window_length=WINDOW_LENGTH)
policy = BoltzmannQPolicy()
optimizer = Adam(lr=1e-3)
metrics = ['mae']
warmup = 10
target_model_update = 1e-2

# Load the agent who should solve the task.
agent = init_dqn()

# It's time to learn something.
agent.fit(env, nb_steps=5000, visualize=False, verbose=2)





