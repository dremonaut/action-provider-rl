import gym
import numpy as np
from keras.layers import Dense, Activation, Input, concatenate
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.core import Processor
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy

from ap.action_provider import Action, ActionProvider
from learning_agents.adqn import ADQNAgent

"""

 Used to assess the performance of our ADQN algorithm on the (in fact rather simple) cart pole problem.

"""


ENV_NAME = 'FrozenLake8x8-v0'
WINDOW_LENGTH = 1


class FrozenLakeActionProvider(ActionProvider):
    def actions(self, state):
        return [Action([0]), Action([1]), Action([2]), Action([3])]


class FrozenLakeProcessor(Processor):
    def process_action(self, action):
        return action.params[0]


def init_dqn():
    model = Sequential()
    model.add(Input(input_shape=(1,)))
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
    inputs_state = Input(shape=(1,))
    #flatten_state = Flatten()(inputs_state)
    inputs_action = Input(shape=(1,))
    merged = concatenate([inputs_state, inputs_action])
    hidden_1 = Dense(32, activation='relu')(merged)
    hidden_2 = Dense(16, activation='relu')(hidden_1)
    hidden_3 = Dense(16, activation='relu')(hidden_2)
    output = Dense(1, activation='linear')(hidden_3)
    model = Model(inputs=[inputs_state, inputs_action], outputs=output)

    adqn_agent = ADQNAgent(model=model, memory=memory, action_provider=FrozenLakeActionProvider(),
                           processor=FrozenLakeProcessor(), policy=policy,
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
agent = init_adqn()

# It's time to learn something.
agent.fit(env, nb_steps=50000, visualize=False, verbose=2)





