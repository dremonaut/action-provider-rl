import numpy as np
import random
from rl.util import clone_model


class Observation(object):
    """
        A tuple of (raw sensory input, measurements)
    """

    def __init__(self, raw_features, measurements):
        self.raw_features = raw_features
        self.measurements = measurements

    def flatten(self):
        """
        Flattens the tuple to a single feature list.
        :return: flattened observation, i.e., a one-dimensional list of features.
        """
        flattened = np.concatenate([self.raw_features, self.measurements])
        return flattened


class Memory(object):
    """
        Memory of prior experiences.
    """

    def append(self, experience):
        """
        Adds a new experience.
        :param experience: Experience()
        :return: void
        """
        raise NotImplementedError

    def sample(self, batch_size):
        """
        Samples a random experience batch from memory.

        :param batch_size: max number of experiences to sample.
        :return: list of 'Experience'. |list| <= batch_size.
        """
        raise NotImplementedError


class DefaultMemory(Memory):

    def __init__(self, max_length):
        self.max_length = max_length
        self.experiences = []

    def append(self, experience):
        if len(self.experiences) == self.max_length:
            del self.experiences[0]
        self.experiences.append(experience)

    def sample(self, batch_size):
        if len(self.experiences) < batch_size:
            return self.experiences[:]
        samples = random.sample(self.experiences, batch_size)
        return samples


class TrainSample(object):
    """
     Sample for training the neural network.
    """

    def __init__(self, observation, action, goal_params):
        """
        :param observation: array of features as well as array of present measurements
        :param action: array of features describing the action
        :param goal_params: array of goal params
        :param future_measurements: array of measurements.
        """
        self.step_counter = 0
        self.current_offset = 0
        self.observation = observation
        self.action = action
        self.goal_params = goal_params
        self.future_measurements = []


class Policy(object):
    """
    Action selection on the basis of estimated action quality values.
    """

    def select_action(self, expected_action_qualities):
        """
        Selects an action index.
        :param expected_action_qualities: list of quality measures (one value per action).
        :return: the idx of action to choose.
        """
        raise NotImplementedError


class DecreasingEpsilonGreedyPolicy(Policy):

    def __init__(self, steps, start_eps, end_eps):
        self.eps = start_eps
        self.step_count = 0
        self.step_size = (start_eps - end_eps) / steps

    def select_action(self, expected_action_qualities):
        assert expected_action_qualities.ndim == 1
        nb_actions = expected_action_qualities.shape[0]

        if np.random.uniform() < self.eps:
            action = np.random.random_integers(0, nb_actions - 1)
        else:
            action = np.argmax(expected_action_qualities)

        self.eps -= self.step_size

        return action


def process_future_measurement(present_observation, future_measurements):
    future_measurement_diffs = [(future_measurement - present_observation.measurements)
                                for future_measurement in future_measurements]
    return future_measurement_diffs


class ADFPAgent(object):

    def __init__(self, policy, model, action_provider, memory, goal_function, temporal_offsets, batch_size=64,
                 target_model_update=100, default_measurements=None):
        """

        :param policy:
        :param model:
        :param action_provider:
        :param memory:
        :param goal_function:
        :param temporal_offsets: list of ascending ints indicating the temporal offsets.
        :param default_measurements: Used to fill the future_measurements, iff episode ended before. If 'None', the
                                    latest observed measurements before episode end will be used.
        """
        self.policy = policy
        self.model = model # Inputs of models need to be of form [observation, action, goal_params]
        self.action_provider = action_provider
        self.memory = memory
        self.batch_size = batch_size
        self.temporal_offsets = temporal_offsets
        self.goal_function = goal_function
        self.target_model = clone_model(self.model, {})
        self.step = 0
        self.target_model_update = target_model_update
        self.default_measurements = default_measurements

        self.samples = []

        self.current_metrics = []

    def forward(self, observation: Observation, goal_params):
        """
        Chooses an action to execute.
        :param observation:
        :param goal_params:
        :return:
        """

        self.step += 1
        # Select action to execute
        action = self.select_action(observation, goal_params)
        # Init new train sample
        self.samples.append(TrainSample(observation=observation, action=action, goal_params=goal_params))

        return action

    def backward(self, measurements, terminal):
        """
        Processes the results obtained by the previously executed step.
        :param measurements:
        :param future_measurements:
        :param terminal:
        :return:
        """

        metrics = [np.nan for _ in self.metrics_names]

        # Update samples
        for sample in self.samples:
            sample.step_counter += 1

            if sample.step_counter == self.temporal_offsets[sample.current_offset]:
                sample.future_measurements.append(measurements)
                sample.current_offset += 1

            if terminal:
                remaining_offsets = len(self.temporal_offsets) - sample.current_offset
                # fill
                default = measurements if self.default_measurements is None else self.default_measurements
                for i in range(remaining_offsets):
                    sample.future_measurements.append(default)
                sample.current_offset = len(self.temporal_offsets)

            if sample.current_offset == len(self.temporal_offsets):
                self.memory.append(sample)
                self.samples.remove(sample)

        # Train the network on a single stochastic batch.
        experience_samples = self.memory.sample(self.batch_size)
        if len(experience_samples) == 0:
            return metrics
        metrics = self.train_on_batch(experience_samples)

        # Update target model if necessary
        if self.step % self.target_model_update == 0:
            self.target_model.set_weights(self.model.get_weights())

        return metrics

    def select_action(self, observation, current_goal_params):
        # Obtain all possible actions
        action_candidates = self.action_provider.actions(observation.raw_features)
        action_candidates_params = [action.params[0] for action in action_candidates]  # TODO Externalize processing

        #  Construct inputs for actions
        observation_batch = np.array([observation.flatten()] * len(action_candidates))
        action_batch = np.array(action_candidates_params)
        goal_param_batch = np.array([current_goal_params] * len(action_candidates))

        # Use inputs for prediction
        future_measurement_diffs_per_action = self.target_model.predict_on_batch([observation_batch,
                                                                           action_batch, goal_param_batch])
        # TODO verify that order is maintained

        expected_action_qualities = np.array([self.goal_function(self.group_measurements(future_measurement_diffs),
                                                        current_goal_params)
                                     for future_measurement_diffs in future_measurement_diffs_per_action])

        # let policy choose
        chosen_action = action_candidates[self.policy.select_action(expected_action_qualities)]

        return chosen_action

    def train_on_batch(self, experiences):
        # Construct batch
        observation_batch = []
        action_batch = []
        goal_param_batch = []
        future_measurement_diff_batch = []

        for e in experiences:
            observation_batch.append(e.observation.flatten())
            action_batch.append(e.action.params[0])
            goal_param_batch.append(e.goal_params)
            future_measurement_diff_batch.append(
                self.ungroup_measurements(process_future_measurement(e.observation, e.future_measurements)))

        return self.model.train_on_batch([np.array(observation_batch), np.array(action_batch),
                                          np.array(goal_param_batch)], np.array(future_measurement_diff_batch))

    def compile(self, optimizer, metrics=[]):
        self.target_model.compile(optimizer='sgd', loss='mse')
        self.model.compile(optimizer=optimizer, loss="mse", metrics=metrics)

    def group_measurements(self, future_measurement_diffs):
        # [] to [[]*len(self.temporal_offsets)]
        measurement_length = int(len(future_measurement_diffs)/len(self.temporal_offsets))
        transformed_list = []
        idx = 0
        while idx <= len(future_measurement_diffs) - measurement_length:
            transformed_list.append(future_measurement_diffs[idx:idx+measurement_length])
            idx += measurement_length

        return transformed_list

    def ungroup_measurements(self, future_measurement_diffs):
        transformed_list = [m for sublist in future_measurement_diffs for m in sublist]
        return transformed_list

    @property
    def metrics_names(self):
        return self.model.metrics_names[:] # TODO use keras_rl policies + self.policy.metrics_names[:]
