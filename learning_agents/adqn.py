from learning_agents.gdqn import *


class ADQNAgent(GDQNAgent):

    def __init__(self, model, memory, action_provider, window_length=1, policy=EpsGreedyQPolicy(),
                 gamma=.99, batch_size=32, nb_steps_warmup=1000, train_interval=1, memory_interval=1,
                 target_model_update=10000, delta_range=(-np.inf, np.inf), enable_double_dqn=True,
                 custom_model_objects={}, processor=None):

        super(ADQNAgent, self).__init__(model, memory, window_length, policy, gamma, batch_size,
                                        nb_steps_warmup, train_interval, memory_interval, target_model_update,
                                        delta_range, enable_double_dqn, custom_model_objects, processor)

        self.action_provider = action_provider

    def q_values(self, state, serialized_candidates):
        """
        Determines the Q-values for a state and batch of action candidates.
        :param state:
        :param serialized_candidates:
        :return: an array of Q-values.
        """
        state_action_pairs = [np.repeat([state], len(serialized_candidates), axis=0), serialized_candidates]
        q_values = self.model.predict_on_batch(state_action_pairs).flatten()  # TODO Maybe some caching here ?
        return q_values

    @staticmethod
    def serialize_candidates(action_candidates):
        return np.array([action.params for action in action_candidates])

    def select_action(self, state):
        action_candidates = self.action_provider.actions(state)
        q_values = self.q_values(state, self.serialize_candidates(action_candidates))
        action_idx = self.policy.select_action(q_values=q_values)
        return action_candidates[action_idx]

    def fuzzy_max_action(self, state, action_candidates):
        """
        Implements the 1/e stopping rule for maximizing the probability to find the best candidate
        in a secretary problem.

        :param action_candidates:
        :return:
        """
        shuffled_candidates = self.serialize_candidates(action_candidates[:])
        shuffle(shuffled_candidates)
        # reject the first n/e candidates. Remember the best of them.
        no_candidates_to_reject = int(len(shuffled_candidates) / math.e)
        candidates_to_reject = shuffled_candidates[:no_candidates_to_reject]
        q_values = self.q_values(state, candidates_to_reject)
        idx = np.argmax(q_values.flatten())
        q_max_rejected = q_values[idx]
        max_rejected = candidates_to_reject[idx]

        # go on and stop at the best candidate who is better than every preceded candidate.
        remaining_candidates = shuffled_candidates[no_candidates_to_reject:]
        best_candidate = max_rejected
        for action in remaining_candidates:
            state_action_pair = [np.array([state]), np.array([action])]
            q = self.model.predict(state_action_pair).flatten()[0]
            if q > q_max_rejected:
                best_candidate = action
                break

        return best_candidate

    def max_action(self, state, action_candidates):
        serialized_candidates = self.serialize_candidates(action_candidates)
        q_values = self.q_values(state, serialized_candidates)
        idx = np.argmax(q_values.flatten())
        return serialized_candidates[idx]

    def subset_max_action(self, state, action_candidates):
        random = np.random.randint(0, len(action_candidates) - 1, 5)
        filtered_candidates = [action_candidates[i] for i in random]
        return self.max_action(state, filtered_candidates)

    def max_q_batch(self, state_batch):

        if self.enable_double_dqn:
            # According to the paper "Deep Reinforcement Learning with Double Q-learning"
            # (van Hasselt et al., 2015), in Double DQN, the online network predicts the actions
            # while the target network is used to estimate the Q value.
            assert (self.batch_size == len(state_batch))
            # Calculate the optimal action per state using the online network
            action_batch_list = []  # np.empty(shape=(len(state_batch),2)) # [action(s1), action(s2), ..., action(sn)]
            for i in range(len(state_batch)):
                s = state_batch[i]
                action_candidates = self.action_provider.actions(s)
                # action_batch[i] = self.max_action(s, action_candidates)
                action_batch_list.append(self.max_action(s, action_candidates))
            action_batch = np.array(action_batch_list)

            # Calculate the Q-values of the supposed optimal actions using the target network
            state_action_pairs = [state_batch, action_batch]
            target_q_values = self.target_model.predict_on_batch(state_action_pairs)
            return target_q_values.flatten()
        else:
            # TODO
            # Compute the q_values given state1, and extract the maximum for each sample in the batch.
            # We perform this prediction on the target_model instead of the model for reasons
            # outlined in Mnih (2015). In short: it makes the algorithm more stable.
            target_q_values = self.target_model.predict_on_batch(
                state_batch)  # TODO: Find out possible actions and add them to inputs.
            assert target_q_values.shape == (self.batch_size, self.nb_actions)
            return np.max(target_q_values, axis=1).flatten()

    def train_on_batch(self, state0_batch, action_batch, targets):
        serialized_actions = self.serialize_candidates(action_batch)
        return self.model.train_on_batch([np.array(state0_batch), serialized_actions], targets)

    @property
    def metrics_names(self):
        return self.model.metrics_names[:] + self.policy.metrics_names[:]
