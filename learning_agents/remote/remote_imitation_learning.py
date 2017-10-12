from builtins import NotImplementedError
from remote.thrift.thrift_interface.ai_interface import AgentObserverService
from keras.models import Model
import keras.models
from keras.callbacks import ModelCheckpoint, Callback, History
import random
import numpy as np
import os
import sys
import pickle


class Processor(object):

    def process_observation(self, state):
        raise NotImplementedError

    def process_action(self, action):
        raise NotImplementedError

    def serialize_action(self, action):
        raise NotImplementedError


class Memory(object):

    def append(self, state, action):
        raise NotImplementedError

    def sample(self, batch_size):
        raise NotImplementedError


class DefaultMemory(Memory):

    def __init__(self, max_length):
        self.max_length = max_length
        self.experiences = []

    def append(self, state, action):
        if len(self.experiences) == self.max_length:
            del self.experiences[0]
        self.experiences.append((state, action))

    def sample(self, batch_size):
        raise NotImplementedError


class NotWindowingMemory(DefaultMemory):

    def __init__(self, max_length):
        super(NotWindowingMemory, self).__init__(max_length)

    def sample(self, batch_size):
        if len(self.experiences) <= batch_size:
            return self.experiences[:]
        return random.sample(self.experiences, batch_size)


class WindowingMemory(DefaultMemory):

    def __init__(self, max_length, window_size):
        self.window_size = window_size
        super(WindowingMemory, self).__init__(max_length)

    def sample(self, batch_size):
        # case len(experiences) < window_size
        if len(self.experiences) < self.window_size:
            diff = self.window_size - len(self.experiences)
            sample = self.experiences[:]
            sample.extend([self.experiences[len(self.experiences)-1]] * diff)
            return [sample]

        # else
        indices = list(range(self.window_size, len(self.experiences)))
        samples = random.sample(indices, batch_size)
        windows = []
        for sample_index in samples:
            state = [exp[0] for exp in self.experiences[sample_index-self.window_size:sample_index]]
            flat_state = [val for sublist in state for val in sublist] # state.flatten() #sum(state, [])
            action = self.experiences[sample_index][1]
            windows.append((flat_state, action))

        return windows


class CustomHistory(Callback):

    def __init__(self):
        self.history = History()
        self.history.on_train_begin()

    def on_epoch_end(self, epoch, logs=None):
        self.history.on_epoch_end(epoch=epoch, logs=logs)


def experiences_to_training_data(experiences):
    return [np.array([sample[0] for sample in experiences])], \
           [np.array([np.array(sample[1]) for sample in experiences])]


def memory_generator(memory):
    """
    Generator which continuously provides varying batches of training data sampled from memory.
    :param memory:
    :return:
    """
    while True:
        samples = memory.sample(32)
        states, actions = experiences_to_training_data(samples)
        yield (states, actions)


class ImitationServiceHandler(AgentObserverService.Iface):

    def __init__(self, model: Model, processor, folder_path, epochs, batch_size=32,
                 memory=WindowingMemory(max_length=100000, window_size=5),
                 memory_threshold=10000, no_validation_samples=400, checkpoint_period=10):
        self.model = model
        self.memory = memory
        self.procesor = processor
        self.batch_size = batch_size
        self.memory_threshold = memory_threshold
        self.validation_set = []
        self.no_validation_samples = no_validation_samples
        self.no_epoch = 0
        self.epochs = epochs
        # create the needed directories if it does not already exist
        self.folder_path = folder_path
        checkpoint_path = folder_path + '/checkpoints'
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        self.checkpointer = ModelCheckpoint(filepath= checkpoint_path + '/tmp_model.hdf5', verbose=0, save_best_only=True,
                                            period=checkpoint_period)
        self.agg_history = CustomHistory()

    def update(self, state, action):
        # Build train example and put it in memory
        deserialized_state = self.procesor.process_observation(state)
        deserialized_action = self.procesor.process_action(action)
        self.memory.append(deserialized_state, deserialized_action)

        self.model.fit_generator(generator=memory_generator(self.memory), steps_per_epoch=5000)

    def batchUpdate(self, states, actions):
        for idx, state in enumerate(states):
            self.memory.append(self.procesor.process_observation(state), self.procesor.process_action(actions[idx]))
        # TODO generalize for matching train method as well
        if len(self.memory.experiences) >= self.memory_threshold:
            if len(self.validation_set) == 0:
                # TODO delete samples which are used for validation.
                validation_experiences = self.memory.sample(self.no_validation_samples)
                self.validation_set = experiences_to_training_data(validation_experiences)
                # del(self.memory.experiences[:self.no_validation_samples])
            train_data = self.memory.sample(len(self.memory.experiences)-5) #self.memory.experiences[:]
            states, actions = experiences_to_training_data(train_data)
            self.model.fit(x=states, y=actions, validation_data=self.validation_set, shuffle=True,
                           callbacks=[self.agg_history, self.checkpointer], initial_epoch=self.no_epoch,
                           epochs=self.no_epoch+1)
            self.no_epoch += 1
            if self.no_epoch >= self.epochs:
                # shutdown
                self.save()
                sys.exit(0)

    def imitatedMove(self, state):
        # just ask the model for a prediction
        deserialized_state = self.procesor.process_observation(state)
        action = self.model.predict(deserialized_state)
        serialized_action = self.procesor.serialize_action(action)
        return serialized_action

    def save(self):
        self.model.save(self.folder_path + '/model.hdf5')
        # save history
        with open(self.folder_path + '/trainHistory', 'wb') as file_pi:
            pickle.dump(self.agg_history, file_pi)
        # TODO Save config file containing processor and memory setting, ...


def load_imitation_handler(directory, processor, new_folder_path):
    model = keras.models.load_model(directory + 'model.hdf5')
    # TODO save and load processor, how to handle folder_path? Versioning?
    # TODO other params should be initially set to those of the loaded version and only optionally modyfied.
    return ImitationServiceHandler(model=model, processor=processor, folder_path=new_folder_path)


class ImitationEvaluator(AgentObserverService.Iface):
    """

    """

    def __init__(self, model_to_evaluate:Model, processor:Processor, file_path, no_batches):
        self.model_to_evaluate = model_to_evaluate
        self.processor = processor
        self.history = {}
        for metric in model_to_evaluate.metrics_names:
            self.history[metric] = []
        self.file_path = file_path
        self.no_batches = no_batches
        self.batch_count = 0

    def update(self, state, action):
        raise NotImplementedError

    def batchUpdate(self, states, actions):
        # transform data
        ser_states = []
        ser_actions = []
        for idx, state in enumerate(states):
            ser_states.append(self.processor.process_observation(state))
            ser_actions.append(self.processor.process_action(actions[idx]))

        # test the model
        metrics = self.model_to_evaluate.metrics_names
        result = self.model_to_evaluate.test_on_batch(x=[np.array(ser_states)], y=[np.array(ser_actions)])

        eval_str = ''
        for idx, metric in enumerate(metrics):
            eval_str += metric + ': ' + str(result[idx]) + ' '
            self.history[metric].append(result[idx])
        print(eval_str)

        self.batch_count += 1

        # process results
        if self.batch_count >= self.no_batches:
            self.save()
            avg = np.average(self.history['mean_absolute_error'])
            std = np.std(self.history['mean_absolute_error'])
            print('avg_mae: ' + str(avg) + ' ' + 'std_mae: ' + str(std))
            sys.exit(0)

    def save(self):
        with open(self.file_path, 'wb') as file_pi:
            pickle.dump(self.history, file_pi)