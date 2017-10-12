class Action(object):
    def __init__(self, params):
        self.params = params


class ActionProvider(object):

    def actions(self, state):
        """
        Determines a candidate set of executable actions for a given state.
        :param state:
        :return: list of actions
        """
        raise NotImplementedError