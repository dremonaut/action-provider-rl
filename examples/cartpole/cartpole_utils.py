from rl.core import Processor

from ap.action_provider import ActionProvider, Action


class CartPoleActionProvider(ActionProvider):
    def actions(self, state):
        return [Action([0]), Action([1])]


class CartPoleActionProviderV2(ActionProvider):
    """
    do not drive in the direction the pole is falling.
    """
    def actions(self, observation):
        state = observation
        pole_angle = state[2]
        if pole_angle < -0.0872:
            return[Action([1])]
        if pole_angle > 0.0872:
            return[Action([0])]
        return [Action([0]), Action([1])]


class CartPoleProcessor(Processor):
    def process_action(self, action):
        return action.params[0]
