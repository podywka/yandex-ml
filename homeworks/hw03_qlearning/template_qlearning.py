import numpy as np
import random
from collections import defaultdict


def my_softmax(values: np.ndarray, T=1.):
    """
    Compute softmax with temperature for numerical stability.
    
    Args:
        values: np.array of shape (n,) - input values
        T: float - temperature parameter (default 1.0)
    
    Returns:
        np.array of shape (n,) - softmax probabilities
    """
    exps = np.exp((values - np.max(values)) / T)
    probas = exps / np.sum(exps)
    return probas


class QLearningAgent:
    def __init__(self, alpha, discount, get_legal_actions, temp=1.):
        """
        Q-Learning Agent
        based on https://inst.eecs.berkeley.edu/~cs188/sp19/projects.html
        Instance variables you have access to
          - self.alpha (learning rate)
          - self.discount (discount rate aka gamma)
          - self.temp (softmax temperature)

        Functions you should use
          - self.get_legal_actions(state) {state, hashable -> list of actions, each is hashable}
            which returns legal actions for a state
          - self.get_qvalue(state,action)
            which returns Q(state,action)
          - self.set_qvalue(state,action,value)
            which sets Q(state,action) := value
        !!!Important!!!
        Note: please avoid using self._qValues directly.
            There's a special self.get_qvalue/set_qvalue for that.
        """

        self.get_legal_actions = get_legal_actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.discount = discount
        self.temp = temp

    def get_qvalue(self, state, action):
        return self._qvalues[state][action]

    def set_qvalue(self, state, action, value):
        self._qvalues[state][action] = value

    def get_value(self, state):
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return 0.0
        value = max(self.get_qvalue(state, a) for a in possible_actions)
        return value

    def update(self, state, action, reward, next_state):
        gamma = self.discount
        learning_rate = self.alpha
        qvalue = (1 - learning_rate) * self.get_qvalue(state, action) + \
                 learning_rate * (reward + gamma * self.get_value(next_state))
        self.set_qvalue(state, action, qvalue)

    def get_best_action(self, state):
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return None
        best_action = max(possible_actions, key=lambda a: self.get_qvalue(state, a))
        return best_action

    def get_softmax_policy(self, state):
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return None
        q_values = np.array([self.get_qvalue(state, a) for a in possible_actions])
        probabilities = my_softmax(q_values, T=self.temp)
        return dict(zip(possible_actions, probabilities))

    def get_action(self, state):
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return None
        action_probs = self.get_softmax_policy(state)
        actions, probs = zip(*action_probs.items())
        chosen_action = np.random.choice(actions, p=probs)
        return chosen_action


class EVSarsaAgent(QLearningAgent):
    def get_value(self, state):
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return 0.0
        action_probs = self.get_softmax_policy(state)
        value = sum(action_probs[a] * self.get_qvalue(state, a) for a in possible_actions)
        return value
