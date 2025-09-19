# coding: utf-8

import numpy as np
from collections import defaultdict

n_states = 500 # for Taxi-v3
n_actions = 6 # for Taxi-v3

def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    """
    Select states and actions from games that have rewards >= percentile
    :param states_batch: list of lists of states, states_batch[session_i][t]
    :param actions_batch: list of lists of actions, actions_batch[session_i][t]
    :param rewards_batch: list of rewards, rewards_batch[session_i]
    :param percentile: percentile threshold for elite selection

    :returns: elite_states,elite_actions, both 1D lists of states and respective actions from elite sessions

    Please return elite states and actions in their original order
    [i.e. sorted by session number and timestep within session]

    If you are confused, see examples below. Please don't assume that states are integers
    (they will become different later).
    """
    reward_threshold = np.percentile(rewards_batch, percentile)
    elite_states, elite_actions = [], []
    for session_states, session_actions, session_reward in zip(states_batch, actions_batch, rewards_batch):
        if session_reward >= reward_threshold:
            elite_states.extend(session_states)
            elite_actions.extend(session_actions)
    assert elite_states is not None and elite_actions is not None
    return elite_states, elite_actions


def update_policy(elite_states, elite_actions, n_states, n_actions):
    """
    Given old policy and a list of elite states/actions from select_elites,
    return new updated policy where each action probability is proportional to

    policy[s_i,a_i] ~ #[occurences of si and ai in elite states/actions]

    Don't forget to normalize policy to get valid probabilities and handle 0/0 case.
    In case you never visited a state, set probabilities for all actions to 1./n_actions

    :param elite_states: 1D list of states from elite sessions
    :param elite_actions: 1D list of actions from elite sessions
    :param n_states: number of states in the environment
    :param n_actions: number of actions in the environment

    :returns: new_policy: np.array of shape (n_states, n_actions)
    """
    s_a_freq = defaultdict(int)
    s_freq = defaultdict(int)

    for s, a in zip(elite_states, elite_actions):
        s_a_freq[(s, a)] += 1
        s_freq[s] += 1

    new_policy = np.zeros((n_states, n_actions), dtype=np.float64)
    for s in range(n_states):
        if s_freq[s] > 0:
            for a in range(n_actions):
                new_policy[s, a] = s_a_freq[(s, a)] / s_freq[s]
        else:
            new_policy[s, :] = 1.0 / n_actions

    assert new_policy is not None
    return new_policy


def generate_session(env, policy, t_max=int(10**4)):
    """
    Play game until end or for t_max ticks.
    :param env: gym environment
    :param policy: an array of shape [n_states,n_actions] with action probabilities
    :param t_max: maximum number of steps
    :returns: list of states, list of actions and sum of rewards
    """
    states, actions = [], []
    total_reward = 0.

    s, info = env.reset()

    for t in range(t_max):
        # sample action from policy
        a = np.random.choice(n_actions, p=policy[s])
        new_s, r, done, trunc, info = env.step(a)

        # record state, action, reward
        states.append(s)
        actions.append(a)
        total_reward += r

        s = new_s
        if done or trunc:
            break

    assert states and actions
    return states, actions, total_reward
