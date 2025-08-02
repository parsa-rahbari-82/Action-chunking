"""
Apparently to install box2d-py you need to have swing.
"""

import itertools
import random

import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=False)  # Deterministic


def generate_macro_actions(lengths=(2, 3), primitive_actions=(0, 1, 2, 3)):
    macros = {}
    idx = 0
    for l in lengths:
        for seq in itertools.product(primitive_actions, repeat=l):
            macros[idx] = list(seq)
            idx += 1
    return macros


macro_actions = generate_macro_actions()
num_states = env.observation_space.n
num_macros = len(macro_actions)

q_table = np.zeros((num_states, num_macros))

alpha = 0.1
gamma = 0.99  # discount factor
epsilon = 1.0  # exploration rate
epsilon_decay = 0.995
min_epsilon = 0.01
episodes = 1000

rewards = []

for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        # Epsilon greedy selection
        if random.uniform(0, 1) < epsilon:
            macro_idx = random.choice(list(macro_actions.keys()))
        else:
            macro_idx = np.argmax(q_table[state])

        start_state = state
        macro_seq = macro_actions[macro_idx]
        for action in macro_seq:
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state
            if done:
                break

        # Q-learning update
        old_value = q_table[start_state, macro_idx]
        next_max = np.max(q_table[state])
        new_value = old_value + alpha * (reward + gamma * next_max - old_value)
        q_table[start_state, macro_idx] = new_value

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    rewards.append(total_reward)

    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(rewards)
        print(f"Episode {episode+1}: avg reward = {avg_reward}, epsilon = {epsilon}")
