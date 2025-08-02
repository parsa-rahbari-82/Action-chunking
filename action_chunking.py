"""
Apparently to install box2d-py you need to have swing.
"""

import numpy as np
import gymnasium as gym
import random

env = gym.make("FrozenLake-v1", is_slippery=False)  # Deterministic

q_table = np.zeros((env.observation_space.n, env.action_space.n))

alpha = 0.1
gamma = 0.99       # discount factor
epsilon = 1.0      # exploration rate
epsilon_decay = 0.995
min_epsilon = 0.01
episodes = 1000

for episode in range(episodes):
    state, _ = env.reset()
    done = False

    while not done:
        # Epsilon greedy selection
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, truncated, _ = env.step(action)

        # Q-learning update
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        new_value = old_value + alpha * (reward + gamma * next_max - old_value)
        q_table[state, action] = new_value

        state = next_state

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

print("Test")
