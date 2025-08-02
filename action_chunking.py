"""
Apparently to install box2d-py you need to have swing.
"""

import itertools
import random

import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")  # Deterministic


def generate_macro_actions(lengths=(2, 3), primitive_actions=(0, 1, 2, 3)):
    # Not Practical for large maps/envs
    macros = {}
    idx = 0
    for l in lengths:
        for seq in itertools.product(primitive_actions, repeat=l):
            macros[idx] = list(seq)
            idx += 1
    return macros


def execute_macro(env, macro, gamma):
    discounted_reward = 0.0
    undiscounted_reward = 0.0
    discount = 1.0
    steps_taken = 0

    for action in macro:
        next_state, reward, done, truncated, _ = env.step(action)

        discounted_reward += discount * reward
        undiscounted_reward += reward
        discount *= gamma
        steps_taken += 1

        if done or truncated:
            return next_state, discounted_reward, undiscounted_reward, True, steps_taken

    return next_state, discounted_reward, undiscounted_reward, False, steps_taken


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
        next_state, disc_reward, undis_reward, done, steps_taken = execute_macro(
            env, macro_seq, gamma
        )
        total_reward += undis_reward

        # Q-learning update
        old_value = q_table[start_state, macro_idx]
        next_max = np.max(q_table[next_state])
        temporal_difference_target = disc_reward + (gamma**steps_taken) * next_max
        new_value = old_value + alpha * (temporal_difference_target - old_value)
        q_table[start_state, macro_idx] = new_value

        state = next_state

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    rewards.append(total_reward)

    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(rewards[-100:])  # last 100 episodes
        print(f"Episode {episode+1}: avg reward = {avg_reward}, epsilon = {epsilon}")

env.close()


def test_policy(q_table, macro_actions, gamma, test_episodes=5):
    print("Testing...")
    test_env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
    for _ in range(test_episodes):
        state, _ = test_env.reset()
        done = False
        test_env.render()

        while not done:
            macro_idx = np.argmax(q_table[state])
            macro_sequence = macro_actions[macro_idx]

            next_state, _, _, done, _ = execute_macro(test_env, macro_sequence, gamma)

            state = next_state
            test_env.render()

    test_env.close()


test_policy(q_table, macro_actions, gamma)
