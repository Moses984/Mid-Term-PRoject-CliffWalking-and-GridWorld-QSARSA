# custom_gridworld_q_sarsa_separate.py
"""
Custom GridWorld environment + separate Q-Learning, SARSA, and comparison plots.
Author: [Your Name]
For: Midterm Project - Task 2
"""

import gymnasium as gym
from gymnasium import Env, spaces
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import random
import os
import pickle

# --------------------- Custom GridWorld Env ---------------------
class CustomGridWorld(Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, rows=5, cols=5, traps=None, trap_penalty=-100, max_steps=None):
        self.rows = rows
        self.cols = cols
        self.traps = traps if traps else []
        self.trap_penalty = trap_penalty
        self.start = (rows - 1, 0)
        self.goal = (0, cols - 1)
        self.max_steps = max_steps if max_steps else rows * cols * 4

        self.observation_space = spaces.Discrete(rows * cols)
        self.action_space = spaces.Discrete(4)
        self._to_index = lambda r, c: r * self.cols + c
        self._to_pos = lambda idx: (idx // self.cols, idx % self.cols)
        self.state = None
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self._to_index(*self.start)
        self.step_count = 0
        return int(self.state), {}

    def step(self, action):
        r, c = self._to_pos(int(self.state))
        nr, nc = r, c
        if action == 0: nr = max(0, r - 1)       # up
        elif action == 1: nc = min(self.cols - 1, c + 1)  # right
        elif action == 2: nr = min(self.rows - 1, r + 1)  # down
        elif action == 3: nc = max(0, c - 1)     # left
        next_state = self._to_index(nr, nc)
        reward = -1
        done = False

        if (nr, nc) in self.traps:
            reward = self.trap_penalty
            done = True
        elif (nr, nc) == self.goal:
            reward = 0
            done = True

        self.state = int(next_state)
        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True
        return int(self.state), float(reward), bool(done), False, {}

    def render(self, mode="human"):
        grid = [[" " for _ in range(self.cols)] for _ in range(self.rows)]
        for (tr, tc) in self.traps:
            grid[tr][tc] = "X"
        gr, gc = self.goal; grid[gr][gc] = "G"
        sr, sc = self.start; grid[sr][sc] = "S"
        cr, cc = self._to_pos(self.state)
        if (cr, cc) not in [self.goal, self.start] and (cr, cc) not in self.traps:
            grid[cr][cc] = "A"
        for row in grid[::-1]:
            print(" | ".join(row))
        print("-" * (self.cols * 4))

# --------------------- Utilities ---------------------
def epsilon_greedy(q_table, state, n_actions, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    return int(np.argmax(q_table[state]))

def smooth(x, window=10):
    if window <= 1:
        return np.array(x)
    return np.convolve(x, np.ones(window)/window, mode="valid")

def evaluate_policy(env, q_table, episodes=100, max_steps=1000):
    total_rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        for _ in range(max_steps):
            action = int(np.argmax(q_table[state]))
            state, reward, done, _, _ = env.step(action)
            ep_reward += reward
            if done: break
        total_rewards.append(ep_reward)
    return np.mean(total_rewards), np.std(total_rewards)

# --------------------- Algorithms ---------------------
def q_learning(env, episodes=1000, alpha=0.5, gamma=1.0, epsilon=0.1, epsilon_decay=0.9995, min_epsilon=0.01):
    n_states, n_actions = env.observation_space.n, env.action_space.n
    q_table = np.zeros((n_states, n_actions))
    rewards = []
    eps = epsilon
    for ep in trange(episodes, desc="Q-Learning"):
        state, _ = env.reset()
        done, ep_reward = False, 0
        while not done:
            a = epsilon_greedy(q_table, state, n_actions, eps)
            next_state, r, done, _, _ = env.step(a)
            q_table[state, a] += alpha * (r + gamma * np.max(q_table[next_state]) - q_table[state, a])
            state = next_state
            ep_reward += r
        rewards.append(ep_reward)
        eps = max(min_epsilon, eps * epsilon_decay)
    return q_table, rewards

def sarsa(env, episodes=1000, alpha=0.5, gamma=1.0, epsilon=0.1, epsilon_decay=0.9995, min_epsilon=0.01):
    n_states, n_actions = env.observation_space.n, env.action_space.n
    q_table = np.zeros((n_states, n_actions))
    rewards = []
    eps = epsilon
    for ep in trange(episodes, desc="SARSA"):
        state, _ = env.reset()
        a = epsilon_greedy(q_table, state, n_actions, eps)
        done, ep_reward = False, 0
        while not done:
            next_state, r, done, _, _ = env.step(a)
            next_a = epsilon_greedy(q_table, next_state, n_actions, eps)
            q_table[state, a] += alpha * (r + gamma * q_table[next_state, next_a] - q_table[state, a])
            state, a = next_state, next_a
            ep_reward += r
        rewards.append(ep_reward)
        eps = max(min_epsilon, eps * epsilon_decay)
    return q_table, rewards

# --------------------- Plotting helpers ---------------------
def plot_rewards_single(rewards, algo_name, color, outdir="results", window=20):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(8,5))
    plt.plot(smooth(rewards, window=window), color=color)
    plt.xlabel("Episode (smoothed)")
    plt.ylabel("Total reward")
    plt.title(f"{algo_name} Learning Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{algo_name.lower()}_learning_curve.png"), dpi=200)
    plt.show()

def plot_rewards_comparison(rewards_q, rewards_sarsa, outdir="results", window=20):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(9,6))
    plt.plot(smooth(rewards_q, window=window), label="Q-Learning")
    plt.plot(smooth(rewards_sarsa, window=window), label="SARSA")
    plt.xlabel("Episode (smoothed)")
    plt.ylabel("Total reward")
    plt.title("Q-Learning vs SARSA – Custom GridWorld")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "comparison_learning_curves.png"), dpi=200)
    plt.show()

# --------------------- Main Experiment ---------------------
def main():
    env = CustomGridWorld(rows=5, cols=5, traps=[(4,2), (2,3)], trap_penalty=-100)
    episodes = 1000
    alpha, gamma = 0.5, 1.0
    epsilon, epsilon_decay, min_epsilon = 0.1, 0.9995, 0.01

    # Train Q-Learning
    q_q, rewards_q = q_learning(env, episodes, alpha, gamma, epsilon, epsilon_decay, min_epsilon)
    mean_q, std_q = evaluate_policy(env, q_q)
    print(f"Q-Learning greedy eval: mean={mean_q:.2f}, std={std_q:.2f}")
    plot_rewards_single(rewards_q, "Q-Learning", "blue")

    # Train SARSA
    q_s, rewards_s = sarsa(env, episodes, alpha, gamma, epsilon, epsilon_decay, min_epsilon)
    mean_s, std_s = evaluate_policy(env, q_s)
    print(f"SARSA greedy eval: mean={mean_s:.2f}, std={std_s:.2f}")
    plot_rewards_single(rewards_s, "SARSA", "green")

    # Comparison Plot
    plot_rewards_comparison(rewards_q, rewards_s)

if __name__ == "__main__":
    main()
