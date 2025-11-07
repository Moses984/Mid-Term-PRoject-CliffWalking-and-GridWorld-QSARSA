# cliff_q_sarsa.py
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import random
import os
import pickle

# ---------- Utilities ----------
def epsilon_greedy(q_table, state, n_actions, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    return int(np.argmax(q_table[state]))

def evaluate_policy(env, q_table, episodes=100, max_steps=1000):
    """Run greedy policy (no exploration) and return average cumulative reward."""
    total_rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        steps = 0
        while not done and steps < max_steps:
            action = int(np.argmax(q_table[state]))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            state = next_state
            steps += 1
        total_rewards.append(ep_reward)
    return np.mean(total_rewards), np.std(total_rewards)

# ---------- Q-Learning ----------
def q_learning(env, episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1, epsilon_decay=0.999, min_epsilon=0.01):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.zeros((n_states, n_actions), dtype=float)
    rewards_per_episode = []
    for ep in trange(episodes, desc="Q-learning"):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action = epsilon_greedy(q_table, state, n_actions, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            best_next_action = int(np.argmax(q_table[next_state]))
            td_target = reward + gamma * q_table[next_state, best_next_action]
            td_delta = td_target - q_table[state, action]
            q_table[state, action] += alpha * td_delta

            state = next_state
            ep_reward += reward

        rewards_per_episode.append(ep_reward)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    return q_table, rewards_per_episode

# ---------- SARSA ----------
def sarsa(env, episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1, epsilon_decay=0.999, min_epsilon=0.01):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.zeros((n_states, n_actions), dtype=float)
    rewards_per_episode = []
    for ep in trange(episodes, desc="SARSA"):
        state, _ = env.reset()
        action = epsilon_greedy(q_table, state, n_actions, epsilon)
        done = False
        ep_reward = 0
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_action = epsilon_greedy(q_table, next_state, n_actions, epsilon)

            td_target = reward + gamma * q_table[next_state, next_action]
            td_delta = td_target - q_table[state, action]
            q_table[state, action] += alpha * td_delta

            state, action = next_state, next_action
            ep_reward += reward

        rewards_per_episode.append(ep_reward)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    return q_table, rewards_per_episode

# ---------- Main: train, evaluate, plot ----------
def smooth(x, window=10):
    if window <= 1:
        return np.array(x)
    ret = np.convolve(x, np.ones(window)/window, mode='valid')
    return ret

def main():
    env_name = "CliffWalking-v0"
    env = gym.make(env_name, render_mode=None)  # render_mode="human" to see GUI (may require X11)
    episodes = 1000
    seed = 0
    np.random.seed(seed)
    random.seed(seed)

    # Hyperparameters
    alpha = 0.5
    gamma = 1.0  # cliffwalking commonly uses gamma=1 (episodic undiscounted)
    epsilon = 0.1
    epsilon_decay = 0.9995
    min_epsilon = 0.01

    # Train Q-Learning
    q_qlearning, rewards_q = q_learning(env, episodes=episodes,
                                       alpha=alpha, gamma=gamma,
                                       epsilon=epsilon, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon)

    # Train SARSA
    q_sarsa, rewards_sarsa = sarsa(env, episodes=episodes,
                                   alpha=alpha, gamma=gamma,
                                   epsilon=epsilon, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon)

    # Save Q-tables
    os.makedirs("results", exist_ok=True)
    with open("results/q_qlearning.pkl", "wb") as f:
        pickle.dump(q_qlearning, f)
    with open("results/q_sarsa.pkl", "wb") as f:
        pickle.dump(q_sarsa, f)

    # Evaluate greedy policy
    mean_q, std_q = evaluate_policy(env, q_qlearning, episodes=100)
    mean_s, std_s = evaluate_policy(env, q_sarsa, episodes=100)
    print(f"Q-learning greedy eval: mean return={mean_q:.2f} std={std_q:.2f}")
    print(f"SARSA greedy eval: mean return={mean_s:.2f} std={std_s:.2f}")

    #SARSA plot
    plt.figure()
    plt.plot(smooth(rewards_sarsa, window=20))
    plt.title("CliffWalking - SARSA")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig("results/sarsa_rewards.png", dpi=200)
    plt.show()

    # Q learning plot
    plt.figure()
    plt.plot(smooth(rewards_q, window=20))
    plt.title("CliffWalking - Q-Learning")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig("results/q_learning_rewards.png", dpi=200)
    plt.show()

    # Plot learning curves (smoothed)
    plt.figure(figsize=(10,6))
    plt.plot(smooth(rewards_q, window=20), label="Q-Learning (smoothed)")
    plt.plot(smooth(rewards_sarsa, window=20), label="SARSA (smoothed)")
    plt.xlabel("Episode (smoothed)")
    plt.ylabel("Episode reward")
    plt.title("CliffWalking: Q-Learning vs SARSA")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/learning_curves.png", dpi=200)
    plt.show()

    env.close()

if __name__ == "__main__":
    main()

