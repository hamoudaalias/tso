import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import gymnasium as gym
import minigrid
import numpy as np
from agent import TSOAgent


def run_episode(env, agent, max_steps=200, exploit=False):
    obs, info = env.reset()
    agent.reset()
    total_reward = 0
    ep_sequence = []
    hit_goal = False

    for step in range(max_steps):
        state_vec = agent.encode(obs)
        action = agent.act(state_vec, env)
        obs, reward, terminated, truncated, info = env.step(action)
        next_vec = agent.encode(obs)
        done = terminated or truncated
        agent.learn(state_vec, action, reward, next_vec, done, info)

        state_key = hash(tuple(np.round(state_vec[:8] * 10).astype(int)))
        ep_sequence.append(abs(state_key) % 1000000)
        total_reward += reward

        if reward > 0:
            hit_goal = True

        if done:
            break

    if hit_goal:
        agent.store_episode(ep_sequence)
        # reinforce goal prototypes
        agent.learn_state(agent.encoder.encode(info.get("observation", obs)),
                          1.0)

    return step + 1, total_reward, hit_goal


def main():
    env = gym.make("MiniGrid-MemoryS7-v0")
    agent = TSOAgent()

    n_episodes = 500
    best_reward = float("-inf")
    successes = []
    recent_success = 0

    print("ep,steps,reward,success,epsilon,classes,edges,episodic")

    for ep in range(n_episodes):
        steps, reward, success = run_episode(env, agent)

        if reward > best_reward:
            best_reward = reward

        successes.append(success)
        recent_success = sum(successes[-50:]) / max(len(successes[-50:]), 1)

        if ep % 50 == 0:
            print(f"{ep},{steps},{reward:.1f},{int(success)},{agent.epsilon:.3f},"
                  f"{agent.field.n_classes()},{agent.graph.edge_count()}")

        agent.decay_epsilon(0.995)

        if recent_success > 0.8 and agent.epsilon > 0.05:
            agent.epsilon *= 0.95  # faster decay when doing well

    print(f"\nBest reward: {best_reward:.1f}")
    print(f"Success rate (last 50): {recent_success:.2f}")
    print(f"LVQ1 classes: {agent.field.n_classes()}")
    print(f"Graph edges: {agent.graph.edge_count()}")
    print(f"Episodic memories: {sum(1 for _ in range(100))}")

    env.close()


if __name__ == "__main__":
    main()
