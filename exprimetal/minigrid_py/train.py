"""
Entraîne l'agent TSO sur un environnement MiniGrid.
Usage: python train.py <env_name> [n_episodes]
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import gymnasium as gym
import minigrid
from tso_minigrid_agent import TSOAgent


def run_episode(env, agent, max_steps=200):
    obs, _ = env.reset()
    agent.reset()
    seq = []
    for step in range(max_steps):
        v = agent.featurize(obs)
        a = agent.act(v)
        obs, r, term, trunc, _ = env.step(a)
        nv = agent.featurize(obs)
        done = term or trunc
        agent.learn(v, a, r, nv, done)
        seq.append(v)
        if done:
            break
    if r > 0:
        agent.store_episode(seq)
    return step + 1, r, term


def main():
    env_name = sys.argv[1] if len(sys.argv) > 1 else "MiniGrid-MemoryS7-v0"
    n_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 500

    env = gym.make(env_name)
    agent = TSOAgent()

    print("ep,steps,reward,success,eps,classes,edges")
    successes = []
    best = float("-inf")

    for ep in range(n_ep):
        steps, reward, success = run_episode(env, agent)
        successes.append(int(success))
        if reward > best:
            best = reward
        sr = sum(successes[-100:]) / min(len(successes), 100)
        if ep % 50 == 0 or (ep > 0 and ep % 10 == 0 and sr > 0.3):
            print(f"{ep},{steps},{reward},{int(success)},{agent.epsilon:.3f},"
                  f"{agent.field.n_classes()},{len(agent.q)}")
        agent.decay(0.995)

    sr_final = sum(successes[-200:]) / min(len(successes), 200)
    print(f"\n=== Results: {env_name} ===")
    print(f"Episodes: {n_ep}")
    print(f"Best reward: {best}")
    print(f"Success rate (last 200): {sr_final:.2f}")
    print(f"LVQ1 classes: {agent.field.n_classes()}")
    print(f"Q-table size: {len(agent.q)}")

    env.close()


if __name__ == "__main__":
    main()
