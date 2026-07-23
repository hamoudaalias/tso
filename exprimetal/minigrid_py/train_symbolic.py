"""
Entraînement TSO avec encodeur symbolique + curiosité Φ.
Usage: python train_symbolic.py <env_name> [n_episodes]
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.setrecursionlimit(10000)
import numpy as np
import gymnasium as gym
import minigrid
from agent_symbolic import TSOSymbolicAgent


def run_episode(env, agent, max_steps=300):
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
    agent = TSOSymbolicAgent()

    print("ep,steps,reward,success,eps,classes,q_size,edges", flush=True)
    successes = []
    t0 = __import__('time').time()

    for ep in range(n_ep):
        steps, reward, success = run_episode(env, agent)
        successes.append(int(success))
        sr = sum(successes[-50:]) / min(len(successes), 50)

        if ep % 10 == 0:
            dt = __import__('time').time() - t0
            print(f"{ep},{steps},{reward:.2f},{int(success)},{agent.epsilon:.3f},"
                  f"{agent.field.n_classes()},{len(agent.q)},{agent.graph.edge_count()},"
                  f"dt={dt:.1f}s", flush=True)

        agent.decay(0.995)

    sr_final = sum(successes[-200:]) / min(len(successes), 200)
    print(f"\n=== {env_name} ===", flush=True)
    print(f"Success rate (last 200): {sr_final:.2f}", flush=True)
    print(f"LVQ1 classes: {agent.field.n_classes()}", flush=True)
    print(f"Q-table: {len(agent.q)}", flush=True)
    print(f"Graph edges: {agent.graph.edge_count()}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
