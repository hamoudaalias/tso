"""
Démo TSO jouant à MiniGrid Memory-S7 avec visualisation.
Montre : LVQ1 one-shot, mémoire épisodique, LIF reservoir, Graph topologique.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import gymnasium as gym
import minigrid
from tso_minigrid_agent import TSOAgent


def demo():
    env = gym.make("MiniGrid-MemoryS7-v0", render_mode="rgb_array")
    agent = TSOAgent(epsilon=0.1)

    print("=== TSO — MiniGrid Memory-S7 ===")
    print("Composants: LIF + LVQ1 (one-shot) + Épisodique + Graph + Q-learning")

    for ep in range(10):
        obs, _ = env.reset()
        agent.reset()
        seq = []

        print(f"\n--- Épisode {ep+1} ---")
        for step in range(100):
            v = agent.featurize(obs)
            a = agent.act(v)
            obs, r, term, trunc, info = env.step(a)
            nv = agent.featurize(obs)
            done = term or trunc
            agent.learn(v, a, r, nv, done)
            seq.append(v)

            cls, dist = agent.field.predict_with_distance(v.tolist())
            ctx = agent.ctx.as_slice()
            recalled = agent.episodic.recall(ctx) if len(ctx) > 2 else None

            if step % 10 == 0:
                print(f"  step {step:3d} | action={a} | reward={r:.2f} | "
                      f"LVQ1 class={cls} dist={dist:.3f} | "
                      f"episodic_next={recalled}")

            if done:
                break

        if r > 0:
            agent.store_episode(seq)
            print(f"  ✅ SUCCÈS en {step+1} steps, reward={r:.4f}")

        protos = agent.field.get_prototypes()
        print(f"  Prototypes LVQ1: {[len(p) for p in protos]}")
        print(f"  Mémoires épisodiques: (stockées)")
        agent.decay(0.95)

    env.close()

    print(f"\n=== Résumé final ===")
    print(f"LVQ1 classes: {agent.field.n_classes()}")
    print(f"Q-table: {len(agent.q)} états")
    print(f"Epsilon: {agent.epsilon:.3f}")


if __name__ == "__main__":
    demo()
