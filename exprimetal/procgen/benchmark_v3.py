#!/usr/bin/env python3
"""
TSO v3 — Benchmark 16 Procgen (rapide).
V1 16 protos + Cerebellum Hebbian + AssociativeMemory.
Compare v3 vs aléatoire sur 3 runs par env.
"""
import sys, numpy as np, gym, procgen
sys.path.insert(0, "exprimetal/procgen")
from tso_pyo3 import DualLIFState, ActionMotor, AssociativeMemory, Cerebellum
from retina_v3 import develop_v1, encode_v1

ALL_ENVS = [
    "bigfish", "bossfight", "caveflyer", "chaser", "climber", "coinrun",
    "dodgeball", "fruitbot", "heist", "jumper", "leaper", "maze",
    "miner", "ninja", "plunder", "starpilot",
]


def make_action_vecs(dim):
    vecs = []
    for i in range(15):
        v = np.zeros(dim, dtype=np.float64)
        np.random.seed(i * 7 + 13)
        idxs = np.random.choice(dim, min(6, dim), replace=False)
        v[idxs] = 0.2
        vecs.append(v.tolist())
    return vecs


def run_v3(env, v1, max_steps=250):
    dim = 64
    lif = DualLIFState(dim, 0.95, 0.5)
    cb = Cerebellum(dim, 15, 0.05, 0.3)
    mem = AssociativeMemory()
    motor = ActionMotor(0.7)
    av = make_action_vecs(dim)
    obs = env.reset()
    total = 0.0
    for step in range(max_steps):
        vec = encode_v1(v1, obs)
        lif.step(vec, False)
        concept = lif.get_slow_state()
        rc = mem.recall_with_sim(concept)
        bonuses = [0.0] * 15
        if rc is not None and rc[1] > 0.35:
            bonuses[rc[0]] = 0.5
        cb_a = cb.forward(concept)
        bonuses[cb_a] += 0.3
        action, _ = motor.select_with_bonus(lif, av, bonuses)
        obs, r, done, info = env.step(action)
        total += r
        cb.learn(concept, action, r)
        if r > 0:
            mem.store(concept, action)
        if done:
            break
    return total


def run_rnd(env, max_steps=250):
    obs = env.reset()
    total = 0.0
    for step in range(max_steps):
        a = env.action_space.sample()
        obs, r, done, info = env.step(a)
        total += r
        if done:
            break
    return total


def main():
    print("=== V1 (16 protos, 200 frames) ===")
    cortex = {}
    for en in ALL_ENVS:
        sys.stdout.write(f"  {en}...")
        sys.stdout.flush()
        e = gym.make(f"procgen-{en}-v0", start_level=0, num_levels=50)
        cx = develop_v1(e, n_steps=200, n_protos=16)
        cortex[en] = cx
        e.close()
        print(f" {cx.n_classes()} protos")

    print("\n=== Benchmark v3 vs Aléatoire ===")
    print(f"{'Env':<15} {'TSO v3':>8} {'ALEA':>8} {'Δ':>8}")
    print("-" * 41)
    wins = 0
    for en in ALL_ENVS:
        cx = cortex[en]
        e3 = gym.make(f"procgen-{en}-v0", start_level=0, num_levels=100)
        er = gym.make(f"procgen-{en}-v0", start_level=0, num_levels=100)
        t = np.mean([run_v3(e3, cx) for _ in range(3)])
        d = np.mean([run_rnd(er) for _ in range(3)])
        e3.close()
        er.close()
        df = t - d
        fl = "v3" if df > 0.1 else ("RND" if df < -0.1 else "=")
        if fl == "v3":
            wins += 1
        print(f"{en:<15} {t:>+8.3f} {d:>+8.3f} {df:>+8.3f}  {fl}")

    print(f"\nv3 gagne {wins}/16")


if __name__ == "__main__":
    main()
