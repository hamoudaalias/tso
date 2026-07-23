#!/usr/bin/env python3
"""
Benchmark TSO vs RANDOM sur les 16 environnements Procgen.
5 épisodes chacun, max 500 steps.
Encodeur: downsample 8x8 grayscale + flatten (64D)
TSO: DualLIF + ActionMotor + AssociativeMemory
"""
import sys, gym, procgen, numpy as np
sys.path.insert(0, '/Users/hamoudaalias/Desktop/project/tso/exprimetal/procgen')
from tso_pyo3 import DualLIFState, ActionMotor, AssociativeMemory

ALL_ENVS = ['bigfish','bossfight','caveflyer','chaser','climber','coinrun',
            'dodgeball','fruitbot','heist','jumper','leaper','maze',
            'miner','ninja','plunder','starpilot']

def encode(obs):
    gray = np.mean(obs, axis=2).astype(np.float64) / 255.0
    return gray[::8, ::8].flatten().tolist()

def run_tso(env, max_steps=500):
    mem, dim = AssociativeMemory(), 64
    lif = DualLIFState(dim, 0.9, 0.5)
    motor = ActionMotor(0.7)
    av = [[0.3 if i==j else 0.0 for j in range(dim)] for i in range(15)]
    obs = env.reset()
    tr, stored = 0.0, False
    for step in range(max_steps):
        vec = encode(obs)
        lif.step(vec, False)
        if not stored and step > 10 and step % 30 == 0:
            mem.store(vec, 1); stored = True
        if stored:
            a, _ = motor.select_with_bonus(lif, av, [0.5]*15)
        else:
            a, _ = motor.select(lif, av)
        obs, r, done, info = env.step(a)
        tr += r
        if done: break
    return tr

def run_random(env, max_steps=500):
    obs = env.reset()
    tr = 0.0
    for step in range(max_steps):
        a = env.action_space.sample()
        obs, r, done, info = env.step(a)
        tr += r
        if done: break
    return tr

print(f"{'Env':<15} {'TSO':>8} {'RANDOM':>8} {'DIFF':>8}")
print("-" * 42)

for env_name in ALL_ENVS:
    env_tso = gym.make(f'procgen-{env_name}-v0', start_level=0, num_levels=100)
    env_rnd = gym.make(f'procgen-{env_name}-v0', start_level=0, num_levels=100)
    tso_r = np.mean([run_tso(env_tso) for _ in range(5)])
    rnd_r = np.mean([run_random(env_rnd) for _ in range(5)])
    env_tso.close(); env_rnd.close()
    diff = tso_r - rnd_r
    print(f"{env_name:<15} {tso_r:>8.2f} {rnd_r:>8.2f} {diff:>+8.2f}")
