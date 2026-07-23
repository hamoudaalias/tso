#!/usr/bin/env python3
"""
Benchmark TSO vs RANDOM sur les 16 Procgen — AVEC RÉTINE TSO.
Extraction de bords (Sobel) + couleur moyenne + pooling spatial 8×8 → ~192D.
"""
import sys, gym, procgen, numpy as np
sys.path.insert(0, '/Users/hamoudaalias/Desktop/project/tso/exprimetal/procgen')
from tso_pyo3 import DualLIFState, ActionMotor, AssociativeMemory

ALL_ENVS = ['bigfish','bossfight','caveflyer','chaser','climber','coinrun',
            'dodgeball','fruitbot','heist','jumper','leaper','maze',
            'miner','ninja','plunder','starpilot']

class TsoRetina:
    """Rétine TSO bio-inspirée : bords Sobel + couleur par blocs 8×8."""
    def __init__(self, grid=8):
        self.grid = grid

    def encode(self, rgb):
        h, w = rgb.shape[:2]
        gray = np.mean(rgb, axis=2).astype(np.float64)
        # Approximant Sobel avec diff numpy (pas de scipy)
        edges_x = np.abs(np.diff(gray, axis=1))
        edges_y = np.abs(np.diff(gray, axis=0))
        edges = np.zeros_like(gray)
        edges[:, :-1] += edges_x
        edges[:-1, :] += edges_y
        bs = h // self.grid
        feats = []
        for i in range(self.grid):
            for j in range(self.grid):
                y1, x1 = i*bs, j*bs
                block = rgb[y1:y1+bs, x1:x1+bs]
                eb = edges[y1:y1+bs, x1:x1+bs]
                feats.append(eb.mean() / 255.0)
                feats.extend((block.mean(axis=(0,1)) / 255.0).tolist())
        return np.array(feats, dtype=np.float64).tolist()

def run_tso(env, retina, max_steps=500):
    dim = 8 * 8 * 4  # edge + RGB = 4 channels per block
    mem = AssociativeMemory()
    lif = DualLIFState(dim, 0.9, 0.5)
    motor = ActionMotor(0.7)
    av = [[0.3 if i==j else 0.0 for j in range(dim)] for i in range(15)]
    obs = env.reset()
    tr, stored = 0.0, False
    for step in range(max_steps):
        vec = retina.encode(obs)
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

retina = TsoRetina(grid=8)
print(f"{'Env':<15} {'TSO_retina':>10} {'RANDOM':>8} {'DIFF':>8}")
print("-" * 45)

for env_name in ALL_ENVS:
    env_tso = gym.make(f'procgen-{env_name}-v0', start_level=0, num_levels=100)
    env_rnd = gym.make(f'procgen-{env_name}-v0', start_level=0, num_levels=100)
    tso_r = np.mean([run_tso(env_tso, retina) for _ in range(5)])
    rnd_r = np.mean([run_random(env_rnd) for _ in range(5)])
    env_tso.close(); env_rnd.close()
    diff = tso_r - rnd_r
    flag = "TSO" if diff > 0.1 else ("RND" if diff < -0.1 else "=")
    print(f"{env_name:<15} {tso_r:>10.2f} {rnd_r:>8.2f} {diff:>+8.2f}  {flag}")
