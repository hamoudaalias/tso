import sys, gym, procgen, numpy as np
sys.path.insert(0, '/Users/hamoudaalias/Desktop/project/tso/exprimetal/procgen')
from tso_pyo3 import DualLIFState, ActionMotor, AssociativeMemory

def encode(obs):
    gray = np.mean(obs, axis=2).astype(np.float64) / 255.0
    small = gray[::8, ::8]
    return small.flatten().tolist()

print("=== procgen-maze: TSO NORMAL ===")
env = gym.make('procgen-maze-v0', start_level=0, num_levels=100)
dim = 64
for ep in range(5):
    mem = AssociativeMemory()
    lif = DualLIFState(dim, 0.9, 0.5)
    motor = ActionMotor(0.7)
    av = [[0.3 if i==j else 0.0 for j in range(dim)] for i in range(15)]
    obs = env.reset()
    tr, stored = 0.0, False
    for step in range(500):
        vec = encode(obs)
        lif.step(vec, False)
        if not stored and step > 10 and step % 30 == 0:
            mem.store(vec, 1)
            stored = True
        if stored:
            bonuses = [0.5]*15
            a, _ = motor.select_with_bonus(lif, av, bonuses)
        else:
            a, _ = motor.select(lif, av)
        obs, r, done, info = env.step(a)
        tr += r
        if done:
            break
    print("  Ep {}: reward={:.3f}, steps={}".format(ep+1, tr, step+1))
env.close()

print("\n=== procgen-maze: RANDOM ===")
env = gym.make('procgen-maze-v0', start_level=0, num_levels=100)
for ep in range(5):
    obs = env.reset()
    tr = 0.0
    for step in range(500):
        a = env.action_space.sample()
        obs, r, done, info = env.step(a)
        tr += r
        if done:
            break
    print("  Ep {}: reward={:.3f}, steps={}".format(ep+1, tr, step+1))
env.close()
