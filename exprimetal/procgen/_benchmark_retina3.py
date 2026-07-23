#!/usr/bin/env python3
"""
Benchmark TSO sur 16 Procgen — Rétine + Mémoire associative.
Store: (features_avant_reward, action) quand reward>0.
Recall: recall_with_sim, threshold 0.5 → biais ActionMotor.
"""
import sys, gym, procgen, numpy as np, math
sys.path.insert(0, '/Users/hamoudaalias/Desktop/project/tso/exprimetal/procgen')
from tso_pyo3 import DualLIFState, ActionMotor, AssociativeMemory

ALL_ENVS = ['bigfish','bossfight','caveflyer','chaser','climber','coinrun',
            'dodgeball','fruitbot','heist','jumper','leaper','maze',
            'miner','ninja','plunder','starpilot']

class TsoRetina:
    def __init__(self, grid=8): self.grid = grid
    def encode(self, rgb):
        h,w = rgb.shape[:2]; g = np.mean(rgb, axis=2).astype(np.float64)
        ex = np.abs(np.diff(g, axis=1)); ey = np.abs(np.diff(g, axis=0))
        e = np.zeros_like(g); e[:,:-1] += ex; e[:-1,:] += ey
        bs = h//self.grid; f = []
        for i in range(self.grid):
            for j in range(self.grid):
                y,x = i*bs, j*bs
                b = rgb[y:y+bs, x:x+bs]; eb = e[y:y+bs, x:x+bs]
                f.append(eb.mean()/255.0)
                f.extend((b.mean(axis=(0,1))/255.0).tolist())
        return np.array(f, dtype=np.float64).tolist()

def make_av(dim):
    dirs = [(0,-1),(1,0),(0,1),(-1,0)]
    vecs = []
    for i in range(15):
        v = np.zeros(dim, dtype=np.float64)
        if i < 4:
            dx,dy = dirs[i]; a = math.atan2(dy,dx)
            v[0] = math.cos(a); v[1] = math.sin(a)
            v[2] = 0.3
        else:
            np.random.seed(i*7+13)
            v[np.random.randint(0, dim, 8)] = 0.2
        vecs.append(v.tolist())
    return vecs

def run_tso(env, retina, max_steps=500):
    dim = 8*8*4; av = make_av(dim)
    mem = AssociativeMemory()
    lif = DualLIFState(dim, 0.9, 0.5)
    motor = ActionMotor(0.7)
    obs = env.reset()
    tr = 0.0
    for step in range(max_steps):
        vec = retina.encode(obs)
        lif.step(vec, False)
        # Recall: meilleure action stockée pour features similaires?
        rcall = mem.recall_with_sim(vec)
        if rcall is not None and rcall[1] > 0.4:
            best_a = rcall[0]
            bonuses = [0.0]*15
            bonuses[best_a] = 0.5
            a, _ = motor.select_with_bonus(lif, av, bonuses)
        else:
            a, _ = motor.select(lif, av)
        obs, r, done, info = env.step(a)
        tr += r
        if r > 0:
            mem.store(vec, a)
        if done: break
    return tr

def run_random(env, max_steps=500):
    obs = env.reset(); tr = 0.0
    for step in range(max_steps):
        a = env.action_space.sample()
        obs, r, done, info = env.step(a); tr += r
        if done: break
    return tr

retina = TsoRetina(8)
print('{:<15} {:>10} {:>8} {:>8}'.format('Env','TSO_mem','RANDOM','DIFF'))
print('-'*45)
for env_name in ALL_ENVS:
    env_tso = gym.make('procgen-{}-v0'.format(env_name), start_level=0, num_levels=100)
    env_rnd = gym.make('procgen-{}-v0'.format(env_name), start_level=0, num_levels=100)
    tso_r = np.mean([run_tso(env_tso, retina) for _ in range(5)])
    rnd_r = np.mean([run_random(env_rnd) for _ in range(5)])
    env_tso.close(); env_rnd.close()
    diff = tso_r - rnd_r
    flag = 'TSO' if diff > 0.1 else ('RND' if diff < -0.1 else '=')
    print('{:<15} {:>10.2f} {:>8.2f} {:>+8.2f}  {}'.format(env_name, tso_r, rnd_r, diff, flag))
