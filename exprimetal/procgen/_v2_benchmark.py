#!/usr/bin/env python3
"""TSO v2 : V1 Cortex + Cerebellum (Hebbian learning) sur 16 Procgen."""
import sys, gym, procgen, numpy as np
sys.path.insert(0, '/Users/hamoudaalias/Desktop/project/tso/exprimetal/procgen')
from tso_pyo3 import DualLIFState, AssociativeMemory, AttractorField, Cerebellum

ALL_ENVS = ['bigfish','bossfight','caveflyer','chaser','climber','coinrun',
            'dodgeball','fruitbot','heist','jumper','leaper','maze',
            'miner','ninja','plunder','starpilot']

def get_blocks(rgb):
    h,w = rgb.shape[:2]; g = np.mean(rgb, axis=2).astype(np.float64)
    ex = np.abs(np.diff(g, axis=1)); ey = np.abs(np.diff(g, axis=0))
    e = np.zeros_like(g); e[:,:-1]+=ex; e[:-1,:]+=ey
    bs = h//8; blks = []
    for i in range(8):
        for j in range(8):
            y,x = i*bs, j*bs
            b = rgb[y:y+bs, x:x+bs]; eb = e[y:y+bs, x:x+bs]
            blks.append([eb.mean()/255.0] + (b.mean(axis=(0,1))/255.0).tolist())
    return np.array(blks, dtype=np.float64)

def develop_v1(env, n_steps=500):
    field = None; n_protos = 16; init = False
    for ep in range(n_steps//50 + 1):
        obs = env.reset() if hasattr(env, 'reset') else env.reset()
        blks = get_blocks(obs)
        for b in blks:
            if not init:
                if field is None: field = AttractorField(4, n_protos, 3, 0.05)
                field.add_class(b.tolist())
                init = (field.n_classes() >= n_protos)
            else:
                field.train_step(b.tolist(), field.predict(b.tolist()))
    return field

def encode(field, rgb):
    blks = get_blocks(rgb); vec = np.zeros(64, dtype=np.float64)
    for i,b in enumerate(blks):
        if i < 64: vec[i] = field.predict(b.tolist()) / 15.0
    return vec.tolist()

def run_tso_v2(env, field, ms=500):
    dim = 64; cb = Cerebellum(dim, 15, 0.05, 0.3)
    mem = AssociativeMemory(); lif = DualLIFState(dim, 0.9, 0.5)
    obs = env.reset() if hasattr(env, 'reset') else env.reset()
    tr = 0.0
    for st in range(ms):
        vec = encode(field, obs); lif.step(vec, False)
        rc = mem.recall_with_sim(vec)
        if rc is not None and rc[1] > 0.35:
            a = rc[0]
        else:
            a = cb.forward(vec)
        obs, r, done, info = env.step(a); tr += r
        cb.learn(vec, a, r)
        if r > 0: mem.store(vec, a)
        if done: break
    return tr

def run_rand(env, ms=500):
    obs = env.reset() if hasattr(env, 'reset') else env.reset(); tr = 0.0
    for st in range(ms):
        a = env.action_space.sample()
        obs,r,done,info = env.step(a); tr += r
        if done: break
    return tr

print("=== Phase 1: Developpement V1 ===")
cortex = {}
for en in ALL_ENVS:
    sys.stdout.write(f"  {en}..."); sys.stdout.flush()
    env = gym.make('procgen-{}-v0'.format(en), start_level=0, num_levels=100)
    cx = develop_v1(env, n_steps=500); cortex[en] = cx; env.close()
    print(f" {cx.n_classes()} protos")

print("\n=== Phase 2: TSO v2 vs RANDOM ===")
print('{:<15} {:>10} {:>8} {:>8}'.format('Env','TSO v2','RANDOM','DIFF'))
print('-'*47)
for en in ALL_ENVS:
    cx = cortex[en]
    e1 = gym.make('procgen-{}-v0'.format(en), start_level=0, num_levels=100)
    e2 = gym.make('procgen-{}-v0'.format(en), start_level=0, num_levels=100)
    t = np.mean([run_tso_v2(e1, cx) for _ in range(5)])
    d = np.mean([run_rand(e2) for _ in range(5)])
    e1.close(); e2.close()
    df = t - d; fl = 'TSO' if df > 0.1 else ('RND' if df < -0.1 else '=')
    print('{:<15} {:>10.2f} {:>8.2f} {:>+8.2f}  {}'.format(en, t, d, df, fl))
