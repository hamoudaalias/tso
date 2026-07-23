"""
Ablation test OneShot-v0 : WorkingMemory natif (normal vs amnésique).
Layout-aware navigation : contournement par (9,1) pour matching à y=1.
"""
import sys, random
sys.path.insert(0, "exprimetal/minigrid_py")
import numpy as np
from tso_pyo3 import WorkingMemory
from oneshot_env import OneShotEnv

def obj_vec(tname, cname, s=0):
    oh = {"ball":[1,0,0,0], "key":[0,1,0,0]}.get(tname, [0,0,0,0])
    ci = {"red":0,"green":1,"blue":2,"purple":3,"yellow":4,"grey":5}.get(cname, 0)
    return np.array(oh + [ci/6.0, s/3.0])

def extract(obs):
    img = obs["image"]
    objs = []
    for y in range(7):
        for x in range(7):
            t, c, s = int(img[y,x,0]), int(img[y,x,1]), int(img[y,x,2])
            tn = {5:"key", 6:"ball"}.get(t)
            cn = {0:"red",1:"green",2:"blue",3:"purple",4:"yellow",5:"grey"}.get(c)
            if tn is None: continue
            dx, dy = (x-3)/3.5, (y-3)/3.5
            objs.append((obj_vec(tn, cn, s), dx, dy, tn, cn))
    return objs

def identify_matching(objs, wm, thresh=0.85):
    for vec, dx, dy, tn, cn in objs:
        res = wm.recall(vec.tolist())
        if res is not None and res[1] > thresh:
            return dx, dy, tn, cn
    return None

def run_episode(env, amnesic=False):
    uw = env.unwrapped
    wm = WorkingMemory(6, 0.99, 0.5)
    obs, _ = env.reset()
    stored = False
    step = 0
    i = 0
    plan = [2]*10 + [0, 2, 1]  # nav to (12,3) facing right

    while True:
        if amnesic:
            wm.reset()

        if not stored:
            objs = extract(obs)
            if objs:
                wm.observe([o[0].tolist() for o in objs])
                stored = True

        if i < len(plan):
            a = plan[i]; i += 1
        else:
            objs = extract(obs)
            match = identify_matching(objs, wm) if objs else None
            if match:
                dx, dy, tn, cn = match
                if abs(dy) < 0.3:
                    a = 3  # PICKUP direct (matching at y=3)
                else:
                    # Bypass par (9,1) : (12,3)→9,1→12,1→PICKUP
                    plan = [1, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 3]
                    i = 0; a = plan[i]; i += 1
            else:
                rnd = random.random()
                a = 2 if rnd < 0.6 else (0 if rnd < 0.8 else 1)

        obs, r, term, trunc, _ = env.step(a)
        step += 1
        if term or trunc:
            return r > 0, step

def main():
    env = OneShotEnv(max_steps=100)
    for label, amnesic in [("NORMAL", False), ("AMNÉSIQUE", True)]:
        results = []
        print(f"\n=== {label} ===")
        for ep in range(50):
            ok, steps = run_episode(env, amnesic)
            tag = "CORRECT" if ok else "WRONG"
            results.append(tag)
            sys.stdout.write(f"\rEp {ep+1:2d}: {tag:8s} step={steps:2d}")
            sys.stdout.flush()
        print()
        c = results.count("CORRECT"); w = results.count("WRONG")
        print(f"{label}: CORRECT {c}/50 ({c*2}%) | WRONG {w}/50")

if __name__ == "__main__":
    main()
