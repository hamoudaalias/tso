"""
Test OneShot-v0 : l'agent marche à droite, stocke la cible,
puis reconnaît le matching objet via AssociativeMemory (k-NN cosinus).
"""
import sys
sys.path.insert(0, "exprimetal/minigrid_py")
import numpy as np
from tso_pyo3 import AssociativeMemory
from oneshot_env import OneShotEnv


def obj_vec(t, c, s=0):
    oh = {"ball": [1,0,0,0], "key": [0,1,0,0]}.get(t, [0,0,0,0])
    col = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}.get(c, 0)
    return np.array(oh + [col / 6.0, s / 3.0], dtype=np.float64)


def extract(obs):
    img = obs["image"]
    objs = []
    for y in range(7):
        for x in range(7):
            t = int(img[y, x, 0])
            c = int(img[y, x, 1])
            s = int(img[y, x, 2])
            tname = {5: "key", 6: "ball"}.get(t)
            cname = {0: "red", 1: "green", 2: "blue", 3: "purple", 4: "yellow", 5: "grey"}.get(c)
            if tname is None:
                continue
            dx, dy = (x - 3) / 3.5, (y - 3) / 3.5
            objs.append((obj_vec(tname, cname, s), dx, dy, tname, cname))
    return objs


def main():
    env = OneShotEnv(max_steps=50)
    mem = AssociativeMemory()

    results = []
    for ep in range(50):
        obs, _ = env.reset()
        mem = AssociativeMemory()
        stored = False
        picked = False
        target_type = None

        for step in range(50):
            objs = extract(obs)

            # Stocker la cible au premier objet vu
            if not stored and objs:
                vec, _, _, tname, cname = objs[0]
                mem.store(vec.tolist(), 0)
                target_type = (tname, cname)
                stored = True

            # Vérifier s'il y a un objet devant nous (dx ~ 0)
            obj_ahead = None
            for vec, dx, dy, tname, cname in objs:
                if abs(dx) < 0.3 and dy < 0.3:  # juste devant
                    # Vérifier la similarité avec la mémoire
                    result = mem.recall_with_sim(vec.tolist())
                    if result is not None:
                        data, sim = result
                        if sim > 0.8:
                            obj_ahead = (tname, cname, sim)

            # Décision
            if obj_ahead and not picked:
                a = 3  # PICKUP
                picked = True
            else:
                a = 2  # FORWARD

            obs, r, term, trunc, _ = env.step(a)
            if term or trunc:
                if r > 0:
                    results.append("CORRECT")
                else:
                    results.append("WRONG")
                break
        else:
            results.append("TIMEOUT")

        print(f"Ep {ep+1:2d}: {results[-1]:8s} target={target_type} step={step+1:2d}")

    correct = results.count("CORRECT")
    wrong = results.count("WRONG")
    timeout = results.count("TIMEOUT")
    print(f"\nCORRECT: {correct}/50 ({correct*2}%)")
    print(f"WRONG:   {wrong}/50")
    print(f"TIMEOUT: {timeout}/50")
    print(f"\nOne-shot matching: {'✅ RÉUSSI' if correct > 40 else '⚠️ PARTIEL'}")


if __name__ == "__main__":
    main()
