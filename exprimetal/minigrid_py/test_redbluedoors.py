"""
DoorMatch-v0 — Test mémoire épisodique TSO.
Portes de choix sur 2 colonnes différentes (x=1 et x=7).
Navigation scriptée garantie vers les deux portes.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import gymnasium as gym
from tso_pyo3 import AssociativeMemory
from door_match_env import DoorMatchEnv

gym.register(id="MiniGrid-DoorMatch-v0", entry_point="door_match_env:DoorMatchEnv")

COLOR_IDS = {"red": 0, "blue": 2}


def run_episode(env, amnesic=False, max_steps=80):
    obs, _ = env.reset()
    uw = env.unwrapped
    grid = uw.grid
    w, h = grid.width, grid.height

    mem = AssociativeMemory()
    start_cid = COLOR_IDS[uw.start_door.color]

    # Stocker la couleur de départ (one-hot)
    if not amnesic:
        v = np.zeros(6, dtype=np.float64)
        v[start_cid] = 1.0
        mem.store(v.tolist(), start_cid)

    # Phase 1 : navigation scriptée vers (7,10) [au-dessus de la porte droite]
    # (1,4)↑ → toggle D(1,3) → forward×3 → right → forward×6 → down → forward×9
    actions = [
        5,  # toggle (1,3)
        2,  # forward to (1,3)
        2,  # forward to (1,2)
        2,  # forward to (1,1)
        1,  # turn right
        2, 2, 2, 2, 2, 2,  # forward 6× to (7,1)
        1,  # turn down
        2, 2, 2, 2, 2, 2, 2, 2, 2,  # forward 9× to (7,10)
    ]

    for a in actions:
        obs, r, term, trunc, _ = env.step(a)
        if r > 0: return True, 0
        if r < 0: return False, 0

    # Phase 2 : porte droite en (7,11) — vérifier la couleur
    right_door = grid.get(7, 11)
    right_cid = COLOR_IDS[right_door.color]

    if amnesic:
        toggle_right = np.random.random() < 0.5
    else:
        q = np.zeros(6)
        q[right_cid] = 1.0
        r = mem.recall_with_sim(q.tolist())
        toggle_right = r is not None and r[1] > 0.5

    if toggle_right:
        obs, r, term, trunc, _ = env.step(5)  # toggle → either SUCCESS or FAILURE
        if r > 0: return True, 0
        if r < 0: return False, 0

    # Phase 3 : la porte droite n'a pas été ouverte → aller à la porte gauche
    # Tourner à droite (face LEFT), avancer 6× to (1,10), tourner à gauche, avancer
    env.step(1)  # turn right → face LEFT
    for _ in range(6):
        env.step(2)  # forward to (1,10)
    env.step(0)  # turn left → face DOWN
    env.step(2)  # forward to (1,11)

    # Porte gauche en (1,11)
    left_door = grid.get(1, 11)
    left_cid = COLOR_IDS[left_door.color]

    if amnesic:
        toggle_left = np.random.random() < 0.5
    else:
        q = np.zeros(6)
        q[left_cid] = 1.0
        r = mem.recall_with_sim(q.tolist())
        toggle_left = r is not None and r[1] > 0.5

    obs, r, term, trunc, _ = env.step(5)  # toggle left door
    if r > 0: return True, 0
    if r < 0: return False, 0

    return False, max_steps


def main():
    env = gym.make("MiniGrid-DoorMatch-v0")

    for label, amnesic in [("NORMAL", False), ("AMNÉSIQUE", True)]:
        successes = []
        print(f"\n=== {label} ===")
        for ep in range(100):
            ok, _ = run_episode(env, amnesic=amnesic)
            successes.append(ok)
            if (ep + 1) % 10 == 0:
                sr = sum(successes) / len(successes) * 100
                sys.stdout.write(f"\rEp {ep+1:3d}/100: {sum(successes)}/{ep+1} ({sr:.0f}%)")
                sys.stdout.flush()
        print()
        sr = sum(successes) / 100 * 100
        print(f"{label}: {sr:.0f}%")
        print(f"  > 50% ? {'OUI ✓' if sr > 50 else 'NON ✗'}")

    env.close()


if __name__ == "__main__":
    main()
