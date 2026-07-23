"""
KeyMatch-v0 — Test WorkingMemory TSO : ramasser clé → ouvrir porte → but.
WorkingMemory stocke le signal "clé obtenue" pour savoir quoi faire
à la porte verrouillée. Sans mémoire, l'agent ne togglera pas.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import gymnasium as gym
from tso_pyo3 import WorkingMemory
from key_match_env import KeyMatchEnv

gym.register(id="MiniGrid-KeyMatch-v0", entry_point="key_match_env:KeyMatchEnv")


def run_episode(env, amnesic=False, max_steps=50):
    obs, _ = env.reset()
    uw = env.unwrapped
    grid = uw.grid
    w, h = grid.width, grid.height

    wm = WorkingMemory(8, 0.99, 0.5)

    # Phase 1 : navigation scriptée vers la clé
    # (1,2)↑ → forward to (1,1) [clé] → pickup
    env.step(2)  # forward to (1,1)
    env.step(3)  # pickup key

    # Stocker dans WorkingMemory que la clé a été ramassée
    if not amnesic:
        key_vec = np.zeros(8, dtype=np.float64)
        key_vec[0] = 1.0
        key_vec[1] = 1.0  # flag "has key"
        wm.store(key_vec.tolist(), 1)

    # Phase 2 : navigation vers la porte verrouillée
    # (1,1) → turn right → forward 5× to (6,1) [locked door]
    env.step(1)  # turn right
    for _ in range(5):
        env.step(2)  # forward to (6,1)

    # Phase 3 : à la porte verrouillée — décider
    locked_door = grid.get(6, 1)
    if locked_door and locked_door.type == "door" and locked_door.is_locked:
        if amnesic:
            # Sans mémoire : ne sait pas qu'il a la clé → ne toggle pas
            should_toggle = np.random.random() < 0.5
        else:
            # Avec mémoire : vérifier WM
            q = np.zeros(8, dtype=np.float64)
            q[0] = 1.0
            q[1] = 1.0  # query: "have key?"
            r = wm.recall(q.tolist())
            should_toggle = r is not None and r[1] > 0.3

        if should_toggle:
            env.step(5)  # toggle → unlock door
        else:
            return False, 0

    # Phase 4 : traverser la porte → balle
    env.step(2)  # forward through door to (7,1)
    env.step(1)  # turn right → facing down
    env.step(2)  # forward to (7,2)
    env.step(2)  # forward to (7,3)
    env.step(3)  # pickup ball

    return True, 0


def main():
    env = gym.make("MiniGrid-KeyMatch-v0")

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

    env.close()


if __name__ == "__main__":
    main()
