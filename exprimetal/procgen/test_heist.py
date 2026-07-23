"""
ProcgenHeist v2 — Ablation WorkingMemory. Phase unique continue.

Agent explore avec forward-bias. Store quand exit visible.
Quand gem collectée + exit stored → agent utilise recall(obs) pour
reconnaître le couloir de sortie et forcer FORWARD.

NORMAL: mémoire persistante → reconnaissance du couloir exit.
AMNÉSIQUE: reset chaque step → pas de reconnaissance.
"""
import sys, random
sys.path.insert(0, "exprimetal/procgen")
import numpy as np
from tso_pyo3 import AssociativeMemory
from procgen_maze_env import ProcgenHeistEnv


def encode(obs):
    return (obs.flatten().astype(np.float64) / 5.0).tolist()


def run_episode(env, amnesic=False, max_steps=200):
    mem = AssociativeMemory()
    obs, _ = env.reset()
    exit_stored = False
    gem_collected = False
    step = 0

    for _ in range(max_steps):
        if amnesic:
            mem = AssociativeMemory()
            exit_stored = False

        # Store exit vector on first sight
        if not exit_stored and 5 in obs:
            mem.store(encode(obs), 1)
            exit_stored = True

        # Track gem
        if not gem_collected and 3 in obs:
            gem_collected = True

        # Decision: recall-based or exploration
        recall = mem.recall_with_sim(encode(obs)) if exit_stored else None

        if recall is not None and recall[1] > 0.80:
            action = 2
        else:
            # Exploration: forward bias + random turns
            if obs[1, 2] != 1 and random.random() < 0.7:
                action = 2
            elif obs[2, 3] != 1 and random.random() < 0.5:
                action = 1
            elif obs[2, 1] != 1:
                action = 0
            else:
                action = random.choice([0, 1, 2])

        obs, r, term, trunc, _ = env.step(action)
        step += 1

        if term:
            return True, step
        if trunc:
            return False, step

    return False, step


def main():
    env = ProcgenHeistEnv(max_steps=200)
    for label, amnesic in [("NORMAL", False), ("AMNÉSIQUE", True)]:
        results = []
        print(f"\n=== {label} ===")
        for ep in range(100):
            ok, steps = run_episode(env, amnesic)
            tag = "CORRECT" if ok else "WRONG"
            results.append(tag)
            sys.stdout.write(f"\rEp {ep+1:3d}: {tag:8s} step={steps:3d}")
            sys.stdout.flush()
        print()
        c = results.count("CORRECT")
        print(f"{label}: CORRECT {c}/100 ({c}%) | WRONG {100-c}/100")


if __name__ == "__main__":
    main()
