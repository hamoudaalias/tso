"""
Procgen Heist réel — test WorkingMemory TSO sur pixels 64×64×3.
Encodeur : downsample 16×16 + flatten RGB → 768D vecteur.
Store : quand le niveau est complété (reward > 0), stocker l'obs.
Ablation : NORMAL (mémoire persistante) vs AMNÉSIQUE (reset chaque step).
"""
import sys, random
sys.path.insert(0, "exprimetal/procgen")
import numpy as np
import gym, procgen
from tso_pyo3 import AssociativeMemory, DualLIFState, ActionMotor


def encode(obs):
    """Downsample 64×64×3 → 16×16×3 → flatten 768D, normalize to [0,1]."""
    small = obs[::4, ::4, :].astype(np.float64) / 255.0
    return small.flatten().tolist()


def run_episode(env, amnesic=False, max_steps=500):
    mem = AssociativeMemory()
    dim = 768
    lif = DualLIFState(dim, 0.9, 0.5)
    motor = ActionMotor(0.7)
    obs = env.reset()
    exit_seen = False
    step = 0

    # Action embeddings (15 actions Procgen → 15 vecteurs)
    action_vecs = []
    for i in range(15):
        v = np.zeros(dim, dtype=np.float64)
        v[i * (dim // 15):(i + 1) * (dim // 15)] = 0.2
        action_vecs.append(v.tolist())

    for _ in range(max_steps):
        if amnesic:
            mem = AssociativeMemory()
            lif = DualLIFState(dim, 0.9, 0.5)
            exit_seen = False

        vec = encode(obs)
        lif.step(vec, False)

        # Store observation when level complete
        # (in heist, reward > 0 means gem stolen / level complete)
        # We use the current obs as "target memory"
        if not exit_seen:
            # Every step, store current obs in memory
            # (agent will recognize this view later)
            pass

        # Action selection via ActionMotor
        action, score = motor.select(lif, action_vecs)
        
        obs, reward, done, info = env.step(action)
        step += 1

        # Store the observation when level completes
        if not exit_seen and reward > 0:
            mem.store(vec, 1)
            exit_seen = True
            # Continue exploring — need to return to exit after collecting gem
            # but in real procgen, the episode ends when gem is collected
            # so we can't test return-to-exit here

        if done:
            return reward > 0, step

    return False, step


def main():
    env = gym.make('procgen-heist-v0', start_level=0, num_levels=100)
    
    for label, amnesic in [("NORMAL", False), ("AMNÉSIQUE", True)]:
        results = []
        print(f"\n=== {label} ===")
        for ep in range(50):
            ok, steps = run_episode(env, amnesic)
            tag = "CORRECT" if ok else "WRONG"
            results.append(tag)
            sys.stdout.write(f"\rEp {ep+1:2d}: {tag:8s} step={steps:3d}")
            sys.stdout.flush()
        print()
        c = results.count("CORRECT")
        print(f"{label}: CORRECT {c}/50 ({c*2}%) | WRONG {50-c}/50")

    env.close()


if __name__ == "__main__":
    main()
