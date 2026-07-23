"""
TSO sur Procgen réel — heist & maze.
Features visuelles simplifiées + TSO WorkingMemory + ActionMotor.
NORMAL vs AMNÉSIQUE vs ALEA.
"""
import sys, random, numpy as np
sys.path.insert(0, "exprimetal/procgen")
import gym, procgen
from tso_pyo3 import AssociativeMemory, DualLIFState, ActionMotor


def extract_features(obs):
    """Features visuelles 48D depuis pixels 64×64×3.
    - 4×4 grid de blocs 16×16 : moyenne RGB (3×16=48D)
    - Normalisé [0,1]
    """
    h, w = 4, 4
    block_h, block_w = 64 // h, 64 // w
    feats = []
    for i in range(h):
        for j in range(w):
            block = obs[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w, :]
            feats.extend(block.mean(axis=(0, 1)).tolist())
    return [v / 255.0 for v in feats]


def run_tso_episode(env, amnesic=False, max_steps=1000):
    mem = AssociativeMemory()
    dim = 48
    lif = DualLIFState(dim, 0.95, 0.5)
    motor = ActionMotor(0.6)
    n_actions = 15

    action_vecs = []
    for i in range(n_actions):
        v = np.zeros(dim, dtype=np.float64)
        v[i % dim] = 0.5
        action_vecs.append(v.tolist())

    obs = env.reset()
    total_reward = 0
    exit_stored = False

    for step in range(max_steps):
        if amnesic:
            mem = AssociativeMemory()
            lif = DualLIFState(dim, 0.95, 0.5)
            exit_stored = False

        vec = extract_features(obs)
        lif.step(vec, False)

        if not exit_stored and step > 10 and step % 20 == 0:
            mem.store(vec, 1)
            exit_stored = True

        recall = mem.recall(vec) if exit_stored else None

        if recall is not None:
            bonuses = [0.0] * n_actions
            bonuses[6] = 0.2  # UP
            action, _ = motor.select_with_bonus(lif, action_vecs, bonuses)
        else:
            action, _ = motor.select(lif, action_vecs)

        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward, step


def run_random_episode(env, max_steps=1000):
    obs = env.reset()
    total_reward = 0
    for step in range(max_steps):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward, step


def main():
    for env_name in ['heist', 'maze']:
        print(f"\n=== procgen-{env_name} ===")

        for label, amnesic in [("NORMAL", False), ("AMNÉSIQUE", True), ("ALEA", None)]:
            env = gym.make(f'procgen-{env_name}-v0', start_level=0, num_levels=50)
            rewards = []
            print(f"\n  {label}:")
            for ep in range(50):
                if label == "ALEA":
                    r, s = run_random_episode(env)
                else:
                    r, s = run_tso_episode(env, amnesic=amnesic)
                rewards.append(r)
                tag = "OK" if r > 0 else "NO"
                sys.stdout.write(f"\r    Ep {ep+1:2d}: reward={r:+.3f} {tag}")
                sys.stdout.flush()
            print()
            successes = sum(1 for r in rewards if r > 0)
            print(f"    Mean reward: {np.mean(rewards):+.3f} | Success: {successes}/50 ({successes*2}%)")
            env.close()


if __name__ == "__main__":
    main()
